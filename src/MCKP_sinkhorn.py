"""
Sinkhorn-OT with Chen et al. (MCKP) quadratic cost for dynamic mixed-precision bit allocation.
- Cost C[i,j] = ΔL_i(b_j) ≈ 0.5 * tr(H_i) * sigma2_i(b_j)   (Chen+ 2021)
- sigma2_i(b) = (Δ(b)^2)/12,   Δ(b) = 2*maxabs(w_i)/(2^b - 1)  (per-tensor symmetric quant)
- Column marginal b = softmax(phi) is learnable; mean-bit soft constraint.
- Learnable layer×bit bias Θ to tilt the OT plan during QAT.
- Forward uses mixture of quantized weights with STE so gradients flow to P and W.
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Iterable, Optional, Callable
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import qutil
from torch.utils.data import DataLoader
import diff_sinkhorn as dOT
import HAWQ2_fisher as h2
from  Dynamic_sinkhorn import estimate_trH_empirical_fisher, ChenAllocator
# ---------------------------
# STE quantization
# ---------------------------
class _RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x): return torch.round(x)
    @staticmethod
    def backward(ctx, g): return g

def ste_round(x: torch.Tensor) -> torch.Tensor:
    return _RoundSTE.apply(x)

def quantize_per_tensor_symmetric_ste(w: torch.Tensor, bits: int) -> torch.Tensor:
    if bits >= 32:  # no quantization
        return w
    qmax = (1 << (bits - 1)) - 1
    # Dynamic per-forward scale from maxabs; keep grad for scale; STE on rounding only
    scale = w.abs().amax() / (qmax + 1e-12)
    inv = 1.0 / (scale + 1e-12)
    z = ste_round(w * inv)
    z = torch.clamp(z, min=-qmax-1, max=qmax)
    return z * scale

# ---------------------------
# Differentiable Sinkhorn (log-domain, autograd-friendly)
# ---------------------------

def sinkhorn_log_diff(cost: torch.Tensor,
                      a: torch.Tensor,
                      b: torch.Tensor,
                      epsilon: float = 0.02,
                      iters: int = 200) -> torch.Tensor:
    """
    Compute transport plan P with gradients wrt cost and b.
    a,b are row/col marginals (sum=1).
    """
    dtype = cost.dtype
    K = -cost / epsilon
    f = torch.zeros(cost.size(0), device=cost.device, dtype=dtype)
    g = torch.zeros(cost.size(1), device=cost.device, dtype=dtype)

    def logsumexp(x, dim=-1):
        m, _ = torch.max(x, dim=dim, keepdim=True)
        return (m + torch.log(torch.sum(torch.exp(x - m), dim=dim, keepdim=True))).squeeze(dim)

    log_a = torch.log(a + 1e-40).to(dtype)
    log_b = torch.log(b + 1e-40).to(dtype)

    for _ in range(iters):
        f = log_a - logsumexp(K + g.unsqueeze(0), dim=1)
        g = log_b - logsumexp((K + f.unsqueeze(1)).transpose(0,1), dim=1)

    P = torch.exp(K + f.unsqueeze(1) + g.unsqueeze(0))
    P = P / (P.sum() + 1e-40)
    return P

# ---------------------------
# Differentiable allocator (learns Θ and φ)
# ---------------------------
@dataclass
class DynOTCfg:
    bits: List[int]
    epsilon: float = 0.02
    iters: int = 200
    avg_bits_target: float = 6.0     # soft budget target
    budget_weight: float = 1e-3      # weight for (E[bits]-target)^2
    entropy_weight: float = 0.0      # optional entropy on column b

class ChenDifferentiableAllocator(nn.Module):
    """
    Learnable:
      - theta: [L,B] bias (negative cost) to tilt assignments
      - phi:   [B]   column marginal b = softmax(phi)
    Fixed/updated buffers:
      - a: row marginal (size-weighted)
      - trH: layer Hessian trace estimates (update periodically)
      - wmax: layer max abs weights (update each step/epoch)
    Cost:
      C[i,j] = 0.5 * trH[i] * ( (2*wmax[i]/(2^b_j-1))^2 / 12 )
      Then effective cost: C_eff = C - theta
    """
    def __init__(self, layer_names: List[str], layer_sizes: List[int], bits: List[int], device: torch.device):
        super().__init__()
        self.layer_names = layer_names
        self.layer_sizes = layer_sizes
        self.bits = bits
        self.L, self.B = len(layer_names), len(bits)
        n = torch.tensor(layer_sizes, dtype=torch.float32, device=device)
        self.register_buffer("a", (n / n.sum()).detach())
        self.register_buffer("trH", torch.full((self.L,), 1.0, device=device))
        self.register_buffer("wmax", torch.full((self.L,), 1.0, device=device))
        # learnables
        self.theta = nn.Parameter(torch.zeros(self.L, self.B, device=device))
        self.phi = nn.Parameter(torch.zeros(self.B, device=device))

    @torch.no_grad()
    def update_trH(self, tr_map: Dict[str, float]):
        vals = [max(float(tr_map[nm]), 1e-12) for nm in self.layer_names]
        t = torch.tensor(vals, device=self.a.device, dtype=torch.float32)
        self.trH.copy_(t)

    @torch.no_grad()
    def update_wmax_from_model(self, model: nn.Module):
        vals = []
        for nm in self.layer_names:
            m = get_module_by_name(model, nm)
            assert hasattr(m, "weight") and m.weight is not None
            vals.append(float(m.weight.detach().abs().max().item()) + 1e-12)
        self.wmax.copy_(torch.tensor(vals, device=self.a.device, dtype=torch.float32))

    def column_marginal(self) -> torch.Tensor:
        return F.softmax(self.phi, dim=0)

    def build_cost(self) -> torch.Tensor:
        return deltaL_cost_matrix(self.trH.tolist(), self.wmax.tolist(), self.bits, self.a.device, dtype=torch.float32)

    def compute_P(self, cfg: DynOTCfg) -> Tuple[torch.Tensor, torch.Tensor]:
        C = self.build_cost()
        b = self.column_marginal()
        C_eff = C - self.theta
        P = sinkhorn_log_diff(C_eff, self.a, b, epsilon=cfg.epsilon, iters=cfg.iters)
        return P, b

    def budget_regularizer(self, P: torch.Tensor, cfg: DynOTCfg) -> torch.Tensor:
        bits_t = torch.tensor(self.bits, dtype=P.dtype, device=P.device)  # [B]
        Eb = torch.sum(self.a.unsqueeze(1) * P * bits_t.unsqueeze(0))     # expected bits (param-mass weighted)
        reg = cfg.budget_weight * (Eb - cfg.avg_bits_target) ** 2
        if cfg.entropy_weight > 0:
            b = self.column_marginal()
            reg = reg - cfg.entropy_weight * (-(b * (b.clamp_min(1e-12)).log()).sum())
        return reg

# ---------------------------
# Quant layers that read P[i,:] and mix quantized weights
# ---------------------------

class OTQuantLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, layer_index: int = 0,
                 allocator: ChenDifferentiableAllocator = None, bits: List[int] = None):
        super().__init__(in_features, out_features, bias=bias)
        self.layer_index = layer_index
        self.allocator = allocator
        self.bits = bits

    def forward(self, x):
        P, _ = self.allocator._current_P  # set before forward
        probs = P[self.layer_index]       # [B]
        mix = 0.0
        for j, b in enumerate(self.bits):
            Wq = quantize_per_tensor_symmetric_ste(self.weight, b)
            mix = mix + probs[j] * Wq
        return F.linear(x, mix, self.bias)

class OTQuantConv2d(nn.Conv2d):
    def __init__(self, *args, layer_index: int = 0,
                 allocator: ChenDifferentiableAllocator = None, bits: List[int] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer_index = layer_index
        self.allocator = allocator
        self.bits = bits

    def forward(self, x):
        P, _ = self.allocator._current_P
        probs = P[self.layer_index]
        mix = 0.0
        for j, b in enumerate(self.bits):
            Wq = quantize_per_tensor_symmetric_ste(self.weight, b)
            mix = mix + probs[j] * Wq
        return F.conv2d(x, mix, self.bias, self.stride, self.padding, self.dilation, self.groups)

def wrap_model_with_ot_quant(model: nn.Module,
                             allocator: ChenDifferentiableAllocator,
                             bits: List[int]) -> nn.Module:
    idx = 0
    def _replace(mod: nn.Module):
        nonlocal idx
        for name, child in list(mod.named_children()):
            if isinstance(child, nn.Conv2d):
                q = OTQuantConv2d(
                    child.in_channels, child.out_channels, kernel_size=child.kernel_size,
                    stride=child.stride, padding=child.padding, dilation=child.dilation,
                    groups=child.groups, bias=(child.bias is not None),
                    layer_index=idx, allocator=allocator, bits=bits
                )
                q.weight = child.weight
                q.bias = child.bias
                setattr(mod, name, q)
                idx += 1
            elif isinstance(child, nn.Linear):
                q = OTQuantLinear(
                    child.in_features, child.out_features, bias=(child.bias is not None),
                    layer_index=idx, allocator=allocator, bits=bits
                )
                q.weight = child.weight
                q.bias = child.bias
                setattr(mod, name, q)
                idx += 1
            else:
                _replace(child)
    _replace(model)
    return model


# (Optional) Hutchinson estimator sketch (heavier; not fully optimized)
def hutchinson_trH_step(model: nn.Module,
                        loss_callback: Callable[[nn.Module, Tuple, Dict], torch.Tensor],
                        batch) -> Dict[str, float]:
    """
    One-step Hutchinson trace estimate using hvp with a single Rademacher probe.
    Heavy for large models; provided as a reference.
    """
    model.zero_grad(set_to_none=True)
    args, kwargs = (batch if isinstance(batch, (list, tuple)) else (batch,)), {}
    loss = loss_callback(model, args, kwargs)
    grads = torch.autograd.grad(loss, [p for p in model.parameters() if p.requires_grad], create_graph=True)
    # build probe v with same shapes
    v_list = [torch.randint_like(g, low=0, high=2).float().mul_(2).sub_(1) for g in grads]
    # compute (Hv) via grad( grads·v )
    dot = sum([(g * v).sum() for g, v in zip(grads, v_list)])
    hvp = torch.autograd.grad(dot, [p for p in model.parameters() if p.requires_grad], retain_graph=False)
    # accumulate per layer: sum(H_ii) ≈ v ⊙ (Hv) summed over params
    traces = {}
    idx = 0
    for name, m in qutil.iter_quant_layers(model):
        layer_params = [p for p in m.parameters() if p.requires_grad]
        n_elems = sum(p.numel() for p in layer_params)
        # map hvp/grads slices accordingly
        layer_v = []
        layer_hv = []
        for p in layer_params:
            vp = v_list[idx]; hv = hvp[idx]; idx += 1
            layer_v.append(vp.reshape(-1))
            layer_hv.append(hv.reshape(-1))
        vcat = torch.cat(layer_v); hvcat = torch.cat(layer_hv)
        tr_est = torch.sum(vcat * hvcat).item()
        traces[name] = max(tr_est, 1e-12)
    return traces

# ---------------------------
# Training helper
# ---------------------------

def forward_with_dynamic_bits(model: nn.Module,
                              allocator: ChenDifferentiableAllocator,
                              cfg: DynOTCfg,
                              task_loss_callback: Callable[[nn.Module, Tuple, Dict], torch.Tensor],
                              batch) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # refresh wmax for sigma^2
    allocator.update_wmax_from_model(model)
    # compute P (keeps graph for backprop)
    allocator._current_P = allocator.compute_P(cfg)  # (P,b)
    # task forward
    args, kwargs = (batch if isinstance(batch, (list, tuple)) else (batch,)), {}
    task_loss = task_loss_callback(model, args, kwargs)  # scalar
    # budget reg
    P, _ = allocator._current_P
    budget_reg = allocator.budget_regularizer(P, cfg)
    total = task_loss + budget_reg
    return total, {"task_loss": task_loss, "budget_reg": budget_reg, "P": P.detach().cpu()}

def  sinkhorn_MCKP_allocate(base:nn.Module,
                              train_loader: DataLoader, 
                              device, bits: List[int], avg_bits: float,
                              steps: int = 50, lr: float = 1e-4,
                              chen: bool = False) -> Dict[str,int]:
    # meta
    names, sizes = [], []
    for name, m in qutil.iter_quant_layers(base):
        names.append(name); sizes.append(m.weight.numel())
    # allocator
    if chen:
        alloc = ChenDifferentiableAllocator(names, sizes, bits, device).to(device)
        # initial trH
        tr_map = estimate_trH_empirical_fisher(base, train_loader, device, batches=2)
        alloc.update_trH(tr_map)
    else:
        alloc = dOT.DifferentiableAllocator(names, sizes, bits, device).to(device)
        s_map = h2.empirical_fisher_sensitivity(base, train_loader, device, batches=2)
        alloc.update_sensitivity(s_map)
    # wrap model with mixture modules

    if chen:
        # just reuse mixture layers using diff allocator for STE forward;
        # we can still read P from Chen allocator by injecting alloc._P before forward
        pass
    model = wrap_model_with_ot_quant(base,alloc,bits).to(device)
    
    ce = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(list(model.parameters()) + list(alloc.parameters()), lr=lr)
    it = iter(train_loader)
    for step in range(steps):
        try: x,y = next(it)
        except StopIteration:
            it = iter(train_loader); x,y = next(it)
        x,y = x.to(device), y.to(device)
        opt.zero_grad()
        # refresh stats
        if chen: alloc.update_wmax_from_model(model)
        # compute P
        P = alloc.compute_P(eps=0.02, iters=150)
        alloc._P = P  # inject
        logits = model(x)
        loss = ce(logits, y)
        loss = loss + (alloc.budget_reg(P, avg_bits, weight=1e-3))
        loss.backward()
        opt.step()
        if step % 20 == 0:
            with torch.no_grad():
                bits_t = torch.tensor(bits, device=device, dtype=P.dtype)
                Eb = (alloc.a.unsqueeze(1)*P*bits_t.unsqueeze(0)).sum().item()
            print(f"[dynamic {'Chen' if chen else 'Diff'}] step {step} Eb≈{Eb:.3f}")

    # Harden: take argmax per row
    with torch.no_grad():
        P = alloc.compute_P(eps=0.02, iters=200)
        idxs = P.argmax(dim=1).tolist()
    assignment = {names[i]: bits[idxs[i]] for i in range(len(names))}
    return assignment


# ---------------------------
# Minimal example (classification + FakeData)
# ---------------------------

if __name__ == "__main__":
    try:
        from torchvision import models, datasets, transforms
        from torch.utils.data import DataLoader
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Data
        tfm = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])
        ds = datasets.FakeData(size=128, image_size=(3,224,224), num_classes=10, transform=tfm)
        dl = DataLoader(ds, batch_size=16, shuffle=True)

        # Model
        model = models.resnet18(num_classes=10).to(device)

        # Task loss callback (CE)
        ce = nn.CrossEntropyLoss()
        def task_loss_cb(m, args, kwargs):
            x, y = args[0].to(device), args[1].to(device)
            logits = m(x)
            return ce(logits, y)

        # Layer meta
        layer_names, layer_sizes = [], []
        for name, mod in qutil.iter_quant_layers(model):
            layer_names.append(name)
            layer_sizes.append(qutil.param_count(mod))

        bits = [2,4,6,8]
        alloc = ChenDifferentiableAllocator(layer_names, layer_sizes, bits, device).to(device)
        model = wrap_model_with_ot_quant(model, alloc, bits).to(device)

        # Initial tr(H) by empirical Fisher (few batches)
        tr_map = estimate_trH_empirical_fisher(model, dl, task_loss_cb, device, batches=2)
        alloc.update_trH(tr_map)

        cfg = DynOTCfg(bits=bits, epsilon=0.02, iters=150, avg_bits_target=6.0, budget_weight=1e-3)
        opt = torch.optim.Adam(list(model.parameters()) + list(alloc.parameters()), lr=1e-4)

        # Train loop (short demo)
        for epoch in range(2):
            for i, batch in enumerate(dl):
                # (Optional) refresh tr(H) occasionally
                if i % 10 == 0:
                    tr_map = estimate_trH_empirical_fisher(model, [batch], task_loss_cb, device, batches=1)
                    alloc.update_trH(tr_map)

                opt.zero_grad()
                total, logs = forward_with_dynamic_bits(model, alloc, cfg, task_loss_cb, batch)
                total.backward()
                opt.step()

            with torch.no_grad():
                P, b = alloc.compute_P(cfg)
                Eb = (alloc.a.unsqueeze(1) * P * torch.tensor(bits, device=device).unsqueeze(0)).sum()
                print(f"[epoch {epoch}] mean_bits≈{Eb.item():.3f}  budget_reg={logs['budget_reg'].item():.2e}")

    except Exception as e:
        print("Demo needs torchvision; install it to run the example. Error:", e)
