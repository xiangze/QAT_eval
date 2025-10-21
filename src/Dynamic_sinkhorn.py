import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Iterable, Optional, Callable
from dataclasses import dataclass
from torch.utils.data import DataLoader
import torch.nn.functional as F
import HAWQ2_fisher as h2
import util
#import diff_sinkhorn as dOT

class _RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x): return torch.round(x)
    @staticmethod
    def backward(ctx, g): return g

def ste_round(x: torch.Tensor) -> torch.Tensor: return _RoundSTE.apply(x)

def quantize_per_tensor_symmetric_ste(w: torch.Tensor, bits: int) -> torch.Tensor:
    if bits >= 32: return w
    qmax = (1 << (bits - 1)) - 1
    scale = w.abs().amax() / (qmax + 1e-12)
    inv = 1.0 / (scale + 1e-12)
    z = ste_round(w * inv)
    z = torch.clamp(z, min=-qmax-1, max=qmax)
    return z * scale

def sinkhorn_log_diff(cost, a, b, epsilon=0.02, iters=200):
    K = -cost/epsilon
    f = torch.zeros(cost.size(0), device=cost.device, dtype=cost.dtype)
    g = torch.zeros(cost.size(1), device=cost.device, dtype=cost.dtype)
    def lse(x, dim=-1):
        m,_ = torch.max(x, dim=dim, keepdim=True)
        return (m + torch.log(torch.sum(torch.exp(x-m), dim=dim, keepdim=True))).squeeze(dim)
    log_a = torch.log(a+1e-40).to(cost.dtype); log_b = torch.log(b+1e-40).to(cost.dtype)
    for _ in range(iters):
        f = log_a - lse(K + g.unsqueeze(0), dim=1)
        g = log_b - lse((K + f.unsqueeze(1)).transpose(0,1), dim=1)
    P = torch.exp(K + f.unsqueeze(1) + g.unsqueeze(0))
    return P / (P.sum() + 1e-40)

@dataclass
class DynOTCfg:
    bits: List[int]
    epsilon: float = 0.02
    iters: int = 200
    avg_bits_target: float = 6.0     # soft budget target
    budget_weight: float = 1e-3      # weight for (E[bits]-target)^2
    entropy_weight: float = 0.0      # optional entropy on column b

class DiffAllocator(nn.Module):
    def __init__(self, layer_names: List[str], layer_sizes: List[int], bits: List[int], device):
        super().__init__()
        self.layer_names = layer_names; self.layer_sizes = layer_sizes; self.bits = bits
        self.L, self.B = len(layer_names), len(bits)
        n = torch.tensor(layer_sizes, dtype=torch.float32, device=device)
        self.register_buffer("a", (n/n.sum()).detach())
        self.theta = nn.Parameter(torch.zeros(self.L, self.B, device=device))
        self.phi = nn.Parameter(torch.zeros(self.B, device=device))
        self.register_buffer("sens", torch.full((self.L,), 1.0/self.L, device=device))
        self.register_buffer("err", torch.tensor([2.0**(-2*b) for b in bits], dtype=torch.float32, device=device))

    @torch.no_grad()
    def update_sensitivity(self, s_map: Dict[str,float]):
        s = torch.tensor([s_map[nm] for nm in self.layer_names], dtype=torch.float32, device=self.a.device)
        s = s / (s.sum() + 1e-12); self.sens.copy_(s)

    def build_cost(self):
        n = torch.tensor(self.layer_sizes, dtype=torch.float32, device=self.a.device)
        return (n*self.sens).unsqueeze(1)*self.err.unsqueeze(0)

    def b(self): return F.softmax(self.phi, dim=0)

    def compute_P(self, eps=0.02, iters=200):
        return sinkhorn_log_diff(self.build_cost() - self.theta, self.a, self.b(), eps, iters)

    def budget_reg(self, P, target_avg, weight):
        bits_t = torch.tensor(self.bits, dtype=P.dtype, device=P.device)
        Eb = torch.sum(self.a.unsqueeze(1) * P * bits_t.unsqueeze(0))
        return weight * (Eb - target_avg)**2

class DifferentiableAllocator(DiffAllocator):
    """
    Learnable:
      - theta: [L,B] bias that tilts assignment (acts like negative cost)
      - phi:   [B]   controls column marginal b = softmax(phi)
    Fixed:
      - a: row marginal (size-weighted)
      - bits: candidate bit-widths
      - sens, n: build cost C = n*s*err(bits)
    """
    def __init__(self, layer_names: List[str], layer_sizes: List[int], bits: List[int], device: torch.device):
        super().__init__(layer_names,layer_sizes,bits,device)
        # self.layer_names = layer_names
        # self.layer_sizes = layer_sizes
        # self.bits = bits
        # self.L = len(layer_names)
        # self.B = len(bits)
        # n = torch.tensor(layer_sizes, dtype=torch.float32, device=device)
        # self.register_buffer("a", (n / n.sum()).detach())  # row marginal
        # # learnable params
        # self.theta = nn.Parameter(torch.zeros(self.L, self.B, device=device))
        # self.phi = nn.Parameter(torch.zeros(self.B, device=device))  # b = softmax(phi)
        # # buffers updated externally
        # self.register_buffer("sens", torch.full((self.L,), 1.0/self.L, device=device))
        # self.register_buffer("err", torch.tensor([2.0**(-2*b) for b in bits], dtype=torch.float32, device=device))

    def column_marginal(self) -> torch.Tensor:
        return F.softmax(self.phi, dim=0)  # [B]
    
    def compute_P(self, eps=0.02, iters=200):
        b = self.column_marginal()        # [B], learnable
        # bias theta enters as negative cost (larger theta -> prefer that cell)
        C_eff = self.build_cost()  - self.theta            # differentiable wrt theta
        return  sinkhorn_log_diff(C_eff, self.a, b, epsilon=eps, iters=iters)  # [L,B]

    def budget_regularizer(self, P: torch.Tensor, cfg: DynOTCfg) -> torch.Tensor:
        # Expected bits over "param mass" (use a as row weights)
        bits_t = torch.tensor(self.bits, dtype=P.dtype, device=P.device)  # [B]
        # per-layer expected bits: P[i]·b_vec, then weight by a[i]
        Eb = torch.sum(self.a.unsqueeze(1) * P * bits_t.unsqueeze(0))
        reg = cfg.budget_weight * (Eb - cfg.avg_bits_target)**2
        if cfg.entropy_weight > 0:
            b = self.column_marginal()
            reg = reg - cfg.entropy_weight * (-(b * (b.clamp_min(1e-12)).log()).sum())
        return reg

class ChenDifferentiableAllocator(DiffAllocator):
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
        super().__init__(layer_names,layer_sizes,bits,device)
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

    def build_cost(self) -> torch.Tensor:
        return deltaL_cost_matrix(self.trH.tolist(), self.wmax.tolist(), self.bits, self.a.device, dtype=torch.float32)

    def compute_P(self, cfg: DynOTCfg) -> Tuple[torch.Tensor, torch.Tensor]:
        b = self.column_marginal()
        C_eff = self.build_cost() - self.theta
        P = sinkhorn_log_diff(C_eff, self.a, b, epsilon=cfg.epsilon, iters=cfg.iters)
        return P, b

    def budget_reg(self, P: torch.Tensor, cfg: DynOTCfg) -> torch.Tensor:
        bits_t = torch.tensor(self.bits, dtype=P.dtype, device=P.device)  # [B]
        Eb = torch.sum(self.a.unsqueeze(1) * P * bits_t.unsqueeze(0))     # expected bits (param-mass weighted)
        reg = cfg.budget_weight * (Eb - cfg.avg_bits_target) ** 2
        if cfg.entropy_weight > 0:
            b = self.column_marginal()
            reg = reg - cfg.entropy_weight * (-(b * (b.clamp_min(1e-12)).log()).sum())
        return reg

# ---------------------------
# Chen cost (MCKP quadratic) builder
# ---------------------------
def deltaL_cost_matrix(trH: List[float],   # tr(H_i) per layer
                       wmax: List[float],  # max|w_i| per layer
                       bits: List[int],
                       device: torch.device,
                       dtype=torch.float32) -> torch.Tensor:
    """
    C[i,j] = 0.5 * trH[i] * ((2*wmax[i]/(2^b-1))**2 / 12)
    """
    L, B = len(trH), len(bits)
    tr = torch.tensor(trH, device=device, dtype=dtype).unsqueeze(1)   # [L,1]
    w = torch.tensor(wmax, device=device, dtype=dtype).unsqueeze(1)   # [L,1]
    denom = torch.tensor([float((1 << b) - 1) for b in bits], device=device, dtype=dtype).unsqueeze(0)  # [1,B]
    delta = (2.0 * w) / denom  # [L,B]
    sigma2 = (delta * delta) / 12.0
    C = 0.5 * tr * sigma2
    return C  # [L,B]

class OTQLinear(nn.Linear):
    def __init__(self, in_f, out_f, bias=True, idx=0, alloc: DiffAllocator=None, bits=None):
        super().__init__(in_f, out_f, bias=bias); self.idx=idx; self.alloc=alloc; self.bits=bits
    def forward(self, x):
        P = self.alloc._P
        probs = P[self.idx]
        mix = 0.0
        for j,b in enumerate(self.bits):
            Wq = quantize_per_tensor_symmetric_ste(self.weight, b)
            mix = mix + probs[j]*Wq
        return F.linear(x, mix, self.bias)

class OTQConv2d(nn.Conv2d):
    def __init__(self, *args, idx=0, alloc: DiffAllocator=None, bits=None, **kw):
        super().__init__(*args, **kw); self.idx=idx; self.alloc=alloc; self.bits=bits
    def forward(self, x):
        P = self.alloc._P
        probs = P[self.idx]
        mix = 0.0
        for j,b in enumerate(self.bits):
            Wq = quantize_per_tensor_symmetric_ste(self.weight, b)
            mix = mix + probs[j]*Wq
        return F.conv2d(x, mix, self.bias, self.stride, self.padding, self.dilation, self.groups)


# Chen cost allocator (dynamic)
class ChenAllocator(DiffAllocator):
    def __init__(self, layer_names: List[str], layer_sizes: List[int], bits: List[int], device):
        super().__init__(layer_names,layer_sizes,bits,device)
        self.layer_names = layer_names; self.layer_sizes = layer_sizes; self.bits=bits
        self.L, self.B = len(layer_names), len(bits)
        n = torch.tensor(layer_sizes, dtype=torch.float32, device=device)
        self.register_buffer("a", (n/n.sum()).detach())
        self.register_buffer("trH", torch.full((self.L,), 1.0, device=device))
        self.register_buffer("wmax", torch.full((self.L,), 1.0, device=device))
        self.theta = nn.Parameter(torch.zeros(self.L, self.B, device=device))
        self.phi   = nn.Parameter(torch.zeros(self.B, device=device))

    @torch.no_grad()
    def update_trH(self, tr_map: Dict[str,float]):
        vals = [max(float(tr_map[nm]),1e-12) for nm in self.layer_names]
        self.trH.copy_(torch.tensor(vals, device=self.a.device, dtype=torch.float32))

    @torch.no_grad()
    def update_wmax_from_model(self, model: nn.Module):
        vals=[]
        for nm in self.layer_names:
            # traverse
            mod = model
            for tk in nm.split("."):
                if tk.isdigit(): mod=mod[int(tk)]
                else: mod=getattr(mod, tk)
            vals.append(float(mod.weight.detach().abs().max().item())+1e-12)
        self.wmax.copy_(torch.tensor(vals, device=self.a.device, dtype=torch.float32))

    def b(self): return F.softmax(self.phi, dim=0)

    def build_cost(self):
        L,B=self.L,self.B
        tr = self.trH.unsqueeze(1)                   # [L,1]
        w  = self.wmax.unsqueeze(1)                  # [L,1]
        denom = torch.tensor([(1<<b)-1 for b in self.bits], device=self.a.device, dtype=torch.float32).unsqueeze(0)  # [1,B]
        delta = 2.0*w/denom
        sigma2 = (delta*delta)/12.0
        return 0.5*tr*sigma2

    def compute_P(self, eps=0.02, iters=200):
        return sinkhorn_log_diff(self.build_cost() - self.theta, self.a, self.b(), eps, iters)

    def budget_reg(self, P, target_avg, weight):
        bits_t = torch.tensor(self.bits, dtype=P.dtype, device=P.device)
        Eb = torch.sum(self.a.unsqueeze(1)*P*bits_t.unsqueeze(0))
        return weight * (Eb - target_avg)**2

def wrap_with_mixture_modules(model: nn.Module, alloc: DiffAllocator, bits: List[int]) -> nn.Module:
    idx=0
    def _rep(mod: nn.Module):
        nonlocal idx
        for name, ch in list(mod.named_children()):
            if isinstance(ch, nn.Conv2d):
                q = OTQConv2d(ch.in_channels, ch.out_channels, kernel_size=ch.kernel_size,
                              stride=ch.stride, padding=ch.padding, dilation=ch.dilation,
                              groups=ch.groups, bias=(ch.bias is not None),
                              idx=idx, alloc=alloc, bits=bits)
                q.weight = ch.weight; q.bias=ch.bias
                setattr(mod, name, q); idx+=1
            elif isinstance(ch, nn.Linear):
                q = OTQLinear(ch.in_features, ch.out_features, bias=(ch.bias is not None),
                              idx=idx, alloc=alloc, bits=bits)
                q.weight = ch.weight; q.bias=ch.bias
                setattr(mod, name, q); idx+=1
            else:
                _rep(ch)
    _rep(model); return model

# Short QAT to learn allocator then harden
def dynamic_sinkhorn_allocate(base:nn.Module,
                              train_loader: DataLoader, 
                              device, bits: List[int], avg_bits: float,
                              steps: int = 50, lr: float = 1e-4,
                              chen: bool = False,
                              MCKP=False) -> Dict[str,int]:
    names, sizes = [],[] #list(zip( [name, m.weight.numel()] for name, m in util.iter_quant_layers(base)))
    for name, m in util.iter_quant_layers(base):
        names.append(name);  
        sizes.append(m.weight.numel())

    # allocator
    if chen:
        if(MCKP):
            alloc = ChenDifferentiableAllocator(names, sizes, bits, device).to(device)
        else:
            alloc = ChenAllocator(names, sizes, bits, device).to(device)
        # initial trH
        alloc.update_trH(estimate_trH_empirical_fisher(base, train_loader, device, batches=2))
        # just reuse mixture layers using diff allocator for STE forward;
        # we can still read P from Chen allocator by injecting alloc._P before forward
    else:
        if(MCKP):
            alloc = DifferentiableAllocator(names, sizes, bits, device).to(device)
        else:
            alloc = DiffAllocator(names, sizes, bits, device).to(device)
        alloc.update_sensitivity(h2.empirical_fisher_sensitivity(base, train_loader, device, batches=2))

    model = base.to(device)
    model = wrap_with_mixture_modules(model, alloc, bits).to(device)

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
        P = alloc.compute_P(eps=0.02, iters=150)
        if(P is None):
            print(f"P is None, alloc is {type(alloc)}");exit()
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

# Empirical Fisher trace (sum of grad^2 per layer)
def estimate_trH_empirical_fisher(model: nn.Module, loader: DataLoader, device, batches=1) -> Dict[str,float]:
    ce = nn.CrossEntropyLoss()
    model.train()
    sens = {name: 0.0 for name,_ in util.iter_quant_layers(model)}
    counts = {k:0 for k in sens.keys()}
    it = iter(loader)
    for _ in range(batches):
        try: x,y = next(it)
        except StopIteration: break
        x,y = x.to(device), y.to(device)
        for p in model.parameters():
            if p.grad is not None: p.grad.zero_()
        logits = model(x); loss = ce(logits, y)
        loss.backward()
        for name, m in util.iter_quant_layers(model):
            g2 = 0.0
            for p in m.parameters():
                if p.grad is None: continue
                g2 += torch.sum(p.grad.detach()**2).item()
            sens[name] += g2; counts[name]+=1
    out={}
    for k,v in sens.items():
        if counts[k]>0: v/=counts[k]
        out[k]=max(v,1e-12)
    return out
