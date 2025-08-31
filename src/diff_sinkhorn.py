"""
Differentiable Sinkhorn OT for dynamic mixed-precision bit allocation (QAT-ready).
- Backprop through Sinkhorn to learn bit distribution and layer-bit biases.
- Soft budget via (E[bits]-B_avg)^2 penalty.
- Mixture-of-quantized-weights with STE so gradients reach both P and W.
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Iterable, Optional, Callable
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import util

# ---------------------------
# STE helpers
# ---------------------------

class _RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)
    @staticmethod
    def backward(ctx, g):
        return g  # straight-through

def ste_round(x: torch.Tensor) -> torch.Tensor:
    return _RoundSTE.apply(x)

def quantize_per_tensor_symmetric_ste(w: torch.Tensor, bits: int) -> torch.Tensor:
    if bits >= 32:
        return w
    qmax = (1 << (bits - 1)) - 1
    # scale の勾配は流しにくいので detatch はしない（STE のみ丸めに適用）
    scale = w.abs().amax() / (qmax + 1e-12)
    inv = 1.0 / (scale + 1e-12)
    z = ste_round(w * inv)
    z = torch.clamp(z, min=-qmax-1, max=qmax)
    return z * scale

# ---------------------------
# Differentiable Sinkhorn (log-domain)
# ---------------------------

def sinkhorn_log_diff(
    cost: torch.Tensor,   # [L,B], differentiable
    a: torch.Tensor,      # [L], sums to 1 (row marginal, const)
    b: torch.Tensor,      # [B], sums to 1 (col marginal, learnable via softmax)
    epsilon: float = 0.02,
    iters: int = 200,
) -> torch.Tensor:
    """
    Returns transport plan P with gradients w.r.t. cost and b.
    K = exp((Theta - C)/eps) 形式にしたければ cost 側で調整すること。
    """
    K = -cost / epsilon  # [L,B]
    f = torch.zeros(cost.size(0), device=cost.device, dtype=cost.dtype)
    g = torch.zeros(cost.size(1), device=cost.device, dtype=cost.dtype)

    def logsumexp(x, dim=-1):
        m, _ = torch.max(x, dim=dim, keepdim=True)
        return (m + torch.log(torch.sum(torch.exp(x - m), dim=dim, keepdim=True))).squeeze(dim)

    log_a = torch.log(a + 1e-40).to(cost.dtype)
    log_b = torch.log(b + 1e-40).to(cost.dtype)

    for _ in range(iters):
        f = log_a - logsumexp(K + g.unsqueeze(0), dim=1)
        g = log_b - logsumexp((K + f.unsqueeze(1)).transpose(0,1), dim=1)

    P = torch.exp(K + f.unsqueeze(1) + g.unsqueeze(0))
    P = P / (P.sum() + 1e-40)
    return P

# ---------------------------
# Differentiable allocator (learns column b and bias Θ)
# ---------------------------

@dataclass
class DynOTCfg:
    bits: List[int]
    epsilon: float = 0.02
    iters: int = 200
    avg_bits_target: float = 6.0
    budget_weight: float = 1e-3  # weight for (E[bits]-avg)^2
    entropy_weight: float = 0.0  # optional: encourage high-entropy b

class DifferentiableAllocator(nn.Module):
    """
    Learnable:
      - theta: [L,B] bias that tilts assignment (acts like negative cost)
      - phi:   [B]   controls column marginal b = softmax(phi)
    Fixed:
      - a: row marginal (size-weighted)
      - bits: candidate bit-widths
      - sens, n: build cost C = n*s*err(bits)
    """
    def __init__(self,
                 layer_names: List[str],
                 layer_sizes: List[int],
                 bits: List[int],
                 device: torch.device):
        super().__init__()
        self.layer_names = layer_names
        self.layer_sizes = layer_sizes
        self.bits = bits
        self.L = len(layer_names)
        self.B = len(bits)
        n = torch.tensor(layer_sizes, dtype=torch.float32, device=device)
        self.register_buffer("a", (n / n.sum()).detach())  # row marginal
        # learnable params
        self.theta = nn.Parameter(torch.zeros(self.L, self.B, device=device))
        self.phi = nn.Parameter(torch.zeros(self.B, device=device))  # b = softmax(phi)
        # buffers updated externally
        self.register_buffer("sens", torch.full((self.L,), 1.0/self.L, device=device))
        self.register_buffer("err", torch.tensor([2.0**(-2*b) for b in bits], dtype=torch.float32, device=device))

    @torch.no_grad()
    def update_sensitivity(self, sens_map: Dict[str, float]):
        s = torch.tensor([sens_map[nm] for nm in self.layer_names], dtype=torch.float32, device=self.a.device)
        s = s / (s.sum() + 1e-12)
        self.sens.copy_(s)

    def build_cost(self) -> torch.Tensor:
        # C[i,j] = n_i * s_i * err(b_j)  (scaled; gradient flows via theta in compute_P)
        n = torch.tensor(self.layer_sizes, dtype=torch.float32, device=self.a.device)
        C = (n * self.sens).unsqueeze(1) * self.err.unsqueeze(0)  # [L,B]
        return C

    def column_marginal(self) -> torch.Tensor:
        return F.softmax(self.phi, dim=0)  # [B]

    def compute_P(self, cfg: DynOTCfg) -> Tuple[torch.Tensor, torch.Tensor]:
        C = self.build_cost()  # [L,B], no params
        b = self.column_marginal()        # [B], learnable
        # bias theta enters as negative cost (larger theta -> prefer that cell)
        C_eff = C - self.theta            # differentiable wrt theta
        P = sinkhorn_log_diff(C_eff, self.a, b, epsilon=cfg.epsilon, iters=cfg.iters)  # [L,B]
        return P, b

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

# ---------------------------
# Quant wrappers (don’t mutate parameters; use mixture in forward)
# ---------------------------

class OTQuantLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, layer_index: int = 0,
                 allocator: DifferentiableAllocator = None, bits: List[int] = None):
        super().__init__(in_features, out_features, bias=bias)
        self.layer_index = layer_index
        self.allocator = allocator
        self.bits = bits

    def forward(self, x):
        assert self.allocator is not None and self.bits is not None, "Allocator not set"
        # get row probs P_i,: with gradients
        P, _ = self.allocator._current_P  # injected before forward
        probs = P[self.layer_index]       # [B]
        # build mixture of quantized weights
        W = self.weight
        mix = 0.0
        for j, b in enumerate(self.bits):
            Wq = quantize_per_tensor_symmetric_ste(W, b)
            mix = mix + probs[j] * Wq
        return F.linear(x, mix, self.bias)

class OTQuantConv2d(nn.Conv2d):
    def __init__(self, *args, layer_index: int = 0,
                 allocator: DifferentiableAllocator = None, bits: List[int] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer_index = layer_index
        self.allocator = allocator
        self.bits = bits

    def forward(self, x):
        assert self.allocator is not None and self.bits is not None, "Allocator not set"
        P, _ = self.allocator._current_P
        probs = P[self.layer_index]
        W = self.weight
        mix = 0.0
        for j, b in enumerate(self.bits):
            Wq = quantize_per_tensor_symmetric_ste(W, b)
            mix = mix + probs[j] * Wq
        return F.conv2d(x, mix, self.bias, self.stride, self.padding, self.dilation, self.groups)

# ---------------------------
# Builder: wrap a model’s Conv/Linear with OT-quant modules
# ---------------------------

def wrap_model_with_ot_quant(model: nn.Module, allocator: DifferentiableAllocator, bits: List[int]) -> nn.Module:
    """
    Replace Conv2d/Linear with OTQuant* modules preserving weights/bias.
    The allocator will be injected to each layer.
    """
    layer_names = [name for name, m in util.iter_quant_layers(model) ]

    # recursive replace
    idx = 0
    def _replace(mod: nn.Module, prefix: str = ""):
        nonlocal idx
        for name, child in list(mod.named_children()):
            full = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Conv2d):
                q = OTQuantConv2d(
                    child.in_channels, child.out_channels, kernel_size=child.kernel_size,
                    stride=child.stride, padding=child.padding, dilation=child.dilation,
                    groups=child.groups, bias=(child.bias is not None),
                    layer_index=idx, allocator=allocator, bits=bits
                )
                # copy params
                q.weight = child.weight
                q.bias = child.bias
                setattr(mod, name, q)
                idx += 1
            elif isinstance(child, nn.Linear):
                q = OTQuantLinear(child.in_features, child.out_features,
                                  bias=(child.bias is not None),
                                  layer_index=idx, allocator=allocator, bits=bits)
                q.weight = child.weight
                q.bias = child.bias
                setattr(mod, name, q)
                idx += 1
            else:
                _replace(child, full)
    _replace(model)
    return model

# ---------------------------
# Sensitivity (plug-in): empirical Fisher of mixed objective (optional)
# ---------------------------

def fisher_sensitivity_step(
    model: nn.Module,
    batch,
    device: torch.device,
    loss_callback: Callable[[nn.Module, Tuple, Dict], Dict[str, torch.Tensor]],
    alpha: float = 1.0,
) -> Dict[str, float]:
    """1ステップで grad^2 を測る（移動平均に載せて使う想定）"""
    model.train()
    if isinstance(batch, (list, tuple)):
        args, kwargs = batch, {}
    elif isinstance(batch, dict):
        args, kwargs = (), batch
    else:
        args, kwargs = (batch,), {}

    # zero grad
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()

    outs = loss_callback(model, args, kwargs)  # {"LA","LF"}
    mixed = alpha * outs.get("LA", 0.0) + outs.get("LF", 0.0)
    mixed.backward()

    # collect
    sens = {}
    for name, m in util.iter_quant_layers(model):
        g2_sum, n = 0.0, 0
        for p in m.parameters():
            if p.grad is None: continue
            g2_sum += torch.sum(p.grad.detach()**2).item()
            n += p.numel()
        sens[name] = (g2_sum / max(1, n))
    # normalize
    ssum = sum(sens.values()) + 1e-12
    for k in sens.keys():
        sens[k] /= ssum
    # zero
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()
    return sens

# ---------------------------
# Training step helper
# ---------------------------

@dataclass
class TrainCfg:
    epsilon: float = 0.02
    iters: int = 200
    avg_bits: float = 6.0
    budget_w: float = 1e-3
    entropy_w: float = 0.0
    alpha: float = 1.0  # LA vs LF

def forward_with_dynamic_bits(
    model: nn.Module,
    allocator: DifferentiableAllocator,
    dyn_cfg: DynOTCfg,
    loss_callback: Callable[[nn.Module, Tuple, Dict], Dict[str, torch.Tensor]],
    batch,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    1) P,b を計算して各量子化層に注入
    2) モデル前向き（各層が P 行を参照して混合量子化）
    3) 予算正則化を合算
    """
    # compute P (keeps graph)
    allocator._current_P = allocator.compute_P(dyn_cfg)  # (P,b)
    # forward with loss_callback
    if isinstance(batch, (list, tuple)):
        args, kwargs = batch, {}
    elif isinstance(batch, dict):
        args, kwargs = (), batch
    else:
        args, kwargs = (batch,), {}
    outs = loss_callback(model, args, kwargs)  # {"LA","LF",...}
    task_loss = outs.get("LA", 0.0) + outs.get("LF", 0.0)
    budget_reg = allocator.budget_regularizer(allocator._current_P[0], dyn_cfg)
    total = task_loss + budget_reg
    outs["budget_reg"] = budget_reg
    outs["total_loss"] = total
    return total, outs

if __name__ =="__main__":
    import torch
    import torch.nn as nn
    from torchvision import models, datasets, transforms
    from torch.utils.data import DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tfm = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])
    ds = datasets.FakeData(size=128, image_size=(3,224,224), num_classes=10, transform=tfm)
    dl = DataLoader(ds, batch_size=16, shuffle=True)

    # モデルと損失（LA のみの簡易版。必要なら LF=クリティカル目的を加えてください）
    model = models.resnet18(num_classes=10).to(device)
    ce = nn.CrossEntropyLoss()
    def loss_cb(m, args, kwargs):
        x, y = args[0].to(device), args[1].to(device)
        logits = m(x)
        return {"LA": ce(logits, y), "LF": torch.tensor(0.0, device=device)}

    # レイヤ情報
    layer_names, layer_sizes = [], []
    for name, mod in model.named_modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear)):
            layer_names.append(name)
            layer_sizes.append(sum(p.numel() for p in mod.parameters() if p.requires_grad))

    bits = [2,4,6,8]
    alloc = DifferentiableAllocator(layer_names, layer_sizes, bits, device).to(device)

    # モデルを量子化ラッパに置換（学習中は混合量子化で forward）
    model = wrap_model_with_ot_quant(model, alloc, bits).to(device)

    # 設定
    dyn_cfg = DynOTCfg(bits=bits, epsilon=0.02, iters=150,
                    avg_bits_target=6.0, budget_weight=1e-3, entropy_weight=0.0)
    opt = torch.optim.Adam(list(model.parameters()) + list(alloc.parameters()), lr=1e-4)

    # 簡易トレインループ
    ma_sens = {nm: 1.0/len(layer_names) for nm in layer_names}  # 感度移動平均
    beta = 0.9

    for epoch in range(2):
        for batch in dl:
            # 1) Fisher感度の簡易更新（任意の頻度でOK）
            sens_now = fisher_sensitivity_step(model, batch, device, loss_cb, alpha=1.0)
            ma_sens={beta*ma_sens[k] + (1-beta)*sens_now[k] for k in ma_sens.keys()}
            alloc.update_sensitivity(ma_sens)

            # 2) 前向き（Sinkhorn→混合量子化→損失＋予算正則化）
            opt.zero_grad()
            total, outs = forward_with_dynamic_bits(model, alloc, dyn_cfg, loss_cb, batch)
            total.backward()
            opt.step()

        # ログ（平均ビットの推移など）
        with torch.no_grad():
            P, b = alloc.compute_P(dyn_cfg)
            Eb = (alloc.a.unsqueeze(1) * P * torch.tensor(bits, device=device).unsqueeze(0)).sum()
            print(f"[epoch {epoch}] mean_bits≈{Eb.item():.3f}, "
                f"budget_reg={outs['budget_reg'].item():.2e}")
