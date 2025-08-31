"""
Sinkhorn-OT mixed-precision assignment with Fisher-aware critical-category objectives (HAWQ-V2代替).
- Critical logit/label transformation (Eq. (1) in paper)
- Sensitivity s_i = E[ || d/dw_i ( alpha * L_A + L_F ) ||_2^2 ]  (empirical Fisher; Eq. (8))
- OT/Sinkhorn with size-weighted marginals to approximate ILP budget (Eq. (11))
- Size-aware greedy rounding to hit per-bit parameter capacities
- Fisher-trace regularization term for QAT (Eq. (12))
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Optional, Callable
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------
# Critical-category logit/label transform (classification-friendly)
#   For detection/DETR, supply your own loss callback using same idea.
# -------------------------------
@dataclass
class CriticalTransformCfg:
    n_classes: int                  # 元のクラス数 (背景なし)
    critical_class_ids: List[int]   # 0-based indices of critical classes
    include_background: bool = False  # 分類では通常False。DETRならTrueで背景保持を想定。

def critical_transform_logits_labels(
    logits: torch.Tensor,  # [B, n_classes] (分類). DETR等は各クエリで同様に適用可
    labels: torch.Tensor,  # [B]
    cfg: CriticalTransformCfg,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    式(1)に対応：クリティカルmクラスを保持、非クリティカルは "others" に折りたたみ。
    背景クラスは必要なら最後尾に回す。
    """
    n = cfg.n_classes
    crit = sorted(cfg.critical_class_ids)
    m = len(crit)
    device = logits.device

    # マッピング: old -> new
    # new classes: [critical(0..m-1), others(m), (optional) background(m+1)]
    idx_map = torch.full((n,), m, dtype=torch.long, device=device)  # default -> others
    for new_j, old_j in enumerate(crit):
        idx_map[old_j] = new_j

    # logits 変換
    # others = max over non-critical
    others_mask = torch.ones(n, dtype=torch.bool, device=device)
    others_mask[crit] = False
    crit_mask = ~others_mask

    # gather critical logits
    crit_logits = logits[:, crit_mask]  # [B, m]
    # others as max over non-critical
    others_logit, _ = torch.max(logits[:, others_mask], dim=1, keepdim=True)  # [B,1]

    if cfg.include_background:
        # 背景を仮にクラスnとして持っている場合は別処理が必要
        # ここでは分類想定のため背景は未使用
        new_logits = torch.cat([crit_logits, others_logit], dim=1)
    else:
        new_logits = torch.cat([crit_logits, others_logit], dim=1)  # [B, m+1]

    # labels 変換
    new_labels = idx_map[labels]  # othersはm
    return new_logits, new_labels

# -------------------------------
# Sensitivity: Fisher-diag for (alpha*LA + LF)
# -------------------------------

def _zero_grad(model: nn.Module):
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()

@torch.no_grad()
def _detach_all(model: nn.Module):
    for p in model.parameters():
        p.detach_()

def fisher_sensitivity_mixed_objective(
    model: nn.Module,
    dataloader,
    device: torch.device,
    loss_callback: Callable[[nn.Module, Tuple, Dict], Dict[str, torch.Tensor]],
    # loss_callback must return dict with keys: "LA", "LF"
    alpha: float = 1.0,
    batches: int = 1,
    exclude_layer_name_contains: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    s_i = E_batch[ mean( grad_i(alpha*LA + LF)^2 ) ] を層毎に推定。
    """
    model.train()
    sens = {name: 0.0 for name, _ in iter_quant_layers(model)}
    counts = {name: 0 for name in sens.keys()}
    exclude_layer_name_contains = exclude_layer_name_contains or []

    n_used = 0
    for batch in dataloader:
        if isinstance(batch, (list, tuple)):
            args = batch
            kwargs = {}
        elif isinstance(batch, dict):
            args = ()
            kwargs = batch
        else:
            args = (batch,)
            kwargs = {}

        _zero_grad(model)
        outs = loss_callback(model, args, kwargs)  # {"LA": tensor, "LF": tensor, ...}
        LA = outs["LA"]
        LF = outs["LF"]
        mixed = alpha * LA + LF
        mixed.backward()

        for name, m in iter_quant_layers(model):
            if any(kw in name for kw in exclude_layer_name_contains):
                continue
            g2_sum = 0.0
            n = 0
            for p in m.parameters():
                if p.grad is None:
                    continue
                g2_sum += torch.sum(p.grad.detach() ** 2).item()
                n += p.numel()
            if n > 0:
                sens[name] += g2_sum / n
                counts[name] += 1

        n_used += 1
        if n_used >= batches:
            break

    eps = 1e-12
    for k in sens.keys():
        if counts[k] > 0:
            sens[k] /= counts[k]
        sens[k] = max(sens[k], eps)
    # normalize to sum=1
    ssum = sum(sens.values())
    for k in sens.keys():
        sens[k] /= ssum
    return sens

# -------------------------------
# Bit distribution under avg-bit constraint (max-entropy)
# -------------------------------

def maxent_dist_under_avg(bits: List[int], target_avg: float) -> List[float]:
    b = torch.tensor(bits, dtype=torch.float64)
    lo, hi = -50.0, 50.0
    def mean_bits(lmb):
        w = torch.exp(lmb * b)
        q = w / w.sum()
        return float((q * b).sum())
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        m = mean_bits(mid)
        if m < target_avg:
            lo = mid
        else:
            hi = mid
    lam = 0.5 * (lo + hi)
    w = torch.exp(lam * b)
    q = (w / w.sum()).to(torch.float64)
    return [float(x) for x in q]

# -------------------------------
# Sinkhorn (log-domain)
# -------------------------------

def sinkhorn_log(
    cost: torch.Tensor,    # [L,B]
    a: torch.Tensor,       # [L] (sum=1)
    b: torch.Tensor,       # [B] (sum=1)
    epsilon: float = 0.02,
    iters: int = 500,
) -> torch.Tensor:
    K = -cost / epsilon
    f = torch.zeros(cost.size(0), device=cost.device)
    g = torch.zeros(cost.size(1), device=cost.device)

    def logsumexp(x, dim=-1):
        m, _ = torch.max(x, dim=dim, keepdim=True)
        return (m + torch.log(torch.sum(torch.exp(x - m), dim=dim, keepdim=True))).squeeze(dim)

    log_a = torch.log(a + 1e-40)
    log_b = torch.log(b + 1e-40)
    for _ in range(iters):
        f = log_a - logsumexp(K + g.unsqueeze(0), dim=1)
        g = log_b - logsumexp((K + f.unsqueeze(1)).transpose(0,1), dim=1)
    P = torch.exp(K + f.unsqueeze(1) + g.unsqueeze(0))
    return P / (P.sum() + 1e-40)

# -------------------------------
# Build cost (∆^2 × Fisher) and size-weighted marginals
# -------------------------------

def build_cost_and_marginals(
    sens: Dict[str, float],
    layer_names: List[str],
    layer_sizes: List[int],
    bits: List[int],
    device: torch.device,
    error_model: str = "pow2",  # "pow2": 2^{-2b}
):
    L = len(layer_names)
    B = len(bits)
    s = torch.tensor([sens[nm] for nm in layer_names], dtype=torch.float64, device=device)
    n = torch.tensor(layer_sizes, dtype=torch.float64, device=device)
    if error_model == "pow2":
        err = torch.tensor([2.0 ** (-2 * b) for b in bits], dtype=torch.float64, device=device)
    else:
        raise ValueError("unknown error model")
    C = (s.unsqueeze(1) * n.unsqueeze(1)) * err.unsqueeze(0)  # [L,B]
    # size-weighted a (param mass)
    total_params = n.sum()
    a = (n / total_params).to(torch.float32)  # sum=1
    return C.to(torch.float32), a, n, float(total_params.item())

# -------------------------------
# Size-aware greedy rounding (capacity per bit in param units)
# -------------------------------

def size_aware_rounding(
    P: torch.Tensor,                 # [L,B]
    layer_sizes: List[int],
    bits: List[int],
    target_b: List[float],           # column marginals (fraction of params)
) -> List[int]:
    L, B = P.shape
    items = []
    for i in range(L):
        for j in range(B):
            items.append((float(P[i,j].item()), i, j))
    items.sort(reverse=True, key=lambda x: x[0])

    sizes = [int(x) for x in layer_sizes]
    total = sum(sizes)
    capacity = [int(round(target_b[j] * total)) for j in range(B)]  # param capacity per bit
    assigned = [-1] * L
    used = [0] * B

    for score, i, j in items:
        if assigned[i] != -1:
            continue
        if used[j] + sizes[i] <= capacity[j]:
            assigned[i] = j
            used[j] += sizes[i]
        # early break if capacities are saturated and all layers placed
        if all(u >= c for u, c in zip(used, capacity)) and all(x != -1 for x in assigned):
            break

    # place leftovers to best feasible bit (least overflow)
    for i in range(L):
        if assigned[i] == -1:
            # choose j maximizing P[i,j] / overflow_penalty
            best_j, best_score = None, -1e9
            for j in range(B):
                overflow = max(0, used[j] + sizes[i] - capacity[j])
                score = float(P[i,j].item()) - 1e-9 * overflow
                if score > best_score:
                    best_score, best_j = score, j
            assigned[i] = best_j
            used[best_j] += sizes[i]
    return assigned

# -------------------------------
# Quantization (per-tensor symmetric)
# -------------------------------

def quantize_weight_per_tensor_symmetric(w: torch.Tensor, bits: int) -> torch.Tensor:
    if bits >= 32:
        return w.clone()
    qmax = (1 << (bits - 1)) - 1
    scale = w.detach().abs().max() / (qmax + 1e-12)
    scale = max(scale.item(), 1e-12)
    q = torch.clamp(torch.round(w / scale), min=-qmax-1, max=qmax)
    return q * scale

@dataclass
class QuantAssignment:
    layer_names: List[str]
    bit_indices: List[int]   # index into bits

def apply_weight_quantization_inplace(
    model: nn.Module,
    assignment: QuantAssignment,
    bits: List[int],
    keep_original: bool = True,
    exclude_layer_name_contains: Optional[List[str]] = None,
) -> Dict[str, torch.Tensor]:
    backup = {}
    exclude = exclude_layer_name_contains or []
    name2bit = {name: bits[idx] for name, idx in zip(assignment.layer_names, assignment.bit_indices)}
    for name, m in iter_quant_layers(model):
        if any(kw in name for kw in exclude):
            continue
        b = name2bit.get(name, None)
        if b is None:
            continue
        if hasattr(m, 'weight') and m.weight is not None:
            if keep_original:
                backup[f"{name}.weight"] = m.weight.detach().clone()
            with torch.no_grad():
                q = quantize_weight_per_tensor_symmetric(m.weight.data, b)
                m.weight.data.copy_(q)
    return backup

# -------------------------------
# End-to-end: allocate with Fisher-aware critical objective
# -------------------------------

@dataclass
class OTConfig:
    bits: List[int] = None
    avg_bits: float = 6.0
    epsilon: float = 0.02
    sinkhorn_iters: int = 400
    sens_batches: int = 1
    alpha: float = 1.0  # balance of overall vs critical
    exclude_layer_name_contains: Optional[List[str]] = None  # e.g. ["class_embed","bbox_embed"] for DETR

    def __post_init__(self):
        if self.bits is None:
            self.bits = [2,4,6,8]
        if self.exclude_layer_name_contains is None:
            self.exclude_layer_name_contains = []

@dataclass
class OTResult:
    assignment: QuantAssignment
    P: torch.Tensor
    cost: torch.Tensor
    a: torch.Tensor
    b: torch.Tensor
    layer_names: List[str]
    layer_sizes: List[int]

def allocate_bits_sinkhorn_fisher(
    model: nn.Module,
    dataloader,
    device: torch.device,
    loss_callback: Callable[[nn.Module, Tuple, Dict], Dict[str, torch.Tensor]],
    cfg: OTConfig,
) -> OTResult:
    # 1) Fisher-aware sensitivity for alpha*LA + LF (Eq. (8))
    sens = fisher_sensitivity_mixed_objective(
        model, dataloader, device, loss_callback,
        alpha=cfg.alpha, batches=cfg.sens_batches,
        exclude_layer_name_contains=cfg.exclude_layer_name_contains,
    )

    # 2) layer meta
    layer_names, layer_sizes = [], []
    for name, m in iter_quant_layers(model):
        layer_names.append(name)
        layer_sizes.append(param_count(m))

    # 3) cost and marginals
    C, a, nvec, total_params = build_cost_and_marginals(
        sens, layer_names, layer_sizes, cfg.bits, device, error_model="pow2"
    )

    # 4) choose column marginal b to satisfy E[bits]=avg_bits
    b_list = maxent_dist_under_avg(cfg.bits, cfg.avg_bits)
    b = torch.tensor(b_list, dtype=torch.float32, device=device)  # sum=1 over params

    # 5) Sinkhorn
    P = sinkhorn_log(C, a, b, epsilon=cfg.epsilon, iters=cfg.sinkhorn_iters)  # [L,B]

    # 6) Size-aware rounding to per-bit param capacities
    bit_indices = size_aware_rounding(P, layer_sizes, cfg.bits, b_list)

    return OTResult(
        assignment=QuantAssignment(layer_names=layer_names, bit_indices=bit_indices),
        P=P.detach().cpu(), cost=C.detach().cpu(), a=a.detach().cpu(), b=b.detach().cpu(),
        layer_names=layer_names, layer_sizes=layer_sizes
    )

# -------------------------------
# Fisher-trace regularization for QAT (Eq. (12))
# -------------------------------

def fisher_trace_regularizer(
    model: nn.Module,
    dataloader_iter,
    device: torch.device,
    critical_loss_fn: Callable[[nn.Module, Tuple, Dict], torch.Tensor],
    scale: float = 1.0,
) -> torch.Tensor:
    """
    1バッチで tr(I_F) ≈ sum_j mean( (d LF / d theta_j)^2 ).
    戻り値はスカラーTensor（学習ループで λ * reg を加える）。
    """
    model.train()
    _zero_grad(model)

    # 1 step
    try:
        batch = next(dataloader_iter)
    except StopIteration:
        raise RuntimeError("Exhausted dataloader iterator")

    if isinstance(batch, (list, tuple)):
        args, kwargs = batch, {}
    elif isinstance(batch, dict):
        args, kwargs = (), batch
    else:
        args, kwargs = (batch,), {}

    LF = critical_loss_fn(model, args, kwargs)  # only critical
    LF.backward()

    tr = torch.tensor(0.0, device=device)
    for p in model.parameters():
        if p.grad is not None:
            tr = tr + torch.mean(p.grad.detach() ** 2)
    _zero_grad(model)
    return scale * tr

# -------------------------------
# Example: simple classification loss callback with critical transform
# -------------------------------

def make_classification_loss_callback(
    crit_cfg: Optional[CriticalTransformCfg] = None,
) -> Callable[[nn.Module, Tuple, Dict], Dict[str, torch.Tensor]]:
    """
    LA: 通常のCE、LF: クリティカル変換後のCE
    """
    ce = nn.CrossEntropyLoss()
    def _cb(model: nn.Module, args: Tuple, kwargs: Dict) -> Dict[str, torch.Tensor]:
        x, y = args[0].to(next(model.parameters()).device), args[1].to(next(model.parameters()).device)
        logits = model(x)
        LA = ce(logits, y)
        if crit_cfg is None:
            LF = torch.tensor(0.0, device=logits.device)
        else:
            new_logits, new_y = critical_transform_logits_labels(logits, y, crit_cfg)
            LF = ce(new_logits, new_y)
        return {"LA": LA, "LF": LF}
    return _cb

# DETR用の注意:
#  - loss_callback内でHungarian matching, L_cls, L_boxを計算
#  - 論文通り class/bbox の蒸留を追加するとよい (KL + L1; one-to-one固定)。
#  - 出力FFNの量子化は避ける: exclude_layer_name_contains=["class_embed","bbox_embed"] など。

if __name__ == "__main__":
    from torchvision import datasets, transforms, models
    from torch.utils.data import DataLoader
    import torch, torch.nn as nn

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tfm = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])
    ds = datasets.FakeData(size=128, image_size=(3,224,224), num_classes=10, transform=tfm)
    dl = DataLoader(ds, batch_size=16, shuffle=False)

    model = models.resnet18(num_classes=10).to(device)

    # 例: 0..2 をクリティカルクラスとし Others=1クラスに折り畳む
    crit_cfg = CriticalTransformCfg(n_classes=10, critical_class_ids=[0,1,2], include_background=False)
    loss_cb = make_classification_loss_callback(crit_cfg)

    cfg = OTConfig(
        bits=[2,4,6,8],  avg_bits=6.0,   epsilon=0.02,
        sinkhorn_iters=400, sens_batches=2,
        alpha=0.7,  # L_A と L_F の重み
        exclude_layer_name_contains=[],  # DETRなら ["class_embed","bbox_embed"] 等
    )

    result = allocate_bits_sinkhorn_fisher(model, dl, device, loss_cb, cfg)

    # 割当を適用（必要に応じて除外パターン指定）
    backup = apply_weight_quantization_inplace(
        model,  result.assignment, cfg.bits,
        keep_original=True,
        exclude_layer_name_contains=cfg.exclude_layer_name_contains )

# QAT時に Fisher-trace 正則化を加える例（擬似コード）
# optimizer.zero_grad()
# outs = loss_cb(model, next(iter(dl)), {})   # {"LA","LF"}
# loss = outs["LA"]  # + 他の蒸留ロス等
# reg = fisher_trace_regularizer(model, iter(dl), device,
#                                critical_loss_fn=lambda m,a,k: loss_cb(m,a,k)["LF"],
#                                scale=1.0)
# total_loss = loss + 1e-3 * reg  # λをスケジュール (1e-3→5e-3) 推奨
# total_loss.backward()
# optimizer.step()
