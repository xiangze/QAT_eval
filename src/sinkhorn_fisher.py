# -*- coding: utf-8 -*-
"""
Layerwise bit-width allocation via Optimal Transport (Sinkhorn) instead of Fisher-based HAWQ-V2.
Author: you
Req: torch>=1.12
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import util
import libquantum as q
import HAWQ2_fisher as h2
# ----------------------------------------
# Sensitivity: empirical Fisher (grad^2)
# ----------------------------------------

@torch.no_grad()
def _zero_grad(model: nn.Module):
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()

def layer_sensitivity_empirical_fisher(
    model: nn.Module,
    dataloader,
    loss_fn,
    device: torch.device,
    num_batches: int = 1,
) -> Dict[str, float]:
    """
    経験的 Fisher（= 勾配二乗の平均）で層感度 s_i を推定。
    1〜数バッチで十分な相対比較になることが多い。
    """
    model.train()
    sens = {name: 0.0 for name, _ in util.iter_quant_layers(model)}
    counts = {name: 0 for name in sens.keys()}

    n_used = 0
    for batch in dataloader:
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            x, y = batch[0].to(device), batch[1].to(device)
        else:
            raise ValueError("dataloader should yield (inputs, targets)")

        _zero_grad(model)
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()

        # accumulate grad^2 per layer (weights only)
        for name, m in util.iter_quant_layers(model):
            g2_sum = 0.0
            n = 0
            for p in m.parameters():
                if p.grad is None: 
                    continue
                g2_sum += torch.sum(p.grad.detach()**2).item()
                n += p.numel()
            if n > 0:
                sens[name] += g2_sum / n  # 平均 grad^2
                counts[name] += 1

        n_used += 1
        if n_used >= num_batches:
            break

    # 正規化（相対重要度に）
    # 平滑化用に epsilon を加える
    eps = 1e-12
    for k in sens.keys():
        if counts[k] > 0:
            sens[k] = sens[k] / counts[k]
        sens[k] = max(sens[k], eps)

    total = sum(sens.values())
    for k in sens.keys():
        sens[k] /= total

    return sens  # sum to 1

# ----------------------------------------
# Bit target distribution with mean constraint
# ----------------------------------------

def maxent_bit_distribution(bits: List[int], target_avg: float) -> List[float]:
    """
    平均ビット制約 E[b]=target_avg の下で最大エントロピー分布 q(b)∝exp(λ b) を二分探索で解く。
    """
    b = torch.tensor(bits, dtype=torch.float64)
    # 二分探索範囲（十分広く）
    lo, hi = -50.0, 50.0

    def mean_bits(lmbda: float) -> float:
        w = torch.exp(lmbda * b)
        q = w / w.sum()
        return float((q * b).sum())

    # 単調性を仮定して二分探索
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        m = mean_bits(mid)
        if m < target_avg:
            lo = mid
        else:
            hi = mid
    lmbda = 0.5 * (lo + hi)
    w = torch.exp(lmbda * b)
    q = (w / w.sum()).to(dtype=torch.float64)
    return [float(x) for x in q]

def target_bit_counts(num_layers: int, bits: List[int], avg_bits: float) -> List[int]:
    """
    目標平均ビットを満たす最大エントロピー分布から整数個数を配分（Hare–Niemeyer）。
    """
    q = maxent_bit_distribution(bits, avg_bits)  # sum=1
    raw = [num_layers * qi for qi in q]
    base = [int(math.floor(x)) for x in raw]
    r = num_layers - sum(base)
    # 小数部が大きい順に +1
    frac = sorted(list(enumerate([x - math.floor(x) for x in raw])), key=lambda x: x[1], reverse=True)
    for i in range(r):
        base[frac[i][0]] += 1
    return base  # sum == num_layers

# ----------------------------------------
# Sinkhorn (log-domain stabilized)
# ----------------------------------------

def sinkhorn_transport(
    cost: torch.Tensor,  # [n_layers, n_bits]
    a: torch.Tensor,     # supply, sum=1, shape [n_layers]
    b: torch.Tensor,     # demand, sum=1, shape [n_bits]
    epsilon: float = 0.01,
    n_iters: int = 500,
) -> torch.Tensor:
    """
    ログドメイン Sinkhorn-Knopp。
    P = exp( (-C/ε) + f + g ), with row/col sums = a, b
    """
    device = cost.device
    K = -cost / epsilon  # [L, B]
    f = torch.zeros(cost.shape[0], device=device)
    g = torch.zeros(cost.shape[1], device=device)

    def logsumexp(x, dim=-1):
        m, _ = torch.max(x, dim=dim, keepdim=True)
        return (m + torch.log(torch.sum(torch.exp(x - m), dim=dim, keepdim=True))).squeeze(dim)

    log_a = torch.log(a + 1e-40)
    log_b = torch.log(b + 1e-40)

    for _ in range(n_iters):
        f = log_a - logsumexp(K + g.unsqueeze(0), dim=1)  # row scaling
        g = log_b - logsumexp((K + f.unsqueeze(1)).transpose(0,1), dim=1)  # col scaling

    P = torch.exp(K + f.unsqueeze(1) + g.unsqueeze(0))  # [L, B]
    # 正規化のズレを軽く修正
    P = P / (P.sum() + 1e-40)
    return P

# ----------------------------------------
# Rounding: match target counts
# ----------------------------------------

def greedy_rounding(P: torch.Tensor, target_counts: List[int]) -> List[int]:
    """
    P[i,b] をスコアとして (i,b) を大きい順に選ぶ。
    各層は一度だけ、各ビット幅は target_counts[b] 回だけ選ぶ。
    """
    L, B = P.shape
    # 候補 (score, i, b)
    items: List[Tuple[float,int,int]] = []
    for i in range(L):
        for b in range(B):
            items.append((float(P[i, b].item()), i, b))
    items.sort(reverse=True, key=lambda x: x[0])

    assigned = [-1] * L
    remaining = target_counts[:]

    for score, i, j in items:
        if assigned[i] != -1:
            continue
        if remaining[j] <= 0:
            continue
        assigned[i] = j
        remaining[j] -= 1
        # early exit
        if all(x == 0 for x in remaining):
            break

    # フォールバック（理論上は不要だが保険）
    for i in range(L):
        if assigned[i] == -1:
            j = max(range(B), key=lambda jb: P[i, jb].item() if remaining[jb] > 0 else -1e9)
            assigned[i] = j
            remaining[j] -= 1

    return assigned  # len=L, 値は bit-index

# ----------------------------------------
# 本体
# ----------------------------------------

def build_cost_matrix(
    sens: Dict[str, float],
    layer_names: List[str],
    layer_sizes: List[int],
    bits: List[int],
    device: torch.device,
) -> torch.Tensor:
    """
    C[i,b] = n_i * s_i * error(b) で構築。
    error(b) は量子化雑音の簡易 proxy として 2^{-2b} を使用。
    """
    L = len(layer_names)
    B = len(bits)
    s = torch.tensor([sens[name] for name in layer_names], dtype=torch.float64, device=device)
    n = torch.tensor(layer_sizes, dtype=torch.float64, device=device)
    err = torch.tensor([2.0 ** (-2 * b) for b in bits], dtype=torch.float64, device=device)  # ↓ with bits
    C = (s.unsqueeze(1) * n.unsqueeze(1)) * err.unsqueeze(0)  # [L,B]
    return C.to(dtype=torch.float32)

def ot_allocate_bits_for_model(
    model: nn.Module,
    dataloader,
    loss_fn,
    device: torch.device,
    config: q.OTQuantConfig = q.OTQuantConfig(),
    ) -> q.OTQuantResult:
    # 1) 感度推定
    sens = layer_sensitivity_empirical_fisher(model, dataloader, loss_fn, device, num_batches=config.sens_batches)

    # 2) レイヤ情報
    pairs = [(name, util.param_count(m)) for name, m in util.iter_quant_layers(model)]
    layer_names, layer_sizes = (list(t) for t in zip(*pairs))

    L = len(layer_names)
    bits = config.bits[:]

    # 3) コスト行列
    C = build_cost_matrix(sens, layer_names, layer_sizes, bits, device)

    # 4) 供給/需要分布
    a = torch.full((L,), 1.0 / L, device=device)  # 各層 1/L
    counts = target_bit_counts(L, bits, config.avg_bits)
    b = torch.tensor([c / L for c in counts], dtype=torch.float32, device=device)

    # 5) Sinkhorn
    P = sinkhorn_transport(C, a, b, epsilon=config.epsilon, n_iters=config.sinkhorn_iters)  # [L,B]

    # 6) 整数割当
    bit_indices = greedy_rounding(P, counts)  # [L] in [0..B-1]

    return q.OTQuantResult(
        assignment=q.QuantAssignment(layer_names=layer_names, bit_indices=bit_indices),
        P=P.detach().cpu(),
        target_counts=counts, cost_matrix=C.detach().cpu() )


def build_cost_sens_size_pow2(model: nn.Module, sens: Dict[str,float], bits: List[int], device) -> Tuple[torch.Tensor,List[str],List[int]]:
    names, sizes = [], []
    for name, m in util.iter_quant_layers(model):
        names.append(name)
        sizes.append(m.weight.numel())
    s = torch.tensor([sens[nm] for nm in names], dtype=torch.float64, device=device)
    n = torch.tensor(sizes, dtype=torch.float64, device=device)
    err = torch.tensor([2.0**(-2*b) for b in bits], dtype=torch.float64, device=device)
    C = (s.unsqueeze(1)*n.unsqueeze(1))*err.unsqueeze(0)
    return C.to(torch.float32), names, sizes

def maxent_b_dist(bits: List[int], target_avg: float) -> List[float]:
    b = torch.tensor(bits, dtype=torch.float64)
    lo,hi = -50.0,50.0
    for _ in range(80):
        mid = 0.5*(lo+hi)
        w = torch.exp(mid*b); q=w/w.sum(); m=float((q*b).sum())
        if m < target_avg: lo=mid
        else: hi=mid
    lam = 0.5*(lo+hi); w=torch.exp(lam*b); q=(w/w.sum()).tolist()
    return q

def size_aware_rounding(P: torch.Tensor, sizes: List[int], bits: List[int], col_fracs: List[float]) -> Dict[str,int]:
    L,B = P.shape
    total = sum(sizes)
    capacities = [int(round(total*f)) for f in col_fracs]
    items = []
    for i in range(L):
        for j in range(B):
            items.append((float(P[i,j].item()), i, j))
    items.sort(reverse=True, key=lambda x:x[0])
    assigned = [-1]*L
    used = [0]*B
    for score,i,j in items:
        if assigned[i]!=-1: continue
        if used[j] + sizes[i] <= capacities[j]:
            assigned[i]=j; used[j]+=sizes[i]
        if all(x!=-1 for x in assigned): break
    # fallback
    for i in range(L):
        if assigned[i]==-1:
            j = max(range(B), key=lambda jb: P[i,jb].item())
            assigned[i]=j
    return assigned

"""
全層一括
"""
def ot_hawq_like_allocate(model: nn.Module, loader: DataLoader, device, bits: List[int],
                          avg_bits: float, sens_batches=1, eps=0.02, iters=400) -> Dict[str,int]:
    sens = h2.empirical_fisher_sensitivity(model, loader, device, batches=sens_batches)
    C, names, sizes = build_cost_sens_size_pow2(model, sens, bits, device)
    n = torch.tensor(sizes, dtype=torch.float32, device=device)
    a = n/n.sum()
    col_fracs = maxent_b_dist(bits, avg_bits)
    b = torch.tensor(col_fracs, dtype=torch.float32, device=device)
    P = sinkhorn_transport(C, a, b, eps, iters)  # [L,B]
    idxs = size_aware_rounding(P, sizes, bits, col_fracs)
    return {nm: bits[idxs[i]] for i,nm in enumerate(names)}

def ot_fisher_critical_allocate(model: nn.Module, loader: DataLoader, device, 
                                config: q.OTQuantConfig = q.OTQuantConfig(),) -> Dict[str,int]:
    # For simplicity: if no critical_ids given -> same as OT_HAWQ_like
    return ot_hawq_like_allocate(model, loader, device, config.bits, config.avg_bits, sens_batches=config.sens_batches)


# ----------------------------------------
# Example usage (skeleton)
# ----------------------------------------
if __name__ == "__main__":
    # 例: CIFAR-10 + ResNet18 での使用（ダミー最小例）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # データ（1バッチだけ使う簡易校正）
    tfm = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])
    ds = datasets.FakeData(size=64, image_size=(3,224,224), num_classes=10, transform=tfm)
    dl = DataLoader(ds, batch_size=16, shuffle=False)

    # モデル
    model = models.resnet18(num_classes=10).to(device)
    loss_fn = nn.CrossEntropyLoss()

    # OT によるビット配分
    cfg = q.OTQuantConfig(bits=[2,4,6,8], avg_bits=6.0, epsilon=0.02, sinkhorn_iters=400, sens_batches=1)
    result = ot_allocate_bits_for_model(model, dl, loss_fn, device, cfg)
    
    # 量子化を適用
    backup = q.apply_weight_quantization_inplace(model, result.assignment, cfg.bits, keep_original=True )
    print("Assigned counts per bits:", {b: result.target_counts[i] for i, b in enumerate(cfg.bits)})
    print("First 10 layer -> bit:", [
        (result.assignment.layer_names[i], cfg.bits[result.assignment.bit_indices[i]])
        for i in range(min(10, len(result.assignment.layer_names)))    ])

    #critical ?
    result = ot_fisher_critical_allocate(model,dl,device,cfg)
    backup = q.apply_weight_quantization_inplace(model, result, cfg.bits, keep_original=True )
    # …ここで評価/微調整など…
    # 復元する場合:
    # restore_weights_from_backup(model, backup)

