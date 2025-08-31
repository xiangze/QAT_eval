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

# -------------------------------
# Utilities: collect target layers
# -------------------------------

def iter_quant_layers(model: nn.Module) -> Iterable[Tuple[str, nn.Module]]:
    """Conv2d / Linear を量子化対象として列挙（必要に応じて拡張）"""
    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            yield name, m

def param_count(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

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
    sens = {name: 0.0 for name, _ in iter_quant_layers(model)}
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
        for name, m in iter_quant_layers(model):
            g2_sum = 0.0
            n = 0
            for p in m.parameters():
                if p.grad is None: 
                    continue
                g2_sum += torch.sum(p.grad.detach() ** 2).item()
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
# Quantization (weights per-tensor symmetric uniform)
# ----------------------------------------

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
    bit_indices: List[int]   # 同じ順序で bits[bit_indices[i]] が割当ビット

def apply_weight_quantization_inplace(
    model: nn.Module,
    assignment: QuantAssignment,
    bits: List[int],
    keep_original: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    割当にもとづき Conv/Linear の weight を in-place 量子化。
    keep_original=True のときは元の重みを返す（復元用）。
    """
    backup = {}
    name2bit = {name: bits[idx] for name, idx in zip(assignment.layer_names, assignment.bit_indices)}

    for name, m in iter_quant_layers(model):
        b = name2bit.get(name, None)
        if b is None:
            continue
        # 対象: weight のみ（bias は非量子化）
        if hasattr(m, 'weight') and m.weight is not None:
            if keep_original:
                backup[f"{name}.weight"] = m.weight.detach().clone()
            with torch.no_grad():
                q = quantize_weight_per_tensor_symmetric(m.weight.data, b)
                m.weight.data.copy_(q)

    return backup  # 復元時に用いる

def restore_weights_from_backup(model: nn.Module, backup: Dict[str, torch.Tensor]):
    with torch.no_grad():
        for key, tensor in backup.items():
            # key 例: "layer3.0.conv1.weight"
            # state_dict() と同じキー体系でアクセス
            module_key, param_name = key.rsplit(".", 1)
            # 直接 Module をたどる
            mod = model
            for attr in module_key.split("."):
                if attr.isdigit():
                    mod = mod[int(attr)]  # Sequential 等
                else:
                    mod = getattr(mod, attr)
            getattr(mod, param_name).data.copy_(tensor)

# ----------------------------------------
# End-to-end pipeline
# ----------------------------------------

@dataclass
class OTQuantConfig:
    bits: List[int] = None                  # 例: [2,4,6,8]
    avg_bits: float = 6.0                   # 目標平均
    epsilon: float = 0.01                   # Sinkhorn 温度
    sinkhorn_iters: int = 500
    sens_batches: int = 1                   # 感度推定に使うバッチ数

    def __post_init__(self):
        if self.bits is None:
            self.bits = [2,4,6,8]

@dataclass
class OTQuantResult:
    assignment: QuantAssignment
    P: torch.Tensor
    target_counts: List[int]
    cost_matrix: torch.Tensor

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
    config: OTQuantConfig = OTQuantConfig(),
) -> OTQuantResult:
    # 1) 感度推定
    sens = layer_sensitivity_empirical_fisher(
        model, dataloader, loss_fn, device, num_batches=config.sens_batches
    )

    # 2) レイヤ情報
    layer_names: List[str] = []
    layer_sizes: List[int] = []
    for name, m in iter_quant_layers(model):
        layer_names.append(name)
        layer_sizes.append(param_count(m))

    L = len(layer_names)
    bits = config.bits[:]
    B = len(bits)

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

    return OTQuantResult(
        assignment=QuantAssignment(layer_names=layer_names, bit_indices=bit_indices),
        P=P.detach().cpu(),
        target_counts=counts,
        cost_matrix=C.detach().cpu()
    )

# ----------------------------------------
# Example usage (skeleton)
# ----------------------------------------

if __name__ == "__main__":
    # 例: CIFAR-10 + ResNet18 での使用（ダミー最小例）
    # 実際には DataLoader/モデル定義をあなたの環境に合わせて差し替えてください。
    try:
        from torchvision import datasets, transforms, models
        from torch.utils.data import DataLoader
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # データ（1バッチだけ使う簡易校正）
        tfm = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])
        ds = datasets.FakeData(size=64, image_size=(3,224,224), num_classes=10, transform=tfm)
        dl = DataLoader(ds, batch_size=16, shuffle=False)

        # モデル
        model = models.resnet18(num_classes=10).to(device)
        loss_fn = nn.CrossEntropyLoss()

        # OT によるビット配分
        cfg = OTQuantConfig(bits=[2,4,6,8], avg_bits=6.0, epsilon=0.02, sinkhorn_iters=400, sens_batches=1)
        result = ot_allocate_bits_for_model(model, dl, loss_fn, device, cfg)

        # 量子化を適用
        backup = apply_weight_quantization_inplace(
            model, result.assignment, cfg.bits, keep_original=True
        )
        print("Assigned counts per bits:", {b: result.target_counts[i] for i, b in enumerate(cfg.bits)})
        print("First 10 layer -> bit:", [
            (result.assignment.layer_names[i], cfg.bits[result.assignment.bit_indices[i]])
            for i in range(min(10, len(result.assignment.layer_names)))
        ])

        # …ここで評価/微調整など…
        # 復元する場合:
        # restore_weights_from_backup(model, backup)

    except Exception as e:
        print("Demo failed (install torchvision for the example):", e)
