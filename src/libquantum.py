from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Optional
import torch
import torch.nn as nn
import util

@dataclass
class QuantAssignment:
    layer_names: List[str]
    bit_indices: List[int]   # 同じ順序で bits[bit_indices[i]] が割当ビット

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

    for name, m in util.iter_quant_layers(model):
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
