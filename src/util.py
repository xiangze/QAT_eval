# ---------------------------
# Utilities
# ---------------------------
from typing import List, Dict, Tuple, Iterable, Optional, Callable
import torch.nn as nn

def iter_quant_layers(model: nn.Module) -> Iterable[Tuple[str, nn.Module]]:
    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            yield name, m

def param_count(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

def get_module_by_name(model: nn.Module, dotted: str) -> nn.Module:
    mod = model
    for token in dotted.split("."):
        if token.isdigit():
            mod = mod[int(token)]
        else:
            mod = getattr(mod, token)
    return mod
