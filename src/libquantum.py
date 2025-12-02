from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Optional
import torch
import torch.nn as nn
import qutil

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

    for name, m in qutil.iter_quant_layers(model):
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

from eval_methods_basic import apply_assignment_inplace,evaluate_top1,quantized_weight_size_mb,mean_bits,save_assignment_csv,model_fp32_size_mb,restore_from_backup
import HAWQ2_fisher as h2
import sinkhorn_fisher as OT
import Dynamic_sinkhorn as Dyns
import MCKP_sinkhorn as MCKPs
import os
import json

# Apply MPQ methods
def dumpresults(args, model_ctor,orgmodel,device,val_loader,train_loader,
                methods:list=["HAWQ2_fisher","OT_HAWQ_like","DiffSinkhornDynamic","SinkhornMCKPDynamic"],
                dump=False):
    ## Baseline FP32
    base_acc = evaluate_top1(orgmodel, val_loader, device)
    base_size = model_fp32_size_mb(orgmodel)
    print(f"[FP32] acc={base_acc:.2f}%  size={base_size:.2f} MB")

    results = []
    for meth in methods:
        print(f"\n=== [{meth}] ===")
        # fresh copy
        qmodel = model_ctor().to(device)
        # build assignment
        if meth == "HAWQ2_fisher":
            assignment = h2.hawq2_fisher_allocate(qmodel, val_loader, device, args.bits, args.avg_bits, batches=args.sens_batches)
        elif meth == "OT_HAWQ_like":
            assignment = OT.ot_hawq_like_allocate(qmodel, val_loader, device, args.bits, args.avg_bits,
                                               sens_batches=args.sens_batches, eps=args.sinkhorn_eps, iters=args.sinkhorn_iters)
        elif "Sinkhorn"in meth :
            assignment = Dyns.dynamic_sinkhorn_allocate(qmodel, val_loader, device, args.bits, args.avg_bits,
                                                   steps=args.dynamic_steps, lr=args.dynamic_lr, chen=args.chen,MCKP=("MCKP" in meth))
        elif meth == "OT_Fisher_Critical":
            assignment = OT.ot_fisher_critical_allocate(qmodel, val_loader, device, args.bits, args.avg_bits,
                                                     sens_batches=args.sens_batches, critical_ids=None)
        elif meth == "DiffSinkhornDynamic":
            assignment = Dyns.dynamic_sinkhorn_allocate(model_ctor, train_loader if args.train_cifar10 else val_loader,
                                                   val_loader, device, args.bits, args.avg_bits,
                                                   steps=args.dynamic_steps, lr=args.dynamic_lr, chen=False)
        elif meth == "SinkhornMCKPDynamic":
            assignment = MCKPs.sinkhorn_MCKP_allocate(model_ctor, train_loader if args.train_cifar10 else val_loader,
                                                   val_loader, device, args.bits, args.avg_bits,
                                                   steps=args.dynamic_steps, lr=args.dynamic_lr, chen=True)
        else:
            raise ValueError(meth)

        # apply quantization in-place
        acc = evaluate_top1(qmodel, val_loader, device)
        qsize = quantized_weight_size_mb(qmodel, assignment, include_bias_fp32=True)
        mbits = mean_bits(assignment, qmodel)
        print(f"[{meth}] acc={acc:.2f}%  size={qsize:.2f} MB  mean_bits≈{mbits:.2f}")
        # save assignment
        csv_path = os.path.join(args.out_dir, f"{meth}_assignment.csv")
        save_assignment_csv(csv_path, assignment, qmodel)
        results.append({
            "method": meth,
            "acc": acc,
            "size_MB": qsize,
            "mean_bits": mbits,
            "assignment_csv": csv_path
        })

        if(args.restore):# restore (not needed since we use fresh qmodel each time)
            backup = apply_assignment_inplace(qmodel, assignment)
            restore_from_backup(qmodel, backup)

    if(dump):
        with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
            json.dump(results, f, indent=2)
        print("\n=== Summary ===")
        print(json.dumps(results, indent=2))

    return results