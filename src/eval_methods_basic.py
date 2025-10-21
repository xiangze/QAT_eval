"""
Evaluate multiple layerwise quantization allocation methods on a trained network.
Methods:
  - HAWQ2_fisher (greedy via empirical Fisher)
  - OT_HAWQ_like (Sinkhorn OT with size-weighted marginals)
  - OT_Fisher_Critical (same API; if no critical classes -> same as OT_HAWQ_like)
  - DiffSinkhornDynamic (differentiable Sinkhorn with STE mixture + short QAT)
  - SinkhornMCKPDynamic (Chen et al. quadratic cost + short QAT)

Outputs:
  - Pre-quantization (FP32) accuracy
  - For each method: accuracy, quantized size (MB), mean bits, per-layer assignment CSV
"""

import argparse, json, os, math, csv, time
from dataclasses import dataclass
from typing import List, Dict, Tuple, Iterable, Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import HAWQ2_fisher as h2
import sinkhorn_fisher as OT
import MCKP_sinkhorn as MCKPs
import Dynamic_sinkhorn as Dyns
import util

def quantize_weight_per_tensor_symmetric(w: torch.Tensor, bits: int) -> torch.Tensor:
    if bits >= 32:
        return w.clone()
    qmax = (1 << (bits - 1)) - 1
    scale = w.detach().abs().max() / (qmax + 1e-12)
    scale = max(scale.item(), 1e-12)
    q = torch.clamp(torch.round(w / scale), min=-qmax-1, max=qmax)
    return q * scale

def apply_assignment_inplace(model: nn.Module, assignment: Dict[str, int]) -> Dict[str, torch.Tensor]:
    """
    assignment: {layer_name: bit}
    Returns backup for restoration.
    """
    backup = {}
    with torch.no_grad():
        for name, m in util.util.iter_quant_layers(model):
            b = assignment.get(name, None)
            if b is None: continue
            if hasattr(m, "weight") and m.weight is not None:
                backup[f"{name}.weight"] = m.weight.detach().clone()
                q = quantize_weight_per_tensor_symmetric(m.weight.data, b)
                m.weight.data.copy_(q)
    return backup

def restore_from_backup(model: nn.Module, backup: Dict[str, torch.Tensor]):
    with torch.no_grad():
        for key, tensor in backup.items():
            mod = model
            path, p = key.rsplit(".", 1)
            for token in path.split("."):
                if token.isdigit(): mod = mod[int(token)]
                else: mod = getattr(mod, token)
            getattr(mod, p).data.copy_(tensor)

def model_fp32_size_mb(model: nn.Module) -> float:
    n_params = sum(p.numel() for p in model.parameters())
    return n_params * 32 / 8 / 1e6

def quantized_weight_size_mb(model: nn.Module, assignment: Dict[str, int], include_bias_fp32=True) -> float:
    """
    Size = sum_i (weight_params_i * b_i) + (optional) bias in 32-bit.
    """
    total_bits = 0
    for name, m in util.iter_quant_layers(model):
        b = assignment.get(name, None)
        if b is None: b = 32
        w_params = m.weight.numel() if hasattr(m, "weight") and m.weight is not None else 0
        total_bits += w_params * b
        if include_bias_fp32 and hasattr(m, "bias") and m.bias is not None:
            total_bits += m.bias.numel() * 32
    return total_bits / 8 / 1e6

def save_assignment_csv(path: str, assignment: Dict[str, int], model: nn.Module):
    rows = []
    for name, m in util.iter_quant_layers(model):
        rows.append([name, util.param_count(m), assignment.get(name, 32)])
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer_name", "param_count", "bits"])
        w.writerows(rows)

def mean_bits(assignment: Dict[str, int], model: nn.Module) -> float:
    num = 0
    den = 0
    for name, m in util.iter_quant_layers(model):
        b = assignment.get(name, 32)
        n = m.weight.numel() if hasattr(m, "weight") and m.weight is not None else 0
        num += b * n
        den += n
    return num / max(1, den)

# =========================
# Data & model builders
# =========================

def build_cifar10_loaders(batch=128, num_workers=4, train_aug=False):
    from torchvision import datasets, transforms
    tfm_train = [
        transforms.Resize(224),
        transforms.RandomHorizontalFlip() if train_aug else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ]
    tfm_train = [t for t in tfm_train if not isinstance(t, transforms.Lambda)]
    tfm_train = transforms.Compose(tfm_train)

    tfm_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    train = datasets.CIFAR10(root="./data", train=True, download=True, transform=tfm_train)
    test  = datasets.CIFAR10(root="./data", train=False, download=True, transform=tfm_test)
    return DataLoader(train, batch_size=batch, shuffle=True, num_workers=num_workers, pin_memory=True), \
           DataLoader(test,  batch_size=batch, shuffle=False, num_workers=num_workers, pin_memory=True), 10

def build_imagenet_val_loader(val_dir: str, batch=128, num_workers=8):
    from torchvision import datasets, transforms
    tfm = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    ds = datasets.ImageFolder(val_dir, transform=tfm)
    return DataLoader(ds, batch_size=batch, shuffle=False, num_workers=num_workers, pin_memory=True), 1000

def build_model(name: str, num_classes: int, pretrained_tv: bool, device: torch.device):
    from torchvision import models
    name = name.lower()
    if name == "resnet18":
        if pretrained_tv:
            m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            if m.fc.out_features != num_classes:
                m.fc = nn.Linear(m.fc.in_features, num_classes)
        else:
            m = models.resnet18(num_classes=num_classes)
    elif name == "mobilenet_v2":
        if pretrained_tv:
            m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
            if m.classifier[-1].out_features != num_classes:
                m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
        else:
            m = models.mobilenet_v2(num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported model: {name}")
    return m.to(device)

def train_cifar_quick(model: nn.Module, train_loader, test_loader, device, epochs=2, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()
    for ep in range(epochs):
        model.train()
        for x,y in train_loader:
            x,y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = ce(logits, y)
            loss.backward()
            opt.step()
        acc = evaluate_top1(model, test_loader, device)
        print(f"[train] epoch {ep} CIFAR10 acc={acc:.2f}%")
    return model

@torch.no_grad()
def evaluate_top1(model: nn.Module, loader: DataLoader, device) -> float:
    model.eval()
    corr = 0
    total = 0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        corr += (pred == y).sum().item()
        total += y.numel()
    return 100.0 * corr / max(1,total)


"""
Experiment runner

Method A: HAWQ2_fisher (greedy)
Method B: OT_HAWQ_like
Method C: OT_Fisher_Critical (同 API)  クリティカル変換省略時はOT_HAWQ_likeと同じ
Method D/E: Differentiable Sinkhorn (dynamic) - Minimal inlined versions from previous prototypes
"""
def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    # Data & model
    if args.train_cifar10:
        train_loader, test_loader, ncls = build_cifar10_loaders(batch=args.batch, num_workers=args.workers, train_aug=True)
        model = build_model(args.model, ncls, pretrained_tv=False, device=device)
        model = train_cifar_quick(model, train_loader, test_loader, device, epochs=args.train_epochs, lr=args.lr)
        eval_loader = test_loader
        model_ctor = lambda: build_model(args.model, ncls, pretrained_tv=False, device=device)
    else:
        # torchvision pretrained; need a val loader (ImageNet val path or CIFAR10 test fallback)
        if args.imagenet_val:
            eval_loader, ncls = build_imagenet_val_loader(args.imagenet_val, batch=args.batch, num_workers=args.workers)
            model = build_model(args.model, 1000, pretrained_tv=True, device=device)
            model_ctor = lambda: build_model(args.model, 1000, pretrained_tv=True, device=device)
        else:
            # fallback to CIFAR10 test (head adapted to 10 classes)
            train_loader, test_loader, ncls = build_cifar10_loaders(batch=args.batch, num_workers=args.workers)
            model = build_model(args.model, ncls, pretrained_tv=True, device=device)
            eval_loader = test_loader
            model_ctor = lambda: build_model(args.model, ncls, pretrained_tv=True, device=device)

    # Baseline FP32
    base_acc = evaluate_top1(model, eval_loader, device)
    base_size = model_fp32_size_mb(model)
    print(f"[FP32] acc={base_acc:.2f}%  size={base_size:.2f} MB")

    methods = []
    if "HAWQ2_fisher" in args.methods: methods.append("HAWQ2_fisher")
    if "OT_HAWQ_like" in args.methods: methods.append("OT_HAWQ_like")
    if "OT_Fisher_Critical" in args.methods: methods.append("OT_Fisher_Critical")
    if "DiffSinkhornDynamic" in args.methods: methods.append("DiffSinkhornDynamic")
    if "SinkhornMCKPDynamic" in args.methods: methods.append("SinkhornMCKPDynamic")

    # Iterate methods
    results = []
    for meth in methods:
        print(f"\n=== [{meth}] ===")
        # fresh copy
        qmodel = model_ctor().to(device)
        # build assignment
        if meth == "HAWQ2_fisher":
            assignment = h2.hawq2_fisher_allocate(qmodel, eval_loader, device, args.bits, args.avg_bits, batches=args.sens_batches)
        elif meth == "OT_HAWQ_like":
            assignment = OT.ot_hawq_like_allocate(qmodel, eval_loader, device, args.bits, args.avg_bits,
                                               sens_batches=args.sens_batches, eps=args.sinkhorn_eps, iters=args.sinkhorn_iters)
        elif meth == "OT_Fisher_Critical":
            assignment = OT.ot_fisher_critical_allocate(qmodel, eval_loader, device, args.bits, args.avg_bits,
                                                     sens_batches=args.sens_batches, critical_ids=None)
        elif meth == "DiffSinkhornDynamic":
            assignment = Dyns.dynamic_sinkhorn_allocate(model_ctor, train_loader if args.train_cifar10 else eval_loader,
                                                   eval_loader, device, args.bits, args.avg_bits,
                                                   steps=args.dynamic_steps, lr=args.dynamic_lr, chen=False)
        elif meth == "SinkhornMCKPDynamic":
            assignment = MCKPs.sinkhorn_MCKP_allocate(model_ctor, train_loader if args.train_cifar10 else eval_loader,
                                                   eval_loader, device, args.bits, args.avg_bits,
                                                   steps=args.dynamic_steps, lr=args.dynamic_lr, chen=True)
        else:
            raise ValueError(meth)

        # apply quantization in-place
        backup = apply_assignment_inplace(qmodel, assignment)
        acc = evaluate_top1(qmodel, eval_loader, device)
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
        # restore (not needed since we use fresh qmodel each time)
        # restore_from_backup(qmodel, backup)

    # dump summary
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump({
            "fp32": {"acc": base_acc, "size_MB": base_size},
            "bits": args.bits, "avg_bits": args.avg_bits,
            "results": results
        }, f, indent=2)
    print("\n=== Summary ===")
    print(json.dumps({"fp32":{"acc":base_acc,"size_MB":base_size},"results":results}, indent=2))

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="resnet18", help="resnet18|mobilenet_v2")
    p.add_argument("--cpu", action="store_true")
    # data options
    p.add_argument("--train_cifar10", action="store_true", help="train on CIFAR10 quickly instead of using torchvision pretrained")
    p.add_argument("--train_epochs", type=int, default=2)
    p.add_argument("--imagenet_val", type=str, default="", help="path to ImageNet val dir if evaluating torchvision pretrained")
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)

    # methods to run
    p.add_argument("--methods", type=str, nargs="+",
                   default=["HAWQ2_fisher","OT_HAWQ_like","OT_Fisher_Critical","DiffSinkhornDynamic","SinkhornMCKPDynamic"])

    # quant options
    p.add_argument("--bits", type=int, nargs="+", default=[2,4,6,8])
    p.add_argument("--avg_bits", type=float, default=6.0)
    p.add_argument("--sens_batches", type=int, default=2)

    # sinkhorn
    p.add_argument("--sinkhorn_eps", type=float, default=0.02)
    p.add_argument("--sinkhorn_iters", type=int, default=400)

    # dynamic
    p.add_argument("--dynamic_steps", type=int, default=50)
    p.add_argument("--dynamic_lr", type=float, default=1e-4)

    p.add_argument("--out_dir", type=str, default="./quant_results")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run(args)
