#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

import torchvision
from torchvision import datasets, transforms
from torchvision.models import get_model_weights
import HAWQ2_fisher as h2
import sinkhorn_fisher as OT
import Dynamic_sinkhorn as Dyns
import util
import json
from eval_methods_basic import apply_assignment_inplace,evaluate_top1,quantized_weight_size_mb,mean_bits,save_assignment_csv,model_fp32_size_mb,restore_from_backup

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def infer_input_size_from_weights(weights) -> int:
    # weights.transforms() は Resize/CenterCrop を含む Compose を返す。
    # 代表的な分類モデルなら最終の CenterCrop/Resize の size を224/256相当と仮定。
    # 正確に取り出せない場合はフォールバックで224。
    try:
        t = weights.transforms()
        # 探せたら crop/resize の size を拾う（ざっくり）。
        for tr in reversed(t.transforms if hasattr(t, "transforms") else []):
            if hasattr(tr, "size"):
                size = tr.size
                if isinstance(size, (tuple, list)):
                    return int(size[0])
                return int(size)
    except Exception:
        pass
    return 224

def build_transforms(weights=None, train: bool = True, force_rgb: bool = True) -> transforms.Compose:
    if weights is not None:
        # eval transforms をベースに、train 時は軽い増強に差し替え
        size = infer_input_size_from_weights(weights)
        mean = weights.meta.get("mean", IMAGENET_MEAN)
        std  = weights.meta.get("std", IMAGENET_STD)

        if train:
            tf = [
                transforms.Resize(int(size * 1.14)),  # 256 for 224
                transforms.RandomResizedCrop(size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        else:
            # weights の transforms() は Resize/CenterCrop/ToTensor/Normalize を含むのでそれを使う
            tf = list(weights.transforms().transforms)
    else:
        size = 224
        mean, std = IMAGENET_MEAN, IMAGENET_STD
        if train:
            tf = [
                transforms.Resize(256),
                transforms.RandomResizedCrop(size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        else:
            tf = [
                transforms.Resize(256),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]

    if force_rgb:
        # 先頭に Grayscale->RGB 変換を入れて 1ch データセットにも対応
        tf = [transforms.Grayscale(num_output_channels=3), transforms.Lambda(lambda x: x)] + tf
        # 上の Lambda は PIL→PIL のまま抜けるダミー。Grayscale は PIL 画像にのみ作用。

    return transforms.Compose(tf)

def load_dataset(name: str, root: str, train_tf, eval_tf, val_split: float = 0.1, seed: int = 0):
    name_lower = name.lower()

    # ImageFolder は train/val を root/以下のサブフォルダで切りたいケースが多いので慣習対応
    if name_lower == "imagefolder":
        train_dir = os.path.join(root, "train")
        val_dir = os.path.join(root, "val")
        test_dir = os.path.join(root, "test")
        if os.path.isdir(train_dir) and os.path.isdir(val_dir):
            train_set = datasets.ImageFolder(train_dir, transform=train_tf)
            val_set   = datasets.ImageFolder(val_dir, transform=eval_tf)
            test_set  = datasets.ImageFolder(test_dir, transform=eval_tf) if os.path.isdir(test_dir) else val_set
            return train_set, val_set, test_set
        else:
            # 単一ディレクトリの場合は内部で分割
            full = datasets.ImageFolder(root, transform=train_tf)
            n_val = max(1, int(len(full)*val_split))
            n_train = len(full) - n_val
            g = torch.Generator().manual_seed(seed)
            train_set, val_set = random_split(full, [n_train, n_val], generator=g)
            # 検証/テストでは eval_tf を使う
            val_set.dataset.transform = eval_tf
            test_set = val_set
            return train_set, val_set, test_set

    # 代表的な分類データセット（自動ダウンロード）
    ds_mod = torchvision.datasets

    if name_lower in ["fashionmnist"]:
        DS = getattr(ds_mod, name_upper(name_lower))
        train_set = DS(root=root, train=True, download=True, transform=train_tf)
        test_set  = DS(root=root, train=False, download=True, transform=eval_tf)
        # 検証はテストをそのまま
        return train_set, test_set, test_set

    if name_lower in ["mnist","cifar10", "cifar100", "stl10", "svhn"]:
        if name_lower == "stl10":
            train_set = datasets.STL10(root=root, split="train", download=True, transform=train_tf)
            test_set  = datasets.STL10(root=root, split="test",  download=True, transform=eval_tf)
        elif name_lower == "svhn":
            train_set = datasets.SVHN(root=root, split="train", download=True, transform=train_tf)
            test_set  = datasets.SVHN(root=root, split="test",  download=True, transform=eval_tf)
        else:
            DS = getattr(ds_mod, name_lower.upper())
            train_set = DS(root=root, train=True, download=True, transform=train_tf)
            test_set  = DS(root=root, train=False, download=True, transform=eval_tf)
        return train_set, test_set, test_set

    raise ValueError(f"Unsupported dataset for this simple trainer: {name}. "
                     f"Try one of: ImageFolder, MNIST, FashionMNIST, CIFAR10, CIFAR100, STL10, SVHN.")

def name_upper(name_lower: str) -> str:
    # 'fashionmnist' -> 'FashionMNIST' 等の簡易変換
    return "".join([w.capitalize() for w in name_lower.split()])


def replace_classifier(model: nn.Module, num_classes: int) -> nn.Module:
    # 多くの torchvision 分類モデルに対応
    # ResNet/RegNet/EfficientNet/DenseNet/MobileNet 等をケース分け
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        in_f = model.fc.in_features
        model.fc = nn.Linear(in_f, num_classes)
        return model

    if hasattr(model, "classifier"):
        clf = model.classifier
        if isinstance(clf, nn.Linear):
            model.classifier = nn.Linear(clf.in_features, num_classes)
            return model
        if isinstance(clf, nn.Sequential) and len(clf) > 0:
            # 末尾の Linear を差し替え
            last_idx = None
            for i in reversed(range(len(clf))):
                if isinstance(clf[i], nn.Linear):
                    last_idx = i
                    break
            if last_idx is not None:
                in_f = clf[last_idx].in_features
                new_seq = list(clf)
                new_seq[last_idx] = nn.Linear(in_f, num_classes)
                model.classifier = nn.Sequential(*new_seq)
                return model

    # ViT 等（heads.head）
    if hasattr(model, "heads") and hasattr(model.heads, "head") and isinstance(model.heads.head, nn.Linear):
        in_f = model.heads.head.in_features
        model.heads.head = nn.Linear(in_f, num_classes)
        return model

    # ConvNeXt（classifier[2] が Linear）
    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential) and len(model.classifier) >= 3:
        if isinstance(model.classifier[-1], nn.Linear):
            in_f = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_f, num_classes)
            return model

    raise RuntimeError("Could not find a classifier head to replace; model not supported by this helper.")

def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    return correct / max(1, targets.size(0))

def run_epoch(model, loader, criterion, optimizer, device, train: bool) -> Tuple[float, float]:
    model=model.to(device)
    if train:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    running_correct = 0
    running_total = 0

    with torch.set_grad_enabled(train):
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if train:
                optimizer.zero_grad()

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            if train:
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * labels.size(0)
            running_correct += (outputs.argmax(1) == labels).sum().item()
            running_total += labels.size(0)

    avg_loss = running_loss / max(1, running_total)
    acc = running_correct / max(1, running_total)
    err = 1.0 - acc
    return avg_loss, err

def main(args):
    device = torch.device(args.device)
    if(args.debug):
        args.epochs=1
        args.methods=["DiffSinkhornDynamic","SinkhornMCKPDynamic"]
        args.dataset="MNIST"
        args.batch_size=4

    # 0) 準備
    methods = []
    if "HAWQ2_fisher" in args.methods: methods.append("HAWQ2_fisher")
    if "OT_HAWQ_like" in args.methods: methods.append("OT_HAWQ_like")
    if "OT_Fisher_Critical" in args.methods: methods.append("OT_Fisher_Critical")
    if "DiffSinkhornDynamic" in args.methods: methods.append("DiffSinkhornDynamic")
    if "SinkhornMCKPDynamic" in args.methods: methods.append("SinkhornMCKPDynamic")

    # 1) Weights の取得
    weights = None
    if args.pretrained:
        try:
            # 例: get_model_weights("resnet18") -> ResNet18_Weights Enum、.DEFAULT を使う
            enum = get_model_weights(args.model)
            weights = enum.DEFAULT if hasattr(enum, "DEFAULT") else list(enum)[-1]
        except Exception:
            print(f"[WARN] Could not get weights enum for {args.model}; proceeding without pretrained weights.")
            weights = None

    # 2) Transforms
    train_tf = build_transforms(weights, train=True,  force_rgb=True)
    eval_tf  = build_transforms(weights, train=False, force_rgb=True)

    # 3) Dataset & DataLoaders
    train_set, val_set, test_set = load_dataset(args.dataset, args.data_root, train_tf, eval_tf, args.val_split, args.seed)
    num_classes = len(getattr(train_set, "classes", getattr(getattr(train_set, "dataset", None), "classes", [])))
    if (num_classes is None or num_classes == 0):
        raise RuntimeError("Could not infer num_classes from dataset. Ensure your dataset exposes `.classes` (ImageFolder etc.).")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    # 4) Model
    model_ctor = getattr(torchvision.models, args.model, None)
    if model_ctor is None:
        raise ValueError(f"Unknown model name: {args.model}")
    model = model_ctor(weights=weights).to(device)

    # 最終層をクラス数に合わせて置換
    model = replace_classifier(model, num_classes)

    # 5) Optimizer / Loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 6) Train / Eval Loop
    print(f"Start training: dataset={args.dataset}, model={args.model}, num_classes={num_classes}, device={device}")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_err = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_err     = run_epoch(model, val_loader,   criterion, optimizer, device, train=False)

        print(f"[Epoch {epoch:03d}] "
              f"train_loss={train_loss:.4f}  train_error={train_err:.4%}  "
              f"val_loss={val_loss:.4f}  val_error={val_err:.4%}")
    
    # 6.5) Apply MPQ methods
    ## Baseline FP32
    base_acc = evaluate_top1(model, val_loader, device)
    base_size = model_fp32_size_mb(model)
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

    # dump summary
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\n=== Summary ===")
    print(json.dumps(results, indent=2))

    # 7) Final Test
    test_loss, test_err = run_epoch(model, test_loader, criterion, optimizer, device, train=False)
    print(f"[Test] loss={test_loss:.4f}  error={test_err:.4%}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generic torchvision train/eval script (classification).")
    p.add_argument("--dataset",      type=str, required=True,
                        help="Dataset name: ImageFolder | MNIST | FashionMNIST | CIFAR10 | CIFAR100 | STL10 | SVHN")
    p.add_argument("--data-root",    type=str,   default="data/", help="Dataset root. For ImageFolder, use root/ or root/train,root/val")
    p.add_argument("--model",        type=str,   default="resnet18", help="torchvision.models.* name (e.g., resnet18, efficientnet_b0, mobilenet_v3_small)")
    p.add_argument("--epochs",       type=int,   default=30)
    p.add_argument("--batch-size",   type=int,   default=64)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers",  type=int,   default=4)
    p.add_argument("--device",       type=str,   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--val-split",    type=float, default=0.1, help="Only for ImageFolder single-dir mode.")
    p.add_argument("--seed",         type=int,   default=0)
    p.add_argument("--pretrained",   action="store_true", help="Use DEFAULT weights if available (fine-tuning).")
    # methods to run
    p.add_argument("--methods",      type=str,   nargs="+",
                   default=["HAWQ2_fisher","OT_HAWQ_like","DiffSinkhornDynamic","SinkhornMCKPDynamic"])
    # quant options
    p.add_argument("--bits",         type=int,   nargs="+", default=[2,4,6,8])
    p.add_argument("--avg_bits",     type=float, default=6.0)
    p.add_argument("--sens_batches", type=int,   default=2)
    # sinkhorn
    p.add_argument("--sinkhorn_eps", type=float, default=0.02)
    p.add_argument("--sinkhorn_iters",type=int,  default=400)
    # dynamic
    p.add_argument("--dynamic_steps",type=int,   default=50)
    p.add_argument("--dynamic_lr",   type=float, default=1e-4)
    p.add_argument("--chen",         action="store_true")

    p.add_argument("--out_dir",      type=str, default="./quant_results")
    p.add_argument("--debug",        action="store_true")
    p.add_argument("--restore",      action="store_true")
    args = p.parse_args()
    main(args)
