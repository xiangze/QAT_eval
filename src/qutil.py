# ---------------------------
# Utilities
# ---------------------------
import random, inspect
from typing import List, Dict, Tuple, Iterable, Optional, Callable
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.transforms.functional import InterpolationMode

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

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



# 画像正規化（for ImageNet）
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def _build_transforms(is_train: bool, grayscale_to_rgb: bool,imagesize:int=224) -> transforms.Compose:
    """ResNet50 用の前処理を返す。MNIST系は 1ch→3ch に変換。"""
    ops = []
    if grayscale_to_rgb:
        # PIL Image(L) → 3ch
        ops.append(transforms.Grayscale(num_output_channels=3))
    if is_train:
        # 小さな画像(CIFARなど)にも効くよう RandomResizedCrop を採用
        ops += [
            transforms.RandomResizedCrop(imagesize, scale=(0.8, 1.0), interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
        ]
    else:
        ops += [
            transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
        ]
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    return transforms.Compose(ops)

def _make_dataset(name: str, root: str, train: bool, download: bool, tfm: transforms.Compose ) -> Tuple[torch.utils.data.Dataset, int]:
    """データセットを作り、(dataset, num_classes) を返す。"""
    n = name.lower()
    if n in {"mnist", "fashionmnist", "kmnist"}:
        cls = {
            "mnist": datasets.MNIST,
            "fashionmnist": datasets.FashionMNIST,
            "kmnist": datasets.KMNIST,
        }[n]
        ds = cls(root=root, train=train, transform=tfm, download=download)
        num_classes = 10
        return ds, num_classes
    elif n in {"cifar10", "cifar"}:
        ds = datasets.CIFAR10(root=root, train=train, transform=tfm, download=download)
        num_classes = 10
        return ds, num_classes
    elif n == "cifar100":
        ds = datasets.CIFAR100(root=root, train=train, transform=tfm, download=download)
        num_classes = 100
        return ds, num_classes
    elif n == "svhn":
        split = "train" if train else "test"
        ds = datasets.SVHN(root=root, split=split, transform=tfm, download=download)
        num_classes = 10
        return ds, num_classes
    elif n == "imagefolder":
        # 期待ディレクトリ構成:
        # root/train/<class>/*, root/val/<class>/*
        sub = "train" if train else "val"
        path = os.path.join(root, sub)
        ds = datasets.ImageFolder(path, transform=tfm)
        num_classes = len(ds.classes)
        return ds, num_classes
    else:
        raise ValueError(
            f"Unknown dataset '{name}'. Use one of: "
            "MNIST, FashionMNIST, KMNIST, CIFAR10, CIFAR100, SVHN, ImageFolder"
        )

def _seed_all(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# -------- モデル切り替え周り --------
def _first_non_none(*vals):
    for v in vals:
        if v is not None: return v
    return None


def _build_model(model_name: str, num_classes: int, pretrained: bool) -> nn.Module:
    """
    torchvision の主要な分類モデルを名称で生成し、分類ヘッドを num_classes に付け替える。
    """
    name = model_name.lower()

    # 動的に“存在するものだけ”登録（古い torchvision でも安全）
    CANDIDATES = {}
    def add(key, fn_name, weights_enum_name):
        if hasattr(models, fn_name):
            fn = getattr(models, fn_name)
            weights_enum = getattr(models, weights_enum_name, None)
            CANDIDATES[key] = (fn, weights_enum)

    add("resnet18", "resnet18", "ResNet18_Weights")
    add("resnet34", "resnet34", "ResNet34_Weights")
    add("resnet50", "resnet50", "ResNet50_Weights")
    add("resnet101", "resnet101", "ResNet101_Weights")
    add("resnext50_32x4d", "resnext50_32x4d", "ResNeXt50_32X4D_Weights")
    add("wide_resnet50_2", "wide_resnet50_2", "Wide_ResNet50_2_Weights")
    add("mobilenet_v3_small", "mobilenet_v3_small", "MobileNet_V3_Small_Weights")
    add("mobilenet_v3_large", "mobilenet_v3_large", "MobileNet_V3_Large_Weights")
    add("efficientnet_b0", "efficientnet_b0", "EfficientNet_B0_Weights")
    add("efficientnet_b1", "efficientnet_b1", "EfficientNet_B1_Weights")
    add("efficientnet_b2", "efficientnet_b2", "EfficientNet_B2_Weights")
    add("efficientnet_b3", "efficientnet_b3", "EfficientNet_B3_Weights")
    add("convnext_tiny", "convnext_tiny", "ConvNeXt_Tiny_Weights")
    add("convnext_small", "convnext_small", "ConvNeXt_Small_Weights")
    add("vit_b_16", "vit_b_16", "ViT_B_16_Weights")
    add("swin_t_v2", "swin_t_v2", "Swin_T_V2_Weights")

    if name not in CANDIDATES:
        available = ", ".join(sorted(CANDIDATES.keys()))
        raise ValueError(f"Unknown/unsupported model '{model_name}'. Available: {available}")

    fn, weights_enum = CANDIDATES[name]

    # weights 引数が使える場合は DEFAULT→V2→V1 の順で選択。古いAPIなら pretrained=bool を使う
    weights = None
    if pretrained and weights_enum is not None:
        weights = _first_non_none(
            getattr(weights_enum, "DEFAULT", None),
            getattr(weights_enum, "IMAGENET1K_V2", None),
            getattr(weights_enum, "IMAGENET1K_V1", None),
        )

    try:
        sig = inspect.signature(fn)
        if "weights" in sig.parameters:
            model = fn(weights=weights)
        elif "pretrained" in sig.parameters:
            model = fn(pretrained=pretrained)
        else:
            model = fn()
    except Exception:
        model = fn(pretrained=pretrained) if pretrained else fn()

    # 分類ヘッドの付け替え（モデルごとに場所が違う）
    def _replace_linear(module: nn.Module, attr_path: str):
        # attr_path 例: "fc", "classifier.3", "heads.head", "classifier.2", "head"
        obj = module
        parts = attr_path.split(".")
        for p in parts[:-1]:
            obj = getattr(obj, p)
        last = parts[-1]
        old: nn.Module = getattr(obj, last)
        in_features = old.in_features if isinstance(old, nn.Linear) else None
        if in_features is None:
            # MobileNet/EfficientNet/ConvNeXt など Sequential の末尾を探す
            if hasattr(obj, last) and isinstance(old, nn.Sequential) and len(old) > 0:
                # 末尾の Linear を探す
                for i in range(len(old)-1, -1, -1):
                    if isinstance(old[i], nn.Linear):
                        in_features = old[i].in_features
                        old[i] = nn.Linear(in_features, num_classes)
                        setattr(obj, last, old)
                        return
            raise RuntimeError(f"Could not find Linear layer at '{attr_path}' for {model_name}")
        setattr(obj, last, nn.Linear(in_features, num_classes))

    if name.startswith("resnet") or name.startswith("resnext") or name.startswith("wide_resnet"):
        _replace_linear(model, "fc")
    elif name.startswith("mobilenet"):
        # classifier[-1] が Linear
        _replace_linear(model, "classifier")
    elif name.startswith("efficientnet"):
        _replace_linear(model, "classifier")
    elif name.startswith("convnext"):
        _replace_linear(model, "classifier")
    elif name.startswith("vit"):
        # torchvision ViT は heads.head が最終線形
        if hasattr(model, "heads") and hasattr(model.heads, "head"):
            in_features = model.heads.head.in_features
            model.heads.head = nn.Linear(in_features, num_classes)
        else:
            _replace_linear(model, "heads")
    elif name.startswith("swin"):
        # Swin は head
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Classifier replacement not implemented for '{model_name}'")

    return model

def _top1_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()
