"""
DETR に OT ベースの混合精度量子化（重みのみ）をそのまま流し込める汎用ラッパー。

ポイント:
- 既存の `ot_allocate_bits_for_model(model, dl, loss_fn, device, cfg)` のインターフェースを変えずに DETR を扱えるよう
  モデル側で "loss を返す" ラッパーを被せ、`loss_fn=lambda loss, _ : loss` にする。
- torchvision DETR を優先採用。見つからない場合は Hugging Face DETR で同様に動くラッパーを用意
- 校正データは COCO があれば理想だが、最小動作のためにランダム矩形の合成データセットも同梱。

前提:
- あなたの量子化ユーティリティは `import q` として参照できる（例: q.OTQuantConfig, q.apply_weight_quantization_inplace など）。
- `ot_allocate_bits_for_model` は DataLoader から (inputs, targets) を受け取り、
  `pred = model(inputs)` として forward した出力と targets を `loss_fn(pred, targets)` に渡す想定。
  → 本ファイルでは `model(inputs)` が **スカラー損失** を返すため、`loss_fn` は出力だけを使って返す。

使い方（最小・合成データでの例）:
    python detr_ot_quant_wrapper.py  # そのまま実行で合成データで感度推定→量子化まで走ります

COCO を使う場合（例）:
    python detr_ot_quant_wrapper.py \
        --coco-img /path/to/coco/val2017 \
        --coco-ann /path/to/annotations/instances_val2017.json

Hugging Face DETR を使う場合（torchvision に DETR がない環境など）:
    python detr_ot_quant_wrapper.py --use-hf

注意:
- DETR の loss は学習モードで targets を与えた時のみ計算されます。`ot_allocate_bits_for_model` が eval() を強制する実装でも、
  本ラッパーは forward 内で base.train() をセットして loss パスを通します。
- 位置埋め込みや LayerNorm 等は低ビットで不安定になりやすいため、量子化ユーティリティ側に exclude フィルタがあれば活用してください。
"""

from __future__ import annotations

import argparse
import random
from typing import List, Tuple, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader,random_split
from torchvision import transforms

import libquantum as q
import HAWQ2_fisher as h2
import sinkhorn_fisher as ot

from torchvision.datasets import CocoDetection
import timm 
from timm.models import create_model 

# -----------------------------
# 合成 DETR 校正データセット
# -----------------------------
class RandomDetrCalibDataset(Dataset):
    """DETR 形式 (image, target) を返す合成データ。

    image: Tensor[C,H,W], 値域[0,1]
    target: {"boxes": Tensor[n,4] (x0,y0,x1,y1, 画素座標),
             "labels": Tensor[n], dtype long,
             "image_id": Tensor[1]}
    """

    def __init__(
        self,
        size: int = 32,
        image_size: Tuple[int, int] = (800, 800),
        num_classes: int = 91,
        max_boxes: int = 5,
        min_box_size: int = 40,
        transform: transforms.Compose | None = None,
    ) -> None:
        self.size = size
        self.H, self.W = image_size
        self.num_classes = num_classes
        self.max_boxes = max_boxes
        self.min_box = min_box_size
        self.transform = transform

    def __len__(self) -> int:
        return self.size

    def _rand_box(self) -> Tuple[float, float, float, float]:
        W, H = self.W, self.H
        w = random.randint(self.min_box, max(self.min_box, W // 3))
        h = random.randint(self.min_box, max(self.min_box, H // 3))
        x0 = random.randint(0, max(1, W - w - 1))
        y0 = random.randint(0, max(1, H - h - 1))
        x1 = x0 + w
        y1 = y0 + h
        return float(x0), float(y0), float(x1), float(y1)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        img = torch.rand(3, self.H, self.W)
        n = random.randint(1, self.max_boxes)
        boxes = [self._rand_box() for _ in range(n)]
        labels = [random.randint(1, self.num_classes - 1) for _ in range(n)]
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([idx], dtype=torch.int64),
        }
        if self.transform is not None:
            # 画像だけ簡単に変換（必要なら拡張）
            img = self.transform(img)
        return img, target


def detr_collate_fn(batch: List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]):
    images, targets = list(zip(*batch))
    # DataLoader -> (inputs, dummy_target)
    # 既存の量子化関数が (x,y) を期待する想定に合わせ、y は未使用ダミー
    return (list(images), list(targets)), torch.tensor(0)


# -----------------------------
# DETR 構築（torchvision 優先 / HF 代替）
# -----------------------------

# def build_torchvision_detr(pretrained: bool = True) -> nn.Module | None:
#     try:
#         #from torchvision.models.detection import detr_resnet50, Detr_ResNet50_Weights
#         #weights = Detr_ResNet50_Weights.DEFAULT if pretrained else None
#         #model = detr_resnet50(weights=weights)
#         model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)
#         return model
#     except Exception:
#         return None

class DETRWithLoss(nn.Module):
    """torchvision DETR を loss スカラーを返す形でラップ。
    forward(inputs) で **合計損失のスカラー Tensor** を返す。
    """

    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        self.base = base

    def forward(self, batch: Tuple[List[torch.Tensor], List[Dict[str, torch.Tensor]]]):
        images, targets = batch
        # torchvision DETR の損失計算は train モード・targets 必須
        self.base.train()
        losses: Dict[str, torch.Tensor] = self.base(images, targets)  # type: ignore
        # よく使う損失キーのみ合算（存在するものだけ）
        keys = [k for k in ["loss_ce", "loss_bbox", "loss_giou"] if k in losses]
        if not keys:
            # 万一キー名が異なる場合は全部足す
            loss = sum(losses.values())
        else:
            loss = sum(losses[k] for k in keys)
        return loss


# --- Hugging Face 代替（必要時のみ） ---

def build_hf_detr_wrapper(device: torch.device):
    """Hugging Face DETR を同じく loss スカラーを返す nn.Module にラップ。
    依存: transformers>=4.30 目安。画像前処理も内部で実施。
    """
    from transformers import DetrForObjectDetection, DetrImageProcessor
    from torchvision.transforms import ToPILImage
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").to(device)
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

    class HFDetrWithLoss(nn.Module):
        def __init__(self, model, processor, device):
            super().__init__()
            self.model = model
            self.processor = processor
            self.device = device
            self.to_pil = ToPILImage()

        def forward(self, batch):
            images, targets = batch
            # HF の processor は COCO 風 annotations を期待
            ann_list = []
            for i, t in enumerate(targets):
                boxes = t["boxes"].tolist()
                labels = t["labels"].tolist()
                anns = []
                for (x0, y0, x1, y1), c in zip(boxes, labels):
                    w = max(1.0, x1 - x0)
                    h = max(1.0, y1 - y0)
                    anns.append({
                        "bbox": [float(x0), float(y0), float(w), float(h)],
                        "category_id": int(c),
                        "area": float(w * h),
                        "iscrowd": 0,
                    })
                ann_list.append({"image_id": int(t["image_id"][0].item()), "annotations": anns})
            pil_imgs = [self.to_pil(img.cpu()) for img in images]
            inputs = self.processor(images=pil_imgs, annotations=ann_list, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            self.model.train()
            out = self.model(**inputs)
            return out.loss

    return HFDetrWithLoss(model, processor, device)


def coco_to_detr(item):
            img, anns = item
            # COCO → DETR の (boxes, labels)
            boxes, labels = [], []
            for a in anns:
                x, y, w, h = a["bbox"]
                boxes.append([x, y, x + w, y + h])
                labels.append(a["category_id"])
            target = {
                "boxes": torch.tensor(boxes, dtype=torch.float32),
                "labels": torch.tensor(labels, dtype=torch.int64),
                "image_id": torch.tensor([0], dtype=torch.int64),
            }
            img = transforms.PILToTensor()(img).float() / 255.0
            return img, target

class CocoAsDetr(Dataset):
    def __init__(self, base, size):
        self.base = base
        self.size = min(size, len(base))
    def __len__(self):
        return self.size
    def __getitem__(self, i):
        return coco_to_detr(self.base[i])
    
def make_train_eval_coco(args, train_ratio=0.9):
    """
    args.coco_img : COCO images path
    args.coco_ann : COCO annotations path
    args.calib_size : optional, max size for calibration dataset
    """
    # COCO detection dataset
    base_ds = CocoDetection(args.coco_img, args.coco_ann, transforms=None)
    # --- Train/Eval Split ---
    total = len(base_ds)
    train_size = int(total * train_ratio)
    eval_size = total - train_size

    base_train_ds, base_eval_ds = random_split(base_ds, [train_size, eval_size])

    # --- Wrap each with CocoAsDetr ---
    train_ds = CocoAsDetr(base_train_ds)
    eval_ds  = CocoAsDetr(base_eval_ds)

    return train_ds, eval_ds
    
# 出力がスカラー損失なので loss_fn はそれをそのまま返す
def loss_fn(loss_scalar: torch.Tensor, _dummy_target: Any) -> torch.Tensor:
    return loss_scalar

def report(result,cfg):
    print("Assigned counts per bits:", {b: result.target_counts[i] for i, b in enumerate(cfg.bits)})
    show_n = min(20, len(result.assignment.layer_names))
    pairs = [ (result.assignment.layer_names[i], cfg.bits[result.assignment.bit_indices[i]])  for i in range(show_n)   ]
    print(f"First {show_n} layer -> bit:")
    for name, bit in pairs:
        print(f"  {name:60s} -> {bit}bit")
# -----------------------------
# メイン: 量子化割当て → 重み量子化
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--use-hf", action="store_true", help="torchvision DETR が使えない場合に Hugging Face DETR を使う")
    ap.add_argument("--coco-img", type=str, default=None, help="COCO images ディレクトリ (val2017 など)")
    ap.add_argument("--coco-ann", type=str, default=None, help="COCO annotations JSON (instances_val2017.json など)")
    ap.add_argument("--calib-size", type=int, default=32)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--avg_bits", type=float, default=6.0)
    ap.add_argument("--bits", type=int, nargs="*", default=[2, 4, 6, 8])
    ap.add_argument("--sinkhorn-iters", type=int, default=400)
    ap.add_argument("--sens_batches", type=int, default=2)
    ap.add_argument("--out_dir",      type=str, default="./quant_results_detr")
    ap.add_argument("--basicq", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.coco_img and args.coco_ann:
        # 任意: 本番は COCO を推奨（インストール要: pycocotools）
        tfm = None  # DETR は内部でサイズ正規化するためここでは生画像でも可
        if(args.train==""):
            ds = CocoDetection(args.coco_img, args.coco_ann, transforms=tfm)
            calib_ds = CocoAsDetr(ds, args.calib_size)
            el = DataLoader(calib_ds, batch_size=args.batch, shuffle=False, collate_fn=detr_collate_fn)                        
            tl=None
        else:
            train_ds, eval_ds = make_train_eval_coco(args)
            tl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, collate_fn=detr_collate_fn)
            el = DataLoader(eval_ds, batch_size=args.batch, shuffle=False, collate_fn=detr_collate_fn)
    else:
        calib_ds = RandomDetrCalibDataset(size=args.calib_size)
        el = DataLoader(calib_ds, batch_size=args.batch, shuffle=True, collate_fn=detr_collate_fn)
        #el = DataLoader(calib_ds, batch_size=args.batch, shuffle=True)
        tl=None

    # --- モデル ---
    if not args.use_hf:
        base = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True) 
        if base is None:
            print("[WARN] torchvision DETR が見つかりません。--use-hf を付けるか torchvision を更新してください。")
            return
        model = DETRWithLoss(base).to(device)
    else:
        model = build_hf_detr_wrapper(device)
    #学習

    # --- 量子化設定 & 実行 ---
    if(args.basicq):
        cfg = q.OTQuantConfig(
            bits=args.bits,
            avg_bits=args.avg_bits,
            epsilon=0.02,
            sinkhorn_iters=args.sinkhorn_iters,
            sens_batches=args.sens_batches,
            # 量子化ユーティリティに除外フックがあるなら活用推奨:
            # exclude_types=[nn.LayerNorm, nn.Embedding],
            # exclude_names=[".*pos_embed.*", ".*query_embed.*"],
        )

        result = ot.ot_allocate_bits_for_model(model, tl, loss_fn, device, cfg)  # type: ignore[name-defined]

        # ラッパー (model) に対して量子化適用（base を直接渡すと名前がずれる場合があるため注意）
        if(args.restore):
            backup = q.apply_weight_quantization_inplace(model, result.assignment, cfg.bits, keep_original=True)
            q.restore_from_backup(model, backup)

        report(result,cfg)
    else:
        q.dumpresults(args,model_ctor=model,orgmodel=model,device=device,val_loader=el,train_loader=tl,
                     methods=["HAWQ2_fisher","OT_HAWQ_like","DiffSinkhornDynamic","SinkhornMCKPDynamic"], dump=True)

    # 参考: 推論時は model.base をそのまま使用（weights は量子化後のものに置換済み）
    # model.base.eval(); predictions = model.base([image_tensor])

if __name__ == "__main__":
    main()
