"""
TextImageDataset — 统一的图文对数据集，支持以下三种数据来源：

1. **COCO Captions（推荐评测/小规模训练）**
   目录结构：
       data_root/
           train2017/                       # 图像
           val2017/
           annotations/captions_train2017.json
           annotations/captions_val2017.json

2. **通用 CSV / JSONL（推荐 CC3M / CC12M / 自定义数据）**
   - CSV: 每行 `image_path,caption`（支持表头）
   - JSONL: 每行 `{"image": "...", "caption": "..."}`
   image_path 可以是相对 root 的相对路径，或绝对路径。

3. **Dummy fallback**：当数据路径不存在时回退到随机张量 + 固定文本，
   便于在没有数据的环境下做 smoke test（与 ImageLabelDataset 行为一致）。

返回值（每个 sample）：
    image:           [3, H, W]  Tensor（已归一化）
    input_ids:       [L]        LongTensor，由 tokenizer 处理后定长
    attention_mask:  [L]        LongTensor（1 = 有效 token，0 = padding）

约定：tokenizer 在 Dataset 构造时传入并复用，避免每次 __getitem__ 都重新初始化。
"""

from __future__ import annotations

import csv
import json
import os
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# ============================================================================
# Helpers
# ============================================================================

_DUMMY_CAPTIONS = [
    "a photo of a cat sitting on a sofa",
    "a dog running through a green field",
    "a beautiful sunset over the ocean",
    "a person riding a bicycle on a city street",
    "a slice of pizza on a wooden table",
    "an astronaut floating in outer space",
    "a red car parked in front of a building",
    "a cup of coffee next to a notebook",
]


def _default_transform(input_size: int = 256, train: bool = True) -> transforms.Compose:
    """与 train.py 中 ImageNet 流程保持一致的默认 transform。"""
    if train:
        return transforms.Compose([
            transforms.Resize(int(input_size * 1.1)),
            transforms.RandomCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize(int(input_size * 1.1)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# ============================================================================
# TextImageDataset
# ============================================================================

class TextImageDataset(Dataset):
    """
    Args:
        root:           数据集根目录
        format:         'coco' | 'csv' | 'jsonl' | 'auto'
                        'auto' 会按 root 下的文件自动探测
        split:          'train' | 'val' （仅 COCO 模式有意义）
        tokenizer:      HuggingFace tokenizer 实例（必填）
        max_text_len:   tokenize 后的最大长度（统一 padding 到该长度）
        transform:      torchvision transform（None 则使用默认）
        max_samples:    最多加载 N 条（调试用）
        captions_per_image: COCO 每图 5 caption，是否随机选 1 条
                        - True: 训练时随机
                        - False: 取第一条（确定性，用于 val）
    """

    def __init__(
        self,
        root: str,
        format: str = "auto",
        split: str = "train",
        tokenizer: Optional[Any] = None,
        max_text_len: int = 77,
        transform: Optional[Callable] = None,
        max_samples: Optional[int] = None,
        captions_per_image: bool = True,
        input_size: int = 256,
    ):
        if tokenizer is None:
            raise ValueError("TextImageDataset requires a HuggingFace tokenizer instance.")

        self.root = root
        self.split = split
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        self.captions_per_image = captions_per_image
        self.transform = transform or _default_transform(input_size, train=(split == "train"))

        self._items: List[Tuple[str, Any]] = []   # (image_path, caption_or_caption_list)
        self._is_dummy = False

        # 自动探测格式
        fmt = format
        if fmt == "auto":
            fmt = self._detect_format(root, split)

        if fmt == "coco":
            self._load_coco(root, split)
        elif fmt == "csv":
            self._load_csv(root, split)
        elif fmt == "jsonl":
            self._load_jsonl(root, split)
        elif fmt == "dummy":
            self._is_dummy = True
            self._dummy_len = max_samples or 1024
        else:
            raise ValueError(f"Unsupported dataset format: {fmt}")

        if max_samples and not self._is_dummy and max_samples < len(self._items):
            self._items = self._items[:max_samples]

    # ------------------------------------------------------------------
    # Format detection
    # ------------------------------------------------------------------
    @staticmethod
    def _detect_format(root: str, split: str) -> str:
        """根据目录结构自动判断格式；找不到则回退 dummy。"""
        if not root or not os.path.isdir(root):
            return "dummy"

        # COCO: annotations/captions_{split}{year}.json
        ann_dir = os.path.join(root, "annotations")
        if os.path.isdir(ann_dir):
            for fn in os.listdir(ann_dir):
                if fn.startswith(f"captions_{split}") and fn.endswith(".json"):
                    return "coco"

        # CSV / JSONL: 同名文件 split.{csv|jsonl}
        for ext in ("csv", "jsonl", "tsv"):
            candidate = os.path.join(root, f"{split}.{ext}")
            if os.path.exists(candidate):
                return "jsonl" if ext == "jsonl" else "csv"

        return "dummy"

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------
    def _load_coco(self, root: str, split: str) -> None:
        """COCO Captions JSON 格式：每图 5 条 caption。"""
        ann_dir = os.path.join(root, "annotations")
        ann_path = None
        for fn in os.listdir(ann_dir):
            if fn.startswith(f"captions_{split}") and fn.endswith(".json"):
                ann_path = os.path.join(ann_dir, fn)
                break
        if ann_path is None:
            raise FileNotFoundError(f"No COCO captions JSON found for split={split} in {ann_dir}")

        with open(ann_path, "r") as f:
            data = json.load(f)

        # image_id -> file_name
        id2file: Dict[int, str] = {img["id"]: img["file_name"] for img in data["images"]}
        # image_id -> [captions]
        id2caps: Dict[int, List[str]] = {}
        for ann in data["annotations"]:
            id2caps.setdefault(ann["image_id"], []).append(ann["caption"].strip())

        # 推断图像目录：annotations 文件名末尾如 captions_train2017.json → train2017/
        image_dir_name = Path(ann_path).stem.split("_", 1)[1]   # 'train2017'
        image_dir = os.path.join(root, image_dir_name)

        for img_id, fname in id2file.items():
            caps = id2caps.get(img_id)
            if not caps:
                continue
            self._items.append((os.path.join(image_dir, fname), caps))

        print(f"[TextImageDataset:coco] loaded {len(self._items)} images "
              f"({sum(len(c) for _, c in self._items)} captions) from {ann_path}")

    def _load_csv(self, root: str, split: str) -> None:
        """CSV 格式：image_path,caption（首行可为表头）。"""
        path = None
        for ext in ("csv", "tsv"):
            candidate = os.path.join(root, f"{split}.{ext}")
            if os.path.exists(candidate):
                path = candidate
                delim = "," if ext == "csv" else "\t"
                break
        if path is None:
            raise FileNotFoundError(f"No CSV/TSV found for split={split} in {root}")

        with open(path, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=delim)
            rows = list(reader)

        # 跳过表头（如果首行不是合法路径就当表头）
        if rows and not self._looks_like_path(rows[0][0]):
            rows = rows[1:]

        for row in rows:
            if len(row) < 2:
                continue
            img_rel, cap = row[0].strip(), row[1].strip()
            if not img_rel or not cap:
                continue
            img_path = img_rel if os.path.isabs(img_rel) else os.path.join(root, img_rel)
            self._items.append((img_path, cap))

        print(f"[TextImageDataset:csv] loaded {len(self._items)} (image, caption) pairs from {path}")

    def _load_jsonl(self, root: str, split: str) -> None:
        """JSONL 格式：每行 {'image': ..., 'caption': ...}。"""
        path = os.path.join(root, f"{split}.jsonl")
        if not os.path.exists(path):
            raise FileNotFoundError(f"JSONL not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                img_rel = obj.get("image") or obj.get("image_path") or obj.get("file_name")
                cap = obj.get("caption") or obj.get("text")
                if not img_rel or not cap:
                    continue
                img_path = img_rel if os.path.isabs(img_rel) else os.path.join(root, img_rel)
                self._items.append((img_path, cap.strip()))

        print(f"[TextImageDataset:jsonl] loaded {len(self._items)} pairs from {path}")

    @staticmethod
    def _looks_like_path(s: str) -> bool:
        return ("/" in s or "\\" in s or s.lower().endswith(
            (".jpg", ".jpeg", ".png", ".webp", ".bmp")))

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return self._dummy_len if self._is_dummy else len(self._items)

    def _tokenize(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        enc = self.tokenizer(
            text,
            max_length=self.max_text_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return enc["input_ids"][0], enc["attention_mask"][0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self._is_dummy:
            image = torch.rand(3, 256, 256)
            caption = _DUMMY_CAPTIONS[idx % len(_DUMMY_CAPTIONS)]
        else:
            img_path, caps = self._items[idx]
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception:
                # 损坏图像 → 退回随机张量 + dummy caption（避免训练中断）
                image = torch.rand(3, 256, 256)
                caps = _DUMMY_CAPTIONS[idx % len(_DUMMY_CAPTIONS)]

            if not isinstance(image, torch.Tensor):
                image = self.transform(image)

            if isinstance(caps, list):
                caption = random.choice(caps) if self.captions_per_image else caps[0]
            else:
                caption = caps

        input_ids, attn_mask = self._tokenize(caption)

        # 用 dict 返回：避免与 (image, label) 元组混淆，且方便 DataLoader 自动 collate
        return {
            "image": image,
            "input_ids": input_ids.long(),
            "attention_mask": attn_mask.long(),
        }


# ============================================================================
# Factory
# ============================================================================

def build_text_image_dataset(
    root: str,
    split: str,
    tokenizer: Any,
    *,
    format: str = "auto",
    max_text_len: int = 77,
    input_size: int = 256,
    max_samples: Optional[int] = None,
    captions_per_image: Optional[bool] = None,
) -> TextImageDataset:
    """统一入口；split 默认决定 caption 选取策略。"""
    if captions_per_image is None:
        captions_per_image = (split == "train")
    return TextImageDataset(
        root=root,
        format=format,
        split=split,
        tokenizer=tokenizer,
        max_text_len=max_text_len,
        transform=_default_transform(input_size, train=(split == "train")),
        max_samples=max_samples,
        captions_per_image=captions_per_image,
        input_size=input_size,
    )
