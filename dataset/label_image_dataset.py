"""
ImageLabelDataset — 图文对数据集，用于 ImageNet 等离散类别条件训练（L2I）。

数据来源：
  1. **ImageNet 风格 ImageFolder**（推荐）
     目录结构：
         data_root/
             train/<class_name>/*.jpg
             val/<class_name>/*.jpg
     由 torchvision.datasets.ImageFolder 自动按子目录名归类。

  2. **Dummy fallback**：当 `data_root/<split>` 不存在或无法解析时，
     回退到随机张量 + 循环 label，便于在没有数据的环境下做 smoke test
     （与 dataset.text_image_dataset.TextImageDataset 行为保持一致）。

返回值（每个 sample）：
    image:  [3, H, W]  Tensor
    label:  int        在 [0, num_classes) 范围内
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import torch
from torch.utils.data import Dataset
from torchvision import transforms, datasets


class ImageLabelDataset(Dataset):
    """ImageNet or dummy (image, label) dataset."""

    def __init__(self, root: str, split: str = "train", transform=None,
                 num_classes: int = 1000, max_samples: Optional[int] = None):
        self.num_classes = num_classes
        self.transform = transform or transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(256), transforms.ToTensor(),
        ])

        imagenet_path = os.path.join(root, split)
        if os.path.isdir(imagenet_path):
            try:
                self.dataset = datasets.ImageFolder(imagenet_path, transform=self.transform)
                self.use_real = True
                if max_samples and max_samples < len(self.dataset):
                    self.dataset = torch.utils.data.Subset(
                        self.dataset, list(range(max_samples))
                    )
            except Exception:
                self.use_real = False
                self.length = max_samples or 1000
        else:
            self.use_real = False
            self.length = max_samples or 1000

    def __len__(self) -> int:
        return len(self.dataset) if self.use_real else self.length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        if self.use_real:
            return self.dataset[idx]
        return torch.rand(3, 256, 256), idx % self.num_classes


def build_image_label_dataset(
    root: str,
    split: str = "train",
    transform=None,
    num_classes: int = 1000,
    max_samples: Optional[int] = None,
) -> ImageLabelDataset:
    """与 build_text_image_dataset 对应的轻量工厂，便于在 train.py 中统一调用风格。"""
    return ImageLabelDataset(
        root=root, split=split, transform=transform,
        num_classes=num_classes, max_samples=max_samples,
    )
