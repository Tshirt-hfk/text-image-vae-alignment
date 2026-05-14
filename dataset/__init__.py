"""Dataset utilities for AlignmentVAE.

  - ImageLabelDataset / build_image_label_dataset：ImageNet 等离散类别条件训练（L2I）
  - TextImageDataset  / build_text_image_dataset ：COCO / CSV / JSONL 图文对训练（T2I）
"""

from .label_image_dataset import ImageLabelDataset, build_image_label_dataset
from .text_image_dataset import TextImageDataset, build_text_image_dataset

__all__ = [
    "ImageLabelDataset", "build_image_label_dataset",
    "TextImageDataset", "build_text_image_dataset",
]
