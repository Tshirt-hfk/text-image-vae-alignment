"""
Label-conditioned encoder → N(μ, σ²) [B, h, w, D].

EmbeddingLabelEncoder：int label → (μ, logσ)  (轻量、无 Transformer)

输出形状 [B, h, w, latent_dim] 与 VAE encoder 完全对齐，
便于 AlignmentVAE 上层做 KL 对齐而无需感知条件类型。

与 model.text_encoder.SpatialTextEncoder 是平行关系：
  - cond_type='label' → 使用本文件的 EmbeddingLabelEncoder
  - cond_type='text'  → 使用 model.text_encoder 的 SpatialTextEncoder
"""

from typing import Tuple

import torch
import torch.nn as nn


class EmbeddingLabelEncoder(nn.Module):
    """
    轻量级标签编码器：直接用 nn.Embedding 把 label 映射到每个空间位置的
    高斯分布参数 (μ, logσ)，无 Transformer。

    label (int)
        ├── mu_table        [num_classes, h*w*latent_dim]  → reshape → [B, h, w, D]
        └── logsigma_table  同上

    适用场景：
      - ImageNet 类别条件训练（默认）
      - 数据量小或 num_classes 少时的快速 baseline
      - 验证"对齐 KL"在不依赖大型条件编码器情况下的有效性
    """

    def __init__(self, input_size: int = 256, patch_size: int = 16,
                 num_classes: int = 1000, latent_dim: int = 16,
                 init_logsigma: float = -2.0):
        super().__init__()

        self.hw = input_size // patch_size
        self.num_patches = self.hw * self.hw
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        per_label_dim = self.num_patches * latent_dim

        # 两个独立 Embedding 表分别保存 μ 和 logσ
        self.mu_table = nn.Embedding(num_classes + 1, per_label_dim)
        self.logsigma_table = nn.Embedding(num_classes + 1, per_label_dim)

        # μ ≈ 0、logσ ≈ init_logsigma：与 VAE encoder 初始 σ≈1 对称，
        # KL 初值最小、无方向偏置
        nn.init.normal_(self.mu_table.weight, std=0.02)
        nn.init.constant_(self.logsigma_table.weight, init_logsigma)

    def forward(self, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            labels: [B] integer class labels
        Returns:
            mu, logsigma: [B, h, w, latent_dim]
        """
        B = labels.shape[0]
        h = w = self.hw
        D = self.latent_dim

        mu = self.mu_table(labels).reshape(B, h, w, D)
        logsigma = self.logsigma_table(labels).reshape(B, h, w, D)
        logsigma = torch.clamp(logsigma, min=-5.0, max=2.0)
        return mu, logsigma

    def sample(self, mu: torch.Tensor, logsigma: torch.Tensor) -> torch.Tensor:
        """Reparameterization: z = μ + σ·ε."""
        return mu + torch.exp(logsigma) * torch.randn_like(mu)
