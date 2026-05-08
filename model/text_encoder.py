"""
Spatial Label Encoder → N(μ, σ²) [B, h, w, D].

Architecture: label → LabelEmbedder → conditioning
    learnable spatial tokens → AdaLN Transformer (RoPE, SwiGLU, in-context injection)
    → μ_head, logσ_head → spatial Gaussian matching VAE latent shape.
"""

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from util.model_util import VisionRotaryEmbeddingFast, get_2d_sincos_pos_embed, RMSNorm


# ============================================================================
# Transformer Components
# ============================================================================

def modulate(x, shift, scale):
    """AdaLN modulation: x * (1 + scale) + shift."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class LabelEmbedder(nn.Module):
    """Class label → vector embedding."""

    def __init__(self, num_classes: int, hidden_size: int):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes + 1, hidden_size)
        self.num_classes = num_classes

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        return self.embedding_table(labels)


class Attention(nn.Module):
    """Multi-head self-attention with RoPE and QK-norm."""

    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_norm=True,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.q_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rope):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        q, k = self.q_norm(q), self.k_norm(k)
        q, k = rope(q), rope(k)

        # Use PyTorch SDPA (auto-selects FlashAttention / xformers / math backend)
        drop_p = self.attn_drop.p if self.training else 0.0
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=drop_p)

        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward network."""

    def __init__(self, dim: int, hidden_dim: int, drop=0.0, bias=True):
        super().__init__()
        hidden_dim = int(hidden_dim * 2 / 3)
        self.w12 = nn.Linear(dim, 2 * hidden_dim, bias=bias)
        self.w3 = nn.Linear(hidden_dim, dim, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        return self.w3(self.drop(F.silu(x1) * x2))


class AdaLNBlock(nn.Module):
    """Transformer block with adaLN conditioning, RoPE attention, SwiGLU FFN."""

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True,
                              qk_norm=True, attn_drop=attn_drop, proj_drop=proj_drop)
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        self.mlp = SwiGLUFFN(hidden_size, int(hidden_size * mlp_ratio), drop=proj_drop)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 4 * hidden_size, bias=True),
        )

    def forward(self, x, c, feat_rope=None):
        shift_msa, scale_msa, shift_mlp, scale_mlp = self.adaLN_modulation(c).chunk(4, dim=-1)
        x = x + self.attn(modulate(self.norm1(x), shift_msa, scale_msa), rope=feat_rope)
        x = x + self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


# ============================================================================
# Spatial Label Encoder
# ============================================================================

class SpatialLabelEncoder(nn.Module):
    """
    label → LabelEmbedder → conditioning
    learnable spatial tokens + pos_embed → AdaLN blocks → reshape → (μ, logσ)
    Output: [B, h, w, latent_dim] matching VAE encoder output.
    """

    def __init__(self, input_size: int = 256, patch_size: int = 16,
                 num_classes: int = 1000, hidden_size: int = 768,
                 latent_dim: int = 16, depth: int = 12, num_heads: int = 12,
                 mlp_ratio: float = 4.0, attn_drop: float = 0.0,
                 proj_drop: float = 0.0, in_context_len: int = 32,
                 in_context_start: int = 4, init_logsigma: float = -2.0):
        super().__init__()

        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.in_context_len = in_context_len
        self.in_context_start = in_context_start
        self.hw = input_size // patch_size
        self.num_patches = self.hw * self.hw

        # Label embedding
        self.y_embedder = LabelEmbedder(num_classes, hidden_size)

        # Learnable spatial tokens + fixed sin-cos pos embed
        self.spatial_tokens = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size))
        nn.init.normal_(self.spatial_tokens, std=0.02)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, hidden_size), requires_grad=False)

        # In-context label injection tokens
        if in_context_len > 0:
            self.in_context_posemb = nn.Parameter(
                torch.zeros(1, in_context_len, hidden_size), requires_grad=True)
            nn.init.normal_(self.in_context_posemb, std=0.02)

        # RoPE (with / without prefix tokens)
        half_head = hidden_size // num_heads // 2
        self.feat_rope = VisionRotaryEmbeddingFast(
            dim=half_head, pt_seq_len=self.hw, num_cls_token=0)
        self.feat_rope_incontext = VisionRotaryEmbeddingFast(
            dim=half_head, pt_seq_len=self.hw, num_cls_token=in_context_len)

        # Transformer blocks (dropout in middle layers only)
        self.blocks = nn.ModuleList([
            AdaLNBlock(
                hidden_size, num_heads, mlp_ratio=mlp_ratio,
                attn_drop=attn_drop if (depth // 4 <= i < depth // 4 * 3) else 0.0,
                proj_drop=proj_drop if (depth // 4 <= i < depth // 4 * 3) else 0.0,
            ) for i in range(depth)
        ])

        # Gaussian output heads
        self.norm_final = RMSNorm(hidden_size)
        self.mu_head = nn.Linear(hidden_size, latent_dim)
        self.logsigma_head = nn.Linear(hidden_size, latent_dim)

        self._initialize_weights(init_logsigma)

    def _initialize_weights(self, init_logsigma: float):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Sin-cos positional embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], self.hw)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Zero-out adaLN modulation
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Gaussian heads: zero mu, logsigma bias = init_logsigma
        nn.init.zeros_(self.mu_head.weight)
        nn.init.zeros_(self.mu_head.bias)
        nn.init.zeros_(self.logsigma_head.weight)
        nn.init.constant_(self.logsigma_head.bias, init_logsigma)

    def forward(self, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            labels: [B] integer class labels
        Returns:
            mu, logsigma: [B, h, w, latent_dim]
        """
        B = labels.shape[0]
        c = self.y_embedder(labels)
        x = self.spatial_tokens.expand(B, -1, -1) + self.pos_embed

        for i, block in enumerate(self.blocks):
            if self.in_context_len > 0 and i == self.in_context_start:
                ic_tokens = c.unsqueeze(1).expand(-1, self.in_context_len, -1)
                x = torch.cat([ic_tokens + self.in_context_posemb, x], dim=1)
            rope = self.feat_rope if i < self.in_context_start else self.feat_rope_incontext
            x = block(x, c, rope)

        if self.in_context_len > 0:
            x = x[:, self.in_context_len:]

        x = self.norm_final(x)
        mu = self.mu_head(x)
        logsigma = torch.clamp(self.logsigma_head(x), min=-5.0, max=2.0)

        h = w = self.hw
        return mu.reshape(B, h, w, -1), logsigma.reshape(B, h, w, -1)

    def sample(self, mu: torch.Tensor, logsigma: torch.Tensor) -> torch.Tensor:
        """Reparameterization: z = μ + σ·ε."""
        return mu + torch.exp(logsigma) * torch.randn_like(mu)


# ============================================================================
# Embedding-only Label Encoder (Transformer-free baseline)
# ============================================================================

class EmbeddingLabelEncoder(nn.Module):
    """
    轻量级标签编码器：跳过 Transformer，直接用 nn.Embedding 把 label 映射到
    每个空间位置的高斯分布参数 (μ, logσ)。

    label (int)
        ├── mu_table   [num_classes, h*w*latent_dim]  → reshape → [B, h, w, D]
        └── logsigma_table  同上

    适用场景：
      - 与基于 Transformer 的 SpatialLabelEncoder 做消融对比
      - 数据量小或 num_classes 少时的更快 baseline
      - 验证"对齐 KL"是否真的需要表达力强的标签编码器
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

        # 初始化：μ ≈ 0、logσ ≈ init_logsigma，与 SpatialLabelEncoder 的初始
        # 输出（mu_head 全零、logsigma_head bias=init_logsigma）保持一致，
        # 保证两种编码器在训练初期具有相近的起点。
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


# ============================================================================
# Factory functions
# ============================================================================

def SpatialLabelEncoder_B(input_size=256, num_classes=1000, latent_dim=16, **kw):
    """Base — 6 layers, 768 hidden, 12 heads."""
    return SpatialLabelEncoder(
        input_size=input_size, patch_size=16, num_classes=num_classes,
        hidden_size=768, latent_dim=latent_dim, depth=6, num_heads=12,
        in_context_len=32, in_context_start=4, **kw)

def SpatialLabelEncoder_L(input_size=256, num_classes=1000, latent_dim=16, **kw):
    """Large — 12 layers, 1024 hidden, 16 heads."""
    return SpatialLabelEncoder(
        input_size=input_size, patch_size=16, num_classes=num_classes,
        hidden_size=1024, latent_dim=latent_dim, depth=12, num_heads=16,
        in_context_len=32, in_context_start=8, **kw)

def SpatialLabelEncoder_H(input_size=256, num_classes=1000, latent_dim=16, **kw):
    """Huge — 16 layers, 1280 hidden, 16 heads."""
    return SpatialLabelEncoder(
        input_size=input_size, patch_size=16, num_classes=num_classes,
        hidden_size=1280, latent_dim=latent_dim, depth=16, num_heads=16,
        in_context_len=32, in_context_start=10, **kw)


def EmbeddingLabelEncoder_E(input_size=256, num_classes=1000, latent_dim=16, **kw):
    """Embedding-only baseline — 无 Transformer，直接查表得高斯参数。"""
    # 只接受 EmbeddingLabelEncoder 真正用到的 kwargs，过滤掉 build_model
    # 透传过来的 hidden_size/depth/num_heads 等 Transformer 专属参数。
    accepted = {k: v for k, v in kw.items() if k in ('init_logsigma',)}
    return EmbeddingLabelEncoder(
        input_size=input_size, patch_size=16, num_classes=num_classes,
        latent_dim=latent_dim, **accepted)


SpatialLabelEncoder_models = {
    'SLE-B': SpatialLabelEncoder_B,
    'SLE-L': SpatialLabelEncoder_L,
    'SLE-H': SpatialLabelEncoder_H,
    'SLE-E': EmbeddingLabelEncoder_E,   # E = Embedding-only baseline
}
