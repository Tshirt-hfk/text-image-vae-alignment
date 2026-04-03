"""
Shared model utilities: RoPE, positional embeddings, RMSNorm.

Provides:
    - VisionRotaryEmbeddingFast: 2-D rotary position embedding for vision transformers.
    - get_2d_sincos_pos_embed: 2-D sinusoidal-cosine fixed positional embedding.
    - RMSNorm: Root Mean Square Layer Normalization.
"""

import math
import numpy as np
import torch
import torch.nn as nn


# ============================================================================
# RMSNorm
# ============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (no centering bias)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


# ============================================================================
# 2-D Sin-Cos Positional Embedding
# ============================================================================

def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int, cls_token: bool = False) -> np.ndarray:
    """
    Generate 2D sin-cos positional embedding.

    Args:
        embed_dim: embedding dimension (must be even).
        grid_size: height (= width) of the grid.
        cls_token: if True, prepend a zero vector for the [CLS] token.

    Returns:
        pos_embed: [grid_size*grid_size, embed_dim]  (or [1 + grid_size*grid_size, embed_dim] with cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # (w, h) order
    grid = np.stack(grid, axis=0)  # [2, grid_size, grid_size]

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = _get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def _get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: np.ndarray) -> np.ndarray:
    assert embed_dim % 2 == 0
    emb_h = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # [H*W, D/2]
    emb_w = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # [H*W, D/2]
    pos_embed = np.concatenate([emb_h, emb_w], axis=1)  # [H*W, D]
    return pos_embed


def _get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    """
    embed_dim: output dimension for each position
    pos: list of positions to be encoded, shape (M,) or flattened grid
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega  # [D/2]

    pos = pos.reshape(-1)  # [M]
    out = np.einsum("m,d->md", pos, omega)  # [M, D/2]

    emb_sin = np.sin(out)  # [M, D/2]
    emb_cos = np.cos(out)  # [M, D/2]
    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # [M, D]
    return emb


# ============================================================================
# Vision Rotary Position Embedding (RoPE)
# ============================================================================

class VisionRotaryEmbeddingFast(nn.Module):
    """
    2-D Rotary Position Embedding for Vision Transformers.

    Splits the head dimension into two halves, applying separate rotary
    embeddings for height and width positions.

    Args:
        dim:           half-head dimension (head_dim // 2).
        pt_seq_len:    grid side length (number of patches per side, e.g. 16).
        num_cls_token: number of class / prefix tokens that precede spatial tokens.
    """

    def __init__(self, dim: int, pt_seq_len: int = 16, num_cls_token: int = 0):
        super().__init__()
        self.dim = dim
        self.pt_seq_len = pt_seq_len
        self.num_cls_token = num_cls_token

        # Build and cache the cos/sin tables
        freqs = self._compute_freqs(dim, pt_seq_len)
        self.register_buffer("freqs_cos", freqs.cos(), persistent=False)
        self.register_buffer("freqs_sin", freqs.sin(), persistent=False)

    @staticmethod
    def _compute_freqs(dim: int, seq_len: int) -> torch.Tensor:
        """Compute 2-D rotary frequencies for a grid of size seq_len x seq_len."""
        # 1-D frequency basis
        theta = 1.0 / (10000 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))

        # Positions along each axis
        grid_pos = torch.arange(seq_len, dtype=torch.float32)

        # Outer product → [seq_len, dim//2]
        freqs_1d = torch.einsum("n,d->nd", grid_pos, theta)

        # 2-D grid (height, width)
        freqs_h = freqs_1d.unsqueeze(1).expand(-1, seq_len, -1)  # [H, W, dim//2]
        freqs_w = freqs_1d.unsqueeze(0).expand(seq_len, -1, -1)  # [H, W, dim//2]

        # Concatenate height and width frequencies and flatten spatial dims
        freqs = torch.cat([freqs_h, freqs_w], dim=-1)  # [H, W, dim]
        freqs = freqs.reshape(seq_len * seq_len, -1)   # [H*W, dim]
        return freqs

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Apply rotary embedding to query or key tensor.

        Args:
            t: [B, num_heads, N, head_dim]  where N may include cls/prefix tokens.

        Returns:
            Tensor of same shape with rotary embedding applied.
        """
        seq_len = t.shape[2]
        rot_dim = self.freqs_cos.shape[-1]

        # If there are cls/prefix tokens, leave them unrotated
        if self.num_cls_token > 0:
            t_cls = t[:, :, :self.num_cls_token]
            t_spatial = t[:, :, self.num_cls_token:]
        else:
            t_cls = None
            t_spatial = t

        # Split last dim into rotary and pass-through parts
        t_rot = t_spatial[..., :rot_dim]
        t_pass = t_spatial[..., rot_dim:]

        # Reshape for complex rotation: [..., rot_dim] → [..., rot_dim//2, 2]
        t_rot = t_rot.reshape(*t_rot.shape[:-1], -1, 2)

        # Get the freq tables matching spatial sequence length
        spatial_len = t_spatial.shape[2]
        freqs_cos = self.freqs_cos[:spatial_len].unsqueeze(0).unsqueeze(0)  # [1, 1, N, dim]
        freqs_sin = self.freqs_sin[:spatial_len].unsqueeze(0).unsqueeze(0)

        freqs_cos = freqs_cos.reshape(*freqs_cos.shape[:-1], -1, 2)  # [..., dim//2, 2]
        freqs_sin = freqs_sin.reshape(*freqs_sin.shape[:-1], -1, 2)

        # Apply rotation
        t_rot_out = torch.stack([
            t_rot[..., 0] * freqs_cos[..., 0] - t_rot[..., 1] * freqs_sin[..., 0],
            t_rot[..., 1] * freqs_cos[..., 1] + t_rot[..., 0] * freqs_sin[..., 1],
        ], dim=-1)
        t_rot_out = t_rot_out.reshape(*t_spatial.shape[:-1], rot_dim)

        # Reassemble
        t_spatial = torch.cat([t_rot_out, t_pass], dim=-1)

        if t_cls is not None:
            t = torch.cat([t_cls, t_spatial], dim=2)
        else:
            t = t_spatial

        return t
