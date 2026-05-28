"""
ViT-VAE — pure Transformer tokenizer (ViT/JIT-style).

    Encoder: [B, 3, H, W]
        ── PatchEmbed (Conv2d k=p,s=p)  → [B, N, hidden]
        ── + 2D sin-cos PE
        ── N × Transformer (RoPE + SwiGLU)
        ── RMSNorm
        ── (mu_head | logsigma_head)    → (μ, logσ) [B, h, w, latent_dim]

    Decoder: z [B, h, w, latent_dim]
        ── latent_proj (Linear)         → [B, N, hidden]
        ── + 2D sin-cos PE
        ── N × Transformer (RoPE + SwiGLU)
        ── RMSNorm
        ── patch_head (Linear)          → [B, N, patch² · 3]
        ── unpatchify                   → [B, 3, H, W]

接口与 model.vae.VAE 完全一致 —— 可作为 AlignmentVAE 的 `vae=` 直接替换品。
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from util.model_util import (
    RMSNorm,
    VisionRotaryEmbeddingFast,
    get_2d_sincos_pos_embed,
)
from .text_encoder import Attention, SwiGLUFFN


# ============================================================================
# Patch (un)embed
# ============================================================================

class PatchEmbed(nn.Module):
    """[B, C, H, W] → [B, N, hidden]  via a single strided conv (= ViT patchify)."""

    def __init__(self, img_size: int = 256, patch_size: int = 16,
                 in_channels: int = 3, hidden_size: int = 768):
        super().__init__()
        assert img_size % patch_size == 0, \
            f"img_size {img_size} must be divisible by patch_size {patch_size}"
        self.img_size = img_size
        self.patch_size = patch_size
        self.hw = img_size // patch_size
        self.num_patches = self.hw * self.hw

        self.proj = nn.Conv2d(in_channels, hidden_size,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)                              # [B, hidden, h, w]
        x = x.flatten(2).transpose(1, 2).contiguous() # [B, N, hidden]
        return x


def unpatchify(tokens: torch.Tensor, patch_size: int,
               hw: int, channels: int = 3) -> torch.Tensor:
    """[B, N, P²·C] → [B, C, H, W]  (inverse of PatchEmbed)."""
    B, N, _ = tokens.shape
    assert N == hw * hw, f"got N={N}, expected hw²={hw*hw}"
    p = patch_size
    x = tokens.reshape(B, hw, hw, p, p, channels)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous()   # [B, C, hw, p, hw, p]
    x = x.reshape(B, channels, hw * p, hw * p)
    return x


# ============================================================================
# Plain Transformer block (no AdaLN — encoder/decoder are unconditional)
# ============================================================================

class TransformerBlock(nn.Module):
    """Pre-RMSNorm Transformer block: Attn(RoPE+QK-Norm) → SwiGLU FFN."""

    def __init__(self, hidden_size: int, num_heads: int,
                 mlp_ratio: float = 4.0, attn_drop: float = 0.0,
                 proj_drop: float = 0.0):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = Attention(
            hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True,
            attn_drop=attn_drop, proj_drop=proj_drop,
        )
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        self.mlp = SwiGLUFFN(hidden_size, int(hidden_size * mlp_ratio),
                             drop=proj_drop)

    def forward(self, x: torch.Tensor, rope) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), rope=rope)
        x = x + self.mlp(self.norm2(x))
        return x


# ============================================================================
# Encoder / Decoder
# ============================================================================

class ViTVAEEncoder(nn.Module):
    """Image → Transformer → (μ, logσ) [B, h, w, latent_dim]."""

    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 16,
        resolution: int = 256,
        patch_size: int = 16,
        hidden_size: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        double_z: bool = True,
        init_logsigma: float = 0.0,
        fixed_logsigma: Optional[float] = None,
    ):
        super().__init__()
        # 固定 σ 模式（论文 UL 风格）：去掉 logsigma_head，σ_img 钉死为常数 exp(fixed_logsigma)
        if fixed_logsigma is not None:
            double_z = False

        self.latent_dim = latent_dim
        self.double_z = double_z
        self.fixed_logsigma = (float(fixed_logsigma)
                               if fixed_logsigma is not None else None)
        self.hw = resolution // patch_size
        self.num_patches = self.hw * self.hw
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        # patchify
        self.patch_embed = PatchEmbed(
            img_size=resolution, patch_size=patch_size,
            in_channels=in_channels, hidden_size=hidden_size,
        )

        # 2D sin-cos PE (fixed, non-learnable — same convention as SpatialTextEncoder)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, hidden_size), requires_grad=False)

        # 2D RoPE on q/k (per head, half-head as in SpatialTextEncoder)
        half_head = hidden_size // num_heads // 2
        self.feat_rope = VisionRotaryEmbeddingFast(
            dim=half_head, pt_seq_len=self.hw, num_cls_token=0)

        # Transformer backbone
        self.blocks = nn.ModuleList([
            TransformerBlock(
                hidden_size, num_heads, mlp_ratio=mlp_ratio,
                attn_drop=attn_drop if (depth // 4 <= i < depth // 4 * 3) else 0.0,
                proj_drop=proj_drop if (depth // 4 <= i < depth // 4 * 3) else 0.0,
            )
            for i in range(depth)
        ])
        self.norm_final = RMSNorm(hidden_size, eps=1e-6)

        # Heads. We split μ / logσ into two independent linears (cleaner init
        # than a single linear → chunk; allows zero-init on μ and small-bias on logσ)
        if double_z:
            self.mu_head = nn.Linear(hidden_size, latent_dim)
            self.logsigma_head = nn.Linear(hidden_size, latent_dim)
        else:
            # Deterministic encoder: only μ; logσ ≡ 0
            self.mu_head = nn.Linear(hidden_size, latent_dim)
            self.logsigma_head = None

        self._initialize_weights(init_logsigma)

    def _initialize_weights(self, init_logsigma: float):
        def _basic_init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.view(m.weight.size(0), -1))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.apply(_basic_init)

        # Fixed 2-D sin-cos PE
        pe = get_2d_sincos_pos_embed(self.hidden_size, self.hw)
        self.pos_embed.data.copy_(torch.from_numpy(pe).float().unsqueeze(0))

        # Heads: init μ near zero, logσ to a small constant so that
        # σ ≈ exp(init_logsigma) right at start (matches CNN VAE conv_out bias=0).
        nn.init.zeros_(self.mu_head.weight)
        nn.init.zeros_(self.mu_head.bias)
        if self.logsigma_head is not None:
            nn.init.zeros_(self.logsigma_head.weight)
            nn.init.constant_(self.logsigma_head.bias, init_logsigma)

    def forward(self, x: torch.Tensor, return_z: bool = True):
        """Returns (z, μ, logσ) if return_z else (μ, logσ).  All in [B, h, w, D]."""
        B = x.shape[0]
        h = self.patch_embed(x) + self.pos_embed                  # [B, N, hidden]

        for blk in self.blocks:
            h = blk(h, rope=self.feat_rope)

        h = self.norm_final(h)

        mu = self.mu_head(h)                                       # [B, N, D]
        if self.logsigma_head is not None:
            logsigma = self.logsigma_head(h)
            logsigma = torch.clamp(logsigma, min=-5.0, max=2.0)
        elif self.fixed_logsigma is not None:
            # UL 风格：σ_img 钉死为常量，不参与训练
            logsigma = torch.full_like(mu, self.fixed_logsigma)
        else:
            logsigma = torch.zeros_like(mu)

        # Match VAE encoder output convention: [B, h, w, D]
        mu = mu.reshape(B, self.hw, self.hw, self.latent_dim)
        logsigma = logsigma.reshape(B, self.hw, self.hw, self.latent_dim)

        if return_z:
            z = mu + torch.exp(logsigma) * torch.randn_like(mu)
            return z, mu, logsigma
        return mu, logsigma


class ViTVAEDecoder(nn.Module):
    """z [B, h, w, latent_dim] → Transformer → unpatchify → [B, 3, H, W]."""

    def __init__(
        self,
        latent_dim: int = 16,
        out_channels: int = 3,
        resolution: int = 256,
        patch_size: int = 16,
        hidden_size: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.hw = resolution // patch_size
        self.num_patches = self.hw * self.hw
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        # latent → hidden
        self.latent_proj = nn.Linear(latent_dim, hidden_size)

        # 2D PE (independent from encoder's; recommended in JIT-style decoders)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, hidden_size), requires_grad=False)

        # 2D RoPE
        half_head = hidden_size // num_heads // 2
        self.feat_rope = VisionRotaryEmbeddingFast(
            dim=half_head, pt_seq_len=self.hw, num_cls_token=0)

        # Backbone
        self.blocks = nn.ModuleList([
            TransformerBlock(
                hidden_size, num_heads, mlp_ratio=mlp_ratio,
                attn_drop=attn_drop if (depth // 4 <= i < depth // 4 * 3) else 0.0,
                proj_drop=proj_drop if (depth // 4 <= i < depth // 4 * 3) else 0.0,
            )
            for i in range(depth)
        ])
        self.norm_final = RMSNorm(hidden_size, eps=1e-6)

        # Per-token linear projection back to a flattened RGB patch
        self.patch_head = nn.Linear(hidden_size, patch_size * patch_size * out_channels)

        self._initialize_weights()

    def _initialize_weights(self):
        def _basic_init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.apply(_basic_init)

        pe = get_2d_sincos_pos_embed(self.hidden_size, self.hw)
        self.pos_embed.data.copy_(torch.from_numpy(pe).float().unsqueeze(0))

        # Zero-init final patch head so reconstructions start at "mean image" —
        # avoids large gradient spikes early in training (DiT-style trick).
        nn.init.zeros_(self.patch_head.weight)
        nn.init.zeros_(self.patch_head.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: [B, h, w, latent_dim]  →  x_recon: [B, 3, H, W]"""
        B = z.shape[0]
        # [B, h, w, D] → [B, N, D]
        z_seq = z.reshape(B, self.num_patches, self.latent_dim)
        h = self.latent_proj(z_seq) + self.pos_embed

        for blk in self.blocks:
            h = blk(h, rope=self.feat_rope)

        h = self.norm_final(h)

        # [B, N, P²·C] → [B, C, H, W]
        patches = self.patch_head(h)
        return unpatchify(patches, self.patch_size, self.hw, self.out_channels)


# ============================================================================
# Full ViT-VAE
# ============================================================================

class ViTVAE(nn.Module):
    """
    Pure-Transformer VAE (ViT / JIT-style tokenizer).

    `vae.encoder` / `vae.decoder` / `vae.forward` mirror `model.vae.VAE` exactly,
    so AlignmentVAE / SpatialTextEncoder / EmbeddingLabelEncoder work unchanged.
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 16,
        resolution: int = 256,
        patch_size: int = 16,
        # encoder
        enc_hidden_size: int = 768,
        enc_depth: int = 12,
        enc_num_heads: int = 12,
        # decoder
        dec_hidden_size: int = 768,
        dec_depth: int = 12,
        dec_num_heads: int = 12,
        # shared
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        double_z: bool = True,
        init_logsigma: float = 0.0,
        fixed_logsigma: Optional[float] = None,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.patch_size = patch_size
        self.hw = resolution // patch_size

        self.encoder = ViTVAEEncoder(
            in_channels=in_channels, latent_dim=latent_dim,
            resolution=resolution, patch_size=patch_size,
            hidden_size=enc_hidden_size, depth=enc_depth, num_heads=enc_num_heads,
            mlp_ratio=mlp_ratio, attn_drop=attn_drop, proj_drop=proj_drop,
            double_z=double_z, init_logsigma=init_logsigma,
            fixed_logsigma=fixed_logsigma,
        )
        self.decoder = ViTVAEDecoder(
            latent_dim=latent_dim, out_channels=in_channels,
            resolution=resolution, patch_size=patch_size,
            hidden_size=dec_hidden_size, depth=dec_depth, num_heads=dec_num_heads,
            mlp_ratio=mlp_ratio, attn_drop=attn_drop, proj_drop=proj_drop,
        )

    def forward(self, x: torch.Tensor):
        """Returns (x_recon, z, μ, logσ, kl)  — drop-in for model.vae.VAE.forward."""
        z, mu, logsigma = self.encoder(x)
        x_recon = self.decoder(z)
        # KL(q || N(0,I))  with logsigma = log(σ): sum over latent dims, mean over batch
        kl_per_elem = -0.5 * (1 + 2 * logsigma - mu.pow(2) - (2 * logsigma).exp())
        kl = kl_per_elem.sum(dim=(1, 2, 3)).mean()
        return x_recon, z, mu, logsigma, kl

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode to (μ, logσ) without sampling."""
        return self.encoder(x, return_z=False)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


# ============================================================================
# Factory functions  (parameter budgets roughly matched to VAE-{B,L,H})
# ============================================================================

def ViTVAE_B(latent_dim: int = 64, resolution: int = 256, patch_size=16, **kw) -> ViTVAE:
    """Base — patch 16, hidden 768, depth 12, heads 12 (≈ ViT-B per side)."""
    return ViTVAE(
        latent_dim=latent_dim, resolution=resolution, patch_size=patch_size,
        enc_hidden_size=768, enc_depth=12, enc_num_heads=12,
        dec_hidden_size=768, dec_depth=12, dec_num_heads=12,
        mlp_ratio=4.0, **kw,
    )


def ViTVAE_L(latent_dim: int = 64, resolution: int = 256, patch_size=16, **kw) -> ViTVAE:
    """Large — patch 16, hidden 1024, depth 16, heads 16."""
    return ViTVAE(
        latent_dim=latent_dim, resolution=resolution, patch_size=patch_size,
        enc_hidden_size=1024, enc_depth=16, enc_num_heads=16,
        dec_hidden_size=1024, dec_depth=16, dec_num_heads=16,
        mlp_ratio=4.0, **kw,
    )


def ViTVAE_H(latent_dim: int = 64, resolution: int = 256, patch_size=16, **kw) -> ViTVAE:
    """Huge — patch 16, hidden 1280, depth 24, heads 16."""
    return ViTVAE(
        latent_dim=latent_dim, resolution=resolution, patch_size=patch_size,
        enc_hidden_size=1280, enc_depth=24, enc_num_heads=16,
        dec_hidden_size=1280, dec_depth=24, dec_num_heads=16,
        mlp_ratio=4.0, **kw,
    )


ViTVAE_models = {
    'ViTVAE-B': ViTVAE_B,
    'ViTVAE-L': ViTVAE_L,
    'ViTVAE-H': ViTVAE_H,
}
