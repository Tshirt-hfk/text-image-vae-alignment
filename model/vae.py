"""
Flux2-style VAE with Spatial Gaussian Latent Space.

    Encoder: [B, 3, H, W] → N(μ, σ²) [B, H//16, W//16, D]
    Decoder: [B, H//16, W//16, D] → [B, 3, H, W]

Architecture: GroupNorm-32, SiLU, self-attention at bottleneck,
mid blocks (Res-Attn-Res), 4 downsample stages = 16× compression.
"""

from typing import List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def Normalize(in_channels: int, num_groups: int = 32) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


# ============================================================================
# Core blocks
# ============================================================================

class ResBlock(nn.Module):
    """x → norm1 → SiLU → conv1 → norm2 → SiLU → conv2 + skip → out"""

    def __init__(self, in_channels: int, out_channels: Optional[int] = None):
        super().__init__()
        out_channels = out_channels or in_channels
        self.norm1 = Normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = Normalize(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip_proj = (nn.Conv2d(in_channels, out_channels, 1)
                          if in_channels != out_channels else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip_proj(x)


class AttnBlock(nn.Module):
    """Single-head self-attention on spatial feature maps.
    Uses F.scaled_dot_product_attention for FlashAttention / memory-efficient backends."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.norm = Normalize(in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, 1)
        self.k = nn.Conv2d(in_channels, in_channels, 1)
        self.v = nn.Conv2d(in_channels, in_channels, 1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        q, k, v = self.q(h), self.k(h), self.v(h)
        B, C, H, W = q.shape

        # Reshape to [B, 1, H*W, C] for SDPA (single head)
        q = q.reshape(B, C, H * W).permute(0, 2, 1).unsqueeze(1)  # [B, 1, N, C]
        k = k.reshape(B, C, H * W).permute(0, 2, 1).unsqueeze(1)
        v = v.reshape(B, C, H * W).permute(0, 2, 1).unsqueeze(1)

        # Use PyTorch's optimized SDPA (auto-selects FlashAttention / xformers / math)
        h = F.scaled_dot_product_attention(q, k, v)
        h = h.squeeze(1).permute(0, 2, 1).reshape(B, C, H, W)
        return x + self.proj_out(h)


class Downsample(nn.Module):
    """2× spatial downsampling with stride-2 conv (asymmetric padding)."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (0, 1, 0, 1), mode='constant', value=0))


class Upsample(nn.Module):
    """2× spatial upsampling with nearest interpolation + conv."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.interpolate(x, scale_factor=2.0, mode='nearest'))


# ============================================================================
# Encoder / Decoder
# ============================================================================

class VAEEncoder(nn.Module):
    """
    [B, 3, H, W] → conv_in → {ResBlock × N, [Attn], Downsample} × 4
    → Mid(Res-Attn-Res) → norm → SiLU → conv_out → split → (μ, logσ) [B, h, w, D]

    `fixed_logsigma`（默认 None）：
        - None  → 与原行为一致：double_z=True 学习 (μ, logσ)；double_z=False 时 logσ≡0。
        - float → 论文 UL (Heek et al., 2026) 风格的「deterministic encoder + 固定噪声」：
                  encoder 只输出 μ（强制 double_z=False，conv_out 通道砍半），
                  σ_img 被钉死为常数 exp(fixed_logsigma)。
                  推荐值 -2.5（σ≈0.082，对齐论文 λ(0)=5）。
    """

    def __init__(self, in_channels: int = 3, latent_dim: int = 16, ch: int = 128,
                 ch_mult: Optional[List[int]] = None, num_res_blocks: int = 2,
                 attn_resolutions: Optional[List[int]] = None,
                 resolution: int = 256, double_z: bool = True,
                 fixed_logsigma: Optional[float] = None):
        super().__init__()
        ch_mult = ch_mult or [1, 2, 4, 4]
        attn_resolutions = attn_resolutions or [16]

        # 固定 σ 模式：自动关闭 double_z，避免浪费 conv_out 一半的通道
        if fixed_logsigma is not None:
            double_z = False

        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.latent_dim = latent_dim
        self.double_z = double_z
        self.fixed_logsigma = (float(fixed_logsigma)
                               if fixed_logsigma is not None else None)

        self.conv_in = nn.Conv2d(in_channels, ch, 3, padding=1)

        # Downsampling stages
        self.down = nn.ModuleList()
        curr_res, in_ch = resolution, ch
        for i_level in range(self.num_resolutions):
            block, attn = nn.ModuleList(), nn.ModuleList()
            out_ch = ch * ch_mult[i_level]
            for _ in range(num_res_blocks):
                block.append(ResBlock(in_ch, out_ch))
                in_ch = out_ch
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(in_ch))
            down = nn.Module()
            down.block, down.attn = block, attn
            down.downsample = Downsample(in_ch)
            curr_res //= 2
            self.down.append(down)

        # Mid block
        self.mid_block_1 = ResBlock(in_ch, in_ch)
        self.mid_attn = AttnBlock(in_ch)
        self.mid_block_2 = ResBlock(in_ch, in_ch)

        # Output
        self.norm_out = Normalize(in_ch)
        out_z = 2 * latent_dim if double_z else latent_dim
        self.conv_out = nn.Conv2d(in_ch, out_z, 3, padding=1)
        nn.init.zeros_(self.conv_out.bias)

    def forward(self, x: torch.Tensor, return_z: bool = True) -> Tuple[torch.Tensor, ...]:
        """
        Returns (z, μ, logσ) if return_z else (μ, logσ).
        All outputs have shape [B, h, w, D].
        """
        h = self.conv_in(x)

        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h)
                if len(self.down[i_level].attn) > i_block:
                    h = self.down[i_level].attn[i_block](h)
            h = self.down[i_level].downsample(h)

        h = self.mid_block_2(self.mid_attn(self.mid_block_1(h)))
        h = self.conv_out(F.silu(self.norm_out(h)))

        if self.double_z:
            mu, logvar = torch.chunk(h, 2, dim=1)
            logsigma = 0.5 * logvar          # log(σ²) → log(σ)
            logsigma = torch.clamp(logsigma, min=-5.0, max=2.0)
        elif self.fixed_logsigma is not None:
            # UL 风格：σ_img 钉死为常量，不参与训练（detach 不需要，因为没有可学习参数）
            mu = h
            logsigma = torch.full_like(mu, self.fixed_logsigma)
        else:
            mu, logsigma = h, torch.zeros_like(h)

        mu = mu.permute(0, 2, 3, 1)
        logsigma = logsigma.permute(0, 2, 3, 1)

        if return_z:
            z = mu + torch.exp(logsigma) * torch.randn_like(mu)
            return z, mu, logsigma
        return mu, logsigma


class VAEDecoder(nn.Module):
    """
    z [B, h, w, D] → permute → conv_in → Mid(Res-Attn-Res)
    → {ResBlock × (N+1), [Attn], Upsample} × 4 → norm → SiLU → conv_out → [B, 3, H, W]
    """

    def __init__(self, latent_dim: int = 16, out_channels: int = 3, ch: int = 128,
                 ch_mult: Optional[List[int]] = None, num_res_blocks: int = 2,
                 attn_resolutions: Optional[List[int]] = None, resolution: int = 256):
        super().__init__()
        ch_mult = ch_mult or [1, 2, 4, 4]
        attn_resolutions = attn_resolutions or [16]

        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks

        bottom_res = resolution // (2 ** self.num_resolutions)
        in_ch = ch * ch_mult[-1]

        self.conv_in = nn.Conv2d(latent_dim, in_ch, 3, padding=1)

        # Mid block
        self.mid_block_1 = ResBlock(in_ch, in_ch)
        self.mid_attn = AttnBlock(in_ch)
        self.mid_block_2 = ResBlock(in_ch, in_ch)

        # Upsampling stages
        self.up = nn.ModuleList()
        curr_res = bottom_res
        for i_level in reversed(range(self.num_resolutions)):
            block, attn = nn.ModuleList(), nn.ModuleList()
            out_ch = ch * ch_mult[i_level]
            for _ in range(num_res_blocks + 1):
                block.append(ResBlock(in_ch, out_ch))
                in_ch = out_ch
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(in_ch))
            up = nn.Module()
            up.block, up.attn = block, attn
            up.upsample = Upsample(in_ch)
            curr_res *= 2
            self.up.append(up)

        self.norm_out = Normalize(in_ch)
        self.conv_out = nn.Conv2d(in_ch, out_channels, 3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: [B, h, w, D] → x_recon: [B, C, H, W]"""
        h = self.conv_in(z.permute(0, 3, 1, 2))
        h = self.mid_block_2(self.mid_attn(self.mid_block_1(h)))

        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > i_block:
                    h = self.up[i_level].attn[i_block](h)
            h = self.up[i_level].upsample(h)

        return self.conv_out(F.silu(self.norm_out(h)))


# ============================================================================
# Full VAE
# ============================================================================

class VAE(nn.Module):
    """
    Flux2-style VAE: Image → Encoder → N(μ,σ²) → sample z → Decoder → Image.
    No skip connections between encoder and decoder.
    """

    def __init__(self, in_channels: int = 3, latent_dim: int = 16, ch: int = 128,
                 ch_mult: Optional[List[int]] = None, num_res_blocks: int = 2,
                 attn_resolutions: Optional[List[int]] = None,
                 resolution: int = 256, double_z: bool = True,
                 fixed_logsigma: Optional[float] = None):
        super().__init__()
        ch_mult = ch_mult or [1, 2, 4, 4]
        attn_resolutions = attn_resolutions or [16]
        self.latent_dim = latent_dim

        self.encoder = VAEEncoder(
            in_channels=in_channels, latent_dim=latent_dim, ch=ch,
            ch_mult=ch_mult, num_res_blocks=num_res_blocks,
            attn_resolutions=attn_resolutions, resolution=resolution,
            double_z=double_z, fixed_logsigma=fixed_logsigma,
        )
        self.decoder = VAEDecoder(
            latent_dim=latent_dim, out_channels=in_channels, ch=ch,
            ch_mult=ch_mult, num_res_blocks=num_res_blocks,
            attn_resolutions=attn_resolutions, resolution=resolution,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (x_recon, z, mu, logsigma, kl)."""
        z, mu, logsigma = self.encoder(x)
        x_recon = self.decoder(z)
        # KL(q||p) with logsigma = log(σ): sum over latent dims, mean over batch
        kl_per_elem = -0.5 * (1 + 2 * logsigma - mu.pow(2) - (2 * logsigma).exp())
        kl = kl_per_elem.sum(dim=(1, 2, 3)).mean()
        return x_recon, z, mu, logsigma, kl

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode to (mu, logsigma) without sampling."""
        return self.encoder(x, return_z=False)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


# ============================================================================
# Factory functions
# ============================================================================

def VAE_B(latent_dim=16, resolution=256, **kw):
    """Base — ch=128, ch_mult=[1,2,4,4], 2 res_blocks, attn@16."""
    return VAE(latent_dim=latent_dim, ch=128, ch_mult=[1, 2, 4, 4],
               num_res_blocks=2, attn_resolutions=[16], resolution=resolution, **kw)

def VAE_L(latent_dim=16, resolution=256, **kw):
    """Large — ch=192, attn@32,16."""
    return VAE(latent_dim=latent_dim, ch=192, ch_mult=[1, 2, 4, 4],
               num_res_blocks=2, attn_resolutions=[32, 16], resolution=resolution, **kw)

def VAE_H(latent_dim=16, resolution=256, **kw):
    """Huge — ch=256, 3 res_blocks, attn@32,16."""
    return VAE(latent_dim=latent_dim, ch=256, ch_mult=[1, 2, 4, 4],
               num_res_blocks=3, attn_resolutions=[32, 16], resolution=resolution, **kw)

VAE_models = {'VAE-B': VAE_B, 'VAE-L': VAE_L, 'VAE-H': VAE_H}
