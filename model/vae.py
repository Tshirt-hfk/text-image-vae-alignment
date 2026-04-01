"""
VAE Encoder that outputs a Gaussian distribution (μ, logσ), not a point.
VAE Decoder reconstructs image from a sampled latent z ~ N(μ, σ²).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Residual block with GroupNorm and SiLU activation."""
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        h = F.silu(self.norm2(h))
        h = self.conv2(h)
        return h + self.skip(x)


class VAEEncoder(nn.Module):
    """
    Encoder that outputs a Gaussian distribution over latent space.
    Outputs (μ, logσ) instead of a single point z.
    
    Reparameterization trick: z = μ + σ · ε,  ε ~ N(0, I)
    """
    def __init__(self, in_channels=3, latent_dim=768, hidden_dims=[128, 256, 512, 512]):
        super().__init__()
        self.latent_dim = latent_dim
        
        self.conv_in = nn.Conv2d(in_channels, hidden_dims[0], 3, padding=1)
        
        self.blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        
        for i, dim in enumerate(hidden_dims):
            self.blocks.append(nn.Sequential(
                ResBlock(dim),
                ResBlock(dim),
            ))
            if i < len(hidden_dims) - 1:
                self.downs.append(nn.Conv2d(dim, dim, 3, stride=2, padding=1))
        
        last_dim = hidden_dims[-1]
        self.norm_final = nn.GroupNorm(8, last_dim)
        
        # TWO heads: mean and log-variance
        self.mu_head = nn.Conv2d(last_dim, latent_dim, 3, padding=1)
        self.logsigma_head = nn.Conv2d(last_dim, latent_dim, 3, padding=1)
        
    def forward(self, x, return_z=True):
        """
        Args:
            x: [B, C, H, W] input image
            return_z: if True, also return reparameterized sample z ~ N(μ, σ²)
        
        Returns:
            mu:     [B, latent_dim] mean of latent distribution
            logsigma:[B, latent_dim] log standard deviation
            z:      [B, latent_dim] sampled latent (if return_z=True)
            skips:  list of skip connection feature maps
        """
        skips = []
        h = self.conv_in(x)
        
        for block, down in zip(self.blocks, self.downs):
            h = block(h)
            skips.append(h)
            h = down(h)
        
        # Final block (no downsample)
        h = self.blocks[-1](h)
        skips.append(h)
        
        h = F.silu(self.norm_final(h))
        
        mu = self.mu_head(h).mean(dim=[2, 3])      # [B, D]
        logsigma = self.logsigma_head(h).mean(dim=[2, 3])  # [B, D]
        
        # Clamp for stability
        logsigma = torch.clamp(logsigma, min=-5.0, max=2.0)
        
        if return_z:
            sigma = torch.exp(logsigma)
            eps = torch.randn_like(mu)
            z = mu + sigma * eps  # reparameterized sample
            return z, mu, logsigma, skips
        
        return mu, logsigma, skips


class VAEDecoder(nn.Module):
    """
    Decoder that reconstructs image from a latent z.
    Uses skip connections from encoder for high-frequency detail.
    """
    def __init__(self, latent_dim=768, hidden_dims=[128, 256, 512, 512], out_channels=3):
        super().__init__()
        self.hidden_dims_rev = list(reversed(hidden_dims))
        
        self.from_latent = nn.ConvTranspose2d(latent_dim, self.hidden_dims_rev[0], 4, stride=2, padding=1)
        
        self.blocks = nn.ModuleList()
        self.ups = nn.ModuleList()
        
        for i in range(len(self.hidden_dims_rev) - 1):
            self.blocks.append(nn.Sequential(
                ResBlock(self.hidden_dims_rev[i]),
                ResBlock(self.hidden_dims_rev[i]),
            ))
            self.ups.append(nn.ConvTranspose2d(self.hidden_dims_rev[i], self.hidden_dims_rev[i+1], 4, stride=2, padding=1))
        
        self.blocks.append(nn.Sequential(
            ResBlock(self.hidden_dims_rev[-1]),
            ResBlock(self.hidden_dims_rev[-1]),
        ))
        
        self.norm_out = nn.GroupNorm(8, self.hidden_dims_rev[-1])
        self.conv_out = nn.Conv2d(self.hidden_dims_rev[-1], out_channels, 3, padding=1)
        
    def forward(self, z, skips):
        """
        Args:
            z: [B, latent_dim] latent vector (deterministic, from reparameterization)
            skips: list of skip connection feature maps from encoder
        
        Returns:
            x_recon: [B, C, H, W] reconstructed image
        """
        skips = list(reversed(skips))
        
        # Expand z to spatial feature map
        h = self.from_latent(z[:, :, None, None])  # [B, dim_rev[0], 2, 2]
        
        for i, (block, up) in enumerate(zip(self.blocks, self.ups)):
            skip = skips[i]
            if h.shape[2:] != skip.shape[2:]:
                h = F.interpolate(h, size=skip.shape[2:], mode='bilinear', align_corners=False)
            h = h + skip * 0.5
            h = block(h)
            if i < len(self.ups):
                h = up(h)
        
        h = h + skips[-1] * 0.5
        h = self.blocks[-1](h)
        h = F.silu(self.norm_out(h))
        x_recon = self.conv_out(h)
        
        return x_recon


class VAE(nn.Module):
    """
    Full VAE: Encoder outputs Gaussian → Reparameterize → Decoder
    Standard VAE training: L_recon + β·KL(q(z|x)||N(0,I))
    """
    def __init__(self, in_channels=3, latent_dim=768, hidden_dims=[128, 256, 512, 512]):
        super().__init__()
        self.encoder = VAEEncoder(in_channels, latent_dim, hidden_dims)
        self.decoder = VAEDecoder(latent_dim, hidden_dims, in_channels)
        
    def forward(self, x):
        """Standard VAE forward. Returns reconstruction and KL loss components."""
        z, mu, logsigma, skips = self.encoder(x)
        x_recon = self.decoder(z, skips)
        
        # Standard VAE KL: KL(q(z|x) || N(0,I))
        # = -0.5 * sum(1 + log(σ²) - μ² - σ²)
        kl = -0.5 * torch.mean(1 + logsigma - mu.pow(2) - logsigma.exp())
        
        return x_recon, z, mu, logsigma, kl
    
    def encode(self, x):
        """Encode to (mu, logsigma) — no sampling."""
        mu, logsigma, skips = self.encoder(x, return_z=False)
        return mu, logsigma, skips
    
    def decode(self, z, skips):
        """Decode from a pre-computed z."""
        return self.decoder(z, skips)
