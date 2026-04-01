"""
VAE Encoder + Decoder for image latent compression.
Encoder outputs a latent point z (not a distribution — KL is handled separately).
Decoder uses skip connections from encoder for high-frequency detail preservation.
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
        
        # Skip connection projection if channel dims differ
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x):
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        h = F.silu(self.norm2(h))
        h = self.conv2(h)
        return h + self.skip(x)


class VAEEncoder(nn.Module):
    """
    Hierarchical encoder that outputs a single latent point z_img.
    No KL head here — KL loss is computed externally against N(0,I).
    Skip connections are returned for decoder use.
    """
    def __init__(self, in_channels=3, latent_dim=768, hidden_dims=[128, 256, 512, 512]):
        super().__init__()
        
        # Input conv
        self.conv_in = nn.Conv2d(in_channels, hidden_dims[0], 3, padding=1)
        
        # Hierarchical downsampling blocks
        self.blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.skip_channels = []  # Store channel dims for decoder skip connections
        
        for i, dim in enumerate(hidden_dims):
            self.blocks.append(nn.Sequential(
                ResBlock(dim),
                ResBlock(dim),
            ))
            self.skip_channels.append(dim)
            
            # Downsample except last block
            if i < len(hidden_dims) - 1:
                self.downs.append(nn.Conv2d(dim, dim, 3, stride=2, padding=1))
        
        # Final layers before latent
        last_dim = hidden_dims[-1]
        self.norm_final = nn.GroupNorm(8, last_dim)
        self.conv_final = nn.Conv2d(last_dim, latent_dim, 3, padding=1)
        
        # Spatial size after all downsampling: H/2^(len(hidden_dims)-1)
        # e.g. 512 → 64 if 4 blocks with downsample each time
        
    def forward(self, x):
        """
        Returns:
            z: [B, latent_dim] latent point
            skips: list of [B, C, H, W] feature maps for decoder skip connections
        """
        skips = []
        
        h = self.conv_in(x)  # [B, hidden_dims[0], H, W]
        
        for i, (block, down) in enumerate(zip(self.blocks, self.downs)):
            h = block(h)
            skips.append(h)
            h = down(h)  # downsample
        
        # Final block (no downsample)
        h = self.blocks[-1](h)
        skips.append(h)
        
        # Project to latent
        h = F.silu(self.norm_final(h))
        z = self.conv_final(h)  # [B, latent_dim, H', W']
        
        # Global average pooling → single latent point
        z = z.mean(dim=[2, 3])  # [B, latent_dim]
        
        return z, skips


class VAEDecoder(nn.Module):
    """
    Decoder that reconstructs image from latent z.
    Uses skip connections from encoder for high-frequency detail.
    """
    def __init__(self, latent_dim=768, hidden_dims=[128, 256, 512, 512], out_channels=3):
        super().__init__()
        
        # Reverse hidden_dims for decoder (upsampling)
        self.hidden_dims_rev = list(reversed(hidden_dims))
        
        # Project from latent to feature map
        self.from_latent = nn.ConvTranspose2d(latent_dim, self.hidden_dims_rev[0], 4, stride=2, padding=1)
        
        # Upsampling blocks with skip connections
        self.blocks = nn.ModuleList()
        self.ups = nn.ModuleList()
        
        for i in range(len(self.hidden_dims_rev) - 1):
            self.blocks.append(nn.Sequential(
                ResBlock(self.hidden_dims_rev[i]),
                ResBlock(self.hidden_dims_rev[i]),
            ))
            # Upsample
            self.ups.append(nn.ConvTranspose2d(self.hidden_dims_rev[i], self.hidden_dims_rev[i+1], 4, stride=2, padding=1))
        
        # Final block
        self.blocks.append(nn.Sequential(
            ResBlock(self.hidden_dims_rev[-1]),
            ResBlock(self.hidden_dims_rev[-1]),
        ))
        
        # Output conv
        self.norm_out = nn.GroupNorm(8, self.hidden_dims_rev[-1])
        self.conv_out = nn.Conv2d(self.hidden_dims_rev[-1], out_channels, 3, padding=1)
        
    def forward(self, z, skips):
        """
        Args:
            z: [B, latent_dim] latent point
            skips: list of feature maps from encoder (skip connections)
        
        Returns:
            x_recon: [B, 3, H, W] reconstructed image
        """
        # Reverse skips to match decoder order
        skips = list(reversed(skips))
        
        # Expand latent to feature map
        h = self.from_latent(z[:, :, None, None])  # [B, hidden_dims_rev[0], 2, 2]
        
        for i, (block, up) in enumerate(zip(self.blocks, self.ups)):
            # Add skip connection (channel attention via 1x1 conv if dims differ)
            skip = skips[i]
            
            # Handle spatial size mismatch (decoder upsamples, skip stays same spatial size)
            if h.shape[2:] != skip.shape[2:]:
                # Interpolate h to match skip spatial size
                h = F.interpolate(h, size=skip.shape[2:], mode='bilinear', align_corners=False)
            
            # Channel dim mismatch: project skip if needed
            if h.shape[1] != skip.shape[1]:
                skip = nn.functional.pad if h.shape[1] > skip.shape[1] else nn.Identity()
            
            h = h + skip * 0.5  # Additive skip connection (stabilized)
            h = block(h)
            
            if i < len(self.ups):
                h = up(h)
        
        # Final skip
        h = h + skips[-1] * 0.5
        h = self.blocks[-1](h)
        
        # Output
        h = F.silu(self.norm_out(h))
        x_recon = self.conv_out(h)
        
        return x_recon


class VAE(nn.Module):
    """
    Full VAE: Encoder → latent point → Decoder
    """
    def __init__(self, in_channels=3, latent_dim=768, hidden_dims=[128, 256, 512, 512]):
        super().__init__()
        self.encoder = VAEEncoder(in_channels, latent_dim, hidden_dims)
        self.decoder = VAEDecoder(latent_dim, hidden_dims, in_channels)
        
    def forward(self, x):
        z, skips = self.encoder(x)
        x_recon = self.decoder(z, skips)
        return x_recon, z
    
    def encode(self, x):
        z, _ = self.encoder(x)
        return z
    
    def decode(self, z):
        # For decoder-only use (requires skips to be passed externally)
        # This is a placeholder — full decode needs skips from a forward pass
        raise NotImplementedError("Use forward() for decoding with skip connections")
