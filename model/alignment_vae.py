"""
AlignmentVAE: Combined model for text-conditioned image VAE.

Architecture:
    Text  → Text Encoder → Gaussian(μ_text, σ²_text)
    Image → VAE Encoder  → z_img (point in latent space)
    z_img + μ_text, σ_text → VAE Decoder → Reconstructed Image

Training:
    - L_recon: MSE between original and reconstructed image
    - L_kl: Regularize z_img toward N(0, I)  
    - L_align: Maximize p(z_img | text) = N(z_img; μ_text, σ²_text·I)

Inference (text → image):
    Sample z ~ N(μ_text, σ²_text·I) 
    z → VAE Decoder → Generated Image

Inference (image → text → image):
    z_img = VAE Encoder(x)
    x_recon = VAE Decoder(z_img)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .vae import VAE
from .text_encoder import GaussianTextEncoder


class AlignmentVAE(nn.Module):
    """
    Full Alignment VAE model.
    
    Combines:
    - VAE for image compression/decompression
    - Gaussian Text Encoder for semantic region representation
    - Combined loss for joint training
    """
    
    def __init__(
        self,
        latent_dim=768,
        image_channels=3,
        hidden_dims=[128, 256, 512, 512],
        text_encoder=None,
        text_encoder_pretrained="ViT-L/14",
        text_encoder_source="openai",
        freeze_text_encoder=True,
        text_gaussian_hidden_dim=None,
    ):
        super().__init__()
        
        # === VAE ===
        self.vae = VAE(
            in_channels=image_channels,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
        )
        
        # === Text Encoder with Gaussian output ===
        if text_encoder is not None:
            self.text_encoder = text_encoder
        else:
            # Build from scratch (will need to be initialized with a real CLIP model)
            self.text_encoder = GaussianTextEncoder(
                text_encoder=None,  # Must be set externally
                latent_dim=latent_dim,
                hidden_dim=text_gaussian_hidden_dim,
                freeze_base=freeze_text_encoder,
            )
        
        # Store config
        self.latent_dim = latent_dim
        self.image_channels = image_channels
        self.hidden_dims = hidden_dims
        
    def forward(self, x, texts, return_latents=False):
        """
        Forward pass with alignment loss.
        
        Args:
            x: [B, C, H, W] input images
            texts: [B] list of text strings (or tokenized)
            return_latents: if True, also return latents for analysis
        
        Returns:
            x_recon: reconstructed images
            loss: total loss (for training)
            metrics: dict of per-component losses
        """
        # === Image VAE encoding ===
        z_img, skips = self.vae.encoder(x)
        
        # === Text encoding → Gaussian ===
        mu_text, logsigma_text = self.text_encoder(texts)
        
        # === VAE decoding (using image skips) ===
        x_recon = self.vae.decoder(z_img, skips)
        
        # === Compute losses ===
        # Reconstruction loss
        L_recon = F.mse_loss(x_recon, x)
        
        # KL regularizer: z_img should be well-behaved
        L_kl = 0.5 * torch.mean(z_img ** 2)
        
        # Alignment loss: maximize p(z_img | text)
        sigma_text = torch.exp(torch.clamp(logsigma_text, min=-5, max=2))
        diff = z_img - mu_text
        weighted_mse = torch.sum(diff ** 2 / (2 * sigma_text ** 2 + 1e-8), dim=-1)
        log_det = torch.sum(logsigma_text, dim=-1)
        L_align = torch.mean(weighted_mse + log_det)
        
        # Total loss
        loss = L_recon + 1e-6 * L_kl + 0.1 * L_align
        
        metrics = {
            "loss_total": loss.item(),
            "loss_recon": L_recon.item(),
            "loss_kl": L_kl.item(),
            "loss_alignment": L_align.item(),
            "mean_sigma": sigma_text.mean().item(),
            "mean_mahalanobis": weighted_mse.mean().item(),
        }
        
        if return_latents:
            return x_recon, loss, metrics, {
                "z_img": z_img.detach(),
                "mu_text": mu_text.detach(),
                "sigma_text": sigma_text.detach(),
                "x_recon": x_recon.detach(),
            }
        
        return x_recon, loss, metrics
    
    @torch.no_grad()
    def generate_from_text(self, texts, z_img=None, cfg_scale=1.0, num_samples=1):
        """
        Generate image from text description.
        
        Args:
            texts: string or list of strings
            z_img: optional image latent for img2img (torch tensor)
            cfg_scale: classifier-free guidance scale (>1 = more text adherence)
            num_samples: number of images to generate per text
        
        Returns:
            generated images
        """
        self.eval()
        
        if isinstance(texts, str):
            texts = [texts]
        
        # Encode text → Gaussian
        mu_text, logsigma_text = self.text_encoder(texts)
        sigma_text = torch.exp(torch.clamp(logsigma_text, min=-5, max=2))
        
        # Sample from text Gaussian
        if z_img is None:
            # Pure text-to-image: sample z ~ N(μ, σ²)
            eps = torch.randn_like(mu_text)
            z = mu_text + cfg_scale * sigma_text * eps  # apply CFG
        else:
            # Img2img: interpolate between z_img and text distribution
            if z_img.dim() == 2:
                z_img = z_img.unsqueeze(0).expand(len(texts), -1)
            eps = torch.randn_like(mu_text)
            z_sampled = mu_text + sigma_text * eps
            z = cfg_scale * z_sampled + (1 - cfg_scale) * z_img
        
        # Decode (need skips — use zeros for pure generation)
        B = z.shape[0]
        # For pure generation, we need a dummy skip connection strategy
        # The decoder needs proper skip channels — for now, use forward pass
        # through encoder of a dummy/reconstructed image to get skips
        # This is a simplified version; proper impl would cache skip dims
        
        # Simple approach: broadcast z and use a learned skip generator
        h = z[:, :, None, None]
        h = self.vae.decoder.from_latent(h)
        
        # Use cached skips from encoder if available, else zeros
        if hasattr(self, '_cached_skips') and self._cached_skips is not None:
            skips = self._cached_skips
        else:
            # Generate with empty skips (less quality but works)
            skips = [torch.zeros_like(h)] * (len(self.vae.decoder.blocks) + 1)
        
        x_gen = self.vae.decoder._simple_forward(z, skips if hasattr(self.vae.decoder, '_simple_forward') else None)
        
        return x_gen
    
    @torch.no_grad()
    def reconstruct_image(self, x):
        """
        Standard VAE reconstruction (image → latent → image).
        """
        self.eval()
        z, skips = self.vae.encoder(x)
        x_recon = self.vae.decoder(z, skips)
        return x_recon
    
    @torch.no_grad()
    def encode_text_to_gaussian(self, texts):
        """
        Encode text to Gaussian distribution parameters.
        
        Returns:
            mu: [B, D] mean of semantic region
            sigma: [B, D] standard deviation (not log)
        """
        self.eval()
        mu, logsigma = self.text_encoder(texts)
        return mu, torch.exp(logsigma)
    
    @torch.no_grad()
    def semantic_interpolate(self, text1, text2, num_steps=10):
        """
        Interpolate between two text descriptions in semantic space.
        
        Args:
            text1: start text
            text2: end text  
            num_steps: number of interpolation steps
        
        Returns:
            List of generated images
        """
        mu1, sigma1 = self.encode_text_to_gaussian([text1])
        mu2, sigma2 = self.encode_text_to_gaussian([text2])
        
        alphas = torch.linspace(0, 1, num_steps).view(-1, 1)
        
        # Interpolate means
        mu_interp = (1 - alphas) * mu1 + alphas * mu2
        
        # Interpolate sigmas (geometric mean in log space)
        logsigma_interp = (1 - alphas) * torch.log(sigma1) + alphas * torch.log(sigma2)
        sigma_interp = torch.exp(logsigma_interp)
        
        # Sample from interpolated Gaussians
        eps = torch.randn_like(mu_interp)
        z_interp = mu_interp + sigma_interp * eps
        
        # Decode each z
        images = []
        for i in range(num_steps):
            # Simplified: decode each individually
            z_i = z_interp[i:i+1]
            h = z_i[:, :, None, None]
            x_gen = self.vae.decoder.from_latent(h)
            # Placeholder decode (full impl needs proper skip handling)
            images.append(x_gen)
        
        return images
