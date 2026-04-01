"""
AlignmentVAE: Image and Text both encoded as Gaussian distributions.

Core idea:
    Image → VAE Encoder → N(μ_img, σ²_img·I)
    Text  → Text Encoder → N(μ_text, σ²_text·I)
    
    Maximize p(image_gaussian | text_gaussian)
    = Minimize KL(N(μ_img, σ²_img) || N(μ_text, σ²_text))

Architecture:
    Text  → Gaussian Text Encoder → (μ_text, logσ_text)
    Image → Gaussian VAE Encoder  → (μ_img, logσ_img)
                                    ↓
                            Reparameterize: z ~ N(μ, σ²)
                                    ↓
    z + skips → VAE Decoder → Reconstructed Image

Loss:
    L_total = L_recon + β·L_KL_VAE + γ·L_KL_alignment

    where:
    L_recon        = MSE(x, x̂)                          — pixel reconstruction
    L_KL_VAE       = KL(q(z|x)||N(0,I))                  — standard VAE prior regularization
    L_KL_alignment = KL(N(μ_img,σ²_img) || N(μ_text,σ²_text)) — image-to-text semantic alignment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .vae import VAE
from .text_encoder import GaussianTextEncoder


class AlignmentVAE(nn.Module):
    """
    Alignment VAE with Gaussian image and text encodings.
    
    Both image and text are encoded as Gaussian distributions.
    The KL divergence between them serves as the alignment loss.
    """
    
    def __init__(
        self,
        latent_dim=768,
        image_channels=3,
        hidden_dims=[128, 256, 512, 512],
        text_encoder=None,
        freeze_text_encoder=True,
        text_gaussian_hidden_dim=None,
    ):
        super().__init__()
        
        # === Image VAE with Gaussian output ===
        self.vae = VAE(
            in_channels=image_channels,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
        )
        
        # === Text Encoder with Gaussian output ===
        self.text_encoder = text_encoder
        
        self.latent_dim = latent_dim
        
    def forward(self, x, texts, return_latents=False):
        """
        Forward pass with Gaussian alignment loss.
        
        Args:
            x:      [B, C, H, W] input images
            texts:  [B] list of text strings
            return_latents: if True, return all latent info
        
        Returns:
            x_recon:     reconstructed images
            loss:        total loss
            metrics:     dict of per-component losses
            latents:     (optional) dict of all latent variables
        """
        # === Image encoding → Gaussian distribution ===
        z, mu_img, logsigma_img, skips = self.vae.encoder(x)  # z ~ N(μ_img, σ²_img)
        
        # === Text encoding → Gaussian distribution ===
        mu_text, logsigma_text = self.text_encoder(texts)
        
        # === Image reconstruction from sampled z ===
        x_recon = self.vae.decoder(z, skips)
        
        # === Compute losses ===
        
        # 1. Reconstruction loss
        L_recon = F.mse_loss(x_recon, x)
        
        # 2. Standard VAE KL: regularize toward N(0, I)
        L_kl_vae = -0.5 * torch.mean(1 + logsigma_img - mu_img.pow(2) - logsigma_img.exp())
        
        # 3. Alignment KL: KL(N(μ_img, σ²_img) || N(μ_text, σ²_text))
        #    This is the core alignment — minimize this to maximize p(img|text)
        L_kl_align = self._kl_gaussian(mu_img, logsigma_img, mu_text, logsigma_text)
        
        # Total loss
        loss = L_recon + 1e-6 * L_kl_vae + 0.1 * L_kl_align
        
        metrics = {
            "loss_total": loss.item(),
            "loss_recon": L_recon.item(),
            "loss_kl_vae": L_kl_vae.item(),
            "loss_kl_align": L_kl_align.item(),
            "mean_mu_img_norm": mu_img.norm(dim=-1).mean().item(),
            "mean_sigma_img": torch.exp(logsigma_img).mean().item(),
            "mean_sigma_text": torch.exp(logsigma_text).mean().item(),
        }
        
        if return_latents:
            return x_recon, loss, metrics, {
                "z": z.detach(),
                "mu_img": mu_img.detach(),
                "logsigma_img": logsigma_img.detach(),
                "mu_text": mu_text.detach(),
                "logsigma_text": logsigma_text.detach(),
                "x_recon": x_recon.detach(),
            }
        
        return x_recon, loss, metrics
    
    def _kl_gaussian(self, mu1, logsigma1, mu2, logsigma2):
        """
        KL(N(μ1, σ1²) || N(μ2, σ2²)) — closed form for diagonal Gaussians.
        
        KL = Σ_d [ log(σ2_d/σ1_d) + (σ1_d² + (μ1_d - μ2_d)²) / (2σ2_d²) - 0.5 ]
        
        This is the negative log probability of image Gaussian under text Gaussian.
        Minimizing this ⟺ Maximizing p(image | text).
        """
        sigma1 = torch.exp(logsigma1)
        sigma2 = torch.exp(logsigma2)
        
        # Term 1: log(σ2/σ1) = log(σ2) - log(σ1)
        log_ratio = logsigma2 - logsigma1  # [B, D]
        
        # Term 2: (σ1² + (μ1-μ2)²) / (2σ2²)
        var_ratio = (sigma1 ** 2 + (mu1 - mu2) ** 2) / (2 * sigma2 ** 2 + 1e-8)  # [B, D]
        
        # Full KL per dimension
        kl_per_dim = log_ratio + var_ratio - 0.5  # [B, D]
        
        return kl_per_dim.sum(dim=-1).mean()  # [B] → scalar
    
    @torch.no_grad()
    def generate_from_text(self, texts, cfg_scale=1.0):
        """
        Generate image from text description.
        
        Args:
            texts:     string or list of strings
            cfg_scale: >1 forces stricter adherence to text semantics
        
        Returns:
            generated images
        """
        self.eval()
        
        if isinstance(texts, str):
            texts = [texts]
        
        # Encode text → Gaussian
        mu_text, logsigma_text = self.text_encoder(texts)
        sigma_text = torch.exp(torch.clamp(logsigma_text, min=-5, max=2))
        
        # Sample z ~ N(μ_text, σ²_text)
        eps = torch.randn_like(mu_text)
        z = mu_text + sigma_text * eps
        
        # Decode (no skips available for pure generation, use zeros)
        B, D = z.shape
        dummy_skips = [torch.zeros(B, dim, 4, 4, device=z.device) 
                       for dim in self.vae.decoder.hidden_dims_rev]
        
        x_gen = self.vae.decoder(z, dummy_skips)
        return x_gen
    
    @torch.no_grad()
    def reconstruct_image(self, x):
        """Standard VAE reconstruction: image → latent → image."""
        self.eval()
        z, mu, logsigma, skips = self.vae.encoder(x)
        x_recon = self.vae.decoder(z, skips)
        return x_recon
    
    @torch.no_grad()
    def encode_image_to_gaussian(self, x):
        """
        Encode image to Gaussian parameters.
        
        Returns:
            mu:     [B, D] mean
            sigma:  [B, D] standard deviation
        """
        self.eval()
        mu, logsigma, skips = self.vae.encoder(x, return_z=False)
        return mu, torch.exp(logsigma)
    
    @torch.no_grad()
    def encode_text_to_gaussian(self, texts):
        """
        Encode text to Gaussian parameters.
        
        Returns:
            mu:     [B, D] mean
            sigma:  [B, D] standard deviation
        """
        self.eval()
        mu, logsigma = self.text_encoder(texts)
        return mu, torch.exp(logsigma)
    
    @torch.no_grad()
    def semantic_interpolate(self, text1, text2, num_steps=8):
        """
        Interpolate between two text descriptions in semantic Gaussian space.
        """
        mu1, sigma1 = self.encode_text_to_gaussian([text1])
        mu2, sigma2 = self.encode_text_to_gaussian([text2])
        
        alphas = torch.linspace(0, 1, num_steps).view(-1, 1)
        
        # Linear interpolation of means
        mu_interp = (1 - alphas) * mu1 + alphas * mu2
        
        # Geometric interpolation of sigmas (in log space)
        logsigma_interp = (1 - alphas) * torch.log(sigma1) + alphas * torch.log(sigma2)
        sigma_interp = torch.exp(logsigma_interp)
        
        # Sample from interpolated Gaussians
        eps = torch.randn_like(mu_interp)
        z_interp = mu_interp + sigma_interp * eps
        
        # Decode
        B, D = mu_interp.shape
        dummy_skips = [torch.zeros(B, dim, 4, 4, device=mu_interp.device) 
                       for dim in self.vae.decoder.hidden_dims_rev]
        
        images = []
        for i in range(num_steps):
            img = self.vae.decoder(z_interp[i:i+1], dummy_skips)
            images.append(img)
        
        return images
