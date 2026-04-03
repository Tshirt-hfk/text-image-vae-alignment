"""
AlignmentVAE: Image and Label encoded as Spatial Gaussian distributions.

Architecture:
    Image → VAE Encoder          → N(μ_img, σ²_img)   [B, h, w, D]
    Label → SpatialLabelEncoder  → N(μ_label, σ²_label) [B, h, w, D]

Loss:
    L = L_recon + λ_kl·L_KL_VAE + λ_align·L_KL_align + λ_ent·L_label_entropy
"""

from typing import List, Tuple, Optional, Dict, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from .vae import VAE
from .text_encoder import SpatialLabelEncoder


class AlignmentVAE(nn.Module):
    """
    Alignment VAE with Flux2-style image VAE and spatial Gaussian label encoder.

    Both encoders output [B, h, w, D] spatial Gaussians.
    KL divergence at each spatial position serves as the alignment loss.
    """

    def __init__(
        self,
        input_size: int = 256,
        latent_dim: int = 16,
        image_channels: int = 3,
        num_classes: int = 1000,
        # Pre-built sub-modules (preferred)
        vae: Optional[nn.Module] = None,
        label_encoder: Optional[nn.Module] = None,
        # VAE fallback params (used only if vae is None)
        ch: int = 128,
        ch_mult: Optional[List[int]] = None,
        num_res_blocks: int = 2,
        attn_resolutions: Optional[List[int]] = None,
        double_z: bool = True,
        # Label encoder fallback params (used only if label_encoder is None)
        label_hidden_size: int = 768,
        label_depth: int = 12,
        label_num_heads: int = 12,
        label_mlp_ratio: float = 4.0,
        label_in_context_len: int = 32,
        label_in_context_start: int = 4,
        # Loss weights (configurable, no longer hardcoded)
        weight_recon: float = 1.0,
        weight_kl: float = 1e-4,
        weight_alignment: float = 0.1,
        weight_label_entropy: float = 0.01,
    ):
        super().__init__()

        self.input_size = input_size
        self.latent_dim = latent_dim
        self.patch_size = 16  # 4 downsample stages → 16× compression

        # Loss weights
        self.weight_recon = weight_recon
        self.weight_kl = weight_kl
        self.weight_alignment = weight_alignment
        self.weight_label_entropy = weight_label_entropy

        # Image VAE
        if vae is not None:
            self.vae = vae
        else:
            self.vae = VAE(
                in_channels=image_channels,
                latent_dim=latent_dim,
                ch=ch,
                ch_mult=ch_mult or [1, 2, 4, 4],
                num_res_blocks=num_res_blocks,
                attn_resolutions=attn_resolutions or [16],
                resolution=input_size,
                double_z=double_z,
            )

        # Spatial Label Encoder
        if label_encoder is not None:
            self.label_encoder = label_encoder
        else:
            self.label_encoder = SpatialLabelEncoder(
                input_size=input_size,
                patch_size=self.patch_size,
                num_classes=num_classes,
                hidden_size=label_hidden_size,
                latent_dim=latent_dim,
                depth=label_depth,
                num_heads=label_num_heads,
                mlp_ratio=label_mlp_ratio,
                in_context_len=label_in_context_len,
                in_context_start=label_in_context_start,
            )

    def forward(
        self,
        x: torch.Tensor,
        labels: torch.Tensor,
        return_latents: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor, Dict],
        Tuple[torch.Tensor, torch.Tensor, Dict, Dict],
    ]:
        """
        Args:
            x:              [B, C, H, W] input images
            labels:         [B] integer class labels
            return_latents: if True, also return all latent variables

        Returns:
            x_recon: [B, C, H, W]
            loss:    scalar total loss
            metrics: dict of per-component losses
            latents: (optional) dict of latent variables
        """
        # Encode image → spatial Gaussian
        z, mu_img, logsigma_img = self.vae.encoder(x)

        # Encode label → spatial Gaussian
        mu_label, logsigma_label = self.label_encoder(labels)

        # Decode
        x_recon = self.vae.decoder(z)

        # --- Losses ---
        L_recon = F.mse_loss(x_recon, x)

        # KL(q||p) with logsigma = log(σ): sum over latent dims, mean over batch
        kl_per_elem = -0.5 * (
            1 + 2 * logsigma_img - mu_img.pow(2) - (2 * logsigma_img).exp()
        )
        L_kl_vae = kl_per_elem.sum(dim=(1, 2, 3)).mean()

        L_kl_align = self._kl_gaussian(
            mu_img, logsigma_img, mu_label, logsigma_label
        )

        # Label entropy: sum over latent dims, mean over batch
        L_label_entropy = logsigma_label.sum(dim=(1, 2, 3)).mean()

        w_recon = self.weight_recon * L_recon
        w_kl = self.weight_kl * L_kl_vae
        w_align = self.weight_alignment * L_kl_align
        w_label_ent = self.weight_label_entropy * L_label_entropy
        loss = w_recon + w_kl + w_align + w_label_ent

        metrics = {
            "loss_total": loss.item(),
            "loss_recon": w_recon.item(),
            "loss_kl_vae": w_kl.item(),
            "loss_alignment": w_align.item(),
            "loss_label_entropy": w_label_ent.item(),
            "mean_mu_img_norm": mu_img.norm(dim=-1).mean().item(),
            "mean_sigma_img": torch.exp(logsigma_img).mean().item(),
            "mean_sigma_label": torch.exp(logsigma_label).mean().item(),
        }

        if return_latents:
            return x_recon, loss, metrics, {
                "z": z.detach(),
                "mu_img": mu_img.detach(),
                "logsigma_img": logsigma_img.detach(),
                "mu_label": mu_label.detach(),
                "logsigma_label": logsigma_label.detach(),
                "x_recon": x_recon.detach(),
            }

        return x_recon, loss, metrics

    @staticmethod
    def _kl_gaussian(
        mu1: torch.Tensor, logsigma1: torch.Tensor,
        mu2: torch.Tensor, logsigma2: torch.Tensor,
    ) -> torch.Tensor:
        """KL(N(μ1,σ1²) || N(μ2,σ2²)), sum over latent dims, mean over batch."""
        var_ratio = (logsigma1.exp() ** 2 + (mu1 - mu2) ** 2) / (2 * logsigma2.exp() ** 2 + 1e-8)
        kl_per_elem = logsigma2 - logsigma1 + var_ratio - 0.5
        return kl_per_elem.sum(dim=(1, 2, 3)).mean()

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate_from_label(self, labels: torch.Tensor) -> torch.Tensor:
        """Label → sample z from N(μ_label, σ²_label) → decode."""
        self.eval()
        mu, logsigma = self.label_encoder(labels)
        sigma = torch.exp(torch.clamp(logsigma, min=-5, max=2))
        z = mu + sigma * torch.randn_like(mu)
        return self.vae.decoder(z)

    @torch.no_grad()
    def reconstruct_image(self, x: torch.Tensor) -> torch.Tensor:
        """Image → encode → decode."""
        self.eval()
        z, _, _ = self.vae.encoder(x)
        return self.vae.decoder(z)

    @torch.no_grad()
    def encode_image_to_gaussian(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (mu, sigma) for image spatial Gaussian."""
        self.eval()
        mu, logsigma = self.vae.encoder(x, return_z=False)
        return mu, torch.exp(logsigma)

    @torch.no_grad()
    def encode_label_to_gaussian(self, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (mu, sigma) for label spatial Gaussian."""
        self.eval()
        mu, logsigma = self.label_encoder(labels)
        return mu, torch.exp(logsigma)

    @torch.no_grad()
    def label_interpolate(
        self, label1: int, label2: int,
        num_steps: int = 8, device: str = "cuda",
    ) -> List[torch.Tensor]:
        """Interpolate between two labels in spatial Gaussian space."""
        self.eval()
        l1 = torch.tensor([label1], device=device)
        l2 = torch.tensor([label2], device=device)
        mu1, sigma1 = self.encode_label_to_gaussian(l1)
        mu2, sigma2 = self.encode_label_to_gaussian(l2)

        images = []
        for alpha in torch.linspace(0, 1, num_steps):
            mu_i = (1 - alpha) * mu1 + alpha * mu2
            sigma_i = torch.exp((1 - alpha) * torch.log(sigma1) + alpha * torch.log(sigma2))
            z = mu_i + sigma_i * torch.randn_like(mu_i)
            images.append(self.vae.decoder(z))
        return images
