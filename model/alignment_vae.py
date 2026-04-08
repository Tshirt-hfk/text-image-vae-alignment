"""
AlignmentVAE — L = L_recon + λ_fwd·KL(q_img||q_label) + λ_rev·KL(q_label||q_img)
"""

from typing import List, Tuple, Optional, Dict, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from .vae import VAE
from .text_encoder import SpatialLabelEncoder


class AlignmentVAE(nn.Module):

    def __init__(
        self,
        input_size: int = 256,
        latent_dim: int = 16,
        image_channels: int = 3,
        num_classes: int = 1000,
        vae: Optional[nn.Module] = None,
        label_encoder: Optional[nn.Module] = None,
        # VAE fallback params
        ch: int = 128,
        ch_mult: Optional[List[int]] = None,
        num_res_blocks: int = 2,
        attn_resolutions: Optional[List[int]] = None,
        double_z: bool = True,
        # Label encoder fallback params
        label_hidden_size: int = 768,
        label_depth: int = 12,
        label_num_heads: int = 12,
        label_mlp_ratio: float = 4.0,
        label_in_context_len: int = 32,
        label_in_context_start: int = 4,
        # Loss weights
        weight_recon: float = 1.0,
        weight_alignment: float = 0.1,
        weight_alignment_reverse: float = 0.0,
    ):
        super().__init__()

        self.input_size = input_size
        self.latent_dim = latent_dim
        self.patch_size = 16

        self.weight_recon = weight_recon
        self.weight_alignment = weight_alignment
        self.weight_alignment_reverse = weight_alignment_reverse

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
        z, mu_img, logsigma_img = self.vae.encoder(x)
        mu_label, logsigma_label = self.label_encoder(labels)
        x_recon = self.vae.decoder(z)

        # L_recon
        L_recon = F.mse_loss(x_recon, x)

        # KL(q_img || q_label)  — forward direction
        L_kl_align = self._kl_gaussian(
            mu_img, logsigma_img, mu_label, logsigma_label
        )

        # KL(q_label || q_img)  — reverse direction
        L_kl_reverse = self._kl_gaussian(
            mu_label, logsigma_label, mu_img, logsigma_img
        )

        # Weighted sum
        w_recon = self.weight_recon * L_recon
        w_align = self.weight_alignment * L_kl_align
        w_kl_rev = self.weight_alignment_reverse * L_kl_reverse
        loss = w_recon + w_align + w_kl_rev

        metrics = {
            "loss_total": loss.item(),
            "loss_recon": w_recon.item(),
            "loss_alignment": w_align.item(),
            "loss_kl_reverse": w_kl_rev.item(),
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
    def _kl_gaussian(mu1, logsigma1, mu2, logsigma2):
        """KL(N(μ1,σ1²) || N(μ2,σ2²)), mean over all dims."""
        var_ratio = (logsigma1.exp() ** 2 + (mu1 - mu2) ** 2) / (2 * logsigma2.exp() ** 2 + 1e-8)
        kl_per_elem = logsigma2 - logsigma1 + var_ratio - 0.5
        return kl_per_elem.mean()

    # ---- Inference ----

    @torch.no_grad()
    def generate_from_label(self, labels: torch.Tensor) -> torch.Tensor:
        self.eval()
        mu, logsigma = self.label_encoder(labels)
        sigma = torch.exp(torch.clamp(logsigma, min=-5, max=2))
        z = mu + sigma * torch.randn_like(mu)
        return self.vae.decoder(z)

    @torch.no_grad()
    def reconstruct_image(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        z, _, _ = self.vae.encoder(x)
        return self.vae.decoder(z)

    @torch.no_grad()
    def encode_image_to_gaussian(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.eval()
        mu, logsigma = self.vae.encoder(x, return_z=False)
        return mu, torch.exp(logsigma)

    @torch.no_grad()
    def encode_label_to_gaussian(self, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.eval()
        mu, logsigma = self.label_encoder(labels)
        return mu, torch.exp(logsigma)

    @torch.no_grad()
    def label_interpolate(
        self, label1: int, label2: int,
        num_steps: int = 8, device: str = "cuda",
    ) -> List[torch.Tensor]:
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
