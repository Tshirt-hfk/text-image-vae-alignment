"""
Gaussian Alignment Loss for spatial distributions.

KL(N(μ1,σ1²) || N(μ2,σ2²)) closed form (diagonal Gaussian):
    = Σ_d [ log(σ2_d/σ1_d) + (σ1_d² + (μ1_d-μ2_d)²) / (2σ2_d²) - 0.5 ]

Both image and label are encoded as [B, h, w, D] spatial Gaussians.
KL is computed per element, summed over latent dims (h, w, D), and averaged over batch.
"""

import torch
import torch.nn as nn


class GaussianAlignmentLoss(nn.Module):
    """
    KL divergence between two diagonal Gaussian distributions.

    Supports [B, h, w, D] (spatial) or [B, D] (flat) inputs.
    Optionally computes symmetric KL = 0.5 * (KL_fwd + KL_rev).
    """

    def __init__(self, bidirectional: bool = False, temperature: float = 1.0):
        super().__init__()
        self.bidirectional = bidirectional
        self.temperature = temperature

    def kl_gaussian(
        self,
        mu1: torch.Tensor, logsigma1: torch.Tensor,
        mu2: torch.Tensor, logsigma2: torch.Tensor,
    ) -> torch.Tensor:
        """KL(N(μ1,σ1²) || N(μ2,σ2²)), sum over latent dims, mean over batch."""
        sigma1 = torch.exp(torch.clamp(logsigma1, min=-5, max=5))
        sigma2 = torch.exp(torch.clamp(logsigma2, min=-5, max=5))
        log_ratio = logsigma2 - logsigma1
        var_term = (sigma1 ** 2 + (mu1 - mu2) ** 2) / (2 * sigma2 ** 2 + 1e-8)
        kl_per_elem = log_ratio + var_term - 0.5
        # sum over latent dims, mean over batch; supports [B, h, w, D] or [B, D]
        if kl_per_elem.dim() == 4:
            return kl_per_elem.sum(dim=(1, 2, 3)).mean()
        else:
            return kl_per_elem.sum(dim=1).mean()

    def forward(
        self,
        mu_img: torch.Tensor, logsigma_img: torch.Tensor,
        mu_text: torch.Tensor, logsigma_text: torch.Tensor,
    ):
        """
        Args:
            mu_img, logsigma_img:   [B, ...] image Gaussian params
            mu_text, logsigma_text: [B, ...] text/label Gaussian params

        Returns:
            loss:    scalar alignment loss
            metrics: dict with diagnostic values
        """
        T = self.temperature
        kl_forward = self.kl_gaussian(
            mu_img / T, logsigma_img / T,
            mu_text / T, logsigma_text / T,
        )

        metrics = {
            "kl_forward": kl_forward.item(),
            "mu_img_norm": mu_img.norm(dim=-1).mean().item(),
            "mu_text_norm": mu_text.norm(dim=-1).mean().item(),
            "sigma_img_mean": torch.exp(logsigma_img).mean().item(),
            "sigma_text_mean": torch.exp(logsigma_text).mean().item(),
        }

        if self.bidirectional:
            kl_reverse = self.kl_gaussian(
                mu_text / T, logsigma_text / T,
                mu_img / T, logsigma_img / T,
            )
            loss = (kl_forward + kl_reverse) * 0.5
            metrics["kl_reverse"] = kl_reverse.item()
        else:
            loss = kl_forward

        return loss, metrics
