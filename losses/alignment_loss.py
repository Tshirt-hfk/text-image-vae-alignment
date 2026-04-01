"""
Alignment Loss: Image Gaussian and Text Gaussian.

Core idea (master's latest revision):
    Image encoded as Gaussian: N(μ_img, σ²_img·I)
    Text  encoded as Gaussian: N(μ_text, σ²_text·I)
    
    Maximize p(image_gaussian | text_gaussian)
    ⟺ Minimize KL(N(μ_img, σ²_img) || N(μ_text, σ²_text))

KL(N(μ1, σ1²) || N(μ2, σ2²)) closed form:
    = Σ_d [ log(σ2_d/σ1_d) + (σ1_d² + (μ1_d-μ2_d)²) / (2σ2_d²) - 0.5 ]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianAlignmentLoss(nn.Module):
    """
    KL divergence between image Gaussian and text Gaussian distributions.
    
    L = KL(N_img || N_text) = E_{z~N_img}[log N_img(z) - log N_text(z)]
                            = KL closed form above
    
    Minimizing this pushes the image distribution toward the text distribution:
    - μ_img → μ_text (means align)
    - σ_img → σ_text (variances match)
    
    This is equivalent to maximizing the probability of the image Gaussian
    being consistent with the text-encoded semantic region.
    """
    
    def __init__(self, bidirectional=False, temperature=1.0):
        """
        Args:
            bidirectional: If True, also compute KL(text||image) and average.
                          This enforces I ⟂ T (mutual subspace membership).
        """
        super().__init__()
        self.bidirectional = bidirectional
        self.temperature = temperature
    
    def kl_gaussian(self, mu1, logsigma1, mu2, logsigma2):
        """
        KL(N(μ1, σ1²) || N(μ2, σ2²)) for diagonal Gaussians.
        
        Args:
            mu1, logsigma1: [B, D] first distribution (image)
            mu2, logsigma2: [B, D] second distribution (text)
        
        Returns:
            kl: scalar KL divergence
        """
        sigma1 = torch.exp(torch.clamp(logsigma1, min=-5, max=5))
        sigma2 = torch.exp(torch.clamp(logsigma2, min=-5, max=5))
        
        # log(σ2/σ1) per dimension
        log_ratio = logsigma2 - logsigma1
        
        # (σ1² + (μ1-μ2)²) / (2σ2²) per dimension
        var_term = (sigma1 ** 2 + (mu1 - mu2) ** 2) / (2 * sigma2 ** 2 + 1e-8)
        
        # Full KL per dimension
        kl_per_dim = log_ratio + var_term - 0.5
        
        return kl_per_dim.sum(dim=-1).mean()  # sum over dims, mean over batch
    
    def forward(self, mu_img, logsigma_img, mu_text, logsigma_text):
        """
        Args:
            mu_img:      [B, D] image Gaussian mean
            logsigma_img:[B, D] image Gaussian log standard deviation
            mu_text:     [B, D] text Gaussian mean
            logsigma_text:[B, D] text Gaussian log standard deviation
        
        Returns:
            loss: scalar alignment loss
            metrics: dict
        """
        T = self.temperature
        
        # Forward KL: image → text (what we want to minimize)
        kl_forward = self.kl_gaussian(
            mu_img / T, logsigma_img / T,
            mu_text / T, logsigma_text / T
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
                mu_img / T, logsigma_img / T
            )
            loss = (kl_forward + kl_reverse) * 0.5
            metrics["kl_reverse"] = kl_reverse.item()
        else:
            loss = kl_forward
        
        return loss, metrics


class TotalLoss(nn.Module):
    """
    Combined loss for Alignment VAE.
    
    L_total = λ_recon · MSE(x, x̂)
            + λ_kl_vae · KL(q(z|x) || N(0,I))
            + λ_align · KL(N_img || N_text)
    
    Args:
        weight_recon: weight for reconstruction loss
        weight_kl: weight for VAE KL prior regularization
        weight_alignment: weight for alignment KL
        bidirectional: whether to use symmetric alignment
    """
    
    def __init__(
        self,
        weight_recon=1.0,
        weight_kl=1e-6,
        weight_alignment=0.1,
        bidirectional=False,
        temperature=1.0,
    ):
        super().__init__()
        self.weight_recon = weight_recon
        self.weight_kl = weight_kl
        self.weight_alignment = weight_alignment
        
        self.align_loss = GaussianAlignmentLoss(
            bidirectional=bidirectional,
            temperature=temperature,
        )
    
    def forward(self, x, x_recon, mu_img, logsigma_img, mu_text, logsigma_text, kl_vae=None):
        """
        Args:
            x:           [B, C, H, W] original images
            x_recon:     [B, C, H, W] reconstructed images
            mu_img:      [B, D] image Gaussian mean
            logsigma_img:[B, D] image Gaussian log std
            mu_text:     [B, D] text Gaussian mean
            logsigma_text:[B, D] text Gaussian log std
            kl_vae:      optional precomputed VAE KL (from encoder)
        
        Returns:
            loss:   total combined loss
            metrics: dict of per-component losses
        """
        # Reconstruction
        L_recon = F.mse_loss(x_recon, x)
        
        # VAE KL prior regularization
        if kl_vae is not None:
            L_kl_vae = kl_vae
        else:
            L_kl_vae = -0.5 * torch.mean(1 + logsigma_img - mu_img.pow(2) - logsigma_img.exp())
        
        # Alignment KL: image Gaussian → text Gaussian
        L_align, align_metrics = self.align_loss(mu_img, logsigma_img, mu_text, logsigma_text)
        
        # Combine
        loss = (
            self.weight_recon * L_recon
            + self.weight_kl * L_kl_vae
            + self.weight_alignment * L_align
        )
        
        metrics = {
            "loss_total": loss.item(),
            "loss_recon": L_recon.item(),
            "loss_kl_vae": L_kl_vae.item(),
            "loss_alignment": L_align.item(),
            **{f"align_{k}": v for k, v in align_metrics.items()},
        }
        
        return loss, metrics
