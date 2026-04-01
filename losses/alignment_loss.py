"""
Alignment Loss: Maximize probability of image latent under text Gaussian distribution.

Core idea (from discussion):
  - Image encoded as a POINT in latent space
  - Text encoded as a GAUSSIAN DISTRIBUTION N(μ_text, σ²_text·I)
  - Training objective: maximize p(z_img | text) = N(z_img; μ_text, σ²_text·I)
  
Equivalently minimize: -log N(z_img; μ_text, σ²_text·I)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianAlignmentLoss(nn.Module):
    """
    Loss that encourages the image latent point to have high probability
    density under the text-encoded Gaussian distribution.
    
    L_alignment = -log p(z_img | text)
                   = Σ [(z_img_i - μ_i)² / (2σ_i²)] + Σ log(σ_i) + const
    
    This is essentially a weighted MSE, where the weighting is inversely
    proportional to σ² (dimensions with smaller σ get more emphasis).
    
    Additionally, we can combine with:
    - VAE KL loss: KL(q(z|x) || N(0,I)) to regularize latent space
    - Bidirectional alignment: also push text samples toward image latents
    """
    
    def __init__(self, temperature=1.0, bidirectional=False):
        """
        Args:
            temperature: Scale factor for the Mahalanobis distance.
                        Higher T → softer alignment (less penalty for远离μ)
            bidirectional: If True, also compute text→image alignment (symmetric)
        """
        super().__init__()
        self.temperature = temperature
        self.bidirectional = bidirectional
        
    def forward(self, z_img, mu_text, logsigma_text, z_img_from_text=None, mu_img=None, logsigma_img=None):
        """
        Args:
            z_img: [B, D] image latent point from VAE encoder
            mu_text: [B, D] text Gaussian mean
            logsigma_text: [B, D] text Gaussian log standard deviation
        
        Optional (if bidirectional=True):
            z_img_from_text: [B, D] sampled from text Gaussian
            mu_img: [B, D] image Gaussian mean (from img encoder)
            logsigma_img: [B, D] image Gaussian log std (from img encoder)
        
        Returns:
            loss: scalar alignment loss
            metrics: dict of per-component losses for logging
        """
        sigma_text = torch.exp(logsigma_text)
        D = z_img.shape[-1]
        
        # === Forward alignment: Image → Text Gaussian ===
        # log p(z_img | text) = -0.5 * Σ [(z_i - μ_i)² / σ_i²] - Σ log(σ_i) + C
        diff = z_img - mu_text
        
        # Weighted MSE (inverse variance weighting)
        weighted_mse = torch.sum(diff ** 2 / (2 * sigma_text ** 2 * self.temperature + 1e-8), dim=-1)
        log_det = torch.sum(logsigma_text, dim=-1)
        
        log_prob = -(weighted_mse + log_det)
        L_alignment = -log_prob.mean()
        
        metrics = {
            "loss_alignment_forward": L_alignment.item(),
            "mean_mahalanobis_dist": weighted_mse.mean().item(),
            "mean_logsigma": logsigma_text.mean().item(),
        }
        
        # === Bidirectional: also align text to image ===
        if self.bidirectional and z_img_from_text is not None and mu_img is not None:
            sigma_img = torch.exp(logsigma_img)
            
            # Text sample should have high prob under image Gaussian
            diff_rev = z_img_from_text - mu_img
            weighted_mse_rev = torch.sum(diff_rev ** 2 / (2 * sigma_img ** 2 * self.temperature + 1e-8), dim=-1)
            log_det_rev = torch.sum(logsigma_img, dim=-1)
            log_prob_rev = -(weighted_mse_rev + log_det_rev)
            
            L_alignment_rev = -log_prob_rev.mean()
            L_alignment = (L_alignment + L_alignment_rev) * 0.5
            
            metrics["loss_alignment_reverse"] = L_alignment_rev.item()
        
        return L_alignment, metrics


class TotalLoss(nn.Module):
    """
    Combined loss for Alignment VAE training.
    
    L_total = λ_recon · MSE(x, x̂)
            + λ_kl · KL(q(z|x) || N(0, I))
            + λ_align · L_alignment(z_img, μ_text, σ_text)
    """
    
    def __init__(
        self,
        weight_recon=1.0,
        weight_kl=1e-6,
        weight_alignment=0.1,
        temperature=1.0,
        bidirectional=False,
    ):
        super().__init__()
        self.weight_recon = weight_recon
        self.weight_kl = weight_kl
        self.weight_alignment = weight_alignment
        
        self.alignment_loss = GaussianAlignmentLoss(
            temperature=temperature,
            bidirectional=bidirectional,
        )
        
    def forward(self, x, x_recon, z_img, mu_text, logsigma_text, 
                z_img_prior=None, kl_weight=None):
        """
        Args:
            x: [B, C, H, W] original images
            x_recon: [B, C, H, W] reconstructed images
            z_img: [B, D] image latent from VAE encoder
            mu_text: [B, D] text Gaussian mean
            logsigma_text: [B, D] text Gaussian log std
            z_img_prior: optional, for KL against standard Gaussian
        
        Returns:
            loss: total loss
            metrics: dict of per-component losses
        """
        # Reconstruction loss (pixel-space MSE)
        L_recon = F.mse_loss(x_recon, x)
        
        # KL divergence against standard Gaussian N(0, I)
        # KL(q(z|x) || N(0,I)) ≈ -0.5 * Σ [1 + log(σ²) - μ² - σ²]
        if z_img_prior is not None:
            # z_img already encodes mean and variance info
            # Simple KL: we don't have explicit μ_kl, σ_kl from encoder
            # So we use a simplified approach: just regularize z_img toward N(0,I)
            # This is weaker than true VAE KL but sufficient for our setup
            L_kl = torch.mean(z_img ** 2) * 0.5  # ⟂ Encourage z near origin
        else:
            L_kl = torch.tensor(0.0, device=x.device)
        
        # Alignment loss: maximize p(z_img | text)
        L_alignment, align_metrics = self.alignment_loss(
            z_img, mu_text, logsigma_text
        )
        
        # Combine
        L_total = (
            self.weight_recon * L_recon
            + self.weight_kl * L_kl
            + self.weight_alignment * L_alignment
        )
        
        metrics = {
            "loss_total": L_total.item(),
            "loss_recon": L_recon.item(),
            "loss_kl": L_kl.item(),
            "loss_alignment": L_alignment.item(),
            **{f"align_{k}": v for k, v in align_metrics.items()},
        }
        
        return L_total, metrics
