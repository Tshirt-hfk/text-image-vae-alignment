"""
Text Encoder with Gaussian Output.

Instead of outputting a single embedding vector,
this encoder outputs a Gaussian distribution parameters (μ, logσ)
representing the semantic region described by the text.

Architecture:
    Text → Base Encoder (CLIP/T5) → [μ_head, logσ_head] → N(μ, σ²I)
"""

import torch
import torch.nn as nn


class GaussianTextEncoder(nn.Module):
    """
    Text encoder that outputs a Gaussian distribution over latent space.
    
    The Gaussian models the "semantic region" described by the text:
    - μ: center of the semantic region
    - σ: radius/spread of the region (learned, not fixed)
    
    During training, the image latent z_img is encouraged to have
    high probability density under this distribution: p(z_img | text)
    """
    def __init__(
        self,
        text_encoder,
        latent_dim=768,
        hidden_dim=None,
        freeze_base=True,
    ):
        super().__init__()
        
        self.base_encoder = text_encoder  # e.g. CLIP text encoder (frozen)
        self.latent_dim = latent_dim
        self.freeze_base = freeze_base
        
        # Get base encoder embedding dimension
        if hasattr(text_encoder, 'text_projection'):
            # CLIP
            self.base_dim = text_encoder.text_projection.shape[-1]
        elif hasattr(text_encoder, 'config'):
            # T5/Transformer
            self.base_dim = text_encoder.config.d_model
        else:
            self.base_dim = latent_dim
        
        hidden_dim = hidden_dim or max(self.base_dim, latent_dim)
        
        # Adapter network: base embedding → Gaussian parameters
        # Two separate heads for μ and logσ (allow independent control)
        self.net = nn.Sequential(
            nn.Linear(self.base_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logsigma_head = nn.Linear(hidden_dim, latent_dim)
        
        # Initialize σ to small values (tight distributions)
        nn.init.zeros_(self.logsigma_head.weight)
        nn.init.constant_(self.logsigma_head.bias, -2.0)  # σ ≈ 0.14
        
    def forward(self, text):
        """
        Args:
            text: text input (string list or tokenized tensor depending on base encoder)
        
        Returns:
            mu: [B, latent_dim] mean of text semantic region
            logsigma: [B, latent_dim] log standard deviation
        """
        if self.freeze_base:
            with torch.no_grad():
                h = self.base_encoder(text)
        else:
            h = self.base_encoder(text)
        
        # Project to base dimension if needed
        if hasattr(self.base_encoder, 'text_projection'):
            # CLIP: output is already projected
            h = h / h.norm(dim=-1, keepdim=True)  # normalize CLIP embeddings
        
        h = self.net(h)
        
        mu = self.mu_head(h)
        logsigma = self.logsigma_head(h)
        
        # Clamp σ for numerical stability
        # σ ∈ [e^-5, e^2] ≈ [0.007, 7.4]
        logsigma = torch.clamp(logsigma, min=-5.0, max=2.0)
        
        return mu, logsigma
    
    def sample(self, mu, logsigma):
        """
        Reparameterization trick: z = μ + σ · ε,  ε ~ N(0, I)
        Only used if we want stochastic text representations.
        For alignment, we use the deterministic μ directly.
        """
        sigma = torch.exp(logsigma)
        eps = torch.randn_like(mu)
        return mu + sigma * eps
    
    def log_prob(self, z, mu, logsigma):
        """
        Compute log probability of z under N(μ, σ²I).
        
        log p(z|μ,σ) = -Σ [(z_i - μ_i)² / (2σ_i²)] - Σ log(σ_i) - (D/2)log(2π)
        
        Args:
            z: [B, D] points to evaluate
            mu: [B, D] distribution mean
            logsigma: [B, D] log standard deviation
        
        Returns:
            log_prob: [B] log probability for each point
        """
        sigma = torch.exp(logsigma)
        D = z.shape[-1]
        
        # Mahalanobis distance (diagonal covariance)
        diff = z - mu
        mahalanobis = torch.sum(diff ** 2 / (2 * sigma ** 2 + 1e-8), dim=-1)
        
        # Log determinant of covariance matrix (diag → sum of log σ)
        log_det = torch.sum(logsigma, dim=-1)
        
        # Log probability (constant term omitted)
        log_prob = -mahalanobis - log_det
        
        return log_prob


def build_text_encoder(model_name="ViT-L/14", pretrained="openai", device="cuda", freeze=True):
    """
    Build a CLIP-based text encoder with Gaussian output heads.
    
    Args:
        model_name: CLIP model name (e.g. "ViT-L/14", "ViT-B/32")
        pretrained: Pretrained weight source ("openai", "laion", etc.)
        device: Device to load model on
        freeze: Whether to freeze the base CLIP encoder
    
    Returns:
        GaussianTextEncoder wrapping CLIP
    """
    import open_clip
    
    # Load base CLIP model
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        device=device,
    )
    text_encoder = model.tokenizer  # we'll use the model's encode_text
    
    # Wrap in our Gaussian encoder
    class CLIPTextEncoder:
        """Thin wrapper to make CLIP model compatible with GaussianTextEncoder."""
        def __init__(self, model):
            self.model = model
            self.text_projection = model.text_projection
            self.tokenizer = model.tokenizer
        
        def __call__(self, text, device):
            """
            Encode text strings to embeddings.
            text: list of strings
            """
            if isinstance(text, str):
                text = [text]
            
            # Tokenize
            tokens = self.tokenizer(text).to(device)
            
            # Encode
            with torch.no_grad():
                text_features = self.model.encode_text(tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            return text_features
        
        def to(self, device):
            self.model = self.model.to(device)
            return self
    
    clip_encoder = CLIPTextEncoder(model)
    
    latent_dim = model.text_projection.shape[-1]
    gaussian_encoder = GaussianTextEncoder(
        text_encoder=clip_encoder,
        latent_dim=latent_dim,
        freeze_base=freeze,
    )
    
    return gaussian_encoder, clip_encoder, preprocess
