"""
Training script for AlignmentVAE.

Both image and text are encoded as Gaussian distributions.
Loss = MSE reconstruction + VAE KL + Alignment KL(image || text)

Usage:
    python train.py --config configs/default.yaml
"""

import os
import sys
import argparse
import yaml
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from model import AlignmentVAE, build_text_encoder
from losses import TotalLoss


class ImageCaptionDataset(Dataset):
    """
    Dataset returning (image, caption) pairs.
    Falls back to dummy data if no real data available.
    """
    
    def __init__(self, root, caption_file, transform=None, max_samples=None):
        self.root = root
        self.transform = transform or transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        
        if caption_file and os.path.exists(caption_file):
            import pandas as pd
            df = pd.read_csv(caption_file)
            self.captions = df['caption'].tolist()
            self.image_paths = df['image_path'].tolist()
        else:
            # Dummy captions for testing
            self.captions = [
                "a cat sitting on a couch",
                "a dog running in the park",
                "a red bicycle next to a tree",
                "a person riding a skateboard",
                "a bowl of fruit on a table",
                "a sunset over the ocean",
                "a city skyline at night",
                "a bird flying in the sky",
            ] * 25
            self.image_paths = [None] * len(self.captions)
        
        if max_samples:
            self.captions = self.captions[:max_samples]
            self.image_paths = self.image_paths[:max_samples]
    
    def __len__(self):
        return len(self.captions)
    
    def __getitem__(self, idx):
        caption = self.captions[idx]
        
        if self.image_paths[idx] is not None:
            from PIL import Image
            img_path = os.path.join(self.root, self.image_paths[idx])
            try:
                img = Image.open(img_path).convert("RGB")
                if self.transform:
                    img = self.transform(img)
            except Exception:
                img = torch.rand(3, 256, 256)
        else:
            img = torch.rand(3, 256, 256)
        
        return img, caption


def build_model(cfg, device):
    """Build AlignmentVAE with Gaussian text encoder."""
    
    vae_cfg = cfg.get('model', {})
    latent_dim = vae_cfg.get('latent_dim', 768)
    
    # === Text encoder (CLIP + Gaussian heads) ===
    text_gaussian = None
    try:
        text_enc, clip_base, preprocess = build_text_encoder(
            model_name=cfg.get('text_encoder_model', 'ViT-L/14'),
            pretrained=cfg.get('text_encoder_pretrained', 'openai'),
            device=device,
            freeze=cfg.get('freeze_text_encoder', True),
        )
        
        # Wrap CLIP for use with GaussianTextEncoder
        class CLIPWrapper:
            def __init__(self, clip_model, device):
                self.clip_model = clip_model
                self.text_projection = clip_model.model.text_projection
                self._device = device
            
            def __call__(self, text, device=None):
                if device is None:
                    device = self._device
                if isinstance(text, str):
                    text = [text]
                tokens = self.clip_model.tokenizer(text).to(device)
                with torch.no_grad():
                    h = self.clip_model.model.encode_text(tokens)
                    h = h / h.norm(dim=-1, keepdim=True)
                return h
            
            def to(self, d):
                self._device = d
                self.clip_model.model = self.clip_model.model.to(d)
                return self
        
        clip_wrapper = CLIPWrapper(clip_base, device)
        
        from model.text_encoder import GaussianTextEncoder
        text_gaussian = GaussianTextEncoder(
            text_encoder=clip_wrapper,
            latent_dim=latent_dim,
            hidden_dim=vae_cfg.get('text_gaussian_hidden', latent_dim),
            freeze_base=vae_cfg.get('freeze_text_encoder', True),
        )
        print(f"[OK] CLIP text encoder loaded: {cfg.get('text_encoder_model', 'ViT-L/14')}")
        
    except Exception as e:
        print(f"[WARN] Could not load CLIP: {e}")
        print("[WARN] Using random text embeddings (no semantic alignment)")
        # Dummy text encoder: random embeddings
        class DummyTextGaussian(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.dim = dim
                self.net = nn.Sequential(nn.Linear(dim, dim), nn.GELU())
                self.mu_head = nn.Linear(dim, dim)
                self.logsigma_head = nn.Linear(dim, dim)
                nn.init.zeros_(self.logsigma_head.weight)
                nn.init.constant_(self.logsigma_head.bias, -2.0)
            
            def forward(self, text):
                B = len(text) if isinstance(text, list) else 1
                h = torch.randn(B, self.dim)
                h = self.net(h)
                mu = self.mu_head(h)
                logsigma = torch.clamp(self.logsigma_head(h), min=-5, max=2)
                return mu, logsigma
        
        text_gaussian = DummyTextGaussian(latent_dim)
    
    # === AlignmentVAE ===
    model = AlignmentVAE(
        latent_dim=latent_dim,
        image_channels=vae_cfg.get('image_channels', 3),
        hidden_dims=vae_cfg.get('hidden_dims', [128, 256, 512, 512]),
        text_encoder=text_gaussian,
        freeze_text_encoder=vae_cfg.get('freeze_text_encoder', True),
        text_gaussian_hidden_dim=vae_cfg.get('text_gaussian_hidden', latent_dim),
    )
    
    return model


def train(cfg):
    """Main training loop."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    
    # === Model ===
    model = build_model(cfg, device)
    model = model.to(device)
    
    # === Optimizer ===
    train_cfg = cfg.get('training', {})
    lr = train_cfg.get('lr', 1e-4)
    lr_text_adapter = train_cfg.get('lr_text_adapter', 5e-5)
    
    vae_params = list(model.vae.parameters())
    text_params = list(model.text_encoder.parameters())
    
    optimizer = optim.AdamW([
        {'params': vae_params, 'lr': lr},
        {'params': text_params, 'lr': lr_text_adapter},
    ], weight_decay=train_cfg.get('weight_decay', 0.01))
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train_cfg.get('epochs', 100)
    )
    
    # === Loss ===
    loss_fn = TotalLoss(
        weight_recon=train_cfg.get('weight_recon', 1.0),
        weight_kl=train_cfg.get('weight_kl', 1e-6),
        weight_alignment=train_cfg.get('weight_alignment', 0.1),
        bidirectional=False,
        temperature=train_cfg.get('alignment_temp', 1.0),
    ).to(device)
    
    # === Dataset ===
    dataset = ImageCaptionDataset(
        root=train_cfg.get('data_root', './data'),
        caption_file=train_cfg.get('caption_file', './data/captions.csv'),
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=train_cfg.get('batch_size', 16),
        shuffle=True,
        num_workers=train_cfg.get('num_workers', 4),
        pin_memory=True,
    )
    
    # === Training loop ===
    epochs = train_cfg.get('epochs', 100)
    log_interval = train_cfg.get('log_interval', 100)
    grad_clip = train_cfg.get('grad_clip', 1.0)
    
    global_step = 0
    
    for epoch in range(epochs):
        model.train()
        epoch_losses = {k: 0.0 for k in ['loss_total', 'loss_recon', 'loss_kl_vae', 'loss_alignment']}
        n_batches = 0
        
        for batch_idx, (images, captions) in enumerate(dataloader):
            images = images.to(device)
            
            # Forward pass
            x_recon, loss, metrics = model(images, captions)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            optimizer.step()
            
            # Accumulate
            for k in epoch_losses:
                if k in metrics:
                    epoch_losses[k] += metrics[k]
            n_batches += 1
            
            if global_step % log_interval == 0:
                print(f"[Epoch {epoch+1}/{epochs}] [Step {global_step}] "
                      f"total={metrics['loss_total']:.4f} | "
                      f"recon={metrics['loss_recon']:.4f} | "
                      f"align={metrics['loss_alignment']:.4f} | "
                      f"σ_img={metrics.get('sigma_img_mean', 0):.3f} | "
                      f"σ_txt={metrics.get('sigma_text_mean', 0):.3f}")
            
            global_step += 1
        
        # Epoch summary
        print(f"\n=== Epoch {epoch+1} ===")
        for k, v in epoch_losses.items():
            print(f"  avg_{k}: {v/n_batches:.4f}")
        
        scheduler.step()
        
        # Save checkpoint
        if (epoch + 1) % train_cfg.get('save_interval', 10) == 0:
            ckpt_path = f"checkpoints/alignment_vae_epoch{epoch+1}.pt"
            os.makedirs("checkpoints", exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'config': cfg,
            }, ckpt_path)
            print(f"[OK] Saved: {ckpt_path}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    args = parser.parse_args()
    
    cfg_path = Path(args.config)
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        print(f"[OK] Config loaded: {args.config}")
    else:
        print(f"[WARN] Config not found: {args.config}, using defaults")
        cfg = {}
    
    train(cfg)


if __name__ == '__main__':
    main()
