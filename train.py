"""
Training script for AlignmentVAE.

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
import open_clip

from model import AlignmentVAE, build_text_encoder
from losses import TotalLoss


class ImageCaptionDataset(Dataset):
    """
    Dataset that returns (image, caption) pairs.
    Supports COCO, LAION, or dummy data for testing.
    """
    
    def __init__(self, root, caption_file, transform=None, max_samples=None):
        self.root = root
        self.transform = transform or transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        
        # Load captions
        if caption_file and os.path.exists(caption_file):
            import pandas as pd
            df = pd.read_csv(caption_file)
            self.captions = df['caption'].tolist()
            self.image_paths = df['image_path'].tolist()
        else:
            # Dummy data for testing
            self.captions = [
                "a cat sitting on a couch",
                "a dog running in the park",
                "a red bicycle next to a tree",
                "a person riding a skateboard",
                "a bowl of fruit on a table",
            ] * 20
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
            except Exception as e:
                # Fallback: create dummy image
                img = torch.rand(3, 256, 256)
        else:
            # Dummy image for testing
            img = torch.rand(3, 256, 256)
        
        return img, caption


class DummyTextEncoder:
    """
    Dummy text encoder for testing without CLIP.
    Maps text to a random embedding, but the Gaussian heads still work.
    """
    def __init__(self, embed_dim=768):
        self.embed_dim = embed_dim
        self.text_projection = torch.eye(embed_dim)  # dummy
    
    def __call__(self, text, device):
        if isinstance(text, str):
            text = [text]
        B = len(text)
        return torch.randn(B, self.embed_dim, device=device)
    
    def to(self, device):
        return self


def build_model(cfg, device):
    """Build AlignmentVAE model with text encoder."""
    
    # === Build text encoder (CLIP-based Gaussian encoder) ===
    try:
        # Try to load real CLIP
        text_enc, clip_base, preprocess = build_text_encoder(
            model_name=cfg.get('text_encoder_model', 'ViT-L/14'),
            pretrained=cfg.get('text_encoder_pretrained', 'openai'),
            device=device,
            freeze=cfg.get('freeze_text_encoder', True),
        )
        print(f"[OK] Loaded CLIP text encoder: {cfg.get('text_encoder_model', 'ViT-L/14')}")
        
        # For the GaussianTextEncoder wrapper, we need a callable that takes text
        class CLIPWrapper:
            def __init__(self, clip_model, preprocess):
                self.clip_model = clip_model
                self.preprocess = preprocess
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
        
        clip_wrapper = CLIPWrapper(clip_base, preprocess)
        
        from model.text_encoder import GaussianTextEncoder
        latent_dim = cfg.get('latent_dim', 768)
        text_gaussian = GaussianTextEncoder(
            text_encoder=clip_wrapper,
            latent_dim=latent_dim,
            hidden_dim=cfg.get('text_gaussian_hidden', latent_dim),
            freeze_base=cfg.get('freeze_text_encoder', True),
        )
        print("[OK] Gaussian text encoder ready")
        
    except Exception as e:
        print(f"[WARN] Could not load CLIP ({e}), using dummy text encoder")
        text_gaussian = None
    
    # === Build AlignmentVAE ===
    vae_cfg = cfg.get('model', {})
    model = AlignmentVAE(
        latent_dim=vae_cfg.get('latent_dim', 768),
        image_channels=vae_cfg.get('image_channels', 3),
        hidden_dims=vae_cfg.get('hidden_dims', [128, 256, 512, 512]),
        text_encoder=text_gaussian,
        freeze_text_encoder=vae_cfg.get('freeze_text_encoder', True),
        text_gaussian_hidden_dim=vae_cfg.get('text_gaussian_hidden', 768),
    )
    
    return model, text_gaussian


def train(cfg):
    """Main training loop."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    
    # === Model ===
    model, text_encoder = build_model(cfg, device)
    model = model.to(device)
    
    # === Optimizer ===
    train_cfg = cfg.get('training', {})
    lr = train_cfg.get('lr', 1e-4)
    lr_text_adapter = train_cfg.get('lr_text_adapter', 5e-5)
    
    # Different LR for text Gaussian heads vs rest
    vae_params = list(model.vae.parameters())
    text_adapter_params = list(text_encoder.net.parameters()) + \
                          list(text_encoder.mu_head.parameters()) + \
                          list(text_encoder.logsigma_head.parameters()) if text_encoder else []
    
    optimizer = optim.AdamW([
        {'params': vae_params, 'lr': lr},
        {'params': text_adapter_params, 'lr': lr_text_adapter},
    ], weight_decay=train_cfg.get('weight_decay', 0.01))
    
    # === Scheduler ===
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train_cfg.get('epochs', 100)
    )
    
    # === Loss ===
    loss_fn = TotalLoss(
        weight_recon=train_cfg.get('weight_recon', 1.0),
        weight_kl=train_cfg.get('weight_kl', 1e-6),
        weight_alignment=train_cfg.get('weight_alignment', 0.1),
        temperature=train_cfg.get('alignment_temp', 1.0),
    ).to(device)
    
    # === Dataset ===
    dataset_cfg = cfg.get('dataset', {})
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
        epoch_losses = {k: 0.0 for k in [
            'loss_total', 'loss_recon', 'loss_kl', 'loss_alignment'
        ]}
        
        for batch_idx, (images, captions) in enumerate(dataloader):
            images = images.to(device)
            
            # Forward pass
            x_recon, loss, metrics = model(images, captions)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            optimizer.step()
            
            # Accumulate losses
            for k in epoch_losses:
                if k in metrics:
                    epoch_losses[k] += metrics[k]
            
            # Logging
            if global_step % log_interval == 0:
                print(f"[Epoch {epoch+1}/{epochs}] "
                      f"[Step {global_step}] "
                      f"loss_total={metrics['loss_total']:.4f} | "
                      f"recon={metrics['loss_recon']:.4f} | "
                      f"align={metrics['loss_alignment']:.4f} | "
                      f"sigma={metrics.get('mean_sigma', 0):.3f}")
            
            global_step += 1
        
        # Epoch summary
        n_batches = len(dataloader)
        print(f"\n=== Epoch {epoch+1} Summary ===")
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
                'scheduler_state': scheduler.state_dict(),
                'config': cfg,
            }, ckpt_path)
            print(f"[OK] Checkpoint saved: {ckpt_path}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()
    
    # Load config
    cfg_path = Path(args.config)
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        print(f"[OK] Loaded config from {args.config}")
    else:
        print(f"[WARN] Config not found: {args.config}, using defaults")
        cfg = {}
    
    train(cfg)


if __name__ == '__main__':
    main()
