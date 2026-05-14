"""
Text-conditioned encoder -> N(mu, sigma^2) [B, h, w, D]  (T2I).

  - TextEncoderWrapper: 包装 HuggingFace CLIP / T5，统一接口
  - SpatialTextEncoder: AdaLN Transformer + RoPE + in-context 注入

输出形状 [B, h, w, latent_dim] 与 VAE encoder 完全对齐，
便于 AlignmentVAE 上层做 KL 对齐而无需感知条件类型。

与 model.label_encoder.EmbeddingLabelEncoder 是平行关系：
  - cond_type='label' -> 使用 EmbeddingLabelEncoder
  - cond_type='text'  -> 使用本文件的 SpatialTextEncoder
"""

import math
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from util.model_util import VisionRotaryEmbeddingFast, get_2d_sincos_pos_embed, RMSNorm


# ============================================================================
# Transformer Components
# ============================================================================

def modulate(x, shift, scale):
    """AdaLN modulation: x * (1 + scale) + shift."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class Attention(nn.Module):
    """Multi-head self-attention with RoPE and QK-norm."""

    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_norm=True,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.q_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rope):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        q, k = self.q_norm(q), self.k_norm(k)
        q, k = rope(q), rope(k)

        # Use PyTorch SDPA (auto-selects FlashAttention / xformers / math backend)
        drop_p = self.attn_drop.p if self.training else 0.0
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=drop_p)

        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward network."""

    def __init__(self, dim: int, hidden_dim: int, drop=0.0, bias=True):
        super().__init__()
        hidden_dim = int(hidden_dim * 2 / 3)
        self.w12 = nn.Linear(dim, 2 * hidden_dim, bias=bias)
        self.w3 = nn.Linear(hidden_dim, dim, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        return self.w3(self.drop(F.silu(x1) * x2))


class AdaLNBlock(nn.Module):
    """Transformer block with adaLN conditioning, RoPE attention, SwiGLU FFN."""

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True,
                              qk_norm=True, attn_drop=attn_drop, proj_drop=proj_drop)
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        self.mlp = SwiGLUFFN(hidden_size, int(hidden_size * mlp_ratio), drop=proj_drop)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 4 * hidden_size, bias=True),
        )

    def forward(self, x, c, feat_rope=None):
        shift_msa, scale_msa, shift_mlp, scale_mlp = self.adaLN_modulation(c).chunk(4, dim=-1)
        x = x + self.attn(modulate(self.norm1(x), shift_msa, scale_msa), rope=feat_rope)
        x = x + self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


# ============================================================================
# Text-conditioned encoder (T2I support)
# ============================================================================
#
# 与 model.label_encoder.EmbeddingLabelEncoder（int label → 高斯潜空间）平行，
# 下面这套组件用于「文本 caption → 高斯潜空间」的 T2I 训练流程：
#
#   caption (str)
#     → tokenizer (HF CLIP/T5 tokenizer)
#     → input_ids, attention_mask
#     → TextEncoderWrapper（CLIP/T5，可冻结）
#         ├── pooled      [B, D_text]    → AdaLN conditioning c
#         └── seq_features [B, L, D_text] → in-context tokens
#     → SpatialTextEncoder
#         spatial tokens [B, h*w, hidden_size] + 2D PE
#         → AdaLN Transformer (RoPE + SwiGLU + in-context)
#         → mu_head / logsigma_head
#         → (μ, logσ) [B, h, w, latent_dim]   # 形状与 VAE encoder 完全一致
#
# 这样 AlignmentVAE 上层逻辑（KL 对齐、重建损失）完全不需要改动，仅替换
# label_encoder 字段即可在 ImageNet → COCO/CC3M 之间切换。
# ============================================================================


class TextEncoderWrapper(nn.Module):
    """
    包装 HuggingFace 文本编码器（默认 CLIPTextModel），统一对外接口。

    输出：
        seq_features:  [B, L, D_text]
        pooled:        [B, D_text]
        attention_mask: [B, L]   （透传，便于下游做掩码）

    说明：
      - 默认冻结所有参数（与 SD / Imagen 等 T2I 模型的常见做法一致）
      - 对 CLIP：pooled = EOS token 的 hidden state
      - 对 T5：pooled = 经过 attention mask 的 mean pooling
    """

    SUPPORTED = {
        "clip": "openai/clip-vit-base-patch32",
        "clip-large": "openai/clip-vit-large-patch14",
        "t5-small": "google/t5-v1_1-small",
        "t5-base": "google/t5-v1_1-base",
    }

    def __init__(
        self,
        model_name: str = "clip",
        pretrained_path: Optional[str] = None,
        freeze: bool = True,
    ):
        super().__init__()

        # 解析模型路径
        if pretrained_path:
            hf_path = pretrained_path
            kind = "t5" if "t5" in pretrained_path.lower() else "clip"
        elif model_name in self.SUPPORTED:
            hf_path = self.SUPPORTED[model_name]
            kind = "t5" if "t5" in model_name else "clip"
        else:
            # 透传任意 HF repo 名，根据名字猜类型
            hf_path = model_name
            kind = "t5" if "t5" in model_name.lower() else "clip"

        self.kind = kind
        self.hf_path = hf_path

        # 延迟 import，避免无 transformers 环境下整个模块都崩
        if kind == "clip":
            from transformers import CLIPTextModel
            self.encoder = CLIPTextModel.from_pretrained(hf_path)
            self.hidden_dim = self.encoder.config.hidden_size
        else:
            from transformers import T5EncoderModel
            self.encoder = T5EncoderModel.from_pretrained(hf_path)
            self.hidden_dim = self.encoder.config.d_model

        if freeze:
            self.encoder.eval()
            for p in self.encoder.parameters():
                p.requires_grad = False
        self.frozen = freeze

    def train(self, mode: bool = True):
        """冻结时强制 encoder 处于 eval 模式（防止 dropout / norm running stats 变化）。"""
        super().train(mode)
        if self.frozen:
            self.encoder.eval()
        return self

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ctx = torch.no_grad() if self.frozen else torch.enable_grad()
        with ctx:
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            seq = out.last_hidden_state                         # [B, L, D]

            if self.kind == "clip":
                # CLIP 的标准做法：pooled = EOS（id 最大处的 hidden state）
                eos_index = input_ids.argmax(dim=-1)
                pooled = seq[torch.arange(seq.size(0), device=seq.device), eos_index]
            else:
                # T5：mean-pool 有效 token
                mask_f = attention_mask.unsqueeze(-1).float()
                pooled = (seq * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)

        return seq, pooled, attention_mask


class SpatialTextEncoder(nn.Module):
    """
    Text-conditioned 空间高斯编码器（基于 AdaLN Transformer + RoPE + in-context 注入）。

    Args:
        text_dim:           外部 text encoder 输出的特征维度（D_text）
        text_encoder:       可选的 TextEncoderWrapper 实例。若为 None，则
                            forward 时必须传入预先编码好的 (seq, pooled, mask)
        max_text_len:       in-context 注入的最大 text token 数（截断/补齐）
    """

    def __init__(
        self,
        input_size: int = 256,
        patch_size: int = 16,
        text_dim: int = 512,
        text_encoder: Optional[TextEncoderWrapper] = None,
        max_text_len: int = 77,
        hidden_size: int = 768,
        latent_dim: int = 16,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        in_context_start: int = 4,
        init_logsigma: float = -2.0,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.text_dim = text_dim
        self.max_text_len = max_text_len
        self.in_context_len = max_text_len
        self.in_context_start = in_context_start
        self.hw = input_size // patch_size
        self.num_patches = self.hw * self.hw

        # —— 持有可选的 text encoder ——
        # 若由外部统一管理，可设为 None 以节省显存（适合训练时所有 sample 共享一个 encoder）
        self.text_encoder = text_encoder
        if text_encoder is not None:
            assert text_encoder.hidden_dim == text_dim, (
                f"text_encoder.hidden_dim={text_encoder.hidden_dim} != text_dim={text_dim}"
            )

        # —— 文本特征投影：D_text → hidden_size ——
        self.text_proj_seq = nn.Linear(text_dim, hidden_size)     # token-wise 投影 → in-context
        self.text_proj_pool = nn.Linear(text_dim, hidden_size)    # pooled 投影 → AdaLN c

        # —— 主干：spatial tokens + 2D PE + AdaLN blocks ——
        self.spatial_tokens = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size))
        nn.init.normal_(self.spatial_tokens, std=0.02)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, hidden_size), requires_grad=False)

        self.in_context_posemb = nn.Parameter(
            torch.zeros(1, max_text_len, hidden_size), requires_grad=True)
        nn.init.normal_(self.in_context_posemb, std=0.02)

        half_head = hidden_size // num_heads // 2
        self.feat_rope = VisionRotaryEmbeddingFast(
            dim=half_head, pt_seq_len=self.hw, num_cls_token=0)
        self.feat_rope_incontext = VisionRotaryEmbeddingFast(
            dim=half_head, pt_seq_len=self.hw, num_cls_token=max_text_len)

        self.blocks = nn.ModuleList([
            AdaLNBlock(
                hidden_size, num_heads, mlp_ratio=mlp_ratio,
                attn_drop=attn_drop if (depth // 4 <= i < depth // 4 * 3) else 0.0,
                proj_drop=proj_drop if (depth // 4 <= i < depth // 4 * 3) else 0.0,
            ) for i in range(depth)
        ])

        self.norm_final = RMSNorm(hidden_size)
        self.mu_head = nn.Linear(hidden_size, latent_dim)
        self.logsigma_head = nn.Linear(hidden_size, latent_dim)

        self._initialize_weights(init_logsigma)

    def _initialize_weights(self, init_logsigma: float):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        # 仅对自定义子模块初始化，避免覆盖 HF text_encoder 权重
        for name, module in self.named_modules():
            if name.startswith("text_encoder"):
                continue
            _basic_init(module)

        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], self.hw)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.zeros_(self.mu_head.weight)
        nn.init.zeros_(self.mu_head.bias)
        nn.init.zeros_(self.logsigma_head.weight)
        nn.init.constant_(self.logsigma_head.bias, init_logsigma)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def _resolve_text_features(
        self, cond: Any
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        cond 可以是：
          1) dict {input_ids, attention_mask} —— 由内置 text_encoder 编码
          2) tuple/dict 含 (seq, pooled, mask)  —— 直接使用预编码特征
        统一返回 (seq, pooled, mask)。
        """
        if isinstance(cond, dict) and "seq_features" in cond:
            return cond["seq_features"], cond["pooled"], cond["attention_mask"]

        if isinstance(cond, dict) and "input_ids" in cond:
            assert self.text_encoder is not None, (
                "SpatialTextEncoder collected raw token ids but no internal text_encoder was provided."
            )
            return self.text_encoder(cond["input_ids"], cond["attention_mask"])

        if isinstance(cond, (tuple, list)) and len(cond) == 3:
            return cond  # (seq, pooled, mask)

        raise TypeError(
            f"Unsupported cond type for SpatialTextEncoder: {type(cond)}. "
            "Expected dict with input_ids/attention_mask or precomputed features."
        )

    def forward(self, cond: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            cond: 见 _resolve_text_features
        Returns:
            mu, logsigma: [B, h, w, latent_dim]
        """
        seq_feats, pooled, mask = self._resolve_text_features(cond)
        B = seq_feats.size(0)
        L = seq_feats.size(1)

        # —— 截断 / 补齐到 max_text_len ——
        if L > self.max_text_len:
            seq_feats = seq_feats[:, : self.max_text_len]
            mask = mask[:, : self.max_text_len]
        elif L < self.max_text_len:
            pad_len = self.max_text_len - L
            seq_feats = F.pad(seq_feats, (0, 0, 0, pad_len))
            mask = F.pad(mask, (0, pad_len))

        # —— 投影到 hidden_size ——
        c = self.text_proj_pool(pooled)                          # [B, hidden_size]
        ic_tokens = self.text_proj_seq(seq_feats)                # [B, L, hidden_size]

        # 对 padding token：将其 hidden state 置零（保留可学习的位置编码作为占位）
        # 这是简化版「无 attention mask」处理：让 attention 形式上仍是 dense，
        # 但 padding 处的内容信号为 0，模型自然学会忽略。
        ic_tokens = ic_tokens * mask.unsqueeze(-1).to(ic_tokens.dtype)
        ic_tokens = ic_tokens + self.in_context_posemb

        # —— 主干：spatial tokens + AdaLN blocks ——
        x = self.spatial_tokens.expand(B, -1, -1) + self.pos_embed

        for i, block in enumerate(self.blocks):
            if i == self.in_context_start:
                x = torch.cat([ic_tokens, x], dim=1)
            rope = self.feat_rope if i < self.in_context_start else self.feat_rope_incontext
            x = block(x, c, rope)

        x = x[:, self.max_text_len:]   # 去掉 in-context 部分

        x = self.norm_final(x)
        mu = self.mu_head(x)
        logsigma = torch.clamp(self.logsigma_head(x), min=-5.0, max=2.0)

        h = w = self.hw
        return mu.reshape(B, h, w, -1), logsigma.reshape(B, h, w, -1)

    def sample(self, mu: torch.Tensor, logsigma: torch.Tensor) -> torch.Tensor:
        return mu + torch.exp(logsigma) * torch.randn_like(mu)


# —— 文本编码器变体工厂 ——————————————————————————————————————

def SpatialTextEncoder_B(
    input_size=256, latent_dim=16, text_dim=512, max_text_len=77,
    text_encoder: Optional[TextEncoderWrapper] = None, **kw,
):
    """Base — 6 layers, 768 hidden, 12 heads。"""
    return SpatialTextEncoder(
        input_size=input_size, patch_size=16, text_dim=text_dim,
        text_encoder=text_encoder, max_text_len=max_text_len,
        hidden_size=768, latent_dim=latent_dim, depth=6, num_heads=12,
        in_context_start=4, **kw,
    )


def SpatialTextEncoder_L(
    input_size=256, latent_dim=16, text_dim=512, max_text_len=77,
    text_encoder: Optional[TextEncoderWrapper] = None, **kw,
):
    """Large — 12 layers, 1024 hidden, 16 heads。"""
    return SpatialTextEncoder(
        input_size=input_size, patch_size=16, text_dim=text_dim,
        text_encoder=text_encoder, max_text_len=max_text_len,
        hidden_size=1024, latent_dim=latent_dim, depth=12, num_heads=16,
        in_context_start=8, **kw,
    )


def SpatialTextEncoder_H(
    input_size=256, latent_dim=16, text_dim=512, max_text_len=77,
    text_encoder: Optional[TextEncoderWrapper] = None, **kw,
):
    """Huge — 16 layers, 1280 hidden, 16 heads。"""
    return SpatialTextEncoder(
        input_size=input_size, patch_size=16, text_dim=text_dim,
        text_encoder=text_encoder, max_text_len=max_text_len,
        hidden_size=1280, latent_dim=latent_dim, depth=16, num_heads=16,
        in_context_start=10, **kw,
    )


SpatialTextEncoder_models = {
    'STE-B': SpatialTextEncoder_B,
    'STE-L': SpatialTextEncoder_L,
    'STE-H': SpatialTextEncoder_H,
}
