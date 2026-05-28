# Text-Image VAE Spatial Alignment

将图像和**条件输入（类别标签 / 自然语言 caption）**编码到**同一空间高斯潜空间** `[B, H/16, W/16, D]`，通过 KL 散度对齐两者的分布，从而支持：

- **条件生成**：标签 / caption → 高斯参数 → 采样 → 解码
- **图像重建**：图像 → VAE 潜变量 → 解码
- **条件间插值生成**：在两个条件的高斯分布间线性插值后解码

> 通过 `cond_type` 配置开关无缝切换两种模式：
> - **Label 模式（默认）**：ImageNet 1000 类条件 → 图像
> - **Text 模式（T2I）**：COCO Captions / CC3M 等图文对 → 文生图

## 架构总览

```
                                            ┌───────────────────────┐
Image (RGB)  ──► VAE Encoder (4× ↓2 conv)  ──►  N(μ_img,   σ_img²)  │
                                            │   [B, h, w, D]        │
                                            ├───── KL 对齐 ─────────┤
Label (int)  ──► Embedding Label Encoder ──►  N(μ_label, σ_label²)  │
                                            └─────────┬─────────────┘
                                                      │
                                z = μ_img + σ_img · ε │ (重参数化)
                                                      ▼
                                            VAE Decoder ──► Recon Image

Loss = w_recon · MSE(x, x̂)
     + w_align · KL(q_img || q_label)         # 前向：图像分布逼近条件分布
     + w_align_reverse · KL(q_label || q_img) # 反向：稳定 μ 对齐、防止 σ 过冲
```

### 核心组件

- [`VAE`](model/vae.py:246-287)：Flux2 风格的 4 级 Conv2d 下采样 / 上采样（共 16× 压缩），ResBlock + 可选 [`AttnBlock`](model/vae.py:44-70)（PyTorch SDPA / FlashAttention），中间层 `Res-Attn-Res` 瓶颈。
- [`EmbeddingLabelEncoder`](model/label_encoder.py)：**Label 模式**默认编码器。两张 `nn.Embedding` 表把 int label 映射到每个空间位置的 `(μ, logσ)`，参数量小、训练快。
- [`SpatialTextEncoder`](model/text_encoder.py) + [`TextEncoderWrapper`](model/text_encoder.py)：**T2I 模式**。基于 [`AdaLNBlock`](model/text_encoder.py)（adaLN 调制 + 2D-RoPE + SwiGLU FFN + RMSNorm + QK-norm）的 Transformer 主干，配合 HuggingFace CLIP / T5 文本编码器。
- [`AlignmentVAE`](model/alignment_vae.py)：组合模型，封装训练 forward 与多种推理接口。
- [`GaussianAlignmentLoss`](loss/alignment_loss.py:15-78)：可单独使用的 KL 对齐损失模块（支持双向、温度缩放）。

### 预设变体

VAE：[`VAE_models`](model/vae.py:309) — `VAE-B`（128ch）/ `VAE-L`（192ch）/ `VAE-H`（256ch，**默认**），均为 `ch_mult=[1,2,4,4]`，attn 在 16×16 分辨率，VAE-L/H 额外在 32×32 加 attn。

Spatial Text Encoder（仅 T2I）：[`SpatialTextEncoder_models`](model/text_encoder.py) — `STE-B`（depth=6, dim=768）/ `STE-L`（12, 1024）/ `STE-H`（16, 1280）。`TextEncoderWrapper` 把 caption 编为 `(seq_features, pooled, mask)`，`pooled` 走 AdaLN，`seq_features` 走 in-context tokens。

## 损失函数

实现位于 [`AlignmentVAE.forward`](model/alignment_vae.py:81-133)，对角高斯 KL 闭式解：

```
KL(N(μ₁,σ₁²) || N(μ₂,σ₂²)) = log(σ₂/σ₁) + (σ₁² + (μ₁-μ₂)²)/(2σ₂²) - 1/2
```

逐元素计算后对所有维度（batch + 空间 + 通道）取均值。`logσ` 通过 `clamp(-5, 2)` 约束数值稳定。

| 损失分量 | 公式 | 默认权重 | 作用 |
|---------|------|:---:|------|
| 重建损失 | `MSE(x_recon, x)` | 1.0 | 保证图像重建质量 |
| 前向对齐 KL | `KL(q_img \|\| q_label)` | 1.0 | 拉近图像分布到条件分布 |
| 反向对齐 KL | `KL(q_label \|\| q_img)` | 0.1 | 稳定 μ 对齐，防止 σ 过冲 |

### 固定 σ_img（论文 UL 风格，默认开启）

参考 [Unified Latents (Heek et al., 2026)](https://arxiv.org/abs/2602.17270) Sec 3.1 与 Ablation D：**可学习的 σ_img 容易塌缩或不稳定**，论文把 encoder 改成"确定性 μ + 固定噪声"，对齐到 prior 能解码到的最小 noise level（λ(0)=5，σ≈0.082）。

本项目三份 yaml 已默认启用该模式：

```yaml
model:
  fixed_logsigma_img: -2.5     # σ_img = exp(-2.5) ≈ 0.082（默认）
  # double_z 会被自动 force 为 false（encoder 输出通道砍半）
```

启用后（默认）：
- VAE encoder（CNN/ViT 均支持）的 `conv_out` / `logsigma_head` 只输出 μ，节省一半计算；
- σ_img 在 forward 时由 `torch.full_like(mu, fixed_logsigma_img)` 即时生成，**不参与训练**；
- KL 损失天然 well-defined：[`_kl_gaussian`](model/alignment_vae.py:135-140) 中 σ_img² 退化为常量，梯度只流向 μ_img 和 label encoder；
- 反向 KL 把 σ_label 也推向同一量级，模拟论文"对齐 encoder 噪声 & prior 噪声"的效果。

如需回退到「VAE 学习 σ_img」的传统行为，把 `fixed_logsigma_img` 设为 `null` 即可（`double_z` 会重新生效）。

## 安装

```bash
pip install -r requirements.txt
```

依赖：`torch>=2.0`、`torchvision>=0.15`、`accelerate>=0.25`、`tensorboard`、`scipy`、`pyyaml`、`pillow`、`tqdm`。T2I 模式额外需要 `transformers>=4.30`。

> 训练脚本统一基于 **Hugging Face accelerate** 启动，封装了 DDP / AMP / GradScaler / 梯度累积 / checkpoint 分片。

## 数据准备

| 模式 | 期望布局 | 加载器 |
|---|---|---|
| **Label**（ImageNet） | `data_root/{train,val}/<class>/*.jpg` | [`ImageLabelDataset`](dataset/label_image_dataset.py) |
| **T2I — COCO 2017** | `data_root/{train2017,val2017}/` + `annotations/captions_*.json` | [`TextImageDataset`](dataset/text_image_dataset.py)（自动探测） |
| **T2I — CSV / JSONL** | `train.csv` 每行 `image_path,caption`<br>`train.jsonl` 每行 `{"image": ..., "caption": ...}` | 同上 |

数据路径不存在时，两种数据集都会自动回退到 dummy（随机张量），便于快速 smoke test。

## 启动训练

```bash
# (可选) 一次性生成默认 accelerate 配置
accelerate config

# 单卡
accelerate launch --num_processes 1 train.py --config configs/imagenet_l2i.yaml

# 单机 8 卡（推荐 bf16，需 Ampere/Hopper+）
accelerate launch --multi_gpu --num_processes 8 --mixed_precision bf16 \
    train.py --config configs/imagenet_l2i.yaml

# 断点续训
accelerate launch --multi_gpu --num_processes 8 \
    train.py --config configs/imagenet_l2i.yaml --resume ./output_dir
```

> bf16 与 fp16 等价吞吐但**无需 GradScaler**，对 KL 损失更稳定，强烈推荐。GPU 不支持 bf16（V100 / T4）时改用 `--mixed_precision fp16`，accelerate 会自动启用 GradScaler。
>
> `--resume` 可指向 `output_dir`（自动选 `checkpoint-last/`）或具体 `checkpoint-{name}/`。

T2I 训练相同，把 config 换成 [`configs/coco_t2i.yaml`](configs/coco_t2i.yaml) 并改其中的 `data_root`。

### 常用参数

完整参数见 yaml 与 `accelerate launch train.py --help`。

| 参数 | 说明 |
|------|------|
| `--batch_size` / `--accum_iter` | 单卡 batch 与梯度累积；有效 batch = `batch × accum × num_processes` |
| `--lr` / `--blr` | 绝对 LR / 基准 LR（`lr = blr × eff_batch / 256`），二选一 |
| `--lr_schedule` | `cosine`（默认） / `constant`，配合 `--warmup_epochs` |
| `--grad_clip` | 梯度裁剪阈值（≤0 关闭），默认 1.0 |
| `--amp_dtype` | `fp16` / `bf16` / `fp32`，映射到 accelerate `mixed_precision` |
| `--use_compile` | 启用 `torch.compile(mode='reduce-overhead')` |
| `--resume` | 续训目录（accelerate 标准格式） |

> AMP dtype 优先级：`accelerate launch --mixed_precision` > `--amp_dtype` > yaml `amp_dtype` > 默认 `fp16`。
>
> 默认开启的性能项：fused AdamW、`gradient_as_bucket_view=True`、`persistent_workers`、`prefetch_factor=4`、`accelerator.accumulate` 上下文（梯度累积步内自动跳过 DDP all-reduce 并对 loss 自动除以 `accum_iter`）。

## 推理示例

> accelerate checkpoint 是一个**目录**（`output_dir/checkpoint-best/`），里面是 `model.safetensors` 等分片文件。下面用 [`accelerate.utils.load_checkpoint_in_model`](https://huggingface.co/docs/accelerate) 加载，无需重新构造 `Accelerator`。

### Label 模式

```python
import torch
from accelerate.utils import load_checkpoint_in_model
from model import AlignmentVAE
from model.vae import VAE_H
from model.label_encoder import EmbeddingLabelEncoder

vae = VAE_H(latent_dim=64, resolution=256)
label_encoder = EmbeddingLabelEncoder(input_size=256, num_classes=1000, latent_dim=64)
model = AlignmentVAE(input_size=256, latent_dim=64, num_classes=1000,
                     vae=vae, label_encoder=label_encoder).cuda().eval()

load_checkpoint_in_model(model, "output_dir/checkpoint-best")

labels = torch.tensor([5, 10, 42], device="cuda")
images = model.generate_from_label(labels)         # [3, 3, 256, 256]
```

### T2I 模式（caption → image）

只展示与 Label 模式不同的部分（构造 `text_encoder` / `SpatialTextEncoder` 并把 caption 包成 `cond` dict）：

```python
from model.text_encoder import SpatialTextEncoder_B, TextEncoderWrapper
from transformers import AutoTokenizer

text_encoder = TextEncoderWrapper(model_name="clip", freeze=True)
label_encoder = SpatialTextEncoder_B(
    input_size=256, latent_dim=64,
    text_dim=text_encoder.hidden_dim, max_text_len=77,
    text_encoder=text_encoder,
)
# vae / model 构造 / load_checkpoint_in_model 同上，num_classes 任意填 1

tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
enc = tokenizer(["a photo of a cat sitting on a sofa",
                 "a beautiful sunset over the ocean"],
                max_length=77, padding="max_length",
                truncation=True, return_tensors="pt").to("cuda")
cond = {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}

images = model.generate_from_label(cond)           # [2, 3, 256, 256]
```

### 通用接口（两种模式相同）

```python
recon = model.reconstruct_image(x)                       # 图像重建
mu_i, sigma_i = model.encode_image_to_gaussian(x)        # 拿中间高斯
mu_c, sigma_c = model.encode_label_to_gaussian(cond_or_labels)
# Label 专属：frames = model.label_interpolate(label1=5, label2=42, num_steps=8, device="cuda")
```

## 评估与 Checkpoint

[`evaluate`](train.py:615-690) 在每个 epoch 后于验证集计算重建质量（MSE / PSNR / SSIM，见 [`MetricsComputer`](train.py:272-357)）和潜空间诊断（`mean_sigma_img/label`、双向 KL）。可选 FID（`compute_fid: true`）会基于 InceptionV3 pool3 (2048-d) 特征计算 **Recon FID**（原图 vs 重建）和 **Gen FID**（原图 vs 条件 → 图像）；多卡通过 `accelerator.gather` 跨 rank 聚合。所有指标写入 `output_dir/tensorboard/`。

[`main`](train.py:781-1007) 自动保存三类 checkpoint 目录到 `--output_dir`：

- `checkpoint-last/` — 每 `save_freq` 个 epoch 与最后 epoch 覆写
- `checkpoint-{epoch}/` — 每 50 个 epoch 归档
- `checkpoint-best/` — 验证 loss 创新低时保存

每个目录由 [`accelerator.save_state`](train.py:473-481) 写入：分片后的 model / optimizer / GradScaler / RNG，外加我们附带的 `meta.pt`（`epoch` + `args`）。续训用 `--resume`；推理用 `load_checkpoint_in_model`。

## 文件结构

```
.
├── model/
│   ├── vae.py               # VAE Encoder/Decoder + VAE_B/L/H
│   ├── label_encoder.py     # EmbeddingLabelEncoder（label 模式默认）
│   ├── text_encoder.py      # SpatialTextEncoder + TextEncoderWrapper（T2I）
│   └── alignment_vae.py     # AlignmentVAE 组合模型
├── dataset/
│   ├── label_image_dataset.py  # L2I 数据集（ImageFolder / dummy）
│   └── text_image_dataset.py   # T2I 数据集（COCO / CSV / JSONL / dummy）
├── loss/alignment_loss.py    # GaussianAlignmentLoss
├── util/                     # MetricLogger / lr_sched / RoPE 等工具
├── configs/
│   ├── imagenet_l2i.yaml     # ImageNet L2I 默认配置
│   └── coco_t2i.yaml         # COCO T2I 配置
├── train.py                  # 训练入口（accelerate）
├── requirements.txt
└── README.md
```
