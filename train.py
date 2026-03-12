"""
Autoresearch vision pretraining with DPS patch selection.
Single-GPU, single-file implementation for AIMS cellphone fingerprint dataset.

Usage: uv run train.py
"""
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import time
import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet18

from prepare import NUM_CLASSES, TIME_BUDGET, make_dataloader, evaluate_accuracy_with_counts, create_cached_dataset, create_val_dataset, make_val_dataloader


# ---------------------------------------------------------------------------
# DPS (Differentiable Patch Selection) Module
# ---------------------------------------------------------------------------

class PerturbedTopK(nn.Module):
    """Differentiable Top-K relaxation using perturbation-based gradients."""

    def __init__(self, k: int, num_samples: int = 500, sigma: float = 0.05):
        super().__init__()
        self.num_samples = num_samples
        self.sigma = sigma
        self.k = k

    def forward(self, x):
        return PerturbedTopKFunction.apply(x, self.k, self.num_samples, self.sigma)


class PerturbedTopKFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, k: int, num_samples: int = 500, sigma: float = 0.05):
        b, d = x.shape

        noise = torch.normal(mean=0.0, std=1.0, size=(b, num_samples, d)).to(x.device)
        perturbed_x = x[:, None, :] + noise * sigma

        topk_results = torch.topk(perturbed_x, k=k, dim=-1, sorted=False)
        indices = topk_results.indices
        indices = torch.sort(indices, dim=-1).values

        perturbed_output = torch.nn.functional.one_hot(indices, num_classes=d).float()
        indicators = perturbed_output.mean(dim=1)

        ctx.k = k
        ctx.num_samples = num_samples
        ctx.sigma = sigma
        ctx.perturbed_output = perturbed_output
        ctx.noise = noise

        return indicators

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output is None:
            return tuple([None] * 5)

        noise_gradient = ctx.noise
        expected_gradient = (
            torch.einsum("bnkd,bne->bkde", ctx.perturbed_output, noise_gradient)
            / ctx.num_samples
            / ctx.sigma
        ) * float(ctx.k)

        grad_input = torch.einsum("bkd,bkde->be", grad_output, expected_gradient)

        return (grad_input,) + tuple([None] * 5)


class Scorer(nn.Module):
    """Score network for patch importance - ResNet18 based."""

    def __init__(self, n_channels=3):
        super().__init__()

        resnet = resnet18(weights="IMAGENET1K_V1")

        self.scorer = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            nn.Conv2d(128, 1, kernel_size=3, padding=0),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )
        del resnet

    def forward(self, x):
        x = self.scorer(x)
        return x.squeeze(1)


class DPS(nn.Module):
    """Differentiable Patch Selection - extracts k patches from high-res images."""

    def __init__(self, n_channels, high_size, low_size, score_size, k, num_samples, sigma, patch_size, device):
        super().__init__()

        self.patch_size = patch_size
        self.k = k
        self.device = device

        self.scorer = Scorer(n_channels).to(device)
        self.TOPK = PerturbedTopK(k, num_samples, sigma)

        h, w = high_size
        self.h_score, self.w_score = score_size

        self.scale_h = None  # Computed dynamically in forward
        self.scale_w = None
        self.padding = None
        self.downscale_transform = transforms.Resize(low_size)

    def forward(self, x_high):
        b, c = x_high.shape[:2]
        patch_size = self.patch_size
        device = self.device

        x_low = self.downscale_transform(x_high).to(device)

        # Score patches
        scores_2d = self.scorer(x_low)

        # Handle both 3D (B, H, W) and 4D (B, C, H, W) outputs
        if scores_2d.dim() == 3:
            scores_2d = scores_2d.unsqueeze(1)  # Add channel dim

        # Dynamically compute score dimensions
        _, _, h_score, w_score = scores_2d.shape
        self.h_score, self.w_score = h_score, w_score

        # Compute scale factors and padding dynamically
        h, w = x_high.shape[2], x_high.shape[3]
        self.scale_h = h // self.h_score
        self.scale_w = w // self.w_score
        padded_h = self.scale_h * self.h_score + patch_size - 1
        padded_w = self.scale_w * self.w_score + patch_size - 1
        top_pad = (patch_size - self.scale_h) // 2
        left_pad = (patch_size - self.scale_w) // 2
        bottom_pad = padded_h - top_pad - h
        right_pad = padded_w - left_pad - w
        self.padding = (left_pad, right_pad, top_pad, bottom_pad)

        scores_1d = scores_2d.view(b, -1)

        # Normalize scores to [0, 1]
        scores_min = scores_1d.min(dim=-1, keepdim=True)[0]
        scores_max = scores_1d.max(dim=-1, keepdim=True)[0]
        scores_1d = (scores_1d - scores_min) / (scores_max - scores_min + 1e-5)

        # Get soft indicators via differentiable Top-K
        indicators = self.TOPK(scores_1d).view(b, self.k, self.h_score, self.w_score)
        self.indicators = indicators

        # Extract patches with weighted combination
        x_high_pad = torch.nn.functional.pad(x_high, self.padding, "constant", 0)
        patches = torch.zeros((b, self.k, c, patch_size, patch_size), device=device)

        for i in range(self.h_score):
            for j in range(self.w_score):
                start_h = i * self.scale_h
                start_w = j * self.scale_w

                current_patches = x_high_pad[:, :, start_h:start_h + patch_size, start_w:start_w + patch_size]
                weight = indicators[:, :, i, j]
                patches += torch.einsum('bchw,bk->bkchw', current_patches, weight)

        return patches.view(-1, c, patch_size, patch_size), indicators


# ---------------------------------------------------------------------------
# ViT-style Feature Extractor with Cross-Attention Pooling
# ---------------------------------------------------------------------------

class ScaledDotProductAttention(nn.Module):
    """Standard scaled dot-product attention."""

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        attn = self.dropout(torch.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        return output


class MultiHeadCrossAttention(nn.Module):
    """Cross-attention with learned queries for patch aggregation."""

    def __init__(self, n_token, n_head, d_model, d_k, d_v, attn_dropout=0.1, dropout=0.1):
        super().__init__()
        self.n_token = n_token
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        # Learnable query tokens
        self.q = nn.Parameter(torch.empty((1, n_token, d_model)))
        q_init_val = math.sqrt(1 / d_k)
        nn.init.uniform_(self.q, a=-q_init_val, b=q_init_val)

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5, attn_dropout=attn_dropout)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        n_token = self.n_token
        b, len_seq = x.shape[:2]

        q = self.w_qs(self.q).view(1, n_token, n_head, d_k)
        k = self.w_ks(x).view(b, len_seq, n_head, d_k)
        v = self.w_vs(x).view(b, len_seq, n_head, d_v)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        x = self.attention(q, k, v)
        x = x.transpose(1, 2).contiguous().view(b, n_token, -1)
        x = self.dropout(self.fc(x))
        x += self.q
        x = self.layer_norm(x)
        return x


class PositionwiseFeedForward(nn.Module):
    """Position-wise feed-forward network with residual connection."""

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.w_2(torch.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual
        x = self.layer_norm(x)
        return x


class EncoderLayer(nn.Module):
    """Transformer encoder layer with cross-attention."""

    def __init__(self, n_token, d_model, d_inner, n_head, d_k, d_v, attn_dropout=0.1, dropout=0.1):
        super().__init__()
        self.n_token = n_token
        self.crs_attn = MultiHeadCrossAttention(n_token, n_head, d_model, d_k, d_v, attn_dropout=attn_dropout, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, x):
        x = self.crs_attn(x)
        x = self.pos_ffn(x)
        return x


class Transformer(nn.Module):
    """Stack of transformer encoder layers."""

    def __init__(self, n_layer, n_token, n_head, d_k, d_v, d_model, d_inner, attn_dropout=0.1, dropout=0.1):
        super().__init__()
        self.layer_stack = nn.ModuleList([
            EncoderLayer(n_token[i], d_model, d_inner, n_head, d_k, d_v, attn_dropout=attn_dropout, dropout=dropout)
            for i in range(n_layer)
        ])

    def forward(self, x):
        for enc_layer in self.layer_stack:
            x = enc_layer(x)
        return x


class PatchAggregator(nn.Module):
    """ViT-style patch aggregator with cross-attention pooling.

    Takes k patch features from DPS and aggregates them into a single representation
    using learned query tokens and cross-attention.
    """

    def __init__(self, input_dim, num_classes, n_layer=2, n_token=1, n_head=8,
                 d_k=64, d_v=64, d_model=512, d_inner=2048, attn_dropout=0.1, dropout=0.1,
                 num_patches=5):
        super().__init__()

        self.input_dim = input_dim
        self.n_token = n_token
        self.n_layer = n_layer
        self.num_patches = num_patches  # k value from DPS

        # Project patch features to model dimension
        self.embedding = nn.Linear(input_dim, d_model)

        # Position encoding for patch sequence (k patches)
        self.pos_encoder = nn.Parameter(torch.zeros(1, num_patches, d_model))

        # Transformer layers
        self.transformer = Transformer(
            n_layer,
            n_token=(n_token,) * n_layer,
            n_head=n_head,
            d_k=d_k,
            d_v=d_v,
            d_model=d_model,
            d_inner=d_inner,
            attn_dropout=attn_dropout,
            dropout=dropout
        )

        # Classification head
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        """
        Args:
            x: Patch features of shape (batch * k, input_dim)
        Returns:
            Logits of shape (batch, num_classes)
        """
        # Reshape to (batch, k, input_dim) - k is num_patches
        batch_size = x.shape[0] // self.num_patches
        x = x.view(batch_size, self.num_patches, self.input_dim)

        # Project to model dimension
        x = self.embedding(x)

        # Add positional encoding
        x = x + self.pos_encoder

        # Transform
        x = self.transformer(x)

        # Mean pooling over tokens
        x = x.mean(dim=1)

        # Classify
        x = self.fc(x)
        return x


# ---------------------------------------------------------------------------
# Main Model: DPS + Backbone + Aggregator
# ---------------------------------------------------------------------------

class DPSViTModel(nn.Module):
    """Complete model: DPS patch selection + ViT backbone + cross-attention aggregator."""

    def __init__(self, config, device="cuda"):
        super().__init__()
        self.config = config

        # DPS module
        self.dps = DPS(
            n_channels=3,
            high_size=config.high_size,
            low_size=config.low_size,
            score_size=config.score_size,
            k=config.k,
            num_samples=config.num_samples,
            sigma=config.sigma,
            patch_size=config.patch_size,
            device="cuda"
        )

        # Simple ViT backbone for patch feature extraction (from scratch, not pretrained)
        self.backbone = SimpleViTBackbone(
            patch_size=config.patch_size,
            embed_dim=config.backbone_embed_dim,
            depth=config.backbone_depth,
            n_heads=config.backbone_heads
        )

        # Patch aggregator
        self.aggregator = PatchAggregator(
            input_dim=config.backbone_embed_dim,
            num_classes=config.num_classes,
            n_layer=config.agg_n_layer,
            n_token=config.agg_n_token,
            n_head=config.agg_n_head,
            d_k=config.agg_d_k,
            d_v=config.agg_d_v,
            d_model=config.agg_d_model,
            d_inner=config.agg_d_inner,
            num_patches=config.k
        )

    def forward(self, x):
        """
        Args:
            x: Input images of shape (B, C, H, W)
        Returns:
            Logits of shape (B, num_classes)
        """
        # Extract patches via DPS
        patches, indicators = self.dps(x)

        # Get patch features from backbone
        patch_features = self.backbone(patches)

        # Aggregate patch features
        logits = self.aggregator(patch_features)

        return logits


class SimpleViTBackbone(nn.Module):
    """Simple ViT backbone for processing individual patches.

    Processes each patch independently and outputs a feature vector.
    """

    def __init__(self, patch_size=448, embed_dim=512, depth=4, n_heads=8):
        super().__init__()

        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # Patch embedding - use fixed kernel size of 16x16 (standard ViT)
        kernel_size = 16
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=kernel_size, stride=kernel_size)

        # Number of patches per side after convolution
        self.num_patches_per_side = (patch_size - kernel_size) // kernel_size + 1
        self.num_patches = self.num_patches_per_side ** 2

        # Position embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

        # Transformer blocks
        self.blocks = nn.ModuleList([
            ViTBlock(embed_dim, n_heads) for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Global average pooling -> feature vector
        self.head = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        """
        Args:
            x: Patches of shape (B*k, C, patch_size, patch_size)
        Returns:
            Features of shape (B*k, embed_dim)
        """
        B, C, H, W = x.shape

        # Patch embedding
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)

        # Add positional embedding
        x = x + self.pos_embed

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Global average pooling
        x = x.mean(dim=1)
        x = self.head(x)

        return x


class ViTBlock(nn.Module):
    """Standard ViT transformer block."""

    def __init__(self, embed_dim, n_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)

        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Model Configuration
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    # Image configuration - for full AIMS dataset (5100/2 = 2550, then /2 = 1275 after 2x downscale)
    high_size: tuple = (1275, 1275)  # Half-resolution input
    low_size: tuple = (448, 448)     # Scaled for scorer
    score_size: tuple = (12, 12)     # Score map resolution (computed dynamically at runtime)
    patch_size: int = 224            # Size of extracted patches

    # DPS configuration
    k: int = 5                       # Number of patches to select
    num_samples: int = 100           # MC samples for perturbed TopK
    sigma: float = 0.05              # Perturbation std

    # Backbone configuration
    backbone_embed_dim: int = 512
    backbone_depth: int = 4
    backbone_heads: int = 8

    # Aggregator configuration
    agg_n_layer: int = 2
    agg_n_token: int = 1
    agg_n_head: int = 8
    agg_d_k: int = 64
    agg_d_v: int = 64
    agg_d_model: int = 512
    agg_d_inner: int = 2048

    # Training configuration
    num_classes: int = NUM_CLASSES


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

# Model
DEPTH = 4
ASPECT_RATIO = 96
HEAD_DIM = 128

# Optimization
TOTAL_BATCH_SIZE = 32
EMBEDDING_LR = 0.001
UNEMBEDDING_LR = 0.001
MATRIX_LR = 0.001
SCALAR_LR = 0.001
WEIGHT_DECAY = 0.01
ADAM_BETAS = (0.9, 0.999)
WARMUP_RATIO = 0.1
WARMDOWN_RATIO = 0.5
FINAL_LR_FRAC = 0.0

DEVICE_BATCH_SIZE = 8  # Batch size - increase if you have enough VRAM



def main():
    """Main training function."""
    t_start = time.time()
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.set_float32_matmul_precision("high")
    device = "cuda"
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    # Load data
    print("Loading training dataset...")
    train_images, train_labels, class_names = create_cached_dataset()
    print(f"Loaded {len(train_images)} training images")

    print("Loading validation dataset...")
    val_images, val_labels, _ = create_val_dataset()
    print(f"Loaded {len(val_images)} validation images")

    # Build model config
    config = ModelConfig()
    print(f"Model config: {asdict(config)}")

    # Create model
    model = DPSViTModel(config, device="cuda").to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {num_params:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=EMBEDDING_LR, weight_decay=WEIGHT_DECAY, betas=ADAM_BETAS)

    # Create dataloaders
    train_loader = make_dataloader(train_images, train_labels, DEVICE_BATCH_SIZE, shuffle=True)
    val_loader = make_val_dataloader(val_images, val_labels, DEVICE_BATCH_SIZE)

    print(f"Time budget: {TIME_BUDGET}s")
    print(f"Device batch size: {DEVICE_BATCH_SIZE}")

    # ---------------------------------------------------------------------------
    # Training Loop
    # ---------------------------------------------------------------------------

    smooth_train_loss = 0
    total_training_time = 0
    step = 0
    best_val_acc = 0.0

    while True:
        torch.cuda.synchronize()
        t0 = time.time()

        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            with autocast_ctx:
                logits = model(images)
                loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * images.size(0)

            preds = logits.argmax(dim=-1)
            epoch_correct += (preds == labels).sum().item()
            epoch_total += labels.size(0)

            step += 1

        # Calculate epoch metrics
        epoch_loss /= epoch_total
        epoch_acc = epoch_correct / epoch_total

        torch.cuda.synchronize()
        t1 = time.time()
        dt = t1 - t0
        total_training_time += dt

        # Logging - print after each epoch
        ema_beta = 0.9
        smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * epoch_loss

        pct_done = 100 * min(total_training_time / TIME_BUDGET, 1.0)
        remaining = max(0, TIME_BUDGET - total_training_time)

        print(f"step {step:05d} ({pct_done:.1f}%) | loss: {epoch_loss:.4f} | acc: {epoch_acc:.4f} | dt: {dt:.1f}s | remaining: {remaining:.0f}s", flush=True)

        # Evaluate on validation set periodically
        if step % 50 == 0 or total_training_time >= TIME_BUDGET * 0.9:
            model.eval()
            torch.cuda.empty_cache()
            with autocast_ctx:
                val_acc, val_correct, val_total = evaluate_accuracy_with_counts(model, val_loader, "cuda")
            print(f"  Val acc: {val_acc:.4f} ({val_correct}/{val_total})")

            if val_acc > best_val_acc:
                best_val_acc = val_acc

        # Time's up
        if total_training_time >= TIME_BUDGET:
            break

    print()  # newline after training log

    # Final evaluation
    model.eval()
    with autocast_ctx:
        final_val_acc, final_val_correct, final_val_total = evaluate_accuracy_with_counts(model, val_loader, "cuda")

    # ---------------------------------------------------------------------------
    # Final Summary
    # ---------------------------------------------------------------------------

    t_end = time.time()
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    print("---")
    print(f"val_accuracy:     {final_val_acc:.6f}")
    print(f"val_correct:      {final_val_correct}")
    print(f"val_total:        {final_val_total}")
    print(f"best_val_acc:     {best_val_acc:.6f}")
    print(f"training_seconds: {total_training_time:.1f}")
    print(f"total_seconds:    {t_end - t_start:.1f}")
    print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
    print(f"num_steps:        {step}")
    print(f"num_params_M:     {num_params / 1e6:.1f}")


if __name__ == "__main__":
    main()
