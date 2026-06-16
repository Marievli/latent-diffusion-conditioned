"""
U-Net backbone for latent diffusion with cross-attention conditioning.

Architecture follows the design in Ho et al. (2020) and Rombach et al. (2022),
with modifications for efficient cross-attention injection at multiple resolutions.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Positional / time embeddings
# ──────────────────────────────────────────────────────────────────────────────

def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal position embedding for diffusion timesteps."""
    assert dim % 2 == 0, "Embedding dimension must be even."
    half = dim // 2
    freqs = torch.exp(
        -math.log(10_000) * torch.arange(half, dtype=torch.float32, device=timesteps.device) / (half - 1)
    )
    args = timesteps[:, None].float() * freqs[None]
    return torch.cat([args.sin(), args.cos()], dim=-1)


class TimestepEmbedding(nn.Module):
    """Projects sinusoidal timestep embedding into model hidden dimension."""

    def __init__(self, in_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, t_emb: torch.Tensor) -> torch.Tensor:
        return self.net(t_emb)


# ──────────────────────────────────────────────────────────────────────────────
# Core building blocks
# ──────────────────────────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    """
    Residual block with group normalisation and optional time-step conditioning.

    Uses GroupNorm (G=32) following the recommendation in Ho et al.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        groups: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.time_proj = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_channels * 2))

        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))

        # scale-shift conditioning from timestep embedding
        scale, shift = self.time_proj(t_emb).unsqueeze(-1).unsqueeze(-1).chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale) + shift

        h = self.conv2(self.dropout(F.silu(h)))
        return h + self.skip(x)


class CrossAttention(nn.Module):
    """
    Multi-head cross-attention for injecting conditioning context (e.g. CLIP embeddings).

    Keys and values are projected from the context; queries from the spatial features.
    """

    def __init__(self, query_dim: int, context_dim: int, heads: int = 8, head_dim: int = 64) -> None:
        super().__init__()
        inner_dim = heads * head_dim
        self.heads = heads
        self.scale = head_dim ** -0.5

        self.norm = nn.LayerNorm(query_dim)
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_flat = x.view(B, C, H * W).permute(0, 2, 1)  # (B, HW, C)
        x_norm = self.norm(x_flat)

        q = self.to_q(x_norm)
        k = self.to_k(context)
        v = self.to_v(context)

        # reshape for multi-head attention
        def split_heads(t: torch.Tensor) -> torch.Tensor:
            t = t.view(B, -1, self.heads, t.shape[-1] // self.heads)
            return t.permute(0, 2, 1, 3)  # (B, heads, seq, head_dim)

        q, k, v = map(split_heads, (q, k, v))

        attn = torch.softmax((q @ k.transpose(-2, -1)) * self.scale, dim=-1)
        out = (attn @ v).permute(0, 2, 1, 3).contiguous().view(B, H * W, -1)
        out = self.to_out(out).permute(0, 2, 1).view(B, -1, H, W)
        return x + out


class SpatialTransformer(nn.Module):
    """Wraps cross-attention with a self-attention layer and feed-forward network."""

    def __init__(self, channels: int, context_dim: int, heads: int = 8) -> None:
        super().__init__()
        self.cross_attn = CrossAttention(channels, context_dim, heads=heads)
        self.ff = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Linear(channels * 4, channels),
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        x = self.cross_attn(x, context)
        B, C, H, W = x.shape
        x_flat = x.view(B, C, H * W).permute(0, 2, 1)
        x_flat = x_flat + self.ff(x_flat)
        return x_flat.permute(0, 2, 1).view(B, C, H, W)


# ──────────────────────────────────────────────────────────────────────────────
# Down / Up sampling blocks
# ──────────────────────────────────────────────────────────────────────────────

class Downsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


# ──────────────────────────────────────────────────────────────────────────────
# U-Net
# ──────────────────────────────────────────────────────────────────────────────

class UNet(nn.Module):
    """
    Conditional U-Net for latent diffusion.

    Args:
        in_channels:    Number of latent channels (typically 4 for VAE latents).
        model_channels: Base channel width; subsequent levels multiply this.
        channel_mults:  Multiplicative factors per resolution level.
        context_dim:    Dimensionality of the conditioning context (e.g. 768 for CLIP ViT-L).
        num_res_blocks: ResBlocks per resolution level.
        attn_resolutions: Resolutions (as spatial side lengths) at which to apply attention.
        dropout:        Dropout rate inside ResBlocks.
        heads:          Number of attention heads in cross-attention layers.
    """

    def __init__(
        self,
        in_channels: int = 4,
        model_channels: int = 128,
        channel_mults: tuple[int, ...] = (1, 2, 4, 4),
        context_dim: int = 768,
        num_res_blocks: int = 2,
        attn_resolutions: tuple[int, ...] = (16, 8),
        dropout: float = 0.1,
        heads: int = 8,
    ) -> None:
        super().__init__()

        time_emb_dim = model_channels * 4
        self.time_embed = TimestepEmbedding(model_channels, time_emb_dim)

        self.input_proj = nn.Conv2d(in_channels, model_channels, 3, padding=1)

        # ── encoder ──
        self.down_blocks: nn.ModuleList = nn.ModuleList()
        self.down_samples: nn.ModuleList = nn.ModuleList()
        ch = model_channels
        skip_channels: list[int] = [ch]
        resolution = 64  # assumed spatial size of latents

        for mult in channel_mults:
            out_ch = model_channels * mult
            level_blocks: list[nn.Module] = []
            for _ in range(num_res_blocks):
                level_blocks.append(ResBlock(ch, out_ch, time_emb_dim, dropout=dropout))
                if resolution in attn_resolutions:
                    level_blocks.append(SpatialTransformer(out_ch, context_dim, heads=heads))
                skip_channels.append(out_ch)
                ch = out_ch
            self.down_blocks.append(nn.ModuleList(level_blocks))
            self.down_samples.append(Downsample(ch))
            resolution //= 2

        # ── bottleneck ──
        self.mid_res1 = ResBlock(ch, ch, time_emb_dim, dropout=dropout)
        self.mid_attn = SpatialTransformer(ch, context_dim, heads=heads)
        self.mid_res2 = ResBlock(ch, ch, time_emb_dim, dropout=dropout)

        # ── decoder ──
        self.up_blocks: nn.ModuleList = nn.ModuleList()
        self.up_samples: nn.ModuleList = nn.ModuleList()

        for mult in reversed(channel_mults):
            out_ch = model_channels * mult
            self.up_samples.append(Upsample(ch))
            resolution *= 2
            level_blocks = []
            for _ in range(num_res_blocks + 1):
                skip_ch = skip_channels.pop()
                level_blocks.append(ResBlock(ch + skip_ch, out_ch, time_emb_dim, dropout=dropout))
                if resolution in attn_resolutions:
                    level_blocks.append(SpatialTransformer(out_ch, context_dim, heads=heads))
                ch = out_ch
            self.up_blocks.append(nn.ModuleList(level_blocks))

        self.out_norm = nn.GroupNorm(32, ch)
        self.out_proj = nn.Conv2d(ch, in_channels, 3, padding=1)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:          Noisy latent, shape (B, C, H, W).
            timesteps:  Diffusion timestep indices, shape (B,).
            context:    Conditioning embeddings, shape (B, seq_len, context_dim).

        Returns:
            Predicted noise, same shape as `x`.
        """
        t_emb = self.time_embed(sinusoidal_embedding(timesteps, self.time_embed.net[0].in_features))

        h = self.input_proj(x)
        skips: list[torch.Tensor] = [h]

        # encoder
        for level_blocks, down in zip(self.down_blocks, self.down_samples):
            for block in level_blocks:
                if isinstance(block, ResBlock):
                    h = block(h, t_emb)
                else:
                    h = block(h, context)
                skips.append(h)
            h = down(h)

        # bottleneck
        h = self.mid_res1(h, t_emb)
        h = self.mid_attn(h, context)
        h = self.mid_res2(h, t_emb)

        # decoder
        for up, level_blocks in zip(self.up_samples, self.up_blocks):
            h = up(h)
            for block in level_blocks:
                if isinstance(block, ResBlock):
                    h = torch.cat([h, skips.pop()], dim=1)
                    h = block(h, t_emb)
                else:
                    h = block(h, context)

        return self.out_proj(F.silu(self.out_norm(h)))
