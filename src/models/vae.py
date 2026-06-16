"""
Thin wrapper around a pretrained VAE for latent encoding / decoding.

Operates in the latent space defined by Rombach et al. (2022):
pixel space → (encode) → latent ∈ R^{B×4×H/8×W/8} → (decode) → pixel space.

The VAE weights are always kept frozen; only the diffusion backbone is trained.
"""

from __future__ import annotations

from typing import Union

import torch
import torch.nn as nn


_SCALE_FACTOR = 0.18215  # empirical scaling constant from Stable Diffusion


class VAE(nn.Module):
    """
    Pretrained VAE encoder/decoder with frozen weights.

    Args:
        model_name: HuggingFace repo id for the pretrained VAE, e.g.
                    "stabilityai/sd-vae-ft-mse".
    """

    def __init__(self, model_name: str = "stabilityai/sd-vae-ft-mse") -> None:
        super().__init__()
        from diffusers import AutoencoderKL

        self.vae: AutoencoderKL = AutoencoderKL.from_pretrained(model_name)
        for p in self.vae.parameters():
            p.requires_grad_(False)

    # ── encode ────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode pixel images to latents.

        Args:
            images: Normalised images in [-1, 1], shape (B, 3, H, W).

        Returns:
            Latent codes, shape (B, 4, H/8, W/8).
        """
        dist = self.vae.encode(images).latent_dist
        z = dist.sample() * _SCALE_FACTOR
        return z

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Decode latents back to pixel space.

        Args:
            latents: Shape (B, 4, H/8, W/8).

        Returns:
            Reconstructed images in [-1, 1], shape (B, 3, H, W).
        """
        return self.vae.decode(latents / _SCALE_FACTOR).sample
