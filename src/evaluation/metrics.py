"""
Generation quality metrics: FID and LPIPS.

Both metrics follow standard protocols from the literature.
FID uses Inception-v3 features; LPIPS uses AlexNet by default.
"""

from __future__ import annotations

import logging
from typing import Generator

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Fréchet Inception Distance
# ──────────────────────────────────────────────────────────────────────────────

class FIDScore:
    """
    Computes FID between a set of real and generated images.

    Usage::

        fid = FIDScore(device="cuda")
        fid.update_real(real_loader)
        fid.update_fake(generated_images)   # (N, 3, H, W) tensor in [-1, 1]
        score = fid.compute()

    Requires ``torchmetrics`` (``pip install torchmetrics[image]``).
    """

    def __init__(self, device: str | torch.device = "cpu") -> None:
        from torchmetrics.image.fid import FrechetInceptionDistance

        self.metric = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
        self.device = device

    def update_real(self, loader: DataLoader) -> None:
        for images, _ in loader:
            self.metric.update(self._normalise(images.to(self.device)), real=True)

    def update_fake(self, images: torch.Tensor) -> None:
        self.metric.update(self._normalise(images.to(self.device)), real=False)

    def compute(self) -> float:
        return float(self.metric.compute().item())

    def reset(self) -> None:
        self.metric.reset()

    @staticmethod
    def _normalise(x: torch.Tensor) -> torch.Tensor:
        """Convert [-1, 1] images to [0, 1] for torchmetrics."""
        return (x.clamp(-1, 1) + 1) / 2


# ──────────────────────────────────────────────────────────────────────────────
# LPIPS (Learned Perceptual Image Patch Similarity)
# ──────────────────────────────────────────────────────────────────────────────

class LPIPSScore:
    """
    Computes mean LPIPS between pairs of images.

    Requires ``lpips`` (``pip install lpips``).
    """

    def __init__(self, net: str = "alex", device: str | torch.device = "cpu") -> None:
        import lpips

        self.fn = lpips.LPIPS(net=net).to(device)
        self.device = device

    @torch.no_grad()
    def __call__(self, imgs_a: torch.Tensor, imgs_b: torch.Tensor) -> float:
        """
        Args:
            imgs_a, imgs_b: Tensors of shape (N, 3, H, W) in [-1, 1].

        Returns:
            Mean LPIPS distance as a Python float.
        """
        imgs_a = imgs_a.to(self.device)
        imgs_b = imgs_b.to(self.device)
        return float(self.fn(imgs_a, imgs_b).mean().item())


# ──────────────────────────────────────────────────────────────────────────────
# CLIP-based alignment score
# ──────────────────────────────────────────────────────────────────────────────

class CLIPScore:
    """
    Measures semantic alignment between generated images and their text prompts.

    A higher score indicates the image better matches the prompt.
    Requires ``transformers``.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: str | torch.device = "cpu",
    ) -> None:
        from transformers import CLIPModel, CLIPProcessor

        self.model = CLIPModel.from_pretrained(model_name).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.device = device

    @torch.no_grad()
    def __call__(self, images: torch.Tensor, prompts: list[str]) -> float:
        """
        Args:
            images:  (N, 3, H, W) in [-1, 1].
            prompts: List of N text prompts.

        Returns:
            Mean cosine similarity in [0, 1].
        """
        # CLIP expects PIL images or uint8 numpy arrays
        pil_images = [
            _tensor_to_pil(img) for img in images.clamp(-1, 1)
        ]
        inputs = self.processor(text=prompts, images=pil_images, return_tensors="pt", padding=True).to(self.device)
        out = self.model(**inputs)

        img_emb = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
        txt_emb = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
        similarities = (img_emb * txt_emb).sum(dim=-1)
        return float(similarities.mean().item())


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _tensor_to_pil(tensor: torch.Tensor):
    from PIL import Image

    arr = ((tensor.permute(1, 2, 0).cpu().numpy() + 1) * 127.5).clip(0, 255).astype("uint8")
    return Image.fromarray(arr)
