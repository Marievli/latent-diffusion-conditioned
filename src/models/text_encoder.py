"""
CLIP text encoder wrapper for conditioning the diffusion model.

Freezes the CLIP backbone by default and exposes a clean interface for
returning per-token embeddings (for cross-attention) or pooled embeddings.
"""

from __future__ import annotations

from typing import Union

import torch
import torch.nn as nn


class CLIPTextEncoder(nn.Module):
    """
    Thin wrapper around a pretrained CLIP text encoder.

    Keeps the backbone frozen and optionally projects the output to a
    custom context dimension via a learned linear layer.

    Args:
        model_name:    HuggingFace model identifier, e.g. "openai/clip-vit-large-patch14".
        context_dim:   If set, projects CLIP hidden dim → context_dim.
        max_length:    Maximum token sequence length (default 77, matching CLIP).
        freeze:        Whether to freeze CLIP weights.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
        context_dim: int | None = None,
        max_length: int = 77,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        from transformers import CLIPTextModel, CLIPTokenizer

        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.encoder = CLIPTextModel.from_pretrained(model_name)
        self.max_length = max_length

        clip_dim = self.encoder.config.hidden_size  # 768 for ViT-L
        self.proj = nn.Linear(clip_dim, context_dim) if context_dim and context_dim != clip_dim else nn.Identity()

        if freeze:
            self._freeze()

    # ── public API ────────────────────────────────────────────────────────────

    def forward(
        self,
        texts: list[str],
        device: Union[str, torch.device] = "cpu",
    ) -> torch.Tensor:
        """
        Tokenise and encode a list of text prompts.

        Returns:
            Token-level embeddings of shape (B, seq_len, context_dim), suitable
            for cross-attention conditioning in the U-Net.
        """
        tokens = self.tokenizer(
            texts,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        with torch.set_grad_enabled(not self._is_frozen()):
            out = self.encoder(**tokens)

        embeddings = out.last_hidden_state  # (B, seq_len, clip_dim)
        return self.proj(embeddings)

    @torch.no_grad()
    def encode_unconditional(self, batch_size: int, device: Union[str, torch.device] = "cpu") -> torch.Tensor:
        """Return zero-filled unconditional embeddings for classifier-free guidance."""
        dummy = self.forward([""] * batch_size, device=device)
        return torch.zeros_like(dummy)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _freeze(self) -> None:
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.encoder.eval()

    def _is_frozen(self) -> bool:
        return not any(p.requires_grad for p in self.encoder.parameters())
