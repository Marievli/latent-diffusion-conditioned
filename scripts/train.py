#!/usr/bin/env python
"""
Training entry point.

Usage:
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --config configs/default.yaml --resume outputs/ckpt_step_0010000.pt
"""

from __future__ import annotations

import argparse
import logging
import os
import types
from pathlib import Path

import yaml
import torch
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train latent diffusion model")
    parser.add_argument("--config", type=Path, default="configs/default.yaml")
    parser.add_argument("--resume", type=Path, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--debug", action="store_true", help="Run with 32 samples for quick sanity-check")
    return parser.parse_args()


def load_config(path: Path) -> types.SimpleNamespace:
    with open(path) as f:
        raw = yaml.safe_load(f)

    # Flatten nested config into a single SimpleNamespace for convenience
    flat: dict = {}
    for section in raw.values():
        flat.update(section)
    return types.SimpleNamespace(**flat)


def build_dataloader(cfg, split: str = "train") -> DataLoader:
    from src.data.dataset import CaptionedImageDataset, collate_fn

    manifest = cfg.manifest_path if split == "train" else cfg.val_manifest_path
    dataset = CaptionedImageDataset(
        manifest_path=manifest,
        image_root=cfg.image_root,
        image_size=cfg.image_size,
        augment=(split == "train"),
        max_samples=32 if getattr(cfg, "debug", False) else None,
    )
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=(split == "train"),
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        collate_fn=collate_fn,
        drop_last=(split == "train"),
    )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    if args.debug:
        cfg.debug = True
        cfg.num_epochs = 1
        cfg.log_every = 1
        cfg.save_every = 9999
        cfg.sample_every = 9999
        cfg.use_wandb = False
        logger.warning("Debug mode enabled — running with 32 samples.")

    if args.resume:
        cfg.resume_from = str(args.resume)

    logger.info(f"Config: {args.config}")
    logger.info(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    # ── build components ──────────────────────────────────────────────────────
    from src.models.unet import UNet
    from src.models.diffusion import NoiseScheduler
    from src.models.vae import VAE
    from src.models.text_encoder import CLIPTextEncoder
    from src.training.trainer import DiffusionTrainer

    unet = UNet(
        in_channels=cfg.in_channels,
        model_channels=cfg.model_channels,
        channel_mults=tuple(cfg.channel_mults),
        context_dim=cfg.context_dim,
        num_res_blocks=cfg.num_res_blocks,
        attn_resolutions=tuple(cfg.attn_resolutions),
        dropout=cfg.dropout,
        heads=cfg.heads,
    )
    scheduler = NoiseScheduler(
        num_timesteps=cfg.num_timesteps,
        beta_start=cfg.beta_start,
        beta_end=cfg.beta_end,
        schedule=cfg.schedule,
    )
    vae = VAE(model_name=cfg.vae_model)
    text_encoder = CLIPTextEncoder(model_name=cfg.clip_model, context_dim=cfg.context_dim)

    n_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    logger.info(f"U-Net parameters: {n_params / 1e6:.1f}M")

    train_loader = build_dataloader(cfg, split="train")
    val_loader = build_dataloader(cfg, split="val") if hasattr(cfg, "val_manifest_path") else None

    trainer = DiffusionTrainer(
        unet=unet,
        vae=vae,
        text_encoder=text_encoder,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        output_dir=cfg.output_dir,
    )
    trainer.train()


if __name__ == "__main__":
    main()
