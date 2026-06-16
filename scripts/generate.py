#!/usr/bin/env python
"""
Generate images from text prompts using a trained checkpoint.

Usage:
    python scripts/generate.py \\
        --checkpoint outputs/ckpt_final.pt \\
        --config     configs/default.yaml  \\
        --prompts    "a mountain at sunset" "a futuristic city at night" \\
        --steps      50 \\
        --guidance   7.5 \\
        --out        results/
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
import torchvision.utils as vutils
import yaml
import types

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Text-to-image generation")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--config", type=Path, default="configs/default.yaml")
    p.add_argument("--prompts", nargs="+", required=True)
    p.add_argument("--steps", type=int, default=50, help="Number of DDIM sampling steps")
    p.add_argument("--guidance", type=float, default=7.5, help="Classifier-free guidance scale")
    p.add_argument("--eta", type=float, default=0.0, help="DDIM eta (0 = deterministic)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=Path("results"))
    return p.parse_args()


def load_config(path: Path) -> types.SimpleNamespace:
    with open(path) as f:
        raw = yaml.safe_load(f)
    flat: dict = {}
    for section in raw.values():
        flat.update(section)
    return types.SimpleNamespace(**flat)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    args.out.mkdir(parents=True, exist_ok=True)

    # ── load models ───────────────────────────────────────────────────────────
    from src.models.unet import UNet
    from src.models.diffusion import NoiseScheduler
    from src.models.vae import VAE
    from src.models.text_encoder import CLIPTextEncoder

    unet = UNet(
        in_channels=cfg.in_channels,
        model_channels=cfg.model_channels,
        channel_mults=tuple(cfg.channel_mults),
        context_dim=cfg.context_dim,
        num_res_blocks=cfg.num_res_blocks,
        attn_resolutions=tuple(cfg.attn_resolutions),
        dropout=0.0,  # disabled at inference
        heads=cfg.heads,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    # load EMA weights if available
    if "ema_shadow" in ckpt:
        unet.load_state_dict(
            {k: v for k, v in ckpt["ema_shadow"].items()},
            strict=False,
        )
        logger.info("Loaded EMA weights from checkpoint.")
    else:
        unet.load_state_dict(ckpt["unet"])
    unet.eval()

    scheduler = NoiseScheduler(
        num_timesteps=cfg.num_timesteps,
        beta_start=cfg.beta_start,
        beta_end=cfg.beta_end,
        schedule=cfg.schedule,
    ).to(device)

    vae = VAE(cfg.vae_model).to(device)
    text_encoder = CLIPTextEncoder(cfg.clip_model, context_dim=cfg.context_dim).to(device)

    # ── encode prompts ────────────────────────────────────────────────────────
    prompts = args.prompts
    context = text_encoder(prompts, device=device)
    uncond = text_encoder.encode_unconditional(len(prompts), device=device)

    logger.info(f"Generating {len(prompts)} image(s) with {args.steps} DDIM steps…")

    # ── sample ────────────────────────────────────────────────────────────────
    latent_size = cfg.image_size // 8
    shape = (len(prompts), cfg.in_channels, latent_size, latent_size)

    latents = scheduler.ddim_sample(
        model=unet,
        shape=shape,
        num_inference_steps=args.steps,
        eta=args.eta,
        context=context,
        uncond_context=uncond,
        guidance_scale=args.guidance,
        device=device,
    )
    images = vae.decode(latents).clamp(-1, 1)

    # ── save ──────────────────────────────────────────────────────────────────
    for i, (img, prompt) in enumerate(zip(images, prompts)):
        # slug the prompt for a readable filename
        slug = "_".join(prompt.lower().split())[:50]
        path = args.out / f"{i:03d}_{slug}.png"
        vutils.save_image((img + 1) / 2, path)
        logger.info(f"Saved → {path}")

    # also save a grid for quick overview
    grid_path = args.out / "grid.png"
    vutils.save_image((images + 1) / 2, grid_path, nrow=min(4, len(images)))
    logger.info(f"Grid saved → {grid_path}")


if __name__ == "__main__":
    main()
