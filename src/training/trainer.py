"""
Training loop for the latent diffusion model.

Features:
  - Mixed-precision (fp16 / bf16) via torch.amp
  - Gradient accumulation for large effective batch sizes
  - Exponential Moving Average (EMA) of model weights
  - Periodic validation loss tracking
  - Weights & Biases logging with image samples
  - Checkpoint saving / resuming
"""

from __future__ import annotations

import logging
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Exponential Moving Average
# ──────────────────────────────────────────────────────────────────────────────

class EMA:
    """Maintains an exponential moving average copy of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {
            name: param.data.clone() for name, param in model.named_parameters() if param.requires_grad
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if not param.requires_grad or name not in self.shadow:
                continue
            self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def apply(self, model: nn.Module) -> None:
        """Copy EMA weights into model (for evaluation / sampling)."""
        for name, param in model.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name])


# ──────────────────────────────────────────────────────────────────────────────
# Trainer
# ──────────────────────────────────────────────────────────────────────────────

class DiffusionTrainer:
    """
    Encapsulates the full training lifecycle for latent diffusion.

    Args:
        unet:                 Noise-prediction U-Net.
        vae:                  Frozen VAE for encoding images to latents.
        text_encoder:         Frozen CLIP text encoder.
        scheduler:            NoiseScheduler instance.
        train_loader:         DataLoader yielding (image_batch, caption_batch).
        val_loader:           Optional validation DataLoader.
        cfg:                  Config dataclass / SimpleNamespace with training hyperparams.
        output_dir:           Directory for checkpoints and samples.
    """

    def __init__(
        self,
        unet: nn.Module,
        vae: nn.Module,
        text_encoder: nn.Module,
        scheduler: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        cfg,
        output_dir: str | Path = "outputs",
    ) -> None:
        self.unet = unet
        self.vae = vae
        self.text_encoder = text_encoder
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._move_to_device()

        self.optimizer = AdamW(
            self.unet.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        self.lr_scheduler = CosineAnnealingLR(self.optimizer, T_max=cfg.num_epochs)
        self.ema = EMA(self.unet, decay=cfg.ema_decay)

        self.scaler = torch.cuda.amp.GradScaler(enabled=cfg.mixed_precision)
        self.autocast_ctx = (
            torch.cuda.amp.autocast(dtype=torch.bfloat16)
            if cfg.mixed_precision
            else nullcontext()
        )

        self._wandb_run = None
        if cfg.use_wandb:
            self._init_wandb()

    # ── training entry point ─────────────────────────────────────────────────

    def train(self) -> None:
        global_step = self._load_checkpoint()

        for epoch in range(self.cfg.num_epochs):
            self.unet.train()
            epoch_loss = 0.0
            self.optimizer.zero_grad()

            for step, (images, captions) in enumerate(self.train_loader):
                loss = self._train_step(images, captions)
                epoch_loss += loss

                # gradient accumulation
                if (step + 1) % self.cfg.grad_accum_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.unet.parameters(), self.cfg.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    self.ema.update(self.unet)
                    global_step += 1

                    if global_step % self.cfg.log_every == 0:
                        avg = epoch_loss / (step + 1)
                        logger.info(f"[epoch {epoch} | step {global_step}] loss={avg:.4f}")
                        self._log({"train/loss": avg, "train/lr": self.optimizer.param_groups[0]["lr"]}, global_step)

                    if global_step % self.cfg.save_every == 0:
                        self._save_checkpoint(global_step)

                    if global_step % self.cfg.sample_every == 0 and self.cfg.sample_prompts:
                        self._log_samples(global_step)

            self.lr_scheduler.step()

            if self.val_loader:
                val_loss = self._validate()
                logger.info(f"[epoch {epoch}] val_loss={val_loss:.4f}")
                self._log({"val/loss": val_loss}, global_step)

        self._save_checkpoint(global_step, final=True)

    # ── private helpers ───────────────────────────────────────────────────────

    def _train_step(self, images: torch.Tensor, captions: list[str]) -> float:
        images = images.to(self.device)

        with self.autocast_ctx:
            latents = self.vae.encode(images)
            context = self.text_encoder(captions, device=self.device)
            loss = self.scheduler.loss(self.unet, latents, context, p_uncond=self.cfg.p_uncond)
            loss = loss / self.cfg.grad_accum_steps

        self.scaler.scale(loss).backward()
        return loss.item() * self.cfg.grad_accum_steps

    @torch.no_grad()
    def _validate(self) -> float:
        self.unet.eval()
        total, n = 0.0, 0
        for images, captions in self.val_loader:
            images = images.to(self.device)
            latents = self.vae.encode(images)
            context = self.text_encoder(captions, device=self.device)
            loss = self.scheduler.loss(self.unet, latents, context, p_uncond=0.0)
            total += loss.item()
            n += 1
        return total / max(n, 1)

    @torch.no_grad()
    def _log_samples(self, step: int) -> None:
        if not self.cfg.sample_prompts:
            return
        self.unet.eval()
        self.ema.apply(self.unet)

        context = self.text_encoder(self.cfg.sample_prompts, device=self.device)
        uncond = self.text_encoder.encode_unconditional(len(self.cfg.sample_prompts), device=self.device)

        shape = (len(self.cfg.sample_prompts), 4, self.cfg.image_size // 8, self.cfg.image_size // 8)
        latents = self.scheduler.ddim_sample(
            self.unet,
            shape=shape,
            num_inference_steps=self.cfg.val_inference_steps,
            context=context,
            uncond_context=uncond,
            guidance_scale=self.cfg.guidance_scale,
            device=self.device,
        )
        images = self.vae.decode(latents).clamp(-1, 1)

        if self._wandb_run:
            import wandb
            grid = [(img.permute(1, 2, 0).cpu().numpy() * 127.5 + 127.5).astype("uint8") for img in images]
            self._wandb_run.log(
                {"samples": [wandb.Image(img, caption=cap) for img, cap in zip(grid, self.cfg.sample_prompts)]},
                step=step,
            )

        self.unet.train()

    def _save_checkpoint(self, step: int, final: bool = False) -> None:
        tag = "final" if final else f"step_{step:07d}"
        path = self.output_dir / f"ckpt_{tag}.pt"
        torch.save(
            {
                "step": step,
                "unet": self.unet.state_dict(),
                "ema_shadow": self.ema.shadow,
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.lr_scheduler.state_dict(),
            },
            path,
        )
        logger.info(f"Checkpoint saved → {path}")

    def _load_checkpoint(self) -> int:
        ckpt_path = getattr(self.cfg, "resume_from", None)
        if not ckpt_path:
            return 0
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.unet.load_state_dict(ckpt["unet"])
        self.ema.shadow = ckpt["ema_shadow"]
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.lr_scheduler.load_state_dict(ckpt["scheduler"])
        step = ckpt["step"]
        logger.info(f"Resumed from {ckpt_path} (step {step})")
        return step

    def _move_to_device(self) -> None:
        self.unet.to(self.device)
        self.vae.to(self.device)
        self.text_encoder.to(self.device)
        self.scheduler.to(self.device)

    def _log(self, metrics: dict, step: int) -> None:
        if self._wandb_run:
            self._wandb_run.log(metrics, step=step)

    def _init_wandb(self) -> None:
        try:
            import wandb
            self._wandb_run = wandb.init(
                project=self.cfg.wandb_project,
                name=self.cfg.wandb_run_name,
                config=vars(self.cfg),
            )
        except ImportError:
            logger.warning("wandb not installed; skipping W&B logging.")
