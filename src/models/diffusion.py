"""
DDPM noise schedule with DDIM deterministic sampler.

References:
  - Ho et al., "Denoising Diffusion Probabilistic Models" (2020), arXiv:2006.11239
  - Song et al., "Denoising Diffusion Implicit Models" (2020), arXiv:2010.02502
"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Union


class NoiseScheduler(nn.Module):
    """
    Linear or cosine beta schedule with pre-computed diffusion coefficients.

    All tensors are registered as buffers so they travel with the model to the
    correct device automatically.
    """

    def __init__(
        self,
        num_timesteps: int = 1_000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        schedule: str = "cosine",
    ) -> None:
        super().__init__()
        self.T = num_timesteps

        betas = self._build_schedule(schedule, num_timesteps, beta_start, beta_end)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", alphas_cumprod.sqrt())
        self.register_buffer("sqrt_one_minus_alphas_cumprod", (1.0 - alphas_cumprod).sqrt())
        self.register_buffer(
            "posterior_variance",
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )

    # ── schedule builders ────────────────────────────────────────────────────

    @staticmethod
    def _build_schedule(
        schedule: str,
        T: int,
        beta_start: float,
        beta_end: float,
    ) -> torch.Tensor:
        if schedule == "linear":
            return torch.linspace(beta_start, beta_end, T)
        if schedule == "cosine":
            steps = torch.arange(T + 1) / T
            f = torch.cos((steps + 0.008) / 1.008 * torch.tensor(torch.pi / 2)) ** 2
            betas = 1.0 - f[1:] / f[:-1]
            return betas.clamp(0, 0.999)
        raise ValueError(f"Unknown schedule: {schedule!r}. Choose 'linear' or 'cosine'.")

    # ── forward diffusion (q) ─────────────────────────────────────────────────

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample noisy latent at timestep t via the closed-form forward process.

        Returns:
            (x_t, noise) — noisy sample and the noise that was added.
        """
        noise = noise if noise is not None else torch.randn_like(x0)
        sqrt_a = self._extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_1ma = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        return sqrt_a * x0 + sqrt_1ma * noise, noise

    # ── DDPM reverse step ────────────────────────────────────────────────────

    @torch.no_grad()
    def p_sample(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        guidance_scale: float = 7.5,
        uncond_context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Single DDPM reverse step with optional classifier-free guidance."""
        predicted_noise = self._guided_prediction(
            model, x_t, t, context, guidance_scale, uncond_context
        )

        beta = self._extract(self.betas, t, x_t.shape)
        sqrt_1ma = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        sqrt_recip_a = (1.0 - beta).rsqrt()

        mean = sqrt_recip_a * (x_t - beta / sqrt_1ma * predicted_noise)
        var = self._extract(self.posterior_variance, t, x_t.shape)

        noise = torch.randn_like(x_t)
        nonzero = (t > 0).float().view(-1, 1, 1, 1)
        return mean + nonzero * var.sqrt() * noise

    # ── DDIM sampling ────────────────────────────────────────────────────────

    @torch.no_grad()
    def ddim_sample(
        self,
        model: nn.Module,
        shape: tuple[int, ...],
        num_inference_steps: int = 50,
        eta: float = 0.0,
        context: Optional[torch.Tensor] = None,
        guidance_scale: float = 7.5,
        uncond_context: Optional[torch.Tensor] = None,
        device: Union[str, torch.device] = "cpu",
    ) -> torch.Tensor:
        """
        DDIM deterministic sampler for fast inference.

        Args:
            model:               Noise-prediction network.
            shape:               Output shape (B, C, H, W) in latent space.
            num_inference_steps: Number of DDIM steps (typically 20–50).
            eta:                 Stochasticity parameter (0 = fully deterministic).
            context:             Conditional embeddings (B, seq, dim).
            guidance_scale:      CFG scale; 1.0 disables guidance.
            uncond_context:      Unconditional embeddings for CFG.
            device:              Target device.

        Returns:
            Denoised latent tensor of shape `shape`.
        """
        step_indices = np.linspace(0, self.T - 1, num_inference_steps, dtype=int)[::-1]
        x = torch.randn(shape, device=device)

        for i, t_idx in enumerate(step_indices):
            t = torch.full((shape[0],), t_idx, device=device, dtype=torch.long)

            eps = self._guided_prediction(model, x, t, context, guidance_scale, uncond_context)

            alpha = self.alphas_cumprod[t_idx]
            alpha_prev = self.alphas_cumprod[step_indices[i + 1]] if i + 1 < len(step_indices) else torch.ones(1)

            sigma = eta * ((1 - alpha_prev) / (1 - alpha) * (1 - alpha / alpha_prev)).sqrt()
            x0_pred = (x - (1 - alpha).sqrt() * eps) / alpha.sqrt()
            dir_xt = (1 - alpha_prev - sigma ** 2).clamp(min=0).sqrt() * eps
            noise = sigma * torch.randn_like(x) if eta > 0 else 0

            x = alpha_prev.sqrt() * x0_pred + dir_xt + noise

        return x

    # ── helpers ───────────────────────────────────────────────────────────────

    def _guided_prediction(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        context: Optional[torch.Tensor],
        scale: float,
        uncond_context: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Classifier-free guidance noise prediction."""
        if scale == 1.0 or uncond_context is None:
            return model(x_t, t, context)
        # batch uncond and cond for a single forward pass
        x_double = torch.cat([x_t, x_t])
        t_double = torch.cat([t, t])
        ctx_double = torch.cat([uncond_context, context])
        out_uncond, out_cond = model(x_double, t_double, ctx_double).chunk(2)
        return out_uncond + scale * (out_cond - out_uncond)

    @staticmethod
    def _extract(a: torch.Tensor, t: torch.Tensor, shape: tuple) -> torch.Tensor:
        """Gather coefficients at timestep t and reshape to broadcast with `shape`."""
        out = a.gather(0, t)
        return out.view(t.shape[0], *([1] * (len(shape) - 1)))

    def loss(
        self,
        model: nn.Module,
        x0: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        p_uncond: float = 0.1,
    ) -> torch.Tensor:
        """
        Simple diffusion loss (MSE on predicted noise) with random CFG dropout.

        Args:
            model:    Noise-prediction network.
            x0:       Clean latents, shape (B, C, H, W).
            context:  Conditioning embeddings.
            p_uncond: Probability of dropping the condition (CFG training).

        Returns:
            Scalar loss.
        """
        B = x0.shape[0]
        t = torch.randint(0, self.T, (B,), device=x0.device)
        x_t, noise = self.q_sample(x0, t)

        # classifier-free guidance dropout
        if context is not None and p_uncond > 0:
            mask = torch.rand(B, device=x0.device) < p_uncond
            context = context.clone()
            context[mask] = 0.0  # zero-out condition → unconditional

        predicted = model(x_t, t, context)
        return F.mse_loss(predicted, noise)


import torch.nn.functional as F  # noqa: E402 (placed after class to avoid circular-looking import)
