"""Unit tests for the U-Net and diffusion components."""

import pytest
import torch
from src.models.unet import UNet, sinusoidal_embedding
from src.models.diffusion import NoiseScheduler


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def small_unet() -> UNet:
    """Minimal U-Net for fast tests (no GPU required)."""
    return UNet(
        in_channels=4,
        model_channels=32,
        channel_mults=(1, 2),
        context_dim=64,
        num_res_blocks=1,
        attn_resolutions=(8,),
        dropout=0.0,
        heads=2,
    )


@pytest.fixture
def scheduler() -> NoiseScheduler:
    return NoiseScheduler(num_timesteps=100, schedule="cosine")


@pytest.fixture
def batch():
    B, C, H, W = 2, 4, 16, 16
    x = torch.randn(B, C, H, W)
    t = torch.randint(0, 100, (B,))
    ctx = torch.randn(B, 10, 64)  # (batch, seq_len, context_dim)
    return x, t, ctx


# ──────────────────────────────────────────────────────────────────────────────
# Sinusoidal embedding
# ──────────────────────────────────────────────────────────────────────────────

def test_sinusoidal_embedding_shape():
    t = torch.arange(8)
    emb = sinusoidal_embedding(t, dim=128)
    assert emb.shape == (8, 128)


def test_sinusoidal_embedding_deterministic():
    t = torch.tensor([5, 10])
    assert torch.allclose(sinusoidal_embedding(t, 64), sinusoidal_embedding(t, 64))


# ──────────────────────────────────────────────────────────────────────────────
# U-Net forward pass
# ──────────────────────────────────────────────────────────────────────────────

def test_unet_output_shape(small_unet, batch):
    x, t, ctx = batch
    out = small_unet(x, t, ctx)
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"


def test_unet_no_context(small_unet, batch):
    """U-Net should run without conditioning (unconditional mode)."""
    x, t, _ = batch
    out = small_unet(x, t, context=None)
    assert out.shape == x.shape


def test_unet_gradient_flow(small_unet, batch):
    x, t, ctx = batch
    out = small_unet(x, t, ctx)
    loss = out.mean()
    loss.backward()
    for name, p in small_unet.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"No gradient for {name}"


# ──────────────────────────────────────────────────────────────────────────────
# Noise scheduler
# ──────────────────────────────────────────────────────────────────────────────

def test_scheduler_buffer_shapes(scheduler):
    assert scheduler.betas.shape == (100,)
    assert scheduler.alphas_cumprod.shape == (100,)
    assert (scheduler.betas > 0).all()
    assert (scheduler.betas < 1).all()


def test_q_sample_shape(scheduler):
    x0 = torch.randn(4, 4, 8, 8)
    t = torch.randint(0, 100, (4,))
    x_t, noise = scheduler.q_sample(x0, t)
    assert x_t.shape == x0.shape
    assert noise.shape == x0.shape


def test_q_sample_at_t0_close_to_original(scheduler):
    """At t=0, the noisy sample should be very close to x0."""
    x0 = torch.randn(2, 4, 8, 8)
    t = torch.zeros(2, dtype=torch.long)
    noise = torch.zeros_like(x0)
    x_t, _ = scheduler.q_sample(x0, t, noise=noise)
    # sqrt(alpha_bar_0) ≈ 1 for cosine schedule
    assert torch.allclose(x_t, x0, atol=0.05)


def test_diffusion_loss(small_unet, scheduler):
    x0 = torch.randn(2, 4, 16, 16)
    ctx = torch.randn(2, 5, 64)
    loss = scheduler.loss(small_unet, x0, ctx)
    assert loss.item() > 0
    assert not torch.isnan(loss)


# ──────────────────────────────────────────────────────────────────────────────
# DDIM sampling (smoke test — no GPU, very few steps)
# ──────────────────────────────────────────────────────────────────────────────

def test_ddim_sample_shape(small_unet, scheduler):
    shape = (2, 4, 16, 16)
    ctx = torch.randn(2, 5, 64)
    uncond = torch.zeros_like(ctx)
    out = scheduler.ddim_sample(
        small_unet,
        shape=shape,
        num_inference_steps=5,
        context=ctx,
        uncond_context=uncond,
        guidance_scale=2.0,
    )
    assert out.shape == shape


def test_ddim_deterministic(small_unet, scheduler):
    """Same seed → same output when eta=0."""
    shape = (1, 4, 8, 8)
    ctx = torch.randn(1, 5, 64)

    torch.manual_seed(0)
    out_a = scheduler.ddim_sample(small_unet, shape, num_inference_steps=3, context=ctx, eta=0.0)
    torch.manual_seed(0)
    out_b = scheduler.ddim_sample(small_unet, shape, num_inference_steps=3, context=ctx, eta=0.0)
    assert torch.allclose(out_a, out_b)
