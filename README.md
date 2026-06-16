# Latent Diffusion with Text Conditioning

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<your-username>/latent-diffusion-conditioned/blob/main/demo.ipynb)
[![CI](https://github.com/<your-username>/latent-diffusion-conditioned/actions/workflows/ci.yml/badge.svg)](https://github.com/<your-username>/latent-diffusion-conditioned/actions)

A clean, research-grade implementation of a **text-conditioned latent diffusion model** built from scratch in PyTorch. The model operates in the compressed latent space of a pretrained VAE and conditions generation on language via a frozen CLIP text encoder — following the design of Rombach et al. (2022) with a self-contained, fully modular codebase.

Supports both **DDPM stochastic** and **DDIM deterministic** sampling, **classifier-free guidance (CFG)**, mixed-precision training, EMA weight tracking, and W&B logging — all configurable from a single YAML file.

---

## Architecture overview

```
Text prompt
    │
    ▼
CLIP text encoder (frozen)
    │  token embeddings  (B, 77, 768)
    ▼
┌─────────────────────────────────┐
│           U-Net                 │
│  encoder → bottleneck → decoder │
│  ResBlocks + CrossAttention     │◄── timestep embedding
└─────────────────────────────────┘
    │  predicted noise  (B, 4, H/8, W/8)
    ▼
Reverse diffusion (DDPM / DDIM)
    │  clean latent
    ▼
VAE decoder (frozen)
    │
    ▼
Generated image  (B, 3, H, W)
```

**Key design choices:**

- **Latent space**: a pretrained VAE compresses 256×256 images into 32×32×4 latents, reducing the sequence length the diffusion model must process by 64×.
- **Conditioning**: cross-attention layers at multiple U-Net resolutions allow every spatial position to attend over the full token sequence from CLIP.
- **CFG**: during training, conditioning is randomly dropped with probability `p_uncond`; at inference the unconditional and conditional predictions are interpolated with a guidance scale.
- **DDIM**: a 50-step deterministic sampler replaces the full 1 000-step DDPM chain, enabling fast inference without retraining.

---

## Project structure

```
latent-diffusion-conditioned/
├── src/
│   ├── models/
│   │   ├── unet.py           # U-Net backbone with cross-attention
│   │   ├── diffusion.py      # DDPM schedule + DDIM sampler + loss
│   │   ├── vae.py            # Frozen VAE wrapper
│   │   └── text_encoder.py   # Frozen CLIP text encoder
│   ├── data/
│   │   └── dataset.py        # CSV-manifest and HuggingFace dataset adapters
│   ├── training/
│   │   └── trainer.py        # Training loop, EMA, mixed precision, logging
│   └── evaluation/
│       └── metrics.py        # FID, LPIPS, CLIP score
├── configs/
│   └── default.yaml          # All hyperparameters in one place
├── scripts/
│   ├── train.py              # Training entry point
│   └── generate.py           # Inference / image generation
├── tests/
│   └── test_unet.py          # Unit tests for U-Net and scheduler
├── .github/workflows/ci.yml  # GitHub Actions: lint + test
├── Dockerfile
├── pyproject.toml
└── requirements.txt
```

---

## Interactive demo

`demo.ipynb` runs **entirely on CPU** without pretrained weights or external data. It covers:

- U-Net instantiation and parameter breakdown
- Single forward pass with shape verification
- Noise schedule visualisation (cosine vs linear)
- Progressive signal degradation via the forward process
- End-to-end DDIM sampling loop (20 steps)
- Diffusion MSE loss computation
- Full unit test suite

Open directly in Colab with the badge above, or run locally:

```bash
pip install jupyter
jupyter notebook demo.ipynb
```

---

## Quickstart

### 1 — Install dependencies

```bash
git clone https://github.com/<your-username>/latent-diffusion-conditioned.git
cd latent-diffusion-conditioned

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

> **GPU note**: the `requirements.txt` installs the default PyTorch build. For a specific CUDA version:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```

### 2 — Prepare data

Create a CSV file with two columns:

```
image_path,caption
images/00001.jpg,a golden retriever puppy running on a beach
images/00002.jpg,a minimalist kitchen with white marble countertops
...
```

Point to it in `configs/default.yaml` under `data.manifest_path`.

### 3 — Train

```bash
python scripts/train.py --config configs/default.yaml
```

Resume from a checkpoint:

```bash
python scripts/train.py --config configs/default.yaml --resume outputs/ckpt_step_0010000.pt
```

Quick sanity-check (32 samples, 1 epoch, no W&B):

```bash
python scripts/train.py --config configs/default.yaml --debug
```

### 4 — Generate

```bash
python scripts/generate.py \
    --checkpoint outputs/ckpt_final.pt \
    --config     configs/default.yaml  \
    --prompts    "a mountain lake at dusk" "an abstract expressionist painting" \
    --steps      50 \
    --guidance   7.5 \
    --out        results/
```

Outputs are saved as individual PNGs and a single grid image under `results/`.

---

## Docker

Build and run in a fully isolated environment:

```bash
docker build -t latent-diffusion .

# Training (mount local data and output directories)
docker run --gpus all \
    -v $(pwd)/data:/workspace/data \
    -v $(pwd)/outputs:/workspace/outputs \
    latent-diffusion

# Generation
docker run --gpus all \
    -v $(pwd)/outputs:/workspace/outputs \
    -v $(pwd)/results:/workspace/results \
    latent-diffusion \
    python scripts/generate.py \
        --checkpoint outputs/ckpt_final.pt \
        --prompts "a cyberpunk cityscape at night"
```

---

## Configuration

All hyperparameters live in `configs/default.yaml`. Key sections:

| Section    | Key parameters |
|------------|---------------|
| `data`     | `image_size`, `manifest_path`, `augment` |
| `model`    | `model_channels`, `channel_mults`, `attn_resolutions`, `schedule` |
| `training` | `learning_rate`, `batch_size`, `grad_accum_steps`, `ema_decay`, `p_uncond` |
| `sampling` | `guidance_scale`, `val_inference_steps`, `sample_prompts` |
| `logging`  | `use_wandb`, `save_every`, `sample_every` |

To override individual values without editing the file:

```bash
python scripts/train.py --config configs/default.yaml
# then edit the YAML or create a new config that extends the default
```

---

## Evaluation

FID, LPIPS, and CLIP alignment score are implemented in `src/evaluation/metrics.py` and can be run independently:

```python
from src.evaluation.metrics import FIDScore, LPIPSScore, CLIPScore

fid = FIDScore(device="cuda")
fid.update_real(real_loader)
fid.update_fake(generated_images)   # (N, 3, H, W) in [-1, 1]
print(fid.compute())                # lower is better

clip_score = CLIPScore(device="cuda")
print(clip_score(generated_images, prompts))  # higher is better
```

---

## Tests

```bash
pytest tests/ -v
pytest tests/ --cov=src --cov-report=term-missing
```

CI runs automatically on every push via GitHub Actions (`.github/workflows/ci.yml`).

---

## References

- Ho et al., *Denoising Diffusion Probabilistic Models* (NeurIPS 2020). [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)
- Song et al., *Denoising Diffusion Implicit Models* (ICLR 2021). [arXiv:2010.02502](https://arxiv.org/abs/2010.02502)
- Rombach et al., *High-Resolution Image Synthesis with Latent Diffusion Models* (CVPR 2022). [arXiv:2112.10752](https://arxiv.org/abs/2112.10752)
- Radford et al., *Learning Transferable Visual Models From Natural Language Supervision* (ICML 2021). [arXiv:2103.00020](https://arxiv.org/abs/2103.00020)

---

## License

MIT
