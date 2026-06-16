# ──────────────────────────────────────────────────────────────────────────────
# Base image: CUDA 12.1 + cuDNN 8 + Ubuntu 22.04
# ──────────────────────────────────────────────────────────────────────────────
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Prevent interactive prompts during apt-get
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3-pip \
        git wget curl \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# ── Python dependencies ───────────────────────────────────────────────────────
WORKDIR /workspace
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Copy source ───────────────────────────────────────────────────────────────
COPY . .

# ── Make scripts executable ───────────────────────────────────────────────────
RUN chmod +x scripts/*.py

# ── Default command: run training ─────────────────────────────────────────────
CMD ["python", "scripts/train.py", "--config", "configs/default.yaml"]
