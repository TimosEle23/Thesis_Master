# =============================================================================
# Full training + analysis Docker image for TGNN Benchmark
# Supports: CPU training, MPS passthrough (macOS), analysis, all experiments
# =============================================================================
#
# Build:   docker compose build
# Shell:   docker compose run --rm thesis
# Phase 1: docker compose run --rm phase1
# Phase 2: docker compose run --rm phase2
# Analyse: docker compose run --rm analyse
# =============================================================================
FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PyTorch (CPU) — for GPU, override with CUDA wheel at build time
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        torch==2.5.1 torchvision==0.20.1 \
        --index-url https://download.pytorch.org/whl/cpu

# Install PyTorch Geometric + extensions
# torch-sparse needs torch visible at build time, so use --no-build-isolation
RUN pip install --no-cache-dir --no-build-isolation torch-sparse torch-scatter && \
    pip install --no-cache-dir torch-geometric==2.7.0 && \
    pip install --no-cache-dir torch-geometric-temporal==0.56.2

# Install remaining dependencies
RUN pip install --no-cache-dir \
    "numpy>=1.24,<2" "pandas>=2.0" "scipy>=1.10" "scikit-learn>=1.3" \
    pyyaml omegaconf \
    matplotlib seaborn \
    tqdm tensorboard \
    aeon requests

# Copy project code (volumes override this in dev)
COPY . .

CMD ["/bin/bash"]
