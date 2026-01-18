FROM nvcr.io/nvidia/cuda:13.0.1-devel-ubuntu24.04

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    python3-pip \
    ffmpeg \
    libsm6 \
    libxext6 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set up Python environment
ENV VIRTUAL_ENV=/opt/venv
RUN python3.12 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 13.0 support
RUN pip install --index-url https://download.pytorch.org/whl/cu130 \
    torch~=2.7 \
    torchaudio

# Install all Python dependencies
RUN pip install \
    # ltx-core dependencies
    einops \
    numpy \
    transformers \
    safetensors \
    accelerate \
    "scipy>=1.14" \
    # ltx-pipelines dependencies
    av \
    tqdm \
    pillow \
    # ltx-server dependencies
    "fastapi>=0.100.0" \
    "uvicorn[standard]" \
    python-multipart \
    boto3 \
    "pydantic>=2.0" \
    pydantic-settings

WORKDIR /app

# Copy source code directly (no build step needed)
COPY packages/ltx-core/src/ltx_core /app/ltx_core
COPY packages/ltx-pipelines/src/ltx_pipelines /app/ltx_pipelines
COPY packages/ltx-server/src/ltx_server /app/ltx_server

# Add app directory to Python path
ENV PYTHONPATH="/app:${PYTHONPATH}"

# Create temp directory
RUN mkdir -p /tmp/ltx

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "-m", "uvicorn", "ltx_server.main:app", "--host", "0.0.0.0", "--port", "8000"]
