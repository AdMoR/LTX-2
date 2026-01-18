FROM nvcr.io/nvidia/cuda:13.0.1-devel-ubuntu24.04
ARG DEBIAN_FRONTEND=noninteractive

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-venv \
    python3-pip \
    ffmpeg \
    libsm6 \
    libxext6 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y


# Set up Python environment
ENV VIRTUAL_ENV=/opt/venv
RUN python3.12 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install uv for fast package management
RUN pip install uv

WORKDIR /app

# Copy workspace files
COPY pyproject.toml uv.lock ./
COPY packages/ ./packages/

# Install dependencies
RUN uv sync

# Create temp directory
RUN mkdir -p /tmp/ltx

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["uv", "run", "uvicorn", "ltx_server.main:app", "--host", "0.0.0.0", "--port", "8000"]
