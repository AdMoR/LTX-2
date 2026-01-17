# LTX Server

FastAPI server for LTX-2 video generation pipelines.

## Features

- Async job queue with polling for long-running generation tasks
- S3/MinIO storage for generated videos
- Configurable pipeline loading at startup
- Multipart form upload for image/video conditioning
- OpenAPI documentation

## Quick Start

```bash
# Set environment variables
export LTX_PIPELINES=distilled,ic_lora
export LTX_CHECKPOINT_PATH=/models/ltx2.safetensors
export LTX_GEMMA_ROOT=/models/gemma
export S3_BUCKET=ltx-outputs
export S3_ENDPOINT=http://localhost:9000

# Run server
uvicorn ltx_server.main:app --host 0.0.0.0 --port 8000
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/generate/{pipeline}` | Submit generation job |
| `GET` | `/jobs/{job_id}` | Get job status and result URL |
| `GET` | `/jobs` | List recent jobs |
| `DELETE` | `/jobs/{job_id}` | Cancel pending job |
| `GET` | `/pipelines` | List available pipelines |
| `GET` | `/health` | Health check |

## Docker

See the root `docker-compose.yml` for containerized deployment with GPU support.
