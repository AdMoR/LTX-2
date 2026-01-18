# LTX Server

FastAPI server for LTX-2 video generation pipelines.

## Features

- **Async Job Queue**: Long-running generation tasks run asynchronously with polling
- **S3/MinIO Storage**: Generated videos stored with presigned URL access
- **Multiple Pipelines**: Supports distilled, two-stage, IC-LoRA, and keyframe interpolation
- **Image Conditioning**: Multipart form upload for image/video conditioning
- **OpenAPI Documentation**: Auto-generated interactive docs at `/docs`

## Quick Start

```bash
# Set environment variables
export LTX_PIPELINES=distilled,ic_lora
export LTX_CHECKPOINT_PATH=/models/ltx2.safetensors
export LTX_GEMMA_ROOT=/models/gemma
export S3_BUCKET=ltx-outputs
export S3_ENDPOINT=http://localhost:9000
export AWS_ACCESS_KEY_ID=your-access-key
export AWS_SECRET_ACCESS_KEY=your-secret-key

# Run server
uvicorn ltx_server.main:app --host 0.0.0.0 --port 8000
```

---

## Configuration

### Server Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8000` | Server port |

### Pipeline Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `LTX_PIPELINES` | `distilled` | Comma-separated list of pipelines to load |
| `LTX_CHECKPOINT_PATH` | `/models/ltx2.safetensors` | Path to LTX-2 model checkpoint |
| `LTX_GEMMA_ROOT` | `/models/gemma` | Path to Gemma text encoder directory |
| `LTX_SPATIAL_UPSAMPLER_PATH` | `/models/upsampler.safetensors` | Path to spatial upsampler model |
| `LTX_DISTILLED_LORA_PATH` | `None` | Path to distilled LoRA for two-stage pipelines |
| `LTX_FP8_TRANSFORMER` | `false` | Enable FP8 mode for transformer |
| `LTX_TEXT_ENCODER_CPU` | `false` | Load text encoder on CPU |
| `LTX_TEXT_ENCODER_8BIT` | `false` | Use 8-bit quantization for text encoder |
| `LTX_TEXT_ENCODER_4BIT` | `false` | Use 4-bit quantization for text encoder |

### S3/MinIO Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `S3_BUCKET` | `ltx-outputs` | S3 bucket name for storing outputs |
| `S3_ENDPOINT` | `http://minio:9000` | S3/MinIO endpoint URL |
| `S3_REGION` | `us-east-1` | S3 region |
| `AWS_ACCESS_KEY_ID` | - | AWS/MinIO access key |
| `AWS_SECRET_ACCESS_KEY` | - | AWS/MinIO secret key |
| `S3_PRESIGNED_URL_EXPIRY` | `3600` | Presigned URL expiry time (seconds) |

---

## API Reference

### Health Check

#### `GET /health`

Check server health and status.

**Response:**

```json
{
  "status": "healthy",
  "version": "0.1.0",
  "pipelines_loaded": ["distilled", "ic_lora"],
  "gpu_available": true
}
```

---

### Pipelines

#### `GET /pipelines`

List all available pipelines with their capabilities.

**Response:**

```json
{
  "pipelines": [
    {
      "name": "distilled",
      "description": "Fast two-stage generation without CFG",
      "supports_image_conditioning": true,
      "supports_video_conditioning": false,
      "supports_negative_prompt": false,
      "is_two_stage": true
    },
    {
      "name": "ic_lora",
      "description": "IC-LoRA pipeline with video conditioning",
      "supports_image_conditioning": true,
      "supports_video_conditioning": true,
      "supports_negative_prompt": false,
      "is_two_stage": false
    }
  ]
}
```

---

### Jobs

#### `GET /jobs`

List recent generation jobs.

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `limit` | int | `50` | Number of jobs to return (1-100) |
| `offset` | int | `0` | Pagination offset |

**Response:**

```json
{
  "jobs": [
    {
      "job_id": "abc123",
      "status": "completed",
      "pipeline": "distilled",
      "created_at": "2024-01-15T10:30:00Z",
      "started_at": "2024-01-15T10:30:01Z",
      "completed_at": "2024-01-15T10:31:30Z",
      "video_url": "https://s3.example.com/video.mp4?...",
      "audio_url": null,
      "error": null,
      "progress": 1.0
    }
  ],
  "total": 1
}
```

#### `GET /jobs/{job_id}`

Get status and results for a specific job.

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `job_id` | string | Unique job identifier |

**Response:**

```json
{
  "job_id": "abc123",
  "status": "processing",
  "pipeline": "distilled",
  "created_at": "2024-01-15T10:30:00Z",
  "started_at": "2024-01-15T10:30:01Z",
  "completed_at": null,
  "video_url": null,
  "audio_url": null,
  "error": null,
  "progress": 0.45
}
```

**Job Status Values:**

| Status | Description |
|--------|-------------|
| `pending` | Job queued, waiting for processing |
| `processing` | Currently generating video |
| `completed` | Successfully completed, video URL available |
| `failed` | Generation failed, check `error` field |
| `cancelled` | Job was cancelled |

#### `DELETE /jobs/{job_id}`

Cancel a pending job.

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `job_id` | string | Unique job identifier |

**Response:** Returns the job with status `cancelled`.

**Error:** Returns 400 if job is already processing or completed.

---

### Generation Endpoints

All generation endpoints accept `multipart/form-data` and return a `JobResponse`.

#### Common Parameters

These parameters are shared across all generation endpoints:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | string | **required** | Text prompt describing desired video |
| `seed` | int | `42` | Random seed for reproducible generation |
| `height` | int | `768` | Video height in pixels |
| `width` | int | `1152` | Video width in pixels |
| `num_frames` | int | `97` | Number of frames (`num_frames = 8k + 1`) |
| `frame_rate` | float | `24.0` | Output video frame rate |
| `enhance_prompt` | bool | `false` | Enhance prompt using text encoder |

#### Image Conditioning (Optional)

For pipelines that support image conditioning:

| Parameter | Type | Description |
|-----------|------|-------------|
| `images` | file[] | One or more image files |
| `image_frame_indices` | int[] | Target frame indices for each image |
| `image_strengths` | float[] | Conditioning strength for each image (0.0-1.0) |

---

#### `POST /generate/distilled`

Fast two-stage generation without classifier-free guidance.

**Parameters:** Common parameters only.

**Example:**

```bash
curl -X POST http://localhost:8000/generate/distilled \
  -F "prompt=A serene lake at sunset with mountains in the background" \
  -F "seed=42" \
  -F "num_frames=97"
```

**With image conditioning:**

```bash
curl -X POST http://localhost:8000/generate/distilled \
  -F "prompt=A cat walking through a garden" \
  -F "images=@first_frame.png" \
  -F "image_frame_indices=0" \
  -F "image_strengths=1.0"
```

---

#### `POST /generate/ti2vid_one_stage`

Single-stage text/image-to-video generation with classifier-free guidance.

**Additional Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `negative_prompt` | string | `"worst quality, inconsistent motion, blurry, jittery, distorted"` | What to avoid in generation |
| `num_inference_steps` | int | `40` | Number of denoising steps |
| `cfg_guidance_scale` | float | `3.0` | Classifier-free guidance scale |

**Example:**

```bash
curl -X POST http://localhost:8000/generate/ti2vid_one_stage \
  -F "prompt=A rocket launching into space" \
  -F "negative_prompt=blurry, low quality" \
  -F "num_inference_steps=50" \
  -F "cfg_guidance_scale=4.0"
```

---

#### `POST /generate/ti2vid_two_stages`

Two-stage text/image-to-video generation with upscaling.

**Additional Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `negative_prompt` | string | `"worst quality, inconsistent motion, blurry, jittery, distorted"` | What to avoid in generation |
| `num_inference_steps` | int | `40` | Number of denoising steps (stage 1) |
| `cfg_guidance_scale` | float | `3.0` | CFG scale (stage 1) |

**Example:**

```bash
curl -X POST http://localhost:8000/generate/ti2vid_two_stages \
  -F "prompt=Ocean waves crashing on rocky shore" \
  -F "height=768" \
  -F "width=1152" \
  -F "num_frames=97"
```

---

#### `POST /generate/ic_lora`

IC-LoRA pipeline with optional video conditioning for style transfer and video-to-video generation.

**Additional Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `video_conditioning` | file | `None` | Video file for conditioning |
| `video_conditioning_strength` | float | `1.0` | Strength of video conditioning |

**Example with video conditioning:**

```bash
curl -X POST http://localhost:8000/generate/ic_lora \
  -F "prompt=A cyberpunk city at night with neon lights" \
  -F "video_conditioning=@input_video.mp4" \
  -F "video_conditioning_strength=0.8"
```

---

#### `POST /generate/keyframe_interpolation`

Keyframe interpolation between provided images.

**Additional Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `negative_prompt` | string | `"worst quality, inconsistent motion, blurry, jittery, distorted"` | What to avoid |
| `num_inference_steps` | int | `40` | Number of denoising steps |
| `cfg_guidance_scale` | float | `3.0` | Classifier-free guidance scale |

**Example (interpolating between two keyframes):**

```bash
curl -X POST http://localhost:8000/generate/keyframe_interpolation \
  -F "prompt=A flower blooming in timelapse" \
  -F "images=@keyframe_start.png" \
  -F "images=@keyframe_end.png" \
  -F "image_frame_indices=0" \
  -F "image_frame_indices=96" \
  -F "image_strengths=1.0" \
  -F "image_strengths=1.0" \
  -F "num_frames=97"
```

---

## Response Format

### Job Response

All generation endpoints return a `JobResponse`:

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "pending",
  "pipeline": "distilled",
  "created_at": "2024-01-15T10:30:00Z",
  "started_at": null,
  "completed_at": null,
  "video_url": null,
  "audio_url": null,
  "error": null,
  "progress": null
}
```

### Error Response

```json
{
  "detail": "Pipeline 'unknown' is not loaded. Available: ['distilled']"
}
```

---

## Typical Workflow

1. **Submit a job:**

   ```bash
   curl -X POST http://localhost:8000/generate/distilled \
     -F "prompt=A beautiful sunset over mountains"
   ```

2. **Poll for completion:**

   ```bash
   curl http://localhost:8000/jobs/{job_id}
   ```

3. **Download result** when `status` is `completed`:

   ```bash
   curl -o output.mp4 "{video_url}"
   ```

---

## Docker

See the root `docker-compose.yml` for containerized deployment with GPU support.

```bash
# Build and run
docker-compose up -d

# View logs
docker-compose logs -f ltx-server
```

---

## OpenAPI Documentation

Interactive API documentation is available at:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`
- **OpenAPI JSON**: `http://localhost:8000/openapi.json`
