"""Generation endpoints for all pipelines."""

import logging
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from ltx_server.models.responses import JobResponse
from ltx_server.services.media import process_image_uploads, process_video_conditioning_upload

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/generate", tags=["generation"])


@router.post("/ti2vid_one_stage", response_model=JobResponse)
async def generate_ti2vid_one_stage(
    request: Request,
    prompt: Annotated[str, Form(description="Text prompt describing desired video")],
    seed: Annotated[int, Form()] = 42,
    height: Annotated[int, Form()] = 768,
    width: Annotated[int, Form()] = 1152,
    num_frames: Annotated[int, Form()] = 97,
    frame_rate: Annotated[float, Form()] = 24.0,
    negative_prompt: Annotated[str, Form()] = "worst quality, inconsistent motion, blurry, jittery, distorted",
    num_inference_steps: Annotated[int, Form()] = 40,
    cfg_guidance_scale: Annotated[float, Form()] = 3.0,
    enhance_prompt: Annotated[bool, Form()] = False,
    images: Annotated[list[UploadFile] | None, File()] = None,
    image_frame_indices: Annotated[list[int] | None, Form()] = None,
    image_strengths: Annotated[list[float] | None, Form()] = None,
) -> JobResponse:
    """Submit a single-stage text/image-to-video generation job with CFG."""
    pipeline_name = "ti2vid_one_stage"
    _validate_pipeline_loaded(request, pipeline_name)

    # Process uploaded images
    media_service = request.app.state.media_service
    image_conditions = await process_image_uploads(
        media_service, images, image_frame_indices, image_strengths
    )

    # Create job
    params = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "seed": seed,
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "frame_rate": frame_rate,
        "num_inference_steps": num_inference_steps,
        "cfg_guidance_scale": cfg_guidance_scale,
        "enhance_prompt": enhance_prompt,
    }

    return await _submit_job(request, pipeline_name, params, image_conditions)


@router.post("/ti2vid_two_stages", response_model=JobResponse)
async def generate_ti2vid_two_stages(
    request: Request,
    prompt: Annotated[str, Form(description="Text prompt describing desired video")],
    seed: Annotated[int, Form()] = 42,
    height: Annotated[int, Form()] = 768,
    width: Annotated[int, Form()] = 1152,
    num_frames: Annotated[int, Form()] = 97,
    frame_rate: Annotated[float, Form()] = 24.0,
    negative_prompt: Annotated[str, Form()] = "worst quality, inconsistent motion, blurry, jittery, distorted",
    num_inference_steps: Annotated[int, Form()] = 40,
    cfg_guidance_scale: Annotated[float, Form()] = 3.0,
    enhance_prompt: Annotated[bool, Form()] = False,
    images: Annotated[list[UploadFile] | None, File()] = None,
    image_frame_indices: Annotated[list[int] | None, Form()] = None,
    image_strengths: Annotated[list[float] | None, Form()] = None,
) -> JobResponse:
    """Submit a two-stage text/image-to-video generation job."""
    pipeline_name = "ti2vid_two_stages"
    _validate_pipeline_loaded(request, pipeline_name)

    media_service = request.app.state.media_service
    image_conditions = await process_image_uploads(
        media_service, images, image_frame_indices, image_strengths
    )

    params = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "seed": seed,
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "frame_rate": frame_rate,
        "num_inference_steps": num_inference_steps,
        "cfg_guidance_scale": cfg_guidance_scale,
        "enhance_prompt": enhance_prompt,
    }

    return await _submit_job(request, pipeline_name, params, image_conditions)


@router.post("/distilled", response_model=JobResponse)
async def generate_distilled(
    request: Request,
    prompt: Annotated[str, Form(description="Text prompt describing desired video")],
    seed: Annotated[int, Form()] = 42,
    height: Annotated[int, Form()] = 768,
    width: Annotated[int, Form()] = 1152,
    num_frames: Annotated[int, Form()] = 97,
    frame_rate: Annotated[float, Form()] = 24.0,
    enhance_prompt: Annotated[bool, Form()] = False,
    images: Annotated[list[UploadFile] | None, File()] = None,
    image_frame_indices: Annotated[list[int] | None, Form()] = None,
    image_strengths: Annotated[list[float] | None, Form()] = None,
) -> JobResponse:
    """Submit a distilled two-stage generation job (fast, no CFG)."""
    pipeline_name = "distilled"
    _validate_pipeline_loaded(request, pipeline_name)

    media_service = request.app.state.media_service
    image_conditions = await process_image_uploads(
        media_service, images, image_frame_indices, image_strengths
    )

    params = {
        "prompt": prompt,
        "seed": seed,
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "frame_rate": frame_rate,
        "enhance_prompt": enhance_prompt,
    }

    return await _submit_job(request, pipeline_name, params, image_conditions)


@router.post("/ic_lora", response_model=JobResponse)
async def generate_ic_lora(
    request: Request,
    prompt: Annotated[str, Form(description="Text prompt describing desired video")],
    seed: Annotated[int, Form()] = 42,
    height: Annotated[int, Form()] = 768,
    width: Annotated[int, Form()] = 1152,
    num_frames: Annotated[int, Form()] = 97,
    frame_rate: Annotated[float, Form()] = 24.0,
    enhance_prompt: Annotated[bool, Form()] = False,
    video_conditioning_strength: Annotated[float, Form()] = 1.0,
    images: Annotated[list[UploadFile] | None, File()] = None,
    image_frame_indices: Annotated[list[int] | None, Form()] = None,
    image_strengths: Annotated[list[float] | None, Form()] = None,
    video_conditioning: Annotated[UploadFile | None, File()] = None,
) -> JobResponse:
    """Submit an IC-LoRA generation job with video conditioning."""
    pipeline_name = "ic_lora"
    _validate_pipeline_loaded(request, pipeline_name)

    media_service = request.app.state.media_service
    image_conditions = await process_image_uploads(
        media_service, images, image_frame_indices, image_strengths
    )
    video_conditions = await process_video_conditioning_upload(
        media_service, video_conditioning, video_conditioning_strength
    )

    params = {
        "prompt": prompt,
        "seed": seed,
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "frame_rate": frame_rate,
        "enhance_prompt": enhance_prompt,
    }

    return await _submit_job(
        request, pipeline_name, params, image_conditions, video_conditions
    )


@router.post("/keyframe_interpolation", response_model=JobResponse)
async def generate_keyframe_interpolation(
    request: Request,
    prompt: Annotated[str, Form(description="Text prompt describing desired video")],
    seed: Annotated[int, Form()] = 42,
    height: Annotated[int, Form()] = 768,
    width: Annotated[int, Form()] = 1152,
    num_frames: Annotated[int, Form()] = 97,
    frame_rate: Annotated[float, Form()] = 24.0,
    negative_prompt: Annotated[str, Form()] = "worst quality, inconsistent motion, blurry, jittery, distorted",
    num_inference_steps: Annotated[int, Form()] = 40,
    cfg_guidance_scale: Annotated[float, Form()] = 3.0,
    enhance_prompt: Annotated[bool, Form()] = False,
    images: Annotated[list[UploadFile] | None, File()] = None,
    image_frame_indices: Annotated[list[int] | None, Form()] = None,
    image_strengths: Annotated[list[float] | None, Form()] = None,
) -> JobResponse:
    """Submit a keyframe interpolation generation job."""
    pipeline_name = "keyframe_interpolation"
    _validate_pipeline_loaded(request, pipeline_name)

    media_service = request.app.state.media_service
    image_conditions = await process_image_uploads(
        media_service, images, image_frame_indices, image_strengths
    )

    params = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "seed": seed,
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "frame_rate": frame_rate,
        "num_inference_steps": num_inference_steps,
        "cfg_guidance_scale": cfg_guidance_scale,
        "enhance_prompt": enhance_prompt,
    }

    return await _submit_job(request, pipeline_name, params, image_conditions)


def _validate_pipeline_loaded(request: Request, pipeline_name: str) -> None:
    """Check if the requested pipeline is loaded."""
    pipeline_registry = request.app.state.pipeline_registry
    if not pipeline_registry.is_loaded(pipeline_name):
        loaded = pipeline_registry.list_loaded()
        raise HTTPException(
            status_code=400,
            detail=f"Pipeline '{pipeline_name}' is not loaded. Available: {loaded}",
        )


async def _submit_job(
    request: Request,
    pipeline_name: str,
    params: dict,
    images: list[tuple[str, int, float]],
    video_conditioning: list[tuple[str, float]] | None = None,
) -> JobResponse:
    """Create and submit a generation job."""
    job_manager = request.app.state.job_manager
    storage = request.app.state.storage

    job = job_manager.create_job(
        pipeline=pipeline_name,
        params=params,
        images=images,
        video_conditioning=video_conditioning,
    )

    await job_manager.submit_job(job)

    logger.info(f"Created job {job.job_id} for pipeline {pipeline_name}")
    return job.to_response(storage)
