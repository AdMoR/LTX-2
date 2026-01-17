"""Pydantic request models for generation endpoints."""

from pydantic import BaseModel, Field


class BaseGenerateRequest(BaseModel):
    """Base request model with common generation parameters."""

    prompt: str = Field(..., description="Text prompt describing the desired video content")
    seed: int = Field(default=42, description="Random seed for reproducible generation")
    height: int = Field(default=768, description="Height of the generated video in pixels")
    width: int = Field(default=1152, description="Width of the generated video in pixels")
    num_frames: int = Field(
        default=97,
        description="Number of frames to generate (num_frames = 8k + 1)",
    )
    frame_rate: float = Field(default=24.0, description="Frame rate of the generated video")
    enhance_prompt: bool = Field(
        default=False,
        description="Whether to enhance the prompt using the text encoder",
    )


class TI2VidOneStageGenerateRequest(BaseGenerateRequest):
    """Request model for single-stage text/image-to-video generation with CFG."""

    negative_prompt: str = Field(
        default="worst quality, inconsistent motion, blurry, jittery, distorted",
        description="Negative prompt describing what should not appear",
    )
    num_inference_steps: int = Field(
        default=40,
        description="Number of denoising steps",
    )
    cfg_guidance_scale: float = Field(
        default=3.0,
        description="Classifier-free guidance scale",
    )


class TI2VidTwoStagesGenerateRequest(BaseGenerateRequest):
    """Request model for two-stage text/image-to-video generation."""

    negative_prompt: str = Field(
        default="worst quality, inconsistent motion, blurry, jittery, distorted",
        description="Negative prompt describing what should not appear",
    )
    num_inference_steps: int = Field(
        default=40,
        description="Number of denoising steps for stage 1",
    )
    cfg_guidance_scale: float = Field(
        default=3.0,
        description="Classifier-free guidance scale for stage 1",
    )


class DistilledGenerateRequest(BaseGenerateRequest):
    """Request model for distilled two-stage generation (no CFG)."""

    pass


class ICLoraGenerateRequest(BaseGenerateRequest):
    """Request model for IC-LoRA pipeline with video conditioning."""

    # video_conditioning is handled via multipart form upload
    video_conditioning_strength: float = Field(
        default=1.0,
        description="Strength of video conditioning",
    )


class KeyframeInterpolationGenerateRequest(BaseGenerateRequest):
    """Request model for keyframe interpolation pipeline."""

    negative_prompt: str = Field(
        default="worst quality, inconsistent motion, blurry, jittery, distorted",
        description="Negative prompt describing what should not appear",
    )
    num_inference_steps: int = Field(
        default=40,
        description="Number of denoising steps for stage 1",
    )
    cfg_guidance_scale: float = Field(
        default=3.0,
        description="Classifier-free guidance scale for stage 1",
    )


# Mapping of pipeline names to request models
PIPELINE_REQUEST_MODELS: dict[str, type[BaseGenerateRequest]] = {
    "ti2vid_one_stage": TI2VidOneStageGenerateRequest,
    "ti2vid_two_stages": TI2VidTwoStagesGenerateRequest,
    "distilled": DistilledGenerateRequest,
    "ic_lora": ICLoraGenerateRequest,
    "keyframe_interpolation": KeyframeInterpolationGenerateRequest,
}
