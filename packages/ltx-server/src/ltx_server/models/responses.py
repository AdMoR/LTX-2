"""Pydantic response models for API endpoints."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    """Status of a generation job."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobResponse(BaseModel):
    """Response model for a single job."""

    job_id: str = Field(..., description="Unique job identifier")
    status: JobStatus = Field(..., description="Current job status")
    pipeline: str = Field(..., description="Pipeline used for generation")
    created_at: datetime = Field(..., description="Job creation timestamp")
    started_at: datetime | None = Field(None, description="Processing start timestamp")
    completed_at: datetime | None = Field(None, description="Completion timestamp")
    video_url: str | None = Field(None, description="Presigned URL to download video")
    audio_url: str | None = Field(None, description="Presigned URL to download audio")
    error: str | None = Field(None, description="Error message if job failed")
    progress: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Generation progress (0.0 to 1.0)",
    )


class JobListResponse(BaseModel):
    """Response model for listing jobs."""

    jobs: list[JobResponse] = Field(..., description="List of jobs")
    total: int = Field(..., description="Total number of jobs")


class PipelineInfo(BaseModel):
    """Information about an available pipeline."""

    name: str = Field(..., description="Pipeline identifier")
    description: str = Field(..., description="Pipeline description")
    supports_image_conditioning: bool = Field(
        default=True,
        description="Whether pipeline supports image conditioning",
    )
    supports_video_conditioning: bool = Field(
        default=False,
        description="Whether pipeline supports video conditioning",
    )
    supports_negative_prompt: bool = Field(
        default=False,
        description="Whether pipeline uses negative prompt / CFG",
    )
    is_two_stage: bool = Field(
        default=False,
        description="Whether pipeline uses two-stage generation",
    )


class PipelinesResponse(BaseModel):
    """Response model for listing available pipelines."""

    pipelines: list[PipelineInfo] = Field(..., description="Available pipelines")


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str = Field(default="healthy", description="Server health status")
    version: str = Field(..., description="Server version")
    pipelines_loaded: list[str] = Field(..., description="Currently loaded pipelines")
    gpu_available: bool = Field(..., description="Whether GPU is available")


class ErrorResponse(BaseModel):
    """Response model for errors."""

    detail: str = Field(..., description="Error message")
