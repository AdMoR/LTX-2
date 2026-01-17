"""Media service for handling file uploads and temporary storage."""

import logging
import shutil
import uuid
from pathlib import Path

from fastapi import UploadFile

from ltx_server.config import get_settings

logger = logging.getLogger(__name__)


class MediaService:
    """Service for handling uploaded media files."""

    def __init__(self, temp_dir: Path | None = None):
        """Initialize media service with temp directory."""
        self.temp_dir = temp_dir or get_settings().temp_dir
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def save_upload(self, file: UploadFile, prefix: str = "upload") -> Path:
        """
        Save an uploaded file to temporary storage.

        Args:
            file: FastAPI UploadFile
            prefix: Prefix for the saved filename

        Returns:
            Path to the saved file
        """
        # Generate unique filename
        file_id = str(uuid.uuid4())[:8]
        suffix = Path(file.filename).suffix if file.filename else ""
        filename = f"{prefix}_{file_id}{suffix}"
        file_path = self.temp_dir / filename

        # Save file
        logger.info(f"Saving uploaded file to {file_path}")
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        return file_path

    async def save_upload_async(self, file: UploadFile, prefix: str = "upload") -> Path:
        """
        Save an uploaded file asynchronously.

        Args:
            file: FastAPI UploadFile
            prefix: Prefix for the saved filename

        Returns:
            Path to the saved file
        """
        # Generate unique filename
        file_id = str(uuid.uuid4())[:8]
        suffix = Path(file.filename).suffix if file.filename else ""
        filename = f"{prefix}_{file_id}{suffix}"
        file_path = self.temp_dir / filename

        # Save file
        logger.info(f"Saving uploaded file to {file_path}")
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)

        # Reset file position for potential re-reads
        await file.seek(0)

        return file_path

    def cleanup_file(self, file_path: Path) -> None:
        """Remove a temporary file."""
        if file_path.exists():
            logger.info(f"Cleaning up file {file_path}")
            file_path.unlink()

    def cleanup_directory(self, dir_path: Path) -> None:
        """Remove a temporary directory and its contents."""
        if dir_path.exists():
            logger.info(f"Cleaning up directory {dir_path}")
            shutil.rmtree(dir_path, ignore_errors=True)

    def create_job_directory(self, job_id: str) -> Path:
        """Create a directory for a specific job's temp files."""
        job_dir = self.temp_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        return job_dir

    def get_job_directory(self, job_id: str) -> Path:
        """Get the directory for a specific job."""
        return self.temp_dir / job_id


class UploadedImageCondition:
    """Represents an uploaded image for conditioning."""

    def __init__(self, path: Path, frame_idx: int, strength: float):
        self.path = path
        self.frame_idx = frame_idx
        self.strength = strength

    def to_tuple(self) -> tuple[str, int, float]:
        """Convert to the format expected by pipelines."""
        return (str(self.path), self.frame_idx, self.strength)


class UploadedVideoCondition:
    """Represents an uploaded video for conditioning."""

    def __init__(self, path: Path, strength: float):
        self.path = path
        self.strength = strength

    def to_tuple(self) -> tuple[str, float]:
        """Convert to the format expected by pipelines."""
        return (str(self.path), self.strength)


async def process_image_uploads(
    media_service: MediaService,
    images: list[UploadFile] | None,
    frame_indices: list[int] | None,
    strengths: list[float] | None,
) -> list[tuple[str, int, float]]:
    """
    Process uploaded images into conditioning tuples.

    Args:
        media_service: Media service for saving files
        images: List of uploaded image files
        frame_indices: Frame index for each image
        strengths: Conditioning strength for each image

    Returns:
        List of (path, frame_idx, strength) tuples
    """
    if not images:
        return []

    # Default values
    frame_indices = frame_indices or [0] * len(images)
    strengths = strengths or [1.0] * len(images)

    # Validate lengths
    if len(images) != len(frame_indices) or len(images) != len(strengths):
        raise ValueError(
            f"Mismatched lengths: images={len(images)}, "
            f"frame_indices={len(frame_indices)}, strengths={len(strengths)}"
        )

    result = []
    for image, frame_idx, strength in zip(images, frame_indices, strengths, strict=True):
        path = await media_service.save_upload_async(image, prefix="img")
        result.append((str(path), frame_idx, strength))

    return result


async def process_video_conditioning_upload(
    media_service: MediaService,
    video: UploadFile | None,
    strength: float = 1.0,
) -> list[tuple[str, float]]:
    """
    Process uploaded video conditioning file.

    Args:
        media_service: Media service for saving files
        video: Uploaded video file
        strength: Conditioning strength

    Returns:
        List with single (path, strength) tuple, or empty list
    """
    if not video:
        return []

    path = await media_service.save_upload_async(video, prefix="vid")
    return [(str(path), strength)]
