"""Job manager for async generation task handling."""

import asyncio
import logging
import tempfile
import traceback
import uuid
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any

from ltx_pipelines.utils.constants import AUDIO_SAMPLE_RATE
from ltx_pipelines.utils.media_io import encode_video

from ltx_server.config import get_settings
from ltx_server.models.responses import JobResponse, JobStatus
from ltx_server.services.pipeline_registry import PipelineRegistry
from ltx_server.services.storage import StorageService

logger = logging.getLogger(__name__)


class Job:
    """Internal job representation."""

    def __init__(
        self,
        job_id: str,
        pipeline: str,
        params: dict[str, Any],
        images: list[tuple[str, int, float]] | None = None,
        video_conditioning: list[tuple[str, float]] | None = None,
    ):
        self.job_id = job_id
        self.pipeline = pipeline
        self.params = params
        self.images = images or []
        self.video_conditioning = video_conditioning or []
        self.status = JobStatus.PENDING
        self.created_at = datetime.utcnow()
        self.started_at: datetime | None = None
        self.completed_at: datetime | None = None
        self.video_key: str | None = None
        self.audio_key: str | None = None
        self.error: str | None = None
        self.progress: float | None = None

    def to_response(self, storage: StorageService) -> JobResponse:
        """Convert to API response with presigned URLs."""
        video_url = None
        audio_url = None

        if self.video_key:
            video_url = storage.get_presigned_url(self.video_key)
        if self.audio_key:
            audio_url = storage.get_presigned_url(self.audio_key)

        return JobResponse(
            job_id=self.job_id,
            status=self.status,
            pipeline=self.pipeline,
            created_at=self.created_at,
            started_at=self.started_at,
            completed_at=self.completed_at,
            video_url=video_url,
            audio_url=audio_url,
            error=self.error,
            progress=self.progress,
        )


class JobManager:
    """Manages generation jobs with an in-memory queue."""

    def __init__(
        self,
        pipeline_registry: PipelineRegistry,
        storage: StorageService,
        max_jobs: int = 1000,
    ):
        """
        Initialize job manager.

        Args:
            pipeline_registry: Registry of loaded pipelines
            storage: Storage service for uploads
            max_jobs: Maximum number of jobs to keep in memory
        """
        self.pipeline_registry = pipeline_registry
        self.storage = storage
        self.max_jobs = max_jobs
        self._jobs: OrderedDict[str, Job] = OrderedDict()
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._worker_task: asyncio.Task | None = None
        self._shutdown = False
        self._settings = get_settings()

    async def start(self) -> None:
        """Start the background worker."""
        logger.info("Starting job manager worker")
        self._shutdown = False
        self._worker_task = asyncio.create_task(self._worker_loop())

    async def stop(self) -> None:
        """Stop the background worker."""
        logger.info("Stopping job manager worker")
        self._shutdown = True
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

    def create_job(
        self,
        pipeline: str,
        params: dict[str, Any],
        images: list[tuple[str, int, float]] | None = None,
        video_conditioning: list[tuple[str, float]] | None = None,
    ) -> Job:
        """
        Create a new generation job.

        Args:
            pipeline: Pipeline name
            params: Generation parameters
            images: Image conditioning list
            video_conditioning: Video conditioning list

        Returns:
            The created job
        """
        job_id = str(uuid.uuid4())
        job = Job(
            job_id=job_id,
            pipeline=pipeline,
            params=params,
            images=images,
            video_conditioning=video_conditioning,
        )

        # Enforce max jobs limit
        while len(self._jobs) >= self.max_jobs:
            oldest_id, _ = self._jobs.popitem(last=False)
            logger.info(f"Evicted old job: {oldest_id}")

        self._jobs[job_id] = job
        return job

    async def submit_job(self, job: Job) -> None:
        """Submit a job to the processing queue."""
        await self._queue.put(job.job_id)
        logger.info(f"Job {job.job_id} submitted to queue")

    def get_job(self, job_id: str) -> Job | None:
        """Get a job by ID."""
        return self._jobs.get(job_id)

    def list_jobs(self, limit: int = 50, offset: int = 0) -> tuple[list[Job], int]:
        """List jobs with pagination."""
        all_jobs = list(self._jobs.values())
        # Most recent first
        all_jobs.reverse()
        return all_jobs[offset : offset + limit], len(all_jobs)

    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a pending job.

        Returns True if job was cancelled, False if not found or not pending.
        """
        job = self._jobs.get(job_id)
        if not job or job.status != JobStatus.PENDING:
            return False

        job.status = JobStatus.CANCELLED
        job.completed_at = datetime.utcnow()
        return True

    async def _worker_loop(self) -> None:
        """Background worker that processes jobs from the queue."""
        logger.info("Worker loop started")

        while not self._shutdown:
            try:
                # Wait for a job with timeout to allow shutdown checks
                try:
                    job_id = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue

                job = self._jobs.get(job_id)
                if not job:
                    logger.warning(f"Job {job_id} not found in store")
                    continue

                if job.status == JobStatus.CANCELLED:
                    logger.info(f"Job {job_id} was cancelled, skipping")
                    continue

                await self._process_job(job)

            except asyncio.CancelledError:
                logger.info("Worker loop cancelled")
                break
            except Exception as e:
                logger.error(f"Worker loop error: {e}")
                await asyncio.sleep(1)

        logger.info("Worker loop stopped")

    async def _process_job(self, job: Job) -> None:
        """Process a single generation job."""
        logger.info(f"Processing job {job.job_id} with pipeline {job.pipeline}")

        job.status = JobStatus.PROCESSING
        job.started_at = datetime.utcnow()
        job.progress = 0.0

        try:
            # Run generation in a thread pool to avoid blocking
            video, audio = await asyncio.get_event_loop().run_in_executor(
                None,
                self._run_generation,
                job,
            )

            # Encode and upload video
            job.progress = 0.8
            video_key, audio_key = await asyncio.get_event_loop().run_in_executor(
                None,
                self._encode_and_upload,
                job,
                video,
                audio,
            )

            job.video_key = video_key
            job.audio_key = audio_key
            job.status = JobStatus.COMPLETED
            job.progress = 1.0
            logger.info(f"Job {job.job_id} completed successfully")

        except Exception as e:
            logger.error(f"Job {job.job_id} failed: {e}\n{traceback.format_exc()}")
            job.status = JobStatus.FAILED
            job.error = str(e)

        job.completed_at = datetime.utcnow()

    def _run_generation(self, job: Job) -> tuple[Any, Any]:
        """Run the actual generation (called in thread pool)."""
        params = job.params.copy()
        params["images"] = job.images

        # Add video conditioning for ic_lora pipeline
        if job.pipeline == "ic_lora" and job.video_conditioning:
            params["video_conditioning"] = job.video_conditioning

        return self.pipeline_registry.generate(job.pipeline, **params)

    def _encode_and_upload(self, job: Job, video, audio) -> tuple[str, str | None]:
        """Encode video and upload to S3 (called in thread pool)."""
        # Create temp directory for this job
        temp_dir = self._settings.temp_dir / job.job_id
        temp_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Encode video to file
            video_path = temp_dir / "output.mp4"
            frame_rate = job.params.get("frame_rate", 24.0)

            encode_video(
                video=video,
                fps=frame_rate,
                audio=audio,
                audio_sample_rate=AUDIO_SAMPLE_RATE,
                output_path=str(video_path),
                video_chunks_number=1,
            )

            # Upload to S3
            video_key = f"videos/{job.job_id}/output.mp4"
            self.storage.upload_file(video_path, video_key, "video/mp4")

            return video_key, None

        finally:
            # Cleanup temp files
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)
