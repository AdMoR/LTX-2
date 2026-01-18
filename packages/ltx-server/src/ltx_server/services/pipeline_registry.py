"""Pipeline registry for loading and managing LTX pipelines."""

import logging
from collections.abc import Iterator
from typing import Any

import torch

from ltx_pipelines import (
    DistilledPipeline,
    ICLoraPipeline,
    KeyframeInterpolationPipeline,
    TI2VidOneStagePipeline,
    TI2VidTwoStagesPipeline,
)
from ltx_server.config import PipelineSettings, get_settings
from ltx_server.models.responses import PipelineInfo
from ltx_server.services.model_cache import SharedModelCache

logger = logging.getLogger(__name__)


# Pipeline metadata for API responses
PIPELINE_METADATA: dict[str, PipelineInfo] = {
    "ti2vid_one_stage": PipelineInfo(
        name="ti2vid_one_stage",
        description="Single-stage text/image-to-video generation with CFG guidance",
        supports_image_conditioning=True,
        supports_video_conditioning=False,
        supports_negative_prompt=True,
        is_two_stage=False,
    ),
    "ti2vid_two_stages": PipelineInfo(
        name="ti2vid_two_stages",
        description="Two-stage text/image-to-video with CFG and distilled upscaling",
        supports_image_conditioning=True,
        supports_video_conditioning=False,
        supports_negative_prompt=True,
        is_two_stage=True,
    ),
    "distilled": PipelineInfo(
        name="distilled",
        description="Fast two-stage distilled generation (no CFG)",
        supports_image_conditioning=True,
        supports_video_conditioning=False,
        supports_negative_prompt=False,
        is_two_stage=True,
    ),
    "ic_lora": PipelineInfo(
        name="ic_lora",
        description="Two-stage generation with IC-LoRA video conditioning",
        supports_image_conditioning=True,
        supports_video_conditioning=True,
        supports_negative_prompt=False,
        is_two_stage=True,
    ),
    "keyframe_interpolation": PipelineInfo(
        name="keyframe_interpolation",
        description="Keyframe-based video interpolation with CFG",
        supports_image_conditioning=True,
        supports_video_conditioning=False,
        supports_negative_prompt=True,
        is_two_stage=True,
    ),
}


class PipelineRegistry:
    """Registry for loading and managing video generation pipelines.

    Uses a SharedModelCache to load all models once and share them across pipelines.
    This significantly reduces memory usage compared to loading models per-pipeline.
    """

    def __init__(self, settings: PipelineSettings | None = None):
        """Initialize the registry with pipeline settings."""
        self.settings = settings or get_settings().pipeline
        self._pipelines: dict[str, Any] = {}
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model_cache: SharedModelCache | None = None

    @property
    def device(self) -> torch.device:
        """Get the device used for inference."""
        return self._device

    @property
    def gpu_available(self) -> bool:
        """Check if GPU is available."""
        return torch.cuda.is_available()

    @property
    def model_cache(self) -> SharedModelCache:
        """Get the shared model cache."""
        if self._model_cache is None:
            raise RuntimeError("Model cache not initialized. Call load_pipelines() first.")
        return self._model_cache

    def load_pipelines(self) -> None:
        """Load all configured pipelines using shared model cache."""
        enabled = self.settings.enabled_pipelines
        logger.info(f"Loading pipelines: {enabled}")

        # Determine if we need spatial upsampler and distilled transformer
        needs_upsampler = any(
            p in enabled for p in ["distilled", "ic_lora", "ti2vid_two_stages", "keyframe_interpolation"]
        )
        needs_distilled = any(
            p in enabled for p in ["ti2vid_two_stages", "keyframe_interpolation"]
        )

        # Validate required paths
        if needs_upsampler and not self.settings.spatial_upsampler_path:
            raise ValueError(
                f"spatial_upsampler_path is required for pipelines: {enabled}"
            )
        if needs_distilled and not self.settings.distilled_lora_path:
            raise ValueError(
                f"distilled_lora_path is required for pipelines: {enabled}"
            )

        # Create shared model cache with all required models
        logger.info("Initializing shared model cache...")
        self._model_cache = SharedModelCache(
            checkpoint_path=self.settings.checkpoint_path,
            gemma_root=self.settings.gemma_root,
            spatial_upsampler_path=self.settings.spatial_upsampler_path if needs_upsampler else None,
            distilled_lora_path=self.settings.distilled_lora_path if needs_distilled else None,
            device=self._device,
            fp8transformer=self.settings.fp8_transformer,
            text_encoder_device=self.settings.text_encoder_device,
            text_encoder_8bit=self.settings.text_encoder_8bit,
            text_encoder_4bit=self.settings.text_encoder_4bit,
        )
        logger.info("Shared model cache initialized")

        # Load pipelines using the shared cache
        for name in enabled:
            if name not in PIPELINE_METADATA:
                logger.warning(f"Unknown pipeline: {name}, skipping")
                continue

            try:
                self._load_pipeline(name)
                logger.info(f"Loaded pipeline: {name}")
            except Exception as e:
                logger.error(f"Failed to load pipeline {name}: {e}")
                raise

    def _load_pipeline(self, name: str) -> None:
        """Load a specific pipeline by name using shared model cache."""
        if name == "ti2vid_one_stage":
            self._pipelines[name] = TI2VidOneStagePipeline(
                device=self._device,
                model_cache=self._model_cache,
            )

        elif name == "ti2vid_two_stages":
            self._pipelines[name] = TI2VidTwoStagesPipeline(
                device=self._device,
                model_cache=self._model_cache,
            )

        elif name == "distilled":
            self._pipelines[name] = DistilledPipeline(
                device=self._device,
                model_cache=self._model_cache,
            )

        elif name == "ic_lora":
            self._pipelines[name] = ICLoraPipeline(
                device=self._device,
                model_cache=self._model_cache,
            )

        elif name == "keyframe_interpolation":
            self._pipelines[name] = KeyframeInterpolationPipeline(
                device=self._device,
                model_cache=self._model_cache,
            )

        else:
            raise ValueError(f"Unknown pipeline: {name}")

    def get_pipeline(self, name: str) -> Any:
        """Get a loaded pipeline by name."""
        if name not in self._pipelines:
            raise KeyError(f"Pipeline '{name}' not loaded. Available: {list(self._pipelines.keys())}")
        return self._pipelines[name]

    def list_loaded(self) -> list[str]:
        """List names of loaded pipelines."""
        return list(self._pipelines.keys())

    def list_available(self) -> list[PipelineInfo]:
        """List metadata for all loaded pipelines."""
        return [PIPELINE_METADATA[name] for name in self._pipelines.keys()]

    def is_loaded(self, name: str) -> bool:
        """Check if a pipeline is loaded."""
        return name in self._pipelines

    @torch.no_grad()
    def generate(
        self,
        pipeline_name: str,
        **kwargs,
    ) -> tuple[Iterator[torch.Tensor], torch.Tensor]:
        """
        Run generation on a pipeline.

        Args:
            pipeline_name: Name of the pipeline to use
            **kwargs: Generation parameters passed to the pipeline

        Returns:
            Tuple of (video_iterator, audio_tensor)
        """
        pipeline = self.get_pipeline(pipeline_name)
        return pipeline(**kwargs)
