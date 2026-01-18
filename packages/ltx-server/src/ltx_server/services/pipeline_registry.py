"""Pipeline registry for loading and managing LTX pipelines."""

import logging
from collections.abc import Iterator
from typing import Any

import torch

from ltx_core.loader import LoraPathStrengthAndSDOps
from ltx_pipelines import (
    DistilledPipeline,
    ICLoraPipeline,
    KeyframeInterpolationPipeline,
    TI2VidOneStagePipeline,
    TI2VidTwoStagesPipeline,
)
from ltx_server.config import PipelineSettings, get_settings
from ltx_server.models.responses import PipelineInfo

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
    """Registry for loading and managing video generation pipelines."""

    def __init__(self, settings: PipelineSettings | None = None):
        """Initialize the registry with pipeline settings."""
        self.settings = settings or get_settings().pipeline
        self._pipelines: dict[str, Any] = {}
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def device(self) -> torch.device:
        """Get the device used for inference."""
        return self._device

    @property
    def gpu_available(self) -> bool:
        """Check if GPU is available."""
        return torch.cuda.is_available()

    def load_pipelines(self) -> None:
        """Load all configured pipelines."""
        enabled = self.settings.enabled_pipelines
        logger.info(f"Loading pipelines: {enabled}")

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
        """Load a specific pipeline by name."""
        text_encoder_device = self.settings.text_encoder_device

        if name == "ti2vid_one_stage":
            self._pipelines[name] = TI2VidOneStagePipeline(
                checkpoint_path=self.settings.checkpoint_path,
                gemma_root=self.settings.gemma_root,
                loras=[],
                device=self._device,
                fp8transformer=self.settings.fp8_transformer,
                text_encoder_device=text_encoder_device,
                text_encoder_8bit=self.settings.text_encoder_8bit,
                text_encoder_4bit=self.settings.text_encoder_4bit,
            )

        elif name == "ti2vid_two_stages":
            if not self.settings.distilled_lora_path or not self.settings.spatial_upsampler_path:
                raise ValueError(
                    "ti2vid_two_stages requires distilled_lora_path and spatial_upsampler_path"
                )
            distilled_lora = [LoraPathStrengthAndSDOps(self.settings.distilled_lora_path, 1.0, {})]
            self._pipelines[name] = TI2VidTwoStagesPipeline(
                checkpoint_path=self.settings.checkpoint_path,
                distilled_lora=distilled_lora,
                spatial_upsampler_path=self.settings.spatial_upsampler_path,
                gemma_root=self.settings.gemma_root,
                loras=[],
                device=self._device,
                fp8transformer=self.settings.fp8_transformer,
                text_encoder_device=text_encoder_device,
                text_encoder_8bit=self.settings.text_encoder_8bit,
                text_encoder_4bit=self.settings.text_encoder_4bit,
            )

        elif name == "distilled":
            if not self.settings.spatial_upsampler_path:
                raise ValueError("distilled requires spatial_upsampler_path")
            self._pipelines[name] = DistilledPipeline(
                checkpoint_path=self.settings.checkpoint_path,
                gemma_root=self.settings.gemma_root,
                spatial_upsampler_path=self.settings.spatial_upsampler_path,
                loras=[],
                device=self._device,
                fp8transformer=self.settings.fp8_transformer,
                text_encoder_device=text_encoder_device,
                text_encoder_8bit=self.settings.text_encoder_8bit,
                text_encoder_4bit=self.settings.text_encoder_4bit,
            )

        elif name == "ic_lora":
            if not self.settings.spatial_upsampler_path:
                raise ValueError("ic_lora requires spatial_upsampler_path")
            self._pipelines[name] = ICLoraPipeline(
                checkpoint_path=self.settings.checkpoint_path,
                spatial_upsampler_path=self.settings.spatial_upsampler_path,
                gemma_root=self.settings.gemma_root,
                loras=[],
                device=self._device,
                fp8transformer=self.settings.fp8_transformer,
                text_encoder_device=text_encoder_device,
                text_encoder_8bit=self.settings.text_encoder_8bit,
                text_encoder_4bit=self.settings.text_encoder_4bit,
            )

        elif name == "keyframe_interpolation":
            if not self.settings.distilled_lora_path or not self.settings.spatial_upsampler_path:
                raise ValueError(
                    "keyframe_interpolation requires distilled_lora_path and spatial_upsampler_path"
                )
            distilled_lora = [LoraPathStrengthAndSDOps(self.settings.distilled_lora_path, 1.0, {})]
            self._pipelines[name] = KeyframeInterpolationPipeline(
                checkpoint_path=self.settings.checkpoint_path,
                distilled_lora=distilled_lora,
                spatial_upsampler_path=self.settings.spatial_upsampler_path,
                gemma_root=self.settings.gemma_root,
                loras=[],
                device=self._device,
                fp8transformer=self.settings.fp8_transformer,
                text_encoder_device=text_encoder_device,
                text_encoder_8bit=self.settings.text_encoder_8bit,
                text_encoder_4bit=self.settings.text_encoder_4bit,
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
