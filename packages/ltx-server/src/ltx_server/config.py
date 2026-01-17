"""Server configuration using pydantic-settings."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ServerSettings(BaseSettings):
    """Server configuration."""

    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")


class PipelineSettings(BaseSettings):
    """Pipeline configuration."""

    model_config = SettingsConfigDict(env_prefix="LTX_")

    pipelines: str = Field(
        default="distilled",
        description="Comma-separated list of pipelines to load",
    )
    checkpoint_path: str = Field(
        default="/models/ltx2.safetensors",
        description="Path to LTX-2 model checkpoint",
    )
    gemma_root: str = Field(
        default="/models/gemma",
        description="Path to Gemma text encoder root directory",
    )
    spatial_upsampler_path: str | None = Field(
        default="/models/upsampler.safetensors",
        description="Path to spatial upsampler model",
    )
    distilled_lora_path: str | None = Field(
        default=None,
        description="Path to distilled LoRA for two-stage pipelines",
    )
    fp8_transformer: bool = Field(
        default=False,
        description="Enable FP8 mode for transformer",
    )
    text_encoder_cpu: bool = Field(
        default=False,
        description="Load text encoder on CPU",
    )
    text_encoder_8bit: bool = Field(
        default=False,
        description="Use 8-bit quantization for text encoder",
    )
    text_encoder_4bit: bool = Field(
        default=False,
        description="Use 4-bit quantization for text encoder",
    )

    @property
    def enabled_pipelines(self) -> list[str]:
        """Get list of enabled pipeline names."""
        return [p.strip() for p in self.pipelines.split(",") if p.strip()]

    @property
    def text_encoder_device(self) -> str | None:
        """Get text encoder device based on settings."""
        return "cpu" if self.text_encoder_cpu else None


class S3Settings(BaseSettings):
    """S3/MinIO storage configuration."""

    model_config = SettingsConfigDict(env_prefix="S3_")

    bucket: str = Field(default="ltx-outputs", description="S3 bucket name")
    endpoint: str = Field(
        default="http://minio:9000",
        description="S3 endpoint URL",
    )
    region: str = Field(default="us-east-1", description="S3 region")
    access_key_id: str = Field(
        default="",
        alias="AWS_ACCESS_KEY_ID",
        description="AWS access key ID",
    )
    secret_access_key: str = Field(
        default="",
        alias="AWS_SECRET_ACCESS_KEY",
        description="AWS secret access key",
    )
    presigned_url_expiry: int = Field(
        default=3600,
        description="Presigned URL expiry time in seconds",
    )


class Settings(BaseSettings):
    """Combined application settings."""

    server: ServerSettings = Field(default_factory=ServerSettings)
    pipeline: PipelineSettings = Field(default_factory=PipelineSettings)
    s3: S3Settings = Field(default_factory=S3Settings)

    # Temp directory for processing
    temp_dir: Path = Field(
        default=Path("/tmp/ltx"),
        description="Temporary directory for processing",
    )


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
