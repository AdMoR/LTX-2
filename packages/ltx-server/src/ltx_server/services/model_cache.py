"""Shared model cache for LTX pipelines."""

import hashlib
import logging
import threading
from dataclasses import replace

import torch

from ltx_core.loader.primitives import LoraPathStrengthAndSDOps
from ltx_core.loader.registry import StateDictRegistry
from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder as Builder
from ltx_core.model.audio_vae import (
    AUDIO_VAE_DECODER_COMFY_KEYS_FILTER,
    VOCODER_COMFY_KEYS_FILTER,
    AudioDecoder,
    AudioDecoderConfigurator,
    Vocoder,
    VocoderConfigurator,
)
from ltx_core.model.transformer import (
    LTXV_MODEL_COMFY_RENAMING_MAP,
    LTXV_MODEL_COMFY_RENAMING_WITH_TRANSFORMER_LINEAR_DOWNCAST_MAP,
    UPCAST_DURING_INFERENCE,
    LTXModelConfigurator,
    X0Model,
)
from ltx_core.model.upsampler import LatentUpsampler, LatentUpsamplerConfigurator
from ltx_core.model.video_vae import (
    VAE_DECODER_COMFY_KEYS_FILTER,
    VAE_ENCODER_COMFY_KEYS_FILTER,
    VideoDecoder,
    VideoDecoderConfigurator,
    VideoEncoder,
    VideoEncoderConfigurator,
)
from ltx_core.text_encoders.gemma import (
    AV_GEMMA_TEXT_ENCODER_KEY_OPS,
    AVGemmaTextEncoderModel,
    AVGemmaTextEncoderModelConfigurator,
    module_ops_from_gemma_root,
)

logger = logging.getLogger(__name__)


def _get_gpu_memory_info() -> tuple[float, float, float, float]:
    """Get GPU memory usage in GB. Returns (pytorch_allocated, pytorch_reserved, vram_used, total)."""
    if not torch.cuda.is_available():
        return 0.0, 0.0, 0.0, 0.0
    
    # PyTorch's view of memory (only tracks PyTorch allocations)
    pytorch_allocated = torch.cuda.memory_allocated() / 1024**3
    pytorch_reserved = torch.cuda.memory_reserved() / 1024**3
    
    # Actual VRAM usage (like nvidia-smi, includes bitsandbytes, CUDA context, etc.)
    free, total = torch.cuda.mem_get_info()
    vram_used = (total - free) / 1024**3
    total_gb = total / 1024**3
    
    return pytorch_allocated, pytorch_reserved, vram_used, total_gb


def _get_model_dtype_info(model: torch.nn.Module) -> dict[str, int]:
    """Get dtype distribution of model parameters."""
    dtype_counts: dict[str, int] = {}
    for param in model.parameters():
        dtype_name = str(param.dtype).replace("torch.", "")
        dtype_counts[dtype_name] = dtype_counts.get(dtype_name, 0) + param.numel()
    return dtype_counts


def _format_dtype_info(dtype_counts: dict[str, int]) -> str:
    """Format dtype counts as a readable string with percentages."""
    total = sum(dtype_counts.values())
    if total == 0:
        return "no parameters"
    
    parts = []
    for dtype, count in sorted(dtype_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / total
        if pct >= 1.0:  # Only show dtypes with >= 1%
            parts.append(f"{dtype}: {pct:.1f}%")
    return ", ".join(parts) if parts else "unknown"


def _get_model_device(model: torch.nn.Module) -> str:
    """Get the device of a model's first parameter."""
    for param in model.parameters():
        return str(param.device)
    return "unknown"


def _log_model_info(model_name: str, model: torch.nn.Module) -> None:
    """Log model info including device, precision, and memory usage."""
    pytorch_alloc, pytorch_res, vram_used, total = _get_gpu_memory_info()
    dtype_info = _get_model_dtype_info(model)
    dtype_str = _format_dtype_info(dtype_info)
    device_str = _get_model_device(model)
    
    # Calculate model size in MB
    param_count = sum(p.numel() for p in model.parameters())
    param_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
    
    logger.info(
        f"  ✓ {model_name} loaded | "
        f"device={device_str}, precision=[{dtype_str}], "
        f"params={param_count/1e6:.1f}M, size={param_size_mb:.1f}MB | "
        f"PyTorch: {pytorch_alloc:.2f}GB alloc, VRAM: {vram_used:.2f}GB/{total:.2f}GB"
    )


class SharedModelCache:
    """
    Centralized cache for all models used by LTX pipelines.

    Loads models once at startup and keeps them on GPU permanently.
    Shared components (VAE, text encoder, vocoder, upsampler) are loaded once.
    Pre-builds known transformer variants (base and distilled LoRA).
    User LoRA transformers are built on-demand using cached state dicts.
    """

    def __init__(
        self,
        checkpoint_path: str,
        gemma_root: str,
        spatial_upsampler_path: str | None = None,
        distilled_lora_path: str | None = None,
        device: torch.device | None = None,
        fp8transformer: bool = False,
        text_encoder_device: torch.device | str | None = None,
        text_encoder_8bit: bool = False,
        text_encoder_4bit: bool = False,
    ):
        """
        Initialize the shared model cache.

        Args:
            checkpoint_path: Path to the main LTX checkpoint
            gemma_root: Path to Gemma text encoder weights
            spatial_upsampler_path: Optional path to spatial upsampler checkpoint
            distilled_lora_path: Optional path to distilled LoRA for pre-building
            device: Target device for models (default: cuda if available)
            fp8transformer: Whether to use FP8 quantization for transformers
            text_encoder_device: Device for text encoder (None=same as device, "cpu"=CPU)
            text_encoder_8bit: Use 8-bit quantization for text encoder
            text_encoder_4bit: Use 4-bit quantization for text encoder
        """
        self.checkpoint_path = checkpoint_path
        self.gemma_root = gemma_root
        self.spatial_upsampler_path = spatial_upsampler_path
        self.distilled_lora_path = distilled_lora_path
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16
        self.fp8transformer = fp8transformer
        self.text_encoder_8bit = text_encoder_8bit
        self.text_encoder_4bit = text_encoder_4bit

        # Set text encoder device
        if text_encoder_device is None:
            self.text_encoder_device = self.device
        elif isinstance(text_encoder_device, str):
            self.text_encoder_device = torch.device(text_encoder_device)
        else:
            self.text_encoder_device = text_encoder_device

        # Registry for caching state dicts (used for building LoRA variants)
        self._registry = StateDictRegistry()

        # Thread safety for on-demand LoRA builds
        self._lora_build_lock = threading.Lock()
        self._lora_transformer_cache: dict[str, X0Model] = {}

        # Initialize builders
        self._init_builders()

        # Load all shared models
        logger.info("Loading shared models into cache...")
        self._load_shared_models()
        logger.info("Shared model cache initialized")

    def _init_builders(self) -> None:
        """Initialize model builders."""
        # Transformer builder (base, no LoRA)
        self._transformer_builder = Builder(
            model_path=self.checkpoint_path,
            model_class_configurator=LTXModelConfigurator,
            model_sd_ops=LTXV_MODEL_COMFY_RENAMING_MAP,
            loras=(),
            registry=self._registry,
        )

        # Video VAE builders
        self._vae_decoder_builder = Builder(
            model_path=self.checkpoint_path,
            model_class_configurator=VideoDecoderConfigurator,
            model_sd_ops=VAE_DECODER_COMFY_KEYS_FILTER,
            registry=self._registry,
        )

        self._vae_encoder_builder = Builder(
            model_path=self.checkpoint_path,
            model_class_configurator=VideoEncoderConfigurator,
            model_sd_ops=VAE_ENCODER_COMFY_KEYS_FILTER,
            registry=self._registry,
        )

        # Audio VAE builders
        self._audio_decoder_builder = Builder(
            model_path=self.checkpoint_path,
            model_class_configurator=AudioDecoderConfigurator,
            model_sd_ops=AUDIO_VAE_DECODER_COMFY_KEYS_FILTER,
            registry=self._registry,
        )

        self._vocoder_builder = Builder(
            model_path=self.checkpoint_path,
            model_class_configurator=VocoderConfigurator,
            model_sd_ops=VOCODER_COMFY_KEYS_FILTER,
            registry=self._registry,
        )

        # Text encoder builder
        self._text_encoder_builder = Builder(
            model_path=self.checkpoint_path,
            model_class_configurator=AVGemmaTextEncoderModelConfigurator,
            model_sd_ops=AV_GEMMA_TEXT_ENCODER_KEY_OPS,
            registry=self._registry,
            module_ops=module_ops_from_gemma_root(
                self.gemma_root,
                device=self.text_encoder_device,
                quantize_8bit=self.text_encoder_8bit,
                quantize_4bit=self.text_encoder_4bit,
            ),
        )

        # Spatial upsampler builder (optional)
        if self.spatial_upsampler_path is not None:
            self._upsampler_builder = Builder(
                model_path=self.spatial_upsampler_path,
                model_class_configurator=LatentUpsamplerConfigurator,
                registry=self._registry,
            )

    def _load_shared_models(self) -> None:
        """Load all shared models into memory."""
        initial_pytorch, _, initial_vram, total = _get_gpu_memory_info()
        logger.info(f"Starting model loading (VRAM: {initial_vram:.2f}GB / {total:.2f}GB)")
        logger.info(f"Configuration: fp8_transformer={self.fp8transformer}, "
                    f"text_encoder_8bit={self.text_encoder_8bit}, "
                    f"text_encoder_4bit={self.text_encoder_4bit}")

        # Load text encoder
        logger.info("Loading text encoder...")
        self._text_encoder = self._build_text_encoder()
        _log_model_info("Text encoder", self._text_encoder)

        # Load video VAE
        logger.info("Loading video encoder...")
        self._video_encoder = self._vae_encoder_builder.build(
            device=self.device, dtype=self.dtype
        ).to(self.device).eval()
        _log_model_info("Video encoder", self._video_encoder)

        logger.info("Loading video decoder...")
        self._video_decoder = self._vae_decoder_builder.build(
            device=self.device, dtype=self.dtype
        ).to(self.device).eval()
        _log_model_info("Video decoder", self._video_decoder)

        # Load audio components
        logger.info("Loading audio decoder...")
        self._audio_decoder = self._audio_decoder_builder.build(
            device=self.device, dtype=self.dtype
        ).to(self.device).eval()
        _log_model_info("Audio decoder", self._audio_decoder)

        logger.info("Loading vocoder...")
        self._vocoder = self._vocoder_builder.build(
            device=self.device, dtype=self.dtype
        ).to(self.device).eval()
        _log_model_info("Vocoder", self._vocoder)

        # Load spatial upsampler (optional)
        if self.spatial_upsampler_path is not None:
            logger.info("Loading spatial upsampler...")
            self._spatial_upsampler = self._upsampler_builder.build(
                device=self.device, dtype=self.dtype
            ).to(self.device).eval()
            _log_model_info("Spatial upsampler", self._spatial_upsampler)
        else:
            self._spatial_upsampler = None

        # Pre-build transformer variants
        logger.info("Loading base transformer (no LoRA)...")
        self._transformer_base = self._build_transformer(loras=())
        _log_model_info("Base transformer", self._transformer_base)

        if self.distilled_lora_path is not None:
            logger.info("Loading distilled transformer...")
            distilled_lora = (LoraPathStrengthAndSDOps(self.distilled_lora_path, 1.0, {}),)
            self._transformer_distilled = self._build_transformer(loras=distilled_lora)
            _log_model_info("Distilled transformer", self._transformer_distilled)
        else:
            self._transformer_distilled = None

        # Final summary
        final_pytorch, final_reserved, final_vram, total = _get_gpu_memory_info()
        logger.info(
            f"✅ Model loading complete | "
            f"PyTorch: {final_pytorch:.2f}GB alloc (+{final_pytorch - initial_pytorch:.2f}GB) | "
            f"VRAM: {final_vram:.2f}GB/{total:.2f}GB (+{final_vram - initial_vram:.2f}GB)"
        )

    def _build_text_encoder(self) -> AVGemmaTextEncoderModel:
        """Build the text encoder with proper dtype handling."""
        use_quantization = self.text_encoder_8bit or self.text_encoder_4bit
        if use_quantization:
            te_dtype = self.dtype
        elif self.text_encoder_device.type == "cpu":
            te_dtype = torch.float16
        else:
            te_dtype = self.dtype

        encoder = self._text_encoder_builder.build(
            device=self.text_encoder_device, dtype=te_dtype
        )

        # With quantization, model is already on GPU via device_map="auto"
        if not use_quantization:
            encoder = encoder.to(self.text_encoder_device)

        return encoder.eval()

    def _build_transformer(self, loras: tuple[LoraPathStrengthAndSDOps, ...]) -> X0Model:
        """Build a transformer with optional LoRAs."""
        builder = replace(self._transformer_builder, loras=loras)

        if self.fp8transformer:
            fp8_builder = replace(
                builder,
                module_ops=(UPCAST_DURING_INFERENCE,),
                model_sd_ops=LTXV_MODEL_COMFY_RENAMING_WITH_TRANSFORMER_LINEAR_DOWNCAST_MAP,
            )
            return X0Model(fp8_builder.build(device=self.device)).to(self.device).eval()
        else:
            return X0Model(
                builder.build(device=self.device, dtype=self.dtype)
            ).to(self.device).eval()

    def _lora_cache_key(self, loras: list[LoraPathStrengthAndSDOps]) -> str:
        """Generate a cache key for a set of LoRAs."""
        parts = []
        for lora in sorted(loras, key=lambda x: x.path):
            parts.append(f"{lora.path}:{lora.strength}")
        key_str = "|".join(parts)
        return hashlib.sha256(key_str.encode()).hexdigest()[:16]

    # =========================================================================
    # Public accessors for shared models
    # =========================================================================

    def get_text_encoder(self) -> AVGemmaTextEncoderModel:
        """Get the cached text encoder."""
        return self._text_encoder

    def get_video_encoder(self) -> VideoEncoder:
        """Get the cached video encoder."""
        return self._video_encoder

    def get_video_decoder(self) -> VideoDecoder:
        """Get the cached video decoder."""
        return self._video_decoder

    def get_audio_decoder(self) -> AudioDecoder:
        """Get the cached audio decoder."""
        return self._audio_decoder

    def get_vocoder(self) -> Vocoder:
        """Get the cached vocoder."""
        return self._vocoder

    def get_spatial_upsampler(self) -> LatentUpsampler:
        """Get the cached spatial upsampler."""
        if self._spatial_upsampler is None:
            raise ValueError(
                "Spatial upsampler not available. "
                "Provide spatial_upsampler_path when creating SharedModelCache."
            )
        return self._spatial_upsampler

    def get_transformer_base(self) -> X0Model:
        """Get the cached base transformer (no LoRA)."""
        return self._transformer_base

    def get_transformer_distilled(self) -> X0Model:
        """Get the cached distilled transformer."""
        if self._transformer_distilled is None:
            raise ValueError(
                "Distilled transformer not available. "
                "Provide distilled_lora_path when creating SharedModelCache."
            )
        return self._transformer_distilled

    def build_transformer_with_loras(
        self, loras: list[LoraPathStrengthAndSDOps]
    ) -> X0Model:
        """
        Build a transformer with custom LoRAs.

        Results are cached so subsequent requests with the same LoRAs
        return the cached model.

        Args:
            loras: List of LoRA configurations to apply

        Returns:
            Transformer model with LoRAs applied
        """
        if not loras:
            return self._transformer_base

        cache_key = self._lora_cache_key(loras)

        with self._lora_build_lock:
            if cache_key not in self._lora_transformer_cache:
                logger.info(f"Building transformer with LoRAs: {[l.path for l in loras]}")
                self._lora_transformer_cache[cache_key] = self._build_transformer(
                    loras=tuple(loras)
                )
                _log_memory(f"Transformer with LoRAs ({cache_key})")
            return self._lora_transformer_cache[cache_key]

    @property
    def has_spatial_upsampler(self) -> bool:
        """Check if spatial upsampler is available."""
        return self._spatial_upsampler is not None

    @property
    def has_distilled_transformer(self) -> bool:
        """Check if distilled transformer is available."""
        return self._transformer_distilled is not None
