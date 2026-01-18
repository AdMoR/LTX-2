import logging
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import torch

from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.components.guiders import CFGGuider
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.protocols import DiffusionStepProtocol
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.loader import LoraPathStrengthAndSDOps
from ltx_core.model.audio_vae import decode_audio as vae_decode_audio
from ltx_core.model.upsampler import upsample_video
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_core.model.video_vae import decode_video as vae_decode_video
from ltx_core.text_encoders.gemma import encode_text
from ltx_core.types import LatentState, VideoPixelShape
from ltx_pipelines.utils import ModelLedger
from ltx_pipelines.utils.args import default_2_stage_arg_parser
from ltx_pipelines.utils.constants import (
    AUDIO_SAMPLE_RATE,
    STAGE_2_DISTILLED_SIGMA_VALUES,
)
from ltx_pipelines.utils.helpers import (
    assert_resolution,
    cleanup_memory,
    denoise_audio_video,
    euler_denoising_loop,
    generate_enhanced_prompt,
    get_device,
    guider_denoising_func,
    image_conditionings_by_replacing_latent,
    log_generation_stage,
    simple_denoising_func,
)
from ltx_pipelines.utils.media_io import encode_video
from ltx_pipelines.utils.types import PipelineComponents

if TYPE_CHECKING:
    from ltx_server.services.model_cache import SharedModelCache

device = get_device()


class TI2VidTwoStagesPipeline:
    """
    Two-stage text/image-to-video generation pipeline.
    Stage 1 generates video at the target resolution with CFG guidance, then
    Stage 2 upsamples by 2x and refines using a distilled LoRA for higher
    quality output. Supports optional image conditioning via the images parameter.

    Can be initialized in two ways:
    1. With a SharedModelCache (for server use with shared models)
    2. With individual model paths (for standalone/CLI use)
    """

    def __init__(
        self,
        checkpoint_path: str | None = None,
        distilled_lora: list[LoraPathStrengthAndSDOps] | None = None,
        spatial_upsampler_path: str | None = None,
        gemma_root: str | None = None,
        loras: list[LoraPathStrengthAndSDOps] | None = None,
        device: torch.device = device,
        fp8transformer: bool = False,
        text_encoder_device: torch.device | str | None = None,
        text_encoder_8bit: bool = False,
        text_encoder_4bit: bool = False,
        *,
        model_cache: "SharedModelCache | None" = None,
    ):
        self.device = device
        self.dtype = torch.bfloat16
        self._model_cache = model_cache
        self._loras = loras or []

        if model_cache is not None:
            # Server mode: use shared model cache
            self._stage_1_model_ledger = None
            self._stage_2_model_ledger = None
        else:
            # Standalone mode: create ModelLedgers
            if checkpoint_path is None or gemma_root is None or spatial_upsampler_path is None:
                raise ValueError(
                    "checkpoint_path, gemma_root, and spatial_upsampler_path are required "
                    "when not using a SharedModelCache"
                )
            if not distilled_lora:
                raise ValueError("distilled_lora is required when not using a SharedModelCache")

            self._stage_1_model_ledger = ModelLedger(
                dtype=self.dtype,
                device=device,
                checkpoint_path=checkpoint_path,
                gemma_root_path=gemma_root,
                spatial_upsampler_path=spatial_upsampler_path,
                loras=self._loras,
                fp8transformer=fp8transformer,
                text_encoder_device=text_encoder_device,
                text_encoder_8bit=text_encoder_8bit,
                text_encoder_4bit=text_encoder_4bit,
            )

            self._stage_2_model_ledger = self._stage_1_model_ledger.with_loras(
                loras=distilled_lora,
            )

        self.pipeline_components = PipelineComponents(
            dtype=self.dtype,
            device=device,
        )

    def _get_models(self) -> dict[str, Any]:
        """Get all required models from cache or ledger."""
        if self._model_cache is not None:
            # Stage 1: user LoRAs (or base if no LoRAs)
            if self._loras:
                transformer_stage1 = self._model_cache.build_transformer_with_loras(self._loras)
            else:
                transformer_stage1 = self._model_cache.get_transformer_base()
            # Stage 2: distilled transformer
            transformer_stage2 = self._model_cache.get_transformer_distilled()

            return {
                "text_encoder": self._model_cache.get_text_encoder(),
                "video_encoder": self._model_cache.get_video_encoder(),
                "video_decoder": self._model_cache.get_video_decoder(),
                "audio_decoder": self._model_cache.get_audio_decoder(),
                "vocoder": self._model_cache.get_vocoder(),
                "spatial_upsampler": self._model_cache.get_spatial_upsampler(),
                "transformer_stage1": transformer_stage1,
                "transformer_stage2": transformer_stage2,
            }
        else:
            return {
                "text_encoder": self._stage_1_model_ledger.text_encoder(),
                "video_encoder": self._stage_1_model_ledger.video_encoder(),
                "video_decoder": self._stage_2_model_ledger.video_decoder(),
                "audio_decoder": self._stage_2_model_ledger.audio_decoder(),
                "vocoder": self._stage_2_model_ledger.vocoder(),
                "spatial_upsampler": self._stage_2_model_ledger.spatial_upsampler(),
                "transformer_stage1": self._stage_1_model_ledger.transformer(),
                "transformer_stage2": self._stage_2_model_ledger.transformer(),
            }

    @property
    def _uses_shared_cache(self) -> bool:
        """Check if using shared model cache."""
        return self._model_cache is not None

    @torch.no_grad()
    def __call__(  # noqa: PLR0913
        self,
        prompt: str,
        negative_prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        num_inference_steps: int,
        cfg_guidance_scale: float,
        images: list[tuple[str, int, float]],
        tiling_config: TilingConfig | None = None,
        enhance_prompt: bool = False,
    ) -> tuple[Iterator[torch.Tensor], torch.Tensor]:
        assert_resolution(height=height, width=width, is_two_stage=True)

        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        stepper = EulerDiffusionStep()
        cfg_guider = CFGGuider(cfg_guidance_scale)
        dtype = torch.bfloat16

        # Get all models (from cache or ledger)
        models = self._get_models()
        text_encoder = models["text_encoder"]
        video_encoder = models["video_encoder"]
        transformer_stage1 = models["transformer_stage1"]
        transformer_stage2 = models["transformer_stage2"]
        spatial_upsampler = models["spatial_upsampler"]
        video_decoder = models["video_decoder"]
        audio_decoder = models["audio_decoder"]
        vocoder = models["vocoder"]

        log_generation_stage("Starting generation")

        if enhance_prompt:
            prompt = generate_enhanced_prompt(
                text_encoder, prompt, images[0][0] if len(images) > 0 else None, seed=seed
            )
        context_p, context_n = encode_text(text_encoder, prompts=[prompt, negative_prompt])
        v_context_p, a_context_p = context_p
        v_context_n, a_context_n = context_n

        log_generation_stage("Text encoding complete", {
            "v_context_p": v_context_p,
            "v_context_n": v_context_n,
            "a_context_p": a_context_p,
            "a_context_n": a_context_n,
        })

        # Only cleanup if not using shared cache
        if not self._uses_shared_cache:
            torch.cuda.synchronize()
            del text_encoder
            cleanup_memory()

        torch.cuda.synchronize()
        cleanup_memory()

        # Stage 1: Initial low resolution video generation.
        sigmas = LTX2Scheduler().execute(steps=num_inference_steps).to(dtype=torch.float32, device=self.device)

        def first_stage_denoising_loop(
            sigmas: torch.Tensor, video_state: LatentState, audio_state: LatentState, stepper: DiffusionStepProtocol
        ) -> tuple[LatentState, LatentState]:
            return euler_denoising_loop(
                sigmas=sigmas,
                video_state=video_state,
                audio_state=audio_state,
                stepper=stepper,
                denoise_fn=guider_denoising_func(
                    cfg_guider,
                    v_context_p,
                    v_context_n,
                    a_context_p,
                    a_context_n,
                    transformer=transformer_stage1,  # noqa: F821
                ),
            )

        stage_1_output_shape = VideoPixelShape(
            batch=1,
            frames=num_frames,
            width=width // 2,
            height=height // 2,
            fps=frame_rate,
        )
        stage_1_conditionings = image_conditionings_by_replacing_latent(
            images=images,
            height=stage_1_output_shape.height,
            width=stage_1_output_shape.width,
            video_encoder=video_encoder,
            dtype=dtype,
            device=self.device,
        )
        video_state, audio_state = denoise_audio_video(
            output_shape=stage_1_output_shape,
            conditionings=stage_1_conditionings,
            noiser=noiser,
            sigmas=sigmas,
            stepper=stepper,
            denoising_loop_fn=first_stage_denoising_loop,
            components=self.pipeline_components,
            dtype=dtype,
            device=self.device,
        )

        log_generation_stage("Stage 1 denoising complete", {
            "video_latent": video_state.latent,
            "audio_latent": audio_state.latent,
        })

        if not self._uses_shared_cache:
            torch.cuda.synchronize()
            del transformer_stage1
            cleanup_memory()

        torch.cuda.synchronize()
        cleanup_memory()

        # Stage 2: Upsample and refine the video at higher resolution with distilled LORA.
        upscaled_video_latent = upsample_video(
            latent=video_state.latent[:1],
            video_encoder=video_encoder,
            upsampler=spatial_upsampler,
        )

        log_generation_stage("Spatial upsampling complete", {
            "upscaled_video_latent": upscaled_video_latent,
        })

        if not self._uses_shared_cache:
            torch.cuda.synchronize()
            cleanup_memory()

        distilled_sigmas = torch.Tensor(STAGE_2_DISTILLED_SIGMA_VALUES).to(self.device)

        def second_stage_denoising_loop(
            sigmas: torch.Tensor, video_state: LatentState, audio_state: LatentState, stepper: DiffusionStepProtocol
        ) -> tuple[LatentState, LatentState]:
            return euler_denoising_loop(
                sigmas=sigmas,
                video_state=video_state,
                audio_state=audio_state,
                stepper=stepper,
                denoise_fn=simple_denoising_func(
                    video_context=v_context_p,
                    audio_context=a_context_p,
                    transformer=transformer_stage2,  # noqa: F821
                ),
            )

        stage_2_output_shape = VideoPixelShape(batch=1, frames=num_frames, width=width, height=height, fps=frame_rate)
        stage_2_conditionings = image_conditionings_by_replacing_latent(
            images=images,
            height=stage_2_output_shape.height,
            width=stage_2_output_shape.width,
            video_encoder=video_encoder,
            dtype=dtype,
            device=self.device,
        )
        video_state, audio_state = denoise_audio_video(
            output_shape=stage_2_output_shape,
            conditionings=stage_2_conditionings,
            noiser=noiser,
            sigmas=distilled_sigmas,
            stepper=stepper,
            denoising_loop_fn=second_stage_denoising_loop,
            components=self.pipeline_components,
            dtype=dtype,
            device=self.device,
            noise_scale=distilled_sigmas[0],
            initial_video_latent=upscaled_video_latent,
            initial_audio_latent=audio_state.latent,
        )

        log_generation_stage("Stage 2 denoising complete", {
            "video_latent": video_state.latent,
            "audio_latent": audio_state.latent,
        })

        # Only cleanup if not using shared cache
        if not self._uses_shared_cache:
            torch.cuda.synchronize()
            del transformer_stage2
            del video_encoder
            cleanup_memory()

        log_generation_stage("Starting VAE decode")
        decoded_video = vae_decode_video(video_state.latent, video_decoder, tiling_config)
        log_generation_stage("Video VAE decode complete")

        decoded_audio = vae_decode_audio(audio_state.latent, audio_decoder, vocoder)
        log_generation_stage("Audio decode complete")


        return decoded_video, decoded_audio


@torch.inference_mode()
def main() -> None:
    logging.getLogger().setLevel(logging.INFO)
    parser = default_2_stage_arg_parser()
    args = parser.parse_args()
    text_encoder_device = "cpu" if args.text_encoder_cpu else None
    pipeline = TI2VidTwoStagesPipeline(
        checkpoint_path=args.checkpoint_path,
        distilled_lora=args.distilled_lora,
        spatial_upsampler_path=args.spatial_upsampler_path,
        gemma_root=args.gemma_root,
        loras=args.lora,
        fp8transformer=args.enable_fp8,
        text_encoder_device=text_encoder_device,
        text_encoder_8bit=args.text_encoder_8bit,
        text_encoder_4bit=args.text_encoder_4bit,
    )
    tiling_config = TilingConfig.default()
    video_chunks_number = get_video_chunks_number(args.num_frames, tiling_config)
    video, audio = pipeline(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        frame_rate=args.frame_rate,
        num_inference_steps=args.num_inference_steps,
        cfg_guidance_scale=args.cfg_guidance_scale,
        images=args.images,
        tiling_config=tiling_config,
    )

    encode_video(
        video=video,
        fps=args.frame_rate,
        audio=audio,
        audio_sample_rate=AUDIO_SAMPLE_RATE,
        output_path=args.output_path,
        video_chunks_number=video_chunks_number,
    )


if __name__ == "__main__":
    main()
