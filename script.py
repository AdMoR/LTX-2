#!/usr/bin/env python3
"""
LTX-2 Video-to-Video Pipeline Script
Optimized for 16GB VRAM / 32GB RAM

Usage:
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python v2v_inference.py \
        --prompt "A cat walking on the beach" \
        --input-video input.mp4 \
        --output output.mp4
"""

import argparse
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

# --------------------------------------------------------------------------
# 1. Model Download Helpers
# --------------------------------------------------------------------------

HF_REPO_ID = "Lightricks/LTX-2"
GEMMA_REPO_ID = "google/gemma-3-12b-it-qat-q4_0-unquantized"

# Files needed for video-to-video with FP8
REQUIRED_FILES = {
    "checkpoint": "ltx-2-19b-distilled-fp8.safetensors",
    "upsampler": "ltx-2-spatial-upscaler-x2-1.0.safetensors",
    # IC-LoRA for video-to-video control (choose one based on your use case)
    "ic_lora_depth": "ltx-2-19b-ic-lora-depth-control.safetensors",
    "ic_lora_canny": "ltx-2-19b-ic-lora-canny-control.safetensors",
}


def download_models(cache_dir: Path) -> dict[str, Path]:
    """Download required models from HuggingFace Hub if not present."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    print("Checking and downloading required models...")

    # Download main LTX-2 files
    for key, filename in REQUIRED_FILES.items():
        local_path = cache_dir / filename
        if local_path.exists():
            print(f"  ✓ {filename} (cached)")
            paths[key] = local_path
        else:
            print(f"  ↓ Downloading {filename}...")
            # Determine repo based on file type
            repo_id = HF_REPO_ID
            if "ic-lora" in filename:
                # IC-LoRAs are in separate repos
                if "depth" in filename:
                    repo_id = "Lightricks/LTX-2-19b-IC-LoRA-Depth-Control"
                elif "canny" in filename:
                    repo_id = "Lightricks/LTX-2-19b-IC-LoRA-Canny-Control"
            paths[key] = Path(
                hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    local_dir=cache_dir,
                )
            )
            print(f"  ✓ {filename}")

    # Download Gemma text encoder
    gemma_dir = cache_dir / "gemma-3-12b-it"
    if gemma_dir.exists() and any(gemma_dir.iterdir()):
        print(f"  ✓ Gemma text encoder (cached)")
        paths["gemma"] = gemma_dir
    else:
        print("  ↓ Downloading Gemma text encoder (this may take a while)...")
        from huggingface_hub import snapshot_download

        paths["gemma"] = Path(
            snapshot_download(
                repo_id=GEMMA_REPO_ID,
                local_dir=gemma_dir,
            )
        )
        print("  ✓ Gemma text encoder")

    return paths


# --------------------------------------------------------------------------
# 2. Video-to-Video Pipeline
# --------------------------------------------------------------------------


def run_video_to_video(
    prompt: str,
    input_video: str,
    output_path: str,
    model_paths: dict[str, Path],
    seed: int = 42,
    height: int = 544,
    width: int = 960,
    num_frames: int = 97,
    frame_rate: float = 25.0,
    conditioning_strength: float = 0.8,
    ic_lora_type: str = "depth",  # "depth" or "canny"
    text_encoder_4bit: bool = False,
    text_encoder_8bit: bool = False,
) -> None:
    """Run video-to-video generation with IC-LoRA."""
    from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
    from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
    from ltx_pipelines.ic_lora import ICLoraPipeline
    from ltx_pipelines.utils.constants import AUDIO_SAMPLE_RATE
    from ltx_pipelines.utils.media_io import encode_video

    print(f"\n{'=' * 60}")
    print("LTX-2 Video-to-Video Generation")
    print(f"{'=' * 60}")
    print(f"Input: {input_video}")
    print(f"Output: {output_path}")
    print(f"Prompt: {prompt}")
    print(f"Resolution: {width}x{height}")
    print(f"Frames: {num_frames} @ {frame_rate} fps")
    print(f"IC-LoRA: {ic_lora_type}")
    print(f"Conditioning strength: {conditioning_strength}")
    te_mode = "4-bit" if text_encoder_4bit else "8-bit" if text_encoder_8bit else "full precision"
    print(f"Text encoder: {te_mode}")
    print(f"Seed: {seed}")
    print(f"{'=' * 60}\n")

    # Select IC-LoRA based on type
    ic_lora_key = f"ic_lora_{ic_lora_type}"
    if ic_lora_key not in model_paths:
        raise ValueError(f"IC-LoRA type '{ic_lora_type}' not downloaded. Available: depth, canny")

    lora = LoraPathStrengthAndSDOps(
        path=str(model_paths[ic_lora_key]),
        strength=1.0,
        sd_ops=LTXV_LORA_COMFY_RENAMING_MAP,
    )

    print(f"Loading pipeline with FP8 transformer and {te_mode} text encoder...")
    pipeline = ICLoraPipeline(
        checkpoint_path=str(model_paths["checkpoint"]),
        spatial_upsampler_path=str(model_paths["upsampler"]),
        gemma_root=str(model_paths["gemma"]),
        loras=[lora],
        fp8transformer=True,  # Enable FP8 for transformer (~10GB VRAM)
        text_encoder_4bit=text_encoder_4bit,  # 4-bit: ~6GB VRAM
        text_encoder_8bit=text_encoder_8bit,  # 8-bit: ~12GB VRAM
    )

    print("Running inference...")
    tiling_config = TilingConfig.default()
    video_chunks_number = get_video_chunks_number(num_frames, tiling_config)

    video, audio = pipeline(
        prompt=prompt,
        seed=seed,
        height=height,
        width=width,
        num_frames=num_frames,
        frame_rate=frame_rate,
        images=[],  # No image conditioning, using video conditioning
        video_conditioning=[(input_video, conditioning_strength)],
        tiling_config=tiling_config,
    )

    print("Encoding output video...")
    encode_video(
        video=video,
        fps=frame_rate,
        audio=audio,
        audio_sample_rate=AUDIO_SAMPLE_RATE,
        output_path=output_path,
        video_chunks_number=video_chunks_number,
    )

    print(f"\n✓ Video saved to: {output_path}")


# --------------------------------------------------------------------------
# 3. Main Entry Point
# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LTX-2 Video-to-Video Generation (16GB VRAM optimized)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument("--input-video", type=str, required=True, help="Input video path")
    parser.add_argument("--output", type=str, required=True, help="Output video path")
    parser.add_argument(
        "--model-cache",
        type=str,
        default="./models",
        help="Directory to cache downloaded models",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--height",
        type=int,
        default=544,
        help="Video height (divisible by 64 for two-stage)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=960,
        help="Video width (divisible by 64 for two-stage)",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=97,
        help="Number of frames (must be 8k+1)",
    )
    parser.add_argument("--frame-rate", type=float, default=25.0, help="Frame rate")
    parser.add_argument(
        "--conditioning-strength",
        type=float,
        default=0.8,
        help="How strongly to follow the input video (0.0-1.0)",
    )
    parser.add_argument(
        "--ic-lora-type",
        type=str,
        choices=["depth", "canny"],
        default="depth",
        help="IC-LoRA type for video conditioning",
    )
    parser.add_argument(
        "--text-encoder-4bit",
        action="store_true",
        help="Use 4-bit NF4 quantization for Gemma text encoder (~6GB VRAM). Recommended for 16GB systems.",
    )
    parser.add_argument(
        "--text-encoder-8bit",
        action="store_true",
        help="Use 8-bit quantization for Gemma text encoder (~12GB VRAM).",
    )

    args = parser.parse_args()

    # Validate mutually exclusive options
    if args.text_encoder_4bit and args.text_encoder_8bit:
        parser.error("Cannot use both --text-encoder-4bit and --text-encoder-8bit")

    # Check CUDA availability
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for inference")

    # Show GPU info
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")

    if gpu_mem < 15:
        print(
            "⚠️  Warning: Less than 16GB VRAM detected. "
            "Consider reducing resolution or frame count."
        )

    # Download models
    cache_dir = Path(args.model_cache).resolve()
    model_paths = download_models(cache_dir)

    # Run inference
    run_video_to_video(
        prompt=args.prompt,
        input_video=args.input_video,
        output_path=args.output,
        model_paths=model_paths,
        seed=args.seed,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        frame_rate=args.frame_rate,
        conditioning_strength=args.conditioning_strength,
        ic_lora_type=args.ic_lora_type,
        text_encoder_4bit=args.text_encoder_4bit,
        text_encoder_8bit=args.text_encoder_8bit,
    )


if __name__ == "__main__":
    main()
