"""
LTX-2 Video Generation Client

A Streamlit-based visual client for the LTX-2 video generation server.
"""

import io
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

import requests
import streamlit as st

# =============================================================================
# Configuration
# =============================================================================

DEFAULT_SERVER_URL = "http://localhost:8000"
POLL_INTERVAL = 2.0  # seconds


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class PipelineConfig:
    """Configuration for a pipeline."""

    name: str
    display_name: str
    description: str
    supports_negative_prompt: bool = False
    supports_video_conditioning: bool = False
    supports_cfg: bool = False


# Pipeline configurations
PIPELINES = {
    "distilled": PipelineConfig(
        name="distilled",
        display_name="⚡ Distilled (Fast)",
        description="Fast two-stage generation without CFG. Best for quick iterations.",
    ),
    "ti2vid_one_stage": PipelineConfig(
        name="ti2vid_one_stage",
        display_name="🎬 Single-Stage CFG",
        description="Single-stage generation with classifier-free guidance for higher quality.",
        supports_negative_prompt=True,
        supports_cfg=True,
    ),
    "ti2vid_two_stages": PipelineConfig(
        name="ti2vid_two_stages",
        display_name="🎥 Two-Stage CFG",
        description="Two-stage generation with upscaling for best quality.",
        supports_negative_prompt=True,
        supports_cfg=True,
    ),
    "ic_lora": PipelineConfig(
        name="ic_lora",
        display_name="🔄 IC-LoRA",
        description="Video-to-video generation with style transfer capabilities.",
        supports_video_conditioning=True,
    ),
    "keyframe_interpolation": PipelineConfig(
        name="keyframe_interpolation",
        display_name="🖼️ Keyframe Interpolation",
        description="Interpolate between keyframe images to create smooth transitions.",
        supports_negative_prompt=True,
        supports_cfg=True,
    ),
}


# =============================================================================
# API Client
# =============================================================================


class LTXClient:
    """Client for LTX Server API."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def health_check(self) -> dict | None:
        """Check server health."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            response.raise_for_status()
            return response.json()
        except requests.RequestException:
            return None

    def get_pipelines(self) -> list[dict]:
        """Get available pipelines."""
        try:
            response = requests.get(f"{self.base_url}/pipelines", timeout=5)
            response.raise_for_status()
            return response.json().get("pipelines", [])
        except requests.RequestException:
            return []

    def submit_job(
        self,
        pipeline: str,
        params: dict[str, Any],
        images: list[tuple[bytes, str, int, float]] | None = None,
        video_conditioning: tuple[bytes, str, float] | None = None,
    ) -> dict | None:
        """Submit a generation job."""
        url = f"{self.base_url}/generate/{pipeline}"

        # Build form data
        form_data = {}
        for key, value in params.items():
            if value is not None:
                form_data[key] = (None, str(value))

        files = []

        # Add images if provided
        if images:
            for img_data, filename, frame_idx, strength in images:
                files.append(("images", (filename, img_data, "image/png")))
                form_data[f"image_frame_indices"] = (None, str(frame_idx))
                form_data[f"image_strengths"] = (None, str(strength))

        # Add video conditioning if provided
        if video_conditioning:
            video_data, filename, strength = video_conditioning
            files.append(("video_conditioning", (filename, video_data, "video/mp4")))
            form_data["video_conditioning_strength"] = (None, str(strength))

        try:
            response = requests.post(url, data=form_data, files=files, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            st.error(f"Failed to submit job: {e}")
            return None

    def get_job(self, job_id: str) -> dict | None:
        """Get job status."""
        try:
            response = requests.get(f"{self.base_url}/jobs/{job_id}", timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.RequestException:
            return None

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a job."""
        try:
            response = requests.delete(f"{self.base_url}/jobs/{job_id}", timeout=10)
            return response.status_code == 200
        except requests.RequestException:
            return False


# =============================================================================
# UI Components
# =============================================================================


def setup_page():
    """Configure the Streamlit page."""
    st.set_page_config(
        page_title="LTX-2 Video Generator",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom CSS for a distinctive look
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Outfit:wght@300;400;500;600;700&display=swap');
        
        :root {
            --bg-primary: #0a0a0f;
            --bg-secondary: #12121a;
            --bg-tertiary: #1a1a25;
            --accent-primary: #00d9ff;
            --accent-secondary: #ff3366;
            --accent-gradient: linear-gradient(135deg, #00d9ff 0%, #ff3366 100%);
            --text-primary: #f0f0f5;
            --text-secondary: #8888aa;
            --border-color: #2a2a3a;
        }
        
        .stApp {
            background: var(--bg-primary);
            font-family: 'Outfit', sans-serif;
        }
        
        .main .block-container {
            padding-top: 2rem;
            max-width: 1400px;
        }
        
        h1, h2, h3 {
            font-family: 'Outfit', sans-serif;
            font-weight: 600;
            background: var(--accent-gradient);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .stSidebar {
            background: var(--bg-secondary);
            border-right: 1px solid var(--border-color);
        }
        
        .stSidebar .stMarkdown {
            color: var(--text-primary);
        }
        
        .stButton > button {
            background: var(--accent-gradient);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.75rem 2rem;
            font-family: 'Outfit', sans-serif;
            font-weight: 500;
            font-size: 1rem;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(0, 217, 255, 0.3);
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0, 217, 255, 0.4);
        }
        
        .stTextInput > div > div > input,
        .stTextArea > div > div > textarea,
        .stSelectbox > div > div > div {
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            color: var(--text-primary);
            font-family: 'JetBrains Mono', monospace;
        }
        
        .stSlider > div > div > div {
            background: var(--accent-gradient);
        }
        
        .status-card {
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1rem 0;
        }
        
        .status-pending { border-left: 4px solid #ffaa00; }
        .status-processing { border-left: 4px solid #00d9ff; }
        .status-completed { border-left: 4px solid #00ff88; }
        .status-failed { border-left: 4px solid #ff3366; }
        
        .metric-value {
            font-family: 'JetBrains Mono', monospace;
            font-size: 2rem;
            font-weight: 600;
            color: var(--accent-primary);
        }
        
        .pipeline-badge {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            border-radius: 20px;
            font-size: 0.85rem;
            color: var(--text-secondary);
        }
        
        .video-container {
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1rem;
            display: flex;
            justify-content: center;
        }
        
        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header():
    """Render the page header."""
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("# 🎬 LTX-2 Video Generator")
        st.markdown(
            "<p style='color: #8888aa; margin-top: -1rem;'>Create stunning AI-generated videos with text and image prompts</p>",
            unsafe_allow_html=True,
        )


def render_server_status(client: LTXClient):
    """Render server connection status."""
    health = client.health_check()

    if health:
        status_color = "#00ff88"
        status_text = "Connected"
        gpu_status = "🟢 GPU" if health.get("gpu_available") else "🔴 CPU"
        pipelines = health.get("pipelines_loaded", [])

        st.sidebar.markdown(
            f"""
            <div style='padding: 1rem; background: #1a1a25; border-radius: 8px; border: 1px solid #2a2a3a;'>
                <div style='display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;'>
                    <span style='width: 8px; height: 8px; background: {status_color}; border-radius: 50%;'></span>
                    <span style='color: {status_color}; font-weight: 500;'>{status_text}</span>
                </div>
                <div style='color: #8888aa; font-size: 0.85rem;'>
                    {gpu_status} • v{health.get('version', '?')}
                </div>
                <div style='color: #8888aa; font-size: 0.8rem; margin-top: 0.5rem;'>
                    Loaded: {', '.join(pipelines) if pipelines else 'None'}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return pipelines
    else:
        st.sidebar.markdown(
            """
            <div style='padding: 1rem; background: #1a1a25; border-radius: 8px; border: 1px solid #ff3366;'>
                <div style='display: flex; align-items: center; gap: 0.5rem;'>
                    <span style='width: 8px; height: 8px; background: #ff3366; border-radius: 50%;'></span>
                    <span style='color: #ff3366; font-weight: 500;'>Disconnected</span>
                </div>
                <div style='color: #8888aa; font-size: 0.85rem; margin-top: 0.5rem;'>
                    Cannot reach server
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return []


def render_pipeline_selector(loaded_pipelines: list[str]) -> PipelineConfig | None:
    """Render pipeline selection."""
    st.sidebar.markdown("### Pipeline")

    # Filter to only show loaded pipelines
    available = {k: v for k, v in PIPELINES.items() if k in loaded_pipelines}

    if not available:
        st.sidebar.warning("No pipelines available")
        return None

    selected = st.sidebar.selectbox(
        "Select pipeline",
        options=list(available.keys()),
        format_func=lambda x: available[x].display_name,
        label_visibility="collapsed",
    )

    if selected:
        config = available[selected]
        st.sidebar.markdown(
            f"<p style='color: #8888aa; font-size: 0.85rem;'>{config.description}</p>",
            unsafe_allow_html=True,
        )
        return config

    return None


def render_generation_params(pipeline: PipelineConfig) -> dict[str, Any]:
    """Render generation parameter controls."""
    params = {}

    st.sidebar.markdown("### Generation")

    # Prompt
    params["prompt"] = st.sidebar.text_area(
        "Prompt",
        placeholder="Describe your video...",
        height=100,
    )

    # Negative prompt (if supported)
    if pipeline.supports_negative_prompt:
        with st.sidebar.expander("Negative Prompt"):
            params["negative_prompt"] = st.text_area(
                "What to avoid",
                value="worst quality, inconsistent motion, blurry, jittery, distorted",
                height=80,
                label_visibility="collapsed",
            )

    st.sidebar.markdown("### Video Settings")

    col1, col2 = st.sidebar.columns(2)
    with col1:
        params["width"] = st.number_input("Width", value=1152, step=64, min_value=256)
    with col2:
        params["height"] = st.number_input("Height", value=768, step=64, min_value=256)

    col1, col2 = st.sidebar.columns(2)
    with col1:
        params["num_frames"] = st.number_input(
            "Frames",
            value=97,
            step=8,
            min_value=9,
            help="Must be 8k+1 (e.g., 9, 17, 25, 33, ...)",
        )
    with col2:
        params["frame_rate"] = st.number_input(
            "FPS", value=24.0, step=1.0, min_value=1.0
        )

    # Advanced settings
    with st.sidebar.expander("Advanced"):
        params["seed"] = st.number_input("Seed", value=42, min_value=0)
        params["enhance_prompt"] = st.checkbox("Enhance prompt", value=False)

        if pipeline.supports_cfg:
            params["num_inference_steps"] = st.slider(
                "Inference steps", min_value=10, max_value=100, value=40
            )
            params["cfg_guidance_scale"] = st.slider(
                "CFG scale", min_value=1.0, max_value=10.0, value=3.0, step=0.5
            )

    return params


def render_image_conditioning() -> list[tuple[bytes, str, int, float]]:
    """Render image conditioning upload."""
    images = []

    with st.expander("🖼️ Image Conditioning", expanded=False):
        st.markdown(
            "<p style='color: #8888aa; font-size: 0.85rem;'>Upload images to condition specific frames</p>",
            unsafe_allow_html=True,
        )

        uploaded_files = st.file_uploader(
            "Upload images",
            type=["png", "jpg", "jpeg", "webp"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        if uploaded_files:
            for i, file in enumerate(uploaded_files):
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    st.image(file, width=150)
                with col2:
                    frame_idx = st.number_input(
                        f"Frame #{i + 1}",
                        value=0 if i == 0 else 96,
                        min_value=0,
                        key=f"frame_idx_{i}",
                    )
                with col3:
                    strength = st.slider(
                        f"Strength #{i + 1}",
                        min_value=0.0,
                        max_value=1.0,
                        value=1.0,
                        key=f"strength_{i}",
                    )

                images.append((file.getvalue(), file.name, frame_idx, strength))

    return images


def render_video_conditioning() -> tuple[bytes, str, float] | None:
    """Render video conditioning upload for IC-LoRA."""
    with st.expander("🎥 Video Conditioning", expanded=False):
        st.markdown(
            "<p style='color: #8888aa; font-size: 0.85rem;'>Upload a video for style transfer / video-to-video</p>",
            unsafe_allow_html=True,
        )

        uploaded_video = st.file_uploader(
            "Upload video",
            type=["mp4", "webm", "mov"],
            label_visibility="collapsed",
        )

        if uploaded_video:
            st.video(uploaded_video)
            strength = st.slider(
                "Conditioning strength",
                min_value=0.0,
                max_value=1.0,
                value=1.0,
            )
            return (uploaded_video.getvalue(), uploaded_video.name, strength)

    return None


def render_job_status(job: dict):
    """Render job status card."""
    status = job.get("status", "unknown")
    progress = job.get("progress")

    status_class = f"status-{status}"
    status_emoji = {
        "pending": "⏳",
        "processing": "🔄",
        "completed": "✅",
        "failed": "❌",
        "cancelled": "🚫",
    }.get(status, "❓")

    st.markdown(
        f"""
        <div class='status-card {status_class}'>
            <div style='display: flex; justify-content: space-between; align-items: center;'>
                <div>
                    <span style='font-size: 1.25rem;'>{status_emoji}</span>
                    <span style='font-weight: 600; color: #f0f0f5; margin-left: 0.5rem;'>{status.upper()}</span>
                </div>
                <span class='pipeline-badge'>{job.get('pipeline', 'unknown')}</span>
            </div>
            <div style='color: #8888aa; font-size: 0.85rem; margin-top: 0.5rem;'>
                Job ID: <code style='color: #00d9ff;'>{job.get('job_id', 'unknown')[:8]}...</code>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if progress is not None and status == "processing":
        st.progress(progress, text=f"Progress: {int(progress * 100)}%")


def render_video_result(job: dict):
    """Render the generated video."""
    video_url = job.get("video_url")

    if video_url:
        st.markdown("### 🎬 Generated Video")
        st.video(video_url)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                f"[⬇️ Download Video]({video_url})",
                unsafe_allow_html=True,
            )

        audio_url = job.get("audio_url")
        if audio_url:
            with col2:
                st.markdown(
                    f"[⬇️ Download Audio]({audio_url})",
                    unsafe_allow_html=True,
                )


def poll_job(client: LTXClient, job_id: str) -> dict | None:
    """Poll job status until completion."""
    status_placeholder = st.empty()
    progress_placeholder = st.empty()

    while True:
        job = client.get_job(job_id)
        if not job:
            st.error("Failed to fetch job status")
            return None

        status = job.get("status")

        with status_placeholder.container():
            render_job_status(job)

        if status in ["completed", "failed", "cancelled"]:
            return job

        time.sleep(POLL_INTERVAL)


# =============================================================================
# Main Application
# =============================================================================


def main():
    """Main application entry point."""
    setup_page()

    # Initialize session state
    if "server_url" not in st.session_state:
        st.session_state.server_url = DEFAULT_SERVER_URL
    if "current_job" not in st.session_state:
        st.session_state.current_job = None

    # Sidebar: Server configuration
    st.sidebar.markdown("## ⚙️ Server")
    st.session_state.server_url = st.sidebar.text_input(
        "Server URL",
        value=st.session_state.server_url,
        label_visibility="collapsed",
    )

    client = LTXClient(st.session_state.server_url)

    # Server status
    loaded_pipelines = render_server_status(client)

    # Pipeline selector
    pipeline = render_pipeline_selector(loaded_pipelines)

    # Generation params
    if pipeline:
        params = render_generation_params(pipeline)

        st.sidebar.markdown("---")

        # Generate button
        generate_clicked = st.sidebar.button(
            "🚀 Generate Video",
            use_container_width=True,
            disabled=not params.get("prompt"),
        )
    else:
        params = {}
        generate_clicked = False

    # Main content area
    render_header()

    if not loaded_pipelines:
        st.warning(
            "⚠️ Cannot connect to server. Please check the server URL and ensure the server is running."
        )
        st.code(
            f"# Start the server:\nuvicorn ltx_server.main:app --host 0.0.0.0 --port 8000",
            language="bash",
        )
        return

    # Conditioning inputs (main area)
    col1, col2 = st.columns(2)

    with col1:
        images = render_image_conditioning()

    with col2:
        video_cond = None
        if pipeline and pipeline.supports_video_conditioning:
            video_cond = render_video_conditioning()

    st.markdown("---")

    # Handle generation
    if generate_clicked and pipeline:
        if not params.get("prompt"):
            st.error("Please enter a prompt")
        else:
            with st.spinner("Submitting job..."):
                job = client.submit_job(
                    pipeline=pipeline.name,
                    params=params,
                    images=images if images else None,
                    video_conditioning=video_cond,
                )

            if job:
                st.session_state.current_job = job
                st.success(f"Job submitted: {job.get('job_id', 'unknown')[:8]}...")

                # Poll for completion
                result = poll_job(client, job["job_id"])

                if result:
                    st.session_state.current_job = result

                    if result.get("status") == "completed":
                        render_video_result(result)
                    elif result.get("status") == "failed":
                        st.error(f"Generation failed: {result.get('error', 'Unknown error')}")

    # Show previous result if exists
    elif st.session_state.current_job:
        job = st.session_state.current_job
        render_job_status(job)

        if job.get("status") == "completed":
            render_video_result(job)


if __name__ == "__main__":
    main()
