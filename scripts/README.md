# LTX Scripts

Utility scripts for working with LTX-2.

## LTX Client

A visual Streamlit client for the LTX-2 video generation server.

![LTX Client](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)

### Installation

```bash
pip install -r requirements.txt
```

### Usage

1. **Start the LTX Server** (in another terminal):

```bash
cd packages/ltx-server
uvicorn ltx_server.main:app --host 0.0.0.0 --port 8000
```

2. **Run the client**:

```bash
streamlit run scripts/ltx_client.py
```

3. Open `http://localhost:8501` in your browser.

### Features

- **Pipeline Selection**: Choose from available generation pipelines
- **Parameter Configuration**: Adjust all generation settings
- **Image Conditioning**: Upload keyframe images with per-frame strength control
- **Video Conditioning**: Upload conditioning videos for IC-LoRA
- **Real-time Progress**: Live job status updates with progress bar
- **Video Preview**: View and download generated videos directly in the UI

### Screenshots

The client provides a modern, dark-themed interface with:

- Server status indicator
- Pipeline selector with descriptions
- Video dimension and frame controls
- Advanced settings (seed, CFG scale, inference steps)
- Image/video upload with drag-and-drop
- Progress tracking and video playback

### Configuration

The default server URL is `http://localhost:8000`. You can change this in the sidebar.

Environment variables:
- `STREAMLIT_SERVER_PORT`: Change the client port (default: 8501)
