# Open Broadcast

A free, open-source alternative to NVIDIA Broadcast that runs entirely on CPU. No GPU required.

## Features

- 🎙️ **AI Noise Suppression** — Remove background noise from your microphone in real-time (RNNoise)
- 🖼️ **Background Removal / Blur** — Replace or blur your webcam background (MediaPipe Selfie Segmentation)
- 📐 **Auto Frame** — Automatic face tracking and centering (MediaPipe Face Detection)
- 📷 **Virtual Camera** — Output to a virtual camera for use in Zoom, Discord, Meet, etc.
- 🎛️ **System Tray UI** — Minimal desktop app with easy controls

## Requirements

- Python 3.10+
- Linux (v4l2loopback for virtual camera) or Windows (OBS Virtual Camera)
- Webcam
- Microphone

## Quick Start

```bash
# Clone
git clone https://github.com/ClawrissaBot/open-broadcast.git
cd open-broadcast

# Install dependencies
pip install -e .

# Linux: Load virtual camera kernel module
sudo modprobe v4l2loopback devices=1 card_label="Open Broadcast" exclusive_caps=1

# Run
open-broadcast
```

## Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌──────────────┐
│  Webcam      │────▶│  Video Pipeline   │────▶│ Virtual Cam  │
│  (cv2)       │     │  - Segmentation   │     │ (v4l2loopback│
└─────────────┘     │  - Auto Frame     │     │  / OBS)      │
                     └──────────────────┘     └──────────────┘

┌─────────────┐     ┌──────────────────┐     ┌──────────────┐
│  Microphone  │────▶│  Audio Pipeline   │────▶│ Virtual Mic  │
│  (sounddevice│     │  - RNNoise        │     │ (PulseAudio  │
│   / pyaudio) │     │                   │     │  / pipewire) │
└─────────────┘     └──────────────────┘     └──────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    System Tray UI (PySide6)                  │
│  [✓ Noise Suppression] [✓ Background Blur] [✓ Auto Frame]  │
└─────────────────────────────────────────────────────────────┘
```

## Models Used

| Feature | Model | License | Runs on |
|---------|-------|---------|---------|
| Noise Suppression | [RNNoise](https://github.com/xiph/rnnoise) | BSD-3 | CPU |
| Background Segmentation | [MediaPipe Selfie Segmentation](https://ai.google.dev/edge/mediapipe/solutions/vision/image_segmenter) | Apache 2.0 | CPU |
| Face Detection | [MediaPipe Face Detection](https://ai.google.dev/edge/mediapipe/solutions/vision/face_detector) | Apache 2.0 | CPU |

## License

MIT
