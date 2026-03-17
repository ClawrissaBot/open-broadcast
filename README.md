# Open Broadcast

A free, open-source alternative to NVIDIA Broadcast that runs entirely on CPU. No GPU required.

## Features

- 🎙️ **AI Noise Suppression** — DeepFilterNet (full-band 48kHz, closest thing to RTX Voice on CPU)
- 🛡️ **Voice Activity Detection** — Silero VAD gates processing during silence (saves CPU)
- 🖼️ **Background Removal / Blur** — Replace or blur your webcam background (MediaPipe Selfie Segmentation)
- 📐 **Auto Frame** — Automatic face tracking and centering (MediaPipe Face Detection)
- 📷 **Virtual Camera** — Output to a virtual camera for use in Zoom, Discord, Meet, etc.
- 🎛️ **System Tray UI** — Minimal desktop app with easy controls
- 🪟 **Cross-Platform** — Linux + Windows support

## Requirements

- Python 3.10+
- PyTorch (CPU build is fine)
- Linux (PulseAudio/PipeWire + v4l2loopback) or Windows (VB-Audio Virtual Cable + OBS Virtual Cam)
- Webcam + Microphone

## Quick Start

```bash
# Clone
git clone https://github.com/ClawrissaBot/open-broadcast.git
cd open-broadcast

# Install PyTorch CPU
pip install torch torchaudio -f https://download.pytorch.org/whl/cpu/torch_stable.html

# Install Open Broadcast
pip install -e .

# Linux: Load virtual camera kernel module
sudo modprobe v4l2loopback devices=1 card_label="Open Broadcast" exclusive_caps=1

# Run
open-broadcast
```

### Windows Setup

```powershell
# Install PyTorch CPU
pip install torch torchaudio -f https://download.pytorch.org/whl/cpu/torch_stable.html

# Install Open Broadcast
pip install -e .

# Install VB-Audio Virtual Cable for virtual mic output
# Download from: https://vb-audio.com/Cable/

# Run
open-broadcast
```

## Audio Pipeline

The audio pipeline uses a two-stage architecture for quality + efficiency:

```
┌─────────────┐     ┌──────────────┐     ┌───────────────────┐     ┌──────────────┐
│  Microphone  │────▶│  Silero VAD  │────▶│  DeepFilterNet    │────▶│ Virtual Mic  │
│              │     │  (is speech?)│     │  (noise removal)  │     │              │
└─────────────┘     └──────────────┘     └───────────────────┘     └──────────────┘
                     16kHz check          48kHz full-band
                     ~0.5ms               ~2ms per frame
```

**Silero VAD** detects whether you're speaking. During silence, it gates the output (near-silent, saves CPU). When speech is detected, **DeepFilterNet** runs full noise suppression at 48kHz — handling non-stationary noise (voices, TV, traffic) much better than RNNoise.

A 300ms holdover prevents choppy cuts at the end of sentences.

## Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌──────────────┐
│  Webcam      │────▶│  Video Pipeline   │────▶│ Virtual Cam  │
│  (cv2)       │     │  - Segmentation   │     │ (v4l2loopback│
└─────────────┘     │  - Auto Frame     │     │  / OBS VCam) │
                     └──────────────────┘     └──────────────┘

┌─────────────┐     ┌──────────────────┐     ┌──────────────┐
│  Microphone  │────▶│  Audio Pipeline   │────▶│ Virtual Mic  │
│  (sounddevice│     │  - Silero VAD     │     │ (PulseAudio  │
│   )          │     │  - DeepFilterNet  │     │  / VB-Cable) │
└─────────────┘     └──────────────────┘     └──────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    System Tray UI (PySide6)                  │
│  [✓ Noise Suppression] [✓ VAD] [✓ Bg Blur] [✓ Auto Frame] │
└─────────────────────────────────────────────────────────────┘
```

## Models Used

- **[DeepFilterNet](https://github.com/Rikorose/DeepFilterNet)** — Deep neural network for full-band (48kHz) noise suppression. Handles non-stationary noise (voices, music, traffic). Real-time factor ~0.19 on i5 CPU. *License: MIT/Apache 2.0*
- **[Silero VAD](https://github.com/snakers4/silero-vad)** — Voice Activity Detection. Gates noise suppression during silence to save CPU. *License: MIT*
- **[MediaPipe Selfie Segmentation](https://ai.google.dev/edge/mediapipe/solutions/vision/image_segmenter)** — Background removal/blur. *License: Apache 2.0*
- **[MediaPipe Face Detection](https://ai.google.dev/edge/mediapipe/solutions/vision/face_detector)** — Auto-frame face tracking. *License: Apache 2.0*

## Configuration

```python
# In the tray UI or config file:
noise_suppression = True   # Enable/disable DeepFilterNet
vad_enabled = True          # Enable/disable Silero VAD gating
vad_threshold = 0.5         # VAD sensitivity (0.0-1.0, lower = more sensitive)
```

## Platform Notes

**Linux:**
- Audio: PulseAudio or PipeWire (sounddevice auto-detects)
- Virtual mic: `pactl load-module module-null-sink sink_name=OpenBroadcast`
- Virtual cam: `v4l2loopback` kernel module

**Windows:**
- Audio: WASAPI (sounddevice auto-detects)
- Virtual mic: [VB-Audio Virtual Cable](https://vb-audio.com/Cable/) (free)
- Virtual cam: OBS Virtual Camera or [Unity Capture](https://github.com/schellingb/UnityCapture)

## License

MIT
