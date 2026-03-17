"""Audio pipeline — noise suppression via DeepFilterNet + Silero VAD, virtual mic output.

Supports Linux (PulseAudio/PipeWire) and Windows (WASAPI via sounddevice).
"""

from __future__ import annotations

import platform
import threading
import time
from typing import TYPE_CHECKING

import numpy as np

try:
    import sounddevice as sd

    HAS_SD = True
except ImportError:
    HAS_SD = False

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if TYPE_CHECKING:
    from open_broadcast.pipeline.manager import PipelineConfig


class SileroVAD:
    """Voice Activity Detection using Silero VAD (torch or ONNX)."""

    SAMPLE_RATE = 16000  # Silero works at 16kHz
    WINDOW_SAMPLES = 512  # 32ms at 16kHz

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self._model = None
        self._available = False
        self._load_model()

    def _load_model(self):
        if not HAS_TORCH:
            print("[vad] PyTorch not available — VAD disabled")
            return
        try:
            self._model, _ = torch.hub.load(
                "snakers4/silero-vad", "silero_vad", trust_repo=True
            )
            self._model.eval()
            self._available = True
            print("[vad] Silero VAD loaded")
        except Exception as e:
            print(f"[vad] Failed to load Silero VAD: {e}")

    @property
    def available(self) -> bool:
        return self._available

    def is_speech(self, audio_16k: np.ndarray) -> bool:
        """Check if audio chunk contains speech. Expects float32 mono at 16kHz."""
        if not self._available:
            return True  # Pass-through if VAD unavailable

        with torch.no_grad():
            tensor = torch.from_numpy(audio_16k).float()
            prob = self._model(tensor, self.SAMPLE_RATE).item()
        return prob >= self.threshold

    def reset(self):
        """Reset VAD state between segments."""
        if self._available and hasattr(self._model, "reset_states"):
            self._model.reset_states()


class DeepFilterNetProcessor:
    """Real-time noise suppression using DeepFilterNet."""

    SAMPLE_RATE = 48000
    # DeepFilterNet processes in hop-size chunks (480 samples = 10ms at 48kHz)
    HOP_SIZE = 480
    # Process in blocks of this many hops for efficiency
    BLOCK_HOPS = 10  # 100ms blocks

    def __init__(self):
        self._model = None
        self._df_state = None
        self._available = False
        self._load_model()

    def _load_model(self):
        try:
            from df.enhance import init_df

            self._model, self._df_state, _ = init_df()
            self._available = True
            print("[audio] DeepFilterNet loaded (48kHz full-band)")
        except ImportError:
            print("[audio] deepfilternet not installed — noise suppression disabled")
            print("[audio] Install: pip install deepfilternet")
        except Exception as e:
            print(f"[audio] DeepFilterNet init failed: {e}")

    @property
    def available(self) -> bool:
        return self._available

    def process(self, audio: np.ndarray) -> np.ndarray:
        """Process a block of float32 mono audio at 48kHz. Returns denoised audio."""
        if not self._available:
            return audio

        try:
            from df.enhance import enhance
            import torch

            # DeepFilterNet expects [channels, samples] tensor
            tensor = torch.from_numpy(audio).float().unsqueeze(0)
            enhanced = enhance(self._model, self._df_state, tensor)
            return enhanced.squeeze(0).numpy()
        except Exception as e:
            print(f"[audio] DeepFilterNet process error: {e}")
            return audio


class AudioPipeline:
    """Captures mic input, applies VAD-gated noise suppression, outputs to virtual sink.

    Architecture:
        Mic → Silero VAD (16kHz check) → DeepFilterNet (48kHz denoise) → Virtual Mic

    When VAD detects no speech, the pipeline can either:
      - Pass through silence (saves CPU)
      - Still process (catches partial words)

    Cross-platform:
      - Linux: sounddevice + PulseAudio/PipeWire virtual sink
      - Windows: sounddevice + WASAPI, virtual cable (VB-Audio/etc.)
    """

    SAMPLE_RATE = 48000
    CHANNELS = 1
    # Process 100ms blocks for a good balance of latency and efficiency
    BLOCK_SIZE = 4800  # 100ms at 48kHz

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._dfn = DeepFilterNetProcessor()
        self._vad = SileroVAD(threshold=config.vad_threshold)
        self._stream = None
        self._running = False
        self._speech_active = False
        self._silence_frames = 0
        # Hold speech state for a few frames after VAD says silence (avoid choppy cuts)
        self._holdover_frames = 3  # ~300ms holdover

    def start(self):
        if not HAS_SD:
            print("[audio] sounddevice not available — install: pip install sounddevice")
            return

        if not self.config.noise_suppression:
            print("[audio] Noise suppression disabled in config")
            return

        if not self._dfn.available:
            print("[audio] No noise suppression backend available")
            return

        self._running = True

        def callback(indata, outdata, frames, time_info, status):
            if status:
                print(f"[audio] {status}")

            audio = indata[:, 0].copy()

            # VAD check — downsample to 16kHz for Silero
            if self._vad.available and self.config.vad_enabled:
                audio_16k = self._resample(audio, self.SAMPLE_RATE, SileroVAD.SAMPLE_RATE)
                is_speech = self._vad.is_speech(audio_16k)

                if is_speech:
                    self._speech_active = True
                    self._silence_frames = 0
                else:
                    self._silence_frames += 1
                    if self._silence_frames > self._holdover_frames:
                        self._speech_active = False

                # If no speech, output near-silence (not zero — avoids click artifacts)
                if not self._speech_active:
                    outdata[:, 0] = audio * 0.01  # -40dB gate
                    return

            # Apply DeepFilterNet noise suppression
            processed = self._dfn.process(audio)
            outdata[:, 0] = processed

        try:
            self._stream = sd.Stream(
                samplerate=self.SAMPLE_RATE,
                blocksize=self.BLOCK_SIZE,
                channels=self.CHANNELS,
                dtype="float32",
                callback=callback,
                device=(self.config.input_device, self.config.output_device),
            )
            self._stream.start()
            backend = "DeepFilterNet"
            if self._vad.available and self.config.vad_enabled:
                backend += " + Silero VAD"
            print(f"[audio] Noise suppression active ({backend})")
            print(f"[audio] Platform: {platform.system()}")
        except Exception as e:
            print(f"[audio] Failed to start: {e}")
            self._suggest_platform_fix(e)

    def stop(self):
        self._running = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
        if self._vad.available:
            self._vad.reset()

    def update_config(self, config: PipelineConfig):
        self.config = config
        self._vad.threshold = config.vad_threshold

    @staticmethod
    def _resample(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
        """Simple resampling via linear interpolation (good enough for VAD)."""
        if src_rate == dst_rate:
            return audio
        ratio = dst_rate / src_rate
        n_samples = int(len(audio) * ratio)
        indices = np.linspace(0, len(audio) - 1, n_samples)
        return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)

    @staticmethod
    def _suggest_platform_fix(error: Exception):
        """Print platform-specific troubleshooting hints."""
        system = platform.system()
        if system == "Linux":
            print("[audio] Hint: Make sure PulseAudio or PipeWire is running")
            print("[audio] For virtual mic: pactl load-module module-null-sink sink_name=OpenBroadcast")
        elif system == "Windows":
            print("[audio] Hint: Install VB-Audio Virtual Cable for virtual mic output")
            print("[audio] Download: https://vb-audio.com/Cable/")
        elif system == "Darwin":
            print("[audio] Hint: Install BlackHole for virtual audio on macOS")
            print("[audio] brew install blackhole-2ch")
