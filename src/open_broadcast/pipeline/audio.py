"""Audio pipeline — noise suppression via RNNoise, virtual mic output."""

from __future__ import annotations

import ctypes
import ctypes.util
import threading
import struct
from typing import TYPE_CHECKING

import numpy as np

try:
    import sounddevice as sd
    HAS_SD = True
except ImportError:
    HAS_SD = False

if TYPE_CHECKING:
    from open_broadcast.pipeline.manager import PipelineConfig


class RNNoise:
    """Wrapper around librnnoise for real-time noise suppression."""

    FRAME_SIZE = 480  # RNNoise expects 480 samples (10ms at 48kHz)

    def __init__(self):
        self._lib = None
        self._state = None
        self._load_library()

    def _load_library(self):
        """Try to load librnnoise."""
        lib_path = ctypes.util.find_library("rnnoise")
        if lib_path is None:
            # Try common paths
            for path in ["librnnoise.so", "librnnoise.so.0", "librnnoise.dylib"]:
                try:
                    self._lib = ctypes.CDLL(path)
                    break
                except OSError:
                    continue
        else:
            self._lib = ctypes.CDLL(lib_path)

        if self._lib is None:
            print("[audio] librnnoise not found — noise suppression disabled")
            print("[audio] Install: apt install librnnoise0 (Debian/Ubuntu)")
            return

        # Setup function signatures
        self._lib.rnnoise_create.restype = ctypes.c_void_p
        self._lib.rnnoise_create.argtypes = [ctypes.c_void_p]

        self._lib.rnnoise_destroy.restype = None
        self._lib.rnnoise_destroy.argtypes = [ctypes.c_void_p]

        self._lib.rnnoise_process_frame.restype = ctypes.c_float
        self._lib.rnnoise_process_frame.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
        ]

        self._state = self._lib.rnnoise_create(None)

    @property
    def available(self) -> bool:
        return self._lib is not None and self._state is not None

    def process(self, samples: np.ndarray) -> np.ndarray:
        """Process a frame of 480 float32 samples. Returns denoised samples."""
        if not self.available:
            return samples

        # RNNoise expects float values in [-32768, 32767] range
        scaled = (samples * 32767).astype(np.float32)
        in_buf = scaled.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        out_arr = np.zeros(self.FRAME_SIZE, dtype=np.float32)
        out_buf = out_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        self._lib.rnnoise_process_frame(self._state, out_buf, in_buf)

        return out_arr / 32767.0

    def close(self):
        if self._state and self._lib:
            self._lib.rnnoise_destroy(self._state)
            self._state = None


class AudioPipeline:
    """Captures mic input, applies noise suppression, outputs to virtual sink."""

    SAMPLE_RATE = 48000
    CHANNELS = 1
    FRAME_SIZE = RNNoise.FRAME_SIZE

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._rnnoise = RNNoise()
        self._stream = None
        self._running = False

    def start(self):
        if not HAS_SD:
            print("[audio] sounddevice not available")
            return

        if not self.config.noise_suppression:
            return

        self._running = True

        def callback(indata, outdata, frames, time_info, status):
            if status:
                print(f"[audio] {status}")

            audio = indata[:, 0].copy()

            # Process in FRAME_SIZE chunks
            processed = np.zeros_like(audio)
            for i in range(0, len(audio) - self.FRAME_SIZE + 1, self.FRAME_SIZE):
                chunk = audio[i:i + self.FRAME_SIZE]
                processed[i:i + self.FRAME_SIZE] = self._rnnoise.process(chunk)

            outdata[:, 0] = processed

        try:
            self._stream = sd.Stream(
                samplerate=self.SAMPLE_RATE,
                blocksize=self.FRAME_SIZE,
                channels=self.CHANNELS,
                dtype="float32",
                callback=callback,
                device=(self.config.input_device, self.config.output_device),
            )
            self._stream.start()
            print("[audio] Noise suppression active")
        except Exception as e:
            print(f"[audio] Failed to start: {e}")

    def stop(self):
        self._running = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
        self._rnnoise.close()

    def update_config(self, config: PipelineConfig):
        self.config = config
