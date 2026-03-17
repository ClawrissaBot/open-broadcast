"""Pipeline manager — orchestrates video and audio processing."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from open_broadcast.pipeline.video import VideoPipeline
from open_broadcast.pipeline.audio import AudioPipeline


@dataclass
class PipelineConfig:
    # Video
    camera_index: int = 0
    background_mode: str = "blur"  # "blur", "remove", "replace", "off"
    blur_strength: int = 21
    background_image: str | None = None
    auto_frame: bool = False
    auto_frame_zoom: float = 1.4
    auto_frame_smoothing: float = 0.85

    # Audio
    noise_suppression: bool = True
    input_device: int | None = None
    output_device: int | None = None


class PipelineManager:
    """Manages video and audio processing pipelines."""

    def __init__(self):
        self.config = PipelineConfig()
        self._video: VideoPipeline | None = None
        self._audio: AudioPipeline | None = None
        self._running = False

    @property
    def running(self) -> bool:
        return self._running

    def start(self):
        if self._running:
            return

        self._video = VideoPipeline(self.config)
        self._audio = AudioPipeline(self.config)

        self._video.start()
        self._audio.start()
        self._running = True

    def stop(self):
        if not self._running:
            return

        if self._video:
            self._video.stop()
        if self._audio:
            self._audio.stop()
        self._running = False

    def update_config(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self.config, k):
                setattr(self.config, k, v)

        # Hot-reload pipelines if running
        if self._running:
            if self._video:
                self._video.update_config(self.config)
            if self._audio:
                self._audio.update_config(self.config)
