"""Video pipeline — webcam capture, segmentation, auto-frame, virtual camera output."""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

import cv2
import numpy as np
import mediapipe as mp

if TYPE_CHECKING:
    from open_broadcast.pipeline.manager import PipelineConfig

try:
    import pyvirtualcam
    HAS_VCAM = True
except ImportError:
    HAS_VCAM = False


class VideoPipeline:
    """Processes webcam frames: segmentation + auto-frame → virtual camera."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        # MediaPipe models (lazy init)
        self._segmenter = None
        self._face_detector = None

        # Auto-frame state
        self._frame_center = None
        self._smoothed_center = None

    def start(self):
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)

    def update_config(self, config: PipelineConfig):
        self.config = config

    def _init_models(self):
        """Initialize MediaPipe models."""
        mp_selfie = mp.solutions.selfie_segmentation
        self._segmenter = mp_selfie.SelfieSegmentation(model_selection=1)

        if self.config.auto_frame:
            mp_face = mp.solutions.face_detection
            self._face_detector = mp_face.FaceDetection(
                model_selection=0, min_detection_confidence=0.7
            )

    def _run(self):
        """Main video processing loop."""
        self._init_models()

        cap = cv2.VideoCapture(self.config.camera_index)
        if not cap.isOpened():
            print(f"[video] Failed to open camera {self.config.camera_index}")
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

        vcam = None
        if HAS_VCAM:
            try:
                vcam = pyvirtualcam.Camera(width=width, height=height, fps=fps, fmt=pyvirtualcam.PixelFormat.BGR)
                print(f"[video] Virtual camera: {vcam.device}")
            except Exception as e:
                print(f"[video] Virtual camera unavailable: {e}")

        while not self._stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            # Background segmentation
            if self.config.background_mode != "off":
                frame = self._apply_background(frame)

            # Auto-frame
            if self.config.auto_frame:
                frame = self._apply_auto_frame(frame, width, height)

            # Output to virtual camera
            if vcam:
                vcam.send(frame)
                vcam.sleep_until_next_frame()

        cap.release()
        if vcam:
            vcam.close()
        if self._segmenter:
            self._segmenter.close()
        if self._face_detector:
            self._face_detector.close()

    def _apply_background(self, frame: np.ndarray) -> np.ndarray:
        """Apply background removal/blur using MediaPipe segmentation."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._segmenter.process(rgb)
        mask = result.segmentation_mask

        # Smooth mask edges
        mask = cv2.GaussianBlur(mask, (7, 7), 0)
        mask_3ch = np.stack([mask] * 3, axis=-1)

        if self.config.background_mode == "blur":
            bg = cv2.GaussianBlur(frame, (self.config.blur_strength * 2 + 1,) * 2, 0)
        elif self.config.background_mode == "remove":
            bg = np.zeros_like(frame)
        elif self.config.background_mode == "replace" and self.config.background_image:
            bg = cv2.imread(self.config.background_image)
            if bg is not None:
                bg = cv2.resize(bg, (frame.shape[1], frame.shape[0]))
            else:
                bg = np.zeros_like(frame)
        else:
            return frame

        output = (frame * mask_3ch + bg * (1 - mask_3ch)).astype(np.uint8)
        return output

    def _apply_auto_frame(self, frame: np.ndarray, width: int, height: int) -> np.ndarray:
        """Auto-frame: detect face and crop/zoom to center it."""
        if self._face_detector is None:
            return frame

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._face_detector.process(rgb)

        if result.detections:
            det = result.detections[0]
            bbox = det.location_data.relative_bounding_box
            cx = bbox.xmin + bbox.width / 2
            cy = bbox.ymin + bbox.height / 2

            target = np.array([cx, cy])
            if self._smoothed_center is None:
                self._smoothed_center = target
            else:
                s = self.config.auto_frame_smoothing
                self._smoothed_center = s * self._smoothed_center + (1 - s) * target

            # Crop around smoothed center
            zoom = self.config.auto_frame_zoom
            crop_w = int(width / zoom)
            crop_h = int(height / zoom)

            cx_px = int(self._smoothed_center[0] * width)
            cy_px = int(self._smoothed_center[1] * height)

            x1 = max(0, min(cx_px - crop_w // 2, width - crop_w))
            y1 = max(0, min(cy_px - crop_h // 2, height - crop_h))

            cropped = frame[y1:y1 + crop_h, x1:x1 + crop_w]
            return cv2.resize(cropped, (width, height))

        return frame
