"""System tray application with controls."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtWidgets import (
    QSystemTrayIcon, QMenu, QWidget, QVBoxLayout, QHBoxLayout,
    QCheckBox, QComboBox, QSlider, QLabel, QPushButton, QGroupBox,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon, QAction

if TYPE_CHECKING:
    from open_broadcast.pipeline.manager import PipelineManager


class ControlPanel(QWidget):
    """Main control panel window."""

    def __init__(self, manager: PipelineManager):
        super().__init__()
        self.manager = manager
        self.setWindowTitle("Open Broadcast")
        self.setFixedWidth(380)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # --- Video section ---
        video_group = QGroupBox("Video")
        video_layout = QVBoxLayout(video_group)

        # Background mode
        bg_row = QHBoxLayout()
        bg_row.addWidget(QLabel("Background:"))
        self.bg_combo = QComboBox()
        self.bg_combo.addItems(["Blur", "Remove", "Replace", "Off"])
        self.bg_combo.currentTextChanged.connect(self._on_bg_changed)
        bg_row.addWidget(self.bg_combo)
        video_layout.addLayout(bg_row)

        # Blur strength
        blur_row = QHBoxLayout()
        blur_row.addWidget(QLabel("Blur:"))
        self.blur_slider = QSlider(Qt.Orientation.Horizontal)
        self.blur_slider.setRange(5, 99)
        self.blur_slider.setSingleStep(2)
        self.blur_slider.setValue(21)
        self.blur_slider.valueChanged.connect(self._on_blur_changed)
        blur_row.addWidget(self.blur_slider)
        video_layout.addLayout(blur_row)

        # Auto-frame
        self.auto_frame_cb = QCheckBox("Auto Frame (face tracking)")
        self.auto_frame_cb.toggled.connect(
            lambda v: self.manager.update_config(auto_frame=v)
        )
        video_layout.addWidget(self.auto_frame_cb)

        layout.addWidget(video_group)

        # --- Audio section ---
        audio_group = QGroupBox("Audio")
        audio_layout = QVBoxLayout(audio_group)

        self.noise_cb = QCheckBox("Noise Suppression (RNNoise)")
        self.noise_cb.setChecked(True)
        self.noise_cb.toggled.connect(
            lambda v: self.manager.update_config(noise_suppression=v)
        )
        audio_layout.addWidget(self.noise_cb)

        layout.addWidget(audio_group)

        # --- Controls ---
        btn_row = QHBoxLayout()
        self.start_btn = QPushButton("▶ Start")
        self.start_btn.clicked.connect(self._toggle)
        btn_row.addWidget(self.start_btn)
        layout.addLayout(btn_row)

    def _on_bg_changed(self, text: str):
        self.manager.update_config(background_mode=text.lower())

    def _on_blur_changed(self, value: int):
        # Ensure odd value for GaussianBlur
        value = value if value % 2 == 1 else value + 1
        self.manager.update_config(blur_strength=value)

    def _toggle(self):
        if self.manager.running:
            self.manager.stop()
            self.start_btn.setText("▶ Start")
        else:
            self.manager.start()
            self.start_btn.setText("⏹ Stop")


class SystemTrayApp(QSystemTrayIcon):
    """System tray icon with quick toggles."""

    def __init__(self, manager: PipelineManager):
        super().__init__()
        self.manager = manager
        self.panel = ControlPanel(manager)

        self.setToolTip("Open Broadcast")
        self._build_menu()
        self.activated.connect(self._on_activated)

    def _build_menu(self):
        menu = QMenu()

        show_action = QAction("Show Controls", menu)
        show_action.triggered.connect(self.panel.show)
        menu.addAction(show_action)

        menu.addSeparator()

        quit_action = QAction("Quit", menu)
        quit_action.triggered.connect(self._quit)
        menu.addAction(quit_action)

        self.setContextMenu(menu)

    def _on_activated(self, reason):
        if reason == QSystemTrayIcon.ActivationReason.Trigger:
            self.panel.show()
            self.panel.raise_()

    def _quit(self):
        self.manager.stop()
        from PySide6.QtWidgets import QApplication
        QApplication.quit()
