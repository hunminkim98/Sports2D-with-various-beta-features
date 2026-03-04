#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Qt realtime display backend for Sports2D."""

from __future__ import annotations

import sys
import time
from collections import deque
from typing import Any, Dict, Optional

import cv2

from PySide6.QtCore import Qt, QEventLoop
from PySide6.QtGui import QFont, QImage, QKeyEvent, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


STYLE = """
QWidget {
    background-color: #0B1220;
    color: #EAF0FF;
    font-family: "Segoe UI";
}
QLabel {
    background-color: transparent;
}
QFrame#appBar {
    background-color: #121B2E;
    border: 1px solid #2A3658;
    border-radius: 12px;
}
QFrame#statusBar {
    background-color: #121B2E;
    border: 1px solid #2A3658;
    border-radius: 10px;
}
QFrame#videoStage {
    background-color: #121B2E;
    border: 1px solid #2A3658;
    border-radius: 14px;
}
QLabel#badge {
    border-radius: 8px;
    padding: 4px 8px;
    color: #08131F;
    background-color: #19C3A6;
    font-weight: 700;
}
QLabel#videoHint {
    color: #9FB0D1;
    font-size: 12px;
}
QLabel#overlayText {
    color: #EAF0FF;
    font-size: 12px;
}
QLabel#metricLabel {
    color: #9FB0D1;
    font-size: 12px;
}
QLabel#metricValue {
    color: #EAF0FF;
    font-family: "Consolas";
    font-size: 13px;
    font-weight: 600;
}
QGroupBox {
    border: 1px solid #2A3658;
    border-radius: 12px;
    margin-top: 10px;
    background-color: #1A2540;
    font-weight: 600;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 2px 8px;
    color: #9FB0D1;
}
QPushButton {
    border-radius: 10px;
    border: 1px solid #2A3658;
    background-color: #233156;
    color: #EAF0FF;
    padding: 8px 10px;
    font-weight: 600;
}
QPushButton:hover {
    background-color: #2B3B66;
}
QPushButton#stopBtn {
    background-color: #5E2431;
    border-color: #7D3042;
}
QPushButton#stopBtn:hover {
    background-color: #723244;
}
QListWidget {
    background-color: #121B2E;
    border: 1px solid #2A3658;
    border-radius: 10px;
    padding: 4px;
    color: #9FB0D1;
    font-size: 11px;
}
"""


class VideoCanvas(QLabel):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._source_pixmap: Optional[QPixmap] = None
        self._fill_mode = False
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(640, 360)
        self.setStyleSheet("background-color: #0A101C; border-radius: 12px; border: 1px solid #2A3658;")

    def set_fill_mode(self, enabled: bool) -> None:
        self._fill_mode = enabled
        self._rescale()

    def set_source_pixmap(self, pixmap: QPixmap) -> None:
        self._source_pixmap = pixmap
        self._rescale()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._rescale()

    def _rescale(self) -> None:
        if self._source_pixmap is None or self.width() < 2 or self.height() < 2:
            return
        scaled = self._source_pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatioByExpanding if self._fill_mode else Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(scaled)


class UmFitRealtimeWindow(QMainWindow):
    def __init__(
        self,
        window_title: str,
        display_width: int,
        display_height: int,
        model_name: str,
        runtime_backend: str,
        webcam_id: Optional[int],
        save_video: bool,
        frame_size: Optional[tuple[int, int]],
    ) -> None:
        super().__init__()
        self._actions: deque[str] = deque()
        self._is_fullscreen = False
        self._session_state = "Initializing"
        self._event_counter = 0
        self._elapsed_started = time.perf_counter()
        self._normal_geometry = None

        self.setWindowTitle(window_title)
        self.setMinimumSize(1280, 720)
        self.resize(max(1280, int(display_width or 1440)), max(720, int(display_height or 900)))

        self._build_ui(model_name, runtime_backend, webcam_id, save_video, frame_size)
        self.setStyleSheet(STYLE)
        self._apply_state_style("Initializing")
        self.add_event("Session initialized")

    def _build_ui(
        self,
        model_name: str,
        runtime_backend: str,
        webcam_id: Optional[int],
        save_video: bool,
        frame_size: Optional[tuple[int, int]],
    ) -> None:
        root = QWidget(self)
        self.setCentralWidget(root)
        root.setObjectName("rootView")
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(10)
        self._root_layout = root_layout

        app_bar = QFrame()
        app_bar.setObjectName("appBar")
        app_bar.setFixedHeight(56)
        self.app_bar = app_bar
        app_bar_layout = QHBoxLayout(app_bar)
        app_bar_layout.setContentsMargins(14, 8, 14, 8)

        left_box = QHBoxLayout()
        brand = QLabel("UmFit")
        brand.setFont(QFont("Segoe UI", 16, QFont.Weight.DemiBold))
        badge = QLabel("Realtime")
        badge.setObjectName("badge")
        left_box.addWidget(brand)
        left_box.addSpacing(8)
        left_box.addWidget(badge)
        left_box.addStretch(1)

        self.state_label = QLabel("Initializing")
        self.state_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.state_label.setFont(QFont("Segoe UI", 13, QFont.Weight.DemiBold))

        self.headline_info_label = QLabel("FPS -- | Model -- | Backend -- | Cam --")
        self.headline_info_label.setObjectName("overlayText")
        self.headline_info_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        app_bar_layout.addLayout(left_box, 1)
        app_bar_layout.addWidget(self.state_label, 1)
        app_bar_layout.addWidget(self.headline_info_label, 1)

        body = QWidget()
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(12)
        self.body_container = body
        self.body_layout = body_layout

        left_stage = QFrame()
        left_stage.setObjectName("videoStage")
        self.left_stage = left_stage
        left_layout = QVBoxLayout(left_stage)
        left_layout.setContentsMargins(12, 12, 12, 12)
        left_layout.setSpacing(8)
        self.left_layout = left_layout

        overlay_top = QHBoxLayout()
        self.live_label = QLabel("LIVE")
        self.live_label.setObjectName("badge")
        self.persons_label = QLabel("Persons 0")
        self.persons_label.setObjectName("overlayText")
        self.persons_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        overlay_top.addWidget(self.live_label)
        overlay_top.addStretch(1)
        overlay_top.addWidget(self.persons_label)

        self.video_canvas = VideoCanvas()
        self.video_hint_label = QLabel("Q Stop | Space Pause | F11 Fullscreen | Esc Exit Fullscreen")
        self.video_hint_label.setObjectName("videoHint")

        left_layout.addLayout(overlay_top)
        left_layout.addWidget(self.video_canvas, 1)
        left_layout.addWidget(self.video_hint_label)

        right_panel = QWidget()
        self.right_panel = right_panel
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)

        control_group = QGroupBox("Session Controls")
        control_layout = QVBoxLayout(control_group)
        self.webcam_label = QLabel(f"Webcam ID: {webcam_id if webcam_id is not None else '-'}")
        self.webcam_label.setObjectName("metricLabel")
        self.model_label = QLabel(f"Pose Model: {model_name or '-'}")
        self.model_label.setObjectName("metricLabel")
        self.backend_label = QLabel(f"Runtime Backend: {runtime_backend or '-'}")
        self.backend_label.setObjectName("metricLabel")

        btn_row = QHBoxLayout()
        self.start_btn = QPushButton("Start/Resume")
        self.pause_btn = QPushButton("Pause")
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setObjectName("stopBtn")
        btn_row.addWidget(self.start_btn)
        btn_row.addWidget(self.pause_btn)
        btn_row.addWidget(self.stop_btn)

        control_layout.addWidget(self.webcam_label)
        control_layout.addWidget(self.model_label)
        control_layout.addWidget(self.backend_label)
        control_layout.addLayout(btn_row)

        metrics_group = QGroupBox("Runtime Metrics")
        metrics_layout = QGridLayout(metrics_group)
        self.metric_ui_fps = self._metric_value("--")
        self.metric_inference = self._metric_value("--")
        self.metric_persons = self._metric_value("0")
        self.metric_dropped = self._metric_value("0")
        self._metric_row(metrics_layout, 0, "UI FPS", self.metric_ui_fps)
        self._metric_row(metrics_layout, 1, "Inference ms", self.metric_inference)
        self._metric_row(metrics_layout, 2, "Detected Persons", self.metric_persons)
        self._metric_row(metrics_layout, 3, "Dropped Frames", self.metric_dropped)

        events_group = QGroupBox("Events")
        events_layout = QVBoxLayout(events_group)
        self.events_list = QListWidget()
        self.events_list.setMinimumHeight(72)
        self.events_list.setMaximumHeight(96)
        events_layout.addWidget(self.events_list)

        right_layout.addWidget(control_group)
        right_layout.addWidget(metrics_group)
        right_layout.addWidget(events_group)
        right_layout.addStretch(1)

        body_layout.addWidget(left_stage, 72)
        body_layout.addWidget(right_panel, 28)

        status_bar = QFrame()
        status_bar.setObjectName("statusBar")
        status_bar.setFixedHeight(28)
        self.status_bar = status_bar
        status_layout = QHBoxLayout(status_bar)
        status_layout.setContentsMargins(12, 2, 12, 2)
        status_layout.setSpacing(8)

        frame_size_text = "-"
        if frame_size is not None and len(frame_size) == 2:
            frame_size_text = f"{frame_size[0]}x{frame_size[1]}"

        self.source_status_label = QLabel(f"Source: webcam | Resolution: {frame_size_text}")
        self.source_status_label.setObjectName("overlayText")
        self.save_status_label = QLabel("Saving video" if save_video else "Not saving")
        self.save_status_label.setObjectName("overlayText")
        self.save_status_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        status_layout.addWidget(self.source_status_label, 1)
        status_layout.addWidget(self.save_status_label, 1)

        root_layout.addWidget(app_bar)
        root_layout.addWidget(body, 1)
        root_layout.addWidget(status_bar)

        self.start_btn.clicked.connect(lambda: self._emit_action("resume"))
        self.pause_btn.clicked.connect(lambda: self._emit_action("toggle_pause"))
        self.stop_btn.clicked.connect(lambda: self._emit_action("stop"))

    def _metric_row(self, layout: QGridLayout, row: int, label: str, value: QLabel) -> None:
        label_widget = QLabel(label)
        label_widget.setObjectName("metricLabel")
        layout.addWidget(label_widget, row, 0)
        layout.addWidget(value, row, 1)

    def _metric_value(self, value: str) -> QLabel:
        label = QLabel(value)
        label.setObjectName("metricValue")
        return label

    def _emit_action(self, action: str) -> None:
        self._actions.append(action)

    def pop_actions(self) -> list[str]:
        actions = list(self._actions)
        self._actions.clear()
        return actions

    def add_event(self, message: str) -> None:
        self._event_counter += 1
        timestamp = time.strftime("%H:%M:%S")
        self.events_list.insertItem(0, f"[{timestamp}] {message}")
        while self.events_list.count() > 3:
            self.events_list.takeItem(self.events_list.count() - 1)

    def set_video_frame(self, frame_bgr) -> None:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        height, width, _ = rgb.shape
        image = QImage(rgb.data, width, height, rgb.strides[0], QImage.Format.Format_RGB888).copy()
        self.video_canvas.set_source_pixmap(QPixmap.fromImage(image))

    def _apply_state_style(self, state: str) -> None:
        normalized = state.strip().lower()
        if normalized == "live":
            self.live_label.setStyleSheet("background-color: #19C3A6; color: #08131F; border-radius: 8px; padding: 4px 8px; font-weight: 700;")
        elif normalized == "paused":
            self.live_label.setStyleSheet("background-color: #F4B740; color: #08131F; border-radius: 8px; padding: 4px 8px; font-weight: 700;")
        elif normalized in {"camera lost", "error"}:
            self.live_label.setStyleSheet("background-color: #FF5D5D; color: #08131F; border-radius: 8px; padding: 4px 8px; font-weight: 700;")
        else:
            self.live_label.setStyleSheet("background-color: #9FB0D1; color: #08131F; border-radius: 8px; padding: 4px 8px; font-weight: 700;")

    def set_session_state(self, state: str) -> None:
        if not state:
            return
        if state != self._session_state:
            self._session_state = state
            self.add_event(f"State changed: {state}")
        self.state_label.setText(state)
        self.live_label.setText(state.upper())
        self._apply_state_style(state)

    def update_stats(self, stats: Dict[str, Any]) -> None:
        if not stats:
            return

        state = stats.get("state")
        if isinstance(state, str):
            self.set_session_state(state)

        persons = int(stats.get("detected_persons", 0))
        self.persons_label.setText(f"Persons {persons}")
        self.metric_persons.setText(str(persons))

        dropped_frames = int(stats.get("dropped_frames", 0))
        self.metric_dropped.setText(str(dropped_frames))

        ui_fps = stats.get("ui_fps")
        if ui_fps is not None:
            self.metric_ui_fps.setText(f"{float(ui_fps):.1f}")

        inference_ms = stats.get("inference_ms")
        if inference_ms is not None:
            self.metric_inference.setText(f"{float(inference_ms):.1f}")

        model_name = stats.get("model")
        if model_name:
            self.model_label.setText(f"Pose Model: {model_name}")

        backend = stats.get("backend")
        if backend:
            self.backend_label.setText(f"Runtime Backend: {backend}")

        webcam_id = stats.get("webcam_id")
        if webcam_id is not None:
            self.webcam_label.setText(f"Webcam ID: {webcam_id}")

        if "save_status" in stats:
            self.save_status_label.setText(str(stats["save_status"]))

        if "camera_resolution" in stats:
            self.source_status_label.setText(f"Source: webcam | Resolution: {stats['camera_resolution']}")

        elapsed_seconds = stats.get("elapsed_seconds")
        elapsed_text = "--"
        if elapsed_seconds is not None:
            elapsed_text = f"{int(elapsed_seconds // 60):02d}:{int(elapsed_seconds % 60):02d}"

        headline_text = (
            f"FPS {self.metric_ui_fps.text()} | "
            f"Model {model_name or '-'} | "
            f"Backend {backend or '-'} | "
            f"Cam {webcam_id if webcam_id is not None else '-'} | "
            f"Time {elapsed_text}"
        )
        self.headline_info_label.setText(headline_text)

    def _set_fullscreen(self, enabled: bool) -> None:
        if enabled == self._is_fullscreen:
            return

        if enabled:
            self._normal_geometry = self.geometry()
            self._root_layout.setContentsMargins(0, 0, 0, 0)
            self._root_layout.setSpacing(0)
            self.body_layout.setContentsMargins(0, 0, 0, 0)
            self.body_layout.setSpacing(0)
            self.left_layout.setContentsMargins(0, 0, 0, 0)
            self.left_layout.setSpacing(0)

            self.app_bar.hide()
            self.right_panel.hide()
            self.status_bar.hide()
            self.live_label.hide()
            self.persons_label.hide()
            self.video_hint_label.hide()
            self.left_stage.setStyleSheet("background-color: #000000; border: none; border-radius: 0px;")
            self.video_canvas.setStyleSheet("background-color: #000000; border: none; border-radius: 0px;")
            self.video_canvas.set_fill_mode(True)
            self.showFullScreen()
            self.add_event("Entered fullscreen")
        else:
            self.showNormal()
            if self._normal_geometry is not None:
                self.setGeometry(self._normal_geometry)
            self._root_layout.setContentsMargins(12, 12, 12, 12)
            self._root_layout.setSpacing(10)
            self.body_layout.setContentsMargins(0, 0, 0, 0)
            self.body_layout.setSpacing(12)
            self.left_layout.setContentsMargins(12, 12, 12, 12)
            self.left_layout.setSpacing(8)

            self.app_bar.show()
            self.right_panel.show()
            self.status_bar.show()
            self.live_label.show()
            self.persons_label.show()
            self.video_hint_label.show()
            self.left_stage.setStyleSheet("")
            self.video_canvas.setStyleSheet("background-color: #0A101C; border-radius: 12px; border: 1px solid #2A3658;")
            self.video_canvas.set_fill_mode(False)
            self.add_event("Exited fullscreen")

        self._is_fullscreen = enabled

    def keyPressEvent(self, event: QKeyEvent) -> None:
        key = event.key()
        if key == Qt.Key.Key_Escape:
            if self._is_fullscreen:
                self._set_fullscreen(False)
            event.accept()
            return
        if key == Qt.Key.Key_Q:
            self._emit_action("stop")
            event.accept()
            return
        if key == Qt.Key.Key_Space:
            self._emit_action("toggle_pause")
            event.accept()
            return
        if key == Qt.Key.Key_F11:
            self._set_fullscreen(not self._is_fullscreen)
            event.accept()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event) -> None:
        self._emit_action("stop")
        event.accept()


class QtRealtimeDisplay:
    backend_name = "qt"

    def __init__(
        self,
        window_title: str,
        display_width: Optional[int],
        display_height: Optional[int],
        model_name: str,
        runtime_backend: str,
        webcam_id: Optional[int],
        save_video: bool,
        frame_size: Optional[tuple[int, int]],
    ) -> None:
        self.app = QApplication.instance() or QApplication(sys.argv[:1])
        self.window = UmFitRealtimeWindow(
            window_title=window_title,
            display_width=display_width or 1440,
            display_height=display_height or 900,
            model_name=model_name,
            runtime_backend=runtime_backend,
            webcam_id=webcam_id,
            save_video=save_video,
            frame_size=frame_size,
        )
        self.window.show()
        self._closed = False
        self.poll_events(delay_ms=1)

    def render(self, frame, stats: Optional[Dict[str, Any]] = None) -> None:
        if self._closed:
            return
        self.window.set_video_frame(frame)
        if stats:
            self.window.update_stats(stats)

    def poll_events(self, delay_ms: int = 1) -> Dict[str, bool]:
        if self._closed:
            return {
                "stop": True,
                "toggle_pause": False,
                "resume": False,
                "toggle_fullscreen": False,
            }

        deadline = time.perf_counter() + max(0, int(delay_ms)) / 1000.0
        while time.perf_counter() < deadline:
            self.app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 5)
        self.app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 1)

        events = {
            "stop": False,
            "toggle_pause": False,
            "resume": False,
            "toggle_fullscreen": False,
        }
        for action in self.window.pop_actions():
            if action in events:
                events[action] = True
        return events

    def set_session_state(self, state: str) -> None:
        if not self._closed:
            self.window.set_session_state(state)

    def close(self) -> None:
        if self._closed:
            return
        self.window.close()
        self.app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 10)
        self._closed = True
