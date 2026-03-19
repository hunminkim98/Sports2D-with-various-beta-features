#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Qt hybrid review editors for manual pose and ball correction."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from Sports2D.Utilities.hybrid_editor import (
    DERIVED_KEYPOINT_NAMES,
    MIN_ZOOM_VIEW_SPAN_PX,
    ZOOM_STEP_FACTOR,
    VideoFrameNavigator,
    _normalize_pose_manual_mask,
    _find_neighbor_keypoint_position,
    _open_video_capture,
    _selected_track_review_state,
    _score_to_rgb,
    _status_color,
    augment_pose_arrays_with_derived_keypoints,
    build_ball_issue_list,
    build_pose_issue_list,
    refresh_pose_derived_keypoints,
)


try:  # pragma: no cover - optional dependency
    from PySide6.QtCore import QEventLoop, QRectF, Qt, QTimer, Signal
    from PySide6.QtGui import QColor, QImage, QKeyEvent, QPainter, QPen
    from PySide6.QtWidgets import (
        QApplication,
        QDialog,
        QFrame,
        QHBoxLayout,
        QLabel,
        QListWidget,
        QListWidgetItem,
        QPushButton,
        QSlider,
        QVBoxLayout,
        QWidget,
    )
    QT_AVAILABLE = True
    QT_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - optional dependency
    QEventLoop = None
    QRectF = None
    Qt = None
    QTimer = None
    Signal = None
    QColor = None
    QImage = None
    QKeyEvent = None
    QPainter = None
    QPen = None
    QApplication = None
    QDialog = None
    QFrame = None
    QHBoxLayout = None
    QLabel = None
    QListWidget = None
    QListWidgetItem = None
    QPushButton = None
    QSlider = None
    QVBoxLayout = None
    QWidget = None
    QT_AVAILABLE = False
    QT_IMPORT_ERROR = exc


EDITOR_STYLE = """
QWidget {
    background-color: #0B1220;
    color: #EAF0FF;
    font-family: "Segoe UI";
}
QFrame#videoStage {
    background-color: #121B2E;
    border: 1px solid #2A3658;
    border-radius: 14px;
}
QFrame#sidePanel {
    background-color: #121B2E;
    border: 1px solid #2A3658;
    border-radius: 14px;
}
QLabel#header {
    color: #EAF0FF;
    font-size: 15px;
    font-weight: 700;
}
QLabel#subtle {
    color: #9FB0D1;
    font-size: 12px;
}
QLabel#status {
    color: #EAF0FF;
    font-size: 12px;
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
QPushButton#primary {
    background-color: #1D6FD8;
    border-color: #2F80ED;
}
QPushButton#primary:hover {
    background-color: #237AE8;
}
QListWidget {
    background-color: #0F1728;
    border: 1px solid #2A3658;
    border-radius: 10px;
    padding: 4px;
    color: #9FB0D1;
    font-size: 11px;
}
QSlider::groove:horizontal {
    border: 1px solid #2A3658;
    height: 8px;
    background: #0F1728;
    border-radius: 4px;
}
QSlider::handle:horizontal {
    background: #2F80ED;
    border: 1px solid #5EA8FF;
    width: 16px;
    margin: -5px 0;
    border-radius: 8px;
}
"""

PoseReviewDialog = None
BallReviewDialog = None


def _require_qt():
    if not QT_AVAILABLE:  # pragma: no cover - environment specific
        raise ImportError(f"PySide6 is required for the Qt hybrid editor: {QT_IMPORT_ERROR}")


def _probe_video_fps(video_file_path) -> float:
    cap = cv2.VideoCapture(str(Path(video_file_path)))
    if not cap.isOpened():
        return 30.0
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    finally:
        cap.release()
    if not np.isfinite(fps) or fps <= 1.0:
        return 30.0
    return min(fps, 60.0)


def _qcolor_from_status(status: str) -> "QColor":
    return QColor(_status_color(status))


def _qcolor_from_score(score: Optional[float]) -> "QColor":
    r, g, b = _score_to_rgb(score)
    return QColor(int(round(r * 255)), int(round(g * 255)), int(round(b * 255)))


def _clamp_view_box(
    left: float,
    top: float,
    width: float,
    height: float,
    image_width: float,
    image_height: float,
) -> Tuple[float, float, float, float]:
    width = float(np.clip(width, 1e-6, max(image_width, 1e-6)))
    height = float(np.clip(height, 1e-6, max(image_height, 1e-6)))
    max_left = max(0.0, float(image_width) - width)
    max_top = max(0.0, float(image_height) - height)
    left = float(np.clip(left, 0.0, max_left))
    top = float(np.clip(top, 0.0, max_top))
    return left, top, width, height


def _translate_view_box(
    left: float,
    top: float,
    width: float,
    height: float,
    widget_dx: float,
    widget_dy: float,
    target_width: float,
    target_height: float,
    image_width: float,
    image_height: float,
) -> Tuple[float, float, float, float]:
    image_dx = float(widget_dx) * float(width) / max(float(target_width), 1e-9)
    image_dy = float(widget_dy) * float(height) / max(float(target_height), 1e-9)
    return _clamp_view_box(
        left=float(left) - image_dx,
        top=float(top) - image_dy,
        width=float(width),
        height=float(height),
        image_width=float(image_width),
        image_height=float(image_height),
    )


if QT_AVAILABLE:  # pragma: no branch

    class ReviewVideoCanvas(QWidget):
        imageClicked = Signal(float, float)
        rightClicked = Signal()

        def __init__(self, parent: Optional[QWidget] = None) -> None:
            super().__init__(parent)
            self._qimage: Optional[QImage] = None
            self._view_rect: Optional[QRectF] = None
            self._pan_active = False
            self._pan_anchor_widget: Optional[Tuple[float, float]] = None
            self._pan_anchor_view: Optional[QRectF] = None
            self.setMinimumSize(820, 460)
            self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            self.setMouseTracking(True)

        def set_frame(self, frame_bgr) -> None:
            rgb = cv2.cvtColor(np.asarray(frame_bgr), cv2.COLOR_BGR2RGB)
            height, width, _ = rgb.shape
            image = QImage(rgb.data, width, height, rgb.strides[0], QImage.Format.Format_RGB888).copy()
            reset_view = self._qimage is None or self._qimage.size() != image.size()
            self._qimage = image
            if reset_view:
                self._view_rect = QRectF(0.0, 0.0, float(width), float(height))
            self.update()

        def image_size(self) -> Tuple[float, float]:
            if self._qimage is None:
                return 0.0, 0.0
            return float(self._qimage.width()), float(self._qimage.height())

        def reset_zoom(self) -> None:
            width, height = self.image_size()
            if width > 0 and height > 0:
                self._view_rect = QRectF(0.0, 0.0, width, height)
                self.update()

        def paintEvent(self, _event) -> None:
            painter = QPainter(self)
            painter.fillRect(self.rect(), QColor("#08111F"))
            if self._qimage is None or self._view_rect is None:
                return

            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

            target_rect = self._target_rect()
            painter.drawImage(target_rect, self._qimage, self._view_rect)
            self._paint_overlay(painter, target_rect, self._view_rect)

        def mousePressEvent(self, event) -> None:
            if event.button() == Qt.MouseButton.LeftButton:
                image_xy = self._widget_to_image(event.position().x(), event.position().y())
                if image_xy is None:
                    return
                self.imageClicked.emit(image_xy[0], image_xy[1])
                event.accept()
                return
            if event.button() == Qt.MouseButton.RightButton:
                self.rightClicked.emit()
                event.accept()
                return
            if event.button() == Qt.MouseButton.MiddleButton and self._qimage is not None and self._view_rect is not None:
                self._pan_active = True
                self._pan_anchor_widget = (float(event.position().x()), float(event.position().y()))
                self._pan_anchor_view = QRectF(self._view_rect)
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
                event.accept()
                return
            return super().mousePressEvent(event)

        def mouseMoveEvent(self, event) -> None:
            if not self._pan_active or self._pan_anchor_widget is None or self._pan_anchor_view is None or self._qimage is None:
                return super().mouseMoveEvent(event)
            target_rect = self._target_rect()
            left, top, width, height = _translate_view_box(
                left=float(self._pan_anchor_view.left()),
                top=float(self._pan_anchor_view.top()),
                width=float(self._pan_anchor_view.width()),
                height=float(self._pan_anchor_view.height()),
                widget_dx=float(event.position().x()) - self._pan_anchor_widget[0],
                widget_dy=float(event.position().y()) - self._pan_anchor_widget[1],
                target_width=float(target_rect.width()),
                target_height=float(target_rect.height()),
                image_width=float(self._qimage.width()),
                image_height=float(self._qimage.height()),
            )
            self._view_rect = QRectF(left, top, width, height)
            self.update()
            event.accept()

        def mouseReleaseEvent(self, event) -> None:
            if event.button() == Qt.MouseButton.MiddleButton and self._pan_active:
                self._pan_active = False
                self._pan_anchor_widget = None
                self._pan_anchor_view = None
                self.unsetCursor()
                event.accept()
                return
            return super().mouseReleaseEvent(event)

        def wheelEvent(self, event) -> None:
            if self._qimage is None or self._view_rect is None:
                return super().wheelEvent(event)
            image_xy = self._widget_to_image(event.position().x(), event.position().y())
            if image_xy is None:
                return super().wheelEvent(event)
            delta = event.angleDelta().y()
            if delta == 0:
                return
            zoom_factor = ZOOM_STEP_FACTOR ** (-1 if delta > 0 else 1)
            self._zoom_to(image_xy[0], image_xy[1], zoom_factor)
            event.accept()

        def _target_rect(self) -> QRectF:
            if self._qimage is None:
                return QRectF()
            image_w = float(self._qimage.width())
            image_h = float(self._qimage.height())
            if image_w <= 0 or image_h <= 0:
                return QRectF()

            widget_w = max(float(self.width()), 1.0)
            widget_h = max(float(self.height()), 1.0)
            scale = min(widget_w / image_w, widget_h / image_h)
            draw_w = image_w * scale
            draw_h = image_h * scale
            offset_x = (widget_w - draw_w) * 0.5
            offset_y = (widget_h - draw_h) * 0.5
            return QRectF(offset_x, offset_y, draw_w, draw_h)

        def _widget_to_image(self, widget_x: float, widget_y: float) -> Optional[Tuple[float, float]]:
            if self._qimage is None or self._view_rect is None:
                return None
            target_rect = self._target_rect()
            if not target_rect.contains(widget_x, widget_y):
                return None

            rel_x = (widget_x - target_rect.left()) / max(target_rect.width(), 1e-9)
            rel_y = (widget_y - target_rect.top()) / max(target_rect.height(), 1e-9)
            image_x = self._view_rect.left() + rel_x * self._view_rect.width()
            image_y = self._view_rect.top() + rel_y * self._view_rect.height()
            return float(image_x), float(image_y)

        def _image_to_widget(self, image_x: float, image_y: float) -> Optional[Tuple[float, float]]:
            if self._qimage is None or self._view_rect is None:
                return None
            if not self._view_rect.contains(image_x, image_y):
                return None

            target_rect = self._target_rect()
            rel_x = (float(image_x) - self._view_rect.left()) / max(self._view_rect.width(), 1e-9)
            rel_y = (float(image_y) - self._view_rect.top()) / max(self._view_rect.height(), 1e-9)
            return (
                float(target_rect.left() + rel_x * target_rect.width()),
                float(target_rect.top() + rel_y * target_rect.height()),
            )

        def _zoom_to(self, image_x: float, image_y: float, zoom_factor: float) -> None:
            if self._qimage is None or self._view_rect is None:
                return
            image_w = float(self._qimage.width())
            image_h = float(self._qimage.height())
            current = self._view_rect
            min_w = min(MIN_ZOOM_VIEW_SPAN_PX, image_w)
            min_h = min(MIN_ZOOM_VIEW_SPAN_PX, image_h)

            target_w = float(np.clip(current.width() * zoom_factor, min_w, image_w))
            target_h = float(np.clip(current.height() * zoom_factor, min_h, image_h))

            rel_x = 0.5 if current.width() <= 0 else float(np.clip((image_x - current.left()) / current.width(), 0.0, 1.0))
            rel_y = 0.5 if current.height() <= 0 else float(np.clip((image_y - current.top()) / current.height(), 0.0, 1.0))

            left = image_x - rel_x * target_w
            top = image_y - rel_y * target_h
            left, top, target_w, target_h = _clamp_view_box(
                left=left,
                top=top,
                width=target_w,
                height=target_h,
                image_width=image_w,
                image_height=image_h,
            )
            self._view_rect = QRectF(left, top, target_w, target_h)
            self.update()

        def _paint_overlay(self, painter: QPainter, target_rect: QRectF, source_rect: QRectF) -> None:
            del painter, target_rect, source_rect


    class PoseVideoCanvas(ReviewVideoCanvas):
        def __init__(self, parent: Optional[QWidget] = None) -> None:
            super().__init__(parent)
            self._frame_x = np.array([], dtype=float)
            self._frame_y = np.array([], dtype=float)
            self._frame_scores = np.array([], dtype=float)
            self._issues = []
            self._issue_by_index = {}
            self._selected_index: Optional[int] = None
            self._keypoint_names: List[str] = []

        def set_pose_state(
            self,
            frame_x,
            frame_y,
            frame_scores,
            issues,
            selected_index: Optional[int],
            keypoint_names: Sequence[str],
        ) -> None:
            self._frame_x = np.asarray(frame_x, dtype=float)
            self._frame_y = np.asarray(frame_y, dtype=float)
            self._frame_scores = np.asarray(frame_scores, dtype=float)
            self._issues = list(issues or [])
            self._issue_by_index = {issue["index"]: issue for issue in self._issues}
            self._selected_index = None if selected_index is None else int(selected_index)
            self._keypoint_names = list(keypoint_names or [])
            self.update()

        def _paint_overlay(self, painter: QPainter, target_rect: QRectF, source_rect: QRectF) -> None:
            del target_rect, source_rect
            font = painter.font()
            font.setPointSize(8)
            painter.setFont(font)

            for kp_idx, keypoint_name in enumerate(self._keypoint_names):
                issue = self._issue_by_index.get(kp_idx)
                x_value = self._frame_x[kp_idx] if kp_idx < len(self._frame_x) else np.nan
                y_value = self._frame_y[kp_idx] if kp_idx < len(self._frame_y) else np.nan
                score = self._frame_scores[kp_idx] if kp_idx < len(self._frame_scores) else np.nan

                if issue is not None and issue["status"] == "missing":
                    ghost_xy = issue.get("ghost_xy")
                    if ghost_xy is None:
                        continue
                    widget_xy = self._image_to_widget(ghost_xy[0], ghost_xy[1])
                    if widget_xy is None:
                        continue
                    pen = QPen(_qcolor_from_status("missing"))
                    pen.setWidth(2)
                    painter.setPen(pen)
                    painter.drawLine(widget_xy[0] - 8, widget_xy[1] - 8, widget_xy[0] + 8, widget_xy[1] + 8)
                    painter.drawLine(widget_xy[0] - 8, widget_xy[1] + 8, widget_xy[0] + 8, widget_xy[1] - 8)
                    if self._selected_index == kp_idx:
                        painter.drawText(widget_xy[0] + 8, widget_xy[1] - 10, keypoint_name)
                    continue

                if not (np.isfinite(x_value) and np.isfinite(y_value)):
                    continue

                widget_xy = self._image_to_widget(x_value, y_value)
                if widget_xy is None:
                    continue

                if issue is not None and issue["status"] == "manually_edited":
                    fill_color = _qcolor_from_status("manually_edited")
                    edge_color = QColor("#0B132B")
                    radius = 6.5
                    edge_width = 2
                elif issue is not None and issue["status"] == "derived":
                    fill_color = _qcolor_from_status("derived")
                    edge_color = QColor("#2D3436")
                    radius = 5.0
                    edge_width = 1
                elif issue is not None and issue["status"] == "low_confidence":
                    fill_color = QColor(0, 0, 0, 0)
                    edge_color = _qcolor_from_status("low_confidence")
                    radius = 6.0
                    edge_width = 2
                else:
                    fill_color = _qcolor_from_score(score if np.isfinite(score) else None)
                    edge_color = QColor("#0B132B")
                    radius = 4.5
                    edge_width = 1

                pen = QPen(edge_color)
                pen.setWidth(edge_width)
                painter.setPen(pen)
                painter.setBrush(fill_color)
                painter.drawEllipse(QRectF(widget_xy[0] - radius, widget_xy[1] - radius, radius * 2.0, radius * 2.0))

                if self._selected_index == kp_idx:
                    highlight_pen = QPen(QColor("#00C2FF"))
                    highlight_pen.setWidth(2)
                    painter.setPen(highlight_pen)
                    painter.setBrush(Qt.BrushStyle.NoBrush)
                    painter.drawEllipse(QRectF(widget_xy[0] - radius - 7, widget_xy[1] - radius - 7, (radius + 7) * 2.0, (radius + 7) * 2.0))
                    painter.drawText(widget_xy[0] + 8, widget_xy[1] - 10, keypoint_name)


    class BallVideoCanvas(ReviewVideoCanvas):
        def __init__(self, parent: Optional[QWidget] = None) -> None:
            super().__init__(parent)
            self._boxes = np.empty((0, 4), dtype=float)
            self._center = None
            self._manual_override = False

        def set_ball_state(self, boxes, center, manual_override: bool) -> None:
            self._boxes = np.asarray(boxes, dtype=float).reshape(-1, 4) if len(np.asarray(boxes).reshape(-1)) > 0 else np.empty((0, 4), dtype=float)
            self._center = None if center is None else (float(center[0]), float(center[1]))
            self._manual_override = bool(manual_override)
            self.update()

        def _paint_overlay(self, painter: QPainter, target_rect: QRectF, source_rect: QRectF) -> None:
            del target_rect, source_rect
            box_pen = QPen(QColor("#F39C12"))
            box_pen.setWidth(2)
            painter.setPen(box_pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            for box in self._boxes:
                top_left = self._image_to_widget(float(box[0]), float(box[1]))
                bottom_right = self._image_to_widget(float(box[2]), float(box[3]))
                if top_left is None or bottom_right is None:
                    continue
                rect = QRectF(
                    min(top_left[0], bottom_right[0]),
                    min(top_left[1], bottom_right[1]),
                    abs(bottom_right[0] - top_left[0]),
                    abs(bottom_right[1] - top_left[1]),
                )
                painter.drawRect(rect)

            if self._center is None:
                return

            center_xy = self._image_to_widget(self._center[0], self._center[1])
            if center_xy is None:
                return
            edge = QColor("#F7F7F7")
            fill = _qcolor_from_status("manual_ball_override") if self._manual_override else QColor("#111111")
            painter.setPen(QPen(edge, 2))
            painter.setBrush(fill)
            painter.drawEllipse(QRectF(center_xy[0] - 7, center_xy[1] - 7, 14, 14))


    class HybridReviewDialogBase(QDialog):
        def __init__(
            self,
            video_file_path,
            frame_range,
            frame_count: int,
            window_title: str,
            title_text: str,
            instructions_text: str,
        ) -> None:
            super().__init__()
            self.video_file_path = video_file_path
            self.frame_range = frame_range
            self.frame_count = max(int(frame_count), 1)
            self.current_frame_idx = 0
            self.playing = False
            self.frame_navigator = VideoFrameNavigator(
                _open_video_capture(video_file_path),
                start_frame=int(frame_range[0]),
                cache_size=48,
                sequential_window=6,
            )
            self.video_fps = _probe_video_fps(video_file_path)

            self.setWindowTitle(window_title)
            self.resize(1520, 920)
            self.setStyleSheet(EDITOR_STYLE)

            root_layout = QHBoxLayout(self)
            root_layout.setContentsMargins(12, 12, 12, 12)
            root_layout.setSpacing(12)

            left_stage = QFrame()
            left_stage.setObjectName("videoStage")
            left_layout = QVBoxLayout(left_stage)
            left_layout.setContentsMargins(12, 12, 12, 12)
            left_layout.setSpacing(8)

            self.canvas = self._create_canvas()
            self.canvas.imageClicked.connect(self._on_canvas_clicked)
            self.canvas.rightClicked.connect(self._on_canvas_right_clicked)
            self.frame_label = QLabel("")
            self.frame_label.setObjectName("subtle")
            left_layout.addWidget(self.canvas, 1)
            left_layout.addWidget(self.frame_label)

            slider_row = QHBoxLayout()
            self.prev_button = QPushButton("Prev")
            self.play_button = QPushButton("Play")
            self.next_button = QPushButton("Next")
            self.reset_zoom_button = QPushButton("Reset Zoom")
            self.frame_slider = QSlider(Qt.Orientation.Horizontal)
            self.frame_slider.setMinimum(0)
            self.frame_slider.setMaximum(self.frame_count - 1)
            slider_row.addWidget(self.prev_button)
            slider_row.addWidget(self.play_button)
            slider_row.addWidget(self.next_button)
            slider_row.addWidget(self.reset_zoom_button)
            slider_row.addWidget(self.frame_slider, 1)
            left_layout.addLayout(slider_row)

            right_panel = QFrame()
            right_panel.setObjectName("sidePanel")
            right_layout = QVBoxLayout(right_panel)
            right_layout.setContentsMargins(12, 12, 12, 12)
            right_layout.setSpacing(8)

            title_label = QLabel(title_text)
            title_label.setObjectName("header")
            instructions_label = QLabel(instructions_text)
            instructions_label.setObjectName("subtle")
            instructions_label.setWordWrap(True)
            self.status_label = QLabel("")
            self.status_label.setObjectName("status")
            self.status_label.setWordWrap(True)
            self.issue_list = QListWidget()

            button_row = QHBoxLayout()
            self.hide_button = QPushButton("Hide")
            self.restore_button = QPushButton("Restore")
            self.ok_button = QPushButton("OK")
            self.ok_button.setObjectName("primary")
            button_row.addWidget(self.hide_button)
            button_row.addWidget(self.restore_button)
            button_row.addStretch(1)
            button_row.addWidget(self.ok_button)

            right_layout.addWidget(title_label)
            right_layout.addWidget(instructions_label)
            right_layout.addWidget(self.status_label)
            right_layout.addWidget(self.issue_list, 1)
            right_layout.addLayout(button_row)

            root_layout.addWidget(left_stage, 72)
            root_layout.addWidget(right_panel, 28)

            self.playback_timer = QTimer(self)
            self.playback_timer.setInterval(max(16, int(round(1000.0 / max(self.video_fps, 1.0)))))

            self.prev_button.clicked.connect(self._on_prev)
            self.play_button.clicked.connect(self._toggle_playback)
            self.next_button.clicked.connect(self._on_next)
            self.reset_zoom_button.clicked.connect(self.canvas.reset_zoom)
            self.hide_button.clicked.connect(self._on_hide)
            self.restore_button.clicked.connect(self._on_restore)
            self.ok_button.clicked.connect(self.accept)
            self.frame_slider.valueChanged.connect(self._on_slider_changed)
            self.issue_list.itemClicked.connect(self._on_issue_clicked)
            self.playback_timer.timeout.connect(self._advance_frame)

            self._render_current_frame()

        def _create_canvas(self) -> ReviewVideoCanvas:
            raise NotImplementedError

        def _update_canvas_for_frame(self, frame_bgr, frame_idx: int) -> None:
            raise NotImplementedError

        def _build_status_lines(self, frame_idx: int) -> List[str]:
            raise NotImplementedError

        def _build_issue_entries(self, frame_idx: int) -> List[dict]:
            raise NotImplementedError

        def _handle_canvas_click(self, image_x: float, image_y: float) -> None:
            raise NotImplementedError

        def _handle_issue_selection(self, issue_index) -> None:
            raise NotImplementedError

        def _handle_hide(self) -> None:
            raise NotImplementedError

        def _handle_restore(self) -> None:
            raise NotImplementedError

        def _handle_right_click(self) -> None:
            return

        def _render_current_frame(self) -> None:
            frame_bgr = self.frame_navigator.get_frame(self.current_frame_idx)
            self.canvas.set_frame(frame_bgr)
            self._update_canvas_for_frame(frame_bgr, self.current_frame_idx)
            self.status_label.setText("\n".join(self._build_status_lines(self.current_frame_idx)))
            self._populate_issue_list(self._build_issue_entries(self.current_frame_idx))
            actual_frame_idx = int(self.frame_range[0]) + int(self.current_frame_idx)
            self.frame_label.setText(
                f"Frame {self.current_frame_idx + 1}/{self.frame_count} | source frame {actual_frame_idx} | playback {self.video_fps:.1f} fps"
            )

        def _populate_issue_list(self, entries: Sequence[dict]) -> None:
            self.issue_list.blockSignals(True)
            self.issue_list.clear()
            for entry in entries:
                item = QListWidgetItem(entry["text"])
                item.setForeground(QColor(entry.get("color", "#EAF0FF")))
                item.setData(Qt.ItemDataRole.UserRole, entry.get("data"))
                self.issue_list.addItem(item)
            self.issue_list.blockSignals(False)

        def _set_playing(self, enabled: bool) -> None:
            enabled = bool(enabled and self.frame_count > 1)
            self.playing = enabled
            if enabled:
                self.playback_timer.start()
                self.play_button.setText("Pause")
            else:
                self.playback_timer.stop()
                self.play_button.setText("Play")

        def _toggle_playback(self) -> None:
            self._set_playing(not self.playing)

        def _advance_frame(self) -> None:
            if self.current_frame_idx >= self.frame_count - 1:
                self._set_playing(False)
                return
            self.frame_slider.setValue(self.current_frame_idx + 1)

        def _on_prev(self) -> None:
            self._set_playing(False)
            self.frame_slider.setValue(max(0, self.current_frame_idx - 1))

        def _on_next(self) -> None:
            self._set_playing(False)
            self.frame_slider.setValue(min(self.frame_count - 1, self.current_frame_idx + 1))

        def _on_slider_changed(self, value: int) -> None:
            self.current_frame_idx = int(value)
            self._render_current_frame()

        def _on_canvas_clicked(self, image_x: float, image_y: float) -> None:
            self._set_playing(False)
            self._handle_canvas_click(float(image_x), float(image_y))
            self._render_current_frame()

        def _on_canvas_right_clicked(self) -> None:
            self._set_playing(False)
            self._handle_right_click()
            self._render_current_frame()

        def _on_issue_clicked(self, item: QListWidgetItem) -> None:
            self._set_playing(False)
            self._handle_issue_selection(item.data(Qt.ItemDataRole.UserRole))
            self._render_current_frame()

        def _on_hide(self) -> None:
            self._set_playing(False)
            self._handle_hide()
            self._render_current_frame()

        def _on_restore(self) -> None:
            self._set_playing(False)
            self._handle_restore()
            self._render_current_frame()

        def keyPressEvent(self, event: QKeyEvent) -> None:
            key = event.key()
            if key == Qt.Key.Key_Space:
                self._toggle_playback()
                event.accept()
                return
            if key == Qt.Key.Key_Left:
                self._on_prev()
                event.accept()
                return
            if key == Qt.Key.Key_Right:
                self._on_next()
                event.accept()
                return
            if key in {Qt.Key.Key_Return, Qt.Key.Key_Enter}:
                self.accept()
                event.accept()
                return
            if key == Qt.Key.Key_Escape:
                self.reject()
                event.accept()
                return
            super().keyPressEvent(event)

        def closeEvent(self, event) -> None:
            self._set_playing(False)
            self.frame_navigator.close()
            event.accept()


    class PoseReviewDialog(HybridReviewDialogBase):
        def __init__(
            self,
            video_file_path,
            frame_range,
            person_x_raw,
            person_y_raw,
            person_scores_raw,
            keypoint_names: Sequence[str],
            keypoint_threshold: float,
            manual_mask=None,
            window_title: str = "Hybrid pose review",
        ) -> None:
            self.person_x_raw, self.person_y_raw, self.person_scores_raw, self.keypoint_names = augment_pose_arrays_with_derived_keypoints(
                person_x_raw,
                person_y_raw,
                person_scores_raw,
                keypoint_names,
            )
            self.original_x = self.person_x_raw.copy()
            self.original_y = self.person_y_raw.copy()
            self.original_scores = self.person_scores_raw.copy()
            self.keypoint_threshold = float(keypoint_threshold)
            self.manual_mask = _normalize_pose_manual_mask(manual_mask, self.person_scores_raw.shape)
            self.selected_keypoint_index: Optional[int] = None
            self.current_issues = []
            self.current_issue_by_index = {}

            instructions = (
                "Click an issue or keypoint to select it.\n"
                "Click in the video to move the selected keypoint.\n"
                "Wheel zoom, middle-drag pan, right-click deselect.\n"
                "Use Space to play/pause and arrow keys to step."
            )
            super().__init__(
                video_file_path=video_file_path,
                frame_range=frame_range,
                frame_count=len(self.person_x_raw),
                window_title=window_title,
                title_text="Hybrid Pose Review (Qt)",
                instructions_text=instructions,
            )

        def _create_canvas(self) -> PoseVideoCanvas:
            return PoseVideoCanvas()

        def _issues_for_frame(self, frame_idx: int):
            issues = build_pose_issue_list(
                self.person_x_raw[frame_idx],
                self.person_y_raw[frame_idx],
                self.person_scores_raw[frame_idx],
                keypoint_names=self.keypoint_names,
                keypoint_threshold=self.keypoint_threshold,
                manual_mask_frame=self.manual_mask[frame_idx],
                frame_index=frame_idx,
                full_x_series=self.person_x_raw,
                full_y_series=self.person_y_raw,
            )
            return issues, {issue["index"]: issue for issue in issues}

        def _update_canvas_for_frame(self, frame_bgr, frame_idx: int) -> None:
            del frame_bgr
            refreshed_x, refreshed_y, refreshed_scores = refresh_pose_derived_keypoints(
                self.person_x_raw,
                self.person_y_raw,
                self.person_scores_raw,
                self.keypoint_names,
            )
            self.person_x_raw[:] = refreshed_x
            self.person_y_raw[:] = refreshed_y
            self.person_scores_raw[:] = refreshed_scores
            self.current_issues, self.current_issue_by_index = self._issues_for_frame(frame_idx)
            self.canvas.set_pose_state(
                self.person_x_raw[frame_idx],
                self.person_y_raw[frame_idx],
                self.person_scores_raw[frame_idx],
                self.current_issues,
                self.selected_keypoint_index,
                self.keypoint_names,
            )

        def _build_status_lines(self, frame_idx: int) -> List[str]:
            selected_idx = self.selected_keypoint_index
            selected_name = self.keypoint_names[selected_idx] if selected_idx is not None else "None"
            selected_issue = self.current_issue_by_index.get(selected_idx) if selected_idx is not None else None
            selected_status = selected_issue["status"] if selected_issue is not None else "normal"
            flagged = ", ".join(issue["status"] for issue in self.current_issues[:4]) or "no issues"
            return [
                f"Selected keypoint: {selected_name} ({selected_status})",
                f"Flagged: {flagged}",
                f"Local frame index: {frame_idx}",
            ]

        def _build_issue_entries(self, frame_idx: int) -> List[dict]:
            del frame_idx
            if len(self.current_issues) == 0:
                return [{"text": "No flagged keypoints.", "color": "#9FB0D1", "data": None}]
            entries = []
            for issue in self.current_issues:
                score_text = ""
                if issue.get("score") is not None:
                    score_text = f" ({issue['score']:.2f})"
                entries.append(
                    {
                        "text": f"{issue['keypoint']}: {issue['status']}{score_text}",
                        "color": _status_color(issue["status"]),
                        "data": issue["index"],
                    }
                )
            return entries

        def _select_nearest_keypoint(self, frame_idx: int, x_click: float, y_click: float) -> Optional[int]:
            best_idx = None
            best_dist = 20.0
            frame_x = self.person_x_raw[frame_idx]
            frame_y = self.person_y_raw[frame_idx]
            for idx, (x_value, y_value) in enumerate(zip(frame_x, frame_y)):
                if not (np.isfinite(x_value) and np.isfinite(y_value)):
                    ghost_xy = _find_neighbor_keypoint_position(
                        frame_idx,
                        self.person_x_raw[:, idx],
                        self.person_y_raw[:, idx],
                    )
                    if ghost_xy is None:
                        continue
                    x_value, y_value = ghost_xy
                dist = float(np.hypot(x_value - x_click, y_value - y_click))
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx
            return best_idx

        def _handle_canvas_click(self, image_x: float, image_y: float) -> None:
            frame_idx = self.current_frame_idx
            selected_idx = self.selected_keypoint_index
            nearest_idx = self._select_nearest_keypoint(frame_idx, image_x, image_y)
            if nearest_idx is not None and (selected_idx is None or nearest_idx != selected_idx):
                self.selected_keypoint_index = int(nearest_idx)
                return

            if selected_idx is None:
                return
            if self.keypoint_names[selected_idx] in DERIVED_KEYPOINT_NAMES:
                return

            self.person_x_raw[frame_idx, selected_idx] = float(image_x)
            self.person_y_raw[frame_idx, selected_idx] = float(image_y)
            self.person_scores_raw[frame_idx, selected_idx] = max(1.0, self.keypoint_threshold)
            self.manual_mask[frame_idx, selected_idx] = True

        def _handle_issue_selection(self, issue_index) -> None:
            if issue_index is None:
                return
            self.selected_keypoint_index = int(issue_index)

        def _handle_right_click(self) -> None:
            self.selected_keypoint_index = None

        def _handle_hide(self) -> None:
            selected_idx = self.selected_keypoint_index
            if selected_idx is None or self.keypoint_names[selected_idx] in DERIVED_KEYPOINT_NAMES:
                return
            frame_idx = self.current_frame_idx
            self.person_x_raw[frame_idx, selected_idx] = np.nan
            self.person_y_raw[frame_idx, selected_idx] = np.nan
            self.person_scores_raw[frame_idx, selected_idx] = np.nan
            self.manual_mask[frame_idx, selected_idx] = True

        def _handle_restore(self) -> None:
            selected_idx = self.selected_keypoint_index
            if selected_idx is None or self.keypoint_names[selected_idx] in DERIVED_KEYPOINT_NAMES:
                return
            frame_idx = self.current_frame_idx
            self.person_x_raw[frame_idx, selected_idx] = self.original_x[frame_idx, selected_idx]
            self.person_y_raw[frame_idx, selected_idx] = self.original_y[frame_idx, selected_idx]
            self.person_scores_raw[frame_idx, selected_idx] = self.original_scores[frame_idx, selected_idx]
            self.manual_mask[frame_idx, selected_idx] = False

        def get_result(self):
            return self.person_x_raw, self.person_y_raw, self.person_scores_raw, self.manual_mask


    class BallReviewDialog(HybridReviewDialogBase):
        def __init__(
            self,
            video_file_path,
            frame_range,
            ball_centers,
            ball_boxes,
            ball_scores,
            ball_tracks,
            selected_ball_ids,
            score_threshold: float = 0.1,
            window_title: str = "Hybrid ball review",
        ) -> None:
            self.ball_centers = list(ball_centers)
            self.original_centers = [None if center is None else tuple(center) for center in ball_centers]
            self.ball_visible = [center is not None for center in ball_centers]
            self.original_visible = list(self.ball_visible)
            self.manual_override_mask = [False for _ in ball_centers]
            self.ball_boxes = list(ball_boxes)
            self.ball_scores = list(ball_scores)
            self.ball_tracks = list(ball_tracks)
            self.selected_ball_ids = list(selected_ball_ids)
            self.score_threshold = float(score_threshold)
            self.current_issues = []
            self.current_source_track_id = None

            instructions = (
                "Click in the video to place the ball center.\n"
                "Use Hide or Restore for the current frame.\n"
                "Wheel zoom, middle-drag pan.\n"
                "Use Space to play/pause and arrow keys to step."
            )
            super().__init__(
                video_file_path=video_file_path,
                frame_range=frame_range,
                frame_count=len(self.ball_centers),
                window_title=window_title,
                title_text="Hybrid Ball Review (Qt)",
                instructions_text=instructions,
            )

        def _create_canvas(self) -> BallVideoCanvas:
            return BallVideoCanvas()

        def _update_canvas_for_frame(self, frame_bgr, frame_idx: int) -> None:
            del frame_bgr
            frame_center = self.ball_centers[frame_idx]
            selected_id = self.selected_ball_ids[frame_idx] if frame_idx < len(self.selected_ball_ids) else None
            frame_tracks = self.ball_tracks[frame_idx] if frame_idx < len(self.ball_tracks) else []
            _, track_score, track_visible, source_track_id = _selected_track_review_state(
                frame_tracks,
                selected_id,
                frame_center=frame_center,
            )
            self.current_source_track_id = source_track_id
            self.current_issues = build_ball_issue_list(
                frame_center,
                score=track_score,
                score_threshold=self.score_threshold,
                manual_override=self.manual_override_mask[frame_idx],
                visible=self.ball_visible[frame_idx],
                track_missing=bool(selected_id is not None and not track_visible and frame_center is None),
            )
            boxes = self.ball_boxes[frame_idx] if frame_idx < len(self.ball_boxes) else np.empty((0, 4))
            self.canvas.set_ball_state(boxes, frame_center if self.ball_visible[frame_idx] else None, self.manual_override_mask[frame_idx])

        def _build_status_lines(self, frame_idx: int) -> List[str]:
            selected_id = self.selected_ball_ids[frame_idx] if frame_idx < len(self.selected_ball_ids) else None
            issue_text = ", ".join(issue["status"] for issue in self.current_issues) or "no issues"
            return [
                f"Selected track: {selected_id}",
                (
                    f"Visible source track: {self.current_source_track_id}"
                    if self.current_source_track_id is not None
                    and selected_id is not None
                    and int(self.current_source_track_id) != int(selected_id)
                    else (
                        "Visible source track: same as selected"
                        if self.current_source_track_id is not None and selected_id is not None
                        else "Visible source track: none"
                    )
                ),
                f"Flagged: {issue_text}",
                f"Local frame index: {frame_idx}",
            ]

        def _build_issue_entries(self, frame_idx: int) -> List[dict]:
            del frame_idx
            if len(self.current_issues) == 0:
                return [{"text": "No flagged ball issues.", "color": "#9FB0D1", "data": None}]
            entries = []
            for issue in self.current_issues:
                if issue["status"] == "low_confidence_ball":
                    text = f"low_confidence_ball ({issue['score']:.2f} < {issue['threshold']:.2f})"
                else:
                    text = issue["status"]
                entries.append({"text": text, "color": _status_color(issue["status"]), "data": None})
            return entries

        def _handle_canvas_click(self, image_x: float, image_y: float) -> None:
            frame_idx = self.current_frame_idx
            self.ball_centers[frame_idx] = (int(round(float(image_x))), int(round(float(image_y))))
            self.ball_visible[frame_idx] = True
            self.manual_override_mask[frame_idx] = True

        def _handle_issue_selection(self, issue_index) -> None:
            del issue_index

        def _handle_hide(self) -> None:
            frame_idx = self.current_frame_idx
            self.ball_centers[frame_idx] = None
            self.ball_visible[frame_idx] = False
            self.manual_override_mask[frame_idx] = True

        def _handle_restore(self) -> None:
            frame_idx = self.current_frame_idx
            self.ball_centers[frame_idx] = self.original_centers[frame_idx]
            self.ball_visible[frame_idx] = self.original_visible[frame_idx]
            self.manual_override_mask[frame_idx] = False

        def get_result(self):
            return self.ball_centers, self.ball_visible, self.manual_override_mask


def review_pose_sequence_qt(
    video_file_path,
    frame_range,
    person_x_raw,
    person_y_raw,
    person_scores_raw,
    keypoint_names: Sequence[str],
    keypoint_threshold: float,
    manual_mask=None,
    window_title: str = "Hybrid pose review",
):
    _require_qt()
    original_x = np.asarray(person_x_raw, dtype=float).copy()
    original_y = np.asarray(person_y_raw, dtype=float).copy()
    original_scores = np.asarray(person_scores_raw, dtype=float).copy()
    original_manual_mask = _normalize_pose_manual_mask(manual_mask, original_scores.shape)
    app = QApplication.instance() or QApplication(sys.argv[:1])
    dialog = PoseReviewDialog(
        video_file_path=video_file_path,
        frame_range=frame_range,
        person_x_raw=person_x_raw,
        person_y_raw=person_y_raw,
        person_scores_raw=person_scores_raw,
        keypoint_names=keypoint_names,
        keypoint_threshold=keypoint_threshold,
        manual_mask=manual_mask,
        window_title=window_title,
    )
    dialog.show()
    dialog.raise_()
    dialog.activateWindow()
    dialog_result = dialog.exec()
    app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 10)
    accepted_code = getattr(getattr(QDialog, "DialogCode", object), "Accepted", 1)
    if dialog_result != accepted_code:
        return original_x, original_y, original_scores, original_manual_mask
    return dialog.get_result()


def review_ball_sequence_qt(
    video_file_path,
    frame_range,
    ball_centers,
    ball_boxes,
    ball_scores,
    ball_tracks,
    selected_ball_ids,
    score_threshold: float = 0.1,
    window_title: str = "Hybrid ball review",
):
    _require_qt()
    original_centers = [None if center is None else tuple(center) for center in ball_centers]
    original_visible = [center is not None for center in original_centers]
    original_manual_override = [False for _ in original_centers]
    app = QApplication.instance() or QApplication(sys.argv[:1])
    dialog = BallReviewDialog(
        video_file_path=video_file_path,
        frame_range=frame_range,
        ball_centers=ball_centers,
        ball_boxes=ball_boxes,
        ball_scores=ball_scores,
        ball_tracks=ball_tracks,
        selected_ball_ids=selected_ball_ids,
        score_threshold=score_threshold,
        window_title=window_title,
    )
    dialog.show()
    dialog.raise_()
    dialog.activateWindow()
    dialog_result = dialog.exec()
    app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 10)
    accepted_code = getattr(getattr(QDialog, "DialogCode", object), "Accepted", 1)
    if dialog_result != accepted_code:
        return original_centers, original_visible, original_manual_override
    return dialog.get_result()
