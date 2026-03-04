#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Realtime display backends used by Sports2D webcam/video preview."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Type

import cv2

DEFAULT_REALTIME_WINDOW_TITLE = "UmFit realtime"


class BaseRealtimeDisplay:
    """Minimal display interface used by the processing loop."""

    backend_name = "base"

    def render(self, frame, stats: Optional[Dict[str, Any]] = None) -> None:
        raise NotImplementedError

    def poll_events(self, delay_ms: int = 1) -> Dict[str, bool]:
        return {
            "stop": False,
            "toggle_pause": False,
            "resume": False,
            "toggle_fullscreen": False,
        }

    def set_session_state(self, state: str) -> None:
        _ = state

    def close(self) -> None:
        pass


class OpenCVRealtimeDisplay(BaseRealtimeDisplay):
    """OpenCV-backed realtime window implementation."""

    backend_name = "opencv"

    def __init__(
        self,
        window_title: str,
        display_width: Optional[int] = None,
        display_height: Optional[int] = None,
    ) -> None:
        self.window_title = window_title or DEFAULT_REALTIME_WINDOW_TITLE
        flags = cv2.WINDOW_NORMAL
        if hasattr(cv2, "WINDOW_KEEPRATIO"):
            flags |= cv2.WINDOW_KEEPRATIO
        cv2.namedWindow(self.window_title, flags)
        if display_width and display_height:
            cv2.resizeWindow(self.window_title, int(display_width), int(display_height))

    def render(self, frame, stats: Optional[Dict[str, Any]] = None) -> None:
        _ = stats
        cv2.imshow(self.window_title, frame)

    def poll_events(self, delay_ms: int = 1) -> Dict[str, bool]:
        key = cv2.waitKey(max(1, int(delay_ms))) & 0xFF
        events = {
            "stop": False,
            "toggle_pause": False,
            "resume": False,
            "toggle_fullscreen": False,
        }

        if key in (ord("q"), 27):
            events["stop"] = True
        elif key == ord(" "):
            events["toggle_pause"] = True
        elif key in (ord("r"), ord("s")):
            events["resume"] = True
        return events

    def close(self) -> None:
        try:
            cv2.destroyWindow(self.window_title)
        except cv2.error:
            cv2.destroyAllWindows()


def normalize_realtime_backend_name(backend: Optional[str]) -> str:
    if backend is None:
        return "opencv"
    normalized = str(backend).strip().lower()
    return normalized if normalized else "opencv"


def _load_qt_display_class() -> Type[BaseRealtimeDisplay]:
    from Sports2D.Utilities.realtime_qt import QtRealtimeDisplay

    return QtRealtimeDisplay


def create_realtime_display(
    backend: Optional[str],
    window_title: str,
    display_width: Optional[int] = None,
    display_height: Optional[int] = None,
    model_name: str = "",
    runtime_backend: str = "",
    webcam_id: Optional[int] = None,
    save_video: bool = False,
    frame_size: Optional[tuple[int, int]] = None,
) -> BaseRealtimeDisplay:
    """Create a realtime display backend with robust fallback behavior."""

    selected_backend = normalize_realtime_backend_name(backend)
    if selected_backend == "qt":
        try:
            qt_display_cls = _load_qt_display_class()
            return qt_display_cls(
                window_title=window_title,
                display_width=display_width,
                display_height=display_height,
                model_name=model_name,
                runtime_backend=runtime_backend,
                webcam_id=webcam_id,
                save_video=save_video,
                frame_size=frame_size,
            )
        except Exception as exc:
            logging.warning(
                "Qt realtime UI backend could not be initialized (%s). Falling back to OpenCV. "
                "Install with: pip install sports2d[ui]",
                exc,
            )
            selected_backend = "opencv"

    if selected_backend != "opencv":
        logging.warning(
            "Unknown realtime_ui_backend '%s'. Falling back to OpenCV.",
            selected_backend,
        )

    return OpenCVRealtimeDisplay(
        window_title=window_title,
        display_width=display_width,
        display_height=display_height,
    )
