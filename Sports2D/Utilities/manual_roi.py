"""Manual ROI selection and coordinate conversion helpers."""

from __future__ import annotations

import copy
import logging
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np


ROI = Tuple[int, int, int, int]


def get_manual_roi_screen_size(default: Tuple[int, int] = (1920, 1080)) -> Tuple[int, int]:
    """Best-effort screen size lookup for ROI dialogs."""
    try:
        import tkinter as tk
    except Exception:
        return default

    try:
        root = tk.Tk()
        root.withdraw()
        screen_width = int(root.winfo_screenwidth())
        screen_height = int(root.winfo_screenheight())
        root.destroy()
        if screen_width <= 0 or screen_height <= 0:
            return default
        return screen_width, screen_height
    except Exception:
        return default


def fit_frame_to_screen(frame: np.ndarray, screen_width: int, screen_height: int, margin: int = 120):
    """Resize a frame for ROI display while preserving aspect ratio."""
    frame_height, frame_width = frame.shape[:2]
    max_width = max(1, int(screen_width) - int(margin))
    max_height = max(1, int(screen_height) - int(margin))
    if frame_width <= max_width and frame_height <= max_height:
        return frame.copy(), 1.0

    width_ratio = max_width / float(frame_width)
    height_ratio = max_height / float(frame_height)
    scale = min(width_ratio, height_ratio)
    scaled_width = max(1, int(round(frame_width * scale)))
    scaled_height = max(1, int(round(frame_height * scale)))
    resized = cv2.resize(frame, (scaled_width, scaled_height), interpolation=cv2.INTER_AREA)
    return resized, float(scale)


def scale_selection_to_full_frame(selection: Sequence[float], scale: float) -> Optional[ROI]:
    """Convert a scaled-window xywh ROI selection back to full-frame xyxy coordinates."""
    roi_xyxy = xywh_to_xyxy(selection)
    if roi_xyxy is None:
        return None
    if scale is None or scale <= 0:
        scale = 1.0
    inv_scale = 1.0 / float(scale)
    x1, y1, x2, y2 = roi_xyxy
    return (
        int(round(x1 * inv_scale)),
        int(round(y1 * inv_scale)),
        int(round(x2 * inv_scale)),
        int(round(y2 * inv_scale)),
    )


def xywh_to_xyxy(selection: Sequence[float]) -> Optional[ROI]:
    """Convert an OpenCV-style xywh selection to xyxy."""
    if selection is None:
        return None
    arr = np.asarray(selection, dtype=np.float32).reshape(-1)
    if arr.size != 4 or not np.all(np.isfinite(arr)):
        return None
    x, y, width, height = arr.tolist()
    return (int(round(x)), int(round(y)), int(round(x + width)), int(round(y + height)))


def normalize_roi_xyxy(
    roi: Optional[Sequence[float]],
    frame_shape,
    padding_px: int = 0,
) -> Optional[ROI]:
    """Clamp an xyxy ROI to the frame bounds and reject empty selections."""
    if roi is None or frame_shape is None:
        return None

    arr = np.asarray(roi, dtype=np.float32).reshape(-1)
    if arr.size != 4 or not np.all(np.isfinite(arr)):
        return None

    frame_height, frame_width = frame_shape[:2]
    padding = max(0, int(padding_px))
    x1, y1, x2, y2 = [int(round(v)) for v in arr.tolist()]
    x1 -= padding
    y1 -= padding
    x2 += padding
    y2 += padding

    x1 = int(np.clip(x1, 0, frame_width))
    y1 = int(np.clip(y1, 0, frame_height))
    x2 = int(np.clip(x2, 0, frame_width))
    y2 = int(np.clip(y2, 0, frame_height))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def union_rois(*rois: Optional[Sequence[int]]) -> Optional[ROI]:
    """Return the smallest ROI covering every provided ROI."""
    valid = [tuple(int(v) for v in roi) for roi in rois if roi is not None]
    if len(valid) == 0:
        return None
    return (
        min(roi[0] for roi in valid),
        min(roi[1] for roi in valid),
        max(roi[2] for roi in valid),
        max(roi[3] for roi in valid),
    )


def expand_roi_with_context(
    roi: Optional[Sequence[int]],
    frame_shape,
    scale: float = 2.5,
    min_size: int = 128,
) -> Optional[ROI]:
    """Expand a tight ROI with extra context while staying inside frame bounds."""
    if roi is None or frame_shape is None:
        return None

    frame_height, frame_width = frame_shape[:2]
    x1, y1, x2, y2 = [float(v) for v in roi]
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0

    target_width = min(float(frame_width), max(width * float(scale), float(min_size)))
    target_height = min(float(frame_height), max(height * float(scale), float(min_size)))
    half_width = target_width / 2.0
    half_height = target_height / 2.0

    expanded = (
        int(round(center_x - half_width)),
        int(round(center_y - half_height)),
        int(round(center_x + half_width)),
        int(round(center_y + half_height)),
    )
    return normalize_roi_xyxy(expanded, frame_shape, padding_px=0)


def translate_roi_to_local(roi: Optional[Sequence[int]], parent_roi: Optional[Sequence[int]]) -> Optional[ROI]:
    """Translate a full-frame ROI into the local coordinates of its parent ROI."""
    if roi is None or parent_roi is None:
        return None
    x1, y1, x2, y2 = [int(v) for v in roi]
    parent_x1, parent_y1, _, _ = [int(v) for v in parent_roi]
    return (
        x1 - parent_x1,
        y1 - parent_y1,
        x2 - parent_x1,
        y2 - parent_y1,
    )


def crop_frame_to_roi(frame: np.ndarray, roi: Optional[Sequence[int]]) -> np.ndarray:
    """Crop a frame to the selected ROI."""
    if roi is None:
        return frame
    x1, y1, x2, y2 = [int(v) for v in roi]
    return frame[y1:y2, x1:x2]


def offset_keypoints_to_full_frame(keypoints, roi: Optional[Sequence[int]]):
    """Translate keypoints from ROI-local coordinates back to full-frame coordinates."""
    keypoints_arr = np.asarray(keypoints, dtype=np.float32)
    if keypoints_arr.size == 0 or roi is None:
        return keypoints_arr.copy()

    x1, y1, _, _ = [float(v) for v in roi]
    shifted = keypoints_arr.copy()
    valid_x = np.isfinite(shifted[..., 0])
    valid_y = np.isfinite(shifted[..., 1])
    shifted[..., 0][valid_x] += x1
    shifted[..., 1][valid_y] += y1
    return shifted


def offset_xyxy_boxes_to_full_frame(boxes, roi: Optional[Sequence[int]]) -> np.ndarray:
    """Translate xyxy boxes from ROI-local coordinates back to full-frame coordinates."""
    box_arr = np.asarray(boxes, dtype=np.float32)
    if box_arr.size == 0:
        return np.empty((0, 4), dtype=np.float32)
    if box_arr.ndim == 1:
        box_arr = box_arr.reshape(1, -1)
    if box_arr.shape[1] < 4:
        return np.empty((0, 4), dtype=np.float32)
    shifted = box_arr[:, :4].astype(np.float32, copy=True)
    if roi is None:
        return shifted
    x1, y1, _, _ = [float(v) for v in roi]
    shifted[:, [0, 2]] += x1
    shifted[:, [1, 3]] += y1
    return shifted


def boxes_center_inside_roi(boxes, roi: Optional[Sequence[int]]) -> np.ndarray:
    """Return a mask selecting boxes whose center lies inside the ROI."""
    box_arr = np.asarray(boxes, dtype=np.float32)
    if box_arr.size == 0:
        return np.zeros((0,), dtype=bool)
    if box_arr.ndim == 1:
        box_arr = box_arr.reshape(1, -1)
    if roi is None:
        return np.ones((len(box_arr),), dtype=bool)
    x1, y1, x2, y2 = [float(v) for v in roi]
    centers_x = (box_arr[:, 0] + box_arr[:, 2]) / 2.0
    centers_y = (box_arr[:, 1] + box_arr[:, 3]) / 2.0
    return (
        (centers_x >= x1)
        & (centers_x <= x2)
        & (centers_y >= y1)
        & (centers_y <= y2)
    )


def offset_detection_meta_to_full_frame(meta, roi: Optional[Sequence[int]]):
    """Translate detection metadata boxes from ROI-local coordinates to full-frame space."""
    meta_dict = dict(meta or {})
    if len(meta_dict) == 0:
        return meta_dict

    shifted = {}
    for key, value in meta_dict.items():
        if key in {"boxes", "person_boxes", "ball_boxes"}:
            shifted[key] = offset_xyxy_boxes_to_full_frame(value, roi)
        elif key == "sam3_ball_meta" and isinstance(value, dict):
            shifted[key] = offset_detection_meta_to_full_frame(value, roi)
        elif isinstance(value, np.ndarray):
            shifted[key] = value.copy()
        else:
            shifted[key] = copy.deepcopy(value)
    return shifted


def select_manual_rois(
    frame: np.ndarray,
    detect_ball: bool = False,
    padding_px: int = 0,
    window_prefix: str = "Manual ROI",
):
    """Collect static person and optional ball ROIs from the user."""
    display_frame = np.asarray(frame).copy()
    screen_width, screen_height = get_manual_roi_screen_size()
    person_display_frame, person_scale = fit_frame_to_screen(
        display_frame,
        screen_width=screen_width,
        screen_height=screen_height,
    )
    person_selection = cv2.selectROI(
        f"{window_prefix} - Select person ROI",
        person_display_frame,
        showCrosshair=False,
        fromCenter=False,
    )
    cv2.destroyWindow(f"{window_prefix} - Select person ROI")
    person_roi = normalize_roi_xyxy(
        scale_selection_to_full_frame(person_selection, person_scale),
        display_frame.shape,
        padding_px=padding_px,
    )
    if person_roi is None:
        logging.warning(
            "manual_roi=true but no valid person ROI was selected. Falling back to full-frame inference."
        )
        return {"person_roi": None, "ball_roi": None}

    ball_roi = None
    if detect_ball:
        ball_display = display_frame.copy()
        x1, y1, x2, y2 = person_roi
        cv2.rectangle(ball_display, (x1, y1), (x2, y2), (0, 165, 255), 2)
        ball_display_frame, ball_scale = fit_frame_to_screen(
            ball_display,
            screen_width=screen_width,
            screen_height=screen_height,
        )
        ball_selection = cv2.selectROI(
            f"{window_prefix} - Select ball ROI",
            ball_display_frame,
            showCrosshair=False,
            fromCenter=False,
        )
        cv2.destroyWindow(f"{window_prefix} - Select ball ROI")
        ball_roi = normalize_roi_xyxy(
            scale_selection_to_full_frame(ball_selection, ball_scale),
            display_frame.shape,
            padding_px=padding_px,
        )
        if ball_roi is None:
            logging.warning(
                "detect_ball=true but no valid ball ROI was selected. Reusing the person ROI for ball detection."
            )
            ball_roi = person_roi

    return {"person_roi": person_roi, "ball_roi": ball_roi}
