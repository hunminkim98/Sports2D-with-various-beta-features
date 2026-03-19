#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Hybrid review helpers for manual pose and ball correction."""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


DERIVED_KEYPOINT_SEQUENCE = ("Hip", "Neck")
DERIVED_KEYPOINT_NAMES = set(DERIVED_KEYPOINT_SEQUENCE)
DERIVED_KEYPOINT_SOURCES = {
    "Hip": ("LHip", "RHip"),
    "Neck": ("LShoulder", "RShoulder"),
}
POSE_ISSUE_PRIORITY = {
    "missing": 0,
    "low_confidence": 1,
    "manually_edited": 2,
    "derived": 3,
}
BALL_ISSUE_PRIORITY = {
    "missing_ball": 0,
    "track_gap": 1,
    "low_confidence_ball": 2,
    "manual_ball_override": 3,
}
ZOOM_STEP_FACTOR = 1.2
MIN_ZOOM_VIEW_SPAN_PX = 24.0


def _rowwise_nanmean(values):
    values = np.asarray(values, dtype=float)
    finite_mask = np.isfinite(values)
    sums = np.where(finite_mask, values, 0.0).sum(axis=1)
    counts = finite_mask.sum(axis=1)
    result = np.full((values.shape[0],), np.nan, dtype=float)
    valid_rows = counts > 0
    result[valid_rows] = sums[valid_rows] / counts[valid_rows]
    return result


def refresh_pose_derived_keypoints(person_x_raw, person_y_raw, person_scores_raw, keypoint_names: Sequence[str]):
    """
    Recompute read-only derived markers such as Hip/Neck from their source keypoints.
    """

    keypoint_names = list(keypoint_names or [])
    person_x_raw = np.asarray(person_x_raw, dtype=float).copy()
    person_y_raw = np.asarray(person_y_raw, dtype=float).copy()
    person_scores_raw = np.asarray(person_scores_raw, dtype=float).copy()

    squeezed = person_x_raw.ndim == 1
    if squeezed:
        person_x_raw = person_x_raw.reshape(1, -1)
        person_y_raw = person_y_raw.reshape(1, -1)
        person_scores_raw = person_scores_raw.reshape(1, -1)

    for derived_name in DERIVED_KEYPOINT_SEQUENCE:
        if derived_name not in keypoint_names:
            continue

        derived_idx = keypoint_names.index(derived_name)
        source_names = DERIVED_KEYPOINT_SOURCES.get(derived_name, ())
        if not source_names or not all(source_name in keypoint_names for source_name in source_names):
            person_x_raw[:, derived_idx] = np.nan
            person_y_raw[:, derived_idx] = np.nan
            person_scores_raw[:, derived_idx] = np.nan
            continue

        source_indices = [keypoint_names.index(source_name) for source_name in source_names]
        person_x_raw[:, derived_idx] = _rowwise_nanmean(person_x_raw[:, source_indices])
        person_y_raw[:, derived_idx] = _rowwise_nanmean(person_y_raw[:, source_indices])
        person_scores_raw[:, derived_idx] = _rowwise_nanmean(person_scores_raw[:, source_indices])

    if squeezed:
        return person_x_raw.reshape(-1), person_y_raw.reshape(-1), person_scores_raw.reshape(-1)
    return person_x_raw, person_y_raw, person_scores_raw


def augment_pose_arrays_with_derived_keypoints(person_x_raw, person_y_raw, person_scores_raw, keypoint_names: Sequence[str]):
    """
    Append derived-marker columns if needed and populate them from the source markers.
    """

    keypoint_names = list(keypoint_names or [])
    person_x_raw = np.asarray(person_x_raw, dtype=float).copy()
    person_y_raw = np.asarray(person_y_raw, dtype=float).copy()
    person_scores_raw = np.asarray(person_scores_raw, dtype=float).copy()

    squeezed = person_x_raw.ndim == 1
    if squeezed:
        person_x_raw = person_x_raw.reshape(1, -1)
        person_y_raw = person_y_raw.reshape(1, -1)
        person_scores_raw = person_scores_raw.reshape(1, -1)

    for derived_name in DERIVED_KEYPOINT_SEQUENCE:
        if derived_name in keypoint_names:
            continue
        nan_column = np.full((person_x_raw.shape[0], 1), np.nan, dtype=float)
        person_x_raw = np.concatenate([person_x_raw, nan_column], axis=1)
        person_y_raw = np.concatenate([person_y_raw, nan_column.copy()], axis=1)
        person_scores_raw = np.concatenate([person_scores_raw, nan_column.copy()], axis=1)
        keypoint_names.append(derived_name)

    person_x_raw, person_y_raw, person_scores_raw = refresh_pose_derived_keypoints(
        person_x_raw,
        person_y_raw,
        person_scores_raw,
        keypoint_names,
    )
    if squeezed:
        return (
            person_x_raw.reshape(-1),
            person_y_raw.reshape(-1),
            person_scores_raw.reshape(-1),
            keypoint_names,
        )
    return person_x_raw, person_y_raw, person_scores_raw, keypoint_names


def _normalize_pose_manual_mask(manual_mask, target_shape):
    target_shape = tuple(int(dim) for dim in target_shape)
    if manual_mask is None:
        return np.zeros(target_shape, dtype=bool)

    manual_mask = np.asarray(manual_mask, dtype=bool)
    if manual_mask.shape == target_shape:
        return manual_mask.copy()

    normalized = np.zeros(target_shape, dtype=bool)
    common_rows = min(normalized.shape[0], manual_mask.shape[0] if manual_mask.ndim > 0 else 0)
    common_cols = min(normalized.shape[1], manual_mask.shape[1] if manual_mask.ndim > 1 else 0)
    if common_rows > 0 and common_cols > 0:
        normalized[:common_rows, :common_cols] = manual_mask[:common_rows, :common_cols]
    return normalized


def evaluate_pose_frame(
    raw_x,
    raw_y,
    raw_scores,
    keypoint_threshold: float,
    average_threshold: float,
    keypoint_number_threshold: float,
):
    """
    Apply Sports2D frame-level pose thresholding to raw keypoint arrays.

    Returns filtered x/y/scores plus a rejection reason when the person should be
    removed for the frame.
    """

    raw_x = np.asarray(raw_x, dtype=float).copy()
    raw_y = np.asarray(raw_y, dtype=float).copy()
    raw_scores = np.asarray(raw_scores, dtype=float).copy()

    invalid_mask = np.isnan(raw_scores) | (raw_scores < float(keypoint_threshold))
    filtered_x = np.where(invalid_mask, np.nan, raw_x)
    filtered_y = np.where(invalid_mask, np.nan, raw_y)
    filtered_scores = np.where(invalid_mask, np.nan, raw_scores)

    valid_scores = filtered_scores[~np.isnan(filtered_scores)]
    required_keypoints = len(filtered_scores) * float(keypoint_number_threshold)
    enough_good_keypoints = len(valid_scores) >= required_keypoints
    average_score = float(np.nanmean(valid_scores)) if len(valid_scores) > 0 else 0.0
    average_score_is_enough = average_score >= float(average_threshold)

    rejection_reason = None
    if not enough_good_keypoints:
        rejection_reason = "too_few_keypoints"
    elif not average_score_is_enough:
        rejection_reason = "low_average_confidence"

    if rejection_reason is not None:
        filtered_x[:] = np.nan
        filtered_y[:] = np.nan
        filtered_scores[:] = np.nan

    return filtered_x, filtered_y, filtered_scores, rejection_reason


def _find_neighbor_keypoint_position(frame_idx: int, x_series, y_series):
    """Return the nearest finite `(x, y)` value around `frame_idx` if one exists."""

    x_series = np.asarray(x_series, dtype=float).reshape(-1)
    y_series = np.asarray(y_series, dtype=float).reshape(-1)
    frame_count = min(len(x_series), len(y_series))
    for offset in range(1, frame_count):
        for candidate_idx in (frame_idx - offset, frame_idx + offset):
            if candidate_idx < 0 or candidate_idx >= frame_count:
                continue
            x_value = x_series[candidate_idx]
            y_value = y_series[candidate_idx]
            if np.isfinite(x_value) and np.isfinite(y_value):
                return float(x_value), float(y_value)
    return None


def build_pose_issue_list(
    frame_x,
    frame_y,
    frame_scores,
    keypoint_names: Sequence[str],
    keypoint_threshold: float,
    manual_mask_frame=None,
    derived_keypoint_names: Optional[Sequence[str]] = None,
    frame_index: Optional[int] = None,
    full_x_series=None,
    full_y_series=None,
):
    """
    Build a sorted issue list for one pose frame.

    Each entry includes keypoint name, status, and optional diagnostic metadata.
    """

    frame_x = np.asarray(frame_x, dtype=float).reshape(-1)
    frame_y = np.asarray(frame_y, dtype=float).reshape(-1)
    frame_scores = np.asarray(frame_scores, dtype=float).reshape(-1)
    keypoint_names = list(keypoint_names or [])
    if manual_mask_frame is None:
        manual_mask_frame = np.zeros_like(frame_scores, dtype=bool)
    manual_mask_frame = np.asarray(manual_mask_frame, dtype=bool).reshape(-1)
    derived_names = {
        str(name)
        for name in (derived_keypoint_names or DERIVED_KEYPOINT_NAMES)
    }

    issues = []
    for idx, keypoint_name in enumerate(keypoint_names):
        score = frame_scores[idx] if idx < len(frame_scores) else np.nan
        x_value = frame_x[idx] if idx < len(frame_x) else np.nan
        y_value = frame_y[idx] if idx < len(frame_y) else np.nan
        is_manual = bool(idx < len(manual_mask_frame) and manual_mask_frame[idx])
        is_missing = not (np.isfinite(x_value) and np.isfinite(y_value))
        is_low_confidence = np.isfinite(score) and float(score) < float(keypoint_threshold)
        is_derived = keypoint_name in derived_names

        status = None
        if is_missing:
            status = "missing"
        elif is_low_confidence:
            status = "low_confidence"
        elif is_manual:
            status = "manually_edited"
        elif is_derived:
            status = "derived"

        if status is None:
            continue

        issue = {
            "index": int(idx),
            "keypoint": str(keypoint_name),
            "status": status,
            "score": float(score) if np.isfinite(score) else None,
            "threshold": float(keypoint_threshold),
            "manual": is_manual,
            "derived": is_derived,
            "editable": not is_derived,
            "frame_index": int(frame_index) if frame_index is not None else None,
        }
        if status == "missing" and frame_index is not None and full_x_series is not None and full_y_series is not None:
            issue["ghost_xy"] = _find_neighbor_keypoint_position(
                int(frame_index),
                np.asarray(full_x_series, dtype=float)[:, idx],
                np.asarray(full_y_series, dtype=float)[:, idx],
            )
        issues.append(issue)

    return sorted(
        issues,
        key=lambda item: (
            POSE_ISSUE_PRIORITY.get(item["status"], 99),
            item["index"],
        ),
    )


def build_ball_issue_list(
    center,
    score=None,
    score_threshold: float = 0.1,
    manual_override: bool = False,
    visible: bool = True,
    track_missing: bool = False,
):
    """Build a sorted issue list for one ball frame."""

    issues = []
    has_center = center is not None and len(np.asarray(center).reshape(-1)) >= 2
    if not visible or not has_center:
        issues.append({"status": "missing_ball"})
    if track_missing:
        issues.append({"status": "track_gap"})
    if score is not None:
        try:
            score_value = float(score)
        except (TypeError, ValueError):
            score_value = float("nan")
        if np.isfinite(score_value) and score_value < float(score_threshold):
            issues.append(
                {
                    "status": "low_confidence_ball",
                    "score": score_value,
                    "threshold": float(score_threshold),
                }
            )
    if manual_override:
        issues.append({"status": "manual_ball_override"})

    return sorted(
        issues,
        key=lambda item: BALL_ISSUE_PRIORITY.get(item["status"], 99),
    )


def _status_color(status: str):
    colors = {
        "missing": "#D64545",
        "low_confidence": "#E2A13B",
        "manually_edited": "#2F80ED",
        "derived": "#7F8C8D",
        "missing_ball": "#D64545",
        "track_gap": "#9B59B6",
        "low_confidence_ball": "#E2A13B",
        "manual_ball_override": "#2F80ED",
    }
    return colors.get(str(status), "#2D3436")


def _score_to_rgb(score: Optional[float]):
    if score is None or not np.isfinite(score):
        return (0.6, 0.6, 0.6)
    score_clamped = float(np.clip(score, 0.0, 1.0))
    return (
        1.0 - score_clamped,
        0.35 + 0.55 * score_clamped,
        0.15,
    )


def _open_video_capture(video_file_path):
    cap = cv2.VideoCapture(str(Path(video_file_path)))
    if not cap.isOpened():
        raise ValueError(f"Could not open video for hybrid editor: {video_file_path}")
    return cap


def _video_axis_limits(frame) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    height, width = np.asarray(frame).shape[:2]
    return (-0.5, float(width) - 0.5), (float(height) - 0.5, -0.5)


class VideoFrameNavigator:
    """Frame reader optimized for local stepping through compressed videos."""

    def __init__(self, cap, start_frame: int = 0, cache_size: int = 32, sequential_window: int = 4):
        self.cap = cap
        self.start_frame = int(start_frame)
        self.cache_size = max(4, int(cache_size))
        self.sequential_window = max(1, int(sequential_window))
        self.frame_cache: Dict[int, np.ndarray] = {}
        self.cache_order: List[int] = []
        self.stream_frame_idx: Optional[int] = None

    def _touch_cache(self, actual_frame_idx: int):
        if actual_frame_idx in self.cache_order:
            self.cache_order.remove(actual_frame_idx)
            self.cache_order.append(actual_frame_idx)

    def _cache_put(self, actual_frame_idx: int, frame):
        self.frame_cache[actual_frame_idx] = frame.copy()
        self._touch_cache(actual_frame_idx)
        if actual_frame_idx not in self.cache_order:
            self.cache_order.append(actual_frame_idx)
        while len(self.frame_cache) > self.cache_size:
            oldest_frame = self.cache_order.pop(0)
            self.frame_cache.pop(oldest_frame, None)

    def _cache_get(self, actual_frame_idx: int):
        frame = self.frame_cache.get(actual_frame_idx)
        if frame is not None:
            self._touch_cache(actual_frame_idx)
        return frame

    def _seek_and_read(self, actual_frame_idx: int):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(actual_frame_idx))
        success, frame = self.cap.read()
        if not success:
            raise ValueError(f"Could not read frame {actual_frame_idx}")
        self.stream_frame_idx = int(actual_frame_idx)
        self._cache_put(actual_frame_idx, frame)
        return self.frame_cache[actual_frame_idx]

    def _read_forward(self, steps: int):
        frame = None
        for _ in range(int(steps)):
            success, frame = self.cap.read()
            if not success:
                next_frame_idx = 0 if self.stream_frame_idx is None else self.stream_frame_idx + 1
                raise ValueError(f"Could not read frame {next_frame_idx}")
            next_frame_idx = 0 if self.stream_frame_idx is None else self.stream_frame_idx + 1
            self.stream_frame_idx = int(next_frame_idx)
            self._cache_put(self.stream_frame_idx, frame)
        if frame is None or self.stream_frame_idx is None:
            raise ValueError("Could not advance to the requested frame.")
        return self.frame_cache[self.stream_frame_idx]

    def get_frame(self, local_frame_idx: int):
        actual_frame_idx = self.start_frame + int(local_frame_idx)
        cached_frame = self._cache_get(actual_frame_idx)
        if cached_frame is not None:
            return cached_frame

        if self.stream_frame_idx is not None:
            delta = int(actual_frame_idx - self.stream_frame_idx)
            if 0 < delta <= self.sequential_window:
                return self._read_forward(delta)

        return self._seek_and_read(actual_frame_idx)

    def close(self):
        self.cap.release()


class FrameRenderController:
    """Collapse multiple frame requests to the latest pending target."""

    def __init__(self):
        self.pending_frame_idx: Optional[int] = None
        self.rendered_frame_idx: Optional[int] = None

    def request(self, frame_idx: int):
        self.pending_frame_idx = int(frame_idx)

    def consume(self):
        if self.pending_frame_idx is None:
            return None
        frame_idx = int(self.pending_frame_idx)
        self.pending_frame_idx = None
        if frame_idx == self.rendered_frame_idx:
            return None
        self.rendered_frame_idx = frame_idx
        return frame_idx


def _create_frame_slider(slider_cls, ax, label: str, frame_count: int):
    slider_kwargs = {
        "ax": ax,
        "label": label,
        "valmin": 0,
        "valmax": max(0, int(frame_count) - 1),
        "valinit": 0,
        "valstep": 1,
    }
    try:
        return slider_cls(dragging=False, **slider_kwargs)
    except TypeError:
        return slider_cls(**slider_kwargs)


def _compute_zoomed_limits(
    current_limits,
    focus_value: Optional[float],
    zoom_factor: float,
    bounds_limits,
    min_span: float = MIN_ZOOM_VIEW_SPAN_PX,
):
    current_start, current_end = [float(value) for value in current_limits]
    bounds_start, bounds_end = [float(value) for value in bounds_limits]
    is_inverted = current_start > current_end

    current_min, current_max = sorted((current_start, current_end))
    bounds_min, bounds_max = sorted((bounds_start, bounds_end))
    bounds_span = max(bounds_max - bounds_min, 1e-9)
    min_span = float(np.clip(min_span, 1e-6, bounds_span))

    span = max(current_max - current_min, 1e-9)
    target_span = float(np.clip(span * float(zoom_factor), min_span, bounds_span))
    if np.isclose(target_span, bounds_span):
        new_min, new_max = bounds_min, bounds_max
    else:
        if focus_value is None or not np.isfinite(focus_value):
            focus = current_min + span * 0.5
        else:
            focus = float(np.clip(focus_value, current_min, current_max))

        relative_focus = float(np.clip((focus - current_min) / span, 0.0, 1.0))
        new_min = focus - relative_focus * target_span
        new_max = new_min + target_span
        if new_min < bounds_min:
            shift = bounds_min - new_min
            new_min += shift
            new_max += shift
        if new_max > bounds_max:
            shift = new_max - bounds_max
            new_min -= shift
            new_max -= shift
        new_min = max(bounds_min, new_min)
        new_max = min(bounds_max, new_max)

    return (new_max, new_min) if is_inverted else (new_min, new_max)


def _zoom_axis_on_scroll(ax, event, bounds_x, bounds_y):
    if event.inaxes != ax:
        return False

    step = getattr(event, "step", None)
    if step is None:
        button = str(getattr(event, "button", "")).lower()
        if button == "up":
            step = 1
        elif button == "down":
            step = -1
        else:
            return False
    if not np.isfinite(step) or float(step) == 0.0:
        return False

    zoom_factor = float(ZOOM_STEP_FACTOR ** (-float(step)))
    ax.set_xlim(_compute_zoomed_limits(ax.get_xlim(), event.xdata, zoom_factor, bounds_x))
    ax.set_ylim(_compute_zoomed_limits(ax.get_ylim(), event.ydata, zoom_factor, bounds_y))
    ax.figure.canvas.draw_idle()
    return True


def normalize_hybrid_ui_backend(ui_backend: Optional[str]) -> str:
    if ui_backend is None:
        return "auto"
    normalized = str(ui_backend).strip().lower()
    if normalized in {"", "auto"}:
        return "auto"
    if normalized in {"qt", "matplotlib"}:
        return normalized
    logging.warning("Unknown hybrid_ui_backend '%s'. Falling back to auto.", ui_backend)
    return "auto"


def _load_qt_hybrid_editor_module():
    return importlib.import_module("Sports2D.Utilities.hybrid_editor_qt")


def _review_pose_sequence_matplotlib(
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
    """
    Review and optionally edit a single person's raw pose sequence.

    Interaction:
    - Click an issue or a visible keypoint to select it.
    - Click again on the video to move the selected keypoint.
    - Use Hide to remove the selected keypoint for the frame.
    - Use Restore to revert the selected keypoint to the automatic estimate.
    """

    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, Slider

    person_x_raw, person_y_raw, person_scores_raw, keypoint_names = augment_pose_arrays_with_derived_keypoints(
        person_x_raw,
        person_y_raw,
        person_scores_raw,
        keypoint_names,
    )
    original_x = person_x_raw.copy()
    original_y = person_y_raw.copy()
    original_scores = person_scores_raw.copy()
    manual_mask = _normalize_pose_manual_mask(manual_mask, person_scores_raw.shape)

    start_frame, _ = frame_range
    frame_navigator = VideoFrameNavigator(
        _open_video_capture(video_file_path),
        start_frame=start_frame,
        cache_size=32,
        sequential_window=4,
    )
    render_controller = FrameRenderController()

    selected = {"keypoint_index": None}
    dynamic_artists = {"markers": [], "labels": [], "issues": []}
    issue_artist_map = {}

    first_frame = frame_navigator.get_frame(0)
    fig = plt.figure(figsize=(13, 8), num=window_title)
    ax_video = plt.axes([0.05, 0.14, 0.63, 0.78])
    ax_video.axis("off")
    img_plot = ax_video.imshow(cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB))
    default_x_limits, default_y_limits = _video_axis_limits(first_frame)
    ax_video.set_xlim(default_x_limits)
    ax_video.set_ylim(default_y_limits)
    ax_video.set_autoscale_on(False)

    ax_slider = plt.axes([0.05, 0.06, 0.63, 0.04])
    frame_slider = _create_frame_slider(Slider, ax_slider, "frame", len(person_x_raw))

    ax_status = plt.axes([0.71, 0.82, 0.25, 0.12])
    ax_status.axis("off")
    status_text = ax_status.text(0.0, 1.0, "", va="top", fontsize=10)

    ax_issues = plt.axes([0.71, 0.22, 0.25, 0.56])
    ax_issues.axis("off")

    ax_prev = plt.axes([0.71, 0.14, 0.07, 0.05])
    ax_next = plt.axes([0.79, 0.14, 0.07, 0.05])
    ax_hide = plt.axes([0.87, 0.14, 0.07, 0.05])
    ax_restore = plt.axes([0.71, 0.08, 0.11, 0.05])
    ax_ok = plt.axes([0.84, 0.08, 0.10, 0.05])
    prev_button = Button(ax_prev, "Prev")
    next_button = Button(ax_next, "Next")
    hide_button = Button(ax_hide, "Hide")
    restore_button = Button(ax_restore, "Restore")
    ok_button = Button(ax_ok, "OK")
    try:
        render_timer = fig.canvas.new_timer(interval=35)
    except Exception:
        render_timer = None

    def clear_dynamic_artists():
        for artist_group in dynamic_artists.values():
            for artist in artist_group:
                try:
                    artist.remove()
                except Exception:
                    pass
            artist_group.clear()
        issue_artist_map.clear()

    def get_current_frame_index() -> int:
        return int(frame_slider.val)

    def select_nearest_keypoint(frame_idx: int, x_click: float, y_click: float):
        best_idx = None
        best_dist = 20.0
        frame_x = person_x_raw[frame_idx]
        frame_y = person_y_raw[frame_idx]
        for idx, (x_value, y_value) in enumerate(zip(frame_x, frame_y)):
            if not (np.isfinite(x_value) and np.isfinite(y_value)):
                ghost_xy = _find_neighbor_keypoint_position(
                    frame_idx,
                    person_x_raw[:, idx],
                    person_y_raw[:, idx],
                )
                if ghost_xy is None:
                    continue
                x_value, y_value = ghost_xy
            dist = float(np.hypot(x_value - x_click, y_value - y_click))
            if dist < best_dist:
                best_dist = dist
                best_idx = idx
        return best_idx

    def render_frame(frame_idx: int):
        refreshed_x, refreshed_y, refreshed_scores = refresh_pose_derived_keypoints(
            person_x_raw,
            person_y_raw,
            person_scores_raw,
            keypoint_names,
        )
        person_x_raw[:] = refreshed_x
        person_y_raw[:] = refreshed_y
        person_scores_raw[:] = refreshed_scores
        frame_bgr = frame_navigator.get_frame(frame_idx)
        img_plot.set_data(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        clear_dynamic_artists()

        issues = build_pose_issue_list(
            person_x_raw[frame_idx],
            person_y_raw[frame_idx],
            person_scores_raw[frame_idx],
            keypoint_names=keypoint_names,
            keypoint_threshold=keypoint_threshold,
            manual_mask_frame=manual_mask[frame_idx],
            frame_index=frame_idx,
            full_x_series=person_x_raw,
            full_y_series=person_y_raw,
        )
        issue_by_index = {issue["index"]: issue for issue in issues}

        line_height = 0.042
        title_artist = ax_issues.text(0.0, 1.0, "Frame issues", va="top", fontsize=11, fontweight="bold")
        dynamic_artists["issues"].append(title_artist)
        if len(issues) == 0:
            no_issue_artist = ax_issues.text(0.0, 0.94, "No flagged keypoints.", va="top", fontsize=9)
            dynamic_artists["issues"].append(no_issue_artist)
        else:
            for line_idx, issue in enumerate(issues):
                score_text = ""
                if issue.get("score") is not None:
                    score_text = f" ({issue['score']:.2f})"
                body = f"{issue['keypoint']}: {issue['status']}{score_text}"
                artist = ax_issues.text(
                    0.0,
                    0.94 - line_idx * line_height,
                    body,
                    va="top",
                    fontsize=8.5,
                    color=_status_color(issue["status"]),
                    picker=True,
                )
                issue_artist_map[artist] = issue["index"]
                dynamic_artists["issues"].append(artist)

        for kp_idx, keypoint_name in enumerate(keypoint_names):
            issue = issue_by_index.get(kp_idx)
            x_value = person_x_raw[frame_idx, kp_idx]
            y_value = person_y_raw[frame_idx, kp_idx]
            score = person_scores_raw[frame_idx, kp_idx]
            if issue is not None and issue["status"] == "missing":
                ghost_xy = issue.get("ghost_xy")
                if ghost_xy is not None:
                    marker = ax_video.scatter(
                        [ghost_xy[0]],
                        [ghost_xy[1]],
                        marker="x",
                        s=90,
                        c=_status_color("missing"),
                        linewidths=1.8,
                        zorder=4,
                    )
                    dynamic_artists["markers"].append(marker)
                    if selected["keypoint_index"] == kp_idx:
                        label = ax_video.text(
                            ghost_xy[0] + 6,
                            ghost_xy[1] + 6,
                            keypoint_name,
                            color=_status_color("missing"),
                            fontsize=8,
                            zorder=5,
                        )
                        dynamic_artists["labels"].append(label)
                continue

            if not (np.isfinite(x_value) and np.isfinite(y_value)):
                continue

            if issue is not None and issue["status"] == "manually_edited":
                facecolor = _status_color("manually_edited")
                edgecolor = "#0B132B"
                linewidth = 1.6
                size = 55
            elif issue is not None and issue["status"] == "derived":
                facecolor = _status_color("derived")
                edgecolor = "#2D3436"
                linewidth = 1.0
                size = 36
            elif issue is not None and issue["status"] == "low_confidence":
                facecolor = "none"
                edgecolor = _status_color("low_confidence")
                linewidth = 1.8
                size = 52
            else:
                facecolor = _score_to_rgb(score)
                edgecolor = "#0B132B"
                linewidth = 0.8
                size = 34

            marker = ax_video.scatter(
                [x_value],
                [y_value],
                s=size,
                facecolors=facecolor,
                edgecolors=edgecolor,
                linewidths=linewidth,
                zorder=4,
            )
            dynamic_artists["markers"].append(marker)

            if selected["keypoint_index"] == kp_idx:
                highlight = ax_video.scatter(
                    [x_value],
                    [y_value],
                    s=size + 85,
                    facecolors="none",
                    edgecolors="#00C2FF",
                    linewidths=2.0,
                    zorder=5,
                )
                label = ax_video.text(
                    x_value + 6,
                    y_value + 6,
                    keypoint_name,
                    color="#00C2FF",
                    fontsize=8,
                    zorder=6,
                )
                dynamic_artists["markers"].append(highlight)
                dynamic_artists["labels"].append(label)

        selected_idx = selected["keypoint_index"]
        selected_name = keypoint_names[selected_idx] if selected_idx is not None else "None"
        selected_issue = issue_by_index.get(selected_idx) if selected_idx is not None else None
        selected_status = selected_issue["status"] if selected_issue is not None else "normal"
        status_text.set_text(
            "Hybrid Pose Review\n"
            "1. Click an issue or keypoint to select it.\n"
            "2. Click in the video to move the selected keypoint.\n"
            "3. Use Hide or Restore for the selected frame.\n"
            "4. Scroll to zoom around the cursor.\n"
            "Legend: missing=red x, low_conf=amber, manual=blue, derived=gray.\n"
            f"Selected: {selected_name} ({selected_status})"
        )
        fig.canvas.draw_idle()
        render_controller.rendered_frame_idx = int(frame_idx)

    def flush_pending_redraw():
        if render_timer is not None:
            render_timer.stop()
        frame_idx = render_controller.consume()
        if frame_idx is None:
            return
        render_frame(frame_idx)

    def redraw(*_args, force=False):
        if force:
            render_frame(get_current_frame_index())
            return

        render_controller.request(get_current_frame_index())
        if render_timer is None:
            flush_pending_redraw()
            return
        render_timer.stop()
        render_timer.start()

    def on_click(event):
        if event.inaxes != ax_video or event.xdata is None or event.ydata is None:
            return

        frame_idx = get_current_frame_index()
        selected_idx = selected["keypoint_index"]
        nearest_idx = select_nearest_keypoint(frame_idx, float(event.xdata), float(event.ydata))
        if nearest_idx is not None and (selected_idx is None or nearest_idx != selected_idx):
            selected["keypoint_index"] = int(nearest_idx)
            redraw(force=True)
            return

        if selected_idx is not None and keypoint_names[selected_idx] not in DERIVED_KEYPOINT_NAMES:
            person_x_raw[frame_idx, selected_idx] = float(event.xdata)
            person_y_raw[frame_idx, selected_idx] = float(event.ydata)
            person_scores_raw[frame_idx, selected_idx] = max(1.0, float(keypoint_threshold))
            manual_mask[frame_idx, selected_idx] = True
            redraw(force=True)
            return

    def on_pick(event):
        if event.artist not in issue_artist_map:
            return
        selected["keypoint_index"] = int(issue_artist_map[event.artist])
        redraw(force=True)

    def on_prev(_event):
        frame_slider.set_val(max(0, get_current_frame_index() - 1))

    def on_next(_event):
        frame_slider.set_val(min(len(person_x_raw) - 1, get_current_frame_index() + 1))

    def on_hide(_event):
        selected_idx = selected["keypoint_index"]
        if selected_idx is None or keypoint_names[selected_idx] in DERIVED_KEYPOINT_NAMES:
            return
        frame_idx = get_current_frame_index()
        person_x_raw[frame_idx, selected_idx] = np.nan
        person_y_raw[frame_idx, selected_idx] = np.nan
        person_scores_raw[frame_idx, selected_idx] = np.nan
        manual_mask[frame_idx, selected_idx] = True
        redraw(force=True)

    def on_restore(_event):
        selected_idx = selected["keypoint_index"]
        if selected_idx is None or keypoint_names[selected_idx] in DERIVED_KEYPOINT_NAMES:
            return
        frame_idx = get_current_frame_index()
        person_x_raw[frame_idx, selected_idx] = original_x[frame_idx, selected_idx]
        person_y_raw[frame_idx, selected_idx] = original_y[frame_idx, selected_idx]
        person_scores_raw[frame_idx, selected_idx] = original_scores[frame_idx, selected_idx]
        manual_mask[frame_idx, selected_idx] = False
        redraw(force=True)

    def on_ok(_event):
        plt.close(fig)

    def on_scroll(event):
        _zoom_axis_on_scroll(ax_video, event, default_x_limits, default_y_limits)

    if render_timer is not None:
        render_timer.add_callback(flush_pending_redraw)
    frame_slider.on_changed(redraw)
    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("pick_event", on_pick)
    fig.canvas.mpl_connect("scroll_event", on_scroll)
    prev_button.on_clicked(on_prev)
    next_button.on_clicked(on_next)
    hide_button.on_clicked(on_hide)
    restore_button.on_clicked(on_restore)
    ok_button.on_clicked(on_ok)

    render_frame(0)
    plt.show()
    frame_navigator.close()
    return person_x_raw, person_y_raw, person_scores_raw, manual_mask


def _normalize_ball_center_value(center):
    if center is None:
        return None
    center_arr = np.asarray(center, dtype=float).reshape(-1)
    if len(center_arr) < 2:
        return None
    if not np.isfinite(center_arr[0]) or not np.isfinite(center_arr[1]):
        return None
    return (int(round(float(center_arr[0]))), int(round(float(center_arr[1]))))


def _selected_track_review_state(frame_tracks, selected_track_id, frame_center=None):
    if selected_track_id is None:
        return None, None, False, None
    selected_track_id = int(selected_track_id)
    visible_tracks = []
    selected_track = None
    for track in frame_tracks or []:
        track_id = int(track.get("id", -1))
        track_center = _normalize_ball_center_value(track.get("center"))
        if track_id == selected_track_id:
            selected_track = track
        if bool(track.get("visible", False)) and track_center is not None:
            visible_tracks.append((track, track_center))

    if selected_track is not None:
        selected_center = _normalize_ball_center_value(selected_track.get("center"))
        if bool(selected_track.get("visible", False)) and selected_center is not None:
            return selected_center, selected_track.get("score"), True, selected_track_id

    review_center = _normalize_ball_center_value(frame_center)
    if review_center is not None and len(visible_tracks) > 0:
        source_track, source_center = min(
            visible_tracks,
            key=lambda item: float(np.linalg.norm(np.asarray(item[1], dtype=float) - np.asarray(review_center, dtype=float))),
        )
        return source_center, source_track.get("score"), True, int(source_track.get("id"))

    if selected_track is not None:
        return _normalize_ball_center_value(selected_track.get("center")), selected_track.get("score"), bool(selected_track.get("visible", False)), None

    return None, None, False, None


def _recenter_box(box, center):
    if box is None or center is None:
        return None
    box_arr = np.asarray(box, dtype=float).reshape(-1)
    if len(box_arr) < 4:
        return None
    width = float(box_arr[2] - box_arr[0])
    height = float(box_arr[3] - box_arr[1])
    center_x, center_y = float(center[0]), float(center[1])
    return np.array(
        [
            center_x - width * 0.5,
            center_y - height * 0.5,
            center_x + width * 0.5,
            center_y + height * 0.5,
        ],
        dtype=np.float32,
    )


def apply_ball_override_to_tracks(frame_tracks, selected_track_id, center, visible):
    """Apply a manual ball override onto a frame's tracked-ball metadata."""

    updated_tracks = []
    selected_track_id = None if selected_track_id is None else int(selected_track_id)
    matched = False
    for track in frame_tracks or []:
        updated_track = dict(track)
        if selected_track_id is not None and int(updated_track.get("id", -1)) == selected_track_id:
            matched = True
            if visible and center is not None:
                updated_track["center"] = (int(round(float(center[0]))), int(round(float(center[1]))))
                updated_track["visible"] = True
                updated_track["missing"] = 0
                updated_track["box"] = _recenter_box(updated_track.get("box"), updated_track["center"])
            else:
                updated_track["center"] = None
                updated_track["visible"] = False
                updated_track["missing"] = int(updated_track.get("missing", 0)) + 1
                updated_track["box"] = None
        updated_tracks.append(updated_track)

    if not matched and selected_track_id is not None:
        updated_tracks.append(
            {
                "id": selected_track_id,
                "center": (int(round(float(center[0]))), int(round(float(center[1])))) if visible and center is not None else None,
                "box": None,
                "score": float("nan"),
                "visible": bool(visible and center is not None),
                "missing": 0 if visible and center is not None else 1,
            }
        )
    return updated_tracks


def _review_ball_sequence_matplotlib(
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
    """Review and optionally edit the selected ball timeline."""

    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, Slider

    ball_centers = list(ball_centers)
    original_centers = [None if center is None else tuple(center) for center in ball_centers]
    ball_visible = [center is not None for center in ball_centers]
    original_visible = list(ball_visible)
    manual_override_mask = [False for _ in ball_centers]

    start_frame, _ = frame_range
    frame_navigator = VideoFrameNavigator(
        _open_video_capture(video_file_path),
        start_frame=start_frame,
        cache_size=32,
        sequential_window=4,
    )
    render_controller = FrameRenderController()

    first_frame = frame_navigator.get_frame(0)
    fig = plt.figure(figsize=(12, 8), num=window_title)
    ax_video = plt.axes([0.05, 0.14, 0.68, 0.78])
    ax_video.axis("off")
    img_plot = ax_video.imshow(cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB))
    default_x_limits, default_y_limits = _video_axis_limits(first_frame)
    ax_video.set_xlim(default_x_limits)
    ax_video.set_ylim(default_y_limits)
    ax_video.set_autoscale_on(False)

    ax_slider = plt.axes([0.05, 0.06, 0.68, 0.04])
    frame_slider = _create_frame_slider(Slider, ax_slider, "frame", len(ball_centers))

    ax_status = plt.axes([0.77, 0.46, 0.2, 0.44])
    ax_status.axis("off")
    status_text = ax_status.text(0.0, 1.0, "", va="top", fontsize=10)

    ax_prev = plt.axes([0.77, 0.14, 0.07, 0.05])
    ax_next = plt.axes([0.85, 0.14, 0.07, 0.05])
    ax_hide = plt.axes([0.77, 0.08, 0.07, 0.05])
    ax_restore = plt.axes([0.85, 0.08, 0.07, 0.05])
    ax_ok = plt.axes([0.77, 0.02, 0.15, 0.05])
    prev_button = Button(ax_prev, "Prev")
    next_button = Button(ax_next, "Next")
    hide_button = Button(ax_hide, "Hide")
    restore_button = Button(ax_restore, "Restore")
    ok_button = Button(ax_ok, "OK")
    try:
        render_timer = fig.canvas.new_timer(interval=35)
    except Exception:
        render_timer = None

    dynamic_artists = []

    def clear_dynamic_artists():
        for artist in dynamic_artists:
            try:
                artist.remove()
            except Exception:
                pass
        dynamic_artists.clear()

    def current_index():
        return int(frame_slider.val)

    def render_frame(frame_idx: int):
        frame_bgr = frame_navigator.get_frame(frame_idx)
        img_plot.set_data(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        clear_dynamic_artists()

        frame_center = ball_centers[frame_idx]
        selected_id = selected_ball_ids[frame_idx] if frame_idx < len(selected_ball_ids) else None
        frame_tracks = ball_tracks[frame_idx] if frame_idx < len(ball_tracks) else []
        track_center, track_score, track_visible, source_track_id = _selected_track_review_state(
            frame_tracks,
            selected_id,
            frame_center=frame_center,
        )
        effective_score = track_score
        issues = build_ball_issue_list(
            frame_center,
            score=effective_score,
            score_threshold=score_threshold,
            manual_override=manual_override_mask[frame_idx],
            visible=ball_visible[frame_idx],
            track_missing=bool(selected_id is not None and not track_visible and frame_center is None),
        )

        boxes = np.asarray(ball_boxes[frame_idx], dtype=float).reshape(-1, 4) if frame_idx < len(ball_boxes) else np.empty((0, 4))
        for box in boxes:
            x1, y1, x2, y2 = box
            rect = plt.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=1.0,
                edgecolor="#F39C12",
                facecolor="none",
                zorder=3,
            )
            ax_video.add_patch(rect)
            dynamic_artists.append(rect)

        if frame_center is not None and ball_visible[frame_idx]:
            marker_color = _status_color("manual_ball_override") if manual_override_mask[frame_idx] else "#111111"
            marker = ax_video.scatter(
                [frame_center[0]],
                [frame_center[1]],
                s=90,
                c=marker_color,
                edgecolors="#F7F7F7",
                linewidths=1.2,
                zorder=4,
            )
            dynamic_artists.append(marker)

        status_lines = [
            "Hybrid Ball Review",
            "Click in the video to set the ball center.",
            "Use Hide or Restore for the current frame.",
            "Scroll to zoom around the cursor.",
            f"Selected track: {selected_id}",
        ]
        if source_track_id is not None and selected_id is not None and int(source_track_id) != int(selected_id):
            status_lines.append(f"Visible source track: {source_track_id}")
        for issue in issues:
            if issue["status"] == "low_confidence_ball":
                status_lines.append(
                    f"- low_confidence_ball ({issue['score']:.2f} < {issue['threshold']:.2f})"
                )
            else:
                status_lines.append(f"- {issue['status']}")
        if len(issues) == 0:
            status_lines.append("- no issues")
        status_text.set_text("\n".join(status_lines))
        fig.canvas.draw_idle()
        render_controller.rendered_frame_idx = int(frame_idx)

    def flush_pending_redraw():
        if render_timer is not None:
            render_timer.stop()
        frame_idx = render_controller.consume()
        if frame_idx is None:
            return
        render_frame(frame_idx)

    def redraw(*_args, force=False):
        if force:
            render_frame(current_index())
            return

        render_controller.request(current_index())
        if render_timer is None:
            flush_pending_redraw()
            return
        render_timer.stop()
        render_timer.start()

    def on_click(event):
        if event.inaxes != ax_video or event.xdata is None or event.ydata is None:
            return
        frame_idx = current_index()
        ball_centers[frame_idx] = (int(round(float(event.xdata))), int(round(float(event.ydata))))
        ball_visible[frame_idx] = True
        manual_override_mask[frame_idx] = True
        redraw(force=True)

    def on_prev(_event):
        frame_slider.set_val(max(0, current_index() - 1))

    def on_next(_event):
        frame_slider.set_val(min(len(ball_centers) - 1, current_index() + 1))

    def on_hide(_event):
        frame_idx = current_index()
        ball_centers[frame_idx] = None
        ball_visible[frame_idx] = False
        manual_override_mask[frame_idx] = True
        redraw(force=True)

    def on_restore(_event):
        frame_idx = current_index()
        ball_centers[frame_idx] = original_centers[frame_idx]
        ball_visible[frame_idx] = original_visible[frame_idx]
        manual_override_mask[frame_idx] = False
        redraw(force=True)

    def on_ok(_event):
        plt.close(fig)

    def on_scroll(event):
        _zoom_axis_on_scroll(ax_video, event, default_x_limits, default_y_limits)

    if render_timer is not None:
        render_timer.add_callback(flush_pending_redraw)
    frame_slider.on_changed(redraw)
    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("scroll_event", on_scroll)
    prev_button.on_clicked(on_prev)
    next_button.on_clicked(on_next)
    hide_button.on_clicked(on_hide)
    restore_button.on_clicked(on_restore)
    ok_button.on_clicked(on_ok)

    render_frame(0)
    plt.show()
    frame_navigator.close()
    return ball_centers, ball_visible, manual_override_mask


def review_pose_sequence(
    video_file_path,
    frame_range,
    person_x_raw,
    person_y_raw,
    person_scores_raw,
    keypoint_names: Sequence[str],
    keypoint_threshold: float,
    manual_mask=None,
    window_title: str = "Hybrid pose review",
    ui_backend: Optional[str] = None,
):
    selected_backend = normalize_hybrid_ui_backend(ui_backend)
    if selected_backend in {"auto", "qt"}:
        try:
            qt_module = _load_qt_hybrid_editor_module()
            return qt_module.review_pose_sequence_qt(
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
        except Exception as exc:
            logging.warning(
                "Qt hybrid pose editor unavailable (%s). Falling back to Matplotlib editor.",
                exc,
            )

    return _review_pose_sequence_matplotlib(
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


def review_ball_sequence(
    video_file_path,
    frame_range,
    ball_centers,
    ball_boxes,
    ball_scores,
    ball_tracks,
    selected_ball_ids,
    score_threshold: float = 0.1,
    window_title: str = "Hybrid ball review",
    ui_backend: Optional[str] = None,
):
    selected_backend = normalize_hybrid_ui_backend(ui_backend)
    if selected_backend in {"auto", "qt"}:
        try:
            qt_module = _load_qt_hybrid_editor_module()
            return qt_module.review_ball_sequence_qt(
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
        except Exception as exc:
            logging.warning(
                "Qt hybrid ball editor unavailable (%s). Falling back to Matplotlib editor.",
                exc,
            )

    return _review_ball_sequence_matplotlib(
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
