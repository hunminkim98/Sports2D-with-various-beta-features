#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

try:
    from scipy.signal import butter, filtfilt
except ImportError:  # pragma: no cover - Pose2Sim normally provides scipy
    butter = None
    filtfilt = None


GRAVITY_MPS2 = 9.81
PELVIS_TRUNK_ALPHA = 0.20
GRF_FILTER_CUTOFF_HZ = 6.0
GRF_FILTER_ORDER = 4
MIN_FLIGHT_TIME_S = 0.08
FOOT_CONTACT_HEIGHT_M = 0.03
CONTACT_VELOCITY_THRESHOLD_MPS = 0.50
COM_PROXY_METHOD = "pelvis_trunk_proxy_v1"


def _marker_triplet(trc_data, marker_name):
    if marker_name not in trc_data.columns:
        return None
    marker_data = trc_data.loc[:, trc_data.columns == marker_name]
    marker_data = pd.DataFrame(marker_data).copy()
    if marker_data.shape[1] < 3:
        return None
    return marker_data.iloc[:, :3]


def _marker_axis(trc_data, marker_name, axis="y"):
    marker_data = _marker_triplet(trc_data, marker_name)
    if marker_data is None:
        return None
    axis_idx = {"x": 0, "y": 1, "z": 2}.get(str(axis).lower())
    if axis_idx is None:
        raise ValueError(f"Unsupported axis '{axis}'. Expected one of: x, y, z.")
    return marker_data.iloc[:, axis_idx].astype(float).reset_index(drop=True)


def _point_from_name_or_pair(person_x, person_y, keypoint_names, marker_name, pair_names):
    keypoint_names = list(keypoint_names)
    if marker_name in keypoint_names:
        idx = keypoint_names.index(marker_name)
        x_value = float(person_x[idx])
        y_value = float(person_y[idx])
        if np.isfinite(x_value) and np.isfinite(y_value):
            return np.array([x_value, y_value], dtype=float)

    points = []
    for name in pair_names:
        if name not in keypoint_names:
            continue
        idx = keypoint_names.index(name)
        x_value = float(person_x[idx])
        y_value = float(person_y[idx])
        if np.isfinite(x_value) and np.isfinite(y_value):
            points.append((x_value, y_value))
    if len(points) == 0:
        return None
    return np.nanmean(np.asarray(points, dtype=float), axis=0)


def estimate_pelvis_trunk_com_xy_px(person_x, person_y, keypoint_names, alpha=PELVIS_TRUNK_ALPHA):
    hip_point = _point_from_name_or_pair(
        person_x,
        person_y,
        keypoint_names,
        marker_name="Hip",
        pair_names=("LHip", "RHip"),
    )
    neck_point = _point_from_name_or_pair(
        person_x,
        person_y,
        keypoint_names,
        marker_name="Neck",
        pair_names=("LShoulder", "RShoulder"),
    )
    if hip_point is None or neck_point is None:
        return None
    com_point = hip_point + float(alpha) * (neck_point - hip_point)
    if not np.all(np.isfinite(com_point)):
        return None
    return tuple(np.round(com_point).astype(int).tolist())


def estimate_pelvis_trunk_com_y(trc_data, alpha=PELVIS_TRUNK_ALPHA):
    hip_y = _marker_axis(trc_data, "Hip", axis="y")
    neck_y = _marker_axis(trc_data, "Neck", axis="y")
    if hip_y is None or neck_y is None:
        raise ValueError("Pelvis-trunk CoM proxy requires Hip and Neck marker triplets.")
    return hip_y + float(alpha) * (neck_y - hip_y)


def _available_support_marker_names(marker_names):
    groups = [
        ("LBigToe", "RBigToe"),
        ("LToe", "RToe"),
        ("LAnkle", "RAnkle"),
    ]
    for group in groups:
        available = [name for name in group if name in marker_names]
        if len(available) > 0:
            return available
    return []


def _support_side_point_px(person_x, person_y, keypoint_names, side_prefix):
    keypoint_names = list(keypoint_names)
    side_prefix = str(side_prefix).strip().upper()
    toe_names = (f"{side_prefix}BigToe", f"{side_prefix}Toe")
    heel_names = (f"{side_prefix}Heel",)
    ankle_names = (f"{side_prefix}Ankle",)

    toe_point = _point_from_name_or_pair(
        person_x,
        person_y,
        keypoint_names,
        marker_name=toe_names[0],
        pair_names=toe_names,
    )
    heel_point = _point_from_name_or_pair(
        person_x,
        person_y,
        keypoint_names,
        marker_name=heel_names[0],
        pair_names=heel_names,
    )
    if toe_point is not None and heel_point is not None:
        return 0.5 * (toe_point + heel_point)
    if toe_point is not None:
        return toe_point
    if heel_point is not None:
        return heel_point
    ankle_point = _point_from_name_or_pair(
        person_x,
        person_y,
        keypoint_names,
        marker_name=ankle_names[0],
        pair_names=ankle_names,
    )
    if ankle_point is not None:
        return ankle_point
    return None


def _support_height_series(trc_data):
    marker_names = _available_support_marker_names(trc_data.columns)
    if len(marker_names) == 0:
        return None
    support_series = []
    for marker_name in marker_names:
        marker_y = _marker_axis(trc_data, marker_name, axis="y")
        if marker_y is not None:
            support_series.append(marker_y.to_numpy(dtype=float))
    if len(support_series) == 0:
        return None
    return np.nanmin(np.vstack(support_series), axis=0)


def _interpolate_nan_series(values):
    values = np.asarray(values, dtype=float).copy()
    if len(values) == 0 or np.all(~np.isfinite(values)):
        return values
    valid_idx = np.flatnonzero(np.isfinite(values))
    if len(valid_idx) == len(values):
        return values
    all_idx = np.arange(len(values))
    values[~np.isfinite(values)] = np.interp(
        all_idx[~np.isfinite(values)],
        valid_idx,
        values[valid_idx],
    )
    return values


def lowpass_signal(values, fps, cutoff_hz=GRF_FILTER_CUTOFF_HZ, order=GRF_FILTER_ORDER):
    values = _interpolate_nan_series(values)
    if len(values) == 0 or not np.any(np.isfinite(values)):
        return values
    if butter is None or filtfilt is None:
        raise RuntimeError(
            "motion.vertical_jump requires scipy.signal butter/filtfilt for CoM filtering."
        )
    if fps is None or fps <= 0:
        return values
    nyquist_hz = 0.5 * float(fps)
    if cutoff_hz <= 0 or cutoff_hz >= nyquist_hz:
        return values
    b, a = butter(int(order), float(cutoff_hz) / nyquist_hz, btype="low")
    padlen = 3 * (max(len(a), len(b)) - 1)
    if len(values) <= padlen:
        return values
    return filtfilt(b, a, values)


def detect_vertical_jump_events(
    trc_data_m,
    com_velocity_y,
    raw_vgrf_n,
    body_weight_n,
    fps,
    foot_contact_height_m=FOOT_CONTACT_HEIGHT_M,
    contact_velocity_threshold_mps=CONTACT_VELOCITY_THRESHOLD_MPS,
    min_flight_time_s=MIN_FLIGHT_TIME_S,
):
    support_height = _support_height_series(trc_data_m)
    min_flight_frames = max(1, int(round(float(min_flight_time_s) * float(fps))))
    takeoff_frame = None
    landing_frame = None

    if support_height is not None:
        airborne_mask = support_height > float(foot_contact_height_m)
        contact_mask = support_height <= float(foot_contact_height_m)

        for frame_idx in range(0, max(0, len(airborne_mask) - min_flight_frames + 1)):
            if not np.all(airborne_mask[frame_idx:frame_idx + min_flight_frames]):
                continue
            velocity_value = float(com_velocity_y[frame_idx]) if frame_idx < len(com_velocity_y) else np.nan
            if np.isnan(velocity_value) or velocity_value >= -float(contact_velocity_threshold_mps):
                takeoff_frame = frame_idx
                break

        if takeoff_frame is not None:
            landing_search_start = min(len(contact_mask), takeoff_frame + min_flight_frames)
            for frame_idx in range(landing_search_start, len(contact_mask)):
                if contact_mask[frame_idx]:
                    landing_frame = frame_idx
                    break

    if takeoff_frame is None:
        positive_velocity_candidates = np.where(
            np.asarray(com_velocity_y, dtype=float) >= max(0.0, 0.5 * float(contact_velocity_threshold_mps))
        )[0]
        if len(positive_velocity_candidates) > 0:
            propulsive_start_frame = int(positive_velocity_candidates[0])
            low_force_candidates = np.where(
                (raw_vgrf_n <= 0.1 * float(body_weight_n))
                & (np.arange(len(raw_vgrf_n)) > propulsive_start_frame)
            )[0]
            takeoff_frame = int(low_force_candidates[0]) if len(low_force_candidates) > 0 else None
    if landing_frame is None and takeoff_frame is not None:
        landing_candidates = np.where(raw_vgrf_n[min(len(raw_vgrf_n), takeoff_frame + 1):] >= 0.5 * float(body_weight_n))[0]
        if len(landing_candidates) > 0:
            landing_frame = int(landing_candidates[0] + min(len(raw_vgrf_n), takeoff_frame + 1))

    if takeoff_frame is not None:
        takeoff_frame = int(np.clip(takeoff_frame, 0, len(raw_vgrf_n) - 1))
    if landing_frame is not None:
        landing_frame = int(np.clip(landing_frame, 0, len(raw_vgrf_n) - 1))
        if takeoff_frame is not None and landing_frame <= takeoff_frame:
            landing_frame = None

    return takeoff_frame, landing_frame


def _zero_flight_phase(vgrf_n, takeoff_frame, landing_frame):
    vgrf_zeroed = np.asarray(vgrf_n, dtype=float).copy()
    if takeoff_frame is None:
        return vgrf_zeroed
    zero_start = int(takeoff_frame)
    zero_end = int(landing_frame) if landing_frame is not None else len(vgrf_zeroed)
    vgrf_zeroed[zero_start:zero_end] = 0.0
    return vgrf_zeroed


def _apply_vgrf_constraints(vgrf_n, takeoff_frame, landing_frame):
    vgrf_constrained = _zero_flight_phase(vgrf_n, takeoff_frame, landing_frame)
    finite_mask = np.isfinite(vgrf_constrained)
    vgrf_constrained[finite_mask] = np.maximum(vgrf_constrained[finite_mask], 0.0)
    return vgrf_constrained


def _json_safe_value(value):
    if isinstance(value, dict):
        return {str(k): _json_safe_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_safe_value(v) for v in value.tolist()]
    if value is None:
        return None
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def analyze_vertical_jump_trial(
    trc_data_m,
    mass_kg,
    fps,
    alpha=PELVIS_TRUNK_ALPHA,
    cutoff_hz=GRF_FILTER_CUTOFF_HZ,
    filter_order=GRF_FILTER_ORDER,
):
    trc_data_m = pd.DataFrame(trc_data_m).copy()
    if "time" in trc_data_m.columns and len(trc_data_m) > 0:
        time_s = np.asarray(trc_data_m["time"], dtype=float)
    else:
        time_s = np.arange(len(trc_data_m), dtype=float) / float(fps)
    if len(time_s) >= 2:
        dt = float(np.nanmedian(np.diff(time_s)))
        effective_fps = float(1.0 / dt) if dt > 0 else float(fps)
    else:
        effective_fps = float(fps)
        dt = float(1.0 / effective_fps) if effective_fps > 0 else 0.0

    com_y_raw = estimate_pelvis_trunk_com_y(trc_data_m, alpha=alpha).to_numpy(dtype=float)
    if not np.any(np.isfinite(com_y_raw)):
        raise ValueError("Pelvis-trunk CoM proxy could not be computed from Hip/Neck markers.")
    com_y_filtered = lowpass_signal(
        com_y_raw,
        fps=effective_fps,
        cutoff_hz=cutoff_hz,
        order=filter_order,
    )
    velocity_y = np.gradient(com_y_filtered, dt) if len(com_y_filtered) > 1 and dt > 0 else np.zeros_like(com_y_filtered)
    acceleration_y = np.gradient(velocity_y, dt) if len(velocity_y) > 1 and dt > 0 else np.zeros_like(velocity_y)
    body_weight_n = float(mass_kg) * GRAVITY_MPS2
    raw_vgrf_n = float(mass_kg) * acceleration_y + body_weight_n

    takeoff_frame, landing_frame = detect_vertical_jump_events(
        trc_data_m,
        velocity_y,
        raw_vgrf_n,
        body_weight_n,
        fps=effective_fps,
    )
    vgrf_n = _apply_vgrf_constraints(raw_vgrf_n, takeoff_frame, landing_frame)

    pre_takeoff_stop = int(takeoff_frame) if takeoff_frame is not None else len(vgrf_n)
    pre_takeoff_stop = max(1, pre_takeoff_stop)
    lowest_com_frame = int(np.nanargmin(com_y_filtered[:pre_takeoff_stop])) if pre_takeoff_stop > 0 else 0
    peak_vgrf_n = float(np.nanmax(vgrf_n[:pre_takeoff_stop])) if pre_takeoff_stop > 0 else float(np.nanmax(vgrf_n))
    peak_vgrf_bw = float(peak_vgrf_n / body_weight_n) if body_weight_n > 0 else None

    net_impulse_ns = None
    if takeoff_frame is not None and takeoff_frame > lowest_com_frame:
        net_impulse_ns = float(np.trapz(
            vgrf_n[lowest_com_frame:takeoff_frame + 1] - body_weight_n,
            time_s[lowest_com_frame:takeoff_frame + 1],
        ))

    rfd_n_per_s = None
    peak_frame = None
    if pre_takeoff_stop > 0:
        peak_frame = int(np.nanargmax(vgrf_n[:pre_takeoff_stop]))
        min_before_peak_frame = int(np.nanargmin(vgrf_n[:peak_frame + 1]))
        delta_t = float((peak_frame - min_before_peak_frame) / effective_fps) if effective_fps > 0 else 0.0
        if delta_t > 0:
            rfd_n_per_s = float((vgrf_n[peak_frame] - vgrf_n[min_before_peak_frame]) / delta_t)

    metrics = {
        "body_weight_n": body_weight_n,
        "peak_vgrf_n": peak_vgrf_n,
        "peak_vgrf_bw": peak_vgrf_bw,
        "net_impulse_ns": net_impulse_ns,
        "rfd_n_per_s": rfd_n_per_s,
        "takeoff_frame": takeoff_frame,
        "takeoff_time_s": float(time_s[takeoff_frame]) if takeoff_frame is not None and takeoff_frame < len(time_s) else None,
        "landing_frame": landing_frame,
        "landing_time_s": float(time_s[landing_frame]) if landing_frame is not None and landing_frame < len(time_s) else None,
        "lowest_com_frame": lowest_com_frame,
        "lowest_com_time_s": float(time_s[lowest_com_frame]) if lowest_com_frame < len(time_s) else None,
        "peak_frame": peak_frame,
        "com_proxy_method": COM_PROXY_METHOD,
        "pelvis_trunk_alpha": float(alpha),
        "filter_cutoff_hz": float(cutoff_hz),
        "filter_order": int(filter_order),
    }

    return {
        "time_s": time_s,
        "com_y_raw": com_y_raw,
        "com_y_filtered": com_y_filtered,
        "velocity_y": velocity_y,
        "acceleration_y": acceleration_y,
        "raw_vgrf_n": raw_vgrf_n,
        "vgrf_n": vgrf_n,
        "body_weight_n": body_weight_n,
        "takeoff_frame": takeoff_frame,
        "landing_frame": landing_frame,
        "lowest_com_frame": lowest_com_frame,
        "metrics": metrics,
    }


def write_grf_trc(time_s, vgrf_n, trc_path, fps=30):
    trc_path = Path(trc_path)
    time_s = np.asarray(time_s, dtype=float)
    vgrf_n = np.asarray(vgrf_n, dtype=float)
    trc_data = pd.DataFrame(
        np.column_stack([
            time_s,
            np.zeros_like(vgrf_n),
            vgrf_n,
            np.zeros_like(vgrf_n),
        ]),
        columns=["time", "GRF", "GRF", "GRF"],
    ).reset_index(drop=True)

    data_rate = camera_rate = orig_data_rate = fps
    num_frames = len(trc_data)
    num_markers = 1
    header_trc = [
        "PathFileType\t4\t(X/Y/Z)\t" + str(trc_path),
        "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames",
        "\t".join(map(str, [data_rate, camera_rate, num_frames, num_markers, "N", orig_data_rate, 0, num_frames])),
        "Frame#\tTime\tGRF\t\t\t",
        "\t\tX1\tY1\tZ1",
    ]

    with open(trc_path, "w", encoding="utf-8") as trc_o:
        for line in header_trc:
            trc_o.write(line + "\n")
        trc_data.to_csv(trc_o, sep="\t", index=True, header=None, lineterminator="\n")


def write_grf_metrics_json(metrics, json_path):
    json_path = Path(json_path)
    with open(json_path, "w", encoding="utf-8") as json_o:
        json.dump(_json_safe_value(metrics), json_o, ensure_ascii=True, indent=2)


def estimate_grf_arrow_anchor_px(
    person_x,
    person_y,
    keypoint_names,
    floor_x_origin=0.0,
    floor_y_origin=None,
    floor_angle=0.0,
):
    keypoint_names = list(keypoint_names)
    left_point = _support_side_point_px(person_x, person_y, keypoint_names, "L")
    right_point = _support_side_point_px(person_x, person_y, keypoint_names, "R")

    support_points = []
    if left_point is not None:
        support_points.append(left_point)
    if right_point is not None:
        support_points.append(right_point)
    if len(support_points) == 0:
        return None
    support_points = np.asarray(support_points, dtype=float)
    anchor_x = float(np.nanmean(support_points[:, 0]))
    support_y = float(np.nanmax(support_points[:, 1]))
    if floor_y_origin is None or not np.isfinite(float(floor_y_origin)):
        anchor_y = support_y
    else:
        floor_y = float(floor_y_origin) - np.tan(float(floor_angle)) * (anchor_x - float(floor_x_origin))
        anchor_y = max(support_y, float(floor_y))
    return tuple(np.round([anchor_x, anchor_y]).astype(int).tolist())


def resolve_vgrf_arrow_base_length_px(frame_height, min_length_px=120.0, height_ratio=(1.0 / 6.0)):
    frame_height = float(frame_height)
    if not np.isfinite(frame_height) or frame_height <= 0:
        return int(round(float(min_length_px)))
    return int(round(max(float(min_length_px), frame_height * float(height_ratio))))


def project_force_to_arrow_length_px(force_n, body_weight_n, base_length_px=120.0, max_visual_bw=3.5):
    if body_weight_n is None or not np.isfinite(body_weight_n) or body_weight_n <= 0:
        return 0
    if force_n is None or not np.isfinite(force_n):
        return 0
    force_ratio = np.clip(float(force_n) / float(body_weight_n), 0.0, float(max_visual_bw))
    return int(round(float(base_length_px) * force_ratio))


def draw_com_proxy_overlay(img, com_point, color=(0, 255, 255), radius=6):
    if com_point is None:
        return img
    x_coord, y_coord = map(int, com_point)
    cv2.circle(img, (x_coord, y_coord), int(max(2, radius)), (255, 255, 255), -1, lineType=cv2.LINE_AA)
    cv2.circle(img, (x_coord, y_coord), int(max(1, radius - 2)), tuple(color), -1, lineType=cv2.LINE_AA)
    return img


def draw_vgrf_arrow_overlay(
    img,
    anchor_point,
    force_n,
    body_weight_n,
    direction=(0.0, -1.0),
    color=(0, 0, 255),
    base_length_px=None,
    max_visual_bw=3.5,
    thickness=3,
):
    if anchor_point is None:
        return img
    if base_length_px is None:
        base_length_px = resolve_vgrf_arrow_base_length_px(img.shape[0])
    arrow_length_px = project_force_to_arrow_length_px(
        force_n,
        body_weight_n,
        base_length_px=base_length_px,
        max_visual_bw=max_visual_bw,
    )
    if arrow_length_px <= 0:
        return img

    direction = np.asarray(direction, dtype=float)
    if not np.all(np.isfinite(direction)) or np.linalg.norm(direction) == 0:
        direction = np.array([0.0, -1.0], dtype=float)
    direction = direction / np.linalg.norm(direction)
    tip_point = np.round(np.asarray(anchor_point, dtype=float) + direction * float(arrow_length_px)).astype(int)
    anchor_point = tuple(int(v) for v in anchor_point)
    tip_point = tuple(int(v) for v in tip_point.tolist())
    stroke_thickness = max(1, int(round(float(thickness) * 2.5)))
    outline_thickness = stroke_thickness + 1
    tip_length = 0.10
    cv2.arrowedLine(img, anchor_point, tip_point, (0, 0, 0), outline_thickness, cv2.LINE_AA, tipLength=tip_length)
    cv2.arrowedLine(img, anchor_point, tip_point, tuple(color), stroke_thickness, cv2.LINE_AA, tipLength=tip_length)
    return img
