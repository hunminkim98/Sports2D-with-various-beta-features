#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

try:
    from scipy.signal import butter, filtfilt, savgol_filter
except ImportError:  # pragma: no cover - Pose2Sim normally provides scipy
    butter = None
    filtfilt = None
    savgol_filter = None


GRAVITY_MPS2 = 9.81
PELVIS_TRUNK_ALPHA = 0.20
GRF_FILTER_CUTOFF_HZ = 6.0
GRF_FILTER_ORDER = 4
GRF_DERIVATIVE_WINDOW_SECONDS = 0.28
GRF_DERIVATIVE_POLYORDER = 2
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
    big_toe_name = f"{side_prefix}BigToe"
    lateral_forefoot_names = (f"{side_prefix}SmallToe", f"{side_prefix}5Meta")
    toe_tip_name = f"{side_prefix}Toe"
    heel_names = (f"{side_prefix}Heel",)
    ankle_names = (f"{side_prefix}Ankle",)

    big_toe_point = _point_from_name_or_pair(
        person_x,
        person_y,
        keypoint_names,
        marker_name=big_toe_name,
        pair_names=(big_toe_name,),
    )
    lateral_forefoot_point = None
    for marker_name in lateral_forefoot_names:
        lateral_forefoot_point = _point_from_name_or_pair(
            person_x,
            person_y,
            keypoint_names,
            marker_name=marker_name,
            pair_names=(marker_name,),
        )
        if lateral_forefoot_point is not None:
            break
    toe_tip_point = _point_from_name_or_pair(
        person_x,
        person_y,
        keypoint_names,
        marker_name=toe_tip_name,
        pair_names=(toe_tip_name,),
    )

    if big_toe_point is not None and lateral_forefoot_point is not None:
        forefoot_point = 0.5 * (big_toe_point + lateral_forefoot_point)
    elif big_toe_point is not None and toe_tip_point is not None:
        forefoot_point = 0.5 * (big_toe_point + toe_tip_point)
    elif big_toe_point is not None:
        forefoot_point = big_toe_point
    elif lateral_forefoot_point is not None:
        forefoot_point = lateral_forefoot_point
    else:
        forefoot_point = toe_tip_point
    heel_point = _point_from_name_or_pair(
        person_x,
        person_y,
        keypoint_names,
        marker_name=heel_names[0],
        pair_names=heel_names,
    )

    # Anchor the GRF arrow at the midpoint of each foot by blending the heel
    # with the forefoot center (big toe + small toe when both exist).
    if forefoot_point is not None and heel_point is not None:
        return 0.5 * (forefoot_point + heel_point)
    if forefoot_point is not None:
        return forefoot_point
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


def _resolve_derivative_window_length(num_samples, fps, window_seconds=GRF_DERIVATIVE_WINDOW_SECONDS):
    num_samples = int(num_samples)
    if num_samples < 5:
        return None
    if fps is None or fps <= 0:
        target = 5
    else:
        target = int(round(float(window_seconds) * float(fps)))
    target = max(5, target)
    if target % 2 == 0:
        target += 1
    max_window = num_samples if num_samples % 2 == 1 else num_samples - 1
    if max_window < 5:
        return None
    return min(target, max_window)


def estimate_com_derivatives(
    com_y_filtered,
    dt,
    fps,
    window_seconds=GRF_DERIVATIVE_WINDOW_SECONDS,
    polyorder=GRF_DERIVATIVE_POLYORDER,
):
    com_y_filtered = np.asarray(com_y_filtered, dtype=float)
    if len(com_y_filtered) <= 1 or dt <= 0:
        zeros = np.zeros_like(com_y_filtered)
        return zeros, zeros

    window_length = _resolve_derivative_window_length(
        len(com_y_filtered), fps, window_seconds=window_seconds
    )
    if (
        savgol_filter is None
        or window_length is None
        or window_length <= int(polyorder)
    ):
        velocity_y = np.gradient(com_y_filtered, dt)
        acceleration_y = np.gradient(velocity_y, dt)
        return velocity_y, acceleration_y

    velocity_y = savgol_filter(
        com_y_filtered,
        window_length=window_length,
        polyorder=int(polyorder),
        deriv=1,
        delta=float(dt),
        mode="interp",
    )
    acceleration_y = savgol_filter(
        com_y_filtered,
        window_length=window_length,
        polyorder=int(polyorder),
        deriv=2,
        delta=float(dt),
        mode="interp",
    )
    return velocity_y, acceleration_y


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


def _marker_triplet_array(trc_data, marker_name):
    marker_data = _marker_triplet(trc_data, marker_name)
    if marker_data is None:
        return None
    return marker_data.to_numpy(dtype=float, copy=True)


def _point_series_from_name_or_pair(trc_data, marker_name, pair_names):
    if marker_name in trc_data.columns:
        return _marker_triplet_array(trc_data, marker_name)

    point_arrays = []
    for name in pair_names:
        point_array = _marker_triplet_array(trc_data, name)
        if point_array is not None:
            point_arrays.append(point_array)
    if len(point_arrays) == 0:
        return None
    if len(point_arrays) == 1:
        return point_arrays[0]
    return np.nanmean(np.stack(point_arrays, axis=0), axis=0)


def _support_side_point_series_m(trc_data_m, side_prefix):
    side_prefix = str(side_prefix).strip().upper()
    big_toe_name = f"{side_prefix}BigToe"
    lateral_forefoot_names = (f"{side_prefix}SmallToe", f"{side_prefix}5Meta")
    toe_tip_name = f"{side_prefix}Toe"
    heel_name = f"{side_prefix}Heel"
    ankle_name = f"{side_prefix}Ankle"

    big_toe_series = _point_series_from_name_or_pair(
        trc_data_m,
        marker_name=big_toe_name,
        pair_names=(big_toe_name,),
    )
    lateral_forefoot_series = None
    for marker_name in lateral_forefoot_names:
        lateral_forefoot_series = _point_series_from_name_or_pair(
            trc_data_m,
            marker_name=marker_name,
            pair_names=(marker_name,),
        )
        if lateral_forefoot_series is not None:
            break
    toe_tip_series = _point_series_from_name_or_pair(
        trc_data_m,
        marker_name=toe_tip_name,
        pair_names=(toe_tip_name,),
    )

    if big_toe_series is not None and lateral_forefoot_series is not None:
        forefoot_series = 0.5 * (big_toe_series + lateral_forefoot_series)
    elif big_toe_series is not None and toe_tip_series is not None:
        forefoot_series = 0.5 * (big_toe_series + toe_tip_series)
    elif big_toe_series is not None:
        forefoot_series = big_toe_series
    elif lateral_forefoot_series is not None:
        forefoot_series = lateral_forefoot_series
    else:
        forefoot_series = toe_tip_series

    heel_series = _point_series_from_name_or_pair(
        trc_data_m,
        marker_name=heel_name,
        pair_names=(heel_name,),
    )
    if forefoot_series is not None and heel_series is not None:
        return 0.5 * (forefoot_series + heel_series)
    if forefoot_series is not None:
        return forefoot_series
    if heel_series is not None:
        return heel_series
    return _point_series_from_name_or_pair(
        trc_data_m,
        marker_name=ankle_name,
        pair_names=(ankle_name,),
    )


def estimate_shared_cop_series_m(trc_data_m):
    trc_data_m = pd.DataFrame(trc_data_m).copy()
    left_series = _support_side_point_series_m(trc_data_m, "L")
    right_series = _support_side_point_series_m(trc_data_m, "R")

    point_series = [series for series in [left_series, right_series] if series is not None]
    if len(point_series) == 0:
        raise ValueError(
            "Inverse dynamics requires support markers (toe/heel or ankle) to estimate CoP."
        )

    if len(point_series) == 1:
        cop_series = np.asarray(point_series[0], dtype=float)
    else:
        cop_series = np.nanmean(np.stack(point_series, axis=0), axis=0)

    cop_series = np.asarray(cop_series, dtype=float)
    if cop_series.ndim != 2 or cop_series.shape[1] < 3:
        raise ValueError("Estimated CoP proxy must resolve to an Nx3 meter-space series.")

    cop_series = cop_series[:, :3].copy()
    for axis_idx in range(cop_series.shape[1]):
        cop_series[:, axis_idx] = _interpolate_nan_series(cop_series[:, axis_idx])

    if not np.all(np.isfinite(cop_series)):
        raise ValueError("Inverse dynamics CoP proxy contains unresolved non-finite values.")

    return cop_series


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
    velocity_y, acceleration_y = estimate_com_derivatives(
        com_y_filtered,
        dt=dt,
        fps=effective_fps,
    )
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


def build_external_loads_mot_data(time_s, total_vgrf_n, cop_xyz_m):
    time_s = np.asarray(time_s, dtype=float).reshape(-1)
    total_vgrf_n = np.asarray(total_vgrf_n, dtype=float).reshape(-1)
    cop_xyz_m = np.asarray(cop_xyz_m, dtype=float)

    if len(time_s) == 0:
        raise ValueError("Inverse dynamics requires a non-empty GRF time series.")
    if len(total_vgrf_n) != len(time_s):
        raise ValueError("Inverse dynamics GRF length must match the provided time series.")
    if cop_xyz_m.ndim != 2 or cop_xyz_m.shape[0] != len(time_s) or cop_xyz_m.shape[1] < 3:
        raise ValueError("Inverse dynamics CoP proxy must be an Nx3 array aligned with time.")
    if not np.all(np.isfinite(time_s)):
        raise ValueError("Inverse dynamics time series contains non-finite values.")
    if not np.all(np.isfinite(total_vgrf_n)):
        raise ValueError("Inverse dynamics GRF contains non-finite values.")
    if not np.all(np.isfinite(cop_xyz_m[:, :3])):
        raise ValueError("Inverse dynamics CoP proxy contains non-finite values.")

    half_force_n = 0.5 * total_vgrf_n
    cop_x = cop_xyz_m[:, 0]
    # ExternalLoads points are expressed in the ground frame, so clamp the
    # vertical coordinate onto the ground plane instead of applying the force
    # at the floating support-midpoint height.
    cop_y = np.zeros_like(total_vgrf_n)
    cop_z = cop_xyz_m[:, 2]
    zeros = np.zeros_like(total_vgrf_n)

    return pd.DataFrame(
        {
            "time": time_s,
            "ground_force_l_vx": zeros,
            "ground_force_l_vy": half_force_n,
            "ground_force_l_vz": zeros,
            "ground_force_l_px": cop_x,
            "ground_force_l_py": cop_y,
            "ground_force_l_pz": cop_z,
            "ground_torque_l_x": zeros,
            "ground_torque_l_y": zeros,
            "ground_torque_l_z": zeros,
            "ground_force_r_vx": zeros,
            "ground_force_r_vy": half_force_n,
            "ground_force_r_vz": zeros,
            "ground_force_r_px": cop_x,
            "ground_force_r_py": cop_y,
            "ground_force_r_pz": cop_z,
            "ground_torque_r_x": zeros,
            "ground_torque_r_y": zeros,
            "ground_torque_r_z": zeros,
        }
    )


def write_opensim_mot(data, mot_path, name="OpenSimData", in_degrees=False):
    data = pd.DataFrame(data).copy()
    if "time" not in data.columns:
        raise ValueError("OpenSim MOT export requires a leading 'time' column.")

    mot_path = Path(mot_path)
    n_rows = len(data)
    n_columns = data.shape[1]
    header_mot = [
        str(name),
        "version=1",
        f"nRows={n_rows}",
        f"nColumns={n_columns}",
        f"inDegrees={'yes' if in_degrees else 'no'}",
        "",
        "Units are S.I. units (second, meters, Newtons, ...)",
        "If the header above contains a line with 'inDegrees', this indicates whether rotational values are in degrees (yes) or radians (no).",
        "",
        "endheader",
        "	".join(map(str, data.columns.tolist())),
    ]

    with open(mot_path, "w", encoding="utf-8") as mot_o:
        for line in header_mot:
            mot_o.write(line + "\n")
        data.to_csv(mot_o, sep="\t", index=False, header=None, lineterminator="\n")

    return data


def read_opensim_storage_file(storage_path):
    storage_path = Path(storage_path)
    with open(storage_path, "r", encoding="utf-8") as storage_i:
        lines = storage_i.readlines()

    header_end_idx = None
    for idx, line in enumerate(lines):
        if line.strip().lower() == "endheader":
            header_end_idx = idx
            break
    if header_end_idx is None or header_end_idx + 1 >= len(lines):
        raise ValueError(f"Could not parse OpenSim storage header from {storage_path}.")

    header_line_idx = header_end_idx + 1
    data_lines = [line for line in lines[header_line_idx:] if line.strip()]
    if len(data_lines) == 0:
        return pd.DataFrame()

    column_names = data_lines[0].strip().split()
    rows = [line.strip().split() for line in data_lines[1:]]
    if len(rows) == 0:
        return pd.DataFrame(columns=column_names)
    return pd.DataFrame(rows, columns=column_names).apply(pd.to_numeric, errors="coerce")


def select_joint_contribution_columns(id_columns):
    columns = list(id_columns) if id_columns is not None else []
    hip_columns = []
    knee_columns = []

    for column_name in columns:
        col_lower = str(column_name).lower()
        if "hip" in col_lower and ("flexion" in col_lower or "moment" in col_lower):
            hip_columns.append(column_name)
        if "knee" in col_lower and (
            "angle" in col_lower or "flexion" in col_lower or "moment" in col_lower
        ):
            knee_columns.append(column_name)

    if len(hip_columns) == 0:
        hip_columns = [column_name for column_name in columns if "hip" in str(column_name).lower()]
    if len(knee_columns) == 0:
        knee_columns = [column_name for column_name in columns if "knee" in str(column_name).lower()]

    return hip_columns, knee_columns


def select_sagittal_joint_contribution_columns(id_columns):
    columns = list(id_columns) if id_columns is not None else []
    column_lookup = {str(column_name).lower(): column_name for column_name in columns}

    hip_required = [
        "hip_flexion_r_moment",
        "hip_flexion_l_moment",
    ]
    knee_required = [
        "knee_angle_r_moment",
        "knee_angle_l_moment",
    ]

    hip_columns = [
        column_lookup[column_name]
        for column_name in hip_required
        if column_name in column_lookup
    ]
    knee_columns = [
        column_lookup[column_name]
        for column_name in knee_required
        if column_name in column_lookup
    ]

    return hip_columns, knee_columns


def _calculate_joint_contribution_from_selected_columns(
    id_df,
    hip_columns,
    knee_columns,
    start_frame,
    end_frame,
    frame_rate,
    definition=None,
):
    id_df = pd.DataFrame(id_df).copy()
    if "time" not in id_df.columns:
        raise ValueError("Joint contribution calculation requires a time column in the ID storage.")
    if frame_rate is None or float(frame_rate) <= 0:
        raise ValueError("Joint contribution calculation requires a positive frame_rate.")
    if len(hip_columns) == 0 or len(knee_columns) == 0:
        raise ValueError(
            f"Could not find hip/knee columns. Available: {list(id_df.columns)}"
        )

    times = np.asarray(id_df["time"], dtype=float)
    if len(times) == 0:
        raise ValueError("Joint contribution calculation requires non-empty ID data.")

    dt = 1.0 / float(frame_rate)
    start_time = float(start_frame) * dt
    end_time = float(end_frame) * dt
    start_idx = int(np.searchsorted(times, start_time))
    end_idx = int(np.searchsorted(times, end_time))
    if end_idx <= start_idx:
        end_idx = len(times)

    hip_moments_sum = np.zeros(len(times), dtype=float)
    for column_name in hip_columns:
        hip_moments_sum += np.abs(np.asarray(id_df[column_name], dtype=float))

    knee_moments_sum = np.zeros(len(times), dtype=float)
    for column_name in knee_columns:
        knee_moments_sum += np.abs(np.asarray(id_df[column_name], dtype=float))

    phase_times = times[start_idx:end_idx]
    hip_contraction = hip_moments_sum[start_idx:end_idx]
    knee_contraction = knee_moments_sum[start_idx:end_idx]
    if len(phase_times) == 0:
        phase_times = times
        hip_contraction = hip_moments_sum
        knee_contraction = knee_moments_sum

    hip_integral = float(np.trapz(hip_contraction, x=phase_times))
    knee_integral = float(np.trapz(knee_contraction, x=phase_times))
    total_integral = hip_integral + knee_integral

    if total_integral > 0:
        hip_pct = (hip_integral / total_integral) * 100.0
        knee_pct = (knee_integral / total_integral) * 100.0
    else:
        hip_pct = 50.0
        knee_pct = 50.0

    dominant_strategy = "Hip Dominant" if hip_pct > knee_pct else "Knee Dominant"

    result = {
        "success": True,
        "hip_columns": hip_columns,
        "knee_columns": knee_columns,
        "hip_moment_integral_Nms": hip_integral,
        "knee_moment_integral_Nms": knee_integral,
        "hip_contribution_pct": round(hip_pct, 1),
        "knee_contribution_pct": round(knee_pct, 1),
        "dominant_strategy": dominant_strategy,
        "contraction_start_frame": int(start_frame),
        "contraction_end_frame": int(end_frame),
        "frame_rate_hz": float(frame_rate),
    }
    if definition is not None:
        result["definition"] = definition
    return result


def calculate_joint_contribution_from_id_storage(
    id_df,
    start_frame,
    end_frame,
    frame_rate,
):
    hip_columns, knee_columns = select_joint_contribution_columns(id_df.columns)
    return _calculate_joint_contribution_from_selected_columns(
        id_df,
        hip_columns,
        knee_columns,
        start_frame,
        end_frame,
        frame_rate,
    )


def calculate_sagittal_joint_contribution_from_id_storage(
    id_df,
    start_frame,
    end_frame,
    frame_rate,
):
    columns = list(pd.DataFrame(id_df).columns)
    column_lookup = {str(column_name).lower(): column_name for column_name in columns}
    required_columns = [
        "hip_flexion_r_moment",
        "hip_flexion_l_moment",
        "knee_angle_r_moment",
        "knee_angle_l_moment",
    ]
    missing_columns = [
        column_name for column_name in required_columns if column_name not in column_lookup
    ]
    if missing_columns:
        raise ValueError(
            "Sagittal joint contribution requires columns: "
            + ", ".join(missing_columns)
        )

    hip_columns, knee_columns = select_sagittal_joint_contribution_columns(columns)
    return _calculate_joint_contribution_from_selected_columns(
        id_df,
        hip_columns,
        knee_columns,
        start_frame,
        end_frame,
        frame_rate,
        definition="sagittal_only",
    )


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
    anchor_point = np.nanmean(support_points, axis=0)
    if not np.all(np.isfinite(anchor_point)):
        return None
    return tuple(np.round(anchor_point).astype(int).tolist())


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
