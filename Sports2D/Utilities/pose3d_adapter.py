#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Adapter utilities to convert Sports2D 2D keypoints into FMPose3D-ready inputs.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


FMPOSE3D_ROOT = Path(__file__).resolve().parent.parent / "FMPose3D"
if str(FMPOSE3D_ROOT) not in sys.path:
    sys.path.insert(0, str(FMPOSE3D_ROOT))

from fmpose3d.common.camera import normalize_screen_coordinates
from fmpose3d.lib.preprocess import coco_h36m


COCO_17_JOINT_NAMES: Tuple[str, ...] = (
    "Nose",
    "LEye",
    "REye",
    "LEar",
    "REar",
    "LShoulder",
    "RShoulder",
    "LElbow",
    "RElbow",
    "LWrist",
    "RWrist",
    "LHip",
    "RHip",
    "LKnee",
    "RKnee",
    "LAnkle",
    "RAnkle",
)

H36M_17_JOINT_NAMES: Tuple[str, ...] = (
    "Hip",
    "RHip",
    "RKnee",
    "RAnkle",
    "LHip",
    "LKnee",
    "LAnkle",
    "Spine",
    "Thorax",
    "Nose",
    "Head",
    "LShoulder",
    "LElbow",
    "LWrist",
    "RShoulder",
    "RElbow",
    "RWrist",
)

KEYPOINT_ALIASES: Dict[str, Tuple[str, ...]] = {
    "Nose": ("Nose", "nose"),
    "LEye": ("LEye", "left_eye", "LeftEye"),
    "REye": ("REye", "right_eye", "RightEye"),
    "LEar": ("LEar", "left_ear", "LeftEar"),
    "REar": ("REar", "right_ear", "RightEar"),
    "LShoulder": ("LShoulder", "left_shoulder"),
    "RShoulder": ("RShoulder", "right_shoulder"),
    "LElbow": ("LElbow", "left_elbow"),
    "RElbow": ("RElbow", "right_elbow"),
    "LWrist": ("LWrist", "left_wrist"),
    "RWrist": ("RWrist", "right_wrist"),
    "LHip": ("LHip", "left_hip"),
    "RHip": ("RHip", "right_hip"),
    "LKnee": ("LKnee", "left_knee"),
    "RKnee": ("RKnee", "right_knee"),
    "LAnkle": ("LAnkle", "left_ankle"),
    "RAnkle": ("RAnkle", "right_ankle"),
}

MIRROR_FALLBACKS: Dict[str, str] = {
    "LEye": "REye",
    "REye": "LEye",
    "LEar": "REar",
    "REar": "LEar",
    "LShoulder": "RShoulder",
    "RShoulder": "LShoulder",
    "LElbow": "RElbow",
    "RElbow": "LElbow",
    "LWrist": "RWrist",
    "RWrist": "LWrist",
    "LHip": "RHip",
    "RHip": "LHip",
    "LKnee": "RKnee",
    "RKnee": "LKnee",
    "LAnkle": "RAnkle",
    "RAnkle": "LAnkle",
}


def _find_column_name(columns: Sequence[str], canonical_name: str) -> str | None:
    col_lookup = {str(col).lower(): str(col) for col in columns}
    for alias in KEYPOINT_ALIASES.get(canonical_name, (canonical_name,)):
        resolved = col_lookup.get(alias.lower())
        if resolved is not None:
            return resolved
    return None


def _interpolate_fill(arr_1d: np.ndarray) -> np.ndarray:
    series = pd.Series(arr_1d, dtype="float32")
    return (
        series.interpolate(method="linear", limit_direction="both")
        .ffill()
        .bfill()
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )


def _build_coco17_keypoints(
    x_df: pd.DataFrame,
    y_df: pd.DataFrame,
) -> Tuple[np.ndarray, List[str]]:
    nb_frames = len(x_df.index)
    coco_xy = np.full((nb_frames, 17, 2), np.nan, dtype=np.float32)
    missing: List[str] = []
    joint_to_idx = {name: idx for idx, name in enumerate(COCO_17_JOINT_NAMES)}

    for idx, joint_name in enumerate(COCO_17_JOINT_NAMES):
        x_col = _find_column_name(x_df.columns, joint_name)
        y_col = _find_column_name(y_df.columns, joint_name)
        if x_col is None or y_col is None:
            missing.append(joint_name)
            continue
        coco_xy[:, idx, 0] = x_df[x_col].to_numpy(dtype=np.float32)
        coco_xy[:, idx, 1] = y_df[y_col].to_numpy(dtype=np.float32)

    for missing_joint in missing:
        if missing_joint in ("LEye", "REye", "LEar", "REar") and "Nose" not in missing:
            nose_idx = joint_to_idx["Nose"]
            miss_idx = joint_to_idx[missing_joint]
            coco_xy[:, miss_idx, :] = coco_xy[:, nose_idx, :]
            continue

        mirror_joint = MIRROR_FALLBACKS.get(missing_joint)
        if mirror_joint is None or mirror_joint in missing:
            continue
        miss_idx = joint_to_idx[missing_joint]
        mirror_idx = joint_to_idx[mirror_joint]
        coco_xy[:, miss_idx, :] = coco_xy[:, mirror_idx, :]

    for joint_idx in range(coco_xy.shape[1]):
        coco_xy[:, joint_idx, 0] = _interpolate_fill(coco_xy[:, joint_idx, 0])
        coco_xy[:, joint_idx, 1] = _interpolate_fill(coco_xy[:, joint_idx, 1])

    return np.nan_to_num(coco_xy, nan=0.0, posinf=0.0, neginf=0.0), missing


def prepare_fmpose3d_input_from_xy(
    x_df: pd.DataFrame,
    y_df: pd.DataFrame,
    cam_width: int,
    cam_height: int,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Convert Sports2D per-person XY trajectories to normalized H36M-17 input for FMPose3D.

    Returns:
        keypoints_h36m_norm: (T, 17, 2) float32
        valid_frames: 1D indices from coco_h36m
        missing_coco_joints: list of missing source joints
    """

    if cam_width <= 0 or cam_height <= 0:
        raise ValueError(f"Invalid image size for normalization: {cam_width}x{cam_height}")

    coco17_xy, missing = _build_coco17_keypoints(x_df, y_df)
    h36m_xy, valid_frames = coco_h36m(coco17_xy.astype(np.float32))
    h36m_norm = normalize_screen_coordinates(h36m_xy, w=cam_width, h=cam_height)
    h36m_norm = np.nan_to_num(h36m_norm, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return h36m_norm, np.asarray(valid_frames, dtype=np.int32), missing

