#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Fix OpenMP runtime conflict: multiple copies of libiomp5md.dll
# This occurs when PyTorch, NumPy(MKL), onnxruntime, etc. load their own OpenMP runtime
# Must be set BEFORE importing numpy, torch, or any library that uses OpenMP
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


"""
    ##############################################################
    ## Compute pose and angles from video or webcam input       ##
    ##############################################################
    
    Detects 2D joint centers from a video or a webcam with RTMLib.
    Computes selected joint and segment angles. 
    Optionally saves processed image files and video file.
    Optionally saves processed poses as a TRC file, and angles as a MOT file (OpenSim compatible).

    This scripts:
    - loads skeleton information
    - reads stream from a video or a webcam
    - sets up the RTMLib pose tracker from RTMlib with specified parameters
    - detects poses within the selected time range
    - tracks people so that their IDs are consistent across frames
    - retrieves the keypoints with high enough confidence, and only keeps the persons with enough high-confidence keypoints
    - computes joint and segment angles, and flips those on the left/right side them if the respective foot is pointing to the left
    - draws bounding boxes around each person with their IDs
    - draws joint and segment angles on the body, and writes the values either near the joint/segment, or on the upper-left of the image with a progress bar
    - draws the skeleton and the keypoints, with a green to red color scale to account for their confidence
    - optionally show processed images, saves them, or saves them as a video
    - interpolates missing pose and angle sequences if gaps are not too large
    - filters them with the selected filter and parameters
    - optionally plots pose and angle data before and after processing for comparison
    - optionally saves poses for each person as a trc file, and angles as a mot file
        
    ⚠ Warning ⚠ 
    - The pose detection is only as good as the pose estimation algorithm, i.e., it is not perfect.
    - It will lead to reliable results only if the persons move in the 2D plane (sagittal or frontal plane).
    - The persons need to be filmed as perpendicularly as possible from their side.
    If you need research-grade markerless joint kinematics, consider using several cameras,
    and constraining angles to a biomechanically accurate model. See Pose2Sim for example: 
    https://github.com/perfanalytics/pose2sim
        
    INPUTS:
    - a video or a webcam
    - a dictionary obtained from a configuration file (.toml extension)
    - a skeleton model
    
    OUTPUTS:
    - one trc file of joint coordinates per detected person
    - one mot file of joint angles per detected person
    - image files, video
    - a logs.txt file 
"""


## INIT
from pathlib import Path
import sys
import logging
import json
import ast
import copy
import shutil
import os
import re
import platform
import time
import tempfile
import unicodedata
from importlib.metadata import version
from datetime import datetime
import itertools as it
from tqdm import tqdm
from collections import defaultdict
from anytree import RenderTree

import numpy as np
import pandas as pd
import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib import patheffects
from rtmlib.tools.object_detection.post_processings import nms

# SynthPose integration - lazy import to avoid dependency issues
SYNTHPOSE_AVAILABLE = False
try:
    from Sports2D.Utilities.synthpose_skeleton import (
        create_synthpose_skeleton,
        SYNTHPOSE_MARKER_ALIASES,
        SYNTHPOSE_KEYPOINT_NAMES,
        SYNTHPOSE_SKELETON_LINKS,
    )

    SYNTHPOSE_AVAILABLE = True
except ImportError:
    SYNTHPOSE_MARKER_ALIASES = {}

# Unified pose backend abstraction
from Sports2D.Utilities.pose_backend import create_pose_backend
from Sports2D.Utilities.realtime_display import create_realtime_display
from Sports2D.Utilities.hybrid_editor import (
    apply_ball_override_to_tracks,
    augment_pose_arrays_with_derived_keypoints,
    evaluate_pose_frame,
    review_ball_sequence,
    review_pose_sequence,
)
from Sports2D.Utilities.manual_roi import select_manual_rois
from Sports2D.Utilities.ball_blender import write_ball_blender_helper
from Sports2D.Utilities.sam3_detector import (
    PERSON_CLASS_ID as SAM3_PERSON_CLASS_ID,
    SPORTS_BALL_CLASS_ID as SAM3_BALL_CLASS_ID,
)
from Sports2D.Utilities.motion import (
    analyze_vertical_jump_trial,
    build_external_loads_mot_data,
    calculate_joint_contribution_from_id_storage,
    calculate_sagittal_joint_contribution_from_id_storage,
    draw_com_proxy_overlay,
    draw_vgrf_arrow_overlay,
    estimate_grf_arrow_anchor_px,
    estimate_pelvis_trunk_com_xy_px,
    estimate_shared_cop_series_m,
    read_opensim_storage_file,
    write_grf_metrics_json,
    write_grf_trc,
    write_opensim_mot,
)

from Sports2D.Utilities.common import *
from Pose2Sim.common import *
from Pose2Sim.skeletons import *
from Pose2Sim.calibration import toml_write
from Pose2Sim.poseEstimation import setup_model_class_mode, setup_backend_device
from Pose2Sim.triangulation import indices_of_first_last_non_nan_chunks
from Pose2Sim.personAssociation import *
from Pose2Sim.filtering import *

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
np.set_printoptions(legacy="1.21")  # otherwise prints np.float64(3.0) rather than 3.0
import warnings  # Silence numpy and CoreML warnings

warnings.filterwarnings(
    "ignore", category=RuntimeWarning, message="Mean of empty slice"
)
warnings.filterwarnings(
    "ignore", category=RuntimeWarning, message="All-NaN slice encountered"
)
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message="invalid value encountered in scalar divide",
)
warnings.filterwarnings(
    "ignore",
    message=".*Input.*has a dynamic shape.*but the runtime shape.*has zero elements.*",
)


# Not safe, but to be used until OpenMMLab/RTMlib's SSL certificates are updated
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

# ============================================================
# Pose Drawing Functions
# ============================================================


def _mask_pose_for_drawing(all_X, all_Y, all_scores, threshold):
    """
    Return copies of pose arrays with points below the draw threshold masked out.
    """

    threshold = float(threshold)
    masked_X, masked_Y, masked_scores = [], [], []
    for X, Y, scores in zip(all_X, all_Y, all_scores):
        X_arr = np.asarray(X, dtype=float).copy()
        Y_arr = np.asarray(Y, dtype=float).copy()
        score_arr = np.asarray(scores, dtype=float).copy()
        invalid_mask = (
            np.isnan(X_arr)
            | np.isnan(Y_arr)
            | np.isnan(score_arr)
            | (score_arr < threshold)
        )
        masked_X.append(np.where(invalid_mask, np.nan, X_arr))
        masked_Y.append(np.where(invalid_mask, np.nan, Y_arr))
        masked_scores.append(np.where(invalid_mask, np.nan, score_arr))
    return masked_X, masked_Y, masked_scores


def draw_pose(
    img,
    all_X,
    all_Y,
    all_scores,
    pose_model,
    keypoint_names=None,
    backend_name="rtmlib",
    thickness=1,
    kpt_threshold=0.3,
    keypoint_draw_threshold=None,
    skeleton_draw_threshold=None,
):
    """
    Unified pose drawing function that works with any backend.

    For RTMLib (26/133 keypoints):
        - Uses Pose2Sim's draw_keypts/draw_skel functions

    For SynthPose (52 keypoints):
        - Uses custom styling (HALPE26 colored circles, others white diamonds)

    INPUTS:
    - img: OpenCV image (BGR)
    - all_X, all_Y: List of x,y coordinates per person
    - all_scores: List of confidence scores per person
    - pose_model: Skeleton tree structure (anytree Node)
    - keypoint_names: List of keypoint names (required for SynthPose)
    - backend_name: 'rtmlib' or 'synthpose'
    - thickness: Line thickness
    - kpt_threshold: Legacy keypoint confidence threshold alias
    - keypoint_draw_threshold: Display threshold for keypoint markers
    - skeleton_draw_threshold: Display threshold for skeleton lines

    OUTPUT:
    - img: Image with drawn keypoints and skeleton
    """
    if keypoint_draw_threshold is None:
        keypoint_draw_threshold = float(kpt_threshold)
    if skeleton_draw_threshold is None:
        skeleton_draw_threshold = float(keypoint_draw_threshold)

    draw_X_keypoints, draw_Y_keypoints, draw_scores_keypoints = _mask_pose_for_drawing(
        all_X,
        all_Y,
        all_scores,
        keypoint_draw_threshold,
    )
    draw_X_skeleton, draw_Y_skeleton, draw_scores_skeleton = _mask_pose_for_drawing(
        all_X,
        all_Y,
        all_scores,
        skeleton_draw_threshold,
    )

    if backend_name == "synthpose":
        img = _draw_synthpose_keypoints(
            img,
            draw_X_keypoints,
            draw_Y_keypoints,
            draw_scores_keypoints,
            keypoint_names=keypoint_names,
            thickness=thickness,
            threshold=keypoint_draw_threshold,
        )
        img = _draw_synthpose_skeleton(
            img,
            draw_X_skeleton,
            draw_Y_skeleton,
            draw_scores_skeleton,
            pose_model,
            thickness=thickness,
            threshold=skeleton_draw_threshold,
        )
    else:
        # RTMLib: use Pose2Sim functions
        img = draw_keypts(
            img,
            draw_X_keypoints,
            draw_Y_keypoints,
            draw_scores_keypoints,
            cmap_str="RdYlGn",
        )
        img = draw_skel(img, draw_X_skeleton, draw_Y_skeleton, pose_model)
    return img


def _ensure_xyxy_boxes(boxes):
    """
    Normalize detector boxes to Nx4 xyxy float32.
    """
    if boxes is None:
        return np.empty((0, 4), dtype=np.float32)
    boxes = np.asarray(boxes)
    if boxes.size == 0:
        return np.empty((0, 4), dtype=np.float32)
    if boxes.ndim == 1:
        boxes = boxes.reshape(1, -1)
    if boxes.shape[1] < 4:
        return np.empty((0, 4), dtype=np.float32)
    return boxes[:, :4].astype(np.float32, copy=False)


def _ensure_score_vector(scores, expected_len=None):
    """
    Normalize detector confidence scores to a float32 vector.
    """
    if scores is None:
        score_arr = np.empty((0,), dtype=np.float32)
    else:
        score_arr = np.asarray(scores, dtype=np.float32).reshape(-1)

    if expected_len is None:
        return score_arr

    expected_len = max(0, int(expected_len))
    if expected_len == 0:
        return np.empty((0,), dtype=np.float32)
    if len(score_arr) == expected_len:
        return score_arr
    if len(score_arr) == 0:
        return np.full((expected_len,), np.nan, dtype=np.float32)
    if len(score_arr) > expected_len:
        return score_arr[:expected_len]
    padded = np.full((expected_len,), np.nan, dtype=np.float32)
    padded[: len(score_arr)] = score_arr
    return padded


def _resolve_draw_likelihood_threshold(config_value, fallback):
    """
    Parse a display-only pose draw threshold while preserving a numeric fallback.
    """

    if config_value in [None, ""]:
        return float(fallback)
    try:
        return float(np.clip(float(config_value), 0.0, 1.0))
    except (TypeError, ValueError):
        return float(fallback)


def _ensure_mask_list(masks, expected_len=None):
    """
    Normalize instance masks to a list of 2D arrays.
    """
    if masks is None:
        return []

    if isinstance(masks, np.ndarray):
        if masks.size == 0:
            return []
        if masks.ndim == 2:
            mask_values = [masks]
        elif masks.ndim >= 3:
            mask_values = [masks[i] for i in range(masks.shape[0])]
        else:
            return []
    else:
        try:
            mask_values = list(masks)
        except TypeError:
            return []

    normalized = []
    for mask in mask_values:
        mask_arr = np.asarray(mask)
        if mask_arr.size == 0:
            return []
        mask_arr = np.squeeze(mask_arr)
        if mask_arr.ndim != 2:
            return []
        normalized.append(mask_arr)

    if expected_len is not None and len(normalized) != int(expected_len):
        return []
    return normalized


def draw_sam3_mask_overlay(
    img,
    detection_meta,
    alpha=0.22,
    person_color=(70, 170, 110),
    ball_color=(0, 165, 255),
):
    """
    Draw semi-transparent SAM3 instance masks for person and ball detections.
    """
    detection_meta = detection_meta or {}
    alpha = float(np.clip(alpha, 0.0, 1.0))
    if alpha <= 0.0:
        return img

    classes = np.asarray(detection_meta.get("classes", []), dtype=np.int32).reshape(-1)
    if len(classes) == 0:
        return img

    raw_masks = detection_meta.get("masks")
    masks = _ensure_mask_list(raw_masks, expected_len=len(classes))
    if raw_masks is not None and len(masks) != len(classes):
        logging.debug(
            "Skipping SAM3 mask overlay due to mask/class mismatch: masks=%s classes=%s",
            len(masks),
            len(classes),
        )
        return img

    if len(masks) == 0:
        return img

    img_height, img_width = img.shape[:2]
    overlay = img.copy()
    has_overlay = False

    for class_id, mask in zip(classes, masks):
        if int(class_id) == SAM3_PERSON_CLASS_ID:
            color = person_color
        elif int(class_id) == SAM3_BALL_CLASS_ID:
            color = ball_color
        else:
            continue

        mask_bool = np.asarray(mask) > 0
        if mask_bool.shape != (img_height, img_width):
            mask_bool = cv2.resize(
                mask_bool.astype(np.uint8),
                (img_width, img_height),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        if not mask_bool.any():
            continue

        overlay[mask_bool] = color
        has_overlay = True

    if not has_overlay:
        return img

    return cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0)


def filter_sam3_detection_meta_classes(detection_meta, allowed_class_ids):
    """
    Keep only the requested SAM3 classes from detection metadata.
    """
    detection_meta = detection_meta or {}
    allowed = {int(class_id) for class_id in (allowed_class_ids or [])}
    empty = {
        "boxes": np.empty((0, 4), dtype=np.float32),
        "classes": np.empty((0,), dtype=np.int32),
        "scores": np.empty((0,), dtype=np.float32),
        "person_boxes": np.empty((0, 4), dtype=np.float32),
        "ball_boxes": np.empty((0, 4), dtype=np.float32),
        "ball_scores": np.empty((0,), dtype=np.float32),
        "class_names": np.empty((0,), dtype=object),
        "prompt_indices": np.empty((0,), dtype=np.int32),
        "masks": [],
    }
    if not allowed:
        return empty

    classes = np.asarray(detection_meta.get("classes", []), dtype=np.int32).reshape(-1)
    if len(classes) == 0:
        return empty

    keep_mask = np.isin(classes, list(allowed))
    if not np.any(keep_mask):
        return empty

    boxes = _ensure_xyxy_boxes(detection_meta.get("boxes"))
    scores = _ensure_score_vector(
        detection_meta.get("scores"), expected_len=len(classes)
    )
    class_names = np.asarray(
        detection_meta.get("class_names", []), dtype=object
    ).reshape(-1)
    if len(class_names) != len(classes):
        class_names = np.empty((len(classes),), dtype=object)
    prompt_indices = np.asarray(
        detection_meta.get("prompt_indices", []), dtype=np.int32
    ).reshape(-1)
    if len(prompt_indices) != len(classes):
        prompt_indices = np.full((len(classes),), -1, dtype=np.int32)

    raw_masks = detection_meta.get("masks")
    masks = _ensure_mask_list(raw_masks, expected_len=len(classes))
    filtered_masks = (
        [masks[i] for i, keep in enumerate(keep_mask) if keep]
        if len(masks) == len(classes)
        else []
    )

    filtered_classes = classes[keep_mask].astype(np.int32, copy=False)
    filtered_boxes = (
        boxes[keep_mask]
        if len(boxes) == len(classes)
        else np.empty((0, 4), dtype=np.float32)
    )
    filtered_scores = (
        scores[keep_mask]
        if len(scores) == len(classes)
        else np.empty((0,), dtype=np.float32)
    )
    filtered_names = class_names[keep_mask]
    filtered_prompt_indices = prompt_indices[keep_mask].astype(np.int32, copy=False)
    ball_mask = filtered_classes == SAM3_BALL_CLASS_ID

    return {
        "boxes": filtered_boxes.astype(np.float32, copy=False),
        "classes": filtered_classes,
        "scores": filtered_scores.astype(np.float32, copy=False),
        "person_boxes": np.empty((0, 4), dtype=np.float32),
        "ball_boxes": filtered_boxes[ball_mask].astype(np.float32, copy=False),
        "ball_scores": filtered_scores[ball_mask].astype(np.float32, copy=False),
        "class_names": filtered_names,
        "prompt_indices": filtered_prompt_indices,
        "masks": filtered_masks,
    }


def _sam3_mask_available(mask_meta_or_flag):
    """
    Return whether SAM3 produced ball mask-backed detections for a frame.

    Supports both legacy dict metadata and the lightweight bool flag used by the
    memory-safe export path.
    """
    if isinstance(mask_meta_or_flag, (bool, np.bool_)):
        return bool(mask_meta_or_flag)

    detection_meta = mask_meta_or_flag or {}
    if not isinstance(detection_meta, dict):
        return False

    if len(_ensure_mask_list(detection_meta.get("masks"))) > 0:
        return True
    if len(_ensure_xyxy_boxes(detection_meta.get("ball_boxes"))) > 0:
        return True

    classes = np.asarray(detection_meta.get("classes", []), dtype=np.int32).reshape(-1)
    return bool(np.any(classes == SAM3_BALL_CLASS_ID))


def _json_safe_float(value):
    """
    Convert numeric values to JSON-safe floats, mapping NaN/inf to None.
    """
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    return value


def _json_safe_int(value):
    """
    Convert numeric values to JSON-safe ints, mapping invalid values to None.
    """
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_ball_center(center):
    """
    Normalize an `(x, y)` center to integer pixel coordinates.
    """
    if center is None:
        return None
    center_arr = np.asarray(center, dtype=np.float32).reshape(-1)
    if len(center_arr) < 2 or np.isnan(center_arr[:2]).any():
        return None
    return (
        int(round(float(center_arr[0]))),
        int(round(float(center_arr[1]))),
    )


def _serialize_box_xyxy(box):
    """
    Convert a detector box to a JSON-safe `[x1, y1, x2, y2]` list.
    """
    if box is None:
        return None
    box_arr = _ensure_xyxy_boxes([box])
    if len(box_arr) == 0:
        return None
    return [_json_safe_float(coord) for coord in box_arr[0]]


def _find_ball_track_nearest_center(frame_ball_tracks, center, visible_only=True):
    """
    Find the tracked-ball entry whose center is nearest to the requested center.
    """
    center = _normalize_ball_center(center)
    if center is None:
        return None

    candidate_tracks = []
    for track in frame_ball_tracks or []:
        track_center = _normalize_ball_center(track.get("center"))
        if track_center is None:
            continue
        if visible_only and not track.get("visible", False):
            continue
        candidate_tracks.append((track, track_center))
    if len(candidate_tracks) == 0:
        return None

    center_arr = np.asarray(center, dtype=np.float32)
    best_track = None
    best_distance = None
    for track, track_center in candidate_tracks:
        distance = float(
            np.linalg.norm(center_arr - np.asarray(track_center, dtype=np.float32))
        )
        if best_distance is None or distance < best_distance:
            best_track = track
            best_distance = distance
    return best_track


def _resolve_selected_ball_source_track(
    frame_ball_tracks, selected_track_id=None, center=None, max_recovery_dist=None
):
    """
    Resolve the visible raw track that best matches the selected ball timeline center.
    """
    frame_ball_tracks = frame_ball_tracks or []
    center = _normalize_ball_center(center)
    selected_track_id = _json_safe_int(selected_track_id)
    if center is None:
        return None

    def _distance_to_center(track):
        track_center = _normalize_ball_center((track or {}).get("center"))
        if track_center is None:
            return None
        return float(
            np.linalg.norm(
                np.asarray(track_center, dtype=np.float32)
                - np.asarray(center, dtype=np.float32)
            )
        )

    selected_track = None
    if selected_track_id is not None:
        for track in frame_ball_tracks:
            track_id = _json_safe_int(track.get("id"))
            if track_id == selected_track_id:
                selected_track = track
                break

    if (
        selected_track is not None
        and selected_track.get("visible", False)
        and _normalize_ball_center(selected_track.get("center")) is not None
    ):
        if max_recovery_dist is None:
            return selected_track
        selected_dist = _distance_to_center(selected_track)
        if selected_dist is not None and selected_dist <= float(max_recovery_dist):
            return selected_track

    nearest_track = _find_ball_track_nearest_center(
        frame_ball_tracks,
        center,
        visible_only=True,
    )
    if nearest_track is None:
        return None
    if max_recovery_dist is None:
        return nearest_track
    nearest_dist = _distance_to_center(nearest_track)
    if nearest_dist is None or nearest_dist > float(max_recovery_dist):
        return None
    return nearest_track


def build_ball_export_record(
    frame_ball_center,
    frame_ball_boxes,
    frame_ball_scores,
    frame_ball_tracks=None,
    frame_selected_ball_id=None,
    frame_sam3_ball_mask_meta=None,
    multi_id_tracking=False,
    max_recovery_dist=None,
):
    """
    Build a single-frame export record for the selected ball.
    """
    frame_ball_tracks = frame_ball_tracks or []
    ball_boxes = _ensure_xyxy_boxes(frame_ball_boxes)
    ball_scores = _ensure_score_vector(frame_ball_scores, expected_len=len(ball_boxes))
    center = _normalize_ball_center(frame_ball_center)
    mask_available = _sam3_mask_available(frame_sam3_ball_mask_meta)

    record = {
        "visible": False,
        "track_id": _json_safe_int(frame_selected_ball_id),
        "source_track_id": None,
        "score": None,
        "center_xy": None,
        "box_xyxy": None,
        "ball_keypoints_2d": [None, None, None],
        "mask_available": bool(mask_available),
    }

    if multi_id_tracking:
        source_track = _resolve_selected_ball_source_track(
            frame_ball_tracks,
            selected_track_id=frame_selected_ball_id,
            center=center,
            max_recovery_dist=max_recovery_dist,
        )
        if source_track is not None:
            source_track_id = _json_safe_int(source_track.get("id"))
            source_track_center = _normalize_ball_center(source_track.get("center"))
            if source_track_center is None:
                source_track_center = center
            if source_track_center is None or not source_track.get("visible", False):
                return record
            center = source_track_center
            score = _json_safe_float(source_track.get("score"))
            if record["track_id"] is None:
                record["track_id"] = source_track_id
            record["source_track_id"] = source_track_id
            record["visible"] = True
            record["center_xy"] = list(center) if center is not None else None
            record["box_xyxy"] = _serialize_box_xyxy(source_track.get("box"))
            record["score"] = score
            record["ball_keypoints_2d"] = [
                _json_safe_float(center[0]) if center is not None else None,
                _json_safe_float(center[1]) if center is not None else None,
                score,
            ]
            return record

    if center is None:
        return record

    record["visible"] = True
    record["center_xy"] = [int(center[0]), int(center[1])]

    if len(ball_boxes) > 0:
        candidate_centers = np.asarray(
            extract_ball_centers({"ball_boxes": ball_boxes}),
            dtype=np.float32,
        ).reshape(-1, 2)
        if len(candidate_centers) > 0:
            center_arr = np.asarray(center, dtype=np.float32)
            nearest_idx = int(
                np.argmin(
                    np.linalg.norm(candidate_centers - center_arr[None, :], axis=1)
                )
            )
            record["box_xyxy"] = _serialize_box_xyxy(ball_boxes[nearest_idx])
            record["score"] = _json_safe_float(
                ball_scores[nearest_idx] if nearest_idx < len(ball_scores) else None
            )

    record["ball_keypoints_2d"] = [
        _json_safe_float(center[0]),
        _json_safe_float(center[1]),
        record["score"],
    ]
    return record


def build_ball_export_series(
    all_frames_time,
    all_frames_ball_centers,
    all_frames_ball_boxes,
    all_frames_ball_scores,
    all_frames_ball_tracks,
    all_frames_selected_ball_ids,
    all_frames_sam3_ball_mask_meta,
    frame_offset=0,
    multi_id_tracking=False,
    max_recovery_dist=None,
):
    """
    Build frame-aligned export records for the selected ball.
    """
    frame_offset = int(frame_offset)
    entries = []
    frame_count = len(all_frames_time)
    for frame_idx in range(frame_count):
        entries.append(
            {
                "frame_index": frame_offset + frame_idx,
                "time": _json_safe_float(
                    all_frames_time.iloc[frame_idx]
                    if frame_idx < len(all_frames_time)
                    else None
                ),
                "ball": build_ball_export_record(
                    all_frames_ball_centers[frame_idx]
                    if frame_idx < len(all_frames_ball_centers)
                    else None,
                    all_frames_ball_boxes[frame_idx]
                    if frame_idx < len(all_frames_ball_boxes)
                    else None,
                    all_frames_ball_scores[frame_idx]
                    if frame_idx < len(all_frames_ball_scores)
                    else None,
                    frame_ball_tracks=all_frames_ball_tracks[frame_idx]
                    if frame_idx < len(all_frames_ball_tracks)
                    else None,
                    frame_selected_ball_id=all_frames_selected_ball_ids[frame_idx]
                    if frame_idx < len(all_frames_selected_ball_ids)
                    else None,
                    frame_sam3_ball_mask_meta=all_frames_sam3_ball_mask_meta[frame_idx]
                    if frame_idx < len(all_frames_sam3_ball_mask_meta)
                    else None,
                    multi_id_tracking=multi_id_tracking,
                    max_recovery_dist=max_recovery_dist,
                ),
            }
        )
    return entries


def write_ball_pose_json(ball_export_series, pose_ball_output_dir, output_prefix):
    """
    Write per-frame ball JSON files into the pose_ball directory.
    """
    pose_ball_output_dir.mkdir(parents=True, exist_ok=True)
    for entry in ball_export_series:
        frame_index = int(entry.get("frame_index", 0))
        json_path = pose_ball_output_dir / f"{output_prefix}_{frame_index:06d}.json"
        payload = {
            "version": 1.0,
            "frame_index": frame_index,
            "time": entry.get("time"),
            "balls": [entry.get("ball", {})],
        }
        with open(json_path, "w", encoding="utf-8") as json_o:
            json.dump(payload, json_o, ensure_ascii=True, indent=2)


def build_ball_trc_data(ball_export_series, index=None, marker_name="ball"):
    """
    Build a 3-column TRC marker table `(X, Y, Z)` from ball export records.
    """
    if index is None:
        index = range(len(ball_export_series))
    length = len(index)
    x = np.full((length,), np.nan, dtype=np.float64)
    y = np.full((length,), np.nan, dtype=np.float64)
    z = np.full((length,), np.nan, dtype=np.float64)

    for row_idx in range(min(length, len(ball_export_series))):
        center = (ball_export_series[row_idx] or {}).get("ball", {}).get("center_xy")
        if center is None or len(center) < 2:
            continue
        if center[0] is None or center[1] is None:
            continue
        x[row_idx] = float(center[0])
        y[row_idx] = float(center[1])
        z[row_idx] = 0.0

    return pd.DataFrame(
        np.column_stack([x, y, z]),
        index=index,
        columns=[marker_name, marker_name, marker_name],
    )


def append_ball_marker_to_trc_data(trc_data, ball_trc_data, marker_name="ball"):
    """
    Return a copy of TRC data with an extra trailing ball marker triplet.
    """
    if ball_trc_data is None or len(ball_trc_data) == 0:
        return trc_data.copy()

    ball_trc_data = pd.DataFrame(ball_trc_data).copy()
    if ball_trc_data.shape[1] < 3:
        return trc_data.copy()

    aligned_ball = ball_trc_data.reindex(trc_data.index).iloc[:, :3].copy()
    aligned_ball.columns = [marker_name, marker_name, marker_name]
    return pd.concat([trc_data.copy(), aligned_ball], axis=1)


def build_public_meter_trc_data(
    trc_data, marker_aliases=None, ball_trc_data=None, marker_name="ball"
):
    """
    Build the public meter-space TRC table without trimming its original row count.

    Internal consumers such as jump analysis and IK may still use a trimmed/rebased
    subset, but the final exported `_m.trc` should preserve the same sample count
    and time axis as the source px-space TRC whenever meter export is enabled.
    """

    public_trc_data = append_trc_marker_aliases(
        trc_data,
        marker_aliases=marker_aliases,
    )
    if ball_trc_data is not None and len(ball_trc_data) > 0:
        public_trc_data = append_ball_marker_to_trc_data(
            public_trc_data,
            ball_trc_data,
            marker_name=marker_name,
        )
    return public_trc_data


def append_trc_marker_aliases(trc_data, marker_aliases=None):
    """
    Return a copy of TRC data with aliased marker triplets appended.

    This is used to expose SynthPose foot markers under the HALPE/OpenPose
    names expected by Pose2Sim marker augmentation without changing the
    underlying runtime keypoint schema.
    """
    marker_aliases = dict(marker_aliases or {})
    if len(marker_aliases) == 0:
        return trc_data.copy()

    trc_data = pd.DataFrame(trc_data).copy()
    marker_names = list(trc_data.columns[1::3]) if len(trc_data.columns) > 1 else []
    existing_markers = {str(name) for name in marker_names}
    alias_triplets = []

    for source_name, alias_name in marker_aliases.items():
        if source_name not in existing_markers or alias_name in existing_markers:
            continue
        source_triplet = trc_data.loc[:, trc_data.columns == source_name]
        if source_triplet.shape[1] < 3:
            continue
        alias_triplet = source_triplet.iloc[:, :3].copy()
        alias_triplet.columns = [alias_name, alias_name, alias_name]
        alias_triplets.append(alias_triplet)
        existing_markers.add(alias_name)

    if len(alias_triplets) == 0:
        return trc_data
    return pd.concat([trc_data] + alias_triplets, axis=1)


BODY_WITH_FEET_OPENSIM_BRIDGE_MARKERS = [
    "Hip",
    "RHip",
    "RKnee",
    "RAnkle",
    "RBigToe",
    "RSmallToe",
    "RHeel",
    "LHip",
    "LKnee",
    "LAnkle",
    "LBigToe",
    "LSmallToe",
    "LHeel",
    "Neck",
    "Head",
    "Nose",
    "RShoulder",
    "RElbow",
    "RWrist",
    "LShoulder",
    "LElbow",
    "LWrist",
]


def _build_body_with_feet_opensim_bridge_trc_data(trc_data):
    """
    Build the original 22-marker body_with_feet TRC contract for the OpenSim bridge.
    """

    trc_data = pd.DataFrame(trc_data).copy()
    bridge_parts = [trc_data.iloc[:, :1].copy()]
    marker_names = list(trc_data.columns[1::3]) if len(trc_data.columns) > 1 else []
    existing_markers = {str(name) for name in marker_names}

    missing_markers = [
        marker_name
        for marker_name in BODY_WITH_FEET_OPENSIM_BRIDGE_MARKERS
        if marker_name not in existing_markers
    ]
    if missing_markers:
        raise ValueError(
            "body_with_feet OpenSim bridge TRC is missing required markers: "
            + ", ".join(missing_markers)
        )

    for marker_name in BODY_WITH_FEET_OPENSIM_BRIDGE_MARKERS:
        marker_triplet = trc_data.loc[:, trc_data.columns == marker_name]
        if marker_triplet.shape[1] < 3:
            raise ValueError(
                f"body_with_feet OpenSim bridge marker '{marker_name}' does not have a full XYZ triplet."
            )
        marker_triplet = marker_triplet.iloc[:, :3].copy()
        marker_triplet.columns = [marker_name, marker_name, marker_name]
        bridge_parts.append(marker_triplet)

    return pd.concat(bridge_parts, axis=1)


def _resolve_opensim_bridge_trc_data(pose_model_name, trc_data):
    """
    Return the TRC schema that should be staged into Pose2Sim/OpenSim.
    """

    normalized_name = str(pose_model_name or "").strip().lower()
    if normalized_name == "body_with_feet":
        return _build_body_with_feet_opensim_bridge_trc_data(trc_data)
    return pd.DataFrame(trc_data).copy()


def _resolve_meter_conversion_trc_data(pose_model_name, trc_data):
    """
    Return the TRC schema that should drive px->meter conversion and meter exports.

    `body_with_feet` originally used the sparse 22-marker contract all the way
    through height estimation, px->meter conversion, public `_m.trc` export,
    and OpenSim staging. Keep that schema here so scaling matches the upstream
    repo instead of converting the current dense 26-marker HALPE view first.
    """

    normalized_name = str(pose_model_name or "").strip().lower()
    if normalized_name == "body_with_feet":
        return _build_body_with_feet_opensim_bridge_trc_data(trc_data)
    return pd.DataFrame(trc_data).copy()


def _resolve_pose2sim_pose_model_name(pose_model_name):
    """
    Translate Sports2D runtime pose-model names to Pose2Sim/OpenSim bridge names.

    Sports2D can run full SynthPose 52-keypoint inference at runtime, but the
    Pose2Sim kinematics stack only recognizes its own canonical skeleton names
    such as HALPE_26 and COCO_133. The bridge must therefore remap runtime
    model names before marker augmentation or inverse kinematics starts.
    """

    normalized_name = str(pose_model_name or "body_with_feet").strip().lower()
    pose2sim_model_names = {
        "body_with_feet": "HALPE_26",
        "whole_body_wrist": "COCO_133_WRIST",
        "whole_body": "COCO_133",
        "body": "COCO_17",
        "hand": "HAND_21",
        "face": "FACE_106",
        "animal": "ANIMAL2D_17",
        "synthpose": "HALPE_26",
        "synthpose_base": "HALPE_26",
    }
    return pose2sim_model_names.get(
        normalized_name, str(pose_model_name or "BODY_WITH_FEET").strip().upper()
    )


def _configure_pose2sim_kinematics_bridge(
    pose2sim_config_dict, pose_model_name, feet_on_floor
):
    """
    Apply Sports2D-to-Pose2Sim bridge settings needed for marker augmentation / IK.
    """

    pose2sim_config_dict["markerAugmentation"]["feet_on_floor"] = feet_on_floor
    resolved_pose_model_name = _resolve_pose2sim_pose_model_name(pose_model_name)
    pose2sim_config_dict["pose"]["pose_model"] = resolved_pose_model_name
    return resolved_pose_model_name


ESTIMATED_GRF_BODY_NAMES = {
    "l": "calcn_l",
    "r": "calcn_r",
}


def _resolve_inverse_dynamics_requested(kinematics_cfg):
    kinematics_cfg = dict(kinematics_cfg or {})
    if "inverse_dynamics" in kinematics_cfg:
        return bool(kinematics_cfg.get("inverse_dynamics"))
    if "Inverse_Dynamics" in kinematics_cfg:
        return bool(kinematics_cfg.get("Inverse_Dynamics"))
    return False


def _resolve_inverse_dynamics_gate(
    inverse_dynamics_requested,
    do_ik,
    vertical_jump_requested,
    vertical_jump_enabled,
    to_meters,
    save_angles,
    calculate_angles,
):
    if not inverse_dynamics_requested:
        return False, None
    if not do_ik:
        return False, "kinematics.inverse_dynamics=true requires kinematics.do_ik=true. Skipping inverse dynamics."
    if not to_meters:
        return False, "kinematics.inverse_dynamics=true requires px_to_meters_conversion.to_meters=true. Skipping inverse dynamics."
    if not vertical_jump_requested:
        return False, "kinematics.inverse_dynamics=true requires motion.vertical_jump=true. Skipping inverse dynamics."
    if not vertical_jump_enabled:
        return False, "kinematics.inverse_dynamics=true could not use the GRF estimator for this run. Skipping inverse dynamics."
    if not save_angles or not calculate_angles:
        return False, "kinematics.inverse_dynamics=true requires save_angles=true and calculate_angles=true so inverse kinematics produces a motion file. Skipping inverse dynamics."
    return True, None


def _resolve_inverse_dynamics_cop_series(trc_data_m_export_i, inverse_dynamics_enabled):
    if not inverse_dynamics_enabled:
        return None
    return estimate_shared_cop_series_m(trc_data_m_export_i)


def _resolve_inverse_dynamics_artifact_paths(ik_mot_path):
    ik_mot_path = Path(ik_mot_path)
    if ik_mot_path.name.endswith("_ik.mot"):
        base_stem = ik_mot_path.stem[:-3]
    else:
        base_stem = ik_mot_path.stem
    return {
        "grf_mot": ik_mot_path.parent / f"{base_stem}_grf.mot",
        "external_loads_xml": ik_mot_path.parent / f"{base_stem}_ExternalLoads.xml",
        "inverse_dynamics_sto": ik_mot_path.parent / f"{base_stem}_id.sto",
        "metadata_json": ik_mot_path.parent / f"{base_stem}_id_metadata.json",
    }


def _resolve_inverse_dynamics_workspace_stem(trc_name, opensim_workspace_info):
    original_stem = Path(trc_name).stem
    if opensim_workspace_info is None:
        return original_stem

    staged_stem_map = dict(opensim_workspace_info.get("staged_stem_map", {}))
    for staged_stem, restored_stem in staged_stem_map.items():
        if restored_stem == original_stem:
            return staged_stem
    return original_stem


def _serialize_inverse_dynamics_cop_series(cop_xyz_m):
    if cop_xyz_m is None:
        return None
    cop_xyz_m = np.asarray(cop_xyz_m, dtype=float)
    if cop_xyz_m.ndim != 2 or cop_xyz_m.shape[1] < 3:
        return None
    cop_xyz_m = cop_xyz_m[:, :3].copy()
    if not np.all(np.isfinite(cop_xyz_m)):
        return None
    return cop_xyz_m


def _build_inverse_dynamics_metadata_payload(
    *,
    trc_name,
    ik_motion_file,
    scaled_model_file,
    external_loads_mot_file,
    external_loads_xml_file,
    inverse_dynamics_file,
    metrics,
    success,
    error=None,
):
    metadata = {
        "source": "estimated_vertical_grf",
        "assumptions": {
            "vertical_axis": "y",
            "bilateral_force_split": "50:50",
            "shared_cop_proxy": True,
            "horizontal_forces_zero": True,
            "free_torques_zero": True,
        },
        "trc_name": trc_name,
        "ik_motion_file": ik_motion_file,
        "scaled_model_file": scaled_model_file,
        "external_loads_mot_file": external_loads_mot_file,
        "external_loads_xml_file": external_loads_xml_file,
        "inverse_dynamics_file": inverse_dynamics_file,
        "metrics": metrics,
        "success": bool(success),
    }
    if error:
        metadata["error"] = str(error)
    return metadata


def _populate_joint_contribution_metadata(
    metadata,
    id_storage_df,
    *,
    start_frame,
    end_frame,
    frame_rate,
):
    try:
        metadata["joint_contribution"] = calculate_joint_contribution_from_id_storage(
            id_storage_df,
            start_frame=start_frame,
            end_frame=end_frame,
            frame_rate=frame_rate,
        )
    except Exception as joint_contribution_exc:
        metadata["joint_contribution"] = {
            "success": False,
            "error": str(joint_contribution_exc),
        }

    try:
        metadata["joint_contribution_sagittal"] = (
            calculate_sagittal_joint_contribution_from_id_storage(
                id_storage_df,
                start_frame=start_frame,
                end_frame=end_frame,
                frame_rate=frame_rate,
            )
        )
    except Exception as sagittal_exc:
        metadata["joint_contribution_sagittal"] = {
            "success": False,
            "definition": "sagittal_only",
            "error": str(sagittal_exc),
        }

    return metadata


def _write_estimated_grf_external_loads_xml(osim, xml_path, grf_mot_path):
    xml_path = Path(xml_path)
    grf_mot_path = Path(grf_mot_path)

    external_loads = osim.ExternalLoads()
    external_loads.setName("estimated_vertical_grf")
    external_loads.setDataFileName(grf_mot_path.name)

    for side in ["l", "r"]:
        external_force = osim.ExternalForce()
        external_force.setName(f"EstimatedGRF_{side.upper()}")
        external_force.setAppliedToBodyName(ESTIMATED_GRF_BODY_NAMES[side])
        external_force.setForceExpressedInBodyName("ground")
        external_force.setPointExpressedInBodyName("ground")
        external_force.setForceIdentifier(f"ground_force_{side}_v")
        external_force.setPointIdentifier(f"ground_force_{side}_p")
        external_force.setTorqueIdentifier(f"ground_torque_{side}_")
        external_force.set_appliesForce(True)
        external_loads.cloneAndAppend(external_force)

    external_loads.printToXML(str(xml_path))
    return xml_path


def _run_estimated_grf_inverse_dynamics(
    osim,
    model_path,
    ik_mot_path,
    external_loads_xml_path,
    output_sto_path,
    start_time,
    end_time,
):
    model_path = Path(model_path)
    ik_mot_path = Path(ik_mot_path)
    external_loads_xml_path = Path(external_loads_xml_path)
    output_sto_path = Path(output_sto_path)

    id_tool = osim.InverseDynamicsTool()
    id_tool.setModelFileName(str(model_path))
    id_tool.setCoordinatesFileName(str(ik_mot_path))
    id_tool.setExternalLoadsFileName(str(external_loads_xml_path))
    id_tool.setStartTime(float(start_time))
    id_tool.setEndTime(float(end_time))
    id_tool.setResultsDir(str(output_sto_path.parent))
    id_tool.setOutputGenForceFileName(output_sto_path.name)
    id_tool.run()
    return output_sto_path


def _contains_non_ascii_path(path_like):
    """
    Return True when the provided path contains non-ASCII characters.
    """

    return any(ord(char) > 127 for char in str(path_like))


def _sanitize_ascii_opensim_stem(value, fallback="opensim_item"):
    """
    Normalize a filename stem to a conservative ASCII-only representation.
    """

    normalized = (
        unicodedata.normalize("NFKD", str(value))
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", normalized).strip("._-")
    return normalized or str(fallback)


def _should_use_ascii_safe_opensim_workspace(output_dir, trc_files):
    """
    Return True when OpenSim should avoid the native output path.
    """

    return _contains_non_ascii_path(output_dir) or any(
        _contains_non_ascii_path(trc_file) for trc_file in (trc_files or [])
    )


def _restore_opensim_artifact_name(file_name, staged_stem_map):
    """
    Convert an ASCII-staged OpenSim artifact name back to its original stem.
    """

    artifact_path = Path(file_name)
    for staged_stem, original_stem in sorted(
        staged_stem_map.items(), key=lambda item: len(item[0]), reverse=True
    ):
        if artifact_path.stem == staged_stem or artifact_path.stem.startswith(
            staged_stem + "_"
        ):
            restored_stem = original_stem + artifact_path.stem[len(staged_stem) :]
            return f"{restored_stem}{artifact_path.suffix}"
    return artifact_path.name


def _stage_opensim_input_trcs(
    trc_files, destination_dir, bridge_trc_data_by_name=None, fps=30
):
    """
    Stage the OpenSim input TRCs into the destination directory.
    """

    destination_dir = Path(destination_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    bridge_trc_data_by_name = dict(bridge_trc_data_by_name or {})

    staged_paths = []
    for trc_file in trc_files:
        trc_file = Path(trc_file)
        staged_path = destination_dir / trc_file.name
        if staged_path.exists():
            os.remove(staged_path)
        if trc_file.name in bridge_trc_data_by_name:
            make_trc_with_trc_data(bridge_trc_data_by_name[trc_file.name], staged_path, fps=fps)
        else:
            shutil.copy2(trc_file, staged_path)
        staged_paths.append(staged_path)
    return staged_paths


def _create_ascii_safe_opensim_workspace(
    trc_files, bridge_trc_data_by_name=None, fps=30
):
    """
    Stage TRC inputs into an ASCII-only temporary workspace for OpenSim.
    """

    temp_parent = Path(tempfile.gettempdir())
    if _contains_non_ascii_path(temp_parent):
        temp_parent = Path("C:/Temp") if os.name == "nt" else Path("/tmp")
    temp_parent.mkdir(parents=True, exist_ok=True)

    workspace_root = Path(
        tempfile.mkdtemp(prefix="sports2d_opensim_", dir=str(temp_parent))
    )
    pose3d_dir = workspace_root / "pose-3d"
    kinematics_dir = workspace_root / "kinematics"
    pose3d_dir.mkdir(parents=True, exist_ok=True)
    kinematics_dir.mkdir(parents=True, exist_ok=True)

    staged_stem_map = {}
    staged_input_names = set()
    used_stems = set()

    for idx, trc_file in enumerate(trc_files):
        base_stem = _sanitize_ascii_opensim_stem(
            trc_file.stem, fallback=f"opensim_trial_{idx:02d}"
        )
        staged_stem = base_stem
        suffix_id = 1
        while staged_stem in used_stems:
            staged_stem = f"{base_stem}_{suffix_id:02d}"
            suffix_id += 1
        used_stems.add(staged_stem)

        staged_path = pose3d_dir / f"{staged_stem}{trc_file.suffix}"
        bridge_trc_data_by_name = dict(bridge_trc_data_by_name or {})
        if trc_file.name in bridge_trc_data_by_name:
            make_trc_with_trc_data(
                bridge_trc_data_by_name[trc_file.name], staged_path, fps=fps
            )
        else:
            shutil.copy2(trc_file, staged_path)
        staged_stem_map[staged_stem] = trc_file.stem
        staged_input_names.add(staged_path.name)

    return {
        "root_dir": workspace_root,
        "pose3d_dir": pose3d_dir,
        "kinematics_dir": kinematics_dir,
        "staged_stem_map": staged_stem_map,
        "staged_input_names": staged_input_names,
    }


def _move_ascii_safe_opensim_outputs(
    workspace_info, final_pose3d_dir, final_kinematics_dir
):
    """
    Restore OpenSim artifacts from the ASCII-safe workspace to the final output dirs.
    """

    pose3d_dir = workspace_info["pose3d_dir"]
    kinematics_dir = workspace_info["kinematics_dir"]
    staged_input_names = set(workspace_info["staged_input_names"])
    staged_stem_map = dict(workspace_info["staged_stem_map"])
    final_pose3d_dir = Path(final_pose3d_dir)
    final_kinematics_dir = Path(final_kinematics_dir)
    final_pose3d_dir.mkdir(parents=True, exist_ok=True)
    final_kinematics_dir.mkdir(parents=True, exist_ok=True)

    for directory in [pose3d_dir, kinematics_dir]:
        for file_path in directory.glob("*"):
            if not file_path.is_file():
                continue
            restored_name = _restore_opensim_artifact_name(
                file_path.name, staged_stem_map
            )
            destination_dir = (
                final_pose3d_dir if directory == pose3d_dir else final_kinematics_dir
            )
            destination_path = destination_dir / restored_name
            if destination_path.exists():
                os.remove(destination_path)
            if file_path.suffix.lower() == ".xml":
                xml_text = file_path.read_text(encoding="utf-8")
                for staged_stem, original_stem in sorted(
                    staged_stem_map.items(),
                    key=lambda item: len(item[0]),
                    reverse=True,
                ):
                    xml_text = xml_text.replace(staged_stem, original_stem)
                destination_path.write_text(xml_text, encoding="utf-8")
                os.remove(file_path)
            else:
                shutil.move(str(file_path), destination_path)

    shutil.rmtree(workspace_info["root_dir"], ignore_errors=True)


def strip_auxiliary_trc_markers(Q_coords, keypoints_names, ignored_marker_names=None):
    """
    Remove non-pose markers such as a trailing ball marker from loaded TRCs.
    """
    ignored = {
        str(marker_name).strip().lower()
        for marker_name in (ignored_marker_names or [])
        if str(marker_name).strip()
    }
    keypoints_names = list(keypoints_names or [])
    if not ignored or len(keypoints_names) == 0:
        return Q_coords, keypoints_names

    filtered_names = [
        marker_name
        for marker_name in keypoints_names
        if str(marker_name).strip().lower() not in ignored
    ]
    if len(filtered_names) == len(keypoints_names):
        return Q_coords, keypoints_names

    return Q_coords.loc[:, filtered_names].copy(), filtered_names


def _remap_pose_model_ids_by_keypoint_names(pose_model, keypoint_names):
    """
    Return a pose-model copy whose node ids match the current keypoint tensor order.

    anytree traversal order is not guaranteed to match the processed pose tensor
    column order. Saved overlay rendering indexes keypoint arrays through node.id,
    so ids must be reassigned by keypoint name rather than preorder position.
    """
    from anytree import PreOrderIter

    remapped_pose_model = copy.deepcopy(pose_model)
    name_to_idx = {
        str(name): idx for idx, name in enumerate(list(keypoint_names or []))
    }

    for node in PreOrderIter(remapped_pose_model):
        node_name = str(getattr(node, "name", ""))
        node.id = name_to_idx.get(node_name)

    return remapped_pose_model


def _select_pose_keypoint_columns(
    frame_values, source_keypoint_names, target_keypoint_names
):
    """
    Return pose-array columns reordered to the requested keypoint-name schema.
    """

    source_keypoint_names = list(source_keypoint_names or [])
    target_keypoint_names = list(target_keypoint_names or [])
    column_indices = []
    missing_names = []
    for keypoint_name in target_keypoint_names:
        if keypoint_name not in source_keypoint_names:
            missing_names.append(keypoint_name)
            continue
        column_indices.append(source_keypoint_names.index(keypoint_name))

    if missing_names:
        raise ValueError(
            f"Pose review output is missing required keypoints: {missing_names}"
        )

    return np.take(np.asarray(frame_values), column_indices, axis=-1)


def extract_ball_centers(detection_meta):
    """
    Extract integer ball centers from xyxy detector boxes.

    INPUTS:
    - detection_meta: dict containing optional `ball_boxes` key in xyxy format.

    OUTPUTS:
    - List of `(x, y)` integer centers.
    """
    detection_meta = detection_meta or {}
    ball_boxes = _ensure_xyxy_boxes(detection_meta.get("ball_boxes"))
    centers = []
    for x1, y1, x2, y2 in ball_boxes:
        if np.isnan([x1, y1, x2, y2]).any():
            continue
        cx = int(round((x1 + x2) * 0.5))
        cy = int(round((y1 + y2) * 0.5))
        centers.append((cx, cy))
    return centers


def _centers_to_keypoints(centers):
    """
    Convert Nx2 centers to keypoint tensor expected by sort_people_sports2d: Nx1x2.
    """
    centers = np.asarray(centers, dtype=np.float32)
    if centers.size == 0:
        return np.empty((0, 1, 2), dtype=np.float32)
    if centers.ndim == 1:
        centers = centers.reshape(1, -1)
    if centers.shape[1] < 2:
        return np.empty((0, 1, 2), dtype=np.float32)
    return centers[:, :2].reshape(-1, 1, 2).astype(np.float32, copy=False)


def _parse_ball_selection_mode(value, default="auto"):
    """
    Parse ball track selection mode ('auto' or 'id').
    """
    mode = str(default if value is None else value).strip().lower()
    if mode not in ["auto", "id"]:
        logging.warning(
            "Invalid ball_selection_mode '%s'. Falling back to '%s'.",
            value,
            default,
        )
        mode = default
    return mode


def _parse_ball_ordering_method(value, default="first_detected"):
    """
    Parse ball ordering method for auto track selection.
    """
    supported_methods = [
        "on_click",
        "highest_likelihood",
        "largest_size",
        "smallest_size",
        "greatest_displacement",
        "least_displacement",
        "first_detected",
        "last_detected",
    ]
    method = str(default if value is None else value).strip().lower()
    if method not in supported_methods:
        logging.warning(
            "Invalid ball_ordering_method '%s'. Falling back to '%s'.",
            value,
            default,
        )
        method = default
    return method


def _parse_motion_person_selection_target(value, default="auto"):
    """
    Parse the target motion class used by motion-specific person selection.
    """
    supported_targets = ["auto", "broad_jump", "sprint_start", "etc"]
    target = str(default if value is None else value).strip().lower()
    if target not in supported_targets:
        logging.warning(
            "Invalid motion.person_selection_target '%s'. Falling back to '%s'.",
            value,
            default,
        )
        target = default
    return target


def _parse_ball_detector_backend(value, synthpose_detector=None, default="same"):
    """
    Parse ball detector backend.

    Supported modes:
    - same: reuse the main synthpose detector
    - sam3: add a dedicated SAM3 sports-ball detector

    For convenience, specifying the same detector name as `synthpose_detector`
    is treated as `same`.
    """
    backend = str(default if value is None else value).strip().lower()
    detector_name = str(synthpose_detector or "").strip().lower()
    if backend in {"same", "sam3"}:
        return backend
    if detector_name and backend == detector_name:
        return "same"
    logging.warning(
        "Unsupported ball_detector_backend '%s'. Falling back to '%s'.",
        value,
        default,
    )
    return default


def _parse_ball_selected_id(value):
    """
    Parse selected ball track ID. Negative values disable explicit selection.
    """
    try:
        parsed = int(value)
    except Exception:
        return None
    return parsed if parsed >= 0 else None


def _parse_video_codec(value, default="mp4v"):
    """
    Parse output video codec.

    Supported values:
    - mp4v: MPEG-4 Part 2 (OpenCV-native default)
    - h264: H.264/AVC (finalized via ffmpeg)
    """

    codec = str(default if value is None else value).strip().lower()
    aliases = {
        "h264": "h264",
        "h.264": "h264",
        "avc1": "h264",
        "mp4v": "mp4v",
        "mpeg4": "mp4v",
    }
    codec = aliases.get(codec, codec)
    if codec not in ["h264", "mp4v"]:
        raise ValueError(
            f"Invalid video_codec '{value}'. Supported values are 'mp4v' and 'h264'."
        )
    return codec


def _predict_ball_track_keypoints(
    previous_keypoints,
    previous_track_ids,
    previous_missing_counts,
    track_velocities_by_id,
):
    """
    Predict the next center for each tracked ball from its last observed center and velocity.
    """
    previous_keypoints = np.asarray(previous_keypoints, dtype=np.float32)
    if len(previous_keypoints) == 0:
        return np.empty((0, 1, 2), dtype=np.float32)

    predicted_keypoints = np.array(previous_keypoints, dtype=np.float32, copy=True)
    for row_idx in range(len(predicted_keypoints)):
        if row_idx >= len(previous_track_ids):
            continue
        track_id = int(previous_track_ids[row_idx])
        velocity = np.asarray(
            (track_velocities_by_id or {}).get(track_id, (0.0, 0.0)),
            dtype=np.float32,
        ).reshape(-1)
        if (
            len(velocity) < 2
            or not np.all(np.isfinite(velocity[:2]))
            or np.isnan(predicted_keypoints[row_idx]).any()
        ):
            continue
        missing_count = (
            int(previous_missing_counts[row_idx])
            if row_idx < len(previous_missing_counts)
            else 0
        )
        predicted_keypoints[row_idx, 0, 0] += velocity[0] * float(
            max(1, missing_count + 1)
        )
        predicted_keypoints[row_idx, 0, 1] += velocity[1] * float(
            max(1, missing_count + 1)
        )
    return predicted_keypoints


def _update_ball_track_velocity(
    track_velocities_by_id, track_id, last_observed_kp, current_kp, previous_missing=0
):
    """
    Update a tracked ball's velocity estimate from its latest observation.
    """
    track_velocities_by_id = (
        track_velocities_by_id if isinstance(track_velocities_by_id, dict) else {}
    )
    track_id = int(track_id)
    previous_velocity = np.asarray(
        track_velocities_by_id.get(track_id, (0.0, 0.0)),
        dtype=np.float32,
    ).reshape(-1)
    if len(previous_velocity) < 2 or not np.all(np.isfinite(previous_velocity[:2])):
        previous_velocity = np.zeros((2,), dtype=np.float32)

    last_observed_kp = np.asarray(last_observed_kp, dtype=np.float32).reshape(-1)
    current_kp = np.asarray(current_kp, dtype=np.float32).reshape(-1)
    if (
        len(last_observed_kp) < 2
        or len(current_kp) < 2
        or not np.all(np.isfinite(last_observed_kp[:2]))
        or not np.all(np.isfinite(current_kp[:2]))
    ):
        track_velocities_by_id[track_id] = (0.0, 0.0)
        return track_velocities_by_id

    steps = float(max(1, int(previous_missing) + 1))
    observed_velocity = (current_kp[:2] - last_observed_kp[:2]) / steps
    if not np.all(np.isfinite(observed_velocity[:2])):
        track_velocities_by_id[track_id] = (0.0, 0.0)
        return track_velocities_by_id
    if np.linalg.norm(previous_velocity[:2]) < 1e-3:
        blended_velocity = observed_velocity
    else:
        smoothing = 0.2 if int(previous_missing) > 0 else 0.45
        blended_velocity = (
            smoothing * previous_velocity[:2] + (1.0 - smoothing) * observed_velocity
        )
    if not np.all(np.isfinite(blended_velocity[:2])):
        track_velocities_by_id[track_id] = (0.0, 0.0)
        return track_velocities_by_id
    track_velocities_by_id[track_id] = (
        float(blended_velocity[0]),
        float(blended_velocity[1]),
    )
    return track_velocities_by_id


def _decay_ball_track_velocity(track_velocities_by_id, track_id, decay=0.95):
    """
    Decay a tracked ball's velocity estimate while it is temporarily missing.
    """
    if not isinstance(track_velocities_by_id, dict):
        return {}
    track_id = int(track_id)
    previous_velocity = np.asarray(
        track_velocities_by_id.get(track_id, (0.0, 0.0)),
        dtype=np.float32,
    ).reshape(-1)
    if len(previous_velocity) < 2 or not np.all(np.isfinite(previous_velocity[:2])):
        track_velocities_by_id[track_id] = (0.0, 0.0)
        return track_velocities_by_id
    track_velocities_by_id[track_id] = (
        float(decay * previous_velocity[0]),
        float(decay * previous_velocity[1]),
    )
    return track_velocities_by_id


def dedupe_ball_detections(
    ball_boxes, ball_scores=None, iou_threshold=0.8, center_eps_px=6.0
):
    """
    Collapse same-frame near-duplicate ball boxes before tracking.
    """
    ball_boxes = _ensure_xyxy_boxes(ball_boxes)
    ball_scores = _ensure_score_vector(ball_scores, expected_len=len(ball_boxes))
    if len(ball_boxes) <= 1:
        return ball_boxes, ball_scores

    widths = np.maximum(0.0, ball_boxes[:, 2] - ball_boxes[:, 0])
    heights = np.maximum(0.0, ball_boxes[:, 3] - ball_boxes[:, 1])
    areas = widths * heights
    centers = np.column_stack(
        (
            (ball_boxes[:, 0] + ball_boxes[:, 2]) * 0.5,
            (ball_boxes[:, 1] + ball_boxes[:, 3]) * 0.5,
        )
    ).astype(np.float32, copy=False)
    sortable_scores = np.where(np.isfinite(ball_scores), ball_scores, -np.inf)
    order = np.argsort(-sortable_scores, kind="stable")
    suppressed = np.zeros((len(ball_boxes),), dtype=bool)
    keep_indices = []

    for order_idx, box_idx in enumerate(order):
        if suppressed[box_idx]:
            continue
        keep_indices.append(int(box_idx))
        x1, y1, x2, y2 = ball_boxes[box_idx]
        area = max(0.0, float(areas[box_idx]))
        center = centers[box_idx]
        for other_idx in order[order_idx + 1 :]:
            if suppressed[other_idx]:
                continue
            ox1, oy1, ox2, oy2 = ball_boxes[other_idx]
            inter_x1 = max(float(x1), float(ox1))
            inter_y1 = max(float(y1), float(oy1))
            inter_x2 = min(float(x2), float(ox2))
            inter_y2 = min(float(y2), float(oy2))
            inter_w = max(0.0, inter_x2 - inter_x1)
            inter_h = max(0.0, inter_y2 - inter_y1)
            inter_area = inter_w * inter_h
            union_area = area + max(0.0, float(areas[other_idx])) - inter_area
            iou = inter_area / union_area if union_area > 0.0 else 0.0
            center_dist = float(np.linalg.norm(center - centers[other_idx]))
            similar_area = (
                area <= 0.0
                or max(0.0, float(areas[other_idx])) <= 0.0
                or min(area, float(areas[other_idx]))
                >= 0.55 * max(area, float(areas[other_idx]))
            )
            if iou >= float(iou_threshold) or (
                center_dist <= float(center_eps_px) and similar_area
            ):
                suppressed[other_idx] = True

    keep_indices = np.asarray(sorted(keep_indices), dtype=np.int64)
    return (
        ball_boxes[keep_indices].astype(np.float32, copy=False),
        ball_scores[keep_indices].astype(np.float32, copy=False),
    )


def track_balls_sports2d(
    ball_boxes,
    previous_keypoints,
    previous_track_ids,
    previous_missing_counts,
    next_track_id,
    ball_scores=None,
    track_velocities_by_id=None,
    max_dist=120.0,
    max_missing_frames=12,
):
    """
    Associate ball detections across frames using Sports2D Hungarian tracker.

    INPUTS:
    - ball_boxes: Nx4 xyxy detections for current frame.
    - ball_scores: detector confidence scores aligned with ball_boxes.
    - previous_keypoints: previous tracking state as Nx1x2 centers.
    - previous_track_ids: list of track IDs aligned with previous_keypoints.
    - previous_missing_counts: per-track consecutive missing frame counters.
    - next_track_id: next integer ID to assign.
    - max_dist: association distance threshold in pixels (None disables threshold).
    - max_missing_frames: drop tracks missing for more than this many frames.

    OUTPUTS:
    - tracked_balls: list of dicts with keys id, center, box, visible, missing.
    - updated_keypoints, updated_track_ids, updated_missing_counts, next_track_id
    """
    ball_boxes = _ensure_xyxy_boxes(ball_boxes)
    ball_scores = _ensure_score_vector(ball_scores, expected_len=len(ball_boxes))
    centers = np.asarray(
        extract_ball_centers({"ball_boxes": ball_boxes}), dtype=np.float32
    )
    current_keypoints = _centers_to_keypoints(centers)

    previous_track_ids = list(previous_track_ids or [])
    previous_missing_counts = list(previous_missing_counts or [])
    track_velocities_by_id = (
        track_velocities_by_id if isinstance(track_velocities_by_id, dict) else {}
    )
    if previous_keypoints is None:
        previous_keypoints = np.empty((0, 1, 2), dtype=np.float32)
    previous_keypoints = np.asarray(previous_keypoints, dtype=np.float32)

    if len(previous_track_ids) != len(previous_keypoints):
        previous_track_ids = previous_track_ids[: len(previous_keypoints)]
    if len(previous_missing_counts) != len(previous_keypoints):
        previous_missing_counts = previous_missing_counts[: len(previous_keypoints)]

    if len(previous_keypoints) == 0:
        tracked_balls = []
        updated_track_ids = []
        updated_missing_counts = []
        if len(current_keypoints) == 0:
            return (
                tracked_balls,
                current_keypoints,
                updated_track_ids,
                updated_missing_counts,
                int(next_track_id),
            )
        for idx in range(len(current_keypoints)):
            track_id = int(next_track_id)
            next_track_id += 1
            track_velocities_by_id[track_id] = (0.0, 0.0)
            center = (
                int(round(float(current_keypoints[idx, 0, 0]))),
                int(round(float(current_keypoints[idx, 0, 1]))),
            )
            tracked_balls.append(
                {
                    "id": track_id,
                    "center": center,
                    "box": ball_boxes[idx].astype(np.float32, copy=False),
                    "score": float(ball_scores[idx])
                    if idx < len(ball_scores)
                    else float("nan"),
                    "visible": True,
                    "missing": 0,
                }
            )
            updated_track_ids.append(track_id)
            updated_missing_counts.append(0)
        return (
            tracked_balls,
            current_keypoints,
            updated_track_ids,
            updated_missing_counts,
            int(next_track_id),
        )

    predicted_previous_keypoints = _predict_ball_track_keypoints(
        previous_keypoints,
        previous_track_ids,
        previous_missing_counts,
        track_velocities_by_id,
    )

    sorted_prev_keypoints, sorted_keypoints, sorted_ids = sort_people_sports2d(
        predicted_previous_keypoints,
        current_keypoints,
        scores=None,
        max_dist=max_dist,
    )

    n_prev = len(previous_track_ids)
    tracked_balls = []
    updated_keypoints = []
    updated_track_ids = []
    updated_missing_counts = []
    active_track_ids = set()
    rejected_current_indices = []

    for row_idx, curr_idx in enumerate(np.asarray(sorted_ids, dtype=np.int64)):
        if row_idx < n_prev:
            track_id = int(previous_track_ids[row_idx])
            prev_missing = (
                int(previous_missing_counts[row_idx])
                if row_idx < len(previous_missing_counts)
                else 0
            )
            previous_velocity = tuple(track_velocities_by_id.get(track_id, (0.0, 0.0)))
            last_observed_kp = previous_keypoints[row_idx]
            predicted_kp = sorted_prev_keypoints[row_idx]
        else:
            track_id = int(next_track_id)
            next_track_id += 1
            prev_missing = 0
            previous_velocity = (0.0, 0.0)
            last_observed_kp = np.full((1, 2), np.nan, dtype=np.float32)
            predicted_kp = np.full((1, 2), np.nan, dtype=np.float32)

        is_visible = int(curr_idx) >= 0 and int(curr_idx) < len(ball_boxes)
        # Once a raw track has gone missing, force any later reappearance to start a new raw ID.
        # Selected-ball continuity is handled separately from raw detector-fragment IDs.
        if is_visible and row_idx < n_prev and prev_missing > 0:
            rejected_idx = int(curr_idx)
            if rejected_idx not in rejected_current_indices:
                rejected_current_indices.append(rejected_idx)
            is_visible = False
        if (
            is_visible
            and row_idx < n_prev
            and max_dist is not None
            and prev_missing <= 0
        ):
            observed_center = (
                last_observed_kp[0] if last_observed_kp.ndim == 2 else last_observed_kp
            )
            current_center = (
                current_keypoints[int(curr_idx), 0]
                if current_keypoints[int(curr_idx)].ndim == 2
                else current_keypoints[int(curr_idx)]
            )
            if np.all(
                np.isfinite(np.asarray(observed_center, dtype=np.float32)[:2])
            ) and np.all(np.isfinite(np.asarray(current_center, dtype=np.float32)[:2])):
                observed_jump = float(
                    np.linalg.norm(
                        np.asarray(current_center, dtype=np.float32)[:2]
                        - np.asarray(observed_center, dtype=np.float32)[:2]
                    )
                )
                if observed_jump > float(max_dist):
                    rejected_idx = int(curr_idx)
                    if rejected_idx not in rejected_current_indices:
                        rejected_current_indices.append(rejected_idx)
                    is_visible = False
        missing_count = 0 if is_visible else prev_missing + 1
        if missing_count > int(max_missing_frames):
            continue

        if is_visible:
            curr_idx = int(curr_idx)
            kp = current_keypoints[curr_idx]
            center = (
                int(round(float(kp[0, 0]))),
                int(round(float(kp[0, 1]))),
            )
            box = ball_boxes[curr_idx].astype(np.float32, copy=False)
            score = (
                float(ball_scores[curr_idx])
                if curr_idx < len(ball_scores)
                else float("nan")
            )
            track_velocities_by_id = _update_ball_track_velocity(
                track_velocities_by_id,
                track_id,
                last_observed_kp[0] if last_observed_kp.ndim == 2 else last_observed_kp,
                kp[0] if kp.ndim == 2 else kp,
                previous_missing=prev_missing,
            )
        else:
            kp = last_observed_kp
            center = None
            box = None
            score = float("nan")
            track_velocities_by_id = _decay_ball_track_velocity(
                track_velocities_by_id,
                track_id,
            )

        tracked_balls.append(
            {
                "id": track_id,
                "center": center,
                "box": box,
                "score": score,
                "visible": bool(is_visible),
                "missing": int(missing_count),
                "predicted_center": _normalize_ball_center(
                    predicted_kp[0] if predicted_kp.ndim == 2 else predicted_kp
                ),
                "velocity": previous_velocity
                if not is_visible
                else tuple(track_velocities_by_id.get(track_id, (0.0, 0.0))),
            }
        )
        updated_keypoints.append(kp)
        updated_track_ids.append(track_id)
        updated_missing_counts.append(int(missing_count))
        active_track_ids.add(track_id)

    if len(rejected_current_indices) > 0:
        for curr_idx in rejected_current_indices:
            track_id = int(next_track_id)
            next_track_id += 1
            kp = current_keypoints[curr_idx]
            center = (
                int(round(float(kp[0, 0]))),
                int(round(float(kp[0, 1]))),
            )
            track_velocities_by_id[track_id] = (0.0, 0.0)
            tracked_balls.append(
                {
                    "id": track_id,
                    "center": center,
                    "box": ball_boxes[curr_idx].astype(np.float32, copy=False),
                    "score": float(ball_scores[curr_idx])
                    if curr_idx < len(ball_scores)
                    else float("nan"),
                    "visible": True,
                    "missing": 0,
                    "predicted_center": None,
                    "velocity": (0.0, 0.0),
                }
            )
            updated_keypoints.append(kp)
            updated_track_ids.append(track_id)
            updated_missing_counts.append(0)
            active_track_ids.add(track_id)

    if len(updated_keypoints) > 0:
        updated_keypoints = np.asarray(updated_keypoints, dtype=np.float32).reshape(
            -1, 1, 2
        )
    else:
        updated_keypoints = np.empty((0, 1, 2), dtype=np.float32)

    for stale_track_id in list(track_velocities_by_id.keys()):
        if int(stale_track_id) not in active_track_ids:
            track_velocities_by_id.pop(int(stale_track_id), None)

    return (
        tracked_balls,
        updated_keypoints,
        updated_track_ids,
        updated_missing_counts,
        int(next_track_id),
    )


def _update_ball_track_stats(track_stats_by_id, tracked_balls, frame_index):
    """
    Update per-track statistics used by ball ordering methods.
    """
    track_stats_by_id = track_stats_by_id or {}
    for track in tracked_balls or []:
        if "id" not in track:
            continue
        track_id = int(track.get("id"))
        stats = track_stats_by_id.get(track_id)
        if stats is None:
            stats = {
                "first_seen_frame": int(frame_index),
                "last_seen_frame": int(frame_index),
                "visible_count": 0,
                "area_sum": 0.0,
                "area_count": 0,
                "score_sum": 0.0,
                "score_count": 0,
                "displacement_sum": 0.0,
                "last_center": None,
            }
            track_stats_by_id[track_id] = stats
        else:
            stats["last_seen_frame"] = int(frame_index)

        if not track.get("visible", False):
            continue
        center = track.get("center")
        if center is None:
            continue

        center = (int(center[0]), int(center[1]))
        stats["visible_count"] += 1

        box = track.get("box")
        if box is not None:
            box_arr = _ensure_xyxy_boxes([box])
            if len(box_arr) > 0:
                x1, y1, x2, y2 = box_arr[0]
                area = max(0.0, float((x2 - x1) * (y2 - y1)))
                if np.isfinite(area):
                    stats["area_sum"] += area
                    stats["area_count"] += 1

        score = track.get("score")
        if score is not None:
            score = float(score)
            if np.isfinite(score):
                stats["score_sum"] += score
                stats["score_count"] += 1

        last_center = stats.get("last_center")
        if last_center is not None:
            stats["displacement_sum"] += float(
                np.hypot(center[0] - last_center[0], center[1] - last_center[1])
            )
        stats["last_center"] = center

    return track_stats_by_id


def _has_ball_confidence_stats(track_ids, track_stats_by_id):
    """
    Whether any candidate track has detector confidence stats.
    """
    for track_id in track_ids or []:
        stats = (track_stats_by_id or {}).get(int(track_id), {})
        if int(stats.get("score_count", 0)) > 0:
            return True
    return False


def _rank_ball_track_ids(
    track_ids, track_stats_by_id, ordering_method="first_detected"
):
    """
    Rank candidate track IDs according to ordering method.
    """
    candidate_ids = [int(track_id) for track_id in (track_ids or [])]
    if len(candidate_ids) == 0:
        return []

    method = str(ordering_method or "first_detected").strip().lower()
    if method == "on_click":
        method = "first_detected"
    if method == "highest_likelihood" and not _has_ball_confidence_stats(
        candidate_ids, track_stats_by_id
    ):
        method = "first_detected"

    def _first_seen(track_id):
        stats = (track_stats_by_id or {}).get(int(track_id), {})
        return int(stats.get("first_seen_frame", int(1e9)))

    def _metric(track_id):
        stats = (track_stats_by_id or {}).get(int(track_id), {})
        first_seen = _first_seen(track_id)
        if method == "last_detected":
            return (-first_seen, int(track_id))
        if method == "highest_likelihood":
            score_count = int(stats.get("score_count", 0))
            mean_score = (
                (float(stats.get("score_sum", 0.0)) / score_count)
                if score_count > 0
                else float("-inf")
            )
            return (-mean_score, first_seen, int(track_id))
        if method == "largest_size":
            area_count = int(stats.get("area_count", 0))
            mean_area = (
                (float(stats.get("area_sum", 0.0)) / area_count)
                if area_count > 0
                else float("-inf")
            )
            return (-mean_area, first_seen, int(track_id))
        if method == "smallest_size":
            area_count = int(stats.get("area_count", 0))
            mean_area = (
                (float(stats.get("area_sum", 0.0)) / area_count)
                if area_count > 0
                else float("inf")
            )
            return (mean_area, first_seen, int(track_id))
        if method == "greatest_displacement":
            return (
                -float(stats.get("displacement_sum", 0.0)),
                first_seen,
                int(track_id),
            )
        if method == "least_displacement":
            return (
                float(stats.get("displacement_sum", 0.0)),
                first_seen,
                int(track_id),
            )
        return (first_seen, int(track_id))

    return sorted(candidate_ids, key=_metric)


def _update_selected_ball_motion_state(
    previous_center, previous_velocity, current_center, smoothing=0.6, decay=0.6
):
    """
    Update selected-ball center/velocity state without fabricating visibility.
    """
    next_center = previous_center
    next_velocity = previous_velocity
    current_center = _normalize_ball_center(current_center)

    if current_center is not None:
        if previous_center is not None:
            frame_velocity = (
                float(current_center[0] - previous_center[0]),
                float(current_center[1] - previous_center[1]),
            )
            if previous_velocity is None:
                next_velocity = frame_velocity
            else:
                next_velocity = (
                    float(
                        smoothing * previous_velocity[0]
                        + (1.0 - smoothing) * frame_velocity[0]
                    ),
                    float(
                        smoothing * previous_velocity[1]
                        + (1.0 - smoothing) * frame_velocity[1]
                    ),
                )
        next_center = current_center
    elif previous_velocity is not None:
        next_velocity = (
            float(decay * previous_velocity[0]),
            float(decay * previous_velocity[1]),
        )

    return next_center, next_velocity


def select_ball_track_id(
    tracked_balls,
    selection_mode="auto",
    requested_track_id=None,
    previous_selected_id=None,
    previous_selected_center=None,
    previous_selected_velocity=None,
    ordering_method="first_detected",
    track_stats_by_id=None,
    max_recovery_dist=None,
):
    """
    Select active ball track ID and its center for trajectory rendering.

    OUTPUTS:
    - selected_track_id: int or None
    - selected_center: `(x, y)` or None
    """
    tracked_balls = tracked_balls or []
    track_stats_by_id = track_stats_by_id or {}
    tracks_by_id = {
        int(track.get("id")): track for track in tracked_balls if "id" in track
    }
    visible_tracks = [
        track
        for track in tracked_balls
        if track.get("visible", False) and track.get("center") is not None
    ]

    if selection_mode == "id":
        selected_track_id = (
            requested_track_id
            if requested_track_id is not None
            else previous_selected_id
        )
        if selected_track_id is None:
            return None, None
        track = tracks_by_id.get(int(selected_track_id))
        if track is None or not track.get("visible", False):
            return int(selected_track_id), None
        return int(selected_track_id), tuple(track.get("center"))

    # Keep stable selection by default; 'last_detected' intentionally tracks the latest arrivals.
    if ordering_method != "last_detected" and previous_selected_id is not None:
        previous_track = tracks_by_id.get(int(previous_selected_id))
        if previous_track is not None:
            if (
                previous_track.get("visible", False)
                and previous_track.get("center") is not None
            ):
                previous_track_center = tuple(previous_track.get("center"))
                if previous_selected_center is None or max_recovery_dist is None:
                    return int(previous_selected_id), previous_track_center
                gated_center = select_ball_center(
                    [previous_track_center],
                    previous_center=previous_selected_center,
                    max_jump_px=max_recovery_dist,
                    previous_velocity=previous_selected_velocity,
                )
                if gated_center is not None:
                    return int(previous_selected_id), previous_track_center
        if previous_selected_center is not None and len(visible_tracks) > 0:
            recovered_center = select_ball_center(
                [tuple(track.get("center")) for track in visible_tracks],
                previous_center=previous_selected_center,
                max_jump_px=max_recovery_dist,
                previous_velocity=previous_selected_velocity,
            )
            recovered_track = _find_ball_track_nearest_center(
                visible_tracks,
                recovered_center,
                visible_only=True,
            )
            if (
                recovered_track is not None
                and recovered_track.get("center") is not None
            ):
                return int(previous_selected_id), tuple(recovered_track.get("center"))
        return int(previous_selected_id), None

    active_track_ids = [
        int(track.get("id")) for track in tracked_balls if "id" in track
    ]
    visible_track_ids = [
        int(track.get("id")) for track in visible_tracks if "id" in track
    ]
    candidate_track_ids = (
        visible_track_ids if len(visible_track_ids) > 0 else active_track_ids
    )
    if len(candidate_track_ids) == 0:
        return None, None

    ranked_track_ids = _rank_ball_track_ids(
        candidate_track_ids,
        track_stats_by_id=track_stats_by_id,
        ordering_method=ordering_method,
    )
    if len(ranked_track_ids) == 0:
        return None, None

    selected_track_id = int(ranked_track_ids[0])
    selected_track = tracks_by_id.get(selected_track_id)
    if (
        selected_track is None
        or not selected_track.get("visible", False)
        or selected_track.get("center") is None
    ):
        return selected_track_id, None
    return selected_track_id, tuple(selected_track.get("center"))


def _ball_track_color(
    track_id, base_color=(0, 0, 0), selected=False, has_selected=False
):
    """
    Deterministic BGR color for ball track ID.
    """
    if track_id is None:
        return tuple(base_color)
    palette = (
        int((37 * int(track_id) + 29) % 200 + 30),
        int((97 * int(track_id) + 53) % 200 + 30),
        int((57 * int(track_id) + 91) % 200 + 30),
    )
    blended = tuple(
        int(np.clip(0.35 * int(base_color[idx]) + 0.65 * palette[idx], 0, 255))
        for idx in range(3)
    )
    if selected:
        return blended
    if has_selected:
        return tuple(int(np.clip(0.55 * c, 0, 255)) for c in blended)
    return blended


def select_ball_center(
    candidates,
    previous_center=None,
    max_jump_px=120,
    min_movement_px=2.0,
    previous_velocity=None,
    lock_radius_px=35.0,
    switch_margin_px=8.0,
):
    """
    Select a single center candidate with temporal continuity constraints.

    INPUTS:
    - candidates: list of `(x, y)` centers for current frame.
    - previous_center: previous accepted center or None.
    - max_jump_px: max allowed movement from previous center. None disables gating.
    - min_movement_px: tiny-jitter suppression threshold.
    - previous_velocity: optional `(vx, vy)` estimated from recent frames.
    - lock_radius_px: keep tracking within this radius unless a clearly better candidate appears.
    - switch_margin_px: required score improvement to switch outside lock radius.

    OUTPUTS:
    - selected `(x, y)` or `None` if rejected/unavailable.
    """
    if not candidates:
        return None
    if previous_center is None:
        return tuple(candidates[0])

    candidates_arr = np.asarray(candidates, dtype=np.float32).reshape(-1, 2)
    previous_center_arr = np.asarray(previous_center, dtype=np.float32).reshape(
        2,
    )
    if previous_velocity is None:
        previous_velocity_arr = np.zeros(2, dtype=np.float32)
    else:
        previous_velocity_arr = np.asarray(previous_velocity, dtype=np.float32).reshape(
            2,
        )
    predicted_center = previous_center_arr + previous_velocity_arr

    dists_prev = np.linalg.norm(candidates_arr - previous_center_arr[None, :], axis=1)
    dists_pred = np.linalg.norm(candidates_arr - predicted_center[None, :], axis=1)
    speed = float(np.linalg.norm(previous_velocity_arr))
    use_prediction = speed >= 0.75
    if use_prediction:
        continuity_scores = 0.25 * dists_prev + 0.75 * dists_pred
    else:
        continuity_scores = dists_prev.copy()

    lock_radius = max(float(min_movement_px), float(lock_radius_px))
    near_idx = np.where(dists_prev <= lock_radius)[0]
    far_idx = np.where(dists_prev > lock_radius)[0]
    if len(near_idx) > 0:
        near_sorted = near_idx[np.argsort(continuity_scores[near_idx])]
        locked_near = int(near_idx[np.argmin(dists_prev[near_idx])])
        best_near = int(near_sorted[0])
        primary_idx = locked_near
        # Avoid unstable flip-flop between nearby candidates unless improvement is clear.
        if (
            continuity_scores[best_near] + float(switch_margin_px)
            < continuity_scores[locked_near]
        ):
            primary_idx = best_near

        far_sorted = (
            far_idx[np.argsort(continuity_scores[far_idx])]
            if len(far_idx) > 0
            else np.array([], dtype=np.int64)
        )
        if len(far_sorted) > 0:
            best_far = int(far_sorted[0])
            # Keep a lock on nearby candidates unless far candidate is clearly better.
            if (
                continuity_scores[best_far] + float(switch_margin_px)
                < continuity_scores[primary_idx]
            ):
                primary_idx = best_far

            ordered_rest = [
                idx
                for idx in np.concatenate((near_sorted, far_sorted))
                if int(idx) != int(primary_idx)
            ]
            ordered_idx = np.asarray(
                [int(primary_idx)] + [int(i) for i in ordered_rest], dtype=np.int64
            )
        else:
            ordered_rest = [idx for idx in near_sorted if int(idx) != int(primary_idx)]
            ordered_idx = np.asarray(
                [int(primary_idx)] + [int(i) for i in ordered_rest], dtype=np.int64
            )
    else:
        ordered_idx = np.argsort(continuity_scores)

    selected_idx = int(ordered_idx[0])
    for idx in ordered_idx:
        idx = int(idx)
        if dists_prev[idx] >= float(min_movement_px):
            selected_idx = idx
            break

    selected = (
        int(round(float(candidates_arr[selected_idx, 0]))),
        int(round(float(candidates_arr[selected_idx, 1]))),
    )
    selected_dist = float(dists_prev[selected_idx])

    if max_jump_px is not None and selected_dist > float(max_jump_px):
        return None
    if selected_dist < float(min_movement_px):
        return (
            int(round(float(previous_center_arr[0]))),
            int(round(float(previous_center_arr[1]))),
        )
    return selected


def _parse_ball_max_jump_px(value, default=120.0):
    """
    Parse ball max-jump config. Supports numeric values and 'none'.
    """
    if value is None:
        return float(default)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ["", "none", "null"]:
            return None
    try:
        parsed = float(value)
        if parsed <= 0:
            return None
        return parsed
    except Exception:
        logging.warning(
            "Invalid ball_max_jump_px '%s'. Falling back to %s.",
            value,
            default,
        )
        return float(default)


def _parse_ball_color(color_value, default=(0, 0, 0)):
    """
    Parse BGR color triplet and clip to uint8 range.
    """
    if not isinstance(color_value, (list, tuple)) or len(color_value) < 3:
        return tuple(default)
    try:
        parsed = [int(np.clip(int(v), 0, 255)) for v in color_value[:3]]
    except Exception:
        return tuple(default)
    return tuple(parsed)


def draw_ball_overlay(
    img,
    ball_boxes,
    ball_center,
    trail_points,
    color=(0, 0, 0),
    radius=4,
    trail_alpha=0.35,
    tracked_balls=None,
    selected_track_id=None,
    show_ids=False,
):
    """
    Draw optional ball bbox, IDs, center, and trajectory trail on image.
    """
    ball_boxes = _ensure_xyxy_boxes(ball_boxes)
    tracked_balls = tracked_balls or []
    has_selected = selected_track_id is not None

    if len(tracked_balls) > 0:
        for track in tracked_balls:
            if not track.get("visible", False):
                continue
            box = track.get("box")
            if box is None:
                continue
            box = _ensure_xyxy_boxes([box])
            if len(box) == 0:
                continue
            x1, y1, x2, y2 = box[0]
            track_id = int(track.get("id"))
            is_selected = has_selected and track_id == int(selected_track_id)
            track_color = _ball_track_color(
                track_id,
                base_color=color,
                selected=is_selected,
                has_selected=has_selected,
            )
            thickness = 3 if is_selected else 2
            cv2.rectangle(
                img, (int(x1), int(y1)), (int(x2), int(y2)), track_color, thickness
            )
            if show_ids:
                label = f"ball {track_id}"
                label_pos = (int(x1), max(0, int(y1) - 7))
                cv2.putText(
                    img,
                    label,
                    label_pos,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    img,
                    label,
                    label_pos,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    track_color,
                    1,
                    cv2.LINE_AA,
                )
    else:
        for x1, y1, x2, y2 in ball_boxes:
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

    if trail_points and len(trail_points) >= 2:
        overlay = img.copy()
        line_thickness = max(1, int(round(radius * 0.75)))
        for p1, p2 in zip(trail_points[:-1], trail_points[1:]):
            cv2.line(overlay, tuple(p1), tuple(p2), color, line_thickness, cv2.LINE_AA)
        img = cv2.addWeighted(
            overlay,
            float(np.clip(trail_alpha, 0.0, 1.0)),
            img,
            1.0 - float(np.clip(trail_alpha, 0.0, 1.0)),
            0,
        )

    if ball_center is not None:
        cv2.circle(img, tuple(ball_center), int(max(1, radius)), color, -1)
    return img


# SynthPose-specific drawing functions (private)
def _draw_synthpose_keypoints(
    img, all_X, all_Y, all_scores, keypoint_names=None, thickness=1, threshold=0.3
):
    """
    Draw SynthPose 52 keypoints with special styling:
    - HALPE26 (bodywithfeet): Colored circles at normal size
    - Other anatomical markers: 1/2 size diamonds (white color)

    HALPE26 bodywithfeet keypoints:
    - COCO17 body (0-16): Nose, Eyes, Ears, Shoulders, Elbows, Wrists, Hips, Knees, Ankles
    - Foot keypoints (40-47): R5Meta, L5Meta, RToe, LToe, RBigToe, LBigToe, LHeel, RHeel

    Uses keypoint NAMES (not IDs) to distinguish HALPE26 from other anatomical markers,
    so it works correctly even after keypoint IDs are reordered.

    INPUTS:
    - img: OpenCV image
    - all_X: list of x coordinates for each person
    - all_Y: list of y coordinates for each person
    - all_scores: list of scores for each person
    - keypoint_names: list of keypoint names (required for correct HALPE26/Anatomical distinction)
    - thickness: int. Line thickness

    OUTPUT:
    - img: Image with drawn keypoints
    """
    from Sports2D.Utilities.synthpose_skeleton import (
        SYNTHPOSE_KEYPOINT_COLORS,
        SYNTHPOSE_HALPE26_BODYWITHFEET_NAMES,
        SYNTHPOSE_KEYPOINT_NAMES,
    )

    radius = thickness + 4  # Same as default in Pose2Sim

    for person_id, (X, Y, scores) in enumerate(zip(all_X, all_Y, all_scores)):
        if np.isnan(X).all():
            continue

        for kp_id in range(len(X)):
            if np.isnan(X[kp_id]) or np.isnan(Y[kp_id]) or np.isnan(scores[kp_id]):
                continue

            x_coord, y_coord = int(X[kp_id]), int(Y[kp_id])
            score = scores[kp_id]

            # Skip if score is below threshold
            if score < float(threshold):
                continue

            # Determine if this is HALPE26 bodywithfeet or other anatomical marker based on keypoint NAME
            is_halpe26 = False
            kp_name = None
            if keypoint_names is not None and kp_id < len(keypoint_names):
                kp_name = keypoint_names[kp_id]
                is_halpe26 = kp_name in SYNTHPOSE_HALPE26_BODYWITHFEET_NAMES
            else:
                # Fallback: use ID-based check (only works for original IDs)
                # HALPE26 = COCO17 (0-16) + Foot keypoints (40-47)
                is_halpe26 = kp_id < 17 or (40 <= kp_id <= 47)

            if not is_halpe26:
                # Other anatomical marker: 1/2 size diamond (white color)
                diamond_radius = radius // 2
                points = np.array(
                    [
                        [x_coord, y_coord - diamond_radius],  # top
                        [x_coord + diamond_radius, y_coord],  # right
                        [x_coord, y_coord + diamond_radius],  # bottom
                        [x_coord - diamond_radius, y_coord],  # left
                    ],
                    np.int32,
                )
                cv2.fillPoly(img, [points], (255, 255, 255))  # White
            else:
                # HALPE26 bodywithfeet marker: Colored circle at normal size
                # Get color based on keypoint name (lookup original ID in SYNTHPOSE_KEYPOINT_NAMES)
                if kp_name is not None:
                    try:
                        original_id = SYNTHPOSE_KEYPOINT_NAMES.index(kp_name)
                        color = SYNTHPOSE_KEYPOINT_COLORS[original_id]
                    except (ValueError, IndexError):
                        color = (0, 255, 0)  # Default green
                else:
                    color = (
                        SYNTHPOSE_KEYPOINT_COLORS[kp_id]
                        if kp_id < len(SYNTHPOSE_KEYPOINT_COLORS)
                        else (0, 255, 0)
                    )
                cv2.circle(img, (x_coord, y_coord), radius, tuple(color), -1)

    return img


def _draw_synthpose_skeleton(
    img, all_X, all_Y, all_scores, pose_model, thickness=1, threshold=0.3
):
    """
    Draw SynthPose skeleton using pose_model tree structure.
    Uses parent-child relationships from anytree, not hardcoded ID links,
    so it works correctly even after keypoint IDs are reordered.

    INPUTS:
    - img: OpenCV image
    - all_X: list of x coordinates for each person
    - all_Y: list of y coordinates for each person
    - all_scores: list of keypoint scores for each person
    - pose_model: Skeleton tree structure (anytree Node)
    - thickness: int. Line thickness
    - threshold: float. Minimum endpoint score required to draw a line

    OUTPUT:
    - img: Image with drawn skeleton
    """
    from Sports2D.Utilities.synthpose_skeleton import SYNTHPOSE_KEYPOINT_COLORS
    from anytree import PreOrderIter

    for person_id, (X, Y, scores) in enumerate(zip(all_X, all_Y, all_scores)):
        if np.isnan(X).all():
            continue

        # Use anytree parent-child relationships (works with reordered IDs)
        for node in PreOrderIter(pose_model):
            if node.parent is None:
                continue  # Skip root node

            child_id = node.id
            parent_id = node.parent.id

            # Skip if either node has no valid ID
            if child_id is None or parent_id is None:
                continue

            # Skip if out of bounds
            if child_id >= len(X) or parent_id >= len(X):
                continue

            x1, y1 = X[parent_id], Y[parent_id]
            x2, y2 = X[child_id], Y[child_id]
            score1 = scores[parent_id]
            score2 = scores[child_id]

            if (
                np.isnan(x1)
                or np.isnan(y1)
                or np.isnan(x2)
                or np.isnan(y2)
                or np.isnan(score1)
                or np.isnan(score2)
                or score1 < float(threshold)
                or score2 < float(threshold)
            ):
                continue

            # Determine color: gray for anatomical markers, colored for COCO17
            # Note: After reordering, we can't rely on original IDs for color
            # Use a consistent gray/white for skeleton lines
            color = (200, 200, 200)  # Gray for all skeleton lines

            cv2.line(
                img, (int(x1), int(y1)), (int(x2), int(y2)), tuple(color), thickness
            )

    return img


CORRECTION_2D_TO_3D = 1.063  # Corrective factor for height calculation: segments do not perfectly lie in the 2D plane and look shorter than in 3D
DEFAULT_MASS = 70
DEFAULT_HEIGHT = 1.7

## AUTHORSHIP INFORMATION
__author__ = "David Pagnon, HunMin Kim"
__copyright__ = "Copyright 2023, Sports2D"
__credits__ = ["David Pagnon"]
__license__ = "BSD 3-Clause License"
__version__ = version("sports2d")
__maintainer__ = "David Pagnon"
__email__ = "contact@david-pagnon.com"
__status__ = "Development"


# FUNCTIONS
def setup_webcam(webcam_id, vid_output_path, input_size):
    """
    Set up webcam capture with OpenCV.

    INPUTS:
    - webcam_id: int. The ID of the webcam to capture from
    - input_size: tuple. The size of the input frame (width, height)

    OUTPUTS:
    - cap: cv2.VideoCapture. The webcam capture object
    - out_vid: cv2.VideoWriter. The video writer object
    - cam_width: int. The actual width of the webcam frame
    - cam_height: int. The actual height of the webcam frame
    - fps: int. The frame rate of the webcam
    """

    # On Windows, try multiple backends because virtual cameras may not support
    # index capture on a specific backend.
    selected_backend = "default"
    cap = None
    if platform.system() == "Windows":
        backend_candidates = [
            ("DSHOW", cv2.CAP_DSHOW),
            ("MSMF", cv2.CAP_MSMF),
            ("default", None),
        ]
        for backend_name, backend_api in backend_candidates:
            test_cap = (
                cv2.VideoCapture(webcam_id, backend_api)
                if backend_api is not None
                else cv2.VideoCapture(webcam_id)
            )
            if not test_cap.isOpened():
                test_cap.release()
                continue
            # Ensure backend can really produce frames (some backends open but never read).
            ok, _ = test_cap.read()
            if ok:
                cap = test_cap
                selected_backend = backend_name
                break
            test_cap.release()
    else:
        cap = cv2.VideoCapture(webcam_id)

    if cap is None or not cap.isOpened():
        raise ValueError(
            f"Error: Could not open webcam #{webcam_id} with available backends. "
            "If you use a virtual camera (e.g., Camo), verify that the app is running and try another webcam_id."
        )
    if platform.system() == "Windows":
        logging.info(f"Webcam #{webcam_id} opened with backend: {selected_backend}")

    # set width and height to closest available for the webcam
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, input_size[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, input_size[1])
    cam_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0:
        fps = 30

    if cam_width != input_size[0] or cam_height != input_size[1]:
        logging.warning(
            f"Warning: Your webcam does not support {input_size[0]}x{input_size[1]} resolution. Resolution set to the closest supported one: {cam_width}x{cam_height}."
        )

    # fourcc MJPG produces very large files but is faster. If it is too slow, consider using it and then converting the video to h264
    # try:
    #     fourcc = cv2.VideoWriter_fourcc(*'avc1') # =h264. better compression and quality but may fail on some systems
    #     out_vid = cv2.VideoWriter(vid_output_path, fourcc, fps, (cam_width, cam_height))
    #     if not out_vid.isOpened():
    #         raise ValueError("Failed to open video writer with 'avc1' (h264)")
    # except Exception:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_vid = cv2.VideoWriter(vid_output_path, fourcc, fps, (cam_width, cam_height))
    # logging.info("Failed to open video writer with 'avc1' (h264). Using 'mp4v' instead.")

    return cap, out_vid, cam_width, cam_height, fps


def setup_video(video_file_path, vid_output_path, save_vid):
    """
    Set up video capture with OpenCV.

    INPUTS:
    - video_file_path: Path. The path to the video file
    - save_vid: bool. Whether to save the video output
    - vid_output_path: Path. The path to save the video output

    OUTPUTS:
    - cap: cv2.VideoCapture. The video capture object
    - out_vid: cv2.VideoWriter. The video writer object
    - cam_width: int. The width of the video
    - cam_height: int. The height of the video
    - fps: int. The frame rate of the video
    """

    if video_file_path.name == video_file_path.stem:
        raise ValueError(
            "Please set video_input to 'webcam' or to a video file (with extension) in Config.toml"
        )
    try:
        cap = cv2.VideoCapture(str(video_file_path.absolute()))
        if not cap.isOpened():
            raise
    except:
        raise NameError(
            f"{video_file_path} is not a video. Check video_dir and video_input in your Config.toml file."
        )

    cam_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_vid = None
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0:
        fps = 30
    if save_vid:
        # try:
        #     fourcc = cv2.VideoWriter_fourcc(*'avc1') # =h264. better compression and quality but may fail on some systems
        #     out_vid = cv2.VideoWriter(str(vid_output_path.absolute()), fourcc, fps, (cam_width, cam_height))
        #     if not out_vid.isOpened():
        #         raise ValueError("Failed to open video writer with 'avc1' (h264)")
        # except Exception:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_vid = cv2.VideoWriter(
            str(vid_output_path.absolute()), fourcc, fps, (cam_width, cam_height)
        )
        # logging.info("Failed to open video writer with 'avc1' (h264). Using 'mp4v' instead.")

    return cap, out_vid, cam_width, cam_height, fps


def flip_left_right_direction(
    person_X, L_R_direction_idx, keypoints_names, keypoints_ids
):
    """
    Flip the points to the right or left for more consistent angle calculation
    depending on which direction the person is facing

    INPUTS:
    - person_X: list of x coordinates
    - L_R_direction_idx: list of indices of the left toe, left heel, right toe, right heel
    - keypoints_names: list of keypoint names (see skeletons.py)
    - keypoints_ids: list of keypoint ids (see skeletons.py)

    OUTPUTS:
    - person_X_flipped: list of x coordinates after flipping
    """

    Ltoe_idx, LHeel_idx, Rtoe_idx, RHeel_idx = L_R_direction_idx
    right_orientation = person_X[Rtoe_idx] - person_X[RHeel_idx]
    left_orientation = person_X[Ltoe_idx] - person_X[LHeel_idx]
    global_orientation = right_orientation + left_orientation

    person_X_flipped = person_X.copy()
    if left_orientation < 0:
        for k in keypoints_names:
            if k.startswith("L"):
                keypt_idx = keypoints_ids[keypoints_names.index(k)]
                person_X_flipped[keypt_idx] = person_X_flipped[keypt_idx] * -1
    if right_orientation < 0:
        for k in keypoints_names:
            if k.startswith("R"):
                keypt_idx = keypoints_ids[keypoints_names.index(k)]
                person_X_flipped[keypt_idx] = person_X_flipped[keypt_idx] * -1
    if global_orientation < 0:
        for k in keypoints_names:
            if not k.startswith("L") and not k.startswith("R"):
                keypt_idx = keypoints_ids[keypoints_names.index(k)]
                person_X_flipped[keypt_idx] = person_X_flipped[keypt_idx] * -1

    return person_X_flipped


def _resolve_person_visible_side_frame(
    person_X, visible_side_str, has_toe_heel, L_R_direction_idx=None
):
    """
    Resolve the per-frame visible side using the original Sports2D semantics.
    """

    visible_side_str = str(visible_side_str or "auto").strip().lower()
    if visible_side_str == "auto":
        if has_toe_heel and L_R_direction_idx is not None:
            Ltoe_idx, LHeel_idx, Rtoe_idx, RHeel_idx = L_R_direction_idx
            right_orientation = person_X[Rtoe_idx] - person_X[RHeel_idx]
            left_orientation = person_X[Ltoe_idx] - person_X[LHeel_idx]
            global_orientation = right_orientation + left_orientation
            return "right" if global_orientation >= 0 else "left"
        return "right"
    return visible_side_str


def _apply_visible_side_whole_body_flip(
    person_X, visible_side_frame, keypoints_names, keypoints_ids
):
    """
    Apply the original visible_side-driven whole-body flip to the X coordinates.
    """

    person_X_flipped = np.asarray(person_X, dtype=float).copy()
    visible_side_frame = str(visible_side_frame or "right").strip().lower()

    if visible_side_frame in ["right", "none"]:
        return person_X_flipped
    if visible_side_frame == "left":
        return -person_X_flipped
    if visible_side_frame in ["front", "back"]:
        negate_prefix = "R" if visible_side_frame == "front" else "L"
        for keypoint_name in keypoints_names:
            if keypoint_name.startswith(negate_prefix):
                keypoint_idx = keypoints_ids[keypoints_names.index(keypoint_name)]
                person_X_flipped[keypoint_idx] = -person_X_flipped[keypoint_idx]
        return person_X_flipped
    return person_X_flipped


def compute_angle(
    ang_name, person_X_flipped, person_Y, angle_dict, keypoints_ids, keypoints_names
):
    """
    Compute the angles from the 2D coordinates of the keypoints.
    Takes into account which side the participant is facing.
    Takes into account the offset and scaling of the angle from angle_dict.
    Requires points_to_angles function (see common.py)

    INPUTS:
    - ang_name: str. The name of the angle to compute
    - person_X_flipped: list of x coordinates after flipping if needed
    - person_Y: list of y coordinates
    - angle_dict: dict. The dictionary of angles to compute (name: [keypoints, type, offset, scaling])
    - keypoints_ids: list of keypoint ids (see skeletons.py)
    - keypoints_names: list of keypoint names (see skeletons.py)

    OUTPUTS:
    - ang: float. The computed angle
    """

    ang_params = angle_dict.get(ang_name)
    if ang_params is not None:
        try:
            if ang_name in ["pelvis", "trunk", "shoulders"]:
                angle_coords = [
                    [
                        np.abs(
                            person_X_flipped[keypoints_ids[keypoints_names.index(kpt)]]
                        ),
                        person_Y[keypoints_ids[keypoints_names.index(kpt)]],
                    ]
                    for kpt in ang_params[0]
                ]
            else:
                angle_coords = [
                    [
                        person_X_flipped[keypoints_ids[keypoints_names.index(kpt)]],
                        person_Y[keypoints_ids[keypoints_names.index(kpt)]],
                    ]
                    for kpt in ang_params[0]
                ]
            ang = fixed_angles(angle_coords, ang_name)
        except:
            ang = np.nan
    else:
        ang = np.nan

    return ang


def _upsert_derived_pose_keypoint(
    derived_name, person_X, person_Y, person_scores, keypoint_names
):
    """
    Recompute a derived keypoint from its source markers and upsert it into the frame arrays.
    """

    source_map = {
        "Hip": ("LHip", "RHip"),
        "Neck": ("LShoulder", "RShoulder"),
    }
    keypoint_names = list(keypoint_names)
    if derived_name == "Head":
        if (
            "LEye" in keypoint_names
            and "REye" in keypoint_names
            and np.all(
                np.isfinite(
                    [
                        person_X[keypoint_names.index("LEye")],
                        person_Y[keypoint_names.index("LEye")],
                        person_X[keypoint_names.index("REye")],
                        person_Y[keypoint_names.index("REye")],
                    ]
                )
            )
        ):
            left_eye_idx = keypoint_names.index("LEye")
            right_eye_idx = keypoint_names.index("REye")
            left_eye = np.asarray(
                [person_X[left_eye_idx], person_Y[left_eye_idx]], dtype=float
            )
            right_eye = np.asarray(
                [person_X[right_eye_idx], person_Y[right_eye_idx]], dtype=float
            )
            eye_center = (left_eye + right_eye) * 0.5
            eye_distance = float(np.linalg.norm(left_eye - right_eye))
            x_value = float(eye_center[0])
            y_value = float(eye_center[1] - eye_distance * 0.8)
            score_value = float(
                np.nanmean([person_scores[left_eye_idx], person_scores[right_eye_idx]])
            )
        elif "Nose" in keypoint_names:
            nose_idx = keypoint_names.index("Nose")
            x_value = float(person_X[nose_idx])
            y_value = float(person_Y[nose_idx])
            score_value = float(person_scores[nose_idx])
        else:
            x_value = np.nan
            y_value = np.nan
            score_value = np.nan
    else:
        source_names = source_map.get(derived_name)
        if source_names is None:
            return person_X, person_Y, person_scores, list(keypoint_names)
        if not all(source_name in keypoint_names for source_name in source_names):
            x_value = np.nan
            y_value = np.nan
            score_value = np.nan
        else:
            idx_a = keypoint_names.index(source_names[0])
            idx_b = keypoint_names.index(source_names[1])
            x_candidates = np.asarray([person_X[idx_a], person_X[idx_b]], dtype=float)
            y_candidates = np.asarray([person_Y[idx_a], person_Y[idx_b]], dtype=float)
            score_candidates = np.asarray(
                [person_scores[idx_a], person_scores[idx_b]], dtype=float
            )
            x_value = (
                float(np.nanmean(x_candidates))
                if np.any(np.isfinite(x_candidates))
                else np.nan
            )
            y_value = (
                float(np.nanmean(y_candidates))
                if np.any(np.isfinite(y_candidates))
                else np.nan
            )
            score_value = (
                float(np.nanmean(score_candidates))
                if np.any(np.isfinite(score_candidates))
                else np.nan
            )
    if not np.isfinite(x_value) or not np.isfinite(y_value):
        x_value = np.nan
        y_value = np.nan
        score_value = np.nan

    if derived_name in keypoint_names:
        derived_idx = keypoint_names.index(derived_name)
        if derived_idx >= len(person_X):
            pad_width = derived_idx + 1 - len(person_X)
            nan_pad = np.full((pad_width,), np.nan, dtype=float)
            person_X = np.append(person_X, nan_pad)
            person_Y = np.append(person_Y, nan_pad.copy())
            person_scores = np.append(person_scores, nan_pad.copy())
        person_X[derived_idx] = x_value
        person_Y[derived_idx] = y_value
        person_scores[derived_idx] = score_value
    else:
        person_X = np.append(person_X, x_value)
        person_Y = np.append(person_Y, y_value)
        person_scores = np.append(person_scores, score_value)
        keypoint_names.append(derived_name)

    return person_X, person_Y, person_scores, keypoint_names


def _recompute_pose_frame_from_raw(
    raw_person_X,
    raw_person_Y,
    raw_person_scores,
    raw_keypoint_names,
    keypoint_likelihood_threshold,
    average_likelihood_threshold,
    keypoint_number_threshold,
    flip_left_right,
    L_R_direction_idx,
    angle_names,
    calculate_angles,
    visible_side_person="auto",
    use_visible_side_whole_body_flip=False,
    has_toe_heel=False,
):
    """
    Recompute filtered coordinates, flipped coordinates, and angles from raw pose values.
    """

    person_X, person_Y, person_scores, _ = evaluate_pose_frame(
        raw_person_X,
        raw_person_Y,
        raw_person_scores,
        keypoint_likelihood_threshold,
        average_likelihood_threshold,
        keypoint_number_threshold,
    )

    keypoint_names = list(raw_keypoint_names)
    for derived_name in ["Hip", "Neck", "Head"]:
        if derived_name in keypoint_names:
            continue
        person_X, person_Y, person_scores, keypoint_names = (
            _upsert_derived_pose_keypoint(
                derived_name,
                person_X,
                person_Y,
                person_scores,
                keypoint_names,
            )
        )

    keypoint_ids = list(range(len(keypoint_names)))
    person_visible_side_frame = visible_side_person
    if use_visible_side_whole_body_flip:
        person_visible_side_frame = _resolve_person_visible_side_frame(
            person_X,
            visible_side_person,
            has_toe_heel,
            L_R_direction_idx=L_R_direction_idx if has_toe_heel else None,
        )
        person_X_flipped = _apply_visible_side_whole_body_flip(
            person_X.copy(),
            person_visible_side_frame,
            keypoints_names=keypoint_names,
            keypoints_ids=keypoint_ids,
        )
    elif flip_left_right and L_R_direction_idx is not None:
        person_X_flipped = flip_left_right_direction(
            person_X.copy(), L_R_direction_idx, keypoint_names, keypoint_ids
        )
    else:
        person_X_flipped = person_X.copy()

    if calculate_angles:
        person_angles = []
        for ang_name in angle_names:
            ang_params = angle_dict.get(ang_name)
            kpts = ang_params[0] if ang_params is not None else []
            if not any(item not in keypoint_names for item in kpts):
                ang = compute_angle(
                    ang_name,
                    person_X_flipped,
                    person_Y,
                    angle_dict,
                    keypoint_ids,
                    keypoint_names,
                )
            else:
                ang = np.nan
            person_angles.append(ang)
        if (
            not use_visible_side_whole_body_flip
            and person_visible_side_frame == "left"
            and not flip_left_right
        ):
            person_angles = list(-np.array(person_angles, dtype=float))
    else:
        person_angles = []

    return (
        np.asarray(person_X, dtype=float),
        np.asarray(person_Y, dtype=float),
        np.asarray(person_scores, dtype=float),
        np.asarray(person_X_flipped, dtype=float),
        np.asarray(person_angles, dtype=float),
        keypoint_names,
    )


def _recompute_pose_timelines_from_raw(
    raw_frames_X,
    raw_frames_Y,
    raw_frames_scores,
    raw_keypoint_names,
    keypoint_likelihood_threshold,
    average_likelihood_threshold,
    keypoint_number_threshold,
    flip_left_right,
    L_R_direction_idx,
    angle_names,
    calculate_angles,
    visible_side_person="auto",
    use_visible_side_whole_body_flip=False,
    has_toe_heel=False,
):
    """
    Recompute timeline arrays for one selected person after manual edits.
    """

    frame_count = len(raw_frames_X)
    expected_keypoint_names = list(raw_keypoint_names)
    for derived_name in ["Hip", "Neck", "Head"]:
        if derived_name not in expected_keypoint_names:
            expected_keypoint_names.append(derived_name)

    filtered_X = np.full(
        (frame_count, len(expected_keypoint_names)), np.nan, dtype=float
    )
    filtered_Y = np.full_like(filtered_X, np.nan)
    filtered_scores = np.full_like(filtered_X, np.nan)
    filtered_X_flipped = np.full_like(filtered_X, np.nan)
    if calculate_angles:
        filtered_angles = np.full((frame_count, len(angle_names)), np.nan, dtype=float)
    else:
        filtered_angles = np.empty((frame_count, 0), dtype=float)

    for frame_idx in range(frame_count):
        (
            person_X,
            person_Y,
            person_scores,
            person_X_flipped,
            person_angles,
            keypoint_names,
        ) = _recompute_pose_frame_from_raw(
            raw_frames_X[frame_idx],
            raw_frames_Y[frame_idx],
            raw_frames_scores[frame_idx],
            raw_keypoint_names,
            keypoint_likelihood_threshold,
            average_likelihood_threshold,
            keypoint_number_threshold,
            flip_left_right,
            L_R_direction_idx,
            angle_names,
            calculate_angles,
            visible_side_person=visible_side_person,
            use_visible_side_whole_body_flip=use_visible_side_whole_body_flip,
            has_toe_heel=has_toe_heel,
        )
        if keypoint_names != expected_keypoint_names:
            raise ValueError(
                f"Hybrid recompute produced an inconsistent keypoint schema: "
                f"{keypoint_names} != {expected_keypoint_names}"
            )
        filtered_X[frame_idx] = person_X
        filtered_Y[frame_idx] = person_Y
        filtered_scores[frame_idx] = person_scores
        filtered_X_flipped[frame_idx] = person_X_flipped
        if calculate_angles:
            filtered_angles[frame_idx] = person_angles

    return (
        filtered_X,
        filtered_Y,
        filtered_scores,
        filtered_X_flipped,
        filtered_angles,
        expected_keypoint_names,
    )


def draw_dotted_line(
    img,
    start,
    direction,
    length,
    color=(0, 255, 0),
    gap=7,
    dot_length=3,
    thickness=thickness,
):
    """
    Draw a dotted line with on a cv2 image

    INPUTS:
    - img: opencv image
    - start: np.array. The starting point of the line
    - direction: np.array. The direction of the line
    - length: int. The length of the line
    - color: tuple. The color of the line
    - gap: int. The distance between each dot
    - dot_length: int. The length of each dot
    - thickness: int. The thickness of the line

    OUTPUT:
    - img: image with the dotted line
    """

    for i in range(0, length, gap):
        line_start = start + direction * i
        line_end = line_start + direction * dot_length
        cv2.line(
            img,
            tuple(line_start.astype(int)),
            tuple(line_end.astype(int)),
            color,
            thickness,
        )


def draw_angles(
    img,
    valid_X,
    valid_Y,
    valid_angles,
    valid_X_flipped,
    keypoints_ids,
    keypoints_names,
    angle_names,
    display_angle_values_on=["body", "list"],
    colors=[(255, 0, 0), (0, 255, 0), (0, 0, 255)],
    fontSize=0.3,
    thickness=1,
):
    """
    Draw angles on the image.
    Angles are displayed as a list on the image and/or on the body.

    INPUTS:
    - img: opencv image
    - valid_X: list of list of x coordinates
    - valid_Y: list of list of y coordinates
    - valid_angles: list of list of angles
    - valid_X_flipped: list of list of x coordinates after flipping if needed
    - keypoints_ids: list of keypoint ids (see skeletons.py)
    - keypoints_names: list of keypoint names (see skeletons.py)
    - angle_names: list of angle names
    - display_angle_values_on: list of str. 'body' and/or 'list'
    - colors: list of colors to cycle through

    OUTPUT:
    - img: image with angles
    """

    color_cycle = it.cycle(colors)
    for person_id, (X, Y, angles, X_flipped) in enumerate(
        zip(valid_X, valid_Y, valid_angles, valid_X_flipped)
    ):
        c = next(color_cycle)
        if not np.isnan(X).all():
            # person label
            if "list" in display_angle_values_on:
                person_label_position = (
                    int(10 + fontSize * 150 / 0.3 * person_id),
                    int(fontSize * 50),
                )
                cv2.putText(
                    img,
                    f"person {person_id}",
                    person_label_position,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    fontSize + 0.2,
                    (255, 255, 255),
                    thickness + 1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    img,
                    f"person {person_id}",
                    person_label_position,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    fontSize + 0.2,
                    c,
                    thickness,
                    cv2.LINE_AA,
                )

            # angle lines, names and values
            ang_label_line = 1
            for k, ang in enumerate(angles):
                if not np.isnan(ang):
                    ang_name = angle_names[k]
                    ang_params = angle_dict.get(ang_name)
                    if ang_params is not None:
                        kpts = ang_params[0]
                        if not any(
                            item not in keypoints_names + ["Neck", "Hip"]
                            for item in kpts
                        ):
                            ang_coords = np.array(
                                [
                                    [
                                        X[keypoints_ids[keypoints_names.index(kpt)]],
                                        Y[keypoints_ids[keypoints_names.index(kpt)]],
                                    ]
                                    for kpt in ang_params[0]
                                    if kpt in keypoints_names
                                ]
                            )
                            X_flipped = np.append(X_flipped, X[len(X_flipped) :])
                            X_flipped_coords = [
                                X_flipped[keypoints_ids[keypoints_names.index(kpt)]]
                                for kpt in ang_params[0]
                                if kpt in keypoints_names
                            ]
                            flip = (
                                -1
                                if any(x_flipped < 0 for x_flipped in X_flipped_coords)
                                else 1
                            )
                            flip = (
                                1
                                if ang_name in ["pelvis", "trunk", "shoulders"]
                                else flip
                            )
                            right_angle = True if ang_params[2] == 90 else False

                            # Draw angle
                            if not np.any(np.isnan(ang_coords)):
                                if len(ang_coords) == 2:  # segment angle
                                    app_point, vec = draw_segment_angle(
                                        img, ang_coords, flip
                                    )
                                else:  # joint angle
                                    app_point, vec1, vec2 = draw_joint_angle(
                                        img, ang_coords, flip, right_angle
                                    )

                                # Write angle on body
                                if "body" in display_angle_values_on:
                                    if len(ang_coords) == 2:  # segment angle
                                        write_angle_on_body(
                                            img,
                                            ang,
                                            app_point,
                                            vec,
                                            np.array([1, 0]),
                                            dist=20,
                                            color=(255, 255, 255),
                                            fontSize=fontSize,
                                            thickness=thickness,
                                        )
                                    else:  # joint angle
                                        write_angle_on_body(
                                            img,
                                            ang,
                                            app_point,
                                            vec1,
                                            vec2,
                                            dist=40,
                                            color=(0, 255, 0),
                                            fontSize=fontSize,
                                            thickness=thickness,
                                        )

                                # Write angle as a list on image with progress bar
                                if "list" in display_angle_values_on:
                                    if len(ang_coords) == 2:  # segment angle
                                        ang_label_line = write_angle_as_list(
                                            img,
                                            ang,
                                            ang_name,
                                            person_label_position,
                                            ang_label_line,
                                            color=(255, 255, 255),
                                            fontSize=fontSize,
                                            thickness=thickness,
                                        )
                                    else:
                                        ang_label_line = write_angle_as_list(
                                            img,
                                            ang,
                                            ang_name,
                                            person_label_position,
                                            ang_label_line,
                                            color=(0, 255, 0),
                                            fontSize=fontSize,
                                            thickness=thickness,
                                        )

    return img


def draw_segment_angle(img, ang_coords, flip):
    """
    Draw a segment angle on the image.

    INPUTS:
    - img: opencv image
    - ang_coords: np.array. The 2D coordinates of the keypoints
    - flip: int. Whether the angle should be flipped

    OUTPUT:
    - app_point: np.array. The point where the angle is displayed
    - unit_segment_direction: np.array. The unit vector of the segment direction
    - img: image with the angle
    """

    if not np.any(np.isnan(ang_coords)):
        app_point = np.int32(np.mean(ang_coords, axis=0))

        # segment line
        segment_direction = np.int32(ang_coords[0]) - np.int32(ang_coords[1])
        if (segment_direction == 0).all():
            return app_point, np.array([0, 0])
        unit_segment_direction = segment_direction / np.linalg.norm(segment_direction)
        cv2.line(
            img,
            app_point,
            np.int32(app_point + unit_segment_direction * 20),
            (255, 255, 255),
            thickness,
        )

        # horizontal line
        cv2.line(
            img,
            app_point,
            (np.int32(app_point[0]) + flip * 20, np.int32(app_point[1])),
            (255, 255, 255),
            thickness,
        )

        return app_point, unit_segment_direction


def draw_joint_angle(img, ang_coords, flip, right_angle):
    """
    Draw a joint angle on the image.

    INPUTS:
    - img: opencv image
    - ang_coords: np.array. The 2D coordinates of the keypoints
    - flip: int. Whether the angle should be flipped
    - right_angle: bool. Whether the angle should be offset by 90 degrees

    OUTPUT:
    - app_point: np.array. The point where the angle is displayed
    - unit_segment_direction: np.array. The unit vector of the segment direction
    - unit_parentsegment_direction: np.array. The unit vector of the parent segment direction
    - img: image with the angle
    """

    if not np.any(np.isnan(ang_coords)):
        app_point = np.int32(ang_coords[1])

        segment_direction = np.int32(ang_coords[0] - ang_coords[1])
        parentsegment_direction = np.int32(ang_coords[-2] - ang_coords[-1])
        if (segment_direction == 0).all() or (parentsegment_direction == 0).all():
            return app_point, np.array([0, 0]), np.array([0, 0])

        if right_angle:
            segment_direction = np.array(
                [-flip * segment_direction[1], flip * segment_direction[0]]
            )
            segment_direction, parentsegment_direction = (
                parentsegment_direction,
                segment_direction,
            )

        # segment line
        unit_segment_direction = segment_direction / np.linalg.norm(segment_direction)
        cv2.line(
            img,
            app_point,
            np.int32(app_point + unit_segment_direction * 40),
            (0, 255, 0),
            thickness,
        )

        # parent segment dotted line
        unit_parentsegment_direction = parentsegment_direction / np.linalg.norm(
            parentsegment_direction
        )
        draw_dotted_line(
            img,
            app_point,
            unit_parentsegment_direction,
            40,
            color=(0, 255, 0),
            gap=7,
            dot_length=3,
            thickness=thickness,
        )

        # arc
        start_angle = np.degrees(
            np.arctan2(unit_segment_direction[1], unit_segment_direction[0])
        )
        end_angle = np.degrees(
            np.arctan2(unit_parentsegment_direction[1], unit_parentsegment_direction[0])
        )
        if abs(end_angle - start_angle) > 180:
            if end_angle > start_angle:
                start_angle += 360
            else:
                end_angle += 360
        cv2.ellipse(
            img, app_point, (20, 20), 0, start_angle, end_angle, (0, 255, 0), thickness
        )

        return app_point, unit_segment_direction, unit_parentsegment_direction


def write_angle_on_body(
    img,
    ang,
    app_point,
    vec1,
    vec2,
    dist=40,
    color=(255, 255, 255),
    fontSize=0.3,
    thickness=1,
):
    """
    Write the angle on the body.

    INPUTS:
    - img: opencv image
    - ang: float. The angle value to display
    - app_point: np.array. The point where the angle is displayed
    - vec1: np.array. The unit vector of the first segment
    - vec2: np.array. The unit vector of the second segment
    - dist: int. The distance from the origin where to write the angle
    - color: tuple. The color of the angle

    OUTPUT:
    - img: image with the angle
    """

    vec_sum = vec1 + vec2
    if (vec_sum == 0.0).all():
        return
    unit_vec_sum = vec_sum / np.linalg.norm(vec_sum)
    text_position = np.int32(app_point + unit_vec_sum * dist)
    cv2.putText(
        img,
        f"{ang:.1f}",
        text_position,
        cv2.FONT_HERSHEY_SIMPLEX,
        fontSize,
        (0, 0, 0),
        thickness + 1,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        f"{ang:.1f}",
        text_position,
        cv2.FONT_HERSHEY_SIMPLEX,
        fontSize,
        color,
        thickness,
        cv2.LINE_AA,
    )


def write_angle_as_list(
    img,
    ang,
    ang_name,
    person_label_position,
    ang_label_line,
    color=(255, 255, 255),
    fontSize=0.3,
    thickness=1,
):
    """
    Write the angle as a list on the image with a progress bar.

    INPUTS:
    - img: opencv image
    - ang: float. The value of the angle to display
    - ang_name: str. The name of the angle
    - person_label_position: tuple. The position of the person label
    - ang_label_line: int. The line where to write the angle
    - color: tuple. The color of the angle

    OUTPUT:
    - ang_label_line: int. The updated line where to write the next angle
    - img: image with the angle
    """

    if not np.any(np.isnan(ang)):
        # angle names and values
        ang_label_position = (
            person_label_position[0],
            person_label_position[1] + int((ang_label_line) * 40 * fontSize),
        )
        ang_value_position = (
            ang_label_position[0] + int(250 * fontSize),
            ang_label_position[1],
        )
        cv2.putText(
            img,
            f"{ang_name}:",
            ang_label_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            fontSize,
            (0, 0, 0),
            thickness + 1,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            f"{ang_name}:",
            ang_label_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            fontSize,
            color,
            thickness,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            f"{ang:.1f}",
            ang_value_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            fontSize,
            (0, 0, 0),
            thickness + 1,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            f"{ang:.1f}",
            ang_value_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            fontSize,
            color,
            thickness,
            cv2.LINE_AA,
        )

        # progress bar
        ang_percent = int(ang * 50 / 180)
        y_crop, y_crop_end = (
            ang_value_position[1] - int(35 * fontSize),
            ang_value_position[1],
        )
        x_crop, x_crop_end = (
            ang_label_position[0] + int(300 * fontSize),
            ang_label_position[0]
            + int(300 * fontSize)
            + int(ang_percent * fontSize / 0.3),
        )
        if ang_percent < 0:
            x_crop, x_crop_end = x_crop_end, x_crop
        img_crop = img[y_crop:y_crop_end, x_crop:x_crop_end]
        if img_crop.size > 0:
            white_rect = np.ones(img_crop.shape, dtype=np.uint8) * 255
            alpha_rect = cv2.addWeighted(img_crop, 0.6, white_rect, 0.4, 1.0)
            img[y_crop:y_crop_end, x_crop:x_crop_end] = alpha_rect

        ang_label_line += 1

    return ang_label_line


def load_pose_file(Q_coords):
    """
    Load 2D keypoints from a dataframe of XYZ coordinates

    INPUTS:
    - Q_coords: pd.DataFrame. The dataframe of XYZ coordinates

    OUTPUTS:
    - keypoints_all: np.array. The keypoints in the format (Nframes, 1, Nmarkers, 2)
    - scores_all: np.array. The scores in the format (Nframes, 1, Nmarkers)
    """

    Z_cols = np.array(
        [[3 * i, 3 * i + 1] for i in range(len(Q_coords.columns) // 3)]
    ).ravel()
    Q_coords_xy = Q_coords.iloc[:, Z_cols]
    kpt_number = len(Q_coords_xy.columns) // 2

    # shape (Nframes, 2*Nmarkers) --> (Nframes, 1, Nmarkers, 2)
    keypoints_all = np.array(Q_coords_xy).reshape(len(Q_coords_xy), 1, kpt_number, 2)
    # shape (Nframes, 1, Nmarkers)
    scores_all = np.ones((len(Q_coords), 1, kpt_number))

    return keypoints_all, scores_all


def trc_data_from_XYZtime(X, Y, Z, time):
    """
    Constructs trc_data from 3D coordinates and time.

    INPUTS:
    - X: pd.DataFrame. The x coordinates of the keypoints
    - Y: pd.DataFrame. The y coordinates of the keypoints
    - Z: pd.DataFrame. The z coordinates of the keypoints
    - time: pd.Series. The time series for the coordinates

    OUTPUT:
    - trc_data: pd.DataFrame. Dataframe of trc data
    """

    columns_to_concat = []
    for kpt in range(len(X.columns)):
        columns_to_concat.extend([X.iloc[:, kpt], Y.iloc[:, kpt], Z.iloc[:, kpt]])
    trc_data = pd.concat([time] + columns_to_concat, axis=1)

    return trc_data


def reset_trc_frame_time_origin(trc_data):
    """
    Reset TRC row numbering to start at frame 0 and, when present, time 0.

    This is used for trimmed meter exports so the saved TRC starts from a
    local origin even if the valid motion segment begins later in the source.
    """

    trc_data = pd.DataFrame(trc_data).copy().reset_index(drop=True)
    if not trc_data.empty and "time" in trc_data.columns:
        trc_data["time"] = trc_data["time"] - trc_data["time"].iloc[0]
    return trc_data


def make_trc_with_trc_data(trc_data, trc_path, fps=30):
    """
    Write a TRC file from a DataFrame of time and coordinates

    INPUTS:
    - trc_data: pd.DataFrame. The time and coordinates of the keypoints.
                    The column names must be 'time', 'kpt1', 'kpt1', 'kpt1', 'kpt2', 'kpt2', 'kpt2', ...
    - trc_path: Path. The path to the TRC file to save
    - fps: float. The framerate of the video

    OUTPUT:
    - None
    """

    DataRate = CameraRate = OrigDataRate = fps
    NumFrames = len(trc_data)
    NumMarkers = (len(trc_data.columns) - 1) // 3
    keypoint_names = trc_data.columns[1::3]
    header_trc = [
        "PathFileType\t4\t(X/Y/Z)\t" + str(trc_path),
        "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames",
        "\t".join(
            map(
                str,
                [
                    DataRate,
                    CameraRate,
                    NumFrames,
                    NumMarkers,
                    "m",
                    OrigDataRate,
                    0,
                    NumFrames,
                ],
            )
        ),
        "Frame#\tTime\t" + "\t\t\t".join(keypoint_names) + "\t\t\t",
        "\t\t"
        + "\t".join(
            [f"X{i + 1}\tY{i + 1}\tZ{i + 1}" for i in range(len(keypoint_names))]
        ),
    ]

    with open(trc_path, "w") as trc_o:
        [trc_o.write(line + "\n") for line in header_trc]
        trc_data.to_csv(trc_o, sep="\t", index=True, header=None, lineterminator="\n")


def make_mot_with_angles(angles, time, mot_path):
    """
    Write a mot file from angles and time, compatible with OpenSim.

    INPUTS:
    - angles: pd.DataFrame. The angles to write
    - time: pd.Series. The time series for the angles
    - mot_path: str. The path where to save the mot file

    OUTPUT:
    - angles: pd.DataFrame. The data that has been written to the MOT file
    """

    # Header
    nRows, nColumns = angles.shape
    angle_names = angles.columns
    header_mot = [
        "Coordinates",
        "version=1",
        f"{nRows=}",
        f"{nColumns=}",
        "inDegrees=yes",
        "",
        "Units are S.I. units (second, meters, Newtons, ...)",
        "If the header above contains a line with 'inDegrees', this indicates whether rotational values are in degrees (yes) or radians (no).",
        "",
        "endheader",
        "time\t" + "\t".join(angle_names),
    ]

    # Write file
    angles.insert(0, "time", time)
    with open(mot_path, "w") as mot_o:
        [mot_o.write(line + "\n") for line in header_mot]
        angles.to_csv(mot_o, sep="\t", index=False, header=None, lineterminator="\n")

    return angles


def pose_plots(trc_data_unfiltered, trc_data, person_id, show=True):
    """
    Displays trc filtered and unfiltered data for comparison

    INPUTS:
    - trc_data_unfiltered: pd.DataFrame. The unfiltered trc data
    - trc_data: pd.DataFrame. The filtered trc data
    - person_id: int. The ID of the person
    - show: bool. Whether to show the plots

    OUTPUT:
    - matplotlib window with tabbed figures for each keypoint
    """

    os_name = platform.system()
    mpl.rc("figure", max_open_warning=0)
    if show:
        if os_name == "Windows":
            mpl.use("qt5agg")  # windows
        pw = plotWindow()
        pw.MainWindow.setWindowTitle("Person" + str(person_id) + " coordinates")
    else:
        mpl.use("Agg")  # Otherwise fails on Hugging-face
        figures_list = []

    keypoints_names = trc_data.columns[1::3]
    for id, keypoint in enumerate(keypoints_names):
        f = plt.figure()
        if show:
            if os_name == "Windows":
                f.canvas.manager.window.setWindowTitle(keypoint + " Plot")
            elif os_name == "Darwin":
                f.canvas.manager.set_window_title(keypoint + " Plot")

        axX = plt.subplot(211)
        plt.plot(
            trc_data_unfiltered.iloc[:, 0],
            trc_data_unfiltered.iloc[:, id * 3 + 1],
            label="unfiltered",
        )
        plt.plot(trc_data.iloc[:, 0], trc_data.iloc[:, id * 3 + 1], label="filtered")
        plt.setp(axX.get_xticklabels(), visible=False)
        axX.set_ylabel(keypoint + " X")
        plt.legend()

        axY = plt.subplot(212)
        plt.plot(
            trc_data_unfiltered.iloc[:, 0],
            trc_data_unfiltered.iloc[:, id * 3 + 2],
            label="unfiltered",
        )
        plt.plot(trc_data.iloc[:, 0], trc_data.iloc[:, id * 3 + 2], label="filtered")
        axY.set_xlabel("Time (seconds)")
        axY.set_ylabel(keypoint + " Y")

        if show:
            pw.addPlot(keypoint, f)
        else:
            figures_list.append((keypoint, f))

    if show:
        pw.show()
        return pw
    else:
        return figures_list


def angle_plots(angle_data_unfiltered, angle_data, person_id, show=True):
    """
    Displays angle filtered and unfiltered data for comparison

    INPUTS:
    - angle_data_unfiltered: pd.DataFrame. The unfiltered angle data
    - angle_data: pd.DataFrame. The filtered angle data

    OUTPUT:
    - matplotlib window with tabbed figures for each angle
    """

    os_name = platform.system()
    mpl.rc("figure", max_open_warning=0)
    if show:
        if os_name == "Windows":
            mpl.use("qt5agg")  # windows
        pw = plotWindow()
        pw.MainWindow.setWindowTitle("Person" + str(person_id) + " angles")
    else:
        mpl.use("Agg")  # Otherwise fails on Hugging-face
        figures_list = []

    angles_names = angle_data.columns[1:]
    for id, angle in enumerate(angles_names):
        f = plt.figure()
        if show:
            if os_name == "Windows":
                f.canvas.manager.window.setWindowTitle(angle + " Plot")  # windows
            elif os_name == "Darwin":  # macOS
                f.canvas.manager.set_window_title(angle + " Plot")  # mac

        ax = plt.subplot(111)
        plt.plot(
            angle_data_unfiltered.iloc[:, 0],
            angle_data_unfiltered.iloc[:, id + 1],
            label="unfiltered",
        )
        plt.plot(angle_data.iloc[:, 0], angle_data.iloc[:, id + 1], label="filtered")

        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel(angle + " (°)")
        plt.legend()

        if show:
            pw.addPlot(angle, f)
        else:
            figures_list.append((angle, f))

    if show:
        pw.show()
        return pw
    else:
        return figures_list


def get_personIDs_with_highest_scores(all_frames_scores, nb_persons_to_detect):
    """
    Get the person IDs with the highest scores

    INPUTS:
    - all_frames_scores: array of scores for all frames, all persons, all keypoints
    - nb_persons_to_detect: int or 'all'. The number of persons to detect

    OUTPUT:
    - selected_persons: list of int. The person IDs with the highest scores
    """

    # Get the person with the highest scores over all frames and all keypoints
    score_means = np.nansum(np.nanmean(all_frames_scores, axis=0), axis=1)
    selected_persons = (-score_means).argsort()[:nb_persons_to_detect]

    return selected_persons


def get_personIDs_in_detection_order(nb_persons_to_detect, reverse=False):
    """
    Get the person IDs in the order of detection

    INPUTS:
    - nb_persons_to_detect: int. The number of persons to detect
    - reverse: bool. Whether to reverse the order of detection

    OUTPUT:
    - selected_persons: list of int. The person IDs in the order of detection
    """

    selected_persons = list(range(nb_persons_to_detect))
    if reverse:
        selected_persons = selected_persons[::-1]

    return selected_persons


def get_personIDs_with_largest_size(
    all_frames_X_homog,
    all_frames_Y_homog,
    nb_persons_to_detect,
    reverse=False,
    vertical=False,
):
    """
    Get the person IDs with the largest size

    INPUTS:
    - all_frames_X_homog: shape (Nframes, Npersons, Nkpts)
    - all_frames_Y_homog: shape (Nframes, Npersons, Nkpts)
    - nb_persons_to_detect: int. The number of persons to detect
    - reverse: bool. Whether to reverse the order of detection from smallest to largest size
    - vertical: bool. Whether to compute the size in the vertical direction only

    OUTPUT:
    - selected_persons: list of int. The person IDs with the largest size
    """

    # average size over all keypoints (axis=2) and all frames (axis=0) for each person (axis=1)
    y_sizes = np.array(
        [
            np.nanmean(
                np.nanmax(all_frames_Y_homog, axis=2)
                - np.nanmin(all_frames_Y_homog, axis=2),
                axis=0,
            )
        ][0]
    )
    if vertical:
        sizes = y_sizes
    else:
        x_sizes = np.array(
            [
                np.nanmean(
                    np.nanmax(all_frames_X_homog, axis=2)
                    - np.nanmin(all_frames_X_homog, axis=2),
                    axis=0,
                )
            ][0]
        )
        sizes = np.sqrt(x_sizes**2 + y_sizes**2)

    if not reverse:  # greatest to smallest size
        sizes = -sizes

    selected_persons = sizes.argsort()[:nb_persons_to_detect]

    return selected_persons


def get_personIDs_with_greatest_displacement(
    all_frames_X_homog,
    all_frames_Y_homog,
    nb_persons_to_detect,
    reverse=False,
    horizontal=True,
):
    """
    Get the person IDs with the greatest displacement

    INPUTS:
    - all_frames_X_homog: shape (Nframes, Npersons, Nkpts)
    - all_frames_Y_homog: shape (Nframes, Npersons, Nkpts)
    - nb_persons_to_detect: int. The number of persons to detect
    - reverse: bool. Whether to reverse the order of detection from smallest to greatest displacement
    - horizontal: bool. Whether to compute the displacement in the horizontal direction

    OUTPUT:
    - selected_persons: list of int. The person IDs with the greatest displacement
    """

    # Average position over all keypoints to shape (Npersons, Nframes, Ndims)
    mean_pos_X_kpts = np.nanmean(all_frames_X_homog, axis=2)

    # Compute sum of distances from one frame to the next
    if horizontal:
        max_dist_traveled = abs(np.nansum(np.diff(mean_pos_X_kpts, axis=0), axis=0))
    else:
        mean_pos_Y_kpts = np.nanmean(all_frames_Y_homog, axis=2)
        pos_XY = np.stack((mean_pos_X_kpts.T, mean_pos_Y_kpts.T), axis=-1)
        max_dist_traveled = np.nansum(
            [
                euclidean_distance(m, p)
                for (m, p) in zip(pos_XY[:, 1:, :], pos_XY[:, :-1, :])
            ],
            axis=1,
        )
    max_dist_traveled = np.where(np.isinf(max_dist_traveled), 0, max_dist_traveled)

    if not reverse:  # greatest to smallest displacement
        max_dist_traveled = -max_dist_traveled

    selected_persons = (max_dist_traveled).argsort()[:nb_persons_to_detect]

    return selected_persons


def _clip01(value):
    """
    Clip a scalar to [0, 1], treating missing values as 0.
    """
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(value):
        return 0.0
    return float(np.clip(value, 0.0, 1.0))


def _safe_nanmedian(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(np.median(values))


def _keypoint_series(all_frames_X_person, all_frames_Y_person, keypoint_names, names):
    """
    Return frame-wise mean XY for the requested keypoint names.
    """
    keypoint_names = list(keypoint_names or [])
    indices = [keypoint_names.index(name) for name in names if name in keypoint_names]
    if len(indices) == 0:
        return None

    xs = np.asarray(all_frames_X_person[:, indices], dtype=float)
    ys = np.asarray(all_frames_Y_person[:, indices], dtype=float)
    valid_x = np.sum(np.isfinite(xs), axis=1)
    valid_y = np.sum(np.isfinite(ys), axis=1)
    mean_x = np.divide(
        np.nansum(xs, axis=1),
        valid_x,
        out=np.full((xs.shape[0],), np.nan, dtype=float),
        where=valid_x > 0,
    )
    mean_y = np.divide(
        np.nansum(ys, axis=1),
        valid_y,
        out=np.full((ys.shape[0],), np.nan, dtype=float),
        where=valid_y > 0,
    )
    return np.column_stack([mean_x, mean_y])


def _pose_center_series(all_frames_X_person, all_frames_Y_person):
    """
    Return frame-wise mean XY across all finite keypoints.
    """
    xs = np.asarray(all_frames_X_person, dtype=float)
    ys = np.asarray(all_frames_Y_person, dtype=float)
    valid_x = np.sum(np.isfinite(xs), axis=1)
    valid_y = np.sum(np.isfinite(ys), axis=1)
    mean_x = np.divide(
        np.nansum(xs, axis=1),
        valid_x,
        out=np.full((xs.shape[0],), np.nan, dtype=float),
        where=valid_x > 0,
    )
    mean_y = np.divide(
        np.nansum(ys, axis=1),
        valid_y,
        out=np.full((ys.shape[0],), np.nan, dtype=float),
        where=valid_y > 0,
    )
    return np.column_stack([mean_x, mean_y])


def _first_last_valid_xy(series):
    """
    Return first and last finite XY rows from a frame-wise series.
    """
    if series is None:
        return None, None
    series = np.asarray(series, dtype=float)
    if series.ndim != 2 or series.shape[1] < 2:
        return None, None
    valid_rows = np.where(np.all(np.isfinite(series[:, :2]), axis=1))[0]
    if valid_rows.size == 0:
        return None, None
    return series[valid_rows[0], :2], series[valid_rows[-1], :2]


def _longest_true_run(mask):
    """
    Return the inclusive (start, end, length) for the longest True run.
    """
    mask = np.asarray(mask, dtype=bool)
    best_start, best_end, best_len = None, None, 0
    current_start = None
    for idx, value in enumerate(mask):
        if value and current_start is None:
            current_start = idx
        if (not value or idx == len(mask) - 1) and current_start is not None:
            current_end = idx if value and idx == len(mask) - 1 else idx - 1
            current_len = current_end - current_start + 1
            if current_len > best_len:
                best_start, best_end, best_len = (
                    current_start,
                    current_end,
                    current_len,
                )
            current_start = None
    return best_start, best_end, best_len


def _broad_jump_airborne_motion_features(
    hip,
    left_foot,
    right_foot,
    body_height,
):
    """
    Detect broad-jump-specific flight with horizontal motion.

    From 2D pose only, flight is approximated as a contiguous interval where
    both feet are lifted above the observed foot-ground baseline. During that
    same interval, the hip center must move horizontally in either direction.
    """
    default = {
        "broad_jump_has_airborne_interval": False,
        "broad_jump_has_airborne_x_motion": False,
        "broad_jump_condition_met": False,
        "broad_jump_airborne_frame_count": 0,
        "broad_jump_airborne_x_displacement": 0.0,
        "broad_jump_airborne_x_displacement_norm": 0.0,
        "broad_jump_foot_clearance": 0.0,
        "broad_jump_foot_clearance_norm": 0.0,
        "broad_jump_airborne_start_frame": None,
        "broad_jump_airborne_end_frame": None,
    }

    if hip is None or left_foot is None or right_foot is None:
        return default

    hip = np.asarray(hip, dtype=float)
    left_foot = np.asarray(left_foot, dtype=float)
    right_foot = np.asarray(right_foot, dtype=float)
    if (
        hip.ndim != 2
        or left_foot.ndim != 2
        or right_foot.ndim != 2
        or hip.shape[0] == 0
    ):
        return default

    frame_count = min(hip.shape[0], left_foot.shape[0], right_foot.shape[0])
    hip = hip[:frame_count, :2]
    left_foot = left_foot[:frame_count, :2]
    right_foot = right_foot[:frame_count, :2]

    valid = (
        np.all(np.isfinite(hip), axis=1)
        & np.all(np.isfinite(left_foot), axis=1)
        & np.all(np.isfinite(right_foot), axis=1)
    )
    if int(np.sum(valid)) < 3:
        return default

    # Image y grows downward. The lower foot of the two approximates support;
    # high percentiles approximate stance/ground frames seen in the sequence.
    support_y = np.maximum(left_foot[:, 1], right_foot[:, 1])
    valid_support_y = support_y[valid]
    if valid_support_y.size < 3:
        return default

    ground_y = float(np.nanpercentile(valid_support_y, 85))
    min_clearance_px = max(0.07 * float(body_height), 4.0)
    clearance = ground_y - support_y
    airborne_mask = valid & (clearance >= min_clearance_px)
    start_frame, end_frame, airborne_count = _longest_true_run(airborne_mask)
    min_airborne_frames = max(2, min(5, int(np.ceil(0.05 * int(np.sum(valid))))))
    has_airborne_interval = airborne_count >= min_airborne_frames

    x_displacement = 0.0
    x_displacement_norm = 0.0
    has_airborne_x_motion = False
    if has_airborne_interval and start_frame is not None and end_frame is not None:
        start_hip = hip[start_frame]
        end_hip = hip[end_frame]
        if np.all(np.isfinite(start_hip)) and np.all(np.isfinite(end_hip)):
            x_displacement = float(end_hip[0] - start_hip[0])
            x_displacement_norm = abs(x_displacement) / max(float(body_height), 1e-6)
            min_x_motion_px = max(0.20 * float(body_height), 5.0)
            has_airborne_x_motion = abs(x_displacement) >= min_x_motion_px

    max_clearance = (
        float(np.nanmax(clearance[airborne_mask])) if np.any(airborne_mask) else 0.0
    )
    condition_met = bool(has_airborne_interval and has_airborne_x_motion)
    return {
        "broad_jump_has_airborne_interval": bool(has_airborne_interval),
        "broad_jump_has_airborne_x_motion": bool(has_airborne_x_motion),
        "broad_jump_condition_met": condition_met,
        "broad_jump_airborne_frame_count": int(airborne_count),
        "broad_jump_airborne_x_displacement": float(x_displacement),
        "broad_jump_airborne_x_displacement_norm": float(x_displacement_norm),
        "broad_jump_foot_clearance": float(max_clearance),
        "broad_jump_foot_clearance_norm": float(
            max_clearance / max(float(body_height), 1e-6)
        ),
        "broad_jump_airborne_start_frame": (
            int(start_frame) if start_frame is not None and has_airborne_interval else None
        ),
        "broad_jump_airborne_end_frame": (
            int(end_frame) if end_frame is not None and has_airborne_interval else None
        ),
    }


def _sprint_start_motion_features(
    hip,
    left_heel,
    right_heel,
    body_height,
    scores=None,
    fps=30.0,
):
    """
    Detect sprint-start-specific horizontal motion and heel alternation.

    Sprint-start requires all three condition signals:
    fast horizontal hip/body motion, vertical heel oscillation, and left/right
    heel alternation. Stance posture features are kept out of this rule.
    """
    default = {
        "sprint_start_fast_horizontal_motion": False,
        "sprint_start_heel_vertical_oscillation": False,
        "sprint_start_heel_alternation": False,
        "sprint_start_condition_met": False,
        "sprint_start_peak_frame": None,
        "sprint_start_window_start_frame": None,
        "sprint_start_window_end_frame": None,
        "sprint_start_window_valid_frame_count": 0,
        "sprint_start_window_presence_ratio": 0.0,
        "sprint_start_window_mean_confidence": 0.0,
        "sprint_start_peak_speed_norm": 0.0,
        "sprint_start_horizontal_displacement_norm": 0.0,
        "sprint_start_horizontal_speed_norm": 0.0,
        "sprint_start_left_heel_y_range_norm": 0.0,
        "sprint_start_right_heel_y_range_norm": 0.0,
        "sprint_start_heel_antiphase_ratio": 0.0,
        "sprint_start_heel_y_correlation": 0.0,
    }
    if hip is None or left_heel is None or right_heel is None:
        return default

    hip = np.asarray(hip, dtype=float)
    left_heel = np.asarray(left_heel, dtype=float)
    right_heel = np.asarray(right_heel, dtype=float)
    if (
        hip.ndim != 2
        or left_heel.ndim != 2
        or right_heel.ndim != 2
        or hip.shape[0] == 0
    ):
        return default

    frame_count = min(hip.shape[0], left_heel.shape[0], right_heel.shape[0])
    hip = hip[:frame_count, :2]
    left_heel = left_heel[:frame_count, :2]
    right_heel = right_heel[:frame_count, :2]
    fps = float(fps) if fps is not None else 30.0
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0

    valid = (
        np.all(np.isfinite(hip), axis=1)
        & np.all(np.isfinite(left_heel), axis=1)
        & np.all(np.isfinite(right_heel), axis=1)
    )
    if int(np.sum(valid)) < 4:
        return default

    body_height = max(float(body_height), 1e-6)

    # Sprint starts are short, fast events. A whole-video median frame-to-frame
    # speed is dominated by stationary/occluded frames and misses the runner.
    # Instead, find each track's local peak horizontal hip speed, then evaluate
    # the 1 second lookback window ending at that peak.
    lag_frames = max(1, int(round(0.10 * fps)))
    speeds = np.full((frame_count,), np.nan, dtype=float)
    for frame_idx in range(lag_frames, frame_count):
        start_idx = frame_idx - lag_frames
        if not (valid[start_idx] and valid[frame_idx]):
            continue
        interval_valid = valid[start_idx : frame_idx + 1]
        if float(np.mean(interval_valid)) < 0.5:
            continue
        dx = abs(float(hip[frame_idx, 0] - hip[start_idx, 0]))
        elapsed_seconds = lag_frames / fps
        speeds[frame_idx] = dx / max(elapsed_seconds, 1e-6) / body_height

    if not np.any(np.isfinite(speeds)):
        return default

    peak_frame = int(np.nanargmax(speeds))
    peak_speed_norm = float(speeds[peak_frame])
    window_frames = max(3, int(round(1.0 * fps)))
    window_start = max(0, peak_frame - window_frames + 1)
    window_end = peak_frame
    window_slice = slice(window_start, window_end + 1)
    window_valid = valid[window_slice]
    window_length = max(1, int(window_end - window_start + 1))
    window_valid_count = int(np.sum(window_valid))
    window_presence_ratio = float(window_valid_count / window_length)
    if window_valid_count < 3:
        return {
            **default,
            "sprint_start_peak_frame": int(peak_frame),
            "sprint_start_window_start_frame": int(window_start),
            "sprint_start_window_end_frame": int(window_end),
            "sprint_start_window_valid_frame_count": int(window_valid_count),
            "sprint_start_window_presence_ratio": float(window_presence_ratio),
            "sprint_start_peak_speed_norm": float(peak_speed_norm),
            "sprint_start_horizontal_speed_norm": float(peak_speed_norm),
        }

    window_indices = np.arange(window_start, window_end + 1)[window_valid]
    hip_x = hip[window_indices, 0]
    left_y = left_heel[window_indices, 1]
    right_y = right_heel[window_indices, 1]

    if scores is not None:
        score_arr = np.asarray(scores, dtype=float)
        if score_arr.ndim == 2 and score_arr.shape[0] >= frame_count:
            window_scores = score_arr[window_slice]
            finite_scores = window_scores[np.isfinite(window_scores)]
            window_mean_confidence = (
                float(np.mean(finite_scores)) if finite_scores.size > 0 else 0.0
            )
        else:
            window_mean_confidence = 0.0
    else:
        window_mean_confidence = 0.0

    first_hip_x = float(hip_x[0])
    last_hip_x = float(hip_x[-1])
    horizontal_displacement_norm = abs(last_hip_x - first_hip_x) / body_height
    hip_dx = np.abs(np.diff(hip_x))
    horizontal_speed_norm = (
        float(np.nanmax(hip_dx)) * fps / body_height if hip_dx.size > 0 else 0.0
    )
    fast_horizontal_motion = (
        peak_speed_norm >= 1.20 and horizontal_displacement_norm >= 0.35
    )

    left_range_norm = (
        float(np.nanmax(left_y) - np.nanmin(left_y)) / body_height
        if left_y.size > 0
        else 0.0
    )
    right_range_norm = (
        float(np.nanmax(right_y) - np.nanmin(right_y)) / body_height
        if right_y.size > 0
        else 0.0
    )
    heel_vertical_oscillation = left_range_norm >= 0.12 and right_range_norm >= 0.12

    left_dy = np.diff(left_y)
    right_dy = np.diff(right_y)
    min_heel_step = max(0.03 * body_height, 2.0)
    moving = (np.abs(left_dy) >= min_heel_step) & (
        np.abs(right_dy) >= min_heel_step
    )
    if np.any(moving):
        antiphase_ratio = float(
            np.mean(np.sign(left_dy[moving]) != np.sign(right_dy[moving]))
        )
    else:
        antiphase_ratio = 0.0

    centered_left_y = left_y - float(np.mean(left_y))
    centered_right_y = right_y - float(np.mean(right_y))
    denom = float(np.linalg.norm(centered_left_y) * np.linalg.norm(centered_right_y))
    heel_y_correlation = (
        float(np.dot(centered_left_y, centered_right_y) / denom)
        if denom > 1e-6
        else 0.0
    )
    heel_alternation = antiphase_ratio >= 0.50 or heel_y_correlation <= -0.35
    condition_met = bool(
        fast_horizontal_motion
        and heel_vertical_oscillation
        and heel_alternation
        and window_valid_count >= 3
    )

    return {
        "sprint_start_fast_horizontal_motion": bool(fast_horizontal_motion),
        "sprint_start_heel_vertical_oscillation": bool(heel_vertical_oscillation),
        "sprint_start_heel_alternation": bool(heel_alternation),
        "sprint_start_condition_met": condition_met,
        "sprint_start_peak_frame": int(peak_frame),
        "sprint_start_window_start_frame": int(window_start),
        "sprint_start_window_end_frame": int(window_end),
        "sprint_start_window_valid_frame_count": int(window_valid_count),
        "sprint_start_window_presence_ratio": float(window_presence_ratio),
        "sprint_start_window_mean_confidence": float(window_mean_confidence),
        "sprint_start_peak_speed_norm": float(peak_speed_norm),
        "sprint_start_horizontal_displacement_norm": float(
            horizontal_displacement_norm
        ),
        "sprint_start_horizontal_speed_norm": float(horizontal_speed_norm),
        "sprint_start_left_heel_y_range_norm": float(left_range_norm),
        "sprint_start_right_heel_y_range_norm": float(right_range_norm),
        "sprint_start_heel_antiphase_ratio": float(antiphase_ratio),
        "sprint_start_heel_y_correlation": float(heel_y_correlation),
    }


def _motion_track_bbox_metrics(all_frames_X_person, all_frames_Y_person):
    """
    Compute median pose-bbox width, height, and area for one person track.
    """
    widths, heights, areas = [], [], []
    for frame_X, frame_Y in zip(all_frames_X_person, all_frames_Y_person):
        frame_X = np.asarray(frame_X, dtype=float)
        frame_Y = np.asarray(frame_Y, dtype=float)
        valid = np.isfinite(frame_X) & np.isfinite(frame_Y)
        if not np.any(valid):
            continue
        width = float(np.nanmax(frame_X[valid]) - np.nanmin(frame_X[valid]))
        height = float(np.nanmax(frame_Y[valid]) - np.nanmin(frame_Y[valid]))
        if width <= 0 and height <= 0:
            continue
        widths.append(max(width, 0.0))
        heights.append(max(height, 0.0))
        areas.append(max(width, 0.0) * max(height, 0.0))
    return {
        "median_width": _safe_nanmedian(widths),
        "median_height": _safe_nanmedian(heights),
        "median_area": _safe_nanmedian(areas),
    }


def _motion_track_features(
    all_frames_X_person,
    all_frames_Y_person,
    all_frames_scores_person,
    keypoint_names,
    fps=30.0,
):
    """
    Compute rule-based motion-selection features for one tracked person slot.

    The first-stage quality gates use presence, confidence, and pose-bbox size.
    Motion scores are only used after those gates pass.
    """
    all_frames_X_person = np.asarray(all_frames_X_person, dtype=float)
    all_frames_Y_person = np.asarray(all_frames_Y_person, dtype=float)
    scores = np.asarray(all_frames_scores_person, dtype=float)
    if all_frames_X_person.ndim != 2 or all_frames_Y_person.ndim != 2:
        return {
            "presence_ratio": 0.0,
            "mean_confidence": float("nan"),
            "median_area": float("nan"),
            "median_height": float("nan"),
            "broad_jump_score": 0.0,
            "sprint_start_score": 0.0,
            "label": "etc",
        }

    coord_presence = np.any(
        np.isfinite(all_frames_X_person) & np.isfinite(all_frames_Y_person),
        axis=1,
    )
    score_presence = (
        np.any(np.isfinite(scores), axis=1)
        if scores.ndim == 2 and scores.shape[0] == all_frames_X_person.shape[0]
        else coord_presence
    )
    presence_ratio = (
        float(np.mean(coord_presence & score_presence))
        if all_frames_X_person.shape[0] > 0
        else 0.0
    )
    finite_scores = scores[np.isfinite(scores)]
    mean_confidence = (
        float(np.mean(finite_scores)) if finite_scores.size > 0 else float("nan")
    )

    bbox_metrics = _motion_track_bbox_metrics(
        all_frames_X_person, all_frames_Y_person
    )
    body_height = bbox_metrics["median_height"]
    if not np.isfinite(body_height) or body_height <= 1e-6:
        body_height = max(bbox_metrics.get("median_width", float("nan")), 1.0)
    if not np.isfinite(body_height) or body_height <= 1e-6:
        body_height = 1.0

    hip = _keypoint_series(
        all_frames_X_person,
        all_frames_Y_person,
        keypoint_names,
        ["Hip", "MidHip", "CHip", "RHip", "LHip"],
    )
    if hip is None:
        hip = _pose_center_series(all_frames_X_person, all_frames_Y_person)

    trunk_top = _keypoint_series(
        all_frames_X_person,
        all_frames_Y_person,
        keypoint_names,
        ["Neck", "Head", "Nose", "RShoulder", "LShoulder"],
    )
    left_foot = _keypoint_series(
        all_frames_X_person,
        all_frames_Y_person,
        keypoint_names,
        ["LBigToe", "LHeel", "LAnkle", "LFoot"],
    )
    right_foot = _keypoint_series(
        all_frames_X_person,
        all_frames_Y_person,
        keypoint_names,
        ["RBigToe", "RHeel", "RAnkle", "RFoot"],
    )
    left_heel = _keypoint_series(
        all_frames_X_person,
        all_frames_Y_person,
        keypoint_names,
        ["LHeel"],
    )
    right_heel = _keypoint_series(
        all_frames_X_person,
        all_frames_Y_person,
        keypoint_names,
        ["RHeel"],
    )

    first_hip, last_hip = _first_last_valid_xy(hip)
    valid_hip = hip[np.all(np.isfinite(hip[:, :2]), axis=1)]
    if first_hip is None or last_hip is None or valid_hip.size == 0:
        hip_net_horizontal = 0.0
        hip_vertical_arc = 0.0
        hip_speed_ratio = 0.0
    else:
        hip_net_horizontal = abs(float(last_hip[0] - first_hip[0])) / body_height
        hip_vertical_arc = (
            float(np.nanmax(valid_hip[:, 1]) - np.nanmin(valid_hip[:, 1]))
            / body_height
        )
        hip_dx = np.abs(np.diff(valid_hip[:, 0]))
        if hip_dx.size >= 4:
            split = max(1, hip_dx.size // 3)
            early_speed = float(np.median(hip_dx[:split]))
            late_speed = float(np.median(hip_dx[-split:]))
            hip_speed_ratio = late_speed / (early_speed + 1e-6)
        else:
            hip_speed_ratio = 0.0

    first_left_foot, last_left_foot = _first_last_valid_xy(left_foot)
    first_right_foot, last_right_foot = _first_last_valid_xy(right_foot)
    if first_left_foot is not None and first_right_foot is not None:
        initial_foot_sep = abs(float(first_left_foot[0] - first_right_foot[0]))
        initial_foot_sep_norm = initial_foot_sep / body_height
        initial_foot_y = float(np.nanmean([first_left_foot[1], first_right_foot[1]]))
    else:
        initial_foot_sep_norm = float("nan")
        initial_foot_y = float("nan")

    if (
        first_left_foot is not None
        and last_left_foot is not None
        and first_right_foot is not None
        and last_right_foot is not None
    ):
        left_dx = float(last_left_foot[0] - first_left_foot[0])
        right_dx = float(last_right_foot[0] - first_right_foot[0])
        same_direction = np.sign(left_dx) == np.sign(right_dx)
        max_dx = max(abs(left_dx), abs(right_dx), 1e-6)
        foot_sync = (min(abs(left_dx), abs(right_dx)) / max_dx) if same_direction else 0.0
    else:
        foot_sync = 0.0

    first_trunk, _ = _first_last_valid_xy(trunk_top)
    if first_hip is not None and first_trunk is not None:
        trunk_lean = abs(float(first_trunk[0] - first_hip[0])) / (
            abs(float(first_trunk[1] - first_hip[1])) + 1e-6
        )
    else:
        trunk_lean = 0.0

    if first_hip is not None and np.isfinite(initial_foot_y):
        hip_to_feet_ratio = abs(float(initial_foot_y - first_hip[1])) / body_height
    else:
        hip_to_feet_ratio = float("nan")

    broad_jump_flight = _broad_jump_airborne_motion_features(
        hip, left_foot, right_foot, body_height
    )
    broad_jump_score = 1.0 if broad_jump_flight["broad_jump_condition_met"] else 0.0

    sprint_start_motion = _sprint_start_motion_features(
        hip,
        left_heel,
        right_heel,
        body_height,
        scores=scores,
        fps=fps,
    )
    sprint_start_score = (
        1.0 if sprint_start_motion["sprint_start_condition_met"] else 0.0
    )

    if broad_jump_score > 0.0:
        label = "broad_jump"
    elif sprint_start_score > 0.0:
        label = "sprint_start"
    else:
        label = "etc"
    return {
        "presence_ratio": presence_ratio,
        "mean_confidence": mean_confidence,
        "median_area": bbox_metrics["median_area"],
        "median_height": bbox_metrics["median_height"],
        "hip_net_horizontal": hip_net_horizontal,
        "hip_vertical_arc": hip_vertical_arc,
        "initial_foot_sep": initial_foot_sep_norm,
        "foot_sync": _clip01(foot_sync),
        "trunk_lean": trunk_lean,
        "hip_to_feet_ratio": hip_to_feet_ratio,
        "hip_speed_ratio": hip_speed_ratio,
        **broad_jump_flight,
        **sprint_start_motion,
        "broad_jump_score": float(broad_jump_score),
        "sprint_start_score": float(sprint_start_score),
        "label": label,
    }


def _motion_specific_target_score(features, target):
    target = _parse_motion_person_selection_target(target)
    broad_score = float(features.get("broad_jump_score", 0.0))
    sprint_score = float(features.get("sprint_start_score", 0.0))
    if target == "broad_jump":
        return broad_score
    if target == "sprint_start":
        return sprint_score
    if target == "etc":
        return 1.0 - max(broad_score, sprint_score)
    return max(broad_score, sprint_score)


def resolve_personIDs_for_motion_specific(
    all_frames_X_homog,
    all_frames_Y_homog,
    all_frames_scores_homog,
    keypoint_names,
    nb_persons_to_detect,
    target="auto",
    presence_threshold=0.8,
    confidence_threshold=0.3,
    size_min_ratio=0.35,
    motion_score_threshold=0.35,
    fps=30.0,
):
    """
    Resolve motion-specific person ordering with first-stage gate filters.

    Presence ratio, mean confidence, and pose-bbox size ratio are binary gates:
    above-threshold tracks become candidates, but higher gate values do not
    dominate the motion-specific ranking except as tie-breakers.
    """
    diagnostics = {
        "used_fallback": False,
        "fallback_reason": None,
        "target": _parse_motion_person_selection_target(target),
        "eligible_person_ids": [],
        "ranked_person_ids": [],
        "features_by_person": {},
    }

    all_frames_X_homog = np.asarray(all_frames_X_homog, dtype=float)
    all_frames_Y_homog = np.asarray(all_frames_Y_homog, dtype=float)
    all_frames_scores_homog = np.asarray(all_frames_scores_homog, dtype=float)
    if (
        all_frames_X_homog.ndim != 3
        or all_frames_Y_homog.ndim != 3
        or all_frames_scores_homog.ndim != 3
        or all_frames_X_homog.shape[1] == 0
    ):
        diagnostics["used_fallback"] = True
        diagnostics["fallback_reason"] = "no_person_tracks"
        diagnostics["selected_persons"] = []
        return [], diagnostics

    person_count = all_frames_X_homog.shape[1]
    features_by_person = {}
    median_areas = []
    for person_idx in range(person_count):
        features = _motion_track_features(
            all_frames_X_homog[:, person_idx, :],
            all_frames_Y_homog[:, person_idx, :],
            all_frames_scores_homog[:, person_idx, :],
            keypoint_names,
            fps=fps,
        )
        features_by_person[int(person_idx)] = features
        area = float(features.get("median_area", float("nan")))
        median_areas.append(area if np.isfinite(area) and area > 0 else 0.0)

    max_median_area = max(median_areas) if len(median_areas) > 0 else 0.0
    eligible_person_ids = []
    for person_idx, features in features_by_person.items():
        area = float(features.get("median_area", 0.0))
        size_ratio = area / max_median_area if max_median_area > 0 else 0.0
        features["size_ratio"] = float(size_ratio)
        target_for_gate = diagnostics["target"]
        sprint_gate = (
            target_for_gate in {"sprint_start", "auto"}
            and bool(features.get("sprint_start_condition_met", False))
        )
        if sprint_gate:
            # Sprint-start runners are short-lived and often split into several
            # tracker IDs. Do not require whole-video presence or whole-video
            # bbox-size gates here; gate only the local peak-speed window.
            window_confidence = float(
                features.get("sprint_start_window_mean_confidence", 0.0)
            )
            if (
                window_confidence >= float(confidence_threshold)
                and int(features.get("sprint_start_window_valid_frame_count", 0)) >= 3
            ):
                eligible_person_ids.append(int(person_idx))
            continue

        if target_for_gate == "sprint_start":
            continue

        if (
            float(features.get("presence_ratio", 0.0)) >= float(presence_threshold)
            and float(features.get("mean_confidence", 0.0))
            >= float(confidence_threshold)
            and size_ratio >= float(size_min_ratio)
        ):
            eligible_person_ids.append(int(person_idx))

    target = diagnostics["target"]
    motion_margin = 0.05
    ranked_person_ids = []
    for person_idx in eligible_person_ids:
        features = features_by_person[person_idx]
        broad_score = float(features.get("broad_jump_score", 0.0))
        sprint_score = float(features.get("sprint_start_score", 0.0))
        if target == "auto":
            if features.get("broad_jump_condition_met", False):
                target_score = broad_score
                features["label"] = "broad_jump"
            else:
                target_score = max(broad_score, sprint_score)
            if (
                target_score < float(motion_score_threshold)
                or (
                    not features.get("broad_jump_condition_met", False)
                    and abs(broad_score - sprint_score) < motion_margin
                )
            ):
                features["label"] = "etc"
                continue
        else:
            target_score = _motion_specific_target_score(features, target)
            if target != "etc" and target_score < float(motion_score_threshold):
                continue
            if target == "etc" and target_score < float(motion_score_threshold):
                continue
        features["target_score"] = float(target_score)
        ranked_person_ids.append(int(person_idx))

    def _motion_specific_rank_key(person_idx):
        features = features_by_person[person_idx]
        sprint_start_tiebreak = target == "sprint_start" or (
            target == "auto" and features.get("label") == "sprint_start"
        )
        if sprint_start_tiebreak:
            return (
                -float(features.get("target_score", 0.0)),
                -float(features.get("sprint_start_peak_speed_norm", 0.0)),
                -float(
                    features.get("sprint_start_horizontal_displacement_norm", 0.0)
                ),
                -float(features.get("sprint_start_window_mean_confidence", 0.0)),
                -float(features.get("sprint_start_window_presence_ratio", 0.0)),
                -float(features.get("size_ratio", 0.0)),
                int(person_idx),
            )
        broad_jump_tiebreak = (
            target == "broad_jump"
            or (
                target == "auto"
                and features.get("label") == "broad_jump"
                and features.get("broad_jump_condition_met", False)
            )
        )
        if broad_jump_tiebreak:
            return (
                -float(features.get("target_score", 0.0)),
                -float(features.get("size_ratio", 0.0)),
                -float(features.get("median_area", 0.0)),
                -float(features.get("mean_confidence", 0.0)),
                -float(features.get("presence_ratio", 0.0)),
                int(person_idx),
            )
        return (
            -float(features.get("target_score", 0.0)),
            -float(features.get("mean_confidence", 0.0)),
            -float(features.get("size_ratio", 0.0)),
            -float(features.get("presence_ratio", 0.0)),
            int(person_idx),
        )

    ranked_person_ids = sorted(ranked_person_ids, key=_motion_specific_rank_key)

    def _sprint_start_near_miss_rank_key(person_idx):
        features = features_by_person[person_idx]
        return (
            -float(features.get("sprint_start_peak_speed_norm", 0.0)),
            -float(features.get("sprint_start_horizontal_displacement_norm", 0.0)),
            -float(features.get("sprint_start_window_mean_confidence", 0.0)),
            -float(features.get("mean_confidence", 0.0)),
            -float(features.get("size_ratio", 0.0)),
            int(person_idx),
        )

    sprint_start_near_miss_ids = []
    if target == "sprint_start":
        for person_idx, features in features_by_person.items():
            window_confidence = float(
                features.get("sprint_start_window_mean_confidence", 0.0)
            )
            peak_speed = float(features.get("sprint_start_peak_speed_norm", 0.0))
            displacement = float(
                features.get("sprint_start_horizontal_displacement_norm", 0.0)
            )
            if (
                window_confidence >= float(confidence_threshold)
                and int(features.get("sprint_start_window_valid_frame_count", 0)) >= 3
                and bool(
                    features.get("sprint_start_heel_vertical_oscillation", False)
                )
                and bool(features.get("sprint_start_heel_alternation", False))
                and (peak_speed >= 1.20 or displacement >= 0.35)
            ):
                sprint_start_near_miss_ids.append(int(person_idx))
        sprint_start_near_miss_ids = sorted(
            sprint_start_near_miss_ids, key=_sprint_start_near_miss_rank_key
        )

    diagnostics["eligible_person_ids"] = eligible_person_ids
    diagnostics["ranked_person_ids"] = ranked_person_ids
    diagnostics["sprint_start_near_miss_ids"] = sprint_start_near_miss_ids
    diagnostics["features_by_person"] = features_by_person
    if len(eligible_person_ids) == 0:
        diagnostics["used_fallback"] = True
        diagnostics["fallback_reason"] = (
            "no_motion_specific_candidate"
            if target == "sprint_start"
            else "gate_filters_removed_all_candidates"
        )
        if target == "sprint_start" and len(sprint_start_near_miss_ids) > 0:
            diagnostics["fallback_reason"] = "sprint_start_local_window_near_miss"
            diagnostics["selected_persons"] = sprint_start_near_miss_ids[
                : int(nb_persons_to_detect)
            ]
        else:
            diagnostics["selected_persons"] = list(
                map(
                    int,
                    get_personIDs_with_highest_scores(
                        all_frames_scores_homog,
                        nb_persons_to_detect,
                    ),
                )
            )
        return diagnostics["selected_persons"], diagnostics

    if len(ranked_person_ids) == 0:
        diagnostics["used_fallback"] = True
        diagnostics["fallback_reason"] = "no_motion_specific_candidate"
        if target == "sprint_start" and len(sprint_start_near_miss_ids) > 0:
            diagnostics["fallback_reason"] = "sprint_start_local_window_near_miss"
            diagnostics["selected_persons"] = sprint_start_near_miss_ids[
                : int(nb_persons_to_detect)
            ]
        else:
            diagnostics["selected_persons"] = list(
                map(
                    int,
                    get_personIDs_with_highest_scores(
                        all_frames_scores_homog,
                        nb_persons_to_detect,
                    ),
                )
            )
        return diagnostics["selected_persons"], diagnostics

    selected_persons = ranked_person_ids[: int(nb_persons_to_detect)]
    diagnostics["selected_persons"] = selected_persons
    return selected_persons, diagnostics


def get_personIDs_for_medicine_ball(
    all_frames_X_homog,
    all_frames_Y_homog,
    all_frames_scores_homog,
    all_frames_ball_centers,
    nb_persons_to_detect,
    presence_threshold=0.95,
    opening_window_frames=10,
):
    """
    Rank tracked person slots for medicine-ball throws.

    Persons must satisfy the long-run presence threshold first. The remaining
    candidates are ranked by smallest mean distance to the selected ball over
    the opening frame window.

    OUTPUTS:
    - selected_persons: ordered list of eligible person slot indices
    - diagnostics: dict with presence/distance bookkeeping for logging/tests
    """

    scores = np.asarray(all_frames_scores_homog, dtype=float)
    if scores.ndim != 3 or scores.shape[0] == 0 or scores.shape[1] == 0:
        return [], {
            "presence_ratios": np.asarray([], dtype=float),
            "eligible_person_ids": [],
            "ranked_person_ids": [],
            "usable_ball_frame_count": 0,
            "window_frame_count": 0,
        }

    all_frames_X_homog = np.asarray(all_frames_X_homog, dtype=float)
    all_frames_Y_homog = np.asarray(all_frames_Y_homog, dtype=float)
    frame_count, person_count = scores.shape[:2]
    window_frame_count = min(
        int(opening_window_frames), int(frame_count), len(all_frames_ball_centers or [])
    )

    presence_mask = np.any(np.isfinite(scores), axis=2)
    presence_ratios = (
        np.mean(presence_mask, axis=0)
        if frame_count > 0
        else np.zeros((person_count,), dtype=float)
    )
    eligible_person_ids = [
        int(person_idx)
        for person_idx in range(person_count)
        if float(presence_ratios[person_idx]) >= float(presence_threshold)
    ]

    x_valid_counts = np.sum(np.isfinite(all_frames_X_homog), axis=2)
    y_valid_counts = np.sum(np.isfinite(all_frames_Y_homog), axis=2)
    person_centers_x = np.divide(
        np.nansum(all_frames_X_homog, axis=2),
        x_valid_counts,
        out=np.full((frame_count, person_count), np.nan, dtype=float),
        where=x_valid_counts > 0,
    )
    person_centers_y = np.divide(
        np.nansum(all_frames_Y_homog, axis=2),
        y_valid_counts,
        out=np.full((frame_count, person_count), np.nan, dtype=float),
        where=y_valid_counts > 0,
    )
    score_valid_counts = np.sum(np.isfinite(scores), axis=(0, 2))
    mean_scores = np.divide(
        np.nansum(scores, axis=(0, 2)),
        score_valid_counts,
        out=np.full((person_count,), np.nan, dtype=float),
        where=score_valid_counts > 0,
    )

    usable_ball_frame_count = 0
    for frame_idx in range(window_frame_count):
        if _normalize_ball_center(all_frames_ball_centers[frame_idx]) is not None:
            usable_ball_frame_count += 1

    mean_distances = {}
    distance_frame_counts = {}
    for person_idx in eligible_person_ids:
        frame_distances = []
        for frame_idx in range(window_frame_count):
            ball_center = _normalize_ball_center(all_frames_ball_centers[frame_idx])
            if ball_center is None:
                continue

            person_center_x = float(person_centers_x[frame_idx, person_idx])
            person_center_y = float(person_centers_y[frame_idx, person_idx])
            if not np.isfinite(person_center_x) or not np.isfinite(person_center_y):
                continue

            frame_distances.append(
                float(
                    np.hypot(
                        person_center_x - float(ball_center[0]),
                        person_center_y - float(ball_center[1]),
                    )
                )
            )

        distance_frame_counts[int(person_idx)] = len(frame_distances)
        mean_distances[int(person_idx)] = (
            float(np.mean(frame_distances))
            if len(frame_distances) > 0
            else float("inf")
        )

    ranked_person_ids = sorted(
        [
            int(person_idx)
            for person_idx in eligible_person_ids
            if np.isfinite(float(mean_distances.get(int(person_idx), float("inf"))))
        ],
        key=lambda person_idx: (
            float(mean_distances.get(int(person_idx), float("inf"))),
            -float(presence_ratios[person_idx]),
            -float(mean_scores[person_idx])
            if np.isfinite(mean_scores[person_idx])
            else float("inf"),
            int(person_idx),
        ),
    )

    selected_persons = ranked_person_ids[: int(nb_persons_to_detect)]
    diagnostics = {
        "presence_ratios": presence_ratios,
        "eligible_person_ids": eligible_person_ids,
        "ranked_person_ids": ranked_person_ids,
        "mean_distances": mean_distances,
        "distance_frame_counts": distance_frame_counts,
        "usable_ball_frame_count": int(usable_ball_frame_count),
        "window_frame_count": int(window_frame_count),
    }
    return selected_persons, diagnostics


def resolve_personIDs_for_medicine_ball(
    all_frames_X_homog,
    all_frames_Y_homog,
    all_frames_scores_homog,
    all_frames_ball_centers,
    nb_persons_to_detect,
    detect_ball=True,
    ball_ordering_method="first_detected",
):
    """
    Resolve medicine-ball person ordering with a deterministic fallback.

    OUTPUTS:
    - selected_persons: list of tracked person slot indices
    - diagnostics: dict describing fallback/eligibility details
    """

    diagnostics = {
        "used_fallback": False,
        "fallback_reason": None,
    }
    if not detect_ball:
        diagnostics["used_fallback"] = True
        diagnostics["fallback_reason"] = "detect_ball=false"
        diagnostics["selected_persons"] = get_personIDs_with_highest_scores(
            all_frames_scores_homog,
            nb_persons_to_detect,
        )
        return diagnostics["selected_persons"], diagnostics

    if str(ball_ordering_method or "").strip().lower() == "on_click":
        diagnostics["used_fallback"] = True
        diagnostics["fallback_reason"] = "ball_ordering_method='on_click'"
        diagnostics["selected_persons"] = get_personIDs_with_highest_scores(
            all_frames_scores_homog,
            nb_persons_to_detect,
        )
        return diagnostics["selected_persons"], diagnostics

    selected_persons, medicine_ball_stats = get_personIDs_for_medicine_ball(
        all_frames_X_homog,
        all_frames_Y_homog,
        all_frames_scores_homog,
        all_frames_ball_centers,
        nb_persons_to_detect=nb_persons_to_detect,
    )
    diagnostics.update(medicine_ball_stats)
    diagnostics["selected_persons"] = selected_persons
    if int(diagnostics.get("usable_ball_frame_count", 0)) == 0:
        diagnostics["used_fallback"] = True
        diagnostics["fallback_reason"] = "opening_window_has_no_selected_ball"
        diagnostics["selected_persons"] = get_personIDs_with_highest_scores(
            all_frames_scores_homog,
            nb_persons_to_detect,
        )
        return diagnostics["selected_persons"], diagnostics
    if len(selected_persons) == 0:
        diagnostics["used_fallback"] = True
        diagnostics["fallback_reason"] = (
            "medicine_ball_presence_gate_removed_all_candidates"
        )
        diagnostics["selected_persons"] = get_personIDs_with_highest_scores(
            all_frames_scores_homog,
            nb_persons_to_detect,
        )
        return diagnostics["selected_persons"], diagnostics

    return selected_persons, diagnostics


def get_personIDs_on_click(
    video_file_path, frame_range, all_frames_X_homog, all_frames_Y_homog
):
    """
    Get the person IDs on click in the image

    INPUTS:
    - video_file_path: path to video file
    - frame_range: tuple (start_frame, end_frame)
    - all_frames_X_homog: shape (Nframes, Npersons, Nkpts)
    - all_frames_Y_homog: shape (Nframes, Npersons, Nkpts)

    OUTPUT:
    - selected_persons: list of int. The person IDs selected by the user
    """

    # Reorganize the coordinates to shape (Nframes, Npersons, Nkpts, Ndims)
    all_pose_coords = np.stack((all_frames_X_homog, all_frames_Y_homog), axis=-1)

    # Select person IDs on click on video/image
    selected_persons = select_persons_on_vid(
        video_file_path, frame_range, all_pose_coords
    )

    return selected_persons


def _build_ball_click_pose_coords(all_frames_ball_tracks):
    """
    Build click-select pose-like coordinates for ball tracks from tracked ball boxes.

    OUTPUTS:
    - all_pose_coords: shape (Nframes, Ntracks, 2, 2) with 2 keypoints per track (bbox corners)
    - ordered_track_ids: list of track IDs in first-appearance order
    """
    ordered_track_ids = []
    known_track_ids = set()
    for frame_tracks in all_frames_ball_tracks or []:
        for track in frame_tracks or []:
            if "id" not in track:
                continue
            track_id = int(track.get("id"))
            if track_id in known_track_ids:
                continue
            known_track_ids.add(track_id)
            ordered_track_ids.append(track_id)

    n_frames = len(all_frames_ball_tracks or [])
    n_tracks = len(ordered_track_ids)
    all_pose_coords = np.full((n_frames, n_tracks, 2, 2), np.nan, dtype=np.float32)
    if n_tracks == 0:
        return all_pose_coords, ordered_track_ids

    track_slot = {track_id: idx for idx, track_id in enumerate(ordered_track_ids)}
    for frame_idx, frame_tracks in enumerate(all_frames_ball_tracks or []):
        for track in frame_tracks or []:
            if "id" not in track:
                continue
            slot_idx = track_slot.get(int(track.get("id")))
            if slot_idx is None:
                continue
            box = track.get("box")
            if box is None:
                continue
            box_arr = _ensure_xyxy_boxes([box])
            if len(box_arr) == 0:
                continue
            x1, y1, x2, y2 = box_arr[0]
            all_pose_coords[frame_idx, slot_idx, 0] = [x1, y1]
            all_pose_coords[frame_idx, slot_idx, 1] = [x2, y2]
    return all_pose_coords, ordered_track_ids


def get_ball_trackIDs_on_click(video_file_path, frame_range, all_frames_ball_tracks):
    """
    Select one ball track ID by clicking ball boxes on a post-processing UI.
    """
    all_pose_coords, ordered_track_ids = _build_ball_click_pose_coords(
        all_frames_ball_tracks
    )
    if len(ordered_track_ids) == 0:
        return []

    selected_slots = select_persons_on_vid(
        video_file_path,
        frame_range,
        all_pose_coords,
        candidate_labels=[f"ball {track_id}" for track_id in ordered_track_ids],
        window_title="Select ball track to follow",
        selection_label="Selected ball",
        allow_multiple=False,
    )
    return [
        int(ordered_track_ids[slot_idx])
        for slot_idx in selected_slots
        if int(slot_idx) >= 0 and int(slot_idx) < len(ordered_track_ids)
    ]


def stitch_selected_ball_timeline(
    all_frames_ball_tracks, selected_track_id, max_jump_px=None
):
    """
    Rebuild a selected-ball timeline from one raw seed track ID across raw ID splits.
    """
    frame_count = len(all_frames_ball_tracks or [])
    if selected_track_id is None or frame_count == 0:
        return [], []

    selected_track_id = int(selected_track_id)
    stitch_max_jump_px = 120.0 if max_jump_px is None else float(max_jump_px)
    stitched_ids = [None for _ in range(frame_count)]
    stitched_centers = [None for _ in range(frame_count)]

    # Forward pass: start once the seed track becomes visible and keep continuity afterwards.
    forward_center = None
    forward_velocity = None
    forward_active = False
    for frame_idx, frame_tracks in enumerate(all_frames_ball_tracks or []):
        seed_track = next(
            (
                track
                for track in frame_tracks or []
                if int(track.get("id", -1)) == selected_track_id
                and track.get("visible", False)
                and track.get("center") is not None
            ),
            None,
        )
        if seed_track is not None:
            forward_active = True
        if not forward_active:
            continue

        selected_id, selected_center = select_ball_track_id(
            frame_tracks,
            selection_mode="auto",
            previous_selected_id=selected_track_id,
            previous_selected_center=forward_center,
            previous_selected_velocity=forward_velocity,
            ordering_method="first_detected",
            max_recovery_dist=stitch_max_jump_px,
            track_stats_by_id={},
        )
        stitched_ids[frame_idx] = (
            selected_track_id if selected_id is None else int(selected_id)
        )
        stitched_centers[frame_idx] = _normalize_ball_center(selected_center)
        forward_center, forward_velocity = _update_selected_ball_motion_state(
            forward_center,
            forward_velocity,
            stitched_centers[frame_idx],
        )

    # Backward pass: recover compatible earlier fragments before the first seed appearance.
    backward_center = None
    backward_velocity = None
    backward_active = False
    for frame_idx in range(frame_count - 1, -1, -1):
        frame_tracks = all_frames_ball_tracks[frame_idx]
        seed_track = next(
            (
                track
                for track in frame_tracks or []
                if int(track.get("id", -1)) == selected_track_id
                and track.get("visible", False)
                and track.get("center") is not None
            ),
            None,
        )
        if seed_track is not None:
            backward_active = True
        if not backward_active:
            continue

        reverse_velocity = None
        if backward_velocity is not None:
            reverse_velocity = (
                -float(backward_velocity[0]),
                -float(backward_velocity[1]),
            )
        selected_id, selected_center = select_ball_track_id(
            frame_tracks,
            selection_mode="auto",
            previous_selected_id=selected_track_id,
            previous_selected_center=backward_center,
            previous_selected_velocity=reverse_velocity,
            ordering_method="first_detected",
            max_recovery_dist=stitch_max_jump_px,
            track_stats_by_id={},
        )
        recovered_center = _normalize_ball_center(selected_center)
        if seed_track is None and recovered_center is None:
            continue
        if stitched_ids[frame_idx] is None:
            stitched_ids[frame_idx] = (
                selected_track_id if selected_id is None else int(selected_id)
            )
        if stitched_centers[frame_idx] is None:
            stitched_centers[frame_idx] = recovered_center
        backward_center, backward_velocity = _update_selected_ball_motion_state(
            backward_center,
            backward_velocity,
            recovered_center,
        )

    return stitched_ids, stitched_centers


def select_persons_on_vid(
    video_file_path,
    frame_range,
    all_pose_coords,
    candidate_labels=None,
    window_title="Select the persons to analyze in the desired order",
    selection_label="Selected",
    allow_multiple=True,
):
    """
    Interactive UI to select tracks from a video by clicking on bounding boxes.

    INPUTS:
    - video_file_path: path to video file
    - frame_range: tuple (start_frame, end_frame)
    - all_pose_coords: keypoints coordinates. shape (Nframes, Ntracks, Nkpts, Ndims)
    - candidate_labels: optional labels for candidate tracks
    - window_title: UI window title
    - selection_label: status text prefix
    - allow_multiple: allow selecting multiple IDs when True

    OUTPUT:
    - selected_persons : list with indices of selected tracks
    """

    BACKGROUND_COLOR = "white"
    SLIDER_COLOR = "#4682B4"
    SLIDER_EDGE_COLOR = (0.5, 0.5, 0.5, 0.5)
    UNSELECTED_COLOR = (1, 1, 1, 0.1)
    LINE_UNSELECTED_COLOR = "white"
    LINE_SELECTED_COLOR = "darkorange"

    def _format_selected_text():
        if len(selected_persons) == 0:
            return "None"
        return ", ".join([str(candidate_labels[idx]) for idx in selected_persons])

    def get_frame(frame_idx):
        """Get frame with caching"""
        actual_frame_idx = start_frame + frame_idx

        if actual_frame_idx in frame_cache:
            cache_order.remove(actual_frame_idx)
            cache_order.append(actual_frame_idx)
            return frame_cache[actual_frame_idx]

        cap.set(cv2.CAP_PROP_POS_FRAMES, actual_frame_idx)
        success, frame = cap.read()
        if not success:
            raise ValueError(f"Could not read frame {actual_frame_idx}")

        frame_cache[actual_frame_idx] = frame.copy()
        cache_order.append(actual_frame_idx)

        while len(frame_cache) > cache_size:
            oldest_frame = cache_order.pop(0)
            if oldest_frame in frame_cache:
                del frame_cache[oldest_frame]

        return frame

    def update_frame(val):
        frame_idx = int(frame_slider.val)
        frame = get_frame(frame_idx)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        for items in [rects, annotations]:
            for item in items:
                item.remove()
            items.clear()

        for person_idx, bbox in enumerate(all_bboxes[frame_idx]):
            if ~np.isnan(bbox).any():
                x_min, y_min, x_max, y_max = bbox.astype(int)
                rect = plt.Rectangle(
                    (x_min, y_min),
                    x_max - x_min,
                    y_max - y_min,
                    linewidth=1,
                    edgecolor="white",
                    facecolor=UNSELECTED_COLOR,
                    linestyle="-",
                    path_effects=[patheffects.withSimplePatchShadow()],
                    zorder=2,
                )
                ax_video.add_patch(rect)
                rects.append(rect)

                annotation = ax_video.text(
                    x_min,
                    y_min - 10,
                    f"{candidate_labels[person_idx]}",
                    color=LINE_UNSELECTED_COLOR,
                    fontsize=7,
                    fontweight="normal",
                    bbox=dict(
                        facecolor=UNSELECTED_COLOR,
                        edgecolor=LINE_UNSELECTED_COLOR,
                        boxstyle="square,pad=0.3",
                    ),
                    path_effects=[patheffects.withSimplePatchShadow()],
                    zorder=3,
                )
                annotations.append(annotation)
            else:
                rect = plt.Rectangle((np.nan, np.nan), np.nan, np.nan)
                ax_video.add_patch(rect)
                rects.append(rect)

        img_plot.set_data(frame_rgb)
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes != ax_video:
            return

        frame_idx = int(frame_slider.val)
        x, y = event.xdata, event.ydata

        for person_idx, bbox in enumerate(all_bboxes[frame_idx]):
            if ~np.isnan(bbox).any():
                x_min, y_min, x_max, y_max = bbox.astype(int)
                if x_min <= x <= x_max and y_min <= y <= y_max:
                    if person_idx in selected_persons:
                        rects[person_idx].set_linewidth(1)
                        rects[person_idx].set_edgecolor(LINE_UNSELECTED_COLOR)
                        selected_persons.remove(person_idx)
                    else:
                        if not allow_multiple:
                            for selected_idx in selected_persons.copy():
                                if selected_idx < len(rects):
                                    rects[selected_idx].set_linewidth(1)
                                    rects[selected_idx].set_edgecolor(
                                        LINE_UNSELECTED_COLOR
                                    )
                            selected_persons.clear()
                        rects[person_idx].set_linewidth(2)
                        rects[person_idx].set_edgecolor(LINE_SELECTED_COLOR)
                        selected_persons.append(person_idx)

                    status_text.set_text(
                        f"{selection_label}: {_format_selected_text()}"
                    )
                    fig.canvas.draw_idle()
                    break

    def on_hover(event):
        if event.inaxes != ax_video:
            return

        frame_idx = int(frame_slider.val)
        x, y = event.xdata, event.ydata

        for person_idx, bbox in enumerate(all_bboxes[frame_idx]):
            if person_idx >= len(rects):
                continue
            if ~np.isnan(bbox).any():
                x_min, y_min, x_max, y_max = bbox.astype(int)
                if x_min <= x <= x_max and y_min <= y <= y_max:
                    rects[person_idx].set_linewidth(2)
                    rects[person_idx].set_edgecolor(LINE_SELECTED_COLOR)
                    rects[person_idx].set_facecolor((1, 1, 0, 0.2))
                else:
                    rects[person_idx].set_facecolor(UNSELECTED_COLOR)
                    if person_idx in selected_persons:
                        rects[person_idx].set_linewidth(2)
                        rects[person_idx].set_edgecolor(LINE_SELECTED_COLOR)
                    else:
                        rects[person_idx].set_linewidth(1)
                        rects[person_idx].set_edgecolor(LINE_UNSELECTED_COLOR)
                fig.canvas.draw_idle()

    def on_ok(event):
        plt.close(fig)

    cap = cv2.VideoCapture(video_file_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_file_path}")
    start_frame, end_frame = frame_range

    frame_cache = {}
    cache_size = 20
    cache_order = []

    selected_persons = []
    n_frames, n_persons = all_pose_coords.shape[0], all_pose_coords.shape[1]
    if candidate_labels is None:
        candidate_labels = list(range(n_persons))
    else:
        candidate_labels = list(candidate_labels)
    if len(candidate_labels) < n_persons:
        candidate_labels = candidate_labels + list(
            range(len(candidate_labels), n_persons)
        )

    all_bboxes = []
    for frame_idx in range(n_frames):
        frame_bboxes = []
        for person_idx in range(n_persons):
            keypoints = all_pose_coords[frame_idx, person_idx]
            valid_keypoints = keypoints[~np.isnan(keypoints).all(axis=1)]
            if len(valid_keypoints) > 0:
                x_min, y_min = np.min(valid_keypoints, axis=0)
                x_max, y_max = np.max(valid_keypoints, axis=0)
                frame_bboxes.append((x_min, y_min, x_max, y_max))
            else:
                frame_bboxes.append((np.nan, np.nan, np.nan, np.nan))
        all_bboxes.append(frame_bboxes)
    all_bboxes = np.array(all_bboxes)

    first_frame = get_frame(0)
    frame_height, frame_width = first_frame.shape[:2]
    is_vertical = frame_height > frame_width
    if is_vertical:
        fig_height = frame_height / 250
    else:
        fig_height = max(frame_height / 300, 6)
    fig = plt.figure(figsize=(8, fig_height), num=window_title)
    fig.patch.set_facecolor(BACKGROUND_COLOR)

    video_axes_height = 0.7 if is_vertical else 0.6
    ax_video = plt.axes([0.1, 0.2, 0.8, video_axes_height])
    ax_video.axis("off")
    ax_video.set_facecolor(BACKGROUND_COLOR)

    frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    rects, annotations = [], []
    for person_idx, bbox in enumerate(all_bboxes[0]):
        if ~np.isnan(bbox).any():
            x_min, y_min, x_max, y_max = bbox.astype(int)
            rect = plt.Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                linewidth=1,
                edgecolor=LINE_UNSELECTED_COLOR,
                facecolor=UNSELECTED_COLOR,
                linestyle="-",
                path_effects=[patheffects.withSimplePatchShadow()],
                zorder=2,
            )
            ax_video.add_patch(rect)
            annotation = ax_video.text(
                x_min,
                y_min - 10,
                f"{candidate_labels[person_idx]}",
                color=LINE_UNSELECTED_COLOR,
                fontsize=7,
                fontweight="normal",
                bbox=dict(
                    facecolor=UNSELECTED_COLOR,
                    edgecolor=LINE_UNSELECTED_COLOR,
                    boxstyle="square,pad=0.3",
                    path_effects=[patheffects.withSimplePatchShadow()],
                ),
                zorder=3,
            )
            rects.append(rect)
            annotations.append(annotation)
    img_plot = ax_video.imshow(frame_rgb)

    ax_slider = plt.axes(
        [
            ax_video.get_position().x0,
            ax_video.get_position().y0 - 0.05,
            ax_video.get_position().width,
            0.04,
        ]
    )
    ax_slider.set_facecolor(BACKGROUND_COLOR)
    frame_slider = Slider(
        ax=ax_slider,
        label="",
        valmin=0,
        valmax=len(all_pose_coords) - 1,
        valinit=0,
        valstep=1,
        valfmt=None,
    )
    frame_slider.poly.set_edgecolor(SLIDER_EDGE_COLOR)
    frame_slider.poly.set_facecolor(SLIDER_COLOR)
    frame_slider.poly.set_linewidth(1)
    frame_slider.valtext.set_visible(False)

    ax_status = plt.axes(
        [
            ax_video.get_position().x0,
            ax_video.get_position().y0 - 0.1,
            2 * ax_video.get_position().width / 3,
            0.04,
        ]
    )
    ax_status.axis("off")
    status_text = ax_status.text(
        0.0, 0.5, f"{selection_label}: None", color="black", fontsize=10
    )

    ax_button = plt.axes(
        [
            ax_video.get_position().x0 + 3 * ax_video.get_position().width / 4,
            ax_video.get_position().y0 - 0.1,
            ax_video.get_position().width / 4,
            0.04,
        ]
    )
    ok_button = Button(ax_button, "OK", color=BACKGROUND_COLOR)

    frame_slider.on_changed(update_frame)
    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("motion_notify_event", on_hover)
    ok_button.on_clicked(on_ok)

    plt.show()

    return selected_persons


def compute_floor_line(
    trc_data,
    score_data,
    keypoint_names=["LBigToe", "RBigToe"],
    toe_speed_below=7,
    score_threshold=0.3,
):
    """
    Compute the floor line equation, angle, and direction
    from the feet keypoints when they have zero speed.

    N.B.: Y coordinates point downwards

    INPUTS:
    - trc_data: pd.DataFrame. The trc data
    - keypoint_names: list of str. The names of the keypoints to use
    - toe_speed_below: float. The speed threshold (px/frame) below which the keypoints are considered as not moving

    OUTPUT:
    - angle: float. The angle of the floor line in radians
    - xy_origin: list. The origin of the floor line
    - gait_direction: float. Left if < 0, 'right' otherwise
    """

    # Retrieve zero-speed coordinates for the foot
    low_speeds_X, low_speeds_Y = [], []
    gait_direction_val = []
    for kpt in keypoint_names:
        # Remove frames without data
        trc_data_kpt = trc_data[kpt].iloc[:, :2]
        score_data_kpt = score_data[kpt]
        start, end = indices_of_first_last_non_nan_chunks(
            score_data_kpt, chunk_choice_method="all"
        )
        trc_data_kpt_trim = trc_data_kpt.iloc[start:end].reset_index(drop=True)
        score_data_kpt_trim = score_data_kpt.iloc[start:end].reset_index(drop=True)

        # Compute euclidean speed
        speeds = np.linalg.norm(trc_data_kpt_trim.diff(), axis=1)

        # Remove speeds with low confidence
        speeds = np.where(score_data_kpt_trim > score_threshold, speeds, np.nan)

        # Get coordinates with low speeds, high
        low_speeds_coords = trc_data_kpt_trim[speeds < toe_speed_below]
        low_speeds_coords = low_speeds_coords[low_speeds_coords != 0]

        low_speeds_X_kpt = low_speeds_coords.iloc[:, 0].tolist()
        low_speeds_X += low_speeds_X_kpt
        low_speeds_Y += low_speeds_coords.iloc[:, 1].tolist()

        # gait direction (between [-1,1])
        X_trend_val = np.polyfit(range(len(low_speeds_X_kpt)), low_speeds_X_kpt, 1)[0]
        gait_direction_kpt = (
            X_trend_val
            * len(low_speeds_X_kpt)
            / (np.max(low_speeds_X_kpt) - np.min(low_speeds_X_kpt))
        )
        gait_direction_val.append(gait_direction_kpt)

    # Fit a line to the zero-speed coordinates
    floor_line = np.polyfit(low_speeds_X, low_speeds_Y, 1)  # (slope, intercept)
    angle = -np.arctan(floor_line[0])  # angle of the floor line in radians
    xy_origin = [0, floor_line[1]]  # origin of the floor line

    # Gait direction
    gait_direction = np.mean(gait_direction_val)

    return angle, xy_origin, gait_direction


def get_distance_from_camera(
    perspective_value=10,
    perspective_unit="distance_m",
    calib_file=None,
    height_px=1,
    height_m=1,
    cam_width=1,
    cam_height=1,
):
    """
    Compute the distance between the camera and the person based on the chosen perspective unit.

    INPUTS:
    - perspective_value: Value associated with the chosen perspective unit.
    - perspective_unit: Unit used to compute the distance. Can be 'distance_m', 'f_px', 'fov_rad', 'fov_deg', or 'from_calib'.
    - calib_file: Path to the toml calibration file.
    - height_px: Height of the person in pixels.
    - height_m: Height of the first person in meters.
    - cam_width: Width of the camera frame in pixels.
    - cam_height: Height of the camera frame in pixels.

    OUTPUTS:
    - distance_m: Distance between the camera and the person in meters.
    """

    if perspective_unit == "from_calib":
        if not calib_file:
            perspective_unit = "distance_m"
            distance_m = 10.0
            logging.warning(
                f"No calibration file provided. Using a default distance of {distance_m} m between the camera and the person to convert px to meters."
            )
        else:
            calib_params_dict = retrieve_calib_params(calib_file)
            f_px = calib_params_dict["K"][0][0][0]
            distance_m = f_px / height_px * height_m
    elif perspective_unit == "distance_m":
        distance_m = perspective_value
    elif perspective_unit == "f_px":
        f_px = perspective_value
        distance_m = f_px / height_px * height_m
    elif perspective_unit == "fov_rad":
        fov_rad = perspective_value
        f_px = max(cam_width, cam_height) / 2 / np.tan(fov_rad / 2)
        distance_m = f_px / height_px * height_m
    elif perspective_unit == "fov_deg":
        fov_rad = np.radians(perspective_value)
        f_px = max(cam_width, cam_height) / 2 / np.tan(fov_rad / 2)
        distance_m = f_px / height_px * height_m

    return distance_m


def get_floor_params(
    floor_angle="auto",
    xy_origin=["auto"],
    calib_file=None,
    height_px=1,
    height_m=1,
    fps=30,
    trc_data=pd.DataFrame(),
    score_data=pd.DataFrame(),
    toe_speed_below=1,
    score_threshold=0.5,
    cam_width=1,
    cam_height=1,
):
    """
    Compute the floor angle and the xy_origin based on calibration file, kinematics, or user input.

    INPUTS:
    - floor_angle: Method to compute the floor angle. Can be 'auto', 'from_calib', 'from_kinematics', or a numeric value in degrees.
    - xy_origin: Method to compute the xy_origin. Can be ['auto'], ['from_calib'], ['from_kinematics'], or a list of two numeric values in pixels [cx, cy].
    - calib_file: Path to a toml calibration file.
    - height_px: Height of the person in pixels.
    - height_m: Height of the first person in meters.
    - fps: Framerate of the video in frames per second. Used if estimating floor line from kinematics.
    - trc_data: DataFrame containing the pose data in pixels for one person. Used if estimating floor line from kinematics.
    - score_data: DataFrame containing the keypoint scores for one person. Used if estimating floor line from kinematics.
    - toe_speed_below: Speed below which the foot is considered to be stationary, in m/s. Used if estimating floor line from kinematics.
    - score_threshold: Minimum average keypoint score to consider a frame for floor line estimation.
    - cam_width: Width of the camera frame in pixels. Used if failed to estimate floor line from kinematics.
    - cam_height: Height of the camera frame in pixels. Used if failed to estimate floor line from kinematics.

    OUTPUTS:
    - floor_angle_estim: Estimated floor angle in radians.
    - xy_origin_estim: Estimated xy_origin as a list of two numeric values in pixels [cx, cy].
    - gait_direction: Estimated gait direction. 'left' if < 0, 'right' otherwise.
    """

    # Estimate floor angle from the calibration file
    if floor_angle == "from_calib" or xy_origin == ["from_calib"]:
        if not calib_file:
            if floor_angle == "from_calib":
                floor_angle = "auto"
            if xy_origin == ["from_calib"]:
                xy_origin = ["auto"]
            logging.warning(
                f"No calibration file provided. Estimating floor angle and xy_origin from the pose of the first selected person."
            )
        else:
            calib_params_dict = retrieve_calib_params(calib_file)

            R90z = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
            R270x = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])

            R_cam = cv2.Rodrigues(calib_params_dict["R"][0])[0]
            T_cam = np.array(calib_params_dict["T"][0])
            R_world, T_world = world_to_camera_persp(R_cam, T_cam)
            Rfloory = R90z.T @ R_world @ R270x.T
            T_world = R90z.T @ T_world
            floor_angle_calib = np.arctan2(Rfloory[0, 2], Rfloory[0, 0])

            cu = calib_params_dict["K"][0][0][2]
            cv = calib_params_dict["K"][0][1][2]
            cx = 0.0
            cy = cv + T_world[2] * height_px / height_m
            xy_origin_calib = [cx, cy]

    # Estimate xy_origin from the line formed by the toes when they are on the ground (where speed = 0)
    px_per_m = height_px / height_m
    toe_speed_below_px_frame = (
        toe_speed_below * px_per_m / fps
    )  # speed below which the foot is considered to be stationary
    try:
        if all(key in trc_data for key in ["LBigToe", "RBigToe"]):
            floor_angle_kin, xy_origin_kin, gait_direction = compute_floor_line(
                trc_data,
                score_data,
                keypoint_names=["LBigToe", "RBigToe"],
                toe_speed_below=toe_speed_below_px_frame,
                score_threshold=score_threshold,
            )
        else:
            floor_angle_kin, xy_origin_estim, gait_direction = compute_floor_line(
                trc_data,
                score_data,
                keypoint_names=["LAnkle", "RAnkle"],
                toe_speed_below=toe_speed_below_px_frame,
                score_threshold=score_threshold,
            )
            xy_origin_kin[1] = (
                xy_origin_kin[1] + 0.13 * px_per_m
            )  # approx. height of the ankle above the floor
            logging.warning(
                f"The RBigToe and LBigToe are missing from your pose estimation model. Using ankles - 13 cm to compute the floor line."
            )
    except:
        floor_angle_kin = 0
        xy_origin_kin = cam_width / 2, cam_height / 2
        gait_direction = 1
        logging.warning(
            f"Could not estimate the floor angle, xy_origin, and visible from person {0}. Make sure that the full body is visible. Using floor angle = 0°, xy_origin = [{cam_width / 2}, {cam_height / 2}] px, and visible_side = right."
        )

    # Determine final floor angle estimation
    if floor_angle == "from_calib":
        floor_angle_estim = floor_angle_calib
    elif floor_angle in ["auto", "from_kinematics"]:
        floor_angle_estim = floor_angle_kin
    else:
        try:
            floor_angle_estim = np.radians(float(floor_angle))
        except:
            raise ValueError(
                f'Invalid floor_angle: {floor_angle}. Must be "auto", "from_calib", "from_kinematics", or a numeric value in degrees.'
            )

    # Determine final xy_origin estimation
    if xy_origin == ["from_calib"]:
        xy_origin_estim = xy_origin_calib
    elif xy_origin in [["auto"], ["from_kinematics"]]:
        xy_origin_estim = xy_origin_kin
    else:
        try:
            xy_origin_estim = [float(v) for v in xy_origin]
        except:
            raise ValueError(
                f'Invalid xy_origin: {xy_origin}. Must be "auto", "from_calib", "from_kinematics", or a list of two numeric values in pixels.'
            )

    return floor_angle_estim, xy_origin_estim, gait_direction


def convert_px_to_meters(
    Q_coords_kpt,
    first_person_height,
    height_px,
    distance_m,
    cam_width,
    cam_height,
    cx,
    cy,
    floor_angle,
    visible_side="none",
):
    """
    Convert pixel coordinates to meters.
    Corrects for floor angle, floor level, and depth perspective errors.

    INPUTS:
    - Q_coords_kpt: pd.DataFrame. The xyz coordinates of a keypoint in pixels, with z filled with zeros
    - first_person_height: float. The height of the person in meters
    - height_px: float. The height of the person in pixels
    - cam_width: float. The width of the camera frame in pixels
    - cam_height: float. The height of the camera frame in pixels
    - distance_m: float. The distance between the camera and the person in meters
    - cx, cy: float. The origin of the image in pixels
    - floor_angle: float. The angle of the floor in radians
    - visible_side: str. The side of the person that is visible ('right', 'left', 'front', 'back', 'none')

    OUTPUT:
    - Q_coords_kpt_m: pd.DataFrame. The XYZ coordinates of a keypoint in meters
    """

    # u,v coordinates
    u = Q_coords_kpt.iloc[:, 0]
    v = Q_coords_kpt.iloc[:, 1]
    cu = cam_width / 2
    cv = cam_height / 2

    # Normative Z coordinates
    marker_name = Q_coords_kpt.columns[0]
    if (
        "marker_Z_positions" in globals()
        and visible_side != "none"
        and marker_name in marker_Z_positions[visible_side].keys()
    ):
        Z = u.copy()
        Z[:] = marker_Z_positions[visible_side][marker_name]
    else:
        Z = np.zeros_like(u)

    ## Compute X and Y coordinates in meters
    # X =   first_person_height / height_px * (u-cu)
    # Y = - first_person_height / height_px * (v-cv)
    ## With floor angle and level correction:
    # X =   first_person_height / height_px * ((u-cx)*np.cos(floor_angle) + (v-cy)*np.sin(floor_angle))
    # Y = - first_person_height / height_px * ((v-cy)*np.cos(floor_angle) + (u-cx)*np.sin(floor_angle))
    ## With floor angle and level correction, and depth perspective correction:
    scaling_factor = first_person_height / height_px
    X = scaling_factor * (
        ((u - cx) - Z / distance_m * (u - cu)) * np.cos(floor_angle)
        + ((v - cy) - Z / distance_m * (v - cv)) * np.sin(floor_angle)
    )
    Y = -scaling_factor * (
        ((v - cy) - Z / distance_m * (v - cv)) * np.cos(floor_angle)
        - ((u - cx) - Z / distance_m * (u - cu)) * np.sin(floor_angle)
    )

    # Assemble results
    Q_coords_kpt_m = pd.DataFrame(np.array([X, Y, Z]).T, columns=Q_coords_kpt.columns)

    return Q_coords_kpt_m


def process_fun(config_dict, video_file, time_range, frame_rate, result_dir):
    """
    Detect 2D joint centers from a video or a webcam with RTMLib.
    Compute selected joint and segment angles.
    Optionally save processed image files and video file.
    Optionally save processed poses as a TRC file, and angles as a MOT file (OpenSim compatible).

    This scripts:
    - loads skeleton information
    - reads stream from a video or a webcam
    - sets up the RTMLib pose tracker from RTMlib with specified parameters
    - detects poses within the selected time range
    - tracks people so that their IDs are consistent across frames
    - retrieves the keypoints with high enough confidence, and only keeps the persons with enough high-confidence keypoints
    - computes joint and segment angles, and flips those on the left/right side them if the respective foot is pointing to the left
    - draws bounding boxes around each person with their IDs
    - draws joint and segment angles on the body, and writes the values either near the joint/segment, or on the upper-left of the image with a progress bar
    - draws the skeleton and the keypoints, with a green to red color scale to account for their confidence
    - optionally show processed images, saves them, or saves them as a video
    - interpolates missing pose and angle sequences if gaps are not too large
    - filters them with the selected filter and parameters
    - optionally plots pose and angle data before and after processing for comparison
    - optionally saves poses for each person as a trc file, and angles as a mot file

    ⚠ Warning ⚠
    - The pose detection is only as good as the pose estimation algorithm, i.e., it is not perfect.
    - It will lead to reliable results only if the persons move in the 2D plane (sagittal or frontal plane).
    - The persons need to be filmed as perpendicularly as possible from their side.
    If you need research-grade markerless joint kinematics, consider using several cameras,
    and constraining angles to a biomechanically accurate model. See Pose2Sim for example:
    https://github.com/perfanalytics/pose2sim

    INPUTS:
    - a video or a webcam
    - a dictionary obtained from a configuration file (.toml extension)
    - a skeleton model

    OUTPUTS:
    - one trc file of joint coordinates per detected person
    - one mot file of joint angles per detected person
    - image files, video
    - a logs.txt file
    """

    # Base parameters
    video_dir = Path(config_dict.get("base").get("video_dir"))

    nb_persons_to_detect = config_dict.get("base").get("nb_persons_to_detect")
    if nb_persons_to_detect != "all":
        try:
            nb_persons_to_detect = int(nb_persons_to_detect)
            if nb_persons_to_detect < 1:
                logging.warning(
                    'nb_persons_to_detect must be "all" or > 1. Detecting all persons instead.'
                )
                nb_persons_to_detect = "all"
        except:
            logging.warning(
                'nb_persons_to_detect must be "all" or an integer. Detecting all persons instead.'
            )
            nb_persons_to_detect = "all"

    person_ordering_method = config_dict.get("base").get("person_ordering_method")

    first_person_height = config_dict.get("base").get("first_person_height")
    visible_side = config_dict.get("base").get("visible_side")
    if isinstance(visible_side, str):
        visible_side = [visible_side]

    # Pose from file
    load_trc_px = config_dict.get("base").get("load_trc_px")
    if load_trc_px == "":
        load_trc_px = None
    else:
        load_trc_px = Path(load_trc_px).resolve()
    compare = config_dict.get("base").get("compare")

    # Webcam settings
    webcam_id = config_dict.get("base").get("webcam_id")
    input_size = config_dict.get("base").get("input_size")

    # Output settings
    show_realtime_results = config_dict.get("base").get("show_realtime_results")
    realtime_ui_backend = config_dict.get("base").get("realtime_ui_backend", "opencv")
    realtime_window_title = config_dict.get("base").get(
        "realtime_window_title", "UmFit realtime"
    )
    save_vid = config_dict.get("base").get("save_vid")
    video_codec = _parse_video_codec(config_dict.get("base").get("video_codec", "mp4v"))
    save_img = config_dict.get("base").get("save_img")
    save_pose = config_dict.get("base").get("save_pose")
    calculate_angles = config_dict.get("base").get("calculate_angles")
    save_angles = config_dict.get("base").get("save_angles")
    hybrid_mode = bool(config_dict.get("base").get("hybrid_mode", False))
    hybrid_review_pose = bool(config_dict.get("base").get("hybrid_review_pose", True))
    hybrid_review_ball = bool(config_dict.get("base").get("hybrid_review_ball", True))
    hybrid_ui_backend = config_dict.get("base").get("hybrid_ui_backend", "matplotlib")

    # Pose_advanced settings
    slowmo_factor = config_dict.get("pose").get("slowmo_factor")
    pose_model = config_dict.get("pose").get("pose_model")
    mode = config_dict.get("pose").get("mode")
    det_frequency = config_dict.get("pose").get("det_frequency")
    backend = config_dict.get("pose").get("backend")
    device = config_dict.get("pose").get("device")
    tracking_mode = config_dict.get("pose").get("tracking_mode")
    synthpose_detector = config_dict.get("pose").get(
        "synthpose_detector", "yolox"
    )  # 'yolox', 'yolo26', 'rtdetr', 'rtdetrv4', or 'sam3'
    if tracking_mode == "deepsort":
        from deep_sort_realtime.deepsort_tracker import DeepSort

        deepsort_params = config_dict.get("pose").get("deepsort_params")
        try:
            deepsort_params = ast.literal_eval(deepsort_params)
        except:  # if within single quotes instead of double quotes when run with sports2d --mode """{dictionary}"""
            deepsort_params = (
                deepsort_params.strip("'")
                .replace("\n", "")
                .replace(" ", "")
                .replace(",", '", "')
                .replace(":", '":"')
                .replace("{", '{"')
                .replace("}", '"}')
                .replace('":"/', ":/")
                .replace('":"\\', ":\\")
            )
            deepsort_params = re.sub(
                r'"\[([^"]+)",\s?"([^"]+)\]"', r"[\1,\2]", deepsort_params
            )  # changes "[640", "640]" to [640,640]
            deepsort_params = json.loads(deepsort_params)
        deepsort_tracker = DeepSort(**deepsort_params)
        deepsort_tracker.tracker.tracks.clear()

    keypoint_likelihood_threshold = config_dict.get("pose").get(
        "keypoint_likelihood_threshold"
    )
    draw_keypoint_likelihood_threshold = _resolve_draw_likelihood_threshold(
        config_dict.get("pose").get("draw_keypoint_likelihood_threshold"),
        keypoint_likelihood_threshold,
    )
    draw_skeleton_likelihood_threshold = _resolve_draw_likelihood_threshold(
        config_dict.get("pose").get("draw_skeleton_likelihood_threshold"),
        draw_keypoint_likelihood_threshold,
    )
    average_likelihood_threshold = config_dict.get("pose").get(
        "average_likelihood_threshold"
    )
    keypoint_number_threshold = config_dict.get("pose").get("keypoint_number_threshold")
    max_distance = config_dict.get("pose").get("max_distance", None)
    detect_ball = bool(config_dict.get("pose").get("detect_ball", False))
    ball_trail_length = max(
        1, int(config_dict.get("pose").get("ball_trail_length", 20))
    )
    ball_trail_alpha = float(
        np.clip(config_dict.get("pose").get("ball_trail_alpha", 0.35), 0.0, 1.0)
    )
    ball_radius = max(1, int(config_dict.get("pose").get("ball_radius", 4)))
    ball_detection_threshold = float(
        np.clip(config_dict.get("pose").get("ball_detection_threshold", 0.1), 0.01, 0.9)
    )
    ball_max_jump_px = _parse_ball_max_jump_px(
        config_dict.get("pose").get("ball_max_jump_px", 120)
    )
    ball_color = _parse_ball_color(config_dict.get("pose").get("ball_color", [0, 0, 0]))
    ball_tracking_mode = (
        str(config_dict.get("pose").get("ball_tracking_mode", "sports2d"))
        .strip()
        .lower()
    )
    ball_ordering_method = _parse_ball_ordering_method(
        config_dict.get("pose").get("ball_ordering_method", "first_detected"),
    )
    ball_selection_mode = _parse_ball_selection_mode(
        config_dict.get("pose").get("ball_selection_mode", "auto"),
    )
    ball_selected_id = _parse_ball_selected_id(
        config_dict.get("pose").get("ball_selected_id", -1)
    )
    ball_tracking_max_distance = _parse_ball_max_jump_px(
        config_dict.get("pose").get("ball_tracking_max_distance", 120),
        default=120.0,
    )
    ball_track_max_missing_frames = max(
        0,
        int(config_dict.get("pose").get("ball_track_max_missing_frames", 12)),
    )
    ball_show_ids = bool(config_dict.get("pose").get("ball_show_ids", True))
    manual_roi = bool(config_dict.get("pose").get("manual_roi", False))
    manual_roi_padding_px = max(
        0,
        int(config_dict.get("pose").get("manual_roi_padding_px", 16)),
    )
    if manual_roi and load_trc_px:
        logging.warning("manual_roi=true is ignored when loading poses from TRC.")
        manual_roi = False
    ball_detector_backend = _parse_ball_detector_backend(
        config_dict.get("pose").get("ball_detector_backend", "same"),
        synthpose_detector=synthpose_detector,
    )
    sam3_show_realtime_masks = bool(
        config_dict.get("pose").get("sam3_show_realtime_masks", False)
    )
    sam3_realtime_mask_alpha = float(
        np.clip(
            config_dict.get("pose").get("sam3_realtime_mask_alpha", 0.22),
            0.0,
            1.0,
        )
    )
    motion_cfg = config_dict.get("motion", {}) or {}
    vertical_jump_requested = bool(motion_cfg.get("vertical_jump", False))
    motion_person_selection_target = _parse_motion_person_selection_target(
        motion_cfg.get("person_selection_target", "auto")
    )
    motion_person_presence_threshold = float(
        np.clip(motion_cfg.get("person_selection_presence_threshold", 0.8), 0.0, 1.0)
    )
    motion_person_confidence_threshold = float(
        np.clip(motion_cfg.get("person_selection_confidence_threshold", 0.3), 0.0, 1.0)
    )
    motion_person_size_min_ratio = float(
        np.clip(motion_cfg.get("person_selection_size_min_ratio", 0.35), 0.0, 1.0)
    )
    motion_person_score_threshold = float(
        np.clip(motion_cfg.get("person_selection_motion_score_threshold", 0.35), 0.0, 1.0)
    )

    # Pixel to meters conversion
    to_meters = config_dict.get("px_to_meters_conversion").get("to_meters")
    make_c3d = config_dict.get("px_to_meters_conversion").get("make_c3d")
    save_calib = config_dict.get("px_to_meters_conversion").get("save_calib")
    # Correct perspective effects
    perspective_value = config_dict.get("px_to_meters_conversion", {}).get(
        "perspective_value", 10.0
    )
    perspective_unit = config_dict.get("px_to_meters_conversion", {}).get(
        "perspective_unit", "distance_m"
    )
    # Calibration from person height
    floor_angle = config_dict.get("px_to_meters_conversion").get(
        "floor_angle", "auto"
    )  # 'auto' or float
    xy_origin = config_dict.get("px_to_meters_conversion").get(
        "xy_origin", ["auto"]
    )  # ['auto'] or [x, y]
    # Calibration from file
    calib_file = config_dict.get("px_to_meters_conversion").get("calib_file")
    if calib_file == "":
        calib_file = None
    else:
        calib_file = Path(calib_file).resolve()
        if not calib_file.is_file():
            raise FileNotFoundError(
                f"Error: Could not find calibration file {calib_file}. Check that the file exists."
            )

    # Angles advanced settings
    display_angle_values_on = config_dict.get("angles").get("display_angle_values_on")
    fontSize = config_dict.get("angles").get("fontSize")
    thickness = 1 if fontSize < 0.8 else 2
    joint_angle_names = config_dict.get("angles").get("joint_angles")
    segment_angle_names = config_dict.get("angles").get("segment_angles")
    angle_names = joint_angle_names + segment_angle_names
    angle_names = [angle_name.lower() for angle_name in angle_names]
    flip_left_right = config_dict.get("angles").get("flip_left_right")
    correct_segment_angles_with_floor_angle = config_dict.get("angles").get(
        "correct_segment_angles_with_floor_angle"
    )
    angle_output_mode = str(
        config_dict.get("angles").get("angle_output_mode", "legacy_continuous")
    ).lower()
    unwrap_angles = config_dict.get("angles").get("unwrap_angles", True)
    if angle_output_mode not in ["legacy_continuous", "bounded_principal"]:
        raise ValueError(
            f"Invalid angle_output_mode: {angle_output_mode}. Must be 'legacy_continuous' or 'bounded_principal'."
        )

    # Post-processing settings
    interpolate = config_dict.get("post-processing").get("interpolate")
    interp_gap_smaller_than = config_dict.get("post-processing").get(
        "interp_gap_smaller_than"
    )
    fill_large_gaps_with = config_dict.get("post-processing").get(
        "fill_large_gaps_with"
    )
    sections_to_keep = config_dict.get("post-processing").get("sections_to_keep")
    min_chunk_size = config_dict.get("post-processing").get("min_chunk_size")
    do_filter = config_dict.get("post-processing").get("filter")
    handle_LR_swap = config_dict.get("post-processing").get("handle_LR_swap", False)
    reject_outliers = config_dict.get("post-processing").get("reject_outliers", False)
    show_plots = config_dict.get("post-processing").get("show_graphs")
    save_plots = config_dict.get("post-processing").get("save_graphs")
    filter_type = config_dict.get("post-processing").get("filter_type")
    butterworth_filter_order = (
        config_dict.get("post-processing").get("butterworth", {}).get("order")
    )
    butterworth_filter_cutoff = (
        config_dict.get("post-processing")
        .get("butterworth", {})
        .get("cut_off_frequency")
    )
    gcv_filter_cutoff = (
        config_dict.get("post-processing")
        .get("gcv_spline", {})
        .get("gcv_cut_off_frequency")
    )
    gcv_smoothing_factor = (
        config_dict.get("post-processing")
        .get("gcv_spline", {})
        .get("gcv_smoothing_factor")
    )
    kalman_filter_trust_ratio = (
        config_dict.get("post-processing").get("kalman", {}).get("trust_ratio")
    )
    kalman_filter_smooth = (
        config_dict.get("post-processing").get("kalman", {}).get("smooth")
    )
    gaussian_filter_kernel = (
        config_dict.get("post-processing").get("gaussian", {}).get("sigma_kernel")
    )
    loess_filter_kernel = (
        config_dict.get("post-processing").get("loess", {}).get("nb_values_used")
    )
    median_filter_kernel = (
        config_dict.get("post-processing").get("median", {}).get("kernel_size")
    )
    butterworthspeed_filter_order = (
        config_dict.get("post-processing").get("butterworth_on_speed", {}).get("order")
    )
    butterworthspeed_filter_cutoff = (
        config_dict.get("post-processing")
        .get("butterworth_on_speed", {})
        .get("cut_off_frequency")
    )

    # Create output directories
    if video_file == "webcam":
        current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_file_stem = f"webcam_{current_date}"
        output_dir_name = f"{video_file_stem}_Sports2D"
        video_file_path = (
            result_dir / output_dir_name / f"webcam_{current_date}_raw.mp4"
        )
    else:
        video_file_stem = video_file.stem
        output_dir_name = f"{video_file_stem}_Sports2D"
        video_file_path = video_dir / video_file
    output_dir = result_dir / output_dir_name
    plots_output_dir = output_dir / f"{output_dir_name}_graphs"
    img_output_dir = output_dir / f"{output_dir_name}_img"
    pose_ball_output_dir = output_dir / "pose_ball"
    vid_output_path = output_dir / f"{output_dir_name}.mp4"
    vid_output_tmp_path = output_dir / f"{output_dir_name}__tmp.mp4"
    pose_output_path = output_dir / f"{output_dir_name}_px.trc"
    pose_output_path_m = output_dir / f"{output_dir_name}_m.trc"
    angles_output_path = output_dir / f"{output_dir_name}_angles.mot"
    output_dir.mkdir(parents=True, exist_ok=True)
    if save_img:
        img_output_dir.mkdir(parents=True, exist_ok=True)
    if save_plots:
        plots_output_dir.mkdir(parents=True, exist_ok=True)

    # Inverse kinematics settings
    do_ik = config_dict.get("kinematics").get("do_ik")
    use_augmentation = config_dict.get("kinematics").get("use_augmentation")
    inverse_dynamics_requested = _resolve_inverse_dynamics_requested(
        config_dict.get("kinematics")
    )
    participant_masses = config_dict.get("kinematics").get("participant_mass")
    participant_masses = (
        participant_masses
        if isinstance(participant_masses, list)
        else [participant_masses]
    )
    feet_on_floor = config_dict.get("kinematics").get("feet_on_floor")
    fastest_frames_to_remove_percent = config_dict.get("kinematics").get(
        "fastest_frames_to_remove_percent"
    )
    large_hip_knee_angles = config_dict.get("kinematics").get("large_hip_knee_angles")
    trimmed_extrema_percent = config_dict.get("kinematics").get(
        "trimmed_extrema_percent"
    )
    close_to_zero_speed_px = config_dict.get("kinematics").get("close_to_zero_speed_px")
    close_to_zero_speed_m = config_dict.get("kinematics").get("close_to_zero_speed_m")
    vertical_jump_enabled = bool(vertical_jump_requested)
    if vertical_jump_enabled and not to_meters:
        logging.warning(
            "motion.vertical_jump=true requires px_to_meters_conversion.to_meters=true. "
            "Skipping vertical jump estimation and overlays."
        )
        vertical_jump_enabled = False
    inverse_dynamics_enabled, inverse_dynamics_skip_reason = _resolve_inverse_dynamics_gate(
        inverse_dynamics_requested=inverse_dynamics_requested,
        do_ik=do_ik,
        vertical_jump_requested=vertical_jump_requested,
        vertical_jump_enabled=vertical_jump_enabled,
        to_meters=to_meters,
        save_angles=save_angles,
        calculate_angles=calculate_angles,
    )
    if inverse_dynamics_skip_reason is not None:
        logging.warning(inverse_dynamics_skip_reason)
    write_meter_pose = bool(save_pose or do_ik or use_augmentation)
    need_floor_corrected_angles = bool(
        to_meters
        and save_angles
        and calculate_angles
        and correct_segment_angles_with_floor_angle
    )
    need_postprocess_pose = bool(
        save_pose
        or vertical_jump_enabled
        or do_ik
        or use_augmentation
        or need_floor_corrected_angles
    )
    need_meter_pose = bool(
        to_meters
        and (write_meter_pose or vertical_jump_enabled or need_floor_corrected_angles)
    )
    # Create a Pose2Sim dictionary and fill in missing keys
    recursivedict = lambda: defaultdict(recursivedict)
    Pose2Sim_config_dict = recursivedict()
    if do_ik or use_augmentation:
        try:
            if use_augmentation:
                from Pose2Sim.markerAugmentation import augment_markers_all
            if do_ik:
                from Pose2Sim.kinematics import kinematics_all
        except ImportError:
            logging.error(
                "OpenSim package is not installed. Please install it to use inverse kinematics or marker augmentation features (see 'Full install' section of the documentation)."
            )
            raise ImportError(
                "OpenSim package is not installed. Please install it to use inverse kinematics or marker augmentation features (see 'Full install' section of the documentation)."
            )

        # Fill Pose2Sim dictionary (height and mass will be filled later)
        Pose2Sim_config_dict["project"]["project_dir"] = str(output_dir)
        Pose2Sim_config_dict["markerAugmentation"]["make_c3d"] = make_c3d
        Pose2Sim_config_dict["kinematics"] = config_dict.get("kinematics")
        # Temporarily recreate Pose2Sim file hierarchy
        pose3d_dir = Path(output_dir) / "pose-3d"
        pose3d_dir.mkdir(parents=True, exist_ok=True)
        kinematics_dir = Path(output_dir) / "kinematics"
        kinematics_dir.mkdir(parents=True, exist_ok=True)

    if do_filter:
        Pose2Sim_config_dict["personAssociation"]["handle_LR_swap"] = handle_LR_swap
        Pose2Sim_config_dict["filtering"]["reject_outliers"] = reject_outliers
        Pose2Sim_config_dict["filtering"]["filter"] = do_filter
        Pose2Sim_config_dict["filtering"]["type"] = filter_type
        Pose2Sim_config_dict["filtering"]["gcv_spline"]["cut_off_frequency"] = (
            gcv_filter_cutoff
        )
        Pose2Sim_config_dict["filtering"]["gcv_spline"]["smoothing_factor"] = (
            gcv_smoothing_factor
        )
        Pose2Sim_config_dict["filtering"]["butterworth"]["cut_off_frequency"] = (
            butterworth_filter_cutoff
        )
        Pose2Sim_config_dict["filtering"]["butterworth"]["order"] = (
            butterworth_filter_order
        )
        Pose2Sim_config_dict["filtering"]["kalman"]["trust_ratio"] = (
            kalman_filter_trust_ratio
        )
        Pose2Sim_config_dict["filtering"]["kalman"]["smooth"] = kalman_filter_smooth
        Pose2Sim_config_dict["filtering"]["gaussian"]["sigma_kernel"] = (
            gaussian_filter_kernel
        )
        Pose2Sim_config_dict["filtering"]["loess"]["nb_values_used"] = (
            loess_filter_kernel
        )
        Pose2Sim_config_dict["filtering"]["median"]["kernel_size"] = (
            median_filter_kernel
        )
        Pose2Sim_config_dict["filtering"]["butterworth_on_speed"]["order"] = (
            butterworthspeed_filter_order
        )
        Pose2Sim_config_dict["filtering"]["butterworth_on_speed"][
            "cut_off_frequency"
        ] = butterworthspeed_filter_cutoff

    # Set up video capture
    if video_file == "webcam":
        cap, out_vid, cam_width, cam_height, fps = setup_webcam(
            webcam_id, vid_output_path, input_size
        )
        frame_rate = fps
        frame_range = [0, sys.maxsize]
        frame_iterator = range(*frame_range)
        logging.warning(
            "Webcam input: the framerate may vary. If results are filtered, Sports2D will use the average framerate as input."
        )
    else:
        cap, out_vid, cam_width, cam_height, fps = setup_video(
            video_file_path, vid_output_path, save_vid
        )
        fps *= slowmo_factor
        start_time = get_start_time_ffmpeg(video_file_path)
        frame_range = (
            [
                int((time_range[0] - start_time) * frame_rate),
                int((time_range[1] - start_time) * frame_rate),
            ]
            if time_range
            else [0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))]
        )
        frame_iterator = tqdm(range(*frame_range))  # use a progress bar

    pending_frame = None
    pending_frame_index = None
    config_dict.setdefault("pose", {}).pop("_manual_person_roi", None)
    config_dict.setdefault("pose", {}).pop("_manual_ball_roi", None)
    if manual_roi and not load_trc_px:
        preview_frame_idx = int(frame_range[0])
        if video_file != "webcam":
            cap.set(cv2.CAP_PROP_POS_FRAMES, preview_frame_idx)
        preview_success, preview_frame = cap.read()
        if not preview_success or preview_frame is None:
            logging.warning(
                "manual_roi=true requested but the preview frame could not be read. Falling back to full-frame inference."
            )
        else:
            selected_rois = select_manual_rois(
                preview_frame,
                detect_ball=detect_ball,
                padding_px=manual_roi_padding_px,
                window_prefix="Sports2D Manual ROI",
            )
            person_roi = selected_rois.get("person_roi")
            ball_roi = selected_rois.get("ball_roi")
            if person_roi is not None:
                config_dict.setdefault("pose", {})["_manual_person_roi"] = list(
                    person_roi
                )
                if detect_ball and ball_roi is not None:
                    config_dict["pose"]["_manual_ball_roi"] = list(ball_roi)
                logging.info(
                    "Manual ROI enabled: person_roi=%s%s",
                    tuple(person_roi),
                    f", ball_roi={tuple(ball_roi)}"
                    if detect_ball and ball_roi is not None
                    else "",
                )
                if (
                    detect_ball
                    and str(pose_model).lower() in ["synthpose", "synthpose_base"]
                    and ball_detector_backend == "same"
                    and ball_roi is not None
                    and tuple(ball_roi) != tuple(person_roi)
                ):
                    logging.info(
                        "ball_detector_backend='same' will use the union of person_roi and ball_roi inside the shared detector."
                    )
            pending_frame = preview_frame.copy()
            pending_frame_index = preview_frame_idx

    motion_floor_angle_overlay = 0.0
    motion_floor_origin_overlay = (cam_width / 2.0, cam_height / 2.0)
    # Select the appropriate model based on the model_type
    logging.info("\nEstimating pose...")
    pose_model_name = pose_model
    backend_name = "rtmlib"
    display_runtime_backend = "initializing"
    realtime_display = None
    display_paused = False
    stop_requested = False
    dropped_frames = 0
    session_start_perf = time.perf_counter()

    if show_realtime_results:
        try:
            screen_width, screen_height = get_screen_size()
            display_width, display_height = calculate_display_size(
                cam_width, cam_height, screen_width, screen_height, margin=50
            )
        except Exception:
            display_width, display_height = cam_width, cam_height

        realtime_display = create_realtime_display(
            backend=realtime_ui_backend,
            window_title=realtime_window_title,
            display_width=display_width,
            display_height=display_height,
            model_name=pose_model_name,
            runtime_backend=display_runtime_backend,
            webcam_id=webcam_id if video_file == "webcam" else None,
            save_video=(save_vid or video_file == "webcam"),
            frame_size=(cam_width, cam_height),
        )
        realtime_display.set_session_state("Initializing")

    # Check if SynthPose is requested
    use_synthpose = pose_model_name.lower() in ["synthpose", "synthpose_base"]

    if use_synthpose:
        # SynthPose mode
        if not SYNTHPOSE_AVAILABLE:
            raise ImportError(
                "SynthPose is not available. Install with: pip install torch transformers\n"
                "Or use a different pose_model (body_with_feet, whole_body, body, etc.)"
            )
        synthpose_mode = "huge" if pose_model_name.lower() == "synthpose" else "base"
        # Use full 52-keypoint SynthPose skeleton for angle calculation and visualization
        pose_model = create_synthpose_skeleton()
        ModelClass = None  # Not used for SynthPose
        mode = synthpose_mode
        logging.info(
            f"Using SynthPose ({synthpose_mode}) - VitPose from HuggingFace Transformers with 52 keypoints"
        )
    else:
        # RTMLib mode (default)
        pose_model, ModelClass, mode = setup_model_class_mode(
            pose_model, mode, config_dict
        )

    # Select device and backend
    if use_synthpose:
        # SynthPose uses PyTorch directly, not ONNX - skip Pose2Sim backend setup
        # Device will be auto-detected by the pose backend using torch.cuda.is_available()
        pass  # Keep original device value from config ('auto', 'cuda', 'cpu', etc.)
    else:
        # RTMLib uses ONNX backends
        backend, device = setup_backend_device(backend=backend, device=device)

    # Skip pose estimation or set it up:
    if load_trc_px:
        if not "_px" in str(load_trc_px):
            logging.error(f"\n{load_trc_px} file needs to be in px, not in meters.")
        logging.info(
            f"\nUsing a pose file instead of running pose estimation and tracking: {load_trc_px}."
        )
        display_runtime_backend = "trc"
        backend_name = "trc"
        # Load pose file in px
        Q_coords, _, time_col, keypoints_names, _ = read_trc(load_trc_px)
        Q_coords, keypoints_names = strip_auxiliary_trc_markers(
            Q_coords,
            keypoints_names,
            ignored_marker_names=("ball",),
        )
        t0 = time_col[0]
        tf = time_col.iloc[-1]
        keypoints_ids = [i for i in range(len(keypoints_names))]
        keypoints_all, scores_all = load_pose_file(Q_coords)

        for pre, _, node in RenderTree(pose_model):
            if node.name in keypoints_names:
                node.id = keypoints_names.index(node.name)
        if time_range:
            frame_range = [
                abs(time_col - time_range[0]).idxmin(),
                abs(time_col - time_range[1]).idxmin() + 1,
            ]
        else:
            frame_range = [0, len(Q_coords)]
        frame_iterator = tqdm(range(*frame_range))

    else:
        t0 = 0
        tf = (
            (cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1) / fps
            if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0
            else float("inf")
        )

        # Set up pose tracker using unified backend
        try:
            pose_tracker = create_pose_backend(config_dict)
            backend_name = pose_tracker.backend_name
            display_runtime_backend = backend_name
            use_synthpose = backend_name == "synthpose"
            logging.info(
                f"{pose_tracker.backend_name} tracker initialized with {pose_tracker.num_keypoints} keypoints"
            )
        except Exception as e:
            logging.error(f"Error: Pose estimation backend initialization failed: {e}")
            raise ValueError(
                f"Error: Pose estimation backend initialization failed: {e}"
            )

        if hasattr(pose_tracker, "prepare_video_context"):
            pose_tracker.prepare_video_context(
                video_file_path=video_file_path if video_file != "webcam" else None,
                frame_range=frame_range,
                input_kind="webcam" if video_file == "webcam" else "video",
            )

        keypoints_names = list(getattr(pose_tracker, "keypoint_names", []) or [])
        if len(keypoints_names) == 0:
            indexed_names = sorted(
                [
                    (int(node.id), str(node.name))
                    for _, _, node in RenderTree(pose_model)
                    if node.id is not None
                ],
                key=lambda item: item[0],
            )
            keypoints_names = [name for _, name in indexed_names]
        keypoints_ids = list(range(len(keypoints_names)))
        kpt_id_max = len(keypoints_names)
        pose_model_with_output_ids = _remap_pose_model_ids_by_keypoint_names(
            pose_model,
            keypoints_names,
        )

        logging.info(
            f"Persons detection is run every {det_frequency} frames (pose estimation is run at every frame). Tracking is done with {tracking_mode}."
        )

        if tracking_mode == "deepsort":
            logging.info(f"Deepsort parameters: {deepsort_params}.")
        if tracking_mode not in ["deepsort", "sports2d"]:
            logging.warning(
                f"Tracking mode {tracking_mode} is not implemented. 'sports2d' is recommended."
            )
        logging.info(
            f"{'All persons are' if nb_persons_to_detect == 'all' else f'{nb_persons_to_detect} persons are' if nb_persons_to_detect > 1 else '1 person is'} analyzed. Person ordering method is {person_ordering_method}."
        )
        logging.info(
            f"{keypoint_likelihood_threshold=}, "
            f"{draw_keypoint_likelihood_threshold=}, "
            f"{draw_skeleton_likelihood_threshold=}, "
            f"{average_likelihood_threshold=}, "
            f"{keypoint_number_threshold=}"
        )

    ball_overlay_enabled = detect_ball and not load_trc_px
    if detect_ball and load_trc_px:
        logging.warning("detect_ball=true is ignored when loading poses from TRC.")
    if ball_overlay_enabled and not hasattr(pose_tracker, "last_detections"):
        logging.warning(
            "detect_ball=true requested but backend does not expose detection metadata. Disabling ball overlay."
        )
        ball_overlay_enabled = False
    ball_multi_id_tracking = ball_overlay_enabled and ball_tracking_mode == "sports2d"
    if ball_overlay_enabled and ball_tracking_mode != "sports2d":
        logging.warning(
            "Unsupported ball_tracking_mode '%s'. Falling back to legacy single-ball selection.",
            ball_tracking_mode,
        )
    if ball_multi_id_tracking and ball_selection_mode == "auto":
        logging.info(
            "Ball auto-selection ordering method is '%s'.", ball_ordering_method
        )
    if (
        ball_multi_id_tracking
        and ball_selection_mode == "id"
        and ball_selected_id is None
    ):
        logging.warning(
            "ball_selection_mode='id' requires ball_selected_id >= 0. Falling back to auto selection."
        )
        ball_selection_mode = "auto"
    uses_sam3_mask_source = use_synthpose and (
        str(synthpose_detector).strip().lower() == "sam3"
        or (detect_ball and ball_detector_backend == "sam3")
    )
    sam3_realtime_overlay_enabled = (
        bool(show_realtime_results)
        and uses_sam3_mask_source
        and sam3_show_realtime_masks
    )
    sam3_ball_mask_flag_enabled = bool(ball_overlay_enabled and uses_sam3_mask_source)

    raw_keypoint_names = keypoints_names.copy()
    L_R_direction_idx = None
    if flip_left_right:
        try:
            Ltoe_idx = keypoints_ids[keypoints_names.index("LBigToe")]
            LHeel_idx = keypoints_ids[keypoints_names.index("LHeel")]
            Rtoe_idx = keypoints_ids[keypoints_names.index("RBigToe")]
            RHeel_idx = keypoints_ids[keypoints_names.index("RHeel")]
            L_R_direction_idx = [Ltoe_idx, LHeel_idx, Rtoe_idx, RHeel_idx]
            has_toe_heel = True
        except ValueError:
            logging.warning(
                f"Missing 'LBigToe', 'LHeel', 'RBigToe', 'RHeel' keypoints. flip_left_right will be set to False"
            )
            flip_left_right = False
            has_toe_heel = False
    else:
        has_toe_heel = False

    if calculate_angles:
        for ang_name in angle_names:
            ang_params = angle_dict.get(ang_name)
            kpts = ang_params[0]
            if any(item not in keypoints_names + ["Neck", "Hip"] for item in kpts):
                logging.warning(
                    f"Skipping {ang_name} angle computation because at least one of the following keypoints is not provided by the pose estimation model: {ang_params[0]}."
                )

    # %% ==================================================
    # Process video or webcam feed
    # ====================================================
    logging.info(f"\nProcessing video stream...")
    # logging.info(f"{'Video, ' if save_vid else ''}{'Images, ' if save_img else ''}{'Pose, ' if save_pose else ''}{'Angles ' if save_angles else ''}{'and ' if save_angles or save_img or save_pose or save_vid else ''}Logs will be saved in {result_dir}.")
    (
        all_frames_X,
        all_frames_X_flipped,
        all_frames_Y,
        all_frames_scores,
        all_frames_angles,
    ) = [], [], [], [], []
    all_frames_X_raw, all_frames_Y_raw, all_frames_scores_raw = [], [], []
    all_frames_ball_centers, all_frames_ball_boxes = [], []
    all_frames_ball_scores = []
    all_frames_ball_tracks, all_frames_selected_ball_ids = [], []
    all_frames_sam3_ball_mask_available = []
    ball_trail_points = []
    ball_trail_points_by_id = {}
    ball_previous_center = None
    ball_previous_velocity = None
    ball_selected_track_id = ball_selected_id if ball_selection_mode == "id" else None
    ball_prev_keypoints = np.empty((0, 1, 2), dtype=np.float32)
    ball_track_ids = []
    ball_track_missing_counts = []
    ball_track_velocities_by_id = {}
    next_ball_track_id = 0
    ball_track_stats_by_id = {}
    ball_warned_missing_likelihood = False
    # Keep a valid keypoint schema even when early frames contain no detected persons.
    new_keypoints_names, new_keypoints_ids = (
        keypoints_names.copy(),
        keypoints_ids.copy(),
    )
    frame_processing_times = []
    frame_count = int(pending_frame_index) if pending_frame_index is not None else 0
    if np.isfinite(tf):
        first_frame = max(int(t0 * fps), frame_range[0])
        last_frame = min(int(tf * fps), frame_range[1] - 1)
    else:
        first_frame = frame_range[0]
        last_frame = frame_range[1] - 1
    if first_frame >= last_frame:
        logging.error(
            "Error: No frames to process. Check that your time_range is coherent with the video duration."
        )
        raise ValueError(
            "Error: No frames to process. Check that your time_range is coherent with the video duration."
        )

    consecutive_grab_failures = 0
    while cap.isOpened():
        # Skip to the starting frame
        if frame_count < first_frame:
            cap.read()
            frame_count += 1
            continue

        for frame_nb in frame_iterator:
            if show_realtime_results and realtime_display is not None:
                while display_paused:
                    pause_events = realtime_display.poll_events(delay_ms=30)
                    if pause_events.get("stop"):
                        stop_requested = True
                        break
                    if pause_events.get("toggle_pause") or pause_events.get("resume"):
                        display_paused = False
                        realtime_display.set_session_state("Live")
                        break
                if stop_requested:
                    break

            start_time = datetime.now()
            if pending_frame is not None and frame_count == first_frame:
                success, frame = True, pending_frame.copy()
                pending_frame = None
            else:
                success, frame = cap.read()
            frame_count += 1
            frame_ball_center = None
            frame_ball_boxes = np.empty((0, 4), dtype=np.float32)
            frame_ball_scores = np.empty((0,), dtype=np.float32)
            frame_ball_tracks = []
            frame_selected_ball_id = (
                ball_selected_track_id if ball_multi_id_tracking else None
            )
            frame_sam3_ball_mask_available = False
            detection_meta = {}

            # If frame not grabbed
            if not success:
                logging.warning(f"Failed to grab frame {frame_count - 1}.")
                dropped_frames += 1
                consecutive_grab_failures += 1
                if show_realtime_results and realtime_display is not None:
                    if consecutive_grab_failures >= 3:
                        realtime_display.set_session_state("Camera Lost")
                    fail_events = realtime_display.poll_events(delay_ms=1)
                    if fail_events.get("stop"):
                        stop_requested = True
                        break
                    if fail_events.get("toggle_pause"):
                        display_paused = not display_paused
                        realtime_display.set_session_state(
                            "Paused" if display_paused else "Live"
                        )

                kpt_count = len(new_keypoints_names)
                nan_pose = np.full((1, kpt_count), np.nan)
                raw_nan_pose = np.full((1, len(raw_keypoint_names)), np.nan)
                all_frames_X.append(nan_pose.copy())
                all_frames_X_flipped.append(nan_pose.copy())
                all_frames_Y.append(nan_pose.copy())
                all_frames_scores.append(nan_pose.copy())
                all_frames_X_raw.append(raw_nan_pose.copy())
                all_frames_Y_raw.append(raw_nan_pose.copy())
                all_frames_scores_raw.append(raw_nan_pose.copy())
                all_frames_ball_centers.append(None)
                all_frames_ball_boxes.append(np.empty((0, 4), dtype=np.float32))
                all_frames_ball_scores.append(np.empty((0,), dtype=np.float32))
                all_frames_ball_tracks.append([])
                all_frames_selected_ball_ids.append(frame_selected_ball_id)
                all_frames_sam3_ball_mask_available.append(False)
                if save_angles and calculate_angles:
                    all_frames_angles.append(np.full((1, len(angle_names)), np.nan))
                continue
            consecutive_grab_failures = 0

            # Retrieve pose or Estimate pose and track people
            if load_trc_px:
                if frame_nb >= len(keypoints_all):
                    break
                keypoints = keypoints_all[frame_nb]
                scores = scores_all[frame_nb]
            else:
                # Save video on the fly if the input is a webcam
                if video_file == "webcam":
                    out_vid.write(frame)

                try:  # Frames with no detection cause errors on MacOS CoreMLExecutionProvider
                    # Detect poses
                    keypoints, scores = pose_tracker(frame)
                    if ball_overlay_enabled or sam3_realtime_overlay_enabled:
                        detection_meta = (
                            getattr(pose_tracker, "last_detections", {}) or {}
                        )

                    # Non maximum suppression (at pose level, not detection, and only using likely keypoints)
                    frame_shape = frame.shape
                    mask_scores = np.mean(scores, axis=1) > 0.2

                    likely_keypoints = np.where(
                        mask_scores[:, np.newaxis, np.newaxis], keypoints, np.nan
                    )
                    likely_scores = np.where(mask_scores[:, np.newaxis], scores, np.nan)
                    likely_bboxes = bbox_xyxy_compute(
                        frame_shape, likely_keypoints, padding=0
                    )
                    score_likely_bboxes = np.nanmean(likely_scores, axis=1)

                    valid_indices = np.where(~np.isnan(score_likely_bboxes))[0]
                    if len(valid_indices) > 0:
                        valid_bboxes = likely_bboxes[valid_indices]
                        valid_scores = score_likely_bboxes[valid_indices]
                        keep_valid = nms(valid_bboxes, valid_scores, nms_thr=0.45)
                        keep = valid_indices[keep_valid]
                    else:
                        keep = []
                    keypoints, scores = likely_keypoints[keep], likely_scores[keep]

                    # # Debugging: display detected keypoints on the frame
                    # colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255), (0,255,255), (128,0,0), (0,128,0), (0,0,128), (128,128,0), (128,0,128), (0,128,128)]
                    # bboxes = likely_bboxes[keep]
                    # for person_idx in range(len(keypoints)):
                    #     for kpt_idx, kpt in enumerate(keypoints[person_idx]):
                    #         if not np.isnan(kpt).any():
                    #             cv2.circle(frame, (int(kpt[0]), int(kpt[1])), 3, colors[person_idx%len(colors)], -1)
                    #     if not np.isnan(bboxes[person_idx]).any():
                    #         cv2.rectangle(frame, (int(bboxes[person_idx][0]), int(bboxes[person_idx][1])), (int(bboxes[person_idx][2]), int(bboxes[person_idx][3])), colors[person_idx%len(colors)], 1)
                    # cv2.imshow('UmFit realtime', frame)

                    # Track poses across frames
                    if tracking_mode == "deepsort":
                        keypoints, scores = sort_people_deepsort(
                            keypoints, scores, deepsort_tracker, frame, frame_count
                        )
                    if tracking_mode == "sports2d":
                        if "prev_keypoints" not in locals():
                            prev_keypoints = keypoints
                        prev_keypoints, keypoints, scores = sort_people_sports2d(
                            prev_keypoints,
                            keypoints,
                            scores=scores,
                            max_dist=max_distance,
                        )
                    else:
                        pass

                    # # Debugging: display detected keypoints on the frame
                    # colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255), (0,255,255), (128,0,0), (0,128,0), (0,0,128), (128,128,0), (128,0,128), (0,128,128)]
                    # for person_idx in range(len(keypoints)):
                    #     for kpt_idx, kpt in enumerate(keypoints[person_idx]):
                    #         if not np.isnan(kpt).any():
                    #             cv2.circle(frame, (int(kpt[0]), int(kpt[1])), 3, colors[person_idx%len(colors)], -1)
                    #         # if not np.isnan(bboxes[person_idx]).any():
                    #         #     cv2.rectangle(frame, (int(bboxes[person_idx][0]), int(bboxes[person_idx][1])), (int(bboxes[person_idx][2]), int(bboxes[person_idx][3])), colors[person_idx%len(colors)], 1)
                    #     cv2.imshow('UmFit realtime', frame)
                    # # if (cv2.waitKey(1) & 0xFF) == ord('q') or (cv2.waitKey(1) & 0xFF) == 27:
                    # #     break
                    # # input()
                except Exception as e:
                    logging.debug(
                        "Pose estimation failed at frame %s: %s", frame_count - 1, e
                    )
                    keypoints = np.full((1, kpt_id_max, 2), fill_value=np.nan)
                    scores = np.full((1, kpt_id_max), fill_value=np.nan)
                    detection_meta = {}

                if ball_overlay_enabled:
                    frame_ball_boxes = _ensure_xyxy_boxes(
                        detection_meta.get("ball_boxes")
                    )
                    frame_ball_scores = _ensure_score_vector(
                        detection_meta.get("ball_scores"),
                        expected_len=len(frame_ball_boxes),
                    )
                    frame_ball_boxes, frame_ball_scores = dedupe_ball_detections(
                        frame_ball_boxes,
                        frame_ball_scores,
                    )
                    if ball_multi_id_tracking:
                        previous_selected_track_id = ball_selected_track_id
                        (
                            frame_ball_tracks,
                            ball_prev_keypoints,
                            ball_track_ids,
                            ball_track_missing_counts,
                            next_ball_track_id,
                        ) = track_balls_sports2d(
                            frame_ball_boxes,
                            ball_prev_keypoints,
                            ball_track_ids,
                            ball_track_missing_counts,
                            next_ball_track_id,
                            ball_scores=frame_ball_scores,
                            track_velocities_by_id=ball_track_velocities_by_id,
                            max_dist=ball_tracking_max_distance,
                            max_missing_frames=ball_track_max_missing_frames,
                        )
                        ball_track_stats_by_id = _update_ball_track_stats(
                            ball_track_stats_by_id,
                            frame_ball_tracks,
                            frame_index=frame_nb,
                        )
                        active_ball_track_ids = [
                            int(track.get("id"))
                            for track in frame_ball_tracks
                            if "id" in track
                        ]
                        if ball_selection_mode == "auto":
                            if (
                                ball_ordering_method == "highest_likelihood"
                                and len(active_ball_track_ids) > 0
                                and not ball_warned_missing_likelihood
                                and not _has_ball_confidence_stats(
                                    active_ball_track_ids, ball_track_stats_by_id
                                )
                            ):
                                logging.warning(
                                    "ball_ordering_method='highest_likelihood' requested but detector confidence "
                                    "is unavailable. Falling back to 'first_detected' behavior."
                                )
                                ball_warned_missing_likelihood = True
                        frame_selected_ball_id, selected_center = select_ball_track_id(
                            frame_ball_tracks,
                            selection_mode=ball_selection_mode,
                            requested_track_id=ball_selected_id,
                            previous_selected_id=ball_selected_track_id,
                            previous_selected_center=ball_previous_center,
                            previous_selected_velocity=ball_previous_velocity,
                            ordering_method=ball_ordering_method,
                            track_stats_by_id=ball_track_stats_by_id,
                            max_recovery_dist=ball_tracking_max_distance
                            if ball_tracking_max_distance is not None
                            else ball_max_jump_px,
                        )

                        frame_ball_center = selected_center
                        if frame_ball_center is not None:
                            gate_reference = (
                                ball_previous_center
                                if frame_selected_ball_id == previous_selected_track_id
                                else None
                            )
                            if (
                                gate_reference is not None
                                and ball_max_jump_px is not None
                            ):
                                jump_dist = float(
                                    np.hypot(
                                        frame_ball_center[0] - gate_reference[0],
                                        frame_ball_center[1] - gate_reference[1],
                                    )
                                )
                                if jump_dist > float(ball_max_jump_px):
                                    frame_ball_center = None

                        if frame_selected_ball_id is None:
                            ball_previous_center = None
                            ball_previous_velocity = None
                        elif frame_ball_center is not None:
                            ball_previous_center, ball_previous_velocity = (
                                _update_selected_ball_motion_state(
                                    ball_previous_center,
                                    ball_previous_velocity,
                                    frame_ball_center,
                                )
                            )
                            selected_trail = ball_trail_points_by_id.get(
                                frame_selected_ball_id, []
                            )
                            selected_trail.append(tuple(frame_ball_center))
                            if len(selected_trail) > ball_trail_length:
                                selected_trail = selected_trail[-ball_trail_length:]
                            ball_trail_points_by_id[frame_selected_ball_id] = (
                                selected_trail
                            )
                        elif ball_previous_velocity is not None:
                            ball_previous_center, ball_previous_velocity = (
                                _update_selected_ball_motion_state(
                                    ball_previous_center,
                                    ball_previous_velocity,
                                    None,
                                )
                            )

                        ball_selected_track_id = frame_selected_ball_id
                    else:
                        ball_candidates = extract_ball_centers(
                            {"ball_boxes": frame_ball_boxes}
                        )
                        frame_ball_center = select_ball_center(
                            ball_candidates,
                            previous_center=ball_previous_center,
                            max_jump_px=ball_max_jump_px,
                            previous_velocity=ball_previous_velocity,
                        )
                        if frame_ball_center is not None:
                            if ball_previous_center is not None:
                                frame_velocity = (
                                    float(
                                        frame_ball_center[0] - ball_previous_center[0]
                                    ),
                                    float(
                                        frame_ball_center[1] - ball_previous_center[1]
                                    ),
                                )
                                if ball_previous_velocity is None:
                                    ball_previous_velocity = frame_velocity
                                else:
                                    smoothing = 0.75
                                    ball_previous_velocity = (
                                        smoothing * float(ball_previous_velocity[0])
                                        + (1.0 - smoothing) * frame_velocity[0],
                                        smoothing * float(ball_previous_velocity[1])
                                        + (1.0 - smoothing) * frame_velocity[1],
                                    )
                            else:
                                ball_previous_velocity = (0.0, 0.0)
                            ball_previous_center = frame_ball_center
                            ball_trail_points.append(frame_ball_center)
                            if len(ball_trail_points) > ball_trail_length:
                                ball_trail_points = ball_trail_points[
                                    -ball_trail_length:
                                ]
                        elif ball_previous_velocity is not None:
                            ball_previous_velocity = (
                                0.6 * float(ball_previous_velocity[0]),
                                0.6 * float(ball_previous_velocity[1]),
                            )

                if sam3_ball_mask_flag_enabled:
                    frame_sam3_ball_mask_available = _sam3_mask_available(
                        detection_meta.get("sam3_ball_meta") or detection_meta
                    )

            # Process coordinates and compute angles
            valid_X, valid_Y, valid_scores = [], [], []
            raw_X, raw_Y, raw_scores = [], [], []
            render_X, render_Y, render_scores = [], [], []
            valid_X_flipped, valid_angles_flipped, valid_angles = [], [], []
            for person_idx in range(len(keypoints)):
                if load_trc_px:
                    person_X_raw = keypoints[person_idx][:, 0]
                    person_Y_raw = keypoints[person_idx][:, 1]
                    person_scores_raw = scores[person_idx]
                    person_X = keypoints[person_idx][:, 0]
                    person_Y = keypoints[person_idx][:, 1]
                    person_scores = scores[person_idx]
                    person_render_X, person_render_Y, person_render_scores = (
                        person_X.copy(),
                        person_Y.copy(),
                        person_scores.copy(),
                    )
                else:
                    # Retrieve keypoints and scores for the person, remove low-confidence keypoints
                    person_X_raw = keypoints[person_idx][:, 0]
                    person_Y_raw = keypoints[person_idx][:, 1]
                    person_scores_raw = scores[person_idx]
                    person_X, person_Y, person_scores, _ = evaluate_pose_frame(
                        person_X_raw,
                        person_Y_raw,
                        person_scores_raw,
                        keypoint_likelihood_threshold,
                        average_likelihood_threshold,
                        keypoint_number_threshold,
                    )
                    person_render_X, person_render_Y, person_render_scores = (
                        person_X,
                        person_Y,
                        person_scores,
                    )
                person_visible_side_frame = (
                    visible_side[person_idx] if len(visible_side) > person_idx else "auto"
                )
                # Restore the upstream visible_side whole-body flip for RTMLib/body_with_feet paths.
                if not use_synthpose:
                    person_visible_side_frame = _resolve_person_visible_side_frame(
                        person_X,
                        person_visible_side_frame,
                        has_toe_heel,
                        L_R_direction_idx=L_R_direction_idx if has_toe_heel else None,
                    )
                    person_X_flipped = _apply_visible_side_whole_body_flip(
                        person_X,
                        person_visible_side_frame,
                        keypoints_names=keypoints_names,
                        keypoints_ids=keypoints_ids,
                    )
                elif flip_left_right:
                    person_X_flipped = flip_left_right_direction(
                        person_X, L_R_direction_idx, keypoints_names, keypoints_ids
                    )
                else:
                    person_X_flipped = person_X.copy()

                # Add derived markers needed by downstream pose/IK consumers.
                new_keypoints_names, new_keypoints_ids = (
                    keypoints_names.copy(),
                    keypoints_ids.copy(),
                )
                for kpt in ["Hip", "Neck", "Head"]:
                    if kpt not in new_keypoints_names:
                        person_X_flipped, person_Y, person_scores, new_keypoints_names = _upsert_derived_pose_keypoint(
                            kpt,
                            person_X_flipped,
                            person_Y,
                            person_scores,
                            new_keypoints_names,
                        )
                        person_X, _, _, _ = _upsert_derived_pose_keypoint(
                            kpt,
                            person_X,
                            person_Y,
                            person_scores,
                            new_keypoints_names,
                        )
                        new_keypoints_ids.append(len(person_X_flipped) - 1)

                # Compute angles
                if calculate_angles:
                    person_angles = []
                    for ang_name in angle_names:
                        ang_params = angle_dict.get(ang_name)
                        kpts = ang_params[0]
                        if not any(item not in new_keypoints_names for item in kpts):
                            ang = compute_angle(
                                ang_name,
                                person_X_flipped,
                                person_Y,
                                angle_dict,
                                new_keypoints_ids,
                                new_keypoints_names,
                            )
                        else:
                            ang = np.nan
                        person_angles.append(ang)

                    if (
                        use_synthpose
                        and person_visible_side_frame == "left"
                        and not flip_left_right
                    ):
                        person_angles_flipped = list(-np.array(person_angles))
                    else:
                        person_angles_flipped = person_angles.copy()

                    valid_angles.append(person_angles)
                    valid_angles_flipped.append(person_angles_flipped)
                    valid_X_flipped.append(person_X_flipped)
                raw_X.append(person_X_raw)
                raw_Y.append(person_Y_raw)
                raw_scores.append(person_scores_raw)
                valid_X.append(person_X)
                valid_Y.append(person_Y)
                valid_scores.append(person_scores)
                render_X.append(person_render_X)
                render_Y.append(person_render_Y)
                render_scores.append(person_render_scores)

            # Keep frame arrays homogeneous when no person is detected in a frame.
            if len(valid_X) == 0:
                kpt_count = len(new_keypoints_names)
                raw_kpt_count = len(raw_keypoint_names)
                valid_X = [np.full(kpt_count, np.nan)]
                valid_Y = [np.full(kpt_count, np.nan)]
                valid_scores = [np.full(kpt_count, np.nan)]
                raw_X = [np.full(raw_kpt_count, np.nan)]
                raw_Y = [np.full(raw_kpt_count, np.nan)]
                raw_scores = [np.full(raw_kpt_count, np.nan)]
                render_X = [np.full(kpt_count, np.nan)]
                render_Y = [np.full(kpt_count, np.nan)]
                render_scores = [np.full(kpt_count, np.nan)]
                if calculate_angles:
                    valid_X_flipped = [np.full(kpt_count, np.nan)]
                    nan_angles = np.full(len(angle_names), np.nan)
                    valid_angles = [nan_angles.copy()]
                    valid_angles_flipped = [nan_angles.copy()]

            # Draw keypoints and skeleton
            if show_realtime_results and realtime_display is not None:
                img = frame.copy()
                cv2.putText(
                    img,
                    f"Press 'q' to stop",
                    (cam_width - int(600 * fontSize), cam_height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    fontSize + 0.2,
                    (255, 255, 255),
                    thickness + 1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    img,
                    f"Press 'q' to stop",
                    (cam_width - int(600 * fontSize), cam_height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    fontSize + 0.2,
                    (0, 0, 255),
                    thickness,
                    cv2.LINE_AA,
                )
                if sam3_realtime_overlay_enabled:
                    sam3_overlay_meta = (
                        detection_meta.get("sam3_ball_meta") or detection_meta
                    )
                    img = draw_sam3_mask_overlay(
                        img,
                        sam3_overlay_meta,
                        alpha=sam3_realtime_mask_alpha,
                        ball_color=ball_color,
                    )
                img = draw_bounding_box(
                    img,
                    valid_X,
                    valid_Y,
                    colors=colors,
                    fontSize=fontSize,
                    thickness=thickness,
                )
                # Draw keypoints and skeleton using unified draw_pose function
                if use_synthpose:
                    # For realtime visualization, data is in original SynthPose order (indices 0-51)
                    # Use SYNTHPOSE_KEYPOINT_NAMES for correct name-to-index mapping
                    from Sports2D.Utilities.synthpose_skeleton import (
                        SYNTHPOSE_KEYPOINT_NAMES,
                    )

                    realtime_kpt_names = list(SYNTHPOSE_KEYPOINT_NAMES)
                    # Add derived markers if they were appended to the processed arrays.
                    for kpt in ["Hip", "Neck", "Head"]:
                        if kpt in new_keypoints_names and kpt not in realtime_kpt_names:
                            realtime_kpt_names.append(kpt)
                    kpt_names_for_draw = realtime_kpt_names
                else:
                    kpt_names_for_draw = None
                img = draw_pose(
                    img,
                    render_X,
                    render_Y,
                    render_scores,
                    pose_model_with_output_ids,
                    keypoint_names=kpt_names_for_draw,
                    backend_name=backend_name,
                    thickness=thickness,
                    keypoint_draw_threshold=draw_keypoint_likelihood_threshold,
                    skeleton_draw_threshold=draw_skeleton_likelihood_threshold,
                )
                if calculate_angles:
                    img = draw_angles(
                        img,
                        valid_X,
                        valid_Y,
                        valid_angles_flipped,
                        valid_X_flipped,
                        new_keypoints_ids,
                        new_keypoints_names,
                        angle_names,
                        display_angle_values_on=display_angle_values_on,
                        colors=colors,
                        fontSize=fontSize,
                        thickness=thickness,
                    )
                if ball_overlay_enabled:
                    selected_trail_points = ball_trail_points
                    if ball_multi_id_tracking:
                        selected_trail_points = ball_trail_points_by_id.get(
                            frame_selected_ball_id, []
                        )
                    img = draw_ball_overlay(
                        img,
                        frame_ball_boxes,
                        frame_ball_center,
                        selected_trail_points,
                        color=ball_color,
                        radius=ball_radius,
                        trail_alpha=ball_trail_alpha,
                        tracked_balls=frame_ball_tracks
                        if ball_multi_id_tracking
                        else None,
                        selected_track_id=frame_selected_ball_id
                        if ball_multi_id_tracking
                        else None,
                        show_ids=ball_show_ids and ball_multi_id_tracking,
                    )

                frame_elapsed = max((datetime.now() - start_time).total_seconds(), 1e-6)
                detected_persons = int(
                    sum(
                        0 if np.isnan(np.asarray(person)).all() else 1
                        for person in valid_X
                    )
                )
                if ball_overlay_enabled:
                    if ball_multi_id_tracking:
                        detected_balls = int(
                            sum(1 for t in frame_ball_tracks if t.get("visible", False))
                        )
                    else:
                        detected_balls = int(len(frame_ball_boxes))
                else:
                    detected_balls = 0

                realtime_display.render(
                    img,
                    stats={
                        "state": "Paused" if display_paused else "Live",
                        "ui_fps": 1.0 / frame_elapsed,
                        "inference_ms": frame_elapsed * 1000.0,
                        "detected_persons": detected_persons,
                        "detected_balls": detected_balls,
                        "dropped_frames": dropped_frames,
                        "model": pose_model_name,
                        "backend": display_runtime_backend,
                        "webcam_id": webcam_id if video_file == "webcam" else None,
                        "elapsed_seconds": time.perf_counter() - session_start_perf,
                        "camera_resolution": f"{cam_width}x{cam_height}",
                        "save_status": "Saving video"
                        if (save_vid or video_file == "webcam")
                        else "Not saving",
                    },
                )
                display_events = realtime_display.poll_events(delay_ms=1)
                if display_events.get("toggle_pause"):
                    display_paused = not display_paused
                    realtime_display.set_session_state(
                        "Paused" if display_paused else "Live"
                    )
                if display_events.get("resume"):
                    display_paused = False
                    realtime_display.set_session_state("Live")
                if display_events.get("stop"):
                    stop_requested = True
                    break

                # # Debugging
                # img_output_path = img_output_dir / f'{video_file_stem}_frame{frame_nb:06d}.png'
                # cv2.imwrite(str(img_output_path), img)

            all_frames_X.append(np.array(valid_X))
            all_frames_X_flipped.append(np.array(valid_X_flipped))
            all_frames_Y.append(np.array(valid_Y))
            all_frames_scores.append(np.array(valid_scores))
            all_frames_X_raw.append(np.array(raw_X))
            all_frames_Y_raw.append(np.array(raw_Y))
            all_frames_scores_raw.append(np.array(raw_scores))
            all_frames_ball_centers.append(frame_ball_center)
            all_frames_ball_boxes.append(frame_ball_boxes.copy())
            all_frames_ball_scores.append(frame_ball_scores.copy())
            frame_ball_tracks_snapshot = []
            for track in frame_ball_tracks:
                frame_ball_tracks_snapshot.append(
                    {
                        "id": int(track.get("id")),
                        "center": tuple(track.get("center"))
                        if track.get("center") is not None
                        else None,
                        "box": (
                            np.asarray(track.get("box"), dtype=np.float32).copy()
                            if track.get("box") is not None
                            else None
                        ),
                        "score": (
                            float(track.get("score"))
                            if track.get("score") is not None
                            and np.isfinite(float(track.get("score")))
                            else float("nan")
                        ),
                        "visible": bool(track.get("visible", False)),
                        "missing": int(track.get("missing", 0)),
                    }
                )
            all_frames_ball_tracks.append(frame_ball_tracks_snapshot)
            all_frames_selected_ball_ids.append(frame_selected_ball_id)
            all_frames_sam3_ball_mask_available.append(frame_sam3_ball_mask_available)

            if save_angles and calculate_angles:
                all_frames_angles.append(np.array(valid_angles))
            if (
                video_file == "webcam" and save_vid
            ):  # To adjust framerate of output video
                elapsed_time = (datetime.now() - start_time).total_seconds()
                frame_processing_times.append(elapsed_time)
            if stop_requested:
                break

        # End of the video is reached
        cap.release()
        logging.info(f"Video processing completed.")
        if save_vid or video_file == "webcam":
            out_vid.release()
            if video_file == "webcam":
                vid_output_path.absolute().rename(video_file_path)

        if show_realtime_results and realtime_display is not None:
            realtime_display.close()

    # %% ==================================================
    # Post-processing: Select persons, Interpolate, filter, and save pose and angles
    # ====================================================
    all_frames_X_raw_homog = make_homogeneous(all_frames_X_raw)
    all_frames_Y_raw_homog = make_homogeneous(all_frames_Y_raw)
    all_frames_scores_raw_homog = make_homogeneous(all_frames_scores_raw)
    all_frames_X_homog = make_homogeneous(all_frames_X)
    all_frames_X_homog = all_frames_X_homog[..., new_keypoints_ids]
    if calculate_angles or save_angles:
        all_frames_X_flipped_homog = make_homogeneous(all_frames_X_flipped)
        all_frames_X_flipped_homog = all_frames_X_flipped_homog[..., new_keypoints_ids]
        all_frames_angles_homog = make_homogeneous(all_frames_angles)
    else:
        all_frames_X_flipped_homog = all_frames_X_flipped
        all_frames_angles_homog = all_frames_angles
    all_frames_Y_homog = make_homogeneous(all_frames_Y)
    all_frames_Y_homog = all_frames_Y_homog[..., new_keypoints_ids]
    all_frames_Z_homog = pd.DataFrame(
        np.zeros_like(all_frames_X_homog)[:, 0, :], columns=new_keypoints_names
    )
    all_frames_scores_homog = make_homogeneous(all_frames_scores)
    all_frames_scores_homog = all_frames_scores_homog[..., new_keypoints_ids]

    frame_range = [0, frame_count] if video_file == "webcam" else frame_range
    sample_count = max(0, frame_count - frame_range[0])
    if load_trc_px:
        all_frames_time = pd.Series(
            np.asarray(
                time_col.iloc[frame_range[0] : frame_range[0] + sample_count],
                dtype=float,
            ),
            name="time",
        ).reset_index(drop=True)
    else:
        frame_indices = frame_range[0] + np.arange(sample_count, dtype=float)
        all_frames_time = pd.Series(frame_indices / fps, name="time")
    if load_trc_px:
        selected_persons = [0]
    else:
        # Select persons
        nb_detected_persons = all_frames_scores_homog.shape[1]
        if nb_persons_to_detect == "all":
            nb_persons_to_detect = all_frames_scores_homog.shape[1]
        if nb_detected_persons < nb_persons_to_detect:
            logging.warning(
                f"Less than the {nb_persons_to_detect} required persons were detected. Analyzing all {nb_detected_persons} persons."
            )
            nb_persons_to_detect = nb_detected_persons

        if person_ordering_method == "on_click":
            selected_persons = get_personIDs_on_click(
                video_file_path, frame_range, all_frames_X_homog, all_frames_Y_homog
            )
            if len(selected_persons) == 0:
                logging.warning("No persons selected. Analyzing all detected persons.")
                selected_persons = list(range(nb_detected_persons))
            if len(selected_persons) != nb_persons_to_detect:
                logging.warning(
                    f'You selected more (or less) than the required {nb_persons_to_detect} persons. "nb_persons_to_detect" will be set to {len(selected_persons)}.'
                )
                nb_persons_to_detect = len(selected_persons)
        elif person_ordering_method == "highest_likelihood":
            selected_persons = get_personIDs_with_highest_scores(
                all_frames_scores_homog, nb_persons_to_detect
            )
        elif person_ordering_method == "first_detected":
            selected_persons = get_personIDs_in_detection_order(nb_persons_to_detect)
        elif person_ordering_method == "last_detected":
            selected_persons = get_personIDs_in_detection_order(
                nb_persons_to_detect, reverse=True
            )
        elif person_ordering_method == "largest_size":
            selected_persons = get_personIDs_with_largest_size(
                all_frames_X_homog,
                all_frames_Y_homog,
                nb_persons_to_detect=nb_persons_to_detect,
                vertical=False,
            )
        elif person_ordering_method == "smallest_size":
            selected_persons = get_personIDs_with_largest_size(
                all_frames_X_homog,
                all_frames_Y_homog,
                nb_persons_to_detect=nb_persons_to_detect,
                vertical=False,
                reverse=True,
            )
        elif person_ordering_method == "greatest_displacement":
            selected_persons = get_personIDs_with_greatest_displacement(
                all_frames_X_homog,
                all_frames_Y_homog,
                nb_persons_to_detect=nb_persons_to_detect,
                horizontal=True,
            )
        elif person_ordering_method == "least_displacement":
            selected_persons = get_personIDs_with_greatest_displacement(
                all_frames_X_homog,
                all_frames_Y_homog,
                nb_persons_to_detect=nb_persons_to_detect,
                horizontal=True,
                reverse=True,
            )
        elif person_ordering_method == "medicine_ball":
            selected_persons, medicine_ball_stats = resolve_personIDs_for_medicine_ball(
                all_frames_X_homog,
                all_frames_Y_homog,
                all_frames_scores_homog,
                all_frames_ball_centers,
                nb_persons_to_detect=nb_persons_to_detect,
                detect_ball=detect_ball,
                ball_ordering_method=ball_ordering_method,
            )
            if medicine_ball_stats.get("used_fallback", False):
                fallback_reason = medicine_ball_stats.get(
                    "fallback_reason", "unknown reason"
                )
                logging.warning(
                    "person_ordering_method='medicine_ball' requires an automatic selected-ball timeline. "
                    "Falling back to 'highest_likelihood' because %s.",
                    fallback_reason,
                )
            else:
                eligible_person_ids = list(
                    medicine_ball_stats.get("eligible_person_ids", [])
                )
                if len(selected_persons) < nb_persons_to_detect:
                    logging.warning(
                        "person_ordering_method='medicine_ball' kept %s person(s) after the 95%% presence gate "
                        "(requested %s). Eligible slots: %s.",
                        len(selected_persons),
                        nb_persons_to_detect,
                        eligible_person_ids,
                    )
        elif person_ordering_method == "motion_specific":
            selected_persons, motion_person_stats = (
                resolve_personIDs_for_motion_specific(
                    all_frames_X_homog,
                    all_frames_Y_homog,
                    all_frames_scores_homog,
                    new_keypoints_names,
                    nb_persons_to_detect=nb_persons_to_detect,
                    target=motion_person_selection_target,
                    presence_threshold=motion_person_presence_threshold,
                    confidence_threshold=motion_person_confidence_threshold,
                    size_min_ratio=motion_person_size_min_ratio,
                    motion_score_threshold=motion_person_score_threshold,
                    fps=fps,
                )
            )
            if motion_person_stats.get("used_fallback", False):
                fallback_reason = motion_person_stats.get(
                    "fallback_reason", "unknown reason"
                )
                logging.warning(
                    "person_ordering_method='motion_specific' could not find "
                    "a gate-filtered %s candidate. Falling back to "
                    "'highest_likelihood' because %s.",
                    motion_person_selection_target,
                    fallback_reason,
                )
            elif len(selected_persons) < nb_persons_to_detect:
                logging.warning(
                    "person_ordering_method='motion_specific' kept %s person(s) "
                    "after gate filters and motion scoring (requested %s). "
                    "Eligible slots: %s.",
                    len(selected_persons),
                    nb_persons_to_detect,
                    list(motion_person_stats.get("eligible_person_ids", [])),
                )
        else:
            raise ValueError(
                f"Invalid person_ordering_method: {person_ordering_method}. Must be "
                "'on_click', 'highest_likelihood', 'largest_size', 'smallest_size', "
                "'greatest_displacement', 'least_displacement', 'first_detected', "
                "'last_detected', 'medicine_ball', or 'motion_specific'."
            )
        logging.info(
            f"Reordered persons: IDs of persons {selected_persons} become {list(range(len(selected_persons)))}."
        )

    if (
        ball_multi_id_tracking
        and ball_selection_mode == "auto"
        and ball_ordering_method == "on_click"
    ):
        selected_ball_ids = get_ball_trackIDs_on_click(
            video_file_path,
            frame_range,
            all_frames_ball_tracks,
        )
        if len(selected_ball_ids) == 0:
            logging.warning(
                "No ball selected in on_click UI. Keeping automatic ball selection results."
            )
        else:
            selected_ball_track_id = int(selected_ball_ids[0])
            logging.info(
                "Selected ball track on_click: ball %s", selected_ball_track_id
            )
            stitched_ids, stitched_centers = stitch_selected_ball_timeline(
                all_frames_ball_tracks,
                selected_ball_track_id,
                max_jump_px=ball_tracking_max_distance
                if ball_tracking_max_distance is not None
                else ball_max_jump_px,
            )
            for frame_idx in range(len(all_frames_ball_tracks)):
                if (
                    frame_idx < len(stitched_ids)
                    and stitched_ids[frame_idx] is not None
                ):
                    all_frames_selected_ball_ids[frame_idx] = stitched_ids[frame_idx]
                if frame_idx < len(stitched_centers):
                    all_frames_ball_centers[frame_idx] = stitched_centers[frame_idx]

    if hybrid_mode and len(selected_persons) > 0:
        if hybrid_review_pose:
            logging.info(
                "Hybrid pose review enabled. Opening manual pose editor for selected persons."
            )
            for selected_person_slot, idx_person in enumerate(selected_persons):
                visible_side_person = (
                    visible_side[selected_person_slot]
                    if len(visible_side) > selected_person_slot
                    else "auto"
                )
                review_X_raw, review_Y_raw, review_scores_raw, review_keypoint_names = (
                    augment_pose_arrays_with_derived_keypoints(
                        all_frames_X_raw_homog[:, idx_person, :],
                        all_frames_Y_raw_homog[:, idx_person, :],
                        all_frames_scores_raw_homog[:, idx_person, :],
                        raw_keypoint_names,
                    )
                )
                corrected_X_review, corrected_Y_review, corrected_scores_review, _ = (
                    review_pose_sequence(
                        video_file_path,
                        frame_range,
                        review_X_raw,
                        review_Y_raw,
                        review_scores_raw,
                        keypoint_names=review_keypoint_names,
                        keypoint_threshold=keypoint_likelihood_threshold,
                        manual_mask=np.zeros_like(review_scores_raw, dtype=bool),
                        window_title=f"Hybrid pose review - person {selected_person_slot}",
                        ui_backend=hybrid_ui_backend,
                    )
                )
                corrected_X_raw = _select_pose_keypoint_columns(
                    corrected_X_review,
                    review_keypoint_names,
                    raw_keypoint_names,
                )
                corrected_Y_raw = _select_pose_keypoint_columns(
                    corrected_Y_review,
                    review_keypoint_names,
                    raw_keypoint_names,
                )
                corrected_scores_raw = _select_pose_keypoint_columns(
                    corrected_scores_review,
                    review_keypoint_names,
                    raw_keypoint_names,
                )
                all_frames_X_raw_homog[:, idx_person, :] = corrected_X_raw
                all_frames_Y_raw_homog[:, idx_person, :] = corrected_Y_raw
                all_frames_scores_raw_homog[:, idx_person, :] = corrected_scores_raw

                (
                    corrected_X,
                    corrected_Y,
                    corrected_scores,
                    corrected_X_flipped,
                    corrected_angles,
                    corrected_keypoint_names,
                ) = _recompute_pose_timelines_from_raw(
                    corrected_X_raw,
                    corrected_Y_raw,
                    corrected_scores_raw,
                    raw_keypoint_names,
                    keypoint_likelihood_threshold,
                    average_likelihood_threshold,
                    keypoint_number_threshold,
                    flip_left_right,
                    L_R_direction_idx,
                    angle_names,
                    calculate_angles,
                    visible_side_person=visible_side_person,
                    use_visible_side_whole_body_flip=not use_synthpose,
                    has_toe_heel=has_toe_heel,
                )
                if corrected_keypoint_names != new_keypoints_names:
                    raise ValueError(
                        f"Hybrid pose review changed the keypoint schema unexpectedly: "
                        f"{corrected_keypoint_names} != {new_keypoints_names}"
                    )
                all_frames_X_homog[:, idx_person, :] = corrected_X
                all_frames_Y_homog[:, idx_person, :] = corrected_Y
                all_frames_scores_homog[:, idx_person, :] = corrected_scores
                if calculate_angles or save_angles:
                    all_frames_X_flipped_homog[:, idx_person, :] = corrected_X_flipped
                    all_frames_angles_homog[:, idx_person, :] = corrected_angles

        if hybrid_review_ball and ball_overlay_enabled:
            logging.info("Hybrid ball review enabled. Opening manual ball editor.")
            (
                corrected_ball_centers,
                corrected_ball_visible,
                manual_ball_override_mask,
            ) = review_ball_sequence(
                video_file_path,
                frame_range,
                all_frames_ball_centers,
                all_frames_ball_boxes,
                all_frames_ball_scores,
                all_frames_ball_tracks,
                all_frames_selected_ball_ids,
                score_threshold=ball_detection_threshold,
                window_title="Hybrid ball review",
                ui_backend=hybrid_ui_backend,
            )
            for frame_idx, center in enumerate(corrected_ball_centers):
                if not manual_ball_override_mask[frame_idx]:
                    continue
                all_frames_ball_centers[frame_idx] = (
                    center if corrected_ball_visible[frame_idx] else None
                )
                if ball_multi_id_tracking:
                    all_frames_ball_tracks[frame_idx] = apply_ball_override_to_tracks(
                        all_frames_ball_tracks[frame_idx],
                        all_frames_selected_ball_ids[frame_idx]
                        if frame_idx < len(all_frames_selected_ball_ids)
                        else None,
                        all_frames_ball_centers[frame_idx],
                        corrected_ball_visible[frame_idx],
                    )

    ball_pose_export_enabled = bool(save_pose and ball_overlay_enabled)
    ball_export_series = []
    ball_trc_px = pd.DataFrame()
    public_meter_trc_data_by_name = {}
    trc_data, trc_data_unfiltered, score_data = [], [], []
    first_run_starts_everyone, last_run_ends_everyone = [], []
    vertical_jump_results = []
    if ball_pose_export_enabled:
        ball_export_series = build_ball_export_series(
            all_frames_time,
            all_frames_ball_centers,
            all_frames_ball_boxes,
            all_frames_ball_scores,
            all_frames_ball_tracks,
            all_frames_selected_ball_ids,
            all_frames_sam3_ball_mask_available,
            frame_offset=frame_range[0],
            multi_id_tracking=ball_multi_id_tracking,
            max_recovery_dist=ball_tracking_max_distance
            if ball_tracking_max_distance is not None
            else ball_max_jump_px,
        )
        write_ball_pose_json(ball_export_series, pose_ball_output_dir, output_dir_name)
        ball_blender_helper_path = write_ball_blender_helper(
            output_dir,
            output_dir_name,
            marker_name="ball",
        )
        ball_trc_px = build_ball_trc_data(
            ball_export_series, index=all_frames_time.index, marker_name="ball"
        )
        logging.info(f"Ball pose JSON saved to {pose_ball_output_dir.resolve()}.")
        logging.info(
            f"Blender ball helper saved to {ball_blender_helper_path.resolve()}."
        )

    # %% ==================================================
    # Post-processing pose
    # ====================================================
    (
        all_frames_X_processed,
        all_frames_X_flipped_processed,
        all_frames_Y_processed,
        all_frames_scores_processed,
        all_frames_angles_processed,
    ) = (
        all_frames_X_homog.copy(),
        all_frames_X_flipped_homog.copy(),
        all_frames_Y_homog.copy(),
        all_frames_scores_homog.copy(),
        all_frames_angles_homog.copy(),
    )
    new_visible_side = visible_side.copy()
    if need_postprocess_pose:
        logging.info("\nPost-processing pose:")
        # Process pose for each person
        for i, idx_person in enumerate(selected_persons):
            pose_path_person = pose_output_path.parent / (
                pose_output_path.stem + f"_person{i:02d}.trc"
            )
            all_frames_X_person = pd.DataFrame(
                all_frames_X_processed[:, idx_person, :], columns=new_keypoints_names
            )
            all_frames_Y_person = pd.DataFrame(
                all_frames_Y_processed[:, idx_person, :], columns=new_keypoints_names
            )
            score_data.append(
                pd.DataFrame(
                    all_frames_scores_processed[:, idx_person, :],
                    columns=new_keypoints_names,
                )
            )
            if calculate_angles or save_angles:
                all_frames_X_flipped_person = pd.DataFrame(
                    all_frames_X_flipped_processed[:, idx_person, :],
                    columns=new_keypoints_names,
                )

            # Interpolate
            if not interpolate:
                logging.info(f"- Person {i}: No interpolation.")
                all_frames_X_person_interp = all_frames_X_person
                all_frames_Y_person_interp = all_frames_Y_person
            else:
                logging.info(
                    f"- Person {i}: Interpolating missing sequences if they are smaller than {interp_gap_smaller_than} frames. Large gaps filled with {fill_large_gaps_with}."
                )
                all_frames_X_person_interp = all_frames_X_person.apply(
                    interpolate_zeros_nans,
                    axis=0,
                    args=[interp_gap_smaller_than, "linear"],
                )
                all_frames_Y_person_interp = all_frames_Y_person.apply(
                    interpolate_zeros_nans,
                    axis=0,
                    args=[interp_gap_smaller_than, "linear"],
                )

            # Find the first and last valid chunks of data
            first_run_starts, last_run_ends = [], []
            for col in all_frames_X_person.columns:
                first_run_start, last_run_end = indices_of_first_last_non_nan_chunks(
                    all_frames_X_person_interp[col],
                    min_chunk_size=min_chunk_size,
                    chunk_choice_method=sections_to_keep,
                )
                first_run_starts += [first_run_start]
                last_run_ends += [last_run_end]
            first_run_start_min, last_run_end_max = (
                min(first_run_starts),
                max(last_run_ends),
            )
            first_run_starts_everyone += [first_run_starts]
            last_run_ends_everyone += [last_run_ends]

            # Do not process person if no section of min_chunk_size valid frames in a row
            if (first_run_start_min, last_run_end_max) == (0, 0):
                (
                    all_frames_X_processed[:, idx_person, :],
                    all_frames_X_flipped_processed[:, idx_person, :],
                    all_frames_Y_processed[:, idx_person, :],
                ) = np.nan, np.nan, np.nan
                columns = np.array(
                    [[c] * 3 for c in all_frames_X_person.columns]
                ).flatten()
                trc_data_i = pd.DataFrame(
                    0, index=all_frames_X_person.index, columns=["time"] + list(columns)
                )
                trc_data_i["time"] = all_frames_time
                trc_data.append(trc_data_i)
                trc_data_unfiltered_i = trc_data_i.copy()
                trc_data_unfiltered.append(trc_data_unfiltered_i)
                logging.info(
                    f"  Person {i}: Less than {min_chunk_size} valid frames in a row. Deleting person."
                )
                continue

            # Fill remaining gaps
            if fill_large_gaps_with.lower() == "last_value":
                for col_id, col in enumerate(all_frames_X_person_interp.columns):
                    first_run_start, last_run_end = (
                        first_run_starts[col_id],
                        last_run_ends[col_id],
                    )
                    for coord_df in [
                        all_frames_X_person_interp,
                        all_frames_Y_person_interp,
                        all_frames_Z_homog,
                    ]:
                        coord_df.loc[:first_run_start, col] = np.nan
                        coord_df.loc[last_run_end:, col] = np.nan
                        coord_df.loc[first_run_start:last_run_end, col] = (
                            coord_df.loc[first_run_start:last_run_end, col]
                            .ffill()
                            .bfill()
                        )
            elif fill_large_gaps_with.lower() == "zeros":
                all_frames_X_person_interp.replace(np.nan, 0, inplace=True)
                all_frames_Y_person_interp.replace(np.nan, 0, inplace=True)

            # if handle_LR_swap:
            #     logging.info(f'Handling left-right swaps.')
            #     all_frames_X_person_interp = all_frames_X_person_interp.apply(LR_unswap, axis=0)
            #     all_frames_Y_person_interp = all_frames_Y_person_interp.apply(LR_unswap, axis=0)

            if reject_outliers:
                logging.info("Rejecting outliers with a Hampel filter.")
                all_frames_X_person_interp = all_frames_X_person_interp.apply(
                    hampel_filter, axis=0, args=[round(7 * frame_rate / 30), 2]
                )
                all_frames_Y_person_interp = all_frames_Y_person_interp.apply(
                    hampel_filter, axis=0, args=[round(7 * frame_rate / 30), 2]
                )

            if not do_filter:
                logging.info(f"No filtering.")
                all_frames_X_person_filt = all_frames_X_person_interp
                all_frames_Y_person_filt = all_frames_Y_person_interp
            else:
                if filter_type == ("butterworth" or "butterworth_on_speed"):
                    cutoff = butterworth_filter_cutoff
                    if video_file == "webcam":
                        if cutoff / (fps / 2) >= 1:
                            cutoff_old = cutoff
                            cutoff = fps / (2 + 0.001)
                            args = f"\n{cutoff_old:.1f} Hz cut-off framerate too large for a real-time framerate of {fps:.1f} Hz. Using a cut-off framerate of {cutoff:.1f} Hz instead."
                            butterworth_filter_cutoff = cutoff
                    filt_type = (
                        "Butterworth"
                        if filter_type == "butterworth"
                        else "Butterworth on speed"
                    )
                    args = f"{filt_type} filter, {butterworth_filter_order}th order, {butterworth_filter_cutoff} Hz."
                    frame_rate = fps
                elif filter_type == "gcv_spline":
                    args = f"GVC Spline filter, which automatically evaluates the best trade-off between smoothness and fidelity to data."
                elif filter_type == "kalman":
                    args = f"Kalman filter, trusting measurement {kalman_filter_trust_ratio} times more than the process matrix."
                elif filter_type == "gaussian":
                    args = f"Gaussian filter, Sigma kernel {gaussian_filter_kernel}."
                elif filter_type == "loess":
                    args = f"LOESS filter, window size of {loess_filter_kernel} frames."
                elif filter_type == "median":
                    args = f"Median filter, kernel of {median_filter_kernel}."
                else:
                    logging.error(
                        f"Invalid filter_type: {filter_type}. Must be 'butterworth', 'gcv_spline', 'kalman', 'gaussian', 'loess', or 'median'."
                    )
                    raise ValueError(
                        f"Invalid filter_type: {filter_type}. Must be 'butterworth', 'gcv_spline', 'kalman', 'gaussian', 'loess', or 'median'."
                    )

                logging.info(f"Filtering with {args}")
                all_frames_X_person_filt = all_frames_X_person_interp.apply(
                    filter1d,
                    axis=0,
                    args=[Pose2Sim_config_dict, filter_type, frame_rate],
                )
                all_frames_Y_person_filt = all_frames_Y_person_interp.apply(
                    filter1d,
                    axis=0,
                    args=[Pose2Sim_config_dict, filter_type, frame_rate],
                )

            # Build TRC file
            trc_data_i = trc_data_from_XYZtime(
                all_frames_X_person_filt,
                all_frames_Y_person_filt,
                all_frames_Z_homog,
                all_frames_time,
            )
            trc_data.append(trc_data_i)
            if save_pose and not load_trc_px:
                trc_data_to_write = append_trc_marker_aliases(
                    trc_data_i,
                    marker_aliases=SYNTHPOSE_MARKER_ALIASES,
                )
                if ball_pose_export_enabled:
                    trc_data_to_write = append_ball_marker_to_trc_data(
                        trc_data_to_write,
                        ball_trc_px,
                        marker_name="ball",
                    )
                make_trc_with_trc_data(
                    trc_data_to_write, str(pose_path_person), fps=fps
                )
                logging.info(f"Pose in pixels saved to {pose_path_person.resolve()}.")

            # Plotting coordinates before and after interpolation and filtering
            columns_to_concat = []
            for kpt in range(len(all_frames_X_person.columns)):
                columns_to_concat.extend(
                    [
                        all_frames_X_person.iloc[:, kpt],
                        all_frames_Y_person.iloc[:, kpt],
                        all_frames_Z_homog.iloc[:, kpt],
                    ]
                )
            trc_data_unfiltered_i = pd.concat(
                [all_frames_time] + columns_to_concat, axis=1
            )
            trc_data_unfiltered.append(trc_data_unfiltered_i)
            if not to_meters and (show_plots or save_plots):
                pw = pose_plots(trc_data_unfiltered_i, trc_data_i, i, show=show_plots)
                if save_plots:
                    if show_plots:
                        for n, f in enumerate(pw.figure_handles):
                            dpi = pw.canvases[n].figure.dpi
                            f.set_size_inches(1280 / dpi, 720 / dpi)
                            title = pw.tabs.tabText(n)
                            plot_path = plots_output_dir / (
                                pose_output_path.stem
                                + f"_person{i:02d}_px_{title.replace(' ', '_').replace('/', '_')}.png"
                            )
                            f.savefig(plot_path, dpi=dpi, bbox_inches="tight")
                    else:  # Tabbed plots not used
                        for title, f in pw:
                            dpi = f.dpi
                            f.set_size_inches(1280 / dpi, 720 / dpi)
                            plot_path = plots_output_dir / (
                                pose_output_path.stem
                                + f"_person{i:02d}_px_{title.replace(' ', '_').replace('/', '_')}.png"
                            )
                            f.savefig(plot_path, dpi=dpi, bbox_inches="tight")
                            plt.close(f)
                    logging.info(f"Pose plots (px) saved in {plots_output_dir}.")

            (
                all_frames_X_processed[:, idx_person, :],
                all_frames_Y_processed[:, idx_person, :],
            ) = all_frames_X_person_filt, all_frames_Y_person_filt
            if calculate_angles or save_angles:
                all_frames_X_flipped_processed[:, idx_person, :] = (
                    all_frames_X_flipped_person
                )

        # %% Convert px to meters
        trc_data_m = []
        opensim_bridge_trc_data_by_name = {}
        if need_meter_pose and len(trc_data) > 0:
            logging.info("\nConverting pose to meters:")
            meter_trc_data = [
                _resolve_meter_conversion_trc_data(pose_model_name, trc_data_i)
                for trc_data_i in trc_data
            ]
            meter_trc_data_unfiltered = [
                _resolve_meter_conversion_trc_data(
                    pose_model_name, trc_data_unfiltered_i
                )
                for trc_data_unfiltered_i in trc_data_unfiltered
            ]
            meter_keypoint_names = list(meter_trc_data[0].columns[1::3])

            # Compute height of the first person in pixels
            height_px = CORRECTION_2D_TO_3D * compute_height(
                meter_trc_data[0].iloc[:, 1:],
                meter_keypoint_names,
                fastest_frames_to_remove_percent=fastest_frames_to_remove_percent,
                close_to_zero_speed=close_to_zero_speed_px,
                large_hip_knee_angles=large_hip_knee_angles,
                trimmed_extrema_percent=trimmed_extrema_percent,
            )

            # Compute distance from camera to compensate for perspective effects
            distance_m = get_distance_from_camera(
                perspective_value=perspective_value,
                perspective_unit=perspective_unit,
                calib_file=calib_file,
                height_px=height_px,
                height_m=first_person_height,
                cam_width=cam_width,
                cam_height=cam_height,
            )

            # Compute floor angle and xy_origin to compensate for camera horizon and person position
            floor_angle_estim, xy_origin_estim, gait_direction = get_floor_params(
                floor_angle=floor_angle,
                xy_origin=xy_origin,
                calib_file=calib_file,
                height_px=height_px,
                height_m=first_person_height,
                fps=fps,
                trc_data=meter_trc_data[0],
                score_data=score_data[0],
                toe_speed_below=1,
                score_threshold=average_likelihood_threshold,
                cam_width=cam_width,
                cam_height=cam_height,
            )
            cx, cy = xy_origin_estim
            direction_person0 = (
                "right"
                if gait_direction > 0.3
                else "left"
                if gait_direction < -0.3
                else "front"
            )

            logging.info(
                f"Converting from pixels to meters using a person height of {first_person_height:.2f} in meters (manual input), and of {height_px:.2f} in pixels (calculated)."
            )

            perspective_messages = {
                "distance_m": f"(obtained from a manual input).",
                "f_px": f"(calculated from a focal length of {perspective_value:.2f} m).",
                "fov_deg": f"(calculated from a field of view of {perspective_value:.2f} deg).",
                "fov_rad": f"(calculated from a field of view of {perspective_value:.2f} rad).",
                "from_calib": f"(calculated from a calibration file: {calib_file}).",
            }
            message = perspective_messages.get(perspective_unit, "")
            logging.info(
                f"Perspective effects corrected using a camera-to-person distance of {distance_m:.2f} m {message}"
            )

            floor_angle_messages = {
                "manual": "manual input.",
                "auto": "gait kinematics.",
                "from_kinematics": "gait kinematics.",
                "from_calib": "a calibration file.",
            }
            if isinstance(floor_angle, (int, float)):
                key = "manual"
            else:
                key = floor_angle
            message = floor_angle_messages.get(key, "")
            logging.info(
                f"Camera horizon: {np.degrees(floor_angle_estim):.2f}°, corrected using {message}"
            )

            def get_correction_message(xy_origin):
                if (
                    all(isinstance(o, (int, float)) for o in xy_origin)
                    and len(xy_origin) == 2
                ):
                    return "manual input."
                elif xy_origin == ["auto"] or xy_origin == ["from_kinematics"]:
                    return "gait kinematics."
                elif xy_origin == ["from_calib"]:
                    return "a calibration file."
                else:
                    return "."

            message = get_correction_message(xy_origin)
            logging.info(
                f"Floor level: {cy:.2f} px (from the top of the image), gait starting at {cx:.2f} px in the {direction_person0} direction for the first person. Corrected using {message}\n"
            )

            # Prepare calibration data
            R90z = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
            R270x = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])

            calib_file_path = output_dir / f"{video_file_stem}_Sports2D_calib.toml"

            # name, size, distortions
            N = [video_file_stem]
            S = [[cam_width, cam_height]]
            D = [[0.0, 0.0, 0.0, 0.0]]

            # Intrinsics
            f = height_px / first_person_height * distance_m
            cu = cam_width / 2
            cv = cam_height / 2
            K = np.array([[[f, 0.0, cu], [0.0, f, cv], [0.0, 0.0, 1.0]]])

            # Extrinsics
            Rfloory = np.array(
                [
                    [np.cos(floor_angle_estim), 0.0, np.sin(floor_angle_estim)],
                    [0.0, 1.0, 0.0],
                    [-np.sin(floor_angle_estim), 0.0, np.cos(floor_angle_estim)],
                ]
            )
            R_world = R90z @ Rfloory @ R270x
            T_world = R90z @ np.array(
                [-(cx - cu) / f * distance_m, -distance_m, (cy - cv) / f * distance_m]
            )

            R_cam, T_cam = world_to_camera_persp(R_world, T_world)
            Tvec_cam = T_cam.reshape(1, 3).tolist()
            Rvec_cam = cv2.Rodrigues(R_cam)[0].reshape(1, 3).tolist()

            # Save calibration file
            if save_calib and not calib_file:
                toml_write(calib_file_path, N, S, D, K, Rvec_cam, Tvec_cam)
                logging.info(f"Calibration saved to {calib_file_path}.")

            motion_floor_angle_overlay = float(floor_angle_estim)
            motion_floor_origin_overlay = (float(cx), float(cy))

            # Coordinates in m
            new_visible_side = []
            for i in range(len(trc_data)):
                meter_trc_data_i = meter_trc_data[i]
                meter_trc_data_unfiltered_i = meter_trc_data_unfiltered[i]
                jump_overlay_result = {
                    "body_weight_n": None,
                    "full_vgrf_n": np.full(
                        (len(all_frames_time),), np.nan, dtype=float
                    ),
                    "time_s": None,
                    "vgrf_n": None,
                    "cop_xyz_m": None,
                    "metrics": None,
                    "trc_name": None,
                }
                if not np.array(meter_trc_data_i.iloc[:, 1:] == 0).all():
                    # Automatically determine visible side
                    visible_side_i = (
                        visible_side[i] if len(visible_side) > i else "auto"
                    )  # set to 'auto' if list too short
                    # Set to 'front' if slope of X values between [-5,5]
                    if visible_side_i == "auto":
                        try:
                            if all(
                                key in meter_trc_data_i
                                for key in ["LBigToe", "RBigToe"]
                            ):
                                _, _, gait_direction = compute_floor_line(
                                    meter_trc_data_i,
                                    score_data[i],
                                    keypoint_names=["LBigToe", "RBigToe"],
                                    score_threshold=keypoint_likelihood_threshold,
                                )  # toe_speed_below=1 bu default
                            else:
                                _, _, gait_direction = compute_floor_line(
                                    meter_trc_data_i,
                                    score_data[i],
                                    keypoint_names=["LAnkle", "RAnkle"],
                                    score_threshold=keypoint_likelihood_threshold,
                                )
                                logging.warning(
                                    f"The RBigToe and LBigToe are missing from your model. Gait direction will be determined from the ankle points."
                                )
                            visible_side_i = (
                                "right"
                                if gait_direction > 0.3
                                else "left"
                                if gait_direction < -0.3
                                else "front"
                            )
                            logging.info(
                                f"- Person {i}: Seen from the {visible_side_i}."
                            )
                        except:
                            visible_side_i = "none"
                            logging.warning(
                                f'- Person {i}: Could not automatically find gait direction. Please set visible_side to "front", "back", "left", or "right" for this person. Setting to "none".'
                            )
                    # skip if none
                    elif visible_side_i == "none":
                        logging.info(
                            f'- Person {i}: Keeping output in 2D because "visible_side" is set to "none" for person {i}.'
                        )
                    else:
                        logging.info(f"- Person {i}: Seen from the {visible_side_i}.")

                    # Convert to meters
                    px_to_m_i = [
                        convert_px_to_meters(
                            meter_trc_data_i[kpt_name],
                            first_person_height,
                            height_px,
                            distance_m,
                            cam_width,
                            cam_height,
                            cx,
                            cy,
                            -floor_angle_estim,
                            visible_side=visible_side_i,
                        )
                        for kpt_name in meter_keypoint_names
                    ]
                    trc_data_m_i = pd.concat(
                        [all_frames_time.rename("time")] + px_to_m_i, axis=1
                    )
                    first_run_starts_meter = _select_pose_keypoint_columns(
                        np.asarray(first_run_starts_everyone[i], dtype=int),
                        new_keypoints_names,
                        meter_keypoint_names,
                    )
                    last_run_ends_meter = _select_pose_keypoint_columns(
                        np.asarray(last_run_ends_everyone[i], dtype=int),
                        new_keypoints_names,
                        meter_keypoint_names,
                    )
                    for c_id, c in enumerate(
                        3 * np.arange(len(trc_data_m_i.columns[3::3])) + 1
                    ):  # only X coordinates
                        first_run_start, last_run_end = (
                            int(first_run_starts_meter[c_id]),
                            int(last_run_ends_meter[c_id]),
                        )
                        trc_data_m_i.iloc[:first_run_start, c + 2] = np.nan
                        trc_data_m_i.iloc[last_run_end:, c + 2] = np.nan
                        trc_data_m_i.iloc[first_run_start:last_run_end, c + 2] = (
                            trc_data_m_i.iloc[first_run_start:last_run_end, c + 2]
                            .ffill()
                            .bfill()
                        )
                    trc_data_m_public_i = trc_data_m_i.copy()
                    trc_m_first_trim = trc_data_m_i.isnull().any(axis=1).idxmin()
                    trc_m_last_trim = trc_data_m_i[::-1].isnull().any(axis=1).idxmin()
                    trc_data_m_i = trc_data_m_i.iloc[
                        trc_m_first_trim : trc_m_last_trim + 1, :
                    ]
                    px_to_m_unfiltered_i = [
                        convert_px_to_meters(
                            meter_trc_data_unfiltered_i[kpt_name],
                            first_person_height,
                            height_px,
                            distance_m,
                            cam_width,
                            cam_height,
                            cx,
                            cy,
                            -floor_angle_estim,
                            visible_side=visible_side_i,
                        )
                        for kpt_name in meter_keypoint_names
                    ]
                    trc_data_unfiltered_m_i = pd.concat(
                        [all_frames_time.rename("time")] + px_to_m_unfiltered_i, axis=1
                    )

                    if to_meters and (show_plots or save_plots):
                        pw = pose_plots(
                            trc_data_unfiltered_m_i, trc_data_m_i, i, show=show_plots
                        )
                        if save_plots:
                            if show_plots:
                                for n, f in enumerate(pw.figure_handles):
                                    dpi = pw.canvases[n].figure.dpi
                                    f.set_size_inches(1280 / dpi, 720 / dpi)
                                    title = pw.tabs.tabText(n)
                                    plot_path = plots_output_dir / (
                                        pose_output_path.stem
                                        + f"_person{i:02d}_m_{title.replace(' ', '_').replace('/', '_')}.png"
                                    )
                                    f.savefig(plot_path, dpi=dpi, bbox_inches="tight")
                            else:  # Tabbed plots not used
                                for title, f in pw:
                                    dpi = f.dpi
                                    f.set_size_inches(1280 / dpi, 720 / dpi)
                                    plot_path = plots_output_dir / (
                                        pose_output_path.stem
                                        + f"_person{i:02d}_m_{title.replace(' ', '_').replace('/', '_')}.png"
                                    )
                                    f.savefig(plot_path, dpi=dpi, bbox_inches="tight")
                                    plt.close(f)
                            logging.info(f"Pose plots (m) saved in {plots_output_dir}.")

                    # Rebase trimmed meter exports so saved TRCs always start at frame/time 0.
                    trc_data_m_export_i = reset_trc_frame_time_origin(trc_data_m_i)
                    trc_data_m_file_i = append_trc_marker_aliases(
                        trc_data_m_export_i,
                        marker_aliases=SYNTHPOSE_MARKER_ALIASES,
                    )
                    public_trc_data_m_file_i = build_public_meter_trc_data(
                        trc_data_m_public_i,
                        marker_aliases=SYNTHPOSE_MARKER_ALIASES,
                    )

                    if vertical_jump_enabled:
                        mass_i = (
                            participant_masses[i]
                            if len(participant_masses) > i
                            else DEFAULT_MASS
                        )
                        if len(participant_masses) <= i:
                            logging.warning(
                                f"No mass provided for vertical jump. Using {DEFAULT_MASS} kg as default."
                            )
                        try:
                            jump_result = analyze_vertical_jump_trial(
                                trc_data_m_export_i,
                                mass_kg=mass_i,
                                fps=fps,
                            )
                        except ValueError as exc:
                            logging.warning(
                                "Skipping vertical jump export for person %s: %s",
                                i,
                                exc,
                            )
                        else:
                            cop_xyz_m = None
                            if inverse_dynamics_enabled:
                                try:
                                    cop_xyz_m = _resolve_inverse_dynamics_cop_series(
                                        trc_data_m_export_i,
                                        inverse_dynamics_enabled,
                                    )
                                except ValueError as exc:
                                    logging.warning(
                                        "Skipping inverse dynamics CoP proxy for person %s: %s",
                                        i,
                                        exc,
                                    )
                            full_vgrf_n = np.full(
                                (len(all_frames_time),), np.nan, dtype=float
                            )
                            full_vgrf_n[
                                int(trc_m_first_trim) : int(trc_m_last_trim) + 1
                            ] = jump_result["vgrf_n"]
                            jump_overlay_result = {
                                "body_weight_n": float(jump_result["body_weight_n"]),
                                "full_vgrf_n": full_vgrf_n,
                                "time_s": np.asarray(
                                    jump_result["time_s"], dtype=float
                                ).copy(),
                                "vgrf_n": np.asarray(
                                    jump_result["vgrf_n"], dtype=float
                                ).copy(),
                                "cop_xyz_m": _serialize_inverse_dynamics_cop_series(cop_xyz_m),
                                "metrics": copy.deepcopy(jump_result["metrics"]),
                                "trc_name": None,
                            }
                            grf_stem = (
                                "GRF"
                                if len(selected_persons) == 1
                                else f"GRF_person{i:02d}"
                            )
                            grf_trc_path = output_dir / f"{grf_stem}.trc"
                            grf_metrics_path = output_dir / f"{grf_stem}_metrics.json"
                            write_grf_trc(
                                jump_result["time_s"],
                                jump_result["vgrf_n"],
                                grf_trc_path,
                                fps=fps,
                            )
                            write_grf_metrics_json(
                                jump_result["metrics"], grf_metrics_path
                            )
                            logging.info(
                                "Vertical GRF saved to %s and %s.",
                                grf_trc_path.resolve(),
                                grf_metrics_path.resolve(),
                            )

                    # Write to trc file
                    trc_data_m.append(trc_data_m_export_i)
                    pose_path_person_m_i = pose_output_path.parent / (
                        pose_output_path_m.stem + f"_person{i:02d}.trc"
                    )
                    if vertical_jump_enabled:
                        jump_overlay_result["trc_name"] = pose_path_person_m_i.name
                    if ball_pose_export_enabled:
                        ball_trc_m_i = convert_px_to_meters(
                            ball_trc_px.reindex(trc_data_m_public_i.index),
                            first_person_height,
                            height_px,
                            distance_m,
                            cam_width,
                            cam_height,
                            cx,
                            cy,
                            -floor_angle_estim,
                            visible_side=visible_side_i,
                        )
                        public_trc_data_m_file_i = build_public_meter_trc_data(
                            trc_data_m_public_i,
                            marker_aliases=SYNTHPOSE_MARKER_ALIASES,
                            ball_trc_data=ball_trc_m_i,
                            marker_name="ball",
                        )
                    if write_meter_pose:
                        public_meter_trc_data_by_name[pose_path_person_m_i.name] = (
                            public_trc_data_m_file_i
                        )
                    if do_ik or use_augmentation:
                        opensim_bridge_trc_data_by_name[
                            pose_path_person_m_i.name
                        ] = _resolve_opensim_bridge_trc_data(
                            pose_model_name,
                            trc_data_m_file_i,
                        )
                    if write_meter_pose:
                        make_trc_with_trc_data(
                            trc_data_m_file_i, pose_path_person_m_i, fps=fps
                        )
                    if write_meter_pose and make_c3d:
                        c3d_path = convert_to_c3d(str(pose_path_person_m_i))
                    if write_meter_pose:
                        logging.info(
                            f"Pose in meters saved to {pose_path_person_m_i.resolve()}. {'Also saved in c3d format.' if make_c3d else ''}"
                        )
                else:
                    visible_side_i = "none"
                new_visible_side += [visible_side_i]
                if vertical_jump_enabled:
                    vertical_jump_results.append(jump_overlay_result)
        else:
            new_visible_side = visible_side.copy()

    # %% ==================================================
    # Post-processing angles
    # ====================================================
    if save_angles and calculate_angles:
        logging.info("\nPost-processing angles (without inverse kinematics):")
        logging.info(f"Angle output mode: {angle_output_mode}.")
        logging.info(f"Unwrap angles: {unwrap_angles}.")

        # unwrap angles
        # all_frames_angles_homog = np.unwrap(all_frames_angles_homog, axis=0, period=180) # This give all nan values -> need to mask nans
        if unwrap_angles:
            for i in range(all_frames_angles_homog.shape[1]):  # for each person
                for j in range(all_frames_angles_homog.shape[2]):  # for each angle
                    valid_mask = ~np.isnan(all_frames_angles_homog[:, i, j])
                    ang = np.unwrap(
                        all_frames_angles_homog[valid_mask, i, j], period=180
                    )
                    ang = ang - 360 if ang.mean() > 180 else ang
                    ang = ang + 360 if ang.mean() < -180 else ang
                    all_frames_angles_homog[valid_mask, i, j] = ang

        # Process angles for each person
        for i, idx_person in enumerate(selected_persons):
            angles_path_person = angles_output_path.parent / (
                angles_output_path.stem + f"_person{i:02d}.mot"
            )
            all_frames_angles_person = pd.DataFrame(
                all_frames_angles_homog[:, idx_person, :], columns=angle_names
            )

            # Keep the legacy left-side sign correction only for SynthPose paths.
            if use_synthpose and new_visible_side[i] == "left" and not flip_left_right:
                all_frames_angles_homog[:, idx_person, :] = -all_frames_angles_homog[
                    :, idx_person, :
                ]

            if not interpolate:
                logging.info(f"- Person {i}: No interpolation.")
                all_frames_angles_person_interp = all_frames_angles_person
            else:
                logging.info(
                    f"- Person {i}: Interpolating missing sequences if they are smaller than {interp_gap_smaller_than} frames. Large gaps filled with {fill_large_gaps_with}."
                )
                all_frames_angles_person_interp = all_frames_angles_person.apply(
                    interpolate_zeros_nans,
                    axis=0,
                    args=[interp_gap_smaller_than, "linear"],
                )

            # Find the first and last valid chunks of data
            first_run_starts, last_run_ends = [], []
            for col in all_frames_angles_person.columns:
                first_run_start, last_run_end = indices_of_first_last_non_nan_chunks(
                    all_frames_angles_person_interp[col],
                    min_chunk_size=min_chunk_size,
                    chunk_choice_method=sections_to_keep,
                )
                first_run_starts += [first_run_start]
                last_run_ends += [last_run_end]
            first_run_start_min, last_run_end_max = (
                min(first_run_starts),
                max(last_run_ends),
            )

            # Do not process person if no section of min_chunk_size valid frames in a row
            if (first_run_start_min, last_run_end_max) == (0, 0):
                all_frames_angles_processed[:, idx_person, :] = np.nan
                logging.info(
                    f"  Person {i}: Less than {min_chunk_size} valid frames in a row. Deleting person."
                )
                continue

            # Fill remaining gaps
            if fill_large_gaps_with == "last_value":
                for col_id, col in enumerate(all_frames_angles_person_interp.columns):
                    first_run_start, last_run_end = (
                        first_run_starts[col_id],
                        last_run_ends[col_id],
                    )
                    all_frames_angles_person_interp.loc[:first_run_start, col] = np.nan
                    all_frames_angles_person_interp.loc[last_run_end:, col] = np.nan
                    all_frames_angles_person_interp.loc[
                        first_run_start:last_run_end, col
                    ] = (
                        all_frames_angles_person_interp.loc[
                            first_run_start:last_run_end, col
                        ]
                        .ffill()
                        .bfill()
                    )
            elif fill_large_gaps_with == "zeros":
                all_frames_angles_person_interp.replace(np.nan, 0, inplace=True)

            # Filter
            if reject_outliers:
                logging.info(f"Rejecting outliers with a Hampel filter.")
                all_frames_angles_person_interp = all_frames_angles_person_interp.apply(
                    hampel_filter, axis=0
                )

            if not do_filter:
                logging.info(f"No filtering.")
                all_frames_angles_person_filt = all_frames_angles_person_interp
            else:
                if filter_type == ("butterworth" or "butterworth_on_speed"):
                    cutoff = butterworth_filter_cutoff
                    if video_file == "webcam":
                        if cutoff / (fps / 2) >= 1:
                            cutoff_old = cutoff
                            cutoff = fps / (2 + 0.001)
                            args = f"\n{cutoff_old:.1f} Hz cut-off framerate too large for a real-time framerate of {fps:.1f} Hz. Using a cut-off framerate of {cutoff:.1f} Hz instead."
                            butterworth_filter_cutoff = cutoff
                    filt_type = (
                        "Butterworth"
                        if filter_type == "butterworth"
                        else "Butterworth on speed"
                    )
                    args = f"{filt_type} filter, {butterworth_filter_order}th order, {butterworth_filter_cutoff} Hz."
                    frame_rate = fps
                elif filter_type == "gcv_spline":
                    args = f"GVC Spline filter, which automatically evaluates the best trade-off between smoothness and fidelity to data."
                elif filter_type == "kalman":
                    args = f"Kalman filter, trusting measurement {kalman_filter_trust_ratio} times more than the process matrix."
                elif filter_type == "gaussian":
                    args = f"Gaussian filter, Sigma kernel {gaussian_filter_kernel}."
                elif filter_type == "loess":
                    args = f"LOESS filter, window size of {loess_filter_kernel} frames."
                elif filter_type == "median":
                    args = f"Median filter, kernel of {median_filter_kernel}."
                else:
                    logging.error(
                        f"Invalid filter_type: {filter_type}. Must be 'butterworth', 'gcv_spline', 'kalman', 'gaussian', 'loess', or 'median'."
                    )
                    raise ValueError(
                        f"Invalid filter_type: {filter_type}. Must be 'butterworth', 'gcv_spline', 'kalman', 'gaussian', 'loess', or 'median'."
                    )

                logging.info(f"Filtering with {args}")
                all_frames_angles_person_filt = all_frames_angles_person_interp.apply(
                    filter1d,
                    axis=0,
                    args=[Pose2Sim_config_dict, filter_type, frame_rate],
                )

            # Add floor_angle_estim to segment angles
            if correct_segment_angles_with_floor_angle and to_meters:
                logging.info(
                    f"Correcting segment angles by removing the {round(np.degrees(floor_angle_estim), 2)}° floor angle."
                )
                for ang_name in all_frames_angles_person_filt.columns:
                    if "horizontal" in angle_dict[ang_name][1]:
                        all_frames_angles_person_filt[ang_name] -= np.degrees(
                            floor_angle_estim
                        )

            all_frames_angles_person_export = all_frames_angles_person_filt.copy()
            if angle_output_mode == "bounded_principal":
                for ang_name in all_frames_angles_person_export.columns:
                    ang_params = angle_dict.get(ang_name)
                    if ang_params is None:
                        continue
                    # Keep segment orientations continuous; bound joint angles to principal range.
                    if ang_params[1] in ["flexion", "dorsiflexion"]:
                        all_frames_angles_person_export[ang_name] = (
                            wrap_angle_series_to_principal(
                                all_frames_angles_person_export[ang_name].to_numpy()
                            )
                        )

            # Remove columns with all nan values
            all_frames_angles_processed[:, idx_person, :] = (
                all_frames_angles_person_export
            )
            all_frames_angles_person_export.dropna(axis=1, how="all", inplace=True)
            all_frames_angles_person = all_frames_angles_person[
                all_frames_angles_person_export.columns
            ]

            # Build mot file
            angle_data = make_mot_with_angles(
                all_frames_angles_person_export,
                all_frames_time,
                str(angles_path_person),
            )
            logging.info(f"Angles saved to {angles_path_person.resolve()}.")

            # Plotting angles before and after interpolation and filtering
            all_frames_angles_person.insert(0, "time", all_frames_time)
            if show_plots or save_plots:
                pw = angle_plots(
                    all_frames_angles_person, angle_data, i, show=show_plots
                )  # i = current person
                if save_plots:
                    if show_plots:
                        for n, f in enumerate(pw.figure_handles):
                            dpi = pw.canvases[n].figure.dpi
                            f.set_size_inches(1280 / dpi, 720 / dpi)
                            title = pw.tabs.tabText(n)
                            plot_path = plots_output_dir / (
                                pose_output_path.stem
                                + f"_person{i:02d}_ang_{title.replace(' ', '_').replace('/', '_')}.png"
                            )
                            f.savefig(plot_path, dpi=dpi, bbox_inches="tight")
                    else:  # Tabbed plots not used
                        for title, f in pw:
                            dpi = f.dpi
                            f.set_size_inches(1280 / dpi, 720 / dpi)
                            plot_path = plots_output_dir / (
                                pose_output_path.stem
                                + f"_person{i:02d}_ang_{title.replace(' ', '_').replace('/', '_')}.png"
                            )
                            f.savefig(plot_path, dpi=dpi, bbox_inches="tight")
                            plt.close(f)
                    logging.info(f"Pose plots (m) saved in {plots_output_dir}.")

    # %% ==================================================
    # Save images/video with processed pose and angles
    # ====================================================
    if save_vid or save_img:
        logging.info("\nSaving images of processed pose and angles:")
        if save_vid:
            if vid_output_tmp_path.exists():
                vid_output_tmp_path.unlink()
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out_vid = cv2.VideoWriter(
                str(vid_output_tmp_path.absolute()),
                fourcc,
                fps,
                (cam_width, cam_height),
            )
            if not out_vid.isOpened():
                raise ValueError(
                    f"Failed to open temporary video writer at {vid_output_tmp_path}."
                )

        # Reorder persons
        all_frames_X_processed, all_frames_Y_processed = (
            all_frames_X_processed[:, selected_persons, :],
            all_frames_Y_processed[:, selected_persons, :],
        )
        all_frames_scores_processed = all_frames_scores_processed[
            :, selected_persons, :
        ]
        if save_angles or calculate_angles:
            all_frames_X_flipped_processed = all_frames_X_flipped_processed[
                :, selected_persons, :
            ]
            all_frames_angles_processed = all_frames_angles_processed[
                :, selected_persons, :
            ]

        # Saved overlay arrays are ordered by keypoint name, not anytree traversal.
        pose_model_with_new_ids = _remap_pose_model_ids_by_keypoint_names(
            pose_model,
            new_keypoints_names,
        )
        saved_overlay_keypoint_ids = list(range(len(new_keypoints_names)))
        motion_arrow_direction = np.array([0.0, -1.0], dtype=float)

        # Draw pose and angles on the full processed frame range.
        first_frame = frame_range[0]
        first_trim = 0
        last_trim = all_frames_X_processed.shape[0]
        ball_replay_trail = []
        ball_replay_trails_by_id = {}
        replay_fallback_frame = None
        replay_read_failures = []
        cap = cv2.VideoCapture(video_file_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, first_frame + first_trim)
        for i in range(first_trim, last_trim):
            source_frame_idx = first_frame + i
            success, frame = cap.read()
            if not success:
                replay_read_failures.append(source_frame_idx)
                if replay_fallback_frame is not None:
                    frame = replay_fallback_frame.copy()
                else:
                    frame = np.zeros((cam_height, cam_width, 3), dtype=np.uint8)
                if len(replay_read_failures) <= 3:
                    logging.warning(
                        "Could not read source frame %s while saving overlays. Reusing the last readable frame.",
                        source_frame_idx,
                    )
            else:
                replay_fallback_frame = frame.copy()
            img = frame.copy()
            img = draw_bounding_box(
                img,
                all_frames_X_processed[i],
                all_frames_Y_processed[i],
                colors=colors,
                fontSize=fontSize,
                thickness=thickness,
            )
            # Draw keypoints and skeleton using unified draw_pose function
            img = draw_pose(
                img,
                all_frames_X_processed[i],
                all_frames_Y_processed[i],
                all_frames_scores_processed[i],
                pose_model_with_new_ids,
                keypoint_names=new_keypoints_names if use_synthpose else None,
                backend_name=backend_name,
                thickness=thickness,
                keypoint_draw_threshold=draw_keypoint_likelihood_threshold,
                skeleton_draw_threshold=draw_skeleton_likelihood_threshold,
            )
            if calculate_angles:
                img = draw_angles(
                    img,
                    all_frames_X_processed[i],
                    all_frames_Y_processed[i],
                    all_frames_angles_processed[i],
                    all_frames_X_flipped_processed[i],
                    saved_overlay_keypoint_ids,
                    new_keypoints_names,
                    angle_names,
                    display_angle_values_on=display_angle_values_on,
                    colors=colors,
                    fontSize=fontSize,
                    thickness=thickness,
                )
            if ball_overlay_enabled and i < len(all_frames_ball_centers):
                frame_ball_center = all_frames_ball_centers[i]
                frame_ball_boxes = (
                    all_frames_ball_boxes[i]
                    if i < len(all_frames_ball_boxes)
                    else np.empty((0, 4), dtype=np.float32)
                )
                frame_ball_tracks = (
                    all_frames_ball_tracks[i] if i < len(all_frames_ball_tracks) else []
                )
                frame_selected_ball_id = (
                    all_frames_selected_ball_ids[i]
                    if i < len(all_frames_selected_ball_ids)
                    else None
                )
                if ball_multi_id_tracking:
                    source_track = _resolve_selected_ball_source_track(
                        frame_ball_tracks,
                        selected_track_id=frame_selected_ball_id,
                        center=frame_ball_center,
                        max_recovery_dist=ball_tracking_max_distance
                        if ball_tracking_max_distance is not None
                        else ball_max_jump_px,
                    )
                    selected_center = (
                        tuple(source_track.get("center"))
                        if source_track is not None
                        and source_track.get("center") is not None
                        else None
                    )
                    if selected_center is None and frame_ball_center is not None:
                        selected_center = tuple(frame_ball_center)
                    frame_ball_center = selected_center
                    if (
                        frame_selected_ball_id is not None
                        and selected_center is not None
                    ):
                        replay_trail = ball_replay_trails_by_id.get(
                            frame_selected_ball_id, []
                        )
                        replay_trail.append(selected_center)
                        if len(replay_trail) > ball_trail_length:
                            replay_trail = replay_trail[-ball_trail_length:]
                        ball_replay_trails_by_id[frame_selected_ball_id] = replay_trail
                    draw_trail = ball_replay_trails_by_id.get(
                        frame_selected_ball_id, []
                    )
                else:
                    if frame_ball_center is not None:
                        ball_replay_trail.append(tuple(frame_ball_center))
                        if len(ball_replay_trail) > ball_trail_length:
                            ball_replay_trail = ball_replay_trail[-ball_trail_length:]
                    draw_trail = ball_replay_trail
                img = draw_ball_overlay(
                    img,
                    frame_ball_boxes,
                    frame_ball_center,
                    draw_trail,
                    color=ball_color,
                    radius=ball_radius,
                    trail_alpha=ball_trail_alpha,
                    tracked_balls=frame_ball_tracks if ball_multi_id_tracking else None,
                    selected_track_id=frame_selected_ball_id
                    if ball_multi_id_tracking
                    else None,
                    show_ids=ball_show_ids and ball_multi_id_tracking,
                )
            if vertical_jump_enabled:
                for person_slot, jump_result in enumerate(vertical_jump_results):
                    if person_slot >= all_frames_X_processed.shape[1]:
                        continue
                    person_x = np.asarray(
                        all_frames_X_processed[i, person_slot, :], dtype=float
                    )
                    person_y = np.asarray(
                        all_frames_Y_processed[i, person_slot, :], dtype=float
                    )
                    overlay_color = colors[person_slot % len(colors)]
                    com_point = estimate_pelvis_trunk_com_xy_px(
                        person_x,
                        person_y,
                        new_keypoints_names,
                    )
                    img = draw_com_proxy_overlay(img, com_point, color=overlay_color)

                    body_weight_n = jump_result.get("body_weight_n")
                    full_vgrf_n = jump_result.get("full_vgrf_n")
                    force_n = np.nan
                    if full_vgrf_n is not None and i < len(full_vgrf_n):
                        force_n = float(full_vgrf_n[i])
                    anchor_point = estimate_grf_arrow_anchor_px(
                        person_x,
                        person_y,
                        new_keypoints_names,
                        floor_x_origin=motion_floor_origin_overlay[0],
                        floor_y_origin=motion_floor_origin_overlay[1],
                        floor_angle=motion_floor_angle_overlay,
                    )
                    img = draw_vgrf_arrow_overlay(
                        img,
                        anchor_point,
                        force_n=force_n,
                        body_weight_n=body_weight_n,
                        direction=motion_arrow_direction,
                        color=(0, 0, 255),
                        thickness=max(2, thickness + 1),
                    )

            # Save video or images
            if save_vid:
                out_vid.write(img)
            if save_img:
                cv2.imwrite(
                    str(
                        (
                            img_output_dir
                            / f"{output_dir_name}_{(i + frame_range[0]):06d}.png"
                        )
                    ),
                    img,
                )
        cap.release()
        if replay_read_failures:
            logging.warning(
                "Reused the last readable frame while saving overlays for %d unreadable source frames "
                "(first=%s, last=%s).",
                len(replay_read_failures),
                replay_read_failures[0],
                replay_read_failures[-1],
            )

        if save_vid:
            out_vid.release()
            final_fps = None
            if video_file == "webcam":
                total_processing_time = sum(frame_processing_times)
                if total_processing_time > 0:
                    final_fps = len(frame_processing_times) / total_processing_time
                    logging.info(
                        "Rewriting webcam processed video with average framerate %.3f fps.",
                        final_fps,
                    )
                else:
                    logging.warning(
                        "Could not compute webcam average framerate (no timing samples). "
                        "Keeping the original writer framerate %.3f fps.",
                        fps,
                    )

            try:
                if video_codec == "h264" or final_fps is not None:
                    transcode_video_ffmpeg(
                        vid_output_tmp_path,
                        vid_output_path,
                        codec=video_codec,
                        source_fps=fps if final_fps is not None else None,
                        desired_framerate=final_fps,
                    )
                    if vid_output_tmp_path.exists():
                        vid_output_tmp_path.unlink()
                else:
                    if vid_output_path.exists():
                        vid_output_path.unlink()
                    vid_output_tmp_path.rename(vid_output_path)
            except Exception as e:
                logging.error(
                    "Failed to finalize processed video with codec '%s': %s",
                    video_codec,
                    e,
                )
                raise

            if final_fps is not None:
                fps = final_fps
            logging.info(f"Processed video saved to {vid_output_path.resolve()}.")
        if save_img:
            logging.info(f"Processed images saved to {img_output_dir.resolve()}.")

    # %% ==================================================
    # OpenSim inverse kinematics (and optional marker augmentation)
    # ====================================================
    if do_ik or use_augmentation:
        import opensim as osim

        logging.info("\nPost-processing angles (with inverse kinematics):")
        if not to_meters:
            logging.warning(
                "Skipping marker augmentation and inverse kinematics as to_meters was set to False."
            )
        else:
            native_pose3d_dir = pose3d_dir
            native_kinematics_dir = kinematics_dir
            opensim_workspace_info = None
            opensim_pose3d_dir = native_pose3d_dir
            opensim_kinematics_dir = native_kinematics_dir

            # move all trc files containing _m_ string to pose3d_dir
            if not load_trc_px:
                trc_list = list(output_dir.glob("*_m_*.trc"))
            else:
                trc_list = [pose_path_person_m_i]

            if _should_use_ascii_safe_opensim_workspace(output_dir, trc_list):
                opensim_workspace_info = _create_ascii_safe_opensim_workspace(
                    trc_list,
                    bridge_trc_data_by_name=opensim_bridge_trc_data_by_name,
                    fps=fps,
                )
                opensim_pose3d_dir = opensim_workspace_info["pose3d_dir"]
                opensim_kinematics_dir = opensim_workspace_info["kinematics_dir"]
                Pose2Sim_config_dict["project"]["project_dir"] = str(
                    opensim_workspace_info["root_dir"]
                )
                logging.info(
                    "Using ASCII-safe OpenSim workspace at %s for non-ASCII result paths.",
                    opensim_workspace_info["root_dir"],
                )
            else:
                _stage_opensim_input_trcs(
                    trc_list,
                    opensim_pose3d_dir,
                    bridge_trc_data_by_name=opensim_bridge_trc_data_by_name,
                    fps=fps,
                )

            heights_m, masses = [], []
            for i in range(len(trc_data_m)):
                trc_data_m_i = trc_data_m[i]
                if do_ik and not use_augmentation:
                    logging.info(
                        f"- Person {i}: Running scaling and inverse kinematics without marker augmentation. Set use_augmentation to True if you need it."
                    )
                elif not do_ik and use_augmentation:
                    logging.info(
                        f"- Person {i}: Running marker augmentation without inverse kinematics. Set do_ik to True if you need it."
                    )
                else:
                    logging.info(
                        f"- Person {i}: Running marker augmentation and inverse kinematics."
                    )

                # Delete person if less than 4 valid frames
                pose_path_person = pose_output_path.parent / (
                    pose_output_path.stem + f"_person{i:02d}.trc"
                )
                all_frames_X_person = pd.DataFrame(
                    all_frames_X_homog[:, i, :], columns=new_keypoints_names
                )
                if new_visible_side[i] == "none":
                    logging.info(
                        f'Skipping marker augmentation and inverse kinematics because visible_side is "none".'
                    )
                else:
                    # Provide missing data to Pose2Sim_config_dict
                    trc_data_m_keypoint_names = list(trc_data_m_i.columns[1::3])
                    height_m_i = compute_height(
                        trc_data_m_i.iloc[:, 1:],
                        trc_data_m_keypoint_names,
                        fastest_frames_to_remove_percent=fastest_frames_to_remove_percent,
                        close_to_zero_speed=close_to_zero_speed_m,
                        large_hip_knee_angles=large_hip_knee_angles,
                        trimmed_extrema_percent=trimmed_extrema_percent,
                    )
                    mass_i = (
                        participant_masses[i]
                        if len(participant_masses) > i
                        else DEFAULT_MASS
                    )
                    if len(participant_masses) <= i:
                        logging.warning(
                            f"No mass provided. Using {DEFAULT_MASS} kg as default."
                        )
                    heights_m.append(height_m_i)
                    masses.append(mass_i)

            Pose2Sim_config_dict["project"]["participant_height"] = heights_m
            Pose2Sim_config_dict["project"]["participant_mass"] = masses
            Pose2Sim_config_dict["project"]["frame_range"] = "all"
            resolved_pose2sim_model_name = _configure_pose2sim_kinematics_bridge(
                Pose2Sim_config_dict,
                pose_model_name=pose_model_name,
                feet_on_floor=feet_on_floor,
            )
            if resolved_pose2sim_model_name != str(pose_model_name).strip().upper():
                logging.info(
                    "OpenSim bridge remapped pose_model '%s' -> '%s' for Pose2Sim kinematics.",
                    pose_model_name,
                    resolved_pose2sim_model_name,
                )
            Pose2Sim_config_dict = to_dict(Pose2Sim_config_dict)

            # Marker augmentation
            if use_augmentation:
                logging.info("Running marker augmentation...")
                augment_markers_all(Pose2Sim_config_dict)
                if opensim_workspace_info is not None:
                    logging.info(
                        "Augmented TRC results staged in temporary OpenSim workspace %s.",
                        opensim_pose3d_dir.resolve(),
                    )
                else:
                    logging.info(
                        f"Augmented trc results saved to {opensim_pose3d_dir.resolve()}.\n"
                    )

            if do_ik:
                if not save_angles or not calculate_angles:
                    logging.warning(
                        f"Skipping inverse kinematics because save_angles or calculate_angles is set to False."
                    )
                else:
                    logging.info("Running inverse kinematics...")
                    kinematics_all(Pose2Sim_config_dict)
                    for mot_file in opensim_kinematics_dir.glob("*.mot"):
                        if (mot_file.parent / (mot_file.stem + "_ik.mot")).exists():
                            os.remove(mot_file.parent / (mot_file.stem + "_ik.mot"))
                        os.rename(
                            mot_file, mot_file.parent / (mot_file.stem + "_ik.mot")
                        )
                    if opensim_workspace_info is not None:
                        logging.info(
                            "OpenSim intermediate .osim and .mot artifacts saved to temporary workspace %s.",
                            opensim_kinematics_dir.resolve().parent,
                        )
                    else:
                        logging.info(
                            f".osim model and .mot motion file results saved to {opensim_kinematics_dir.resolve().parent}.\n"
                        )

                    if inverse_dynamics_enabled:
                        logging.info("Running inverse dynamics from estimated vertical GRF...")
                        for person_slot, jump_result in enumerate(vertical_jump_results):
                            if person_slot >= len(selected_persons):
                                continue
                            if not isinstance(jump_result, dict):
                                continue
                            time_s = jump_result.get("time_s")
                            total_vgrf_n = jump_result.get("vgrf_n")
                            cop_xyz_m = jump_result.get("cop_xyz_m")
                            trc_name = jump_result.get("trc_name")
                            if not trc_name:
                                logging.warning(
                                    "Skipping inverse dynamics for person %s because the meter TRC name is unavailable.",
                                    person_slot,
                                )
                                continue

                            workspace_stem = _resolve_inverse_dynamics_workspace_stem(
                                trc_name,
                                opensim_workspace_info,
                            )
                            ik_mot_path = opensim_kinematics_dir / f"{workspace_stem}_ik.mot"
                            model_path = opensim_kinematics_dir / f"{workspace_stem}.osim"
                            artifact_paths = _resolve_inverse_dynamics_artifact_paths(
                                ik_mot_path
                            )
                            metadata = _build_inverse_dynamics_metadata_payload(
                                trc_name=trc_name,
                                ik_motion_file=ik_mot_path.name,
                                scaled_model_file=model_path.name,
                                external_loads_mot_file=artifact_paths["grf_mot"].name,
                                external_loads_xml_file=artifact_paths["external_loads_xml"].name,
                                inverse_dynamics_file=artifact_paths["inverse_dynamics_sto"].name,
                                metrics=jump_result.get("metrics"),
                                success=False,
                            )
                            if time_s is None or total_vgrf_n is None:
                                metadata["error"] = "Estimated vertical GRF export is unavailable."
                                write_grf_metrics_json(
                                    metadata,
                                    artifact_paths["metadata_json"],
                                )
                                logging.warning(
                                    "Skipping inverse dynamics for person %s because no estimated GRF export is available.",
                                    person_slot,
                                )
                                continue
                            if cop_xyz_m is None:
                                metadata["error"] = "Estimated CoP proxy is unavailable for inverse dynamics."
                                write_grf_metrics_json(
                                    metadata,
                                    artifact_paths["metadata_json"],
                                )
                                logging.warning(
                                    "Skipping inverse dynamics for person %s because no CoP proxy is available.",
                                    person_slot,
                                )
                                continue

                            if not ik_mot_path.exists() or not model_path.exists():
                                metadata["error"] = (
                                    f"IK artifacts are missing ({model_path.name}, {ik_mot_path.name})."
                                )
                                write_grf_metrics_json(
                                    metadata,
                                    artifact_paths["metadata_json"],
                                )
                                logging.warning(
                                    "Skipping inverse dynamics for person %s because IK artifacts are missing (%s, %s).",
                                    person_slot,
                                    model_path.name,
                                    ik_mot_path.name,
                                )
                                continue

                            try:
                                external_loads_df = build_external_loads_mot_data(
                                    time_s,
                                    total_vgrf_n,
                                    cop_xyz_m,
                                )
                                ik_storage = read_opensim_storage_file(ik_mot_path)
                                if "time" not in ik_storage.columns:
                                    raise ValueError(
                                        f"Inverse kinematics file {ik_mot_path.name} does not expose a time column."
                                    )
                                ik_time_s = np.asarray(
                                    ik_storage["time"], dtype=float
                                ).reshape(-1)
                                force_time_s = np.asarray(
                                    external_loads_df["time"], dtype=float
                                ).reshape(-1)
                                if len(ik_time_s) != len(force_time_s) or not np.allclose(
                                    ik_time_s,
                                    force_time_s,
                                    atol=1e-6,
                                    rtol=0.0,
                                ):
                                    raise ValueError(
                                        "Inverse dynamics requires the external-load timebase to match the IK motion file exactly."
                                    )

                                write_opensim_mot(
                                    external_loads_df,
                                    artifact_paths["grf_mot"],
                                    name="GroundReactionForces",
                                    in_degrees=False,
                                )
                                _write_estimated_grf_external_loads_xml(
                                    osim,
                                    artifact_paths["external_loads_xml"],
                                    artifact_paths["grf_mot"],
                                )
                                _run_estimated_grf_inverse_dynamics(
                                    osim,
                                    model_path=model_path,
                                    ik_mot_path=ik_mot_path,
                                    external_loads_xml_path=artifact_paths["external_loads_xml"],
                                    output_sto_path=artifact_paths["inverse_dynamics_sto"],
                                    start_time=float(force_time_s[0]),
                                    end_time=float(force_time_s[-1]),
                                )
                                metadata["success"] = True
                            except Exception as exc:
                                metadata["error"] = str(exc)
                                logging.warning(
                                    "Inverse dynamics failed for person %s: %s",
                                    person_slot,
                                    exc,
                                )

                            if metadata.get("success"):
                                _populate_joint_contribution_metadata(
                                    metadata,
                                    read_opensim_storage_file(
                                        artifact_paths["inverse_dynamics_sto"]
                                    ),
                                    start_frame=int(
                                        jump_result.get("metrics", {}).get(
                                            "lowest_com_frame", 0
                                        )
                                    ),
                                    end_frame=int(
                                        jump_result.get("metrics", {}).get(
                                            "takeoff_frame", 0
                                        )
                                    ),
                                    frame_rate=float(fps),
                                )

                            write_grf_metrics_json(
                                metadata,
                                artifact_paths["metadata_json"],
                            )
                            if metadata.get("success"):
                                logging.info(
                                    "Inverse dynamics artifacts saved to %s, %s, %s, and %s.",
                                    artifact_paths["grf_mot"].resolve(),
                                    artifact_paths["external_loads_xml"].resolve(),
                                    artifact_paths["inverse_dynamics_sto"].resolve(),
                                    artifact_paths["metadata_json"].resolve(),
                                )
            # Restore or preserve final pose-3d / kinematics layout.
            osim.Logger.removeFileSink()
            if opensim_workspace_info is not None:
                _move_ascii_safe_opensim_outputs(
                    opensim_workspace_info,
                    native_pose3d_dir,
                    native_kinematics_dir,
                )
                if native_pose3d_dir.exists() and not any(native_pose3d_dir.iterdir()):
                    native_pose3d_dir.rmdir()
                if native_kinematics_dir.exists() and not any(
                    native_kinematics_dir.iterdir()
                ):
                    native_kinematics_dir.rmdir()
                logging.info(
                    "OpenSim pose-3d artifacts saved to %s.",
                    native_pose3d_dir.resolve(),
                )
                logging.info(
                    "OpenSim kinematics artifacts saved to %s.\n",
                    native_kinematics_dir.resolve(),
                )
            else:
                logging.info(
                    "OpenSim pose-3d artifacts saved to %s.",
                    opensim_pose3d_dir.resolve(),
                )
                logging.info(
                    "OpenSim kinematics artifacts saved to %s.\n",
                    opensim_kinematics_dir.resolve(),
                )

    if public_meter_trc_data_by_name:
        for trc_name, trc_data_with_ball in public_meter_trc_data_by_name.items():
            public_m_path = output_dir / trc_name
            make_trc_with_trc_data(trc_data_with_ball, public_m_path, fps=fps)
        logging.info("Pose in meters saved to final TRC outputs.")
