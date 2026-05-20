#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
##################################################
## Pose Estimation Backend Abstraction Layer    ##
##################################################

Unified interface for pose estimation backends (RTMLib, SynthPose, Sapiens2).

This module provides:
- PoseBackend: Abstract base class defining the interface
- RTMLibBackend: ONNX-based pose estimation via Pose2Sim/rtmlib
- SynthPoseBackend: PyTorch-based VitPose estimation
- create_pose_backend(): Factory function for backend creation

Usage:
    from Sports2D.Utilities.pose_backend import create_pose_backend

    config = {'pose': {'pose_model': 'synthpose', 'device': 'auto', ...}}
    backend = create_pose_backend(config)
    keypoints, scores = backend(frame)
"""

from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Optional, Sequence
import numpy as np
import logging

from Sports2D.Utilities.manual_roi import (
    boxes_outside_rois,
    crop_frame_to_roi,
    expand_roi_xyxy,
    normalize_manual_roi_mode,
    offset_keypoints_to_full_frame,
    offset_xyxy_boxes_to_full_frame,
    roi_from_boxes_xyxy,
)
from Sports2D.Utilities.synthpose_skeleton import HALPE26_KEYPOINT_NAMES


# Retry delay for tracker initialization (multi-threading conflicts)
TRACKER_INIT_RETRY_DELAY = 3  # seconds

# COCO class IDs
PERSON_CLASS_ID = 0
SPORTS_BALL_CLASS_ID = 32


## AUTHORSHIP INFORMATION
__author__ = "Sports2D Contributors"
__copyright__ = "Copyright 2024, Sports2D"
__license__ = "BSD 3-Clause License"


def _keypoint_names_in_output_order(skeleton_tree) -> List[str]:
    """
    Return keypoint names ordered by the pose-array index.

    anytree traversal order is structural, but pose tensors are indexed by keypoint id.
    UI layers that display names next to coordinates need this id-sorted order.
    """

    from anytree import PreOrderIter

    indexed_names = [
        (int(node.id), str(node.name))
        for node in PreOrderIter(skeleton_tree)
        if node.id is not None
    ]
    if len(indexed_names) == 0:
        return []

    indexed_names.sort(key=lambda item: item[0])
    ordered_ids = [item[0] for item in indexed_names]
    expected_ids = list(range(len(indexed_names)))
    if ordered_ids == [
        0,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
    ]:
        return list(HALPE26_KEYPOINT_NAMES)

    if ordered_ids != expected_ids:
        logging.warning(
            "Skeleton keypoint ids are not contiguous: %s. "
            "Falling back to placeholder names for missing dense output slots.",
            ordered_ids,
        )

    name_by_id = {idx: name for idx, name in indexed_names}
    max_id = max(ordered_ids)
    return [
        name_by_id.get(idx, f"__unused_keypoint_{idx:02d}") for idx in range(max_id + 1)
    ]


class PoseBackend(ABC):
    """
    Abstract base class for pose estimation backends.

    All pose estimation backends must implement this interface.
    This ensures consistent behavior across RTMLib, SynthPose, and future backends.

    Interface Contract:
        - __call__(frame) -> (keypoints, scores)
        - reset() -> None
        - skeleton_tree -> anytree.Node
        - num_keypoints -> int
        - backend_name -> str
        - keypoint_names -> list
    """

    @abstractmethod
    def __call__(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate poses in a frame.

        Args:
            frame: BGR image array (H, W, 3) from cv2.VideoCapture

        Returns:
            keypoints: np.ndarray shape (N_persons, N_keypoints, 2) - (x, y) coordinates
            scores: np.ndarray shape (N_persons, N_keypoints) - confidence scores
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset tracker state (for new video/stream)."""
        pass

    @property
    @abstractmethod
    def skeleton_tree(self):
        """
        Return anytree Node hierarchy for skeleton structure.

        Returns:
            anytree.Node: Root node of the skeleton tree with .id and .name attributes
        """
        pass

    @property
    @abstractmethod
    def num_keypoints(self) -> int:
        """
        Number of keypoints this backend produces.

        Returns:
            int: Number of keypoints (e.g., 26 for HALPE_26, 52 for SynthPose)
        """
        pass

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """
        Backend identifier string.

        Returns:
            str: 'rtmlib' or 'synthpose'
        """
        pass

    @property
    @abstractmethod
    def keypoint_names(self) -> List[str]:
        """
        List of keypoint names in order.

        Returns:
            List[str]: Keypoint names matching the output order
        """
        pass

    @property
    def last_detections(self) -> Dict[str, np.ndarray]:
        """
        Optional per-frame detection metadata.

        Returns:
            Dict[str, np.ndarray]: Backend-specific detection outputs.
            Empty by default for backends that do not expose detections.
        """
        return {}

    def prepare_video_context(
        self, video_file_path=None, frame_range=None, input_kind="video"
    ) -> None:
        """Optional hook for backends that need file-video context."""
        return None


class _StaticPersonROITracker:
    """Crop person inference to a static ROI and restore full-frame keypoints."""

    def __init__(self, pose_tracker, person_roi: Sequence[int]):
        self._pose_tracker = pose_tracker
        self._person_roi = tuple(int(v) for v in person_roi)
        self._roi_released = False

    def __call__(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self._roi_released:
            return self._pose_tracker(frame)

        keypoints, scores = self._pose_tracker(
            crop_frame_to_roi(frame, self._person_roi)
        )
        keypoints = offset_keypoints_to_full_frame(keypoints, self._person_roi)
        if np.isfinite(np.asarray(keypoints, dtype=np.float32)).any():
            self._roi_released = True
            if hasattr(self._pose_tracker, "reset"):
                self._pose_tracker.reset()
        return keypoints, scores

    def reset(self) -> None:
        self._roi_released = False
        if hasattr(self._pose_tracker, "reset"):
            self._pose_tracker.reset()


class _AdaptivePersonROITracker:
    """Crop person inference to a mutable ROI that follows accepted person detections."""

    def __init__(
        self,
        pose_tracker,
        person_roi: Sequence[int],
        tracking_margin_px: int = 48,
        reacquire_patience: int = 6,
        reacquire_frequency: int = 15,
    ):
        self._pose_tracker = pose_tracker
        self._seed_person_roi = tuple(int(v) for v in person_roi)
        self._tracking_margin_px = max(0, int(tracking_margin_px))
        self._reacquire_patience = max(1, int(reacquire_patience))
        self._reacquire_frequency = max(1, int(reacquire_frequency))
        self.reset()

    @staticmethod
    def _person_boxes_from_keypoints(keypoints) -> np.ndarray:
        kpts = np.asarray(keypoints, dtype=np.float32)
        if kpts.size == 0 or kpts.ndim != 3 or kpts.shape[-1] < 2:
            return np.empty((0, 4), dtype=np.float32)
        person_boxes = []
        for person_kpts in kpts:
            valid = ~np.isnan(person_kpts[:, 0]) & ~np.isnan(person_kpts[:, 1])
            if not np.any(valid):
                continue
            xs = person_kpts[valid, 0]
            ys = person_kpts[valid, 1]
            person_boxes.append([np.min(xs), np.min(ys), np.max(xs), np.max(ys)])
        if not person_boxes:
            return np.empty((0, 4), dtype=np.float32)
        return np.asarray(person_boxes, dtype=np.float32)

    def _should_force_full_frame_reacquire(self) -> bool:
        if self._person_roi_miss_count < self._reacquire_patience:
            return False
        if self._last_full_frame_reacquire_frame is None:
            return True
        return (
            self._frame_count - self._last_full_frame_reacquire_frame
        ) >= self._reacquire_frequency

    def _tracker_uses_detection_cadence(self) -> bool:
        return hasattr(self._pose_tracker, "frame_cnt") and hasattr(
            self._pose_tracker, "det_frequency"
        )

    def _tracker_will_run_detection(self) -> bool:
        if not self._tracker_uses_detection_cadence():
            return True
        det_frequency = max(1, int(getattr(self._pose_tracker, "det_frequency", 1)))
        frame_cnt = int(getattr(self._pose_tracker, "frame_cnt", 0))
        return frame_cnt % det_frequency == 0

    def _clear_tracker_local_state(self) -> None:
        if hasattr(self._pose_tracker, "bboxes_last_frame"):
            self._pose_tracker.bboxes_last_frame = []
        if hasattr(self._pose_tracker, "track_ids_last_frame"):
            self._pose_tracker.track_ids_last_frame = []

    def _schedule_next_roi(self, roi) -> None:
        if roi is None:
            return
        next_roi = tuple(int(v) for v in roi)
        if self._tracker_uses_detection_cadence():
            self._pending_person_roi = next_roi
        else:
            self._active_person_roi = next_roi
            self._pending_person_roi = None

    def _apply_pending_roi_if_needed(self) -> None:
        if self._pending_person_roi is None:
            return
        self._active_person_roi = self._pending_person_roi
        self._pending_person_roi = None

    def __call__(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        frame_shape = frame.shape
        tracker_runs_detection = self._tracker_will_run_detection()
        if tracker_runs_detection:
            self._apply_pending_roi_if_needed()
        force_full_frame = (
            tracker_runs_detection and self._should_force_full_frame_reacquire()
        )
        if tracker_runs_detection:
            inference_roi = (
                None
                if force_full_frame
                else roi_from_boxes_xyxy(
                    [self._active_person_roi],
                    frame_shape,
                    padding_px=0,
                )
                if self._active_person_roi is not None
                else None
            )
        else:
            inference_roi = self._last_inference_roi

        if tracker_runs_detection and inference_roi != self._last_inference_roi:
            self._clear_tracker_local_state()

        keypoints, scores = self._pose_tracker(crop_frame_to_roi(frame, inference_roi))
        if inference_roi is not None:
            keypoints = offset_keypoints_to_full_frame(keypoints, inference_roi)
        self._last_inference_roi = inference_roi

        if tracker_runs_detection:
            person_boxes = self._person_boxes_from_keypoints(keypoints)
            if len(person_boxes) > 0:
                updated_roi = roi_from_boxes_xyxy(
                    person_boxes,
                    frame_shape,
                    padding_px=self._tracking_margin_px,
                )
                self._schedule_next_roi(updated_roi)
                self._person_roi_miss_count = 0
            elif inference_roi is not None:
                self._person_roi_miss_count += 1
                self._schedule_next_roi(
                    expand_roi_xyxy(
                        self._active_person_roi,
                        frame_shape,
                        padding_px=self._tracking_margin_px,
                    )
                )
            if force_full_frame:
                self._last_full_frame_reacquire_frame = self._frame_count

        self._frame_count += 1
        return keypoints, scores

    def reset(self) -> None:
        self._active_person_roi = tuple(int(v) for v in self._seed_person_roi)
        self._pending_person_roi = None
        self._person_roi_miss_count = 0
        self._last_full_frame_reacquire_frame = None
        self._last_inference_roi = None
        self._frame_count = 0
        if hasattr(self._pose_tracker, "reset"):
            self._pose_tracker.reset()


class _RTMLibBallAwareTracker:
    """
    Wrapper tracker that preserves standard RTMLib person pose behavior and
    runs a separate COCO detector for optional ball metadata.
    """

    MODE_TO_COCO_SIZE = {
        "performance": "x",
        "balanced": "m",
        "lightweight": "s",
    }

    COCO_YOLOX_MODELS = {
        "s": "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.onnx",
        "m": "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_m.onnx",
        "l": "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_l.onnx",
        "x": "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_x.onnx",
    }

    def __init__(
        self,
        pose_tracker,
        mode: str,
        det_frequency: int,
        backend: str,
        device: str,
        num_keypoints: int,
        ball_class_ids: Optional[Sequence[int]] = None,
        ball_detection_threshold: float = 0.1,
        ball_nms_score_threshold: float = 0.2,
        ball_roi: Optional[Sequence[int]] = None,
        ball_ignore_zones=None,
    ):
        from rtmlib import YOLOX

        self._pose_tracker = pose_tracker
        self._num_keypoints = int(num_keypoints)
        self._det_frequency = max(1, int(det_frequency))
        self._frame_count = 0
        self._ball_class_ids = set(ball_class_ids or [SPORTS_BALL_CLASS_ID])
        self._ball_roi = (
            tuple(int(v) for v in ball_roi) if ball_roi is not None else None
        )
        self._ball_ignore_zones = self._normalize_runtime_rois(ball_ignore_zones)
        self._ball_roi_released = False
        self.last_detections: Dict[str, np.ndarray] = self._empty_detections()

        requested_size = self.MODE_TO_COCO_SIZE.get(str(mode).lower(), "m")
        detector_url = self.COCO_YOLOX_MODELS.get(
            requested_size, self.COCO_YOLOX_MODELS["m"]
        )
        detector_backend = backend if backend != "auto" else "onnxruntime"
        detector_device = "cuda" if device == "cuda" else "cpu"

        self._ball_detector = YOLOX(
            onnx_model=detector_url,
            model_input_size=(640, 640),
            mode="multiclass",
            nms_thr=float(np.clip(ball_nms_score_threshold, 0.01, 0.9)),
            score_thr=float(np.clip(ball_detection_threshold, 0.01, 0.9)),
            backend=detector_backend,
            device=detector_device,
        )
        logging.info(
            "RTMLib ball detector initialized (weights=coco/%s, backend=%s, device=%s, "
            "score_thr=%s, nms_thr=%s)",
            requested_size,
            detector_backend,
            detector_device,
            float(np.clip(ball_detection_threshold, 0.01, 0.9)),
            float(np.clip(ball_nms_score_threshold, 0.01, 0.9)),
        )

    @staticmethod
    def _empty_detections() -> Dict[str, np.ndarray]:
        return {
            "boxes": np.empty((0, 4), dtype=np.float32),
            "classes": np.empty((0,), dtype=np.int32),
            "scores": np.empty((0,), dtype=np.float32),
            "person_boxes": np.empty((0, 4), dtype=np.float32),
            "ball_boxes": np.empty((0, 4), dtype=np.float32),
            "ball_scores": np.empty((0,), dtype=np.float32),
        }

    @staticmethod
    def _ensure_xyxy(boxes) -> np.ndarray:
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

    @staticmethod
    def _normalize_runtime_rois(rois):
        """Normalize runtime ignore-zone payloads to xyxy integer tuples."""
        normalized = []
        for roi in rois or []:
            arr = np.asarray(roi, dtype=np.float32).reshape(-1)
            if arr.size != 4 or not np.all(np.isfinite(arr)):
                continue
            normalized.append(tuple(int(round(v)) for v in arr.tolist()))
        return normalized

    def _filter_ignored_ball_detections(self, boxes, classes, scores):
        """Remove ball detections whose boxes overlap any configured ignore zone."""
        boxes = self._ensure_xyxy(boxes)
        classes = np.asarray(classes, dtype=np.int32).reshape(-1)
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        ignore_zones = getattr(self, "_ball_ignore_zones", None) or []
        if len(ignore_zones) == 0 or len(boxes) == 0:
            return boxes, classes, scores
        if len(classes) != len(boxes):
            return boxes, classes, scores
        if len(scores) != len(boxes):
            scores = np.full((len(boxes),), np.nan, dtype=np.float32)

        ball_mask = np.isin(classes, list(self._ball_class_ids))
        if not np.any(ball_mask):
            return boxes, classes, scores

        keep_mask = np.ones((len(boxes),), dtype=bool)
        keep_mask[ball_mask] = boxes_outside_rois(boxes[ball_mask], ignore_zones)
        return boxes[keep_mask], classes[keep_mask], scores[keep_mask]

    @staticmethod
    def _person_boxes_from_keypoints(keypoints) -> np.ndarray:
        kpts = np.asarray(keypoints)
        if kpts.size == 0 or kpts.ndim != 3 or kpts.shape[-1] < 2:
            return np.empty((0, 4), dtype=np.float32)
        person_boxes = []
        for person_kpts in kpts:
            valid = ~np.isnan(person_kpts[:, 0]) & ~np.isnan(person_kpts[:, 1])
            if not np.any(valid):
                continue
            xs = person_kpts[valid, 0]
            ys = person_kpts[valid, 1]
            person_boxes.append([np.min(xs), np.min(ys), np.max(xs), np.max(ys)])
        if not person_boxes:
            return np.empty((0, 4), dtype=np.float32)
        return np.asarray(person_boxes, dtype=np.float32)

    def _run_ball_detection(
        self, frame: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        run_detection = (
            self._frame_count % self._det_frequency == 0 or self._frame_count == 0
        )
        if not run_detection:
            return (
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.int32),
                np.empty((0,), dtype=np.float32),
            )

        try:
            active_ball_roi = (
                None if getattr(self, "_ball_roi_released", False) else self._ball_roi
            )
            detector_outputs = self._ball_detector(
                crop_frame_to_roi(frame, active_ball_roi)
            )
        except Exception as e:
            logging.debug(
                "RTMLib ball detector failed on frame %s: %s", self._frame_count, e
            )
            return (
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.int32),
                np.empty((0,), dtype=np.float32),
            )

        if isinstance(detector_outputs, tuple) and len(detector_outputs) >= 2:
            raw_boxes = np.asarray(detector_outputs[0], dtype=np.float32)
            boxes = self._ensure_xyxy(raw_boxes)
            classes = np.asarray(detector_outputs[1], dtype=np.int32).reshape(-1)
            if len(detector_outputs) >= 3:
                scores = np.asarray(detector_outputs[2], dtype=np.float32).reshape(-1)
            elif raw_boxes.ndim == 2 and raw_boxes.shape[1] >= 5:
                scores = raw_boxes[:, 4].astype(np.float32, copy=False)
            else:
                scores = np.full((len(boxes),), np.nan, dtype=np.float32)
        else:
            raw_boxes = np.asarray(detector_outputs, dtype=np.float32)
            boxes = self._ensure_xyxy(raw_boxes)
            classes = np.empty((0,), dtype=np.int32)
            if raw_boxes.ndim == 2 and raw_boxes.shape[1] >= 5:
                scores = raw_boxes[:, 4].astype(np.float32, copy=False)
            else:
                scores = np.full((len(boxes),), np.nan, dtype=np.float32)

        if len(classes) != len(boxes):
            classes = np.empty((0,), dtype=np.int32)
            boxes = np.empty((0, 4), dtype=np.float32)
            scores = np.empty((0,), dtype=np.float32)
        if len(scores) != len(boxes):
            scores = np.full((len(boxes),), np.nan, dtype=np.float32)
        if (
            not getattr(self, "_ball_roi_released", False)
            and self._ball_roi is not None
            and len(boxes) > 0
        ):
            boxes = offset_xyxy_boxes_to_full_frame(boxes, self._ball_roi)
        boxes, classes, scores = self._filter_ignored_ball_detections(
            boxes,
            classes,
            scores,
        )
        if (
            not getattr(self, "_ball_roi_released", False)
            and self._ball_roi is not None
            and len(boxes) > 0
        ):
            self._ball_roi_released = True
        return boxes, classes, scores

    def __call__(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        keypoints, scores = self._pose_tracker(frame)
        person_boxes = self._person_boxes_from_keypoints(keypoints)
        boxes, classes, det_scores = self._run_ball_detection(frame)

        if len(classes) > 0 and len(boxes) > 0:
            ball_mask = np.isin(classes, list(self._ball_class_ids))
            ball_boxes = boxes[ball_mask]
            ball_scores = (
                det_scores[ball_mask]
                if len(det_scores) == len(boxes)
                else np.full((len(ball_boxes),), np.nan, dtype=np.float32)
            )
        else:
            ball_boxes = np.empty((0, 4), dtype=np.float32)
            ball_scores = np.empty((0,), dtype=np.float32)

        self.last_detections = {
            "boxes": boxes,
            "classes": classes,
            "scores": det_scores,
            "person_boxes": person_boxes,
            "ball_boxes": ball_boxes,
            "ball_scores": ball_scores,
        }
        self._frame_count += 1
        return keypoints, scores

    def reset(self) -> None:
        self._frame_count = 0
        self._ball_roi_released = False
        if hasattr(self._pose_tracker, "reset"):
            self._pose_tracker.reset()
        self.last_detections = self._empty_detections()


class RTMLibBackend(PoseBackend):
    """
    RTMLib pose estimation backend using ONNX models via Pose2Sim.

    This backend wraps Pose2Sim's RTMLib integration, providing:
    - ONNX-based inference (fast, cross-platform)
    - Multiple model options (body_with_feet, whole_body, body)
    - Multiple execution providers (onnxruntime, openvino, opencv)

    Usage:
        config = {'pose': {'pose_model': 'body_with_feet', 'mode': 'balanced', ...}}
        backend = RTMLibBackend(config)
        keypoints, scores = backend(frame)
    """

    def __init__(self, config_dict: dict):
        """
        Initialize RTMLib backend.

        Args:
            config_dict: Full configuration dictionary with 'pose' section containing:
                - pose_model: Model name ('body_with_feet', 'whole_body', 'body')
                - mode: Quality mode ('lightweight', 'balanced', 'performance')
                - backend: ONNX provider ('auto', 'onnxruntime', 'openvino', 'opencv')
                - device: Device selection ('auto', 'cuda', 'cpu')
                - det_frequency: Detection frequency (every N frames)

        Raises:
            ValueError: If pose_model or mode is invalid
        """
        from Pose2Sim.poseEstimation import (
            setup_model_class_mode,
            setup_backend_device,
            setup_pose_tracker,
        )

        pose_config = config_dict.get("pose", {})

        # 1. Model and mode setup
        pose_model_name = pose_config.get("pose_model", "body_with_feet")
        mode = pose_config.get("mode", "balanced")

        self._pose_model, self._ModelClass, self._mode = setup_model_class_mode(
            pose_model_name, mode, config_dict
        )

        # 2. Backend and device setup (ONNX providers)
        backend = pose_config.get("backend", "auto")
        device = pose_config.get("device", "auto")
        self._backend, self._device = setup_backend_device(backend, device)

        det_frequency = pose_config.get("det_frequency", 4)
        detect_ball = bool(pose_config.get("detect_ball", False))
        manual_person_roi = pose_config.get("_manual_person_roi")
        manual_ball_roi = pose_config.get("_manual_ball_roi") or manual_person_roi
        manual_roi_mode = normalize_manual_roi_mode(
            pose_config.get("manual_roi_mode", "bootstrap"),
        )
        manual_roi_tracking_margin_px = max(
            0,
            int(pose_config.get("manual_roi_tracking_margin_px", 48)),
        )
        manual_roi_reacquire_patience = max(
            1,
            int(pose_config.get("manual_roi_reacquire_patience", 6)),
        )
        manual_roi_reacquire_frequency = max(
            1,
            int(pose_config.get("manual_roi_reacquire_frequency", 15)),
        )
        manual_ball_ignore_zones = _RTMLibBallAwareTracker._normalize_runtime_rois(
            pose_config.get("_manual_ball_ignore_zones"),
        )
        ball_class_ids = pose_config.get("ball_class_ids", [SPORTS_BALL_CLASS_ID])
        ball_detection_threshold = pose_config.get(
            "ball_detection_threshold",
            0.1,
        )
        ball_nms_score_threshold = pose_config.get(
            "ball_nms_score_threshold",
            0.2,
        )
        if isinstance(ball_class_ids, int):
            ball_class_ids = [ball_class_ids]
        elif isinstance(ball_class_ids, (list, tuple, set)):
            try:
                ball_class_ids = [int(c) for c in ball_class_ids]
            except Exception:
                ball_class_ids = [SPORTS_BALL_CLASS_ID]

        if manual_roi_mode == "adaptive_person" and manual_person_roi is None:
            logging.warning(
                "manual_roi_mode='adaptive_person' requires a manual person ROI. Falling back to bootstrap."
            )
            manual_roi_mode = "bootstrap"

        # Cache keypoint names and count
        self._keypoint_names = _keypoint_names_in_output_order(self._pose_model)
        self._num_keypoints = len(self._keypoint_names)
        self._last_detections: Dict[str, np.ndarray] = {}
        self._supports_ball_detection = False

        # 3. Tracker initialization with retry for multi-threading
        def _init_default_tracker():
            try:
                return setup_pose_tracker(
                    self._ModelClass,
                    det_frequency,
                    self._mode,
                    False,
                    self._backend,
                    self._device,
                )
            except RuntimeError as e:
                # Retry once for multi-threading initialization issues
                import time

                logging.warning(f"RTMLib tracker init retry due to: {e}")
                time.sleep(TRACKER_INIT_RETRY_DELAY)
                return setup_pose_tracker(
                    self._ModelClass,
                    det_frequency,
                    self._mode,
                    False,
                    self._backend,
                    self._device,
                )

        if detect_ball:
            try:
                default_tracker = _init_default_tracker()
                if (
                    manual_person_roi is not None
                    and manual_roi_mode == "adaptive_person"
                ):
                    default_tracker = _AdaptivePersonROITracker(
                        pose_tracker=default_tracker,
                        person_roi=manual_person_roi,
                        tracking_margin_px=manual_roi_tracking_margin_px,
                        reacquire_patience=manual_roi_reacquire_patience,
                        reacquire_frequency=manual_roi_reacquire_frequency,
                    )
                elif manual_person_roi is not None:
                    default_tracker = _StaticPersonROITracker(
                        pose_tracker=default_tracker,
                        person_roi=manual_person_roi,
                    )
                self._tracker = _RTMLibBallAwareTracker(
                    pose_tracker=default_tracker,
                    mode=self._mode,
                    det_frequency=det_frequency,
                    backend=self._backend,
                    device=self._device,
                    num_keypoints=self._num_keypoints,
                    ball_class_ids=ball_class_ids,
                    ball_detection_threshold=ball_detection_threshold,
                    ball_nms_score_threshold=ball_nms_score_threshold,
                    ball_roi=manual_ball_roi,
                    ball_ignore_zones=manual_ball_ignore_zones,
                )
                self._supports_ball_detection = True
            except Exception as e:
                logging.warning(
                    f"Ball detection requested but ball-aware tracker init failed: {e}. "
                    "Falling back to standard RTMLib tracker."
                )
                default_tracker = _init_default_tracker()
                if (
                    manual_person_roi is not None
                    and manual_roi_mode == "adaptive_person"
                ):
                    default_tracker = _AdaptivePersonROITracker(
                        pose_tracker=default_tracker,
                        person_roi=manual_person_roi,
                        tracking_margin_px=manual_roi_tracking_margin_px,
                        reacquire_patience=manual_roi_reacquire_patience,
                        reacquire_frequency=manual_roi_reacquire_frequency,
                    )
                elif manual_person_roi is not None:
                    default_tracker = _StaticPersonROITracker(
                        pose_tracker=default_tracker,
                        person_roi=manual_person_roi,
                    )
                self._tracker = default_tracker
        else:
            default_tracker = _init_default_tracker()
            if manual_person_roi is not None and manual_roi_mode == "adaptive_person":
                default_tracker = _AdaptivePersonROITracker(
                    pose_tracker=default_tracker,
                    person_roi=manual_person_roi,
                    tracking_margin_px=manual_roi_tracking_margin_px,
                    reacquire_patience=manual_roi_reacquire_patience,
                    reacquire_frequency=manual_roi_reacquire_frequency,
                )
            elif manual_person_roi is not None:
                default_tracker = _StaticPersonROITracker(
                    pose_tracker=default_tracker,
                    person_roi=manual_person_roi,
                )
            self._tracker = default_tracker

        logging.info(
            f"RTMLibBackend initialized: model={pose_model_name}, mode={self._mode}, "
            f"backend={self._backend}, device={self._device}, keypoints={self._num_keypoints}, "
            f"ball_detection={self._supports_ball_detection}"
        )

    def __call__(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Run pose estimation. Returns (keypoints, scores)."""
        outputs = self._tracker(frame)
        if isinstance(outputs, tuple) and len(outputs) >= 2:
            keypoints, scores = outputs[0], outputs[1]
        else:
            keypoints, scores = outputs

        if self._supports_ball_detection and hasattr(self._tracker, "last_detections"):
            self._last_detections = getattr(self._tracker, "last_detections", {}) or {}
        else:
            self._last_detections = {}

        return keypoints, scores

    def reset(self) -> None:
        """Reset tracker state."""
        if hasattr(self._tracker, "reset"):
            self._tracker.reset()
        self._last_detections = {}

    @property
    def skeleton_tree(self):
        """Return skeleton tree from pose_model."""
        return self._pose_model

    @property
    def num_keypoints(self) -> int:
        """Return keypoint count."""
        return self._num_keypoints

    @property
    def backend_name(self) -> str:
        return "rtmlib"

    @property
    def keypoint_names(self) -> List[str]:
        """Return keypoint names from skeleton tree."""
        return self._keypoint_names

    @property
    def last_detections(self) -> Dict[str, np.ndarray]:
        return self._last_detections


class SynthPoseBackend(PoseBackend):
    """
    SynthPose backend using VitPose models with PyTorch.

    This backend provides:
    - PyTorch-based inference (GPU-accelerated)
    - 52 keypoints (17 COCO + 35 anatomical markers)
    - Multiple detector options (yolox, yolo26, rtdetr, rtdetrv4, sam3)

    Model Selection:
        SynthPose supports RTMLib-compatible 'mode' parameter for model selection:
        - 'performance': VitPose-huge (most accurate, slower)
        - 'balanced': VitPose-base (good balance of speed/accuracy)
        - 'lightweight': NOT SUPPORTED - falls back to VitPose-base with warning

        Alternatively, use pose_model directly:
        - 'synthpose': VitPose-huge
        - 'synthpose_base': VitPose-base

    Device Selection:
        - 'auto': Auto-detect (CUDA > MPS > CPU)
        - 'cuda': Force NVIDIA GPU
        - 'mps': Force Apple Metal
        - 'cpu': Force CPU

    Usage:
        # Using mode parameter (RTMLib-compatible)
        config = {'pose': {'pose_model': 'synthpose', 'mode': 'performance', ...}}

        # Using explicit pose_model
        config = {'pose': {'pose_model': 'synthpose_base', 'device': 'auto', ...}}

        backend = SynthPoseBackend(config)
        keypoints, scores = backend(frame)
    """

    # Mode to VitPose model mapping (RTMLib compatibility)
    MODE_TO_VITPOSE = {
        "performance": "huge",  # VitPose-huge (most accurate)
        "balanced": "base",  # VitPose-base (balanced)
        "lightweight": "base",  # Fallback to base (lightweight not supported)
    }
    # Explicit SynthPose model-size override mapping
    SYNTHPOSE_MODEL_SIZE_TO_VITPOSE = {
        "performance": "huge",
        "balanced": "base",
        "lightweight": "base",  # lightweight not available in HF model family
        "huge": "huge",
        "base": "base",
    }

    def __init__(self, config_dict: dict):
        """
        Initialize SynthPose backend.

        Args:
            config_dict: Full configuration dictionary with 'pose' section containing:
                - pose_model: 'synthpose' (huge) or 'synthpose_base' (base)
                - mode: RTMLib-compatible mode ('performance', 'balanced', 'lightweight')
                       - 'performance' → VitPose-huge
                       - 'balanced' → VitPose-base
                       - 'lightweight' → VitPose-base (with warning)
                - device: Device selection ('auto', 'cuda', 'cpu', 'mps')
                - det_frequency: Detection frequency (every N frames)
                - keypoint_likelihood_threshold: Confidence threshold
                - synthpose_detector: Detector type ('yolox', 'yolo26', 'rtdetr', 'rtdetrv4', 'sam3')
                - sam3_target: SAM3 prompt preset ('ball' or 'broad_jump')
                - sam3_model_path: Local raw .pt checkpoint or HF id/directory for SAM3
                - sam3_processor_path: Optional processor path for HF SAM3 bundles
                - sam3_runtime: SAM3 runtime backend ('transformers', 'meta', or auto-switched)
                - sam3_show_realtime_masks: Draw SAM3 masks in live preview
                - sam3_realtime_mask_alpha: Live preview SAM3 mask opacity
                - ball_detector_backend: 'same' or 'sam3' for hybrid ball detection
                - save_vid/save_img + detect_ball: Persist SAM3 ball masks for export rendering

        Raises:
            ImportError: If torch/transformers not installed
        """
        from Sports2D.Utilities.synthpose_tracker import SynthPosePoseTracker
        from Sports2D.Utilities.synthpose_skeleton import (
            create_synthpose_skeleton,
            SYNTHPOSE_KEYPOINT_NAMES,
        )

        pose_config = config_dict.get("pose", {})
        pose_model = pose_config.get("pose_model", "synthpose").lower()
        mode = pose_config.get("mode", "balanced").lower()
        synthpose_model_size = pose_config.get("synthpose_model_size", "")
        if synthpose_model_size in [None, "", "auto", "none"]:
            synthpose_model_size = pose_config.get("synthpose_detector_size", "")
            if synthpose_model_size not in [None, "", "auto", "none"]:
                logging.warning(
                    "`synthpose_detector_size` is deprecated. "
                    "Use `synthpose_model_size` to control VitPose model size."
                )
        synthpose_model_size = (
            str(synthpose_model_size).lower()
            if synthpose_model_size not in [None, ""]
            else ""
        )

        # Determine VitPose size:
        # 1) explicit synthpose_model_size override
        # 2) explicit pose_model
        # 3) mode-based fallback
        if synthpose_model_size:
            if synthpose_model_size in self.SYNTHPOSE_MODEL_SIZE_TO_VITPOSE:
                self._mode = self.SYNTHPOSE_MODEL_SIZE_TO_VITPOSE[synthpose_model_size]
            else:
                logging.warning(
                    f"Unknown synthpose_model_size '{synthpose_model_size}'. "
                    "Falling back to pose_model/mode selection."
                )
                synthpose_model_size = ""

        if not synthpose_model_size:
            if pose_model == "synthpose_base":
                # Explicit base model requested
                self._mode = "base"
            elif pose_model == "synthpose":
                # Generic synthpose - use mode parameter for model selection
                if mode == "lightweight":
                    logging.warning(
                        "SynthPose does not support 'lightweight' mode. "
                        "VitPose-base (balanced) will be used instead. "
                        "For maximum accuracy, use mode='performance' (VitPose-huge)."
                    )
                    self._mode = "base"
                elif mode == "performance":
                    self._mode = "huge"
                else:
                    # balanced or any other value defaults to base
                    self._mode = self.MODE_TO_VITPOSE.get(mode, "base")
            else:
                # Unknown pose_model, default to huge
                self._mode = "huge"

        # Detectors must depend on mode parameter
        detector_mode = (
            mode if mode in ["performance", "balanced", "lightweight"] else "balanced"
        )
        if detector_mode != mode:
            logging.warning(
                f"Unsupported mode '{mode}' for SynthPose detector. Using 'balanced'."
            )

        if synthpose_model_size == "lightweight":
            logging.warning(
                "synthpose_model_size='lightweight' maps to VitPose-base "
                "because no lightweight HF VitPose is available."
            )

        # Device selection (config takes priority, 'auto' triggers detection in tracker)
        device = pose_config.get("device", "auto")
        detector_threshold = pose_config.get(
            "person_detection_threshold",
            pose_config.get("keypoint_likelihood_threshold", 0.3),
        )
        ball_detection_threshold = pose_config.get("ball_detection_threshold", 0.1)
        show_realtime_sam3_masks = bool(
            config_dict.get("base", {}).get("show_realtime_results", False)
            and pose_config.get("sam3_show_realtime_masks", False)
        )

        # Initialize tracker
        self._tracker = SynthPosePoseTracker(
            mode=self._mode,
            device=device,
            det_frequency=pose_config.get("det_frequency", 4),
            person_threshold=detector_threshold,
            backend=pose_config.get("backend", "auto"),
            detector=pose_config.get("synthpose_detector", "yolox"),
            detect_ball=bool(pose_config.get("detect_ball", False)),
            ball_class_ids=pose_config.get("ball_class_ids", [SPORTS_BALL_CLASS_ID]),
            ball_detection_threshold=ball_detection_threshold,
            # Detector size is controlled by mode parameter.
            detector_size=detector_mode,
            ball_nms_score_threshold=pose_config.get("ball_nms_score_threshold", 0.2),
            sam3_target=pose_config.get("sam3_target", "ball"),
            sam3_model_path=pose_config.get("sam3_model_path", ""),
            sam3_processor_path=pose_config.get("sam3_processor_path", ""),
            sam3_runtime=pose_config.get("sam3_runtime", "transformers"),
            sam3_store_masks=bool(pose_config.get("sam3_store_masks", False)),
            sam3_show_realtime_masks=show_realtime_sam3_masks,
            sam3_save_ball_masks=False,
            sam3_inference_mode=pose_config.get("sam3_inference_mode", "image"),
            sam3_bootstrap_frames=pose_config.get("sam3_bootstrap_frames", 12),
            sam3_video_refresh_frequency=pose_config.get(
                "sam3_video_refresh_frequency", pose_config.get("det_frequency", 4)
            ),
            sam3_video_reseed_on_loss=bool(
                pose_config.get("sam3_video_reseed_on_loss", True)
            ),
            sam3_video_loss_patience=pose_config.get("sam3_video_loss_patience", 3),
            ball_detector_backend=pose_config.get("ball_detector_backend", "same"),
            manual_person_roi=pose_config.get("_manual_person_roi"),
            manual_ball_roi=pose_config.get("_manual_ball_roi"),
            manual_ball_ignore_zones=_RTMLibBallAwareTracker._normalize_runtime_rois(
                pose_config.get("_manual_ball_ignore_zones"),
            ),
            manual_roi_mode=pose_config.get("manual_roi_mode", "bootstrap"),
            manual_roi_tracking_margin_px=pose_config.get(
                "manual_roi_tracking_margin_px", 48
            ),
            manual_roi_reacquire_patience=pose_config.get(
                "manual_roi_reacquire_patience", 6
            ),
            manual_roi_reacquire_frequency=pose_config.get(
                "manual_roi_reacquire_frequency", 15
            ),
        )

        # Store skeleton tree and keypoint names
        self._skeleton_tree = create_synthpose_skeleton()
        self._keypoint_names = list(SYNTHPOSE_KEYPOINT_NAMES)
        self._last_detections: Dict[str, np.ndarray] = {}

        # Log initialization with mode mapping info
        mode_info = f"config_mode='{mode}'"
        if synthpose_model_size:
            mode_info += f", synthpose_model_size='{synthpose_model_size}'"
        mode_info += f" → vitpose='{self._mode}'"
        logging.info(
            f"SynthPoseBackend initialized: {mode_info}, "
            f"detector={pose_config.get('synthpose_detector', 'yolox')}, "
            f"device={self._tracker.device}, keypoints=52"
        )

    def __call__(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Run pose estimation. Returns (keypoints, scores)."""
        keypoints, scores = self._tracker(frame)
        self._last_detections = getattr(self._tracker, "last_detections", {}) or {}
        return keypoints, scores

    def reset(self) -> None:
        """Reset tracker state."""
        if hasattr(self._tracker, "reset"):
            self._tracker.reset()
        else:
            self._tracker.frame_count = 0
            self._tracker.prev_boxes = None
        self._last_detections = {}

    def prepare_video_context(
        self, video_file_path=None, frame_range=None, input_kind="video"
    ) -> None:
        """Forward file-video context to the tracker when supported."""
        if hasattr(self._tracker, "prepare_video_context"):
            self._tracker.prepare_video_context(
                video_file_path=video_file_path,
                frame_range=frame_range,
                input_kind=input_kind,
            )

    @property
    def skeleton_tree(self):
        """Return 52-keypoint skeleton tree."""
        return self._skeleton_tree

    @property
    def num_keypoints(self) -> int:
        """Always 52 for SynthPose."""
        return 52

    @property
    def backend_name(self) -> str:
        return "synthpose"

    @property
    def keypoint_names(self) -> List[str]:
        """Return 52 SynthPose keypoint names."""
        return self._keypoint_names

    @property
    def last_detections(self) -> Dict[str, np.ndarray]:
        return self._last_detections


def create_pose_backend(config_dict: dict) -> PoseBackend:
    """
    Factory function to create pose backend from configuration.

    Args:
        config_dict: Full configuration dictionary with 'pose' section

    Returns:
        PoseBackend: Configured backend instance (RTMLibBackend, SynthPoseBackend, or Sapiens2Backend)

    Raises:
        ValueError: If pose_model is invalid
        ImportError: If SynthPose dependencies not installed

    Mode Parameter (RTMLib Compatibility):
        The 'mode' parameter works consistently across both backends:

        RTMLib:
            - 'performance': Highest quality ONNX models
            - 'balanced': Good balance of speed/accuracy
            - 'lightweight': Fastest, lower accuracy

        SynthPose:
            - 'performance': VitPose-huge (most accurate)
            - 'balanced': VitPose-base (good balance)
            - 'lightweight': NOT SUPPORTED → falls back to VitPose-base with warning

    Examples:
        # RTMLib backend (default)
        config = {'pose': {'pose_model': 'body_with_feet', 'mode': 'balanced'}}
        backend = create_pose_backend(config)

        # SynthPose backend with mode parameter
        config = {'pose': {'pose_model': 'synthpose', 'mode': 'performance'}}
        backend = create_pose_backend(config)  # Uses VitPose-huge

        # SynthPose backend with explicit model
        config = {'pose': {'pose_model': 'synthpose_base', 'device': 'cuda'}}
        backend = create_pose_backend(config)  # Uses VitPose-base
    """
    pose_config = config_dict.get("pose", {})
    pose_model = pose_config.get("pose_model", "body_with_feet").lower()

    if pose_model == "sapiens2":
        try:
            from Sports2D.Utilities.sapiens2_backend import Sapiens2Backend

            return Sapiens2Backend(config_dict)
        except ImportError as e:
            raise ImportError(
                f"Sapiens2 requires additional dependencies: {e}\n"
                "Install the local Sapiens2 checkout with: pip install -e ./sapiens2\n"
                "Then download pose and detector checkpoints under $SAPIENS_CHECKPOINT_ROOT,\n"
                "or set pose.sapiens2_root / pose.sapiens2_checkpoint / "
                "pose.sapiens2_detector_checkpoint in the config."
            ) from e
    if pose_model in ["synthpose", "synthpose_base"]:
        try:
            return SynthPoseBackend(config_dict)
        except ImportError as e:
            message = str(e)
            if message.startswith("Raw SAM3 checkpoints (.pt/.pth) require"):
                raise ImportError(
                    f"SynthPose requires additional dependencies: {message}\n"
                    "Raw SAM3 checkpoint mode is separate from sports2d[synthpose].\n"
                    "Install the official Meta sam3 package in the same environment,\n"
                    "or switch to sam3_runtime='transformers' with a Hugging Face SAM3 bundle.\n"
                    "Official SAM3 install docs currently target Python 3.12+, "
                    "PyTorch 2.7+, and CUDA 12.6 for the raw-checkpoint runtime."
                ) from e
            if message.startswith(
                "Ultralytics YOLO detectors require the 'ultralytics' package."
            ):
                raise ImportError(
                    f"SynthPose requires additional dependencies: {message}\n"
                    "Install with: pip install sports2d[synthpose,yolo26]\n"
                    "Or: pip install torch transformers ultralytics"
                ) from e
            if message.startswith("Hugging Face SAM3 runtime requires"):
                raise ImportError(
                    f"SynthPose requires additional dependencies: {message}\n"
                    "This means the current transformers install in this environment is too old for SAM3.\n"
                    "Upgrade transformers to a build that exposes Sam3Model/Sam3Processor.\n"
                    "If the latest stable release still does not expose SAM3, install transformers from source:\n"
                    "pip install git+https://github.com/huggingface/transformers"
                ) from e
            raise ImportError(
                f"SynthPose requires additional dependencies: {message}\n"
                "Install with: pip install sports2d[synthpose]\n"
                "Or: pip install torch transformers"
            ) from e
    else:
        return RTMLibBackend(config_dict)
