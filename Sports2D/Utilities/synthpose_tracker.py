#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Fix OpenMP runtime conflict: multiple copies of libiomp5md.dll
# Must be set BEFORE importing numpy, torch, or any library that uses OpenMP
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

'''
    ##################################################
    ## SynthPose Pose Tracker                       ##
    ##################################################
    
    SynthPose pose tracker using:
    - configurable person detection (YOLOX, YOLO26, RT-DETR, RT-DETRv4, or SAM3)
    - VitPose from HuggingFace Transformers for pose estimation (52 keypoints)
    
    This module provides:
    - SynthPosePoseTracker class with __call__(frame) interface
    - Full 52 SynthPose keypoints output (17 COCO + 35 anatomical markers)
    - Keeps detector metadata compatible with the Sports2D processing pipeline
'''

import numpy as np
import logging
from contextlib import contextmanager
from PIL import Image

from Sports2D.Utilities.manual_roi import (
    boxes_center_inside_roi,
    boxes_outside_rois,
    crop_frame_to_roi,
    expand_roi_with_context,
    expand_roi_xyxy,
    normalize_manual_roi_mode,
    normalize_roi_xyxy,
    offset_detection_meta_to_full_frame,
    offset_keypoints_to_full_frame,
    offset_xyxy_boxes_to_full_frame,
    roi_from_boxes_xyxy,
    translate_roi_to_local,
    union_rois,
)
from Sports2D.Utilities.sam3_detector import (
    BALL_ONLY_SAM3_PROMPTS,
    Sam3Detector,
    empty_sam3_detections,
)
from Sports2D.Utilities.sam3_video_detector import Sam3VideoDetector

PERSON_CLASS_ID = 0
SPORTS_BALL_CLASS_ID = 32
SUPPORTED_SYNTHPOSE_DETECTORS = (
    'yolox',
    'yolo26',
    'rtdetr',
    'rtdetrv4',
    'sam3',
)


def _resolve_rtmlib_yolox_device(pose_device: str, backend: str) -> str:
    if pose_device == 'cuda':
        return 'cuda'
    if pose_device == 'mps' and backend == 'onnxruntime':
        return 'mps'
    return 'cpu'


@contextmanager
def _temporary_yolox_coreml_provider(backend: str, device: str):
    if backend != 'onnxruntime' or device != 'mps':
        yield
        return

    try:
        from rtmlib.tools.base import RTMLIB_SETTINGS
    except Exception as exc:
        logging.debug('Could not configure SynthPose YOLOX CoreML provider: %s', exc)
        yield
        return

    onnxruntime_settings = RTMLIB_SETTINGS.get('onnxruntime', {})
    previous_provider = onnxruntime_settings.get('mps')
    onnxruntime_settings['mps'] = (
        'CoreMLExecutionProvider',
        {
            'ModelFormat': 'MLProgram',
            'RequireStaticInputShapes': '1',
        },
    )
    try:
        yield
    finally:
        onnxruntime_settings['mps'] = previous_provider

# Lazy imports to avoid loading heavy dependencies at module load time
_MODELS_LOADED = False
_torch = None
_AutoProcessor = None
_VitPoseForPoseEstimation = None
_RTDetrForObjectDetection = None


def _load_dependencies():
    """Lazily load torch and transformers dependencies."""
    global _MODELS_LOADED, _torch, _AutoProcessor, _VitPoseForPoseEstimation, _RTDetrForObjectDetection
    
    if _MODELS_LOADED:
        return
    
    try:
        import torch
        from transformers import (
            AutoProcessor,
            VitPoseForPoseEstimation,
            RTDetrForObjectDetection,
        )
        
        _torch = torch
        _AutoProcessor = AutoProcessor
        _VitPoseForPoseEstimation = VitPoseForPoseEstimation
        _RTDetrForObjectDetection = RTDetrForObjectDetection
        _MODELS_LOADED = True
        
    except ImportError as e:
        raise ImportError(
            "SynthPose requires PyTorch and HuggingFace Transformers. "
            "Install with: pip install torch transformers"
        ) from e


## AUTHORSHIP INFORMATION
__author__ = "Sports2D Contributors"
__copyright__ = "Copyright 2024, Sports2D"
__license__ = "BSD 3-Clause License"


def _normalize_synthpose_detector(detector, default='yolox'):
    '''
    Normalize and validate the SynthPose detector selection.
    '''
    normalized = str(default if detector is None else detector).strip().lower()
    if normalized in SUPPORTED_SYNTHPOSE_DETECTORS:
        return normalized

    supported = "', '".join(SUPPORTED_SYNTHPOSE_DETECTORS)
    raise ValueError(
        f"Unsupported synthpose_detector '{detector}'. "
        f"Expected one of: '{supported}'."
    )


def _normalize_ball_detector_backend(ball_detector_backend, detector=None, default='same'):
    '''
    Normalize the optional dedicated ball detector backend.

    - 'same' reuses the main detector
    - 'sam3' enables the dedicated SAM3 sports-ball detector
    - using the same detector name as the person detector is treated as 'same'
    '''
    normalized = str(default if ball_detector_backend is None else ball_detector_backend).strip().lower()
    detector_name = str(detector or '').strip().lower()
    if normalized in {'same', 'sam3'}:
        return normalized
    if detector_name and normalized == detector_name:
        return 'same'
    logging.warning(
        "Unsupported ball_detector_backend '%s'. Falling back to '%s'.",
        ball_detector_backend,
        default,
    )
    return default


def _normalize_sam3_inference_mode(sam3_inference_mode, default='image'):
    """Normalize the SAM3 inference-mode selector."""
    normalized = str(default if sam3_inference_mode is None else sam3_inference_mode).strip().lower()
    if normalized in {'image', 'video'}:
        return normalized
    logging.warning(
        "Unsupported sam3_inference_mode '%s'. Falling back to '%s'.",
        sam3_inference_mode,
        default,
    )
    return default


def _adaptive_person_mode_enabled(manual_roi_mode, manual_person_roi):
    """Return True when adaptive person ROI mode has a valid seed ROI."""
    return (
        str(manual_roi_mode or '').strip().lower() == 'adaptive_person'
        and manual_person_roi is not None
    )


class SynthPosePoseTracker:
    '''
    Pose tracker using rtmlib YOLOX + VitPose for 52-keypoint pose estimation.
    
    Usage:
        tracker = SynthPosePoseTracker(mode='huge', device='cuda')
        keypoints, scores = tracker(frame)  # frame is BGR numpy array
        
    The output format matches rtmlib:
        - keypoints: np.array shape (N_persons, 52, 2)
        - scores: np.array shape (N_persons, 52)
    '''
    
    def __init__(self,
                 mode='huge',
                 device='auto',
                 det_frequency=1,
                 person_threshold=0.3,
                 keypoint_threshold=0.3,
                 backend='auto',
                 detector='yolox',
                 detect_ball=False,
                 ball_class_ids=None,
                 ball_detection_threshold=0.1,
                 detector_size='balanced',
                 ball_nms_score_threshold=0.2,
                 sam3_target='ball',
                 sam3_model_path='',
                 sam3_processor_path='',
                 sam3_runtime='transformers',
                 sam3_store_masks=False,
                 sam3_show_realtime_masks=False,
                 sam3_save_ball_masks=False,
                 sam3_inference_mode='image',
                 sam3_bootstrap_frames=12,
                 sam3_video_refresh_frequency=4,
                 sam3_video_reseed_on_loss=True,
                 sam3_video_loss_patience=3,
                 ball_detector_backend='same',
                 manual_person_roi=None,
                 manual_ball_roi=None,
                 manual_ball_ignore_zones=None,
                 manual_roi_mode='bootstrap',
                 manual_roi_tracking_margin_px=48,
                 manual_roi_reacquire_patience=6,
                 manual_roi_reacquire_frequency=15):
        '''
        Initialize SynthPose tracker.

        INPUTS:
        - mode: 'huge' or 'base' - VitPose model size selection
                'huge' = stanfordmimi/synthpose-vitpose-huge-hf (more accurate, slower)
                'base' = stanfordmimi/synthpose-vitpose-base-hf (faster, less accurate)
        - device: Device for PyTorch inference
                  'auto': Auto-detect (CUDA > MPS > CPU)
                  'cuda': Force NVIDIA GPU (raises if unavailable)
                  'mps': Force Apple Metal (macOS only)
                  'cpu': Force CPU inference
                  Note: This differs from RTMLib which uses ONNX providers.
        - det_frequency: Run person detection every N frames (default 1)
        - person_threshold: Confidence threshold for person detection (default 0.3)
        - keypoint_threshold: Confidence threshold for keypoints (default 0.3)
        - backend: Backend for rtmlib YOLOX ('auto', 'onnxruntime', 'openvino', 'opencv')
                   Only used when detector='yolox'. Ignored for 'yolo26', 'rtdetr',
                   'rtdetrv4', and 'sam3'.
        - detector: Person detector selection
                    'yolox': rtmlib YOLOX (RECOMMENDED - fast, reliable)
                    'yolo26': Ultralytics YOLO26 detector
                    'rtdetr': HuggingFace RT-DETR (good accuracy, no local setup)
                    'rtdetrv4': Local RT-DETRv4 (requires engine installation)
                    'sam3': Promptable SAM3 detector (HF bundle or raw .pt checkpoint)
        '''
        
        # Load dependencies
        _load_dependencies()
        
        self.mode = mode.lower()
        self.det_frequency = max(1, det_frequency)
        self.person_threshold = person_threshold
        self.keypoint_threshold = keypoint_threshold
        self.backend = backend
        self.detector_type = _normalize_synthpose_detector(detector)
        self.detect_ball = bool(detect_ball)
        if ball_class_ids is None:
            self.ball_class_ids = [SPORTS_BALL_CLASS_ID]
        elif isinstance(ball_class_ids, (list, tuple, set)):
            self.ball_class_ids = [int(c) for c in ball_class_ids]
        else:
            self.ball_class_ids = [int(ball_class_ids)]
        self.sam3_target = str(sam3_target)
        self.sam3_model_path = str(sam3_model_path or '').strip()
        self.sam3_processor_path = str(sam3_processor_path or '').strip()
        self.sam3_runtime = str(sam3_runtime or 'transformers').strip().lower()
        self.sam3_store_masks = bool(sam3_store_masks)
        self.sam3_show_realtime_masks = bool(sam3_show_realtime_masks)
        self.sam3_save_ball_masks = bool(sam3_save_ball_masks)
        self.sam3_inference_mode = _normalize_sam3_inference_mode(sam3_inference_mode)
        self.sam3_bootstrap_frames = max(1, int(sam3_bootstrap_frames))
        self.sam3_video_refresh_frequency = max(1, int(sam3_video_refresh_frequency))
        self.sam3_video_reseed_on_loss = bool(sam3_video_reseed_on_loss)
        self.sam3_video_loss_patience = max(1, int(sam3_video_loss_patience))
        self.ball_detector_backend = _normalize_ball_detector_backend(
            ball_detector_backend,
            detector=detector,
        )
        self.manual_roi_mode = normalize_manual_roi_mode(manual_roi_mode)
        self.manual_person_roi = self._normalize_runtime_roi(manual_person_roi)
        self.manual_ball_roi = self._normalize_runtime_roi(manual_ball_roi)
        self.manual_ball_ignore_zones = self._normalize_runtime_rois(manual_ball_ignore_zones)
        self.manual_roi_tracking_margin_px = max(0, int(manual_roi_tracking_margin_px))
        self.manual_roi_reacquire_patience = max(1, int(manual_roi_reacquire_patience))
        self.manual_roi_reacquire_frequency = max(1, int(manual_roi_reacquire_frequency))
        self._manual_person_roi_released = False
        self._manual_ball_roi_released = False
        self.active_manual_person_roi = self.manual_person_roi
        self.person_roi_miss_count = 0
        self.last_full_frame_person_reacquire_frame = None
        self.sam3_collect_masks = bool(
            self.sam3_store_masks or self.sam3_show_realtime_masks or self.sam3_save_ball_masks
        )
        self.last_detections = self._empty_detections()
        self.sam3_detector = None
        self.sam3_ball_detector = None
        self.video_input_kind = 'video'
        self.video_file_path = None
        self.video_frame_index_offset = 0
        self.detector = None
        self.detector_size = self._resolve_detector_size(
            detector_size,
            detector=self.detector_type,
        )
        if self.detector_type == 'rtdetrv4':
            self.rtdetrv4_size = self.detector_size if self.detector_size in {'s', 'm', 'l', 'x'} else 'x'
        self.ball_detection_threshold = float(np.clip(ball_detection_threshold, 0.01, 0.9))
        self.ball_nms_score_threshold = float(np.clip(ball_nms_score_threshold, 0.01, 0.9))
        
        # Frame tracking for detection frequency
        self.frame_count = 0
        self.prev_boxes = None
        self._warned_ball_detection_cadence = False
        self._warned_shared_union_roi = False
        
        # Set device for VitPose (PyTorch)
        if device == 'auto':
            if _torch.cuda.is_available():
                self.device = 'cuda'
            elif hasattr(_torch.backends, 'mps') and _torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        else:
            self.device = device
        
        logging.info(
            'SynthPose initializing: hf_model_size=%s (VitPose), '
            'detector_type=%s, detector_size=%s (mode-driven), device=%s, '
            'person_thr=%.3f, ball_thr=%.3f, sam3_target=%s, ball_detector_backend=%s',
            self.mode,
            self.detector_type,
            self.detector_size,
            self.device,
            self.person_threshold,
            self.ball_detection_threshold,
            self.sam3_target,
            self.ball_detector_backend,
        )
        if (
            self.detect_ball
            and self.det_frequency > 1
            and not self._uses_secondary_sam3_ball_detector()
        ):
            logging.warning(
                'detect_ball=true with det_frequency=%s reuses sparse detector metadata on shared-detector paths. '
                'Ball continuity may degrade between detection frames. Use det_frequency=1 or ball_detector_backend=\'sam3\' '
                'for stronger ball-ID stability.',
                self.det_frequency,
            )
            self._warned_ball_detection_cadence = True
        if self.sam3_inference_mode == 'video' and not self._uses_secondary_sam3_ball_detector():
            logging.warning(
                "sam3_inference_mode='video' is currently supported only for the hybrid "
                "ball path (ball_detector_backend='sam3'). Falling back to image mode."
            )
            self.sam3_inference_mode = 'image'
        if self.manual_roi_mode == 'adaptive_person' and self.manual_person_roi is None:
            logging.warning(
                "manual_roi_mode='adaptive_person' requires a manual person ROI. Falling back to bootstrap."
            )
            self.manual_roi_mode = 'bootstrap'
        if self.manual_roi_mode == 'adaptive_person' and self.detect_ball and not self._uses_secondary_sam3_ball_detector():
            logging.warning(
                "manual_roi_mode='adaptive_person' is not supported with "
                "detect_ball=true and ball_detector_backend='same'. Falling back to bootstrap."
            )
            self.manual_roi_mode = 'bootstrap'
        self._reset_manual_roi_runtime_state()
        
        # Load models
        self._load_models()
        
        logging.info(f'SynthPose ready. Output: 52 keypoints (17 COCO + 35 anatomical markers)')

    def _reset_manual_roi_runtime_state(self):
        """Reset mutable ROI runtime state for bootstrap/adaptive modes."""
        self.active_manual_person_roi = self.manual_person_roi
        self.person_roi_miss_count = 0
        self.last_full_frame_person_reacquire_frame = None

    def _adaptive_person_mode_enabled(self):
        """Return True when adaptive person ROI mode is active."""
        return _adaptive_person_mode_enabled(
            getattr(self, 'manual_roi_mode', 'bootstrap'),
            getattr(self, 'manual_person_roi', None),
        )

    def _should_force_full_frame_person_reacquire(self):
        """Return True when adaptive person ROI should bypass local crop for one detection pass."""
        if not self._adaptive_person_mode_enabled():
            return False
        if getattr(self, 'person_roi_miss_count', 0) < getattr(self, 'manual_roi_reacquire_patience', 6):
            return False
        if getattr(self, 'last_full_frame_person_reacquire_frame', None) is None:
            return True
        return (
            (int(self.frame_count) - int(self.last_full_frame_person_reacquire_frame))
            >= getattr(self, 'manual_roi_reacquire_frequency', 15)
        )

    def _update_adaptive_person_roi(self, person_boxes_xyxy, frame_shape):
        """Update the active person ROI from accepted full-frame person boxes."""
        if not self._adaptive_person_mode_enabled():
            return
        updated_roi = roi_from_boxes_xyxy(
            person_boxes_xyxy,
            frame_shape,
            padding_px=getattr(self, 'manual_roi_tracking_margin_px', 48),
        )
        if updated_roi is not None:
            self.active_manual_person_roi = updated_roi
        self.person_roi_miss_count = 0

    def _mark_adaptive_person_roi_miss(self, frame_shape):
        """Expand the active ROI after a local adaptive-person miss."""
        if not self._adaptive_person_mode_enabled():
            return
        self.person_roi_miss_count += 1
        expanded_roi = expand_roi_xyxy(
            self.active_manual_person_roi,
            frame_shape,
            padding_px=getattr(self, 'manual_roi_tracking_margin_px', 48),
        )
        if expanded_roi is not None:
            self.active_manual_person_roi = expanded_roi

    def prepare_video_context(self, *, video_file_path=None, frame_range=None, input_kind='video'):
        """Store file-video context for video-aware SAM3 modes."""
        self.video_input_kind = str(input_kind or 'video').strip().lower()
        self.video_file_path = str(video_file_path) if video_file_path is not None else None
        if isinstance(frame_range, (list, tuple)) and len(frame_range) >= 1:
            self.video_frame_index_offset = int(frame_range[0])
        else:
            self.video_frame_index_offset = 0
        self._reset_manual_roi_runtime_state()
        if self.sam3_ball_detector is not None and hasattr(self.sam3_ball_detector, 'prepare_video_context'):
            self.sam3_ball_detector.prepare_video_context(
                video_file_path=self.video_file_path,
                frame_index_offset=self.video_frame_index_offset,
                input_kind=self.video_input_kind,
            )

    def reset(self):
        """Reset tracker runtime state for a new video session."""
        self.frame_count = 0
        self.prev_boxes = None
        self.last_detections = self._empty_detections()
        self._manual_person_roi_released = False
        self._manual_ball_roi_released = False
        self._reset_manual_roi_runtime_state()
        if self.sam3_ball_detector is not None and hasattr(self.sam3_ball_detector, 'close'):
            self.sam3_ball_detector.close()

    @staticmethod
    def _resolve_detector_size(size_value, detector='yolox'):
        """
        Resolve detector size from explicit size or mode alias.

        Accepted values:
        - YOLOX / RT-DETRv4: performance -> x, balanced -> m, lightweight -> s
        - YOLO26: performance -> x, balanced -> m, lightweight -> n
        """
        detector_name = str(detector or 'yolox').strip().lower()
        if detector_name == 'yolo26':
            mode_to_size = {
                'performance': 'x',
                'balanced': 'm',
                'lightweight': 'n',
            }
            explicit_sizes = {'x', 'l', 'm', 's', 'n'}
            tiny_alias = 'n'
            default_size = 'm'
        else:
            mode_to_size = {
                'performance': 'x',
                'balanced': 'm',
                'lightweight': 's',
            }
            explicit_sizes = {'x', 'l', 'm', 's'}
            tiny_alias = 's'
            default_size = 'm'

        value = str(size_value).lower()
        if value in mode_to_size:
            return mode_to_size[value]
        if value == 'tiny':
            logging.warning(
                "synthpose detector size 'tiny' is deprecated. Using '%s'.",
                tiny_alias,
            )
            return tiny_alias
        if value in explicit_sizes:
            return value

        logging.warning(
            "Unknown synthpose detector size '%s' for detector '%s'. Using '%s'.",
            size_value,
            detector_name,
            default_size,
        )
        return default_size

    @staticmethod
    def _normalize_runtime_roi(roi):
        """Normalize a runtime ROI payload to an xyxy integer tuple."""
        if roi is None:
            return None
        arr = np.asarray(roi, dtype=np.float32).reshape(-1)
        if arr.size != 4 or not np.all(np.isfinite(arr)):
            return None
        return tuple(int(round(v)) for v in arr.tolist())

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

    @staticmethod
    def _coco_to_xyxy_boxes(boxes_coco):
        """Convert COCO xywh boxes to xyxy boxes."""
        boxes_coco = np.asarray(boxes_coco, dtype=np.float32).reshape(-1, 4)
        if len(boxes_coco) == 0:
            return np.empty((0, 4), dtype=np.float32)
        boxes_xyxy = np.zeros((len(boxes_coco), 4), dtype=np.float32)
        boxes_xyxy[:, 0] = boxes_coco[:, 0]
        boxes_xyxy[:, 1] = boxes_coco[:, 1]
        boxes_xyxy[:, 2] = boxes_coco[:, 0] + boxes_coco[:, 2]
        boxes_xyxy[:, 3] = boxes_coco[:, 1] + boxes_coco[:, 3]
        return boxes_xyxy

    def _resolve_manual_rois(self, frame_shape):
        """Resolve configured static ROIs against the current frame shape."""
        if self._adaptive_person_mode_enabled():
            person_roi = None
            if not self._should_force_full_frame_person_reacquire():
                person_roi = normalize_roi_xyxy(
                    getattr(self, 'active_manual_person_roi', None),
                    frame_shape,
                )

            ball_roi = None
            if not getattr(self, '_manual_ball_roi_released', False):
                ball_roi = normalize_roi_xyxy(getattr(self, 'manual_ball_roi', None), frame_shape)
            if self.detect_ball and ball_roi is None and not getattr(self, '_manual_ball_roi_released', False):
                ball_roi = person_roi
            return person_roi, ball_roi

        if (
            not self._uses_secondary_sam3_ball_detector()
            and (
                getattr(self, '_manual_person_roi_released', False)
                or getattr(self, '_manual_ball_roi_released', False)
            )
        ):
            return None, None

        person_roi = None
        if not getattr(self, '_manual_person_roi_released', False):
            person_roi = normalize_roi_xyxy(getattr(self, 'manual_person_roi', None), frame_shape)

        ball_roi = None
        if not getattr(self, '_manual_ball_roi_released', False):
            ball_roi = normalize_roi_xyxy(getattr(self, 'manual_ball_roi', None), frame_shape)

        if self.detect_ball and ball_roi is None and not getattr(self, '_manual_ball_roi_released', False):
            ball_roi = person_roi
        return person_roi, ball_roi

    def _offset_coco_boxes_to_full_frame(self, boxes_coco, roi):
        """Translate COCO xywh boxes from ROI-local to full-frame coordinates."""
        boxes_xyxy = self._coco_to_xyxy_boxes(boxes_coco)
        return self._xyxy_to_coco_boxes(offset_xyxy_boxes_to_full_frame(boxes_xyxy, roi))

    def _offset_coco_boxes_to_local_frame(self, boxes_coco, roi):
        """Translate COCO xywh boxes from full-frame coordinates into ROI-local coordinates."""
        boxes_xyxy = self._coco_to_xyxy_boxes(boxes_coco)
        if len(boxes_xyxy) == 0 or roi is None:
            return boxes_coco
        local_boxes = boxes_xyxy.copy()
        x1, y1, _, _ = [float(v) for v in roi]
        local_boxes[:, [0, 2]] -= x1
        local_boxes[:, [1, 3]] -= y1
        return self._xyxy_to_coco_boxes(local_boxes)

    def _filter_primary_detections_to_rois(self, person_boxes, person_roi=None, ball_roi=None):
        """Restrict primary detector outputs to the manually selected ROI subsets."""
        meta = dict(self.last_detections or self._empty_detections())

        if person_roi is not None:
            person_boxes_xyxy = self._coco_to_xyxy_boxes(person_boxes)
            person_mask = boxes_center_inside_roi(person_boxes_xyxy, person_roi)
            person_boxes_xyxy = person_boxes_xyxy[person_mask]
            person_boxes = self._xyxy_to_coco_boxes(person_boxes_xyxy)
            meta['person_boxes'] = person_boxes_xyxy

        if ball_roi is not None:
            ball_boxes = self._ensure_xyxy_boxes(meta.get('ball_boxes'))
            ball_mask = boxes_center_inside_roi(ball_boxes, ball_roi)
            meta['ball_boxes'] = ball_boxes[ball_mask]
            ball_scores = self._tensor_like_to_numpy(meta.get('ball_scores'), dtype=np.float32).reshape(-1)
            if len(ball_scores) == len(ball_boxes):
                meta['ball_scores'] = ball_scores[ball_mask]
            else:
                meta['ball_scores'] = np.full((np.count_nonzero(ball_mask),), np.nan, dtype=np.float32)

        self.last_detections = meta
        return person_boxes

    def _filter_ball_detection_meta_to_roi(self, ball_meta, ball_roi=None):
        """Restrict secondary ball detections to the user-selected ball ROI."""
        if ball_roi is None:
            return ball_meta or self._empty_detections()

        meta = dict(ball_meta or self._empty_detections())
        boxes = self._ensure_xyxy_boxes(meta.get('boxes'))
        if len(boxes) == 0:
            return meta

        keep_mask = boxes_center_inside_roi(boxes, ball_roi)
        meta['boxes'] = boxes[keep_mask]

        classes = self._tensor_like_to_numpy(meta.get('classes'), dtype=np.int32).reshape(-1)
        if len(classes) == len(boxes):
            meta['classes'] = classes[keep_mask]
        scores = self._tensor_like_to_numpy(meta.get('scores'), dtype=np.float32).reshape(-1)
        if len(scores) == len(boxes):
            meta['scores'] = scores[keep_mask]
        class_names = np.asarray(meta.get('class_names', np.empty((0,), dtype=object)), dtype=object).reshape(-1)
        if len(class_names) == len(boxes):
            meta['class_names'] = class_names[keep_mask]
        prompt_indices = self._tensor_like_to_numpy(meta.get('prompt_indices'), dtype=np.int32).reshape(-1)
        if len(prompt_indices) == len(boxes):
            meta['prompt_indices'] = prompt_indices[keep_mask]

        ball_boxes = self._ensure_xyxy_boxes(meta.get('ball_boxes'))
        ball_keep_mask = boxes_center_inside_roi(ball_boxes, ball_roi)
        meta['ball_boxes'] = ball_boxes[ball_keep_mask]
        ball_scores = self._tensor_like_to_numpy(meta.get('ball_scores'), dtype=np.float32).reshape(-1)
        if len(ball_scores) == len(ball_boxes):
            meta['ball_scores'] = ball_scores[ball_keep_mask]
        return meta

    @staticmethod
    def _filter_mask_entries(masks, keep_mask, expected_len):
        """Filter optional instance masks when they stay aligned with detector boxes."""
        if masks is None:
            return None
        try:
            mask_array = np.asarray(masks)
        except Exception:
            mask_array = None
        if mask_array is not None and mask_array.size > 0:
            if mask_array.ndim == 2:
                mask_values = [mask_array]
            elif mask_array.ndim >= 3:
                mask_values = [mask_array[i] for i in range(mask_array.shape[0])]
            else:
                mask_values = list(masks)
        else:
            mask_values = list(masks) if masks is not None else []
        if len(mask_values) != int(expected_len):
            return masks
        return [mask_values[i] for i, keep in enumerate(np.asarray(keep_mask, dtype=bool)) if keep]

    def _apply_ball_ignore_zones_to_detection_meta(self, detection_meta):
        """Remove ball detections whose boxes overlap configured ignore zones."""
        ignore_zones = getattr(self, 'manual_ball_ignore_zones', None) or []
        if len(ignore_zones) == 0:
            return detection_meta or self._empty_detections()

        meta = dict(detection_meta or self._empty_detections())
        boxes = self._ensure_xyxy_boxes(meta.get('boxes'))
        classes = self._tensor_like_to_numpy(meta.get('classes'), dtype=np.int32).reshape(-1)
        if len(boxes) > 0 and len(classes) == len(boxes):
            ball_mask = np.isin(classes, self.ball_class_ids)
            if np.any(ball_mask):
                keep_mask = np.ones((len(boxes),), dtype=bool)
                keep_mask[ball_mask] = boxes_outside_rois(boxes[ball_mask], ignore_zones)
                meta['boxes'] = boxes[keep_mask]
                meta['classes'] = classes[keep_mask]

                scores = self._tensor_like_to_numpy(meta.get('scores'), dtype=np.float32).reshape(-1)
                if len(scores) == len(boxes):
                    meta['scores'] = scores[keep_mask]
                class_names = np.asarray(
                    meta.get('class_names', np.empty((0,), dtype=object)),
                    dtype=object,
                ).reshape(-1)
                if len(class_names) == len(boxes):
                    meta['class_names'] = class_names[keep_mask]
                prompt_indices = self._tensor_like_to_numpy(
                    meta.get('prompt_indices'),
                    dtype=np.int32,
                ).reshape(-1)
                if len(prompt_indices) == len(boxes):
                    meta['prompt_indices'] = prompt_indices[keep_mask]
                filtered_masks = self._filter_mask_entries(
                    meta.get('masks'),
                    keep_mask,
                    expected_len=len(boxes),
                )
                if filtered_masks is not None:
                    meta['masks'] = filtered_masks

        ball_boxes = self._ensure_xyxy_boxes(meta.get('ball_boxes'))
        if len(ball_boxes) == 0:
            return meta

        ball_keep_mask = boxes_outside_rois(ball_boxes, ignore_zones)
        meta['ball_boxes'] = ball_boxes[ball_keep_mask]
        ball_scores = self._tensor_like_to_numpy(meta.get('ball_scores'), dtype=np.float32).reshape(-1)
        if len(ball_scores) == len(ball_boxes):
            meta['ball_scores'] = ball_scores[ball_keep_mask]
        else:
            meta['ball_scores'] = np.full((np.count_nonzero(ball_keep_mask),), np.nan, dtype=np.float32)
        return meta

    @staticmethod
    def _tensor_like_to_numpy(value, dtype=None):
        '''
        Convert torch / ultralytics tensor-like objects to numpy arrays.
        '''
        if value is None:
            return np.empty((0,), dtype=dtype if dtype is not None else np.float32)

        if hasattr(value, 'detach'):
            value = value.detach()
        if hasattr(value, 'cpu'):
            value = value.cpu()
        if hasattr(value, 'numpy'):
            try:
                value = value.numpy()
            except TypeError:
                pass

        array = np.asarray(value)
        if dtype is not None:
            array = array.astype(dtype, copy=False)
        return array

    @classmethod
    def _ensure_xyxy_boxes(cls, boxes):
        '''
        Normalize detector boxes to Nx4 xyxy float32.
        '''
        box_array = cls._tensor_like_to_numpy(boxes, dtype=np.float32)
        if box_array.size == 0:
            return np.empty((0, 4), dtype=np.float32)
        if box_array.ndim == 1:
            box_array = box_array.reshape(1, -1)
        if box_array.shape[1] < 4:
            return np.empty((0, 4), dtype=np.float32)
        return box_array[:, :4].astype(np.float32, copy=False)

    @staticmethod
    def _xyxy_to_coco_boxes(boxes_xyxy):
        '''
        Convert xyxy boxes to COCO xywh format for VitPose.
        '''
        boxes_xyxy = np.asarray(boxes_xyxy, dtype=np.float32).reshape(-1, 4)
        if len(boxes_xyxy) == 0:
            return np.empty((0, 4), dtype=np.float32)
        boxes_coco = np.zeros((len(boxes_xyxy), 4), dtype=np.float32)
        boxes_coco[:, 0] = boxes_xyxy[:, 0]
        boxes_coco[:, 1] = boxes_xyxy[:, 1]
        boxes_coco[:, 2] = boxes_xyxy[:, 2] - boxes_xyxy[:, 0]
        boxes_coco[:, 3] = boxes_xyxy[:, 3] - boxes_xyxy[:, 1]
        return boxes_coco

    @staticmethod
    def _resolve_class_names(classes, names):
        '''
        Resolve per-box class names from a detector label lookup.
        '''
        classes = np.asarray(classes, dtype=np.int32).reshape(-1)
        if len(classes) == 0 or names is None:
            return np.empty((0,), dtype=object)

        resolved = []
        if isinstance(names, dict):
            for class_id in classes:
                resolved.append(str(names.get(int(class_id), int(class_id))))
        else:
            try:
                name_list = list(names)
            except TypeError:
                return np.empty((0,), dtype=object)
            for class_id in classes:
                idx = int(class_id)
                if 0 <= idx < len(name_list):
                    resolved.append(str(name_list[idx]))
                else:
                    resolved.append(str(idx))
        return np.asarray(resolved, dtype=object)

    def _set_last_detections(self, boxes, classes=None, scores=None,
                             class_names=None, prompt_indices=None,
                             metadata_score_threshold=None,
                             person_score_threshold=None):
        '''
        Normalize detector outputs into the shared metadata contract.
        '''
        boxes = self._ensure_xyxy_boxes(boxes)
        if len(boxes) == 0:
            self.last_detections = self._empty_detections()
            return np.empty((0, 4), dtype=np.float32)

        classes = self._tensor_like_to_numpy(classes, dtype=np.int32).reshape(-1)
        if len(classes) != len(boxes):
            classes = np.full((len(boxes),), PERSON_CLASS_ID, dtype=np.int32)

        scores = self._tensor_like_to_numpy(scores, dtype=np.float32).reshape(-1)
        if len(scores) == 0:
            scores = np.full((len(boxes),), np.nan, dtype=np.float32)
        elif len(scores) != len(boxes):
            padded_scores = np.full((len(boxes),), np.nan, dtype=np.float32)
            limit = min(len(scores), len(boxes))
            padded_scores[:limit] = scores[:limit]
            scores = padded_scores

        class_names = np.asarray(class_names, dtype=object).reshape(-1) if class_names is not None else np.empty((0,), dtype=object)
        if len(class_names) not in {0, len(boxes)}:
            class_names = np.empty((0,), dtype=object)

        prompt_indices = (
            self._tensor_like_to_numpy(prompt_indices, dtype=np.int32).reshape(-1)
            if prompt_indices is not None else np.empty((0,), dtype=np.int32)
        )
        if len(prompt_indices) not in {0, len(boxes)}:
            prompt_indices = np.empty((0,), dtype=np.int32)

        if metadata_score_threshold is not None and len(scores) == len(boxes):
            meta_mask = np.ones((len(scores),), dtype=bool)
            finite_score_mask = np.isfinite(scores)
            meta_mask[finite_score_mask] = scores[finite_score_mask] >= float(metadata_score_threshold)
            boxes = boxes[meta_mask]
            classes = classes[meta_mask]
            scores = scores[meta_mask]
            if len(class_names) == len(meta_mask):
                class_names = class_names[meta_mask]
            if len(prompt_indices) == len(meta_mask):
                prompt_indices = prompt_indices[meta_mask]
            if len(boxes) == 0:
                self.last_detections = self._empty_detections()
                return np.empty((0, 4), dtype=np.float32)

        person_mask = classes == PERSON_CLASS_ID
        if person_score_threshold is not None and len(scores) == len(boxes):
            finite_score_mask = np.isfinite(scores)
            person_mask = person_mask & (
                ~finite_score_mask | (scores >= float(person_score_threshold))
            )
        person_boxes_xyxy = boxes[person_mask]
        if len(person_boxes_xyxy) == 0 and len(boxes) > 0:
            unique_classes = np.unique(classes)
            if len(unique_classes) == 1 and int(unique_classes[0]) == PERSON_CLASS_ID:
                if person_score_threshold is None or np.any(scores >= float(person_score_threshold)):
                    fallback_mask = np.ones((len(boxes),), dtype=bool)
                    if person_score_threshold is not None:
                        fallback_mask = scores >= float(person_score_threshold)
                    person_boxes_xyxy = boxes[fallback_mask]

        if self._primary_detector_handles_ball():
            ball_mask = np.isin(classes, self.ball_class_ids)
            if len(scores) == len(boxes):
                finite_score_mask = np.isfinite(scores)
                ball_mask = ball_mask & (
                    ~finite_score_mask | (scores >= self.ball_detection_threshold)
                )
            ball_boxes_xyxy = boxes[ball_mask]
            ball_scores_xyxy = scores[ball_mask]
        else:
            ball_boxes_xyxy = np.empty((0, 4), dtype=np.float32)
            ball_scores_xyxy = np.empty((0,), dtype=np.float32)

        self.last_detections = {
            'boxes': boxes,
            'classes': classes,
            'scores': scores,
            'person_boxes': person_boxes_xyxy,
            'ball_boxes': ball_boxes_xyxy,
            'ball_scores': ball_scores_xyxy,
            'class_names': class_names,
            'prompt_indices': prompt_indices,
            'sam3_ball_meta': {},
        }
        return self._xyxy_to_coco_boxes(person_boxes_xyxy)

    def _empty_detections(self):
        empty = empty_sam3_detections(store_masks=self.sam3_collect_masks)
        empty['sam3_ball_meta'] = {}
        return empty

    def _uses_secondary_sam3_ball_detector(self):
        return (
            self.detect_ball
            and self.ball_detector_backend == 'sam3'
            and self.detector_type != 'sam3'
        )

    def _primary_detector_handles_ball(self):
        return self.detect_ball and not self._uses_secondary_sam3_ball_detector()
    
    def _load_models(self):
        '''Load person detector (YOLOX, RT-DETR, RT-DETRv4, or SAM3) and VitPose model.'''

        loader_map = {
            'sam3': (
                'Loading SAM3 detector (HF bundle or raw .pt checkpoint)...',
                self._load_sam3_detector,
            ),
            'rtdetrv4': (
                'Loading RT-DETRv4 person detector (local checkpoint)...',
                self._load_rtdetrv4_detector,
            ),
            'rtdetr': (
                'Loading RT-DETR person detector (HuggingFace Transformers, original SynthPose_PM)...',
                self._load_rtdetr_detector,
            ),
            'yolox': (
                'Loading rtmlib YOLOX person detector...',
                self._load_rtmlib_detector,
            ),
            'yolo26': (
                'Loading Ultralytics YOLO26 person detector...',
                self._load_ultralytics_detector,
            ),
        }
        try:
            message, loader = loader_map[self.detector_type]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported synthpose_detector '{self.detector_type}'."
            ) from exc
        logging.info(message)
        loader()

        if self._uses_secondary_sam3_ball_detector():
            logging.info('Loading secondary SAM3 sports-ball detector...')
            self._load_sam3_ball_detector()
        
        # Pose estimator: VitPose from HuggingFace
        if self.mode == 'huge':
            model_name = "stanfordmimi/synthpose-vitpose-huge-hf"
        elif self.mode == 'base':
            model_name = "stanfordmimi/synthpose-vitpose-base-hf"
        else:
            raise ValueError(f"Unknown mode '{self.mode}'. Use 'huge' or 'base'.")
        
        logging.info(f'Loading VitPose model: {model_name}...')
        self.pose_processor = _AutoProcessor.from_pretrained(model_name)
        self.pose_model = _VitPoseForPoseEstimation.from_pretrained(model_name).to(self.device)
        self.pose_model.eval()
        
        logging.info('SynthPose models loaded successfully.')

    def _load_sam3_detector(self):
        '''Load SAM3 detector from a raw checkpoint or HuggingFace-compatible repository.'''
        self.sam3_detector = Sam3Detector(
            model_path=self.sam3_model_path,
            processor_path=self.sam3_processor_path,
            runtime=self.sam3_runtime,
            device=self.device,
            target=self.sam3_target,
            store_masks=self.sam3_collect_masks,
            person_threshold=self.person_threshold,
            ball_detection_threshold=self.ball_detection_threshold,
        )
        logging.info(
            "SAM3 detector loaded (target=%s, runtime=%s, prompts=%s)",
            self.sam3_detector.target,
            self.sam3_detector.runtime,
            self.sam3_detector.prompts,
        )

    def _load_sam3_ball_detector(self):
        '''Load a SAM3 detector dedicated to sports-ball prompts for hybrid person/ball mode.'''
        if self.sam3_inference_mode == 'video':
            self.sam3_ball_detector = Sam3VideoDetector(
                model_path=self.sam3_model_path,
                processor_path=self.sam3_processor_path,
                runtime=self.sam3_runtime,
                device=self.device,
                prompts=BALL_ONLY_SAM3_PROMPTS,
                store_masks=self.sam3_collect_masks,
                person_threshold=self.person_threshold,
                ball_detection_threshold=self.ball_detection_threshold,
                bootstrap_frames=self.sam3_bootstrap_frames,
                refresh_frequency=self.sam3_video_refresh_frequency,
                reseed_on_loss=self.sam3_video_reseed_on_loss,
                loss_patience=self.sam3_video_loss_patience,
            )
            self.sam3_ball_detector.prepare_video_context(
                video_file_path=self.video_file_path,
                frame_index_offset=self.video_frame_index_offset,
                input_kind=self.video_input_kind,
            )
            logging.info(
                "SAM3.1 sports-ball video detector loaded (runtime=%s, prompts=%s, bootstrap_frames=%s)",
                self.sam3_ball_detector.runtime,
                self.sam3_ball_detector.prompts,
                self.sam3_bootstrap_frames,
            )
            return

        self.sam3_ball_detector = Sam3Detector(
            model_path=self.sam3_model_path,
            processor_path=self.sam3_processor_path,
            runtime=self.sam3_runtime,
            device=self.device,
            target='ball',
            prompts=BALL_ONLY_SAM3_PROMPTS,
            store_masks=self.sam3_collect_masks,
            person_threshold=self.person_threshold,
            ball_detection_threshold=self.ball_detection_threshold,
        )
        logging.info(
            "SAM3 sports-ball detector loaded (runtime=%s, prompts=%s)",
            self.sam3_ball_detector.runtime,
            self.sam3_ball_detector.prompts,
        )
    
    def _load_rtmlib_detector(self):
        '''Load rtmlib YOLOX detector.'''
        from rtmlib import YOLOX

        # HumanArt detector is person-only. For detect_ball, use COCO YOLOX.
        if self._primary_detector_handles_ball():
            coco_yolox_models = {
                's': 'https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.onnx',
                'm': 'https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_m.onnx',
                'l': 'https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_l.onnx',
                'x': 'https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_x.onnx',
            }
            model_url = coco_yolox_models.get(self.detector_size, coco_yolox_models['m'])
            input_size = (640, 640)
            score_thr = self.ball_detection_threshold
            nms_thr = self.ball_nms_score_threshold
            detector_mode = 'multiclass'
            weights_variant = 'coco'
            weights_size = self.detector_size
        else:
            humanart_size_map = {
                's': 'm',
                'm': 'm',
                'l': 'm',
                'x': 'x',
            }
            resolved_humanart_size = humanart_size_map.get(self.detector_size, 'm')
            humanart_yolox_models = {
                'm': (
                    'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/'
                    'yolox_m_8xb8-300e_humanart-c2c7a14a.zip',
                    (640, 640),
                ),
                'l': (
                    'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/'
                    'yolox_m_8xb8-300e_humanart-c2c7a14a.zip',
                    (640, 640),
                ),
                'x': (
                    'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/'
                    'yolox_x_8xb8-300e_humanart-a39d44ed.zip',
                    (640, 640),
                ),
            }
            model_url, input_size = humanart_yolox_models[resolved_humanart_size]
            score_thr = self.person_threshold
            nms_thr = 0.45
            detector_mode = 'human'
            weights_variant = 'humanart'
            weights_size = resolved_humanart_size
        
        # Determine backend and device for rtmlib
        rtmlib_backend = self.backend if self.backend != 'auto' else 'onnxruntime'
        rtmlib_device = _resolve_rtmlib_yolox_device(self.device, rtmlib_backend)

        # rtmlib YOLOX API: onnx_model, model_input_size, nms_thr, score_thr, backend, device
        with _temporary_yolox_coreml_provider(rtmlib_backend, rtmlib_device):
            self.detector = YOLOX(
                onnx_model=model_url,
                model_input_size=input_size,
                mode=detector_mode,
                nms_thr=nms_thr,
                score_thr=score_thr,
                backend=rtmlib_backend,
                device=rtmlib_device
            )

        logging.info(
            'rtmlib YOLOX detector loaded (mode=%s, weights=%s/%s, '
            'requested_size=%s, nms_thr=%s, score_thr=%s, backend=%s)',
            detector_mode,
            weights_variant,
            weights_size,
            self.detector_size,
            nms_thr,
            score_thr,
            rtmlib_backend,
        )

    def _load_ultralytics_detector(self):
        '''Load Ultralytics YOLO26 detector lazily.'''
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                "Ultralytics YOLO detectors require the 'ultralytics' package."
            ) from e

        model_name = f'yolo26{self.detector_size}.pt'
        self.detector = YOLO(model_name)
        if hasattr(self.detector, 'to'):
            try:
                self.detector.to(self.device)
            except Exception as e:
                logging.debug(
                    'Could not move Ultralytics detector to device %s: %s',
                    self.device,
                    e,
                )
        logging.info(
            'Ultralytics YOLO26 detector loaded (weights=%s, device=%s)',
            model_name,
            self.device,
        )
    
    def _load_rtdetr_detector(self):
        '''Load RT-DETR detector from HuggingFace Transformers (original SynthPose_PM detector).'''
        
        # RT-DETR model from HuggingFace (same as SynthPose_PM)
        rtdetr_model_name = "PekingU/rtdetr_r50vd_coco_o365"
        
        logging.info(f'Loading RT-DETR model: {rtdetr_model_name}...')
        self.rtdetr_processor = _AutoProcessor.from_pretrained(rtdetr_model_name)
        self.rtdetr_model = _RTDetrForObjectDetection.from_pretrained(rtdetr_model_name).to(self.device)
        self.rtdetr_model.eval()
        
        logging.info(f'RT-DETR detector loaded (PekingU/rtdetr_r50vd_coco_o365)')
    
    def _load_rtdetrv4_detector(self):
        '''Load RT-DETRv4 detector from local checkpoint.'''
        import os
        import sys
        
        # Find the models directory
        models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'rtdetrv4')
        
        # Try to find checkpoint file based on rtdetrv4_size attribute (default: xlarge)
        size = getattr(self, 'rtdetrv4_size', 'x')  # 's', 'm', 'l', 'x'
        checkpoint_names = [
            f'rtdetrv4_{size}.pth',
            f'rtv4_{size}.pth',
            f'rtdetrv4-{size}.pth',
        ]
        
        checkpoint_path = None
        for name in checkpoint_names:
            path = os.path.join(models_dir, name)
            if os.path.exists(path):
                checkpoint_path = path
                break
        
        # Also check for any .pth file if specific name not found
        if checkpoint_path is None:
            pth_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')] if os.path.exists(models_dir) else []
            if pth_files:
                checkpoint_path = os.path.join(models_dir, pth_files[0])
                logging.info(f'Using first available checkpoint: {pth_files[0]}')
        
        if checkpoint_path is None or not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"RT-DETRv4 checkpoint not found in {models_dir}. "
                f"Please download from https://drive.google.com/file/d/19gnkMTgFveJsrOvSmEPQXCTG6v9oQHN3 "
                f"and save as rtdetrv4_x.pth"
            )
        
        logging.info(f'Loading RT-DETRv4 checkpoint: {checkpoint_path}')
        
        # Load model using PyTorch
        import torch
        import torch.nn as nn
        import torchvision.transforms as T
        
        # Try to import RT-DETRv4 engine
        # First, add local RT-DETRv4 repo to sys.path if it exists
        rtdetrv4_repo_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'RT-DETRv4')
        if os.path.exists(rtdetrv4_repo_path) and rtdetrv4_repo_path not in sys.path:
            sys.path.insert(0, rtdetrv4_repo_path)
            logging.info(f'Added RT-DETRv4 repo to sys.path: {rtdetrv4_repo_path}')
        
        try:
            from engine.core import YAMLConfig
        except ImportError:
            # Fallback: simplified loading without YAMLConfig
            logging.warning('RT-DETRv4 engine not found. Using simplified DETR loading.')
            self._load_rtdetrv4_simplified(checkpoint_path)
            return
        
        # Config path for RT-DETRv4
        # First try: configs in RT-DETRv4 repo
        config_path = os.path.join(rtdetrv4_repo_path, 'configs', 'rtv4', f'rtv4_hgnetv2_{size}_coco.yml')
        if not os.path.exists(config_path):
            # Second try: configs copied to rtdetrv4 folder
            config_path = os.path.join(models_dir, 'configs', f'rtv4_hgnetv2_{size}_coco.yml')
        
        if not os.path.exists(config_path):
            logging.warning(f'Config file not found: {config_path}. Using simplified loading.')
            self._load_rtdetrv4_simplified(checkpoint_path)
            return
        
        # Load with YAMLConfig
        cfg = YAMLConfig(config_path, resume=checkpoint_path)
        
        if 'HGNetv2' in cfg.yaml_cfg:
            cfg.yaml_cfg['HGNetv2']['pretrained'] = False
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']
        
        cfg.model.load_state_dict(state)
        
        # Create deploy model
        class RTDETRv4Model(nn.Module):
            def __init__(self, model, postprocessor):
                super().__init__()
                self.model = model.deploy()
                self.postprocessor = postprocessor.deploy()
            
            def forward(self, images, orig_target_sizes):
                outputs = self.model(images)
                outputs = self.postprocessor(outputs, orig_target_sizes)
                return outputs
        
        self.rtdetrv4_model = RTDETRv4Model(cfg.model, cfg.postprocessor).to(self.device)
        self.rtdetrv4_model.eval()
        
        # Transform for preprocessing
        self.rtdetrv4_transform = T.Compose([
            T.Resize((640, 640)),
            T.ToTensor(),
        ])
        
        logging.info(f'RT-DETRv4 detector loaded from {checkpoint_path}')
    
    def _load_rtdetrv4_simplified(self, checkpoint_path):
        '''
        RT-DETRv4 loading without YAMLConfig - raises clear error.

        The fallback to Faster R-CNN has been removed for clarity.
        Users should either:
        1. Use synthpose_detector='yolox' (recommended, fastest)
        2. Use synthpose_detector='rtdetr' (good accuracy, no local setup)
        3. Properly install RT-DETRv4 engine for synthpose_detector='rtdetrv4'
        '''
        raise ImportError(
            "RT-DETRv4 engine not properly installed.\n\n"
            "The RT-DETRv4 detector requires the engine/ directory from the RT-DETRv4 repository.\n\n"
            "Options:\n"
            "  1. Use 'synthpose_detector = \"yolox\"' (RECOMMENDED - fast and reliable)\n"
            "  2. Use 'synthpose_detector = \"rtdetr\"' (good accuracy, HuggingFace-based)\n"
            "  3. Install RT-DETRv4 engine properly:\n"
            "     - Ensure Sports2D/models/RT-DETRv4/engine/ directory exists\n"
            "     - Download model weights to Sports2D/models/rtdetrv4/\n\n"
            "For most users, we recommend: synthpose_detector = 'yolox'"
        )
    
    def __call__(self, frame):
        '''
        Run pose estimation on a frame.
        
        INPUTS:
        - frame: BGR numpy array (H, W, 3) from cv2.VideoCapture
        
        OUTPUTS:
        - keypoints: np.array shape (N_persons, 52, 2)
        - scores: np.array shape (N_persons, 52)
        '''
        import cv2

        self.frame_count += 1
        self.last_detections = self._empty_detections()
        person_roi, ball_roi = self._resolve_manual_rois(frame.shape)
        primary_roi = person_roi
        primary_frame = frame
        person_filter_roi = None
        ball_filter_roi = None

        if person_roi is not None:
            if self.detect_ball and not self._uses_secondary_sam3_ball_detector():
                primary_roi = union_rois(person_roi, ball_roi) or person_roi
                if (
                    ball_roi is not None
                    and primary_roi != person_roi
                    and not self._warned_shared_union_roi
                ):
                    logging.info(
                        "manual_roi=true with a shared detector uses the union of person_roi and ball_roi."
                    )
                    self._warned_shared_union_roi = True
            primary_frame = crop_frame_to_roi(frame, primary_roi)
            person_filter_roi = translate_roi_to_local(person_roi, primary_roi)
            ball_filter_roi = (
                translate_roi_to_local(ball_roi, primary_roi)
                if ball_roi is not None else None
            )

        # Stage 1: Person Detection using YOLOX or RT-DETR
        # Fix: det_frequency=1 means run every frame, det_frequency=2 means every 2nd frame, etc.
        run_detection = (self.frame_count % self.det_frequency == 0) or (self.prev_boxes is None)
        adaptive_person_mode = self._adaptive_person_mode_enabled()
        used_full_frame_person_reacquire = (
            adaptive_person_mode
            and self.manual_person_roi is not None
            and person_roi is None
        )

        if run_detection:
            if used_full_frame_person_reacquire:
                self.last_full_frame_person_reacquire_frame = int(self.frame_count)
            person_boxes = self._detect_persons(primary_frame)
            if primary_roi is not None and not self._uses_secondary_sam3_ball_detector():
                person_boxes = self._filter_primary_detections_to_rois(
                    person_boxes,
                    person_roi=person_filter_roi,
                    ball_roi=ball_filter_roi,
                )
            primary_meta = self.last_detections
            primary_meta_release_check = (
                offset_detection_meta_to_full_frame(primary_meta, primary_roi)
                if primary_roi is not None else dict(primary_meta or self._empty_detections())
            )
            primary_meta_release_check = self._apply_ball_ignore_zones_to_detection_meta(
                primary_meta_release_check,
            )
            full_frame_person_boxes = person_boxes
            if len(person_boxes) > 0 and primary_roi is not None:
                full_frame_person_boxes = self._offset_coco_boxes_to_full_frame(person_boxes, primary_roi)
            self.prev_boxes = full_frame_person_boxes if len(person_boxes) > 0 else None
            if len(person_boxes) > 0 and adaptive_person_mode:
                self._update_adaptive_person_roi(
                    self._coco_to_xyxy_boxes(full_frame_person_boxes),
                    frame.shape,
                )
            elif primary_roi is not None and len(person_boxes) > 0:
                self._manual_person_roi_released = True
                if not self._uses_secondary_sam3_ball_detector():
                    self._manual_ball_roi_released = True
            elif (
                primary_roi is not None
                and not adaptive_person_mode
                and not self._uses_secondary_sam3_ball_detector()
                and len(self._ensure_xyxy_boxes(primary_meta_release_check.get('ball_boxes'))) > 0
            ):
                self._manual_person_roi_released = True
                self._manual_ball_roi_released = True
            elif primary_roi is not None and adaptive_person_mode:
                self._mark_adaptive_person_roi_miss(frame.shape)
        else:
            person_boxes = self.prev_boxes if self.prev_boxes is not None else np.array([])
            self.last_detections = self._empty_detections()
            primary_meta = self.last_detections

        if primary_roi is not None:
            self.last_detections = offset_detection_meta_to_full_frame(primary_meta, primary_roi)
        else:
            self.last_detections = dict(primary_meta or self._empty_detections())
        self.last_detections = self._apply_ball_ignore_zones_to_detection_meta(
            self.last_detections,
        )

        if self._uses_secondary_sam3_ball_detector():
            uses_video_ball_detector = isinstance(getattr(self, 'sam3_ball_detector', None), Sam3VideoDetector)
            ball_inference_roi = None if uses_video_ball_detector else ball_roi
            if ball_roi is not None and not uses_video_ball_detector:
                ball_inference_roi = expand_roi_with_context(
                    ball_roi,
                    frame.shape,
                    scale=2.5,
                    min_size=128,
                )
            ball_frame = crop_frame_to_roi(frame, ball_inference_roi)
            ball_meta = self._detect_balls_sam3(ball_frame)
            if ball_inference_roi is not None and not uses_video_ball_detector:
                ball_meta = offset_detection_meta_to_full_frame(ball_meta, ball_inference_roi)
            ball_meta = self._filter_ball_detection_meta_to_roi(ball_meta, ball_roi=ball_roi)
            ball_meta = self._apply_ball_ignore_zones_to_detection_meta(ball_meta)
            if ball_roi is not None and len(self._ensure_xyxy_boxes(ball_meta.get('ball_boxes'))) > 0:
                self._manual_ball_roi_released = True
            self._merge_secondary_ball_detections(ball_meta)

        # No persons detected
        if len(person_boxes) == 0:
            return np.array([]).reshape(0, 52, 2), np.array([]).reshape(0, 52)

        pose_person_boxes = person_boxes
        if not run_detection and adaptive_person_mode and primary_roi is not None:
            pose_person_boxes = self._offset_coco_boxes_to_local_frame(person_boxes, primary_roi)

        # Stage 2: Pose Estimation using VitPose
        # Convert BGR to RGB PIL Image for VitPose
        frame_rgb = cv2.cvtColor(primary_frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        keypoints, scores = self._estimate_poses(pil_image, pose_person_boxes)

        # Handle empty pose results
        if keypoints.size == 0:
            return np.array([]).reshape(0, 52, 2), np.array([]).reshape(0, 52)

        if primary_roi is not None:
            keypoints = offset_keypoints_to_full_frame(keypoints, primary_roi)
        return keypoints, scores
    
    def _detect_persons(self, frame, height=None, width=None):
        '''
        Detect persons in frame using YOLOX (rtmlib) or RT-DETR (HuggingFace).
        
        INPUTS:
        - frame: BGR numpy array (H, W, 3)
        - height: Frame height (optional, for RT-DETR)
        - width: Frame width (optional, for RT-DETR)
        
        OUTPUTS:
        - person_boxes: np.array shape (N_persons, 4) in COCO format (x, y, w, h)
        '''
        
        detector_map = {
            'sam3': self._detect_persons_sam3,
            'rtdetrv4': self._detect_persons_rtdetrv4,
            'rtdetr': self._detect_persons_rtdetr,
            'yolox': self._detect_persons_yolox,
            'yolo26': self._detect_persons_yolo26,
        }
        try:
            detector_fn = detector_map[self.detector_type]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported synthpose_detector '{self.detector_type}'."
            ) from exc
        return detector_fn(frame)
    
    def _detect_persons_yolox(self, frame):
        '''Detect persons using rtmlib YOLOX.'''

        detector_outputs = self.detector(frame)
        if detector_outputs is None:
            self.last_detections = self._empty_detections()
            return np.array([])

        if isinstance(detector_outputs, tuple) and len(detector_outputs) >= 2:
            bboxes = np.asarray(detector_outputs[0], dtype=np.float32)
            classes = np.asarray(detector_outputs[1], dtype=np.int32).reshape(-1)
            if len(detector_outputs) >= 3:
                detection_scores = np.asarray(detector_outputs[2], dtype=np.float32).reshape(-1)
            elif bboxes.ndim == 2 and bboxes.shape[1] >= 5:
                detection_scores = bboxes[:, 4].astype(np.float32, copy=False)
            else:
                detection_scores = np.full((len(bboxes),), np.nan, dtype=np.float32)
        else:
            bboxes = np.asarray(detector_outputs, dtype=np.float32)
            classes = np.full((len(bboxes),), PERSON_CLASS_ID, dtype=np.int32)
            if bboxes.ndim == 2 and bboxes.shape[1] >= 5:
                detection_scores = bboxes[:, 4].astype(np.float32, copy=False)
            else:
                detection_scores = np.full((len(bboxes),), np.nan, dtype=np.float32)

        if bboxes.size == 0:
            self.last_detections = self._empty_detections()
            return np.array([])

        if bboxes.ndim == 1:
            bboxes = bboxes.reshape(1, -1)
        if bboxes.shape[1] < 4:
            self.last_detections = self._empty_detections()
            return np.array([])

        if bboxes.shape[1] >= 5 and not (isinstance(detector_outputs, tuple) and len(detector_outputs) >= 2):
            score_mask = bboxes[:, 4] >= self.person_threshold
            bboxes = bboxes[score_mask]
            classes = classes[score_mask]
            detection_scores = detection_scores[score_mask] if len(detection_scores) == len(score_mask) else np.full((len(bboxes),), np.nan, dtype=np.float32)
            if len(bboxes) == 0:
                self.last_detections = self._empty_detections()
                return np.array([])

        metadata_score_threshold = (
            min(self.person_threshold, self.ball_detection_threshold)
            if self._primary_detector_handles_ball() else self.person_threshold
        )
        return self._set_last_detections(
            bboxes[:, :4],
            classes=classes,
            scores=detection_scores,
            metadata_score_threshold=metadata_score_threshold,
            person_score_threshold=self.person_threshold,
        )

    def _detect_persons_yolo26(self, frame):
        '''Detect persons using Ultralytics YOLO26.'''
        metadata_threshold = (
            min(self.person_threshold, self.ball_detection_threshold)
            if self._primary_detector_handles_ball() else self.person_threshold
        )
        try:
            results = self.detector(
                frame,
                verbose=False,
                conf=float(metadata_threshold),
            )
        except TypeError:
            results = self.detector(frame)

        if results is None:
            self.last_detections = self._empty_detections()
            return np.array([])

        if isinstance(results, (list, tuple)):
            result = results[0] if len(results) > 0 else None
        else:
            result = results
        if result is None:
            self.last_detections = self._empty_detections()
            return np.array([])

        result_boxes = getattr(result, 'boxes', None)
        boxes = self._ensure_xyxy_boxes(
            getattr(result_boxes, 'xyxy', result_boxes)
        )
        if len(boxes) == 0:
            self.last_detections = self._empty_detections()
            return np.array([])

        classes = self._tensor_like_to_numpy(
            getattr(result_boxes, 'cls', None),
            dtype=np.int32,
        ).reshape(-1)
        scores = self._tensor_like_to_numpy(
            getattr(result_boxes, 'conf', None),
            dtype=np.float32,
        ).reshape(-1)
        class_names = self._resolve_class_names(classes, getattr(result, 'names', None))

        return self._set_last_detections(
            boxes,
            classes=classes,
            scores=scores,
            class_names=class_names,
            metadata_score_threshold=metadata_threshold,
            person_score_threshold=self.person_threshold,
        )

    def _detect_persons_sam3(self, frame):
        '''Detect persons using SAM3 with prompt presets.'''
        import cv2

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        person_boxes_coco, detection_meta = self.sam3_detector.detect_person_boxes(pil_image)
        self.last_detections = detection_meta or self._empty_detections()
        return person_boxes_coco

    def _detect_balls_sam3(self, frame):
        '''Detect sports balls using the secondary SAM3 detector in hybrid mode.'''
        import cv2

        if self.sam3_ball_detector is None:
            return self._empty_detections()

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        if isinstance(self.sam3_ball_detector, Sam3VideoDetector):
            absolute_frame_idx = self.video_frame_index_offset + max(0, int(self.frame_count) - 1)
            return self.sam3_ball_detector.detect(
                pil_image,
                frame_index=absolute_frame_idx,
            ) or self._empty_detections()
        return self.sam3_ball_detector.detect(pil_image) or self._empty_detections()

    def _merge_secondary_ball_detections(self, ball_meta):
        '''Merge SAM3 sports-ball detections into the current detector metadata.'''
        base_meta = dict(self.last_detections or self._empty_detections())
        ball_meta = ball_meta or self._empty_detections()

        base_boxes = np.asarray(base_meta.get('boxes', np.empty((0, 4))), dtype=np.float32).reshape(-1, 4)
        base_classes = np.asarray(base_meta.get('classes', np.empty((0,))), dtype=np.int32).reshape(-1)
        base_scores = np.asarray(base_meta.get('scores', np.empty((0,))), dtype=np.float32).reshape(-1)
        base_class_names = np.asarray(base_meta.get('class_names', np.empty((0,), dtype=object)), dtype=object).reshape(-1)
        base_prompt_indices = np.asarray(base_meta.get('prompt_indices', np.empty((0,))), dtype=np.int32).reshape(-1)

        ball_boxes = np.asarray(ball_meta.get('boxes', np.empty((0, 4))), dtype=np.float32).reshape(-1, 4)
        ball_classes = np.asarray(ball_meta.get('classes', np.empty((0,))), dtype=np.int32).reshape(-1)
        ball_scores = np.asarray(ball_meta.get('scores', np.empty((0,))), dtype=np.float32).reshape(-1)
        ball_class_names = np.asarray(ball_meta.get('class_names', np.empty((0,), dtype=object)), dtype=object).reshape(-1)
        ball_prompt_indices = np.asarray(ball_meta.get('prompt_indices', np.empty((0,))), dtype=np.int32).reshape(-1)

        if len(base_class_names) != len(base_classes):
            base_class_names = np.empty((len(base_classes),), dtype=object)
        if len(base_prompt_indices) != len(base_classes):
            base_prompt_indices = np.full((len(base_classes),), -1, dtype=np.int32)
        if len(ball_class_names) != len(ball_classes):
            ball_class_names = np.empty((len(ball_classes),), dtype=object)
        if len(ball_prompt_indices) != len(ball_classes):
            ball_prompt_indices = np.full((len(ball_classes),), -1, dtype=np.int32)

        base_meta['boxes'] = (
            np.concatenate([base_boxes, ball_boxes], axis=0)
            if len(base_boxes) > 0 or len(ball_boxes) > 0
            else np.empty((0, 4), dtype=np.float32)
        )
        base_meta['classes'] = np.concatenate([base_classes, ball_classes], axis=0)
        base_meta['scores'] = np.concatenate([base_scores, ball_scores], axis=0)
        base_meta['class_names'] = np.concatenate([base_class_names, ball_class_names], axis=0)
        base_meta['prompt_indices'] = np.concatenate([base_prompt_indices, ball_prompt_indices], axis=0)
        base_meta['ball_boxes'] = np.asarray(
            ball_meta.get('ball_boxes', np.empty((0, 4))),
            dtype=np.float32,
        ).reshape(-1, 4)
        base_meta['ball_scores'] = np.asarray(
            ball_meta.get('ball_scores', np.empty((0,))),
            dtype=np.float32,
        ).reshape(-1)
        base_meta['sam3_ball_meta'] = ball_meta
        self.last_detections = base_meta
    
    def _detect_persons_rtdetr(self, frame):
        '''Detect persons using RT-DETR from HuggingFace Transformers (original SynthPose_PM).'''
        import cv2
        
        height, width = frame.shape[:2]
        
        # Convert BGR to RGB PIL Image for RT-DETR
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        # Process image for RT-DETR
        inputs = self.rtdetr_processor(images=pil_image, return_tensors="pt").to(self.device)
        
        with _torch.no_grad():
            outputs = self.rtdetr_model(**inputs)
        
        metadata_threshold = (
            min(self.person_threshold, self.ball_detection_threshold)
            if self._primary_detector_handles_ball() else self.person_threshold
        )

        # Post-process detection results
        results = self.rtdetr_processor.post_process_object_detection(
            outputs, 
            target_sizes=_torch.tensor([(height, width)]),
            threshold=metadata_threshold,
        )
        result = results[0]

        labels = result["labels"].cpu().numpy()
        scores = result["scores"].cpu().numpy()
        all_boxes_voc = result["boxes"].cpu().numpy()
        return self._set_last_detections(
            all_boxes_voc,
            classes=labels,
            scores=scores,
            metadata_score_threshold=metadata_threshold,
            person_score_threshold=self.person_threshold,
        )
    
    def _detect_persons_rtdetrv4(self, frame):
        '''Detect persons using RT-DETRv4 from local checkpoint.'''
        import cv2
        
        height, width = frame.shape[:2]
        
        # Convert BGR to RGB PIL Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        # Preprocess
        im_data = self.rtdetrv4_transform(pil_image).unsqueeze(0).to(self.device)
        orig_size = _torch.tensor([[width, height]]).to(self.device)
        
        # Inference
        with _torch.no_grad():
            labels, boxes, scores = self.rtdetrv4_model(im_data, orig_size)

        labels_0 = labels[0]
        boxes_0 = boxes[0]
        scores_0 = scores[0]
        metadata_threshold = (
            min(self.person_threshold, self.ball_detection_threshold)
            if self._primary_detector_handles_ball() else self.person_threshold
        )
        return self._set_last_detections(
            boxes_0,
            classes=labels_0,
            scores=scores_0,
            metadata_score_threshold=metadata_threshold,
            person_score_threshold=self.person_threshold,
        )

    def _estimate_poses(self, pil_image, person_boxes):
        '''
        Estimate poses for detected persons using VitPose.
        
        INPUTS:
        - pil_image: PIL Image (RGB)
        - person_boxes: np.array shape (N_persons, 4) in COCO format (x, y, w, h)
        
        OUTPUTS:
        - keypoints: np.array shape (N_persons, 52, 2)
        - scores: np.array shape (N_persons, 52)
        '''
        
        # Prepare input for VitPose
        inputs = self.pose_processor(
            pil_image,
            boxes=[person_boxes],
            return_tensors="pt"
        ).to(self.device)
        
        # Run pose estimation
        with _torch.no_grad():
            outputs = self.pose_model(**inputs)
        
        # Post-process results
        pose_results = self.pose_processor.post_process_pose_estimation(
            outputs,
            boxes=[person_boxes]
        )
        image_pose_result = pose_results[0]
        
        # Convert to numpy arrays
        if len(image_pose_result) > 0:
            all_keypoints = []
            all_scores = []
            
            for pose_result in image_pose_result:
                keypoints = np.array(pose_result["keypoints"])  # (52, 2)
                scores = np.array(pose_result["scores"])        # (52,)
                all_keypoints.append(keypoints)
                all_scores.append(scores)
            
            return np.array(all_keypoints), np.array(all_scores)
        
        return np.array([]).reshape(0, 52, 2), np.array([]).reshape(0, 52)
    
    def reset(self):
        '''Reset tracker state (clear previous boxes and frame count).'''
        self.frame_count = 0
        self.prev_boxes = None


def setup_synthpose_tracker(mode='huge', det_frequency=1, device='auto', backend='auto', detector='yolox'):
    '''
    Factory function to create SynthPose tracker.
    Matches the pattern used by setup_pose_tracker() in Sports2D.
    
    INPUTS:
    - mode: 'huge' or 'base' (VitPose model size)
    - det_frequency: Detection frequency (run detection every N frames)
    - device: 'auto', 'cuda', 'cpu', 'mps'
    - backend: 'auto', 'onnxruntime', 'openvino', 'opencv'
    - detector: 'yolox' (rtmlib, faster), 'yolo26' (Ultralytics), 'rtdetr' (HuggingFace), or 'rtdetrv4' (local checkpoint, best accuracy)
    
    OUTPUTS:
    - SynthPosePoseTracker instance
    '''
    return SynthPosePoseTracker(
        mode=mode,
        device=device,
        det_frequency=det_frequency,
        backend=backend,
        detector=detector
    )
