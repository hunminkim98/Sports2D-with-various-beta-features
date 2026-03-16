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
    - rtmlib YOLOX for person detection (same as Sports2D default)
    - VitPose from HuggingFace Transformers for pose estimation (52 keypoints)
    
    This module provides:
    - SynthPosePoseTracker class with __call__(frame) interface
    - Full 52 SynthPose keypoints output (17 COCO + 35 anatomical markers)
    - Uses rtmlib's YOLOX detector for consistency with Sports2D
'''

import numpy as np
import logging
from PIL import Image

from Sports2D.Utilities.sam3_detector import (
    BALL_ONLY_SAM3_PROMPTS,
    Sam3Detector,
    empty_sam3_detections,
)

PERSON_CLASS_ID = 0
SPORTS_BALL_CLASS_ID = 32

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
                 ball_detector_backend='same'):
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
                   Only used when detector='yolox'. Ignored for 'rtdetr' and 'rtdetrv4'.
        - detector: Person detector selection
                    'yolox': rtmlib YOLOX (RECOMMENDED - fast, reliable)
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
        self.detector_type = detector.lower()  # 'yolox' or 'rtdetr'
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
        self.ball_detector_backend = _normalize_ball_detector_backend(
            ball_detector_backend,
            detector=detector,
        )
        self.sam3_collect_masks = bool(
            self.sam3_store_masks or self.sam3_show_realtime_masks or self.sam3_save_ball_masks
        )
        self.last_detections = self._empty_detections()
        self.sam3_detector = None
        self.sam3_ball_detector = None
        self.detector_size = self._resolve_detector_size(detector_size)
        self.ball_detection_threshold = float(np.clip(ball_detection_threshold, 0.01, 0.9))
        self.ball_nms_score_threshold = float(np.clip(ball_nms_score_threshold, 0.01, 0.9))
        
        # Frame tracking for detection frequency
        self.frame_count = 0
        self.prev_boxes = None
        
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
        
        # Load models
        self._load_models()
        
        logging.info(f'SynthPose ready. Output: 52 keypoints (17 COCO + 35 anatomical markers)')

    @staticmethod
    def _resolve_detector_size(size_value):
        """
        Resolve detector size from explicit size or mode alias.

        Accepted values:
        - mode aliases: performance -> x, balanced -> m, lightweight -> s
        - explicit sizes: x, l, m, s
        """
        mode_to_size = {
            'performance': 'x',
            'balanced': 'm',
            'lightweight': 's',
        }
        explicit_sizes = {'x', 'l', 'm', 's'}
        value = str(size_value).lower()
        if value in mode_to_size:
            return mode_to_size[value]
        if value == 'tiny':
            logging.warning(
                "synthpose detector size 'tiny' is deprecated. Using 's'."
            )
            return 's'
        if value in explicit_sizes:
            return value

        logging.warning(
            f"Unknown synthpose detector size '{size_value}'. "
            "Using 'm' (balanced)."
        )
        return 'm'

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
        
        # Person detector: YOLOX (rtmlib), RT-DETR, RT-DETRv4, or SAM3
        if self.detector_type == 'sam3':
            logging.info('Loading SAM3 detector (HF bundle or raw .pt checkpoint)...')
            self._load_sam3_detector()
        elif self.detector_type == 'rtdetrv4':
            logging.info('Loading RT-DETRv4 person detector (local checkpoint)...')
            self._load_rtdetrv4_detector()
        elif self.detector_type == 'rtdetr':
            logging.info('Loading RT-DETR person detector (HuggingFace Transformers, original SynthPose_PM)...')
            self._load_rtdetr_detector()
        else:
            logging.info('Loading rtmlib YOLOX person detector...')
            self._load_rtmlib_detector()

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
        rtmlib_device = 'cuda' if self.device == 'cuda' else 'cpu'
        
        # rtmlib YOLOX API: onnx_model, model_input_size, nms_thr, score_thr, backend, device
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
        height, width = frame.shape[:2]
        self.last_detections = self._empty_detections()
        
        # Stage 1: Person Detection using YOLOX or RT-DETR
        # Fix: det_frequency=1 means run every frame, det_frequency=2 means every 2nd frame, etc.
        run_detection = (self.frame_count % self.det_frequency == 0) or (self.prev_boxes is None)
        
        if run_detection:
            person_boxes = self._detect_persons(frame, height, width)
            if self._uses_secondary_sam3_ball_detector():
                self._merge_secondary_ball_detections(self._detect_balls_sam3(frame))
            self.prev_boxes = person_boxes if len(person_boxes) > 0 else None
        else:
            person_boxes = self.prev_boxes if self.prev_boxes is not None else np.array([])
            self.last_detections = self._empty_detections()
        
        # No persons detected
        if len(person_boxes) == 0:
            return np.array([]).reshape(0, 52, 2), np.array([]).reshape(0, 52)
        
        # Stage 2: Pose Estimation using VitPose
        # Convert BGR to RGB PIL Image for VitPose
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        keypoints, scores = self._estimate_poses(pil_image, person_boxes)
        
        # Handle empty pose results
        if keypoints.size == 0:
            return np.array([]).reshape(0, 52, 2), np.array([]).reshape(0, 52)
        
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
        
        if self.detector_type == 'sam3':
            return self._detect_persons_sam3(frame)
        elif self.detector_type == 'rtdetrv4':
            return self._detect_persons_rtdetrv4(frame)
        elif self.detector_type == 'rtdetr':
            return self._detect_persons_rtdetr(frame)
        else:
            return self._detect_persons_yolox(frame)
    
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

        bboxes = bboxes[:, :4]
        if len(classes) != len(bboxes):
            classes = np.full((len(bboxes),), PERSON_CLASS_ID, dtype=np.int32)
        if len(detection_scores) != len(bboxes):
            detection_scores = np.full((len(bboxes),), np.nan, dtype=np.float32)

        person_mask = classes == PERSON_CLASS_ID
        person_boxes_xyxy = bboxes[person_mask]
        if len(person_boxes_xyxy) == 0 and len(bboxes) > 0:
            # Fallback for detectors whose single-class index is not 0.
            unique_classes = np.unique(classes)
            if len(unique_classes) == 1:
                person_boxes_xyxy = bboxes
        if self._primary_detector_handles_ball():
            ball_mask = np.isin(classes, self.ball_class_ids)
            ball_boxes_xyxy = bboxes[ball_mask]
            ball_scores_xyxy = detection_scores[ball_mask]
        else:
            ball_boxes_xyxy = np.empty((0, 4), dtype=np.float32)
            ball_scores_xyxy = np.empty((0,), dtype=np.float32)

        # Convert from (x1, y1, x2, y2) to COCO format (x, y, w, h) for VitPose
        person_boxes_coco = np.zeros((len(person_boxes_xyxy), 4))
        if len(person_boxes_xyxy) > 0:
            person_boxes_coco[:, 0] = person_boxes_xyxy[:, 0]  # x
            person_boxes_coco[:, 1] = person_boxes_xyxy[:, 1]  # y
            person_boxes_coco[:, 2] = person_boxes_xyxy[:, 2] - person_boxes_xyxy[:, 0]  # width
            person_boxes_coco[:, 3] = person_boxes_xyxy[:, 3] - person_boxes_xyxy[:, 1]  # height

        self.last_detections = {
            'boxes': bboxes,
            'classes': classes,
            'scores': detection_scores,
            'person_boxes': person_boxes_xyxy,
            'ball_boxes': ball_boxes_xyxy,
            'ball_scores': ball_scores_xyxy,
            'class_names': np.empty((0,), dtype=object),
            'prompt_indices': np.empty((0,), dtype=np.int32),
            'sam3_ball_meta': {},
        }

        return person_boxes_coco

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
        
        # Post-process detection results
        results = self.rtdetr_processor.post_process_object_detection(
            outputs, 
            target_sizes=_torch.tensor([(height, width)]),
            threshold=min(self.person_threshold, self.ball_detection_threshold)
        )
        result = results[0]

        labels = result["labels"].cpu().numpy()
        scores = result["scores"].cpu().numpy()
        all_boxes_voc = result["boxes"].cpu().numpy()
        person_mask = (labels == PERSON_CLASS_ID) & (scores >= self.person_threshold)
        person_boxes_voc = all_boxes_voc[person_mask]
        if self._primary_detector_handles_ball():
            ball_mask = np.isin(labels, self.ball_class_ids) & (scores >= self.ball_detection_threshold)
            ball_boxes_voc = all_boxes_voc[ball_mask]
            ball_scores_voc = scores[ball_mask].astype(np.float32, copy=False)
        else:
            ball_boxes_voc = np.empty((0, 4), dtype=np.float32)
            ball_scores_voc = np.empty((0,), dtype=np.float32)

        self.last_detections = {
            'boxes': all_boxes_voc,
            'classes': labels.astype(np.int32, copy=False),
            'scores': scores.astype(np.float32, copy=False),
            'person_boxes': person_boxes_voc,
            'ball_boxes': ball_boxes_voc,
            'ball_scores': ball_scores_voc,
            'class_names': np.empty((0,), dtype=object),
            'prompt_indices': np.empty((0,), dtype=np.int32),
            'sam3_ball_meta': {},
        }
        
        if len(person_boxes_voc) == 0:
            return np.array([])
        
        # Convert from VOC (x1, y1, x2, y2) to COCO format (x, y, w, h) for VitPose
        person_boxes_coco = np.zeros((len(person_boxes_voc), 4))
        person_boxes_coco[:, 0] = person_boxes_voc[:, 0]  # x
        person_boxes_coco[:, 1] = person_boxes_voc[:, 1]  # y
        person_boxes_coco[:, 2] = person_boxes_voc[:, 2] - person_boxes_voc[:, 0]  # width
        person_boxes_coco[:, 3] = person_boxes_voc[:, 3] - person_boxes_voc[:, 1]  # height
        
        return person_boxes_coco
    
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
        person_mask = (labels_0 == PERSON_CLASS_ID) & (scores_0 >= self.person_threshold)
        person_boxes_voc = boxes_0[person_mask].cpu().numpy()
        if self._primary_detector_handles_ball():
            ball_mask = _torch.zeros_like(person_mask, dtype=_torch.bool)
            for cls_id in self.ball_class_ids:
                ball_mask = ball_mask | ((labels_0 == cls_id) & (scores_0 >= self.ball_detection_threshold))
            ball_boxes_voc = boxes_0[ball_mask].cpu().numpy()
            ball_scores_voc = scores_0[ball_mask].cpu().numpy().astype(np.float32, copy=False)
        else:
            ball_boxes_voc = np.empty((0, 4), dtype=np.float32)
            ball_scores_voc = np.empty((0,), dtype=np.float32)

        valid_mask = scores_0 >= self.person_threshold
        self.last_detections = {
            'boxes': boxes_0[valid_mask].cpu().numpy(),
            'classes': labels_0[valid_mask].cpu().numpy().astype(np.int32, copy=False),
            'scores': scores_0[valid_mask].cpu().numpy().astype(np.float32, copy=False),
            'person_boxes': person_boxes_voc,
            'ball_boxes': ball_boxes_voc,
            'ball_scores': ball_scores_voc,
            'class_names': np.empty((0,), dtype=object),
            'prompt_indices': np.empty((0,), dtype=np.int32),
            'sam3_ball_meta': {},
        }
        
        if len(person_boxes_voc) == 0:
            return np.array([])
        
        # Convert from VOC (x1, y1, x2, y2) to COCO format (x, y, w, h) for VitPose
        person_boxes_coco = np.zeros((len(person_boxes_voc), 4))
        person_boxes_coco[:, 0] = person_boxes_voc[:, 0]  # x
        person_boxes_coco[:, 1] = person_boxes_voc[:, 1]  # y
        person_boxes_coco[:, 2] = person_boxes_voc[:, 2] - person_boxes_voc[:, 0]  # width
        person_boxes_coco[:, 3] = person_boxes_voc[:, 3] - person_boxes_voc[:, 1]  # height
        
        return person_boxes_coco

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
    - detector: 'yolox' (rtmlib, faster), 'rtdetr' (HuggingFace), or 'rtdetrv4' (local checkpoint, best accuracy)
    
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
