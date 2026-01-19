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
                 detector='yolox'):
        '''
        Initialize SynthPose tracker.
        
        INPUTS:
        - mode: 'huge' or 'base' - VitPose model size selection
                'huge' = stanfordmimi/synthpose-vitpose-huge-hf (more accurate, slower)
                'base' = stanfordmimi/synthpose-vitpose-base-hf (faster, less accurate)
        - device: 'auto', 'cuda', 'cpu', 'mps' for VitPose
                  'auto' will select CUDA if available, else CPU
        - det_frequency: Run person detection every N frames (default 1)
        - person_threshold: Confidence threshold for person detection (default 0.3)
        - keypoint_threshold: Confidence threshold for keypoints (default 0.3)
        - backend: Backend for rtmlib ('auto', 'onnxruntime', 'openvino', 'opencv')
        - detector: 'yolox' (rtmlib, faster) or 'rtdetr' (HuggingFace Transformers, original SynthPose_PM detector)
        '''
        
        # Load dependencies
        _load_dependencies()
        
        self.mode = mode.lower()
        self.det_frequency = max(1, det_frequency)
        self.person_threshold = person_threshold
        self.keypoint_threshold = keypoint_threshold
        self.backend = backend
        self.detector_type = detector.lower()  # 'yolox' or 'rtdetr'
        
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
        
        logging.info(f'SynthPose initializing with mode={self.mode}, detector={self.detector_type}, device={self.device}')
        
        # Load models
        self._load_models()
        
        logging.info(f'SynthPose ready. Output: 52 keypoints (17 COCO + 35 anatomical markers)')
    
    def _load_models(self):
        '''Load person detector (YOLOX, RT-DETR, or RT-DETRv4) and VitPose model.'''
        
        # Person detector: YOLOX (rtmlib) or RT-DETR (HuggingFace) or RT-DETRv4 (local)
        if self.detector_type == 'rtdetrv4':
            logging.info('Loading RT-DETRv4 person detector (local checkpoint)...')
            self._load_rtdetrv4_detector()
        elif self.detector_type == 'rtdetr':
            logging.info('Loading RT-DETR person detector (HuggingFace Transformers, original SynthPose_PM)...')
            self._load_rtdetr_detector()
        else:
            logging.info('Loading rtmlib YOLOX person detector...')
            self._load_rtmlib_detector()
        
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
    
    def _load_rtmlib_detector(self):
        '''Load rtmlib YOLOX detector.'''
        from rtmlib import YOLOX
        
        # Always use performance mode YOLOX (yolox_x) for best detection quality with SynthPose
        # This matches the 'performance' mode detector in Sports2D/rtmlib
        model_url = 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_x_8xb8-300e_humanart-a39d44ed.zip'
        input_size = (640, 640)
        
        # Determine backend and device for rtmlib
        rtmlib_backend = self.backend if self.backend != 'auto' else 'onnxruntime'
        rtmlib_device = 'cuda' if self.device == 'cuda' else 'cpu'
        
        # rtmlib YOLOX API: onnx_model, model_input_size, nms_thr, score_thr, backend, device
        self.detector = YOLOX(
            onnx_model=model_url,
            model_input_size=input_size,
            score_thr=self.person_threshold,
            backend=rtmlib_backend,
            device=rtmlib_device
        )
        
        logging.info(f'rtmlib YOLOX detector loaded (yolox_x/performance, backend={rtmlib_backend})')
    
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
        '''Simplified RT-DETRv4 loading without YAMLConfig (uses raw checkpoint).'''
        import torch
        import torch.nn as nn
        import torchvision.transforms as T
        from torchvision.models.detection import fasterrcnn_resnet50_fpn
        
        logging.warning('Using fallback detector (Faster R-CNN) since RT-DETRv4 engine is not installed.')
        logging.warning('For full RT-DETRv4 support, install: pip install git+https://github.com/RT-DETRs/RT-DETRv4.git')
        
        # Fallback to torchvision Faster R-CNN for person detection
        self.rtdetrv4_model = fasterrcnn_resnet50_fpn(pretrained=True).to(self.device)
        self.rtdetrv4_model.eval()
        
        self.rtdetrv4_transform = T.Compose([
            T.ToTensor(),
        ])
        
        self._rtdetrv4_is_fallback = True
        logging.info('Fallback Faster R-CNN detector loaded')
    
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
        
        # Stage 1: Person Detection using YOLOX or RT-DETR
        # Fix: det_frequency=1 means run every frame, det_frequency=2 means every 2nd frame, etc.
        run_detection = (self.frame_count % self.det_frequency == 0) or (self.prev_boxes is None)
        
        if run_detection:
            person_boxes = self._detect_persons(frame, height, width)
            self.prev_boxes = person_boxes if len(person_boxes) > 0 else None
        else:
            person_boxes = self.prev_boxes if self.prev_boxes is not None else np.array([])
        
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
        
        if self.detector_type == 'rtdetrv4':
            return self._detect_persons_rtdetrv4(frame)
        elif self.detector_type == 'rtdetr':
            return self._detect_persons_rtdetr(frame)
        else:
            return self._detect_persons_yolox(frame)
    
    def _detect_persons_yolox(self, frame):
        '''Detect persons using rtmlib YOLOX.'''
        
        # rtmlib YOLOX returns boxes in (x1, y1, x2, y2) format
        bboxes = self.detector(frame)
        
        if bboxes is None or len(bboxes) == 0:
            return np.array([])
        
        # Filter by confidence threshold
        if bboxes.shape[1] == 5:  # (x1, y1, x2, y2, score)
            mask = bboxes[:, 4] >= self.person_threshold
            bboxes = bboxes[mask]
        
        if len(bboxes) == 0:
            return np.array([])
        
        # Convert from (x1, y1, x2, y2) to COCO format (x, y, w, h) for VitPose
        person_boxes_coco = np.zeros((len(bboxes), 4))
        person_boxes_coco[:, 0] = bboxes[:, 0]  # x
        person_boxes_coco[:, 1] = bboxes[:, 1]  # y
        person_boxes_coco[:, 2] = bboxes[:, 2] - bboxes[:, 0]  # width
        person_boxes_coco[:, 3] = bboxes[:, 3] - bboxes[:, 1]  # height
        
        return person_boxes_coco
    
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
            threshold=self.person_threshold
        )
        result = results[0]
        
        # Filter for human label (0 in COCO)
        person_mask = result["labels"] == 0
        person_boxes_voc = result["boxes"][person_mask].cpu().numpy()
        
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
        
        # Check if using fallback detector
        if getattr(self, '_rtdetrv4_is_fallback', False):
            return self._detect_persons_rtdetrv4_fallback(frame_rgb, height, width)
        
        # Preprocess
        im_data = self.rtdetrv4_transform(pil_image).unsqueeze(0).to(self.device)
        orig_size = _torch.tensor([[width, height]]).to(self.device)
        
        # Inference
        with _torch.no_grad():
            labels, boxes, scores = self.rtdetrv4_model(im_data, orig_size)
        
        # Filter for person class (label 0 in COCO) and threshold
        person_mask = (labels[0] == 0) & (scores[0] >= self.person_threshold)
        person_boxes_voc = boxes[0][person_mask].cpu().numpy()
        
        if len(person_boxes_voc) == 0:
            return np.array([])
        
        # Convert from VOC (x1, y1, x2, y2) to COCO format (x, y, w, h) for VitPose
        person_boxes_coco = np.zeros((len(person_boxes_voc), 4))
        person_boxes_coco[:, 0] = person_boxes_voc[:, 0]  # x
        person_boxes_coco[:, 1] = person_boxes_voc[:, 1]  # y
        person_boxes_coco[:, 2] = person_boxes_voc[:, 2] - person_boxes_voc[:, 0]  # width
        person_boxes_coco[:, 3] = person_boxes_voc[:, 3] - person_boxes_voc[:, 1]  # height
        
        return person_boxes_coco
    
    def _detect_persons_rtdetrv4_fallback(self, frame_rgb, height, width):
        '''Fallback detection using Faster R-CNN when RT-DETRv4 engine is not available.'''
        import torch
        
        # Convert to tensor
        im_tensor = self.rtdetrv4_transform(Image.fromarray(frame_rgb)).to(self.device)
        
        with torch.no_grad():
            outputs = self.rtdetrv4_model([im_tensor])
        
        # Filter for person class (label 1 in torchvision COCO) and threshold
        boxes = outputs[0]['boxes'].cpu().numpy()
        labels = outputs[0]['labels'].cpu().numpy()
        scores = outputs[0]['scores'].cpu().numpy()
        
        person_mask = (labels == 1) & (scores >= self.person_threshold)
        person_boxes_voc = boxes[person_mask]
        
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
