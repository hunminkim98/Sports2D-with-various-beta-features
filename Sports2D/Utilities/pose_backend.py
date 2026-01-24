#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
    ##################################################
    ## Pose Estimation Backend Abstraction Layer    ##
    ##################################################

    Unified interface for pose estimation backends (RTMLib, SynthPose).

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
'''

from abc import ABC, abstractmethod
from typing import Tuple, List
import numpy as np
import logging


# Retry delay for tracker initialization (multi-threading conflicts)
TRACKER_INIT_RETRY_DELAY = 3  # seconds


## AUTHORSHIP INFORMATION
__author__ = "Sports2D Contributors"
__copyright__ = "Copyright 2024, Sports2D"
__license__ = "BSD 3-Clause License"


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
        from Pose2Sim.poseEstimation import setup_model_class_mode, setup_backend_device, setup_pose_tracker
        from anytree import PreOrderIter

        pose_config = config_dict.get('pose', {})

        # 1. Model and mode setup
        pose_model_name = pose_config.get('pose_model', 'body_with_feet')
        mode = pose_config.get('mode', 'balanced')

        self._pose_model, self._ModelClass, self._mode = setup_model_class_mode(
            pose_model_name,
            mode,
            config_dict
        )

        # 2. Backend and device setup (ONNX providers)
        backend = pose_config.get('backend', 'auto')
        device = pose_config.get('device', 'auto')
        self._backend, self._device = setup_backend_device(backend, device)

        # 3. Tracker initialization with retry for multi-threading
        det_frequency = pose_config.get('det_frequency', 4)
        try:
            self._tracker = setup_pose_tracker(
                self._ModelClass, det_frequency, self._mode,
                False, self._backend, self._device
            )
        except RuntimeError as e:
            # Retry once for multi-threading initialization issues
            import time
            logging.warning(f'RTMLib tracker init retry due to: {e}')
            time.sleep(TRACKER_INIT_RETRY_DELAY)
            self._tracker = setup_pose_tracker(
                self._ModelClass, det_frequency, self._mode,
                False, self._backend, self._device
            )

        # Cache keypoint names and count
        self._keypoint_names = [node.name for node in PreOrderIter(self._pose_model) if node.id is not None]
        self._num_keypoints = len(self._keypoint_names)

        logging.info(f'RTMLibBackend initialized: model={pose_model_name}, mode={self._mode}, '
                     f'backend={self._backend}, device={self._device}, keypoints={self._num_keypoints}')

    def __call__(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Run pose estimation. Returns (keypoints, scores)."""
        return self._tracker(frame)

    def reset(self) -> None:
        """Reset tracker state."""
        if hasattr(self._tracker, 'reset'):
            self._tracker.reset()

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
        return 'rtmlib'

    @property
    def keypoint_names(self) -> List[str]:
        """Return keypoint names from skeleton tree."""
        return self._keypoint_names


class SynthPoseBackend(PoseBackend):
    """
    SynthPose backend using VitPose models with PyTorch.

    This backend provides:
    - PyTorch-based inference (GPU-accelerated)
    - 52 keypoints (17 COCO + 35 anatomical markers)
    - Multiple detector options (yolox, rtdetr, rtdetrv4)

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
        'performance': 'huge',   # VitPose-huge (most accurate)
        'balanced': 'base',      # VitPose-base (balanced)
        'lightweight': 'base',   # Fallback to base (lightweight not supported)
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
                - synthpose_detector: Detector type ('yolox', 'rtdetr', 'rtdetrv4')

        Raises:
            ImportError: If torch/transformers not installed
        """
        from Sports2D.Utilities.synthpose_tracker import SynthPosePoseTracker
        from Sports2D.Utilities.synthpose_skeleton import (
            create_synthpose_skeleton,
            SYNTHPOSE_KEYPOINT_NAMES
        )

        pose_config = config_dict.get('pose', {})
        pose_model = pose_config.get('pose_model', 'synthpose').lower()
        mode = pose_config.get('mode', 'balanced').lower()

        # Determine VitPose size: explicit pose_model takes precedence, then mode
        if pose_model == 'synthpose_base':
            # Explicit base model requested
            self._mode = 'base'
        elif pose_model == 'synthpose':
            # Generic synthpose - use mode parameter for model selection
            if mode == 'lightweight':
                logging.warning(
                    "SynthPose does not support 'lightweight' mode. "
                    "VitPose-base (balanced) will be used instead. "
                    "For maximum accuracy, use mode='performance' (VitPose-huge)."
                )
                self._mode = 'base'
            elif mode == 'performance':
                self._mode = 'huge'
            else:
                # balanced or any other value defaults to base
                self._mode = self.MODE_TO_VITPOSE.get(mode, 'base')
        else:
            # Unknown pose_model, default to huge
            self._mode = 'huge'

        # Device selection (config takes priority, 'auto' triggers detection in tracker)
        device = pose_config.get('device', 'auto')

        # Initialize tracker
        self._tracker = SynthPosePoseTracker(
            mode=self._mode,
            device=device,
            det_frequency=pose_config.get('det_frequency', 4),
            person_threshold=pose_config.get('keypoint_likelihood_threshold', 0.3),
            detector=pose_config.get('synthpose_detector', 'yolox')
        )

        # Store skeleton tree and keypoint names
        self._skeleton_tree = create_synthpose_skeleton()
        self._keypoint_names = list(SYNTHPOSE_KEYPOINT_NAMES)

        # Log initialization with mode mapping info
        mode_info = f"config_mode='{mode}' → vitpose='{self._mode}'"
        logging.info(f'SynthPoseBackend initialized: {mode_info}, '
                     f'detector={pose_config.get("synthpose_detector", "yolox")}, '
                     f'device={self._tracker.device}, keypoints=52')

    def __call__(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Run pose estimation. Returns (keypoints, scores)."""
        return self._tracker(frame)

    def reset(self) -> None:
        """Reset tracker state."""
        self._tracker.frame_count = 0
        self._tracker.prev_boxes = None

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
        return 'synthpose'

    @property
    def keypoint_names(self) -> List[str]:
        """Return 52 SynthPose keypoint names."""
        return self._keypoint_names


def create_pose_backend(config_dict: dict) -> PoseBackend:
    """
    Factory function to create pose backend from configuration.

    Args:
        config_dict: Full configuration dictionary with 'pose' section

    Returns:
        PoseBackend: Configured backend instance (RTMLibBackend or SynthPoseBackend)

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
    pose_config = config_dict.get('pose', {})
    pose_model = pose_config.get('pose_model', 'body_with_feet').lower()

    if pose_model in ['synthpose', 'synthpose_base']:
        try:
            return SynthPoseBackend(config_dict)
        except ImportError as e:
            raise ImportError(
                f"SynthPose requires additional dependencies: {e}\n"
                "Install with: pip install sports2d[synthpose]\n"
                "Or: pip install torch transformers"
            ) from e
    else:
        return RTMLibBackend(config_dict)
