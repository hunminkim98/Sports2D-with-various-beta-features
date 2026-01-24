# SynthPose & RT-DETRv4 Architecture Refactoring Plan

## Overview

**Goal**: Sports2D의 SynthPose/RT-DETRv4 통합을 기존 RTMLib 패턴에 맞게 깔끔하게 리팩토링

**Scope**: 전체 아키텍처 통합 (Backend 추상화, 설정 표준화, 그리기 함수 통합)

**Constraints**:
- API 호환성 유지 (Sports2D.process() 및 CLI 변경 없음)
- 52 keypoints 출력 유지 (HALPE26 매핑은 각도 계산시만 사용)
- Flat config schema 유지 ([pose] 섹션 구조 유지)

---

## Phase 1: Backend 추상화 레이어 생성

### 1.1 PoseBackend 인터페이스 정의
**File**: `Sports2D/Utilities/pose_backend.py` (신규)

```python
from abc import ABC, abstractmethod
from typing import Tuple, Optional
import numpy as np

class PoseBackend(ABC):
    """Pose estimation backend interface.

    All pose estimation backends must implement this interface.
    This ensures consistent behavior across RTMLib, SynthPose, and future backends.
    """

    @abstractmethod
    def __call__(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate poses in a frame.

        Args:
            frame: BGR image array (H, W, 3)

        Returns:
            keypoints: (N_persons, N_keypoints, 2) - (x, y) coordinates
            scores: (N_persons, N_keypoints) - confidence scores
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset tracker state (for new video)."""
        pass

    @property
    @abstractmethod
    def skeleton_tree(self):
        """Return anytree Node hierarchy for skeleton structure."""
        pass

    @property
    @abstractmethod
    def num_keypoints(self) -> int:
        """Number of keypoints this backend produces."""
        pass

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Backend identifier string ('rtmlib' or 'synthpose')."""
        pass

    @property
    @abstractmethod
    def keypoint_names(self) -> list:
        """List of keypoint names in order."""
        pass
```

### 1.2 RTMLibBackend 구현 (상세)
**File**: `Sports2D/Utilities/pose_backend.py`

```python
class RTMLibBackend(PoseBackend):
    """RTMLib pose estimation backend using ONNX models via Pose2Sim."""

    def __init__(self, config_dict: dict):
        """
        Initialize RTMLib backend.

        Args:
            config_dict: Full configuration dictionary with 'pose' section

        Wraps:
            - setup_model_class_mode() -> pose_model, ModelClass, mode
            - setup_backend_device() -> backend, device
            - setup_pose_tracker() -> PoseTracker
        """
        pose_config = config_dict['pose']

        # 1. Model and mode setup
        self._pose_model, self._ModelClass, self._mode = setup_model_class_mode(
            pose_config['pose_model'],
            pose_config['mode'],
            config_dict
        )

        # 2. Backend and device setup (ONNX providers)
        self._backend, self._device = setup_backend_device(
            pose_config.get('backend', 'auto'),
            pose_config.get('device', 'auto')
        )

        # 3. Tracker initialization with retry for multi-threading
        det_frequency = pose_config.get('det_frequency', 4)
        try:
            self._tracker = setup_pose_tracker(
                self._ModelClass, det_frequency, self._mode,
                False, self._backend, self._device
            )
        except Exception:
            import time
            time.sleep(3)
            self._tracker = setup_pose_tracker(
                self._ModelClass, det_frequency, self._mode,
                False, self._backend, self._device
            )

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
        """Return keypoint count (26 for HALPE_26, 133 for COCO_133, etc.)."""
        from anytree import PreOrderIter
        return sum(1 for node in PreOrderIter(self._pose_model) if node.id is not None)

    @property
    def backend_name(self) -> str:
        return 'rtmlib'

    @property
    def keypoint_names(self) -> list:
        """Return keypoint names from skeleton tree."""
        from anytree import PreOrderIter
        return [node.name for node in PreOrderIter(self._pose_model) if node.id is not None]
```

### 1.3 SynthPoseBackend 구현 (상세)
**File**: `Sports2D/Utilities/pose_backend.py`

```python
class SynthPoseBackend(PoseBackend):
    """SynthPose backend using VitPose models with PyTorch."""

    def __init__(self, config_dict: dict):
        """
        Initialize SynthPose backend.

        Args:
            config_dict: Full configuration dictionary

        Device Selection Logic:
            1. If config specifies 'cuda'/'cpu'/'mps' -> use that
            2. If config specifies 'auto' -> auto-detect:
               - torch.cuda.is_available() -> 'cuda'
               - torch.backends.mps.is_available() -> 'mps'
               - else -> 'cpu'
        """
        from Sports2D.Utilities.synthpose_tracker import SynthPosePoseTracker
        from Sports2D.Utilities.synthpose_skeleton import (
            create_synthpose_skeleton,
            SYNTHPOSE_KEYPOINT_NAMES
        )

        pose_config = config_dict['pose']
        pose_model = pose_config['pose_model']

        # Determine VitPose size from pose_model name
        self._mode = 'huge' if pose_model.lower() == 'synthpose' else 'base'

        # Device selection (config takes priority, 'auto' triggers detection)
        device = pose_config.get('device', 'auto')

        # Initialize tracker
        self._tracker = SynthPosePoseTracker(
            mode=self._mode,
            device=device,  # Pass through - tracker handles 'auto' detection
            det_frequency=pose_config.get('det_frequency', 4),
            keypoint_likelihood_threshold=pose_config.get('keypoint_likelihood_threshold', 0.3),
            detector=pose_config.get('synthpose_detector', 'yolox')
        )

        # Store skeleton tree
        self._skeleton_tree = create_synthpose_skeleton()
        self._keypoint_names = list(SYNTHPOSE_KEYPOINT_NAMES)

    def __call__(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Run pose estimation. Returns (keypoints, scores)."""
        return self._tracker(frame)

    def reset(self) -> None:
        """Reset tracker state."""
        self._tracker.frame_count = 0
        self._tracker.prev_boxes = None

    @property
    def skeleton_tree(self):
        return self._skeleton_tree

    @property
    def num_keypoints(self) -> int:
        return 52  # Always 52 for SynthPose

    @property
    def backend_name(self) -> str:
        return 'synthpose'

    @property
    def keypoint_names(self) -> list:
        return self._keypoint_names
```

### 1.4 Backend Factory 함수 (상세)
**File**: `Sports2D/Utilities/pose_backend.py`

```python
def create_pose_backend(config_dict: dict) -> PoseBackend:
    """
    Factory function to create pose backend from config.

    Args:
        config_dict: Full configuration dictionary

    Returns:
        Configured PoseBackend instance (RTMLibBackend or SynthPoseBackend)

    Raises:
        ValueError: If pose_model is invalid
        ImportError: If SynthPose dependencies not installed
    """
    pose_model = config_dict['pose']['pose_model'].lower()

    if pose_model in ['synthpose', 'synthpose_base']:
        try:
            return SynthPoseBackend(config_dict)
        except ImportError as e:
            raise ImportError(
                f"SynthPose requires additional dependencies: {e}\n"
                "Install with: pip install sports2d[synthpose]"
            )
    else:
        return RTMLibBackend(config_dict)
```

**Tasks**:
- [ ] Create `pose_backend.py` with ABC interface
- [ ] Implement `RTMLibBackend` class with retry logic
- [ ] Implement `SynthPoseBackend` class
- [ ] Implement `create_pose_backend()` factory function
- [ ] Add comprehensive docstrings and type hints
- [ ] Add unit tests for factory function

---

## Phase 2: synthpose_skeleton.py 정리

### 2.1 HALPE26 인덱스 중복 제거 (상세)

**현재 상태**: 동일한 매핑이 5번 중복 정의됨
- Line 230: `SYNTHPOSE_HALPE26_BODYWITHFEET_INDICES` 첫 정의
- Line 249: 동일한 내용 반복
- Line 272: 또 다시 반복
- Line 293: 또 다시 반복
- 추가로 여러 함수 내에서 인라인 정의

**해결책**: 모듈 상단에 단일 정의

```python
# ============================================================
# HALPE26 MAPPING - SINGLE SOURCE OF TRUTH
# ============================================================

# SynthPose 52 keypoints -> HALPE26 26 keypoints mapping
# Maps SynthPose keypoint indices to their HALPE26 equivalents
SYNTHPOSE_TO_HALPE26_MAP = {
    # COCO17 body (indices 0-16 in both)
    0: 0,    # Nose
    1: 1,    # LEye
    2: 2,    # REye
    3: 3,    # LEar
    4: 4,    # REar
    5: 5,    # LShoulder
    6: 6,    # RShoulder
    7: 7,    # LElbow
    8: 8,    # RElbow
    9: 9,    # LWrist
    10: 10,  # RWrist
    11: 11,  # LHip
    12: 12,  # RHip
    13: 13,  # LKnee
    14: 14,  # RKnee
    15: 15,  # LAnkle
    16: 16,  # RAnkle
    # Foot keypoints (SynthPose 40-47 -> HALPE26 17-25)
    40: 20,  # R5Meta -> RBigToe (closest)
    41: 23,  # L5Meta -> LBigToe (closest)
    42: 21,  # RToe -> RSmallToe
    43: 24,  # LToe -> LSmallToe
    44: 20,  # RBigToe
    45: 23,  # LBigToe
    46: 25,  # LHeel
    47: 22,  # RHeel
}

# Indices of SynthPose keypoints that map to HALPE26 bodywithfeet
SYNTHPOSE_HALPE26_BODYWITHFEET_INDICES = list(SYNTHPOSE_TO_HALPE26_MAP.keys())

# Names of HALPE26 bodywithfeet keypoints (for name-based lookup)
SYNTHPOSE_HALPE26_BODYWITHFEET_NAMES = {
    'Nose', 'LEye', 'REye', 'LEar', 'REar',
    'LShoulder', 'RShoulder', 'LElbow', 'RElbow', 'LWrist', 'RWrist',
    'LHip', 'RHip', 'LKnee', 'RKnee', 'LAnkle', 'RAnkle',
    'R5Meta', 'L5Meta', 'RToe', 'LToe', 'RBigToe', 'LBigToe', 'LHeel', 'RHeel'
}
```

### 2.2 Virtual Hip Node 동작 확인 (검증 완료)

**분석 결과**: Virtual Hip은 `process.py`에서 자동으로 처리됨

**코드 흐름** (`process.py:2070-2077`):
```python
# Add Neck and Hip if not provided
new_keypoints_names, new_keypoints_ids = keypoints_names.copy(), keypoints_ids.copy()
for kpt in ['Hip', 'Neck']:
    if kpt not in new_keypoints_names:
        person_X_flipped, person_Y, person_scores = add_neck_hip_coords(
            kpt, person_X_flipped, person_Y, person_scores,
            new_keypoints_ids, new_keypoints_names
        )
```

**`add_neck_hip_coords`** (from `Pose2Sim.common`):
- `Hip`: LHip와 RHip의 midpoint로 계산
- `Neck`: LShoulder와 RShoulder의 midpoint로 계산

**결론**: Virtual Hip (`id=None`)은 skeleton tree 구조용이며, 실제 좌표는 runtime에 계산됨. **추가 작업 불필요**.

### 2.3 파일 구조 정리 (Target)

```python
"""
synthpose_skeleton.py - SynthPose 52-keypoint skeleton definition

This module defines the SynthPose skeleton structure with 52 keypoints:
- 17 COCO keypoints (body)
- 35 additional anatomical markers (anatomical)

The skeleton is compatible with Pose2Sim's anytree-based skeleton system.
"""

# ============================================================
# 1. CONSTANTS (Single Source of Truth)
# ============================================================
SYNTHPOSE_KEYPOINT_NAMES = (...)      # 52 keypoint names
SYNTHPOSE_KEYPOINT_COLORS = (...)     # RGB colors for visualization
SYNTHPOSE_TO_HALPE26_MAP = {...}      # Mapping dictionary
SYNTHPOSE_HALPE26_BODYWITHFEET_INDICES = [...]  # Derived from map
SYNTHPOSE_HALPE26_BODYWITHFEET_NAMES = {...}    # Name set

# ============================================================
# 2. SKELETON TREE CREATION
# ============================================================
def create_synthpose_skeleton():
    """Create 52-keypoint skeleton tree structure."""
    ...

# ============================================================
# 3. MAPPING FUNCTIONS
# ============================================================
def map_synthpose_to_halpe26(keypoints, scores):
    """Convert 52 keypoints to 26 HALPE26 keypoints."""
    ...

def get_halpe26_indices():
    """Return indices of HALPE26 keypoints in SynthPose output."""
    return SYNTHPOSE_HALPE26_BODYWITHFEET_INDICES
```

**Tasks**:
- [ ] Move HALPE26 mapping to module top (single definition)
- [ ] Remove 4 duplicate constant definitions
- [ ] Add module docstring
- [ ] Reorganize functions logically
- [ ] Verify no external code depends on removed duplicates

---

## Phase 3: synthpose_tracker.py 리팩토링

### 3.1 Detector Factory 패턴 적용

**현재 문제**: `__init__`에 3개 detector 초기화 로직이 인라인됨

**해결책**: Detector selection을 별도 클래스로 분리

```python
class DetectorType(Enum):
    YOLOX = 'yolox'
    RTDETR = 'rtdetr'
    RTDETRV4 = 'rtdetrv4'

class DetectorFactory:
    """Factory for creating person detectors."""

    @staticmethod
    def create(detector_type: str, device: str, det_frequency: int):
        """
        Create a person detector.

        Args:
            detector_type: 'yolox', 'rtdetr', or 'rtdetrv4'
            device: 'cuda', 'cpu', 'mps', or 'auto'
            det_frequency: Detection frequency (every N frames)

        Returns:
            Detector callable: frame -> List[BBox]

        Raises:
            ValueError: Unknown detector type
            FileNotFoundError: RT-DETRv4 weights not found
        """
        dtype = DetectorType(detector_type.lower())

        if dtype == DetectorType.YOLOX:
            return YOLOXDetector(device, det_frequency)
        elif dtype == DetectorType.RTDETR:
            return RTDETRDetector(device)
        elif dtype == DetectorType.RTDETRV4:
            return RTDETRv4Detector(device)

        raise ValueError(f"Unknown detector type: {detector_type}")
```

### 3.2 Fallback 로직 제거 및 명확한 에러 처리

**현재 문제** (`synthpose_tracker.py:313-332`):
- RT-DETRv4 engine import 실패시 → Faster R-CNN fallback
- 사용자에게 명확한 안내 없음

**해결책**: Fallback 제거, 명확한 에러 메시지 제공

```python
def _load_rtdetrv4_detector(self):
    """Load RT-DETRv4 detector with clear error handling."""

    # Check for model weights
    checkpoint_path = self._find_rtdetrv4_checkpoint()
    if checkpoint_path is None:
        raise FileNotFoundError(
            "RT-DETRv4 checkpoint not found.\n"
            "Please download from: https://github.com/RT-DETRs/RT-DETRv4/releases\n"
            "Place in: Sports2D/models/RT-DETRv4/\n"
            "Expected files: rtdetrv4_x.pth, rtv4_x.pth, or similar"
        )

    # Check for engine module
    try:
        from engine.core import YAMLConfig
    except ImportError:
        raise ImportError(
            "RT-DETRv4 engine not installed.\n"
            "The engine/ directory must be present in Sports2D/models/RT-DETRv4/\n"
            "Alternative: Use synthpose_detector='yolox' (recommended) or 'rtdetr'"
        )

    # Load model (no fallback)
    ...
```

### 3.3 Device 선택 로직 문서화

**현재 상태** (`synthpose_tracker.py:122-130`):
- `device='auto'`일 때 torch 기반 자동 감지
- config의 명시적 device 설정 존중

**결론**: 현재 로직이 올바름. 문서화만 개선.

```python
def __init__(self, mode='huge', device='auto', ...):
    """
    Initialize SynthPose tracker.

    Args:
        device: Device for PyTorch inference
            - 'auto': Auto-detect (CUDA > MPS > CPU)
            - 'cuda': Force CUDA (raises if unavailable)
            - 'mps': Force Apple Metal (raises if unavailable)
            - 'cpu': Force CPU

        Note: This differs from RTMLib which uses ONNX providers.
        RTMLib's 'backend' parameter (onnxruntime/openvino/opencv)
        is not applicable to SynthPose.
    """
```

**Tasks**:
- [ ] Create `DetectorFactory` class
- [ ] Remove `_load_rtdetrv4_simplified()` fallback method
- [ ] Add clear error messages for missing weights/engine
- [ ] Update docstrings for device parameter
- [ ] Test error paths

---

## Phase 4: process.py 통합

### 4.1 Backend 선택 로직 교체

**현재** (`process.py:1810-1895`): 60+ lines의 분기 로직

**목표**: Factory 함수 호출로 단순화

```python
# Before (60+ lines)
use_synthpose = pose_model_name.lower() in ['synthpose', 'synthpose_base']
if use_synthpose:
    # SynthPose specific setup...
    synthpose_mode = 'huge' if pose_model_name == 'synthpose' else 'base'
    pose_model = create_synthpose_skeleton()
    # ... more setup
else:
    # RTMLib specific setup...
    pose_model, ModelClass, mode = setup_model_class_mode(...)
    backend, device = setup_backend_device(...)
    # ... more setup

# After (10 lines)
from Sports2D.Utilities.pose_backend import create_pose_backend

pose_backend = create_pose_backend(config_dict)
pose_model = pose_backend.skeleton_tree
keypoints_names = pose_backend.keypoint_names

# In processing loop:
keypoints, scores = pose_backend(frame)
```

### 4.2 그리기 함수 통합 전략 (상세)

**현재 상태**:
1. `draw_keypts()` / `draw_skel()`: Pose2Sim에서 import (RTMLib용)
2. `draw_synthpose_keypoints()` / `draw_synthpose_skeleton()`: 로컬 정의 (SynthPose용)

**통합 전략**: 로컬 SynthPose 함수만 교체, Pose2Sim import 유지

```python
def draw_pose(
    img: np.ndarray,
    keypoints: np.ndarray,  # (N_persons, N_keypoints, 2)
    scores: np.ndarray,     # (N_persons, N_keypoints)
    pose_backend: PoseBackend,
    thickness: int = 1,
    kpt_threshold: float = 0.3,
) -> np.ndarray:
    """
    Unified pose drawing function.

    Works with any backend by using skeleton_tree for connections
    and keypoint_names for styling decisions.

    For RTMLib (26/133 keypoints):
        - Delegates to Pose2Sim's draw_keypts/draw_skel

    For SynthPose (52 keypoints):
        - Uses custom styling (HALPE26 colored, others white diamonds)
    """
    if pose_backend.backend_name == 'rtmlib':
        # Use existing Pose2Sim functions
        img = draw_keypts(img, keypoints, scores, kpt_threshold)
        img = draw_skel(img, keypoints, pose_backend.skeleton_tree, thickness)
    else:
        # SynthPose: custom styling
        img = _draw_synthpose_keypoints(img, keypoints, scores,
                                         pose_backend.keypoint_names, thickness)
        img = _draw_synthpose_skeleton(img, keypoints,
                                        pose_backend.skeleton_tree, thickness)
    return img

# Private functions (renamed from current public ones)
def _draw_synthpose_keypoints(...): ...
def _draw_synthpose_skeleton(...): ...
```

**변경 범위**:
- Pose2Sim의 `draw_keypts`/`draw_skel` 사용은 유지
- 로컬 `draw_synthpose_*` 함수들을 private으로 리네임
- 통합 `draw_pose()` 함수 추가

### 4.3 변수명 일관성

| 현재 | 변경 후 | 설명 |
|------|---------|------|
| `pose_tracker` (RTMLib) | `pose_backend` | 통일 |
| `synthpose_tracker` (SynthPose) | `pose_backend` | 통일 |
| `use_synthpose` flag | 제거 | Factory가 처리 |

**Tasks**:
- [ ] Replace backend selection with `create_pose_backend()` call
- [ ] Create unified `draw_pose()` function
- [ ] Rename `draw_synthpose_*` to `_draw_synthpose_*` (private)
- [ ] Update variable names for consistency
- [ ] Remove `use_synthpose` flag
- [ ] Test both backends end-to-end

---

## Phase 5: RT-DETRv4 디렉토리 정리

### 5.1 의존성 분석 결과

**Import Chain** (`synthpose_tracker.py` → RT-DETRv4):
```
synthpose_tracker.py
    └── from engine.core import YAMLConfig
            └── engine/core/__init__.py
                    ├── from .yaml_config import YAMLConfig
                    ├── from .workspace import GLOBAL_CONFIG, register, create
                    └── from ._config import BaseConfig

            └── engine/__init__.py
                    ├── from . import optim  # Training only
                    ├── from . import data   # Training only
                    ├── from . import rtv4   # NEEDED for model
                    └── from .backbone import *  # NEEDED for model
```

**필요한 디렉토리**:
```
Sports2D/models/RT-DETRv4/
├── engine/
│   ├── __init__.py        # 수정 필요 (optim, data import 제거)
│   ├── backbone/          # KEEP - 모델 backbone
│   ├── core/              # KEEP - YAMLConfig
│   ├── misc/              # CHECK - 유틸리티
│   └── rtv4/              # KEEP - 모델 구현
├── configs/               # KEEP - 모델 설정
└── __init__.py
```

**제거할 디렉토리**:
```
Sports2D/models/RT-DETRv4/
├── engine/
│   ├── solver/            # REMOVE - Training
│   ├── optim/             # REMOVE - Training
│   └── data/              # REMOVE - Data loading/augmentation
└── tools/                 # REMOVE - 전체 디렉토리
```

### 5.2 engine/__init__.py 수정

```python
# Before
from . import optim  # Training - REMOVE
from . import data   # Training - REMOVE
from . import rtv4   # Inference - KEEP

from .backbone import *  # Inference - KEEP

# After
from . import rtv4
from .backbone import *

from .backbone import (
    get_activation,
    FrozenBatchNorm2d,
    freeze_batch_norm2d,
)
```

### 5.3 sys.path 조작 개선

**현재** (`synthpose_tracker.py:251-254`):
```python
rtdetrv4_repo_path = os.path.join(...)
if os.path.exists(rtdetrv4_repo_path) and rtdetrv4_repo_path not in sys.path:
    sys.path.insert(0, rtdetrv4_repo_path)
```

**개선**: 상대 import로 변경 또는 context manager 사용

```python
@contextmanager
def _rtdetrv4_import_context():
    """Temporarily add RT-DETRv4 to sys.path for import."""
    rtdetrv4_path = Path(__file__).parent.parent / 'models' / 'RT-DETRv4'
    if rtdetrv4_path.exists():
        sys.path.insert(0, str(rtdetrv4_path))
        try:
            yield
        finally:
            sys.path.remove(str(rtdetrv4_path))
    else:
        yield  # No-op if path doesn't exist
```

**Tasks**:
- [ ] Verify `engine/misc/` necessity (test without it)
- [ ] Remove `engine/solver/`, `engine/optim/`, `engine/data/`
- [ ] Remove `tools/` directory
- [ ] Modify `engine/__init__.py` to remove training imports
- [ ] Test RT-DETRv4 inference still works
- [ ] Consider context manager for sys.path manipulation

---

## Phase 6: 설정 및 문서화

### 6.1 Config_demo.toml 개선

```toml
[pose]
## Pose estimation model selection
## --------------------------------
## RTMLib models (ONNX-based, default):
##   - 'body_with_feet': HALPE_26 model with 26 keypoints (recommended)
##   - 'whole_body_wrist': COCO_133 model with 133 keypoints (hands/face)
##   - 'body': COCO_17 model with 17 keypoints (basic body only)
##
## SynthPose models (PyTorch-based, requires 'pip install sports2d[synthpose]'):
##   - 'synthpose': VitPose-huge with 52 keypoints (most accurate, slower)
##   - 'synthpose_base': VitPose-base with 52 keypoints (faster)
pose_model = 'body_with_feet'

## Mode selection
## --------------
## For RTMLib only (ignored for SynthPose):
##   - 'lightweight': Fastest inference, lower accuracy
##   - 'balanced': Good balance of speed and accuracy (recommended)
##   - 'performance': Highest accuracy, slower inference
mode = 'balanced'

## Device selection
## ----------------
## Works for both RTMLib and SynthPose:
##   - 'auto': Auto-detect best available (CUDA > MPS > CPU)
##   - 'cuda': Force NVIDIA GPU (requires CUDA)
##   - 'mps': Force Apple Metal (macOS only)
##   - 'cpu': Force CPU inference
device = 'auto'

## Backend selection (RTMLib only)
## -------------------------------
## ONNX execution provider (ignored for SynthPose):
##   - 'auto': Auto-select best provider
##   - 'onnxruntime': ONNX Runtime (default)
##   - 'openvino': Intel OpenVINO
##   - 'opencv': OpenCV DNN
backend = 'auto'

## Person detector for SynthPose
## -----------------------------
## Only used when pose_model='synthpose' or 'synthpose_base':
##   - 'yolox': Fast YOLOX detector from rtmlib (recommended)
##   - 'rtdetr': RT-DETR from HuggingFace transformers
##   - 'rtdetrv4': RT-DETRv4 local model (requires weights download)
synthpose_detector = 'yolox'
```

### 6.2 CLAUDE.md 업데이트

**추가할 내용**:

```markdown
### Pose Backend System

Sports2D uses a unified backend abstraction for pose estimation:

**Files**:
- `Utilities/pose_backend.py`: Backend interface and implementations
- `Utilities/synthpose_tracker.py`: SynthPose tracker implementation
- `Utilities/synthpose_skeleton.py`: 52-keypoint skeleton definition

**Backend Interface**:
```python
class PoseBackend(ABC):
    def __call__(self, frame) -> (keypoints, scores)
    def reset() -> None
    @property skeleton_tree -> anytree.Node
    @property num_keypoints -> int
    @property backend_name -> str
    @property keypoint_names -> list
```

**Creating a Backend**:
```python
from Sports2D.Utilities.pose_backend import create_pose_backend

config = {'pose': {'pose_model': 'synthpose', ...}}
backend = create_pose_backend(config)
keypoints, scores = backend(frame)
```
```

**Tasks**:
- [ ] Add detailed comments to Config_demo.toml
- [ ] Update CLAUDE.md with backend system documentation
- [ ] Document device/backend parameter differences
- [ ] Add troubleshooting section for common errors

---

## Phase 7: 테스트 코드 업데이트

### 7.1 Backend Interface 테스트

**File**: `test_pose_backend.py` (신규)

```python
import pytest
import numpy as np

class TestPoseBackendFactory:
    """Test backend creation from config."""

    def test_create_rtmlib_backend(self):
        config = {'pose': {'pose_model': 'body_with_feet', 'mode': 'balanced'}}
        backend = create_pose_backend(config)
        assert backend.backend_name == 'rtmlib'
        assert backend.num_keypoints == 26

    def test_create_synthpose_backend(self):
        config = {'pose': {'pose_model': 'synthpose', 'device': 'cpu'}}
        backend = create_pose_backend(config)
        assert backend.backend_name == 'synthpose'
        assert backend.num_keypoints == 52

    def test_invalid_model_raises(self):
        config = {'pose': {'pose_model': 'invalid_model'}}
        with pytest.raises(ValueError):
            create_pose_backend(config)

class TestPoseBackendInterface:
    """Test backend implements required interface."""

    @pytest.fixture
    def mock_frame(self):
        return np.zeros((480, 640, 3), dtype=np.uint8)

    def test_call_returns_correct_shapes(self, backend, mock_frame):
        keypoints, scores = backend(mock_frame)
        # Shape: (N_persons, N_keypoints, 2) and (N_persons, N_keypoints)
        if len(keypoints) > 0:
            assert keypoints.shape[1] == backend.num_keypoints
            assert keypoints.shape[2] == 2
            assert scores.shape[1] == backend.num_keypoints

    def test_skeleton_tree_valid(self, backend):
        from anytree import Node
        assert isinstance(backend.skeleton_tree, Node)

    def test_keypoint_names_match_count(self, backend):
        assert len(backend.keypoint_names) == backend.num_keypoints
```

### 7.2 기존 테스트 호환성 확인

**File**: `Sports2D/Utilities/tests.py`

기존 테스트가 변경 없이 통과해야 함:
- CLI 테스트
- Python API 테스트
- Config 파라미터 테스트

**Tasks**:
- [ ] Create `test_pose_backend.py` with interface tests
- [ ] Add angle calculation tests with SynthPose output
- [ ] Verify existing `tests.py` passes unchanged
- [ ] Add CI marker for SynthPose tests (optional dependencies)

---

## Implementation Order

```
Phase 1 (Backend Abstraction)  ─────────────────────┐
    ↓                                                │
Phase 2 (Skeleton Cleanup) ────┐                     │ Can start
    ↓                          │                     │ in parallel
Phase 3 (Tracker Refactor) ────┼─────────────────────┘
    ↓                          │
Phase 4 (Process Integration) ←┘
    ↓
Phase 5 (RT-DETRv4 Cleanup) ─── Independent
    ↓
Phase 6 (Documentation)
    ↓
Phase 7 (Testing)
```

**권장 실행 순서**:
1. **Phase 1** 먼저 완료 (다른 phase의 기반)
2. **Phase 2, 3** 병렬 진행 가능
3. **Phase 4**는 1, 2, 3 완료 후
4. **Phase 5**는 독립적으로 언제든 진행 가능
5. **Phase 6, 7**은 코드 변경 완료 후

---

## Risk Mitigation

### Breaking Change Prevention
- [ ] 모든 변경 전 기존 테스트 실행 확인
- [ ] CLI 인터페이스 변경 금지
- [ ] `Sports2D.process()` 시그니처 유지

### Rollback Strategy
1. 각 Phase별 별도 커밋
2. Feature branch에서 작업
3. 문제 발생시 해당 커밋 revert

### Validation Checkpoints
- [ ] Phase 1 후: `create_pose_backend()` 양 백엔드 동작 확인
- [ ] Phase 2 후: HALPE26 매핑 동작 확인
- [ ] Phase 3 후: Detector factory 동작 확인
- [ ] Phase 4 후: 전체 파이프라인 테스트 (RTMLib + SynthPose)
- [ ] Phase 5 후: RT-DETRv4 inference 테스트
- [ ] 최종: CI 테스트 전체 통과

---

## Success Criteria

| 기준 | 측정 방법 |
|------|----------|
| 단일 Factory 함수 | `create_pose_backend()` 하나로 모든 백엔드 생성 |
| HALPE26 중복 제거 | 5개 → 1개 정의로 감소 |
| 통합 그리기 함수 | `draw_pose()` 함수가 양 백엔드 처리 |
| RT-DETRv4 정리 | training 코드 제거, inference만 유지 |
| 테스트 통과 | `pytest -v Sports2D/Utilities/tests.py` 전체 통과 |
| API 호환성 | CLI 및 Python API 변경 없음 |
| 문서화 | Config 파라미터 100% 문서화 |

---

## Estimated Changes

| File | Action | Lines |
|------|--------|-------|
| `Utilities/pose_backend.py` | Create | +350 |
| `Utilities/synthpose_skeleton.py` | Refactor | -150 |
| `Utilities/synthpose_tracker.py` | Refactor | -80 |
| `process.py` | Refactor | -120 |
| `models/RT-DETRv4/engine/solver/` | Delete | -1500 |
| `models/RT-DETRv4/engine/optim/` | Delete | -500 |
| `models/RT-DETRv4/engine/data/` | Delete | -800 |
| `models/RT-DETRv4/tools/` | Delete | -1000 |
| `Demo/Config_demo.toml` | Enhance | +50 |
| `CLAUDE.md` | Update | +40 |
| `test_pose_backend.py` | Create | +150 |

**Net Effect**: ~3,500 lines 감소, 가독성/유지보수성 향상
