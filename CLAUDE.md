# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Sports2D is a Python tool that computes 2D joint positions, joint angles, and segment angles from video or webcam input. It uses RTMLib or SynthPose for pose estimation and optionally integrates with OpenSim for inverse kinematics.

## Build and Development Commands

### Installation
```bash
# Quick install from PyPI
pip install sports2d

# Install from source (development)
pip install .

# Full install with OpenSim (required for inverse kinematics)
conda create -n Sports2D python=3.12 -y
conda activate Sports2D
conda install -c opensim-org opensim -y
pip install sports2d

# Install with SynthPose support (VitPose models)
pip install sports2d[synthpose]
```

### Running Tests
```bash
# Run the full test suite
pytest -v Sports2D/Utilities/tests.py

# Run tests with output capture (as CI does)
pytest -v Sports2D/Utilities/tests.py --capture=sys

# Use the test entry point
tests_sports2d
```

### Linting
```bash
# Check for syntax errors (CI blocking)
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics

# Full lint check (non-blocking warnings)
flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
```

### Running the Application
```bash
# Run demo with default parameters
sports2d

# Run on custom video
sports2d --video_input path_to_video.mp4

# Run with config file
sports2d --config Config_demo.toml

# Run from Python
from Sports2D import Sports2D
Sports2D.process(config_dict)
```

### GPU Acceleration (Development)
```bash
# Check CUDA compatibility
nvidia-smi

# Install PyTorch with CUDA (example for CUDA 12.4)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install ONNX Runtime with GPU
pip uninstall onnxruntime
pip install onnxruntime-gpu

# Verify GPU setup
python -c 'import torch; print(torch.cuda.is_available())'
python -c 'import onnxruntime as ort; print(ort.get_available_providers())'
```

## Architecture Overview

### Core Pipeline Flow
1. **Sports2D.py** (`Sports2D/Sports2D.py`): Entry point with CLI argument parsing, configuration management, and the `process()` function that orchestrates video processing
2. **process.py** (`Sports2D/process.py`): Main processing logic with `process_fun()` that handles:
   - Video/webcam reading
   - Pose estimation setup (RTMLib or SynthPose)
   - Person tracking and selection
   - Angle computation
   - Result visualization and output

### Key Entry Points
- `Sports2D.main()` - CLI entry point, parses args and calls `process()`
- `Sports2D.process(config_dict)` - Main API, accepts config dictionary
- `process.process_fun()` - Core video processing loop
- `common.angle_dict` - Joint/segment angle definitions

### Key Dependencies
- **Pose2Sim** (>=0.10.40): Heavy reuse for skeleton definitions (`Pose2Sim.skeletons`), filtering (`Pose2Sim.filtering`), calibration (`Pose2Sim.calibration`), and pose estimation setup (`Pose2Sim.poseEstimation`)
- **RTMLib**: Primary pose estimation backend using ONNX models
- **OpenSim** (optional): For biomechanically accurate inverse kinematics

### Pose Estimation Backends

**RTMLib (default):**
- body_with_feet (HALPE_26) - most common, 26 keypoints
- whole_body_wrist (COCO_133) - includes hands/face
- body (COCO_17) - basic body only
- Modes: `lightweight`, `balanced`, `performance`

**SynthPose (optional, requires `[synthpose]` extras):**
- VitPose-huge/base from Stanford MIMI
- 52 keypoints (17 COCO + 35 anatomical markers)
- Person detectors: `yolox` (recommended), `rtdetr`, `rtdetrv4`
- Files: `Utilities/pose_backend.py`, `Utilities/synthpose_tracker.py`, `Utilities/synthpose_skeleton.py`
- RT-DETRv4 engine (inference-only): `Sports2D/models/RT-DETRv4/engine/`

### Configuration System
The configuration uses TOML files (see `Demo/Config_demo.toml` for full reference):
- `[base]`: Video input, person detection, output settings
- `[pose]`: Pose model, mode, detection frequency, tracking
- `[px_to_meters_conversion]`: Calibration, perspective correction
- `[angles]`: Joint/segment angle selection and display
- `[post-processing]`: Interpolation, filtering options
- `[kinematics]`: OpenSim inverse kinematics settings

### Output Formats
- **TRC files**: Joint coordinates (OpenSim-compatible)
- **MOT files**: Joint angles (OpenSim-compatible)
- **C3D files**: Motion capture format
- **Video/images**: Annotated with skeleton overlay and angles
- **Calibration TOML**: Camera parameters for Pose2Sim

### Utilities Module (`Sports2D/Utilities/`)
- `common.py`: Angle computation dictionaries (`angle_dict`), marker Z positions, helper functions
- `tests.py`: Test workflow covering CLI and Python API
- `pose_backend.py`: **Backend abstraction layer** - unified interface for pose estimation
- `synthpose_tracker.py`: SynthPose person tracking with YOLOX/RT-DETR/RT-DETRv4 detectors
- `synthpose_skeleton.py`: 52-keypoint skeleton definition and HALPE_26 mapping

### Pose Backend System

Sports2D uses a unified backend abstraction (`pose_backend.py`) for pose estimation:

**Interface** (`PoseBackend` ABC):
```python
class PoseBackend(ABC):
    def __call__(self, frame) -> (keypoints, scores)  # (N, K, 2), (N, K)
    def reset() -> None
    @property skeleton_tree -> anytree.Node
    @property num_keypoints -> int
    @property backend_name -> str  # 'rtmlib' or 'synthpose'
    @property keypoint_names -> List[str]
```

**Implementations**:
- `RTMLibBackend`: ONNX-based pose estimation via Pose2Sim/rtmlib
- `SynthPoseBackend`: PyTorch-based VitPose estimation

**Factory Function**:
```python
from Sports2D.Utilities.pose_backend import create_pose_backend

config = {'pose': {'pose_model': 'synthpose', 'device': 'auto', ...}}
backend = create_pose_backend(config)
keypoints, scores = backend(frame)
```

**Device/Backend Parameter Differences**:
| Parameter | RTMLib | SynthPose |
|-----------|--------|-----------|
| `device` | Affects ONNX provider | Affects PyTorch device (cuda/mps/cpu) |
| `backend` | ONNX provider (onnxruntime/openvino/opencv) | **Ignored** (always PyTorch) |
| `mode` | Model quality (lightweight/balanced/performance) | VitPose selection: performance→huge, balanced→base, lightweight→base+warning |
| `synthpose_detector` | **Ignored** | Person detector (yolox/rtdetr/rtdetrv4) |

**Ball Detection (`detect_ball=true`)**:
- Shared option: `ball_detection_threshold` controls confidence cutoff for sports-ball candidates (default `0.1`).
- RTMLib backend keeps person pose detection unchanged and runs a separate COCO multiclass detector for ball metadata.
- SynthPose backend applies separate thresholds for person vs ball filtering in YOLOX/RT-DETR/RT-DETRv4 paths.

## Key Implementation Details

### Person Tracking
Two tracking modes available:
- `sports2d`: Default lightweight tracker using distance-based association
- `deepsort`: More robust for crowded scenes but slower

### Angle Computation
Joint and segment angles are defined in `angle_dict` in `common.py`:
- 4-point angles (e.g., ankle dorsiflexion)
- 3-point angles (e.g., knee flexion)
- 2-point segment angles (e.g., shank angle from horizontal)

### Pixel-to-Meters Conversion
Handles perspective correction using:
- Person height reference (`--first_person_height`)
- Camera-to-person distance, focal length, or field of view (`--perspective_unit`, `--perspective_value`)
- Calibration file with camera intrinsics/extrinsics (`--calib_file`)

## CI/CD Pipeline
- **Platforms**: Ubuntu, Windows, macOS (latest)
- **Python versions**: 3.10, 3.11, 3.12
- **Required checks**: flake8 syntax errors (E9, F63, F7, F82)
- **Test timeout**: 20 minutes
- Tests require OpenSim conda environment

## Version Management
Uses `setuptools-scm` to auto-generate version from git tags. No manual version bumping needed.

## Python Version Requirements
- Requires Python >=3.9 (tested on 3.10, 3.11, 3.12)
