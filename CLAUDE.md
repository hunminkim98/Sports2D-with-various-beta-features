# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Sports2D is a Python tool that computes 2D joint positions, joint angles, and segment angles from video or webcam input. It uses RTMLib for pose estimation and optionally integrates with OpenSim for inverse kinematics.

## Build and Development Commands

### Installation
```bash
# Quick install from PyPI
pip install sports2d

# Install from source (development)
git clone https://github.com/davidpagnon/sports2d.git
cd sports2d
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
# Run the test suite with pytest
pytest -v Sports2D/Utilities/tests.py

# Or use the test entry point
tests_sports2d
```

### Linting
```bash
# Check for syntax errors
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics

# Full lint check
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

## Architecture Overview

### Core Pipeline Flow
1. **Sports2D.py** (`Sports2D/Sports2D.py`): Entry point with CLI argument parsing, configuration management, and the `process()` function that orchestrates video processing
2. **process.py** (`Sports2D/process.py`): Main processing logic with `process_fun()` that handles:
   - Video/webcam reading
   - Pose estimation setup (RTMLib or SynthPose)
   - Person tracking and selection
   - Angle computation
   - Result visualization and output

### Key Dependencies
- **Pose2Sim**: Heavy reuse of utilities from the [Pose2Sim](https://github.com/perfanalytics/pose2sim) project for:
  - Skeleton definitions (`Pose2Sim.skeletons`)
  - Filtering algorithms (`Pose2Sim.filtering`)
  - Calibration file handling (`Pose2Sim.calibration`)
  - Pose estimation setup (`Pose2Sim.poseEstimation`)
- **RTMLib**: Primary pose estimation backend using ONNX models
- **OpenSim** (optional): For biomechanically accurate inverse kinematics

### Skeleton/Pose Model Support
- **RTMLib models**: body_with_feet (HALPE_26), whole_body, body (COCO_17), hand, face, animal
- **SynthPose**: VitPose-huge from Stanford MIMI (52 keypoints mapped to HALPE_26)
- Custom skeletons can be defined in the TOML config file

### Configuration System
The configuration uses TOML files with these main sections:
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
- `common.py`: Angle computation dictionaries, marker Z positions, helper functions
- `tests.py`: Test workflow covering CLI and Python API
- `synthpose_tracker.py` / `synthpose_skeleton.py`: SynthPose VitPose integration

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
- Person height reference
- Camera-to-person distance, focal length, or field of view
- Calibration file with camera intrinsics/extrinsics

## Python Version Requirements
- Requires Python >=3.9 (tested on 3.10, 3.11, 3.12)
