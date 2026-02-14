# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FMPose3D is a research implementation of monocular 3D pose estimation via Flow Matching. It learns a conditional ODE vector field that maps Gaussian noise to 3D poses conditioned on 2D input, generates multiple hypotheses via Euler sampling, and aggregates them using RPEA (Reprojection Error Aggregation) for per-joint selection.

## Build & Development Commands

```bash
# Install with dev dependencies (pytest, black, flake8, isort)
pip install '.[dev]'

# Run all tests
python -m pytest

# Run a single test file
python -m pytest tests/test_model.py

# Run a single test
python -m pytest tests/test_model.py::test_model_forward_shape

# Format code
black --line-length 100 .
isort --profile black --line-length 100 .

# Lint
flake8 .

# Check file headers (CI enforces this)
python scripts/update_headers.py --check

# Fix missing/outdated headers
python scripts/update_headers.py

# Build package
python -m build
```

## Training & Inference

```bash
# Train on Human3.6M (requires dataset/ with .npz files)
sh scripts/FMPose3D_train.sh

# Inference with pre-trained model (requires pre_trained_models/)
sh scripts/FMPose3D_test.sh

# Demo on in-the-wild images (from demo/ directory)
sh demo/vis_in_the_wild.sh
```

## Architecture

### Package Structure

- **`fmpose3d/models/`** - Neural network architecture
  - `model_GAMLP.py` - Main model (`Model` class): GCN + Multi-head Attention + MLP blocks with Gaussian Fourier time embeddings. Input: 2D pose `(B,F,J,2)` + noisy 3D `(B,F,J,3)` + time `(B,1,1,1)` -> velocity field `(B,F,J,3)`
  - `graph_frames.py` - Graph convolution using Human3.6M skeleton topology (17 joints, 4 adjacency matrices)

- **`fmpose3d/aggregation_methods.py`** - Multi-hypothesis aggregation. Key function: `aggregation_RPEA_joint_level` which projects hypotheses to 2D, computes reprojection error, and performs top-k exponential-weighted per-joint selection

- **`fmpose3d/common/`** - Dataset loading, argument parsing, metrics
  - `arguments.py` (`opts` class) - CLI argument parser
  - `utils.py` - `mpjpe_cal`, `p_mpjpe` (Procrustes-aligned), `project_to_2d`
  - `load_data_hm36.py` (`Fusion` class) - PyTorch DataLoader for Human3.6M
  - `h36m_dataset.py` - Dataset parser (subjects S1,S5,S6,S7,S8 train / S9,S11 test)

- **`fmpose3d/lib/`** - External detection pipeline (used by demo only)
  - `hrnet/` - HRNet 2D pose detector
  - `yolov3/` - Human bounding box detection
  - `sort/` - Multi-object tracking
  - `preprocess.py` - COCO-to-H36M keypoint format conversion

- **`scripts/FMPose3D_main.py`** - Training/inference entry point. Supports loading model class from either the installed package or a local file path via `--model_path`

- **`fmpose3d/animals/`** - Parallel subpackage for animal pose estimation with its own models and data pipeline

### Flow Matching Pipeline

1. **Training**: Interpolate between noise `y0 ~ N(0,I)` and ground-truth 3D pose `y1` at random time `t`. Model predicts the velocity field. Loss: MSE between predicted and target velocity.
2. **Inference**: Sample initial noise, solve ODE with Euler method over `eval_sample_steps` (default 3) steps. Repeat for `num_hypothesis` samples.
3. **Aggregation**: RPEA projects each hypothesis to 2D, compares against observed 2D keypoints, selects best per-joint with exponential weighting.

### Model Tensor Shapes

All tensors follow `(B, F, J, C)` convention: Batch, Frames (usually 1), Joints (17), Channels (2 for 2D, 3 for 3D).

## File Header Requirement

All `.py` files (except trivial `__init__.py`) must have this header:

```python
"""
FMPose3D: monocular 3D Pose Estimation via Flow Matching

Official implementation of the paper:
"FMPose3D: monocular 3D Pose Estimation via Flow Matching"
by Ti Wang, Xiaohang Yu, and Mackenzie Weygandt Mathis
Licensed under Apache 2.0
"""
```

CI checks this on every push/PR. Fix with `python scripts/update_headers.py`.

## CI Workflows

- **build.yaml** - Runs `pytest` on Ubuntu + Windows (Python 3.10, PyTorch 2.4.0 CPU)
- **check-headers.yml** - Validates Apache 2.0 headers on all Python files
- **codespell.yml** - Spell checking (ignore list: `fmpose,mpjpe,uvd,xyz,hm36,cpn,dbb`)
- **release-pypi.yml** - Publishes to PyPI on `v*.*.*` tags

## Key Dependencies

PyTorch 2.4.1, torchvision 0.19.1, timm (vision transformers), einops (tensor rearrangement), deeplabcut 3.0.0rc13. Version is defined in `fmpose3d/__init__.py` (`__version__`).
