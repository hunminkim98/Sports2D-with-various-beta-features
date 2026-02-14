# Spec: Replace YOLOv3 + HRNet with YOLOX-X + ViTPose-H

## Overview

Replace the legacy person detector (YOLOv3/Darknet, 2018) and 2D pose estimator (HRNet-W48, 2019) in the FMPose3D demo pipeline with modern, more accurate alternatives:

- **Person Detector**: YOLOX-X (Megvii, 2021) — 99.1M params, 51.5 mAP on COCO
- **2D Pose Estimator**: ViTPose-H (ViT-Huge, 2022) — 632M params, 78.9 AP on COCO

## Architecture Decisions

### 1. Standalone PyTorch (NOT MM-ecosystem)
- No mmcv, mmdet, mmpose, mmengine dependencies
- YOLOX: Install from Megvii official repo (`pip install yolox` from source)
- ViTPose-H: Use Hugging Face Transformers (`VitPoseForPoseEstimation`)
- Keeps dependency footprint small and Windows-compatible

### 2. SORT Tracker: Retained
- Keep SORT for video temporal consistency
- For single-image mode, SORT still runs but has minimal overhead

### 3. Full Replacement (not side-by-side)
- Old `fmpose3d/lib/yolov3/` and `fmpose3d/lib/hrnet/` code is NOT deleted but is no longer imported
- New modules: `fmpose3d/lib/yolox_detector.py` and `fmpose3d/lib/vitpose_estimator.py`
- New unified pipeline: `fmpose3d/lib/gen_kpts.py` (replaces `fmpose3d/lib/hrnet/gen_kpts.py`)

### 4. Input Resolutions
- YOLOX-X: 640x640 (standard for YOLOX, up from 416 for YOLOv3)
- ViTPose-H: 256x192 (standard COCO checkpoint)

## Interface Contract (UNCHANGED)

The downstream pipeline is NOT modified:
- `gen_video_kpts()` still returns `keypoints` (M, T, 17, 2) and `scores` (M, T, 17) in **COCO 17-joint format**
- `h36m_coco_format()` in `preprocess.py` converts to H36M 17 joints (unchanged)
- FMPose3D 3D model receives H36M format (unchanged)

## New Module Design

### `fmpose3d/lib/yolox_detector.py`
- `load_yolox_model(device=None) -> model`
  - Downloads checkpoint if missing via `download_checkpoints.py`
  - Loads YOLOX-X with `get_exp(exp_name="yolox-x")`
  - Returns model on GPU/CPU
- `detect_persons(frame, model, conf_thre=0.3, nms_thre=0.45, input_size=640) -> (bboxs, scores)`
  - Letterbox resize to 640x640
  - Run inference, postprocess (NMS)
  - Filter person class (id=0) only
  - Convert output to `[x1, y1, x2, y2]` format
  - Return `bboxs` (N, 4) and `scores` (N, 1) — same format as old `yolo_human_det()`
  - Return `(None, None)` if no detections

### `fmpose3d/lib/vitpose_estimator.py`
- `load_vitpose_model(device=None) -> (model, processor)`
  - Uses HuggingFace `VitPoseForPoseEstimation.from_pretrained("usyd-community/vitpose-huge-simple")`
  - Returns model + processor
- `estimate_pose(frame, bboxs, model, processor, device=None) -> (keypoints, scores)`
  - Takes BGR frame + person bounding boxes
  - Converts bboxs from [x1,y1,x2,y2] to [x,y,w,h] for HuggingFace API
  - Runs ViTPose inference
  - Returns keypoints (N, 17, 2) in COCO format and scores (N, 17)

### `fmpose3d/lib/gen_kpts.py` (NEW — replaces hrnet/gen_kpts.py)
- `gen_from_image(frame, people_sort, det_model, pose_model, pose_processor, ...) -> (kpts, scores)`
  - Detect persons with YOLOX-X
  - Track with SORT (reuse existing `fmpose3d/lib/sort/sort.py`)
  - Estimate 2D pose with ViTPose-H
  - Return keypoints in COCO format (same shape as before)
- `gen_video_kpts(path, num_peroson=1, gen_output=False, type='image') -> (keypoints, scores)`
  - Main entry point (same signature as before)
  - Loads YOLOX-X + ViTPose-H
  - Iterates frames (image or video)
  - Returns (M, T, 17, 2) keypoints and (M, T, 17) scores

## Checkpoint Management

Update `fmpose3d/lib/checkpoint/download_checkpoints.py`:
- Change `REQUIRED_FILES` to `['yolox_x.pth']` (ViTPose uses HuggingFace auto-download)
- Add download URL for YOLOX-X: `https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_x.pth`
- HuggingFace handles ViTPose-H checkpoint caching automatically

## Dependency Changes

Add to `pyproject.toml`:
- `yolox` (from source or git)
- `transformers>=4.45.0` (for VitPoseForPoseEstimation)

Remove (no longer needed by new pipeline):
- Nothing removed — old deps may still be needed by old code paths

## Files Modified

1. **`fmpose3d/lib/gen_kpts.py`** — NEW: unified detection+pose pipeline
2. **`fmpose3d/lib/yolox_detector.py`** — NEW: YOLOX-X wrapper
3. **`fmpose3d/lib/vitpose_estimator.py`** — NEW: ViTPose-H wrapper
4. **`fmpose3d/__init__.py`** — Update import from new `gen_kpts.py`
5. **`fmpose3d/lib/checkpoint/download_checkpoints.py`** — Update for YOLOX-X checkpoint
6. **`demo/vis_in_the_wild.py`** — Update import to new pipeline
7. **`demo/vis_in_the_wild.sh`** — Update det_dim default (640)
8. **`pyproject.toml`** — Add yolox, transformers dependencies

## Acceptance Criteria

1. `gen_video_kpts()` returns (M, T, 17, 2) keypoints and (M, T, 17) scores
2. `h36m_coco_format(keypoints, scores)` succeeds without assertion errors
3. End-to-end demo produces valid 3D pose output
4. Checkpoint auto-download works for YOLOX-X
5. Both image and video modes work
6. File headers pass CI check
