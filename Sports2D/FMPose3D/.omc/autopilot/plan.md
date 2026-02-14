# Implementation Plan: YOLOX-X + ViTPose-H Replacement

## Step 1: Install Dependencies (sequential, first)
- Install YOLOX from source: `pip install git+https://github.com/Megvii-BaseDetection/YOLOX.git`
- Install/upgrade transformers: `pip install "transformers>=4.45.0"`
- Verify imports work: `python -c "from yolox.exp import get_exp; from transformers import VitPoseForPoseEstimation"`

## Step 2: Create YOLOX-X Detector Wrapper (parallel with Step 3)
**File**: `fmpose3d/lib/yolox_detector.py`

Functions:
- `load_yolox_model(checkpoint_path=None, device=None)` → model
  - Uses `get_exp(exp_name="yolox-x")` to create model
  - Loads checkpoint from cache dir
  - Returns eval model on device
- `detect_persons(frame, model, conf_thre=0.3, nms_thre=0.45, input_size=(640, 640))` → (bboxs, scores)
  - Preprocess with `ValTransform(legacy=False)` + letterbox to 640x640
  - Run model inference
  - `postprocess()` with NMS
  - Filter class_id == 0 (person)
  - Scale boxes back to original image size
  - Return bboxs (N,4) as [x1,y1,x2,y2], scores (N,1)
  - Return (None, None) if no detections

## Step 3: Create ViTPose-H Estimator Wrapper (parallel with Step 2)
**File**: `fmpose3d/lib/vitpose_estimator.py`

Functions:
- `load_vitpose_model(device=None)` → (model, processor)
  - `VitPoseForPoseEstimation.from_pretrained("usyd-community/vitpose-huge-simple")`
  - `AutoProcessor.from_pretrained("usyd-community/vitpose-huge-simple")`
  - Move model to device, eval mode
- `estimate_pose(frame_rgb, bboxs, model, processor, device=None)` → (keypoints, scores)
  - Convert bboxs from [x1,y1,x2,y2] to [x,y,w,h] for HuggingFace API
  - `processor(image, boxes=[bboxs_xywh], return_tensors="pt")`
  - Run model inference
  - `processor.post_process_pose_estimation(outputs, boxes=[bboxs_xywh])`
  - Extract keypoints (N,17,2) and scores (N,17)
  - Return in same format as old pipeline

## Step 4: Create Unified Pipeline (depends on Steps 2+3)
**File**: `fmpose3d/lib/gen_kpts.py`

Functions:
- `gen_from_image(frame, people_sort, det_model, pose_model, pose_processor, det_conf=0.3, num_peroson=1)` → (kpts, scores)
  - Call `detect_persons()` from yolox_detector
  - Handle None detections (fallback to previous frame's bboxes, same pattern as old code)
  - Track with SORT `people_sort.update(bboxs)`
  - Select top `num_peroson` tracked boxes
  - Call `estimate_pose()` from vitpose_estimator
  - Return kpts (num_peroson, 17, 2) and scores (num_peroson, 17)

- `gen_video_kpts(path, num_peroson=1, gen_output=False, type='image')` → (keypoints, scores)
  - Load YOLOX-X model
  - Load ViTPose-H model
  - Initialize SORT tracker
  - Process image or video frames
  - Collect results, transpose to (M, T, 17, 2) and (M, T, 17)
  - Return same format as old pipeline

## Step 5: Update Checkpoint Downloads (parallel with Step 4)
**File**: `fmpose3d/lib/checkpoint/download_checkpoints.py`

Changes:
- Update `REQUIRED_FILES` to `['yolox_x.pth']` (ViTPose uses HF auto-download)
- Add YOLOX-X download URL (GitHub releases)
- Update `download_folder()` to download individual files by URL
- Keep `get_checkpoint_path()` API unchanged

## Step 6: Update Imports & Integration (depends on Step 4)
**Files**:
- `fmpose3d/__init__.py` — Change import from `.lib.hrnet.gen_kpts` to `.lib.gen_kpts`
- `demo/vis_in_the_wild.py` — Change `from fmpose3d.lib.hrnet.gen_kpts import gen_video_kpts as hrnet_pose` to `from fmpose3d.lib.gen_kpts import gen_video_kpts`
- Remove `ensure_checkpoints()` call at top of vis_in_the_wild.py (YOLOX checkpoint handled inside gen_kpts)

## Step 7: Update pyproject.toml
- Add `transformers>=4.45.0` to dependencies
- Note: YOLOX installed from git, not from PyPI (document in README)

## Step 8: Add File Headers
- All new `.py` files must include the Apache 2.0 header

## Execution Strategy

| Step | Agent | Parallel? |
|------|-------|-----------|
| 1 | bash (direct) | First |
| 2 + 3 | executor-high (opus) x2 | Parallel |
| 4 | executor-high (opus) | After 2+3 |
| 5 + 6 + 7 | executor (sonnet) | Parallel after 4 |
| 8 | executor-low (haiku) | Last |
