"""Quick test for the new YOLOX-X + ViTPose-H pipeline."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from fmpose3d.lib.yolox_detector import load_yolox_model, detect_persons
from fmpose3d.lib.vitpose_estimator import load_vitpose_model, estimate_pose
from fmpose3d.lib.gen_kpts import gen_video_kpts
from fmpose3d.lib.preprocess import h36m_coco_format
import cv2

print("=" * 60)
print("TEST 1: YOLOX-X person detection")
print("=" * 60)
det_model = load_yolox_model()
print(f"YOLOX-X model loaded on {next(det_model.parameters()).device}")

img_path = os.path.join(os.path.dirname(__file__), '..', 'demo', 'images', 'running.png')
frame = cv2.imread(img_path)
print(f"Image shape: {frame.shape}")

bboxs, scores = detect_persons(frame, det_model)
if bboxs is not None:
    print(f"Detected {len(bboxs)} person(s)")
    print(f"Bboxs shape: {bboxs.shape}, Scores shape: {scores.shape}")
    print(f"Bboxs: {bboxs}")
    print(f"Scores: {scores.flatten()}")
else:
    print("ERROR: No persons detected!")
    sys.exit(1)

print()
print("=" * 60)
print("TEST 2: ViTPose-H pose estimation")
print("=" * 60)
pose_model, pose_processor = load_vitpose_model()
print("ViTPose-H model loaded")

keypoints, kp_scores = estimate_pose(frame, bboxs, pose_model, pose_processor)
print(f"Keypoints shape: {keypoints.shape}")  # Expected: (N, 17, 2)
print(f"Scores shape: {kp_scores.shape}")      # Expected: (N, 17)
print(f"Keypoints[0] sample (first 5 joints): {keypoints[0][:5]}")
print(f"Scores[0] sample (first 5 joints): {kp_scores[0][:5]}")

print()
print("=" * 60)
print("TEST 3: Full gen_video_kpts pipeline")
print("=" * 60)
full_kpts, full_scores = gen_video_kpts(img_path, num_peroson=1, type='image')
print(f"Full keypoints shape: {full_kpts.shape}")   # Expected: (1, 1, 17, 2)
print(f"Full scores shape: {full_scores.shape}")     # Expected: (1, 1, 17)

print()
print("=" * 60)
print("TEST 4: COCO to H36M conversion")
print("=" * 60)
h36m_kpts, h36m_scores, valid_frames = h36m_coco_format(full_kpts, full_scores)
print(f"H36M keypoints shape: {h36m_kpts.shape}")
print(f"H36M scores shape: {h36m_scores.shape}")
print(f"Valid frames: {valid_frames}")

print()
print("=" * 60)
print("ALL TESTS PASSED!")
print("=" * 60)
