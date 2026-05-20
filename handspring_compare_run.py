from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import toml

from Sports2D import Sports2D


REPO_ROOT = Path(__file__).resolve().parent
VIDEO = REPO_ROOT / "Sports2D" / "Demo" / "handspring.mp4"
CONFIG = REPO_ROOT / "Sports2D" / "Demo" / "Config_demo.toml"
OUT_ROOT = REPO_ROOT / "comparison_runs" / "handspring"


RUNS = {
    "rtmpose": {
        "result_dir": OUT_ROOT / "rtmpose_wholebody_performance",
        "pose": {
            "pose_model": "whole_body",
            "mode": "performance",
            "backend": "onnxruntime",
            "device": "cuda",
            "det_frequency": 1,
        },
    },
    "synthpose": {
        "result_dir": OUT_ROOT / "synthpose_huge_yolox",
        "pose": {
            "pose_model": "synthpose",
            "mode": "performance",
            "synthpose_model_size": "huge",
            "synthpose_detector": "yolox",
            "backend": "onnxruntime",
            "device": "cuda",
            "det_frequency": 1,
        },
    },
    "sapiens2_5b": {
        "result_dir": OUT_ROOT / "sapiens2_5b_yolox",
        "pose": {
            "pose_model": "sapiens2",
            "sapiens2_model_size": "5b",
            "sapiens2_bbox_source": "yolox",
            "sapiens2_yolox_model_size": "x",
            "sapiens2_inference_dtype": "bfloat16",
            "sapiens2_flip_test": False,
            "backend": "onnxruntime",
            "device": "cuda",
            "det_frequency": 1,
        },
    },
    "sapiens2_5b_original308": {
        "result_dir": OUT_ROOT / "sapiens2_5b_yolox_original308",
        "base": {
            "calculate_angles": False,
            "person_ordering_method": "first_detected",
            "save_angles": False,
        },
        "pose": {
            "pose_model": "sapiens2",
            "sapiens2_model_size": "5b",
            "sapiens2_keypoint_schema": "sapiens2_308",
            "sapiens2_bbox_source": "yolox",
            "sapiens2_yolox_model_size": "x",
            "sapiens2_inference_dtype": "bfloat16",
            "sapiens2_flip_test": False,
            "backend": "onnxruntime",
            "device": "cuda",
            "det_frequency": 1,
        },
    },
}

COMPARISON_SOURCES = [
    ("RTMPose whole_body", "rtmpose"),
    ("SynthPose huge", "synthpose"),
    ("Sapiens2 5B native 308", "sapiens2_5b_original308"),
]
COMPARISON_OUTPUT = (
    OUT_ROOT / "handspring_rtmpose_synthpose_sapiens2_5b_original308_comparison.mp4"
)


def build_config(run_name: str) -> dict:
    run = RUNS[run_name]
    config = toml.load(CONFIG)

    base = config.setdefault("base", {})
    base["video_input"] = [str(VIDEO)]
    base["result_dir"] = str(run["result_dir"])
    base["nb_persons_to_detect"] = 1
    base["person_ordering_method"] = "highest_likelihood"
    base["show_realtime_results"] = False
    base["save_vid"] = True
    base["save_img"] = False
    base["save_pose"] = True
    # Keep comparison videos pose-only so RTMPose, SynthPose, and Sapiens2 can
    # be compared directly without Sports2D angle text/arc or bbox overlays.
    base["calculate_angles"] = False
    base["save_angles"] = False
    base["time_range"] = []
    base["compare"] = False
    base.update(run.get("base", {}))

    pose = config.setdefault("pose", {})
    pose.update(run["pose"])
    pose["manual_roi"] = False
    pose.pop("_manual_person_roi", None)
    pose.pop("_manual_ball_roi", None)
    pose["detect_ball"] = False
    pose["ball_detector_backend"] = "same"
    pose["sam3_show_realtime_masks"] = False
    pose["draw_person_bounding_boxes"] = False
    pose["person_detection_threshold"] = 0.3
    pose["keypoint_likelihood_threshold"] = 0.3

    conversion = config.setdefault("px_to_meters_conversion", {})
    conversion["to_meters"] = False
    conversion["make_c3d"] = False
    conversion["save_calib"] = False

    kinematics = config.setdefault("kinematics", {})
    kinematics["do_ik"] = False
    kinematics["use_augmentation"] = False
    kinematics["inverse_dynamics"] = False

    motion = config.setdefault("motion", {})
    motion["vertical_jump"] = False

    angles = config.setdefault("angles", {})
    angles["fontSize"] = 0.2

    return config


def _processed_video_path(run_name: str) -> Path:
    return (
        RUNS[run_name]["result_dir"]
        / "handspring_Sports2D"
        / "handspring_Sports2D.mp4"
    )


def _resize_panel(frame, panel_width: int, panel_height: int):
    resized = cv2.resize(frame, (panel_width, panel_height), interpolation=cv2.INTER_AREA)
    return resized


def _draw_label(frame, label: str):
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 42), (0, 0, 0), -1)
    cv2.putText(
        frame,
        label,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return frame


def compose_original308_comparison() -> Path:
    sources = []
    for label, run_name in COMPARISON_SOURCES:
        path = _processed_video_path(run_name)
        if not path.exists():
            raise FileNotFoundError(
                f"Missing processed video for {run_name}: {path}. "
                f"Run `python handspring_compare_run.py {run_name}` first."
            )
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open processed video: {path}")
        sources.append((label, path, cap))

    fps_values = [cap.get(cv2.CAP_PROP_FPS) or 30.0 for _, _, cap in sources]
    frame_counts = [int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) for _, _, cap in sources]
    widths = [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0) for _, _, cap in sources]
    heights = [int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0) for _, _, cap in sources]
    frame_count = min(frame_counts)
    if frame_count <= 0:
        raise RuntimeError(f"Could not determine comparison frame count: {frame_counts}")
    panel_width = min(width for width in widths if width > 0)
    panel_height = min(height for height in heights if height > 0)
    fps = fps_values[0]

    COMPARISON_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(COMPARISON_OUTPUT),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (panel_width * len(sources), panel_height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create comparison video: {COMPARISON_OUTPUT}")

    try:
        for _ in range(frame_count):
            panels = []
            for label, _, cap in sources:
                ok, frame = cap.read()
                if not ok:
                    frame = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
                panel = _resize_panel(frame, panel_width, panel_height)
                panels.append(_draw_label(panel, label))
            writer.write(np.hstack(panels))
    finally:
        writer.release()
        for _, _, cap in sources:
            cap.release()

    return COMPARISON_OUTPUT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", choices=sorted(RUNS) + ["compose_original308"])
    args = parser.parse_args()

    if args.run == "compose_original308":
        output = compose_original308_comparison()
        print(output)
        return

    if not VIDEO.exists():
        raise FileNotFoundError(VIDEO)
    RUNS[args.run]["result_dir"].mkdir(parents=True, exist_ok=True)
    Sports2D.process(build_config(args.run))


if __name__ == "__main__":
    main()
