#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sapiens2 pose-estimation backend for Sports2D.

This adapter keeps the heavyweight Sapiens2 stack optional and lazy.  When the
Sports2D config selects ``pose.pose_model = 'sapiens2'``, the backend loads the
local/nested Sapiens2 checkout and runs its 308-keypoint top-down pose model.
The output schema is configurable: Sports2D can either map the body/foot subset
into HALPE_26 for the existing biomechanics pipeline, or keep the original
Sapiens2 308-keypoint tensor for dense visual comparison.
"""

from __future__ import annotations

import copy
import importlib.util
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

from Sports2D.Utilities.manual_roi import (
    expand_roi_xyxy,
    normalize_manual_roi_mode,
    normalize_roi_xyxy,
    roi_from_boxes_xyxy,
)
from Sports2D.Utilities.pose_backend import PERSON_CLASS_ID, PoseBackend
from Sports2D.Utilities.synthpose_skeleton import HALPE26_KEYPOINT_NAMES


SAPIENS2_MODEL_SIZES = ("0.4b", "0.8b", "1b", "5b")
SAPIENS2_DEFAULT_DATASET = "shutterstock_goliath_3po"
SAPIENS2_CONFIG_TEMPLATE = (
    "sapiens/pose/configs/keypoints308/{dataset}/"
    "sapiens2_{size}_keypoints308_{dataset}-1024x768.py"
)
SAPIENS2_DEFAULT_DETECTOR_CONFIG = (
    "sapiens/pose/tools/vis/rtmdet_m_640-8xb32_coco-person.py"
)
SAPIENS2_POSE_REPO_TEMPLATE = "facebook/sapiens2-pose-{size}"
SAPIENS2_DETECTOR_REPO = "facebook/sapiens-pose-bbox-detector"
SAPIENS2_SUPPORTED_BBOX_SOURCES = {"detector", "manual_roi", "full_frame", "yolox"}
SAPIENS2_KEYPOINT_SCHEMA_HALPE26 = "halpe26"
SAPIENS2_KEYPOINT_SCHEMA_ORIGINAL = "sapiens2_308"
SAPIENS2_KEYPOINT_SCHEMA_ALIASES = {
    "": SAPIENS2_KEYPOINT_SCHEMA_HALPE26,
    "halpe": SAPIENS2_KEYPOINT_SCHEMA_HALPE26,
    "halpe26": SAPIENS2_KEYPOINT_SCHEMA_HALPE26,
    "halpe_26": SAPIENS2_KEYPOINT_SCHEMA_HALPE26,
    "sports2d": SAPIENS2_KEYPOINT_SCHEMA_HALPE26,
    "mapped": SAPIENS2_KEYPOINT_SCHEMA_HALPE26,
    "original": SAPIENS2_KEYPOINT_SCHEMA_ORIGINAL,
    "raw": SAPIENS2_KEYPOINT_SCHEMA_ORIGINAL,
    "full": SAPIENS2_KEYPOINT_SCHEMA_ORIGINAL,
    "sapiens2": SAPIENS2_KEYPOINT_SCHEMA_ORIGINAL,
    "sapiens2_308": SAPIENS2_KEYPOINT_SCHEMA_ORIGINAL,
    "keypoints308": SAPIENS2_KEYPOINT_SCHEMA_ORIGINAL,
    "308": SAPIENS2_KEYPOINT_SCHEMA_ORIGINAL,
}

# Sapiens2 308-keypoint output uses lower_snake_case names from keypoints308.py.
# Sports2D's downstream angle/export stack expects the HALPE/OpenPose-style names
# used by Pose2Sim and the existing body_with_feet model.
SAPIENS2_TO_HALPE26_DIRECT = {
    "Nose": "nose",
    "LEye": "left_eye",
    "REye": "right_eye",
    "LEar": "left_ear",
    "REar": "right_ear",
    "LShoulder": "left_shoulder",
    "RShoulder": "right_shoulder",
    "LElbow": "left_elbow",
    "RElbow": "right_elbow",
    "LWrist": "left_wrist",
    "RWrist": "right_wrist",
    "LHip": "left_hip",
    "RHip": "right_hip",
    "LKnee": "left_knee",
    "RKnee": "right_knee",
    "LAnkle": "left_ankle",
    "RAnkle": "right_ankle",
    "LBigToe": "left_big_toe",
    "RBigToe": "right_big_toe",
    "LSmallToe": "left_small_toe",
    "RSmallToe": "right_small_toe",
    "LHeel": "left_heel",
    "RHeel": "right_heel",
}


def _repo_root() -> Path:
    """Return the repository root containing ``Sports2D/``."""

    return Path(__file__).resolve().parents[2]


def _normalize_model_size(value: object) -> str:
    """Normalize Sapiens2 model-size aliases to the checkpoint/config suffix."""

    raw = str(value or "0.4b").strip().lower().replace("-", "_")
    raw = raw.removeprefix("sapiens2_").removeprefix("sapiens2")
    raw = raw.removesuffix("_pose").strip("_")
    aliases = {
        "04b": "0.4b",
        "0_4b": "0.4b",
        "400m": "0.4b",
        "08b": "0.8b",
        "0_8b": "0.8b",
        "800m": "0.8b",
        "1_0b": "1b",
        "1.0b": "1b",
        "5_0b": "5b",
        "5.0b": "5b",
    }
    normalized = aliases.get(raw, raw)
    if normalized not in SAPIENS2_MODEL_SIZES:
        raise ValueError(
            "Unsupported sapiens2_model_size "
            f"'{value}'. Expected one of: {', '.join(SAPIENS2_MODEL_SIZES)}."
        )
    return normalized


def _normalize_sapiens2_keypoint_schema(value: object) -> str:
    """Normalize Sapiens2 output keypoint schema aliases."""

    raw = str(value or SAPIENS2_KEYPOINT_SCHEMA_HALPE26).strip().lower()
    normalized = SAPIENS2_KEYPOINT_SCHEMA_ALIASES.get(raw)
    if normalized is None:
        raise ValueError(
            "Unsupported sapiens2_keypoint_schema "
            f"'{value}'. Expected 'halpe26' or 'sapiens2_308'."
        )
    return normalized


def _keypoint_names_from_name_to_id(name_to_id: Dict[str, int]) -> list[str]:
    """Return keypoint names sorted by their tensor id."""

    valid_items = [
        (int(idx), str(name))
        for name, idx in dict(name_to_id or {}).items()
        if idx is not None and int(idx) >= 0
    ]
    if not valid_items:
        return []

    max_id = max(idx for idx, _ in valid_items)
    names = [f"sapiens2_keypoint_{idx:03d}" for idx in range(max_id + 1)]
    for idx, name in valid_items:
        names[idx] = name
    return names


def _load_sapiens2_skeleton_link_names(sapiens2_root: Path) -> list[tuple[str, str]]:
    """
    Load Sapiens2 keypoints308 skeleton links without importing the heavy runtime.

    The config file only defines Python dictionaries, so importing it through
    ``importlib`` avoids requiring torch during lightweight tests or docs checks.
    """

    metainfo_path = (
        Path(sapiens2_root)
        / "sapiens/pose/configs/_base_/keypoints308.py"
    )
    if not metainfo_path.exists():
        logging.warning(
            "Sapiens2 keypoints308 metainfo was not found: %s. "
            "Original 308-keypoint output will still draw points but no skeleton links.",
            metainfo_path,
        )
        return []

    spec = importlib.util.spec_from_file_location(
        "_sports2d_sapiens2_keypoints308", metainfo_path
    )
    if spec is None or spec.loader is None:
        return []
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    dataset_info = getattr(module, "dataset_info", {}) or {}
    skeleton_info = dataset_info.get("skeleton_info", {}) or {}
    link_names = []
    for _, link_info in sorted(
        skeleton_info.items(),
        key=lambda item: int((item[1] or {}).get("id", item[0])),
    ):
        link = (link_info or {}).get("link")
        if not link or len(link) != 2:
            continue
        link_names.append((str(link[0]), str(link[1])))
    return link_names


def _build_sapiens2_original_skeleton(
    keypoint_name_to_id: Dict[str, int],
    skeleton_link_names: Sequence[tuple[str, str]] = (),
):
    """Create a lightweight anytree skeleton for original Sapiens2 308 outputs."""

    from anytree import Node

    keypoint_names = _keypoint_names_from_name_to_id(keypoint_name_to_id)
    root = Node("SAPIENS2_308", id=None)
    for idx, keypoint_name in enumerate(keypoint_names):
        Node(keypoint_name, parent=root, id=idx)

    skeleton_links = []
    valid_link_names = []
    for parent_name, child_name in skeleton_link_names or []:
        if parent_name not in keypoint_name_to_id or child_name not in keypoint_name_to_id:
            continue
        skeleton_links.append(
            (int(keypoint_name_to_id[parent_name]), int(keypoint_name_to_id[child_name]))
        )
        valid_link_names.append((parent_name, child_name))

    root.skeleton_links = skeleton_links
    root.skeleton_link_names = valid_link_names
    root.keypoint_schema = SAPIENS2_KEYPOINT_SCHEMA_ORIGINAL
    return root, keypoint_names


def _expand_candidate_path(path_value: object, base_dirs: Iterable[Path]) -> Path:
    """Resolve a user/default path against reasonable local base directories."""

    path = Path(str(path_value)).expanduser()
    if path.is_absolute():
        return path

    for base_dir in base_dirs:
        candidate = (Path(base_dir) / path).expanduser()
        if candidate.exists():
            return candidate.resolve()
    return (Path.cwd() / path).resolve()


def _parse_bool(value: object, default: bool = True) -> bool:
    if value is None or value == "":
        return bool(default)
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    logging.warning(
        "Invalid boolean value for Sapiens2 option: %r. Using default=%s.",
        value,
        default,
    )
    return bool(default)


def _auto_download_enabled(pose_config: dict) -> bool:
    return _parse_bool(pose_config.get("sapiens2_auto_download", True), default=True)


def _download_hf_checkpoint(
    repo_id: str,
    filename: str,
    target_path: Path,
    description: str,
) -> Path:
    """
    Download a Sapiens2 checkpoint from Hugging Face into the expected local folder.

    The returned path is always usable by the caller.  If a user requested a custom
    missing filename, the Hub filename is downloaded beside it and returned instead
    of silently copying gigabytes to a second name.
    """

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "Automatic Sapiens2 checkpoint download requires huggingface_hub. "
            "Install with `pip install huggingface_hub`, "
            "`pip install -e .[sapiens2]`, or `pip install sports2d[sapiens2]`."
        ) from exc

    target_path = Path(target_path).expanduser()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    logging.info(
        "Downloading %s from Hugging Face (%s/%s) to %s",
        description,
        repo_id,
        filename,
        target_path.parent,
    )
    try:
        downloaded_path = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(target_path.parent),
            )
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download {description} from Hugging Face "
            f"repo '{repo_id}' file '{filename}'."
        ) from exc
    if target_path.name == filename:
        return target_path if target_path.exists() else downloaded_path

    logging.warning(
        "Sapiens2 auto-download fetched %s as %s instead of missing custom path %s. "
        "Using the downloaded Hub filename for this run.",
        description,
        downloaded_path,
        target_path,
    )
    return downloaded_path


def _resolve_sapiens2_root(pose_config: dict) -> Path:
    configured = pose_config.get("sapiens2_root", "")
    if configured not in [None, ""]:
        root = _expand_candidate_path(configured, [Path.cwd(), _repo_root()])
    else:
        root = (_repo_root() / "sapiens2").resolve()

    if not root.exists():
        raise ImportError(
            "Sapiens2 backend requires a local Sapiens2 checkout. "
            "Set pose.sapiens2_root to the sapiens2 repository path or keep the "
            "nested ./sapiens2 checkout next to Sports2D."
        )
    if not (root / "sapiens").exists():
        raise ImportError(
            f"Sapiens2 root '{root}' does not contain a sapiens package. "
            "Set pose.sapiens2_root to the Sapiens2 repository root."
        )
    return root


def _checkpoint_root(pose_config: dict) -> Path:
    configured = pose_config.get("sapiens2_checkpoint_root", "")
    raw_root = (
        configured
        if configured not in [None, ""]
        else os.environ.get("SAPIENS_CHECKPOINT_ROOT", "~/sapiens2_host")
    )
    return Path(str(raw_root)).expanduser()


def _resolve_sapiens2_config_path(
    pose_config: dict, sapiens2_root: Path, model_size: str
) -> Path:
    configured = pose_config.get("sapiens2_config", "")
    default_rel = SAPIENS2_CONFIG_TEMPLATE.format(
        dataset=SAPIENS2_DEFAULT_DATASET,
        size=model_size,
    )
    path_value = configured if configured not in [None, ""] else default_rel
    path = _expand_candidate_path(path_value, [Path.cwd(), sapiens2_root, _repo_root()])
    if not path.exists():
        raise FileNotFoundError(
            f"Sapiens2 pose config was not found: {path}. "
            "Set pose.sapiens2_config or pose.sapiens2_root."
        )
    return path


def _resolve_sapiens2_checkpoint_path(
    pose_config: dict, checkpoint_root: Path, model_size: str
) -> Path:
    configured = pose_config.get("sapiens2_checkpoint", "")
    filename = f"sapiens2_{model_size}_pose.safetensors"
    default_path = checkpoint_root / "pose" / filename
    path_value = configured if configured not in [None, ""] else default_path
    path = _expand_candidate_path(path_value, [Path.cwd(), checkpoint_root, _repo_root()])
    if not path.exists():
        if _auto_download_enabled(pose_config):
            default_repo_id = SAPIENS2_POSE_REPO_TEMPLATE.format(size=model_size)
            repo_id = str(
                pose_config.get("sapiens2_pose_repo", "") or default_repo_id
            ).strip() or default_repo_id
            return _download_hf_checkpoint(
                repo_id=repo_id,
                filename=filename,
                target_path=path,
                description=f"Sapiens2 pose checkpoint ({model_size})",
            )
        raise FileNotFoundError(
            f"Sapiens2 pose checkpoint was not found: {path}. "
            "Automatic download is disabled. Enable pose.sapiens2_auto_download "
            "or set pose.sapiens2_checkpoint."
        )
    return path


def _resolve_sapiens2_detector_config_path(
    pose_config: dict, sapiens2_root: Path
) -> Path:
    configured = pose_config.get("sapiens2_detector_config", "")
    path_value = (
        configured if configured not in [None, ""] else SAPIENS2_DEFAULT_DETECTOR_CONFIG
    )
    path = _expand_candidate_path(path_value, [Path.cwd(), sapiens2_root, _repo_root()])
    if not path.exists():
        raise FileNotFoundError(
            f"Sapiens2 detector config was not found: {path}. "
            "Set pose.sapiens2_detector_config or pose.sapiens2_root."
        )
    return path


def _resolve_sapiens2_detector_checkpoint_path(
    pose_config: dict, checkpoint_root: Path
) -> Path:
    configured = pose_config.get("sapiens2_detector_checkpoint", "")
    filename = "rtmdet_m.pth"
    default_path = checkpoint_root / "detector" / filename
    path_value = configured if configured not in [None, ""] else default_path
    path = _expand_candidate_path(path_value, [Path.cwd(), checkpoint_root, _repo_root()])
    if not path.exists():
        if _auto_download_enabled(pose_config):
            repo_id = str(
                pose_config.get("sapiens2_detector_repo", "")
                or SAPIENS2_DETECTOR_REPO
            ).strip() or SAPIENS2_DETECTOR_REPO
            return _download_hf_checkpoint(
                repo_id=repo_id,
                filename=filename,
                target_path=path,
                description="Sapiens2 RTMDet person detector checkpoint",
            )
        raise FileNotFoundError(
            f"Sapiens2 detector checkpoint was not found: {path}. "
            "Automatic download is disabled. Enable pose.sapiens2_auto_download "
            "or set pose.sapiens2_detector_checkpoint."
        )
    return path


def _resolve_sapiens2_device(device_value: object) -> str:
    """Convert Sports2D's device option into a Sapiens2/PyTorch device string."""

    requested = str(device_value or "auto").strip().lower()
    if requested in {"", "auto"}:
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda:0"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"
    if requested == "cuda":
        return "cuda:0"
    if requested.startswith("cuda:") or requested in {"cpu", "mps"}:
        return requested
    logging.warning(
        "Sapiens2 backend does not support device '%s'. Falling back to CPU.",
        device_value,
    )
    return "cpu"


def _resolve_sapiens2_inference_dtype(torch_module, dtype_value: object):
    """Return an optional torch dtype for Sapiens2 inference."""

    requested = str(dtype_value or "float32").strip().lower()
    aliases = {
        "": "float32",
        "none": "float32",
        "fp32": "float32",
        "float": "float32",
        "fp16": "float16",
        "half": "float16",
        "bf16": "bfloat16",
    }
    requested = aliases.get(requested, requested)
    if requested == "float32":
        return None
    if requested == "float16":
        return torch_module.float16
    if requested == "bfloat16":
        return torch_module.bfloat16
    raise ValueError(
        "Unsupported sapiens2_inference_dtype "
        f"'{dtype_value}'. Expected float32, float16, or bfloat16."
    )


def _nms_xyxy(boxes: np.ndarray, scores: np.ndarray, nms_threshold: float) -> np.ndarray:
    """Small NumPy NMS for detector person boxes."""

    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if len(boxes) == 0:
        return np.empty((0,), dtype=np.int64)

    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter_w = np.maximum(0.0, xx2 - xx1)
        inter_h = np.maximum(0.0, yy2 - yy1)
        inter = inter_w * inter_h
        union = areas[i] + areas[rest] - inter
        iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        order = rest[iou <= float(nms_threshold)]
    return np.asarray(keep, dtype=np.int64)


def _finite_midpoint(
    keypoints: np.ndarray,
    scores: np.ndarray,
    idx_a: int,
    idx_b: int,
) -> Tuple[np.ndarray, float]:
    points = np.asarray([keypoints[idx_a], keypoints[idx_b]], dtype=np.float32)
    point_scores = np.asarray([scores[idx_a], scores[idx_b]], dtype=np.float32)
    finite = np.isfinite(points[:, 0]) & np.isfinite(points[:, 1])
    if not np.any(finite):
        return np.asarray([np.nan, np.nan], dtype=np.float32), np.nan
    return (
        np.nanmean(points[finite], axis=0).astype(np.float32),
        float(np.nanmean(point_scores[finite])),
    )


def _derive_head(
    mapped_keypoints: np.ndarray, mapped_scores: np.ndarray, name_to_idx: Dict[str, int]
) -> Tuple[np.ndarray, float]:
    left_eye_idx = name_to_idx["LEye"]
    right_eye_idx = name_to_idx["REye"]
    nose_idx = name_to_idx["Nose"]
    left_eye = mapped_keypoints[left_eye_idx]
    right_eye = mapped_keypoints[right_eye_idx]
    if np.all(np.isfinite([*left_eye, *right_eye])):
        eye_center = (left_eye + right_eye) * 0.5
        eye_distance = float(np.linalg.norm(left_eye - right_eye))
        return (
            np.asarray([eye_center[0], eye_center[1] - eye_distance * 0.8], dtype=np.float32),
            float(np.nanmean([mapped_scores[left_eye_idx], mapped_scores[right_eye_idx]])),
        )
    if np.all(np.isfinite(mapped_keypoints[nose_idx])):
        return mapped_keypoints[nose_idx].astype(np.float32), float(mapped_scores[nose_idx])
    return np.asarray([np.nan, np.nan], dtype=np.float32), np.nan


def _map_sapiens2_keypoints_to_halpe26(
    keypoints: Sequence[np.ndarray],
    scores: Sequence[np.ndarray],
    sapiens_keypoint_name_to_id: Dict[str, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Map Sapiens2 308-keypoint instances to Sports2D HALPE_26 arrays."""

    keypoints = [np.asarray(instance, dtype=np.float32) for instance in keypoints]
    scores = [np.asarray(instance, dtype=np.float32).reshape(-1) for instance in scores]
    output_keypoints = np.full(
        (len(keypoints), len(HALPE26_KEYPOINT_NAMES), 2), np.nan, dtype=np.float32
    )
    output_scores = np.full(
        (len(keypoints), len(HALPE26_KEYPOINT_NAMES)), np.nan, dtype=np.float32
    )
    halpe_name_to_idx = {name: idx for idx, name in enumerate(HALPE26_KEYPOINT_NAMES)}

    for person_idx, (person_keypoints, person_scores) in enumerate(zip(keypoints, scores)):
        for halpe_name, sapiens_name in SAPIENS2_TO_HALPE26_DIRECT.items():
            src_idx = sapiens_keypoint_name_to_id.get(sapiens_name)
            dst_idx = halpe_name_to_idx[halpe_name]
            if (
                src_idx is None
                or src_idx >= len(person_keypoints)
                or src_idx >= len(person_scores)
            ):
                continue
            output_keypoints[person_idx, dst_idx] = person_keypoints[src_idx, :2]
            output_scores[person_idx, dst_idx] = person_scores[src_idx]

        neck_point, neck_score = _finite_midpoint(
            output_keypoints[person_idx],
            output_scores[person_idx],
            halpe_name_to_idx["LShoulder"],
            halpe_name_to_idx["RShoulder"],
        )
        hip_point, hip_score = _finite_midpoint(
            output_keypoints[person_idx],
            output_scores[person_idx],
            halpe_name_to_idx["LHip"],
            halpe_name_to_idx["RHip"],
        )
        head_point, head_score = _derive_head(
            output_keypoints[person_idx], output_scores[person_idx], halpe_name_to_idx
        )
        for name, point, score in [
            ("Neck", neck_point, neck_score),
            ("Hip", hip_point, hip_score),
            ("Head", head_point, head_score),
        ]:
            dst_idx = halpe_name_to_idx[name]
            output_keypoints[person_idx, dst_idx] = point
            output_scores[person_idx, dst_idx] = score

    return output_keypoints, output_scores


def _format_sapiens2_original_keypoints(
    keypoints: Sequence[np.ndarray],
    scores: Sequence[np.ndarray],
    keypoint_names: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """Return Sapiens2 keypoints in their original dense tensor order."""

    expected_count = len(list(keypoint_names or []))
    output_keypoints = np.full(
        (len(keypoints), expected_count, 2), np.nan, dtype=np.float32
    )
    output_scores = np.full(
        (len(keypoints), expected_count), np.nan, dtype=np.float32
    )

    for person_idx, (person_keypoints, person_scores) in enumerate(zip(keypoints, scores)):
        person_keypoints = np.asarray(person_keypoints, dtype=np.float32)
        person_scores = np.asarray(person_scores, dtype=np.float32).reshape(-1)
        keypoint_count = min(expected_count, len(person_keypoints))
        score_count = min(expected_count, len(person_scores))
        if keypoint_count > 0:
            output_keypoints[person_idx, :keypoint_count] = person_keypoints[
                :keypoint_count, :2
            ]
        if score_count > 0:
            output_scores[person_idx, :score_count] = person_scores[:score_count]
    return output_keypoints, output_scores


class Sapiens2Backend(PoseBackend):
    """Sapiens2 308-keypoint top-down pose backend."""

    def __init__(self, config_dict: dict):
        from Pose2Sim.skeletons import HALPE_26

        pose_config = config_dict.get("pose", {}) or {}
        self._pose_config = pose_config
        self._keypoint_schema = _normalize_sapiens2_keypoint_schema(
            pose_config.get("sapiens2_keypoint_schema", SAPIENS2_KEYPOINT_SCHEMA_HALPE26)
        )
        self._model_size = _normalize_model_size(
            pose_config.get("sapiens2_model_size", "0.4b")
        )
        self._sapiens2_root = _resolve_sapiens2_root(pose_config)
        if str(self._sapiens2_root) not in sys.path:
            sys.path.insert(0, str(self._sapiens2_root))

        self._checkpoint_root = _checkpoint_root(pose_config)
        self._pose_config_path = _resolve_sapiens2_config_path(
            pose_config, self._sapiens2_root, self._model_size
        )
        self._pose_checkpoint_path = _resolve_sapiens2_checkpoint_path(
            pose_config, self._checkpoint_root, self._model_size
        )
        self._device = _resolve_sapiens2_device(pose_config.get("device", "auto"))
        self._det_frequency = max(1, int(pose_config.get("det_frequency", 1)))
        self._person_threshold = float(
            pose_config.get(
                "person_detection_threshold",
                pose_config.get("keypoint_likelihood_threshold", 0.3),
            )
        )
        self._keypoint_threshold = float(
            pose_config.get("keypoint_likelihood_threshold", self._person_threshold)
        )
        self._nms_threshold = float(pose_config.get("sapiens2_nms_threshold", 0.3))
        base_config = config_dict.get("base", {}) or {}
        self._max_person_bboxes = max(
            0, int(base_config.get("nb_persons_to_detect", 0) or 0)
        )
        self._manual_person_roi = pose_config.get("_manual_person_roi")
        self._manual_roi_mode = normalize_manual_roi_mode(
            pose_config.get("manual_roi_mode", "bootstrap")
        )
        self._manual_roi_tracking_margin_px = max(
            0, int(pose_config.get("manual_roi_tracking_margin_px", 48))
        )
        self._manual_roi_reacquire_patience = max(
            1, int(pose_config.get("manual_roi_reacquire_patience", 6))
        )
        self._manual_roi_reacquire_frequency = max(
            1, int(pose_config.get("manual_roi_reacquire_frequency", 15))
        )
        self._active_manual_person_roi = self._manual_person_roi
        self._manual_roi_miss_count = 0
        self._last_full_frame_reacquire_frame: Optional[int] = None
        self._bbox_source = self._normalize_bbox_source(
            pose_config.get("sapiens2_bbox_source", "detector")
            )
        self._frame_count = 0
        self._prev_bboxes: Optional[np.ndarray] = None
        self._prev_bbox_scores: Optional[np.ndarray] = None
        self._last_detections: Dict[str, np.ndarray] = self._empty_detections()
        self._skeleton_tree = None
        self._keypoint_names = []
        self._timing = {
            "bbox_s": 0.0,
            "pose_preprocess_s": 0.0,
            "pose_forward_s": 0.0,
            "pose_decode_s": 0.0,
            "backend_total_s": 0.0,
        }
        self._timing_frames = 0
        self._timing_pose_instances = 0
        self._timing_bboxes = 0

        try:
            import torch
            from sapiens.pose.datasets import UDPHeatmap, parse_pose_metainfo
            from sapiens.pose.models import init_model
        except ImportError as exc:
            raise ImportError(
                "Sapiens2 backend requires the local sapiens2 package and its "
                "dependencies (torch, torchvision, safetensors, and optional mmdet "
                "for RTMDet bboxes). Install the nested checkout with "
                "`pip install -e ./sapiens2` or set pose.sapiens2_root to an "
                "installed Sapiens2 checkout."
            ) from exc

        self._torch = torch
        self._infer_dtype = _resolve_sapiens2_inference_dtype(
            torch, pose_config.get("sapiens2_inference_dtype", "float32")
        )
        self._model = init_model(
            str(self._pose_config_path),
            str(self._pose_checkpoint_path),
            device=self._device,
        )
        if self._infer_dtype is not None:
            self._model.to(dtype=self._infer_dtype)
            if str(self._device).startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
        if (
            self._model.cfg.val_cfg is not None
            and pose_config.get("sapiens2_flip_test") not in [None, ""]
        ):
            self._model.cfg.val_cfg["flip_test"] = _parse_bool(
                pose_config.get("sapiens2_flip_test"), default=True
            )
        num_keypoints = int(getattr(self._model.cfg, "num_keypoints", 0))
        if num_keypoints == 308 or not hasattr(self._model, "pose_metainfo"):
            self._model.pose_metainfo = parse_pose_metainfo(
                dict(
                    from_file=str(
                        self._sapiens2_root
                        / "sapiens/pose/configs/_base_/keypoints308.py"
                    )
                )
            )
        codec_cfg = dict(self._model.cfg.codec)
        codec_type = codec_cfg.pop("type")
        if codec_type != "UDPHeatmap":
            raise ValueError(
                f"Unsupported Sapiens2 pose codec '{codec_type}'. Only UDPHeatmap is supported."
            )
        self._model.codec = UDPHeatmap(**codec_cfg)
        self._sapiens_name_to_id = dict(self._model.pose_metainfo["keypoint_name2id"])
        self._sapiens_keypoint_names = _keypoint_names_from_name_to_id(
            self._sapiens_name_to_id
        )
        if self._keypoint_schema == SAPIENS2_KEYPOINT_SCHEMA_ORIGINAL:
            skeleton_link_names = _load_sapiens2_skeleton_link_names(self._sapiens2_root)
            self._skeleton_tree, self._keypoint_names = (
                _build_sapiens2_original_skeleton(
                    self._sapiens_name_to_id,
                    skeleton_link_names,
                )
            )
        else:
            self._skeleton_tree = copy.deepcopy(HALPE_26)
            self._keypoint_names = list(HALPE26_KEYPOINT_NAMES)

        self._detector = None
        self._inference_detector = None
        self._yolox_detector = None
        if self._bbox_source == "detector":
            self._init_detector(pose_config)
        elif self._bbox_source == "yolox":
            self._init_yolox_detector(pose_config)
        elif self._bbox_source == "manual_roi" and self._manual_person_roi is None:
            logging.warning(
                "sapiens2_bbox_source='manual_roi' requested without manual_roi data. "
                "Falling back to full-frame Sapiens2 pose boxes."
            )
            self._bbox_source = "full_frame"
        if (
            self._bbox_source == "manual_roi"
            and self._manual_roi_mode == "adaptive_person"
            and self._manual_person_roi is None
        ):
            logging.warning(
                "manual_roi_mode='adaptive_person' requires a manual person ROI. "
                "Falling back to bootstrap."
            )
            self._manual_roi_mode = "bootstrap"

        logging.info(
            "Sapiens2Backend initialized: model=sapiens2_%s, bbox_source=%s, "
            "schema=%s, device=%s, dtype=%s, flip_test=%s, keypoints=%s",
            self._model_size,
            self._bbox_source,
            self._keypoint_schema,
            self._device,
            self._infer_dtype or "float32",
            bool(
                self._model.cfg.val_cfg is not None
                and self._model.cfg.val_cfg.get("flip_test", False)
            ),
            self.num_keypoints,
        )

    def _sync_cuda_if_needed(self) -> None:
        """Synchronize CUDA before/after timed pose forward passes."""

        if str(self._device).startswith("cuda") and self._torch.cuda.is_available():
            self._torch.cuda.synchronize()

    def _record_timing(self, name: str, elapsed_s: float) -> None:
        self._timing[name] = self._timing.get(name, 0.0) + float(elapsed_s)

    def _timing_summary(self) -> str:
        frames = max(1, self._timing_frames)
        instances = max(1, self._timing_pose_instances)
        bbox_s = self._timing.get("bbox_s", 0.0)
        preprocess_s = self._timing.get("pose_preprocess_s", 0.0)
        forward_s = self._timing.get("pose_forward_s", 0.0)
        decode_s = self._timing.get("pose_decode_s", 0.0)
        total_s = self._timing.get("backend_total_s", 0.0)
        pose_total_s = preprocess_s + forward_s + decode_s

        return (
            "Sapiens2 inference timing: "
            f"frames={self._timing_frames}, pose_instances={self._timing_pose_instances}, "
            f"bboxes={self._timing_bboxes}, bbox_source={self._bbox_source}, "
            f"device={self._device}, dtype={self._infer_dtype or 'float32'}, "
            f"bbox_stage={bbox_s:.3f}s ({bbox_s / frames * 1000:.2f} ms/frame), "
            f"pose_preprocess={preprocess_s:.3f}s "
            f"({preprocess_s / instances * 1000:.2f} ms/person), "
            f"pose_forward={forward_s:.3f}s "
            f"({forward_s / instances * 1000:.2f} ms/person, "
            f"{self._timing_pose_instances / forward_s if forward_s > 0 else 0.0:.2f} person/s), "
            f"pose_decode={decode_s:.3f}s ({decode_s / instances * 1000:.2f} ms/person), "
            f"pose_total={pose_total_s:.3f}s "
            f"({pose_total_s / instances * 1000:.2f} ms/person), "
            f"backend_inference_total={total_s:.3f}s "
            f"({total_s / frames * 1000:.2f} ms/frame, "
            f"{self._timing_frames / total_s if total_s > 0 else 0.0:.2f} fps)"
        )

    def log_inference_timing_summary(self) -> None:
        """Log accumulated Sapiens2 detector/pose inference timing."""

        if self._timing_frames <= 0:
            return
        logging.info(self._timing_summary())

    @staticmethod
    def _empty_detections() -> Dict[str, np.ndarray]:
        return {
            "boxes": np.empty((0, 4), dtype=np.float32),
            "classes": np.empty((0,), dtype=np.int32),
            "scores": np.empty((0,), dtype=np.float32),
            "person_boxes": np.empty((0, 4), dtype=np.float32),
            "ball_boxes": np.empty((0, 4), dtype=np.float32),
            "ball_scores": np.empty((0,), dtype=np.float32),
        }

    @staticmethod
    def _normalize_bbox_source(value: object) -> str:
        bbox_source = str(value or "detector").strip().lower()
        if bbox_source not in SAPIENS2_SUPPORTED_BBOX_SOURCES:
            raise ValueError(
                "Unsupported sapiens2_bbox_source "
                f"'{value}'. Expected one of: {', '.join(sorted(SAPIENS2_SUPPORTED_BBOX_SOURCES))}."
            )
        return bbox_source

    @staticmethod
    def _mmdet_pipeline(cfg):
        try:
            from mmdet.datasets import transforms
        except Exception:
            return cfg
        if "test_dataloader" not in cfg:
            return cfg
        pipeline = cfg.test_dataloader.dataset.pipeline
        for trans in pipeline:
            if trans.get("type") in dir(transforms):
                trans["type"] = "mmdet." + trans["type"]
        return cfg

    def _init_detector(self, pose_config: dict) -> None:
        detector_config_path = _resolve_sapiens2_detector_config_path(
            pose_config, self._sapiens2_root
        )
        detector_checkpoint_path = _resolve_sapiens2_detector_checkpoint_path(
            pose_config, self._checkpoint_root
        )
        if "mmpretrain" not in sys.modules:
            # Match Sapiens2's CLI workaround for mmpretrain/transformers API drift.
            sys.modules["mmpretrain"] = None
        try:
            from mmdet.apis import inference_detector, init_detector
        except ImportError as exc:
            raise ImportError(
                "Sapiens2 bbox_source='detector' requires mmdet. Install the "
                "Sapiens2 dependencies with `pip install -e .[sapiens2]` or "
                "`pip install sports2d[sapiens2]`, or set pose.sapiens2_bbox_source "
                "to 'manual_roi' or 'full_frame'."
            ) from exc

        self._detector = init_detector(
            str(detector_config_path),
            str(detector_checkpoint_path),
            device=self._device,
        )
        self._detector.cfg = self._mmdet_pipeline(self._detector.cfg)
        self._inference_detector = inference_detector

    @staticmethod
    def _resolve_yolox_detector_size(pose_config: dict) -> str:
        value = str(
            pose_config.get(
                "sapiens2_yolox_model_size",
                pose_config.get("mode", "balanced"),
            )
        ).strip().lower()
        mode_to_size = {
            "performance": "x",
            "balanced": "m",
            "lightweight": "s",
        }
        if value in mode_to_size:
            return mode_to_size[value]
        if value in {"s", "m", "l", "x"}:
            return value
        if value == "tiny":
            logging.warning("sapiens2_yolox_model_size='tiny' is deprecated. Using 's'.")
            return "s"
        logging.warning(
            "Unknown sapiens2_yolox_model_size/mode '%s'. Using 'm'.",
            value,
        )
        return "m"

    def _init_yolox_detector(self, pose_config: dict) -> None:
        try:
            from rtmlib import YOLOX
        except ImportError as exc:
            raise ImportError(
                "Sapiens2 bbox_source='yolox' requires rtmlib. Install Sports2D "
                "runtime dependencies or set pose.sapiens2_bbox_source to "
                "'manual_roi', 'full_frame', or 'detector'."
            ) from exc

        detector_size = self._resolve_yolox_detector_size(pose_config)
        humanart_size_map = {
            "s": "m",
            "m": "m",
            "l": "m",
            "x": "x",
        }
        resolved_humanart_size = humanart_size_map.get(detector_size, "m")
        humanart_yolox_models = {
            "m": (
                "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/"
                "onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip",
                (640, 640),
            ),
            "x": (
                "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/"
                "onnx_sdk/yolox_x_8xb8-300e_humanart-a39d44ed.zip",
                (640, 640),
            ),
        }
        model_url, input_size = humanart_yolox_models[resolved_humanart_size]
        detector_backend = (
            pose_config.get("backend", "auto")
            if pose_config.get("backend", "auto") != "auto"
            else "onnxruntime"
        )
        detector_device = "cuda" if str(self._device).startswith("cuda") else "cpu"
        self._yolox_detector = YOLOX(
            onnx_model=model_url,
            model_input_size=input_size,
            mode="human",
            nms_thr=0.45,
            score_thr=self._person_threshold,
            backend=detector_backend,
            device=detector_device,
        )
        logging.info(
            "Sapiens2 YOLOX detector initialized: weights=humanart/%s, "
            "requested_size=%s, backend=%s, device=%s, score_thr=%s",
            resolved_humanart_size,
            detector_size,
            detector_backend,
            detector_device,
            self._person_threshold,
        )

    def _limit_person_bboxes(
        self, bboxes: np.ndarray, scores: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self._max_person_bboxes <= 0 or len(bboxes) <= self._max_person_bboxes:
            return bboxes, scores

        bboxes = np.asarray(bboxes, dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        areas = np.maximum(0.0, bboxes[:, 2] - bboxes[:, 0]) * np.maximum(
            0.0, bboxes[:, 3] - bboxes[:, 1]
        )
        finite_scores = np.isfinite(scores)
        if np.any(finite_scores):
            rank = np.where(finite_scores, scores, -np.inf)
        else:
            rank = areas
        order = np.argsort(rank)[::-1][: self._max_person_bboxes]
        return bboxes[order], scores[order]

    @staticmethod
    def _full_frame_bbox(frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        height, width = frame.shape[:2]
        return (
            np.asarray([[0, 0, max(0, width - 1), max(0, height - 1)]], dtype=np.float32),
            np.asarray([1.0], dtype=np.float32),
        )

    def _manual_roi_bbox(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self._manual_person_roi is None:
            return self._full_frame_bbox(frame)
        roi = (
            self._active_manual_person_roi
            if self._manual_roi_mode == "adaptive_person"
            else self._manual_person_roi
        )
        roi = normalize_roi_xyxy(roi, frame.shape)
        if roi is None:
            return self._full_frame_bbox(frame)
        return np.asarray([roi], dtype=np.float32), np.asarray([1.0], dtype=np.float32)

    def _adaptive_manual_roi_enabled(self) -> bool:
        return (
            self._bbox_source == "manual_roi"
            and self._manual_roi_mode == "adaptive_person"
            and self._manual_person_roi is not None
        )

    def _should_force_manual_roi_reacquire(self) -> bool:
        if not self._adaptive_manual_roi_enabled():
            return False
        if self._manual_roi_miss_count < self._manual_roi_reacquire_patience:
            return False
        if self._last_full_frame_reacquire_frame is None:
            return True
        return (
            self._frame_count - self._last_full_frame_reacquire_frame
            >= self._manual_roi_reacquire_frequency
        )

    def _update_adaptive_manual_roi(
        self,
        keypoints: np.ndarray,
        scores: np.ndarray,
        frame_shape,
    ) -> None:
        if not self._adaptive_manual_roi_enabled():
            return
        if keypoints is None or scores is None or len(keypoints) == 0:
            self._mark_adaptive_manual_roi_miss(frame_shape)
            return

        person_keypoints = np.asarray(keypoints[0], dtype=np.float32)
        person_scores = np.asarray(scores[0], dtype=np.float32).reshape(-1)
        if person_keypoints.ndim != 2 or person_keypoints.shape[1] < 2:
            self._mark_adaptive_manual_roi_miss(frame_shape)
            return

        keep = (
            np.isfinite(person_keypoints[:, 0])
            & np.isfinite(person_keypoints[:, 1])
            & np.isfinite(person_scores)
            & (person_scores >= self._keypoint_threshold)
        )
        if int(np.sum(keep)) < 4:
            self._mark_adaptive_manual_roi_miss(frame_shape)
            return

        coords = person_keypoints[keep, :2]
        person_box = np.asarray(
            [[coords[:, 0].min(), coords[:, 1].min(), coords[:, 0].max(), coords[:, 1].max()]],
            dtype=np.float32,
        )
        updated_roi = roi_from_boxes_xyxy(
            person_box,
            frame_shape,
            padding_px=self._manual_roi_tracking_margin_px,
        )
        if updated_roi is not None:
            self._active_manual_person_roi = updated_roi
            self._manual_roi_miss_count = 0

    def _mark_adaptive_manual_roi_miss(self, frame_shape) -> None:
        if not self._adaptive_manual_roi_enabled():
            return
        self._manual_roi_miss_count += 1
        expanded_roi = expand_roi_xyxy(
            self._active_manual_person_roi,
            frame_shape,
            padding_px=self._manual_roi_tracking_margin_px,
        )
        if expanded_roi is not None:
            self._active_manual_person_roi = expanded_roi

    def _detector_bboxes(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        should_detect = self._frame_count % self._det_frequency == 0
        if (
            not should_detect
            and self._prev_bboxes is not None
            and len(self._prev_bboxes) > 0
        ):
            return self._prev_bboxes.copy(), self._prev_bbox_scores.copy()

        det_result = self._inference_detector(self._detector, frame)
        pred_instance = det_result.pred_instances.cpu().numpy()
        bboxes = np.asarray(pred_instance.bboxes, dtype=np.float32)
        scores = np.asarray(pred_instance.scores, dtype=np.float32).reshape(-1)
        labels = np.asarray(pred_instance.labels, dtype=np.int32).reshape(-1)
        keep_mask = np.logical_and(labels == PERSON_CLASS_ID, scores > self._person_threshold)
        bboxes = bboxes[keep_mask]
        scores = scores[keep_mask]
        if len(bboxes) > 0:
            keep = _nms_xyxy(bboxes, scores, self._nms_threshold)
            bboxes = bboxes[keep]
            scores = scores[keep]
        else:
            logging.debug(
                "Sapiens2 detector found no person on frame %s; using full-frame bbox.",
                self._frame_count,
            )
            bboxes, scores = self._full_frame_bbox(frame)
            scores[:] = np.nan

        bboxes, scores = self._limit_person_bboxes(bboxes, scores)
        self._prev_bboxes = bboxes.copy()
        self._prev_bbox_scores = scores.copy()
        return bboxes, scores

    def _yolox_bboxes(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        should_detect = self._frame_count % self._det_frequency == 0
        if (
            not should_detect
            and self._prev_bboxes is not None
            and len(self._prev_bboxes) > 0
        ):
            return self._prev_bboxes.copy(), self._prev_bbox_scores.copy()

        detector_outputs = self._yolox_detector(frame)
        if detector_outputs is None:
            bboxes = np.empty((0, 4), dtype=np.float32)
            scores = np.empty((0,), dtype=np.float32)
        elif isinstance(detector_outputs, tuple) and len(detector_outputs) >= 2:
            bboxes = np.asarray(detector_outputs[0], dtype=np.float32)
            classes = np.asarray(detector_outputs[1], dtype=np.int32).reshape(-1)
            if len(detector_outputs) >= 3:
                scores = np.asarray(detector_outputs[2], dtype=np.float32).reshape(-1)
            elif bboxes.ndim == 2 and bboxes.shape[1] >= 5:
                scores = bboxes[:, 4].astype(np.float32, copy=False)
            else:
                scores = np.full((len(bboxes),), np.nan, dtype=np.float32)
            if len(classes) == len(bboxes):
                keep_class = classes == PERSON_CLASS_ID
                bboxes = bboxes[keep_class]
                scores = scores[keep_class]
        else:
            bboxes = np.asarray(detector_outputs, dtype=np.float32)
            if bboxes.ndim == 2 and bboxes.shape[1] >= 5:
                scores = bboxes[:, 4].astype(np.float32, copy=False)
            else:
                scores = np.full((len(bboxes),), np.nan, dtype=np.float32)

        if bboxes.size == 0:
            bboxes = np.empty((0, 4), dtype=np.float32)
            scores = np.empty((0,), dtype=np.float32)
        else:
            if bboxes.ndim == 1:
                bboxes = bboxes.reshape(1, -1)
            bboxes = bboxes[:, :4].astype(np.float32, copy=False)
            if len(scores) != len(bboxes):
                scores = np.full((len(bboxes),), np.nan, dtype=np.float32)
            finite_boxes = np.all(np.isfinite(bboxes), axis=1)
            finite_scores = np.isfinite(scores)
            keep_score = np.logical_or(~finite_scores, scores >= self._person_threshold)
            keep_mask = np.logical_and(finite_boxes, keep_score)
            bboxes = bboxes[keep_mask]
            scores = scores[keep_mask]
            if len(bboxes) > 0:
                finite_scores = np.isfinite(scores)
                nms_scores = np.where(finite_scores, scores, 1.0).astype(np.float32)
                keep = _nms_xyxy(bboxes, nms_scores, self._nms_threshold)
                bboxes = bboxes[keep]
                scores = scores[keep]

        bboxes, scores = self._limit_person_bboxes(bboxes, scores)
        self._prev_bboxes = bboxes.copy()
        self._prev_bbox_scores = scores.copy()
        return bboxes, scores

    def _get_bboxes(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self._bbox_source == "detector":
            return self._detector_bboxes(frame)
        if self._bbox_source == "yolox":
            return self._yolox_bboxes(frame)
        if self._bbox_source == "manual_roi":
            if self._should_force_manual_roi_reacquire():
                self._last_full_frame_reacquire_frame = self._frame_count
                return self._full_frame_bbox(frame)
            return self._manual_roi_bbox(frame)
        return self._full_frame_bbox(frame)

    def _predict_pose_from_bboxes(
        self, frame: np.ndarray, bboxes: np.ndarray
    ) -> Tuple[list[np.ndarray], list[np.ndarray]]:
        if bboxes is None or len(bboxes) == 0:
            return [], []

        preprocess_start = time.perf_counter()
        inputs_list = []
        data_samples_list = []
        for bbox in np.asarray(bboxes, dtype=np.float32):
            data_info = {"img": frame, "bbox": bbox.reshape(1, 4)}
            data_info["bbox_score"] = np.ones(1, dtype=np.float32)
            data = self._model.pipeline(data_info)
            data = self._model.data_preprocessor(data)
            inputs_list.append(data["inputs"])
            data_samples_list.append(data["data_samples"])

        inputs = self._torch.cat(inputs_list, dim=0)
        if self._infer_dtype is not None and inputs.is_floating_point():
            inputs = inputs.to(dtype=self._infer_dtype)
        self._sync_cuda_if_needed()
        self._record_timing(
            "pose_preprocess_s",
            time.perf_counter() - preprocess_start,
        )

        autocast_device_type = "cuda" if str(self._device).startswith("cuda") else "cpu"
        autocast_enabled = (
            self._infer_dtype in (self._torch.float16, self._torch.bfloat16)
            and str(self._device).startswith(("cuda", "cpu"))
        )
        forward_start = time.perf_counter()
        with self._torch.no_grad():
            with self._torch.autocast(
                device_type=autocast_device_type,
                dtype=self._infer_dtype or self._torch.float32,
                enabled=autocast_enabled,
            ):
                pred = self._model(inputs)
                if self._model.cfg.val_cfg is not None and self._model.cfg.val_cfg.get(
                    "flip_test", False
                ):
                    pred_flipped = self._model(inputs.flip(-1)).flip(-1)
                    flip_indices = self._model.pose_metainfo["flip_indices"]
                    pred_flipped = pred_flipped[:, flip_indices]
                    pred = (pred + pred_flipped) / 2.0
        self._sync_cuda_if_needed()
        self._record_timing("pose_forward_s", time.perf_counter() - forward_start)

        decode_start = time.perf_counter()
        pred = pred.float().cpu().numpy()
        keypoints = []
        keypoint_scores = []
        for idx, data_samples in enumerate(data_samples_list):
            instance_keypoints, instance_scores = self._model.codec.decode(pred[idx])
            meta = data_samples["meta"]
            input_size = np.asarray(meta["input_size"], dtype=np.float32).reshape(-1)[:2]
            bbox_center = np.asarray(meta["bbox_center"], dtype=np.float32).reshape(-1)[:2]
            bbox_scale = np.asarray(meta["bbox_scale"], dtype=np.float32).reshape(-1)[:2]
            instance_keypoints = (
                instance_keypoints / input_size * bbox_scale
                + bbox_center
                - 0.5 * bbox_scale
            )
            keypoints.append(np.asarray(instance_keypoints[0], dtype=np.float32))
            keypoint_scores.append(np.asarray(instance_scores[0], dtype=np.float32))
        self._record_timing("pose_decode_s", time.perf_counter() - decode_start)
        return keypoints, keypoint_scores

    def __call__(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        total_start = time.perf_counter()
        bbox_start = time.perf_counter()
        bboxes, bbox_scores = self._get_bboxes(frame)
        self._record_timing("bbox_s", time.perf_counter() - bbox_start)
        self._timing_bboxes += int(len(bboxes))
        keypoints_308, scores_308 = self._predict_pose_from_bboxes(frame, bboxes)
        self._timing_pose_instances += int(len(keypoints_308))
        if self._keypoint_schema == SAPIENS2_KEYPOINT_SCHEMA_ORIGINAL:
            keypoints, scores = _format_sapiens2_original_keypoints(
                keypoints_308,
                scores_308,
                self._keypoint_names,
            )
        else:
            keypoints, scores = _map_sapiens2_keypoints_to_halpe26(
                keypoints_308,
                scores_308,
                self._sapiens_name_to_id,
            )
        self._update_adaptive_manual_roi(keypoints, scores, frame.shape)
        classes = np.full((len(bboxes),), PERSON_CLASS_ID, dtype=np.int32)
        self._last_detections = {
            "boxes": np.asarray(bboxes, dtype=np.float32),
            "classes": classes,
            "scores": np.asarray(bbox_scores, dtype=np.float32),
            "person_boxes": np.asarray(bboxes, dtype=np.float32),
            "ball_boxes": np.empty((0, 4), dtype=np.float32),
            "ball_scores": np.empty((0,), dtype=np.float32),
        }
        self._frame_count += 1
        self._timing_frames += 1
        self._record_timing("backend_total_s", time.perf_counter() - total_start)
        return keypoints, scores

    def reset(self) -> None:
        self._frame_count = 0
        self._prev_bboxes = None
        self._prev_bbox_scores = None
        self._last_detections = self._empty_detections()
        self._active_manual_person_roi = self._manual_person_roi
        self._manual_roi_miss_count = 0
        self._last_full_frame_reacquire_frame = None
        for key in list(self._timing):
            self._timing[key] = 0.0
        self._timing_frames = 0
        self._timing_pose_instances = 0
        self._timing_bboxes = 0

    @property
    def skeleton_tree(self):
        return self._skeleton_tree

    @property
    def num_keypoints(self) -> int:
        return len(self._keypoint_names)

    @property
    def backend_name(self) -> str:
        return "sapiens2"

    @property
    def keypoint_names(self):
        return self._keypoint_names

    @property
    def keypoint_schema(self) -> str:
        return self._keypoint_schema

    @property
    def last_detections(self) -> Dict[str, np.ndarray]:
        return self._last_detections
