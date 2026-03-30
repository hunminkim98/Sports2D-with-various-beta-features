#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Shared SAM3 helper logic for image-mode and video-mode adapters."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple
import logging

import numpy as np


PERSON_CLASS_ID = 0
SPORTS_BALL_CLASS_ID = 32
DEFAULT_SAM3_TARGET = "ball"
BALL_ONLY_SAM3_PROMPTS = ["sports ball"]

SAM3_TARGET_PROMPTS = {
    "ball": ["person", "sports ball"],
    "broad_jump": ["person"],
}

SAM3_TARGET_ALIASES = {
    "broad jump": "broad_jump",
    "broadjump": "broad_jump",
}


def normalize_sam3_target(target: Optional[str]) -> str:
    """Normalize task presets so config accepts a few human-friendly aliases."""
    value = str(target or DEFAULT_SAM3_TARGET).strip().lower().replace("-", "_")
    value = SAM3_TARGET_ALIASES.get(value, value)
    if value not in SAM3_TARGET_PROMPTS:
        logging.warning(
            "Unknown sam3_target '%s'. Falling back to '%s'.",
            target,
            DEFAULT_SAM3_TARGET,
        )
        return DEFAULT_SAM3_TARGET
    return value


def resolve_sam3_prompts(target: Optional[str]) -> List[str]:
    """Return the prompt preset for a SAM3 tracking target."""
    normalized_target = normalize_sam3_target(target)
    return list(SAM3_TARGET_PROMPTS[normalized_target])


def sam3_prompt_to_class_id(prompt: str, prompt_index: int) -> int:
    """Map prompt text onto Sports2D's existing class-id conventions."""
    text = str(prompt).strip().lower()
    if "ball" in text:
        return SPORTS_BALL_CLASS_ID
    if "person" in text:
        return PERSON_CLASS_ID
    return 1000 + int(prompt_index)


def empty_sam3_detections(store_masks: bool = False) -> Dict[str, Any]:
    """Return the normalized detection schema used by Sports2D backends."""
    empty = {
        "boxes": np.empty((0, 4), dtype=np.float32),
        "classes": np.empty((0,), dtype=np.int32),
        "scores": np.empty((0,), dtype=np.float32),
        "person_boxes": np.empty((0, 4), dtype=np.float32),
        "ball_boxes": np.empty((0, 4), dtype=np.float32),
        "ball_scores": np.empty((0,), dtype=np.float32),
        "class_names": np.empty((0,), dtype=object),
        "prompt_indices": np.empty((0,), dtype=np.int32),
    }
    if store_masks:
        empty["masks"] = []
    return empty


def to_numpy(value: Any, dtype=None) -> np.ndarray:
    """Convert tensor-like values to numpy arrays."""
    if value is None:
        return np.asarray([], dtype=dtype)
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    array = np.asarray(value)
    if dtype is not None:
        array = array.astype(dtype, copy=False)
    return array


def ensure_xyxy_boxes(boxes: Any) -> np.ndarray:
    """Normalize boxes to Nx4 xyxy float32."""
    boxes = to_numpy(boxes, dtype=np.float32)
    if boxes.size == 0:
        return np.empty((0, 4), dtype=np.float32)
    if boxes.ndim == 1:
        boxes = boxes.reshape(1, -1)
    if boxes.shape[1] < 4:
        return np.empty((0, 4), dtype=np.float32)
    return boxes[:, :4].astype(np.float32, copy=False)


def ensure_scores(scores: Any, expected_len: int) -> np.ndarray:
    """Normalize scores to a vector aligned with the expected detection count."""
    scores = to_numpy(scores, dtype=np.float32).reshape(-1)
    if len(scores) == expected_len:
        return scores
    if expected_len == 0:
        return np.empty((0,), dtype=np.float32)
    return np.full((expected_len,), np.nan, dtype=np.float32)


def ensure_prompt_indices(prompt_indices: Any, expected_len: int, prompt_index: int) -> np.ndarray:
    """Normalize prompt indices to a vector aligned with the expected detection count."""
    prompt_indices = to_numpy(prompt_indices, dtype=np.int32).reshape(-1)
    if len(prompt_indices) == expected_len:
        return prompt_indices
    if expected_len == 0:
        return np.empty((0,), dtype=np.int32)
    return np.full((expected_len,), int(prompt_index), dtype=np.int32)


def boxes_from_masks(masks: Any) -> np.ndarray:
    """Recover xyxy boxes from binary masks."""
    masks_array = to_numpy(masks)
    if masks_array.size == 0:
        return np.empty((0, 4), dtype=np.float32)
    if masks_array.ndim == 2:
        masks_array = masks_array[None, ...]

    boxes = []
    for mask in masks_array:
        ys, xs = np.where(mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            continue
        boxes.append([xs.min(), ys.min(), xs.max(), ys.max()])

    if not boxes:
        return np.empty((0, 4), dtype=np.float32)
    return np.asarray(boxes, dtype=np.float32)


def xywh_to_xyxy_boxes(boxes_xywh: Any) -> np.ndarray:
    """Convert absolute xywh boxes to absolute xyxy boxes."""
    boxes_xywh = to_numpy(boxes_xywh, dtype=np.float32)
    if boxes_xywh.size == 0:
        return np.empty((0, 4), dtype=np.float32)
    if boxes_xywh.ndim == 1:
        boxes_xywh = boxes_xywh.reshape(1, -1)
    if boxes_xywh.shape[1] < 4:
        return np.empty((0, 4), dtype=np.float32)
    boxes_xyxy = boxes_xywh[:, :4].astype(np.float32, copy=True)
    boxes_xyxy[:, 2] = boxes_xyxy[:, 0] + boxes_xyxy[:, 2]
    boxes_xyxy[:, 3] = boxes_xyxy[:, 1] + boxes_xyxy[:, 3]
    return boxes_xyxy


def extract_prompt_instances(result: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """
    Normalize a single-prompt SAM3 post-process result.

    The public docs guarantee masks; boxes and scores are derived when absent.
    """
    if not isinstance(result, dict):
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            [],
        )

    masks = result.get("masks")
    boxes = ensure_xyxy_boxes(result.get("boxes"))
    scores = ensure_scores(result.get("scores"), expected_len=len(boxes))

    if len(boxes) == 0 and masks is not None:
        boxes = boxes_from_masks(masks)
        scores = ensure_scores(result.get("scores"), expected_len=len(boxes))

    masks_array = to_numpy(masks)
    if masks_array.size == 0:
        masks_list: List[np.ndarray] = []
    elif masks_array.ndim == 2:
        masks_list = [masks_array]
    else:
        masks_list = [masks_array[i] for i in range(masks_array.shape[0])]

    return boxes, scores, masks_list


def extract_video_outputs(outputs: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], np.ndarray]:
    """Normalize a SAM3 video predictor output payload."""
    if not isinstance(outputs, dict):
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            [],
            np.empty((0,), dtype=np.int32),
        )

    masks = outputs.get("out_binary_masks")
    masks_array = to_numpy(masks)
    if masks_array.size == 0:
        masks_list: List[np.ndarray] = []
    elif masks_array.ndim == 2:
        masks_list = [masks_array]
    else:
        masks_list = [masks_array[i] for i in range(masks_array.shape[0])]

    boxes_xywh = outputs.get("out_boxes_xywh")
    boxes = xywh_to_xyxy_boxes(boxes_xywh)
    if len(boxes) == 0 and len(masks_list) > 0:
        boxes = boxes_from_masks(masks_array)

    scores = ensure_scores(outputs.get("scores"), expected_len=len(boxes))
    obj_ids = to_numpy(outputs.get("out_obj_ids"), dtype=np.int32).reshape(-1)
    if len(obj_ids) != len(boxes):
        obj_ids = np.arange(len(boxes), dtype=np.int32)
    return boxes, scores, masks_list, obj_ids


def build_sam3_detection_metadata(
    *,
    boxes: Any,
    scores: Any,
    prompts: Sequence[str],
    prompt_indices: Any,
    masks: Optional[Sequence[np.ndarray]] = None,
    store_masks: bool = False,
) -> Dict[str, Any]:
    """Build the backend-neutral detection metadata schema."""
    normalized_boxes = ensure_xyxy_boxes(boxes)
    normalized_scores = ensure_scores(scores, expected_len=len(normalized_boxes))
    normalized_prompt_indices = ensure_prompt_indices(
        prompt_indices,
        expected_len=len(normalized_boxes),
        prompt_index=0,
    )

    if len(normalized_boxes) == 0:
        empty = empty_sam3_detections(store_masks=store_masks)
        if store_masks:
            empty["masks"] = list(masks or [])
        return empty

    class_ids = []
    class_names = []
    for prompt_index in normalized_prompt_indices:
        prompt_id = int(prompt_index)
        if 0 <= prompt_id < len(prompts):
            prompt_text = prompts[prompt_id]
        else:
            prompt_text = f"prompt_{prompt_id}"
        class_ids.append(sam3_prompt_to_class_id(prompt_text, prompt_id))
        class_names.append(prompt_text)

    class_ids_array = np.asarray(class_ids, dtype=np.int32)
    class_names_array = np.asarray(class_names, dtype=object)
    person_mask = class_ids_array == PERSON_CLASS_ID
    ball_mask = class_ids_array == SPORTS_BALL_CLASS_ID

    metadata = {
        "boxes": normalized_boxes,
        "classes": class_ids_array,
        "scores": normalized_scores,
        "person_boxes": normalized_boxes[person_mask],
        "ball_boxes": normalized_boxes[ball_mask],
        "ball_scores": normalized_scores[ball_mask],
        "class_names": class_names_array,
        "prompt_indices": normalized_prompt_indices,
    }
    if store_masks:
        metadata["masks"] = list(masks or [])
    return metadata


def xyxy_to_coco(boxes: Any) -> np.ndarray:
    """Convert absolute xyxy boxes into the COCO xywh format expected by VitPose."""
    xyxy_boxes = ensure_xyxy_boxes(boxes)
    if len(xyxy_boxes) == 0:
        return np.empty((0, 4), dtype=np.float32)

    coco_boxes = np.zeros((len(xyxy_boxes), 4), dtype=np.float32)
    coco_boxes[:, 0] = xyxy_boxes[:, 0]
    coco_boxes[:, 1] = xyxy_boxes[:, 1]
    coco_boxes[:, 2] = xyxy_boxes[:, 2] - xyxy_boxes[:, 0]
    coco_boxes[:, 3] = xyxy_boxes[:, 3] - xyxy_boxes[:, 1]
    return coco_boxes
