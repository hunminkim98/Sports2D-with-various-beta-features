#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SAM3 detector adapter for SynthPose.

This module keeps SAM3-specific loading and prompt handling separate from the
main tracker so Sports2D can keep its existing PoseBackend contract.
"""

from __future__ import annotations

from pathlib import Path
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


_torch = None
_Sam3Model = None
_Sam3Processor = None
_build_sam3_image_model = None
_OfficialSam3Processor = None


def _load_torch():
    """Import PyTorch only when a SAM3 runtime is actually used."""
    global _torch

    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "SAM3 requires PyTorch in the target environment before using "
            "synthpose_detector='sam3'."
        ) from exc

    _torch = torch
    return _torch


def _load_transformers_dependencies():
    """Import the Hugging Face SAM3 runtime lazily."""
    global _Sam3Model, _Sam3Processor

    if _Sam3Model is not None and _Sam3Processor is not None:
        return

    _load_torch()

    try:
        from transformers import Sam3Model, Sam3Processor
    except ImportError as exc:
        raise ImportError(
            "Hugging Face SAM3 runtime requires a transformers build that exposes "
            "Sam3Model/Sam3Processor. Install that build or use a raw .pt checkpoint "
            "with the official Meta sam3 package."
        ) from exc

    _Sam3Model = Sam3Model
    _Sam3Processor = Sam3Processor


def _load_meta_dependencies():
    """Import the official Meta SAM3 runtime lazily."""
    global _build_sam3_image_model, _OfficialSam3Processor

    if _build_sam3_image_model is not None and _OfficialSam3Processor is not None:
        return

    _load_torch()

    try:
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor as OfficialSam3Processor
    except ImportError as exc:
        raise ImportError(
            "Raw SAM3 checkpoints (.pt/.pth) require the official Meta sam3 package. "
            "Install the package from facebookresearch/sam3 and keep sam3_model_path "
            "pointing at the checkpoint file. If you want to avoid that runtime, "
            "switch to sam3_runtime='transformers' and use a Hugging Face SAM3 bundle "
            "or repo id instead of a raw checkpoint."
        ) from exc

    _build_sam3_image_model = build_sam3_image_model
    _OfficialSam3Processor = OfficialSam3Processor


def is_sam3_checkpoint_path(model_path: Optional[str]) -> bool:
    """Return True when the configured SAM3 model path looks like a raw checkpoint."""
    value = str(model_path or "").strip()
    if not value:
        return False
    return Path(value).expanduser().suffix.lower() in {".pt", ".pth"}


def resolve_sam3_runtime(runtime: Optional[str], model_path: Optional[str]) -> str:
    """
    Resolve the runtime backend from config and model path.

    Raw checkpoint files are served by the official Meta runtime while HF ids/dirs
    continue to use the transformers adapter.
    """
    normalized_runtime = str(runtime or "transformers").strip().lower()
    normalized_runtime = {
        "": "auto",
        "auto": "auto",
        "official": "meta",
        "native": "meta",
        "meta": "meta",
        "transformers": "transformers",
    }.get(normalized_runtime, normalized_runtime)

    if normalized_runtime not in {"auto", "transformers", "meta"}:
        raise ValueError(
            f"Unsupported sam3_runtime '{runtime}'. Use 'transformers', 'meta', or 'auto'."
        )

    is_checkpoint = is_sam3_checkpoint_path(model_path)
    if normalized_runtime == "auto":
        return "meta" if is_checkpoint else "transformers"
    if normalized_runtime == "transformers" and is_checkpoint:
        logging.info(
            "sam3_model_path '%s' looks like a raw checkpoint, switching runtime to 'meta'.",
            model_path,
        )
        return "meta"
    return normalized_runtime


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


def _to_numpy(value: Any, dtype=None) -> np.ndarray:
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


def _ensure_xyxy_boxes(boxes: Any) -> np.ndarray:
    boxes = _to_numpy(boxes, dtype=np.float32)
    if boxes.size == 0:
        return np.empty((0, 4), dtype=np.float32)
    if boxes.ndim == 1:
        boxes = boxes.reshape(1, -1)
    if boxes.shape[1] < 4:
        return np.empty((0, 4), dtype=np.float32)
    return boxes[:, :4].astype(np.float32, copy=False)


def _ensure_scores(scores: Any, expected_len: int) -> np.ndarray:
    scores = _to_numpy(scores, dtype=np.float32).reshape(-1)
    if len(scores) == expected_len:
        return scores
    if expected_len == 0:
        return np.empty((0,), dtype=np.float32)
    return np.full((expected_len,), np.nan, dtype=np.float32)


def _ensure_prompt_indices(prompt_indices: Any, expected_len: int, prompt_index: int) -> np.ndarray:
    prompt_indices = _to_numpy(prompt_indices, dtype=np.int32).reshape(-1)
    if len(prompt_indices) == expected_len:
        return prompt_indices
    if expected_len == 0:
        return np.empty((0,), dtype=np.int32)
    return np.full((expected_len,), int(prompt_index), dtype=np.int32)


def _boxes_from_masks(masks: Any) -> np.ndarray:
    masks_array = _to_numpy(masks)
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


def extract_prompt_instances(result: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """
    Normalize a single-prompt SAM3 post-process result.

    The public docs guarantee masks; boxes/scores are derived when absent.
    """
    if not isinstance(result, dict):
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            [],
        )

    masks = result.get("masks")
    boxes = _ensure_xyxy_boxes(result.get("boxes"))
    scores = _ensure_scores(result.get("scores"), expected_len=len(boxes))

    if len(boxes) == 0 and masks is not None:
        boxes = _boxes_from_masks(masks)
        scores = _ensure_scores(result.get("scores"), expected_len=len(boxes))

    masks_array = _to_numpy(masks)
    if masks_array.size == 0:
        masks_list: List[np.ndarray] = []
    elif masks_array.ndim == 2:
        masks_list = [masks_array]
    else:
        masks_list = [masks_array[i] for i in range(masks_array.shape[0])]

    return boxes, scores, masks_list


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
    normalized_boxes = _ensure_xyxy_boxes(boxes)
    normalized_scores = _ensure_scores(scores, expected_len=len(normalized_boxes))
    normalized_prompt_indices = _ensure_prompt_indices(
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
    xyxy_boxes = _ensure_xyxy_boxes(boxes)
    if len(xyxy_boxes) == 0:
        return np.empty((0, 4), dtype=np.float32)

    coco_boxes = np.zeros((len(xyxy_boxes), 4), dtype=np.float32)
    coco_boxes[:, 0] = xyxy_boxes[:, 0]
    coco_boxes[:, 1] = xyxy_boxes[:, 1]
    coco_boxes[:, 2] = xyxy_boxes[:, 2] - xyxy_boxes[:, 0]
    coco_boxes[:, 3] = xyxy_boxes[:, 3] - xyxy_boxes[:, 1]
    return coco_boxes


class Sam3Detector:
    """Prompt-driven SAM3 detector adapter used by the SynthPose tracker."""

    def __init__(
        self,
        *,
        model_path: str,
        processor_path: str,
        runtime: str = "transformers",
        device: str = "cpu",
        target: str = DEFAULT_SAM3_TARGET,
        prompts: Optional[Sequence[str]] = None,
        store_masks: bool = False,
        person_threshold: float = 0.3,
        ball_detection_threshold: float = 0.1,
        mask_threshold: float = 0.5,
    ):
        if not str(model_path or "").strip():
            raise ValueError(
                "sam3_model_path must be provided when synthpose_detector='sam3' "
                "or ball_detector_backend='sam3'."
            )

        self.model_path = str(model_path).strip()
        self.runtime = resolve_sam3_runtime(runtime, self.model_path)
        self.processor_path = str(processor_path).strip()
        self.device = device
        self.target = normalize_sam3_target(target)
        if prompts is None:
            self.prompts = resolve_sam3_prompts(self.target)
        else:
            self.prompts = [str(prompt).strip() for prompt in prompts if str(prompt).strip()]
            if len(self.prompts) == 0:
                raise ValueError("SAM3 prompts override must contain at least one non-empty prompt.")
        self.store_masks = bool(store_masks)
        self.score_threshold = float(min(person_threshold, ball_detection_threshold))
        self.mask_threshold = float(np.clip(mask_threshold, 0.0, 1.0))
        self._prompt_contexts: List[Dict[str, Any]] = []

        if self.runtime == "transformers":
            _load_transformers_dependencies()
            self.processor_path = self.processor_path or self.model_path
            self.processor = _Sam3Processor.from_pretrained(self.processor_path)
            self.model = _Sam3Model.from_pretrained(self.model_path).to(self.device)
            self.model.eval()
            self._prompt_contexts = self._prepare_prompt_contexts()
        else:
            _load_meta_dependencies()
            self.processor_path = ""
            self.model = _build_sam3_image_model(
                checkpoint_path=self.model_path,
                load_from_HF=False,
                device=self.device,
                eval_mode=True,
            )
            if hasattr(self.model, "eval"):
                self.model.eval()
            self.processor = _OfficialSam3Processor(self.model)

    @staticmethod
    def _move_to_device(batch: Any, device: str) -> Any:
        if hasattr(batch, "to"):
            return batch.to(device)
        if isinstance(batch, dict):
            moved = {}
            for key, value in batch.items():
                moved[key] = value.to(device) if hasattr(value, "to") else value
            return moved
        return batch

    def _prepare_prompt_contexts(self) -> List[Dict[str, Any]]:
        """Cache text features per prompt so repeated frame inference stays cheap."""
        contexts = []
        for prompt in self.prompts:
            text_inputs = self._move_to_device(
                self.processor(text=prompt, return_tensors="pt"),
                self.device,
            )
            text_embeds = None
            attention_mask = text_inputs.get("attention_mask")
            try:
                with _torch.no_grad():
                    text_features = self.model.get_text_features(
                        input_ids=text_inputs["input_ids"],
                        attention_mask=attention_mask,
                    )
                if hasattr(text_features, "pooler_output") and text_features.pooler_output is not None:
                    text_embeds = text_features.pooler_output
                elif isinstance(text_features, tuple):
                    text_embeds = text_features[-1]
                else:
                    text_embeds = text_features
            except Exception as exc:
                logging.debug("SAM3 text feature cache failed for prompt '%s': %s", prompt, exc)

            contexts.append(
                {
                    "prompt": prompt,
                    "text_embeds": text_embeds,
                    "attention_mask": attention_mask,
                }
            )
        return contexts

    def _post_process(self, outputs: Any, original_sizes: Any) -> Dict[str, Any]:
        if original_sizes is None:
            target_sizes = None
        else:
            target_sizes = original_sizes.tolist() if hasattr(original_sizes, "tolist") else original_sizes
        try:
            results = self.processor.post_process_instance_segmentation(
                outputs,
                threshold=self.score_threshold,
                mask_threshold=self.mask_threshold,
                target_sizes=target_sizes,
            )
        except TypeError:
            results = self.processor.post_process_instance_segmentation(
                outputs,
                target_sizes=target_sizes,
            )
        if isinstance(results, list) and len(results) > 0:
            return results[0]
        return {}

    def _run_transformers_prompt(
        self,
        pil_image,
        prompt_context: Dict[str, Any],
    ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
        image_inputs = self._move_to_device(
            self.processor(images=pil_image, return_tensors="pt"),
            self.device,
        )

        with _torch.no_grad():
            if prompt_context["text_embeds"] is not None:
                outputs = self.model(
                    pixel_values=image_inputs["pixel_values"],
                    text_embeds=prompt_context["text_embeds"],
                    attention_mask=prompt_context["attention_mask"],
                )
            else:
                full_inputs = self._move_to_device(
                    self.processor(
                        images=pil_image,
                        text=prompt_context["prompt"],
                        return_tensors="pt",
                    ),
                    self.device,
                )
                image_inputs = full_inputs
                outputs = self.model(**full_inputs)

        original_sizes = image_inputs.get("original_sizes")
        if original_sizes is None:
            original_sizes = [(pil_image.height, pil_image.width)]
        result = self._post_process(outputs, original_sizes)
        return extract_prompt_instances(result)

    def _run_meta_prompt(
        self,
        inference_state: Any,
        prompt: str,
    ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
        result = self.processor.set_text_prompt(
            state=inference_state,
            prompt=prompt,
        )
        return extract_prompt_instances(result)

    def detect(self, pil_image) -> Dict[str, Any]:
        """Run SAM3 for each configured prompt and aggregate the results."""
        all_boxes = []
        all_scores = []
        all_prompt_indices = []
        all_masks: List[np.ndarray] = []

        if self.runtime == "meta":
            inference_state = self.processor.set_image(pil_image)
            prompt_iterable = enumerate(self.prompts)
        else:
            inference_state = None
            prompt_iterable = enumerate(self._prompt_contexts)

        for prompt_index, prompt_payload in prompt_iterable:
            if self.runtime == "meta":
                boxes, scores, masks = self._run_meta_prompt(inference_state, prompt_payload)
            else:
                boxes, scores, masks = self._run_transformers_prompt(pil_image, prompt_payload)
            if len(boxes) == 0:
                continue
            all_boxes.append(boxes)
            all_scores.append(scores)
            all_prompt_indices.append(
                np.full((len(boxes),), prompt_index, dtype=np.int32)
            )
            if self.store_masks:
                all_masks.extend(masks)

        if not all_boxes:
            return empty_sam3_detections(store_masks=self.store_masks)

        return build_sam3_detection_metadata(
            boxes=np.concatenate(all_boxes, axis=0),
            scores=np.concatenate(all_scores, axis=0),
            prompts=self.prompts,
            prompt_indices=np.concatenate(all_prompt_indices, axis=0),
            masks=all_masks if self.store_masks else None,
            store_masks=self.store_masks,
        )

    def detect_person_boxes(self, pil_image) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Return VitPose-ready person boxes plus normalized detection metadata."""
        metadata = self.detect(pil_image)
        return xyxy_to_coco(metadata["person_boxes"]), metadata
