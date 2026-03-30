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

from Sports2D.Utilities.sam3_common import (
    BALL_ONLY_SAM3_PROMPTS,
    DEFAULT_SAM3_TARGET,
    PERSON_CLASS_ID,
    SPORTS_BALL_CLASS_ID,
    build_sam3_detection_metadata,
    empty_sam3_detections,
    ensure_prompt_indices as _ensure_prompt_indices,
    ensure_scores as _ensure_scores,
    ensure_xyxy_boxes as _ensure_xyxy_boxes,
    extract_prompt_instances,
    normalize_sam3_target,
    resolve_sam3_prompts,
    to_numpy as _to_numpy,
    xyxy_to_coco,
)


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
