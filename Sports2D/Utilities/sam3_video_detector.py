#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""SAM3.1 video-predictor adapter for the Sports2D hybrid ball path."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import logging

import numpy as np

from Sports2D.Utilities.sam3_common import (
    BALL_ONLY_SAM3_PROMPTS,
    build_sam3_detection_metadata,
    ensure_xyxy_boxes,
    extract_video_outputs,
)
from Sports2D.Utilities.sam3_detector import Sam3Detector, resolve_sam3_runtime


_build_sam3_video_predictor = None


def _load_meta_video_dependencies():
    """Import the official Meta SAM3.1 video predictor lazily."""
    global _build_sam3_video_predictor

    if _build_sam3_video_predictor is not None:
        return _build_sam3_video_predictor

    try:
        from sam3.model_builder import build_sam3_video_predictor
    except ImportError as exc:
        raise ImportError(
            "SAM3.1 video mode requires the official Meta sam3 package with "
            "build_sam3_video_predictor available. Install the package from "
            "facebookresearch/sam3 and use a local raw checkpoint."
        ) from exc

    _build_sam3_video_predictor = build_sam3_video_predictor
    return _build_sam3_video_predictor


def _xyxy_to_xywh(box: Sequence[float]) -> List[float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


class Sam3VideoDetector:
    """Segment-based SAM3.1 detector/tracker for one logical prompt family."""

    STATE_BOOTSTRAP = "BOOTSTRAP"
    STATE_VIDEO_ACTIVE = "VIDEO_ACTIVE"
    STATE_IMAGE_FALLBACK = "IMAGE_FALLBACK"

    def __init__(
        self,
        *,
        model_path: str,
        processor_path: str = "",
        runtime: str = "meta",
        device: str = "cpu",
        prompts: Optional[Sequence[str]] = None,
        store_masks: bool = False,
        person_threshold: float = 0.3,
        ball_detection_threshold: float = 0.1,
        bootstrap_frames: int = 12,
        refresh_frequency: int = 4,
        reseed_on_loss: bool = True,
        loss_patience: int = 3,
        stable_center_px: float = 40.0,
        stable_area_ratio_min: float = 0.5,
        stable_area_ratio_max: float = 2.0,
    ):
        self.model_path = str(model_path or "").strip()
        self.runtime = resolve_sam3_runtime(runtime, self.model_path)
        if self.runtime != "meta":
            raise ValueError(
                "SAM3.1 video mode currently supports only the official Meta runtime "
                "with a local raw checkpoint."
            )

        self.device = str(device or "cpu")
        self.prompts = [str(p).strip() for p in (prompts or BALL_ONLY_SAM3_PROMPTS) if str(p).strip()]
        self.store_masks = bool(store_masks)
        self.bootstrap_frames = max(1, int(bootstrap_frames))
        self.refresh_frequency = max(1, int(refresh_frequency))
        self.reseed_on_loss = bool(reseed_on_loss)
        self.loss_patience = max(1, int(loss_patience))
        self.stable_center_px = float(stable_center_px)
        self.stable_area_ratio_min = float(stable_area_ratio_min)
        self.stable_area_ratio_max = float(stable_area_ratio_max)

        self.image_detector = Sam3Detector(
            model_path=self.model_path,
            processor_path=processor_path,
            runtime=self.runtime,
            device=self.device,
            prompts=self.prompts,
            store_masks=self.store_masks,
            person_threshold=person_threshold,
            ball_detection_threshold=ball_detection_threshold,
        )
        self.video_file_path: Optional[str] = None
        self.frame_index_offset = 0
        self.input_kind = "video"
        self.mode = self.STATE_BOOTSTRAP
        self._predictor = None
        self._session_id: Optional[str] = None
        self._stream = None
        self._frame_cache: Dict[int, Dict[str, Any]] = {}
        self._seed_history: List[Dict[str, Any]] = []
        self._segment_start_frame: Optional[int] = None
        self._fallback_start_frame: Optional[int] = None
        self._last_stream_frame: Optional[int] = None
        self._missing_prediction_count = 0

    def prepare_video_context(self, *, video_file_path, frame_index_offset=0, input_kind="video"):
        """Prepare the detector for a new file-video run."""
        self.close()
        self.video_file_path = str(video_file_path) if video_file_path is not None else None
        self.frame_index_offset = int(frame_index_offset)
        self.input_kind = str(input_kind or "video").strip().lower()
        self.mode = self.STATE_BOOTSTRAP if self.input_kind == "video" and self.video_file_path else self.STATE_IMAGE_FALLBACK
        self._seed_history = []
        self._segment_start_frame = None
        self._fallback_start_frame = None
        self._missing_prediction_count = 0
        if self.mode == self.STATE_BOOTSTRAP:
            _load_meta_video_dependencies()

    def close(self):
        """Close the active predictor session, if any."""
        self._close_session()
        self._seed_history = []
        self._segment_start_frame = None
        self._fallback_start_frame = None
        self._missing_prediction_count = 0
        self.mode = self.STATE_BOOTSTRAP

    def detect(self, pil_image, frame_index: int) -> Dict[str, Any]:
        """Run the current state-machine step for the requested frame."""
        absolute_frame_idx = int(frame_index)

        if self.input_kind != "video" or not self.video_file_path:
            return self.image_detector.detect(pil_image)

        image_meta = self.image_detector.detect(pil_image)

        if self.mode == self.STATE_VIDEO_ACTIVE:
            video_meta = self._advance_video_to_frame(absolute_frame_idx)
            if self._metadata_has_ball(video_meta):
                self._missing_prediction_count = 0
                if (
                    self.reseed_on_loss
                    and self._segment_start_frame is not None
                    and (absolute_frame_idx - self._segment_start_frame) >= self.refresh_frequency
                    and (absolute_frame_idx - self._segment_start_frame) % self.refresh_frequency == 0
                ):
                    self._transition_to_fallback(absolute_frame_idx)
                return video_meta

            self._missing_prediction_count += 1
            if self._missing_prediction_count < self.loss_patience:
                return image_meta
            self._transition_to_fallback(absolute_frame_idx)

        self._record_seed_candidate(image_meta, absolute_frame_idx)
        if self.mode == self.STATE_BOOTSTRAP and self._bootstrap_window_exhausted(absolute_frame_idx):
            self.mode = self.STATE_IMAGE_FALLBACK
            self._fallback_start_frame = absolute_frame_idx

        if self._should_attempt_seed(absolute_frame_idx):
            stable_seed = self._select_stable_seed()
            if stable_seed is not None:
                self._start_segment(
                    seed_frame_idx=int(stable_seed["frame_index"]),
                    seed_box_xyxy=stable_seed["box_xyxy"],
                )
                video_meta = self._advance_video_to_frame(absolute_frame_idx)
                if video_meta is not None:
                    return video_meta

        return image_meta

    def _should_attempt_seed(self, absolute_frame_idx: int) -> bool:
        if self.mode == self.STATE_BOOTSTRAP:
            return True
        if self.mode != self.STATE_IMAGE_FALLBACK or not self.reseed_on_loss:
            return False
        if self._fallback_start_frame is None:
            self._fallback_start_frame = absolute_frame_idx
        return (absolute_frame_idx - self._fallback_start_frame) % self.refresh_frequency == 0

    def _bootstrap_window_exhausted(self, absolute_frame_idx: int) -> bool:
        return (absolute_frame_idx - self.frame_index_offset + 1) >= self.bootstrap_frames

    def _record_seed_candidate(self, image_meta: Dict[str, Any], absolute_frame_idx: int):
        candidate_box, candidate_score = self._select_primary_ball_candidate(image_meta)
        self._seed_history.append(
            {
                "frame_index": absolute_frame_idx,
                "box_xyxy": candidate_box,
                "score": candidate_score,
            }
        )
        max_history = max(self.bootstrap_frames, self.refresh_frequency) + 2
        if len(self._seed_history) > max_history:
            self._seed_history = self._seed_history[-max_history:]

    def _select_primary_ball_candidate(self, image_meta: Dict[str, Any]) -> Tuple[Optional[List[float]], float]:
        ball_boxes = ensure_xyxy_boxes((image_meta or {}).get("ball_boxes"))
        if len(ball_boxes) == 0:
            return None, float("nan")
        ball_scores = np.asarray((image_meta or {}).get("ball_scores", []), dtype=np.float32).reshape(-1)
        if len(ball_scores) == len(ball_boxes) and np.any(np.isfinite(ball_scores)):
            best_idx = int(np.nanargmax(ball_scores))
            return ball_boxes[best_idx].astype(np.float32).tolist(), float(ball_scores[best_idx])
        return ball_boxes[0].astype(np.float32).tolist(), float("nan")

    def _select_stable_seed(self) -> Optional[Dict[str, Any]]:
        recent_entries = [entry for entry in self._seed_history if entry.get("box_xyxy") is not None]
        if len(recent_entries) < 2:
            return None

        runs: List[Dict[str, Any]] = []
        current_run: List[Dict[str, Any]] = []
        for entry in recent_entries:
            if not current_run:
                current_run = [entry]
                continue
            prev = current_run[-1]
            if self._candidates_match(prev["box_xyxy"], entry["box_xyxy"]):
                current_run.append(entry)
            else:
                runs.append(self._summarize_seed_run(current_run))
                current_run = [entry]
        if current_run:
            runs.append(self._summarize_seed_run(current_run))

        valid_runs = [run for run in runs if run["run_length"] >= 2]
        if len(valid_runs) == 0:
            return None
        valid_runs.sort(
            key=lambda run: (
                -int(run["run_length"]),
                -float(run["mean_score"]) if np.isfinite(run["mean_score"]) else float("inf"),
                int(run["first_frame"]),
            )
        )
        return valid_runs[0]

    def _candidates_match(self, left_box: Sequence[float], right_box: Sequence[float]) -> bool:
        left = ensure_xyxy_boxes(left_box)
        right = ensure_xyxy_boxes(right_box)
        if len(left) == 0 or len(right) == 0:
            return False
        left = left[0]
        right = right[0]
        left_center = ((left[0] + left[2]) / 2.0, (left[1] + left[3]) / 2.0)
        right_center = ((right[0] + right[2]) / 2.0, (right[1] + right[3]) / 2.0)
        center_dist = float(np.hypot(left_center[0] - right_center[0], left_center[1] - right_center[1]))
        left_area = max(1.0, float((left[2] - left[0]) * (left[3] - left[1])))
        right_area = max(1.0, float((right[2] - right[0]) * (right[3] - right[1])))
        area_ratio = right_area / left_area
        return (
            center_dist <= self.stable_center_px
            and self.stable_area_ratio_min <= area_ratio <= self.stable_area_ratio_max
        )

    def _summarize_seed_run(self, run_entries: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        scores = np.asarray([entry["score"] for entry in run_entries], dtype=np.float32)
        finite_scores = scores[np.isfinite(scores)]
        mean_score = float(np.mean(finite_scores)) if len(finite_scores) > 0 else float("nan")
        best_entry = run_entries[-1]
        return {
            "run_length": len(run_entries),
            "mean_score": mean_score,
            "first_frame": int(run_entries[0]["frame_index"]),
            "frame_index": int(best_entry["frame_index"]),
            "box_xyxy": list(best_entry["box_xyxy"]),
        }

    def _start_segment(self, *, seed_frame_idx: int, seed_box_xyxy: Sequence[float]):
        predictor_builder = _load_meta_video_dependencies()
        self._close_session()

        self._predictor = predictor_builder(checkpoint_path=self.model_path)
        response = self._predictor.handle_request(
            {
                "type": "start_session",
                "resource_path": self.video_file_path,
            }
        )
        self._session_id = response["session_id"]
        seed_response = self._predictor.handle_request(
            {
                "type": "add_prompt",
                "session_id": self._session_id,
                "frame_index": int(seed_frame_idx),
                "text": self.prompts[0],
                "bounding_boxes": [_xyxy_to_xywh(seed_box_xyxy)],
                "bounding_box_labels": [1],
            }
        )
        self._frame_cache = {
            int(seed_response["frame_index"]): self._normalize_outputs(seed_response.get("outputs", {}))
        }
        self._stream = self._predictor.handle_stream_request(
            {
                "type": "propagate_in_video",
                "session_id": self._session_id,
                "propagation_direction": "forward",
                "start_frame_index": int(seed_frame_idx),
            }
        )
        self._last_stream_frame = int(seed_frame_idx)
        self._segment_start_frame = int(seed_frame_idx)
        self._missing_prediction_count = 0
        self.mode = self.STATE_VIDEO_ACTIVE
        logging.info(
            "SAM3.1 video segment started at frame %s for prompt '%s'.",
            seed_frame_idx,
            self.prompts[0],
        )

    def _advance_video_to_frame(self, absolute_frame_idx: int) -> Optional[Dict[str, Any]]:
        if absolute_frame_idx in self._frame_cache:
            return self._frame_cache[absolute_frame_idx]
        if self._stream is None:
            return None
        if self._last_stream_frame is not None and absolute_frame_idx < self._last_stream_frame:
            return None

        while absolute_frame_idx not in self._frame_cache:
            try:
                event = next(self._stream)
            except StopIteration:
                return None
            frame_idx = int(event.get("frame_index"))
            self._frame_cache[frame_idx] = self._normalize_outputs(event.get("outputs", {}))
            self._last_stream_frame = frame_idx
            if frame_idx >= absolute_frame_idx:
                break
        return self._frame_cache.get(absolute_frame_idx)

    def _normalize_outputs(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        boxes, scores, masks, _obj_ids = extract_video_outputs(outputs)
        prompt_indices = np.zeros((len(boxes),), dtype=np.int32)
        metadata = build_sam3_detection_metadata(
            boxes=boxes,
            scores=scores,
            prompts=self.prompts,
            prompt_indices=prompt_indices,
            masks=masks if self.store_masks else None,
            store_masks=self.store_masks,
        )
        return metadata

    @staticmethod
    def _metadata_has_ball(metadata: Optional[Dict[str, Any]]) -> bool:
        if not isinstance(metadata, dict):
            return False
        return len(ensure_xyxy_boxes(metadata.get("ball_boxes"))) > 0

    def _transition_to_fallback(self, absolute_frame_idx: int):
        self._close_session()
        self.mode = self.STATE_IMAGE_FALLBACK
        self._fallback_start_frame = int(absolute_frame_idx)
        self._missing_prediction_count = 0

    def _close_session(self):
        if self._predictor is not None and self._session_id is not None:
            try:
                self._predictor.handle_request(
                    {
                        "type": "close_session",
                        "session_id": self._session_id,
                    }
                )
            except Exception as exc:
                logging.debug("SAM3.1 predictor close_session failed: %s", exc)
        if self._predictor is not None and hasattr(self._predictor, "shutdown"):
            try:
                self._predictor.shutdown()
            except Exception as exc:
                logging.debug("SAM3.1 predictor shutdown failed: %s", exc)
        self._predictor = None
        self._session_id = None
        self._stream = None
        self._frame_cache = {}
        self._last_stream_frame = None
