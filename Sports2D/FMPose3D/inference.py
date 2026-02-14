#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Reusable FMPose3D inference helpers for Sports2D integration.
"""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Dict, Sequence, Tuple

import numpy as np
import torch


FMPOSE3D_ROOT = Path(__file__).resolve().parent
if str(FMPOSE3D_ROOT) not in sys.path:
    sys.path.insert(0, str(FMPOSE3D_ROOT))

from fmpose3d.common.camera import camera_to_world
from fmpose3d.models import Model as CFM
try:
    from pre_trained_models.model_GAMLP import Model as LEGACY_CFM
except Exception:
    LEGACY_CFM = None


DEFAULT_WORLD_ROTATION = np.array(
    [0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088],
    dtype=np.float32,
)
DEFAULT_JOINTS_LEFT = [4, 5, 6, 11, 12, 13]
DEFAULT_JOINTS_RIGHT = [1, 2, 3, 14, 15, 16]


def resolve_device(device: str = "auto") -> torch.device:
    req = (device or "auto").lower()
    if req == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if req == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("fmpose3d_device='cuda' requested, but CUDA is not available.")
        return torch.device("cuda")
    if req == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("fmpose3d_device='mps' requested, but MPS is not available.")
        return torch.device("mps")
    if req == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported fmpose3d_device: {device}")


def _build_model_args(frames: int = 1, layers: int = 5) -> SimpleNamespace:
    return SimpleNamespace(
        layers=int(layers),
        channel=512,
        d_hid=1024,
        token_dim=256,
        n_joints=17,
        frames=int(frames),
        pad=(int(frames) - 1) // 2,
        joints_left=DEFAULT_JOINTS_LEFT,
        joints_right=DEFAULT_JOINTS_RIGHT,
    )


def load_fmpose3d_model(
    checkpoint_path: str | Path,
    device: str = "auto",
    frames: int = 1,
) -> Tuple[torch.nn.Module, Dict[str, object]]:
    ckpt = Path(checkpoint_path)
    if not ckpt.is_file():
        raise FileNotFoundError(f"FMPose3D checkpoint not found: {ckpt}")

    torch_device = resolve_device(device)
    checkpoint = torch.load(str(ckpt), map_location=torch_device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, dict):
        raise RuntimeError("Unexpected FMPose3D checkpoint format: expected a state_dict dictionary.")
    keys = set(checkpoint.keys())
    checkpoint_is_legacy = any(k.startswith("encoder.") for k in keys)
    checkpoint_is_modern = any(k.startswith("encoder_pose_2d.") for k in keys)

    candidates = []
    if checkpoint_is_legacy and LEGACY_CFM is not None:
        candidates.append((LEGACY_CFM, _build_model_args(frames=frames, layers=5), "legacy"))
    if checkpoint_is_modern or not checkpoint_is_legacy:
        candidates.append((CFM, _build_model_args(frames=frames, layers=3), "modern"))
    if LEGACY_CFM is not None and not checkpoint_is_legacy:
        candidates.append((LEGACY_CFM, _build_model_args(frames=frames, layers=5), "legacy"))

    errors = []
    for model_cls, model_args, model_kind in candidates:
        model = model_cls(model_args).to(torch_device)
        try:
            try:
                model.load_state_dict(checkpoint, strict=True)
            except RuntimeError:
                cleaned = {k.replace("module.", "", 1): v for k, v in checkpoint.items()}
                model.load_state_dict(cleaned, strict=True)
            model.eval()
            return model, {
                "device": torch_device,
                "pad": model_args.pad,
                "joints_left": list(model_args.joints_left),
                "joints_right": list(model_args.joints_right),
            }
        except RuntimeError as exc:
            errors.append(f"{model_kind}: {exc}")

    details = " | ".join(errors) if errors else "no compatible model candidate found"
    raise RuntimeError(f"Failed to load FMPose3D checkpoint with available model definitions: {details}")


def prepare_fmpose3d_input(
    keypoints_h36m_norm: np.ndarray,
    joints_left: Sequence[int] | None = None,
    joints_right: Sequence[int] | None = None,
    augment: bool = True,
) -> torch.Tensor:
    arr = np.asarray(keypoints_h36m_norm, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[1:] != (17, 2):
        raise ValueError(f"Expected input shape (T,17,2), got {arr.shape}.")
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    jl = list(joints_left or DEFAULT_JOINTS_LEFT)
    jr = list(joints_right or DEFAULT_JOINTS_RIGHT)
    non_flip = arr[np.newaxis, ...]  # (1,T,17,2)

    if augment:
        flip = non_flip.copy()
        flip[..., 0] *= -1
        flip[:, :, jl + jr, :] = flip[:, :, jr + jl, :]
        model_input = np.stack([non_flip[0], flip[0]], axis=0)[np.newaxis, ...]  # (1,2,T,17,2)
    else:
        model_input = non_flip[:, np.newaxis, ...]  # (1,1,T,17,2)

    return torch.from_numpy(model_input.astype(np.float32))


def _euler_sample(
    model: torch.nn.Module,
    pose_2d: torch.Tensor,
    sample_steps: int,
) -> torch.Tensor:
    sample_steps = int(sample_steps)
    if sample_steps < 1:
        raise ValueError("sample_steps must be >= 1")

    y_t = torch.randn(
        pose_2d.size(0),
        pose_2d.size(1),
        pose_2d.size(2),
        3,
        device=pose_2d.device,
        dtype=pose_2d.dtype,
    )
    dt = 1.0 / sample_steps
    for s in range(sample_steps):
        t_s = torch.full(
            (pose_2d.size(0), 1, 1, 1),
            s * dt,
            device=pose_2d.device,
            dtype=pose_2d.dtype,
        )
        v_s = model(pose_2d, y_t, t_s)
        y_t = y_t + dt * v_s
    return y_t


def infer_pose3d_sequence(
    model: torch.nn.Module,
    model_input: torch.Tensor,
    sample_steps: int = 3,
    device: torch.device | None = None,
    joints_left: Sequence[int] | None = None,
    joints_right: Sequence[int] | None = None,
    world_rotation: np.ndarray | None = None,
    world_translation: float = 0.0,
    root_joint: int = 0,
) -> np.ndarray:
    if model_input.ndim != 5 or model_input.shape[-2:] != (17, 2):
        raise ValueError(f"Expected model input shape (B,V,T,17,2), got {tuple(model_input.shape)}")
    if model_input.size(1) < 1:
        raise ValueError("Expected at least one view in model_input (shape B,V,T,17,2).")

    jl = list(joints_left or DEFAULT_JOINTS_LEFT)
    jr = list(joints_right or DEFAULT_JOINTS_RIGHT)
    rot = np.asarray(world_rotation if world_rotation is not None else DEFAULT_WORLD_ROTATION, dtype=np.float32)
    infer_device = device or next(model.parameters()).device
    inp = model_input.to(infer_device)

    with torch.no_grad():
        out_non_flip = _euler_sample(model, inp[:, 0], sample_steps=sample_steps)

        if inp.size(1) > 1:
            out_flip = _euler_sample(model, inp[:, 1], sample_steps=sample_steps)
            out_flip[:, :, :, 0] *= -1
            out_flip[:, :, jl + jr, :] = out_flip[:, :, jr + jl, :]
            output_3d = (out_non_flip + out_flip) / 2.0
        else:
            output_3d = out_non_flip

        output_3d[:, :, root_joint, :] = 0
        pose3d_cam = output_3d[0].detach().cpu().numpy().astype(np.float32)  # (T,17,3)

    pose3d_world = camera_to_world(pose3d_cam, R=rot, t=world_translation).astype(np.float32)
    pose3d_world[:, :, 2] -= np.min(pose3d_world[:, :, 2], axis=1, keepdims=True)
    return np.nan_to_num(pose3d_world, nan=0.0, posinf=0.0, neginf=0.0)
