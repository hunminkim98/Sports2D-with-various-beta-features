<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-02-13 | Updated: 2026-04-05 -->

# AGENTS.md

<!-- MANUAL SECTION -->
## Parent
- See `../AGENTS.md` for higher-priority defaults.

## Scope
- Governs reusable utilities and pipeline glue in `Sports2D/Utilities`.
- Includes backend abstraction, pose conversion, filtering, tests, and logging.

## Design constraints
- Preserve the `PoseBackend` abstraction and its behavior (`__call__`, `reset`, `skeleton_tree`, `num_keypoints`, `backend_name`, `keypoint_names`).
- Keep backend-specific code behind the factory path and avoid unconditional imports of optional SynthPose modules.
- Keep shared constants in a single source of truth and avoid cross-file duplication.
- Prefer deterministic, unit-level changes to `Utilities/tests.py` or related helpers when touch points affect output formats.

## Verification expectations
- Update tests when signatures or default behavior changes in `pose_backend.py`.
- Use `pytest -v Sports2D/Utilities/tests.py` as the baseline for utility changes.
- Keep error messages explicit for optional dependencies (for example missing SynthPose extras).
<!-- AUTO-GENERATED SECTION -->
Last reviewed: 2026-04-05

## Purpose
Shared utility layer for the Sports2D runtime. This folder contains backend abstraction, skeleton metadata, tracking and review helpers, realtime-display adapters, SAM3 plumbing, common math/output helpers, and the main automated test module. Checked-in `*_Sports2D/` bundles, logs, caches, and any nested `pose_ball/` exports here are parent-covered artifacts, excluded from AGENTS scoring, and not child domains.

## Key Files
| File | Description |
|------|-------------|
| `pose_backend.py` | Defines the `PoseBackend` abstraction plus RTMLib and SynthPose backend factories/adapters. |
| `sam3_detector.py` | SAM3 runtime adapter supporting Hugging Face bundles and raw-checkpoint Meta runtime resolution for promptable detection. |
| `synthpose_tracker.py` | SynthPose tracker integration, detector selection, and optional RT-DETR/RT-DETRv4 handling. |
| `synthpose_skeleton.py` | 52-keypoint SynthPose skeleton definition and related constants. |
| `common.py` | Shared angle definitions, calibration/output helpers, and core utility functions reused across the pipeline. |
| `realtime_display.py` | Backend-neutral realtime display factory used by the processing loop. |
| `realtime_qt.py` | Optional PySide6-based realtime UI implementation. |
| `tests.py` | Main pytest module covering CLI/API workflow behavior and utility regressions. |
| `__init__.py` | Utility package init and export surface. |

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `*_Sports2D/` | Generated smoke-test or local-run result bundles, including nested `pose_ball/`, videos, TRCs, MOTs, and plots. Parent-covered, excluded from scoring, no child `AGENTS.md`. |

## For AI Agents

### Working In This Directory
- Keep backend-specific imports behind `pose_backend.py` and tracker helpers unless a broader API migration is deliberate.
- Treat `logs.txt*`, `__pycache__/`, generated `*_Sports2D/` bundles, nested `pose_ball/`, and similar local-run folders as disposable parent-covered artifacts; they should not become hidden dependencies for tests or get child `AGENTS.md` files.

### Testing Requirements
- Update or extend `tests.py` when behavior, signatures, defaults, or export-helper contracts change in utility modules.
- Verify optional-dependency failure paths remain explicit and actionable, especially for SynthPose and PySide6 features.

## Dependencies

### Internal
- `Sports2D/process.py` is the primary consumer of this directory.
- `Sports2D/models/` and `Sports2D/Demo/Config_demo.toml` provide model/setup expectations used by SynthPose-related helpers.

### External
- Core integrations include `Pose2Sim`, `rtmlib`, `numpy`, `opencv-python`, and optional extras such as `torch`, `transformers`, the official Meta `sam3` runtime, and `PySide6`.
