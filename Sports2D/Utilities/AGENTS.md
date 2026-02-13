<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-02-13 | Updated: 2026-02-13 -->

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
Last reviewed: 2026-02-13

Topics in this folder:
- `pose_backend.py`
- `synthpose_tracker.py`
- `synthpose_skeleton.py`
- `common.py`
- `tests.py`
