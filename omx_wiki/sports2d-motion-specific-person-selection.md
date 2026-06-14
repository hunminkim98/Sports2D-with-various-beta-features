---
title: Sports2D motion_specific person selection
tags: ["sports2d", "motion-specific", "person-selection", "broad-jump", "sprint-start", "local-window"]
created: 2026-06-14T03:03:27.074Z
updated: 2026-06-14T04:58:30.000Z
sources: []
links: ["sports2d-motion-specific-person-selection.md"]
category: decision
confidence: medium
schemaVersion: 1
---

# Sports2D motion_specific person selection

## Decision
Use `base.person_ordering_method = 'motion_specific'` for automatic person selection when the target movement should be identified from Sports2D pose coordinates.

The common first-stage gate filters apply only when `person_ordering_method='motion_specific'`, and now differ by target:

- `broad_jump` / general whole-video candidates use the whole-track gates:
  - `presence_ratio >= motion.person_selection_presence_threshold` (default `0.8`)
  - `mean_confidence >= motion.person_selection_confidence_threshold` (default `0.3`)
  - `pose bbox area ratio >= motion.person_selection_size_min_ratio` (default `0.35`)
- `sprint_start` does **not** require whole-video `presence_ratio >= 0.8` or whole-video bbox-size gating because the runner may appear briefly, move fast, or be split across tracker IDs.

These gates are candidate filters, not ranking boosts. A candidate with `presence_ratio=1.0` should not outrank a better motion-specific candidate with `presence_ratio=0.8` only because of higher presence.

## Broad jump rule
After the whole-track gate filters, `broad_jump` is condition-based:

1. There must be a contiguous airborne interval inferred from 2D pose.
   - Flight is approximated as both feet lifted above the observed foot-ground baseline.
2. During that same airborne interval, the hip/body center must move horizontally in either x direction.
3. If multiple candidates satisfy the broad-jump condition, select the largest pose bbox / pose area candidate before confidence.

This intentionally distinguishes broad jump from sprint start: broad jump requires a flight interval and simultaneous horizontal displacement.

## Sprint start local-window rule
`sprint_start` is condition-based and uses a local peak-speed window rather than whole-video presence/bbox gates.

Per person track:

1. Use valid hip + left/right heel coordinates and confidence-qualified frames.
2. Compute horizontal hip speed over a short lag window, normalized by body/pose height (`body-heights/sec`) to reduce perspective effects.
3. Find the track's maximal normalized horizontal speed frame.
4. Evaluate the previous 1 second window ending at that peak frame.
5. Gate the sprint candidate by local-window confidence and enough valid frames, not by whole-video `presence_ratio`.
6. Required motion conditions:
   - fast horizontal hip/body motion
   - heel vertical oscillation
   - left/right heel alternation

Current implementation thresholds/shape:

- speed lag: about `0.10 s`
- local sprint window: `peak_frame - 1.0 s` through `peak_frame`
- minimum local valid frames: `>= 3`
- fast horizontal motion: `peak_speed_norm >= 1.20` and `window_horizontal_displacement_norm >= 0.35`
- heel vertical oscillation: both heel-y ranges `>= 0.12 * body_height`
- heel alternation: anti-phase ratio `>= 0.50` or heel-y correlation `<= -0.35`
- sprint ranking tie-breakers: target score, peak speed, local horizontal displacement, local confidence, local presence, then size

Explicitly do not use these posture/shape helpers for sprint_start:

- initial foot stagger
- trunk lean
- low hip posture
- acceleration ratio

The intended distinction is:

- `broad_jump`: feet airborne together + x movement during that airborne phase
- `sprint_start`: fast horizontal motion + alternating heel-y pendulum pattern in the local peak-speed window

## Implementation touchpoints

- `Sports2D/process.py`
  - `_broad_jump_airborne_motion_features()`
  - `_sprint_start_motion_features()`
  - `resolve_personIDs_for_motion_specific()`
  - `process_fun(... fps=...)` passes video fps into motion-specific scoring
- `Sports2D/Utilities/tests.py`
  - motion_specific gate tests
  - broad_jump flight/x tests
  - sprint_start local-window, static-heel, and in-phase-heel tests
- `Sports2D/Sports2D.py`, `README.md`, `refactory.md`
  - config/help/docs notes about sprint_start local-window exception

## Verification evidence from 2026-06-14 sprint_start retest

The user reported the sprint_start selection now works well after this change.

Fresh verification:

- Windows conda env: `C:\Users\gnsal\miniconda3\envs\sports2d\python.exe`
- `py_compile Sports2D/Sports2D.py Sports2D/process.py Sports2D/Utilities/tests.py`: passed
- `pytest -q Sports2D/Utilities/tests.py -k "motion_specific or config_help_mentions_medicine_ball_person_ordering"`: `10 passed, 173 deselected`
- focused sprint local-window tests: `3 passed, 180 deselected`
- broader utility subset excluding two unrelated environment/demo failures: `181 passed, 2 deselected`
- changed-file flake (`E9,F63,F7,F82`): `0`
- `git diff --check` for touched files: clean
- OMX active state after cleanup: `{"active_modes":[]}`

Demo rerun artifacts:

- config: `Sports2D/Demo/motion_specific_test_outputs/configs/sprint_start_motion_specific_local_window.toml`
- output root: `Sports2D/Demo/motion_specific_test_outputs/sprint_start_local_window/`
- log: `Sports2D/Demo/motion_specific_test_outputs/logs/sprint_start_local_window_run.log`
- result folders: 4, each with processed mp4 + TRC + MOT
- log summary: `Reordered persons` = 4, `Falling back` = 0

Selected tracks in the sprint_start retest:

- `IMG_6549`: selected original person `[2] -> [0]`, fallback `False`
- `IMG_6550`: selected original person `[2] -> [0]`, fallback `False`; whole-video presence was only about `0.679`, proving the sprint local-window path bypasses the old global `presence_ratio >= 0.8` gate
- `IMG_6551`: selected original person `[0] -> [0]`, fallback `False`
- `IMG_6552`: selected original person `[1] -> [0]`, fallback `False`

Known notes:

- Some source videos have unreadable tail frames; overlay saving reused the last readable frame. This did not block output generation.
- Full `Sports2D/Utilities/tests.py` still has two unrelated environment/demo failures in this checkout: missing `Demo/side_Sports2D/side_Sports2D_m_person00.trc` and a `Config_demo` video path issue. The sprint/motion-specific validation passed independently.

## Related pages

- [[sports2d-motion-specific-person-selection]]
