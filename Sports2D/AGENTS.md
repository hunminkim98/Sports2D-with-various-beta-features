<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-02-13 | Updated: 2026-03-02 -->

# AGENTS.md

<!-- MANUAL SECTION -->
## Parent
- See `../AGENTS.md` for repository defaults.

## Scope
- Applies to core code under `Sports2D/` including Python modules and package entry points.
- Prioritize behavior-safe changes in the processing path (`Sports2D.py`, `process.py`).

## Core constraints
- Keep CLI/API defaults backward compatible unless the task explicitly requires deprecation.
- Do not alter `Sports2D` public API behavior without updating `refactory.md` context notes and sample docs where relevant.
- If pose-backend behavior changes, align `Utilities/pose_backend.py` and `CLAUDE.md` guidance together.
- Prefer refactors that preserve the existing `PoseBackend` contract.

## Practical workflow
- Run `pytest -v Sports2D/Utilities/tests.py` after logic edits.
- Run `flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics` before finalizing.
- Add changelog notes or inline rationale when touching dependency/feature boundaries.
<!-- AUTO-GENERATED SECTION -->
Last reviewed: 2026-03-02

Subtrees tracked here:
- `Utilities/`
- `Demo/`
- `models/`
