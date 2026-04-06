<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-02-13 | Updated: 2026-04-05 -->

# AGENTS.md

<!-- MANUAL SECTION -->
## Parent
- See `../AGENTS.md` for repository defaults.

## Scope
- Applies to shipped runtime code under `Sports2D/` including Python modules and package entry points.
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
Last reviewed: 2026-04-05

## Purpose
Shipped runtime source of truth for the published Sports2D CLI and processing pipeline. This directory owns the runtime API, configuration defaults, demo bundle, utility modules, and packaged model-integration notes. It also defines hybrid YOLOX+SAM3 ball handling plus saved `pose_ball/` and TRC ball-marker export behavior.

## Key Files
| File | Description |
|------|-------------|
| `Sports2D.py` | CLI entry point, default configuration map, config loading, and top-level `process()` orchestration. |
| `process.py` | Main video/webcam processing loop, drawing logic, tracking, filtering, output generation, backend dispatch, and saved ball export persistence (`pose_ball/`, TRC ball marker). |
| `__init__.py` | Package export surface used by the console entry point and Python API. |

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `Demo/` | Demo config files, raw sample media, and generated example outputs (see `Demo/AGENTS.md`). |
| `Utilities/` | Shared helpers, backend abstraction, realtime display helpers, and test coverage (see `Utilities/AGENTS.md`). |
| `models/` | Packaged model subtree and detector-specific vendor guidance, distinct from the root-level `models/` stash (see `models/AGENTS.md`). |

## For AI Agents

### Working In This Directory
- Keep `Sports2D.py`, `process.py`, `Demo/Config_demo.toml`, and `README.md` aligned when adding or renaming CLI/config options or changing output bundle contents.
- Treat `process.py` as the behavioral hot path: prefer small, isolated edits and verify optional-backend imports remain lazy where intended.
- Let `Demo/` and `Utilities/` child AGENTS own their source scopes; generated run artifacts beneath them stay parent-covered, not new AGENTS domains.

### Testing Requirements
- Use `pytest -v Sports2D/Utilities/tests.py` after logic changes.
- Use `sports2d` or `sports2d --config Sports2D/Demo/Config_demo.toml` as a smoke path when config or CLI behavior changes.

## Dependencies

### Internal
- `Utilities/pose_backend.py` and `Utilities/common.py` define key abstractions and shared computations used by `process.py`.
- `Demo/Config_demo.toml` and `models/` README notes document supported runtime options.

### External
- Core dependencies include `Pose2Sim`, `numpy`, `opencv-python`, `pandas`, `matplotlib`, and optional extras for SynthPose and realtime Qt UI.
