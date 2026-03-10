<!-- Generated: 2026-02-13 | Updated: 2026-03-10 -->

# AGENTS.md

<!-- MANUAL SECTION -->
## Scope and inheritance
- This file is the root instruction layer for the whole repository.
- Read the nearest `AGENTS.md` first when working in any subdirectory.
- If a child file exists, child instructions override these where they conflict.
- Keep this file stable and small; move changing detail into relevant child files.

## Repository-wide defaults
- Use existing project commands in `CLAUDE.md` when available:
  - `pytest -v Sports2D/Utilities/tests.py`
  - `flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics`
  - `sports2d` for a basic functional smoke check.
- Prefer minimal, reviewable edits over broad refactors.
- Keep AI-oriented guidance separate from user feature docs.
- Never commit secrets, local credentials, or machine-specific test artifacts.

## Directory ownership model
- `Sports2D/` contains core library and execution pipeline.
- `SynthPose_PM/` is an optional subproject with separate runtime expectations.
- `Content/` is artifact and media-only.
- `build/` is generated packaging output.
- `.github/` owns automation and release workflows.

## Maintenance policy
- Keep each AGENTS file concise and scoped.
- Preserve manual sections across regeneration.
- Use stable headings so future tooling can update safely.
<!-- AUTO-GENERATED SECTION -->
Last reviewed: 2026-03-10

## Purpose
Repository root for packaging metadata, user-facing documentation, CI entry points, and high-level agent guidance. Shipped runtime code lives in `Sports2D/`; the other top-level directories mainly support workflows, publication assets, or currently inactive scaffolding. Recent backend work added hybrid SAM3 ball detection, `pose_ball/` result artifacts, and expanded model/runtime guidance that is documented from here downward.

## Key Files
| File | Description |
|------|-------------|
| `pyproject.toml` | Packaging metadata, dependencies, package-data rules, and console entry points (`sports2d`, `tests_sports2d`). |
| `README.md` | Primary installation, usage, and output-format documentation. |
| `CLAUDE.md` | Repo-specific development commands and architecture notes for the backend system. |
| `HANDOFF.md` | Handoff context for the unified pose-backend and SynthPose refactor. |
| `refactory.md` | Korean refactoring notes, including recent angle-output behavior changes. |
| `test_synthpose_integration.py` | Manual integration checks for SynthPose skeleton/tracker wiring. |
| `CITATION.cff` | Citation metadata for releases and academic references. |

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `.github/` | GitHub Actions workflows and release automation (see `.github/AGENTS.md`). |
| `Content/` | Paper sources and media assets used by docs/publication flows (see `Content/AGENTS.md`). |
| `Sports2D/` | Published Python package, CLI, processing pipeline, demo assets, and model notes (see `Sports2D/AGENTS.md`). |
| `src/` | Placeholder clean-architecture scaffold with no tracked runtime source files today (see `src/AGENTS.md`). |

## For AI Agents

### Working In This Directory
- Treat `Sports2D/` as the source of truth for shipped behavior and public entry points.
- Keep top-level docs aligned with CLI/config changes that alter installation, dependencies, or output expectations, especially when result bundle contents change.
- Avoid committing generated logs, caches, or local result artifacts unless the task is explicitly about those outputs.

### Testing Requirements
- Use the baseline commands already recorded in the manual section and `CLAUDE.md`.
- Prefer targeted verification for docs-only or workflow-only edits, but keep lint/syntax gates in mind for code-bearing changes.

## Dependencies

### Internal
- `Sports2D/Utilities/` provides the backend abstraction and tests that most runtime changes eventually touch.
- `.github/workflows/` mirrors the repository's supported install, lint, and test paths.

### External
- Packaging is driven by `setuptools`, `wheel`, and `setuptools-scm`.
- Runtime behavior depends primarily on `Pose2Sim`, `imageio_ffmpeg`, and optional extras such as `torch`, `torchvision`, `transformers`, `PySide6`, and the official Meta `sam3` runtime when raw SAM3 checkpoints are used.

Covered AGENTS files:
- `AGENTS.md`
- `.github/AGENTS.md`
- `.github/workflows/AGENTS.md`
- `Content/AGENTS.md`
- `Sports2D/AGENTS.md`
- `Sports2D/Demo/AGENTS.md`
- `Sports2D/Utilities/AGENTS.md`
- `Sports2D/models/AGENTS.md`
- `Sports2D/models/sam3/AGENTS.md`
- `Sports2D/models/rtdetrv4/AGENTS.md`
- `src/AGENTS.md`

Skipped generated or runtime-state directories:
- `.omx/`, `.pytest_cache/`, `.sisyphus/`, `sports2d.egg-info/`, and `**/__pycache__/`
- Empty placeholder model directory `Sports2D/models/RT-DETRv4/`

Suggested upkeep:
- Refresh the covered list whenever AGENTS files are added or removed.
- Reassess `src/` coverage only if tracked source files are added there.
