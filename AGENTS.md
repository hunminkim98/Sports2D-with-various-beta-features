<!-- Generated: 2026-02-13 | Updated: 2026-05-02 -->

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
- `Sports2D/` is the shipped runtime source of truth, core library, and execution pipeline.
- `Content/` is artifact and media-only.
- `.github/` owns automation and release workflows.
- `models/` is a local artifact stash, currently `models/sam3.1_multiplex.pt`, not a separate AGENTS domain.
- `src/` is an inactive scaffold, parent-covered unless it becomes tracked runtime source.

## Maintenance policy
- Keep each AGENTS file concise and scoped.
- Preserve manual sections across regeneration.
- Use stable headings so future tooling can update safely.
<!-- AUTO-GENERATED SECTION -->
Last reviewed: 2026-05-02

## Purpose
Repository root for Sports2D packaging metadata, user-facing documentation, CI entry points, and high-level agent guidance. `Sports2D/` is the shipped runtime source of truth; `sapiens2/` is a nested Sapiens2 git checkout used for local human-vision, pointmap, and OpenSim-marker experimentation. Other top-level paths are support material, local artifact stashes, inactive scaffold placeholders, or generated local-run artifacts.

## Key Files
| File | Description |
|------|-------------|
| `pyproject.toml` | Packaging metadata, dependencies, package-data rules, and console entry points (`sports2d`, `tests_sports2d`). |
| `README.md` | Primary installation, usage, and output-format documentation. |
| `README_KOREAN.md` | Korean-language project README. |
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
| `sapiens2/` | Nested Sapiens2 git checkout with its own hierarchy for dense/pose tooling, docs, demo media, and tests (see `sapiens2/AGENTS.md`). |
| `models/` | Local artifact stash, currently only `sam3.1_multiplex.pt`; not an AGENTS-covered domain. |
| `src/` | Inactive clean-architecture scaffold placeholder; no local `AGENTS.md`. |

## For AI Agents

### Working In This Directory
- Treat `Sports2D/` as the source of truth for shipped behavior and public entry points.
- Treat `sapiens2/` as a nested repository: read `sapiens2/AGENTS.md` and its descendants before modifying that subtree, and avoid assuming parent-repo git status covers nested changes.
- Keep top-level docs aligned with CLI/config changes that alter installation, dependencies, or output expectations, especially when result bundle contents change.
- Avoid committing generated logs, caches, or local result artifacts unless the task is explicitly about those outputs.

### Testing Requirements
- Use the baseline commands already recorded in the manual section and `CLAUDE.md`.
- Prefer targeted verification for docs-only or workflow-only edits, but keep lint/syntax gates in mind for code-bearing changes.

## Dependencies

### Internal
- `Sports2D/Utilities/` provides the backend abstraction and tests that most runtime changes eventually touch.
- `.github/workflows/` mirrors the repository's supported install, lint, and test paths.
- `sapiens2/` is independent nested source used for Sapiens2 dense/pose workflows and local OpenSim marker experiments.

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
- `sapiens2/AGENTS.md`
- `sapiens2/demo/AGENTS.md`
- `sapiens2/tests/AGENTS.md`
- `sapiens2/.github/**/AGENTS.md`, `sapiens2/docs/**/AGENTS.md`, and `sapiens2/sapiens/**/AGENTS.md` for nested Sapiens2 domains.

Parent-covered / excluded from scoring:
- `src/` remains an inactive scaffold with no local `AGENTS.md`.
- Root `models/` is a local artifact stash, currently `models/sam3.1_multiplex.pt`, not its own AGENTS domain.
- Generated or runtime-state paths such as `.omx/`, `.pytest_cache/`, `.sisyphus/`, `sports2d.egg-info/`, `**/__pycache__/`, top-level local-run folders, and nested result artifacts like `*_Sports2D/`, `pose_ball/`, logs, and caches stay parent-covered and do not get child `AGENTS.md` files.
- `Sports2D/models/RT-DETRv4/` stays a placeholder until intentionally populated.
- `.claude/`, `.codex`, `.omc/`, `.venv-codex/`, `.ruff_cache/`, `sapiens2/outputs/`, `sapiens2/sapiens.egg-info/`, and `sports2d_qt_hybrid_*/` are local settings, runtime state, virtualenv/cache, package metadata, or generated output areas; do not create AGENTS domains there.

Suggested upkeep:
- Refresh the covered list whenever AGENTS files are added or removed.
- Only promote placeholder or artifact areas into new AGENTS domains when they become real tracked source.
