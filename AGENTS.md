<!-- Generated: 2026-02-13 | Updated: 2026-03-02 -->

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
Last reviewed: 2026-03-02

Covered AGENTS files:
- `AGENTS.md`
- `.github/AGENTS.md`
- `.github/workflows/AGENTS.md`
- `Content/AGENTS.md`
- `Sports2D/AGENTS.md`
- `Sports2D/Demo/AGENTS.md`
- `Sports2D/Utilities/AGENTS.md`
- `Sports2D/models/AGENTS.md`
- `Sports2D/models/rtdetrv4/AGENTS.md`

Suggested upkeep:
- Update this list if new AGENTS files are added or moved.
- Refresh parent links whenever folders are reorganized.
