<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-02-13 | Updated: 2026-03-10 -->

# AGENTS.md

<!-- MANUAL SECTION -->
## Parent
- See `../AGENTS.md` for repository defaults.

## Scope
- Governs model assets and model loader integrations under `Sports2D/models`.
- Treat this area as binary-heavy and operationally constrained.

## Constraints
- Do not rewrite or remove packaged model artifacts without explicit justification and checksum/reproducibility notes.
- Prefer pointer-based updates (`README.md`, config, and install docs) for model changes.
- Keep heavy model directories isolated from core code edits unless required by API integration.
- If adding a new model file, update related dependency notes and any installation requirements.

## Validation
- Keep changes here limited to metadata and reference docs unless runtime is being deliberately changed.
- Validate import/loading paths when touch points affect runtime model loading.
<!-- AUTO-GENERATED SECTION -->
Last reviewed: 2026-03-10

## Purpose
Model-asset staging area for detector integrations that are too large or too vendor-specific to live in the main Python source tree. Most changes here should be metadata, setup notes, or clearly intentional binary drops. The tree now includes SAM3 checkpoint staging alongside RT-DETRv4 notes.

## Key Files
| File | Description |
|------|-------------|
| `AGENTS.md` | Directory-level handling rules for model assets and placeholder folders. |

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `RT-DETRv4/` | Empty placeholder for an upstream/vendor RT-DETRv4 engine drop-in; leave untouched unless intentionally populating it. |
| `rtdetrv4/` | README-driven checkpoint/config guidance for local RT-DETRv4 support (see `rtdetrv4/AGENTS.md`). |
| `sam3/` | Local SAM3 checkpoint staging used by hybrid ball-detection flows (see `sam3/AGENTS.md`). |

## For AI Agents

### Working In This Directory
- Document provenance, expected filenames, and setup steps whenever model-related paths change.
- Prefer README/config updates over binary churn; only add or replace weights when the task explicitly requires it.

### Testing Requirements
- After path or naming changes, confirm the runtime lookup logic in `Sports2D/Utilities/synthpose_tracker.py` still points at the documented locations.
- If a previously empty placeholder becomes active, add a child `AGENTS.md` for that subtree at the same time.

## Dependencies

### Internal
- `Sports2D/Utilities/synthpose_tracker.py` and demo configuration files reference this directory's expected layout.

### External
- Upstream RT-DETRv4 code/checkpoints, Hugging Face SAM3 bundles, official Meta SAM3 checkpoints, and any local vendor assets added outside the Python packaging flow.
