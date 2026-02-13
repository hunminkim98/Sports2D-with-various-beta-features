<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-02-13 | Updated: 2026-02-13 -->

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
Last reviewed: 2026-02-13

Tracked model areas:
- `RT-DETRv4/`
- `rtdetrv4/`
