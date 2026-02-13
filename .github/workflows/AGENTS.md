<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-02-13 | Updated: 2026-02-13 -->

# AGENTS.md

<!-- MANUAL SECTION -->
## Parent
- See `../AGENTS.md` for `.github`-level constraints and `../../AGENTS.md` for repository defaults.

## Scope
- Governs workflow definitions in `.github/workflows`.

## Workflow guidance
- Keep CI checks aligned with project commands documented in root guidance.
- Do not broaden permissions unless required for a specific automation change.
- Keep backup or disabled workflow files clearly marked (for example `.bak`).

## Validation
- Verify edited workflow triggers (`on`) and job names remain intentional.
- Ensure changes do not break existing publish or CI flow semantics.

<!-- AUTO-GENERATED SECTION -->
Last reviewed: 2026-02-13

Tracked workflows:
- `continuous-integration.yml`
- `joss_pdf.yml`
- `publish-on-release.yml`
- `sync_to_hf.yml.bak`
