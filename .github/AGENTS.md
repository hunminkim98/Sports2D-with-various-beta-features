<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-02-13 | Updated: 2026-02-13 -->

# AGENTS.md

<!-- MANUAL SECTION -->
## Parent
- See `../AGENTS.md` for repository-wide defaults.

## Scope
- Governs GitHub automation under `.github/`.
- Treat workflow files as CI/CD and release infrastructure.

## Working rules
- Keep workflow edits minimal and reversible.
- Preserve existing trigger intent unless explicitly changing release/testing policy.
- Prefer updating a single workflow per task to reduce blast radius.

## Validation
- Confirm YAML syntax is valid before finalizing.
- Keep workflow names and file names stable unless migration is intentional.

<!-- AUTO-GENERATED SECTION -->
Last reviewed: 2026-02-13

Subtrees tracked here:
- `workflows/`
