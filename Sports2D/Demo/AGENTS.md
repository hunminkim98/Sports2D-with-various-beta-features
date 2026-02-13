<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-02-13 | Updated: 2026-02-13 -->

# AGENTS.md

<!-- MANUAL SECTION -->
## Parent
- See `../AGENTS.md` for repository and package defaults.

## Scope
- Governs demo configs, command examples, and sample outputs under `Sports2D/Demo`.

## Scope-specific rules
- Keep demo config files as executable examples.
- If new CLI flags are added in code, reflect them in `Config_demo.toml` and demo guidance.
- Avoid hardcoding environment-specific paths.

## Maintenance
- Use small, reviewable TOML diffs for configuration changes.
- Pair config changes with `process.py`/`Sports2D.py` behavior changes where needed.

<!-- AUTO-GENERATED SECTION -->
Last reviewed: 2026-02-13

Current tracked demo assets:
- `Config_demo.toml`
