<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-02-13 | Updated: 2026-04-05 -->

# AGENTS.md

<!-- MANUAL SECTION -->
## Parent
- See `../AGENTS.md` for repository and package defaults.

## Scope
- Governs demo configs, command examples, and checked-in sample media under `Sports2D/Demo`.

## Scope-specific rules
- Keep demo config files as executable examples.
- If new CLI flags are added in code, reflect them in `Config_demo.toml` and demo guidance.
- Avoid hardcoding environment-specific paths.

## Maintenance
- Use small, reviewable TOML diffs for configuration changes.
- Pair config changes with `process.py`/`Sports2D.py` behavior changes where needed.

<!-- AUTO-GENERATED SECTION -->
Last reviewed: 2026-04-05

## Purpose
Demo and sample-data directory for Sports2D. Hand-maintained inputs here are the config and calibration examples plus a few raw media files. Generated `*_Sports2D/` bundles, nested `pose_ball/`, logs, caches, and similar local-run folders are parent-covered artifacts, excluded from AGENTS scoring, and should not get child AGENTS files.

## Key Files
| File | Description |
|------|-------------|
| `Config_demo.toml` | Canonical example configuration covering CLI defaults, calibration, output, hybrid ball detection, and optional backend settings. |
| `Calib_demo.toml` | Example calibration data used by demo and perspective-conversion flows. |
| `최서현_re.mp4` | Checked-in local demo clip used for recent manual smoke runs. |

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `*_Sports2D/` | Generated output bundles containing rendered video, `.trc`, `.mot`, optional `.c3d`, nested `pose_ball/`, calibration files, and other export artifacts. Parent-covered, excluded from scoring, no child `AGENTS.md`. |
| `*_오류/` and similar local-run folders | Scratch or troubleshooting directories. Keep parent-covered and unscored unless a task explicitly targets them. |

## For AI Agents

### Working In This Directory
- Edit the hand-maintained TOML configs and raw inputs intentionally; generated `*_Sports2D/` bundles, nested `pose_ball/`, logs, caches, and similar local-run folders stay parent-covered artifacts and should not sprout child `AGENTS.md` files.
- Keep demo examples path-agnostic so the default `sports2d` smoke run remains portable across machines.

### Testing Requirements
- If `Config_demo.toml` changes, verify the CLI still accepts the documented options and that the config remains executable.
- Prefer smoke tests over full artifact regeneration unless the task specifically requires updated sample outputs.

## Dependencies

### Internal
- `Sports2D/Sports2D.py` and `Sports2D/process.py` consume these configs and define the output bundle structure.

### External
- Sample media is processed through the runtime stack (`opencv`, `Pose2Sim`, optional SynthPose dependencies) and may generate OpenSim-compatible `.trc`/`.mot`/`.c3d` outputs.
