<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-02-13 | Updated: 2026-03-10 -->

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
Last reviewed: 2026-03-10

## Purpose
Demo and sample-data directory for Sports2D. It mixes hand-maintained configuration files and raw sample videos with many generated result bundles that demonstrate expected outputs. Recent generated bundles now include `pose_ball/` JSON artifacts when `detect_ball=true` and `save_pose=true`.

## Key Files
| File | Description |
|------|-------------|
| `Config_demo.toml` | Canonical example configuration covering base, pose, calibration, output, hybrid ball detection, and optional backend settings. |
| `Calib_demo.toml` | Example calibration data used by demo and perspective-conversion flows. |
| `nfl.mp4` | Short sample input video used for football-style demo runs. |
| `logs.txt` | Local run log output from demo executions; useful for debugging but not a source of truth. |

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `20260220/` | Raw sample capture set containing `.MOV` and `.mp4` inputs from 2026-02-20. |
| `메디신볼/` | Raw medicine-ball capture set used for local testing. |
| `*_Sports2D/` | Generated output bundles containing rendered video, `.trc`, `.mot`, `.c3d`, `pose_ball/` JSON files, calibration files, and optional graph assets. |

## For AI Agents

### Working In This Directory
- Edit the hand-maintained TOML configs and raw inputs intentionally; avoid bulk changes inside generated `*_Sports2D/` result folders, including `pose_ball/`, unless the task is explicitly about artifacts.
- Keep demo examples path-agnostic so the default `sports2d` smoke run remains portable across machines.

### Testing Requirements
- If `Config_demo.toml` changes, verify the CLI still accepts the documented options and that the config remains executable.
- Prefer smoke tests over full artifact regeneration unless the task specifically requires updated sample outputs.

## Dependencies

### Internal
- `Sports2D/Sports2D.py` and `Sports2D/process.py` consume these configs and define the output bundle structure.

### External
- Sample media is processed through the runtime stack (`opencv`, `Pose2Sim`, optional SynthPose dependencies) and may generate OpenSim-compatible `.trc`/`.mot`/`.c3d` outputs.
