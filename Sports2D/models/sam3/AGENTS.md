<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-03-10 | Updated: 2026-03-10 -->

# AGENTS.md

## Purpose
Checkpoint staging area for SAM3 model assets used by Sports2D's promptable detector integrations. This directory currently holds a local raw checkpoint and is referenced by hybrid ball-detection configurations and loader guidance.

## Key Files
| File | Description |
|------|-------------|
| `sam3.pt` | Local raw SAM3 checkpoint used with the official Meta runtime path when configured explicitly. |

## For AI Agents

### Working In This Directory
- Treat model files here as heavyweight binary artifacts; do not replace or rename them casually.
- Prefer updating docs, config examples, and runtime path guidance before changing checkpoint contents.
- If a Hugging Face-style converted bundle is added later, document the expected folder layout and runtime mode at the same time.

### Testing Requirements
- If paths or filenames change, verify `Sports2D/Utilities/sam3_detector.py` and `Sports2D/Utilities/synthpose_tracker.py` still resolve the documented runtime mode correctly.
- Keep tests focused on path resolution and import guidance; do not attempt real inference in repository CI by default.

## Dependencies

### Internal
- `Sports2D/Utilities/sam3_detector.py` decides whether this directory is used through the Meta runtime or Hugging Face runtime.
- `Sports2D/Demo/Config_demo.toml` and README examples may point to this directory when documenting local SAM3 usage.

### External
- Official Meta `sam3` runtime for raw `.pt` checkpoints.
- Hugging Face `transformers` runtime when a converted local SAM3 bundle is staged here in the future.
