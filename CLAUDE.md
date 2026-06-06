# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository. Ignore files in .gitingore, except CLAUDE.md.

## Project Overview

**SANDesc** (Streamlined Attention-based Network for Descriptor Extraction) is the official code for the 3DV 2026 paper. It is a **descriptor-only** module: it reuses the keypoints from an existing detector (ALIKED, SuperPoint, DISK, etc.) and replaces only their *descriptors*, then pose is recomputed. It does not detect keypoints.

The network ([model/network_descriptor.py](model/network_descriptor.py)) is a UNet-style encoder-decoder that outputs a dense descriptor volume `[B, des_dim, H, W]` (L2-normalized along the channel dim); descriptors are bilinearly/nearest sampled at keypoint locations.

## Architecture

- **Network**: `SANDesc` in [model/network_descriptor.py](model/network_descriptor.py) — 4 down blocks + 4 up blocks. The last `up_output_channels` element is the descriptor dim.
- **Building blocks**: [model/modules.py](model/modules.py) — pre-activation UNet blocks (norm→activ→conv, à la DISK; see arxiv 1603.05027). Toggleable via config: `skip_connection` (adds 2nd block + residual align), `third_block` (adds 3rd block), `spatial_attention` (CBAM channel+spatial attention, adapted from Jongchan/attention-module).
- **Loss**: [losses/triplet_loss.py](losses/triplet_loss.py) — `TripletLoss` with hardest-negative mining (`get_hardest_triplets`). `random_negative_ratio` starts at 1.0 and decays each step, so early training uses random negatives and gradually shifts to hardest. Handles multiscale (multiple GT matches per keypoint) and NaN/invalid descriptors.
- **Matcher**: mutual-nearest-neighbor matching (`MNN`/`Matcher`) lives in [utils/utils_matches.py](utils/utils_matches.py), used to build GT match matrices during training. (An older standalone copy is archived under [_bin/matcher/mnn.py](_bin/matcher/mnn.py) and is no longer imported.)
- **LR schedule**: [lr_scheduler/lr_scheduler.py](lr_scheduler/lr_scheduler.py) — `LrManager`, default `warmup_decay_constant`.

## Training

- Entry point: [train_sandesc.py](train_sandesc.py). Run `python train_sandesc.py`; resume with `python train_sandesc.py resume_from=path/to/checkpoint.pth`.
- **Config**: Hydra, in [configs/](configs/) (`config.yaml` composes `training/`, `model/`, `triplet_loss/` defaults). Override on the CLI Hydra-style. When resuming, the checkpoint's config replaces the current one (only `device`/`resume_from` are preserved).
- Mixed precision (bfloat16 AMP) + GradScaler; logging via Weights & Biases (`use_wandb`).
- Setup is centralized in [utils/helpers.py](utils/helpers.py) (`setup_dataloaders`, `setup_model_and_optimizer`, `setup_loss_and_scaler`, `setup_wrappers`, etc.).
- **Datasets**: [datasets/](datasets/) — MegaDepth/DISK ([dataset_megadepth_disk.py](datasets/dataset_megadepth_disk.py)), IMB ([dataset_imb.py](datasets/dataset_imb.py), used for validation), TerraSky3D ([dataset_terrasky.py](datasets/dataset_terrasky.py)).

## External Dependencies (important)

- **PoseBench** (github.com/mattiadurso/PoseBench) provides `wrappers_manager` — the keypoint-detector "wrappers" (`aliked`, `superpoint`, `random`, ...). [utils/helpers.py](utils/helpers.py) `setup_paths()` adds PoseBench to `sys.path`; set the **`POSEBENCH_PATH`** env var to its location (falls back to `/home/mattia/Desktop/Repos/posebench`). Training and testing require this repo cloned locally.
- **mylib** (github.com/mattiadurso/mylib) — installed via `requirements.txt`.
- **Testing/evaluation** is done through PoseBench, not this repo. There are no unit tests here.

### Dataset paths (env vars)

Each dataset resolves its root lazily via `datasets/dataset_paths.py:resolve_dataset_path` — checking an env var first, then built-in fallback paths, and raising `FileNotFoundError` if none exist. Set these to point at your data:

- **`SANDESC_IMB_PATH`** — IMB validation set
- **`SANDESC_MEGADEPTH_PATH`** — MegaDepth/DISK
- **`SANDESC_TERRASKY_PATH`** — TerraSky3D

The hardcoded `/home/mattia/...` paths remain only as last-resort fallbacks.

## Demo

[demo.ipynb](demo.ipynb) visualizes results using precomputed ALIKED keypoints/descriptors saved under [assets/](assets/), so it runs without re-running the detector. Helpers in [demo_utils.py](demo_utils.py).

## Code Style
- Follow Google Python Style Guide (https://google.github.io/styleguide/pyguide.html)
- Use Black for formatting (line length 88)
- Run pylint before submitting
- Docstrings: always triple double-quote format per PEP 257
- Functions should stay under 40 lines
- Prefer explicit imports over wildcard imports



## Working Style (Karpathy Skills)

Behavioral guidelines to reduce common LLM coding mistakes. **Tradeoff:** these bias toward caution over speed. For trivial tasks, use judgment.

### 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them — don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

### 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it — don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: every changed line should trace directly to the user's request.

### 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
