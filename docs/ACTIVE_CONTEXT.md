# Active Context

This file is the default documentation entry for current development on
`DG_TWFD_v3`.

## Default Reading Set

Read only these docs before code work unless the user explicitly asks for
historical context:

1. `docs/experiments/DG_TWFD_v3/HANDOFF_2026-04-20.md`
2. `docs/experiments/DG_TWFD_v3/PIPELINE.md`
3. `docs/experiments/DG_TWFD_v3/EXPERIMENT_LOG.md`
4. `docs/experiments/DG_TWFD_v3/PAPER_EXPERIMENT_TARGETS.md`
5. `docs/experiments/DG_TWFD_v3/ARCHITECTURE_AND_IMPLEMENTATION.md`

Then inspect the relevant code/config files directly.

## Current Development Rule

- Formal training, sampling, and evaluation happen on the server.
- Local work is for code edits, documentation edits, and lightweight checks.
- Ignore untracked local reference folders by default, especially `refs/`.
- Do not read `docs/archive/` unless a specific historical question requires it.

## Active Code Anchors

- Trainer: `src/dgtd/train_dgtd.py`
- Teacher path: `src/dgtd/teacher.py`
- Residual: `src/dgtd/defect.py`
- Warp: `src/dgtd/warp.py`
- Metrics: `src/dgtd/metrics.py`
- Sampling: `src/dgtd/sample_dgtd.py`
- Eval dispatch: `src/dgfm/evaluators/common.py`
- Main config: `configs/experiment/dgtd_cifar10_v3.yaml`
- Smoke config: `configs/experiment/dgtd_cifar10_v3_smoke.yaml`
