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
6. `docs/experiments/DG_TWFD_v3/NETWORK_AND_RECOVERY.md`

Then inspect the relevant code/config files directly.

## Current Development Rule

- Formal training, sampling, and evaluation happen on the server.
- Local work is for code edits, documentation edits, and lightweight checks.
- Use the repo network profiles: local proxy for normal outbound work, and the
  heavy-download profile for datasets, models, and package installs.
- After meaningful workspace edits, push to git and refresh the small recovery
  snapshot under `/temp/Zhengwei/DG-TWFD-recovery`.
- Ignore untracked local reference folders by default, especially `refs/`.
- Do not read `docs/archive/` unless a specific historical or baseline
  reproduction question requires it.
- Archived low-signal docs under `docs/archive/low_signal_2026-04-25/` are not
  part of the active development context.

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
- Diagnostic config: `configs/experiment/dgtd_cifar10_v3_diag.yaml`
- Teacher endpoint diagnostic: `scripts/diagnose_teacher_endpoints.py`
- Run analysis gate: `scripts/analyze_dgtd_run.py`

## Latest Empirical State

As of 2026-04-26, the DGTD v3 result-first loop is functional but the mainline
is not yet producing usable CIFAR-10 samples.

- `e405b` fast-teacher warped probe:
  - config: `configs/experiment/dgtd_cifar10_v3_probe_fast_teacher.yaml`
  - run: `runs/dgtd_cifar10_v3_probe_fast_teacher_e405b`
  - eval: `eval/dgtd_cifar10_v3_probe_fast_teacher_e405b`
  - best approximate FID@512: `373.26` at 2 steps
  - gate failure: `sample_not_noise_like`
- `e406a` fast-teacher no-warp control:
  - config: `configs/experiment/dgtd_cifar10_v3_probe_fast_teacher_no_warp.yaml`
  - run: `runs/dgtd_cifar10_v3_probe_fast_teacher_no_warp_e406a`
  - eval: `eval/dgtd_cifar10_v3_probe_fast_teacher_no_warp_e406a`
  - best approximate FID@512: `386.56` at 2 steps
  - gate failure: `sample_not_noise_like`
- `e407a` stronger endpoint-anchor long probe:
  - config: `configs/experiment/dgtd_cifar10_v3_probe_anchor1_long.yaml`
  - run: `runs/dgtd_cifar10_v3_probe_anchor1_long_e407a`
  - eval: `eval/dgtd_cifar10_v3_probe_anchor1_long_e407a`
  - best approximate FID@1024: `427.45` at 8 steps
  - gate failure: `sample_not_noise_like`

Operational conclusions:

- Do not launch `e402a` full training yet.
- The previous short-budget scheduler stayed in warmup because trainer
  `total_steps` ignored `train.max_train_batches`; this is fixed in
  `src/dgtd/train_dgtd.py`.
- Online teacher endpoint order is correct and online continuation is dominant.
- Learned warp is stable and slightly better than no-warp, but it is not the
  current bottleneck.
- Current failure is objective/data-coverage level: training losses and defects
  improve while pure-noise rollout samples remain noise-like.
