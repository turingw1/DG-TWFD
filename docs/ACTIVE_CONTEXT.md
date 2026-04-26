# Active Context

This file is the default documentation entry for current development on
`DG_TWFD_v3`.

## Default Reading Set

Read only these docs before current EDM-first code work unless the user
explicitly asks for historical context:

1. `docs/experiments/DG_TWFD_v3/README.md`
2. `docs/experiments/DG_TWFD_v3/DOCS_REGISTRY.md`
3. `docs/experiments/DG_TWFD_v3/EDM_FIRST_SUPERVISION.md`
4. `docs/experiments/DG_TWFD_v3/NETWORK_AND_RECOVERY.md`
5. `docs/experiments/DG_TWFD_v3/BASELINE_STATUS.md`
6. `docs/experiments/DG_TWFD_v3/BASELINE_COMPARISON_GUIDE.md`

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
- Use `docs/experiments/DG_TWFD_v3/DOCS_REGISTRY.md` to classify any new doc.
  Dated filenames are evidence snapshots, not the primary versioning method.

## Active Code Anchors

- EDM-first isolated track: `experiments/edm_first/`
- EDM-first trainer: `experiments/edm_first/train_edm_map.py`
- EDM-first evaluator: `experiments/edm_first/eval_edm_map.py`
- EDM-first watcher: `experiments/edm_first/scripts/watch_eval_checkpoints.sh`
- EDM-first threshold checker:
  `experiments/edm_first/scripts/check_fid_thresholds.py`
- Current e504a config:
  `experiments/edm_first/configs/cifar10_edm_map_onestep_prior_msdefect_8h.yaml`
- Eval dispatch: `src/dgfm/evaluators/common.py`
- Paused DGTD trainer: `src/dgtd/train_dgtd.py`
- Paused DGTD teacher path: `src/dgtd/teacher.py`
- Paused DGTD residual/warp: `src/dgtd/defect.py`, `src/dgtd/warp.py`
- Paused DGTD configs:
  `configs/experiment/dgtd_cifar10_v3.yaml`,
  `configs/experiment/dgtd_cifar10_v3_smoke.yaml`,
  `configs/experiment/dgtd_cifar10_v3_diag.yaml`

## Latest Empirical State

As of 2026-04-26, the DDPM/DGTD result-first loop is functional but is paused as
the primary algorithm route. The active route is now EDM-first continuous
sigma-space distillation with time warp retained.

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

- Do not launch DDPM `e402a` full training or DDPM `oss001` unless a specific
  DDPM revisit is requested.
- The previous short-budget scheduler stayed in warmup because trainer
  `total_steps` ignored `train.max_train_batches`; this is fixed in
  `src/dgtd/train_dgtd.py`.
- Online teacher endpoint order is correct and online continuation is dominant.
- Learned warp is stable and slightly better than no-warp, but it is not the
  current bottleneck.
- Current failure is objective/data-coverage level: training losses and defects
  improve while pure-noise rollout samples remain noise-like.

EDM-first continuous-teacher evidence:

- isolated code path: `experiments/edm_first/`
- `e500c` smoke: 20 training steps, 128-sample eval, clear non-noise CIFAR-like
  grids.
- `e501a` learned-warp train:
  - run: `runs/edm_first_cifar10_warp_e501a`
  - eval: `eval/edm_first_cifar10_warp_e501a`
  - budget: 2000 steps, batch size 64, 128k samples seen
  - approx FID@1024 for `1/2/4/8` steps:
    `339.01 / 106.23 / 53.83 / 39.46`
- `e501a` identity-clock eval on the same checkpoint:
  - eval: `eval/edm_first_cifar10_identity_e501a`
  - approx FID@1024 for `1/2/4/8` steps:
    `339.01 / 124.48 / 61.10 / 38.84`
- `e501ref` official EDM checkpoint reference:
  - eval: `eval/edm_cifar10_public_eval_e501ref`
  - official-protocol FID@1024 for sampler steps `1/2/4/8`:
    `679.611 / 473.607 / 115.246 / 33.0675`
- `e502a` extended continuation:
  - run: `runs/edm_first_cifar10_warp_e502a`
  - eval: `eval/edm_first_cifar10_warp_e502a`
  - resumed from `e501a` best, batch size 128, stopped after the step-5000
    checkpoint for evaluation
  - approx FID@2048 learned warp for `1/2/4/8` steps:
    `337.45 / 151.78 / 56.76 / 35.67`
  - approx FID@2048 identity clock for `1/2/4/8` steps:
    `337.45 / 140.76 / 54.24 / 26.89`

Current algorithm judgement:

- DDPM/discrete teacher is not theoretically impossible, but it is a poor
  current distillation target because interpolation error, teacher cost, and
  weak loss-to-sample correlation dominate the experiment.
- EDM-first is now the main path because continuous sigma transitions produce
  usable samples immediately and let time warp operate on a meaningful
  continuous domain.
- Time warp remains mandatory, but the current passive defect-density warp is
  not yet the right mechanism: it helped `e501a` at 2/4 steps, but `e502a`
  identity-clock eval beat learned warp at 2/4/8 steps. Upgrade it into a
  step-budget-aware schedule component.
