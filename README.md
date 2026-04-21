# DG_TWFD_v3

This branch rebuilds the project around a unified `dgtd_map` objective on top
of the current `dgfm` infrastructure.

## Active scope

- keep the reusable `dgfm` stack for datasets, distributed training, sampling,
  evaluation, and server workflow
- replace the old four-loss explicit-map baseline with one
  warp-weighted DGTD residual
- keep the current project time convention:
  - `0 <= t < s < u <= 1`
  - `0.0` is noisy and `1.0` is clean
- use offline teacher trajectory cache as the primary supervision path

## Start here

- active context:
  - [docs/ACTIVE_CONTEXT.md](docs/ACTIVE_CONTEXT.md)
- branch overview:
  - [docs/experiments/DG_TWFD_v3/README.md](docs/experiments/DG_TWFD_v3/README.md)
- current handoff:
  - [docs/experiments/DG_TWFD_v3/HANDOFF_2026-04-20.md](docs/experiments/DG_TWFD_v3/HANDOFF_2026-04-20.md)

Historical round notes, old `map_branch` docs, and superseded reconstruction
prompts are archived under `docs/archive/context_noise_2026-04-21/`. Do not
read that archive during normal development unless a specific historical lookup
is required.

## Main entrypoints

- train:
  - [scripts/run_train.py](scripts/run_train.py)
- eval:
  - [scripts/run_eval.py](scripts/run_eval.py)
- generic sample:
  - [scripts/run_sample.py](scripts/run_sample.py)
- DGTD sampling:
  - [scripts/run_sample_dgtd.py](scripts/run_sample_dgtd.py)

## Main configs

- full:
  - [configs/experiment/dgtd_cifar10_v3.yaml](configs/experiment/dgtd_cifar10_v3.yaml)
- smoke:
  - [configs/experiment/dgtd_cifar10_v3_smoke.yaml](configs/experiment/dgtd_cifar10_v3_smoke.yaml)
