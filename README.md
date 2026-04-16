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

- branch overview:
  - [docs/experiments/DG_TWFD_v3/README.md](/home/gzwlinux/vscode/gitProject/DG-TWFD/docs/experiments/DG_TWFD_v3/README.md)
- reconstruction prompt:
  - [docs/experiments/DG_TWFD_v3/reconstruction_v3.md](/home/gzwlinux/vscode/gitProject/DG-TWFD/docs/experiments/DG_TWFD_v3/reconstruction_v3.md)
- preserved baseline and teacher context:
  - [docs/experiments/map_branch/baseline/current_losses.md](/home/gzwlinux/vscode/gitProject/DG-TWFD/docs/experiments/map_branch/baseline/current_losses.md)
  - [docs/experiments/map_branch/teacher/teacher_entry.md](/home/gzwlinux/vscode/gitProject/DG-TWFD/docs/experiments/map_branch/teacher/teacher_entry.md)
  - [docs/experiments/map_branch/teacher/cache_schema.md](/home/gzwlinux/vscode/gitProject/DG-TWFD/docs/experiments/map_branch/teacher/cache_schema.md)

## Main entrypoints

- train:
  - [scripts/run_train.py](/home/gzwlinux/vscode/gitProject/DG-TWFD/scripts/run_train.py)
- eval:
  - [scripts/run_eval.py](/home/gzwlinux/vscode/gitProject/DG-TWFD/scripts/run_eval.py)
- generic sample:
  - [scripts/run_sample.py](/home/gzwlinux/vscode/gitProject/DG-TWFD/scripts/run_sample.py)
- DGTD sampling:
  - [scripts/run_sample_dgtd.py](/home/gzwlinux/vscode/gitProject/DG-TWFD/scripts/run_sample_dgtd.py)

## Main configs

- full:
  - [configs/experiment/dgtd_cifar10_v3.yaml](/home/gzwlinux/vscode/gitProject/DG-TWFD/configs/experiment/dgtd_cifar10_v3.yaml)
- smoke:
  - [configs/experiment/dgtd_cifar10_v3_smoke.yaml](/home/gzwlinux/vscode/gitProject/DG-TWFD/configs/experiment/dgtd_cifar10_v3_smoke.yaml)
