# DG_TWFD_v3

This directory is the branch-facing documentation entry for the `DG_TWFD_v3`
reconstruction.

## Scope

- algorithm:
  - unified Defect-Guided Trajectory Distillation on explicit map prediction
- active objective:
  - `train.objective=dgtd_map`
- active time warp:
  - `scheduler.timewarp.type=dgtd_density`
- active teacher source:
  - offline trajectory cache under `target.shard_root`
- preserved infrastructure:
  - dataset loading
  - distributed/server workflow
  - evaluation and few-step sampling

## Reading order

1. [reconstruction_v3.md](/home/gzwlinux/vscode/gitProject/DG-TWFD/docs/experiments/DG_TWFD_v3/reconstruction_v3.md)
2. [../map_branch/HANDOFF_2026-04-16.md](/home/gzwlinux/vscode/gitProject/DG-TWFD/docs/experiments/map_branch/HANDOFF_2026-04-16.md)
3. [../map_branch/baseline/current_losses.md](/home/gzwlinux/vscode/gitProject/DG-TWFD/docs/experiments/map_branch/baseline/current_losses.md)
4. [../map_branch/teacher/teacher_entry.md](/home/gzwlinux/vscode/gitProject/DG-TWFD/docs/experiments/map_branch/teacher/teacher_entry.md)
5. [../map_branch/teacher/cache_schema.md](/home/gzwlinux/vscode/gitProject/DG-TWFD/docs/experiments/map_branch/teacher/cache_schema.md)
6. [../map_branch/A100_PIPELINE.md](/home/gzwlinux/vscode/gitProject/DG-TWFD/docs/experiments/map_branch/A100_PIPELINE.md)

## Active code entrypoints

- trainer:
  - [src/dgtd/train_dgtd.py](/home/gzwlinux/vscode/gitProject/DG-TWFD/src/dgtd/train_dgtd.py)
- warp:
  - [src/dgtd/warp.py](/home/gzwlinux/vscode/gitProject/DG-TWFD/src/dgtd/warp.py)
- defect:
  - [src/dgtd/defect.py](/home/gzwlinux/vscode/gitProject/DG-TWFD/src/dgtd/defect.py)
- cache:
  - [src/dgtd/cache.py](/home/gzwlinux/vscode/gitProject/DG-TWFD/src/dgtd/cache.py)
- sampling:
  - [src/dgtd/sample_dgtd.py](/home/gzwlinux/vscode/gitProject/DG-TWFD/src/dgtd/sample_dgtd.py)
- eval dispatch:
  - [src/dgfm/evaluators/common.py](/home/gzwlinux/vscode/gitProject/DG-TWFD/src/dgfm/evaluators/common.py)

## Active commands

- train:
  - `torchrun --nproc_per_node=2 scripts/run_train.py --config configs/experiment/dgtd_cifar10_v3.yaml --run-root runs/dgtd_cifar10_v3`
- sample:
  - `python scripts/run_sample_dgtd.py --config configs/experiment/dgtd_cifar10_v3.yaml --checkpoint runs/dgtd_cifar10_v3/checkpoints/best.pt --output-dir runs/dgtd_cifar10_v3/samples/steps16 --steps 16`
- eval:
  - `python scripts/run_eval.py --config configs/experiment/dgtd_cifar10_v3.yaml --checkpoint runs/dgtd_cifar10_v3/checkpoints/best.pt --eval-root eval/dgtd_cifar10_v3`
