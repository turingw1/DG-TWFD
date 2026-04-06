# map_branch documentation

This directory is the complete documentation context for the current
`map_branch_ctm_explicit_map` git branch.

## Version

- active experiment version:
  - `v3`
- quick diagnostic version:
  - `map_branch_quick v1`
- experiment family:
  - `fm_cifar10_map_branch`

## Reading order

1. [A100_PIPELINE.md](/home/gzwlinux/vscode/gitProject/DG-TWFD/docs/experiments/map_branch/A100_PIPELINE.md)
   - operational entrypoint
   - environment setup
   - CIFAR-10 and online teacher target setup
   - train / resume / eval / sample / panel

2. [ENVIRONMENT.md](/home/gzwlinux/vscode/gitProject/DG-TWFD/docs/experiments/map_branch/ENVIRONMENT.md)
   - conda environment creation
   - package versions
   - validation commands

3. [MASTER_PLAN.md](/home/gzwlinux/vscode/gitProject/DG-TWFD/docs/experiments/map_branch/MASTER_PLAN.md)
   - branch architecture
   - migration boundary from CTM to `dgfm`
   - module responsibilities

4. [ACCEPTANCE_CHECKLIST.md](/home/gzwlinux/vscode/gitProject/DG-TWFD/docs/experiments/map_branch/ACCEPTANCE_CHECKLIST.md)
   - architecture, functional, numerical, research-readiness checks

5. [TECHNICAL_REPORT.md](/home/gzwlinux/vscode/gitProject/DG-TWFD/docs/experiments/map_branch/TECHNICAL_REPORT.md)
   - condensed technical rationale for the branch design

## Current branch scope

- keep `dgfm` shared infrastructure
- add explicit map branch:
  - `M_theta(x_t, t, s) -> x_s`
- preserve current `dgfm` time semantics:
  - `0 <= t < s <= 1`
- current target builder:
  - `teacher_sampler`
- teacher backend:
  - `diffusers_ddpm`
- retained teacher trajectory:
  - `33` anchor states
- teacher internal rollout:
  - `128` DDIM steps
- current loss stack:
  - direct map loss
  - perceptual loss
  - endpoint few-step teacher loss
- current training/sampling tip:
  - CTM-style preconditioning enabled
- quick verification path:
  - `configs/experiment/fm_cifar10_map_branch_quick.yaml`
- training-time archive mirror:
  - `/temp/Zhengwei/dgfm_runs/<FM_EXP>`

## Active code entrypoints

- experiment config:
  - [configs/experiment/fm_cifar10_map_branch.yaml](/home/gzwlinux/vscode/gitProject/DG-TWFD/configs/experiment/fm_cifar10_map_branch.yaml)
  - [configs/experiment/fm_cifar10_map_branch_quick.yaml](/home/gzwlinux/vscode/gitProject/DG-TWFD/configs/experiment/fm_cifar10_map_branch_quick.yaml)
- trainer:
  - [src/dgfm/trainers/map.py](/home/gzwlinux/vscode/gitProject/DG-TWFD/src/dgfm/trainers/map.py)
- model:
  - [src/dgfm/models/map.py](/home/gzwlinux/vscode/gitProject/DG-TWFD/src/dgfm/models/map.py)
- target builder:
  - [src/dgfm/targets/builder.py](/home/gzwlinux/vscode/gitProject/DG-TWFD/src/dgfm/targets/builder.py)
- teacher rollout:
  - [src/dgfm/teachers/diffusers_ddpm.py](/home/gzwlinux/vscode/gitProject/DG-TWFD/src/dgfm/teachers/diffusers_ddpm.py)
- optional offline trajectory dataset:
  - [src/dgfm/datasets/trajectory.py](/home/gzwlinux/vscode/gitProject/DG-TWFD/src/dgfm/datasets/trajectory.py)
- optional offline trajectory preparation:
  - [scripts/prepare_teacher_trajectories.py](/home/gzwlinux/vscode/gitProject/DG-TWFD/scripts/prepare_teacher_trajectories.py)
- map rollout:
  - [src/dgfm/samplers/map_sampler.py](/home/gzwlinux/vscode/gitProject/DG-TWFD/src/dgfm/samplers/map_sampler.py)
