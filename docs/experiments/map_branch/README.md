# map_branch documentation
<!-- codex resume 019d666f-eadb-76e3-9f90-b12ad101f1c5 -->
This directory is the complete documentation context for the current
`map_branch_ctm_explicit_map` git branch.

## Experiment tracking

- use one stable pipeline document:
  - [A100_PIPELINE.md](/home/gzwlinux/vscode/gitProject/DG-TWFD/docs/experiments/map_branch/A100_PIPELINE.md)
- record every run here:
  - [EXPERIMENT_LOG.md](/home/gzwlinux/vscode/gitProject/DG-TWFD/docs/experiments/map_branch/EXPERIMENT_LOG.md)
- experiment family:
  - `fm_cifar10_map_branch`

## Reading order

1. [A100_PIPELINE.md](/home/gzwlinux/vscode/gitProject/DG-TWFD/docs/experiments/map_branch/A100_PIPELINE.md)
   - operational entrypoint
   - environment setup
   - CIFAR-10 and online teacher target setup
   - train / resume / eval / sample / panel

2. [EXPERIMENT_LOG.md](/home/gzwlinux/vscode/gitProject/DG-TWFD/docs/experiments/map_branch/EXPERIMENT_LOG.md)
   - experiment ids
   - source variant names
   - run naming and short annotations

3. [ENVIRONMENT.md](/home/gzwlinux/vscode/gitProject/DG-TWFD/docs/experiments/map_branch/ENVIRONMENT.md)
   - conda environment creation
   - package versions
   - validation commands

4. [MASTER_PLAN.md](/home/gzwlinux/vscode/gitProject/DG-TWFD/docs/experiments/map_branch/MASTER_PLAN.md)
   - branch architecture
   - migration boundary from CTM to `dgfm`
   - module responsibilities

5. [TIMEWARP_CTM_FINALIZATION_PLAN.md](/home/gzwlinux/vscode/gitProject/DG-TWFD/docs/experiments/map_branch/TIMEWARP_CTM_FINALIZATION_PLAN.md)
   - finalization roadmap for time-warp and CTM-style targets
   - task ordering
   - interface boundaries
   - acceptance criteria for the next major code update

6. [ACCEPTANCE_CHECKLIST.md](/home/gzwlinux/vscode/gitProject/DG-TWFD/docs/experiments/map_branch/ACCEPTANCE_CHECKLIST.md)
   - architecture, functional, numerical, research-readiness checks

7. [TECHNICAL_REPORT.md](/home/gzwlinux/vscode/gitProject/DG-TWFD/docs/experiments/map_branch/TECHNICAL_REPORT.md)
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
- current target sampler:
  - `sampling_mode=ctm_discrete`
  - `start_scales=33`
  - `num_heun_step=17`
  - `sample_s_strategy=uniform`
- current target construction:
  - `target_construction=ctm_consistency`
  - `target_source=ema_model`
  - `target_stop_grad=true`
  - `bridge_source=ema_model_rollout`
- teacher internal rollout:
  - `128` DDIM steps
- current loss stack:
  - CTM-style estimate/target consistency loss
  - perceptual loss
  - endpoint few-step teacher loss
  - optional defect-driven time-warp auxiliary update
- current training/sampling tip:
  - CTM-style preconditioning enabled
- current time-warp status:
  - shared train/sample/eval time-warp pipeline is implemented
  - learnable monotone warp is supported
  - checkpoint restore for learned warp is supported
  - default config remains disabled until an experiment explicitly enables it
- current CTM target status:
  - trainer now uses explicit `estimate / target / stop-grad` semantics
  - `teacher_sampler` now emits `t_dt` and teacher bridge states
  - default bridge state generation now uses EMA rollout when available
  - bridge rollout is still map-branch-native, not yet a CTM-faithful solver
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
