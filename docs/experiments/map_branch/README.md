# map_branch documentation

This directory is the complete documentation context for the current
`map_branch_ctm_explicit_map` git branch.

## Version

- active experiment version:
  - `v1`
- experiment family:
  - `fm_cifar10_map_branch`

## Reading order

1. [A100_PIPELINE.md](/home/gzwlinux/vscode/gitProject/DG-TWFD/docs/experiments/map_branch/A100_PIPELINE.md)
   - operational entrypoint
   - environment setup
   - dataset check
   - train / resume / eval / sample / panel

2. [MASTER_PLAN.md](/home/gzwlinux/vscode/gitProject/DG-TWFD/docs/experiments/map_branch/MASTER_PLAN.md)
   - branch architecture
   - migration boundary from CTM to `dgfm`
   - module responsibilities

3. [ACCEPTANCE_CHECKLIST.md](/home/gzwlinux/vscode/gitProject/DG-TWFD/docs/experiments/map_branch/ACCEPTANCE_CHECKLIST.md)
   - architecture, functional, numerical, research-readiness checks

4. [TECHNICAL_REPORT.md](/home/gzwlinux/vscode/gitProject/DG-TWFD/docs/experiments/map_branch/TECHNICAL_REPORT.md)
   - condensed technical rationale for the branch design

## Current branch scope

- keep `dgfm` shared infrastructure
- add explicit map branch:
  - `M_theta(x_t, t, s) -> x_s`
- preserve current `dgfm` time semantics:
  - `0 <= t < s <= 1`
- first-stage target builder:
  - `analytic_path`

## Active code entrypoints

- experiment config:
  - [configs/experiment/fm_cifar10_map_branch.yaml](/home/gzwlinux/vscode/gitProject/DG-TWFD/configs/experiment/fm_cifar10_map_branch.yaml)
- trainer:
  - [src/dgfm/trainers/map.py](/home/gzwlinux/vscode/gitProject/DG-TWFD/src/dgfm/trainers/map.py)
- model:
  - [src/dgfm/models/map.py](/home/gzwlinux/vscode/gitProject/DG-TWFD/src/dgfm/models/map.py)
- target builder:
  - [src/dgfm/targets/builder.py](/home/gzwlinux/vscode/gitProject/DG-TWFD/src/dgfm/targets/builder.py)
- map rollout:
  - [src/dgfm/samplers/map_sampler.py](/home/gzwlinux/vscode/gitProject/DG-TWFD/src/dgfm/samplers/map_sampler.py)
