# DGFM Map Branch

This branch is focused only on the explicit map-learning experiment built on top
of the current `dgfm` framework.

## Branch scope

- baseline FM code remains available as shared infrastructure
- active research target:
  - explicit map learning
  - `M_theta(x_t, t, s) -> x_s`
- current first-stage target mode:
  - analytic path targets under current `dgfm` time semantics

## Use this documentation set

All branch-facing documentation is under:

- [docs/experiments/map_branch/README.md](/home/gzwlinux/vscode/gitProject/DG-TWFD/docs/experiments/map_branch/README.md)

Start there. The A100 launch instructions are in:

- [docs/experiments/map_branch/A100_PIPELINE.md](/home/gzwlinux/vscode/gitProject/DG-TWFD/docs/experiments/map_branch/A100_PIPELINE.md)

## Branch implementation entrypoints

- train:
  - [scripts/run_train.py](/home/gzwlinux/vscode/gitProject/DG-TWFD/scripts/run_train.py)
- eval:
  - [scripts/run_eval.py](/home/gzwlinux/vscode/gitProject/DG-TWFD/scripts/run_eval.py)
- sample:
  - [scripts/run_sample.py](/home/gzwlinux/vscode/gitProject/DG-TWFD/scripts/run_sample.py)
- multistep qualitative panel:
  - [scripts/run_multistep_panel.py](/home/gzwlinux/vscode/gitProject/DG-TWFD/scripts/run_multistep_panel.py)

## Branch config entrypoint

- [configs/experiment/fm_cifar10_map_branch.yaml](/home/gzwlinux/vscode/gitProject/DG-TWFD/configs/experiment/fm_cifar10_map_branch.yaml)
