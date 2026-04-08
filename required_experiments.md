# Required Experiments for DG-TWFD

## Scope

This file is the executable experiment plan for the **current** `dgfm`
map-branch codebase.

It is intentionally narrower than the long-term paper plan. The immediate
goal is:

- finish algorithmic iteration inside the current framework
- use only experiments that the current code can run directly
- avoid any experiment that depends on missing infrastructure
- make every run reproducible from:
  - one config file
  - one activation command
  - the stable
    [A100_PIPELINE.md](/home/gzwlinux/vscode/gitProject/DG-TWFD/docs/experiments/map_branch/A100_PIPELINE.md)

## Feasibility audit against the original wish list

### Supported now

- CIFAR-10 map-branch training
- explicit map prediction with:
  - residual parameterization
  - direct endpoint parameterization
- CTM-style target contract with:
  - `trajectory_regression`
  - `ctm_consistency`
  - `bridge_source in {teacher, ema_model_rollout, current_model_rollout}`
- static and learned monotone time warps
- few-step FID evaluation from one checkpoint at multiple step counts
- qualitative panels
- rich train/val diagnostics in `train.jsonl`

### Not supported now

- dedicated 2D toy-mechanism package inside the `dgfm` map-branch system
- ImageNet64 teacher and map-branch benchmark path
- Inception Score evaluator
- Recall evaluator
- a standalone held-out semigroup-defect evaluation runner detached from
  training logs
- defect-adaptive curriculum as a first-class training module
- spline warp family

### Required modifications to the original plan

- replace the toy 2D experiment with a **CIFAR-10 smoke mechanism experiment**
- restrict the current systematic iteration to **CIFAR-10 only**
- replace unsupported secondary metrics with:
  - multi-step FID
  - average low-step FID
  - logged composition/warp diagnostics
- replace the unsupported standalone defect evaluator with the currently
  available defect surrogates:
  - `train_timewarp_defect_loss`
  - `val_timewarp_defect_loss`
  - `timewarp_interval_defects`
  - `update_ratio`
  - `update_cosine`

These replacements are necessary because the point of the current phase is to
establish a **working, reproducible algorithmic recipe** before broadening the
benchmark surface.

## Core evaluation rules

- use one checkpoint reused at:
  - `1 / 2 / 4 / 8 / 16 / 32 / 64 / 128 / 256`
- compare ablations under the same:
  - backbone
  - teacher
  - smoke budget or full budget
  - evaluation command
- do not use `--set` overrides in formal runs
- every formal experiment must be backed by:
  - one committed config
  - one `EXP_LOG` row

## Primary metrics for the current phase

- FID at `1 / 2 / 4 / 8 / 16 / 32 / 64 / 128 / 256`
- average low-step FID over:
  - `1 / 2 / 4 / 8 / 16`
- train/val diagnostics from `train.jsonl`

## Required diagnostics

- `train_update_ratio`
- `val_update_ratio`
- `train_update_cosine`
- `val_update_cosine`
- `train_pred_update_abs_mean`
- `val_pred_update_abs_mean`
- `timewarp_time_grid` when warp is enabled
- `timewarp_interval_defects` when warp is enabled

## Executable experiment chain

## Experiment 1. Target-construction mechanism ablation

### Question

Does the new CTM-style target contract help more than plain trajectory
regression, and which bridge source is the right default?

### Variants

- `trajectory_regression`
- `ctm_consistency + teacher bridge`
- `ctm_consistency + ema rollout bridge`
- `ctm_consistency + current-model rollout bridge`

### Configs

- `fm_cifar10_map_branch_s1_e1_traj_reg`
- `fm_cifar10_map_branch_s1_e1_ctm_teacher`
- `fm_cifar10_map_branch_s1_e1_ctm_ema`
- `fm_cifar10_map_branch_s1_e1_ctm_current`

### Acceptance

- identify the best target-construction recipe at smoke scale
- prefer the recipe whose FID improves most consistently from `1` to `16`
  steps

## Experiment 2. Defect-probe run

### Question

When time warp is enabled on top of the best smoke recipe, do defect and warp
diagnostics move in a coherent direction?

### Variant

- `ctm_consistency + ema rollout bridge + learned monotone warp`

### Config

- `fm_cifar10_map_branch_s1_e2_defect_probe`

### Acceptance

- `timewarp_time_grid` becomes non-uniform during training
- defect-related diagnostics decrease or stabilize
- this run is diagnostic-first, not quality-first

## Experiment 3. Prediction target ablation

### Question

Inside the explicit-map family, should the model predict the target directly or
through a residual parameterization?

### Variants

- `prediction_type=residual`
- `prediction_type=direct`

### Configs

- `fm_cifar10_map_branch_s1_e3_pred_residual`
- `fm_cifar10_map_branch_s1_e3_pred_direct`

### Modification from the original plan

The original request included a velocity-style target in the same ablation.
That is **not** a clean matched comparison inside the current framework because
velocity FM lives in a different training objective family and backbone path.
For the current phase, velocity FM remains a cross-family reference, not a
same-table matched ablation.

## Experiment 4. Auxiliary-loss reintroduction check

### Question

After the target-construction upgrade, does reintroducing endpoint loss help or
hurt rollout quality?

### Variants

- no endpoint auxiliary
- endpoint auxiliary on

### Configs

- `fm_cifar10_map_branch_s1_e3_pred_residual`
- `fm_cifar10_map_branch_s1_e4_aux_endpoint_on`

### Acceptance

- keep endpoint only if it improves low-step FID without destabilizing
  longer-step rollout

## Experiment 5. Warp strategy ablation

### Question

Does the method benefit from the right time geometry, and if so, which warp
form is actually useful in the current framework?

### Variants

- identity clock
- static data-dense power warp
- static source-dense power warp
- learned monotone warp

### Configs

- `fm_cifar10_map_branch_s1_e5_warp_identity`
- `fm_cifar10_map_branch_s1_e5_warp_data_dense`
- `fm_cifar10_map_branch_s1_e5_warp_source_dense`
- `fm_cifar10_map_branch_s1_e5_warp_learned`

### Modification from the original plan

The original plan requested spline warp and teacher-style heuristic schedule.
These are not first-class options in the current `dgfm` map-branch scheduler,
so they are deferred.

## Experiment 6. Budget sensitivity

### Question

Once the best smoke recipe is selected, does it remain stable when scaled to
the quick and full training budgets?

### Variants

- smoke:
  - taken from the winning recipe in Experiments 1 to 5
- quick budget
- full budget

### Configs

- winning smoke config from Experiments 1 to 5
- `fm_cifar10_map_branch_s1_e6_budget_quick`
- `fm_cifar10_map_branch_s1_e6_budget_full`

### Modification from the original plan

The original plan requested model-size scaling and ImageNet benchmarking.
Those require additional backbone recipes, teacher definitions, and evaluators
that are not yet in the current map-branch system. The current executable
budget study is therefore **training-budget-only**.

## Current minimum deliverables

- one complete target-construction ablation table
- one defect-probe run with usable diagnostics
- one prediction-target ablation
- one warp-strategy ablation
- one quick/full budget follow-up on the best recipe

## What is deliberately deferred

- ImageNet64
- Recall
- Inception Score
- spline warp
- standalone held-out defect evaluator
- curriculum as a separate training module
- paper-ready baseline comparison table

These are not rejected permanently. They are deferred until the current
algorithmic path becomes stable enough that wider benchmarking is meaningful.
