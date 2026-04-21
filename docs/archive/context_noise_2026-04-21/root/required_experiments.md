# Required Experiments for DG-TWFD

## Scope

This file defines the **current executable experiment surface** after the
external-integration update driven by `plan.md`.

The plan is now split into three layers:

- `Stage 0`
  - server readiness smoke for the current two-A6000 distributed branch
- `Stage 1`
  - algorithm selection inside the current CIFAR-10 map-branch system
- `Stage 2`
  - external-facing validation and infrastructure experiments:
    - official-style `.npz` metrics
    - held-out defect validation
    - ImageNet64 data / baseline smoke

Every formal run in this file must satisfy:

- one committed config under `configs/experiment/`
- one row in
  [EXPERIMENT_LOG.md](/home/gzwlinux/vscode/gitProject/DG-TWFD/docs/experiments/map_branch/EXPERIMENT_LOG.md)
- activation through
  [activate_fm_cifar10.sh](/home/gzwlinux/vscode/gitProject/DG-TWFD/scripts/experiments/activate_fm_cifar10.sh)
- no `--set` overrides in formal runs

## Capability status

### Supported now

- CIFAR-10 explicit-map training and evaluation
- CTM-style target contract with:
  - `trajectory_regression`
  - `ctm_consistency`
  - `bridge_source in {teacher, ema_model_rollout, current_model_rollout}`
- static, learned, and minimum-viable spline-style monotone time warps
- official-style sample export:
  - `.npz` with `arr_0=[N,H,W,C] uint8`
  - optional `labels`
- official-style metric bridge through `torch_fidelity` on `.npz`:
  - FID
  - Inception Score
  - Precision
  - Recall
- held-out defect evaluation from checkpoint + fixed triplets
- ImageNet64 dataset ingestion from:
  - raw ILSVRC-style train folders
  - preprocessed folder
  - preprocessed zip
- ImageNet64 baseline smoke configs

### Explicitly not claimed yet

- CTM-faithful ImageNet64 map-branch teacher integration
- full OpenAI/CTM teacher checkpoint execution inside `dgfm` map-branch
- paper-grade ImageNet64 map-branch reproduction

Those remain outside the current required experiment set. The current code now
captures:

- dataset path
- teacher metadata/config defaults
- official evaluation protocol

but not a finished ImageNet64 map-branch teacher backend.

## Core evaluation rules

- algorithmic ablations reuse one checkpoint at:
  - `1 / 2 / 4 / 8 / 16 / 32 / 64 / 128 / 256`
- official metrics runs export a single `.npz` per step count
- official metrics must use one reference `.npz`
- held-out defect runs must use:
  - fixed seed
  - fixed triplet preset or fixed dense default
- spline warp experiments must be compared against:
  - identity clock
  - learned monotone warp

## Primary metrics

### Stage 1

- FID at `1 / 2 / 4 / 8 / 16 / 32 / 64 / 128 / 256`
- average low-step FID over:
  - `1 / 2 / 4 / 8 / 16`
- train/val diagnostics from `train.jsonl`

### Stage 2

- official FID
- official IS mean/std
- official Precision
- official Recall
- held-out `defect_mean / defect_std / defect_median`
- `defect_by_t_bin`
- `defect_by_step_count`

## Required diagnostics

- `train_update_ratio`
- `val_update_ratio`
- `train_update_cosine`
- `val_update_cosine`
- `train_pred_update_abs_mean`
- `val_pred_update_abs_mean`
- `timewarp_time_grid` when warp is enabled
- `timewarp_interval_defects` when warp is enabled
- held-out defect report json for Stage 2 defect runs

## Stage 0. Server readiness smoke

### Experiment 0. A6000 preflight

Question:
- can the current server run both:
  - the short fullstack teacher-backed smoke
  - the two-GPU DDP initialization smoke
  before a long run starts?

Config:
- `fm_cifar10_map_branch_s0_a6000_fullstack_smoke`

Acceptance:
- one short train run completes
- endpoint rollout executes
- timewarp update executes
- few-step eval completes
- no CUDA allocator / driver issue appears in the preflight
- the distributed smoke in
  [DISTRIBUTED_SMOKE.md](/home/gzwlinux/vscode/gitProject/DG-TWFD/docs/experiments/map_branch/DISTRIBUTED_SMOKE.md)
  passes on GPUs `0,1`

## Stage 1. CIFAR-10 algorithm selection

### Experiment 1. Target-construction mechanism ablation

Question:
- does the CTM-style target contract help more than plain trajectory
  regression, and which bridge source should be the default?

Configs:
- `fm_cifar10_map_branch_s1_e1_traj_reg`
- `fm_cifar10_map_branch_s1_e1_ctm_teacher`
- `fm_cifar10_map_branch_s1_e1_ctm_ema`
- `fm_cifar10_map_branch_s1_e1_ctm_current`

Acceptance:
- select one target-construction recipe at smoke scale
- prefer the recipe whose FID improves most coherently from `1` to `16`
  steps and does not collapse beyond `16`

### Experiment 2. Defect-probe run

Question:
- with the selected smoke recipe, does defect-driven time warp move its
  diagnostics in a coherent direction?

Config:
- `fm_cifar10_map_branch_s1_e2_defect_probe`

Acceptance:
- `timewarp_time_grid` becomes non-uniform
- defect-related diagnostics decrease or stabilize
- this run is diagnostic-first, not quality-first

### Experiment 3. Prediction target ablation

Question:
- inside the explicit-map family, should the model predict residual updates or
  direct endpoints?

Configs:
- `fm_cifar10_map_branch_s1_e3_pred_residual`
- `fm_cifar10_map_branch_s1_e3_pred_direct`

Acceptance:
- keep the parameterization with better low-step FID and more coherent
  update-ratio / update-cosine diagnostics

### Experiment 4. Auxiliary-loss reintroduction

Question:
- after the target-construction upgrade, does endpoint auxiliary loss help or
  hurt rollout quality?

Configs:
- `fm_cifar10_map_branch_s1_e3_pred_residual`
- `fm_cifar10_map_branch_s1_e4_aux_endpoint_on`

Acceptance:
- keep endpoint auxiliary only if it improves low-step FID without harming
  longer-step rollout

### Experiment 5. Warp strategy ablation

Question:
- which time geometry is actually useful in the current framework?

Configs:
- `fm_cifar10_map_branch_s1_e5_warp_identity`
- `fm_cifar10_map_branch_s1_e5_warp_data_dense`
- `fm_cifar10_map_branch_s1_e5_warp_source_dense`
- `fm_cifar10_map_branch_s1_e5_warp_learned`
- `fm_cifar10_map_branch_s1_e5_warp_spline`

Acceptance:
- compare static, learned, and spline-style clocks on the same smoke recipe
- keep warp only if it improves low-step FID or held-out defect trends

### Experiment 6. Budget sensitivity

Question:
- does the selected smoke recipe stay stable at quick and full training
  budgets?

Configs:
- winning smoke config from Experiments 1 to 5
- `fm_cifar10_map_branch_s1_e6_budget_quick`
- `fm_cifar10_map_branch_s1_e6_budget_full`

Acceptance:
- the same recipe should not invert its qualitative ranking when moved from
  smoke to quick/full budgets

## Stage 2. External-facing validation

### Experiment 7. Official `.npz` metric bridge

Question:
- does the selected checkpoint hold up under official-style `.npz` evaluation
  rather than only the in-framework few-step runner?

Config:
- `fm_cifar10_map_branch_s2_official_metrics`

Required command family:
- export `.npz` samples with
  [run_export_samples_npz.py](/home/gzwlinux/vscode/gitProject/DG-TWFD/scripts/run_export_samples_npz.py)
- evaluate them with
  [run_evaluate_metrics.py](/home/gzwlinux/vscode/gitProject/DG-TWFD/scripts/run_evaluate_metrics.py)

Acceptance:
- at least one selected step count must have a completed json report with:
  - FID
  - IS mean/std
  - Precision
  - Recall

### Experiment 8. Held-out defect validation

Question:
- do the selected checkpoints reduce semigroup defect on held-out seeds and
  fixed triplets, rather than only reducing the training-side surrogate?

Config:
- `fm_cifar10_map_branch_s2_defect_eval`

Required command family:
- run
  [run_evaluate_defect.py](/home/gzwlinux/vscode/gitProject/DG-TWFD/scripts/run_evaluate_defect.py)

Acceptance:
- produce one report json per selected checkpoint
- compare at least:
  - winning no-warp recipe
  - winning learned-warp or spline-warp recipe

### Experiment 9. ImageNet64 data / baseline smoke

Question:
- is the system able to ingest ImageNet64-style data and run a class-conditional
  baseline smoke end-to-end?

Config:
- `fm_imagenet64_baseline_smoke`

Required command family:
- prepare data with
  [prepare_imagenet64.py](/home/gzwlinux/vscode/gitProject/DG-TWFD/scripts/prepare_imagenet64.py)
  when needed
- train / eval via the stable pipeline after activation
- optionally export `.npz` samples and run official metrics if the reference
  batch is available

Acceptance:
- dataloaders build successfully
- train/eval run without code changes
- class-conditional sampling path runs and writes labels

## Minimum deliverables

- one complete Stage 1 target-construction table
- one complete warp-strategy table including spline warp
- one quick/full budget follow-up
- one official `.npz` metrics report
- one held-out defect report
- one ImageNet64 baseline smoke run

## Current execution order

1. Finish Stage 1 and lock one CIFAR-10 recipe.
2. Run official `.npz` metrics on the selected checkpoint.
3. Run held-out defect evaluation on the selected checkpoints.
4. Run ImageNet64 baseline smoke to verify the external data/eval path.
