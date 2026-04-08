# MAP Branch Experiment Log

This file is the single operational ledger for the current systematic
experiment phase.

Rules:
- every formal run must map to one committed config under `configs/experiment/`
- activate experiments only through
  [activate_fm_cifar10.sh](/home/gzwlinux/vscode/gitProject/DG-TWFD/scripts/experiments/activate_fm_cifar10.sh)
- do not use `--set` in formal runs
- use the stable
  [A100_PIPELINE.md](/home/gzwlinux/vscode/gitProject/DG-TWFD/docs/experiments/map_branch/A100_PIPELINE.md)
  commands after activation
- old ad-hoc smoke rows are retired; this table is the new source of truth

## Naming convention

- `EXP_VARIANT`
  - exactly the config stem under `configs/experiment/`
- `EXP_TAG`
  - run id such as `e101a` or `e601a`
- `FM_EXP`
  - `${EXP_VARIANT}_${EXP_TAG}`

## Stable pipeline commands

After activation, use these commands without extra overrides.

Train:

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/run_train.py \
  --config $FM_CONFIG \
  --run-root $RUN_ROOT \
  --verbose
```

Eval:

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/run_eval.py \
  --config $FM_CONFIG \
  --checkpoint $CKPT_DIR/best.pt \
  --eval-root $METRIC_ROOT \
  --steps 1 2 4 8 16 32 64 128 256
```

Panel:

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/run_multistep_panel.py \
  --config $FM_CONFIG \
  --checkpoint $CKPT_DIR/best.pt \
  --output-dir $SAMPLE_ROOT/multistep_panel \
  --steps 1 2 4 8 16 32 64 128 256 \
  --num-examples 8 \
  --fixed-seed 42
```

## Active experiment table

| Group | EXP_TAG | EXP_VARIANT | FM_CONFIG | Activate | Purpose | Status |
| --- | --- | --- | --- | --- | --- | --- |
| E1 | e101a | `fm_cifar10_map_branch_s1_e1_traj_reg` | `configs/experiment/fm_cifar10_map_branch_s1_e1_traj_reg.yaml` | `source scripts/experiments/activate_fm_cifar10.sh fm_cifar10_map_branch_s1_e1_traj_reg e101a` | target-construction baseline: plain trajectory regression | planned |
| E1 | e102a | `fm_cifar10_map_branch_s1_e1_ctm_teacher` | `configs/experiment/fm_cifar10_map_branch_s1_e1_ctm_teacher.yaml` | `source scripts/experiments/activate_fm_cifar10.sh fm_cifar10_map_branch_s1_e1_ctm_teacher e102a` | CTM contract with teacher bridge | planned |
| E1 | e103a | `fm_cifar10_map_branch_s1_e1_ctm_ema` | `configs/experiment/fm_cifar10_map_branch_s1_e1_ctm_ema.yaml` | `source scripts/experiments/activate_fm_cifar10.sh fm_cifar10_map_branch_s1_e1_ctm_ema e103a` | CTM contract with EMA rollout bridge | planned |
| E1 | e104a | `fm_cifar10_map_branch_s1_e1_ctm_current` | `configs/experiment/fm_cifar10_map_branch_s1_e1_ctm_current.yaml` | `source scripts/experiments/activate_fm_cifar10.sh fm_cifar10_map_branch_s1_e1_ctm_current e104a` | CTM contract with current-model rollout bridge | planned |
| E2 | e201a | `fm_cifar10_map_branch_s1_e2_defect_probe` | `configs/experiment/fm_cifar10_map_branch_s1_e2_defect_probe.yaml` | `source scripts/experiments/activate_fm_cifar10.sh fm_cifar10_map_branch_s1_e2_defect_probe e201a` | defect-probe run with learnable timewarp diagnostics | planned |
| E3 | e301a | `fm_cifar10_map_branch_s1_e3_pred_residual` | `configs/experiment/fm_cifar10_map_branch_s1_e3_pred_residual.yaml` | `source scripts/experiments/activate_fm_cifar10.sh fm_cifar10_map_branch_s1_e3_pred_residual e301a` | prediction target ablation: residual map | planned |
| E3 | e302a | `fm_cifar10_map_branch_s1_e3_pred_direct` | `configs/experiment/fm_cifar10_map_branch_s1_e3_pred_direct.yaml` | `source scripts/experiments/activate_fm_cifar10.sh fm_cifar10_map_branch_s1_e3_pred_direct e302a` | prediction target ablation: direct endpoint map | planned |
| E4 | e401a | `fm_cifar10_map_branch_s1_e3_pred_residual` | `configs/experiment/fm_cifar10_map_branch_s1_e3_pred_residual.yaml` | `source scripts/experiments/activate_fm_cifar10.sh fm_cifar10_map_branch_s1_e3_pred_residual e401a` | auxiliary ablation baseline: endpoint off | planned |
| E4 | e402a | `fm_cifar10_map_branch_s1_e4_aux_endpoint_on` | `configs/experiment/fm_cifar10_map_branch_s1_e4_aux_endpoint_on.yaml` | `source scripts/experiments/activate_fm_cifar10.sh fm_cifar10_map_branch_s1_e4_aux_endpoint_on e402a` | auxiliary ablation: endpoint on | planned |
| E5 | e501a | `fm_cifar10_map_branch_s1_e5_warp_identity` | `configs/experiment/fm_cifar10_map_branch_s1_e5_warp_identity.yaml` | `source scripts/experiments/activate_fm_cifar10.sh fm_cifar10_map_branch_s1_e5_warp_identity e501a` | warp ablation: identity clock | planned |
| E5 | e502a | `fm_cifar10_map_branch_s1_e5_warp_data_dense` | `configs/experiment/fm_cifar10_map_branch_s1_e5_warp_data_dense.yaml` | `source scripts/experiments/activate_fm_cifar10.sh fm_cifar10_map_branch_s1_e5_warp_data_dense e502a` | warp ablation: static data-dense power warp | planned |
| E5 | e503a | `fm_cifar10_map_branch_s1_e5_warp_source_dense` | `configs/experiment/fm_cifar10_map_branch_s1_e5_warp_source_dense.yaml` | `source scripts/experiments/activate_fm_cifar10.sh fm_cifar10_map_branch_s1_e5_warp_source_dense e503a` | warp ablation: static source-dense power warp | planned |
| E5 | e504a | `fm_cifar10_map_branch_s1_e5_warp_learned` | `configs/experiment/fm_cifar10_map_branch_s1_e5_warp_learned.yaml` | `source scripts/experiments/activate_fm_cifar10.sh fm_cifar10_map_branch_s1_e5_warp_learned e504a` | warp ablation: learnable monotone warp | planned |
| E6 | e601a | `fm_cifar10_map_branch_s1_e6_budget_quick` | `configs/experiment/fm_cifar10_map_branch_s1_e6_budget_quick.yaml` | `source scripts/experiments/activate_fm_cifar10.sh fm_cifar10_map_branch_s1_e6_budget_quick e601a` | budget sensitivity: quick budget | planned |
| E6 | e602a | `fm_cifar10_map_branch_s1_e6_budget_full` | `configs/experiment/fm_cifar10_map_branch_s1_e6_budget_full.yaml` | `source scripts/experiments/activate_fm_cifar10.sh fm_cifar10_map_branch_s1_e6_budget_full e602a` | budget sensitivity: full budget | planned |

## Experiment order

1. Run `E1` first and pick the best target-construction recipe.
2. Run `E2` once to check whether defect diagnostics are coherent.
3. Run `E3` to lock prediction parameterization.
4. Run `E4` to decide whether endpoint should stay.
5. Run `E5` to decide whether any warp strategy is worth keeping.
6. Use the winning recipe to interpret `E6`.

## Expected reading of the outputs

For every row, inspect:
- `$LOG_ROOT/train.jsonl`
- `$METRIC_ROOT/reports/summary.json`
- `$SAMPLE_ROOT/multistep_panel/multistep_panel.png`

Primary decision fields:
- FID trend from `1` to `16`
- whether improvement continues beyond `16`
- `train_update_ratio / val_update_ratio`
- `train_update_cosine / val_update_cosine`
- `timewarp_time_grid` and `timewarp_interval_defects` when warp is enabled
