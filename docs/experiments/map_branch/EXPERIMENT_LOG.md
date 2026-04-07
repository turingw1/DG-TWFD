# MAP Branch Experiment Log

Use this file as the single experiment ledger for the
`map_branch_ctm_explicit_map` branch.

Rules:
- do not rename [A100_PIPELINE.md](/home/gzwlinux/vscode/gitProject/DG-TWFD/docs/experiments/map_branch/A100_PIPELINE.md) for each run
- record every new run here with a stable experiment id and source name
- switch runs through environment variables exported by
  [activate_fm_cifar10.sh](/home/gzwlinux/vscode/gitProject/DG-TWFD/scripts/experiments/activate_fm_cifar10.sh)
- before using the pipeline, select one row here and run its activation command
- keep one row per experiment id
- add short comments under the table when an experiment needs extra context

## Naming convention

- `EXP_VARIANT`
  - source config family such as `map_branch` or `map_branch_quick`
- `EXP_TAG`
  - short experiment id such as `e001`, `e002`, `diag01`
- `EXP_NAME`
  - `${EXP_VARIANT}_${EXP_TAG}`
- `FM_EXP`
  - same as `EXP_NAME`

## Active experiment table

| EXP_TAG | EXP_VARIANT | FM_CONFIG | FM_EXP / EXP_NAME | Activate | Purpose | Status |
| --- | --- | --- | --- | --- | --- | --- |
| e001 | map_branch | `configs/experiment/fm_cifar10_map_branch.yaml` | `fm_cifar10_map_branch_e001` | `source scripts/experiments/activate_fm_cifar10.sh map_branch e001` | CTM-like discrete sampler baseline | planned |
| diag01 | map_branch_quick | `configs/experiment/fm_cifar10_map_branch_quick.yaml` | `fm_cifar10_map_branch_quick_diag01` | `source scripts/experiments/activate_fm_cifar10.sh map_branch_quick diag01` | quick diagnostic before full run | planned |

## Pipeline usage contract

1. Add or update the target row in this file.
2. Run the row's `Activate` command once.
3. Enter [A100_PIPELINE.md](/home/gzwlinux/vscode/gitProject/DG-TWFD/docs/experiments/map_branch/A100_PIPELINE.md).
4. Use the fixed pipeline commands directly with the exported `$FM_CONFIG`, `$RUN_ROOT`, `$CKPT_DIR`, `$METRIC_ROOT`, and `$SAMPLE_ROOT`.

## Notes

### e001

- intended as the first full run after switching away from heuristic pair sampling
- source target policy:
  - `target.sampling_mode=ctm_discrete`
  - `target.start_scales=33`
  - `target.num_heun_step=17`
  - `target.sample_s_strategy=uniform`
- keep endpoint loss enabled as an auxiliary interface

### diag01

- intended to validate direction only
- quick config keeps reduced teacher steps and lighter endpoint supervision
