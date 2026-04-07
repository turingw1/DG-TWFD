# MAP Branch Experiment Log

Use this file as the single experiment ledger for the
`map_branch_ctm_explicit_map` branch.

Rules:
- do not rename [A100_PIPELINE.md](/home/gzwlinux/vscode/gitProject/DG-TWFD/docs/experiments/map_branch/A100_PIPELINE.md) for each run
- record every new run here with a stable experiment id and source name
- switch runs through environment variables exported by
  [activate_fm_cifar10.sh](/home/gzwlinux/vscode/gitProject/DG-TWFD/scripts/experiments/activate_fm_cifar10.sh)
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

| EXP_TAG | EXP_VARIANT | FM_CONFIG | FM_EXP / EXP_NAME | Purpose | Status |
| --- | --- | --- | --- | --- | --- |
| e001 | map_branch | `configs/experiment/fm_cifar10_map_branch.yaml` | `fm_cifar10_map_branch_e001` | CTM-like discrete sampler baseline | planned |
| diag01 | map_branch_quick | `configs/experiment/fm_cifar10_map_branch_quick.yaml` | `fm_cifar10_map_branch_quick_diag01` | quick diagnostic before full run | planned |

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
