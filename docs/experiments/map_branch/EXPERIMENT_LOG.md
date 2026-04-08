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
| tw001 | map_branch_timewarp_probe | `configs/experiment/fm_cifar10_map_branch_timewarp_probe.yaml` | `fm_cifar10_map_branch_timewarp_probe_tw001` | `source scripts/experiments/activate_fm_cifar10.sh map_branch_timewarp_probe tw001` | quick defect-driven timewarp smoke test | planned |
| tws01 | map_branch_timewarp_smoke | `configs/experiment/fm_cifar10_map_branch_timewarp_smoke.yaml` | `fm_cifar10_map_branch_timewarp_smoke_tws01` | `source scripts/experiments/activate_fm_cifar10.sh map_branch_timewarp_smoke tws01` | minimal viability run for timewarp + map diagnostics | planned |
| tws02 | map_branch_timewarp_smoke | `configs/experiment/fm_cifar10_map_branch_timewarp_smoke.yaml` | `fm_cifar10_map_branch_timewarp_smoke_tws02` | `source scripts/experiments/activate_fm_cifar10.sh map_branch_timewarp_smoke tws02` | smoke ablation with timewarp disabled | planned |
| tws03 | map_branch_timewarp_smoke | `configs/experiment/fm_cifar10_map_branch_timewarp_smoke.yaml` | `fm_cifar10_map_branch_timewarp_smoke_tws03` | `source scripts/experiments/activate_fm_cifar10.sh map_branch_timewarp_smoke tws03` | smoke ablation with endpoint loss disabled | planned |

## Pipeline usage contract

1. Add or update the target row in this file.
2. Run the row's `Activate` command once.
3. Enter [A100_PIPELINE.md](/home/gzwlinux/vscode/gitProject/DG-TWFD/docs/experiments/map_branch/A100_PIPELINE.md).
4. Use the fixed pipeline commands directly with the exported `$FM_CONFIG`, `$RUN_ROOT`, `$CKPT_DIR`, `$METRIC_ROOT`, and `$SAMPLE_ROOT`.

For diagnostics that intentionally deviate from the stable pipeline template,
record the exact commands under the experiment note in this file and treat the
note as the source of truth for that run.

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

### tw001

- intended as the first direct validation that learned timewarp can change the
  effective time grid during training
- based on the quick map-branch config to keep turnaround short
- enables:
  - `scheduler.timewarp.enabled=true`
  - `scheduler.timewarp.type=learnable_monotone`
  - `loss.timewarp_weight=1.0`
- expected validation points:
  - `train.jsonl` should show non-uniform `timewarp_time_grid`
  - `train_timewarp_defect_loss` should trend down
  - eval `metrics.json` should export the learned `time_grid`

### tws01

- intended for fast failure detection before spending A100 time on larger runs
- compared with `tw001`, this variant further reduces:
  - teacher internal steps
  - retained scales
  - batch size
  - epoch count
  - train/val batch caps
  - eval sample count
- recommended first pass:
  - train with pipeline default command
  - inspect `logs/train.jsonl`
  - only if the diagnostics look sane, move to `tw001` or a larger run
- activation:
  - `source scripts/experiments/activate_fm_cifar10.sh map_branch_timewarp_smoke tws01`
- train:
```bash
CUDA_VISIBLE_DEVICES=1 python scripts/run_train.py \
  --config $FM_CONFIG \
  --run-root $RUN_ROOT \
  --verbose
```
- eval:
```bash
CUDA_VISIBLE_DEVICES=1 python scripts/run_eval.py \
  --config $FM_CONFIG \
  --checkpoint $CKPT_DIR/best.pt \
  --eval-root $METRIC_ROOT \
  --steps 1 2 4 8 16 32 64 128 256 \
  --fid-samples 1000 \
  --sample-batch-size 16
```
- panel:
```bash
CUDA_VISIBLE_DEVICES=1 python scripts/run_multistep_panel.py \
  --config $FM_CONFIG \
  --checkpoint $CKPT_DIR/best.pt \
  --output-dir $SAMPLE_ROOT/panel_best \
  --steps 1 4 16 64 128 256
```
- expected outputs:
  - `logs/train.jsonl`
  - `$METRIC_ROOT/reports/summary.json`
  - `$SAMPLE_ROOT/panel_best/multistep_panel.png`

### tws02

- purpose:
  - isolate whether learned timewarp materially helps
  - keep the same smoke backbone as `tws01`
  - disable timewarp in both training and evaluation
- activation:
  - `source scripts/experiments/activate_fm_cifar10.sh map_branch_timewarp_smoke tws02`
- train:
```bash
CUDA_VISIBLE_DEVICES=1 python scripts/run_train.py \
  --config $FM_CONFIG \
  --run-root $RUN_ROOT \
  --verbose \
  --set scheduler.timewarp.enabled=false \
  --set loss.timewarp_weight=0.0
```
- eval:
```bash
CUDA_VISIBLE_DEVICES=1 python scripts/run_eval.py \
  --config $FM_CONFIG \
  --checkpoint $CKPT_DIR/best.pt \
  --eval-root $METRIC_ROOT \
  --steps 1 2 4 8 16 32 64 128 256 \
  --fid-samples 1000 \
  --sample-batch-size 16 \
  --set scheduler.timewarp.enabled=false \
  --set loss.timewarp_weight=0.0
```
- panel:
```bash
CUDA_VISIBLE_DEVICES=1 python scripts/run_multistep_panel.py \
  --config $FM_CONFIG \
  --checkpoint $CKPT_DIR/best.pt \
  --output-dir $SAMPLE_ROOT/panel_best \
  --steps 1 4 16 64 128 256 \
  --set scheduler.timewarp.enabled=false \
  --set loss.timewarp_weight=0.0
```
- comparison target:
  - compare directly against `tws01`
  - if outputs are nearly identical, timewarp is not the current bottleneck

### tws03

- purpose:
  - test whether endpoint supervision is fighting the main map objective
  - keep learned timewarp enabled
  - disable endpoint loss only
- activation:
  - `source scripts/experiments/activate_fm_cifar10.sh map_branch_timewarp_smoke tws03`
- train:
```bash
CUDA_VISIBLE_DEVICES=1 python scripts/run_train.py \
  --config $FM_CONFIG \
  --run-root $RUN_ROOT \
  --verbose \
  --set loss.endpoint_weight=0.0
```
- eval:
```bash
CUDA_VISIBLE_DEVICES=1 python scripts/run_eval.py \
  --config $FM_CONFIG \
  --checkpoint $CKPT_DIR/best.pt \
  --eval-root $METRIC_ROOT \
  --steps 1 2 4 8 16 32 64 128 256 \
  --fid-samples 1000 \
  --sample-batch-size 16 \
  --set loss.endpoint_weight=0.0
```
- panel:
```bash
CUDA_VISIBLE_DEVICES=1 python scripts/run_multistep_panel.py \
  --config $FM_CONFIG \
  --checkpoint $CKPT_DIR/best.pt \
  --output-dir $SAMPLE_ROOT/panel_best \
  --steps 1 4 16 64 128 256 \
  --set loss.endpoint_weight=0.0
```
- comparison target:
  - compare directly against `tws01`
  - if panel/FID improve while core loss remains reasonable, endpoint loss is likely conflicting with rollout quality
