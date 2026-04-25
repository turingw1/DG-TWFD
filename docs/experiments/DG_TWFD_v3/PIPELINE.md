# DGTD v3 Pipeline

## 1. Purpose

This document defines the stable server-side execution flow for online-mainline
DGTD v3. Select an experiment from
[EXPERIMENT_LOG.md](../../../docs/experiments/DG_TWFD_v3/EXPERIMENT_LOG.md),
activate it once, then run the unified command families below.

## 2. Environment

On the current workspace/cache/temp server:

```bash
cd /home/ma-user/workspace/Zhengwei/DG-TWFD
source scripts/server/activate_a100_runtime.sh
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /home/ma-user/workspace/Zhengwei/DG-TWFD/.conda_envs/dg_twfd_a100
export PYTHONPATH="$PWD/src:$PWD/refs/edm:${PYTHONPATH:-}"
git checkout DG_TWFD_v3
git pull --ff-only
```

`activate_a100_runtime.sh` applies the `proxy` network profile by default. Use
`bash scripts/server/run_network_profile.sh heavy ...` for dataset/model/package
downloads.

If the repo path differs on another machine, adapt only the path prefix or
source an equivalent runtime activation script. Do not change the experiment
logic below.

## 3. Activate experiment

Pick one row from
[EXPERIMENT_LOG.md](../../../docs/experiments/DG_TWFD_v3/EXPERIMENT_LOG.md)
and source it once.

### Short run

```bash
source scripts/experiments/activate_fm_cifar10.sh dgtd_v3_smoke e401a
```

### Diagnostic run

```bash
source scripts/experiments/activate_fm_cifar10.sh dgtd_v3_diag e401b
```

### Full run

```bash
source scripts/experiments/activate_fm_cifar10.sh dgtd_v3 e402a
```

### Optional ablations

```bash
source scripts/experiments/activate_fm_cifar10.sh dgtd_cifar10_v3_ablation_no_warp_diag e403a
source scripts/experiments/activate_fm_cifar10.sh dgtd_cifar10_v3_ablation_no_warp e403a
source scripts/experiments/activate_fm_cifar10.sh dgtd_cifar10_v3_ablation_warp_no_hf_diag e404a
source scripts/experiments/activate_fm_cifar10.sh dgtd_cifar10_v3_ablation_warp_no_hf e404a
```

After activation, use only:

- `FM_CONFIG`
- `RUN_ROOT`
- `CKPT_DIR`
- `SAMPLE_ROOT`
- `LOG_ROOT`
- `METRIC_ROOT`
- `TRAIN_CUDA_VISIBLE_DEVICES`
- `INFER_CUDA_VISIBLE_DEVICES`
- `NNODES`
- `NODE_RANK`
- `NPROC_PER_NODE`
- `MASTER_ADDR`
- `MASTER_PORT`

## 4. Recommended execution order

1. Gate 0: activate `e401a`, then run train/sample/eval only to verify plumbing.
2. Gate 1: activate `e401b`, then run teacher endpoint diagnosis, train, sample,
   eval, and analysis.
3. Gate 2: launch `e402a` only if `e401b` passes the diagnostic gate.
4. If Gate 1 fails, change one module, rerun the same diag budget, and compare
   against the previous `analysis_report.json`.
5. Defer `oss001` until a checkpoint generates non-noise samples.

Current `e402a` default is tuned to raise GPU memory usage and throughput:

- `train.batch_size=512`
- `model.use_checkpoint=false`
- `eval.fid_batch_size=256`
- `eval.sample_batch_size=64`

## 5. Stable command families

### Train

Use this for the main training launch after activation.

```bash
CUDA_VISIBLE_DEVICES=${TRAIN_CUDA_VISIBLE_DEVICES} torchrun \
  --standalone \
  --nnodes=$NNODES \
  --nproc_per_node=$NPROC_PER_NODE \
  --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  scripts/run_train.py \
  --config $FM_CONFIG \
  --run-root $RUN_ROOT \
  --verbose \
  2>&1 | tee $RUN_ROOT/train.stdout_stderr.txt
```

### Resume

Use this to continue the same experiment from `last.pt`.

```bash
CUDA_VISIBLE_DEVICES=${TRAIN_CUDA_VISIBLE_DEVICES} torchrun \
  --standalone \
  --nnodes=$NNODES \
  --nproc_per_node=$NPROC_PER_NODE \
  --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  scripts/run_train.py \
  --config $FM_CONFIG \
  --run-root $RUN_ROOT \
  --resume $CKPT_DIR/last.pt \
  --verbose \
  2>&1 | tee $RUN_ROOT/train_resume.stdout_stderr.txt
```

### Teacher Endpoint Diagnosis

Run this before the diagnostic training job and keep the JSON under `RUN_ROOT`.

```bash
CUDA_VISIBLE_DEVICES=${INFER_CUDA_VISIBLE_DEVICES} python scripts/diagnose_teacher_endpoints.py \
  --config $FM_CONFIG \
  --output $RUN_ROOT/teacher_endpoint_report.json \
  --batch-size 8 \
  2>&1 | tee $RUN_ROOT/teacher_endpoint_report.stdout_stderr.txt
```

### Sample

Use this to export fixed-seed sample grids and tensors.

Short-run default:

```bash
CUDA_VISIBLE_DEVICES=${INFER_CUDA_VISIBLE_DEVICES} python scripts/run_sample_dgtd.py \
  --config $FM_CONFIG \
  --checkpoint $CKPT_DIR/last.pt \
  --output-dir $SAMPLE_ROOT/steps4 \
  --steps 4 \
  --num-samples 64 \
  --fixed-seed 42 \
  2>&1 | tee $SAMPLE_ROOT/steps4.stdout_stderr.txt
```

Full-run default:

```bash
CUDA_VISIBLE_DEVICES=${INFER_CUDA_VISIBLE_DEVICES} python scripts/run_sample_dgtd.py \
  --config $FM_CONFIG \
  --checkpoint $CKPT_DIR/best.pt \
  --output-dir $SAMPLE_ROOT/steps8 \
  --steps 8 \
  --num-samples 64 \
  --fixed-seed 42 \
  2>&1 | tee $SAMPLE_ROOT/steps8.stdout_stderr.txt
```

### Eval

Use this to generate few-step reports.

Short-run default:

```bash
CUDA_VISIBLE_DEVICES=${INFER_CUDA_VISIBLE_DEVICES} python scripts/run_eval.py \
  --config $FM_CONFIG \
  --checkpoint $CKPT_DIR/last.pt \
  --eval-root $METRIC_ROOT \
  --steps 1 2 4 \
  2>&1 | tee $METRIC_ROOT/eval.stdout_stderr.txt
```

Full-run default:

```bash
CUDA_VISIBLE_DEVICES=${INFER_CUDA_VISIBLE_DEVICES} python scripts/run_eval.py \
  --config $FM_CONFIG \
  --checkpoint $CKPT_DIR/best.pt \
  --eval-root $METRIC_ROOT \
  --steps 1 2 4 8 \
  2>&1 | tee $METRIC_ROOT/eval.stdout_stderr.txt
```

Diagnostic default:

```bash
CUDA_VISIBLE_DEVICES=${INFER_CUDA_VISIBLE_DEVICES} python scripts/run_eval.py \
  --config $FM_CONFIG \
  --checkpoint $CKPT_DIR/last.pt \
  --eval-root $METRIC_ROOT \
  --steps 1 2 4 8 \
  2>&1 | tee $METRIC_ROOT/eval.stdout_stderr.txt
```

### Analysis

Run this after sample and eval. The gate output decides whether a full run is
allowed.

```bash
python scripts/analyze_dgtd_run.py \
  --run-root $RUN_ROOT \
  --eval-root $METRIC_ROOT \
  --config $FM_CONFIG \
  --teacher-endpoint-report $RUN_ROOT/teacher_endpoint_report.json \
  --output $RUN_ROOT/analysis_report.md \
  --json-output $RUN_ROOT/analysis_report.json
```

### OSS Baseline

Use only after a usable checkpoint exists. Generate schedules first, then
evaluate the same checkpoint with the generated grids.

```bash
source scripts/experiments/activate_fm_cifar10.sh dgtd_v3_oss_baseline oss001
mkdir -p $RUN_ROOT/oss_schedules
for steps in 2 4 8; do
  CUDA_VISIBLE_DEVICES=${INFER_CUDA_VISIBLE_DEVICES} python scripts/run_sample_dgtd.py \
    --config $FM_CONFIG \
    --checkpoint /path/to/usable/best.pt \
    --output-dir $SAMPLE_ROOT/oss_steps${steps} \
    --mode mode_b_oss \
    --steps ${steps} \
    --num-samples 64 \
    --schedule-json $RUN_ROOT/oss_schedules/oss_schedule_steps${steps}.json \
    --reference-steps 32 \
    --search-batch-size 256 \
    --search-cost-batch-size 64 \
    --fixed-seed 42
done
CUDA_VISIBLE_DEVICES=${INFER_CUDA_VISIBLE_DEVICES} python scripts/run_eval.py \
  --config $FM_CONFIG \
  --checkpoint /path/to/usable/best.pt \
  --eval-root $METRIC_ROOT/oss \
  --steps 2 4 8 \
  --time-grid-dir $RUN_ROOT/oss_schedules
```

## 6. What to check before launching full run

Confirm all of the following on the short run:

- `continuation_sources.online` is nonzero and is the dominant source
- `online_continuation_rate` is nonzero
- `cached_fallback_rate` is not dominating
- `teacher_endpoint_report.json` passes: `u0` not clean input, `u1` closer to
  clean input than `u0`, endpoint time order is valid
- `best.pt` and `last.pt` exist
- `$LOG_ROOT/train.jsonl` contains the DGTD diagnostics fields
- sample and eval both complete and write outputs
- there is no traceback, NaN, or obvious `q_phi` collapse
- `analysis_report.json.gate_verdict.status` is `pass` or any failure is
  explicitly classified as a plumbing problem

## 7. What to return after each stage

### After short run

Return:

- `$RUN_ROOT/train.stdout_stderr.txt`
- `tail -n 10 $LOG_ROOT/train.jsonl`
- `find $CKPT_DIR -maxdepth 2 -type f | sort`
- `find $SAMPLE_ROOT -maxdepth 3 -type f | sort`
- `find $METRIC_ROOT -maxdepth 4 -type f | sort`
- one compact summary of `continuation_sources`, `online_continuation_rate`,
  `cached_fallback_rate`, `train_direct_teacher_error`,
  `train_direct_bridge_gap`, `train_bridge_state_teacher_error`,
  `train_bridge_u_teacher_error`, `train_teacher_rel_error_mean`,
  `alpha_online_mean/min/max`, `eta`, `beta`, `stage`, `entropy_q_phi`, and
  `kl_qD_qphi`

### After diagnostic run

Return:

- `$RUN_ROOT/teacher_endpoint_report.json`
- `$RUN_ROOT/analysis_report.md`
- `$RUN_ROOT/analysis_report.json`
- `gate_verdict.status`, `gate_verdict.failed`, and `gate_verdict.unknown`
- sample grid paths and eval summary paths for steps `1 2 4 8`

After meaningful log/doc/code updates, refresh the small crash-recovery
metadata:

```bash
bash scripts/server/snapshot_recovery_state.sh
```

For a git-backed checkpoint after reviewing the diff:

```bash
DG_TWFD_COMMIT_MESSAGE="checkpoint: <short description>" bash scripts/server/git_checkpoint.sh --commit-push
```

### After full run

Return:

- `$RUN_ROOT/train.stdout_stderr.txt`
- `tail -n 10 $LOG_ROOT/train.jsonl`
- `find $CKPT_DIR -maxdepth 2 -type f | sort`
- `find $SAMPLE_ROOT -maxdepth 3 -type f | sort`
- `find $METRIC_ROOT -maxdepth 4 -type f | sort`
- `$METRIC_ROOT/reports/summary.json`
- `$METRIC_ROOT/reports/summary.csv`
- each `steps{K}/metrics.json`
- throughput / memory measurements if available

## 8. Failure handling

- train failed:
  - return `train.stdout_stderr.txt`
  - return traceback
  - return any partial `train.jsonl`
- sample failed:
  - return sample stderr
  - return sample directory file list
- eval failed:
  - return eval stderr
  - return eval directory file list
- checkpoint missing:
  - return checkpoint directory listing
  - return train stderr
- required log fields missing:
  - return the latest `train.jsonl` lines showing the absence

## 9. Minimal acceptance rules

### Short run passes if

- train, sample, and eval all complete
- checkpoints are written
- `continuation_sources.online` is the main source
- key DGTD diagnostics are present and readable

### Full run passes if

- training completes without fatal instability
- checkpoints are written and readable
- sample and step-wise eval complete
- returned diagnostics are sufficient for the next research decision
