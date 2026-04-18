# DGTD v3 Pipeline

## 1. Purpose

This document defines the stable server-side execution flow for online-mainline
DGTD v3. Select an experiment from
[EXPERIMENT_LOG.md](/home/gzwlinux/vscode/gitProject/DG-TWFD/docs/experiments/DG_TWFD_v3/EXPERIMENT_LOG.md),
activate it once, then run the unified command families below.

## 2. Environment

If the server environment already exists:

```bash
cd /data2/yl7622/Zhengwei/DG-TWFD
source /data2/yl7622/anaconda/etc/profile.d/conda.sh
conda activate /data2/yl7622/Zhengwei/DG-TWFD/.conda_envs/dgfm_map
git checkout DG_TWFD_v3
git pull --ff-only
```

If the local path differs, adapt only the shell path prefix, not the experiment
logic below.

## 3. Activate experiment

Pick one row from
[EXPERIMENT_LOG.md](/home/gzwlinux/vscode/gitProject/DG-TWFD/docs/experiments/DG_TWFD_v3/EXPERIMENT_LOG.md)
and source it once.

### Short run

```bash
source scripts/experiments/activate_fm_cifar10.sh dgtd_v3_smoke e401a
```

### Full run

```bash
source scripts/experiments/activate_fm_cifar10.sh dgtd_v3 e402a
```

### Optional ablations

```bash
source scripts/experiments/activate_fm_cifar10.sh dgtd_cifar10_v3_ablation_no_warp e403a
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

1. Activate `e401a`.
2. Run short-run train.
3. Run short-run sample.
4. Run short-run eval.
5. Check online continuation source, checkpoint outputs, and core diagnostics.
6. Activate `e402a`.
7. Run full-run train.
8. Run full-run sample.
9. Run full-run eval.
10. Fill the selected row in `EXPERIMENT_LOG.md` and return all evidence.

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

## 6. What to check before launching full run

Confirm all of the following on the short run:

- `continuation_sources.online` is nonzero and is the dominant source
- `online_continuation_rate` is nonzero
- `cached_fallback_rate` is not dominating
- `best.pt` and `last.pt` exist
- `$LOG_ROOT/train.jsonl` contains the DGTD diagnostics fields
- sample and eval both complete and write outputs
- there is no traceback, NaN, or obvious `q_phi` collapse

## 7. What to return after each stage

### After short run

Return:

- `$RUN_ROOT/train.stdout_stderr.txt`
- `tail -n 10 $LOG_ROOT/train.jsonl`
- `find $CKPT_DIR -maxdepth 2 -type f | sort`
- `find $SAMPLE_ROOT -maxdepth 3 -type f | sort`
- `find $METRIC_ROOT -maxdepth 4 -type f | sort`
- one compact summary of:
  - `continuation_sources`
  - `online_continuation_rate`
  - `cached_fallback_rate`
  - `train_direct_teacher_error`
  - `train_direct_bridge_gap`
  - `train_bridge_state_teacher_error`
  - `train_bridge_u_teacher_error`
  - `train_teacher_rel_error_mean`
  - `alpha_online_mean/min/max`
  - `eta`, `beta`, `stage`
  - `entropy_q_phi`
  - `kl_qD_qphi`

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
