# DGTD v3 Experiment Log

This file is the operational ledger for formal DGTD v3 experiments on the
online-mainline branch state.

Rules:

- every formal run must map to one committed config under `configs/experiment/`
- activate experiments only through
  [activate_fm_cifar10.sh](../../../scripts/experiments/activate_fm_cifar10.sh)
- do not use `--set` in formal runs
- use the stable command families from
  [PIPELINE.md](../../../docs/experiments/DG_TWFD_v3/PIPELINE.md)
- `EXP_LOG.md` defines the run identity, config choice, and experiment-specific
  purpose
- `PIPELINE.md` defines the concrete train / sample / eval execution steps

## Naming convention

- `EXP_VARIANT`
  - exactly the config stem under `configs/experiment/`
- `EXP_TAG`
  - run id such as `e401a`
- `FM_EXP`
  - `${EXP_VARIANT}_${EXP_TAG}`

## Stable activation pattern

Activate the selected experiment once:

```bash
source scripts/experiments/activate_fm_cifar10.sh <EXP_VARIANT> <EXP_TAG>
```

This sets the stable variables used by the pipeline:

- `FM_CONFIG`
- `RUN_ROOT`
- `CKPT_DIR`
- `SAMPLE_ROOT`
- `LOG_ROOT`
- `METRIC_ROOT`
- `TRAJ_ROOT`
- `HF_HOME`
- `HF_HUB_CACHE`
- `TORCH_HOME`
- `TRAIN_CUDA_VISIBLE_DEVICES`
- `INFER_CUDA_VISIBLE_DEVICES`
- `NNODES`
- `NODE_RANK`
- `NPROC_PER_NODE`
- `MASTER_ADDR`
- `MASTER_PORT`

Current policy:

- formal runs should use the activate script plus the unified pipeline commands
- short-run and full-run should differ by committed config or by the selected
  checkpoint only
- server-side ad hoc overrides belong to temporary validation docs, not here

## Experiment entries

| Group | EXP_TAG | EXP_VARIANT | FM_CONFIG | Activate | Purpose | Status |
| --- | --- | --- | --- | --- | --- | --- |
| E4 | e401a | `dgtd_v3_smoke` | `configs/experiment/dgtd_cifar10_v3_smoke.yaml` | `source scripts/experiments/activate_fm_cifar10.sh dgtd_v3_smoke e401a` | short-run online-mainline acceptance: verify training, checkpointing, online continuation source, sample, and eval all work end to end | ready |
| E4 | e402a | `dgtd_v3` | `configs/experiment/dgtd_cifar10_v3.yaml` | `source scripts/experiments/activate_fm_cifar10.sh dgtd_v3 e402a` | primary full run: online-mainline DGTD v3 main convergence run with higher-memory throughput-tuned training and few-step quality curve | ready |
| E4 | e403a | `dgtd_cifar10_v3_ablation_no_warp` | `configs/experiment/dgtd_cifar10_v3_ablation_no_warp.yaml` | `source scripts/experiments/activate_fm_cifar10.sh dgtd_cifar10_v3_ablation_no_warp e403a` | ablation: remove learned warp while keeping current online-mainline teacher path | planned |
| E4 | e404a | `dgtd_cifar10_v3_ablation_warp_no_hf` | `configs/experiment/dgtd_cifar10_v3_ablation_warp_no_hf.yaml` | `source scripts/experiments/activate_fm_cifar10.sh dgtd_cifar10_v3_ablation_warp_no_hf e404a` | ablation: keep warp but disable the HF-biased density contribution and metric emphasis path | planned |
| B0 | edm001 | `edm_cifar10_public_eval` | `configs/experiment/edm_cifar10_public_eval.yaml` | `source scripts/experiments/activate_fm_cifar10.sh edm_cifar10_public_eval edm001` | public EDM CIFAR-10 teacher/baseline inference and official FID with selectable sampler steps; uses NVLabs public EDM checkpoint by default | ready |

## Execution order

1. Run `e401a` first and confirm the online-mainline path is stable.
2. Launch `e402a` only after `e401a` shows nonzero `continuation_sources.online`
   and valid train/sample/eval outputs.
3. Use `e403a` and `e404a` only if the main full run needs controlled ablations.

## Public EDM CIFAR-10 baseline

This baseline has no training stage. Activation sets the normal experiment
variables plus:

- `EDM_CIFAR10_NETWORK`
  - default:
    `https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl`
- `EDM_CIFAR10_FID_REF`
  - default:
    `https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz`
- `DNNLIB_CACHE_DIR`
  - default: `$TORCH_HOME/dnnlib`

Activate:

```bash
source scripts/experiments/activate_fm_cifar10.sh edm_cifar10_public_eval edm001
```

Run inference and FID for any step list:

```bash
CUDA_VISIBLE_DEVICES=${INFER_CUDA_VISIBLE_DEVICES} python scripts/run_edm_cifar10_eval.py \
  --config $FM_CONFIG \
  --sample-root $SAMPLE_ROOT \
  --eval-root $METRIC_ROOT \
  --steps 1 2 4 8 18 \
  2>&1 | tee $METRIC_ROOT/edm_eval.stdout_stderr.txt
```

For a quick sample-only smoke, generate a small visual subset and skip FID:

```bash
CUDA_VISIBLE_DEVICES=${INFER_CUDA_VISIBLE_DEVICES} python scripts/run_edm_cifar10_eval.py \
  --config $FM_CONFIG \
  --sample-root $SAMPLE_ROOT \
  --eval-root $METRIC_ROOT \
  --steps 18 \
  --num-samples 64 \
  --batch 64 \
  --skip-fid \
  2>&1 | tee $METRIC_ROOT/edm_sample64_steps18.stdout_stderr.txt
```

For a quick sample + FID plumbing smoke, reduce sample count. This verifies the
sample/eval path but is not comparable to the paper FID because official EDM
FID uses 50,000 generated images:

```bash
CUDA_VISIBLE_DEVICES=${INFER_CUDA_VISIBLE_DEVICES} python scripts/run_edm_cifar10_eval.py \
  --config $FM_CONFIG \
  --sample-root $SAMPLE_ROOT \
  --eval-root $METRIC_ROOT \
  --steps 18 \
  --num-samples 512 \
  --batch 64 \
  2>&1 | tee $METRIC_ROOT/edm_eval512_steps18.stdout_stderr.txt
```

For the paper-comparable CIFAR-10 EDM baseline, use `--steps 18`, which is
NFE 35 under the Heun EDM sampler, and keep 50,000 images:

```bash
CUDA_VISIBLE_DEVICES=${INFER_CUDA_VISIBLE_DEVICES} python scripts/run_edm_cifar10_eval.py \
  --config $FM_CONFIG \
  --sample-root $SAMPLE_ROOT \
  --eval-root $METRIC_ROOT \
  --steps 18 \
  --num-samples 50000 \
  --batch 64 \
  2>&1 | tee $METRIC_ROOT/edm_eval50000_steps18.stdout_stderr.txt
```

Return:

- `$METRIC_ROOT/edm_eval.stdout_stderr.txt`
- `$METRIC_ROOT/reports/summary.json`
- `$METRIC_ROOT/reports/summary.csv`
- each `$METRIC_ROOT/steps{K}/metrics.json`
- sample file listing under `$SAMPLE_ROOT/steps{K}/images`

## Return fields for each experiment

For every formal run, return:

- run identity:
  - `date`
  - `commit`
  - `branch`
  - `EXP_VARIANT`
  - `EXP_TAG`
  - `FM_CONFIG`
  - `RUN_ROOT`
  - `gpu_count`
  - `gpu_type`
- training evidence:
  - `train.stdout_stderr.txt`
  - `tail -n 10 $LOG_ROOT/train.jsonl`
  - checkpoint file list under `$CKPT_DIR`
- sampling / eval evidence:
  - sample file list under `$SAMPLE_ROOT`
  - eval file list under `$METRIC_ROOT`
  - `$METRIC_ROOT/reports/summary.json`
  - `$METRIC_ROOT/reports/summary.csv`
  - each `steps{K}/metrics.json`

## Primary decision fields

The returned `train.jsonl` excerpt must expose these keys using their exact
current names:

- `train_loss`, `val_loss`
- `train_defect`, `val_defect`
- `continuation_sources`
- `online_anchor_used_rate`
- `online_continuation_rate`
- `cached_fallback_rate`
- `train_direct_teacher_error`
- `train_direct_bridge_gap`
- `train_bridge_state_teacher_error`
- `train_bridge_u_teacher_error`
- `train_teacher_rel_error_mean`
- `alpha_online_mean`
- `alpha_online_min`
- `alpha_online_max`
- `eta`
- `beta`
- `stage`
- `entropy_q_phi`
- `kl_qD_qphi`
- `max_qphi_over_qbase`
- `q_phi`
- `q_D`
- `D_bar`
- `K_bar`
- `HF_bar`
- `time_grid`

For performance profiling, return when available:

- `images_per_sec` or `samples_per_sec`
- `seconds_per_epoch`
- `max_gpu_memory_mb`
- `train_online_teacher_traj_sec`
- total wall-clock time

## Quick judgment checklist

Fill this after each experiment:

- `online_continuation_primary`: yes / no
- `cached_fallback_dominant`: yes / no
- `checkpoint_write_ok`: yes / no
- `q_phi_collapse_seen`: yes / no
- `clean_end_diag_abnormal`: yes / no
- `1_to_2_step_improves`: yes / no
- `2_to_4_step_improves_or_stable`: yes / no
- `4_to_8_step_improves_or_stable`: yes / no
- `need_patch`: yes / no
- `need_ablation`: yes / no
- `ready_for_next_stage`: yes / no
