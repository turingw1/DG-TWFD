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
| E4 | e401b | `dgtd_v3_diag` | `configs/experiment/dgtd_cifar10_v3_diag.yaml` | `source scripts/experiments/activate_fm_cifar10.sh dgtd_v3_diag e401b` | fixed-budget diagnostic loop: 3 epochs, max 200 train batches, max 50 val batches, 512-sample eval, endpoint teacher report, gate analysis for `1 2 4 8` | ready |
| E4 | e402a | `dgtd_v3` | `configs/experiment/dgtd_cifar10_v3.yaml` | `source scripts/experiments/activate_fm_cifar10.sh dgtd_v3 e402a` | primary full run: online-mainline DGTD v3 main convergence run with higher-memory throughput-tuned training and few-step quality curve | ready |
| E4 | e403a | `dgtd_cifar10_v3_ablation_no_warp_diag` | `configs/experiment/dgtd_cifar10_v3_ablation_no_warp_diag.yaml` | `source scripts/experiments/activate_fm_cifar10.sh dgtd_cifar10_v3_ablation_no_warp_diag e403a` | Module D diagnostic ablation: same diag budget with learned warp disabled | ready |
| E4 | e403a | `dgtd_cifar10_v3_ablation_no_warp` | `configs/experiment/dgtd_cifar10_v3_ablation_no_warp.yaml` | `source scripts/experiments/activate_fm_cifar10.sh dgtd_cifar10_v3_ablation_no_warp e403a` | Module D full ablation: remove learned warp while keeping online-mainline teacher path | planned |
| E4 | e404a | `dgtd_cifar10_v3_ablation_warp_no_hf_diag` | `configs/experiment/dgtd_cifar10_v3_ablation_warp_no_hf_diag.yaml` | `source scripts/experiments/activate_fm_cifar10.sh dgtd_cifar10_v3_ablation_warp_no_hf_diag e404a` | Module E diagnostic ablation: same diag budget with HF path disabled | ready |
| E4 | e404a | `dgtd_cifar10_v3_ablation_warp_no_hf` | `configs/experiment/dgtd_cifar10_v3_ablation_warp_no_hf.yaml` | `source scripts/experiments/activate_fm_cifar10.sh dgtd_cifar10_v3_ablation_warp_no_hf e404a` | Module E full ablation: keep warp but disable the HF-biased density contribution and metric emphasis path | planned |
| E4 | e405b | `dgtd_cifar10_v3_probe_fast_teacher` | `configs/experiment/dgtd_cifar10_v3_probe_fast_teacher.yaml` | manual server command, same run/eval pipeline | fast-teacher warped probe after short-budget scheduler fix; tests whether the current objective can pass sample-quality gate under a practical online teacher budget | completed: gate failed on `sample_not_noise_like`, best approx FID@512 `373.26` at 2 steps |
| E4 | e406a | `dgtd_cifar10_v3_probe_fast_teacher_no_warp` | `configs/experiment/dgtd_cifar10_v3_probe_fast_teacher_no_warp.yaml` | manual server command, same run/eval pipeline | no-warp control for `e405b`; isolates whether learned density warp is causing the noise-like samples | completed: gate failed on `sample_not_noise_like`, best approx FID@512 `386.56` at 2 steps |
| E4 | e407a | `dgtd_cifar10_v3_probe_anchor1_long` | `configs/experiment/dgtd_cifar10_v3_probe_anchor1_long.yaml` | manual server command, same run/eval pipeline | longer stronger endpoint-anchor probe; tests whether objective stabilization alone produces image structure before full training | completed: gate failed on `sample_not_noise_like`, best approx FID@1024 `427.45` at 8 steps |
| E4 | oss001 | `dgtd_v3_oss_baseline` | `configs/experiment/dgtd_cifar10_v3_oss_baseline.yaml` | `source scripts/experiments/activate_fm_cifar10.sh dgtd_v3_oss_baseline oss001` | post-checkpoint optimal-steps baseline: search global schedules for steps `2 4 8`, compare OSS schedule vs learned-warp schedule on the same usable checkpoint | blocked until usable checkpoint |
| B0 | edm001 | `edm_cifar10_public_eval` | `configs/experiment/edm_cifar10_public_eval.yaml` | `source scripts/experiments/activate_fm_cifar10.sh edm_cifar10_public_eval edm001` | public EDM CIFAR-10 teacher/baseline inference and official FID with selectable sampler steps; uses NVLabs public EDM checkpoint by default | ready |

## Execution order

1. Gate 0: run `e401a` smoke only to validate train/checkpoint/sample/eval plumbing.
2. Gate 1: run `e401b` diag and collect `train.jsonl`, sample grids, 512-sample eval,
   teacher endpoint JSON, and `analysis_report.md/json`.
3. Gate 2: launch `e402a` only if `e401b` passes the diagnostic gate or has a
   documented, non-algorithmic failure.
4. If `e401b` fails, fix one module at a time and rerun the same diag budget
   with the same seed and step list.
5. Run `e403a` no-warp only when `q_phi` collapse or step-curve regression is
   the active hypothesis.
6. Run `e404a` no-HF only after image structure appears but texture/detail is
   the active hypothesis.
7. Run `oss001` only after `e402a` or a later checkpoint produces non-noise
   samples; it is a schedule baseline, not a rescue path for an invalid model.

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
- diagnostic evidence:
  - `$RUN_ROOT/teacher_endpoint_report.json`
  - `$RUN_ROOT/analysis_report.md`
  - `$RUN_ROOT/analysis_report.json`
  - `gate_verdict.status`, `gate_verdict.failed`, and `gate_verdict.unknown`
  - for `oss001`, all `oss_schedule_steps{K}.json` files and OSS/default comparison summaries

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
- `train_noisy_endpoint_error`
- `train_endpoint_anchor_loss`
- `val_endpoint_anchor_loss`
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
