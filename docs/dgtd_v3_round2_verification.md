# DGTD v3 Round-2 Verification

## Verdict

Status: **不通过**

Reason:

1. round-2 的核心静态改动大部分已经落到代码里
2. 但训练 smoke 目前在进入一轮 epoch 之前就会失败，无法产出 `last.pt` 和 `train.jsonl`
3. 因此当前 patch 还不具备“可进入下一轮 full training”的基本 runtime 条件

## Scope

Read and checked:

- `docs/dgtd_v3_round2_patch_notes.md`
- `src/dgtd/defect.py`
- `src/dgtd/teacher.py`
- `src/dgtd/sigma.py`
- `src/dgtd/metrics.py`
- `src/dgtd/train_dgtd.py`
- `src/dgtd/warp.py`
- `src/dgtd/sample_dgtd.py`
- `configs/experiment/dgtd_cifar10_v3.yaml`
- `configs/experiment/dgtd_cifar10_v3_smoke.yaml`
- `tests/test_dgtd.py`

Runtime attempts:

- requested train smoke
- requested sample smoke
- requested eval smoke
- one additional synthetic-checkpoint sample/eval sanity check, only because train smoke failed before checkpoint creation

## 1. Gradient-Path Verification

### 1.1 Code-path result

`compute_dgtd_residual()` now builds:

- `x_s_pred = student(x_t, t, s)` at [src/dgtd/defect.py](/home/gzwlinux/vscode/gitProject/DG-TWFD/src/dgtd/defect.py:26)
- `x_u_direct = student(x_t, t, u)` at [src/dgtd/defect.py](/home/gzwlinux/vscode/gitProject/DG-TWFD/src/dgtd/defect.py:27)
- `bridge_cont = student(x_s_pred, s, u)` at [src/dgtd/defect.py](/home/gzwlinux/vscode/gitProject/DG-TWFD/src/dgtd/defect.py:35)
- `x_u_cont = eta * teacher_cont + (1 - eta) * bridge_cont` when teacher continuation exists at [src/dgtd/defect.py](/home/gzwlinux/vscode/gitProject/DG-TWFD/src/dgtd/defect.py:42)

Trainer side now uses symmetric half-stopgrad:

- `direct_residual = x_u_direct - x_u_cont.detach()` at [src/dgtd/train_dgtd.py](/home/gzwlinux/vscode/gitProject/DG-TWFD/src/dgtd/train_dgtd.py:320)
- `bridge_residual = x_u_cont - x_u_direct.detach()` at [src/dgtd/train_dgtd.py](/home/gzwlinux/vscode/gitProject/DG-TWFD/src/dgtd/train_dgtd.py:321)
- `metric_value = 0.5 * (metric_direct + metric_bridge)` at [src/dgtd/train_dgtd.py](/home/gzwlinux/vscode/gitProject/DG-TWFD/src/dgtd/train_dgtd.py:338)

Pseudo graph:

```text
x_t
|- M(x_t,t,u) --------------------------> term_direct
\- M(x_t,t,s)=x_s_pred
   \- continuation(x_s_pred,s,u)=x_u_cont -> term_bridge
```

This is no longer the old direct-only formulation.

### 1.2 Teacher detach status

Teacher continuation remains detached for online and cached-exact paths:

- online path uses `with torch.no_grad()` and `z.detach()` at [src/dgtd/teacher.py](/home/gzwlinux/vscode/gitProject/DG-TWFD/src/dgtd/teacher.py:71)
- cached exact uses `x_u_cached.detach()` at [src/dgtd/teacher.py](/home/gzwlinux/vscode/gitProject/DG-TWFD/src/dgtd/teacher.py:91)

Cached affine is intentionally teacher-anchored but not fully detached:

- `x_u_cached.detach() + alpha * (z - x_s_cached.detach())` at [src/dgtd/teacher.py](/home/gzwlinux/vscode/gitProject/DG-TWFD/src/dgtd/teacher.py:100)

This means teacher anchors stay frozen, while `z=x_s_pred` can still receive gradient through the affine surrogate.

### 1.3 Autograd spot check

I ran a local toy autograd check against the actual loss structure. Result:

- `cached_affine` with `eta=1.0`: `x_s_pred.grad` was non-zero
- `cached_exact` with `eta=1.0`: `x_s_pred.grad` was zero, `x_u_direct.grad` was non-zero
- `cached_exact` with `eta=0.4`: `x_s_pred.grad` became non-zero again because `(1-eta) * bridge_cont` re-enters `x_u_cont`

So the precise conclusion is:

- direct branch definitely gets gradient now
- continuation branch gets gradient when `x_u_cont` still depends on student output
- this is true for `cached_affine` immediately
- this is only conditionally true for `cached_exact` and `online`

Important caveat:

- warmup sets `eta=1.0` at [src/dgtd/train_dgtd.py](/home/gzwlinux/vscode/gitProject/DG-TWFD/src/dgtd/train_dgtd.py:130)
- therefore any sample landing on `cached_exact` or `online` during warmup still degenerates to direct-only

That means “gradient-path fix” is **partially correct but not unconditional**.

## 2. Teacher Continuation Verification

What is implemented:

- priority starts with online teacher when enabled at [src/dgtd/teacher.py](/home/gzwlinux/vscode/gitProject/DG-TWFD/src/dgtd/teacher.py:71)
- cache states come from `interpolate_state(traj, s/u)` at [src/dgtd/teacher.py](/home/gzwlinux/vscode/gitProject/DG-TWFD/src/dgtd/teacher.py:83)
- exact-cache gate uses per-sample `exact_mask = rel_error <= proximity_rtol` at [src/dgtd/teacher.py](/home/gzwlinux/vscode/gitProject/DG-TWFD/src/dgtd/teacher.py:88)
- affine surrogate uses cached `x_s_teacher` and `x_u_teacher` at [src/dgtd/teacher.py](/home/gzwlinux/vscode/gitProject/DG-TWFD/src/dgtd/teacher.py:100)
- source IDs are explicit: `online/cached_affine/cached_exact/bootstrap` at [src/dgtd/teacher.py](/home/gzwlinux/vscode/gitProject/DG-TWFD/src/dgtd/teacher.py:14)

This part matches the patch note intent.

One nuance:

- the actual implementation is not a global “online -> exact -> affine -> bootstrap” branch tree
- it is `online` first, then affine is built for the full batch, then exact samples overwrite affine samples in-place at [src/dgtd/teacher.py](/home/gzwlinux/vscode/gitProject/DG-TWFD/src/dgtd/teacher.py:102)

Functionally that still yields the intended per-sample priority.

## 3. Sigma Consistency Verification

Confirmed on the required DGTD components:

- teacher affine coefficient uses `self.sigma_schedule.alpha(s, u)` at [src/dgtd/teacher.py](/home/gzwlinux/vscode/gitProject/DG-TWFD/src/dgtd/teacher.py:89)
- `metric_norm()` takes `sigma_fn` and is called with `teacher_adapter.sigma` at [src/dgtd/train_dgtd.py](/home/gzwlinux/vscode/gitProject/DG-TWFD/src/dgtd/train_dgtd.py:325)
- `edm_weight()` uses the same `sigma_fn` at [src/dgtd/train_dgtd.py](/home/gzwlinux/vscode/gitProject/DG-TWFD/src/dgtd/train_dgtd.py:349)
- `min_snr_weight()` uses the same `sigma_fn` at [src/dgtd/train_dgtd.py](/home/gzwlinux/vscode/gitProject/DG-TWFD/src/dgtd/train_dgtd.py:354)
- `low_sigma_hf` segmentation also uses `teacher_adapter.sigma(u)` at [src/dgtd/train_dgtd.py](/home/gzwlinux/vscode/gitProject/DG-TWFD/src/dgtd/train_dgtd.py:426)

Config can switch sigma mode from one place:

- `dgtd.sigma_mode: linear_1mt` at [configs/experiment/dgtd_cifar10_v3.yaml](/home/gzwlinux/vscode/gitProject/DG-TWFD/configs/experiment/dgtd_cifar10_v3.yaml:56)

Remaining residue:

- `metrics._resolve_sigma()` still falls back to `1 - t` if `sigma_fn is None` at [src/dgtd/metrics.py](/home/gzwlinux/vscode/gitProject/DG-TWFD/src/dgtd/metrics.py:24)
- `OfficialExplicitMapUNet._noise_level()` still hardcodes `1 - time_value` at [src/dgfm/models/map.py](/home/gzwlinux/vscode/gitProject/DG-TWFD/src/dgfm/models/map.py:92)

For the four DGTD items requested in this audit, the sigma path is unified. There is still legacy `1 - t` residue outside that exact scope.

## 4. q_D and Warp Stability Verification

Confirmed:

- default `q_D` no longer needs `HF_bar`; config disables it via `qd_use_hf_bar: false` and `qd_hf_weight: 0.0` at [configs/experiment/dgtd_cifar10_v3.yaml](/home/gzwlinux/vscode/gitProject/DG-TWFD/configs/experiment/dgtd_cifar10_v3.yaml:77)
- target density core is now `normalize(D_bar) + curvature_weight * normalize(K_bar)` at [src/dgtd/defect.py](/home/gzwlinux/vscode/gitProject/DG-TWFD/src/dgtd/defect.py:140)
- ratio cap exists and is applied when `qd_ratio_cap > 0` at [src/dgtd/defect.py](/home/gzwlinux/vscode/gitProject/DG-TWFD/src/dgtd/defect.py:146)

New diagnostics are indeed computed:

- `entropy_q_phi` at [src/dgtd/train_dgtd.py](/home/gzwlinux/vscode/gitProject/DG-TWFD/src/dgtd/train_dgtd.py:559)
- `kl_qD_qphi` at [src/dgtd/train_dgtd.py](/home/gzwlinux/vscode/gitProject/DG-TWFD/src/dgtd/train_dgtd.py:560)
- `max_qphi_over_qbase` at [src/dgtd/train_dgtd.py](/home/gzwlinux/vscode/gitProject/DG-TWFD/src/dgtd/train_dgtd.py:561)
- `argmax_q_phi` at [src/dgtd/train_dgtd.py](/home/gzwlinux/vscode/gitProject/DG-TWFD/src/dgtd/train_dgtd.py:562)

Test coverage exists for:

- affine fallback gradient at [tests/test_dgtd.py](/home/gzwlinux/vscode/gitProject/DG-TWFD/tests/test_dgtd.py:173)
- ratio cap at [tests/test_dgtd.py](/home/gzwlinux/vscode/gitProject/DG-TWFD/tests/test_dgtd.py:207)

## 5. Logging and Diagnostics Verification

The trainer really writes the new fields into the epoch payload, not just local variables.

Confirmed in final JSON payload:

- `train_direct_teacher_error` / `val_direct_teacher_error` at [src/dgtd/train_dgtd.py](/home/gzwlinux/vscode/gitProject/DG-TWFD/src/dgtd/train_dgtd.py:688)
- `train_bridge_teacher_error` / `val_bridge_teacher_error` at [src/dgtd/train_dgtd.py](/home/gzwlinux/vscode/gitProject/DG-TWFD/src/dgtd/train_dgtd.py:690)
- `train_direct_bridge_gap` / `val_direct_bridge_gap` at [src/dgtd/train_dgtd.py](/home/gzwlinux/vscode/gitProject/DG-TWFD/src/dgtd/train_dgtd.py:692)
- `continuation_sources` at [src/dgtd/train_dgtd.py](/home/gzwlinux/vscode/gitProject/DG-TWFD/src/dgtd/train_dgtd.py:715)
- `train_noisy_* / train_mid_* / train_clean_*` at [src/dgtd/train_dgtd.py](/home/gzwlinux/vscode/gitProject/DG-TWFD/src/dgtd/train_dgtd.py:716)
- warp diagnostics `q_phi/q_D/D_bar/K_bar/HF_bar/time_grid` at [src/dgtd/train_dgtd.py](/home/gzwlinux/vscode/gitProject/DG-TWFD/src/dgtd/train_dgtd.py:704)

One naming problem:

- `bridge_teacher_error` is computed as `MSE(x_s_pred, x_s_teacher)` at [src/dgtd/train_dgtd.py](/home/gzwlinux/vscode/gitProject/DG-TWFD/src/dgtd/train_dgtd.py:377)
- so it is not actually the teacher error of the bridge continuation at `u`
- the metric is still useful, but the name overstates what it measures

## 6. Smoke Runtime Validation

### 6.1 Requested train smoke

Requested command could not be run exactly as written because:

- `TRAJ_ROOT` was unset
- config fallback `/data2/yl7622/Zhengwei/DG-TWFD/teacher_traj/cifar10_ddpm128_p33` does not exist on this machine

I used repo-local cache instead:

```bash
python scripts/run_train.py \
  --config configs/experiment/dgtd_cifar10_v3_smoke.yaml \
  --run-root /tmp/dgtd_v3_round2_smoke \
  --set target.shard_root=data/teacher_shards/ddpm_cifar10_32 \
  --set dataset.data_root=outputs/debug/datasets/cifar10
```

First failure:

```text
ModuleNotFoundError: No module named 'models'
```

Cause:

- `ensure_flow_matching_image_models_on_path()` points at `root/flow_matching/examples/image`
- repo actually keeps this code under `refs/flow_matching/examples/image`
- see [src/dgfm/models/official_unet.py](/home/gzwlinux/vscode/gitProject/DG-TWFD/src/dgfm/models/official_unet.py:9)

After temporary `PYTHONPATH=refs/flow_matching/examples/image:src` workaround, train failed again:

```text
AttributeError: 'DGTDTrainer' object has no attribute 'stats'
```

Cause:

- `DGTDTrainer` is `@dataclass(slots=True)` at [src/dgtd/train_dgtd.py](/home/gzwlinux/vscode/gitProject/DG-TWFD/src/dgtd/train_dgtd.py:184)
- `_init_stats()` assigns `self.stats`, `self.q_base`, `self.q_target` without declaring them as dataclass fields at [src/dgtd/train_dgtd.py](/home/gzwlinux/vscode/gitProject/DG-TWFD/src/dgtd/train_dgtd.py:201)

Observed train artifacts:

- only `/tmp/dgtd_v3_round2_smoke/logs/config_resolved.yaml`
- no `checkpoints/last.pt`
- no `logs/train.jsonl`

### 6.2 Sample smoke

The requested sample smoke could not be run against `/tmp/dgtd_v3_round2_smoke/checkpoints/last.pt` because train never produced the checkpoint.

I still sanity-checked the sample path using a synthetic checkpoint containing:

- random initialized map model state
- valid `timewarp` state

Command:

```bash
PYTHONPATH=refs/flow_matching/examples/image:src python scripts/run_sample_dgtd.py \
  --config configs/experiment/dgtd_cifar10_v3_smoke.yaml \
  --checkpoint /tmp/dgtd_v3_round2_synth/checkpoints/last.pt \
  --output-dir /tmp/dgtd_v3_round2_synth/sample \
  --steps 4 \
  --set target.shard_root=data/teacher_shards/ddpm_cifar10_32 \
  --set dataset.data_root=outputs/debug/datasets/cifar10
```

Result:

- command completed successfully
- `warp.r_to_t()` path is exercised by `build_mode_a_time_grid()` at [src/dgtd/sample_dgtd.py](/home/gzwlinux/vscode/gitProject/DG-TWFD/src/dgtd/sample_dgtd.py:22)
- produced non-uniform time grid:
  `[0.0, 0.5258825421333313, 0.6777514815330505, 0.8017192482948303, 1.0]`

Produced files:

- `grid.png`
- `samples.pt`
- `time_grid.pt`
- `images/000000.png` through `images/000063.png`

### 6.3 Eval smoke

The requested eval smoke could not be run against the train smoke checkpoint because train failed.

I tried eval on the same synthetic checkpoint. It failed before model evaluation due missing FID weights:

```text
RuntimeError: Unable to download torch-fidelity Inception weights ...
```

This is an environment blocker, not a round-2 DGTD logic regression.

## 7. Acceptance Summary

### Most important positive results

1. The symmetric half-stopgrad residual is genuinely implemented, and `cached_affine` now propagates gradient into the continuation side.
2. Offline teacher continuation is materially improved: cache exact and cache affine are both present, with explicit source labeling.
3. Sigma usage is unified across the requested DGTD weighting path and teacher affine continuation, and the new diagnostics are actually written into the epoch payload.

### Most important remaining risks

1. **Train smoke is currently hard-blocked** by `DGTDTrainer(slots=True)` missing `stats/q_base/q_target` fields, so the patch is not runtime-ready.
2. **Warmup still has a direct-only corner case**: when source is `cached_exact` or `online` and `eta=1.0`, the continuation branch gets no student gradient.
3. `bridge_teacher_error` is misnamed: it measures `x_s_pred` vs `x_s_teacher`, not bridge continuation error at `u`, which weakens diagnosis quality.

## Recommendation

Recommendation: **不要进入 full run**

Required before full training:

1. fix the trainer runtime blockers so train smoke can actually finish
2. rerun train/sample/eval smoke on a real checkpoint and inspect the new `train.jsonl`
3. decide whether warmup should keep `eta=1.0`, given that exact/online samples still collapse to direct-only in that stage
