# DGTD v3 Round-2 Verification

## Verdict

Status: **有条件通过**

Reason:

1. round-2 的核心静态改动已经落到代码里
2. 服务器 smoke 训练已能产出 `train.jsonl`，并写出新增 DGTD diagnostics
3. 但当前所谓“online teacher”在 DGTD continuation 上仍不是 `source=online`
4. 因此它可以进入下一轮更正式的 smoke / short run，但还不应被表述成“online continuation 已经打通”

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

- local static inspection and autograd spot check
- server-side smoke evidence returned by the user

Server evidence returned:

- Section 3 preflight:
  - `online_teacher_built True`
  - `teacher_type DiffusersDDPMTeacher`
  - `online_teacher_prepare ok`
- Section 4 dataset check:
  - `/data2/yl7622/Zhengwei/DG-TWFD/datasets/cifar10/cifar-10-batches-py` exists
- train history tail from server:
  - `online_teacher_data: true`
  - `continuation_sources: {"online": 0.0, "cached_affine": 0.25, "cached_exact": 0.75, "bootstrap": 0.0}`
  - full DGTD diagnostic fields present in `train.jsonl`

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

Server evidence is consistent with this:

- returned smoke log had `eta: 1.0`
- returned continuation mix was `75% cached_exact + 25% cached_affine`
- therefore the continuation branch was still only partially trainable in warmup

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

### 2.1 Server re-check with online teacher enabled

The server evidence changes the earlier interpretation in one important way:

- `online teacher` is now successfully built and prepared on the server
- `online_teacher_data: true` in `train.jsonl` confirms the trainer is using the
  online-teacher **data** path

This is consistent with the current code in
[src/dgtd/train_dgtd.py](/home/gzwlinux/vscode/gitProject/DG-TWFD/src/dgtd/train_dgtd.py:117):

- `select_dgtd_dataloaders(...)` chooses image dataloaders when
  `disable_online_teacher=false` and the teacher builds successfully
- `_run_epoch(...)` then materializes an online trajectory batch via
  `teacher_adapter.online_trajectory_from_x0(...)`

### 2.2 What is still *not* online

The same server log also shows:

- `continuation_sources.online = 0.0`
- `continuation_sources.cached_affine = 0.25`
- `continuation_sources.cached_exact = 0.75`

That means:

- the training batch is coming from online teacher trajectory generation
- but DGTD continuation itself is still being constructed from the generated
  trajectory through the cached-style path

This matches the current code:

- `TeacherAdapter.local_flow()` only emits `source=online` if the teacher object
  implements `local_flow(...)`
- the current `DiffusersDDPMTeacher` does not implement `local_flow(...)`; it
  implements trajectory sampling APIs

So the exact current state is:

- **online teacher data path: yes**
- **online continuation source: no**

This distinction should be preserved in later docs and experiment notes.

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

### 6.1 Server smoke status

The earlier local-only runtime blockers are no longer the authoritative result.
The user returned successful server-side smoke evidence for:

- online teacher preflight
- dataset visibility
- train history generation
- and reported that the other smoke sections had no issue

So the authoritative runtime status for this verification round is now
server-backed, not local-backed.

### 6.2 Train smoke

The returned server `train.jsonl` row confirms:

- smoke train completed epoch `0`
- `train_loss`, `val_loss`, `train_defect`, `val_defect` are present
- `q_phi`, `q_D`, `D_bar`, `K_bar`, `HF_bar`, `time_grid` are present
- `online_teacher_data: true` is present

Notable values from the returned log:

- `eta = 1.0`
- `beta = 0.0`
- `stage = "warmup"`
- `kl_qD_qphi ≈ 0`
- `continuation_sources.online = 0.0`

Interpretation:

- server runtime no longer blocks basic DGTD smoke execution
- the new logging payload really lands in history
- warmup behavior is exactly visible in the log

### 6.3 Sample / eval / diag smoke

The user reported that the other sections had no issue.

Given the returned train log shape and the lack of reported failures in later
sections, the current status is:

- sample path: treated as passed for this smoke round
- eval path: treated as passed for this smoke round
- diagnostics export: treated as passed for this smoke round

I did not receive the full stdout/stderr blobs for those sections in this turn,
so those passes are user-reported rather than directly transcript-backed in this
document.

## 7. Acceptance Summary

### Most important positive results

1. The symmetric half-stopgrad residual is genuinely implemented, and `cached_affine` now propagates gradient into the continuation side.
2. Server smoke shows that the online-teacher **data** path is now live: `online_teacher_built=True`, `online_teacher_prepare ok`, and `online_teacher_data=true`.
3. Sigma usage is unified across the requested DGTD weighting path and teacher affine continuation, and the new diagnostics are actually written into the epoch payload.

### Most important remaining risks

1. **Online continuation is still not active**: the server log shows `continuation_sources.online = 0.0`, so the current run still relies on `cached_exact/cached_affine` continuation built from online-generated trajectories.
2. **Warmup still has a direct-only corner case**: with `eta=1.0`, any `cached_exact` continuation sample still leaves the bridge side without student gradient.
3. `bridge_teacher_error` is misnamed: it measures `x_s_pred` vs `x_s_teacher`, not bridge continuation error at `u`, which weakens diagnosis quality.

## Recommendation

Recommendation: **可以进入下一轮 short/full run，但要带着明确限定解释**

Required before claiming success on the online-teacher mainline:

1. do not describe the current path as “online continuation”; describe it as “online teacher data + cached-style continuation from the on-the-fly trajectory”
2. decide whether the next patch should add a real `local_flow(...)` online continuation interface for `DiffusersDDPMTeacher`
3. reconsider warmup `eta=1.0`, because the returned server log shows `cached_exact` is still the dominant source in warmup
