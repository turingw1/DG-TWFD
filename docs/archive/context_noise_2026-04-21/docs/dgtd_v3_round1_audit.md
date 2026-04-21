# DGTD v3 Round-1 Audit

This document audits the current `DG_TWFD_v3` implementation against
[ARCHITECTURE_AND_IMPLEMENTATION.md](/home/gzwlinux/vscode/gitProject/DG-TWFD/docs/experiments/DG_TWFD_v3/ARCHITECTURE_AND_IMPLEMENTATION.md).

Scope:

- read-only implementation audit
- no training-logic modification
- conclusions must be backed by code or existing artifacts
- where evidence is missing, this document states that explicitly

Primary code files audited:

- `src/dgtd/defect.py`
- `src/dgtd/teacher.py`
- `src/dgtd/metrics.py`
- `src/dgtd/train_dgtd.py`
- `src/dgtd/warp.py`
- `src/dgtd/cache.py`
- `src/dgtd/sample_dgtd.py`
- `src/dgfm/models/map.py`
- `src/dgfm/schedulers/timewarp.py`
- `configs/experiment/dgtd_cifar10_v3.yaml`
- `configs/experiment/dgtd_cifar10_v3_smoke.yaml`
- `configs/model/map_unet.yaml`
- `configs/target/teacher_trajectory.yaml`
- `scripts/run_train.py`
- `scripts/run_sample_dgtd.py`
- `scripts/run_eval.py`

Artifact check performed:

- repo-local `train.jsonl` files were scanned
- repo-local `summary.json` files were scanned
- repo-local `last.pt` / `best.pt` checkpoints were checked only for top-level key structure

Artifact result:

- no repo-local checkpoint with DGTD-specific keys `timewarp`, `dgtd_q_target`, or
  `dgtd_stats` was found
- the repo-local `checkpoints/best.pt` is a legacy-format checkpoint with keys
  `artifacts/cfg_name/models/optimizers/scaler/scheduler/state`, not the current
  `DGTDTrainer` checkpoint schema
- existing `train.jsonl` files under `outputs/debug/runs/*` are older smoke runs and
  do not contain DGTD fields such as `q_phi`, `q_D`, `stage`, `eta`, or `beta`
- therefore, most runtime conclusions below are code-backed, not log-backed

## 1. Current-Implementation vs Spec Consistency Check

Reference spec:

- `docs/experiments/DG_TWFD_v3/ARCHITECTURE_AND_IMPLEMENTATION.md`

### 1.1 Consistent with the spec

1. Time semantics are consistent.
   `src/dgtd/defect.py::compute_dgtd_residual` and
   `src/dgtd/train_dgtd.py::_run_epoch` enforce `t < s < u`, matching the
   architecture document and the current `dgfm` convention.

2. The active teacher path is offline cache first, online teacher disabled by config.
   `configs/experiment/dgtd_cifar10_v3.yaml` sets
   `dgtd.disable_online_teacher: true`, and
   `src/dgtd/teacher.py::build_teacher_adapter` only builds an online teacher when
   that flag is false.

3. The student backbone is the explicit map U-Net described in the spec.
   `configs/model/map_unet.yaml` selects `model.family: official_map_unet`, and
   `src/dgfm/models/map.py::OfficialExplicitMapUNet` implements explicit map
   prediction `M_theta(x_t, t, s) -> x_s`.

4. The warp implementation matches the documented `q_phi` density parameterization.
   `src/dgtd/warp.py::MonotoneDensityWarp` uses `softplus(density_raw) + eps`,
   normalizes it, and exposes both CDF and inverse-CDF sampling utilities.

5. The defect target density formula matches the documented high-level formula.
   `src/dgtd/defect.py::build_target_density` forms
   `q_D ∝ (A + eps)^beta * q_base`, then normalizes.

6. The current sampling path is exactly the documented `Mode A`, and `Mode B`
   remains a stub.
   `src/dgtd/sample_dgtd.py` and `scripts/run_sample_dgtd.py` implement
   `mode_a`, while `mode_b_stub` writes a TODO JSON payload only.

### 1.2 Partially consistent or oversimplified in the spec

1. The spec’s high-level target description is slightly too coarse.
   The architecture doc summarizes target construction as “teacher continuation if
   available, otherwise stopgrad student bootstrap.” The exact code in
   `src/dgtd/defect.py:33-38` is:

   - `teacher_cont is None`:
     `target = self_cont`
   - `teacher_cont is not None`:
     `target = eta * teacher_cont.detach() + (1 - eta) * self_cont`

   The `eta`-mixture with `self_cont` is real and should be treated as part of the
   current objective, not as a minor detail.

2. The spec states that cached continuation is used “if possible,” but does not
   spell out the exact gate.
   The actual cached teacher gate in `src/dgtd/teacher.py:36-43` requires:

   - a trajectory batch is present
   - cached `x_s` and `x_u` can be interpolated
   - `((z.detach() - x_s_cached)^2 / ||x_s_cached||^2) <= proximity_rtol`
     for every sample in the batch
   - the batch-level predicate is `near_mask.all()`, not per-sample acceptance

   This is an important implementation detail omitted by the spec.

3. The spec describes DGTD defect statistics at a high level, but the actual code
   computes `sample_defect` using only `metric_value / ||x_u_teacher - x_t||^2`.
   `src/dgtd/defect.py:49-59` explicitly discards the residual argument with
   `del R`. This is a stronger simplification than the document currently makes
   explicit.

### 1.3 Important implementation details not covered clearly enough in the spec

1. `x_s_pred` is used as a control signal, but under the current config it does not
   receive useful gradient. This is not just a subtlety; it changes what the
   objective is really training. See Section 2.

2. Teacher cached continuation is batch-gated with `near_mask.all()`.
   This makes teacher-anchor usage much rarer than a per-sample cache gate would.

3. The warp triplet proposal is fixed and hand-designed in `r`-space.
   `src/dgtd/warp.py:100-131` samples three interval buckets:

   - short: `[1/64, 1/16]` with probability `0.5`
   - medium: `[1/16, 1/4]` with probability `0.3`
   - long: `[1/4, 1]` with probability `0.2`

   This proposal distribution is operationally important and should be treated as
   part of the algorithm definition.

4. Curvature is a finite-difference proxy over cached image-space states.
   `src/dgtd/cache.py:33-40` computes curvature from first-order velocity
   differences, anchored at interior points only.

5. There is no runtime proof in the repository that the current DGTD v3 code path
   has been executed end-to-end.
   Existing local logs/checkpoints are older smoke artifacts and do not confirm the
   current implementation behavior.

## 2. Gradient Path Audit

This section answers the exact gradient questions for the current code path in
`src/dgtd/defect.py` and `src/dgtd/train_dgtd.py`.

### 2.1 Tensor-by-tensor gradient status

Inside `src/dgtd/defect.py::compute_dgtd_residual`:

1. `x_s_pred = student(x_t, t, s, ...)`
   - requires grad at creation time
   - but its downstream uses cut off the gradient path in the current config

2. `x_u_direct = student(x_t, t, u, ...)`
   - requires grad
   - this is the branch that directly feeds the residual and loss

3. `teacher_cont = teacher_adapter.local_flow(..., x_s_pred, ...)`
   - cached path uses `z.detach()` inside `TeacherAdapter.local_flow`
   - cached return is `x_u_cached.detach()`
   - online path is disabled by config
   - effective current result: no gradient from `teacher_cont`

4. `self_cont = student(x_s_pred.detach(), s, u, ...).detach()`
   - input is detached
   - output is detached
   - no gradient reaches either the bridge call or `x_s_pred`

5. `target`
   - always detached in practice under current config
   - when teacher cache hits:
     `eta * teacher_cont.detach() + (1 - eta) * self_cont`
   - when teacher cache misses:
     `self_cont`

6. `residual = x_u_direct - target`
   - only `x_u_direct` contributes gradient

7. `loss = mean(omega.detach() * metric_norm(residual, u, ...))`
   - `omega` is detached in `src/dgtd/train_dgtd.py:288`
   - metric differentiates through `residual`
   - therefore gradients flow to `x_u_direct` only

### 2.2 Pseudo computation graph

Current graph under `disable_online_teacher=true`:

```text
x_t
 ├─> x_s_pred = M_theta(x_t, t, s)
 │    ├─> local_flow(..., z=x_s_pred)
 │    │    └─ uses z.detach(), returns cached x_u.detach() or None
 │    └─> M_theta(x_s_pred.detach(), s, u).detach() = self_cont
 │
 └─> x_u_direct = M_theta(x_t, t, u)

target =
  if cached teacher hit:
    eta * teacher_cont.detach() + (1 - eta) * self_cont
  else:
    self_cont

residual = x_u_direct - target
metric = metric_norm(residual, u, ...)
loss = mean(omega.detach() * metric)
```

Backward path:

```text
loss
 └─> metric
     └─> residual
         └─> x_u_direct
             └─> M_theta(x_t, t, u)
```

No current backward path:

```text
loss -X-> target
loss -X-> self_cont
loss -X-> teacher_cont
loss -X-> x_s_pred
```

### 2.3 Direct answers

1. Which tensors carry gradient?
   In the current DGTD objective, the effective gradient-carrying branch is
   `x_u_direct = M_theta(x_t, t, u)`.

2. Which tensors are detached?
   - `teacher_adapter.local_flow` uses `z.detach()`
   - cached teacher continuation returns detached cached state
   - `x_s_pred.detach()` is used as input to the bridge call
   - `self_cont` is detached again at output
   - `omega` is detached before multiplying the metric

3. Does `x_s_pred` get substantive gradient?
   No, not under the current config.

4. Is the current loss effectively training only `x_u_direct = M_theta(x_t, t, u)`?
   Yes.

This is the single most important implementation fact in the current branch.

## 3. Teacher Continuation Availability Audit

### 3.1 Source taxonomy in the current code

The code currently permits four conceptual continuation states:

1. `cached_exact`
   Meaning in current implementation:
   `TeacherAdapter.local_flow` returns cached `x_u_cached.detach()` when the cached
   state at `s` is deemed close enough to `x_s_pred`.

2. `online`
   `TeacherAdapter.local_flow` delegates to `online_teacher.local_flow(...)` if an
   online teacher exists and the cached branch did not return.

3. `bootstrap`
   `teacher_cont is None`, so `compute_dgtd_residual` falls back to
   `self_cont = stopgrad(M_theta(x_s_pred, s, u))`.

4. `none`
   Raw `teacher_cont` is `None` before bootstrap fallback is applied.

### 3.2 Exact cached-continuation hit condition

Under `disable_online_teacher=true`, `teacher_cont` can only come from the cached
path in `src/dgtd/teacher.py:36-43`.

The exact condition is:

1. `traj is not None`
2. cached states can be interpolated at `s` and `u`
3. compute per-sample relative error:

   `rel = mean((z.detach() - x_s_cached)^2)`

   `denom = mean(x_s_cached^2)`

   `near_mask = (rel / denom) <= teacher_proximity_rtol`

4. accept the cached continuation only if `near_mask.all()` is true for the whole
   batch

Current threshold:

- `teacher_proximity_rtol = 0.05` from
  `configs/experiment/dgtd_cifar10_v3.yaml`

### 3.3 Is cached continuation likely common?

There is no instrumentation in current logs to measure:

- `cached_exact` hit rate
- `near_mask.mean()`
- `near_mask.all()` frequency
- source proportions per batch or per epoch

Therefore the repository does not contain direct empirical proof.

Code-based expectation:

1. Early training, cached continuation is likely very rare.
   `x_s_pred` comes from an untrained or partially trained student, while the gate
   compares it against the cached teacher state.

2. The gate is stricter than it first appears because it is batch-level.
   Even if several samples in a batch are close enough, a single miss causes the
   entire batch to fall back to bootstrap.

3. Later in training, hit rate may improve, but the current repository provides no
   evidence that it actually does.

Conclusion:

- `online`: currently disabled
- `cached_exact`: possible in code, but probably sparse in practice
- `bootstrap`: almost certainly the dominant source unless instrumentation proves
  otherwise
- `none`: transient internal state only

Status:

- evidence insufficient from current logs
- new instrumentation is required

## 4. Sigma Semantics Audit

### 4.1 Where sigma is defined

There are three relevant sigma definitions in the current code:

1. `src/dgtd/teacher.py:48-50`
   `TeacherAdapter.sigma(t) = clamp(1 - t, min=sigma_min)`

2. `src/dgtd/metrics.py:24-30`
   `_resolve_sigma(t_or_sigma, sigma_fn)` uses the passed `sigma_fn`, otherwise
   falls back to `1 - t`

3. `src/dgfm/models/map.py:91-99`
   `OfficialExplicitMapUNet._noise_level(t) = clamp(1 - t, min=sigma_min)`

### 4.2 Do the major components share the same sigma system?

Yes, internally they do.

1. `metric_norm` uses `sigma_fn=teacher_adapter.sigma` in
   `src/dgtd/train_dgtd.py:267-274`

2. `edm_weight` uses the same `sigma_fn=teacher_adapter.sigma` in
   `src/dgtd/train_dgtd.py:276-280`

3. `min_snr_weight` uses the same `sigma_fn=teacher_adapter.sigma` in
   `src/dgtd/train_dgtd.py:281-285`

4. `lambda_hf(u)` inside `metric_norm` is therefore also driven by the same sigma
   system

5. model preconditioning uses `OfficialExplicitMapUNet._noise_level(t)` in
   `src/dgfm/models/map.py:124-137`, which is again `1 - t`

### 4.3 Is there a `sigma = 1 - t` approximation?

Yes, and in the current branch it is more than an approximation inside training
code: it is the actual sigma definition used by both weighting and
preconditioning.

### 4.4 Is this sigma logic consistent with the cache teacher scheduler?

This is the subtle point.

Internal DGTD consistency:

- yes, the model and loss use the same sigma system

Consistency with the scheduler that generated the cached teacher trajectories:

- not proven

Reason:

1. the cache-generation config in `configs/teacher/sampler.yaml` points to a
   diffusers DDPM teacher with DDIM sampling and `num_inference_steps: 128`

2. the training code never imports or reconstructs the teacher scheduler’s exact
   alpha/sigma table

3. therefore the current branch assumes `sigma(t) = 1 - t` as a surrogate time-noise
   mapping over cached trajectories

Conclusion:

- model preconditioning, `edm_weight`, `min_snr_weight`, and `lambda_hf(u)` are
  internally aligned
- exact alignment with the original teacher scheduler is not guaranteed
- this is a material approximation, especially near the clean end

## 5. Cache and Time-Grid Audit

### 5.1 `t_grid` length

What the code enforces:

- no fixed length is enforced
- `TrajectoryCacheDataset` accepts any `t_grid` length as long as a valid sorted
  `x_grid` exists

What the provided cache-generation config suggests:

- `configs/teacher/sampler.yaml` sets `retain_num_points: 33`

Conclusion:

- expected cache length for the intended teacher pipeline is probably `33`
- current training code does not verify this
- current repository does not include accessible teacher shards, so this cannot be
  confirmed from data

### 5.2 Uniform vs non-uniform `t_grid`

The code supports non-uniform grids.

Evidence:

- `src/dgtd/cache.py::interpolate_state` and `interpolate_curvature` use the actual
  stored times with piecewise-linear interpolation
- no uniform-grid assumption is hard-coded anywhere in cache loading or training

Actual shard spacing:

- cannot be confirmed from the current repository because the configured
  `TRAJ_ROOT` default path was not present on this machine during audit

### 5.3 Clean-end densification

Code evidence:

- none

Data evidence:

- unavailable from the current repository

Conclusion:

- evidence insufficient
- current code would allow clean-end densification because interpolation is
  time-aware, but the repository does not prove whether current teacher shards have
  it

### 5.4 State space: pixel vs latent

Current trajectory states are image-space tensors, not latent-space tensors.

Evidence:

1. `src/dgtd/cache.py:22-30` expects `x_grid` to be `[M, C, H, W]`
2. the dataset config used by current experiments is CIFAR-10 image-space
3. no encoder/decoder or latent model is involved in this DGTD path

### 5.5 Curvature proxy and anchor points

Current proxy in `src/dgtd/cache.py:33-40`:

1. compute finite-difference velocities between consecutive cached states
2. compute `delta_v` between adjacent velocities
3. return `delta_v / vel_norm`

Anchor positions:

- interior points only, aligned to `times[1:-1]`
- no curvature value exists at the two endpoints

Interpolation:

- `src/dgtd/cache.py:172-199` linearly interpolates curvature over those interior
  anchors
- queries outside the anchor range are clamped to the first or last curvature value

### 5.6 Does linear interpolation risk clean-end supervision error?

Yes.

Reasons:

1. `interpolate_state` is piecewise-linear over cached states
2. if the teacher trajectory is nonlinear in image space near the clean end, the
   interpolation error can be largest where intervals are sparse
3. `TeacherAdapter.local_flow` and `x_u_teacher` both rely on those interpolated
   states
4. the repository does not contain the actual shard spacing, so the magnitude of
   the effect is unknown

Conclusion:

- linear interpolation can introduce supervision error near the clean end
- current repository does not provide enough evidence to quantify it

## 6. Defect Density Audit

### 6.1 Exact `q_D` formula

In `src/dgtd/defect.py:85-105`:

1. normalize `q_base`
2. normalize each statistic independently:

   - `D_norm = normalize(D_bar)`
   - `K_norm = normalize(K_bar)`
   - `HF_norm = normalize(HF_bar)`

3. combine:

   `A = D_norm + curvature_weight * K_norm + hf_weight * HF_norm`

4. apply exponent and base prior:

   `q_D = (A + eps)^beta * q_base`

5. normalize `q_D`

Then, in `src/dgtd/train_dgtd.py:317-332`:

6. smooth `q_D` with kernel `[0.25, 0.5, 0.25]`
7. optionally flatten toward `q_base` in the flatten stage

### 6.2 How are `D_bar`, `K_bar`, and `HF_bar` formed?

1. `D_bar`
   - updated from `sample_defect`
   - `sample_defect = metric_value / ||x_u_teacher - x_t||^2`
   - `R` itself is not used directly in `compute_sample_defect`

2. `K_bar`
   - updated from interpolated curvature values from the cache

3. `HF_bar`
   - updated from `high_frequency_norm(laplacian_filter(residual.detach()))`

All three are:

- aggregated into bins by `u`
- averaged per observed bin on the current step
- then EMA-updated via `update_ema_bins`

### 6.3 Is `HF_bar` double-counting with the metric HF term?

Potentially yes.

Current loss path:

- `metric_norm` already adds a low-sigma high-frequency residual term to the
  training objective

Current density-adaptation path:

- `HF_bar` separately records the Laplacian energy of the detached residual and
  contributes to `q_D`

Interpretation:

- this is not identical double-use of the same tensor in the same formula
- but it is a two-level emphasis on high-frequency clean-end error
- therefore it can over-concentrate both optimization and sampling density toward
  clean high-frequency regions

### 6.4 Can `q_phi` collapse toward the clean end?

Yes, in principle.

Why:

1. the base density is already non-uniform under `base_density: logit_normal`
2. low-sigma regions receive extra HF emphasis in the metric
3. low-sigma residual HF also contributes to `HF_bar`, which sharpens `q_D`
4. there is no explicit entropy regularizer or ratio cap

What counters this:

1. `q_D` is smoothed
2. the flatten stage mixes `q_D` back toward `q_base`
3. `q_phi` is optimized by KL to `q_D`, not by unconstrained sharpening

Conclusion:

- collapse is possible
- the current codebase does not log the right diagnostics to know whether it is
  happening

### 6.5 What diagnostics are currently logged?

Logged in `src/dgtd/train_dgtd.py:358-397` and then written to `train.jsonl`:

- `loss`
- `defect`
- `hf`
- `curvature`
- `low_sigma_hf`
- `omega`
- `warp_loss`
- `stage`
- `eta`
- `beta`
- `q_phi`
- `q_D`
- `D_bar`
- `K_bar`
- `HF_bar`
- `time_grid`

Not currently logged:

- `entropy(q_phi)`
- exact per-update `KL(q_D || q_phi)` as a stable diagnostic
- `max(q_phi / q_base)`
- teacher continuation source ratios
- direct teacher error
- bridge teacher error
- direct/bridge gap
- noisy/mid/clean bucket summaries

Important nuance:

- `warp_loss` is only updated on warp-update steps, then divided by
  `batches_seen`; it is therefore a diluted epoch-average, not a clean per-update
  KL diagnostic

## 7. Minimal Instrumentation Plan

This section proposes the minimum extra observability needed before large runs.
It does not change the training objective itself.

### 7.1 Teacher continuation source proportions

Add counters per batch and per epoch:

- `teacher_source_cached`
- `teacher_source_online`
- `teacher_source_bootstrap`
- `teacher_source_none`

Also add:

- `teacher_near_mask_mean`
- `teacher_near_mask_all`
- `teacher_rel_error_mean`

Best insertion point:

- `src/dgtd/teacher.py::local_flow`
- `src/dgtd/defect.py::compute_dgtd_residual`

### 7.2 Direct teacher error

Add a detached metric comparing the direct branch to the cached teacher target:

- `direct_teacher_error_l2 = mean((x_u_direct.detach() - x_u_teacher)^2)`

Optional companion:

- `direct_teacher_error_hf`

Best insertion point:

- after `x_u_teacher = interpolate_state(batch, u)` in
  `src/dgtd/train_dgtd.py::_run_epoch`

### 7.3 Bridge teacher error

Add:

- `x_s_teacher = interpolate_state(batch, s)`
- `bridge_teacher_error_l2 = mean((x_s_pred.detach() - x_s_teacher)^2)`

This is required to know whether the bridge branch is learning anything useful,
even if it is not currently receiving gradient.

### 7.4 Direct vs bridge gap

Add:

- `direct_bridge_gap_l2 = mean((x_u_direct.detach() - self_cont)^2)`

This measures semigroup inconsistency directly.

### 7.5 Warp diagnostics

Add exact epoch diagnostics:

- `entropy_q_phi = -sum(q_phi * log q_phi)`
- `kl_qD_qphi = KL(q_D || q_phi)` without batch averaging dilution
- `max_qphi_over_qbase`
- `min_qphi`
- `argmax_qphi_bin`

### 7.6 Region-wise summaries

Bucket `u` into at least three regions:

- noisy
- mid
- clean

For each region log:

- defect mean
- HF mean
- direct teacher error mean
- bridge teacher error mean
- direct/bridge gap mean

These summaries are much more actionable than only logging full per-bin arrays.

## 8. Blocking Issues Before Full-Scale Training

### P0: do not recommend full run before these are addressed

1. The current objective effectively trains only the direct branch
   `M_theta(x_t, t, u)`.
   The bridge path `x_s_pred` has no useful backward signal under the current
   config. If the intended DGTD formulation requires learning bridge composition,
   this is a core objective mismatch, not a cosmetic issue.

2. There is no instrumentation proving whether teacher continuation is actually
   being used.
   Because the cached gate is batch-level `near_mask.all()`, teacher-anchor usage
   may be near-zero in practice. A long run without measuring this is not
   defensible.

3. There is no repo-local DGTD v3 smoke artifact proving that the current code path
   has run successfully end-to-end with current logging fields and checkpoint
   schema.

### P1: high probability of affecting quality or efficiency

1. Sigma semantics are internally consistent but not proven consistent with the
   original teacher scheduler used to generate the cached trajectories.

2. High-frequency error is emphasized both in the loss and in density adaptation,
   which may over-bias `q_phi` toward clean high-frequency bins.

3. `compute_sample_defect` ignores `R` directly and reduces to
   `metric_value / ||x_u_teacher - x_t||^2`. This may be acceptable, but it is a
   strong modeling choice that has not been instrumented or justified empirically.

4. `warp_loss` is logged in a diluted way and cannot reliably diagnose warp
   adaptation quality.

### P2: can wait until after the main training path is verified

1. `Mode B` DP schedule is still a stub.

2. Cache `t_grid` length and spacing are not validated at load time.

3. Clean-end interpolation error risk is known conceptually but unmeasured.

## Final audit conclusion

The current `DG_TWFD_v3` code matches the broad architecture document in overall
topology:

- explicit map backbone
- offline cached trajectory teacher
- density-based learnable warp
- DGTD-style residual and density adaptation

However, the most important implementation reality is stronger than the current
spec wording:

1. under the current configuration, the loss effectively optimizes only the direct
   branch `M_theta(x_t, t, u)`
2. cached teacher continuation is guarded by a strict batch-level acceptance test
3. the repository does not currently contain the runtime instrumentation needed to
   prove that teacher anchoring and warp adaptation are working as intended

That means the branch is structurally aligned with the spec, but not yet
experiment-ready in a research-auditable sense.
