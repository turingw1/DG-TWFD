# DGTD v3 Round-2 Patch Notes

This document records the minimal-but-critical DGTD v3 changes made in round 2.

Goals of this patch:

1. fix the direct-only gradient path in the unified DGTD residual
2. make offline teacher continuation available beyond exact cache hits
3. unify sigma usage across DGTD weighting and teacher continuation
4. add enough diagnostics to judge whether the new training path is working

## 1. Files changed

Core implementation:

- `src/dgtd/defect.py`
- `src/dgtd/teacher.py`
- `src/dgtd/metrics.py`
- `src/dgtd/train_dgtd.py`
- `src/dgtd/warp.py`
- `src/dgtd/sigma.py`
- `src/dgtd/__init__.py`

Configs:

- `configs/experiment/dgtd_cifar10_v3.yaml`
- `configs/experiment/dgtd_cifar10_v3_smoke.yaml`

Tests:

- `tests/test_dgtd.py`

Documentation:

- `docs/dgtd_v3_round2_patch_notes.md`

## 2. What each change fixes

### 2.1 Symmetric stop-grad residual

Problem before:

- the old implementation built
  `target = teacher_or_bootstrap.detach()`
- therefore the main gradient effectively updated only
  `x_u_direct = M_theta(x_t, t, u)`
- the bridge path `x_s_pred = M_theta(x_t, t, s)` had no useful training signal

Fix:

- `src/dgtd/defect.py` now builds a unified continuation state `x_u_cont`
- `src/dgtd/train_dgtd.py` trains with a symmetric stop-grad estimator

This preserves one residual geometry while giving gradient to both sides.

### 2.2 Offline affine teacher continuation

Problem before:

- when `disable_online_teacher=true`, the teacher anchor only existed when
  `x_s_pred` was close enough to the cached `x_s_teacher`
- otherwise the code fell back to detached student bootstrap
- in practice this made teacher anchoring too fragile

Fix:

- `src/dgtd/teacher.py` now supports an affine local surrogate from the cached
  trajectory
- exact cached hits are still retained when the predicted bridge state is close
  enough
- otherwise the code uses a teacher-anchored affine continuation instead of
  dropping the anchor completely

Priority order is now:

1. online teacher local flow
2. cached exact continuation, when near cache
3. cached affine continuation
4. bootstrap fallback

### 2.3 Unified sigma helper

Problem before:

- the code used `1 - t` in multiple places implicitly
- although they were numerically aligned in the default config, there was no
  single schedule owner

Fix:

- `src/dgtd/sigma.py` now defines the shared `SigmaSchedule`
- `TeacherAdapter`, `edm_weight`, `min_snr_weight`, `lambda_hf`, and the affine
  continuation all consume the same schedule function
- current default is explicit in config:
  `dgtd.sigma_mode: linear_1mt`

### 2.4 Safer `q_D` and better warp diagnostics

Problem before:

- `HF_bar` directly sharpened `q_D` by default
- the warp had no explicit entropy, KL, or base-ratio diagnostics
- there was no way to see whether the sampler had become too peaky

Fix:

- `q_D` now defaults to a more conservative form:
  `normalize(D_bar) + lambda_K * normalize(K_bar)`
- `HF_bar` is configurable but disabled by default in the v3 config
- optional `qd_ratio_cap` is available to bound `q_D / q_base`
- trainer now logs entropy, KL, maximum density ratio, peak bin, stage values,
  and continuation source ratios

## 3. Updated residual formula

Define:

- `x_s_pred = M_theta(x_t, t, s)`
- `x_u_direct = M_theta(x_t, t, u)`
- `x_u_cont = continuation(x_s_pred, s, u)`

Current continuation is:

- if teacher continuation exists:
  `x_u_cont = eta * teacher_cont + (1 - eta) * bridge_cont`
- otherwise:
  `x_u_cont = bridge_cont`

where:

- `bridge_cont = M_theta(x_s_pred, s, u)`
- teacher-provided parts are detached or teacher-anchored

The symmetric stop-grad objective is:

- `term_direct = metric_norm(x_u_direct - stopgrad(x_u_cont), u)`
- `term_bridge = metric_norm(stopgrad(x_u_direct) - x_u_cont, u)`
- `loss = mean(omega * 0.5 * (term_direct + term_bridge))`

This differs from the old implementation in one critical way:

- before, only `x_u_direct` received effective gradient
- now, `x_u_direct` and the continuation branch both receive gradient

Gradient sketch:

```text
x_t -> x_u_direct -----------------------> term_direct
  \-> x_s_pred -> bridge_cont -----------> term_bridge
                \-> cached_affine teacher -> term_bridge
```

With cached affine continuation, the bridge branch also gets gradient through the
teacher-anchored affine map.

## 4. Teacher affine surrogate

For cached teacher states:

- `x_s_teacher`
- `x_u_teacher`

the current offline local surrogate is:

- `teacher_affine_cont(z_s; s, u) = x_u_teacher + alpha(s, u) * (z_s - x_s_teacher)`

Current default:

- `alpha(s, u) = sigma(u) / max(sigma(s), eps)`

Behavior:

- when `z_s` is very close to `x_s_teacher`, the code can still use
  `cached_exact`
- when it is not close enough, the code uses `cached_affine`
- if no teacher continuation is available at all, the code falls back to
  bootstrap

## 5. Unified sigma scheme

Current config-controlled sigma owner:

- `src/dgtd/sigma.py::SigmaSchedule`

Supported modes:

- `linear_1mt`
- `explicit_grid`

Current v3 default:

- `dgtd.sigma_mode: linear_1mt`
- `sigma(t) = clamp(1 - t, min=sigma_min)`

Current users of the shared sigma schedule:

- teacher continuation affine coefficient `alpha(s, u)`
- `metric_norm(..., u)`
- `lambda_hf(u)`
- `edm_weight(u)`
- `min_snr_weight(u)`

## 6. New diagnostics

New trainer diagnostics added in `src/dgtd/train_dgtd.py`:

Global error statistics:

- `direct_teacher_error`
- `bridge_teacher_error`
- `direct_bridge_gap`
- `teacher_rel_error`

Continuation source ratios:

- `source_ratio_online`
- `source_ratio_cached_affine`
- `source_ratio_cached_exact`
- `source_ratio_bootstrap`
- aggregated map under `continuation_sources`

Warp diagnostics:

- `entropy_q_phi`
- `kl_qD_qphi`
- `max_qphi_over_qbase`
- `argmax_q_phi`
- `flatten_mix`
- `eta`
- `beta`
- `mu`

Region-wise train diagnostics over `u`:

- `noisy_defect`, `mid_defect`, `clean_defect`
- `noisy_hf`, `mid_hf`, `clean_hf`
- `noisy_endpoint_error`, `mid_endpoint_error`, `clean_endpoint_error`

## 7. Config changes

Main new DGTD config fields:

```yaml
dgtd:
  symmetric_residual: true
  teacher_continuation_mode: affine_fallback
  teacher_keep_cached_exact: true
  sigma_mode: linear_1mt
  sigma_max: 1.0
  qd_curvature_weight: 0.25
  qd_use_hf_bar: false
  qd_hf_weight: 0.0
  qd_ratio_cap: 0.0
```

Interpretation:

- symmetric residual is enabled by default
- offline teacher anchor now uses affine fallback
- sigma schedule is explicit and shared
- `q_D` is more conservative by default

## 8. Smoke validation commands

Train smoke:

```bash
python scripts/run_train.py \
  --config configs/experiment/dgtd_cifar10_v3_smoke.yaml \
  --run-root /tmp/dgtd_v3_round2_smoke \
  --set target.shard_root=$TRAJ_ROOT
```

Sample smoke:

```bash
python scripts/run_sample_dgtd.py \
  --config configs/experiment/dgtd_cifar10_v3_smoke.yaml \
  --checkpoint /tmp/dgtd_v3_round2_smoke/checkpoints/last.pt \
  --output-dir /tmp/dgtd_v3_round2_smoke/sample \
  --steps 4 \
  --set target.shard_root=$TRAJ_ROOT
```

Eval smoke:

```bash
python scripts/run_eval.py \
  --config configs/experiment/dgtd_cifar10_v3_smoke.yaml \
  --checkpoint /tmp/dgtd_v3_round2_smoke/checkpoints/last.pt \
  --eval-root /tmp/dgtd_v3_round2_smoke/eval \
  --steps 1 2 4 \
  --set target.shard_root=$TRAJ_ROOT
```

Optional diagnostics export:

```bash
python scripts/plot_dgtd_diagnostics.py \
  --history /tmp/dgtd_v3_round2_smoke/logs/train.jsonl \
  --output-dir /tmp/dgtd_v3_round2_smoke/diag
```

## 9. Local verification completed in this round

Run locally:

```bash
python -m py_compile \
  src/dgtd/sigma.py \
  src/dgtd/teacher.py \
  src/dgtd/defect.py \
  src/dgtd/metrics.py \
  src/dgtd/train_dgtd.py \
  src/dgtd/warp.py
```

```bash
pytest tests/test_dgtd.py -q
```

Observed result:

- `py_compile` passed
- `pytest tests/test_dgtd.py -q` passed with `7 passed`
