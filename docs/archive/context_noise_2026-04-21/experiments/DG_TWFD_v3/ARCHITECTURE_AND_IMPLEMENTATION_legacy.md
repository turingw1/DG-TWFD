# DG_TWFD_v3 Architecture And Implementation

This document is the authoritative specification of the current `DG_TWFD_v3`
branch state.

Its purpose is twofold:

1. allow a new engineer to reproduce the current project from scratch
2. allow precise codebase modification without reverse-engineering the source

This document describes the code as it is currently implemented, not the ideal
target state from `reconstruction_v3.md`.

## 1. Project goal

The current branch implements a `dgtd_map` training line on top of the existing
`dgfm` infrastructure.

The active training objective is:

- a single unified DGTD residual
- explicit map prediction `M_theta(x_t, t, s) -> x_s`
- offline frozen teacher trajectory supervision
- learnable monotone time-density warp

Current high-level formula:

- student direct prediction:
  - `x_u_direct = M_theta(x_t, t, u)`
- student bridge prediction:
  - `x_s_pred = M_theta(x_t, t, s)`
- target:
  - `teacher continuation` if available from cached trajectory continuation
  - otherwise `stopgrad(M_theta(x_s_pred, s, u))`
- residual:
  - `R = x_u_direct - target`
- loss:
  - `mean(omega * metric_norm(R, u))`

The implementation deliberately keeps the original `dgfm` runtime pieces for:

- config loading
- distributed setup
- run-root layout
- checkpointing
- evaluation
- few-step sampling

## 2. Current branch topology

The current active branch code path is centered on these files:

- train entry:
  - [scripts/run_train.py](../../../scripts/run_train.py)
- eval entry:
  - [scripts/run_eval.py](../../../scripts/run_eval.py)
- DGTD sample entry:
  - [scripts/run_sample_dgtd.py](../../../scripts/run_sample_dgtd.py)
- config loader:
  - [src/dgfm/config/loader.py](../../../src/dgfm/config/loader.py)
- trainer dispatch:
  - [src/dgfm/trainers/__init__.py](../../../src/dgfm/trainers/__init__.py)
- DGTD trainer:
  - [src/dgtd/train_dgtd.py](../../../src/dgtd/train_dgtd.py)
- DGTD teacher adapter:
  - [src/dgtd/teacher.py](../../../src/dgtd/teacher.py)
- DGTD cache loader:
  - [src/dgtd/cache.py](../../../src/dgtd/cache.py)
- DGTD residual and density logic:
  - [src/dgtd/defect.py](../../../src/dgtd/defect.py)
- DGTD metric weighting:
  - [src/dgtd/metrics.py](../../../src/dgtd/metrics.py)
- DGTD warp:
  - [src/dgtd/warp.py](../../../src/dgtd/warp.py)
- explicit map backbone:
  - [src/dgfm/models/map.py](../../../src/dgfm/models/map.py)
- runtime time-grid bridge:
  - [src/dgfm/schedulers/timewarp.py](../../../src/dgfm/schedulers/timewarp.py)
- generic eval runtime:
  - [src/dgfm/evaluators/common.py](../../../src/dgfm/evaluators/common.py)
  - [src/dgfm/evaluators/runner.py](../../../src/dgfm/evaluators/runner.py)
- map rollout sampler:
  - [src/dgfm/samplers/map_sampler.py](../../../src/dgfm/samplers/map_sampler.py)

## 3. Time semantics

This branch uses the `dgfm` map-branch time convention:

- `t in [0, 1]`
- `0.0` means noisiest
- `1.0` means cleanest

All current DGTD triplets obey:

- `t < s < u`

Interpretation:

- `x_t` is noisier than `x_s`
- `x_s` is noisier than `x_u`
- `u` is the most denoised point in the sampled triplet

This is the opposite orientation of some diffusion pseudocode written in
decreasing-noise notation. The branch handles this explicitly in:

- [src/dgtd/train_dgtd.py](../../../src/dgtd/train_dgtd.py)
- [src/dgtd/defect.py](../../../src/dgtd/defect.py)
- [src/dgtd/warp.py](../../../src/dgtd/warp.py)

Any future code change must preserve this convention unless the whole branch is
rewritten consistently.

## 4. Config system

Config loading is implemented in
[loader.py](../../../src/dgfm/config/loader.py).

Important behavior:

- `base:` loads one parent config
- `includes:` merges additional config fragments from `configs/`
- later entries override earlier entries
- `--set key=value` applies runtime overrides
- bash-style defaults such as `${TRAJ_ROOT:-...}` are expanded

Current main config:

- [configs/experiment/dgtd_cifar10_v3.yaml](../../../configs/experiment/dgtd_cifar10_v3.yaml)

Current smoke config:

- [configs/experiment/dgtd_cifar10_v3_smoke.yaml](../../../configs/experiment/dgtd_cifar10_v3_smoke.yaml)

Key config fragments:

- model:
  - [configs/model/map_unet.yaml](../../../configs/model/map_unet.yaml)
- target cache:
  - [configs/target/teacher_trajectory.yaml](../../../configs/target/teacher_trajectory.yaml)
- eval:
  - [configs/eval/map_branch.yaml](../../../configs/eval/map_branch.yaml)

## 5. Backbone

Current backbone:

- `model.family: official_map_unet`

Implementation:

- [src/dgfm/models/map.py](../../../src/dgfm/models/map.py)

The model is a wrapper around the official `flow_matching` image example U-Net.

Key implementation details:

- input image tensor:
  - `x_t: [B, C, H, W]`
- time arguments:
  - `t: [B]`
  - `s: [B]`
- extra conditioning maps appended to the image channels:
  - one channel for `s`
  - one channel for `delta = s - t`
- current conditioning channel count:
  - `map_conditioning_channels: 2`

Current parameterization:

- `prediction_type: residual`
- residual is `tanh(raw) * residual_tanh_scale`
- optional scaling by `s - t`
- final output is `x_t + residual`

Current preconditioning path is enabled:

- `use_preconditioning: true`
- `time_embed_mode: log_noise`
- `inner_parametrization: edm`
- `outer_parametrization: euler`

This means the active map backbone is not a plain direct image-to-image U-Net.
It uses EDM-style inner and outer scalings before producing the final map output.

If you want to change the backbone:

- edit `configs/model/map_unet.yaml` for hyperparameters
- edit `OfficialExplicitMapUNet` in
  [map.py](../../../src/dgfm/models/map.py)
  for architecture logic
- edit `build_map_model()` in the same file if you add a new family

## 6. Teacher mode

Current teacher mode is:

- offline frozen trajectory cache

This is controlled by:

- [dgtd_cifar10_v3.yaml](../../../configs/experiment/dgtd_cifar10_v3.yaml)
  - `target.builder: trajectory_shard`
  - `dgtd.disable_online_teacher: true`

The branch still includes a `teacher/sampler` config fragment, but current DGTD
training does not depend on online teacher execution.

Teacher adapter behavior is defined in
[teacher.py](../../../src/dgtd/teacher.py).

Current logic:

- `cached_state(traj, t)`:
  - interpolate teacher state from cached trajectory
- `local_flow(traj, s, u, z)`:
  - if `z` is close to cached `x_s`, return cached `x_u`
  - else, if an online teacher exists and provides `local_flow`, use it
  - else return `None`
- `sigma(t)`:
  - approximated as `1 - t`, lower bounded by `sigma_min`

Because `disable_online_teacher=true`, the active path is:

- cached-near-state continuation if possible
- otherwise detached student bootstrap fallback

## 7. Teacher trajectory cache format

Current expected raw shard sample format is:

```python
{
  "t_grid": Tensor[M],
  "x_grid": Tensor[M, C, H, W],
  "y": optional class label,
  "label": optional class label,
}
```

Loaded branch-internal format becomes:

```python
{
  "times": Tensor[M],
  "states": Tensor[M, C, H, W],
  "curvature": Tensor[M-2],
  "cond": optional Tensor[],
}
```

Implemented in:

- [src/dgtd/cache.py](../../../src/dgtd/cache.py)

Important behavior:

- if `train/` or `val/` subdirectories exist under `target.shard_root`, the
  loader uses them
- otherwise it treats the root as a flat shard directory
- each shard must contain a Python `list` of sample dicts
- if `manifest.yaml` exists, it is used for shard counts
- otherwise shard lengths are inferred from the first and last shard

Interpolation behavior:

- `interpolate_state(traj, t)`:
  - per-sample linear interpolation over `times`
- `interpolate_curvature(traj, t)`:
  - per-sample linear interpolation over interior curvature anchors

Curvature is computed on load if not precomputed:

- finite-difference velocities:
  - `v_i = (x_{i+1} - x_i) / (t_{i+1} - t_i)`
- curvature proxy:
  - `||v_{i+1} - v_i||^2 / (eps + ||v_i||^2)`

## 8. Warp module

Warp implementation:

- [src/dgtd/warp.py](../../../src/dgtd/warp.py)

Current class:

- `MonotoneDensityWarp`

Parameters:

- `density_raw: nn.Parameter[num_bins]`
- positive density is `softplus(density_raw) + eps`
- normalized to produce `q_phi`

Exposed methods:

- `density()`
- `cdf()`
- `t_to_r(t)`
- `r_to_t(r)`
- `density_at(t)`
- `sample_triplets(batch_size, device)`
- `kl_to_target_density(q_D)`

Current triplet sampling is done in warped `r` space using short/medium/long
interval mixtures:

- short:
  - probability `0.5`
  - interval range `[1/64, 1/16]`
- medium:
  - probability `0.3`
  - interval range `[1/16, 1/4]`
- long:
  - probability `0.2`
  - interval range `[1/4, 1]`

Current code samples:

- `r_t`
- `r_s = r_t + delta_1`
- `r_u = r_t + delta_2`

which is consistent with the branch time convention `t < s < u`.

Warp integration into the generic scheduler happens in:

- [src/dgfm/schedulers/timewarp.py](../../../src/dgfm/schedulers/timewarp.py)

Branch-specific behavior there:

- `scheduler.timewarp.type: dgtd_density` builds `MonotoneDensityWarp`
- runtime sampling grids are built via `r_to_t()`
- checkpoint load/save treats `timewarp` as a regular module state dict

## 9. Detail-aware metric and weighting

Metric implementation:

- [src/dgtd/metrics.py](../../../src/dgtd/metrics.py)

Current metric:

- base term:
  - `mean(R^2)` over flattened image dimensions
- high-frequency term:
  - Laplacian-filtered residual energy
- total:
  - `l2_term + lambda_hf(u) * hf_term`

Current `lambda_hf(u)`:

- computed from `sigma(u)`
- `lambda_hf_max * exp(-sigma(u)^2 / sigma_detail^2)`

Current importance weighting:

- `edm_weight(u)`
- multiplied by `min_snr_weight(u)`
- multiplied by `p_corr(t)^(-kappa)`
- normalized by mean over batch

This is implemented inside the DGTD training loop, not as a standalone loss
module.

## 10. Residual construction

Residual construction is implemented in
[src/dgtd/defect.py](../../../src/dgtd/defect.py).

Current code path:

```python
x_s_pred = student(x_t, t, s)
x_u_direct = student(x_t, t, u)
teacher_cont = teacher_adapter.local_flow(...)
self_cont = student(x_s_pred.detach(), s, u).detach()

if teacher_cont is None:
    target = self_cont
else:
    target = eta * teacher_cont.detach() + (1 - eta) * self_cont

residual = x_u_direct - target
```

Important implementation detail:

- `teacher_cont` is detached before mixing
- `self_cont` is always detached
- therefore gradients flow only through `x_u_direct`
- bridge prediction `x_s_pred` contributes indirectly through the target
  construction path, but the continuation branch is detached

Current defect statistic:

- `metric_value / (eps + mean((x_u_teacher - x_t)^2))`

Current target density:

- build normalized `A` from
  - `D_bar`
  - `0.25 * K_bar`
  - `0.5 * HF_bar`
- then:
  - `q_D ∝ (A + eps)^beta * q_base`
- optional 1D smoothing with `[0.25, 0.5, 0.25]`

## 11. Training loop

DGTD trainer implementation:

- [src/dgtd/train_dgtd.py](../../../src/dgtd/train_dgtd.py)

Entry path:

1. `scripts/run_train.py`
2. `load_experiment_config()`
3. `resolve_run_roots()`
4. `init_distributed()`
5. `build_trainer()`
6. dispatch to `DGTDTrainer`

### 11.1 Initialization

`DGTDTrainer.run()` does:

- build train/val cache dataloaders
- build map backbone
- wrap with DDP if needed
- build EMA shadow model
- build model optimizer
- build DGTD warp module
- build warp optimizer
- build teacher adapter
- initialize bin statistics
- optionally resume checkpoint state

### 11.2 Stage schedule

Implemented by `_stage_values()` in `train_dgtd.py`.

Current stages:

- warmup
  - `beta = 0`
  - `eta = 1.0`
  - `mu = ema_mu_warmup`
- adaptive
  - `beta` ramps to `beta_final`
  - `eta` decays to `eta_min`
  - `mu = ema_mu_main`
- flatten
  - `beta` softens
  - `flatten_mix` mixes `q_target` back toward `q_base`

### 11.3 Per-batch step

Inside `_run_epoch()`:

1. sample triplets from warp
2. map to `t,s,u`
3. interpolate `x_t`, `x_u_teacher`, `curvature`
4. compute DGTD residual
5. compute metric value
6. compute importance weights
7. optimize student
8. update EMA
9. update `D/K/HF` bin statistics
10. if beyond warmup:
   - periodically rebuild `q_target`
   - periodically optimize warp with KL
11. log scalar metrics and diagnostic arrays

### 11.4 Logging

`logs/train.jsonl` stores per-epoch payloads including:

- `train_loss`
- `val_loss`
- `train_defect`
- `val_defect`
- `train_low_sigma_hf`
- `val_low_sigma_hf`
- `q_phi`
- `q_D`
- `D_bar`
- `K_bar`
- `HF_bar`
- `time_grid`
- `eta`
- `beta`
- `stage`
- timing statistics

### 11.5 Checkpoints

Each checkpoint stores:

- `model`
- `ema_model`
- `optimizer`
- `scaler`
- `timewarp`
- `timewarp_optimizer`
- `dgtd_q_target`
- `dgtd_stats`
- `epoch`
- `best_val`
- `global_step`
- `config`

Written files:

- `checkpoints/last.pt`
- `checkpoints/best.pt`

## 12. Sampling

DGTD sampling entry:

- [scripts/run_sample_dgtd.py](../../../scripts/run_sample_dgtd.py)

Sampling helper:

- [src/dgtd/sample_dgtd.py](../../../src/dgtd/sample_dgtd.py)

Current supported modes:

- `mode_a`
  - implemented
- `mode_b_stub`
  - TODO placeholder only

Current mode A:

- build uniform `r` grid
- map to runtime `t` grid using `warp.r_to_t`
- perform explicit map rollout:
  - `x_{t_{i+1}} = M_theta(x_{t_i}, t_i, t_{i+1})`

Sampling outputs:

- `samples.pt`
- `labels.pt` if class-conditional
- `grid.png`
- per-image PNG dumps
- `time_grid.pt`

## 13. Evaluation

Eval entry:

- [scripts/run_eval.py](../../../scripts/run_eval.py)

Eval runtime:

- [src/dgfm/evaluators/common.py](../../../src/dgfm/evaluators/common.py)
- [src/dgfm/evaluators/runner.py](../../../src/dgfm/evaluators/runner.py)

Current behavior:

- loads EMA model if present
- loads warp state if present
- for each requested step count:
  - generates FID samples
  - saves fixed-seed sample grids
  - computes FID
  - records runtime time grid summary

Current report outputs:

- `eval_root/steps{K}/metrics.json`
- `eval_root/steps{K}/generated_stats.npz`
- `eval_root/reports/summary.json`
- `eval_root/reports/summary.csv`
- `eval_root/reports/best.json`

## 14. Distributed behavior

Distributed helpers:

- [src/dgfm/distributed.py](../../../src/dgfm/distributed.py)

Current behavior:

- if `WORLD_SIZE <= 1`, run single-process
- otherwise:
  - use `nccl` on CUDA
  - use `gloo` on CPU
  - wrap model with DDP
  - all-reduce bin statistics via `torch.distributed`

Cache loaders also use `DistributedSampler` when `WORLD_SIZE > 1`.

## 15. Reproducing the current project from scratch

### 15.1 Required inputs

You need:

- the current repository tree
- a working PyTorch environment
- a valid `TRAJ_ROOT`
- trajectory shards matching the cache format above

### 15.2 Minimal branch-local commands

Smoke train:

```bash
python scripts/run_train.py \
  --config configs/experiment/dgtd_cifar10_v3_smoke.yaml \
  --run-root /tmp/dgtd_v3_smoke \
  --set target.shard_root=$TRAJ_ROOT
```

Smoke sample:

```bash
python scripts/run_sample_dgtd.py \
  --config configs/experiment/dgtd_cifar10_v3_smoke.yaml \
  --checkpoint /tmp/dgtd_v3_smoke/checkpoints/last.pt \
  --output-dir /tmp/dgtd_v3_smoke/sample \
  --steps 4 \
  --set target.shard_root=$TRAJ_ROOT
```

Smoke eval:

```bash
python scripts/run_eval.py \
  --config configs/experiment/dgtd_cifar10_v3_smoke.yaml \
  --checkpoint /tmp/dgtd_v3_smoke/checkpoints/last.pt \
  --eval-root /tmp/dgtd_v3_smoke/eval \
  --steps 1 2 4 \
  --set target.shard_root=$TRAJ_ROOT
```

### 15.3 Full train

```bash
torchrun --nproc_per_node=2 scripts/run_train.py \
  --config configs/experiment/dgtd_cifar10_v3.yaml \
  --run-root runs/dgtd_cifar10_v3 \
  --set target.shard_root=$TRAJ_ROOT
```

## 16. How to modify the codebase precisely

### 16.1 Change backbone

Edit:

- config:
  - [configs/model/map_unet.yaml](../../../configs/model/map_unet.yaml)
- model logic:
  - [src/dgfm/models/map.py](../../../src/dgfm/models/map.py)

If adding a new backbone family:

- add a branch in `build_map_model()`
- add a new config file under `configs/model/`

### 16.2 Change teacher mode

Edit:

- [configs/target/teacher_trajectory.yaml](../../../configs/target/teacher_trajectory.yaml)
- [src/dgtd/teacher.py](../../../src/dgtd/teacher.py)
- [src/dgtd/cache.py](../../../src/dgtd/cache.py)

If enabling online teacher:

- set `dgtd.disable_online_teacher: false`
- implement a real `local_flow()` provider on the teacher side
- ensure `TeacherAdapter.local_flow()` returns the online continuation

### 16.3 Change time semantics

This is branch-wide and dangerous.

You must update all of:

- [src/dgtd/train_dgtd.py](../../../src/dgtd/train_dgtd.py)
- [src/dgtd/defect.py](../../../src/dgtd/defect.py)
- [src/dgtd/warp.py](../../../src/dgtd/warp.py)
- [src/dgtd/metrics.py](../../../src/dgtd/metrics.py)
- [src/dgfm/models/map.py](../../../src/dgfm/models/map.py)

Do not change only one location.

### 16.4 Change warp update behavior

Edit:

- `dgtd.*` schedule fields in
  [dgtd_cifar10_v3.yaml](../../../configs/experiment/dgtd_cifar10_v3.yaml)
- warp implementation:
  - [src/dgtd/warp.py](../../../src/dgtd/warp.py)
- outer-loop update logic:
  - [src/dgtd/train_dgtd.py](../../../src/dgtd/train_dgtd.py)

### 16.5 Change defect metric

Edit:

- [src/dgtd/metrics.py](../../../src/dgtd/metrics.py)
- optionally:
  - [src/dgtd/defect.py](../../../src/dgtd/defect.py)
  - [src/dgtd/train_dgtd.py](../../../src/dgtd/train_dgtd.py)

### 16.6 Change ablations

Current ablation configs are under:

- `configs/experiment/dgtd_*`

The simplest pattern is:

- copy the base config
- change only `dgtd.*` flags and runtime limits
- keep smoke variants separate from full variants

## 17. Current gaps and non-final parts

This branch is functional but not yet fully complete relative to the original
reconstruction prompt.

Current gaps:

- `mode_b_stub` is not implemented
- online teacher local solver is not implemented
- diagnostics are still basic
- server-side smoke and quality validation are still ongoing

Therefore this document is a specification of the current implementation state,
not a claim that the entire research plan is finished.
