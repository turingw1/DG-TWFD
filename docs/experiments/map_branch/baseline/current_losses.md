# Current `dgfm` Baseline Losses

This file is only about the current `dgfm` map-branch line.

Primary code/config sources:

- `src/dgfm/trainers/map.py`
- `src/dgfm/targets/builder.py`
- `src/dgfm/targets/pair_sampling.py`
- `configs/target/teacher_sampler_online.yaml`
- `configs/loss/map_ctm_like.yaml`
- `configs/experiment/fm_cifar10_map_branch_s1_e6_budget_full.yaml`
- `configs/experiment/fm_cifar10_map_branch_s1_e6_budget_fullstack.yaml`

## Scope

There are two practical reference recipes on the current line:

- clean baseline: `e602a`
  - config: `fm_cifar10_map_branch_s1_e6_budget_full`
  - CTM-style target construction on
  - endpoint off
  - learnable timewarp off
- fullstack analysis: `e603a`
  - config: `fm_cifar10_map_branch_s1_e6_budget_fullstack`
  - same CTM-style target construction
  - endpoint on
  - learnable timewarp on

So the current line has one main loss family, with endpoint/timewarp as optional
auxiliary branches.

## 1. `M_theta` flow-map matching loss

Current estimate:

```text
x_s_hat = M_theta(x_t, t, s)
```

Current loss core:

```text
L_match = pixel_weight * L_pixel(x_s_hat, target)
        + perceptual_weight * L_perc(x_s_hat, target)
```

where:

- `L_pixel` is MSE by default:
  - `mean((x_s_hat - target)^2)`
- optional fallback is Huber:
  - `F.huber_loss(..., delta=target.huber_delta)`
- `L_perc` is only active if a perceptual metric is built successfully

Current default weights from `configs/loss/map_ctm_like.yaml`:

```yaml
pixel_weight: 1.0
perceptual_weight: 0.25
```

## 2. CTM-style composition loss

On the current `dgfm` line, there is no separate scalar `L_comp`.

Instead, CTM-style composition is absorbed into the target construction:

```text
target = stop_grad(M_target(x_t_dt, t_dt, s))
```

with the bridge state `x_t_dt` constructed first, then fed into the target
model. In the current clean baseline (`e602a`) that means:

```text
target_construction = ctm_consistency
target_source       = ema_model
target_stop_grad    = true
bridge_source       = ema_model_rollout
bridge_steps        = 1
bridge_stop_grad    = true
```

So the exact current chain is:

```text
x_t_dt   = stop_grad(M_ema(x_t, t, t_dt))
target   = stop_grad(M_ema(x_t_dt, t_dt, s))
x_s_hat  = M_theta(x_t, t, s)
L_match  = L(x_s_hat, target)
```

This is the current line’s CTM-style composition mechanism. It is not exposed
as an additional loss term in `train_loss`; it is baked into the supervised
target itself.

## 3. Boundary / N2N loss

The current `dgfm` line has no boundary corrector module and no N2N boundary
loss branch.

Exact current status:

- there is no `B_psi`
- there is no `L_boundary`
- there is no boundary coefficient in `src/dgfm/trainers/map.py`

What exists instead is an optional endpoint rollout auxiliary:

```text
x_0 ~ teacher noise prior
x_1 = clean endpoint paired with the sampled teacher trajectory
x_1_hat = rollout_with_map(M_theta, x_0, step_count)
L_endpoint = pixel_weight * L_pixel(x_1_hat, x_1)
           + perceptual_weight * L_perc(x_1_hat, x_1)
```

Current coefficients:

- clean baseline `e602a`:
  - `endpoint_weight = 0.0`
- fullstack `e603a`:
  - `endpoint_weight = 0.25`
  - `endpoint_every = 8`
  - `endpoint_batch_size = 64`
  - `endpoint_steps = [8, 16]`
  - `endpoint_step_weights = [0.5, 0.5]`

## 4. Current time-warp / timestep schedule

There are two schedule layers on the current line.

### 4.1 CTM-discrete triplet schedule for teacher targets

The current online teacher sampler retains `start_scales = 33` points. It then
samples `(t, t_dt, s)` in index space:

```text
0 <= t_index < t_dt_index <= s_index <= 32
```

Current default sampling config:

```yaml
sampling_mode: ctm_discrete
start_scales: 33
num_heun_step: 17
num_heun_step_random: true
heun_step_strategy: weighted
heun_step_multiplier: 1.0
sample_s_strategy: uniform
teacher_sampler_sub_batch: 128
```

Exact current sampling logic:

1. sample `num_heun_steps` from `1..17`
   - weighted by `k^1.0`
2. sample `t_index` uniformly from valid starts
3. set `t_dt_index = t_index + num_heun_steps`
4. sample `s_index` uniformly from `[t_dt_index, 32]`

If timewarp is disabled, the retained teacher grid is uniform on `[0, 1]`.

### 4.2 Runtime rollout grid

At train/eval/sample time, the runtime grid is built by:

```text
time_grid = linspace(0, 1, step_count + 1)
```

unless `scheduler.timewarp.enabled=true`, in which case the runtime grid is
warped by the monotone timewarp module.

Current clean baseline `e602a`:

```yaml
scheduler.timewarp.enabled: false
```

So its runtime schedule is exactly a uniform `linspace(0, 1, N+1)`.

Current fullstack `e603a`:

```yaml
scheduler.timewarp.enabled: true
scheduler.timewarp.type: learnable_monotone
scheduler.timewarp.num_bins: 64
scheduler.timewarp.init_bias: 0.0
scheduler.timewarp.lr: 1.0e-3
scheduler.timewarp.weight_decay: 0.0
```

Its auxiliary timewarp loss is:

```text
states = rollout_trajectory_with_map(M_theta, x_0, time_grid)
L_tw_defect  = mean(||M_theta(x_i, t_i, t_{i+2}) - M_theta(x_{i+1}, t_{i+1}, t_{i+2})||^2)
L_tw_balance = mean((delta_i - mean(delta))^2)
L_timewarp   = timewarp_defect_weight * L_tw_defect
             + timewarp_balance_weight * L_tw_balance
```

with current `e603a` coefficients:

```yaml
timewarp_weight: 1.0
timewarp_update_every: 1
timewarp_defect_steps: 4
timewarp_batch_size: 32
timewarp_defect_weight: 1.0
timewarp_balance_weight: 0.1
```

## 5. Current weighting coefficients and training schedule

### Clean baseline `e602a`

Current loss configuration:

```yaml
pixel_weight: 1.0
perceptual_weight: 0.25
endpoint_weight: 0.0
timewarp_weight: 0.0
```

Current target configuration:

```yaml
target_construction: ctm_consistency
target_source: ema_model
target_stop_grad: true
bridge_source: ema_model_rollout
bridge_steps: 1
bridge_stop_grad: true
```

### Fullstack `e603a`

Overrides on top of `e602a`:

```yaml
endpoint_weight: 0.25
timewarp_weight: 1.0
```

with the endpoint/timewarp details listed above.

### Shared train schedule on the current line

The experiment root `configs/experiment/fm_cifar10_map_branch.yaml` currently
uses:

```yaml
train:
  batch_size: 160
  num_workers: 12
  persistent_workers: true
  prefetch_factor: 6
```

The smoke base used by stage-1 configs sets:

```yaml
train:
  epochs: 8
```

and `e602a` inherits that same epoch count.

Optimizer/runtime details from the current trainer:

- model optimizer: `AdamW(model.parameters(), lr=train.lr, weight_decay=train.weight_decay)`
- timewarp optimizer: separate `AdamW` only when timewarp is enabled
- EMA shadow is used as the default `target_source`
- mixed precision is enabled when `runtime.amp=true`
- TF32 is enabled on CUDA when `runtime.tf32=true`

## Practical reading

If the goal is to summarize the current `dgfm` baseline in one sentence, it is:

```text
Train M_theta(x_t, t, s) against a stop-grad EMA-composed CTM-style target on
teacher-sampled triplets, with endpoint and learnable timewarp kept as optional
auxiliary branches rather than core losses.
```
