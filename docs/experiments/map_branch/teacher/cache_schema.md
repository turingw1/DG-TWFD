# Teacher Cache Schema

This file records the current offline teacher-trajectory shard schema used by
the `dgfm` map-branch path.

It describes the output of:

- `scripts/prepare_teacher_trajectories.py`

It only describes the shard format consumed by the current map-branch readers.

## Directory layout

Expected root:

```text
<shard_root>/
  train/
    manifest.yaml
    train_00000.pt
    train_00001.pt
    ...
  val/
    manifest.yaml
    val_00000.pt
    val_00001.pt
    ...
```

## `manifest.yaml`

Current writer fields:

```yaml
split: train
count: 50000
teacher:
  type: sampler
  backend: diffusers_ddpm
  name_or_path: google/ddpm-cifar10-32
  solver: ddim
  num_inference_steps: 128
  retain_num_points: 33
  time_semantics: dgfm_u_grid_ascending_0_noise_1_clean
shards:
  - file: train_00000.pt
    count: 1024
```

## Per-shard payload

Each `.pt` shard is a Python `list[dict]`.

Each sample dict currently contains exactly:

```python
{
    "sample_id": int,
    "t_grid": torch.Tensor,   # shape [M], float32, ascending, 0=noise, 1=clean
    "x_grid": torch.Tensor,   # shape [M, C, H, W], stored as teacher.store_dtype
}
```

Current defaults imply:

- `M = retain_num_points = 33`
- `C = 3`
- `H = W = 32`
- `x_grid` storage dtype is usually `float16`

## Reader-side assumptions

`src/dgfm/datasets/trajectory.py` currently assumes:

- `t_grid` is 1D
- `x_grid` is `[M, C, H, W]`
- if a future shard writes `[1, M, C, H, W]`, the reader squeezes the leading
  singleton dimension
- `t_grid` is sorted before pair sampling

The dataset converts samples into pair supervision:

```python
{
    "x_t": x_grid[t_index].float(),
    "x_s": x_grid[s_index].float(),
    "t": t_grid[t_index].float(),
    "s": t_grid[s_index].float(),
    "x_0": x_grid[0].float(),
    "x_1": x_grid[-1].float(),
}
```

## Current pair-sampling contract for shard mode

When `configs/target/teacher_trajectory.yaml` is used, the current defaults are:

- `pair_short_max: 4`
- `pair_mid_max: 12`
- `pair_long_max: 32`
- `pair_short_weight: 0.55`
- `pair_mid_weight: 0.30`
- `pair_long_weight: 0.15`
- `pair_endpoint_weight: 0.35`
- `high_noise_t_weight: 0.75`
- `high_noise_t_fraction: 0.35`

Those values affect how `TrajectoryShardPairDataset` chooses `(t, s)` from the
stored teacher grid.
