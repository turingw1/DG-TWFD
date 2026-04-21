# Current `dgfm` Baseline Metrics

This file documents the metrics currently produced and used by the `dgfm`
map-branch line.

Primary sources:

- `src/dgfm/trainers/map.py`
- `src/dgfm/evaluators/runner.py`
- `docs/experiments/map_branch/A100_PIPELINE.md`
- `docs/experiments/map_branch/HANDOFF_2026-04-16.md`

## Training / validation metrics

The trainer writes one JSON object per epoch to:

- `logs/train.jsonl`

Current core fields:

- `train_loss`
- `train_pixel_loss`
- `train_perceptual_loss`
- `train_endpoint_loss`
- `train_timewarp_loss`
- `train_timewarp_defect_loss`
- `train_timewarp_balance_loss`
- `val_loss`
- `val_pixel_loss`
- `val_perceptual_loss`
- `val_endpoint_loss`
- `val_timewarp_loss`
- `val_timewarp_defect_loss`
- `val_timewarp_balance_loss`

Current diagnostic fields:

- `train_t_mean`, `train_s_mean`, `train_delta_mean`
- `val_t_mean`, `val_s_mean`, `val_delta_mean`
- `train_target_build_sec`, `train_forward_sec`
- `train_endpoint_sec`, `train_timewarp_sec`
- `val_target_build_sec`, `val_forward_sec`
- `val_endpoint_sec`, `val_timewarp_sec`
- `train_update_ratio`, `train_update_cosine`
- `val_update_ratio`, `val_update_cosine`
- `target_builder`
- `target_construction`
- `target_source`
- `target_stop_grad`
- `bridge_source`
- `target_uses_dataset_images`

When timewarp is enabled, the trainer also logs:

- `timewarp_time_grid`
- `timewarp_delta_min`
- `timewarp_delta_max`
- `timewarp_delta_mean`
- `timewarp_delta_std`
- `timewarp_interval_defects`

## Current checkpoint-selection metric

The current trainer selects `best.pt` by:

- minimal `val_loss`

`val_loss` is exactly the weighted prediction objective used in `_run_epoch(...)`
for validation:

```text
val_loss = L_match + endpoint_weight * L_endpoint
```

When timewarp is enabled, the trainer also computes `val_timewarp_*` statistics,
but those are logged separately and are not added into `val_loss`. There is no
separate standalone `L_comp` term on the current line because CTM-style
composition is already baked into the target.

## Few-step evaluation metrics

The default evaluation runner writes:

- `reports/summary.json`
- `reports/summary.csv`
- `reports/best.json`

Per-step records include:

- `step_count`
- `nfe`
- `fid`
- `num_fid_samples`
- `fid_batch_size`
- `sample_batch_size`
- `elapsed_sec`
- `samples_per_sec`
- `timewarp_enabled`
- `time_grid`
- `delta_min`
- `delta_max`
- `delta_mean`
- `delta_std`

The current default eval config uses:

```yaml
step_counts: [1, 2, 4, 8, 16]
num_fid_samples: 50000
fid_batch_size: 128
sample_batch_size: 16
fixed_grid_size: 64
```

Stage-1 experiment commands often extend the inspected step list to:

```text
1, 2, 4, 8, 16, 32, 64, 128, 256
```

## Official-style external metrics

The current line also supports exported `.npz` evaluation through
`scripts/run_evaluate_metrics.py`.

Current official metrics set:

- `fid`
- `is`
- `precision`
- `recall`

Those are written to:

- `$METRIC_ROOT/official/step16_metrics.json`

## Held-out defect metric

The current line supports a separate held-out defect report through:

- `scripts/run_evaluate_defect.py`

The handoff treats this as an external validation path, separate from the
training-side timewarp surrogate.

Current report family:

- `defect_mean`
- `defect_by_t_bin`
- `defect_by_step_count`

## Numeric evidence already recorded in the handoff

The handoff currently records one confirmed stage-0 fullstack preflight epoch:

```text
train_loss=0.119102
val_loss=0.138993
pixel=0.003461
perc=0.005372
endpoint=0.457192
timewarp=0.003857
samples_per_sec=1.08
elapsed_sec=89.14
```

It also records that the trainer emitted:

- `target_builder=teacher_sampler`
- `construction=ctm_consistency`
- `target_source=ema_model`
- `bridge_source=ema_model_rollout`
- `target_uses_dataset_images=false`
- non-uniform `timewarp_time_grid`
- non-zero `val_timewarp_loss`

## Current practical decision metrics

Per the handoff and pipeline docs, the fields that matter most right now are:

1. few-step FID trend from `1 -> 16`
2. whether quality keeps improving beyond `16`
3. `train_update_ratio` / `val_update_ratio`
4. `train_update_cosine` / `val_update_cosine`
5. `train_target_build_sec` versus `train_forward_sec`
6. `timewarp_time_grid` and `timewarp_interval_defects` when warp is enabled
7. official `fid / is / precision / recall`
8. held-out `defect_mean / defect_by_t_bin / defect_by_step_count`
