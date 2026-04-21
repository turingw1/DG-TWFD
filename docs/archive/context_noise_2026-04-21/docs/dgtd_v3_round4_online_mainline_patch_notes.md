# DGTD v3 Round-4 Online Mainline Patch Notes

This note records the round-4 patch that moves DGTD continuation onto the
online-teacher mainline without switching to an expensive full online local
solver rollout.

## 1. Why the previous online teacher was not real online continuation

Before this patch, the branch already supported:

- online teacher construction
- image dataloaders instead of shard dataloaders
- online trajectory materialization from real `x0` batches

But it did **not** support online continuation inside the DGTD residual.

The old behavior was:

1. the teacher built a trajectory online
2. DGTD interpolated `x_s_teacher` and `x_u_teacher` from that trajectory
3. `TeacherAdapter.local_flow(...)` still labeled the continuation as
   `cached_exact` or `cached_affine`
4. `continuation_sources.online` therefore stayed at `0.0`

So the old "online teacher" was only online at the trajectory/data layer, not at
the continuation layer.

## 2. What changed in round 4

Round 4 makes **online trajectory anchors** the primary DGTD continuation source.

When the current batch trajectory was generated online, the teacher continuation
is now constructed as:

```text
teacher_cont_online(z_s; s,u)
= x_u_teacher_online + alpha_online(s,u) * (z_s - x_s_teacher_online)
```

Properties:

- `x_s_teacher_online` and `x_u_teacher_online` remain teacher-owned anchors
- those anchors are detached
- but the continuation formula still depends on `z_s`
- so the bridge branch keeps gradient even when the teacher weight is high

This is the intended "online-conditioned affine/Jacobian-lite continuation"
mainline.

## 3. Files changed

Core implementation:

- `src/dgtd/sigma.py`
- `src/dgtd/cache.py`
- `src/dgtd/teacher.py`
- `src/dgtd/defect.py`
- `src/dgtd/train_dgtd.py`

Configs:

- `configs/experiment/dgtd_cifar10_v3.yaml`
- `configs/experiment/dgtd_cifar10_v3_smoke.yaml`

Tests:

- `tests/test_dgtd.py`

Documentation:

- `docs/dgtd_v3_round4_online_mainline_patch_notes.md`

## 4. Online continuation definition

### 4.1 Mainline online path

For online-generated trajectories, `TeacherAdapter.local_flow(...)` now:

1. detects that the trajectory anchors are online
2. interpolates `x_s_teacher_online` and `x_u_teacher_online`
3. applies the online affine continuation formula
4. returns `source=online`

This means:

- `continuation_sources.online` can now become non-zero
- and should become the dominant source when online teacher is enabled

### 4.2 Cached fallback

The cached paths are still present, but are now fallback behavior:

- `cached_exact`
- `cached_affine`
- `bootstrap`

They are used when the online trajectory-anchor path is not active or cannot be
constructed.

## 5. `alpha_online` definition

The continuation gain is now centralized in the sigma schedule helper.

Supported `alpha_mode` values:

- `ratio_sigma`
- `clamped_ratio_sigma`
- `identity`
- `power_ratio_sigma`

Default mainline config:

- `alpha_mode: clamped_ratio_sigma`
- `alpha_min: 0.05`
- `alpha_max: 1.0`
- `alpha_power: 1.0`

Default formula:

```text
alpha_online(s,u) = clip(sigma(u) / max(sigma(s), eps), alpha_min, alpha_max)
```

This is intentionally more stable than an unconstrained ratio near the clean end.

## 6. Warmup / `eta` change

The warmup schedule no longer starts at `eta=1.0`.

New mainline default:

- `eta_start: 0.95`
- warmup uses `eta = eta_start`
- adaptive stage interpolates from `eta_start` down to `eta_min`

Why this is better than only lowering `eta`:

- the real fix is that online continuation itself now depends on `z_s`
- so the bridge branch keeps gradient even when teacher influence is strong
- lowering `eta` alone would preserve bridge gradient only by weakening the
  teacher
- the new design keeps the run teacher-dominant **and** bridge-trainable

## 7. Continuation source semantics after the patch

The meanings are now:

- `online`
  - continuation built from online-generated trajectory anchors using the
    affine/Jacobian-lite formula
- `cached_exact`
  - detached exact anchor from fallback trajectory/shard mode
- `cached_affine`
  - fallback affine continuation from fallback trajectory/shard mode
- `bootstrap`
  - no teacher continuation available; continuation is student-only

This removes the old ambiguity where online-generated trajectories still looked
like cached continuation in logs.

## 8. New logging / diagnostics

Added or clarified fields:

- `online_anchor_used_rate`
- `online_continuation_rate`
- `cached_fallback_rate`
- `teacher_rel_error_mean`
- `exact_mask_hit_rate`
- `alpha_online_mean`
- `alpha_online_min`
- `alpha_online_max`
- `train_online_teacher_traj_sec`

Also clarified the old bridge naming:

- old `bridge_teacher_error` semantics were `MSE(x_s_pred, x_s_teacher)`
- this patch renames that to `bridge_state_teacher_error`
- and also logs `bridge_u_teacher_error = MSE(bridge_cont, x_u_teacher)`

## 9. Smoke and short-run commands

### 9.1 Train smoke

```bash
python scripts/run_train.py \
  --config configs/experiment/dgtd_cifar10_v3_smoke.yaml \
  --run-root "$DGTD_V3_R4_ROOT" \
  --set teacher.local_files_only=false \
  2>&1 | tee "$DGTD_V3_R4_ROOT/train.stdout_stderr.txt"
```

Expected key indicators:

- `online_teacher_data=true`
- `continuation_sources.online > 0`
- `online_continuation_rate > 0`

### 9.2 Sample smoke

```bash
python scripts/run_sample_dgtd.py \
  --config configs/experiment/dgtd_cifar10_v3_smoke.yaml \
  --checkpoint "$DGTD_V3_R4_ROOT/checkpoints/last.pt" \
  --output-dir "$DGTD_V3_R4_ROOT/sample" \
  --steps 4
```

### 9.3 Eval smoke

```bash
python scripts/run_eval.py \
  --config configs/experiment/dgtd_cifar10_v3_smoke.yaml \
  --checkpoint "$DGTD_V3_R4_ROOT/checkpoints/last.pt" \
  --eval-root "$DGTD_V3_R4_ROOT/eval" \
  --steps 1 2 4
```

### 9.4 Suggested short run

```bash
python scripts/run_train.py \
  --config configs/experiment/dgtd_cifar10_v3.yaml \
  --run-root "$DGTD_V3_R4_SHORT" \
  --set train.epochs=2 \
  --set train.batch_size=32 \
  --set train.max_train_batches=64 \
  --set train.max_val_batches=16 \
  --set teacher.local_files_only=false
```

For the short run, check:

- `continuation_sources.online`
- `online_continuation_rate`
- `bridge_state_teacher_error`
- `bridge_u_teacher_error`
- `direct_bridge_gap`
- `alpha_online_mean`
- `entropy_q_phi`
- `kl_qD_qphi`

## 10. Local verification run in this patch round

I did not run full train/sample/eval locally in this patch round.

I did run:

```bash
python -m py_compile src/dgtd/teacher.py src/dgtd/sigma.py src/dgtd/cache.py src/dgtd/defect.py src/dgtd/train_dgtd.py tests/test_dgtd.py
pytest tests/test_dgtd.py -q
```

Result:

- compile passed
- `14 passed`
