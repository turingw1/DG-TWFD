# Teacher Entry

This folder documents the teacher path that is currently active for the
`dgfm` map-branch codebase.

## Current online entrypoint

The active online teacher path is:

- target builder: `src/dgfm/targets/builder.py`
- teacher factory: `src/dgfm/teachers/factory.py`
- diffusers backend: `src/dgfm/teachers/diffusers_ddpm.py`

The default config chain is:

- experiment root: `configs/experiment/fm_cifar10_map_branch.yaml`
- target include: `configs/target/teacher_sampler_online.yaml`
- teacher include: `configs/teacher/sampler.yaml`

That means the default map-branch teacher call path is:

1. `TeacherSamplerTargetBuilder.build_from_batch(...)`
2. `build_teacher(config)`
3. `DiffusersDDPMTeacher.sample_trajectory_from_x0(...)`
4. sample CTM-style `(t, t_dt, s)` triplets on the retained teacher grid

## Minimal training-side call

The current teacher sampler is exercised indirectly through map-branch
training. The smallest committed config that still uses this path is:

```bash
python scripts/run_train.py \
  --config configs/experiment/fm_cifar10_map_branch_s1_smoke_base.yaml \
  --run-root /tmp/dgfm_smoke \
  --verbose
```

The full server-side workflow currently used in the handoff is:

```bash
source scripts/experiments/activate_fm_cifar10.sh fm_cifar10_map_branch_s1_e6_budget_full e602a
CUDA_VISIBLE_DEVICES=${TRAIN_CUDA_VISIBLE_DEVICES} torchrun \
  --standalone \
  --nnodes=$NNODES \
  --nproc_per_node=$NPROC_PER_NODE \
  --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  scripts/run_train.py \
  --config $FM_CONFIG \
  --run-root $RUN_ROOT \
  --verbose
```

## Offline shard generation

The map-branch also supports precomputing teacher trajectories into shard
files. The current writer is:

- `scripts/prepare_teacher_trajectories.py`

Example:

```bash
python scripts/prepare_teacher_trajectories.py \
  --config configs/experiment/fm_cifar10_map_branch.yaml \
  --output-root /tmp/cifar10_ddpm128_p33 \
  --batch-size 64
```

This produces:

- `train/train_00000.pt`, `train/manifest.yaml`, ...
- `val/val_00000.pt`, `val/manifest.yaml`, ...

To consume those shards instead of the online sampler, switch the target include
to `configs/target/teacher_trajectory.yaml` and point `target.shard_root` at
the generated folder.

## Time semantics

The current map-branch teacher uses `u` on the public API:

- `u=0.0`: noisiest state
- `u=1.0`: cleanest state

The internal DDPM teacher keeps the usual diffusion-style orientation:

- `tau=1.0`: noisiest state
- `tau=0.0`: cleanest state

So the adapter always uses:

- `tau = 1 - u`

Keep this convention fixed across any later doc or code updates on the current
line.
