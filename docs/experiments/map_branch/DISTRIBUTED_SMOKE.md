# Distributed Smoke

This is a minimal preflight for single-node, two-GPU distributed execution.

Goal:
- verify `cuda:0,1`
- verify `torchrun`
- verify NCCL process-group init
- verify `DistributedSampler`
- verify DDP forward/backward/optimizer step on the current explicit-map model

It does **not** start the full training pipeline.

## Command

```bash
cd /data2/yl7622/Zhengwei/DG-TWFD
conda activate /data2/yl7622/Zhengwei/DG-TWFD/.conda_envs/dgfm_map
source scripts/experiments/activate_fm_cifar10.sh fm_cifar10_map_branch_s0_a6000_fullstack_smoke e001a

CUDA_VISIBLE_DEVICES=0,1 torchrun \
  --standalone \
  --nproc_per_node=2 \
  scripts/run_ddp_smoke.py \
  --config $FM_CONFIG \
  --batch-size 8 \
  --steps 4 \
  --dataset-size 64 \
  --out $RUN_ROOT/logs/ddp_smoke.json
```

## Expected result

Rank 0 should print one json object containing:

- `success: true`
- `world_size: 2`
- `steps: 4`
- `loss_first`
- `loss_last`
- `losses`

The script should also write:

```bash
$RUN_ROOT/logs/ddp_smoke.json
```

## What to return

After running it on the server, return:

1. terminal output
2. `cat $RUN_ROOT/logs/ddp_smoke.json`
3. if it fails:
   - full traceback
   - `nvidia-smi`
