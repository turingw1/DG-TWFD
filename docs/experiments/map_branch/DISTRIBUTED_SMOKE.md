# Distributed Smoke

This is the required preflight before formal two-A6000 training runs on the
server branch.

Goal:
- verify `cuda:0,1`
- verify `torchrun`
- verify NCCL process-group init
- verify `DistributedSampler`
- verify DDP forward/backward/optimizer step on the current explicit-map model

It does **not** start the full teacher-backed training pipeline. Its job is to
verify that `torchrun + NCCL + DDP` can initialize cleanly on GPUs `0,1`.

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

After this passes, formal training should use the distributed train commands in
[A100_PIPELINE.md](/home/gzwlinux/vscode/gitProject/DG-TWFD/docs/experiments/map_branch/A100_PIPELINE.md),
while eval and panel stay single-GPU.

## What to return

After running it on the server, return:

1. terminal output
2. `cat $RUN_ROOT/logs/ddp_smoke.json`
3. if it fails:
   - full traceback
   - `nvidia-smi`
