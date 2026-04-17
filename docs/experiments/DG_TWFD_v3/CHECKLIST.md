# DG_TWFD_v3 Checklist

Status labels:

- `[done]` implemented locally and covered by targeted local checks
- `[partial]` code exists but still needs server validation or follow-up implementation
- `[todo]` not finished yet
- `[blocked]` waiting on server results or user decision

## Core implementation

- `[done]` unified `dgtd_map` training objective is wired into trainer dispatch
- `[done]` `dgtd/warp.py`, `defect.py`, `metrics.py`, `cache.py`, `train_dgtd.py`, `sample_dgtd.py` exist
- `[partial]` `TeacherAdapter.local_flow()` only supports cached-near-state fallback plus detached student bootstrap
- `[todo]` optional online teacher one-step local solver hook is not implemented

## Experiment configs

- `[done]` main config: `configs/experiment/dgtd_cifar10_v3.yaml`
- `[done]` smoke config: `configs/experiment/dgtd_cifar10_v3_smoke.yaml`
- `[done]` ablation config B: unified DGTD without learned warp
- `[done]` ablation config C: DGTD + learned warp, no HF metric
- `[done]` ablation config D: DGTD + learned warp + HF metric
- `[partial]` ablation E is currently a runtime eval/sample override, not a standalone train config

## Diagnostics

- `[done]` basic diagnostics script exists: `scripts/plot_dgtd_diagnostics.py`
- `[partial]` density/bin plots exist
- `[todo]` defect heatmap over time bins
- `[todo]` loss-by-bin plot
- `[todo]` schedule panel for `K=1,2,4,8,16`

## Validation

- `[done]` targeted local tests for DGTD config/warp/cache dispatch
- `[blocked]` smoke train on real trajectory cache
- `[blocked]` smoke sample from trained checkpoint
- `[blocked]` smoke eval on trained checkpoint
- `[blocked]` ablation smoke comparison on server

## First server round

Run these and send back the command outputs plus the generated `logs/train.jsonl` tail and sample directory listing.

1. Smoke train:
   `python scripts/run_train.py --config configs/experiment/dgtd_cifar10_v3_smoke.yaml --run-root /tmp/dgtd_v3_smoke --set target.shard_root=$TRAJ_ROOT`
2. Smoke sample:
   `python scripts/run_sample_dgtd.py --config configs/experiment/dgtd_cifar10_v3_smoke.yaml --checkpoint /tmp/dgtd_v3_smoke/checkpoints/last.pt --output-dir /tmp/dgtd_v3_smoke/sample --steps 4 --set target.shard_root=$TRAJ_ROOT`
3. Smoke eval:
   `python scripts/run_eval.py --config configs/experiment/dgtd_cifar10_v3_smoke.yaml --checkpoint /tmp/dgtd_v3_smoke/checkpoints/last.pt --eval-root /tmp/dgtd_v3_smoke/eval --steps 1 2 4 --set target.shard_root=$TRAJ_ROOT`
