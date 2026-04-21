# DG_TWFD_v3 Checklist

This checklist tracks only implementation and result-facing milestones.

Intermediate smoke commands, deployment-side checks, and temporary validation
instructions live in:

- [DEVELOPMENT_VALIDATION.md](/home/gzwlinux/vscode/gitProject/DG-TWFD/docs/experiments/DG_TWFD_v3/DEVELOPMENT_VALIDATION.md)

Status labels:

- `[done]` implemented locally and covered by targeted local checks
- `[partial]` code exists but still needs server validation or follow-up implementation
- `[todo]` not finished yet
- `[blocked]` waiting on server results or user decision

## Milestone A: Runnable DGTD stack

- `[done]` unified `dgtd_map` training objective is wired into trainer dispatch
- `[done]` `dgtd/warp.py`, `defect.py`, `metrics.py`, `cache.py`, `train_dgtd.py`, `sample_dgtd.py` exist
- `[partial]` `TeacherAdapter.local_flow()` only supports cached-near-state fallback plus detached student bootstrap
- `[todo]` optional online teacher one-step local solver hook is not implemented

## Milestone B: Experiment configs

- `[done]` main config: `configs/experiment/dgtd_cifar10_v3.yaml`
- `[done]` smoke config: `configs/experiment/dgtd_cifar10_v3_smoke.yaml`
- `[done]` ablation config B: unified DGTD without learned warp
- `[done]` ablation config C: DGTD + learned warp, no HF metric
- `[done]` ablation config D: DGTD + learned warp + HF metric
- `[partial]` ablation E is currently a runtime eval/sample override, not a standalone train config

## Milestone C: Diagnostics and visibility

- `[done]` basic diagnostics script exists: `scripts/plot_dgtd_diagnostics.py`
- `[partial]` density/bin plots exist
- `[todo]` defect heatmap over time bins
- `[todo]` loss-by-bin plot
- `[todo]` schedule panel for `K=1,2,4,8,16`

## Milestone D: Server validation

- `[done]` targeted local tests for DGTD config/warp/cache dispatch
- `[blocked]` smoke train on real trajectory cache
- `[blocked]` smoke sample from trained checkpoint
- `[blocked]` smoke eval on trained checkpoint
- `[blocked]` ablation smoke comparison on server

## Milestone E: Final experiment package

- `[todo]` final result-oriented pipeline doc
- `[todo]` final result-oriented experiment log
- `[todo]` final ablation table with runtime and quality summary
