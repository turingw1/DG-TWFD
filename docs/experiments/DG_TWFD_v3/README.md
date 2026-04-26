# DG_TWFD_v3

This directory is the branch-facing documentation entry for the active
`DG_TWFD_v3` implementation and EDM-first experiment track.

Use [DOCS_REGISTRY.md](DOCS_REGISTRY.md) as the canonical documentation map.
Dated filenames are evidence snapshots, not the primary versioning mechanism.

## Scope

- algorithm:
  - unified Defect-Guided Trajectory Distillation on explicit map prediction
- active objective:
  - EDM-first continuous sigma-space distillation in `experiments/edm_first/`
- active time warp:
  - retained as a required architecture component, but `e504a` is the current
    no-warp baseline for one-step endpoint quality
- active teacher source:
  - official EDM CIFAR-10 teacher checkpoint for the current track
- preserved infrastructure:
  - dataset loading
  - distributed/server workflow
  - evaluation and few-step sampling
- current experiment priority:
  - supervise `e504a` until the 50% one-step FID threshold is hit or the trend
    clearly fails
  - compare timewarp against identity on future timewarp-enabled runs
  - keep DDPM/DGTD as paused evidence unless explicitly revisited

## Reading order

1. [../../ACTIVE_CONTEXT.md](../../ACTIVE_CONTEXT.md)
2. [DOCS_REGISTRY.md](DOCS_REGISTRY.md)
3. [EDM_FIRST_SUPERVISION.md](EDM_FIRST_SUPERVISION.md)
4. [NETWORK_AND_RECOVERY.md](NETWORK_AND_RECOVERY.md)
5. [BASELINE_STATUS.md](BASELINE_STATUS.md)
6. [BASELINE_COMPARISON_GUIDE.md](BASELINE_COMPARISON_GUIDE.md)

## Documentation roles

Use [DOCS_REGISTRY.md](DOCS_REGISTRY.md) for the full classification. The
short version is:

- status dashboards:
  - [EDM_FIRST_SUPERVISION.md](EDM_FIRST_SUPERVISION.md)
  - [BASELINE_STATUS.md](BASELINE_STATUS.md)
- operations and recovery:
  - [NETWORK_AND_RECOVERY.md](NETWORK_AND_RECOVERY.md)
  - [A100_SERVER_DEPLOYMENT_2026-04-25.md](A100_SERVER_DEPLOYMENT_2026-04-25.md)
- algorithm evidence/reference:
  - [DDPM_TEACHER_SUITABILITY_2026-04-26.md](DDPM_TEACHER_SUITABILITY_2026-04-26.md)
  - [TIME_COORDINATE_DESIGN_ABLATION.md](TIME_COORDINATE_DESIGN_ABLATION.md)
  - [ARCHITECTURE_AND_IMPLEMENTATION.md](ARCHITECTURE_AND_IMPLEMENTATION.md)
- planning:
  - [PAPER_EXPERIMENT_TARGETS.md](PAPER_EXPERIMENT_TARGETS.md)
  - [PLAN/impreved_to_cinsistent.md](PLAN/impreved_to_cinsistent.md)
- superseded handoff:
  - [HANDOFF_2026-04-20.md](HANDOFF_2026-04-20.md)

Archived context lives under `docs/archive/` and is not part of the default
reading set. Do not read archived docs unless a specific historical lookup or
baseline reproduction request requires it.

## Code And History Reading Map

If the goal is to re-open the implementation and algorithm lineage rather than
re-read branch summaries, use these entry documents directly:

- paused DGTD branch docs:
  - [HANDOFF_2026-04-20.md](HANDOFF_2026-04-20.md)
  - [ARCHITECTURE_AND_IMPLEMENTATION.md](ARCHITECTURE_AND_IMPLEMENTATION.md)
  - [PIPELINE.md](PIPELINE.md)
- A100/server runtime layout:
  - [A100_SERVER_DEPLOYMENT_2026-04-25.md](A100_SERVER_DEPLOYMENT_2026-04-25.md)
- network and crash recovery:
  - [NETWORK_AND_RECOVERY.md](NETWORK_AND_RECOVERY.md)
- CTM baseline code and commands:
  - [../../../refs/ctm/README.md](../../../refs/ctm/README.md)
  - [../../archive/low_signal_2026-04-25/baseline/reproduce_ctm_baselines.sh](../../archive/low_signal_2026-04-25/baseline/reproduce_ctm_baselines.sh)
  - [../../archive/low_signal_2026-04-25/baseline/reproduce_ctm_imagenet64_conda.sh](../../archive/low_signal_2026-04-25/baseline/reproduce_ctm_imagenet64_conda.sh)
- consistency distillation lineage:
  - [../../../refs/consistency_models/README.md](../../../refs/consistency_models/README.md)
  - [../../../refs/consistency_models/cm/karras_diffusion.py](../../../refs/consistency_models/cm/karras_diffusion.py)
  - [../../../refs/consistency_models/cm/train_util.py](../../../refs/consistency_models/cm/train_util.py)
- EDM teacher and sampler lineage:
  - [../../../refs/edm/README.md](../../../refs/edm/README.md)
  - [../../../refs/edm/generate.py](../../../refs/edm/generate.py)
  - [../../../refs/edm/fid.py](../../../refs/edm/fid.py)
  - [../../../refs/edm/experiments/dg_twfd_teacher_proxy/README.md](../../../refs/edm/experiments/dg_twfd_teacher_proxy/README.md)
  - [../../../refs/edm/experiments/dg_twfd_timewarp_analysis/README.md](../../../refs/edm/experiments/dg_twfd_timewarp_analysis/README.md)

## Active code entrypoints

- EDM-first trainer:
  - [../../../experiments/edm_first/train_edm_map.py](../../../experiments/edm_first/train_edm_map.py)
- EDM-first evaluator:
  - [../../../experiments/edm_first/eval_edm_map.py](../../../experiments/edm_first/eval_edm_map.py)
- EDM-first watcher:
  - [../../../experiments/edm_first/scripts/watch_eval_checkpoints.sh](../../../experiments/edm_first/scripts/watch_eval_checkpoints.sh)
- EDM-first threshold check:
  - [../../../experiments/edm_first/scripts/check_fid_thresholds.py](../../../experiments/edm_first/scripts/check_fid_thresholds.py)
- current e504a config:
  - [../../../experiments/edm_first/configs/cifar10_edm_map_onestep_prior_msdefect_8h.yaml](../../../experiments/edm_first/configs/cifar10_edm_map_onestep_prior_msdefect_8h.yaml)

## Paused DGTD code entrypoints

- trainer:
  - [../../../src/dgtd/train_dgtd.py](../../../src/dgtd/train_dgtd.py)
- warp:
  - [../../../src/dgtd/warp.py](../../../src/dgtd/warp.py)
- defect:
  - [../../../src/dgtd/defect.py](../../../src/dgtd/defect.py)
- cache:
  - [../../../src/dgtd/cache.py](../../../src/dgtd/cache.py)
- sampling:
  - [../../../src/dgtd/sample_dgtd.py](../../../src/dgtd/sample_dgtd.py)
- eval dispatch:
  - [../../../src/dgfm/evaluators/common.py](../../../src/dgfm/evaluators/common.py)
- EDM teacher-side experiment roots:
  - [../../../refs/edm/experiments/dg_twfd_teacher_proxy](../../../refs/edm/experiments/dg_twfd_teacher_proxy)
  - [../../../refs/edm/experiments/dg_twfd_timewarp_analysis](../../../refs/edm/experiments/dg_twfd_timewarp_analysis)

## Active commands

Use [EDM_FIRST_SUPERVISION.md](EDM_FIRST_SUPERVISION.md) for the current
training/eval monitoring commands. Do not use the old DGTD train/sample/eval
commands unless the DDPM/DGTD route is explicitly resumed.
