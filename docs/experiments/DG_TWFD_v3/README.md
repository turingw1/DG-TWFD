# DG_TWFD_v3

This directory is the branch-facing documentation entry for the active
`DG_TWFD_v3` implementation.

## Scope

- algorithm:
  - unified Defect-Guided Trajectory Distillation on explicit map prediction
- active objective:
  - `train.objective=dgtd_map`
- active time warp:
  - `scheduler.timewarp.type=dgtd_density`
- active teacher source:
  - online teacher data and online trajectory anchors
  - cached trajectory continuation only as fallback
- preserved infrastructure:
  - dataset loading
  - distributed/server workflow
  - evaluation and few-step sampling

## Reading order

1. [../../ACTIVE_CONTEXT.md](../../ACTIVE_CONTEXT.md)
2. [HANDOFF_2026-04-20.md](HANDOFF_2026-04-20.md)
3. [PIPELINE.md](PIPELINE.md)
4. [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md)
5. [PAPER_EXPERIMENT_TARGETS.md](PAPER_EXPERIMENT_TARGETS.md)
6. [ARCHITECTURE_AND_IMPLEMENTATION.md](ARCHITECTURE_AND_IMPLEMENTATION.md)
7. [A100_MIGRATION_2026-04-25.md](A100_MIGRATION_2026-04-25.md)

## Documentation roles

- `HANDOFF_2026-04-20.md`:
  - current project state, rules, and implementation lineage
- `PIPELINE.md`:
  - stable server command families after experiment activation
- `EXPERIMENT_LOG.md`:
  - accepted experiment rows, tags, commands, and return fields
- `PAPER_EXPERIMENT_TARGETS.md`:
  - paper-facing experiment tables and baseline data plan
- `ARCHITECTURE_AND_IMPLEMENTATION.md`:
  - implementation details for the active DGTD stack
- `A100_MIGRATION_2026-04-25.md`:
  - server migration checklist, pinned revisions, and runtime asset gaps

Archived context lives under `docs/archive/context_noise_2026-04-21/` and is
not part of the default reading set.

## Code And History Reading Map

If the goal is to re-open the implementation and algorithm lineage rather than
re-read branch summaries, use these entry documents directly:

- active DG-TWFD branch docs:
  - [HANDOFF_2026-04-20.md](HANDOFF_2026-04-20.md)
  - [ARCHITECTURE_AND_IMPLEMENTATION.md](ARCHITECTURE_AND_IMPLEMENTATION.md)
  - [PIPELINE.md](PIPELINE.md)
- A100/server migration and missing runtime assets:
  - [A100_MIGRATION_2026-04-25.md](A100_MIGRATION_2026-04-25.md)
- CTM baseline code and commands:
  - [../../../refs/ctm/README.md](../../../refs/ctm/README.md)
  - [../../../docs/baseline/reproduce_ctm_baselines.sh](../../../docs/baseline/reproduce_ctm_baselines.sh)
  - [../../../docs/baseline/reproduce_ctm_imagenet64_conda.sh](../../../docs/baseline/reproduce_ctm_imagenet64_conda.sh)
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

- train:
  - `torchrun --nproc_per_node=2 scripts/run_train.py --config configs/experiment/dgtd_cifar10_v3.yaml --run-root runs/dgtd_cifar10_v3`
- sample:
  - `python scripts/run_sample_dgtd.py --config configs/experiment/dgtd_cifar10_v3.yaml --checkpoint runs/dgtd_cifar10_v3/checkpoints/best.pt --output-dir runs/dgtd_cifar10_v3/samples/steps16 --steps 16`
- eval:
  - `python scripts/run_eval.py --config configs/experiment/dgtd_cifar10_v3.yaml --checkpoint runs/dgtd_cifar10_v3/checkpoints/best.pt --eval-root eval/dgtd_cifar10_v3`
