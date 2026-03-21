# Refactor Master Plan

## Executive Decision

### Decision
Heavily rewrite the training and evaluation surface, but partially reuse the existing core where it is already aligned with the target method.

### Preserve
- `flow_matching/flow_matching/path/*`: probability-path abstraction
- `flow_matching/flow_matching/path/scheduler/*`: scheduler abstraction and schedule transform
- `flow_matching/flow_matching/solver/*`: continuous/discrete solver abstraction
- `flow_matching/flow_matching/utils/model_wrapper.py`: model wrapper contract
- `src/dg_twfd/data/teacher.py`: practical external teacher integration logic
- `scripts/experiments/activate_ddpm_cifar10.sh`: user-facing experiment activation philosophy
- `A100_DDPM_CIFAR10_EXPERIMENT_PIPELINE.md`: workflow style only, not architecture

### Refactor First
- Monolithic entrypoints: `train.py`, `sample.py`
- Coupled trainer logic: `src/dg_twfd/engine/trainer.py`
- Scattered evaluation logic: `sample.py`, `scripts/profile_infer.py`, `scripts/preview_samples.py`
- Ad hoc config layering: `config/default.yaml`, `config/profiles/*.yaml`

### Explicitly Deprecate
- Phase-1-specific DG-TWFD naming as the long-term top-level training abstraction
- Evaluation through standalone sampling scripts only
- Distillation-first assumptions in the core training stack
- Experiment logic embedded in markdown rather than in reusable scripts/modules

## New Target Architecture
The target framework is baseline-first and flow-matching-centered.

### Core boundaries
- `datasets`: train/val/test construction, split policy, transforms, shard datasets, image-folder datasets
- `models`: velocity backbones, wrappers, EMA, optional class-conditioning adapters
- `paths`: `ProbPath`-style objects that construct `x_t` and conditional supervision targets
- `schedulers`: continuous-time schedule families and future learnable time-warp wrappers
- `solvers`: ODE/discrete solvers and unified few-step grids
- `trainers`: one reusable training driver with checkpointing, logging, AMP, DDP, EMA
- `evaluators`: first-class FID/KID/IS/sample-grid/checkpoint-eval subsystem
- `teachers`: pluggable teacher API; baseline runs with `NullTeacher`
- `pipelines`: high-level experiment recipes that map a profile to train/eval/sample flows
- `configs`: one base config + dataset/model/train/eval/teacher overlays
- `scripts`: thin CLI entrypoints only
- `utils`: environment, logging, seed, distributed, filesystem, reporting helpers

### Design rule
The core object in Phase 1 remains a velocity field trained on a probability path. Time-warp is exposed as a scheduler-level hook. Semigroup defect is deferred until a clean explicit `M(t,s,x)` abstraction is available.

## Target Directory Structure

```text
.
├── configs/
│   ├── base.yaml
│   ├── dataset/
│   │   ├── cifar10.yaml
│   │   └── imagenet32.yaml
│   ├── model/
│   │   ├── unet_fm.yaml
│   │   └── dit_fm.yaml
│   ├── path/
│   │   ├── condot.yaml
│   │   └── ot.yaml
│   ├── scheduler/
│   │   ├── condot.yaml
│   │   ├── polynomial.yaml
│   │   └── timewarp_hook.yaml
│   ├── teacher/
│   │   ├── none.yaml
│   │   ├── pretrained.yaml
│   │   └── sampler.yaml
│   ├── eval/
│   │   ├── baseline.yaml
│   │   └── few_step.yaml
│   └── experiment/
│       ├── fm_cifar10_baseline.yaml
│       ├── fm_cifar10_stable.yaml
│       └── fm_imagenet32_baseline.yaml
├── src/dgfm/
│   ├── config/
│   ├── datasets/
│   ├── models/
│   ├── paths/
│   ├── schedulers/
│   ├── solvers/
│   ├── teachers/
│   ├── trainers/
│   ├── evaluators/
│   ├── pipelines/
│   └── utils/
├── scripts/
│   ├── run_train.py
│   ├── run_eval.py
│   ├── run_sample.py
│   ├── build_dataset.py
│   ├── collect_teacher.py
│   └── experiments/
├── docs/
│   ├── REFACTOR_MASTER_PLAN.md
│   ├── FLOW_MATCHING_BASELINE_PIPELINE.md
│   ├── EVALUATION_SYSTEM_PLAN.md
│   └── DATASET_AND_TEACHER_INTERFACE_PLAN.md
├── outputs/
│   ├── runs/
│   ├── eval/
│   ├── samples/
│   └── reports/
└── legacy/
    └── docs_and_phase1_notes/
```

Notes:
- Keep `flow_matching/` vendored as a reference subtree, not the runtime package boundary.
- Keep current `src/dg_twfd/` during transition; phase out by moving reusable pieces into `src/dgfm/`.

## Module Migration Table

| Current path | Role today | Action | Notes |
|---|---|---|---|
| `flow_matching/flow_matching/path/*` | Clean FM path abstractions | Reuse directly | Primary source for new `paths/` |
| `flow_matching/flow_matching/path/scheduler/*` | Schedulers + schedule transform | Reuse directly | Core source for future time-warp hook |
| `flow_matching/flow_matching/solver/*` | ODE/discrete solvers | Reuse directly | Wrap with unified evaluation API |
| `flow_matching/examples/image/train.py` | Baseline image training example | Rewrite | Good reference, poor long-term infra |
| `flow_matching/examples/image/training/*` | Example train/eval loop | Wrap then rewrite | Borrow FM loss flow only |
| `flow_matching/examples/text/*` | Hydra text example | Borrow ideas only | Not a direct base for image pipeline |
| `src/dg_twfd/data/teacher.py` | Teacher interface prototype | Wrap | Seed for new `teachers/` module |
| `src/dg_twfd/data/dataset.py` | Trajectory shard dataset | Wrap then split | Move shard-specific logic to datasets subsystem |
| `src/dg_twfd/data/dataloader.py` | Dataloader builder | Rewrite | Fold into dataset factory |
| `src/dg_twfd/models/student.py` | Conv student | Wrap | Can become one baseline backbone |
| `src/dg_twfd/models/student_dit.py` | DiT-style student | Wrap | Optional future backbone |
| `src/dg_twfd/models/timewarp.py` | Monotone timewarp network | Wrap | Keep as future scheduler-level hook |
| `src/dg_twfd/models/boundary.py` | Boundary corrector | Wrap | Keep as optional wrapper module |
| `src/dg_twfd/losses/*` | Match/defect/warp/boundary losses | Rewrite selectively | Keep only after core map abstractions mature |
| `src/dg_twfd/engine/trainer.py` | Monolithic trainer | Rewrite | Replace with baseline-first trainer + evaluator split |
| `src/dg_twfd/engine/checkpoint.py` | Checkpoint utilities | Wrap | Reuse checkpoint IO pieces |
| `src/dg_twfd/infer/sampler.py` | Custom student sampler | Deprecate for Phase 1 | Replace with solver-driven baseline sampling |
| `src/dg_twfd/infer/schedules.py` | Step schedule helpers | Wrap | Move under evaluators/solvers |
| `train.py` | Entry script | Deprecate | Replace with `scripts/run_train.py` |
| `sample.py` | Entry script | Deprecate | Replace with `scripts/run_sample.py` |
| `scripts/profile_infer.py` | Ad hoc profiling | Rewrite | Fold into evaluators/reporting |
| `scripts/preview_samples.py` | Ad hoc visualization | Wrap | Fold into evaluator visualization stage |
| `A100_DDPM_CIFAR10_EXPERIMENT_PIPELINE.md` | Current workflow manual | Preserve style, rewrite content | Replace with FM baseline pipeline |
| `config/default.yaml` | Flat phase-1 config | Deprecate gradually | Split into composable config groups |
| `config/profiles/*.yaml` | Experiment profiles | Rewrite | Keep names traceable during migration |

## Implementation Phases

### Phase 0: Repo Audit and Freeze Old Scripts
Tasks:
- Freeze current `train.py`, `sample.py`, `src/dg_twfd/*` as legacy baseline
- Add explicit deprecation notes to legacy entrypoints
- Record old-to-new module mapping
- Separate reference trees: `flow_matching/`, `ctm/`, `legacy DG-TWFD`

Expected outputs:
- migration table
- legacy freeze note
- target directory skeleton

Risks:
- accidental breakage of current A100 workflow
- mixed ownership between legacy and new codepaths

Validation:
- current legacy commands still runnable
- new docs clearly distinguish legacy vs target framework

### Phase 1: Clean Baseline Training Path
Tasks:
- Create `src/dgfm/` package
- Implement unified config loading
- Implement baseline trainer around FM velocity supervision
- Implement CIFAR-10 dataset module
- Implement baseline image model factory
- Implement solver-backed sampling entrypoint
- Add one activation workflow matching old pipeline style

Expected outputs:
- baseline CIFAR-10 train/eval/sample path
- single launch command family
- outputs saved under one run root

Risks:
- overfitting the architecture to CIFAR-10 only
- copying example logic without cleaning interfaces

Validation:
- train from scratch on CIFAR-10
- resume training from checkpoint
- sample from checkpoint
- run 1/2/4/8/16-step evaluation from a single CLI

### Phase 2: Full Evaluation System
Tasks:
- Implement evaluator registry
- Add FID as mandatory metric
- Add optional KID/IS
- Add fixed-seed sample grids
- Add checkpoint sweep evaluation
- Add result table export and caching

Expected outputs:
- `run_eval.py`
- `outputs/eval/<exp>/<ckpt>/<nfe>/...`
- aggregate CSV/JSON/Markdown report

Risks:
- duplicated sample generation cost
- mismatch between training-time and eval-time preprocessing

Validation:
- one command evaluates one checkpoint over 1/2/4/8/16 steps
- cache prevents duplicate recomputation
- report table generated deterministically

### Phase 3: Dataset Expansion and Teacher Interface
Tasks:
- Add ImageNet32/ImageNet64 dataset configs
- Add `Teacher` interface and `NullTeacher`
- Add pretrained/sampler-based teacher stubs
- Support teacher-aware training hooks without changing baseline path

Expected outputs:
- teacher registry
- dataset registry
- baseline and teacher-based configs share trainer

Risks:
- coupling teacher logic back into trainer core
- data-root inconsistency across local/A100 environments

Validation:
- baseline runs with `NullTeacher`
- teacher configs load without code changes
- ImageNet dry-run constructs loaders and run paths correctly

### Phase 4: Future-Ready Hooks for Time-Warp
Tasks:
- Insert scheduler wrapper hook into config and trainer
- Add learnable or analytic time-reparameterization interface
- Expose few-step grids via warped schedule generation

Expected outputs:
- `schedulers/timewarp.py`
- optional `model_wrapper` or scheduler wrapper integration

Risks:
- applying time-warp at inference only and invalidating training objective
- schedule state leaking into evaluator only

Validation:
- baseline runs with identity warp
- time-warp config can be enabled without trainer rewrite
- evaluation compares warped vs unwarped few-step curves

### Phase 5: Future-Ready Hooks for Semigroup Defect
Tasks:
- Define explicit `FlowMap` abstraction `M(t,s,x)`
- Add defect loss module operating on map compositions
- Add teacher-map or solver-integrated targets as needed

Expected outputs:
- map-level trainer extension
- no rewrite of dataset/evaluator/teacher config system

Risks:
- expensive map evaluation if implemented naively via ODE solves
- abstraction conflict between velocity fields and learned maps

Validation:
- defect module plugs into trainer as an optional objective
- baseline path remains unchanged when defect is disabled
