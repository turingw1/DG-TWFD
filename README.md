# DG_TWFD_v3

This branch currently tracks an EDM-first continuous distillation route while
keeping the earlier DGTD/DDPM implementation available as reference evidence.

## Active scope

- keep the reusable `dgfm` stack for datasets, distributed training, sampling,
  evaluation, and server workflow
- current active experiments live under `experiments/edm_first/`
- keep time warp as a required final architecture component, but use `e504a` as
  the current no-warp one-step baseline
- keep the current project time convention:
  - `0 <= t < s < u <= 1`
  - `0.0` is noisy and `1.0` is clean
- keep the old DGTD/DDPM path paused unless explicitly revisited

## Start here

- active context:
  - [docs/ACTIVE_CONTEXT.md](docs/ACTIVE_CONTEXT.md)
- branch overview:
  - [docs/experiments/DG_TWFD_v3/README.md](docs/experiments/DG_TWFD_v3/README.md)
- documentation registry:
  - [docs/experiments/DG_TWFD_v3/DOCS_REGISTRY.md](docs/experiments/DG_TWFD_v3/DOCS_REGISTRY.md)
- current EDM-first supervision:
  - [docs/experiments/DG_TWFD_v3/EDM_FIRST_SUPERVISION.md](docs/experiments/DG_TWFD_v3/EDM_FIRST_SUPERVISION.md)

Historical round notes, old `map_branch` docs, and superseded reconstruction
prompts are archived under `docs/archive/context_noise_2026-04-21/`. Do not
read that archive during normal development unless a specific historical lookup
is required.

## Main entrypoints

- EDM-first train:
  - [experiments/edm_first/train_edm_map.py](experiments/edm_first/train_edm_map.py)
- EDM-first eval:
  - [experiments/edm_first/eval_edm_map.py](experiments/edm_first/eval_edm_map.py)
- EDM-first watcher:
  - [experiments/edm_first/scripts/watch_eval_checkpoints.sh](experiments/edm_first/scripts/watch_eval_checkpoints.sh)
- paused DGTD train:
  - [scripts/run_train.py](scripts/run_train.py)
- paused DGTD eval:
  - [scripts/run_eval.py](scripts/run_eval.py)

## Main configs

- current e504a:
  - [experiments/edm_first/configs/cifar10_edm_map_onestep_prior_msdefect_8h.yaml](experiments/edm_first/configs/cifar10_edm_map_onestep_prior_msdefect_8h.yaml)
- paused DGTD full:
  - [configs/experiment/dgtd_cifar10_v3.yaml](configs/experiment/dgtd_cifar10_v3.yaml)
- paused DGTD smoke:
  - [configs/experiment/dgtd_cifar10_v3_smoke.yaml](configs/experiment/dgtd_cifar10_v3_smoke.yaml)
