# MAP Branch Acceptance Checklist

## Level 1. Architecture acceptance

### Pass criteria
- baseline FM still trains with `configs/experiment/fm_cifar10_baseline.yaml`
- map branch is selectable by config:
  - `train.objective = explicit_map`
- map branch target source is config-driven:
  - `target.builder = trajectory_shard`
- train / eval / sample entrypoints remain shared
- checkpoints, logs, samples, metrics stay under run-root / eval-root

### Commands
- baseline:
  ```bash
  CUDA_VISIBLE_DEVICES=1 python scripts/run_train.py --config configs/experiment/fm_cifar10_baseline.yaml --run-root outputs/debug/runs/baseline_arch_smoke --set train.epochs=1 --set train.max_train_batches=2 --set train.max_val_batches=1
  ```
- map:
  ```bash
  CUDA_VISIBLE_DEVICES=1 python scripts/run_train.py --config configs/experiment/fm_cifar10_map_branch.yaml --run-root outputs/debug/runs/map_arch_smoke --set train.epochs=1 --set train.max_train_batches=2 --set train.max_val_batches=1
  ```

### Required outputs
- `checkpoints/last.pt`
- `logs/config_resolved.yaml`
- `logs/train.jsonl`

## Level 2. Functional acceptance

### Pass criteria
- map branch trains end-to-end on CIFAR-10
- teacher trajectories can be prepared under `target.shard_root`
- map branch checkpoint loads successfully
- map branch samples at `1/2/4/8/16`
- map branch runs FID evaluation
- map branch dumps fixed-seed sample grids

### Commands
- trajectory preparation:
  ```bash
  CUDA_VISIBLE_DEVICES=1 python scripts/prepare_teacher_trajectories.py --config configs/experiment/fm_cifar10_map_branch.yaml --output-root <TRAJ_ROOT> --batch-size 64
  ```
- eval:
  ```bash
  CUDA_VISIBLE_DEVICES=1 python scripts/run_eval.py --config configs/experiment/fm_cifar10_map_branch.yaml --checkpoint <CKPT> --eval-root <EVAL_ROOT> --steps 1 2 4 8 16 --fid-samples 5000 --fid-batch-size 128
  ```
- sample:
  ```bash
  CUDA_VISIBLE_DEVICES=1 python scripts/run_sample.py --config configs/experiment/fm_cifar10_map_branch.yaml --checkpoint <CKPT> --output-dir <OUT_DIR> --steps 16 --num-samples 64
  ```
- panel:
  ```bash
  CUDA_VISIBLE_DEVICES=1 python scripts/run_multistep_panel.py --config configs/experiment/fm_cifar10_map_branch.yaml --checkpoint <CKPT> --output-dir <OUT_DIR>/panel --steps 1 2 4 8 16
  ```

### Required outputs
- `<TRAJ_ROOT>/train/manifest.yaml`
- `<TRAJ_ROOT>/val/manifest.yaml`
- `<EVAL_ROOT>/reports/summary.csv`
- `<EVAL_ROOT>/reports/best.json`
- `<EVAL_ROOT>/steps16/fixed_seed_grid.png`
- `<OUT_DIR>/grid.png`
- `<OUT_DIR>/panel/multistep_panel.png`

## Level 3. Numerical acceptance

### Pass criteria
- `train_loss` decreases over epochs
- `val_loss` decreases or stabilizes
- fixed-seed images are non-degenerate
- `4-step` or `8-step` samples are visibly better than noise
- FID evaluation completes successfully
- same fixed seed produces deterministic qualitative outputs

### Required log fields
- `train_loss`
- `val_loss`
- `train_t_mean`
- `train_s_mean`
- `train_delta_mean`
- `target_builder`

### Failure conditions
- NaN loss
- all-black / all-white / noise-like fixed grids after non-trivial training
- missing `best.pt`
- FID evaluation crashes or returns missing records

## Level 4. Research-readiness acceptance

### Pass criteria
- target construction lives in `src/dgfm/targets/*`
- map rollout lives in `src/dgfm/samplers/*`
- trainer selection is config-driven
- explicit insertion point exists for:
  - teacher switching
  - time-warp
  - semigroup defect
- no critical logic hidden in one-off scripts

### Exact files that must exist
- `src/dgfm/trainers/map.py`
- `src/dgfm/models/map.py`
- `src/dgfm/targets/builder.py`
- `src/dgfm/samplers/map_sampler.py`
- `configs/experiment/fm_cifar10_map_branch.yaml`

## Acceptance summary

- PASS only if all four levels pass.
- The first blocking failure is architectural inconsistency, then missing functional outputs, then numerical degeneracy.
