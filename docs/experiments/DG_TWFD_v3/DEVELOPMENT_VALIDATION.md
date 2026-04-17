# DG_TWFD_v3 Development Validation

This file is the only place for intermediate validation, deployment-side
commands, smoke instructions, and stage-by-stage debugging notes.

It is intentionally separate from the final experiment record.

## Separation rules

- Final `EXP_LOG` and final pipeline docs should contain:
  - stable commands only
  - accepted experiment settings only
  - measured outcomes and decisions only
  - result-oriented summaries
- This file should contain:
  - smoke commands
  - temporary server-side overrides
  - deployment checks
  - runtime debugging steps
  - intermediate acceptance gates

## Development stages

### Stage 0: Environment and cache visibility

Goal:
- verify the target cache path is readable
- verify the basic Python entrypoints start

Server checks:
1. `echo $TRAJ_ROOT`
2. `find "$TRAJ_ROOT" -maxdepth 2 | head`
3. `python scripts/run_train.py --help`
4. `python scripts/run_sample_dgtd.py --help`
5. `python scripts/run_eval.py --help`

Return:
- command outputs
- whether cache root has `train/`, `val/`, or flat shards

Pass gate:
- `TRAJ_ROOT` is correct
- entrypoints import successfully

### Stage 1: Runnable smoke

Goal:
- verify training, checkpoint save, DGTD sampling, eval, and diagnostics all run end-to-end

Server checks:
1. smoke train
2. smoke sample
3. smoke eval
4. diagnostics export

Return:
- stdout/stderr for each command
- `tail -n 5 logs/train.jsonl`
- sample/eval/diag file listings

Pass gate:
- `last.pt` exists
- `train.jsonl` has DGTD fields such as `q_phi`, `q_D`, `D_bar`, `eta`, `beta`
- `grid.png` and eval reports exist

### Stage 2: Training profile

Goal:
- understand runtime, memory, throughput, and stability before longer runs

Server checks:
1. short training run with a few epochs
2. `nvidia-smi` snapshots during train
3. optional `time` wrapper around train command

Track:
- wall-clock per epoch
- GPU memory
- batch size actually sustainable
- whether warp updates cause instability

Pass gate:
- no crash over multiple epochs
- stable `loss/defect`
- acceptable throughput for planned experiments

### Stage 3: Quality sanity

Goal:
- verify the model produces non-degenerate samples and few-step metrics behave sensibly

Server checks:
1. sample at `1,2,4,8,16` steps
2. run eval on the same checkpoints
3. export diagnostics

Track:
- sample quality by visual inspection
- FID trend
- whether `q_phi` focuses on high-defect bins without collapsing

Pass gate:
- samples are not degenerate
- multi-step quality is directionally sensible

### Stage 4: Ablation smoke

Goal:
- verify each ablation variant is runnable and meaningfully different

Variants:
- baseline current losses
- DGTD without learned warp
- DGTD + warp without HF metric
- DGTD + warp + HF metric
- DGTD + uniform warped inference

Return:
- config used
- runtime summary
- sample/eval summary

Pass gate:
- every ablation runs
- outputs are distinguishable enough to support later comparison

### Stage 5: Final experiment campaign

Goal:
- move only accepted settings into final pipeline and experiment log

Final artifacts should record:
- chosen config
- final train command
- final eval/sample commands
- resource plan
- result summary

Do not copy:
- temporary smoke commands
- failed overrides
- deployment debugging snippets

## Current first server round

1. Smoke train:
   `python scripts/run_train.py --config configs/experiment/dgtd_cifar10_v3_smoke.yaml --run-root /tmp/dgtd_v3_smoke --set target.shard_root=$TRAJ_ROOT`
2. Smoke sample:
   `python scripts/run_sample_dgtd.py --config configs/experiment/dgtd_cifar10_v3_smoke.yaml --checkpoint /tmp/dgtd_v3_smoke/checkpoints/last.pt --output-dir /tmp/dgtd_v3_smoke/sample --steps 4 --set target.shard_root=$TRAJ_ROOT`
3. Smoke eval:
   `python scripts/run_eval.py --config configs/experiment/dgtd_cifar10_v3_smoke.yaml --checkpoint /tmp/dgtd_v3_smoke/checkpoints/last.pt --eval-root /tmp/dgtd_v3_smoke/eval --steps 1 2 4 --set target.shard_root=$TRAJ_ROOT`
4. Diagnostics:
   `python scripts/plot_dgtd_diagnostics.py --history /tmp/dgtd_v3_smoke/logs/train.jsonl --output-dir /tmp/dgtd_v3_smoke/diag`
