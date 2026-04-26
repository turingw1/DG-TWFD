# Baseline Comparison Guide

Last updated: 2026-04-26

## Table Schema

All baseline rows must use:

```text
dataset,method,step,fid,is,recall,checkpoint,eval_script,notes
```

Canonical CSV directory:

```text
results/baselines/
```

## What Each Baseline Tests

| baseline | role in comparison | main question |
|---|---|---|
| EDM official | teacher/reference diffusion baseline | How good is the public teacher sampler at 1/2/4/8 steps under our evaluator? |
| CTM | external trajectory/consistency model baseline | How far are we from a strong dedicated few-step consistency/trajectory method? |
| Consistency Models / CD | classic consistency distillation baseline | Does our method beat a direct consistency-distilled ImageNet64 reference? |
| AYS | schedule-only baseline | Can a fixed optimized schedule explain gains without learning our time warp? |
| OptimalSteps | checkpoint-specific schedule search baseline | Does a searched global schedule beat our learned warp on the same checkpoint? |
| Entropic Time Schedulers | entropy-based time reparameterization baseline | Does an information-equalized schedule beat our learned time coordinate? |
| TCM | optional newer trajectory-consistency baseline | How far are we from newer SOTA trajectory-consistency models? |

## Current Execution Policy

Baseline jobs run sequentially with conservative batch sizes because the main
`e504a_msdefect` experiment is still using GPU 0. A baseline OOM should kill the
baseline process only; the launcher retries with a smaller batch.

Launcher:

```bash
tmux new-session -d -s baseline_external_full 'cd /home/ma-user/workspace/Zhengwei/DG-TWFD && CUDA_VISIBLE_DEVICES=0 BASELINE_RUN_TAG=external_baselines_full_20260426 bash scripts/baselines/run_external_baselines.sh'
```

Monitor:

```bash
tmux capture-pane -pt baseline_external_full -S -80
tail -f runs/external_baselines_full_20260426/logs/baseline_runner.log
```

The current launcher runs only EDM official CIFAR-10 and ImageNet64 end-to-end.
CTM/CD/TCM need official checkpoint files before generation can start. AYS,
OptimalSteps, and Entropic require schedule integration on top of the relevant
checkpoint/evaluator after EDM official baselines are underway.
