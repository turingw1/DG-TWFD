# EDM-First Supervision Log

Last updated: 2026-04-26

## Active Run

```text
run tag: edm_first_cifar10_onestep_msdefect_e504a
tmux: e504a_msdefect
watcher: e504a_eval_watch
checkpoint interval: 250 steps
watch eval steps: 1, 2, 4, 8, 16
FID sample count: 2048 for watcher diagnostics
```

## Current FID Baseline

The first e504a diagnostic checkpoint is:

```text
runs/edm_first_cifar10_onestep_msdefect_e504a/checkpoints/step250.pt
```

Current 2048-sample FID baseline from:

```text
eval/edm_first_cifar10_onestep_msdefect_e504a_step250_steps16/reports/summary.csv
```

| steps | FID |
|---:|---:|
| 1 | 177.890188 |
| 2 | 46.285510 |
| 4 | 49.293783 |
| 8 | 70.910603 |
| 16 | 86.566645 |

The active one-step decision threshold is a 50% FID reduction from the current
1-step baseline:

```text
1-step FID <= 88.945094
```

Secondary multi-step thresholds for a full-train decision:

| steps | 50% FID target |
|---:|---:|
| 2 | 23.142755 |
| 4 | 24.646892 |
| 8 | 35.455302 |
| 16 | 43.283323 |

Interpretation rule: do not judge full-train value from train loss alone. Wait
until a diagnostic checkpoint reaches the primary one-step threshold, or until a
clear multi-step regression/improvement pattern appears across at least two
consecutive watcher evaluations.

The active watcher is configured to compare future evaluations against this
baseline and write:

```text
eval/<tag>/reports/threshold_verdict.json
eval/<tag>/reports/threshold_verdict.csv
```

The full-train analysis trigger is:

```text
primary_target_met: true
```

## Monitoring Commands

Use these commands from the repo root during the current EDM-first experiment
cycle:

```bash
cd /home/ma-user/workspace/Zhengwei/DG-TWFD
```

### One-Shot Status

Training and watcher sessions:

```bash
tmux ls
```

GPU usage:

```bash
nvidia-smi
```

On this server, some shells may need elevated execution to see tmux/GPU state.
If `tmux` returns `Operation not permitted` or `nvidia-smi` cannot see the GPU,
run the same command from the privileged server shell/Codex execution context.

Latest training records:

```bash
tail -n 20 runs/edm_first_cifar10_onestep_msdefect_e504a/logs/train.jsonl
```

Latest checkpoint files:

```bash
find runs/edm_first_cifar10_onestep_msdefect_e504a/checkpoints -maxdepth 1 -type f -printf '%TY-%Tm-%Td %TH:%TM:%TS %s %p\n' | sort
```

Watcher status:

```bash
tail -n 80 runs/edm_first_cifar10_onestep_msdefect_e504a/logs/watch_eval.log
```

### Live Logs

Training pane:

```bash
tmux attach -t e504a_msdefect
```

Watcher pane:

```bash
tmux attach -t e504a_eval_watch
```

Detach from tmux without stopping the job:

```text
Ctrl-b d
```

Read a pane without attaching:

```bash
tmux capture-pane -pt e504a_msdefect -S -120
tmux capture-pane -pt e504a_eval_watch -S -120
```

### Current FID Tables

Step250 baseline with 16-step included:

```bash
cat eval/edm_first_cifar10_onestep_msdefect_e504a_step250_steps16/reports/summary.csv
```

All e504a summaries:

```bash
find -L eval -maxdepth 4 -type f -path '*edm_first_cifar10_onestep_msdefect_e504a*/reports/summary.csv' | sort
```

Latest watcher-produced summary:

```bash
latest_eval="$(find -L eval -maxdepth 2 -type d -name 'edm_first_cifar10_onestep_msdefect_e504a_step*' | sort -V | tail -1)"
cat "$latest_eval/reports/summary.csv"
```

Latest threshold verdict:

```bash
latest_eval="$(find -L eval -maxdepth 2 -type d -name 'edm_first_cifar10_onestep_msdefect_e504a_step*' | sort -V | tail -1)"
cat "$latest_eval/reports/threshold_verdict.csv"
cat "$latest_eval/reports/threshold_verdict.json"
```

### Sample Grids

List fixed-seed grids:

```bash
find -L eval -path '*edm_first_cifar10_onestep_msdefect_e504a*fixed_seed_grid.png' | sort -V
```

Open the latest grid path in VS Code or copy the printed path into the IDE file
browser:

```bash
find -L eval -path '*edm_first_cifar10_onestep_msdefect_e504a*fixed_seed_grid.png' | sort -V | tail -5
```

### Timewarp Comparison

e504a is a no-warp baseline, so timewarp comparison files are expected to be
absent for this run. For future timewarp-enabled runs, inspect:

```bash
latest_eval="$(find -L eval -maxdepth 2 -type d -name '*step*' | sort -V | tail -1)"
cat "$latest_eval/reports/timewarp_comparison.csv"
cat "$latest_eval/reports/timewarp_comparison.json"
```

Negative `fid_delta_auto_minus_identity` means the learned/default timewarp
beats identity for that step count.

### Backup And Resume Evidence

Check `/temp` evidence for the current baseline:

```bash
find /temp/Zhengwei/DG-TWFD-backups/experiment_evidence/edm_first_cifar10_onestep_msdefect_e504a_step250_steps16 -maxdepth 3 -type f | sort
```

Check git safety backups for unpushed commits:

```bash
find /temp/Zhengwei/DG-TWFD-backups/git_safety -maxdepth 3 -type f | sort
```

### Decision Command

Run this after each watcher eval to decide whether the 50% trigger was hit:

```bash
latest_eval="$(find -L eval -maxdepth 2 -type d -name 'edm_first_cifar10_onestep_msdefect_e504a_step*' | sort -V | tail -1)"
python experiments/edm_first/scripts/check_fid_thresholds.py \
  --baseline-summary eval/edm_first_cifar10_onestep_msdefect_e504a_step250_steps16/reports/summary.json \
  --summary "$latest_eval/reports/summary.json" \
  --out-dir "$latest_eval/reports" \
  --target-ratio 0.5 \
  --primary-step 1
cat "$latest_eval/reports/threshold_verdict.json"
```

### Hourly Supervisor

The long supervision job runs a stronger loop than the checkpoint watcher:
every hour it prints training state, runs FID on the latest checkpoint, writes a
threshold verdict, backs up metrics/grids to `/temp`, and exits with a blocker
report if the target is not met after 7 hours.

Start or restart it:

```bash
tmux kill-session -t e504a_hourly_supervisor 2>/dev/null || true
tmux new-session -d -s e504a_hourly_supervisor \
  'cd /home/ma-user/workspace/Zhengwei/DG-TWFD && \
   DG_TWFD_SUPERVISION_INTERVAL_SECONDS=3600 \
   DG_TWFD_SUPERVISION_MAX_HOURS=7 \
   DG_TWFD_SUPERVISION_FID_SAMPLES=2048 \
   bash experiments/edm_first/scripts/hourly_supervise_edm_first.sh'
```

Watch its output:

```bash
tail -f runs/edm_first_cifar10_onestep_msdefect_e504a/logs/hourly_supervisor.log
```

Check whether it is still running:

```bash
tmux ls | rg 'e504a_hourly_supervisor|e504a_msdefect|e504a_eval_watch'
```

If the one-step 50% FID target is reached, the supervisor calls:

```bash
bash experiments/edm_first/scripts/launch_timewarp_followup.sh <winning_checkpoint>
```

That follow-up uses:

```text
experiments/edm_first/configs/cifar10_edm_map_onestep_prior_msdefect_timewarp_8h.yaml
```

If 7 hours pass without hitting the target, inspect the generated blocker
report:

```bash
find runs/edm_first_cifar10_onestep_msdefect_e504a/reports -name 'hourly_supervision_blockers_*.md' | sort -V | tail -1
```

## Timewarp Tracking Rule

e504a is intentionally an identity-clock run:

```text
timewarp.enabled: false
```

Therefore its timewarp effect is `N/A` and it serves as the no-warp baseline for
the current one-step endpoint objective.

From this point onward, watcher evaluations use steps `1 2 4 8 16`. When a
config enables `timewarp.enabled: true`, the watcher also runs an identity-clock
comparison and writes:

```text
eval/<tag>/reports/timewarp_comparison.csv
eval/<tag>/reports/timewarp_comparison.json
```

The key column is:

```text
fid_delta_auto_minus_identity
```

Negative values mean the learned/default timewarp is better than identity for
that step budget. Positive values mean identity is better or equal and the warp
objective should not be trusted for that budget without modification.
