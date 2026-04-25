# A100 Server Deployment

This note fixes the current server runtime layout for the workspace/cache/temp
setup on the active `DG_TWFD_v3` branch.

It supersedes the archived `A100_MIGRATION_2026-04-25.md` note for routine
development:

- this document defines the actual path policy, symlink policy, and bootstrap
  flow on the current server
- the old migration note remains under `docs/archive/low_signal_2026-04-25/`
  for targeted historical lookup only

## Canonical Path Policy

Use this split:

```text
/home/ma-user/workspace/Zhengwei/DG-TWFD           # canonical code and docs root
/cache/Zhengwei/DG-TWFD-runtime                    # datasets, runs, eval, caches, env payloads
/temp/Zhengwei/DG-TWFD-backups                     # backup copy for checkpoints/logs and tmp snapshots
/tmp/dg_twfd                                       # transient scratch only
```

Rules:

- treat `~/workspace/Zhengwei/DG-TWFD` as the only development root that should
  stay git-syncable
- keep large runtime assets out of the tracked tree
- expose runtime directories back into the repo through symlinks so the repo
  layout remains stable for scripts and configs
- use `/temp/Zhengwei/DG-TWFD-backups` as the second copy for important
  checkpoints and logs so long runs can be resumed after server crashes

The workspace repo should not depend on a second code clone under `/cache` as
its formal source of truth.

## Symlinked Repo Layout

After runtime activation, these paths inside the workspace repo are symlinks:

- `datasets -> /cache/Zhengwei/DG-TWFD-runtime/datasets`
- `runs -> /cache/Zhengwei/DG-TWFD-runtime/runs`
- `eval -> /cache/Zhengwei/DG-TWFD-runtime/eval`
- `results -> /cache/Zhengwei/DG-TWFD-runtime/results`
- `teacher_traj -> /cache/Zhengwei/DG-TWFD-runtime/teacher_traj`
- `.hf_home -> /cache/Zhengwei/DG-TWFD-runtime/.hf_home`
- `.torch -> /cache/Zhengwei/DG-TWFD-runtime/.torch`
- `.conda_envs -> /cache/Zhengwei/DG-TWFD-runtime/.conda_envs`
- `backup_runs -> /temp/Zhengwei/DG-TWFD-backups`

This keeps scripts using repo-relative paths while the data actually lives on
the larger filesystems.

## One-Time Runtime Activation

```bash
cd /home/ma-user/workspace/Zhengwei/DG-TWFD
source scripts/server/activate_a100_runtime.sh
```

This prepares the runtime directories, creates the workspace symlinks, and
exports the main environment variables used by the rest of the project.

Important exported values:

- `PROJ=/home/ma-user/workspace/Zhengwei/DG-TWFD`
- `DATA_ROOT=$PROJ/datasets`
- `RUNS_ROOT=$PROJ/runs`
- `EVAL_ROOT=$PROJ/eval`
- `RESULTS_ROOT=$PROJ/results`
- `TRAJ_ROOT=$PROJ/teacher_traj/cifar10_ddpm128_p33`
- `DG_TWFD_BACKUP_ROOT=$PROJ/backup_runs`
- `DG_TWFD_A100_ENV=$PROJ/.conda_envs/dg_twfd_a100`
- `TMPDIR=/tmp/dg_twfd`
- `PYTHONPATH=$PROJ/src:$PROJ/refs/edm:${PYTHONPATH:-}`

## Single-Environment Bootstrap

Use one shared conda env for the root project and `refs/edm`:

```bash
cd /home/ma-user/workspace/Zhengwei/DG-TWFD
bash scripts/server/bootstrap_a100_single_env.sh
```

This script:

- activates the runtime layout
- clones `refs/edm` into the workspace tree if it is missing
- pins `refs/edm` to commit `6bb90217f80afef811abc11e790bc14fab853922`
- creates the env at `./.conda_envs/dg_twfd_a100`
- by default clones the current `base` conda env on this server so the existing
  CUDA-enabled torch stack is reused instead of downloading it again
- installs the root project with the `teacher` runtime extra plus the extra
  packages used by EDM and FID

Then activate it with:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
source scripts/server/activate_a100_runtime.sh
conda activate /home/ma-user/workspace/Zhengwei/DG-TWFD/.conda_envs/dg_twfd_a100
```

## Smoke Verification

```bash
cd /home/ma-user/workspace/Zhengwei/DG-TWFD
source scripts/server/activate_a100_runtime.sh
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /home/ma-user/workspace/Zhengwei/DG-TWFD/.conda_envs/dg_twfd_a100
python scripts/server/smoke_a100.py
```

Expected:

- root imports succeed
- config resolution uses repo-relative symlinked paths
- `refs/edm` imports succeed
- if CUDA is visible in the current shell, torch reports the A100 correctly

## Dataset And Public Asset Preparation

When the environment passes smoke and you are ready to start downloading:

```bash
cd /home/ma-user/workspace/Zhengwei/DG-TWFD
source scripts/server/activate_a100_runtime.sh
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /home/ma-user/workspace/Zhengwei/DG-TWFD/.conda_envs/dg_twfd_a100
bash scripts/server/prepare_public_assets.sh
```

This script switches to the `heavy` network profile before downloads:

- clears local proxy variables
- sets `HF_ENDPOINT=https://hf-mirror.com`
- sets the Huawei pip mirror and TUNA conda config
- sets `CIFAR10_URL` to a tested OSDN mirror

This will:

- download CIFAR-10 into `datasets/cifar10`
- warm the EDM public checkpoint and FID reference caches

Still manual after that:

- raw ImageNet under `datasets/imagenet_raw`
- preprocessed ImageNet64 under `datasets/imagenet64`
- ImageNet64 reference `.npz`
- ImageNet64 teacher checkpoint

## Backup Principle

Two layers exist:

- formal run backups:
  - `scripts/experiments/activate_fm_cifar10.sh` now defaults
    `DGFM_ARCHIVE_ROOT` to `backup_runs/<FM_EXP>`
  - training checkpoints and JSONL logs are therefore written to both the run
    directory and the backup directory
- tmp snapshots:
  - use `bash scripts/server/backup_tmp_artifacts.sh <label>` when `/tmp/dg_twfd`
    contains non-git artifacts worth keeping

For long runs, the second copy under `/temp/Zhengwei/DG-TWFD-backups` is the
main crash-recovery path.

For code/workspace crash recovery, use:

```bash
bash scripts/server/snapshot_recovery_state.sh
```

This keeps only small git recovery metadata under
`/temp/Zhengwei/DG-TWFD-recovery`; it does not copy datasets, envs, or caches.
