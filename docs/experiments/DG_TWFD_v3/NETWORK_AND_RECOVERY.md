# Network And Crash Recovery Policy

The server has two different network needs:

- normal outbound work should use the local proxy at `http://127.0.0.1:8080`
  for GitHub, OpenAI/Codex, and general external access
- large downloads should not use the local proxy; datasets, Hugging Face
  models, and large package installs should use the server route plus mirrors

Use the repo profiles instead of relying on ad hoc shell exports:

```bash
source scripts/server/activate_a100_runtime.sh
source scripts/server/network_profiles.sh

dg_twfd_net_proxy   # default: local proxy for normal outbound access
dg_twfd_net_heavy   # no proxy; hf-mirror, Huawei pip mirror, TUNA conda mirror
dg_twfd_net_offline # no proxy; force HF/Transformers offline flags
```

For one-off commands:

```bash
bash scripts/server/run_network_profile.sh heavy python scripts/build_dataset.py --dataset cifar10 --data-root datasets/cifar10 --download
bash scripts/server/run_network_profile.sh proxy git ls-remote origin
```

`activate_a100_runtime.sh` applies `DG_TWFD_NETWORK_PROFILE=proxy` by default,
so routine project shells keep the local proxy. Download scripts switch to the
`heavy` profile explicitly.

Mirror defaults:

- Hugging Face: `HF_ENDPOINT=https://hf-mirror.com`
- pip: `PIP_INDEX_URL=http://repo.myhuaweicloud.com/repository/pypi/simple`
- conda: `scripts/server/condarc_heavy_downloads.yaml`
- CIFAR-10 tarball: `https://mirrors.dotsrc.org/osdn/datasets/74526/cifar-10-python.tar.gz`
- ModelScope cache: `/cache/Zhengwei/DG-TWFD-runtime/.modelscope`

If a large asset is not mirrored, do not silently fall back to the local proxy.
Prefer one of:

- place the file under `/cache/Zhengwei/DG-TWFD-runtime/...` and point the
  config/env var to the local path
- set an explicit mirror URL for that asset
- download outside the long-running training session and record the source in
  the experiment log

## Crash Recovery

`/home/ma-user/workspace` and `/cache` can disappear after server crashes.
`/temp` survives, but should only hold small recovery metadata and selected
long-running experiment artifacts.

Rules:

- commit and push workspace code frequently; GitHub is the source of truth for
  code and docs
- keep data, envs, and caches under `/cache` via repo symlinks
- keep important long-run logs/checkpoints under
  `/temp/Zhengwei/DG-TWFD-backups`
- keep small repo recovery metadata under
  `/temp/Zhengwei/DG-TWFD-recovery`
- do not copy datasets, full conda envs, or broad cache trees into `/temp`

Create a recovery snapshot after meaningful code/doc changes:

```bash
bash scripts/server/snapshot_recovery_state.sh
```

This writes:

- `/temp/Zhengwei/DG-TWFD-recovery/latest/dirty.patch`
- `/temp/Zhengwei/DG-TWFD-recovery/latest/untracked_files.tar.gz`, if small
- `/temp/Zhengwei/DG-TWFD-recovery/latest/refs_commits.txt`
- `/temp/Zhengwei/DG-TWFD-recovery/restore_after_crash.sh`
- `/temp/Zhengwei/DG-TWFD-recovery/RESTORE.md`

The script keeps only the latest 3 snapshots by default
(`DG_TWFD_RECOVERY_KEEP_SNAPSHOTS=3`) to avoid turning `/temp` into a cache.

For the normal code checkpoint flow, use:

```bash
bash scripts/server/git_checkpoint.sh
DG_TWFD_COMMIT_MESSAGE="checkpoint: describe change" bash scripts/server/git_checkpoint.sh --commit-push
```

The first command refreshes `/temp` recovery metadata and prints git status.
The second command stages git-relevant files, commits, and pushes only when an
explicit message is provided and repo-local git identity is configured.

Restore after a crash:

```bash
bash /temp/Zhengwei/DG-TWFD-recovery/restore_after_crash.sh
```

Then rerun:

```bash
cd /home/ma-user/workspace/Zhengwei/DG-TWFD
source scripts/server/activate_a100_runtime.sh
bash scripts/clone_reference_repos.sh
```
