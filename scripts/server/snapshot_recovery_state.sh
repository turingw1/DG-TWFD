#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RECOVERY_ROOT="${DG_TWFD_RECOVERY_ROOT:-/temp/Zhengwei/DG-TWFD-recovery}"
MAX_UNTRACKED_MB="${DG_TWFD_RECOVERY_MAX_UNTRACKED_MB:-200}"
KEEP_SNAPSHOTS="${DG_TWFD_RECOVERY_KEEP_SNAPSHOTS:-3}"
CODEX_HOME="${CODEX_HOME:-$HOME/.codex}"
CODEX_SNAPSHOT_MAX_MB="${DG_TWFD_CODEX_SNAPSHOT_MAX_MB:-256}"

cd "$ROOT_DIR"
mkdir -p "$RECOVERY_ROOT"

branch="$(git branch --show-current || true)"
remote_url="$(git remote get-url origin 2>/dev/null || true)"
timestamp="$(date +%Y%m%d_%H%M%S)"
snapshot_dir="$RECOVERY_ROOT/snapshots/$timestamp"
latest_dir="$RECOVERY_ROOT/latest"

mkdir -p "$snapshot_dir"

git status --short --branch > "$snapshot_dir/git_status.txt"
git diff --binary > "$snapshot_dir/dirty.patch"
git submodule status > "$snapshot_dir/submodules.txt" 2>/dev/null || true
git rev-parse HEAD > "$snapshot_dir/head.txt"
printf '%s\n' "$branch" > "$snapshot_dir/branch.txt"
printf '%s\n' "$remote_url" > "$snapshot_dir/remote_url.txt"

find refs -mindepth 1 -maxdepth 1 -type d -print0 2>/dev/null \
  | while IFS= read -r -d '' repo; do
      if git -C "$repo" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        printf '%s %s\n' "$repo" "$(git -C "$repo" rev-parse --short HEAD)"
      fi
    done > "$snapshot_dir/refs_commits.txt"

untracked_list="$snapshot_dir/untracked_files.txt"
git ls-files --others --exclude-standard > "$untracked_list"
untracked_mb=0
if [[ -s "$untracked_list" ]]; then
  untracked_bytes="$(tar -cf - --files-from "$untracked_list" 2>/dev/null | wc -c || echo 0)"
  untracked_mb=$(( (untracked_bytes + 1048575) / 1048576 ))
  if (( untracked_mb <= MAX_UNTRACKED_MB )); then
    tar -czf "$snapshot_dir/untracked_files.tar.gz" --files-from "$untracked_list"
  else
    echo "Skipped untracked tar: ${untracked_mb}MB exceeds ${MAX_UNTRACKED_MB}MB" > "$snapshot_dir/untracked_files.skipped"
  fi
fi

if git rev-parse --verify HEAD >/dev/null 2>&1; then
  tmp_bundle="${TMPDIR:-/tmp}/dg_twfd_repo_head_${timestamp}_$$.bundle"
  git bundle create "$tmp_bundle" HEAD >/dev/null
  cp "$tmp_bundle" "$snapshot_dir/repo_head.bundle"
  rm -f "$tmp_bundle"
fi

codex_mb=0
if [[ -d "$CODEX_HOME" ]]; then
  codex_list="$snapshot_dir/codex_files.txt"
  : > "$codex_list"
  if [[ -f "$CODEX_HOME/config.toml" ]]; then
    printf '%s\n' "config.toml" >> "$codex_list"
  fi
  if [[ -d "$CODEX_HOME/archived_sessions" ]]; then
    find "$CODEX_HOME/archived_sessions" -maxdepth 1 -type f -name '*.jsonl' -printf 'archived_sessions/%f\n' \
      | sort >> "$codex_list"
  fi
  if [[ -s "$codex_list" ]]; then
    codex_bytes="$(tar -C "$CODEX_HOME" -cf - --files-from "$codex_list" 2>/dev/null | wc -c || echo 0)"
    codex_mb=$(( (codex_bytes + 1048575) / 1048576 ))
    if (( codex_mb <= CODEX_SNAPSHOT_MAX_MB )); then
      tar -C "$CODEX_HOME" -czf "$snapshot_dir/codex_home_minimal.tar.gz" --files-from "$codex_list"
    else
      echo "Skipped Codex snapshot: ${codex_mb}MB exceeds ${CODEX_SNAPSHOT_MAX_MB}MB" > "$snapshot_dir/codex_home_minimal.skipped"
    fi
  fi
fi

cat > "$snapshot_dir/RESTORE.md" <<EOF
# DG-TWFD Crash Recovery Snapshot

Snapshot: $timestamp

This directory is intentionally small. It contains git metadata, a binary diff
for tracked dirty files, optional small untracked files, and current refs commit
ids. It does not contain datasets, conda envs, run outputs, or checkpoints.

## Restore

\`\`\`bash
bash /temp/Zhengwei/DG-TWFD-recovery/restore_after_crash.sh
\`\`\`

## Snapshot Contents

- \`remote_url.txt\`: origin URL
- \`branch.txt\`: current branch
- \`head.txt\`: current HEAD
- \`dirty.patch\`: tracked uncommitted changes
- \`untracked_files.tar.gz\`: untracked git-relevant files, if below size cap
- \`refs_commits.txt\`: current reference repository commits
- \`repo_head.bundle\`: current committed repo HEAD, including unpushed commits
- \`codex_home_minimal.tar.gz\`: Codex config and archived sessions, if under
  the size cap
\`\`\`
$(sed 's/^/  /' "$snapshot_dir/refs_commits.txt")
\`\`\`
EOF

cat > "$RECOVERY_ROOT/restore_after_crash.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

RECOVERY_ROOT="${DG_TWFD_RECOVERY_ROOT:-/temp/Zhengwei/DG-TWFD-recovery}"
SNAPSHOT_DIR="${1:-$RECOVERY_ROOT/latest}"
WORKSPACE_ROOT="${DG_TWFD_WORKSPACE_ROOT:-/home/ma-user/workspace/Zhengwei}"
REPO_DIR="$WORKSPACE_ROOT/DG-TWFD"

remote_url="$(cat "$SNAPSHOT_DIR/remote_url.txt")"
branch="$(cat "$SNAPSHOT_DIR/branch.txt")"

mkdir -p "$WORKSPACE_ROOT"
if [[ ! -d "$REPO_DIR/.git" ]]; then
  git clone "$remote_url" "$REPO_DIR"
fi

cd "$REPO_DIR"
git fetch origin || true
if [[ -f "$SNAPSHOT_DIR/repo_head.bundle" ]]; then
  git fetch "$SNAPSHOT_DIR/repo_head.bundle" HEAD:refs/tmp/dg_twfd_recovery_head
  git checkout -B "$branch" refs/tmp/dg_twfd_recovery_head
  git branch --set-upstream-to="origin/$branch" "$branch" 2>/dev/null || true
else
  git checkout "$branch"
  git pull --ff-only origin "$branch" || true
fi

if [[ -s "$SNAPSHOT_DIR/dirty.patch" ]]; then
  git apply --index "$SNAPSHOT_DIR/dirty.patch" || git apply "$SNAPSHOT_DIR/dirty.patch"
fi

if [[ -f "$SNAPSHOT_DIR/untracked_files.tar.gz" ]]; then
  tar -xzf "$SNAPSHOT_DIR/untracked_files.tar.gz" -C "$REPO_DIR"
fi

bash scripts/clone_reference_repos.sh || true
source scripts/server/activate_a100_runtime.sh

if [[ -f "$SNAPSHOT_DIR/codex_home_minimal.tar.gz" ]]; then
  CODEX_HOME="${CODEX_HOME:-$HOME/.codex}"
  mkdir -p "$CODEX_HOME"
  tar -xzf "$SNAPSHOT_DIR/codex_home_minimal.tar.gz" -C "$CODEX_HOME"
fi

cat <<EOM
Recovery bootstrap finished.
Repo: $REPO_DIR
Snapshot: $SNAPSHOT_DIR

Next checks:
  git status --short --branch
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$DG_TWFD_A100_ENV"
  python scripts/server/smoke_a100.py
EOM
EOF
chmod +x "$RECOVERY_ROOT/restore_after_crash.sh"

rm -f "$latest_dir"
ln -s "$snapshot_dir" "$latest_dir"
cp "$snapshot_dir/RESTORE.md" "$RECOVERY_ROOT/RESTORE.md"

mapfile -t old_snapshots < <(find "$RECOVERY_ROOT/snapshots" -mindepth 1 -maxdepth 1 -type d | sort -r)
for old_snapshot in "${old_snapshots[@]:$KEEP_SNAPSHOTS}"; do
  rm -rf "$old_snapshot"
done

echo "Recovery snapshot written"
echo "  snapshot=$snapshot_dir"
echo "  latest=$latest_dir"
echo "  untracked_size_mb=$untracked_mb"
echo "  codex_snapshot_mb=$codex_mb"
echo "  keep_snapshots=$KEEP_SNAPSHOTS"
