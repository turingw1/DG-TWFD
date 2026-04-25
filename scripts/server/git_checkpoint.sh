#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

mode="${1:-status}"

if [[ "${DG_TWFD_SKIP_RECOVERY_SNAPSHOT:-0}" != "1" ]]; then
  bash "$ROOT_DIR/scripts/server/snapshot_recovery_state.sh"
fi

echo "Git workspace state:"
git status --short --branch

git fetch origin

if [[ "$mode" != "--commit-push" ]]; then
  cat <<EOF

No commit was made.
To checkpoint code/docs to git:
  DG_TWFD_COMMIT_MESSAGE="your message" bash scripts/server/git_checkpoint.sh --commit-push
EOF
  exit 0
fi

message="${DG_TWFD_COMMIT_MESSAGE:-}"
if [[ -z "$message" ]]; then
  echo "DG_TWFD_COMMIT_MESSAGE is required for --commit-push" >&2
  exit 2
fi

if [[ -z "$(git config user.name || true)" || -z "$(git config user.email || true)" ]]; then
  cat >&2 <<EOF
Git identity is not configured in this repo.
Set it before committing, for example:
  git config user.name "Your Name"
  git config user.email "your-email@example.com"
EOF
  exit 2
fi

git add -A
if git diff --cached --quiet; then
  echo "No staged changes to commit."
else
  git commit -m "$message"
fi

git push origin HEAD
