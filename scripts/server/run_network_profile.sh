#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [[ $# -lt 2 ]]; then
  cat >&2 <<EOF
Usage:
  bash scripts/server/run_network_profile.sh <proxy|heavy|offline> <command> [args...]

Examples:
  bash scripts/server/run_network_profile.sh heavy python scripts/build_dataset.py --dataset cifar10 --data-root datasets/cifar10 --download
  bash scripts/server/run_network_profile.sh proxy git ls-remote origin
EOF
  exit 2
fi

profile="$1"
shift

source "$ROOT_DIR/scripts/server/activate_a100_runtime.sh" >/dev/null
source "$ROOT_DIR/scripts/server/network_profiles.sh"
dg_twfd_net_apply "$profile"
dg_twfd_net_status

exec "$@"
