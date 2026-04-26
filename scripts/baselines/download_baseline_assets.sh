#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

source scripts/server/activate_a100_runtime.sh >/dev/null
source scripts/server/network_profiles.sh
dg_twfd_net_apply "${DG_TWFD_BASELINE_NETWORK_PROFILE:-proxy}"

asset_root="${BASELINE_ASSET_ROOT:-/cache/Zhengwei/DG-TWFD-runtime/checkpoints/baselines}"
mkdir -p "$asset_root/consistency_models" "$asset_root/ctm" "$asset_root/tcm"

download() {
  local url="$1"
  local output="$2"
  if [[ -s "$output" ]]; then
    echo "exists: $output"
    return 0
  fi
  echo "download: $url -> $output"
  wget -c --tries=20 --timeout=30 --waitretry=5 -O "$output" "$url"
}

if [[ "${DOWNLOAD_CD_IMAGENET64:-1}" == "1" ]]; then
  download \
    "https://openaipublic.blob.core.windows.net/consistency/cd_imagenet64_lpips.pt" \
    "$asset_root/consistency_models/cd_imagenet64_lpips.pt"
  download \
    "https://openaipublic.blob.core.windows.net/consistency/cd_imagenet64_l2.pt" \
    "$asset_root/consistency_models/cd_imagenet64_l2.pt"
fi

cat > "$asset_root/README.md" <<'EOF'
# External Baseline Assets

This directory stores large external baseline checkpoints. It lives under
`/cache`, not git. Important checkpoints/results must be either redownloadable
from this script or separately backed up under `/temp` if they are expensive to
recreate.

Current scripted downloads:

- OpenAI Consistency Models ImageNet64 CD LPIPS
- OpenAI Consistency Models ImageNet64 CD L2

Pending manual/mirror assets:

- CTM CIFAR-10 checkpoint from Google Drive folder
- CTM ImageNet64 checkpoint from Google Drive
- TCM CIFAR-10 and ImageNet64 checkpoints from Google Drive
EOF

python3 scripts/baselines/probe_baseline_assets.py --output results/baselines/asset_probe.json
