#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
source "$ROOT_DIR/scripts/server/activate_a100_runtime.sh" >/dev/null
source "$ROOT_DIR/scripts/server/network_profiles.sh"
dg_twfd_net_apply "${DG_TWFD_ASSET_NETWORK_PROFILE:-proxy}"

cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/refs/edm:$ROOT_DIR/src:${PYTHONPATH:-}"
mkdir -p "$DNNLIB_CACHE_DIR"

"$DG_TWFD_A100_ENV/bin/python" - <<'PY'
import os
import pickle
from pathlib import Path

import dnnlib

network = os.environ.get(
    "EDM_CIFAR10_NETWORK",
    "https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl",
)
fid_ref = os.environ.get(
    "EDM_CIFAR10_FID_REF",
    "https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz",
)
print("DNNLIB_CACHE_DIR", os.environ.get("DNNLIB_CACHE_DIR"))
for url in [network, fid_ref]:
    print("checking", url)
    with dnnlib.util.open_url(url, verbose=True) as handle:
        if url.endswith(".pkl"):
            obj = pickle.load(handle)
            print("  pkl keys:", sorted(obj.keys()))
        else:
            print("  opened")

fid_stats = Path(os.environ.get("DATA_ROOT", "datasets")) / "cifar10/.dgfm_cache/fid_stats_cifar10_test_32_torch_fidelity_inceptionv3_2048.npz"
print("torch-fidelity stats:", fid_stats, fid_stats.exists())
PY
