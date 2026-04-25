#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_DIR/scripts/server/activate_a100_runtime.sh"
source "$ROOT_DIR/scripts/server/network_profiles.sh"
dg_twfd_net_heavy
echo "Using heavy-download network profile for public assets"
dg_twfd_net_status

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required but was not found on PATH" >&2
  exit 1
fi

CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$DG_TWFD_A100_ENV"

mkdir -p "$DNNLIB_CACHE_DIR" "$DATA_ROOT/cifar10" "$RUNTIME_ROOT/checkpoints/teachers"

python "$ROOT_DIR/scripts/build_dataset.py" --dataset cifar10 --data-root "$DATA_ROOT/cifar10" --download

export EDM_CIFAR10_NETWORK="${EDM_CIFAR10_NETWORK_MIRROR_URL:-$EDM_CIFAR10_NETWORK}"
export EDM_CIFAR10_FID_REF="${EDM_CIFAR10_FID_REF_MIRROR_URL:-$EDM_CIFAR10_FID_REF}"

if [[ "${DG_TWFD_PREPARE_EDM_ASSETS:-1}" == "1" ]]; then
if ! python - <<'PY'
import os
from pathlib import Path

import dnnlib

cache = Path(os.environ["DNNLIB_CACHE_DIR"])
cache.mkdir(parents=True, exist_ok=True)

for url in [
    os.environ["EDM_CIFAR10_NETWORK"],
    os.environ["EDM_CIFAR10_FID_REF"],
]:
    handle = dnnlib.util.open_url(url, verbose=True)
    if hasattr(handle, "close"):
        handle.close()

print("Prepared EDM public assets in", cache)
PY
then
  cat >&2 <<EOF
EDM public asset prefetch failed under the heavy-download profile.
This script did not fall back to the local proxy because these are large files.
Set EDM_CIFAR10_NETWORK_MIRROR_URL / EDM_CIFAR10_FID_REF_MIRROR_URL, or set
EDM_CIFAR10_NETWORK / EDM_CIFAR10_FID_REF to local files under /cache.
EOF
fi
else
  echo "Skipped EDM public asset prefetch because DG_TWFD_PREPARE_EDM_ASSETS=0"
fi

cat <<EOF
Prepared:
  CIFAR-10 dataset under $DATA_ROOT/cifar10
  EDM cache under $DNNLIB_CACHE_DIR

Still manual:
  ImageNet raw data under $IMAGENET_RAW_ROOT
  ImageNet64 preprocessed data under $IMAGENET64_PREPROCESSED
  ImageNet64 reference npz at $IMAGENET64_REFERENCE_NPZ
  ImageNet64 teacher checkpoint at $IMAGENET64_TEACHER_CKPT
EOF
