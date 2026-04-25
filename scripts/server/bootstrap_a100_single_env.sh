#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_DIR/scripts/server/activate_a100_runtime.sh"
source "$ROOT_DIR/scripts/server/network_profiles.sh"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required but was not found on PATH" >&2
  exit 1
fi

CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"

ENV_PREFIX="${DG_TWFD_A100_ENV}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
EDM_COMMIT="${EDM_COMMIT:-6bb90217f80afef811abc11e790bc14fab853922}"
LOCAL_EDM_MIRROR="${LOCAL_EDM_MIRROR:-/cache/Zhengwei/DG-TWFD/refs/edm}"
EDM_ROOT="$ROOT_DIR/refs/edm"
BOOTSTRAP_MODE="${BOOTSTRAP_MODE:-clone_base}"
INSTALL_DEV_EXTRAS="${INSTALL_DEV_EXTRAS:-0}"

if [[ ! -d "$EDM_ROOT/.git" ]]; then
  if [[ -d "$LOCAL_EDM_MIRROR/.git" ]]; then
    git clone "$LOCAL_EDM_MIRROR" "$EDM_ROOT"
    git -C "$EDM_ROOT" remote set-url origin https://github.com/turingw1/edm.git
  else
    git clone https://github.com/turingw1/edm.git "$EDM_ROOT"
  fi
fi

git -C "$EDM_ROOT" checkout "$EDM_COMMIT"

dg_twfd_net_heavy
echo "Using heavy-download network profile for conda/pip installs"
dg_twfd_net_status

if [[ ! -d "$ENV_PREFIX" ]]; then
  if [[ "$BOOTSTRAP_MODE" == "clone_base" ]]; then
    conda create -y -p "$ENV_PREFIX" --clone base
  else
    conda create -y -p "$ENV_PREFIX" python="$PYTHON_VERSION" pip
  fi
fi

conda run -p "$ENV_PREFIX" python -m pip install --upgrade pip setuptools wheel
if [[ "$BOOTSTRAP_MODE" != "clone_base" ]]; then
  conda run -p "$ENV_PREFIX" python -m pip install --index-url "$TORCH_INDEX_URL" torch torchvision
fi
conda run -p "$ENV_PREFIX" python -m pip install --no-build-isolation -e "$ROOT_DIR[teacher]"
if [[ "$INSTALL_DEV_EXTRAS" == "1" ]]; then
  conda install -y -p "$ENV_PREFIX" pytest
fi
conda run -p "$ENV_PREFIX" python -m pip install \
  click \
  tqdm \
  pillow \
  scipy \
  requests \
  psutil \
  matplotlib \
  imageio \
  imageio-ffmpeg \
  pyspng \
  numpy \
  torch-fidelity \
  piq \
  modelscope

cat <<EOF
A100 single environment ready.

Activate with:
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  source "$ROOT_DIR/scripts/server/activate_a100_runtime.sh"
  conda activate "$ENV_PREFIX"

Then verify with:
  python "$ROOT_DIR/scripts/server/smoke_a100.py"
EOF
