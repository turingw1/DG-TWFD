#!/usr/bin/env bash

set -euo pipefail

ENV_NAME="${1:-dgfm_map}"
ENV_ROOT="${2:-/cache/${USER}/conda_envs}"
ENV_PREFIX="${ENV_ROOT}/${ENV_NAME}"

echo "Creating conda environment:"
echo "  name:   ${ENV_NAME}"
echo "  prefix: ${ENV_PREFIX}"

mkdir -p "${ENV_ROOT}"
conda create -p "${ENV_PREFIX}" python=3.10 -y

eval "$(conda shell.bash hook)"
conda activate "${ENV_PREFIX}"

python -m pip install --upgrade pip setuptools wheel

# Match the current local reference environment used for map_branch.
python -m pip install \
  torch==2.10.0 torchvision==0.25.0 \
  --index-url https://download.pytorch.org/whl/cu128

python -m pip install \
  PyYAML==6.0.3 \
  numpy==2.2.3 \
  scipy==1.15.3 \
  torch-fidelity==0.4.0 \
  'diffusers>=0.30' \
  'transformers>=4.40' \
  'accelerate>=0.30' \
  'safetensors>=0.4' \
  matplotlib \
  pillow \
  pytest

python -m pip install -e .

echo
echo "Environment ready:"
echo "  prefix: ${ENV_PREFIX}"
echo "Next:"
echo "  conda activate ${ENV_PREFIX}"
echo "  pytest tests/test_dgfm_map_branch.py tests/test_dgfm_teacher_trajectory.py tests/test_dgfm_velocity_model.py tests/test_dgfm_config.py tests/test_dgfm_overrides.py -q"
