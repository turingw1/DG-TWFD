#!/usr/bin/env bash

set -euo pipefail

ENV_NAME="${1:-dgfm_map}"

echo "Creating conda environment: ${ENV_NAME}"

conda create -n "${ENV_NAME}" python=3.10 -y

eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

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
  matplotlib \
  pillow \
  pytest

python -m pip install -e .

echo
echo "Environment ready: ${ENV_NAME}"
echo "Next:"
echo "  conda activate ${ENV_NAME}"
echo "  pytest tests/test_dgfm_map_branch.py tests/test_dgfm_velocity_model.py tests/test_dgfm_config.py tests/test_dgfm_overrides.py -q"
