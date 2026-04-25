#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REFS_DIR="${1:-${ROOT}/refs}"

mkdir -p "${REFS_DIR}"

repos=(
  "flow_matching|https://github.com/facebookresearch/flow_matching.git"
  "consistency_models|https://github.com/openai/consistency_models.git"
  "min_snr|https://github.com/TiankaiHang/Min-SNR-Diffusion-Training.git"
  "optimalsteps|https://github.com/bebebe666/OptimalSteps.git"
  "rectified_diffusion|https://github.com/G-U-N/Rectified-Diffusion.git"
  "edm|https://github.com/turingw1/edm.git"
)

echo "Reference repo target: ${REFS_DIR}"

if [[ -f "${ROOT}/.gitmodules" ]]; then
  echo
  echo "==> tracked submodules: syncing"
  git -C "${ROOT}" submodule sync --recursive
  git -C "${ROOT}" submodule update --init --recursive refs/ctm refs/ctm-cifar10
fi

for item in "${repos[@]}"; do
  name="${item%%|*}"
  url="${item#*|}"
  target="${REFS_DIR}/${name}"

  if [[ -d "${target}/.git" ]]; then
    echo
    echo "==> ${name}: existing git repo, fetching"
    git -C "${target}" remote -v | sed -n '1,2p'
    git -C "${target}" fetch --all --prune
    git -C "${target}" status --short --branch
    continue
  fi

  if [[ -e "${target}" ]]; then
    echo
    echo "==> ${name}: skipped"
    echo "    ${target} exists but is not a git repo. Move it aside or remove it before cloning."
    continue
  fi

  echo
  echo "==> ${name}: cloning"
  git clone "${url}" "${target}"
done

echo
echo "Reference repo setup complete."
