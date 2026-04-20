#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REFS_DIR="${1:-${ROOT}/refs}"

mkdir -p "${REFS_DIR}"

repos=(
  "Min-SNR-Diffusion-Training|https://github.com/TiankaiHang/Min-SNR-Diffusion-Training"
  "OptimalSteps|https://github.com/bebebe666/OptimalSteps"
  "Rectified-Diffusion|https://github.com/G-U-N/Rectified-Diffusion"
  "conditional-flow-matching|https://github.com/atong01/conditional-flow-matching"
  "ctm-cifar10|https://github.com/Kim-Dongjun/ctm-cifar10.git"
  "edm|https://github.com/NVlabs/edm"
)

echo "Reference repo target: ${REFS_DIR}"

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
