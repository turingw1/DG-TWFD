#!/usr/bin/env bash
set -euo pipefail

PROJECT_NAME="${PROJECT_NAME:-DG-TWFD}"
CACHE_ROOT="${CACHE_ROOT:-/cache/Zhengwei/${PROJECT_NAME}}"
mkdir -p "$CACHE_ROOT/datasets"

cat <<'MSG'
DG-TWFD dataset recovery:
- CIFAR-10 is expected under the runtime dataset cache and can be restored by
  the existing project data-loader/download path when needed.
- ImageNet/ImageNet64 are large recoverable assets. Do not store their bodies in
  /temp; place them under /cache/Zhengwei/DG-TWFD/datasets or the compatible
  runtime cache.
MSG
