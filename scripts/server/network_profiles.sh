#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "This script must be sourced, not executed." >&2
  echo "Example: source scripts/server/network_profiles.sh && dg_twfd_net_heavy" >&2
  exit 1
fi

DG_TWFD_LOCAL_PROXY="${DG_TWFD_LOCAL_PROXY:-http://127.0.0.1:8080}"
DG_TWFD_HF_ENDPOINT="${DG_TWFD_HF_ENDPOINT:-https://hf-mirror.com}"
DG_TWFD_PIP_INDEX_URL="${DG_TWFD_PIP_INDEX_URL:-http://repo.myhuaweicloud.com/repository/pypi/simple}"
DG_TWFD_PIP_TRUSTED_HOST="${DG_TWFD_PIP_TRUSTED_HOST:-repo.myhuaweicloud.com}"
DG_TWFD_CIFAR10_URL="${DG_TWFD_CIFAR10_URL:-https://mirrors.dotsrc.org/osdn/datasets/74526/cifar-10-python.tar.gz}"
DG_TWFD_CONDARC="${DG_TWFD_CONDARC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/scripts/server/condarc_heavy_downloads.yaml}"
DG_TWFD_NO_PROXY="${DG_TWFD_NO_PROXY:-localhost,127.0.0.1,::1,0.0.0.0}"

_dg_twfd_unset_proxy_vars() {
  unset http_proxy
  unset https_proxy
  unset ftp_proxy
  unset all_proxy
  unset HTTP_PROXY
  unset HTTPS_PROXY
  unset FTP_PROXY
  unset ALL_PROXY
}

dg_twfd_net_proxy() {
  export http_proxy="$DG_TWFD_LOCAL_PROXY"
  export https_proxy="$DG_TWFD_LOCAL_PROXY"
  export HTTP_PROXY="$DG_TWFD_LOCAL_PROXY"
  export HTTPS_PROXY="$DG_TWFD_LOCAL_PROXY"
  export all_proxy="$DG_TWFD_LOCAL_PROXY"
  export ALL_PROXY="$DG_TWFD_LOCAL_PROXY"
  export no_proxy="$DG_TWFD_NO_PROXY"
  export NO_PROXY="$DG_TWFD_NO_PROXY"
  export HF_ENDPOINT="$DG_TWFD_HF_ENDPOINT"
  export HUGGINGFACE_HUB_ENDPOINT="$DG_TWFD_HF_ENDPOINT"
  export CIFAR10_URL="$DG_TWFD_CIFAR10_URL"
  unset HF_HUB_OFFLINE
  unset TRANSFORMERS_OFFLINE
  if [[ "${CONDARC:-}" == "$DG_TWFD_CONDARC" ]]; then
    unset CONDARC
  fi
  export DG_TWFD_NETWORK_PROFILE="proxy"
}

dg_twfd_net_heavy() {
  _dg_twfd_unset_proxy_vars
  export no_proxy="$DG_TWFD_NO_PROXY"
  export NO_PROXY="$DG_TWFD_NO_PROXY"
  export HF_ENDPOINT="$DG_TWFD_HF_ENDPOINT"
  export HUGGINGFACE_HUB_ENDPOINT="$DG_TWFD_HF_ENDPOINT"
  export PIP_INDEX_URL="$DG_TWFD_PIP_INDEX_URL"
  export PIP_TRUSTED_HOST="$DG_TWFD_PIP_TRUSTED_HOST"
  export CIFAR10_URL="$DG_TWFD_CIFAR10_URL"
  export CONDARC="$DG_TWFD_CONDARC"
  export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-${RUNTIME_ROOT:-/cache/Zhengwei/DG-TWFD-runtime}/.modelscope}"
  unset HF_HUB_OFFLINE
  unset TRANSFORMERS_OFFLINE
  export DG_TWFD_NETWORK_PROFILE="heavy"
}

dg_twfd_net_offline() {
  _dg_twfd_unset_proxy_vars
  export no_proxy="$DG_TWFD_NO_PROXY"
  export NO_PROXY="$DG_TWFD_NO_PROXY"
  export HF_ENDPOINT="$DG_TWFD_HF_ENDPOINT"
  export HUGGINGFACE_HUB_ENDPOINT="$DG_TWFD_HF_ENDPOINT"
  export CIFAR10_URL="$DG_TWFD_CIFAR10_URL"
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export DG_TWFD_NETWORK_PROFILE="offline"
}

dg_twfd_net_apply() {
  case "${1:-${DG_TWFD_NETWORK_PROFILE:-proxy}}" in
    proxy) dg_twfd_net_proxy ;;
    heavy) dg_twfd_net_heavy ;;
    offline) dg_twfd_net_offline ;;
    *)
      echo "Unknown DG-TWFD network profile: $1" >&2
      return 1
      ;;
  esac
}

dg_twfd_net_status() {
  echo "DG-TWFD network profile: ${DG_TWFD_NETWORK_PROFILE:-unset}"
  echo "  http_proxy=${http_proxy:-}"
  echo "  HTTP_PROXY=${HTTP_PROXY:-}"
  echo "  HF_ENDPOINT=${HF_ENDPOINT:-}"
  echo "  HUGGINGFACE_HUB_ENDPOINT=${HUGGINGFACE_HUB_ENDPOINT:-}"
  echo "  PIP_INDEX_URL=${PIP_INDEX_URL:-}"
  echo "  CIFAR10_URL=${CIFAR10_URL:-}"
  echo "  CONDARC=${CONDARC:-}"
  echo "  MODELSCOPE_CACHE=${MODELSCOPE_CACHE:-}"
}
