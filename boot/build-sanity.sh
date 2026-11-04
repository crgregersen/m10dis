#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

usage()
{
    cat <<EOF
Usage: $(basename "$0")

Build the checked CUDA allocation/kernel/readback utility used by the
late-power resume validation. This command does not require sudo.

Environment:
  NVCC  Optional path to an SM50-capable nvcc compiler.

Options:
  -h, --help  Show this help and exit.
EOF
}

case ${1:-} in
    '') ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
esac
[[ $# -eq 0 ]] || { usage >&2; exit 2; }

NVCC=${NVCC:-$(command -v nvcc || true)}
[[ -n $NVCC && -x $NVCC ]] || { echo "nvcc was not found in PATH" >&2; exit 1; }

"$NVCC" --version

install -d -m 0755 "$SCRIPT_DIR/build"
"$NVCC" -O2 \
    -gencode arch=compute_50,code=sm_50 \
    -gencode arch=compute_50,code=compute_50 \
    "$SCRIPT_DIR/cuda_sanity_all_gpus.cu" \
    -o "$SCRIPT_DIR/build/cuda_sanity_all_gpus"

sha256sum \
    "$SCRIPT_DIR/cuda_sanity_all_gpus.cu" \
    "$SCRIPT_DIR/build/cuda_sanity_all_gpus"

