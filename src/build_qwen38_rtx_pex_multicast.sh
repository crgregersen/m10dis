#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
NVIDIA_SRC=${NVIDIA_SRC:-/usr/src/nvidia-srv-580.178.04}
OUT=${OUT:-"$ROOT/build/qwen38_rtx_pex_multicast"}
SASS=${SASS:-"$ROOT/build/qwen38_rtx_pex_multicast.sass"}
mkdir -p "$(dirname "$OUT")" "$(dirname "$SASS")"
nvcc -std=c++17 -O2 -lineinfo \
  -gencode arch=compute_50,code=sm_50 \
  -gencode arch=compute_86,code=sm_86 \
  -I"$ROOT/transport/driver/abi/580.178.04" \
  -I"$NVIDIA_SRC/common/inc" \
  -I"$NVIDIA_SRC/nvidia-uvm" \
  "$ROOT/src/qwen38_rtx_pex_multicast.cu" -o "$OUT" -lcuda
cuobjdump --dump-sass "$OUT" > "$SASS"
sha256sum "$ROOT/src/qwen38_rtx_pex_multicast.cu" \
  "$ROOT/transport/driver/abi/580.178.04/uvm_ioctl.h" \
  "$ROOT/transport/driver/abi/580.178.04/uvm_linux_ioctl.h" \
  "$OUT" "$SASS"
