#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
NVIDIA_SRC=${NVIDIA_SRC:-/usr/src/nvidia-srv-580.178.04}
NVCC=${NVCC:-/usr/bin/nvcc}
OUT=${OUT:-"$ROOT/build/libqwen38_vllm_rtx_peer.so"}
mkdir -p "$(dirname "$OUT")"
"$NVCC" -std=c++17 -O2 -lineinfo -shared -Xcompiler -fPIC \
  -gencode arch=compute_50,code=sm_50 \
  -gencode arch=compute_86,code=sm_86 \
  -I"$ROOT/transport/driver/abi/580.178.04" \
  -I"$ROOT/src" -I"$NVIDIA_SRC/common/inc" -I"$NVIDIA_SRC/nvidia-uvm" \
  "$ROOT/src/qwen38_vllm_rtx_peer.cu" -o "$OUT" -lcuda -ldl
sha256sum "$ROOT/src/qwen38_vllm_rtx_peer.cu" \
  "$ROOT/src/bench_qwen38_rtx_m10_retained_roundtrip.cu" \
  "$ROOT/src/resident_multicast_mapping.cuh" \
  "$ROOT/transport/driver/abi/580.178.04/uvm_ioctl.h" \
  "$ROOT/transport/driver/abi/580.178.04/uvm_linux_ioctl.h" "$OUT"
