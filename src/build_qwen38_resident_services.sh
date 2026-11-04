#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
NVIDIA_SRC=${NVIDIA_SRC:-/usr/src/nvidia-srv-580.178.04}
NVCC=${NVCC:-/usr/bin/nvcc}
OUT_DIR=${OUT_DIR:-"$ROOT/build/resident-services"}
mkdir -p "$OUT_DIR"
common=( -std=c++17 -O2 -lineinfo -Xcompiler=-pthread -DQ38_SRC_DIR="\"$ROOT/src\""
  -gencode arch=compute_50,code=sm_50
  -I"$ROOT/transport/driver/abi/580.178.04" -I"$ROOT/src"
  -I"$NVIDIA_SRC/common/inc" -I"$NVIDIA_SRC/nvidia-uvm" )
"$NVCC" "${common[@]}" "$ROOT/src/resident_routed_persistent_multicast_experimental.cu" \
  -o "$OUT_DIR/resident_routed_service" -lcuda -ldl
"$NVCC" "${common[@]}" "$ROOT/src/resident_shared_persistent_multicast_experimental.cu" \
  -o "$OUT_DIR/resident_shared_service" -lcuda -ldl
sha256sum "$ROOT/src/resident_pdb_export.cuh" \
  "$ROOT/src/resident_routed_persistent_multicast_experimental.cu" \
  "$ROOT/src/resident_shared_persistent_multicast_experimental.cu" \
  "$ROOT/src/resident_multicast_mapping.cuh" \
  "$ROOT/transport/driver/abi/580.178.04/uvm_ioctl.h" \
  "$ROOT/transport/driver/abi/580.178.04/uvm_linux_ioctl.h" \
  "$OUT_DIR/resident_routed_service" "$OUT_DIR/resident_shared_service"
