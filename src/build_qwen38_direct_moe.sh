#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
NVCC=${NVCC:-/usr/bin/nvcc}
HOST_CXX=${HOST_CXX:-/usr/bin/g++-13}
OUT=${OUT:-"$ROOT/build/libqwen38_direct_moe.so"}
mkdir -p "$(dirname "$OUT")"
"$NVCC" -O3 -std=c++17 -ccbin "$HOST_CXX" -arch=sm_50 -lineinfo \
  -Xcompiler=-fPIC -Xcompiler=-mavx2 -DQ38_DIRECT_LIBRARY -shared \
  "$ROOT/src/qwen38_direct_resident_client.cpp" -o "$OUT"
sha256sum "$ROOT/src/qwen38_direct_resident_client.cpp" \
  "$ROOT/src/resident_direct_mailbox.cuh" "$OUT"
