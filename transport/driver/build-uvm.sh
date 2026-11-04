#!/usr/bin/env bash
# Build the patched nvidia-uvm.ko for driver 580.178.04 against the running kernel.
#   build-uvm.sh [out_dir]      (default: <repo>/build/uvm)
# Copies /usr/src/nvidia-srv-580.178.04 (or $NVIDIA_SRC), applies patches/ in order (README.md),
# builds only the nvidia-uvm module and prints its srcversion and SHA-256. Does not install or load it.
set -Eeuo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
NVIDIA_SRC=${NVIDIA_SRC:-/usr/src/nvidia-srv-580.178.04}
KDIR=${KDIR:-/lib/modules/$(uname -r)/build}
OUT=${1:-$ROOT/build/uvm}
PATCHES=(
  580.178.04-glm53-diagnostic-extension.patch
  580.178.04-qwen38-rtx-peer-multiroute.patch
  580.178.04-qwen38-rtx-peer-pte-capacity.patch
  580.178.04-qwen38-pte-lifecycle-cleanup.patch
  580.178.04-qwen38-pte-reap-dead.patch
)
[[ -d $NVIDIA_SRC/nvidia-uvm ]] || { echo "ERROR,no NVIDIA source at $NVIDIA_SRC"; exit 2; }
rm -rf "$OUT/src"; mkdir -p "$OUT"; cp -a "$NVIDIA_SRC" "$OUT/src"
for p in "${PATCHES[@]}"; do
  patch -d "$OUT/src" -p1 --forward --batch < "$HERE/patches/$p" > "$OUT/patch-$p.log" || { echo "ERROR,patch $p failed (see $OUT/patch-$p.log)"; exit 3; }
  echo "PATCHED,$p"
done
# Same invocation as the driver's dkms.conf (builds all modules; only nvidia-uvm.ko is used).
env -u ARCH CC=/usr/bin/gcc make -C "$OUT/src" -j"$(nproc)" NV_EXCLUDE_BUILD_MODULES='' \
  KERNEL_UNAME="$(uname -r)" IGNORE_XEN_PRESENCE=1 IGNORE_CC_MISMATCH=1 SYSSRC="$KDIR" LD=/usr/bin/ld.bfd CONFIG_X86_KERNEL_IBT= \
  modules > "$OUT/build.log" 2>&1 \
  || { echo "ERROR,build failed (see $OUT/build.log)"; exit 4; }
KO=$OUT/src/nvidia-uvm.ko
cp "$KO" "$OUT/nvidia-uvm-m10dis.ko"
echo "BUILT,$OUT/nvidia-uvm-m10dis.ko,srcversion,$(modinfo -F srcversion "$KO"),sha256,$(sha256sum "$KO" | cut -d' ' -f1),vermagic,$(modinfo -F vermagic "$KO" | cut -d' ' -f1)"
