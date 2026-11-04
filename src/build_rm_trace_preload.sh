#!/usr/bin/env bash
# Build the RM ioctl export-capture preload (rm_ioctl_trace_preload.so) into build/.
# The vLLM client and the daemon LD_PRELOAD it to record the RM handles the RTX peer route needs.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
mkdir -p "$ROOT/build"
gcc -shared -fPIC -O2 -Wall -o "$ROOT/build/rm_ioctl_trace_preload.so" "$ROOT/src/rm_ioctl_trace_preload.c" -ldl
echo "BUILT,$ROOT/build/rm_ioctl_trace_preload.so"
