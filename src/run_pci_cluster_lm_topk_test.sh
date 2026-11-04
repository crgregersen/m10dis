#!/usr/bin/env bash
# Offline test of the PCI-cluster LM-head top-k (worker mode 7, pci_cluster_exchange.cuh) in the real
# resident worker kernels (routed v3 and shared v2), compiled for the RTX and run there:
# one persistent worker grid, a synthetic 1-MiB table in its own allocation, control page in mapped
# host memory (10,347-row synthetic LM slice). No M10, PTE, UVM route, mapper or daemon state is touched. The daemon's PLE owner may
# keep its own small context on the RTX; a running vLLM client is refused.
# Each kernel runs twice: plain, then under compute-sanitizer memcheck (catches any out-of-bounds
# table read, e.g. in the error cases that must not read at all).
# Queue: pueue add --group gpu --label <label> -- bash <this script>
set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/m10dis_env.sh"
ROOT=$M10DIS_HOME
REPO=$M10DIS_HOME
RUN=$M10DIS_RUNS/pci-cluster-exchange/lm-topk-$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$RUN"
exec > >(tee -a "$RUN/run.log") 2>&1
echo "RUN_DIR,$RUN"
RTX_UUID=$(python3 "$REPO/src/qwen38_pci_discover.py" --json | python3 -c 'import json,sys;print(json.load(sys.stdin)["rtx"]["uuid"])')
[[ $RTX_UUID == GPU-* ]] || { echo "ERROR,RTX discovery failed"; exit 3; }
echo "TEST_GPU,rtx,$RTX_UUID"
if pgrep -f run_vllm_resident_m10_client.py >/dev/null; then echo 'ERROR,vLLM client running'; exit 4; fi
inc=( -I"$REPO/transport/driver/abi/580.178.04" -I/usr/src/nvidia-srv-580.178.04/common/inc
      -I/usr/src/nvidia-srv-580.178.04/nvidia-uvm -I"$REPO/src" )
for v in routed shared; do
  def=(); [[ $v == shared ]] && def=(-DTEST_SHARED)
  /usr/bin/nvcc -O2 -std=c++17 -gencode arch=compute_86,code=sm_86 "${def[@]}" -lineinfo \
    "$REPO/src/test_pci_cluster_lm_topk.cu" "${inc[@]}" -o "$RUN/test_lm_$v" >> "$RUN/build.log" 2>&1
  # The same sources must also build for the M10 (sm_50) worker cubins.
  /usr/bin/nvcc -O2 -std=c++17 -arch=sm_50 "$REPO/src/resident_${v}_kernel_v$([[ $v == routed ]] && echo 3 || echo 2)_module.cu" \
    "${inc[@]}" --resource-usage -cubin -o "$RUN/${v}_sm50.cubin" >> "$RUN/build.log" 2>&1
done
sha256sum "$RUN"/test_lm_* "$RUN"/*.cubin "$REPO"/src/pci_cluster_exchange.cuh "$REPO"/src/resident_dynamic_routed_v3.cuh \
  "$REPO"/src/resident_dynamic_shared_v2.cuh "$REPO"/src/test_pci_cluster_lm_topk.cu > "$RUN/hashes.sha256"
grep -E 'Function properties for resident_dynamic|Used [0-9]+ registers' "$RUN/build.log" | grep -A1 'resident_dynamic' || true
pass=0
for v in routed shared; do
  echo "=== $v plain"
  if CUDA_VISIBLE_DEVICES=$RTX_UUID "$RUN/test_lm_$v" > "$RUN/$v.plain.log" 2>&1; then :; fi
  cat "$RUN/$v.plain.log"
  grep -q '^PCI_CLUSTER_LM_TOPK_TEST,PASS' "$RUN/$v.plain.log" && pass=$((pass+1))
  echo "=== $v memcheck"
  if CUDA_VISIBLE_DEVICES=$RTX_UUID timeout 600 compute-sanitizer --tool memcheck --error-exitcode 9 \
       "$RUN/test_lm_$v" > "$RUN/$v.memcheck.log" 2>&1; then
    grep -q '^PCI_CLUSTER_LM_TOPK_TEST,PASS' "$RUN/$v.memcheck.log" && pass=$((pass+1))
  fi
  grep -E 'PCI_CLUSTER_LM_TOPK_TEST|ERROR SUMMARY|Invalid|out of bounds' "$RUN/$v.memcheck.log" | head -8
done
echo "PCI_CLUSTER_LM_TOPK_RUNS_PASSED,$pass,of,4"
[[ $pass -eq 4 ]] || { echo 'ERROR,LM top-k test did not PASS in all 4 runs'; exit 1; }
echo "PCI_CLUSTER_LM_TOPK_TEST_RUN_OK,$RUN"
