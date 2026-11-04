#!/usr/bin/env bash
# Offline hardware test of worker mode 8 (batched MoE) against mode 1 on real M10 cards: routed kernel on
# card 0 (leader + 3 peers), shared kernel on card 5. Plain CUDA, no daemon / PTE / route / PEX state.
# Refuses if the resident daemon or any CUDA client runs. Card GPUs from qwen38_pci_discover.py.
# Queue: pueue add --group gpu --label <label> -- bash <this script>
set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/m10dis_env.sh"
ROOT=$M10DIS_HOME
REPO=$M10DIS_HOME
RUN=$M10DIS_RUNS/batched-moe-research/moe-batch-test-$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$RUN"
exec > >(tee -a "$RUN/run.log") 2>&1
echo "RUN_DIR,$RUN"
if pgrep -f 'qwen38_resident_daemon.py serve' >/dev/null; then echo 'ERROR,daemon running'; exit 4; fi
[[ -z $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) ]] || { echo 'ERROR,CUDA clients present'; exit 4; }
read -r ROUTED SHARED < <(python3 "$REPO/src/qwen38_pci_discover.py" --json | python3 -c 'import json,sys;g=json.load(sys.stdin)["groups"];print(",".join(g[0]["gpus"]),",".join(g[5]["gpus"]))')
echo "CARDS,routed,$ROUTED,shared,$SHARED"
inc=( -I"$REPO/transport/driver/abi/580.178.04" -I/usr/src/nvidia-srv-580.178.04/common/inc -I/usr/src/nvidia-srv-580.178.04/nvidia-uvm -I"$REPO/src" )
/usr/bin/nvcc -O2 -std=c++17 -gencode arch=compute_50,code=sm_50 "$REPO/src/test_q38_moe_batch.cu" "${inc[@]}" -o "$RUN/test_routed" > "$RUN/build.log" 2>&1
/usr/bin/nvcc -O2 -std=c++17 -gencode arch=compute_50,code=sm_50 -DTEST_SHARED "$REPO/src/test_q38_moe_batch.cu" "${inc[@]}" -o "$RUN/test_shared" >> "$RUN/build.log" 2>&1
sha256sum "$RUN"/test_* "$REPO"/src/test_q38_moe_batch.cu "$REPO"/src/resident_routed_batch.cuh "$REPO"/src/resident_batch_layout.cuh \
  "$REPO"/src/resident_dynamic_routed_v3.cuh "$REPO"/src/resident_dynamic_shared_v2.cuh "$REPO"/src/resident_cpu_mailbox_abi.cuh > "$RUN/hashes.sha256"
pass=0
"$RUN/test_routed" "$ROUTED" && pass=$((pass+1)) || true
"$RUN/test_shared" "$SHARED" && pass=$((pass+1)) || true
echo "Q38_MOE_BATCH_TESTS_PASSED,$pass,of,2"
[[ $pass -eq 2 ]] || { echo 'ERROR,batch test did not pass'; exit 1; }
echo "Q38_MOE_BATCH_TEST_RUN_OK,$RUN"
