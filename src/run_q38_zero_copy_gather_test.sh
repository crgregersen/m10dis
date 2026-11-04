#!/usr/bin/env bash
# Offline test of the zero-copy embedding gather (Q38_EMBEDDING_ZERO_COPY=1) on the RTX only:
# builds the peer library (runner flags), pins a synthetic 1.27-GB BF16 table, checks eager, CUDA-graph
# replay and out-of-range ids against torch.index_select. No M10, route or daemon state is touched.
# Queue: pueue add --group gpu --label <label> -- bash <this script>
set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/m10dis_env.sh"
ROOT=$M10DIS_HOME
REPO=$M10DIS_HOME
NVIDIA_SRC=/usr/src/nvidia-srv-580.178.04
RUN=$M10DIS_RUNS/pci-cluster-exchange/zero-copy-gather-$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$RUN"
exec > >(tee -a "$RUN/run.log") 2>&1
echo "RUN_DIR,$RUN"
if pgrep -f run_vllm_resident_m10_client.py >/dev/null; then echo 'ERROR,vLLM client running'; exit 4; fi
RTX_UUID=$(python3 "$REPO/src/qwen38_pci_discover.py" --json | python3 -c 'import json,sys;print(json.load(sys.stdin)["rtx"]["uuid"])')
[[ $RTX_UUID == GPU-* ]] || { echo "ERROR,RTX discovery failed"; exit 3; }
echo "TEST_GPU,rtx,$RTX_UUID"
LIB=$RUN/libqwen38_vllm_rtx_peer.so
/usr/bin/nvcc -std=c++17 -O2 -lineinfo -shared -Xcompiler -fPIC -DQ38_RTX_PEER_SMALL_LEAVES -DQ38_RTX_PEER_LEADER_AGG \
  -gencode arch=compute_50,code=sm_50 -gencode arch=compute_86,code=sm_86 \
  -I"$REPO/transport/driver/abi/580.178.04" -I"$REPO/src" -I"$NVIDIA_SRC/common/inc" -I"$NVIDIA_SRC/nvidia-uvm" \
  "$REPO/src/qwen38_vllm_rtx_peer.cu" -o "$LIB" -lcuda -ldl > "$RUN/build.log" 2>&1
sha256sum "$LIB" "$REPO/src/qwen38_cpu_head_hooks.py" "$REPO/src/test_q38_zero_copy_gather.py" > "$RUN/hashes.sha256"
CUDA_VISIBLE_DEVICES=$RTX_UUID "$REPO/.venv-vllm/bin/python" "$REPO/src/test_q38_zero_copy_gather.py" "$LIB" || true
grep -q '^ZERO_COPY_GATHER_TEST,PASS' "$RUN/run.log" || { echo 'ERROR,zero-copy gather test did not PASS'; exit 1; }
echo "ZERO_COPY_GATHER_TEST_RUN_OK,$RUN"
