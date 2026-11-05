#!/usr/bin/env bash
# Hot-swap the resident M10 worker kernels to the PCI-cluster build (routed v3 + shared v2 with
# worker mode 6 row gather and the startup clear of status/exit_reason), through the daemon's
# documented park + RELOAD_KERNELS path. The daemon keeps running; weights, PLE table and worker
# processes stay loaded. Refuses while a vLLM client or a peer client is active.
# Queue: pueue add --group gpu --label <label> -- sudo -n bash <this script>
set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/m10dis_env.sh"
ROOT=$M10DIS_HOME
REPO=$M10DIS_HOME
ENV_FILE=${Q38_FULL_DAEMON_ENV:-$M10DIS_RUNS/qwen38-full-daemon/latest/daemon.env}
PY=$REPO/.venv-vllm/bin/python
DAEMON="$PY $REPO/src/qwen38_resident_daemon.py"
NVCC=/usr/bin/nvcc; NVIDIA_SRC=/usr/src/nvidia-srv-580.178.04
RUN=$M10DIS_RUNS/pci-cluster-exchange/kernel-reload-$(date -u +%Y%m%dT%H%M%SZ)
[[ $EUID -eq 0 ]] || { echo 'ERROR,run as root (sudo -n)'; exit 2; }
mkdir -p "$RUN/build"
exec > >(tee -a "$RUN/run.log") 2>&1
echo "RUN_DIR,$RUN"
# shellcheck disable=SC1090
source "$ENV_FILE"; cp "$ENV_FILE" "$RUN/daemon.env"
CTRL=$Q38_FULL_CONTROL_SOCKET
echo "DAEMON_RUN,$Q38_FULL_RUN"
if pgrep -f run_vllm_resident_m10_client.py >/dev/null; then echo 'ERROR,vLLM client running'; exit 4; fi
$DAEMON status --control-socket "$CTRL" > "$RUN/daemon-status-before.json"
python3 - "$RUN/daemon-status-before.json" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))
assert s.get("status") == "READY" and s.get("layers_ready") in (48, 49), s.get("status")
pc = (s.get("peer_client") or {}).get("status")
assert pc in (None, "NONE"), f"peer client active: {pc}"
print("DAEMON_BEFORE,pid,%s,activity,%s,kernel_reload,%s" % (s.get("pid"), s.get("worker_activity"), json.dumps(s.get("kernel_reload"))[:300]))
PY
inc=( -I"$REPO/transport/driver/abi/580.178.04" -I"$NVIDIA_SRC/common/inc" -I"$NVIDIA_SRC/nvidia-uvm" -I"$REPO/src" )
# Q38_RELOAD_NVCC_FLAGS (2026-11-04): extra nvcc flags for a measurement build, e.g. -DQ38_MR_PROFILE_BUILD (leader-finish
# profile); empty = the default build. Recorded in the run log.
read -r -a extra_flags <<< "${Q38_RELOAD_NVCC_FLAGS:-}"
echo "RELOAD_NVCC_FLAGS,${Q38_RELOAD_NVCC_FLAGS:-none}"
nv() { "$NVCC" -O2 -std=c++17 -arch=sm_50 "${extra_flags[@]}" "$@" "${inc[@]}" --resource-usage -cubin >> "$RUN/build/cubins.log" 2>&1; }
nv "$REPO/src/resident_routed_kernel_v3_module.cu" -o "$RUN/build/routed_v3.cubin"
nv "$REPO/src/resident_shared_kernel_v2_module.cu" -o "$RUN/build/shared_v2.cubin"
sha256sum "$RUN"/build/*.cubin "$REPO"/src/pci_cluster_exchange.cuh "$REPO"/src/resident_dynamic_routed_v3.cuh "$REPO"/src/resident_routed_batch.cuh \
  "$REPO"/src/resident_dynamic_shared_v2.cuh "$REPO"/src/resident_batch_layout.cuh "$REPO"/src/resident_cpu_mailbox_abi.cuh > "$RUN/build-hashes.sha256"
cat "$RUN/build-hashes.sha256"
grep -E 'Function properties for resident_dynamic|Used [0-9]+ registers' "$RUN/build/cubins.log" | grep -A1 resident_dynamic || true
cp "$REPO"/src/pci_cluster_exchange.cuh "$REPO"/src/resident_batch_layout.cuh "$REPO"/src/resident_cpu_mailbox_abi.cuh "$REPO"/src/resident_routed_batch.cuh "$REPO"/src/resident_dynamic_routed_v3.cuh "$REPO"/src/resident_dynamic_shared_v2.cuh "$0" "$RUN/"

$DAEMON park-workers --control-socket "$CTRL" --clients-quiesced > "$RUN/park-workers.json"
cat "$RUN/park-workers.json"; echo
$DAEMON reload-kernels --control-socket "$CTRL" --routed-image "$RUN/build/routed_v3.cubin" \
  --shared-image "$RUN/build/shared_v2.cubin" --clients-quiesced > "$RUN/reload-kernels.json"
cat "$RUN/reload-kernels.json"; echo
python3 -c "import json,sys;r=json.load(open(sys.argv[1]));assert r.get('status')=='ok',r" "$RUN/reload-kernels.json" \
  || { echo 'ERROR,kernel reload failed; daemon left running (see reload-kernels.json)'; exit 10; }
$DAEMON status --control-socket "$CTRL" > "$RUN/daemon-status-after.json"
python3 - "$RUN/daemon-status-after.json" "$RUN/build/routed_v3.cubin" "$RUN/build/shared_v2.cubin" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))
assert s.get("status") == "READY" and s.get("layers_ready") in (48, 49), s.get("status")
assert s.get("worker_activity") == "PARKED", s.get("worker_activity")
kr = json.dumps(s.get("kernel_reload") or {})
assert sys.argv[2] in kr and sys.argv[3] in kr, "daemon does not record the new images"
print("KERNEL_RELOAD_STATE", kr[:500])
print("DAEMON_AFTER,pid,%s,activity,PARKED,layers,%s" % (s.get("pid"), s.get("layers_ready")))
PY
echo "PCI_CLUSTER_KERNEL_RELOAD_OK,$RUN"
