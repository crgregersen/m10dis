#!/usr/bin/env bash
# Swap the loaded nvidia_uvm module for the patched build (run as root after every reboot).
#   swap-uvm.sh <patched nvidia-uvm.ko>
# The stock module loads at boot. This script:
#   - checks the candidate's vermagic against the running kernel,
#   - keeps a copy of the module file currently on disk as the fallback,
#   - stops nvidia-persistenced, refuses to continue while any process holds /dev/nvidia*,
#   - reloads nvidia_uvm with the same module parameters, restores the fallback on any failure,
#   - appends loaded_srcversion= / loaded_candidate_sha256= to $M10DIS_STATE/uvm-swap.log, which
#     run_qwen38_full_daemon_start.sh checks before it starts the daemon.
# Stop m10-fancontrol first (it holds the NVIDIA device nodes; its stop hook sets the fans to full)
# and close nvtop / nvidia-smi watchers. Start m10-fancontrol again afterwards.
set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../src" && pwd)/m10dis_env.sh"
CANDIDATE=${1:?usage: swap-uvm.sh <patched nvidia-uvm.ko>}
[[ $(id -u) -eq 0 ]] || { echo 'Run with sudo.' >&2; exit 2; }
[[ $(modinfo -F vermagic "$CANDIDATE" | awk '{print $1}') == "$(uname -r)" ]] || { echo 'Candidate vermagic does not match the running kernel.' >&2; exit 8; }
STOCK=$(modinfo -n nvidia_uvm)
mkdir -p "$M10DIS_STATE/uvm-swap"
DIR=$M10DIS_STATE/uvm-swap/$(date -u +%Y%m%dT%H%M%SZ); mkdir -p "$DIR"
LOG=$M10DIS_STATE/uvm-swap.log
exec > >(tee -a "$LOG" "$DIR/swap.log") 2>&1
echo "swap_start=$(date --iso-8601=seconds)"
cp --preserve=mode,timestamps "$STOCK" "$DIR/nvidia-uvm-fallback.ko"
echo "fallback_path=$STOCK"
echo "fallback_sha256=$(sha256sum "$STOCK" | cut -d' ' -f1)"
echo "loaded_before_srcversion=$(cat /sys/module/nvidia_uvm/srcversion)"
NEW_SHA=$(sha256sum "$CANDIDATE" | cut -d' ' -f1); NEW_SRC=$(modinfo -F srcversion "$CANDIDATE")
echo "candidate_srcversion=$NEW_SRC"; echo "candidate_sha256=$NEW_SHA"

args=()
for p in /sys/module/nvidia_uvm/parameters/*; do
  [[ -r $p ]] || continue
  name=${p##*/}; value=$(cat "$p")
  printf '%s\t%s\n' "$name" "$value" >> "$DIR/parameters-before.tsv"
  [[ -n $value && $value != '(null)' ]] && args+=("$name=$value")
done

systemctl stop nvidia-persistenced.service; sleep 1
users=$(fuser /dev/nvidia-uvm /dev/nvidiactl /dev/nvidia[0-9]* 2>/dev/null || true)
if [[ -n $users ]]; then
  echo "ERROR: GPU device nodes still in use by pids: $users" >&2
  systemctl start nvidia-persistenced.service || true; exit 10
fi
rmmod nvidia_uvm
if ! insmod "$CANDIDATE" "${args[@]}"; then
  echo 'Candidate insert failed; restoring the fallback module.' >&2
  insmod "$DIR/nvidia-uvm-fallback.ko" "${args[@]}"; systemctl start nvidia-persistenced.service || true; exit 11
fi
actual=$(cat /sys/module/nvidia_uvm/srcversion)
if [[ $actual != "$NEW_SRC" ]]; then
  echo 'Loaded srcversion mismatch; restoring the fallback module.' >&2
  rmmod nvidia_uvm; insmod "$DIR/nvidia-uvm-fallback.ko" "${args[@]}"; systemctl start nvidia-persistenced.service || true; exit 12
fi
echo "loaded_srcversion=$actual"
echo "loaded_candidate_sha256=$NEW_SHA"
echo "loaded_at=$(journalctl -k -b -o short-iso --no-pager | grep 'nvidia_uvm: module uses' | tail -1 | cut -d' ' -f1)"
systemctl start nvidia-persistenced.service
echo 'PASS: patched UVM loaded. Set Q38_UVM_LAST_LOAD to loaded_at above (m10dis.local.env).'
