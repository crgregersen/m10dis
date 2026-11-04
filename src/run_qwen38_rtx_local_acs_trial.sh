#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/m10dis_env.sh"

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [[ $EUID -ne 0 ]]; then
  echo "run as root: sudo $0 ACTIVE_GROUPS.tsv GPU_MAPPINGS.tsv CONTROL_SOCKET PEX_SNAPSHOT_ROOT [COMMAND ...]" >&2
  exit 2
fi
if [[ $# -lt 4 ]]; then
  echo "usage: $0 ACTIVE_GROUPS.tsv GPU_MAPPINGS.tsv CONTROL_SOCKET PEX_SNAPSHOT_ROOT [COMMAND ...]" >&2
  exit 2
fi

# Port addresses are discovered by scanning sysfs (qwen38_pci_discover.py) and
# checked against ACTIVE_GROUPS.tsv; they are never hardcoded. The RTX outer port
# and the target M10 card's outer port are sibling downstream ports on the common
# outer PEX8748. Clear only ACS request and completion redirection
# (PCI_ACS_RR | PCI_ACS_CR = 0x000c); root port and all other branches stay untouched.
source "$ROOT/src/qwen38_pci_env.sh"
q38_discover "$1" || exit 3
read -ra _OUTER <<< "$Q38_GROUP_OUTER_PORTS"
read -ra _INNER <<< "$Q38_GROUP_INNER_PORTS"
PORTS=("${_OUTER[0]}" "$Q38_RTX_OUTER_PORT")
if [[ ${Q38_ACS_ALL_M10_BRANCHES:-0} == 1 ]]; then
  PORTS=("${_OUTER[@]}" "$Q38_RTX_OUTER_PORT")
elif [[ ${Q38_ACS_INCLUDE_OWNER_PORT:-0} == 1 ]]; then
  PORTS+=("${_INNER[0]}")
fi
# Q38_ACS_EXTRA_PORTS (optional): space-separated inner PEX8747 downstream ports for
# same-card GPU->GPU writes (take them from Q38_GROUP_INNER_PORTS). Each must be a
# discovered inner port; saved, cleared, verified and restored like the others.
EXTRA_PORTS=(${Q38_ACS_EXTRA_PORTS:-})
for bdf in "${EXTRA_PORTS[@]}"; do
  [[ " ${_INNER[*]} " == *" $bdf "* ]] || { echo "ERROR,extra ACS port $bdf is not a discovered PEX8747 downstream port"; exit 2; }
  [[ " ${PORTS[*]} " == *" $bdf "* ]] || PORTS+=("$bdf")
done
is_inner_port() { [[ " ${_INNER[*]} " == *" $1 "* ]]; }
MODE=${Q38_RTX_TRIAL_MODE:-rtx}
[[ $MODE == rtx || $MODE == rtx-direct || $MODE == rtx-bus || $MODE == rtx-bus-all || $MODE == peer-layer ]] || {
  echo "ERROR,unsupported trial mode $MODE"; exit 2;
}
if [[ $MODE == peer-layer && $# -le 4 ]]; then
  echo 'ERROR,peer-layer mode requires a command after the four topology arguments'; exit 2
fi
readonly ACS_CTL=ECAP_ACS+6.w
readonly REDIRECT_MASK=0x000c
TRIAL="$M10DIS_RUNS/qwen38-rtx-m10-multicast/acs-route-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$TRIAL"
exec > >(tee -a "$TRIAL/trial.log") 2>&1

exec 9>/run/lock/qwen38-rtx-local-acs.lock
flock -n 9 || { echo 'ERROR,another RTX local-route trial holds the lock'; exit 1; }

declare -A ORIGINAL TEMPORARY
for bdf in "${PORTS[@]}"; do
  class=$(cat "/sys/bus/pci/devices/$bdf/class")
  vendor=$(cat "/sys/bus/pci/devices/$bdf/vendor")
  device=$(cat "/sys/bus/pci/devices/$bdf/device")
  expected_device=0x8748
  is_inner_port "$bdf" && expected_device=0x8747
  [[ $class == 0x060400 && $vendor == 0x10b5 && $device == "$expected_device" ]] || {
    echo "ERROR,$bdf identity changed: class=$class vendor=$vendor device=$device"; exit 1;
  }
  cap=$(setpci -s "$bdf" ECAP_ACS+2.w)
  [[ $cap == b701 ]] || { echo "ERROR,$bdf ACS capability identity unexpected: $cap"; exit 1; }
  ORIGINAL[$bdf]=$(setpci -s "$bdf" "$ACS_CTL")
  orig=$((16#${ORIGINAL[$bdf]}))
  TEMPORARY[$bdf]=$(printf '%04x' "$((orig & ~REDIRECT_MASK))")
done

{
  printf 'utc\t%s\n' "$(date -u --iso-8601=ns)"
  printf 'stage\tbdf\tdevice\toriginal_control\ttemporary_control\tchanged_mask\n'
  for bdf in "${PORTS[@]}"; do
    role=outer_pex8749_downstream; id=10b5:8748
    if is_inner_port "$bdf"; then role=m10_pex8747_downstream; id=10b5:8747; fi
    printf '%s\t%s\t%s\t0x%s\t0x%s\t0x000c\n' \
      "$role" "$bdf" "$id" "${ORIGINAL[$bdf]}" "${TEMPORARY[$bdf]}"
  done
} | tee "$TRIAL/acs-controls.tsv"

capture_route_state() {
  local suffix=$1 bdf
  for bdf in "${PORTS[@]}"; do lspci -D -s "$bdf" -vv; done > "$TRIAL/pci-route-$suffix.txt"
  dmesg --ctime > "$TRIAL/dmesg-$suffix.txt"
}
capture_route_state before

# Check the words again immediately before the first mutation.
for bdf in "${PORTS[@]}"; do
  now=$(setpci -s "$bdf" "$ACS_CTL")
  [[ $now == "${ORIGINAL[$bdf]}" ]] || {
    echo "ERROR,ACS changed after snapshot on $bdf: saved=${ORIGINAL[$bdf]} now=$now"; exit 1;
  }
done

changed=()
restore_controls() {
  local bdf now
  for ((i=${#changed[@]}-1; i>=0; --i)); do
    bdf=${changed[$i]}
    setpci -s "$bdf" "$ACS_CTL=${ORIGINAL[$bdf]}"
    now=$(setpci -s "$bdf" "$ACS_CTL")
    printf 'ACS_RESTORE,bdf,%s,expected,0x%s,observed,0x%s,match,%s\n' \
      "$bdf" "${ORIGINAL[$bdf]}" "$now" "$([[ $now == "${ORIGINAL[$bdf]}" ]] && echo 1 || echo 0)"
    [[ $now == "${ORIGINAL[$bdf]}" ]] || return 1
  done
  changed=()
}

for bdf in "${PORTS[@]}"; do
  setpci -s "$bdf" "$ACS_CTL=${TEMPORARY[$bdf]}"
  changed+=("$bdf")
  now=$(setpci -s "$bdf" "$ACS_CTL")
  printf 'ACS_APPLY,bdf,%s,original,0x%s,requested,0x%s,observed,0x%s\n' \
    "$bdf" "${ORIGINAL[$bdf]}" "${TEMPORARY[$bdf]}" "$now"
  [[ $now == "${TEMPORARY[$bdf]}" ]] || {
    echo "ERROR,ACS readback mismatch; restoring controls before any GPU probe"
    restore_controls
    exit 1
  }
done

set +e
if [[ $MODE == peer-layer ]]; then
  shift 4
  "$@" 2>&1 | tee "$TRIAL/poc.log"
else
  Q38_RTX_MODE="$MODE" Q38_ACS_TRIAL_FILE="$TRIAL/acs-controls.tsv" \
    "$ROOT/src/run_qwen38_rtx_pex_multicast.sh" "$@" 2>&1 | tee "$TRIAL/poc.log"
fi
poc_rc=${PIPESTATUS[0]}
set -e

# The PTE/DMA backing and GPU execution must be demonstrably quiescent before
# restoring route controls. On ambiguity, intentionally leave the saved route
# and resources in place for recovery; do not run an unconditional exit trap.
if ! rg -q '^SAFE_TO_UNMAP,all acquired GPU and PTE objects released$' "$TRIAL/poc.log"; then
  echo "PRESERVE,unsafe or ambiguous PoC cleanup; ACS remains temporary; trial=$TRIAL"
  exit 90
fi

restore_controls
capture_route_state after
printf 'ACS_RESTORE_COMPLETE,utc,%s\n' "$(date -u --iso-8601=ns)"
printf 'POC_EXIT,%d\n' "$poc_rc"

run_dir=$(sed -n 's/^RUN_DIR,//p' "$TRIAL/poc.log" | tail -1)
if [[ -n $run_dir && -d $run_dir ]]; then
  printf '%s\n' "$run_dir" | tee "$TRIAL/poc-artifact-dir.txt"
  echo "POC_ARTIFACT_DIR,$run_dir"
else
  echo 'WARNING,PoC artifact directory could not be resolved from its run log'
fi

if [[ $poc_rc -ne 0 ]]; then exit "$poc_rc"; fi
echo "ACS_TRIAL_RESULT,PASS,artifacts,$TRIAL"
