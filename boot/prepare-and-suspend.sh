#!/usr/bin/env bash

# Phase 1: validate the cold baseline, remove/isolate RTX, isolate late-M10
# late-M10 ports, and enter deep S3. PSU2 must be OFF when this starts.

set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

usage()
{
    cat <<EOF
Usage: $(basename "$0") [--check-only]

Prepare the RTX + M10 late-power workflow and enter deep S3.
PSU2 must be off and only the boot M10 plus RTX may be online.

Options:
  --check-only  Validate prerequisites without changing machine state.
  -h, --help    Show this help and exit.

Normal use:
  sudo $0 --check-only
  sudo $0
EOF
}

CHECK_ONLY=0
case ${1:-} in
    '') ;;
    --check-only) CHECK_ONLY=1 ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
esac
[[ $# -le 1 ]] || { usage >&2; exit 2; }

require_root
require_commands lspci setpci nvidia-smi systemctl modprobe modinfo fuser flock find journalctl
acquire_workflow_lock
validate_static_environment

RTX_GPU_BDF=$(find_unique_bdf "$RTX_GPU_ID")
RTX_AUDIO_BDF=${RTX_GPU_BDF%.*}.1
[[ $(device_id "$RTX_AUDIO_BDF") == "$RTX_AUDIO_ID" ]] || \
    die "expected RTX audio function is absent at $RTX_AUDIO_BDF"
RTX_ROOT_BDF=$(parent_bdf "$RTX_GPU_BDF")
[[ $(device_class "$RTX_ROOT_BDF") == 0x060400 ]] || \
    die "RTX parent $RTX_ROOT_BDF is not a PCI bridge"
select_late_m10_ports

GM107_COUNT=$(pci_count "$GM107_ID")
[[ $GM107_COUNT -eq 4 ]] || \
    die "expected only the boot M10 (4 GM107s) with PSU2 off; found $GM107_COUNT"

for port in "${LATE_PORTS[@]}"; do
    [[ $(gm107_count_under_branch "$port") -eq 0 ]] || \
        die "late devices already exist below $port; PSU2 must be off at cold boot"
done

NVIDIA_COUNT=$(nvidia-smi -L | awk '/^GPU / { count++ } END { print count + 0 }')
[[ $NVIDIA_COUNT -eq 5 ]] || \
    die "expected 4 GM107s plus RTX in nvidia-smi; found $NVIDIA_COUNT devices"

log "baseline valid: kernel $(uname -r), 4 GM107s, RTX $RTX_GPU_BDF via $RTX_ROOT_BDF"
log "late-M10 scan ports: ${LATE_PORTS[*]}"
if [[ " ${DEFAULT_LATE_PORTS[*]} " == *" $RTX_ROOT_BDF "* ]]; then
    log "excluding RTX parent $RTX_ROOT_BDF from M10 scan/reset list"
fi
if [[ $CHECK_ONLY -eq 1 ]]; then
    log "check-only completed; no state was changed"
    exit 0
fi

if [[ -r $STATE_FILE ]]; then
    # shellcheck disable=SC1090
    source "$STATE_FILE"
    [[ ${PHASE:-} == complete ]] || \
        die "an unfinished workflow is recorded in $STATE_FILE (phase ${PHASE:-unknown})"
fi

RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)
RUN_DIR=$RESULT_ROOT/$RUN_ID
BOOT_ID=$(</proc/sys/kernel/random/boot_id)
START_EPOCH=$(date +%s)
install -d -m 0755 "$RUN_DIR"
capture_inventory "$RUN_DIR/before-isolation.txt"
cp -a "$SCRIPT_DIR/common.sh" "$SCRIPT_DIR/prepare-and-suspend.sh" \
    "$SCRIPT_DIR/resume-and-enumerate.sh" "$SCRIPT_DIR/build-sanity.sh" \
    "$SCRIPT_DIR/cuda_sanity_all_gpus.cu" \
    "$RUN_DIR/"
save_bridge_words "$BRIDGE_STATE_FILE" "$RTX_ROOT_BDF" "${LATE_PORTS[@]}"
cp "$BRIDGE_STATE_FILE" "$RUN_DIR/original-bridge-registers.tsv"
write_state preparing
ln -sfn "$RUN_DIR" "$RESULT_ROOT/LATEST"

log "verifying the patched fan driver and switching to fixed cooling"
ensure_patched_fan_driver
start_fixed_cooling
log "stopping NVIDIA users and unloading the driver"
stop_nvidia_stack

log "removing RTX functions from the Linux PCI model"
printf '1\n' >"/sys/bus/pci/devices/$RTX_AUDIO_BDF/remove"
printf '1\n' >"/sys/bus/pci/devices/$RTX_GPU_BDF/remove"
[[ ! -e /sys/bus/pci/devices/$RTX_AUDIO_BDF ]] || die "RTX audio removal failed"
[[ ! -e /sys/bus/pci/devices/$RTX_GPU_BDF ]] || die "RTX GPU removal failed"

log "asserting Link Disable and Secondary Bus Reset on RTX and late-M10 ports"
for bridge in "$RTX_ROOT_BDF" "${LATE_PORTS[@]}"; do
    set_word_bit "$bridge" CAP_EXP+10.w 0x0010
    set_word_bit "$bridge" BRIDGE_CONTROL 0x0040
    bridge_is_isolated "$bridge" || die "failed to isolate $bridge"
done

capture_bridge_words "$RUN_DIR/isolated-bridge-registers.tsv" \
    "$RTX_ROOT_BDF" "${LATE_PORTS[@]}"
capture_inventory "$RUN_DIR/isolated-inventory.txt"
write_state isolated
sync

cat <<EOF

The machine will now enter deep S3.

  1. Wait until the machine is fully asleep.
  2. Turn PSU2 ON.
  3. Wait about five seconds for the connected M10 boards to power up.
  4. Press the computer power button once to wake it.
  5. After Linux resumes, run:

     sudo $SCRIPT_DIR/resume-and-enumerate.sh

Do not turn PSU2 off after wake.
Artifacts: $RUN_DIR
EOF

log "entering deep S3"
systemctl suspend
log "systemctl suspend returned after resume; run resume-and-enumerate.sh"
printf '%s\n' "$(date --iso-8601=ns)" >"$RUN_DIR/suspend-returned-at.txt"

