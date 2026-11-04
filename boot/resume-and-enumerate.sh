#!/usr/bin/env bash

# Phase 2: after S3 wake with PSU2 ON, discover late M10 branches,
# initialize NVIDIA, run checked CUDA work, and restore fan control.

set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

usage()
{
    cat <<EOF
Usage: $(basename "$0") [--verbose] [--link-wait SECONDS] [--slow-links]

Complete the RTX + M10 workflow after waking from deep S3 with PSU2 on.
This releases and scans each late M10, loads NVIDIA, runs checked CUDA work,
and restores adaptive fan control.

Options:
  -v, --verbose  Also print detailed inventories and per-GPU test results.
  --link-wait SECONDS  Link timeout, 1..300 seconds (default: 10).
                      Shared across ports normally; per port with --slow-links.
  --slow-links   Enable and wait for each port sequentially (original behavior).
  -h, --help  Show this help and exit.

Normal use after resume:
  sudo $0

Slower diagnostic run:
  sudo $0 --slow-links --link-wait 30 --verbose
EOF
}

VERBOSE=0
LINK_WAIT_SECONDS=10
SLOW_LINKS=0
while (( $# )); do
    case $1 in
        -v|--verbose) VERBOSE=1 ;;
        --slow-links) SLOW_LINKS=1 ;;
        --link-wait)
            [[ ${2:-} =~ ^[0-9]{1,3}$ ]] || die "--link-wait requires an integer from 1 to 300"
            LINK_WAIT_SECONDS=$((10#$2))
            (( LINK_WAIT_SECONDS >= 1 && LINK_WAIT_SECONDS <= 300 )) || \
                die "--link-wait must be from 1 to 300 seconds"
            shift
            ;;
        -h|--help) usage; exit 0 ;;
        *) usage >&2; exit 2 ;;
    esac
    shift
done

stage()
{
    printf '\n[%s/6] %s\n' "$1" "$2"
}

ok()
{
    printf '       OK  %s\n' "$*"
}

show_if_verbose()
{
    [[ $VERBOSE -eq 1 ]] && cat "$1"
    return 0
}

require_root
require_commands lspci setpci nvidia-smi systemctl modprobe modinfo fuser flock find journalctl udevadm
acquire_workflow_lock
load_state
[[ $PHASE == isolated ]] || die "expected isolated phase, found $PHASE"
[[ $(</proc/sys/kernel/random/boot_id) == "$BOOT_ID" ]] || \
    die "boot ID changed; this is not the matching S3 resume"
validate_static_environment

printf '\nTesla M10 late-power resume\n'
printf '───────────────────────────\n'
printf 'Kernel: %s\n' "$(uname -r)"
printf 'Run:    %s\n' "$RUN_DIR"

[[ -e /sys/bus/pci/devices/$RTX_ROOT_BDF ]] || die "saved RTX parent bridge is absent: $RTX_ROOT_BDF"
[[ $(device_class "$RTX_ROOT_BDF") == 0x060400 ]] || die "saved RTX parent is not a PCI bridge: $RTX_ROOT_BDF"
if [[ -e /sys/bus/pci/devices/$RTX_GPU_BDF ]]; then
    [[ $(device_id "$RTX_GPU_BDF") == "$RTX_GPU_ID" ]] || die "saved RTX BDF now identifies another device"
    [[ $(parent_bdf "$RTX_GPU_BDF") == "$RTX_ROOT_BDF" ]] || \
        die "RTX moved to a different parent bridge since suspend"
    [[ $(device_id "$RTX_AUDIO_BDF") == "$RTX_AUDIO_ID" ]] || die "saved RTX audio BDF is inconsistent"
fi
SAVED_LATE_PORTS_STRING=$LATE_PORTS_STRING
select_late_m10_ports
[[ $SAVED_LATE_PORTS_STRING == "$LATE_PORTS_STRING" ]] || \
    die "saved late-M10 port list does not match the current RTX/PLX topology"

stage 1 "Validate the S3 resume state"
capture_inventory "$RUN_DIR/after-resume-before-release.txt"
capture_bridge_words "$RUN_DIR/after-resume-bridge-registers.tsv" \
    "$RTX_ROOT_BDF" "${LATE_PORTS[@]}"

bridge_is_isolated "$RTX_ROOT_BDF" || \
    die "RTX parent $RTX_ROOT_BDF was not held isolated across S3"
for port in "${LATE_PORTS[@]}"; do
    bridge_is_isolated "$port" || \
        die "$port was not held with Link Disable and Secondary Bus Reset across S3"
done
ok "saved bridge isolation survived S3"

stage 2 "Prepare fixed cooling and a quiet PCI probe window"
# S3 can relocate the EC I/O base. Reprobe even when the correct module is
# already bound: its cached pre-suspend address may no longer be usable.
ensure_patched_fan_driver 1
start_fixed_cooling
udevadm control --stop-exec-queue
UDEV_QUEUE_STOPPED=1
resume_udev_on_exit()
{
    if [[ ${UDEV_QUEUE_STOPPED:-0} -eq 1 ]]; then
        udevadm control --start-exec-queue || true
    fi
}
trap resume_udev_on_exit EXIT

# Firmware normally restores and rediscovers the RTX during S3 resume. Udev may
# therefore have reloaded NVIDIA before this script starts; unload it again.
stop_nvidia_stack
ok "cooling fixed, udev held, NVIDIA stack unloaded"

if [[ ! -e /sys/bus/pci/devices/$RTX_GPU_BDF ]]; then
    log "RTX was not rediscovered automatically; releasing and rescanning $RTX_ROOT_BDF"
    clear_word_bit "$RTX_ROOT_BDF" CAP_EXP+10.w 0x0010
    status=$(wait_for_dll_active "$RTX_ROOT_BDF") || \
        die "RTX root link did not become active (last status $status)"
    clear_word_bit "$RTX_ROOT_BDF" BRIDGE_CONTROL 0x0040
    sleep 1
    printf '1\n' >"/sys/bus/pci/devices/$RTX_ROOT_BDF/rescan"
    sleep 2
fi

[[ -e /sys/bus/pci/devices/$RTX_GPU_BDF ]] || die "RTX GPU did not return"
[[ -e /sys/bus/pci/devices/$RTX_AUDIO_BDF ]] || die "RTX audio function did not return"
pin_device_d0 "$RTX_ROOT_BDF"
pin_device_d0 "$RTX_GPU_BDF"
pin_device_d0 "$RTX_AUDIO_BDF"

stage 3 "Discover late-powered M10s on ${#LATE_PORTS[@]} reserved branches"
printf 'mode=%s\nlink_wait_seconds=%s\n' "$SLOW_LINKS" "$LINK_WAIT_SECONDS" >"$RUN_DIR/link-wait-settings.txt"
if (( SLOW_LINKS )); then
    printf '       Sequential link checks: up to %d seconds per port\n' "$LINK_WAIT_SECONDS"
else
# Train links together while retaining Secondary Bus Reset and the udev hold.
# Empty ports share one bounded wait instead of consuming ten seconds each.
printf '       Checking all reserved links (up to %d seconds total) ... ' "$LINK_WAIT_SECONDS"
for port in "${LATE_PORTS[@]}"; do
    clear_word_bit "$port" CAP_EXP+10.w 0x0010
done
link_wait_deadline=$((SECONDS + LINK_WAIT_SECONDS))
while :; do
    all_links_active=1
    for port in "${LATE_PORTS[@]}"; do
        status=$(bridge_word "$port" CAP_EXP+12.w)
        if (( (16#$status & 0x2000) == 0 )); then
            all_links_active=0
        fi
    done
    if (( all_links_active || SECONDS >= link_wait_deadline )); then
        break
    fi
    sleep 0.1
done
printf 'done\n'
fi
late_index=0
late_card_count=0
for port in "${LATE_PORTS[@]}"; do
    ((late_index+=1))
    printf '       M10 bank %d via %s ... ' "$late_index" "$port"
    if (( SLOW_LINKS )); then
        clear_word_bit "$port" CAP_EXP+10.w 0x0010
        status=$(wait_for_dll_active "$port" "$((LINK_WAIT_SECONDS * 10))") || true
    else
        status=$(bridge_word "$port" CAP_EXP+12.w)
    fi
    if (( (16#$status & 0x2000) == 0 )); then
        # An unused reserved port has no link. Leave it isolated and record it.
        set_word_bit "$port" CAP_EXP+10.w 0x0010
        bridge_is_isolated "$port" || die "failed to re-isolate empty port $port"
        printf 'no active link (left isolated)\n'
        printf '%s\tNO_LINK\t%s\n' "$port" "$status" >>"$RUN_DIR/empty-ports.tsv"
        continue
    fi
    clear_word_bit "$port" BRIDGE_CONTROL 0x0040
    sleep 1
    printf '1\n' >"/sys/bus/pci/devices/$port/rescan"
    sleep 2
    pin_branch_d0 "$port"

    count=$(gm107_count_under_branch "$port")
    gm107s_under_branch "$port" >"$RUN_DIR/${port//[:.]/_}-gm107s.txt"
    [[ $count -eq 4 ]] || die "$port exposed $count GM107s; expected exactly four"
    ((late_card_count+=1))
    printf '%s\tlink_control=%s\tlink_status=%s\tbridge_control=%s\n' \
        "$port" \
        "$(bridge_word "$port" CAP_EXP+10.w)" \
        "$(bridge_word "$port" CAP_EXP+12.w)" \
        "$(bridge_word "$port" BRIDGE_CONTROL)" \
        >>"$RUN_DIR/release-results.tsv"
    printf 'OK (%d GM107s)\n' "$count"
done

# Repeat after the entire hierarchy exists. This proactively prevents the
# internal M10 downstream ports from entering D3hot before NVIDIA probes them.
for port in "${LATE_PORTS[@]}"; do
    pin_branch_d0 "$port"
done

GM107_COUNT=$(pci_count "$GM107_ID")
EXPECTED_GM107_COUNT=$((4 * (1 + late_card_count)))
EXPECTED_GPU_COUNT=$((EXPECTED_GM107_COUNT + 1))
[[ $late_card_count -gt 0 ]] || die "no late M10 link became active; check PSU2 and cabling"
[[ $GM107_COUNT -eq $EXPECTED_GM107_COUNT ]] || die "PCI enumeration found $GM107_COUNT GM107s; expected $EXPECTED_GM107_COUNT"
[[ $(pci_count "$RTX_GPU_ID") -eq 1 ]] || die "RTX GPU is absent after M10 enumeration"

capture_inventory "$RUN_DIR/all-pci-devices-before-nvidia.txt"
capture_bridge_words "$RUN_DIR/released-bridge-registers.tsv" \
    "$RTX_ROOT_BDF" "${LATE_PORTS[@]}"

stage 4 "Initialize NVIDIA on the complete topology"
udevadm control --start-exec-queue
UDEV_QUEUE_STOPPED=0
trap - EXIT
udevadm settle --timeout=20 || true
modprobe nvidia
modprobe nvidia_uvm
systemctl start nvidia-persistenced.service

NVIDIA_COUNT=0
for _ in $(seq 1 60); do
    if NVIDIA_INVENTORY=$(nvidia-smi -L 2>/dev/null); then
        NVIDIA_COUNT=$(awk '/^GPU / { count++ } END { print count + 0 }' <<<"$NVIDIA_INVENTORY")
    else
        NVIDIA_COUNT=0
    fi
    [[ $NVIDIA_COUNT -eq $EXPECTED_GPU_COUNT ]] && break
    sleep 1
done
nvidia-smi -L >"$RUN_DIR/nvidia-smi-L.txt"
show_if_verbose "$RUN_DIR/nvidia-smi-L.txt"
[[ $NVIDIA_COUNT -eq $EXPECTED_GPU_COUNT ]] || die "NVIDIA initialized $NVIDIA_COUNT devices; expected $EXPECTED_GPU_COUNT"

M10_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | grep -c '^Tesla M10$' || true)
RTX_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | grep -c '^NVIDIA GeForce RTX 3070 Ti$' || true)
[[ $M10_COUNT -eq $EXPECTED_GM107_COUNT && $RTX_COUNT -eq 1 ]] || \
    die "unexpected NVIDIA inventory: M10=$M10_COUNT RTX3070Ti=$RTX_COUNT"
ok "$NVIDIA_COUNT devices initialized: $M10_COUNT GM107s plus RTX 3070 Ti"

[[ -x $SANITY_BIN ]] || die "CUDA sanity binary is missing; run $SCRIPT_DIR/build-sanity.sh first"
stage 5 "Run checked CUDA allocation, kernel, and readback"
if ! "$SANITY_BIN" >"$RUN_DIR/cuda-sanity-all-gpus.txt"; then
    tail -40 "$RUN_DIR/cuda-sanity-all-gpus.txt" >&2
    die "CUDA sanity program failed"
fi
show_if_verbose "$RUN_DIR/cuda-sanity-all-gpus.txt"
grep -q "^CUDA_DEVICE_COUNT,$EXPECTED_GPU_COUNT\$" "$RUN_DIR/cuda-sanity-all-gpus.txt" || \
    die "CUDA sanity did not see $EXPECTED_GPU_COUNT devices"
[[ $(grep -c 'match,YES$' "$RUN_DIR/cuda-sanity-all-gpus.txt") -eq $EXPECTED_GPU_COUNT ]] || \
    die "one or more CUDA devices failed checked readback"
ok "$EXPECTED_GPU_COUNT/$EXPECTED_GPU_COUNT devices returned the expected result"

stage 6 "Restore adaptive cooling and capture final health"
restore_adaptive_cooling
sleep 6
fan_summary=$(fan_bank_health) || \
    die "adaptive cooling did not produce sane readings on all six fan banks"

{
    date --iso-8601=ns
    uname -a
    echo '--- services ---'
    systemctl is-active m10-fancontrol.service m10-fixed-fans.service nvidia-persistenced.service || true
    echo '--- NVIDIA details ---'
    nvidia-smi --query-gpu=index,name,uuid,pci.bus_id,temperature.gpu,power.draw,pcie.link.gen.current,pcie.link.width.current --format=csv,noheader
    echo '--- loaded modules ---'
    modinfo -F filename nvidia
    modinfo -F version nvidia
    modinfo -F srcversion nvidia
    modinfo -F filename nvidia_uvm
    modinfo -F version nvidia_uvm
    modinfo -F srcversion nvidia_uvm
    lsmod | grep '^nvidia'
    echo '--- bridge state ---'
    cat "$RUN_DIR/released-bridge-registers.tsv"
    echo '--- relevant kernel messages since workflow start ---'
    journalctl -k -b --since "@$START_EPOCH" --no-pager \
        | grep -Ei 'NVRM|Xid|AER|DMAR.*fault|IOMMU.*fault|fallen off|D3cold|PCIe Bus Error' || true
} >"$RUN_DIR/final-health.txt"
show_if_verbose "$RUN_DIR/final-health.txt"

write_state complete
sha256sum "$RUN_DIR"/* >"$RUN_DIR/SHA256SUMS.tmp"
mv "$RUN_DIR/SHA256SUMS.tmp" "$RUN_DIR/SHA256SUMS"

ok "adaptive fan control restored ($fan_summary)"
ok "detailed health evidence saved"
printf '\nPASS  %d M10 cards and the RTX passed checked CUDA work.\n' "$((GM107_COUNT / 4))"
if [[ -s $RUN_DIR/empty-ports.tsv ]]; then
    printf '      %d reserved branch(es) had no active link; check against connected cards.\n' "$(wc -l <"$RUN_DIR/empty-ports.tsv")"
fi
printf '      PSU2 must remain on while these devices are registered.\n'
printf '      Artifacts: %s\n' "$RUN_DIR"

