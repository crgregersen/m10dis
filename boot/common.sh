#!/usr/bin/env bash

# Shared definitions for the RTX + up-to-eight-M10 S3 late-power workflow.
# This file is sourced by the two operator-facing scripts.

set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RESULT_ROOT=${RESULT_ROOT:-$SCRIPT_DIR/results}
STATE_DIR=${STATE_DIR:-/run/m10-rtx-late-power}
STATE_FILE=$STATE_DIR/state.env
BRIDGE_STATE_FILE=$STATE_DIR/bridge-registers.tsv
LOCK_FILE=$STATE_DIR/workflow.lock
SANITY_BIN=${SANITY_BIN:-$SCRIPT_DIR/build/cuda_sanity_all_gpus}

EXPECTED_KERNEL=${EXPECTED_KERNEL:-$(uname -r)}
NCT6687_MODULE=${NCT6687_MODULE:-/lib/modules/$EXPECTED_KERNEL/extra/nct6687.ko}
# Late branches = PEX8748 downstream ports in slots 0x09-0x13, the same rule the kernel quirk
# (kernel/m10-late-enable-quirks.patch) uses to reserve them. Discovered from sysfs, never hardcoded.
# The boot M10 sits on slot 0x08. Override with LATE_PORTS_OVERRIDE="bdf bdf ...".
discover_late_ports()
{
    local dev bdf slot parent
    for dev in /sys/bus/pci/devices/*; do
        [[ $(<"$dev/vendor") == 0x10b5 && $(<"$dev/device") == 0x8748 ]] || continue
        parent=$(basename "$(dirname "$(readlink -f "$dev")")")
        [[ -e /sys/bus/pci/devices/$parent/vendor && $(<"/sys/bus/pci/devices/$parent/vendor") == 0x10b5 ]] || continue
        bdf=$(basename "$dev"); slot=$((16#${bdf:8:2}))
        ((slot >= 0x09 && slot <= 0x13)) && echo "$bdf"
    done
}
if [[ -n ${LATE_PORTS_OVERRIDE:-} ]]; then
    read -r -a DEFAULT_LATE_PORTS <<< "$LATE_PORTS_OVERRIDE"
else
    mapfile -t DEFAULT_LATE_PORTS < <(discover_late_ports)
fi
GM107_ID=10de:13bd
RTX_GPU_ID=10de:2482
RTX_AUDIO_ID=10de:228b
PEX8748_ID=10b5:8748

# Build the late-M10 scan list from the kernel's reserved branches, excluding
# the bridge that currently parents the RTX. This matters when the RTX itself
# is installed on one of the PLX8748 downstream ports reserved for an M10.
select_late_m10_ports()
{
    local port
    LATE_PORTS=()
    for port in "${DEFAULT_LATE_PORTS[@]}"; do
        [[ $port == "${RTX_ROOT_BDF:-}" ]] && continue
        LATE_PORTS+=("$port")
    done
    ((${#LATE_PORTS[@]} > 0)) || die "no late-M10 branches remain after excluding RTX bridge ${RTX_ROOT_BDF:-unknown}"
    LATE_PORTS_STRING=${LATE_PORTS[*]}
}

log()
{
    printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$*"
}

die()
{
    log "ERROR: $*" >&2
    exit 1
}

require_root()
{
    [[ ${EUID:-$(id -u)} -eq 0 ]] || die "run this script with sudo"
}

require_commands()
{
    local command
    for command in "$@"; do
        command -v "$command" >/dev/null 2>&1 || die "required command is missing: $command"
    done
}

device_id()
{
    local bdf=$1
    [[ -r /sys/bus/pci/devices/$bdf/vendor ]] || return 1
    printf '%s:%s\n' \
        "$(<"/sys/bus/pci/devices/$bdf/vendor")" \
        "$(<"/sys/bus/pci/devices/$bdf/device")" | sed 's/0x//g'
}

device_class()
{
    local bdf=$1
    [[ -r /sys/bus/pci/devices/$bdf/class ]] || return 1
    cat "/sys/bus/pci/devices/$bdf/class"
}

find_unique_bdf()
{
    local pci_id=$1
    local -a matches=()
    mapfile -t matches < <(lspci -D -n -d "$pci_id" | awk '{print $1}')
    [[ ${#matches[@]} -eq 1 ]] || die "expected one $pci_id function, found ${#matches[@]}"
    printf '%s\n' "${matches[0]}"
}

pci_count()
{
    local pci_id=$1
    lspci -D -n -d "$pci_id" | awk 'END { print NR + 0 }'
}

parent_bdf()
{
    local bdf=$1
    basename "$(dirname "$(readlink -f "/sys/bus/pci/devices/$bdf")")"
}

bridge_word()
{
    local bdf=$1
    local register=$2
    setpci -s "$bdf" "$register"
}

set_word_bit()
{
    local bdf=$1
    local register=$2
    local mask=$3
    local value
    value=$(bridge_word "$bdf" "$register")
    setpci -s "$bdf" "$register=$(printf '%04x' $((16#$value | mask)))"
}

clear_word_bit()
{
    local bdf=$1
    local register=$2
    local mask=$3
    local value
    value=$(bridge_word "$bdf" "$register")
    setpci -s "$bdf" "$register=$(printf '%04x' $((16#$value & ~mask)))"
}

bridge_is_isolated()
{
    local bdf=$1
    local link_control bridge_control
    link_control=$(bridge_word "$bdf" CAP_EXP+10.w)
    bridge_control=$(bridge_word "$bdf" BRIDGE_CONTROL)
    (( (16#$link_control & 0x0010) != 0 && (16#$bridge_control & 0x0040) != 0 ))
}

wait_for_dll_active()
{
    local bdf=$1
    local attempts=${2:-100}
    local status=0000
    local attempt
    for attempt in $(seq 1 "$attempts"); do
        status=$(bridge_word "$bdf" CAP_EXP+12.w)
        if (( (16#$status & 0x2000) != 0 )); then
            printf '%s\n' "$status"
            return 0
        fi
        sleep 0.1
    done
    printf '%s\n' "$status"
    return 1
}

pin_device_d0()
{
    local bdf=$1
    local control=/sys/bus/pci/devices/$bdf/power/control
    [[ -e $control ]] || return 0
    printf 'on\n' >"$control"
}

pin_branch_d0()
{
    local bdf=$1
    local base vendor_file device_dir control
    base=$(readlink -f "/sys/bus/pci/devices/$bdf") || return 1
    pin_device_d0 "$bdf"
    while IFS= read -r vendor_file; do
        device_dir=${vendor_file%/vendor}
        control=$device_dir/power/control
        [[ -e $control ]] && printf 'on\n' >"$control"
    done < <(find "$base" -type f -name vendor -print | sort)
}

gm107s_under_branch()
{
    local bdf=$1
    local base vendor_file device_dir vendor device
    base=$(readlink -f "/sys/bus/pci/devices/$bdf") || return 1
    while IFS= read -r vendor_file; do
        device_dir=${vendor_file%/vendor}
        vendor=$(<"$vendor_file")
        device=$(<"$device_dir/device")
        if [[ $vendor == 0x10de && $device == 0x13bd ]]; then
            basename "$device_dir"
        fi
    done < <(find "$base" -type f -name vendor -print | sort)
}

gm107_count_under_branch()
{
    local bdf=$1
    gm107s_under_branch "$bdf" | awk 'END { print NR + 0 }'
}

nvidia_device_nodes_in_use()
{
    local -a nodes=()
    shopt -s nullglob
    nodes=(/dev/nvidia*)
    shopt -u nullglob
    [[ ${#nodes[@]} -eq 0 ]] && return 1
    fuser "${nodes[@]}" >/dev/null 2>&1
}

show_nvidia_device_users()
{
    local -a nodes=()
    shopt -s nullglob
    nodes=(/dev/nvidia*)
    shopt -u nullglob
    [[ ${#nodes[@]} -eq 0 ]] || fuser -v "${nodes[@]}" 2>&1 || true
}

stop_nvidia_stack()
{
    systemctl stop nvidia-persistenced.service 2>/dev/null || true
    if nvidia_device_nodes_in_use; then
        show_nvidia_device_users
        die "NVIDIA device nodes are busy; close nvtop and all CUDA clients"
    fi

    modprobe -r nvidia_uvm 2>/dev/null || true
    modprobe -r nvidia_drm 2>/dev/null || true
    modprobe -r nvidia_modeset 2>/dev/null || true
    modprobe -r nvidia 2>/dev/null || true

    if lsmod | grep -q '^nvidia'; then
        lsmod | grep '^nvidia' >&2
        die "the NVIDIA module stack did not unload completely"
    fi
}

ensure_patched_fan_driver()
{
    local vermagic bound_driver hwmon force_reload=${1:-0}
    bound_driver=
    for hwmon in /sys/class/hwmon/hwmon*; do
        [[ -r $hwmon/name && $(<"$hwmon/name") == nct6683 ]] || continue
        bound_driver=$(basename "$(readlink -f "$hwmon/device/driver")")
    done
    if [[ $force_reload == 0 && $bound_driver == nct6687 ]] && ! awk '$1 == "nct6683" {found=1} END {exit !found}' /proc/modules; then
        return 0
    fi

    [[ -r $NCT6687_MODULE ]] || \
        die "patched fan module is missing: $NCT6687_MODULE"
    vermagic=$(modinfo -F vermagic "$NCT6687_MODULE")
    [[ $vermagic == "$(uname -r) "* ]] || \
        die "patched fan module vermagic does not match $(uname -r): $vermagic"

    # Preserve a safe fixed output while replacing the stock monitoring driver.
    # The hardware retains its programmed PWM while the module is absent.
    systemctl stop m10-fancontrol.service 2>/dev/null || true
    systemctl stop m10-fixed-fans.service 2>/dev/null || true
    if [[ -n $bound_driver && $force_reload == 0 ]]; then
        /usr/local/libexec/m10-fan-full
    fi
    if awk '$1 == "nct6687" {found=1} END {exit !found}' /proc/modules; then
        modprobe -r nct6687
    fi
    if awk '$1 == "nct6683" {found=1} END {exit !found}' /proc/modules; then
        modprobe -r nct6683
    fi

    if ! modprobe nct6687 fan_mask=0x0f; then
        modprobe nct6683 force=1 2>/dev/null || true
        systemctl start m10-fixed-fans.service 2>/dev/null || true
        die "failed to load patched nct6687; attempted stock-driver cooling fallback"
    fi

    bound_driver=
    for hwmon in /sys/class/hwmon/hwmon*; do
        [[ -r $hwmon/name && $(<"$hwmon/name") == nct6683 ]] || continue
        bound_driver=$(basename "$(readlink -f "$hwmon/device/driver")")
    done
    [[ $bound_driver == nct6687 ]] || die "nct6687 loaded but did not bind the fan controller"

    systemctl start m10-fixed-fans.service
}

fan_bank_health()
{
    local label pwm_path rpm_path pwm rpm
    local -a banks=(
        'MB1:/run/m10-hwmon/pwm1:/run/m10-hwmon/fan1_input'
        'MB2:/run/m10-hwmon/pwm3:/run/m10-hwmon/fan4_input'
        'USB1:/run/m10-arctic-hwmon/pwm1:/run/m10-arctic-hwmon/fan1_input'
        'USB2:/run/m10-arctic-hwmon/pwm2:/run/m10-arctic-hwmon/fan2_input'
        'USB3:/run/m10-arctic-hwmon/pwm3:/run/m10-arctic-hwmon/fan3_input'
        'USB4:/run/m10-arctic-hwmon/pwm4:/run/m10-arctic-hwmon/fan4_input'
    )
    local -a summary=()

    /usr/local/libexec/m10-fancontrol-prepare
    for bank in "${banks[@]}"; do
        IFS=: read -r label pwm_path rpm_path <<<"$bank"
        pwm=$(<"$pwm_path")
        rpm=$(<"$rpm_path")
        if ! [[ $pwm =~ ^[0-9]+$ && $rpm =~ ^[0-9]+$ ]] || \
           (( pwm < 1 || pwm > 255 || rpm < 100 || rpm >= 60000 )); then
            printf '%s has invalid PWM/tachometer data: pwm=%s rpm=%s\n' \
                "$label" "$pwm" "$rpm" >&2
            return 1
        fi
        summary+=("$label=${rpm}rpm")
    done
    printf '%s\n' "${summary[*]}"
}

start_fixed_cooling()
{
    systemctl stop m10-fancontrol.service 2>/dev/null || true
    systemctl start m10-fixed-fans.service
}

restore_adaptive_cooling()
{
    systemctl stop m10-fixed-fans.service 2>/dev/null || true
    systemctl start m10-fancontrol.service
}

save_bridge_words()
{
    local output=$1
    shift
    local bdf
    : >"$output"
    printf 'bdf\tlink_control\tlink_status\tbridge_control\n' >>"$output"
    for bdf in "$@"; do
        printf '%s\t%s\t%s\t%s\n' \
            "$bdf" \
            "$(bridge_word "$bdf" CAP_EXP+10.w)" \
            "$(bridge_word "$bdf" CAP_EXP+12.w)" \
            "$(bridge_word "$bdf" BRIDGE_CONTROL)" >>"$output"
    done
}

capture_bridge_words()
{
    local output=$1
    shift
    save_bridge_words "$output" "$@"
}

capture_inventory()
{
    local output=$1
    {
        date --iso-8601=ns
        printf 'boot_id='; cat /proc/sys/kernel/random/boot_id
        uname -a
        printf 'cmdline='; cat /proc/cmdline
        printf 'mem_sleep='; cat /sys/power/mem_sleep
        echo '--- PCI tree ---'
        lspci -tv
        echo '--- NVIDIA PCI functions ---'
        lspci -Dnn | grep -i NVIDIA || true
        echo '--- NVIDIA inventory ---'
        if lsmod | grep -q '^nvidia'; then
            nvidia-smi -L 2>&1 || true
        else
            echo 'NVIDIA modules are unloaded; nvidia-smi intentionally not invoked.'
        fi
        echo '--- NVIDIA modules ---'
        lsmod | grep '^nvidia' || true
    } >"$output"
}

validate_static_environment()
{
    local kernel
    local port
    kernel=$(uname -r)
    [[ $kernel == 7.0.14-m10late4 || $kernel == 7.0.14-m10late8 ]] || \
        die "unsupported kernel $kernel; select m10late4 or m10late8"
    [[ $kernel == "$EXPECTED_KERNEL" ]] || \
        die "running kernel is $kernel; expected $EXPECTED_KERNEL"
    grep -q '\[deep\]' /sys/power/mem_sleep || die "deep S3 is not selected"
    grep -q 'hpbussize=6' /proc/cmdline || die "required PCI bus reservation is absent"
    grep -q 'hpmmiosize=128M' /proc/cmdline || die "required non-prefetchable MMIO reservation is absent"
    grep -q 'hpmmioprefsize=2G' /proc/cmdline || die "required prefetchable MMIO reservation is absent"
    for port in "${DEFAULT_LATE_PORTS[@]}"; do
        [[ -e /sys/bus/pci/devices/$port ]] || die "late port is absent: $port"
        [[ $(device_id "$port") == "$PEX8748_ID" ]] || \
            die "$port is not the expected PEX8748 downstream port"
        [[ $(device_class "$port") == 0x060400 ]] || die "$port is not a PCI bridge"
    done
}

write_state()
{
    local phase=$1
    local temporary=$STATE_FILE.tmp
    umask 077
    {
        printf 'PHASE=%q\n' "$phase"
        printf 'RUN_DIR=%q\n' "$RUN_DIR"
        printf 'RUN_ID=%q\n' "$RUN_ID"
        printf 'BOOT_ID=%q\n' "$BOOT_ID"
        printf 'START_EPOCH=%q\n' "$START_EPOCH"
        printf 'RTX_GPU_BDF=%q\n' "$RTX_GPU_BDF"
        printf 'RTX_AUDIO_BDF=%q\n' "$RTX_AUDIO_BDF"
        printf 'RTX_ROOT_BDF=%q\n' "$RTX_ROOT_BDF"
        printf 'LATE_PORTS_STRING=%q\n' "$LATE_PORTS_STRING"
    } >"$temporary"
    chmod 0600 "$temporary"
    mv -f "$temporary" "$STATE_FILE"
}

load_state()
{
    [[ -r $STATE_FILE ]] || die "resume state is missing: $STATE_FILE"
    # STATE_FILE is created root-only by write_state().
    # shellcheck disable=SC1090
    source "$STATE_FILE"
    [[ -n ${RUN_DIR:-} && -n ${BOOT_ID:-} && -n ${RTX_ROOT_BDF:-} ]] || \
        die "resume state is incomplete"
}

acquire_workflow_lock()
{
    install -d -m 0700 "$STATE_DIR"
    exec 9>"$LOCK_FILE"
    flock -n 9 || die "another late-power workflow is already running"
}

