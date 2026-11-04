#!/usr/bin/env bash

# Install and prove a dedicated lm-sensors fancontrol instance for two Tesla M10s.
# Every check is scoped to pwm1/fan1 and pwm3/fan4; pwm2 is sampled only to prove it did not move.

set -Eeuo pipefail

if (( EUID != 0 )); then
	printf 'Run this installer with sudo.\n' >&2
	exit 1
fi

readonly SOURCE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
readonly SERVICE=m10-fancontrol.service
readonly FORCE_FAILSAFE=/run/m10-fan-force-failsafe
readonly TEST_LOG=/run/m10-fancontrol-install-test.log

committed=0
direct_pid=
hwmon=
cpu_pwm_before=
cpu_enable_before=

say() {
	printf '[m10-fancontrol] %s\n' "$*"
}

locate_hwmon() {
	local -a matches
	mapfile -t matches < <(/usr/bin/grep -lFx nct6683 /sys/class/hwmon/hwmon*/name 2>/dev/null || true)
	if (( ${#matches[@]} != 1 )); then
		printf 'Expected exactly one nct6683 hwmon device; found %d\n' "${#matches[@]}" >&2
		return 1
	fi
	hwmon=${matches[0]%/name}
}

restore_known_baseline() {
	locate_hwmon || return 1
	# Restore the measured firmware state, touching only the M10 fan channels.
	for channel in 1 3; do
		printf '165\n' >"$hwmon/pwm${channel}"
		printf '0\n' >"$hwmon/pwm${channel}_enable"
	done
}

cpu_unchanged() {
	[[ $(<"$hwmon/pwm2") == "$cpu_pwm_before" && \
	   $(<"$hwmon/pwm2_enable") == "$cpu_enable_before" ]]
}

wait_for_pwm() {
	local channel=$1 expected=$2 attempts=${3:-12}
	local i
	for (( i=0; i<attempts; i++ )); do
		[[ $(<"$hwmon/pwm${channel}") == "$expected" ]] && return 0
		/usr/bin/sleep 1
	done
	return 1
}

wait_for_idle_curve() {
	wait_for_pwm 3 70 "${1:-12}" && wait_for_pwm 1 80 "${1:-12}"
}

wait_for_full_speed() {
	wait_for_pwm 3 200 "${1:-12}" && wait_for_pwm 1 200 "${1:-12}"
}

stop_direct_test() {
	if [[ -n ${direct_pid:-} ]] && /usr/bin/kill -0 "$direct_pid" 2>/dev/null; then
		/usr/bin/kill -TERM "$direct_pid" 2>/dev/null || true
		wait "$direct_pid" 2>/dev/null || true
	fi
	direct_pid=
}

on_exit() {
	local status=$?
	/usr/bin/rm -f "$FORCE_FAILSAFE"
	if (( committed == 0 )); then
			stop_direct_test
			/usr/bin/systemctl disable --now "$SERVICE" >/dev/null 2>&1 || true
			restore_known_baseline || true
			if (( status != 0 )); then
				printf '\nInstallation aborted safely. pwm1 and pwm3 were restored to 165/automatic.\n' >&2
				printf 'Diagnostic log: %s\n' "$TEST_LOG" >&2
			fi
		fi
}
trap on_exit EXIT

for required in \
	m10-fancontrol m10-max-temp m10-max-temp-group m10-max-temp-card0 m10-max-temp-card1 \
	m10-fancontrol-prepare m10-fan-full m10-fan-emergency \
	m10-fancontrol.conf m10-fancontrol.service; do
	[[ -f $SOURCE_DIR/$required ]] || {
		printf 'Missing installer input: %s/%s\n' "$SOURCE_DIR" "$required" >&2
		exit 1
	}
done

locate_hwmon
for attribute in pwm1 pwm1_enable fan1_input pwm2 pwm2_enable pwm3 pwm3_enable fan4_input; do
	[[ -e $hwmon/$attribute ]] || {
		printf 'Missing required nct6683 attribute: %s/%s\n' "$hwmon" "$attribute" >&2
		exit 1
	}
done
[[ -w $hwmon/pwm1 && -w $hwmon/pwm1_enable && \
   -w $hwmon/pwm3 && -w $hwmon/pwm3_enable ]] || {
	printf 'pwm1/pwm3 are not writable; the patched nct6683 driver is not active.\n' >&2
	exit 1
}

cpu_pwm_before=$(<"$hwmon/pwm2")
cpu_enable_before=$(<"$hwmon/pwm2_enable")
say "CPU channel guard: pwm2=$cpu_pwm_before, enable=$cpu_enable_before"
say "M10 channels before install: pwm1=$(<"$hwmon/pwm1") enable=$(<"$hwmon/pwm1_enable") fan1=$(<"$hwmon/fan1_input") RPM; pwm3=$(<"$hwmon/pwm3") enable=$(<"$hwmon/pwm3_enable") fan4=$(<"$hwmon/fan4_input") RPM"

say 'Stopping and disabling competing fan controllers'
/usr/bin/systemctl disable --now fan2go.service >/dev/null 2>&1 || true
/usr/bin/systemctl disable --now fancontrol.service >/dev/null 2>&1 || true
/usr/bin/systemctl disable --now "$SERVICE" >/dev/null 2>&1 || true
restore_known_baseline

say 'Installing the dedicated controller'
/usr/bin/install -D -o root -g root -m 0755 "$SOURCE_DIR/m10-fancontrol" /usr/local/sbin/m10-fancontrol
/usr/bin/install -D -o root -g root -m 0755 "$SOURCE_DIR/m10-max-temp" /usr/local/libexec/m10-max-temp
/usr/bin/install -D -o root -g root -m 0755 "$SOURCE_DIR/m10-max-temp-group" /usr/local/libexec/m10-max-temp-group
/usr/bin/install -D -o root -g root -m 0755 "$SOURCE_DIR/m10-max-temp-card0" /usr/local/libexec/m10-max-temp-card0
/usr/bin/install -D -o root -g root -m 0755 "$SOURCE_DIR/m10-max-temp-card1" /usr/local/libexec/m10-max-temp-card1
/usr/bin/install -D -o root -g root -m 0755 "$SOURCE_DIR/m10-fancontrol-prepare" /usr/local/libexec/m10-fancontrol-prepare
/usr/bin/install -D -o root -g root -m 0755 "$SOURCE_DIR/m10-fan-full" /usr/local/libexec/m10-fan-full
/usr/bin/install -D -o root -g root -m 0755 "$SOURCE_DIR/m10-fan-emergency" /usr/local/sbin/m10-fan-emergency
/usr/bin/install -D -o root -g root -m 0644 "$SOURCE_DIR/m10-fancontrol.conf" /etc/m10-fancontrol.conf
/usr/bin/install -D -o root -g root -m 0644 "$SOURCE_DIR/m10-fancontrol.service" /etc/systemd/system/m10-fancontrol.service
/usr/bin/systemctl daemon-reload

/usr/local/libexec/m10-fancontrol-prepare
/usr/local/sbin/m10-fancontrol --check /etc/m10-fancontrol.conf >"$TEST_LOG" 2>&1
if /usr/bin/grep -Ev '^[[:space:]]*(#|$)' /etc/m10-fancontrol.conf | \
	/usr/bin/grep -q 'pwm2'; then
	printf 'Safety check failed: pwm2 appears in the controller configuration.\n' >&2
	exit 1
fi

if [[ -e /run/fancontrol.pid ]]; then
	stale_pid=$(</run/fancontrol.pid)
	if [[ $stale_pid =~ ^[0-9]+$ ]] && /usr/bin/kill -0 "$stale_pid" 2>/dev/null; then
		printf 'Another fancontrol process is running as PID %s.\n' "$stale_pid" >&2
		exit 1
	fi
	/usr/bin/rm -f /run/fancontrol.pid
fi

say 'Running a short foreground test at the calibrated idle curve'
/usr/local/sbin/m10-fancontrol /etc/m10-fancontrol.conf >>"$TEST_LOG" 2>&1 &
direct_pid=$!
if ! wait_for_idle_curve 8; then
	printf 'Foreground test did not select the expected idle PWM values.\n' >&2
	exit 1
fi
[[ $(<"$hwmon/pwm1_enable") == 1 && $(<"$hwmon/pwm3_enable") == 1 ]] || {
	printf 'Foreground test did not put pwm1 and pwm3 into manual mode.\n' >&2
	exit 1
}
(( $(<"$hwmon/fan1_input") > 5000 )) || {
	printf 'fan1 tachometer is unexpectedly low.\n' >&2
	exit 1
}
(( $(<"$hwmon/fan4_input") > 5000 )) || {
	printf 'fan4 tachometer is unexpectedly low.\n' >&2
	exit 1
}
cpu_unchanged || {
	printf 'CPU channel guard failed during foreground test.\n' >&2
	exit 1
}
say "Foreground result: pwm1=80 fan1=$(<"$hwmon/fan1_input") RPM; pwm3=70 fan4=$(<"$hwmon/fan4_input") RPM; CPU channel unchanged"

stop_direct_test
/usr/bin/sleep 1
[[ $(<"$hwmon/pwm1") == 165 && $(<"$hwmon/pwm1_enable") == 0 && \
   $(<"$hwmon/pwm3") == 165 && $(<"$hwmon/pwm3_enable") == 0 ]] || {
	printf 'fancontrol did not restore the known M10 fan baseline after its test.\n' >&2
	exit 1
}

say 'Starting the boot-enabled systemd service'
/usr/bin/systemctl enable --now "$SERVICE" >>"$TEST_LOG" 2>&1
wait_for_idle_curve 10 || {
	printf 'The systemd service did not settle at the expected idle PWM values.\n' >&2
	exit 1
}
/usr/bin/systemctl is-active --quiet "$SERVICE" || {
	printf 'The systemd service is not active.\n' >&2
	exit 1
}
cpu_unchanged || {
	printf 'CPU channel guard failed after service startup.\n' >&2
	exit 1
}

say 'Testing NVIDIA-read failure: both M10 fan banks must rise to measured full speed (raw 200)'
: >"$FORCE_FAILSAFE"
wait_for_full_speed 12 || {
	printf 'The forced temperature-read failure did not select raw PWM 200 on both channels.\n' >&2
	exit 1
}
/usr/bin/rm -f "$FORCE_FAILSAFE"
say "Fail-safe result: pwm1=200 fan1=$(<"$hwmon/fan1_input") RPM; pwm3=200 fan4=$(<"$hwmon/fan4_input") RPM"

say 'Waiting for the three-sample average to return to the normal idle curve'
wait_for_idle_curve 25 || {
	printf 'The controller did not return to the idle curve after recovery.\n' >&2
	exit 1
}
/usr/bin/systemctl is-active --quiet "$SERVICE" || {
	printf 'The service stopped during its fail-safe test.\n' >&2
	exit 1
}
cpu_unchanged || {
	printf 'CPU channel guard failed during fail-safe testing.\n' >&2
	exit 1
}

committed=1
say "Installed successfully: card0=$(( $(/usr/local/libexec/m10-max-temp-card0) / 1000 )) C, card1=$(( $(/usr/local/libexec/m10-max-temp-card1) / 1000 )) C, pwm1=$(<"$hwmon/pwm1") fan1=$(<"$hwmon/fan1_input") RPM, pwm3=$(<"$hwmon/pwm3") fan4=$(<"$hwmon/fan4_input") RPM"
say 'm10-fancontrol.service is enabled and active; pwm2 was never written.'
