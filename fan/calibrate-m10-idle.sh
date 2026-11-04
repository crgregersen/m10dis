#!/usr/bin/env bash

# Safely characterize lower pwm3 values and adopt raw 70 (35%) only if the
# M10 tachometer, temperatures, service, and CPU-channel guard all pass.

set -Eeuo pipefail

if (( EUID != 0 )); then
	printf 'Run this calibration with sudo.\n' >&2
	exit 1
fi

readonly SOURCE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
readonly SERVICE=m10-fancontrol.service
readonly CONFIG=/etc/m10-fancontrol.conf
readonly CANDIDATE=$SOURCE_DIR/m10-fancontrol-idle70.conf
readonly BACKUP=$(mktemp /run/m10-fancontrol.conf.XXXXXX)

committed=0
hwmon=
cpu_pwm_before=
cpu_enable_before=

say() {
	printf '[M10 calibration] %s\n' "$*"
}

locate_hwmon() {
	local -a matches
	mapfile -t matches < <(/usr/bin/grep -lFx nct6683 /sys/class/hwmon/hwmon*/name 2>/dev/null || true)
	if (( ${#matches[@]} != 1 )); then
		printf 'Expected one nct6683 hwmon device; found %d\n' "${#matches[@]}" >&2
		return 1
	fi
	hwmon=${matches[0]%/name}
}

cpu_unchanged() {
	[[ $(<"$hwmon/pwm2") == "$cpu_pwm_before" && \
	   $(<"$hwmon/pwm2_enable") == "$cpu_enable_before" ]]
}

max_gpu_temp() {
	local millic
	millic=$(/usr/local/libexec/m10-max-temp)
	(( millic >= 0 && millic <= 110000 )) || return 1
	printf '%d\n' "$(( millic / 1000 ))"
}

wait_for_pwm() {
	local expected=$1
	local i
	for (( i=0; i<12; i++ )); do
		[[ $(<"$hwmon/pwm3") == "$expected" ]] && return 0
		/usr/bin/sleep 1
	done
	return 1
}

on_exit() {
	local status=$?
	if (( committed == 0 )); then
		printf '\nCalibration did not complete; restoring the previous configuration.\n' >&2
		/usr/bin/install -o root -g root -m 0644 "$BACKUP" "$CONFIG" || true
		/usr/bin/systemctl start "$SERVICE" >/dev/null 2>&1 || true
	fi
	/usr/bin/rm -f "$BACKUP"
	exit "$status"
}
trap on_exit EXIT

[[ -r $CONFIG && -r $CANDIDATE ]] || {
	printf 'The active or candidate configuration is missing.\n' >&2
	exit 1
}
/usr/bin/cp --preserve=mode,ownership,timestamps "$CONFIG" "$BACKUP"

locate_hwmon
for attribute in pwm2 pwm2_enable pwm3 pwm3_enable fan4_input; do
	[[ -e $hwmon/$attribute ]] || {
		printf 'Missing required attribute: %s/%s\n' "$hwmon" "$attribute" >&2
		exit 1
	}
done

/usr/bin/systemctl is-active --quiet "$SERVICE" || {
	printf '%s must be active before calibration.\n' "$SERVICE" >&2
	exit 1
}

cpu_pwm_before=$(<"$hwmon/pwm2")
cpu_enable_before=$(<"$hwmon/pwm2_enable")
say "CPU guard recorded: pwm2=$cpu_pwm_before, enable=$cpu_enable_before"
say "Starting point: pwm3=$(<"$hwmon/pwm3"), fan4=$(<"$hwmon/fan4_input") RPM, hottest GPU=$(max_gpu_temp) C"

# Stopping the production unit invokes its full-speed fail-safe. Immediately
# take manual control of pwm3 at the already-proven raw value 120.
/usr/bin/systemctl stop "$SERVICE"
printf '1\n' >"$hwmon/pwm3_enable"
printf '120\n' >"$hwmon/pwm3"

for pwm in 110 100 90 80 70; do
	printf '%s\n' "$pwm" >"$hwmon/pwm3"
	/usr/bin/sleep 3

	min_rpm=999999
	max_temp=0
	for sample in 1 2 3; do
		rpm=$(<"$hwmon/fan4_input")
		temp=$(max_gpu_temp) || {
			printf 'Could not validate all four M10 temperatures.\n' >&2
			exit 1
		}
		(( rpm < min_rpm )) && min_rpm=$rpm
		(( temp > max_temp )) && max_temp=$temp
		cpu_unchanged || {
			printf 'CPU channel changed during calibration; aborting.\n' >&2
			exit 1
		}
		/usr/bin/sleep 2
	done

	say "raw $pwm: fan4 minimum ${min_rpm} RPM, hottest GPU ${max_temp} C"
	(( min_rpm >= 3500 )) || {
		printf 'fan4 dropped below the 3500 RPM safety floor.\n' >&2
		exit 1
	}
	(( max_temp <= 40 )) || {
		printf 'A GPU exceeded the 40 C calibration ceiling.\n' >&2
		exit 1
	}
done

say 'All lower-speed measurements passed; installing the 35% idle floor'
/usr/bin/install -o root -g root -m 0644 "$CANDIDATE" "$CONFIG"
/usr/bin/systemctl start "$SERVICE"
wait_for_pwm 70 || {
	printf 'The service did not settle at raw PWM 70.\n' >&2
	exit 1
}
/usr/bin/systemctl is-active --quiet "$SERVICE" || {
	printf 'The controller did not remain active.\n' >&2
	exit 1
}
cpu_unchanged || {
	printf 'CPU channel guard failed after service restart.\n' >&2
	exit 1
}

committed=1
say "Complete: pwm3=70, fan4=$(<"$hwmon/fan4_input") RPM, hottest GPU=$(max_gpu_temp) C"
say 'The CPU channel remained unchanged.'
