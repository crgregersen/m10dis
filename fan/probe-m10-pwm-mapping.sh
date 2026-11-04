#!/usr/bin/env bash

set -Eeuo pipefail

readonly LOG=/tmp/m10-pwm-mapping.log
readonly STEP=10
declare -a original_pwm original_enable

if (( EUID != 0 )); then
	printf 'Run this probe as root: sudo %s\n' "$0" >&2
	exit 1
fi

mapfile -t hwmon_names < <(grep -l '^nct6683$' /sys/class/hwmon/hwmon*/name)
if (( ${#hwmon_names[@]} != 1 )); then
	printf 'Expected exactly one nct6683 hwmon device; found %d.\n' \
		"${#hwmon_names[@]}" >&2
	exit 1
fi
readonly H=${hwmon_names[0]%/name}

for channel in 1 2 3; do
	for attribute in pwm pwm_enable; do
		path="$H/${attribute/pwm/pwm${channel}}"
		if [[ ! -r $path || ! -w $path ]]; then
			printf 'Required attribute is not readable and writable: %s\n' "$path" >&2
			exit 1
		fi
	done
	original_pwm[channel]=$(<"$H/pwm${channel}")
	original_enable[channel]=$(<"$H/pwm${channel}_enable")
	if (( original_pwm[channel] < STEP )); then
		printf 'Refusing to reduce pwm%d=%d by %d.\n' \
			"$channel" "${original_pwm[channel]}" "$STEP" >&2
		exit 1
	fi
done

restore_all() {
	local channel
	trap - EXIT INT TERM HUP
	set +e
	for channel in 1 2 3; do
		[[ -v original_pwm[channel] ]] || continue
		printf '%d\n' "${original_pwm[channel]}" >"$H/pwm${channel}"
		printf '%d\n' "${original_enable[channel]}" >"$H/pwm${channel}_enable"
	done
}
trap restore_all EXIT INT TERM HUP

exec > >(tee "$LOG") 2>&1

sample() {
	local label=$1 fan2 fan4 temps max_temp
	read -r fan2 <"$H/fan2_input"
	read -r fan4 <"$H/fan4_input"
	if temps=$(nvidia-smi --query-gpu=temperature.gpu \
			--format=csv,noheader,nounits 2>/dev/null) &&
		max_temp=$(printf '%s\n' "$temps" | sort -nr | head -n1); then
		:
	else
		max_temp=unavailable
	fi
	printf '%(%FT%TZ)T %-19s fan2=%5s RPM fan4=%5s RPM max_gpu=%s C\n' \
		-1 "$label" "$fan2" "$fan4" "$max_temp"
}

printf 'hwmon=%s\n' "$H"
printf 'original: pwm1=%s enable=%s; pwm2=%s enable=%s; pwm3=%s enable=%s\n' \
	"${original_pwm[1]}" "${original_enable[1]}" \
	"${original_pwm[2]}" "${original_enable[2]}" \
	"${original_pwm[3]}" "${original_enable[3]}"
sample baseline

for channel in 1 2 3; do
	test_pwm=$(( original_pwm[channel] - STEP ))
	printf '\nTesting pwm%d only: %d -> %d -> %d\n' \
		"$channel" "${original_pwm[channel]}" "$test_pwm" \
		"${original_pwm[channel]}"

	# Enter manual mode at the already-active value before making the small change.
	printf '%d\n' "${original_pwm[channel]}" >"$H/pwm${channel}"
	printf '1\n' >"$H/pwm${channel}_enable"
	sleep 2
	sample "pwm${channel} manual/base"

	printf '%d\n' "$test_pwm" >"$H/pwm${channel}"
	for count in 1 2 3 4 5; do
		sleep 1
		sample "pwm${channel} test ${count}/5"
	done

	printf '%d\n' "${original_pwm[channel]}" >"$H/pwm${channel}"
	for count in 1 2 3; do
		sleep 1
		sample "pwm${channel} restore ${count}/3"
	done
	printf '%d\n' "${original_enable[channel]}" >"$H/pwm${channel}_enable"
	sleep 2
	sample "pwm${channel} firmware"
done

restore_all
trap - EXIT INT TERM HUP
printf '\nAll PWM values and enable modes restored. Log: %s\n' "$LOG"
