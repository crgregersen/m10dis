# Frankenstein installation notes

Installed: 2026-10-06

- Host kernel: `7.0.0-31-generic`
- Controller: ARCTIC Fan Controller, USB HID `3904:f001`
- Backport revision: `8705c3c6190b0a4b227bea79c89d4a0f43cebcee`
- DKMS package: `arctic-fan-controller/0.1.0`
- Installed module: `/lib/modules/7.0.0-31-generic/updates/dkms/arctic_fan_controller.ko.zst`
- Module autoload: `/etc/modules-load.d/arctic-fan-controller.conf`

The repository's `0001-kernel-api-compat.patch` was removed from
`patches/series`. It changes `<linux/unaligned.h>` to
`<asm/unaligned.h>`, but kernel 7.0 exports the required API from
`<linux/unaligned.h>` and has no matching architecture header.

The remaining patches initialize all cached channels to the controller's 40%
power-on default and apply the upstream dual license.

## Initial test

No fans were connected. Loading the module rebound the device from
`hid-generic` to `arctic_fan` and created an `arctic_fan` hwmon device with ten
`fanN_input` and ten `pwmN` attributes.

- All ten fan inputs reported `0 RPM`.
- All ten PWM caches initially reported raw value `102` (40% of 255).
- Writing raw value `128` to `pwm1` completed with a device ACK.
- Channels 2 through 10 remained at `102`.
- `pwm1` was restored to `102`.
- No related kernel error was logged.

Locate the hwmon directory dynamically because its number can change:

```bash
for d in /sys/class/hwmon/hwmon*; do
    [ "$(cat "$d/name" 2>/dev/null)" = arctic_fan ] && echo "$d"
done
```

Use the raw sysfs PWM values (`0..255`) for control. The installed
`lm-sensors` formatter displays these values as misleading percentages and
labels channel 10 incorrectly, while direct sysfs reads expose all channels
correctly.

