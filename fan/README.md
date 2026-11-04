# M10 fan control

Six fan banks (two on the motherboard's NCT6683, four on an ARCTIC USB controller) follow the hottest of the 24 GM107. Details: [docs/HARDWARE.md](../docs/HARDWARE.md#cooling).

| File | Role |
|---|---|
| `m10-fancontrol.conf` | fancontrol configuration: banks, curve, temperature source |
| `m10-fancontrol.service` | systemd unit; its stop hook sets all fans to full |
| `bin/m10-fancontrol` | lm-sensors `fancontrol` (GPL-2.0-or-later) with the start-up PWM ceiling lowered from 255 to 200, the measured 100 % of this board's NCT6683 |
| `bin/m10-fancontrol-prepare` | resolves the hwmon devices to stable `/run/m10-hwmon*` links before start |
| `bin/m10-max-temp`, `bin/m10-max-temp-group` | hottest GM107 (all, or one card group) from `nvidia-smi` |
| `bin/m10-fan-full`, `bin/m10-fan-emergency`, `bin/m10-fan-fixed`, `bin/m10-fan-io` | manual overrides and raw access |
| `bin/fanstatus` | readable status of all banks and GPU temperatures |
| `install-m10-fancontrol.sh` | installs the above (sbin, libexec, /etc, systemd) |
| `probe-m10-pwm-mapping.sh`, `calibrate-m10-idle.sh` | find which PWM drives which bank; idle calibration |
| `drivers/nct6687/` | patched nct6687 hwmon driver for the board's Nuvoton chip (GPL-2.0-or-later) |
| `drivers/arctic-fan-controller/` | ARCTIC USB fan controller hwmon driver, backported with DKMS scripts |

Never map the CPU fan (`pwm2` on this board) into the M10 curve.

`bin/fanstatus` maps each ARCTIC channel to the board serial of the M10 card it physically cools (`case $channel in …`). That map follows the cards through PCI re-enumeration. Edit it for your own cards (`nvidia-smi --query-gpu=serial,pci.bus_id --format=csv`).
