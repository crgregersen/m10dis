#!/bin/sh
# Uninstall and unregister the DKMS module.
# Usage: sudo ./scripts/dkms-remove.sh
set -eu

if [ "$(id -u)" -ne 0 ]; then
    echo "dkms-remove.sh: must be run as root (use sudo)." >&2
    exit 1
fi

dkms remove arctic-fan-controller/0.1.0 --all
rm -f /etc/modules-load.d/arctic-fan-controller.conf
echo "dkms-remove.sh: module removed. Reload any in-use driver or reboot."
