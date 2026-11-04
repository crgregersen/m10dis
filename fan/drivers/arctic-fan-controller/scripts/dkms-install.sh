#!/bin/sh
# Prepare the DKMS source tree and register/build/install the module with DKMS.
#
# DKMS needs a self-contained source snapshot. This script generates it from
# the pristine upstream source plus the complete Quilt patch stack, so a DKMS
# rebuild (e.g. after a kernel update) never depends on a developer's
# temporary build/ directory, Git, b4, or network access.
#
# Usage: sudo ./scripts/dkms-install.sh
set -eux

REPO_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
DKMS_DIR="$REPO_DIR/dkms"

if [ "$(id -u)" -ne 0 ]; then
    echo "dkms-install.sh: must be run as root (use sudo)." >&2
    exit 1
fi

# 1. Generate the patched source snapshot into dkms/src (self-contained).
rm -rf "$DKMS_DIR/src"
mkdir -p "$DKMS_DIR/src"
cp -- "$REPO_DIR"/upstream/*.c "$DKMS_DIR/src"/
cp -- "$REPO_DIR"/Makefile "$DKMS_DIR/src"/
cp -- "$DKMS_DIR"/dkms.conf "$DKMS_DIR/src"/

# Apply the Quilt stack into the DKMS source snapshot.
# QUILT_PC must be absolute (see prepare.sh).
cat > "$DKMS_DIR/src/.quiltrc" <<EOF
QUILT_PATCHES=$REPO_DIR/patches
QUILT_PC=$DKMS_DIR/src/.pc
QUILT_PATCHES_PREFIX=yes
EOF
if grep -qve '^[[:space:]]*#' -e '^[[:space:]]*$' "$REPO_DIR/patches/series"; then
    (cd "$DKMS_DIR/src" && quilt --quiltrc "$DKMS_DIR/src/.quiltrc" push -a)
fi
rm -f "$DKMS_DIR/src/.quiltrc"
rm -rf "$DKMS_DIR/src/.pc"

# 2. Register the source with DKMS.
#    DKMS 3.x has no --source-dir option: the source snapshot must live in
#    /usr/src/<name>-<version> before "dkms add".
USR_SRC="/usr/src/arctic-fan-controller-0.1.0"
rm -rf "$USR_SRC"
cp -a "$DKMS_DIR/src" "$USR_SRC"

if dkms status | grep -q "^arctic-fan-controller/0.1.0"; then
    dkms remove arctic-fan-controller/0.1.0 --all
fi
dkms add -c "$USR_SRC/dkms.conf" -m arctic-fan-controller -v 0.1.0

# 3. Build and install for the running kernel (DKMS AUTOINSTALL handles
#    future kernels automatically).
dkms build -m arctic-fan-controller -v 0.1.0
dkms install -m arctic-fan-controller -v 0.1.0

# 4. Load the module at boot via systemd modules-load.
echo "arctic_fan_controller" > /etc/modules-load.d/arctic-fan-controller.conf

echo "dkms-install.sh: module installed and configured to load at boot."
echo "Load it now with: sudo modprobe arctic_fan_controller"
