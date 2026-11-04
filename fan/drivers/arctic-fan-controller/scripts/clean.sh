#!/bin/sh
# Remove the generated build tree and any kernel build artifacts.
set -eu

REPO_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$REPO_DIR"

if [ -d "$REPO_DIR/build" ]; then
    KDIR=${KDIR:-/lib/modules/"$(uname -r)"/build}
    make -C "$KDIR" M="$REPO_DIR/build" clean >/dev/null 2>&1 || true
fi

rm -rf "$REPO_DIR/build"
echo "clean.sh: build/ removed"
