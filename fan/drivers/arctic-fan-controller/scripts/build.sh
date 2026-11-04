#!/bin/sh
# Build the Quilt-patched module in build/ against the running kernel
# (or $KDIR if set). Simple wrapper for users who do not want to invoke
# the top-level Makefile directly.
set -eu

REPO_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$REPO_DIR"

./scripts/prepare.sh

KDIR=${KDIR:-/lib/modules/"$(uname -r)"/build}
make -C "$KDIR" M="$REPO_DIR/build" modules

echo "build.sh: module built in $REPO_DIR/build"
