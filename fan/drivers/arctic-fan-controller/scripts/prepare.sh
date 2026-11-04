#!/bin/sh
# Generate build/ from the pristine upstream source plus the Quilt patch stack.
# The build/ directory is disposable: remove it and re-run this script to
# reproduce the same patched tree deterministically.
set -eu

REPO_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$REPO_DIR"

BUILD_DIR="$REPO_DIR/build"

# 1. Remove any previous build tree and recreate it.
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

# 2. Copy the pristine upstream source and the kbuild Makefile.
cp -- "$REPO_DIR"/upstream/*.c "$BUILD_DIR"/
cp -- "$REPO_DIR"/Makefile "$BUILD_DIR"/

# 3. Keep Quilt metadata inside build/, patches in the repository patches/ dir.
#    QUILT_PC must be absolute: with a relative path quilt's SUBDIR discovery
#    would walk up to the repository root and break path resolution.
cat > "$BUILD_DIR/.quiltrc" <<EOF
QUILT_PATCHES=$REPO_DIR/patches
QUILT_PC=$BUILD_DIR/.pc
QUILT_PATCHES_PREFIX=yes
EOF

# 4. Apply the complete Quilt stack; fail loudly if any patch does not apply.
if grep -qve '^[[:space:]]*#' -e '^[[:space:]]*$' "$REPO_DIR/patches/series"; then
    (cd "$BUILD_DIR" && quilt --quiltrc "$BUILD_DIR/.quiltrc" push -a)
else
    echo "prepare.sh: patch series is empty; nothing to apply."
fi

echo "prepare.sh: patched source tree ready in $BUILD_DIR"
