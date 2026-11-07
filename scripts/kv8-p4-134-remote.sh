#!/usr/bin/env bash
# Sync this worktree to gremlin-1 (tar over a SINGLE ssh session) and build
# with the pinned CUDA 13.4 / gcc 14.4 toolchain. The remote /home and the
# nix store both churn between sessions (Longhorn snapshot + channel
# refresh), so toolchain detection and the build must happen in the same
# session. Usage:
#   scripts/kv8-p4-134-remote.sh [ninja-target...]
set -euo pipefail
SRC="$(cd "$(dirname "$0")/.." && pwd)"
RSH="ssh -o ConnectTimeout=15 gremlin-1"

TARGETS="${*:-all}"

TARBALL="$(mktemp -d)"
BOOT_SH=""
cleanup() { [ -n "$TARBALL" ] && rm -rf "$TARBALL"; [ -n "$BOOT_SH" ] && rm -f "$BOOT_SH"; }
trap cleanup EXIT
tar -C "$SRC" --exclude build --exclude build-host --exclude .git \
    --exclude '*.pyc' -cf "$TARBALL/tree.tar" .
echo "tree: $(du -sh "$TARBALL/tree.tar" | cut -f1)"

# Remote bootstrap (base64 to avoid all quoting/ssh-joining hazards). The
# outer 'bash -c' captures the tar stream from ssh stdin into a file, then
# runs the bootstrap with the tar path as $1 and build targets as $@.
BOOT_SH="$(mktemp)"
cat > "$BOOT_SH" <<'EOF'
set -euo pipefail
TARBALL="$1"; shift
pick() { ls -d /nix/store/*"$1"* 2>/dev/null | grep -vE '\.drv|-info|-man' | head -1; }
CM=$(pick "cmake-4.4.3");   [ -n "$CM" ] || { echo "ERROR cmake"  >&2; exit 1; }
NJ=$(pick "ninja-1.13.2");  [ -n "$NJ" ] || { echo "ERROR ninja"  >&2; exit 1; }
GC=$(pick "gcc-wrapper-14.4.0"); [ -n "$GC" ] || { echo "ERROR gcc" >&2; exit 1; }
CU=$(pick "cuda-merged-13.4");   [ -n "$CU" ] || { echo "ERROR cuda" >&2; exit 1; }
[ -x "$GC/bin/g++" ] || { echo "ERROR gcc bin" >&2; exit 1; }
[ -x "$CU/bin/nvcc" ] || { echo "ERROR cuda bin" >&2; exit 1; }
echo "toolchain cmake=$CM ninja=$NJ gcc=$GC cuda=$CU"
export PATH="$CU/bin:$NJ/bin:$PATH"
export CC="$GC/bin/gcc" CXX="$GC/bin/g++"
export CUDA_HOST_COMPILER="$GC/bin/g++"
mkdir -p /home/celes/src
NEW=$(mktemp -d /home/celes/src/.paged-p4.XXXXXX)
tar -xf "$TARBALL" -C "$NEW"
rm -f "$TARBALL"
rm -rf /home/celes/src/paged-p4
mv "$NEW" /home/celes/src/paged-p4
cd /home/celes/src/paged-p4
rm -rf build
"$CM/bin/cmake" -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGRANITE_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_FLAGS="-I$CU/include" -DCMAKE_CXX_FLAGS="-isystem $CU/include"
"$NJ/bin/ninja" -C build "$@"
echo REMOTE_BUILD_OK
EOF
BOOTSTRAP="$(base64 < "$BOOT_SH" | tr -d '\n')"

# shellcheck disable=SC2086
$RSH "bash -c 'T=\$(mktemp /tmp/paged-p4-tree.XXXXXX.tar); cat > \"\$T\"; echo $BOOTSTRAP | base64 -d | bash -s -- \"\$T\" $TARGETS'" \
    < "$TARBALL/tree.tar"
