#!/usr/bin/env bash
# Build the CUDA engine (sm_89) on gremlin-1 from a synced source tree, without
# touching the GPU or the running granite4-rtx service. Run ON gremlin-1:
#
#   SRC=/home/celes/src/paged-shared scripts/p4-shared-remote-build.sh
#
# SRC defaults to the directory this script lives under. The build is
# incremental (CLEAN=1 wipes it). RUN_HOST=1 also runs the host-only ctest
# suites (no GPU). The device tests (granite4-kernel-test, serve) need free
# VRAM and are NOT run here - stop granite4-rtx first and run them by hand.
set -euo pipefail
SRC="${SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$SRC"
[ "${CLEAN:-0}" = "1" ] && rm -rf build
pick() {
  local want="$1" out
  out=$(
    for p in $(ls -d /nix/store/*/bin/"$want" 2>/dev/null); do
      case "$p" in
        *python3* | *triton*) continue ;;
      esac
      echo "$p"
      break
    done
  )
  [ -n "$out" ] || { echo "ERROR: no $want" >&2; exit 1; }
  dirname "$(dirname "$out")"
}
pick_gcc() {
  # CUDA rejects too-new gcc, so prefer the 14.x wrapper if present.
  local out
  out=$(for p in $(ls -d /nix/store/*gcc-wrapper-14*/bin/g++ 2>/dev/null); do echo "$p"; break; done)
  if [ -z "$out" ]; then
    out=$(for p in $(ls -d /nix/store/*gcc-wrapper*/bin/g++ 2>/dev/null); do echo "$p"; break; done)
  fi
  [ -n "$out" ] || { echo "ERROR: no gcc-wrapper" >&2; exit 1; }
  dirname "$(dirname "$out")"
}
CM=$(pick cmake)
NJ=$(pick ninja)
GC=$(pick_gcc)
CXXBIN="$GC/bin/g++"
CU=""
for d in /nix/store/*cuda-merged-12.9 /nix/store/*cuda-merged-*; do
  if [ -e "$d/bin/nvcc" ] && [ -e "$d/include/library_types.h" ]; then
    CU="$d"
    break
  fi
done
[ -n "$CU" ] || { echo "ERROR: no merged CUDA" >&2; exit 1; }
echo "src=$SRC cmake=$CM g++=$CXXBIN cuda=$CU"
export PATH="$CM/bin:$NJ/bin:$GC/bin:$CU/bin:$PATH"
export CPATH="$CU/include" CPLUS_INCLUDE_PATH="$CU/include" C_INCLUDE_PATH="$CU/include"
export LD_LIBRARY_PATH="$CU/lib64:/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDAHOSTCXX="$CXXBIN"
cmake -B build -S . -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER="$CXXBIN" \
  -DCMAKE_CUDA_HOST_COMPILER="$CXXBIN" \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DCMAKE_CUDA_TOOLKIT_ROOT_DIR="$CU" \
  -DGRANITE_ENABLE_CUDA=ON
ninja -C build
if [ "${RUN_HOST:-0}" = "1" ]; then
  ctest --test-dir build --output-on-failure -E 'kernels_device|verify_gemv'
fi
echo "BUILD OK ($SRC/build)"
