#!/usr/bin/env bash
# Remote-side build+test driver, run on gremlin-1 (see kv8-gremlin-build.sh).
# Builds granite4-kernel-math-test (host, no CUDA) and granite4-kernel-test
# (full sm_89 device suite) with the Nix-store toolchain, then runs both.
#   $1 = source dir (default: the script's parent)
#
# Toolchain pinning: cmake / ninja / clang / cuda-merged-12.9 live in the
# Nix store on gremlin-1. We locate them by name, preferring the
# non-pyproject / non-triton variants (those have a working driver + libs).

set -euo pipefail
SRC="${1:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$SRC"

# First store path matching NAME/bin/TOOL that is NOT a pyproject or triton
# wrapper (those lack a real compiler driver / libs).
pick() {
  local want="$1" out
  out=$(
    for p in $(ls -d /nix/store/*/bin/"$want" 2>/dev/null); do
      case "$p" in
        *python3*|*triton*) continue ;;
      esac
      echo "$p"
    done | head -1
  )
  [ -n "$out" ] || { echo "ERROR: no $want in Nix store" >&2; exit 1; }
  dirname "$(dirname "$out")"
}

pick_gcc() {
  # The gcc-wrapper (self-contained sysroot) is what `nix build` would use.
  # The bare gcc-* store path needs the paired glibc-dev on the linker path.
  # CUDA 12.9 rejects gcc > 14, so prefer the 14.x wrapper if present.
  local out
  out=$(
    for p in $(ls -d /nix/store/*gcc-wrapper-14*/bin/"g++" 2>/dev/null); do
      echo "$p"
    done | head -1
  )
  if [ -z "$out" ]; then
    out=$(
      for p in $(ls -d /nix/store/*gcc-wrapper*/bin/"g++" 2>/dev/null); do
        echo "$p"
      done | head -1
    )
  fi
  [ -n "$out" ] || { echo "ERROR: no gcc-wrapper in Nix store" >&2; exit 1; }
  dirname "$(dirname "$out")"
}

CM=$(pick cmake)
NJ=$(pick ninja)
GC=$(pick_gcc)
CXXBIN="$GC/bin/g++"

# CUDA 12.9 (matches the flake's cudatoolkit); fall back to any merged toolkit
# that ships nvcc + headers.
CU=""
for d in /nix/store/*cuda-merged-12.9 /nix/store/*cuda-merged-*; do
  if [ -e "$d/bin/nvcc" ] && [ -e "$d/include/cuda_runtime.h" ]; then CU="$d"; break; fi
done
[ -n "$CU" ] || { echo "ERROR: no merged CUDA toolkit in the Nix store" >&2; exit 1; }

echo "[remote] cmake=$CM"
echo "[remote] ninja=$NJ"
echo "[remote] g++  =$CXXBIN"
echo "[remote] cuda =$CU"
echo "[remote] device: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"

export PATH="$CM/bin:$NJ/bin:$GC/bin:$CU/bin:$PATH"
# Nix's cudatoolkit build hook normally exports these; set them by hand so
# nvcc's host-compiler subprocess finds the CUDA headers.
export CPATH="$CU/include"
export CPLUS_INCLUDE_PATH="$CU/include"
export C_INCLUDE_PATH="$CU/include"
export LD_LIBRARY_PATH="$CU/lib64:/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# Tell nvcc the host compiler (so -x cuda picks the right C++ compiler for the
# host portion of .cu files) and where the CUDA headers live.
export CUDAHOSTCXX="$CXXBIN"

rm -rf build
cmake -B build -S . -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER="$CXXBIN" \
  -DCMAKE_CXX_STANDARD=20 \
  -DCMAKE_CUDA_STANDARD=20 \
  -DCUDAToolkit_ROOT="$CU" \
  -DGRANITE_ENABLE_CUDA=ON

cmake --build build --target granite4-kernel-math-test -j4
cmake --build build --target granite4-kernel-test -j4

echo '=== host math test ==='
./build/engine/granite4-kernel-math-test

echo '=== GPU device kernel test ==='
./build/engine/granite4-kernel-test
