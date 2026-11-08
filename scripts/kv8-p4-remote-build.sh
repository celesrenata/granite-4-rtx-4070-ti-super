#!/usr/bin/env bash
# One-shot clean build + run of the three relevant test binaries on gremlin-1.
set -euo pipefail
SRC=/home/celes/src/kv8
cd "$SRC"
rm -rf build

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
  # CUDA 12.9 rejects gcc > 14, so prefer the 14.x wrapper if present.
  local out
  out=$(
    for p in $(ls -d /nix/store/*gcc-wrapper-14*/bin/g++ 2>/dev/null); do
      echo "$p"
      break
    done
  )
  if [ -z "$out" ]; then
    out=$(
      for p in $(ls -d /nix/store/*gcc-wrapper*/bin/g++ 2>/dev/null); do
        echo "$p"
        break
      done
    )
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
echo "cmake=$CM"
echo "ninja=$NJ"
echo "g++  =$CXXBIN"
echo "cuda =$CU"
echo "device: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"

export PATH="$CM/bin:$NJ/bin:$GC/bin:$CU/bin:$PATH"
export CPATH="$CU/include"
export CPLUS_INCLUDE_PATH="$CU/include"
export C_INCLUDE_PATH="$CU/include"
export LD_LIBRARY_PATH="$CU/lib64:/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDAHOSTCXX="$CXXBIN"

echo "== configure =="
cmake -B build -S . -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER="$CXXBIN" \
  -DCMAKE_CUDA_HOST_COMPILER="$CXXBIN" \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DCMAKE_CUDA_TOOLKIT_ROOT_DIR="$CU" \
  -DGRANITE_ENABLE_CUDA=ON

echo "== build test binaries =="
ninja -C build granite4-kv-pool-test granite4-kernel-math-test granite4-kernel-test

echo "== run kv-pool (host) =="
./build/engine/granite4-kv-pool-test
echo "== run kernel-math (host) =="
./build/engine/granite4-kernel-math-test
echo "== run kernels (device sm_89) =="
./build/engine/granite4-kernel-test
echo "ALL TESTS PASSED"
