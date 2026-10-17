#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
 cat <<'EOF'
Usage: tools/build_llama.sh [hip|vulkan] [source-directory] [build-directory]
Defaults: hip, build/llama-source, build/llama-<backend>.
GPU_ARCH selects the HIP architecture (default gfx1150); BUILD_JOBS defaults to 2.
The direct-call adapter is built in <build-directory>-adapter.
EOF
 exit 0
fi
backend=${1:-hip}
source_dir=${2:-build/llama-source}
output=${3:-build/llama-${backend}}
arch=${GPU_ARCH:-gfx1150}
case "$backend" in hip) flags=(-DGGML_HIP=ON -DAMDGPU_TARGETS="$arch");; vulkan) flags=(-DGGML_VULKAN=ON);; *) echo 'backend must be hip or vulkan' >&2; exit 2;; esac
revision=f12aaa344c971556ec35be054d7aac1d901fd510
if [[ ! -d "$source_dir/.git" ]]; then
 mkdir -p "$source_dir"
 git -C "$source_dir" init
 git -C "$source_dir" remote add origin https://github.com/ggml-org/llama.cpp.git
 git -C "$source_dir" fetch --depth 1 origin "$revision"
 git -C "$source_dir" checkout --detach FETCH_HEAD
fi
[[ "$(git -C "$source_dir" rev-parse HEAD)" == "$revision" ]] || { echo 'llama.cpp revision mismatch' >&2; exit 1; }
cmake -S "$source_dir" -B "$output" -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF "${flags[@]}"
cmake --build "$output" --target mtmd -j "${BUILD_JOBS:-2}"
repo_root=$(cd "$(dirname "$0")/.." && pwd)
cmake -S "$repo_root/tools/llama_adapter" -B "${output}-adapter" -DLLAMA_ROOT="$(cd "$source_dir" && pwd)" -DLLAMA_BUILD="$(cd "$output" && pwd)"
cmake --build "${output}-adapter" -j "${BUILD_JOBS:-2}"
