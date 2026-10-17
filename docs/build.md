# Source builds and offline operation

Use the versions installed by the Dockerfile for the tested environment: ROCm 7.2 and ROCm PyTorch 2.14.1. MIGraphX and LibTorch are native runtime dependencies. Mixing system HIP with the wheel's HIP can cause shutdown crashes; CMake creates a runtime alias directory so both HIP library names resolve to the same library. Keep the installed directory layout intact.

A source build requires CMake >=3.22, C++20, HIP, MIGraphX, Rust/Cargo, and a ROCm LibTorch installation with matching C++ ABI. CMake downloads pinned MIGraphX and JSON headers into the build directory. Cargo uses the checked-in lockfile.

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install 'torch==2.14.1+rocm7.2' 'torchvision==0.29.1+rocm7.2' \
  --index-url https://download.pytorch.org/whl/rocm7.2 --extra-index-url https://pypi.org/simple
pip install '.[prepare,server,test]'
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DTORCH_ROOT="$VIRTUAL_ENV/lib/python3.12/site-packages/torch" \
  -DEG2_GPU_ARCH=gfx1150 -DCMAKE_INSTALL_PREFIX="$PWD/install"
cmake --build build -j2
cmake --install build
export EG2_LIBRARY_PATH="$PWD/install/lib/libeg2.so"
# gfx1150 native attention needs the experimental AOTriton architecture path.
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
```

Set `EG2_CXX11_ABI` to match the selected LibTorch. `EG2_TOKENIZER_LIBRARY` can point to a separately built Rust static archive. Header downloads can be prepopulated under `build/deps` for an offline build. Build for the target architecture, then prepare profiles on that GPU.

`eg2 fetch --output models/google` downloads the pinned source model revision. Authenticate with Hugging Face through its usual credential mechanism if required. `eg2 prepare` and `eg2 prepare-text` accept a local model directory. Once the dependencies, model, and bundle are available, native inference does not require network access.

Model loading and compilation are intentionally outside request handling. Bundles include exported/compiled model weights. Keep `models/`, `artifacts/`, and build caches outside version control. Recompile after changing GPU architecture or ROCm/MIGraphX/LibTorch versions. Inspect `eg2 info --bundle artifacts/my-gpu` for available profiles and execution metadata.

A native smoke test is `install/bin/eg2-example artifacts/my-gpu 1`. It runs directly as a C++ executable. `install/bin/eg2-text-example artifacts/my-gpu` checks native tokenization and text inference without starting Python. The Python package is not required by that executable.

Some gfx1150 shapes lack an FP32 MLIR attention kernel. If MIGraphX reports `No solutions`, preparation retries with standard GPU operators in a fresh process. This fallback still runs on the GPU; the fresh process is necessary because MIGraphX caches compiler environment settings.
