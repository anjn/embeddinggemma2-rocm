# MobileCLIP2-S2 native API

MobileCLIP2-S2 returns normalized float32 vectors with 512 elements. Image inference supports MIGraphX FP16 at **batch 1 or 4**, and Ryzen AI SW 1.9 BF16 at **batch 1**. Text inference uses ONNX Runtime CPU FP32, one query at a time. C/C++ and Python use the same native implementation. No Python interpreter or PyTorch is needed for inference.

These vectors require a separate search index and the matching MobileCLIP text encoder. They cannot be mixed with EmbeddingGemma's 768D vectors. The NVR evaluation found lower Japanese retrieval accuracy and a numerical difference between NPU BF16 and FP32. This API preserves the evaluated execution settings; it does not fix those model/backend limitations.

## Prepare the bundle

Preparation has three steps, each run in its appropriate Python environment. Install this repository into those environments, or set `PYTHONPATH` to its `python` directory. Use absolute bundle paths when moving between directories.

```sh
# Export environment: PyTorch, open_clip_torch 3.3.0, timm 1.0.30,
# tokenizers 0.22.1, ONNX, huggingface-hub, safetensors, ftfy.
# The optional dependency group is: pip install '.[mobileclip-prepare]'
python -m embeddinggemma2_rocm.mobileclip2_prepare export --bundle /path/to/bundle

# ROCm environment with matching MIGraphX Python bindings and shared libraries.
python -m embeddinggemma2_rocm.mobileclip2_prepare gpu --bundle /path/to/bundle

# Ryzen AI SW 1.9 environment.
source /opt/xilinx/xrt/setup.sh
source ~/work/rai/rai19-venv/bin/activate
python -m embeddinggemma2_rocm.mobileclip2_prepare npu --bundle /path/to/bundle
```

`mc2-prepare` is an equivalent installed entry point. Export downloads the pinned `timm/MobileCLIP2-S2-OpenCLIP` revision `ac6b37c8fc40b62b623d09ed228a6f0c9bc29fe6`. Compilation is offline after export. NPU compilation may take many minutes. GPU compilation disables `simplify_algebra` in a child process; caller environment variables and working directory are not changed. A bundle preparation lock prevents concurrent writes.

The bundle contains both ONNX encoders, tokenizer JSON, compiled GPU profiles, an NPU configuration/cache, and checksum manifests. Treat compiled bundles as trusted local executable assets. Prepare on the target runtime/hardware; do not reuse caches across incompatible SDK, driver or GPU/NPU changes. The native loader validates checksums and the NPU ONNX Runtime version. A missing NPU cache is an error; CPU fallback is disabled.

## Build

CMake 3.22+, a C++20 compiler, Rust/Cargo and OpenSSL development headers are required. CMake fetches the pinned nlohmann JSON header. Use the same ROCm installation that compiled the GPU bundle. For this host:

```sh
cmake -S . -B build-mc2 \
  -DCMAKE_BUILD_TYPE=Release \
  -DEG2_BUILD_NATIVE=OFF -DEG2_BUILD_MOBILECLIP2=ON \
  -DROCM_ROOT="$HOME/work/amd-video-analysis-framework/.deps/rocm72/opt/rocm-7.2.0" \
  -DMC2_ORT_ROOT="$HOME/work/rai/rai19-venv/onnxruntime" \
  -DMC2_ORT_LIBRARY="$HOME/work/rai/rai19-venv/onnxruntime/lib/libonnxruntime.so"
cmake --build build-mc2 -j4
```

`MC2_GPU=OFF` or `MC2_NPU=OFF` removes that backend. ONNX Runtime is still needed for CPU text inference. The existing EmbeddingGemma build remains enabled by default; the example disables it to build MobileCLIP alone.

For inference, expose the matching ROCm libraries and Ryzen AI provider dependencies. The Ryzen AI activation script configures the latter. A plain C/C++ executable needs the libraries/environment, but does not run Python. `cmake --install build-mc2 --prefix /path/to/install` installs the library, headers and native examples.

### Using GPU and NPU in one process

On this host, ROCr 7.2 opens `/dev/accel/accel0` while initializing HIP and keeps it open. If HIP starts first, Ryzen AI cannot acquire the NPU (`Device or resource busy`). **Create the NPU context before any GPU/HIP initialization, then create the GPU context while the NPU context remains alive.** This order is tested with both contexts performing inference. Closing a GPU context does not undo HIP initialization. For an application that has already initialized HIP, use a separate NPU worker process or start the application with the NPU-first order. The library does not close file descriptors owned by another runtime or reset global GPU state.

## Python

```python
import numpy as np
from PIL import Image
from embeddinggemma2_rocm import MobileCLIP2Embedder

with MobileCLIP2Embedder("/path/to/bundle", backend="gpu",
                        library="build-mc2/libmobileclip2.so") as model:
    model.prepare()  # optional eager warmup; exclude startup from benchmarks
    image = np.asarray(Image.open("person.jpg").convert("RGB"))
    images = model.encode_images([image])       # [1, 512]
    queries = model.encode_texts(["a person wearing a hat"])
    similarity = images @ queries.T
    print(similarity, model.capabilities)
```

Alternatively set `MC2_LIBRARY_PATH`. `encode_images` accepts individually sized RGB uint8 arrays, a single HWC array, or a BHWC batch. It centers each crop on a black square and applies Pillow-compatible bilinear resizing to 256×256, including downsampling antialiasing. Maximum width/height is 16384. Image decoding belongs to the caller.

`encode_pixels` accepts already preprocessed contiguous float32 `[batch,3,256,256]` values in `[0,1]`. `tokenize` returns int64 `[count,77]`, and `encode_tokens` embeds those tokens. Text is UTF-8, with HTML entity decoding twice, NFC normalization, fullwidth/ligature/curly-quote cleanup, whitespace cleanup and lowercasing. Long queries truncate to 77 tokens including BOS/EOS. OpenCLIP's automatic repair of malformed/mojibake text is not implemented; pass clean UTF-8. Do not add an EmbeddingGemma prompt prefix. Empty strings are valid; empty requests and embedded NULs are rejected.

Unsupported image batch sizes raise `UnsupportedProfile`; there is no implicit padding, splitting or backend fallback. Calls on a context serialize. `close()` is idempotent; inference after closing raises an error. C callers must not race destruction with inference.

## C and C++

Include `<mobileclip2/mobileclip2.h>` or `<mobileclip2/mobileclip2.hpp>` and link `mobileclip2`. `mc2_image` describes an RGB buffer with width, height and byte row stride. The caller owns input/output buffers; output capacity is measured in float elements. Inputs must remain valid until the synchronous call returns. Errors return a status and a caller-provided error string.

```cpp
mobileclip2::Embedder model("/path/to/bundle", "gpu");
mc2_image crop{rgb, width, height, row_stride};
auto image = model.images({crop});
auto text = model.texts({"a person wearing a hat"});
```

See `examples/mobileclip2.c`, `examples/mobileclip2.cpp` and `examples/mobileclip2.py`. Native executables accept `BUNDLE gpu|npu`.

## Validation

```sh
export MC2_LIBRARY_PATH="$PWD/build-mc2/libmobileclip2.so"
export MC2_TEST_BUNDLE="$PWD/artifacts/mobileclip2-api"
PYTHONPATH=python python -m pytest -q tests/test_mobileclip2.py
```

With no `MC2_TEST_BUNDLE`, hardware tests are skipped. With it set, both backends are required: tests cover text parity, preprocessing parity, GPU batch 1/4, NPU batch 1, invalid inputs, resource lifetime and simultaneous GPU/NPU contexts. `tests/mobileclip2_hardware.py` additionally compares the reviewed NVR fixtures against the previous evaluation, with 100 timed calls after at least 30 warmups and two seconds. It reports inference-only latency (including host transfers), preprocessing-inclusive latency, and retrieval accuracy separately.

### Measured API results (Ryzen AI 9 HX 370 / Radeon 890M)

100 calls after warmup, performance power mode, using the 52 reviewed NVR person crops:

| Backend | Batch | Inference ms/batch | Inference ms/image | Images/s | Including preprocessing ms/batch |
|---|---:|---:|---:|---:|---:|
| MIGraphX FP16 | 1 | 6.86 | 6.86 | 146 | 8.00 |
| MIGraphX FP16 | 4 | 23.05 | 5.76 | 174 | 27.00 |
| Ryzen AI BF16 | 1 | 10.90 | 10.90 | 92 | 11.96 |

Inference measurements use `encode_pixels`, including argument validation, host transfers and final normalization. Preprocessing-inclusive measurements use `encode_images`; image decoding is excluded. Startup, compilation and warmup are excluded from both. These numbers are host-specific, not performance guarantees.

Across all 52 crops, cosine similarity to the corresponding previous backend output was at least 0.9999998. Native preprocessing matched the evaluated Pillow inputs (embedding cosine 1.0). Text output matched the previous FP32 reference. Retrieval hit@1 remained 81.8% English / 27.3% Japanese on GPU and 72.7% / 27.3% on NPU. The NPU's mean cosine to FP32 remains 0.84946; it is a backend precision limitation reproduced by this API, not an accuracy fix. This small fixture set is a regression check, not a general person re-identification benchmark.
