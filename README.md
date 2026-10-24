# EmbeddingGemma 2 on AMD GPUs

[日本語](README.ja.md)

Image and text embeddings for `google/embeddinggemma-2`, with a native C ABI, C++ RAII interface, Python binding, HTTP server, and GPU-specific tuning tools. Outputs are 768-dimensional, L2-normalized FP32 vectors.

Image inference uses HIP preprocessing, MIGraphX programs, and LibTorch attention. Text inference uses a prepared LibTorch module and a native Rust tokenizer. The C/C++ runtime does not start a Python interpreter. Python is required for model preparation and the optional HTTP service.

The specialized image recipe targets Radeon 890M (`gfx1150`), budget 70, 16:9 images and vision chunks of 16. Other shapes use the generic compiled path. Other AMD GPUs require their own compilation and validation; they have not been tested here. Models and compiled bundles are downloaded/generated separately.

Multiple image and text profiles can share GPU weights and a maximum-size workspace with `eg2 share`. The HIP tuner searches per-profile kernel settings and can publish validated GPU recipes under `profiles/gpus/`; compatible machines reuse settings without a search. See [tuning and shared profiles](docs/tuning.md).

## MobileCLIP2-S2

An additional native API provides 512D image/text embeddings: MIGraphX FP16 image inference supports batch 1 and 4; Ryzen AI SW 1.9 BF16 supports batch 1. Text runs on CPU FP32. Use `MobileCLIP2Embedder` in Python or `libmobileclip2` from C/C++. See [preparation, build and API examples](docs/mobileclip2.md). The embeddings use a separate index from EmbeddingGemma; existing APIs are unchanged.

## Build and prepare

Linux, a supported AMD GPU, Docker, and a working ROCm host driver are required for the container workflow. Obtain access to the model and accept its upstream terms before downloading it. Set `HF_TOKEN` in the shell if authentication is required.

```sh
docker build --build-arg GPU_ARCH=gfx1150 -t embeddinggemma2-rocm:local .
mkdir -p models artifacts
docker run --rm -e HF_TOKEN -v "$PWD/models:/data/models" \
  embeddinggemma2-rocm:local fetch --output /data/models/google
```

Prepare the profiles needed by the application. Preparation can take substantial time and GPU memory; start with a small batch. Compiled bundles contain model weights and are specific to the GPU/software environment.

```sh
docker run --rm --device=/dev/kfd --device=/dev/dri \
  --group-add "$(stat -c %g /dev/kfd)" \
  -v "$PWD/models:/data/models:ro" -v "$PWD/artifacts:/data/artifacts" \
  embeddinggemma2-rocm:local prepare --model models/google \
  --output artifacts/my-gpu --budgets 70 --batches 1 4

docker run --rm --device=/dev/kfd --device=/dev/dri \
  --group-add "$(stat -c %g /dev/kfd)" \
  -v "$PWD/models:/data/models:ro" -v "$PWD/artifacts:/data/artifacts" \
  embeddinggemma2-rocm:local prepare-text --model models/google \
  --output artifacts/my-gpu --batches 1 4 --lengths 32 128

docker run --rm -v "$PWD/artifacts:/data/artifacts" \
  embeddinggemma2-rocm:local share --bundle artifacts/my-gpu \
  --output artifacts/my-gpu-shared
```

For budget 70, batch 32/128 on gfx1150, the preparation tool selects vision chunks of 16 and attempts the guarded specialized recipe. Inspect preparation output: failed optimization guards retain the compiled fallback. Prepare other aspect ratios with `--aspect WIDTH HEIGHT`; an unprepared ratio returns an unsupported-profile error. There is no implicit crop or distortion to 16:9. Once prepared, serving can run offline with only the bundle mounted.

See [source builds](docs/build.md), [interfaces](docs/api.md), and [tuning and validation](docs/tuning.md).

## Python and native embedding

```python
import numpy as np
from embeddinggemma2_rocm import Embedder

with Embedder("artifacts/my-gpu-shared") as model:
    rgb = np.zeros((4, 288, 512, 3), dtype=np.uint8)
    image_vectors = model.encode_images(rgb, token_budget=70)
    query_vectors = model.encode_texts(["a cat", "a street"], prompt="query")
    print(model.capabilities)
```

Set `EG2_LIBRARY_PATH` to the built or installed `libeg2.so`. All images in one call must have the same dimensions. A request can be split across smaller prepared batches. For budget 280, this release supports at most 32 images per request and prepares execution batches no larger than 32. Larger requests return `EG2_UNSUPPORTED` (Python `UnsupportedProfile`, HTTP 503): both direct batch 128 and four chunks of 32 failed stability validation on gfx1150. Submit separate requests of at most 32 images. Budgets 70 and 140 retain request batches up to 128. A context serializes execution; use independent workers for independent GPUs.

```cpp
#include <eg2/eg2.hpp>
eg2::Context model("artifacts/my-gpu-shared");
auto vectors = model.texts({"a cat", "a street"});
```

The [C header](include/eg2/eg2.h) documents buffer ownership and errors; [examples](examples/) show native programs.

## HTTP server

```sh
docker run --rm --device=/dev/kfd --device=/dev/dri \
  --group-add "$(stat -c %g /dev/kfd)" -p 127.0.0.1:8000:8000 \
  -v "$PWD/artifacts:/data/artifacts:ro" embeddinggemma2-rocm:local \
  serve --bundle artifacts/my-gpu-shared --host 0.0.0.0

curl http://127.0.0.1:8000/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"embeddinggemma-2","input":["a cat"],"input_type":"query"}'
```

`/v1/embeddings` accepts text or structured image inputs. `/health`, `/ready`, and `/v1/models` expose status and capabilities. Image input is an extension of the OpenAI text embeddings protocol; see [API details](docs/api.md). Deploy behind authentication when exposing it beyond a trusted network.

## Speed comparison

The publication matrix uses deterministic synthetic, prescaled **512×288 RGB (16:9)** inputs, token budgets **70/140/280**, and request batches **1/4/32/128**, where supported. A budget is an upper bound; the report records actual image tokens, sequence length, internal vision chunks, precision, and backend settings.

Timing starts with CPU RGB arrays and ends with normalized FP32 embeddings on the CPU. It includes patch preprocessing, host/device transfers, and internal chunking. File decoding, upstream prescaling, model loading, compilation, and warmup are excluded. Native preprocessing uses fused half-pixel bilinear resizing and RGB scaling by 1/255; the PyTorch baseline uses the upstream bicubic antialiased processor. This preprocessing approximation is part of the native performance/quality tradeoff. These are direct library calls, without HTTP overhead. PyTorch and llama.cpp use the same inputs. Each backend runs in a separate process, with at least 3 warmups and 5 repeats in each of two reversed-order rounds.

A separate HIP search now publishes a validated [gfx1150 budget-70/batch-1 recipe](profiles/gpus/gfx1150/budget70-b1.json) at **35.74 ms/image**. The publication matrix below remains the original measurement set; [new raw rounds](benchmarks/results/gfx1150-hip-tuned/) and [validation](docs/validation.md#shared-profiles-and-portable-hip-recipes) document the additional result.

<!-- benchmark-table:start -->

| Token budget | Batch | Native | PyTorch BF16 | llama.cpp HIP | llama.cpp Vulkan |
|---:|---:|---:|---:|---:|---:|
| 70 | 1 | 57.64 | 100.74 | 123.68 | 179.30 |
| 70 | 4 | 50.67 | 83.73 | 111.31 | 138.23 |
| 70 | 32 | 27.23 | 75.30 | 104.51 | 125.78 |
| 70 | 128 | 26.71 | 74.62 | excluded | 132.21 |
| 140 | 1 | 102.50 | 189.69 | 234.46 | 296.23 |
| 140 | 4 | 92.92 | 160.55 | 204.04 | 242.52 |
| 140 | 32 | 82.43 | 151.71 | 206.86 | 251.55 |
| 140 | 128 | 82.48 | 152.30 | excluded | 257.52 |
| 280 | 1 | 241.38 | 370.66 | 576.94 | 636.43 |
| 280 | 4 | 205.52 | 347.39 | 550.68 | 631.52 |
| 280 | 32 | 202.44 | 338.81 | 603.91 | 619.61 |
| 280 | 128 | failed (139) | omitted | excluded | omitted |

Numeric cells are median **ms/image** across two reversed-order rounds. `excluded`: failed numerical validation; `failed (139)`: native stability failure; `omitted`: remaining measurement skipped, or only one round completed. Completed large-batch baselines are retained; no further PyTorch/llama.cpp batch 32/128 runs are required.

At budget 70 / batch 128, native reaches **26.71 ms/image**, **2.79×** PyTorch BF16 and **4.95×** llama.cpp Vulkan throughput on this GPU.

<!-- benchmark-table:end -->

Measured on Radeon 890M (`gfx1150`), ROCm 7.2 and PyTorch 2.14.1+rocm7.2. PyTorch uses BF16 parameters, vision SDPA and eager text attention. Stock llama.cpp uses BF16 GGUF weights, FP16 K/V and Flash Attention on, with HIP and Vulkan shown separately. Actual image tokens are **66/120/264**, with total sequence lengths **70/124/268**. HIP batch 128 is excluded as `validation_failed` after real-photo reversal checks; larger-budget HIP batch 128 remains unvalidated. See [raw rounds](benchmarks/results/gfx1150/), [environment](benchmarks/environment.json), [configuration screening](benchmarks/screening/README.md), and [reproduction instructions](docs/benchmarks.md).

The native budget-70/batch-128 profile passed the fixed **256-image / 1280-caption** retrieval gate: text→image R@1 **86.328125%**, image→text **93.359375%**, against original CPU FP32 **86.25% / 94.140625%**. The losses are −0.078125 and 0.78125 percentage points, within the one-point criterion. Other shapes and GPUs require their own retrieval checks. [Validation details](docs/validation.md) include baseline quality, native integration, generic aspects, tuning and adoption.

## License

This repository's original code is [MIT](LICENSE). Model weights have their own terms. Dependencies retain their licenses; see [third-party notices](THIRD_PARTY_NOTICES.md). Do not redistribute compiled bundles without checking the model terms.
