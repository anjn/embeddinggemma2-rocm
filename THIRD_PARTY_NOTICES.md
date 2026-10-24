# Third-party components

The MIT license in this repository covers its original code. It does not replace the licenses of dependencies, generated model weights, or external datasets.

| Component | Use | License/source |
|---|---|---|
| AMD ROCm / HIP | GPU runtime and compilation | [ROCm](https://github.com/ROCm/ROCm); component licenses apply |
| AMD MIGraphX | Model compiler/runtime; pinned C API header | [MIT](https://github.com/ROCm/AMDMIGraphX/blob/develop/LICENSE) |
| PyTorch / LibTorch | Native tensors, attention, text module | [BSD-style](https://github.com/pytorch/pytorch/blob/main/LICENSE), including bundled notices |
| nlohmann/json 3.11.3 | Native manifest parser | [MIT](https://github.com/nlohmann/json/blob/v3.11.3/LICENSE.MIT) |
| Hugging Face tokenizers 0.22.1 | Rust native tokenization | [Apache-2.0](https://github.com/huggingface/tokenizers/blob/v0.22.1/LICENSE) |
| llama.cpp | Optional baseline, built separately | [MIT](https://github.com/ggml-org/llama.cpp/blob/master/LICENSE) |
| Transformers / Sentence Transformers | Optional model preparation and PyTorch baseline | Apache-2.0; upstream packages retain their notices |
| NumPy, Pillow, FastAPI, Uvicorn | Python binding and HTTP adapter | Upstream licenses and notices in installed distributions |
| google/embeddinggemma-2 | Separately obtained model | Model repository terms; not licensed by this project's MIT license |

MobileCLIP2 adds ONNX Runtime (MIT), OpenCLIP (MIT), timm (Apache-2.0),
OpenSSL (Apache-2.0 for OpenSSL 3), and the Rust html-escape crate (MIT).
The separately downloaded MobileCLIP2 weights and derived tokenizer vocabulary
remain subject to their upstream terms: https://huggingface.co/timm/MobileCLIP2-S2-OpenCLIP
and https://github.com/apple-aiml-research/ml-mobileclip. No weights or tokenizer
vocabulary are included in this repository.

The tokenizer is linked statically. Its transitive dependencies are fixed in `tokenizer/Cargo.lock`; retain the corresponding crate licenses when redistributing binaries. The Docker image also includes system packages and PyTorch's third-party libraries; retain `/usr/share/doc`, package license files, and PyTorch's license/notices in binary redistributions. No upstream model, photo dataset, or compiled model is included in the Git repository.

The GPU recipes and kernels were extracted from the preceding local EmbeddingGemma 2 optimization implementation by this project's author. Their source is included here so no access to that working directory is needed.
