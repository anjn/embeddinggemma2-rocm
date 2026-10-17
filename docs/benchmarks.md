# Reproducible speed comparisons

Run all backends on the same otherwise-idle GPU. Disable other compilation/tuning/inference jobs while timing. Fix driver, power settings, CPU thread counts, library revisions, model precision, and attention settings. Keep these settings with the result JSON files. Benchmark images are deterministic prescaled 512×288 RGB arrays, so no private photo data is needed for timing. The generator uses NumPy seed 812, draws 16 RGB images, and repeats those images for batches 32 and 128; every backend receives the same bytes. Use a separate real retrieval dataset for quality validation.

The CLI `eg2 benchmark` supports `native`, `torch`, and `llama`. It writes a JSON timing/status record and an NPY embedding file. The main matrix uses budgets70/140/280 and batches1/4/32/128. Raw records include actual image tokens and sequence lengths; these may differ from the requested upper budget. Native records expose internal execution chunks. A large request supported through internal chunks is still a request of that batch size.

Build the pinned llama.cpp adapter with `tools/build_llama.sh` (see `--help`/script arguments). The adapter calls libllama/libmtmd directly and includes RGB preprocessing and vision processing. Supply matching BF16 model and multimodal-projector GGUF files. Keep upstream llama.cpp unmodified and record the actual backend (HIP or Vulkan), flash-attention setting and internal batching. Screen configurations before choosing the baseline; do not call an untested backend the fastest available one.

Create a runner JSON, for example:

```json
[
  {"id":"native","backend":"native","command":["eg2"],"args":["--bundle","artifacts/my-gpu"]},
  {"id":"torch","backend":"torch","command":["eg2"],"args":["--model","models/google","--attention","sdpa"]},
  {"id":"llama","backend":"llama","command":["eg2"],"args":["--model","models/model.gguf","--mmproj","models/mmproj.gguf","--llama-library","build-llama/libeg2_llama_adapter.so"]}
]
```

```sh
python benchmarks/run_matrix.py --config runners.json --output benchmarks/results/my-gpu
python benchmarks/render.py benchmarks/results/my-gpu --output benchmarks/comparison.md
```

Each cell is measured in two reversed-order rounds with separate backend processes. Each round has >=3 warmups and >=5 timed repeats. Numeric table values are medians of all timed repeats, expressed in ms/image. Do not aggregate partial rounds or mix inputs, precision, software revisions, or backend configurations under one runner ID. The renderer preserves error states; missing cells are unmeasured rather than unsupported. Validate outputs separately before publishing throughput claims.

The optional baseline build image supplies HIP BLAS headers and Vulkan build dependencies:

```sh
docker build -f tools/Dockerfile.baselines -t embeddinggemma2-rocm:baselines .
```

Run `tools/build_llama.sh hip` or `tools/build_llama.sh vulkan` inside that image with this repository mounted writable. The default pinned llama.cpp commit is `f12aaa344c971556ec35be054d7aac1d901fd510`. The matching GGUF source is `ggml-org/embeddinggemma-2-GGUF` at revision `bfcd298762cc34d0357ece5ebdd31791a3a374d8`, files `embeddinggemma-2-BF16.gguf` and `mmproj-embeddinggemma-2-BF16.gguf`. Download these separately through Hugging Face; they are not included in Git.

The PyTorch runner defaults to BF16 parameters, SDPA vision attention, and eager text attention. On the measured ROCm stack, text SDPA failed the original-FP32 parity check even in FP32; it is not used for timings. `--torch-precision mixed` keeps FP32 parameters with BF16 autocast, and `--torch-precision fp32` selects FP32. These modes are distinct baselines, not interchangeable runs.

For the fixed 512×288 comparison input, the stock llama.cpp processor needs minimum/maximum image-token bounds of 57/70, 111/140, and 249/280. This yields 66/120/264 actual image tokens, matching the Hugging Face/native grids, and 70/124/268 total tokens. Merely setting the upper budget leaves larger-budget inputs at the smallest grid. The adapter checks actual counts and reports a mismatch as unsupported. Stock llama.cpp retains its own bicubic resize/padding behavior; matching token counts does not imply identical pixel tensors.

[The example runner file](../benchmarks/runners.example.json) includes both llama.cpp backends. Its HIP batch-128 exclusion reflects the documented gfx1150 validation failure; review it when validating a different system. `skip_batches` writes an explicit `validation_failed` record with `skip_reason` rather than inventing a speed or an unsupported-API result. The [configuration screen](../benchmarks/screening/README.md) preserves the tested attention choices and their raw rounds.

For budget 280, this release supports at most 32 images per request and prepares execution batches no larger than 32. Larger requests return `EG2_UNSUPPORTED` (Python `UnsupportedProfile`, HTTP 503): both direct batch 128 and four chunks of 32 failed stability validation on gfx1150. Submit separate requests of at most 32 images. Budgets 70 and 140 retain request batches up to 128.
