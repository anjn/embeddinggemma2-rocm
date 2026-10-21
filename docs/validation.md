# Validation on Radeon 890M

[日本語](validation.ja.md)

The measured GPU is AMD Radeon 890M (`gfx1150`) in a Ryzen AI 9 HX 370 system, with 16 GiB physical memory (about 14 GiB visible to Linux plus 2 GiB reserved VRAM) and 8 GiB swap. CPU governor was `performance`, GPU power policy `auto`; no fixed GPU clock or power limit was applied. The host used Linux 6.17.0-1032-oem. Timings describe this machine and its shared-memory behavior.

The container uses ROCm 7.2, PyTorch 2.14.1+rocm7.2 (wheel HIP 7.2.53211), MIGraphX `2.15.0.dev+20250912-17-195-g1afd1b89c`, Transformers 5.20.0 and Sentence Transformers 6.1.0. The system HIP is 7.2.26015; native/LibTorch uses the wheel HIP through the runtime aliases, while llama.cpp HIP uses system HIP. CPU preprocessing uses four threads. Native and PyTorch run in the public Docker environment; the HIP llama.cpp build uses the same ROCm base, and Vulkan runs on the host's RADV driver (Mesa 25.2.8). Model and llama.cpp revisions are pinned in the [benchmark instructions](benchmarks.md) and preparation source.

## Release quality gate

Budget 70, request batch 128, native vision chunk 16, all 256 images and 1280 captions in the [fixed sample](tuning.md):

| R@1 (%) | Original CPU FP32 | Native candidate | Loss (percentage points) |
|---|---:|---:|---:|
| Text → image | 86.250000 | 86.328125 | −0.078125 |
| Image → text | 94.140625 | 93.359375 | 0.781250 |

Both losses satisfy the one-percentage-point gate. The [machine-readable report](../benchmarks/validation/quality-public-r3.json) identifies the exact runtime bundle and input hashes. This test covers the budget-70 specialized profile, not every prepared shape or another GPU. The minimum image cosine to the full upstream FP32 pipeline is 0.984975, while text cosine is 0.99999976. The native bilinear resampler differs from upstream bicubic antialiasing. The tuner's separate 0.995 calibration compares two profiles using the same native preprocessing; it does not assert that the full native pipeline meets 0.995 against upstream.

The BF16 PyTorch image baseline was checked on the same 256 images against fixed original FP32 text queries: R@1 was 86.015625 / 93.750000, losses 0.234375 / 0.390625 percentage points. This is an image-baseline check, distinct from the native two-modality release gate. Vision uses SDPA and text uses eager attention. Text SDPA produced incorrect outputs on this stack and is excluded.

## Native integration

A fresh public-source export of the specialized recipe reproduces the preceding optimized implementation's 128-image output bit for bit. Standalone C and C++ image executables, native C++ text inference, repeated context creation/destruction, image/text switching, multilingual tokenizer parity, and real-GPU HTTP/direct-call parity were exercised. Short text and a 610-token text match the original CPU FP32 reference within approximately 1.6e-7 maximum absolute error. Budget-70 square (1:1) and portrait (9:16) generic profiles also passed the constant-color FP32 check, both with cosine above 0.999986; this is a shape/preprocessing check, not a retrieval gate. CPU HTTP checks cover validation and resource limits; they do not substitute for these GPU checks.

## Stock llama.cpp validation limits

The adapter preserves upstream kernels and processors. HIP and Vulkan both offload the model and vision tower to the GPU, use BF16 GGUF weights and FP16 K/V, and record their actual token counts. Settings were screened with Flash Attention on and off; on was faster at the representative budget-70/batch-4 cell. This screening does not establish the fastest setting for every cell.

On distinct real photos, Vulkan batches 32 and 128 passed singleton/reversal checks (cosine ≥0.999 and reversal max absolute difference ≤0.005). HIP batch 32 passed. HIP batch 128 failed this tolerance with Flash Attention both on and off: on gave minimum reversal cosine 0.998895 and maximum difference 0.005424; off gave 0.999019 and 0.005536. HIP batch-128 speed cells are therefore excluded as `validation_failed`; larger-budget HIP batch-128 cases remain unvalidated. This is not an out-of-memory or unsupported-API claim. Numeric validation reports are in [benchmarks/validation](../benchmarks/validation/).

Stock llama.cpp bicubic resizing and padding differ from both upstream Hugging Face processing and native bilinear processing. Equal token grids and normalized output do not establish equal retrieval quality. At budget 70 and batch 4, image embeddings with fixed original FP32 text queries gave R@1 85.31250 / 94.140625 for HIP and 85.46875 / 94.53125 for Vulkan. Both meet the one-percentage-point criterion in this check; other budgets and batches were not retrieval-validated. See the corresponding [HIP](../benchmarks/validation/llama_hip-quality.json) and [Vulkan](../benchmarks/validation/llama_vulkan-quality.json) reports. GGUF precision describes stored weights; computation follows the upstream backend, including its supported matrix types.

## Tuning and adoption check

A budget-70/batch-4 search over vision chunks 1 and 4 completed both reversed timing rounds. Chunk 4 was selected at 50.69 ms/image, with minimum cosine 0.999956 against the independently compiled standard GPU reference. Its complete retrieval check returned 86.09375 / 93.75000 R@1 (losses 0.15625 / 0.390625 percentage points), then the adoption tool copied it to a new bundle and the runtime-file digest matched. This is a tooling smoke test, separate from the publication matrix; no wider tuning optimum is claimed. See [tuning report](../benchmarks/validation/tuning-smoke.json) and [quality report](../benchmarks/validation/quality-tuned.json).

## Large-request execution limit

The initial direct native budget-280/batch-128 profile terminated with exit code 139 before writing timings ([failure record](../benchmarks/validation/native-budget280-direct128-failure.json)). A short four-chunk smoke check passed, but the full repeated benchmark also terminated with code 139 ([chunked failure](../benchmarks/validation/native-budget280-chunked128-failure.json)). The exact backend cause remains unknown; neither result is relabeled as out of memory. For budget 280, this release supports at most 32 images per request and prepares execution batches no larger than 32. Larger requests return `EG2_UNSUPPORTED` (Python `UnsupportedProfile`, HTTP 503): both direct batch 128 and four chunks of 32 failed stability validation on gfx1150. Submit separate requests of at most 32 images. Budgets 70 and 140 retain request batches up to 128.

At the user’s request, remaining large PyTorch/llama.cpp measurements were omitted after interruptions. Completed two-round measurements are retained; a single completed PyTorch budget-280/batch-128 round is archived but contributes no speed value.

The final container passed all 14 CPU API checks and real-GPU readiness/image/text parity checks. Budget-280 requests of 33 images returned the expected native unsupported status and HTTP 503. A CPU-only 1919×1080 coprime-aspect export used a bounded fixture and matched the default constant-color FP32 reference and 18×33 grid. See [release checks](../benchmarks/validation/release-checks.json).

## Shared profiles and portable HIP recipes

The initial portable recipe covers Radeon 890M / gfx1150, budget 70, request batch 1, vision batch 1, 16:9. Screening evaluated 76 legal single-kernel candidates, then 32 fusion/segmentation combinations and the existing default. Seventeen pipelines compiled; unsupported segmented graphs were rejected by structural guards. The selected GELU (block 128, vector 4) and packed QK/RoPE (block 128, lanes 16) pipeline measured **35.73748 ms/image** over two rounds and reached minimum cosine **0.999925** against independent CPU FP32 with bilinear preprocessing. [Recipe](../profiles/gpus/gfx1150/budget70-b1.json), [raw rounds](../benchmarks/results/gfx1150-hip-tuned/).

The complete 256-image/1280-caption retrieval check returned text→image R@1 **86.484375%** and image→text **93.359375%**, against CPU FP32 **86.25% / 94.140625%**. Losses were **−0.234375 / 0.78125 percentage points**. [Quality report](../benchmarks/validation/quality-hip-gfx1150-b70-b1.json). A new preparation directory selected the published recipe without searching and reproduced the selected embeddings exactly. [Reuse check](../benchmarks/validation/preset-reuse.json). Other GPU architectures have candidate generation and guarded compilation support but no measured recipes in this release.

Shared image/text batches 1 and 4 retained two image profiles and four text profiles over 100 switches. Weight storage, maximum workspace, RGB capacity and LibTorch transient allocation counters remained constant after warmup; outputs matched the legacy bundle. Shared weight storage was **1,672,550,956 bytes**, workspace **120,585,472 bytes**, and transient reserved storage **138,412,032 bytes** for this profile set. These counters exclude code/graph metadata and other processes. [Memory report](../benchmarks/validation/shared-profile-memory.json).

A newly compiled standard-precision batch-1 GPU reference disagreed with independent CPU FP32. Its root cause is not established; the tuner now uses CPU FP32 directly, one image at a time, rather than treating compiler output as ground truth. [Diagnostic](../benchmarks/validation/standard-reference-diagnostic.json). This does not relax the 0.995 candidate calibration gate or the retrieval-quality gate.

The segmented budget-70/batch-128 shared path reproduced all 128 distinct input embeddings exactly. Median latency was 26.52 ms/image for the legacy bundle and 26.73 ms/image for shared storage, a 0.8% increase within the 5% regression limit. [Report](../benchmarks/validation/shared-fast128-regression.json). Reproduce with `python tests/shared_pipeline.py --legacy LEGACY --shared SHARED --output RESULT.json`; both bundles must contain the desired budget-70/batch-128 profile.
