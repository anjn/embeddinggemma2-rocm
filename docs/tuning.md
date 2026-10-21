# GPU tuning and correctness

`eg2 tune --model models/google --output artifacts/tuning --budget 70 --batch 4 --vision-batches 1 4` screens custom HIP kernels and measures complete image pipelines on a ROCm-supported AMD GPU. GEMM and Attention use existing library implementations.

For each legal vision batch, candidates cover blocks of 64/128/256/512 threads, GELU and preprocessing vector widths of 1/2/4/8/16, and QK/RoPE and paired-normalization lane groups of 4/8/16/32/64. Device limits, wave size and divisibility remove invalid candidates; packed QK uses 16 lanes. Each kernel must pass a numerical comparison before timing. The existing default is measured as a baseline. The fastest passing kernel from each family is tested enabled and disabled, with both compiled and segmented Attention execution. This is a hierarchical search, not the Cartesian product of every kernel configuration.

The tuner exhausts the finite set without an overall time limit. Individual compiler and GPU workers have timeouts and separate processes. Every kernel and measurement is checkpointed. Resume with identical arguments plus `--resume artifacts/tuning/tuning/SESSION`; GPU, software, compiler source and search settings must match. Interrupted launches are recorded as failures; a failed GPU health probe stops the search. Logs remain alongside results. `--cache-dir artifacts/previous` reuses compatible compiled stages.

Complete pipelines are measured in forward and reverse candidate order against an independent CPU FP32 reference using native-style bilinear preprocessing. Reference images run one at a time to bound memory. A separate input-order reversal check detects stale output and batch mapping errors. Results include vision batch, kernel settings, environment, measurements and numerical validation. Search fixtures are 512×288 RGB images (16:9).

A minimum cosine similarity of 0.995 is a numeric calibration check, not a retrieval-quality guarantee. The current tuner saves a proposed candidate with `activated=false`; it does not automatically replace a production bundle. Evaluate candidates on a fixed retrieval dataset before adopting them. The intended release criterion is at most one percentage point loss of retrieval R@1 against the original FP32 model, in both text-to-image and image-to-text retrieval, using all 256 images and 1280 captions. Do not use randomly generated benchmark images for this quality decision.

The specialized gfx1150 recipe uses BF16 residuals/linear layers in the vision tower, custom GELU and QK/RoPE kernels, packed layouts, paired normalization, segmented MIGraphX execution and native attention. The original default recipe applies to the guarded budget70/16:9/vision16 shape. The tuner also attempts shape-specific recipes for other prepared profiles; graph guards can reject unsupported compiler output. Each binary rewrite checks expected graph structure and a tuple-output mapping proof. If a guard fails, preparation retains the compiled fallback and logs the failure. A successful compilation alone does not validate model quality.

Run GPU parity checks explicitly:

```sh
python tests/tokenizer_parity.py --bundle artifacts/my-gpu --tokenizer models/google/tokenizer.json
python tests/text_parity.py --bundle artifacts/my-gpu
python tests/image_profile_parity.py --bundle artifacts/my-gpu --budget 140
```

See each script's `--help` for required reference/model arguments. CPU API tests use a fake embedder and do not establish GPU correctness. The preparation reference arrays are local weight/data artifacts and are ignored by Git.

The quality evaluator accepts a local manifest with `images`, each containing a unique `id`, relative `image` path, and `captions` array. It verifies exact dataset size, hashes decoded inputs and captions, and can cache an original CPU FP32 reference:

```sh
python tools/evaluate_quality.py --manifest dataset/manifest.json --data-root dataset \
  --model models/google --bundle artifacts/candidate --reference artifacts/reference.npz \
  --output artifacts/quality.json --budget 70
python tools/activate_profile.py --profile artifacts/tuning/session/profile.json \
  --quality artifacts/quality.json --output artifacts/validated
```

The activation tool requires a new destination, a passing complete report, and matching candidate bundle/profiles. It copies the candidate; it does not modify an existing deployment.

Use `HIP_VISIBLE_DEVICES` to select the preparation/tuning GPU. Preparation uses device 0 within that visible set. Native contexts and the server also accept a device ordinal. Recompile on each GPU architecture and repeat validation; gfx1150 results do not establish support or speed on other GPUs.

To reproduce the fixed sample, run `python tools/prepare_quality_dataset.py`. It uses Flickr8k's pinned `intro/flickr8k` test split at revision `c7a02cb487da0eca729e133194b0f842cb578c08`: sort 1000 rows by filename, select 256 without replacement with NumPy seed 20261016, keep all five captions. The tool saves the uncropped photos in a gray 528×288 letterbox; the evaluator then prescales them to 512×288. Use `--manifest artifacts/quality-data/manifest.json --data-root artifacts/quality-data` for evaluation. Photos and captions remain in ignored local artifacts. Check the dataset's terms before obtaining or redistributing it.

Prepare text profiles inside the selected candidate bundle before evaluating both modalities. The candidate path is `selected.bundle` in the tuning report; adding text profiles does not replace its measured image profile.

The full-model quality gate uses the agreed retrieval R@1 criterion. It also reports cosine similarity to the upstream FP32 pipeline, which includes a different resampler. The tuner's 0.995 numeric calibration compares candidates against CPU FP32 with the same bilinear preprocessing; it is a separate check.

For budget 280, this release supports at most 32 images per request and prepares execution batches no larger than 32. Larger requests return `EG2_UNSUPPORTED` (Python `UnsupportedProfile`, HTTP 503): both direct batch 128 and four chunks of 32 failed stability validation on gfx1150. Submit separate requests of at most 32 images. Budgets 70 and 140 retain request batches up to 128.

## Publish and reuse GPU recipes

After adding text profiles and passing the complete retrieval-quality check for the selected bundle, export the portable recipe:

```sh
eg2 export-tuning --profile artifacts/my-gpu/tuning/SESSION/profile.json \
  --quality artifacts/quality.json --output profiles/gpus/gfx1150/budget70-b4.json
```

Commit the JSON under `profiles/gpus/`. The Python distribution includes this directory. The record contains settings and validation evidence; it excludes model weights, compiled binaries and local paths. Another machine runs ordinary `eg2 prepare`: a matching recipe is selected automatically and compiled locally, without running a search. Exact GPU identity, software versions, model/profile and recipe-source digest must match. Missing or incompatible records select the default compiled path with a reason. `--tuning-profile PATH` requires that specific compatible record; `--no-presets` disables lookup.

## Shared image and text profiles

Convert an existing prepared bundle to a new directory:

```sh
eg2 share --bundle artifacts/my-gpu --output artifacts/my-gpu-shared
eg2 serve --bundle artifacts/my-gpu-shared
```

Conversion preserves the source. Schema 2 bundles externalize compiled literals and scratch storage, deduplicate identical weights, and allocate one workspace sized for the largest prepared profile. Image graphs, text modules and profile metadata remain resident. All calls within a context serialize on the shared execution stream. Image and text paths restore their inputs when switching profiles. The server prepares all profiles at startup; embedding users can call `Embedder.prepare()` / `Context::prepare()` / `eg2_prepare()` explicitly.

Memory does not multiply by the number of profiles: identical weights and workspace are shared. Distinct weight representations, graph/code metadata, and the largest required scratch space still consume memory. RGB input capacity can grow with input resolution. Different contexts and server workers each own their allocations. `capabilities.memory` reports weight, workspace, RGB and transient-pool bytes separately; whole-device memory includes other processes. Schema 1 bundles keep their previous eviction behavior. Add profiles before conversion; prepare a new source bundle and reconvert when changing the profile set.

`python tests/shared_profiles.py --original artifacts/my-gpu --shared artifacts/my-gpu-shared` checks image/text parity and stable allocation counters over 100 switches (requires image and text batches 1 and 4).

The input-order check requires cosine similarity ≥0.9999 and elementwise `atol=rtol=0.001`, recording both observed errors. This accommodates small FP32 GPU reduction differences; it is independent of the 0.995 calibration threshold and the real-image retrieval gate.
