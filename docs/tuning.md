# GPU tuning and correctness

`eg2 tune --model models/google --output artifacts/tuning --budget 70 --batch 128 --vision-batches 1 4 8 16 32` compiles candidate vision chunks, measures the full pipeline in separate processes in forward and reverse order, and compares outputs to a separately compiled standard profile. Divisibility and shape guards limit candidates. Results record the GPU/software fingerprint and measurements.

A minimum cosine similarity of 0.995 is a numeric calibration check, not a retrieval-quality guarantee. The current tuner saves a proposed candidate with `activated=false`; it does not automatically replace a production bundle. Evaluate candidates on a fixed retrieval dataset before adopting them. The intended release criterion is at most one percentage point loss of retrieval R@1 against the original FP32 model, in both text-to-image and image-to-text retrieval, using all 256 images and 1280 captions. Do not use randomly generated benchmark images for this quality decision.

The specialized gfx1150 recipe uses BF16 residuals/linear layers in the vision tower, custom GELU and QK/RoPE kernels, packed layouts, paired normalization, segmented MIGraphX execution and native attention. It applies only to the guarded budget70/16:9/vision16 shape. Each binary rewrite checks expected graph structure and a tuple-output mapping proof. If a guard fails, preparation retains the compiled fallback and logs the failure. A successful compilation alone does not validate model quality.

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

The full-model quality gate uses the agreed retrieval R@1 criterion. It also reports cosine similarity to the upstream FP32 pipeline, which includes a different resampler. The tuner's 0.995 numeric calibration compares candidates against a standard profile with the same native bilinear preprocessing; it is a separate check.

For budget 280, this release supports at most 32 images per request and prepares execution batches no larger than 32. Larger requests return `EG2_UNSUPPORTED` (Python `UnsupportedProfile`, HTTP 503): both direct batch 128 and four chunks of 32 failed stability validation on gfx1150. Submit separate requests of at most 32 images. Budgets 70 and 140 retain request batches up to 128.
