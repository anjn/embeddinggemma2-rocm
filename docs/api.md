# Application interfaces

All embedding outputs are row-major FP32 `[batch,768]`, L2-normalized. Input RGB is contiguous uint8 `[batch,height,width,3]`. Request batches are 1..128. Native calls are synchronous and return after the CPU output buffer is populated. Caller-owned inputs and outputs remain owned by the caller. A context is serialized internally.

C applications include `eg2/eg2.h`, link `eg2`, create a context with a bundle directory, then call `eg2_encode_images`, `eg2_encode_texts`, or `eg2_encode_tokens`. Free it with `eg2_destroy`. The caller supplies an error-message buffer. No exception crosses the C interface. The C++ `eg2::Context` wrapper owns the context and throws on failed operations.

The Python `Embedder` supports context management and explicit `close()`. `encode_texts(..., prompt='query')` adds `task: search result | query: `; `prompt='document'` adds `title: none | text: `; `prompt=None` adds no prefix. Tokenization rejects inputs exceeding `max_length`, at most 8192. Prepared text profiles must cover the resulting length. The tokenizer uses the model's tokenizer.json. Image profiles must cover the exact input aspect ratio; image dimensions themselves can vary.

## HTTP

Text request:

```json
{"model":"embeddinggemma-2","input":["a cat","a street"],"input_type":"query","encoding_format":"float"}
```

Image request (replace the data URL with a real encoded image):

```json
{"model":"embeddinggemma-2","token_budget":70,"input":[{"content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,..."}}]}]}
```

Each image item contains exactly one image. Use a homogeneous text or image batch. Image URLs must be inline base64 data URLs; remote fetching is not implemented. Images in one request must have equal dimensions. The server decodes images but does not prescale them outside the native patch preprocessing. `encoding_format='base64'` encodes each vector as little-endian FP32 bytes. `dimensions` can only be 768.

Invalid inputs return 400; an unavailable prepared profile returns 503; a full queue returns 429. Default queue capacity is 32, and the request-body limit is 256 MiB. The server uses one serialized GPU worker. `/ready` includes bundle and last-execution metadata. `/health` checks process liveness. The server provides neither authentication nor a cross-GPU scheduler.

The context keeps one image execution profile and one text execution profile. Switching budgets or prepared batch/sequence shapes can evict and reload GPU programs. Include this cost when measuring workloads that mix profiles; steady-state benchmark cells intentionally keep one profile warm.

Image preprocessing uses half-pixel bilinear interpolation into the prepared patch grid, RGB scaling by 1/255, and 16×16 patch extraction. It approximates the upstream bicubic antialiased preprocessing. Use retrieval-quality validation on application data when selecting a fast profile.

For budget 280, this release supports at most 32 images per request and prepares execution batches no larger than 32. Larger requests return `EG2_UNSUPPORTED` (Python `UnsupportedProfile`, HTTP 503): both direct batch 128 and four chunks of 32 failed stability validation on gfx1150. Submit separate requests of at most 32 images. Budgets 70 and 140 retain request batches up to 128.
