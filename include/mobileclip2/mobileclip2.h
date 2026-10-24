#ifndef MOBILECLIP2_H
#define MOBILECLIP2_H
#include <stddef.h>
#include <stdint.h>
#if defined(_WIN32)
#define MC2_API __declspec(dllexport)
#else
#define MC2_API __attribute__((visibility("default")))
#endif
#ifdef __cplusplus
extern "C" {
#endif
typedef struct mc2_context mc2_context;
typedef enum {
  MC2_OK = 0,
  MC2_INVALID_ARGUMENT = 1,
  MC2_RUNTIME_ERROR = 2,
  MC2_UNSUPPORTED = 3
} mc2_status;
/* RGB uint8. data must contain at least (height-1)*row_stride+width*3 bytes. */
typedef struct {
  const uint8_t *data;
  size_t width, height, row_stride;
} mc2_image;
/* backend is "gpu" or "npu". Bundle must be prepared on a compatible machine.
   Context calls serialize; destruction must not race another call. No Python
   needed. */
MC2_API mc2_status mc2_create(const char *bundle, const char *backend,
                              mc2_context **out, char *error, size_t capacity);
MC2_API void mc2_destroy(mc2_context *ctx);
MC2_API mc2_status mc2_prepare(mc2_context *ctx, char *error, size_t capacity);
/* GPU accepts exactly 1 or 4 images; NPU exactly 1. Output is count*512 floats.
   Letterbox to square, bilinear resize 256, scale by 1/255; L2-normalized
   output. */
MC2_API mc2_status mc2_encode_images(mc2_context *ctx, const mc2_image *images,
                                     size_t count, float *output,
                                     size_t elements, char *error,
                                     size_t capacity);
/* Preprocessed contiguous float32 [count,3,256,256], values in [0,1]. */
MC2_API mc2_status mc2_encode_pixels(mc2_context *ctx, const float *pixels,
                                     size_t count, float *output,
                                     size_t elements, char *error,
                                     size_t capacity);
/* UTF-8, CLIP BPE, 77 tokens including BOS/EOS; long text is truncated.
   Text encoder runs on CPU, one query at a time, for either image backend. */
MC2_API mc2_status mc2_tokenize(mc2_context *ctx, const char *const *texts,
                                size_t count, int64_t *ids, size_t elements,
                                char *error, size_t capacity);
MC2_API mc2_status mc2_encode_tokens(mc2_context *ctx, const int64_t *ids,
                                     size_t count, float *output,
                                     size_t elements, char *error,
                                     size_t capacity);
MC2_API mc2_status mc2_encode_texts(mc2_context *ctx, const char *const *texts,
                                    size_t count, float *output,
                                    size_t elements, char *error,
                                    size_t capacity);
MC2_API mc2_status mc2_info(mc2_context *ctx, char *json, size_t capacity);
#ifdef __cplusplus
}
#endif
#endif
