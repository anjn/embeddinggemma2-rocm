#ifndef EG2_EG2_H
#define EG2_EG2_H
#include <stddef.h>
#include <stdint.h>
#if defined(_WIN32)
#define EG2_API __declspec(dllexport)
#else
#define EG2_API __attribute__((visibility("default")))
#endif
#ifdef __cplusplus
extern "C" {
#endif
typedef struct eg2_context eg2_context;
typedef enum {
  EG2_OK = 0,
  EG2_INVALID_ARGUMENT = 1,
  EG2_RUNTIME_ERROR = 2,
  EG2_UNSUPPORTED = 3
} eg2_status;
/* Bundle paths are local trusted compiled-model assets. Calls on a context
   serialize. Input/output buffers belong to the caller. No Python interpreter
   is required. */
EG2_API const char *eg2_version(void);
EG2_API eg2_status eg2_create(const char *bundle_path, int device,
                              eg2_context **out, char *error, size_t capacity);
EG2_API void eg2_destroy(eg2_context *context);
/* RGB is contiguous [batch,height,width,3]. Output has batch*768 float
   elements. First call for a shape warms up/captures the graph; exclude it from
   steady-state timing. Budget 280 accepts at most 32 images per request;
   larger requests return EG2_UNSUPPORTED. */
EG2_API eg2_status eg2_encode_images(eg2_context *context, const uint8_t *rgb,
                                     size_t batch, size_t height, size_t width,
                                     int token_budget, float *output,
                                     size_t output_elements, char *error,
                                     size_t capacity);
/* Token arrays are contiguous [batch,sequence], with int64 ids and 0/1
 * attention mask. */
EG2_API eg2_status eg2_encode_tokens(eg2_context *context, const int64_t *ids,
                                     const int64_t *mask, size_t batch,
                                     size_t sequence, float *output,
                                     size_t output_elements, char *error,
                                     size_t capacity);
EG2_API eg2_status eg2_tokenize(eg2_context *, const char *const *texts,
                                size_t count, const char *prefix, int64_t *ids,
                                int64_t *mask, size_t slots, size_t *lengths,
                                char *error, size_t capacity);
EG2_API eg2_status eg2_encode_texts(eg2_context *, const char *const *texts,
                                    size_t count, const char *prefix,
                                    float *output, size_t output_elements,
                                    char *error, size_t capacity);
EG2_API eg2_status eg2_info(eg2_context *context, char *json, size_t capacity);
#ifdef __cplusplus
}
#endif
#endif
