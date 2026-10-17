#include <algorithm>
#include <cmath>
#include <cstring>
#include <ggml-backend.h>
#include <llama.h>
#include <memory>
#include <mtmd.h>
#include <stdexcept>
#include <string>
#include <vector>
using Chunks =
    std::unique_ptr<mtmd_input_chunks, decltype(&mtmd_input_chunks_free)>;
using Bitmap = std::unique_ptr<mtmd_bitmap, decltype(&mtmd_bitmap_free)>;
using Batch = std::unique_ptr<mtmd_batch, decltype(&mtmd_batch_free)>;
struct Adapter {
  llama_model *model = nullptr;
  llama_context *context = nullptr;
  mtmd_context *vision = nullptr;
  int budget, max_batch, actual_tokens = 0, actual_sequence = 0,
                         vision_calls = 0, vision_limit;
  ~Adapter() {
    if (vision)
      mtmd_free(vision);
    if (context)
      llama_free(context);
    if (model)
      llama_model_free(model);
  }
};
static void error(char *e, size_t cap, const char *msg) {
  if (e && cap) {
    std::strncpy(e, msg, cap - 1);
    e[cap - 1] = 0;
  }
}
static bool is_gpu(ggml_backend_dev_t device) {
  auto type = ggml_backend_dev_type(device);
  return type == GGML_BACKEND_DEVICE_TYPE_GPU ||
         type == GGML_BACKEND_DEVICE_TYPE_IGPU;
}
extern "C" void *eg2_llama_create(const char *model, const char *mmproj,
                                  int budget, int batch, int flash,
                                  int vision_chunk, char *err, size_t cap) {
  try {
    ggml_backend_load_all();
    bool found_gpu = false;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++)
      if (is_gpu(ggml_backend_dev_get(i)))
        found_gpu = true;
    if (!found_gpu)
      throw std::runtime_error("No GPU backend available");
    llama_backend_init();
    auto a = std::make_unique<Adapter>();
    a->budget = budget;
    a->max_batch = batch;
    a->vision_limit = vision_chunk;
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
    a->model = llama_model_load_from_file(model, mp);
    if (!a->model)
      throw std::runtime_error("Cannot load GGUF model");
    auto cp = llama_context_default_params();
    cp.n_ctx = (budget + 32) * batch;
    cp.n_batch = cp.n_ctx;
    cp.n_ubatch = cp.n_ctx;
    cp.n_seq_max = batch;
    cp.n_threads = 4;
    cp.n_threads_batch = 4;
    cp.embeddings = true;
    cp.pooling_type = LLAMA_POOLING_TYPE_MEAN;
    cp.attention_type = LLAMA_ATTENTION_TYPE_NON_CAUSAL;
    cp.flash_attn_type =
        flash ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.type_k = GGML_TYPE_F16;
    cp.type_v = GGML_TYPE_F16;
    a->context = llama_init_from_model(a->model, cp);
    if (!a->context)
      throw std::runtime_error("Cannot create llama context");
    auto vp = mtmd_context_params_default();
    vp.use_gpu = true;
    vp.n_threads = 4;
    // Stock resize rounds up to 48-pixel blocks. These lower bounds force
    // 512x288 RGB inputs to the same 66/120/264-token grids as the HF budgets.
    vp.image_min_tokens = budget == 70 ? 57 : budget == 140 ? 111 : 249;
    vp.image_max_tokens = budget;
    vp.flash_attn_type = cp.flash_attn_type;
    vp.warmup = false;
    vp.batch_max_tokens = budget * vision_chunk;
    a->vision = mtmd_init_from_file(mmproj, a->model, vp);
    if (!a->vision)
      throw std::runtime_error("Cannot load GGUF projector");
    if (llama_model_n_embd_out(a->model) != 768 ||
        llama_model_n_embd_inp(a->model) != 512)
      throw std::runtime_error("Unexpected embedding dimensions");
    return a.release();
  } catch (const std::exception &e) {
    error(err, cap, e.what());
    return nullptr;
  }
}
extern "C" void eg2_llama_destroy(void *p) { delete static_cast<Adapter *>(p); }
extern "C" const char *eg2_llama_device() {
  for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
    auto dev = ggml_backend_dev_get(i);
    if (is_gpu(dev))
      return ggml_backend_dev_name(dev);
  }
  return "none";
}
extern "C" const char *eg2_llama_system() { return llama_print_system_info(); }
extern "C" int eg2_llama_info(void *p, int *out) {
  auto *a = static_cast<Adapter *>(p);
  if (!a || !out)
    return 1;
  out[0] = a->actual_tokens;
  out[1] = a->actual_sequence;
  out[2] = a->vision_calls;
  out[3] = a->vision_limit;
  return 0;
}
extern "C" int eg2_llama_encode(void *p, const unsigned char *rgb, int count,
                                int h, int w, float *out, char *err,
                                size_t cap) {
  try {
    auto *a = static_cast<Adapter *>(p);
    if (!a || !rgb || !out || count < 1 || count > a->max_batch)
      throw std::runtime_error("Invalid encode arguments");
    auto memory = llama_get_memory(a->context);
    if (memory)
      llama_memory_clear(memory, true);
    std::vector<Bitmap> bitmaps;
    std::vector<Chunks> chunks;
    std::vector<const mtmd_input_chunk *> media;
    std::string marker = mtmd_default_marker();
    mtmd_input_text text{marker.c_str(), marker.size(), true, true};
    a->actual_tokens = 0;
    a->actual_sequence = 0;
    a->vision_calls = 0;
    for (int i = 0; i < count; i++) {
      Bitmap bitmap(mtmd_bitmap_init(w, h, rgb + size_t(i) * h * w * 3),
                    mtmd_bitmap_free);
      Chunks ch(mtmd_input_chunks_init(), mtmd_input_chunks_free);
      const mtmd_bitmap *ptr = bitmap.get();
      if (mtmd_tokenize(a->vision, ch.get(), &text, &ptr, 1))
        throw std::runtime_error("llama image tokenization failed");
      int seq = 0;
      for (size_t j = 0; j < mtmd_input_chunks_size(ch.get()); j++) {
        auto *c = mtmd_input_chunks_get(ch.get(), j);
        seq += mtmd_input_chunk_get_n_tokens(c);
        if (mtmd_input_chunk_get_type(c) == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
          media.push_back(c);
          if (i == 0)
            a->actual_tokens += mtmd_input_chunk_get_n_tokens(c);
        }
      }
      if (i == 0)
        a->actual_sequence = seq;
      chunks.push_back(std::move(ch));
      bitmaps.push_back(std::move(bitmap));
    }
    // Keep encoded groups alive until the mixed backbone batch has been
    // evaluated.
    std::vector<Batch> batches;
    size_t next = 0;
    while (next < media.size()) {
      Batch b(mtmd_batch_init(a->vision), mtmd_batch_free);
      size_t before = next;
      while (next < media.size() && next - before < size_t(a->vision_limit) &&
             mtmd_batch_add_chunk(b.get(), media[next]) == 0)
        next++;
      if (next == before)
        throw std::runtime_error("Media chunk cannot fit in a vision batch");
      if (mtmd_batch_encode(b.get()))
        throw std::runtime_error("llama vision encode failed");
      batches.push_back(std::move(b));
      a->vision_calls++;
    }
    std::unique_ptr<llama_batch_ext, decltype(&llama_batch_ext_free)> batch(
        llama_batch_ext_init(a->context), llama_batch_ext_free);
    auto set = [&](int idx, int pos) {
      llama_pos coords[4] = {pos, 0, 0, 0};
      if (idx < 0 || !llama_batch_ext_set_pos(batch.get(), idx, coords) ||
          !llama_batch_ext_set_output_embd(batch.get(), idx, true))
        throw std::runtime_error("Cannot build mixed embedding batch");
    };
    for (int i = 0; i < count; i++) {
      int pos = 0;
      for (size_t j = 0; j < mtmd_input_chunks_size(chunks[i].get()); j++) {
        auto *c = mtmd_input_chunks_get(chunks[i].get(), j);
        if (mtmd_input_chunk_get_type(c) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
          size_t n = 0;
          auto *ids = mtmd_input_chunk_get_tokens_text(c, &n);
          for (size_t k = 0; k < n; k++)
            set(llama_batch_ext_add_token(batch.get(), i, ids[k]), pos++);
        } else {
          float *value = nullptr;
          for (auto &b : batches) {
            value = mtmd_batch_get_output_embd(b.get(), c);
            if (value)
              break;
          }
          if (!value)
            throw std::runtime_error("Missing vision embeddings");
          for (size_t k = 0; k < mtmd_input_chunk_get_n_tokens(c); k++) {
            llama_embd embd{value + k * 512, 1, 512};
            set(llama_batch_ext_add_embd(batch.get(), i, embd), pos++);
          }
        }
      }
    }
    if (llama_process(a->context,
                      llama_model_has_encoder(a->model)
                          ? LLAMA_PROCESS_TYPE_ENCODE
                          : LLAMA_PROCESS_TYPE_DECODE,
                      batch.get()))
      throw std::runtime_error("llama backbone encode failed");
    for (int i = 0; i < count; i++) {
      float *v = llama_get_embeddings_seq(a->context, i);
      if (!v)
        throw std::runtime_error("Missing pooled embedding");
      double norm = 0;
      for (int k = 0; k < 768; k++)
        norm += double(v[k]) * v[k];
      norm = std::sqrt(norm);
      if (!(norm > 0))
        throw std::runtime_error("Invalid pooled norm");
      for (int k = 0; k < 768; k++)
        out[i * 768 + k] = v[k] / norm;
    }
    return 0;
  } catch (const std::exception &e) {
    error(err, cap, e.what());
    return 1;
  }
}
