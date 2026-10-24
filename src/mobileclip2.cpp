#include <mobileclip2/mobileclip2.h>
#include <onnxruntime_cxx_api.h>
#ifdef MC2_GPU
#include <migraphx/migraphx.h>
#endif
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <json.hpp>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <openssl/evp.h>
#include <sstream>
#include <stdexcept>
#include <vector>
using json = nlohmann::json;
namespace fs = std::filesystem;
extern "C" void *eg2_tokenizer_new(const char *, char *, size_t);
extern "C" void eg2_tokenizer_free(void *);
extern "C" int mc2_tokenizer_encode(const void *, const char *const *, size_t,
                                    int64_t *, char *, size_t);
namespace {
struct Unsupported : std::runtime_error {
  using runtime_error::runtime_error;
};
void require(bool ok, const char *message) {
  if (!ok)
    throw std::invalid_argument(message);
}
void message(char *p, size_t n, const std::string &s) {
  if (p && n) {
    auto k = std::min(n - 1, s.size());
    memcpy(p, s.data(), k);
    p[k] = 0;
  }
}
template <class F> mc2_status protect(char *e, size_t n, F fn) noexcept {
  try {
    fn();
    message(e, n, "");
    return MC2_OK;
  } catch (const std::invalid_argument &x) {
    message(e, n, x.what());
    return MC2_INVALID_ARGUMENT;
  } catch (const Unsupported &x) {
    message(e, n, x.what());
    return MC2_UNSUPPORTED;
  } catch (const std::exception &x) {
    message(e, n, x.what());
    return MC2_RUNTIME_ERROR;
  } catch (...) {
    message(e, n, "Unknown native exception");
    return MC2_RUNTIME_ERROR;
  }
}
json read_json(const fs::path &p) {
  std::ifstream f(p);
  if (!f)
    throw std::runtime_error("Cannot read " + p.string());
  return json::parse(f);
}
std::string sha256(const fs::path &p) {
  std::ifstream f(p, std::ios::binary);
  if (!f)
    throw std::runtime_error("Cannot read " + p.string());
  std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> c(EVP_MD_CTX_new(),
                                                            EVP_MD_CTX_free);
  if (!c || EVP_DigestInit_ex(c.get(), EVP_sha256(), nullptr) != 1)
    throw std::runtime_error("SHA256 init failed");
  std::array<char, 65536> b;
  while (f) {
    f.read(b.data(), b.size());
    if (EVP_DigestUpdate(c.get(), b.data(), f.gcount()) != 1)
      throw std::runtime_error("SHA256 failed");
  }
  if (!f.eof())
    throw std::runtime_error("Read failed " + p.string());
  unsigned char hash[32];
  unsigned int n = 0;
  EVP_DigestFinal_ex(c.get(), hash, &n);
  std::ostringstream s;
  for (auto v : hash)
    s << std::hex << std::setw(2) << std::setfill('0') << int(v);
  return s.str();
}
void normalize(float *y, size_t count) {
  for (size_t i = 0; i < count; i++) {
    double norm = 0;
    for (size_t j = 0; j < 512; j++) {
      float v = y[i * 512 + j];
      if (!std::isfinite(v))
        throw std::runtime_error("Non-finite embedding");
      norm += double(v) * v;
    }
    if (norm <= 0)
      throw std::runtime_error("Zero embedding");
    float scale = 1 / std::sqrt(norm);
    for (size_t j = 0; j < 512; j++)
      y[i * 512 + j] *= scale;
  }
}
// Pillow's separable bilinear downsampling includes antialiasing and rounds
// each uint8 intermediate. Coefficients use the same 22-bit fixed-point
// convention.
struct Filter {
  int first;
  std::vector<int> weights;
};
std::vector<Filter> filters(int size) {
  std::vector<Filter> result;
  double scale = double(size) / 256, support = std::max(1., scale);
  for (int x = 0; x < 256; x++) {
    double center = (x + .5) * scale;
    int lo = std::max(0, int(center - support + .5)),
        hi = std::min(size, int(center + support + .5));
    Filter f{lo, {}};
    std::vector<double> w;
    double sum = 0;
    for (int k = lo; k < hi; k++) {
      double v = std::max(0., 1 - std::abs((k - center + .5) / support));
      w.push_back(v);
      sum += v;
    }
    for (double v : w)
      f.weights.push_back(int(v / sum * (1 << 22) + .5));
    result.push_back(std::move(f));
  }
  return result;
}
void preprocess(const mc2_image &im, float *dst) {
  require(im.data && im.width && im.height, "Image is empty");
  require(im.width <= 16384 && im.height <= 16384,
          "Image dimensions exceed 16384");
  require(im.row_stride >= im.width * 3 &&
              im.row_stride <= SIZE_MAX / im.height,
          "Invalid RGB row stride");
  int side = int(std::max(im.width, im.height)),
      ox = (side - int(im.width)) / 2, oy = (side - int(im.height)) / 2;
  auto f = filters(side);
  std::vector<uint8_t> tmp(size_t(side) * 256 * 3, 0);
  for (int y = oy; y < oy + int(im.height); y++)
    for (int x = 0; x < 256; x++)
      for (int c = 0; c < 3; c++) {
        int64_t v = 1 << 21;
        for (size_t k = 0; k < f[x].weights.size(); k++) {
          int sx = f[x].first + int(k) - ox;
          if (sx >= 0 && sx < int(im.width))
            v += int64_t(im.data[size_t(y - oy) * im.row_stride + sx * 3 + c]) *
                 f[x].weights[k];
        }
        tmp[(size_t(y) * 256 + x) * 3 + c] =
            uint8_t(std::clamp<int64_t>(v >> 22, 0, 255));
      }
  for (int y = 0; y < 256; y++)
    for (int x = 0; x < 256; x++)
      for (int c = 0; c < 3; c++) {
        int64_t v = 1 << 21;
        for (size_t k = 0; k < f[y].weights.size(); k++)
          v += int64_t(tmp[(size_t(f[y].first + k) * 256 + x) * 3 + c]) *
               f[y].weights[k];
        dst[(c * 256 + y) * 256 + x] =
            float(std::clamp<int64_t>(v >> 22, 0, 255)) / 255.f;
      }
}
#ifdef MC2_GPU
void mg(migraphx_status s) {
  if (s != migraphx_status_success)
    throw std::runtime_error("MIGraphX status " + std::to_string(s));
}
struct Program {
  migraphx_program_t p = nullptr;
  migraphx_program_parameter_shapes_t shapes = nullptr;
  ~Program() {
    if (shapes)
      migraphx_program_parameter_shapes_destroy(shapes);
    if (p)
      migraphx_program_destroy(p);
  }
  void load(const fs::path &path) {
    migraphx_file_options_t opts = nullptr;
    mg(migraphx_file_options_create(&opts));
    auto s = migraphx_load(&p, path.c_str(), opts);
    migraphx_file_options_destroy(opts);
    mg(s);
    mg(migraphx_program_get_parameter_shapes(&shapes, p));
  }
  void run(const float *x, size_t batch, float *y) {
    const_migraphx_shape_t sh = nullptr;
    mg(migraphx_program_parameter_shapes_get(&sh, shapes, "images"));
    migraphx_shape_datatype_t type;
    size_t bytes;
    mg(migraphx_shape_type(&type, sh));
    mg(migraphx_shape_bytes(&bytes, sh));
    if (type != migraphx_shape_float_type ||
        bytes != batch * 3 * 256 * 256 * sizeof(float))
      throw std::runtime_error("Invalid compiled GPU input shape/type");
    migraphx_argument_t arg = nullptr;
    migraphx_program_parameters_t params = nullptr;
    migraphx_arguments_t outputs = nullptr;
    try {
      mg(migraphx_argument_create(&arg, sh, const_cast<float *>(x)));
      mg(migraphx_program_parameters_create(&params));
      mg(migraphx_program_parameters_add(params, "images", arg));
      mg(migraphx_program_run(&outputs, p, params));
      const_migraphx_argument_t out;
      mg(migraphx_arguments_get(&out, outputs, 0));
      const_migraphx_shape_t outsh;
      mg(migraphx_argument_shape(&outsh, out));
      mg(migraphx_shape_type(&type, outsh));
      mg(migraphx_shape_bytes(&bytes, outsh));
      if (type != migraphx_shape_float_type ||
          bytes != batch * 512 * sizeof(float))
        throw std::runtime_error("Invalid compiled GPU output shape/type");
      char *data;
      mg(migraphx_argument_buffer(&data, out));
      memcpy(y, data, bytes);
    } catch (...) {
      if (outputs)
        migraphx_arguments_destroy(outputs);
      if (params)
        migraphx_program_parameters_destroy(params);
      if (arg)
        migraphx_argument_destroy(arg);
      throw;
    }
    migraphx_arguments_destroy(outputs);
    migraphx_program_parameters_destroy(params);
    migraphx_argument_destroy(arg);
  }
};
#endif
struct TokenizerDeleter {
  void operator()(void *p) const { eg2_tokenizer_free(p); }
};
} // namespace
struct mc2_context {
  std::mutex mutex;
  fs::path root;
  std::string backend;
  json manifest;
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "mobileclip2"};
  std::unique_ptr<Ort::Session> text, image;
  std::unique_ptr<void, TokenizerDeleter> tokenizer;
#ifdef MC2_GPU
  std::map<size_t, std::unique_ptr<Program>> gpu;
#endif
  mc2_context(const fs::path &path, const std::string &b)
      : root(fs::absolute(path)), backend(b) {
    require(b == "gpu" || b == "npu", "Backend must be gpu or npu");
#ifndef MC2_GPU
    if (b == "gpu")
      throw Unsupported("GPU backend was not built");
#endif
#ifndef MC2_NPU
    if (b == "npu")
      throw Unsupported("NPU backend was not built");
#endif
    manifest = read_json(root / "manifest.json");
    require(manifest.at("format") == 1 &&
                manifest.at("model") == "MobileCLIP2-S2" &&
                manifest.at("dimension") == 512 &&
                manifest.at("preprocessing") == "letterbox-bilinear-256",
            "Incompatible MobileCLIP bundle");
    for (auto name : {"image", "text", "tokenizer"}) {
      std::string n = name;
      auto path = root / (n + (n == "tokenizer" ? ".json" : ".onnx"));
      if (sha256(path) != manifest.at(n + "_sha256").get<std::string>())
        throw std::runtime_error("Bundle checksum mismatch: " + n);
    }
    char error[4096]{};
    tokenizer.reset(eg2_tokenizer_new((root / "tokenizer.json").c_str(), error,
                                      sizeof error));
    if (!tokenizer)
      throw std::runtime_error(error);
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(4);
    opts.SetInterOpNumThreads(1);
    text =
        std::make_unique<Ort::Session>(env, (root / "text.onnx").c_str(), opts);
    auto meta = read_json(root / (b + ".json"));
    if (meta.at("image_sha256") != manifest.at("image_sha256"))
      throw std::runtime_error("Compiled model does not match ONNX");
    for (auto it = meta.at("artifacts").begin();
         it != meta.at("artifacts").end(); ++it) {
      if (sha256(root / it.key()) != it.value().get<std::string>())
        throw std::runtime_error("Compiled artifact checksum mismatch: " +
                                 it.key());
    }
    if (b == "gpu") {
#ifdef MC2_GPU
      for (size_t batch : {1, 4}) {
        auto p = std::make_unique<Program>();
        p->load(root / ("gpu-b" + std::to_string(batch) + ".mxr"));
        gpu.emplace(batch, std::move(p));
      }
#else
      throw Unsupported("GPU backend was not built");
#endif
    } else {
#ifdef MC2_NPU
      if (meta.at("onnxruntime") != Ort::GetVersionString())
        throw std::runtime_error(
            "NPU cache ONNX Runtime version mismatch; prepare again");
      std::string key = meta.at("cache_key");
      require(!key.empty() && key.find_first_not_of(
                                  "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQR"
                                  "STUVWXYZ0123456789-_") == std::string::npos,
              "Invalid cache key");
      if (!fs::exists(root / "npu-cache" / key / "context.json"))
        throw std::runtime_error(
            "NPU cache missing; run offline preparation first");
      Ort::SessionOptions npu;
      npu.SetIntraOpNumThreads(4);
      npu.AddFreeDimensionOverrideByName("batch", 1);
      npu.AddConfigEntry("session.disable_cpu_ep_fallback", "1");
      npu.AppendExecutionProvider_VitisAI(
          {{"config_file", (root / "npu-config.json").string()},
           {"cache_dir", (root / "npu-cache").string()},
           {"cache_key", key},
           {"enable_cache_file_io_in_mem", "0"}});
      try {
        image = std::make_unique<Ort::Session>(
            env, (root / "image.onnx").c_str(), npu);
      } catch (const Ort::Exception &e) {
        std::string reason = e.what();
        if (reason.find("Device or resource busy") != std::string::npos)
          reason +=
              "; NPU is busy. On ROCr 7.2, create the NPU context before any "
              "HIP/GPU initialization, or use a separate NPU process.";
        throw std::runtime_error(reason);
      }
#else
      throw Unsupported("NPU backend was not built");
#endif
    }
  }
  void batch(size_t n) {
    if (!((backend == "gpu" && (n == 1 || n == 4)) ||
          (backend == "npu" && n == 1)))
      throw Unsupported("Images require GPU batch 1 or 4, or NPU batch 1");
  }
  void pixels(const float *x, size_t n, float *y) {
    batch(n);
    require(x, "Null pixels");
    for (size_t i = 0; i < n * 3 * 256 * 256; i++)
      require(std::isfinite(x[i]) && x[i] >= 0 && x[i] <= 1,
              "Pixels must be finite float32 in [0,1]");
    if (backend == "gpu") {
#ifdef MC2_GPU
      gpu.at(n)->run(x, n, y);
#endif
    } else {
      auto mem =
          Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
      std::array<int64_t, 4> shape{1, 3, 256, 256};
      auto input = Ort::Value::CreateTensor<float>(
          mem, const_cast<float *>(x), 3 * 256 * 256, shape.data(), 4);
      const char *in = "images";
      const char *out = "embeddings";
      auto values =
          image->Run(Ort::RunOptions{nullptr}, &in, &input, 1, &out, 1);
      copy(values[0], y, 512);
    }
    normalize(y, n);
  }
  static void copy(const Ort::Value &v, float *y, size_t n) {
    auto info = v.GetTensorTypeAndShapeInfo();
    if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        info.GetElementCount() != n)
      throw std::runtime_error("Invalid ONNX output");
    memcpy(y, v.GetTensorData<float>(), n * sizeof(float));
  }
  void tokens(const int64_t *ids, size_t n, float *y) {
    require(ids, "Null tokens");
    for (size_t i = 0; i < n * 77; i++)
      require(ids[i] >= 0 && ids[i] < 49408, "Token out of range");
    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::array<int64_t, 2> shape{1, 77};
    const char *in = "tokens";
    const char *out = "embeddings";
    for (size_t i = 0; i < n; i++) {
      auto input = Ort::Value::CreateTensor<int64_t>(
          mem, const_cast<int64_t *>(ids + i * 77), 77, shape.data(), 2);
      auto values =
          text->Run(Ort::RunOptions{nullptr}, &in, &input, 1, &out, 1);
      copy(values[0], y + i * 512, 512);
    }
    normalize(y, n);
  }
  void tokenize(const char *const *texts, size_t n, int64_t *ids) {
    require(texts && ids, "Null text/tokens");
    char e[4096]{};
    int s = mc2_tokenizer_encode(tokenizer.get(), texts, n, ids, e, sizeof e);
    if (s)
      throw std::invalid_argument(e);
  }
};
namespace {
void output(mc2_context *c, size_t n, float *y, size_t elements) {
  require(c && y, "Null context/output");
  require(n > 0 && n <= SIZE_MAX / (512 * sizeof(float)) && elements >= n * 512,
          "Invalid count/output capacity");
}
} // namespace
extern "C" {
mc2_status mc2_create(const char *bundle, const char *backend,
                      mc2_context **out, char *e, size_t cap) {
  if (out)
    *out = nullptr;
  return protect(e, cap, [&] {
    require(bundle && backend && out, "Null create argument");
    auto c = std::make_unique<mc2_context>(bundle, backend);
    *out = c.release();
  });
}
void mc2_destroy(mc2_context *c) { delete c; }
mc2_status mc2_encode_pixels(mc2_context *c, const float *x, size_t n, float *y,
                             size_t elements, char *e, size_t cap) {
  return protect(e, cap, [&] {
    output(c, n, y, elements);
    std::lock_guard lock(c->mutex);
    c->pixels(x, n, y);
  });
}
mc2_status mc2_encode_images(mc2_context *c, const mc2_image *images, size_t n,
                             float *y, size_t elements, char *e, size_t cap) {
  return protect(e, cap, [&] {
    output(c, n, y, elements);
    require(images, "Null images");
    std::lock_guard lock(c->mutex);
    c->batch(n);
    std::vector<float> x(n * 3 * 256 * 256);
    for (size_t i = 0; i < n; i++)
      preprocess(images[i], x.data() + i * 3 * 256 * 256);
    c->pixels(x.data(), n, y);
  });
}
mc2_status mc2_tokenize(mc2_context *c, const char *const *texts, size_t n,
                        int64_t *ids, size_t elements, char *e, size_t cap) {
  return protect(e, cap, [&] {
    require(c && n && n <= SIZE_MAX / 77 && elements >= n * 77,
            "Invalid tokenizer arguments/capacity");
    std::lock_guard lock(c->mutex);
    c->tokenize(texts, n, ids);
  });
}
mc2_status mc2_encode_tokens(mc2_context *c, const int64_t *ids, size_t n,
                             float *y, size_t elements, char *e, size_t cap) {
  return protect(e, cap, [&] {
    output(c, n, y, elements);
    std::lock_guard lock(c->mutex);
    c->tokens(ids, n, y);
  });
}
mc2_status mc2_encode_texts(mc2_context *c, const char *const *texts, size_t n,
                            float *y, size_t elements, char *e, size_t cap) {
  return protect(e, cap, [&] {
    output(c, n, y, elements);
    std::lock_guard lock(c->mutex);
    std::vector<int64_t> ids(n * 77);
    c->tokenize(texts, n, ids.data());
    c->tokens(ids.data(), n, y);
  });
}
mc2_status mc2_prepare(mc2_context *c, char *e, size_t cap) {
  return protect(e, cap, [&] {
    require(c, "Null context");
    std::lock_guard lock(c->mutex);
    std::vector<float> x(4 * 3 * 256 * 256, .5f), y(4 * 512);
    c->pixels(x.data(), 1, y.data());
    if (c->backend == "gpu")
      c->pixels(x.data(), 4, y.data());
    std::array<int64_t, 77> ids{};
    const char *text = "warmup";
    c->tokenize(&text, 1, ids.data());
    c->tokens(ids.data(), 1, y.data());
  });
}
mc2_status mc2_info(mc2_context *c, char *out, size_t cap) {
  return protect(nullptr, 0, [&] {
    require(c && out, "Null context/output");
    std::lock_guard lock(c->mutex);
    json info = c->manifest;
    info["backend"] = c->backend;
    info["image_batches"] = c->backend == "gpu" ? json({1, 4}) : json({1});
    info["text_backend"] = "cpu-fp32";
    auto s = info.dump();
    require(cap > s.size(), "Info buffer too small");
    memcpy(out, s.c_str(), s.size() + 1);
  });
}
}
