#pragma once
#include "mobileclip2.h"
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
namespace mobileclip2 {
class Embedder {
  mc2_context *ctx_ = nullptr;
  static void check(mc2_status s, const char *e) {
    if (s != MC2_OK)
      throw std::runtime_error(e);
  }

public:
  Embedder(const std::string &bundle, const std::string &backend = "gpu") {
    char e[4096]{};
    auto s = mc2_create(bundle.c_str(), backend.c_str(), &ctx_, e, sizeof e);
    check(s, e);
  }
  ~Embedder() { mc2_destroy(ctx_); }
  Embedder(const Embedder &) = delete;
  Embedder &operator=(const Embedder &) = delete;
  Embedder(Embedder &&other) noexcept
      : ctx_(std::exchange(other.ctx_, nullptr)) {}
  Embedder &operator=(Embedder &&other) noexcept {
    if (this != &other) {
      mc2_destroy(ctx_);
      ctx_ = std::exchange(other.ctx_, nullptr);
    }
    return *this;
  }
  void prepare() {
    char e[4096]{};
    auto s = mc2_prepare(ctx_, e, sizeof e);
    check(s, e);
  }
  std::vector<float> images(const std::vector<mc2_image> &x) {
    std::vector<float> y(x.size() * 512);
    char e[4096]{};
    auto s = mc2_encode_images(ctx_, x.data(), x.size(), y.data(), y.size(), e,
                               sizeof e);
    check(s, e);
    return y;
  }
  std::vector<float> texts(const std::vector<std::string> &x) {
    std::vector<const char *> p;
    for (auto &s : x)
      p.push_back(s.c_str());
    std::vector<float> y(x.size() * 512);
    char e[4096]{};
    auto s = mc2_encode_texts(ctx_, p.data(), p.size(), y.data(), y.size(), e,
                              sizeof e);
    check(s, e);
    return y;
  }
};
} // namespace mobileclip2
