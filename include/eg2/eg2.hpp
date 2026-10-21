#pragma once
#include "eg2.h"
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
namespace eg2 {
class Context {
  eg2_context *handle_ = nullptr;
  static void check(eg2_status status, const char *error) {
    if (status != EG2_OK)
      throw std::runtime_error(error);
  }

public:
  explicit Context(const std::string &bundle, int device = 0) {
    char error[4096];
    check(eg2_create(bundle.c_str(), device, &handle_, error, sizeof(error)),
          error);
  }
  ~Context() { eg2_destroy(handle_); }
  Context(const Context &) = delete;
  Context &operator=(const Context &) = delete;
  Context(Context &&other) noexcept
      : handle_(std::exchange(other.handle_, nullptr)) {}
  Context &operator=(Context &&other) noexcept {
    if (this != &other) {
      eg2_destroy(handle_);
      handle_ = std::exchange(other.handle_, nullptr);
    }
    return *this;
  }
  std::vector<float> images(const uint8_t *rgb, size_t batch, size_t height,
                            size_t width, int budget = 70) {
    std::vector<float> out(batch * 768);
    char error[4096];
    check(eg2_encode_images(handle_, rgb, batch, height, width, budget,
                            out.data(), out.size(), error, sizeof(error)),
          error);
    return out;
  }
  std::vector<float>
  texts(const std::vector<std::string> &texts,
        const std::string &prefix = "task: search result | query: ") {
    std::vector<const char *> ptrs;
    for (auto &t : texts)
      ptrs.push_back(t.c_str());
    std::vector<float> out(texts.size() * 768);
    char error[4096];
    check(eg2_encode_texts(handle_, ptrs.data(), ptrs.size(), prefix.c_str(),
                           out.data(), out.size(), error, sizeof(error)),
          error);
    return out;
  }
  void prepare() {
    char error[4096];
    check(eg2_prepare(handle_, error, sizeof(error)), error);
  }
  eg2_context *get() const noexcept { return handle_; }
};
} // namespace eg2
