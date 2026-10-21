#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <ATen/hip/HIPGraph.h>
#include <ATen/hip/MemPool.h>
#include <ATen/ops/_scaled_dot_product_flash_attention.h>
#include <algorithm>
#include <c10/core/InferenceMode.h>
#include <c10/hip/HIPCachingAllocator.h>
#include <c10/hip/HIPGuard.h>
#include <c10/hip/HIPStream.h>
#include <cstring>
#include <eg2/eg2.h>
#include <filesystem>
#include <fstream>
#include <hip/hip_runtime.h>
#include <json.hpp>
#include <map>
#include <memory>
#include <migraphx/migraphx.h>
#include <mutex>
#include <stdexcept>
#include <string>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <torch/script.h>
#include <vector>
namespace fs = std::filesystem;
using json = nlohmann::json;
extern "C" void *eg2_tokenizer_new(const char *, char *, size_t);
extern "C" void eg2_tokenizer_free(void *);
extern "C" int eg2_tokenizer_encode(const void *, const char *const *, size_t,
                                    const char *, int64_t *, int64_t *, size_t,
                                    size_t *, char *, size_t);
extern "C" int eg2_patchify_config(void *, void *, int, int, int, int, int,
                                   void *, int, int);
namespace {
void mg(migraphx_status s) {
  if (s != migraphx_status_success)
    throw std::runtime_error("MIGraphX status " + std::to_string(s));
}
void hip(hipError_t s) {
  if (s != hipSuccess)
    throw std::runtime_error(hipGetErrorString(s));
}
json read_json(const fs::path &p) {
  std::ifstream f(p);
  if (!f)
    throw std::runtime_error("Cannot read " + p.string());
  json j;
  f >> j;
  return j;
}
at::ScalarType dtype(const std::string &s) {
  if (s == "bf16_type" || s == "bfloat16")
    return at::kBFloat16;
  if (s == "half_type" || s == "float16")
    return at::kHalf;
  if (s == "float_type" || s == "float32")
    return at::kFloat;
  if (s == "int64_type" || s == "int64")
    return at::kLong;
  if (s == "int32_type" || s == "int32")
    return at::kInt;
  if (s == "uint8_type" || s == "uint8" || s == "int8_type")
    return at::kByte;
  throw std::runtime_error("Unsupported dtype " + s);
}
at::TensorOptions gpu(int device, at::ScalarType t = at::kByte) {
  return at::TensorOptions().device(at::kCUDA, device).dtype(t);
}
int64_t span(const json &s) {
  auto lens = s.at("lens").get<std::vector<int64_t>>(),
       strides = s.at("strides").get<std::vector<int64_t>>();
  int64_t n = 1;
  for (size_t i = 0; i < lens.size(); i++)
    n += (lens[i] - 1) * strides[i];
  return n * c10::elementSize(dtype(s.at("type")));
}
at::Tensor tensor_file(const fs::path &root, const json &s, int device) {
  auto lens = s.at("lens").get<std::vector<int64_t>>();
  auto t = at::empty(lens, at::TensorOptions().dtype(dtype(s.at("dtype"))));
  std::ifstream f(root / s.at("file").get<std::string>(),
                  std::ios::binary | std::ios::ate);
  if (!f || f.tellg() != static_cast<std::streamoff>(t.nbytes()))
    throw std::runtime_error("Invalid tensor file " + s.dump());
  f.seekg(0);
  f.read(static_cast<char *>(t.data_ptr()), t.nbytes());
  if (!f)
    throw std::runtime_error("Cannot read tensor");
  return t.to(at::Device(at::kCUDA, device));
}
// One serialized context owns stable addresses shared by every profile.
struct SharedMemory {
  fs::path root;
  json config;
  int device;
  at::Tensor arena, rgb;
  size_t cursor = 0, weights_bytes = 0, switches = 0;
  std::string active;
  std::map<std::string, at::Tensor> weights;
  c10::cuda::CUDAStream stream;
  at::cuda::MemPool transient;
  SharedMemory(const fs::path &p, const json &c, int dev)
      : root(p), config(c), device(dev),
        stream(c10::cuda::getStreamFromPool(false, dev)) {
    arena = at::empty({config.at("workspace_bytes").get<int64_t>()}, gpu(dev));
  }
  void reset() { cursor = 0; }
  at::Tensor allocate(at::IntArrayRef sizes, at::ScalarType type) {
    size_t count = 1;
    for (auto n : sizes)
      count *= n;
    size_t bytes = count * c10::elementSize(type);
    if (cursor + bytes > size_t(arena.numel()))
      throw std::runtime_error("Shared workspace manifest is too small");
    auto result = arena.narrow(0, cursor, bytes).view(type).view(sizes);
    cursor += (bytes + 255) / 256 * 256;
    return result;
  }
  at::Tensor weight(const std::string &id) {
    auto spec = config.at("weights").at(id);
    auto key = spec.at("sha256").get<std::string>();
    if (weights.count(key)) {
      if (weights.at(key).nbytes() != spec.at("bytes").get<size_t>())
        throw std::runtime_error("Inconsistent shared weight extent");
      return weights.at(key);
    }
    auto host = at::empty({spec.at("bytes").get<int64_t>()}, at::kByte);
    std::ifstream f(root / spec.at("file").get<std::string>(),
                    std::ios::binary | std::ios::ate);
    if (!f || f.tellg() != static_cast<std::streamoff>(host.nbytes()))
      throw std::runtime_error("Shared weight size mismatch");
    f.seekg(0);
    f.read(static_cast<char *>(host.data_ptr()), host.nbytes());
    if (!f)
      throw std::runtime_error("Cannot read shared weight");
    auto value = host.to(at::Device(at::kCUDA, device));
    weights_bytes += value.nbytes();
    weights.emplace(key, value);
    return value;
  }
  at::Tensor tensor(const fs::path &path, const json &spec) {
    auto key = spec.at("sha256").get<std::string>();
    if (!weights.count(key)) {
      auto value =
          tensor_file(path, spec, device).reshape({-1}).view(at::kByte);
      weights_bytes += value.nbytes();
      weights.emplace(key, value);
    }
    return weights.at(key)
        .view(dtype(spec.at("dtype")))
        .view(spec.at("lens").get<std::vector<int64_t>>());
  }

  void activate(const std::string &key) {
    if (active != key) {
      ++switches;
      active = key;
    }
  }
};
struct PoolGuard {
  SharedMemory *memory;
  explicit PoolGuard(SharedMemory *m) : memory(m) {
    if (memory) {
      auto selected = memory->stream.stream();
      c10::cuda::CUDACachingAllocator::beginAllocateToPool(
          memory->device, memory->transient.id(),
          [selected](hipStream_t s) { return s == selected; });
    }
  }
  ~PoolGuard() {
    if (memory)
      c10::cuda::CUDACachingAllocator::endAllocateToPool(
          memory->device, memory->transient.id());
  }
};
at::Tensor storage(SharedMemory *m, at::IntArrayRef sizes, int dev,
                   at::ScalarType type = at::kByte) {
  return m ? m->allocate(sizes, type) : at::empty(sizes, gpu(dev, type));
}
// Program handles and borrowed shapes stay alive together. All argument storage
// is GPU-resident.
struct Stage {
  migraphx_program_t program = nullptr;
  migraphx_program_parameter_shapes_t shapes = nullptr;
  migraphx_program_parameters_t params = nullptr;
  std::map<std::string, at::Tensor> buffers;
  at::Tensor output;
  Stage(const fs::path &path, int device, bool allocate = true,
        SharedMemory *shared = nullptr) {
    migraphx_file_options_t opts = nullptr;
    mg(migraphx_file_options_create(&opts));
    auto status = migraphx_load(&program, path.c_str(), opts);
    migraphx_file_options_destroy(opts);
    mg(status);
    mg(migraphx_program_get_parameter_shapes(&shapes, program));
    mg(migraphx_program_parameters_create(&params));
    size_t count = 0;
    mg(migraphx_program_parameter_shapes_size(&count, shapes));
    std::vector<const char *> names(count);
    mg(migraphx_program_parameter_shapes_names(names.data(), shapes));
    json memory;
    if (shared)
      memory = read_json(path.string() + ".memory.json");
    for (auto name : names) {
      const_migraphx_shape_t sh;
      mg(migraphx_program_parameter_shapes_get(&sh, shapes, name));
      if (shared && memory.at("parameters").at(name).contains("weight")) {
        buffers[name] = shared->weight(memory["parameters"][name]["weight"]);
        bind(name, buffers[name].data_ptr());
      } else if (allocate ||
                 (shared && std::string(name).find("eg2_workspace_") == 0)) {
        size_t bytes = 0;
        mg(migraphx_shape_bytes(&bytes, sh));
        buffers[name] = storage(shared, {static_cast<int64_t>(bytes)}, device);
        bind(name, buffers[name].data_ptr());
      }
    }
  }

  Stage(const Stage &) = delete;
  ~Stage() {
    if (params)
      migraphx_program_parameters_destroy(params);
    if (shapes)
      migraphx_program_parameter_shapes_destroy(shapes);
    if (program)
      migraphx_program_destroy(program);
  }
  void bind(const std::string &name, void *ptr) {
    const_migraphx_shape_t sh;
    mg(migraphx_program_parameter_shapes_get(&sh, shapes, name.c_str()));
    migraphx_argument_t arg = nullptr;
    mg(migraphx_argument_create(&arg, sh, ptr));
    auto s = migraphx_program_parameters_add(params, name.c_str(), arg);
    migraphx_argument_destroy(arg);
    mg(s);
  }
  at::Tensor run(hipStream_t stream) {
    migraphx_arguments_t results = nullptr;
    mg(migraphx_program_run_async(&results, program, params, stream,
                                  "ihipStream_t"));
    if (!output.defined()) {
      const_migraphx_argument_t arg;
      mg(migraphx_arguments_get(&arg, results, 0));
      char *ptr = nullptr;
      mg(migraphx_argument_buffer(&ptr, arg));
      const_migraphx_shape_t sh;
      mg(migraphx_argument_shape(&sh, arg));
      const size_t *ls;
      size_t n;
      mg(migraphx_shape_lengths(&ls, &n, sh));
      std::vector<int64_t> lens(ls, ls + n);
      const size_t *ss;
      mg(migraphx_shape_strides(&ss, &n, sh));
      std::vector<int64_t> strides(ss, ss + n);
      migraphx_shape_datatype_t ty;
      mg(migraphx_shape_type(&ty, sh));
      auto type =
          ty == migraphx_shape_float_type
              ? at::kFloat
              : (ty == migraphx_shape_bf16_type ? at::kBFloat16 : at::kHalf);
      int device = 0;
      hip(hipGetDevice(&device));
      output =
          at::from_blob(ptr, lens, strides, [](void *) {}, gpu(device, type));
    }
    migraphx_arguments_destroy(results);
    return output;
  }
};
struct Vision {
  json m;
  fs::path root;
  int device;
  bool segmented;
  std::map<std::string, at::Tensor> buffers;
  std::vector<std::unique_ptr<Stage>> stages;
  std::vector<at::Tensor> weights;
  std::vector<std::map<std::string, at::Tensor>> views;
  at::Tensor output;
  std::unique_ptr<Stage> compiled;
  Vision(const fs::path &bundle, const json &config, int dev,
         SharedMemory *shared = nullptr)
      : device(dev) {
    segmented = config.at("kind") == "segmented";
    if (!segmented) {
      compiled = std::make_unique<Stage>(
          bundle / config.at("path").get<std::string>(), device, true, shared);
      return;
    }
    root = bundle / config.at("path").get<std::string>();
    m = read_json(root / "manifest.json");
    for (auto it = m["allocs"].begin(); it != m["allocs"].end(); ++it)
      buffers[it.key()] = storage(shared, {span(it.value())}, device);
    for (auto it = m["params"].begin(); it != m["params"].end(); ++it)
      if (it.value()["parameter"] != "pixel_values")
        buffers[it.key()] =
            storage(shared, {span(it.value()["shape"])}, device);
    for (auto &s : m["stages"])
      stages.push_back(std::make_unique<Stage>(
          root / s["path"].get<std::string>(), device, false, shared));
    for (auto &e : m["events"]) {
      auto spec = m["weight_files"][e["weight"].get<std::string>()];
      weights.push_back(shared ? shared->tensor(root, spec)
                               : tensor_file(root, spec, device));
      std::map<std::string, at::Tensor> vv;
      for (auto it = e["tensors"].begin(); it != e["tensors"].end(); ++it)
        vv[it.key()] = view(it.value());
      views.push_back(vv);
    }
    output = view(m.at("output"));
  }
  at::Tensor view(const json &d) {
    auto &s = d.at("shape");
    auto t = dtype(s.at("type"));
    auto &b = buffers.at(d.at("base").get<std::string>());
    int64_t offset = d.at("offset");
    if (offset < 0 || offset + span(s) > b.numel())
      throw std::runtime_error("Out-of-range arena view");
    auto typed = b.view(t);
    // as_strided takes an offset in the underlying storage, not in this view.
    return typed.as_strided(s.at("lens").get<std::vector<int64_t>>(),
                            s.at("strides").get<std::vector<int64_t>>(),
                            typed.storage_offset() +
                                offset / c10::elementSize(t));
  }
  at::Tensor run(const at::Tensor &patches, hipStream_t stream) {
    if (!segmented) {
      compiled->bind("pixel_values", patches.data_ptr());
      return compiled->run(stream);
    }
    for (auto it = m["params"].begin(); it != m["params"].end(); ++it)
      if (it.value()["parameter"] == "pixel_values")
        buffers[it.key()] = patches;
    for (size_t i = 0; i < stages.size(); i++) {
      auto &s = m["stages"][i];
      for (auto it = s["external"].begin(); it != s["external"].end(); ++it) {
        auto &d = it.value();
        stages[i]->bind(
            it.key(), static_cast<char *>(
                          buffers.at(d["base"].get<std::string>()).data_ptr()) +
                          d["offset"].get<int64_t>());
      }
      stages[i]->run(stream);
      if (i < views.size()) {
        auto &v = views[i];
        at::Tensor x;
        if (m["events"][i]["kind"] == "attention") {
          auto q = v.at("q").transpose(1, 2), k = v.at("k").transpose(1, 2),
               value = v.at("v").transpose(1, 2);
          auto y = std::get<0>(at::_scaled_dot_product_flash_attention(
              q, k, value, 0., false, false, 1.));
          x = y.transpose(1, 2).reshape(
              {patches.size(0), patches.size(1), 768});
        } else
          x = v.at("x");
        at::matmul_out(v.at("output"), x, weights[i]);
      }
    }
    return output;
  }
};
struct Engine {
  fs::path root;
  json config;
  int device, batch, vbatch, budget, gh, gw, tokens;
  SharedMemory *shared;
  Vision vision;
  Stage backbone;
  at::Tensor embeds, mask, rgb, patches, out;
  std::unique_ptr<at::cuda::CUDAGraph> graph;
  at::Tensor host_embeds, host_attention;
  c10::cuda::CUDAStream execution_stream;
  Engine(fs::path p, int dev, SharedMemory *memory = nullptr)
      : root(std::move(p)), config(read_json(root / "bundle.json")),
        device(dev), batch(config.at("batch")),
        vbatch(config.at("vision").at("batch")), budget(config.at("budget")),
        gh(config.at("vision").at("grid")[0]),
        gw(config.at("vision").at("grid")[1]),
        tokens(config.at("vision").at("tokens")), shared(memory),
        vision(root, config.at("vision"), dev, memory),
        backbone(root / config.at("backbone").at("path").get<std::string>(),
                 dev, true, memory),
        execution_stream(memory ? memory->stream
                                : c10::cuda::getStreamFromPool(false, dev)) {
    if ((config.at("schema") != 1 && config.at("schema") != 2) || batch < 1 ||
        batch > 128 || vbatch < 1 || batch % vbatch || gh % 3 || gw % 3 ||
        tokens != gh * gw / 9)
      throw std::runtime_error("Invalid bundle dimensions");
    hipDeviceProp_t props;
    hip(hipGetDeviceProperties(&props, device));
    std::string arch = props.gcnArchName;
    arch = arch.substr(0, arch.find(':'));
    if (config.contains("gpu_arch") && config["gpu_arch"] != arch)
      throw std::runtime_error("Bundle GPU architecture mismatch");
    embeds = tensor_file(root, config.at("tensors").at("template"), device);
    mask = tensor_file(root, config.at("tensors").at("mask"), device);
    if (embeds.sizes() != at::IntArrayRef({batch, tokens + 4, 512}) ||
        mask.sizes() != at::IntArrayRef({batch, tokens + 4}))
      throw std::runtime_error("Template shape mismatch");
    if (shared) {
      host_embeds = embeds.cpu();
      host_attention = mask.cpu();
      embeds = storage(shared, {batch, tokens + 4, 512}, device, at::kFloat);
      mask = storage(shared, {batch, tokens + 4}, device,
                     host_attention.scalar_type());
    }
    patches = storage(shared, {batch, gh * gw, 768}, device, at::kFloat);
    out = storage(shared, {batch, 768}, device, at::kFloat);
    backbone.bind("inputs_embeds", embeds.data_ptr());
    backbone.bind("attention_mask", mask.data_ptr());
  }
  void execute(hipStream_t stream) {
    if (!shared)
      hip(static_cast<hipError_t>(eg2_patchify_config(
          rgb.data_ptr(), patches.data_ptr(), batch, rgb.size(1), rgb.size(2),
          gh, gw, stream,
          config.value("preprocess", json::object()).value("block", 256),
          config.value("preprocess", json::object()).value("vector", 1))));
    for (int start = 0; start < batch; start += vbatch) {
      auto media = vision.run(patches.slice(0, start, start + vbatch), stream);
      embeds.slice(0, start, start + vbatch)
          .slice(1, 2, 2 + tokens)
          .copy_(media.reshape({vbatch, tokens, 512}));
    }
    int bb = config.at("backbone").value("batch", batch);
    for (int start = 0; start < batch; start += bb) {
      backbone.bind("inputs_embeds",
                    embeds.slice(0, start, start + bb).data_ptr());
      backbone.bind("attention_mask",
                    mask.slice(0, start, start + bb).data_ptr());
      out.slice(0, start, start + bb).copy_(backbone.run(stream));
    }
  }
  void encode(const uint8_t *data, int count, int h, int w, float *output) {
    auto stream = execution_stream;
    c10::cuda::CUDAStreamGuard guard(stream);
    if (shared) {
      shared->activate(root.string());
      embeds.copy_(host_embeds);
      mask.copy_(host_attention);
      size_t bytes = size_t(batch) * h * w * 3;
      if (!shared->rgb.defined() || size_t(shared->rgb.numel()) < bytes)
        shared->rgb = at::empty({static_cast<int64_t>(bytes)}, gpu(device));
      rgb = shared->rgb.narrow(0, 0, bytes).view({batch, h, w, 3});
      rgb.zero_();
    } else if (!rgb.defined() || rgb.size(1) != h || rgb.size(2) != w) {
      graph.reset();
      rgb = at::zeros({batch, h, w, 3}, gpu(device));
    }
    hip(hipMemcpyAsync(rgb.data_ptr(), data, size_t(count) * h * w * 3,
                       hipMemcpyHostToDevice, stream.stream()));
    if (shared)
      hip(static_cast<hipError_t>(eg2_patchify_config(
          rgb.data_ptr(), patches.data_ptr(), batch, h, w, gh, gw,
          stream.stream(),
          config.value("preprocess", json::object()).value("block", 256),
          config.value("preprocess", json::object()).value("vector", 1))));
    const bool eager = std::getenv("EG2_DISABLE_GRAPHS") != nullptr;
    if (!graph && !eager) {
      {
        PoolGuard pool(shared);
        for (int i = 0; i < 3; i++)
          execute(stream.stream());
      }
      stream.synchronize();
      graph = std::make_unique<at::cuda::CUDAGraph>();
      graph->capture_begin(shared ? shared->transient.id()
                                  : at::cuda::MempoolId_t{0, 0});
      try {
        execute(stream.stream());
        graph->capture_end();
      } catch (...) {
        try {
          graph->capture_end();
        } catch (...) {
        }
        graph.reset();
        throw;
      }
    }
    if (eager) {
      PoolGuard pool(shared);
      execute(stream.stream());
    } else
      graph->replay();
    hip(hipMemcpyAsync(output, out.data_ptr(),
                       size_t(count) * 768 * sizeof(float),
                       hipMemcpyDeviceToHost, stream.stream()));
    stream.synchronize();
  }
};
struct TextEngine {
  std::unique_ptr<Stage> stage;
  std::shared_ptr<torch::jit::Module> script;
  at::Tensor script_output;
  int batch, sequence, device;
  SharedMemory *shared;
  at::Tensor ids, mask, host_ids, host_mask, embeds, embedding;
  double scale = 1.;
  bool native_embedding = false;
  std::unique_ptr<at::cuda::CUDAGraph> graph;
  TextEngine(const fs::path &root, const json &p, int dev,
             const at::Tensor &table, double embedding_scale,
             std::shared_ptr<torch::jit::Module> script_module,
             SharedMemory *memory = nullptr)
      : script(std::move(script_module)), batch(p.at("batch")),
        sequence(p.at("sequence")), device(dev), shared(memory) {
    ids = storage(shared, {batch, sequence}, dev, at::kLong);
    mask = storage(shared, {batch, sequence}, dev, at::kLong);
    host_ids =
        at::zeros({batch, sequence}, at::TensorOptions().dtype(at::kLong));
    host_mask = at::zeros_like(host_ids);
    if (script) {
      script_output = storage(shared, {batch, 768}, dev, at::kFloat);
      return;
    }
    stage =
        std::make_unique<Stage>(root / p.at("path").get<std::string>(), dev);
    native_embedding = p.value("embedding", std::string()) == "native";
    if (native_embedding) {
      embedding = table;
      scale = embedding_scale;
      embeds = at::empty({batch, sequence, 512}, gpu(dev, at::kFloat));
      stage->bind("inputs_embeds", embeds.data_ptr());
    } else
      stage->bind("input_ids", ids.data_ptr());
    stage->bind("attention_mask", mask.data_ptr());
  }
  void encode(const int64_t *input, const int64_t *attention, int count,
              int length, float *out) {
    auto stream =
        shared ? shared->stream : c10::cuda::getStreamFromPool(false, device);
    c10::cuda::CUDAStreamGuard guard(stream);
    host_ids.zero_();
    host_mask.zero_();
    for (int i = 0; i < count; i++) {
      std::memcpy(host_ids.data_ptr<int64_t>() + i * sequence,
                  input + i * length, length * 8);
      std::memcpy(host_mask.data_ptr<int64_t>() + i * sequence,
                  attention + i * length, length * 8);
    }
    ids.copy_(host_ids);
    mask.copy_(host_mask);
    if (native_embedding)
      embeds.copy_(at::embedding(embedding, ids) * scale);
    torch::jit::GraphOptimizerEnabledGuard jit_guard(false);
    auto execute = [&]() {
      if (script)
        script_output.copy_(script->forward({ids, mask}).toTensor());
      else
        stage->run(stream.stream());
    };
    if (script) {
      PoolGuard pool(shared);
      if (shared)
        shared->activate("text:" + std::to_string(batch) + ":" +
                         std::to_string(sequence));
      execute();
    } else {
      if (!graph) {
        for (int i = 0; i < 3; i++)
          execute();
        stream.synchronize();
        graph = std::make_unique<at::cuda::CUDAGraph>();
        graph->capture_begin();
        try {
          execute();
          graph->capture_end();
        } catch (...) {
          try {
            graph->capture_end();
          } catch (...) {
          }
          graph.reset();
          throw;
        }
      }
      graph->replay();
    }
    hip(hipMemcpyAsync(out, (script ? script_output : stage->output).data_ptr(),
                       size_t(count) * 768 * 4, hipMemcpyDeviceToHost,
                       stream.stream()));
    stream.synchronize();
  }
};
void error_message(char *out, size_t cap, const char *msg) {
  if (out && cap) {
    std::strncpy(out, msg, cap - 1);
    out[cap - 1] = 0;
  }
}
} // namespace
struct eg2_context {
  fs::path root;
  json index;
  json last;
  at::Tensor text_embedding;
  int device;
  std::mutex mutex;
  std::unique_ptr<SharedMemory> shared;
  std::map<std::string, std::unique_ptr<Engine>> engines;
  std::map<std::string, std::unique_ptr<TextEngine>> text_engines;
  std::map<std::string, std::shared_ptr<torch::jit::Module>> script_modules;
  void *tokenizer = nullptr;
  ~eg2_context() {
    if (tokenizer)
      eg2_tokenizer_free(tokenizer);
  }
  eg2_context(fs::path p, int dev) : root(std::move(p)), device(dev) {
    if (fs::exists(root / "index.json"))
      index = read_json(root / "index.json");
    else {
      auto b = read_json(root / "bundle.json");
      index = {{"schema", 1},
               {"profiles",
                json::array(
                    {{{"path", "."},
                      {"budget", b.at("budget")},
                      {"batch", b.at("batch")},
                      {"aspect", b.value("aspect", json::array({16, 9}))}}})}};
    }
    if ((index.value("schema", 0) != 1 && index.value("schema", 0) != 2) ||
        !index.value("profiles", json::array()).is_array())
      throw std::runtime_error("Invalid bundle index schema");
    hipDeviceProp_t props{};
    hip(hipGetDeviceProperties(&props, device));
    std::string arch = props.gcnArchName;
    arch = arch.substr(0, arch.find(':'));
    auto relative = [](const std::string &name) {
      fs::path p(name);
      if (p.empty() || p.is_absolute())
        throw std::runtime_error("Bundle paths must be relative");
      for (auto &part : p)
        if (part == "..")
          throw std::runtime_error("Bundle paths cannot contain '..'");
    };
    for (auto &p : index.value("profiles", json::array())) {
      relative(p.at("path"));
      int b = p.at("batch"), budget = p.at("budget");
      auto aspect = p.value("aspect", json::array({16, 9}));
      if (b < 1 || b > 128 ||
          (budget != 70 && budget != 140 && budget != 280) ||
          (!aspect.is_null() &&
           (!aspect.is_array() || aspect.size() != 2 ||
            aspect[0].get<int>() < 1 || aspect[1].get<int>() < 1)))
        throw std::runtime_error("Invalid image profile");
      auto bundle =
          read_json(root / p.at("path").get<std::string>() / "bundle.json");
      if (bundle.value("gpu_arch", arch) != arch)
        throw std::runtime_error(
            "Bundle GPU architecture mismatch; recompile on this GPU");
      if (bundle.at("batch") != b || bundle.at("budget") != budget)
        throw std::runtime_error("Index and bundle dimensions disagree");
    }
    for (auto &p : index.value("text_profiles", json::array())) {
      relative(p.at("path"));
      int b = p.at("batch"), l = p.at("sequence");
      if (b < 1 || b > 128 || l < 2 || l > 8192)
        throw std::runtime_error("Invalid text profile");
    }
    if (index.contains("tokenizer"))
      relative(index.at("tokenizer"));
    if (index.value("schema", 0) == 2) {
      for (auto &entry : index.at("shared").at("weights"))
        relative(entry.at("file"));
      shared = std::make_unique<SharedMemory>(root, index.at("shared"), device);
    }
  }
};
extern "C" {
const char *eg2_version() { return "0.1.0"; }
eg2_status eg2_device_info(int device, char *out, size_t cap) {
  try {
    hipDeviceProp_t p{};
    hip(hipGetDeviceProperties(&p, device));
    json data = {{"arch", p.gcnArchName},
                 {"name", p.name},
                 {"wave_size", p.warpSize},
                 {"compute_units", p.multiProcessorCount},
                 {"max_threads_per_block", p.maxThreadsPerBlock},
                 {"shared_memory_per_block", p.sharedMemPerBlock},
                 {"total_memory", p.totalGlobalMem}};
    auto text = data.dump();
    if (!out || cap <= text.size())
      return EG2_INVALID_ARGUMENT;
    std::memcpy(out, text.c_str(), text.size() + 1);
    return EG2_OK;
  } catch (const std::exception &e) {
    error_message(out, cap, e.what());
    return EG2_RUNTIME_ERROR;
  }
}
eg2_status eg2_create(const char *path, int device, eg2_context **out,
                      char *err, size_t cap) {
  if (!out || !path || device < 0) {
    error_message(err, cap, "Invalid create arguments");
    return EG2_INVALID_ARGUMENT;
  }
  *out = nullptr;
  try {
    c10::InferenceMode inference;
    hip(hipSetDevice(device));
    // MemPool requires LibTorch allocator initialization even in a C-only host.
    at::globalContext().lazyInitDevice(at::kCUDA);
    at::set_num_threads(4);
    *out = new eg2_context(path, device);
    return EG2_OK;
  } catch (const std::exception &e) {
    error_message(err, cap, e.what());
    return EG2_RUNTIME_ERROR;
  }
}
void eg2_destroy(eg2_context *p) {
  if (!p)
    return;
  try {
    c10::cuda::CUDAGuard guard(p->device);
    delete p;
  } catch (...) { /* Destructors must not throw across the C ABI. */
  }
}
eg2_status eg2_encode_images(eg2_context *ctx, const uint8_t *data,
                             size_t batch, size_t h, size_t w, int budget,
                             float *out, size_t elements, char *err,
                             size_t cap) {
  if (!ctx || !data || !out || batch < 1 || batch > 128 || h < 1 || w < 1 ||
      h > 32768 || w > 32768 || elements < batch * 768 ||
      (budget != 70 && budget != 140 && budget != 280)) {
    error_message(err, cap, "Invalid image arguments");
    return EG2_INVALID_ARGUMENT;
  }
  if (budget == 280 && batch > 32) {
    error_message(err, cap,
                  "Image budget 280 supports at most 32 images per request; "
                  "larger requests failed stability validation");
    return EG2_UNSUPPORTED;
  }
  try {
    std::lock_guard<std::mutex> lock(ctx->mutex);
    c10::InferenceMode inference;
    c10::cuda::CUDAGuard guard(ctx->device);
    json selected;
    size_t best = 129;
    for (auto &p : ctx->index.at("profiles")) {
      auto aspect = p.value("aspect", json::array({16, 9}));
      bool matches = aspect.is_null() || (w * aspect[1].get<size_t>() ==
                                          h * aspect[0].get<size_t>());
      size_t b = p.at("batch");
      if (p.at("budget") == budget && b >= batch && b < best && matches) {
        selected = p;
        best = b;
      }
    }
    if (selected.is_null()) {
      best = 0;
      for (auto &p : ctx->index.at("profiles")) {
        auto aspect = p.value("aspect", json::array({16, 9}));
        bool matches = aspect.is_null() || (w * aspect[1].get<size_t>() ==
                                            h * aspect[0].get<size_t>());
        size_t b = p.at("batch");
        if (p.at("budget") == budget && matches && b > best) {
          selected = p;
          best = b;
        }
      }
    }
    if (selected.is_null()) {
      error_message(err, cap,
                    "No prepared profile for this budget and aspect ratio; run "
                    "eg2 prepare");
      return EG2_UNSUPPORTED;
    }
    std::string key = selected.at("path");
    if (!ctx->shared && !ctx->engines.count(key))
      ctx->engines.clear();
    auto &engine = ctx->engines[key];
    if (!engine) {
      if (ctx->shared)
        ctx->shared->reset();
      engine = std::make_unique<Engine>(ctx->root / key, ctx->device,
                                        ctx->shared.get());
    }
    for (size_t start = 0; start < batch; start += best)
      engine->encode(data + start * h * w * 3, std::min(best, batch - start), h,
                     w, out + start * 768);
    ctx->last = {
        {"modality", "image"},
        {"requested_batch", batch},
        {"execution_batch", best},
        {"vision_batch", engine->vbatch},
        {"chunks", (batch + best - 1) / best},
        {"image_tokens", engine->tokens},
        {"sequence", engine->tokens + 4},
        {"budget", budget},
        {"vision_kind", engine->config.at("vision").at("kind")},
        {"fingerprint", engine->config.value("fingerprint", json::object())}};
    return EG2_OK;
  } catch (const std::exception &e) {
    error_message(err, cap, e.what());
    return EG2_RUNTIME_ERROR;
  }
}
eg2_status eg2_encode_tokens(eg2_context *ctx, const int64_t *ids,
                             const int64_t *mask, size_t count, size_t length,
                             float *out, size_t elements, char *err,
                             size_t cap) {
  if (!ctx || !ids || !mask || !out || count < 1 || count > 128 || length < 1 ||
      length > 8192 || elements < count * 768) {
    error_message(err, cap, "Invalid token arguments");
    return EG2_INVALID_ARGUMENT;
  }
  for (size_t i = 0; i < count; i++) {
    bool active = false;
    for (size_t j = 0; j < length; j++) {
      auto k = i * length + j;
      if (ids[k] < 0 || ids[k] >= 262144 || mask[k] < 0 || mask[k] > 1) {
        error_message(err, cap, "Invalid token id or mask");
        return EG2_INVALID_ARGUMENT;
      }
      active |= mask[k] != 0;
    }
    if (!active) {
      error_message(err, cap, "Empty attention mask");
      return EG2_INVALID_ARGUMENT;
    }
  }
  try {
    std::lock_guard<std::mutex> lock(ctx->mutex);
    c10::InferenceMode inference;
    c10::cuda::CUDAGuard guard(ctx->device);
    json selected;
    size_t cost = SIZE_MAX;
    for (auto &p : ctx->index.value("text_profiles", json::array())) {
      size_t b = p.at("batch"), l = p.at("sequence");
      if (b >= count && l >= length && b * l < cost) {
        selected = p;
        cost = b * l;
      }
    }
    if (selected.is_null()) {
      size_t best_batch = 0;
      for (auto &p : ctx->index.value("text_profiles", json::array())) {
        size_t b = p.at("batch"), l = p.at("sequence");
        if (l >= length &&
            (b > best_batch || (b == best_batch && b * l < cost))) {
          selected = p;
          best_batch = b;
          cost = b * l;
        }
      }
    }
    if (selected.is_null()) {
      error_message(err, cap,
                    "No prepared text profile for this sequence length; run "
                    "eg2 prepare-text");
      return EG2_UNSUPPORTED;
    }
    std::string path = selected.at("path");
    std::string key = path + "|" +
                      std::to_string(selected.at("batch").get<int>()) + "|" +
                      std::to_string(selected.at("sequence").get<int>());
    if (!ctx->shared && !ctx->text_engines.count(key)) {
      ctx->text_engines.clear();
      if (!ctx->script_modules.count(path))
        ctx->script_modules.clear();
    }
    auto &engine = ctx->text_engines[key];
    if (!engine)
      engine = ([&]() {
        if (ctx->shared)
          ctx->shared->reset();
        if (selected.value("embedding", std::string()) == "native" &&
            !ctx->text_embedding.defined())
          ctx->text_embedding = tensor_file(
              ctx->root, ctx->index.at("text_embedding"), ctx->device);
        std::shared_ptr<torch::jit::Module> module;
        if (selected.value("kind", std::string()) == "torchscript") {
          auto &cached = ctx->script_modules[path];
          if (!cached) {
            cached = std::make_shared<torch::jit::Module>(torch::jit::load(
                (ctx->root / path).string(),
                at::Device(ctx->shared ? at::kCPU : at::kCUDA, ctx->device)));
            if (ctx->shared) {
              auto specs = ctx->index.at("shared").at("text_weights").at(path);
              for (auto item : cached->named_parameters(true)) {
                auto spec = specs.at("parameters").at(item.name);
                auto value = ctx->shared->weight(spec.at("weight"))
                                 .view(item.value.scalar_type())
                                 .view(item.value.sizes());
                item.value.set_data(value);
              }
              for (auto item : cached->named_buffers(true)) {
                auto spec = specs.at("buffers").at(item.name);
                auto value = ctx->shared->weight(spec.at("weight"))
                                 .view(item.value.scalar_type())
                                 .view(item.value.sizes());
                item.value.set_data(value);
              }
            }
            cached->eval();
          }
          module = cached;
        }
        return std::make_unique<TextEngine>(
            ctx->root, selected, ctx->device, ctx->text_embedding,
            ctx->index.value("text_embedding", json::object())
                .value("scale", 1.),
            module, ctx->shared.get());
      })();
    size_t b = selected.at("batch");
    for (size_t start = 0; start < count; start += b)
      engine->encode(ids + start * length, mask + start * length,
                     std::min(b, count - start), length, out + start * 768);
    ctx->last = {{"modality", "text"},
                 {"requested_batch", count},
                 {"execution_batch", b},
                 {"chunks", (count + b - 1) / b},
                 {"sequence", selected.at("sequence")}};
    return EG2_OK;
  } catch (const std::exception &e) {
    error_message(err, cap, e.what());
    return EG2_RUNTIME_ERROR;
  }
}
eg2_status eg2_tokenize(eg2_context *ctx, const char *const *texts,
                        size_t count, const char *prefix, int64_t *ids,
                        int64_t *mask, size_t slots, size_t *lengths, char *err,
                        size_t cap) {
  if (!ctx || !texts || !ids || !mask || !lengths || count < 1 || count > 128 ||
      slots < 1 || slots > 8192) {
    error_message(err, cap, "Invalid tokenizer arguments");
    return EG2_INVALID_ARGUMENT;
  }
  try {
    std::lock_guard<std::mutex> lock(ctx->mutex);
    if (!ctx->tokenizer) {
      if (!ctx->index.contains("tokenizer")) {
        error_message(err, cap, "Bundle has no tokenizer");
        return EG2_UNSUPPORTED;
      }
      auto path = ctx->root / ctx->index["tokenizer"].get<std::string>();
      ctx->tokenizer = eg2_tokenizer_new(path.c_str(), err, cap);
      if (!ctx->tokenizer)
        return EG2_RUNTIME_ERROR;
    }
    return eg2_tokenizer_encode(ctx->tokenizer, texts, count, prefix, ids, mask,
                                slots, lengths, err, cap) == 0
               ? EG2_OK
               : EG2_INVALID_ARGUMENT;
  } catch (const std::exception &e) {
    error_message(err, cap, e.what());
    return EG2_RUNTIME_ERROR;
  }
}
eg2_status eg2_encode_texts(eg2_context *ctx, const char *const *texts,
                            size_t count, const char *prefix, float *out,
                            size_t elements, char *err, size_t cap) {
  if (count < 1 || count > 128) {
    error_message(err, cap, "Invalid text count");
    return EG2_INVALID_ARGUMENT;
  }
  try {
    std::vector<int64_t> ids(count * 8192), mask(count * 8192);
    std::vector<size_t> lengths(count);
    auto status = eg2_tokenize(ctx, texts, count, prefix, ids.data(),
                               mask.data(), 8192, lengths.data(), err, cap);
    if (status)
      return status;
    size_t length = *std::max_element(lengths.begin(), lengths.end());
    std::vector<int64_t> compact_ids(count * length),
        compact_mask(count * length);
    for (size_t i = 0; i < count; i++) {
      std::copy_n(ids.data() + i * 8192, length,
                  compact_ids.data() + i * length);
      std::copy_n(mask.data() + i * 8192, length,
                  compact_mask.data() + i * length);
    }
    return eg2_encode_tokens(ctx, compact_ids.data(), compact_mask.data(),
                             count, length, out, elements, err, cap);
  } catch (const std::exception &e) {
    error_message(err, cap, e.what());
    return EG2_RUNTIME_ERROR;
  }
}
eg2_status eg2_prepare(eg2_context *ctx, char *err, size_t cap) {
  if (!ctx) {
    error_message(err, cap, "Invalid context");
    return EG2_INVALID_ARGUMENT;
  }
  try {
    for (auto &p : ctx->index.value("profiles", json::array())) {
      size_t b = p.at("batch");
      int budget = p.at("budget");
      if (budget == 280 && b > 32)
        continue;
      auto aspect = p.value("aspect", json::array({16, 9}));
      if (aspect.is_null())
        aspect = json::array({16, 9});
      size_t w = aspect[0].get<size_t>(), h = aspect[1].get<size_t>();
      size_t scale = std::min(
          size_t(16), std::max(size_t(1), size_t(512) / std::max(w, h)));
      w *= scale;
      h *= scale;
      std::vector<uint8_t> input(b * h * w * 3, 127);
      std::vector<float> output(b * 768);
      auto status = eg2_encode_images(ctx, input.data(), b, h, w, budget,
                                      output.data(), output.size(), err, cap);
      if (status != EG2_OK)
        return status;
    }
    for (auto &p : ctx->index.value("text_profiles", json::array())) {
      size_t b = p.at("batch"), l = p.at("sequence");
      std::vector<int64_t> ids(b * l, 1), mask(b * l, 1);
      std::vector<float> output(b * 768);
      auto status = eg2_encode_tokens(ctx, ids.data(), mask.data(), b, l,
                                      output.data(), output.size(), err, cap);
      if (status != EG2_OK)
        return status;
    }
    return EG2_OK;
  } catch (const std::exception &e) {
    error_message(err, cap, e.what());
    return EG2_RUNTIME_ERROR;
  }
}
eg2_status eg2_info(eg2_context *ctx, char *out, size_t cap) {
  if (!ctx || !out)
    return EG2_INVALID_ARGUMENT;
  try {
    std::lock_guard<std::mutex> lock(ctx->mutex);
    auto info = ctx->index;
    info["last_execution"] = ctx->last;
    info.erase("shared");
    if (ctx->shared) {
      auto &m = *ctx->shared;
      size_t transient_reserved = 0, transient_allocated = 0;
      for (const auto &segment :
           c10::cuda::CUDACachingAllocator::snapshot().segments)
        if (segment.device == ctx->device &&
            segment.owner_private_pool_id == m.transient.id()) {
          transient_reserved += segment.total_size;
          transient_allocated += segment.allocated_size;
        }
      info["memory"] = {
          {"mode", "shared"},
          {"weight_bytes", m.weights_bytes},
          {"transient_reserved_bytes", transient_reserved},
          {"transient_allocated_bytes", transient_allocated},
          {"weight_allocations", m.weights.size()},
          {"workspace_bytes", m.arena.nbytes()},
          {"rgb_capacity_bytes", m.rgb.defined() ? m.rgb.nbytes() : 0},
          {"profile_switches", m.switches},
          {"resident_image_profiles", ctx->engines.size()},
          {"resident_text_profiles", ctx->text_engines.size()},
          {"resident_text_modules", ctx->script_modules.size()}};
    } else
      info["memory"] = {{"mode", "legacy"}};
    hipDeviceProp_t props{};
    if (hipGetDeviceProperties(&props, ctx->device) == hipSuccess)
      info["gpu"] = {{"name", props.name}, {"arch", props.gcnArchName}};
    size_t free_bytes = 0, total_bytes = 0;
    if (hipMemGetInfo(&free_bytes, &total_bytes) == hipSuccess)
      info["device_memory"] = {
          {"free_bytes", free_bytes},
          {"total_bytes", total_bytes},
          {"scope", "whole device, including other processes"}};
    int version = 0;
    if (hipRuntimeGetVersion(&version) == hipSuccess)
      info["hip_runtime_version"] = version;
    auto s = info.dump();
    if (cap <= s.size())
      return EG2_INVALID_ARGUMENT;
    std::memcpy(out, s.c_str(), s.size() + 1);
    return EG2_OK;
  } catch (...) {
    return EG2_RUNTIME_ERROR;
  }
}
}
