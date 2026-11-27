#include "ggml_cpu_backend.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef LIZARD_HAVE_GGML_CPU_BACKEND
#include <ggml-backend.h>
#include <ggml-cpu.h>
#include <ggml.h>
#include <repack.h>
#endif

namespace lizard_native {
namespace {

std::atomic<std::uint64_t> g_matvec_calls {0};
std::atomic<std::uint64_t> g_matvec_failures {0};
std::atomic<std::uint64_t> g_plan_hits {0};
std::atomic<std::uint64_t> g_plan_misses {0};
std::atomic<std::uint64_t> g_matvec_nanoseconds {0};
std::atomic<std::uint64_t> g_weight_bytes {0};
std::atomic<std::uint64_t> g_repacked_bytes {0};
std::atomic<std::uint64_t> g_repacked_tensors {0};
std::atomic<std::uint64_t> g_repack_failures {0};
std::atomic<std::uint64_t> g_grouped_graph_calls {0};
std::atomic<std::uint64_t> g_grouped_graph_operations {0};
std::atomic<std::uint64_t> g_fused_ffn_calls {0};
constexpr std::uint64_t k_default_repack_budget_bytes = 320ull * 1024ull * 1024ull;

std::uint64_t repack_budget_bytes() {
  static const std::uint64_t budget = [] {
    const char *value = std::getenv("LIZARD_GGML_REPACK_MB");
    if (value == nullptr || *value == '\0') {
      // Qualified Run 06 target-host default. This cap repacks 42 hot Q4_K
      // tensors on Llama 3.2 3B while remaining below the matched llama.cpp
      // working-set and private-byte envelopes. Set 0 for a no-repack A/B.
      return k_default_repack_budget_bytes;
    }
    char *end = nullptr;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    if (end == value || *end != '\0' || parsed > 4096ull) {
      return std::numeric_limits<std::uint64_t>::max();
    }
    return static_cast<std::uint64_t>(parsed) * 1024ull * 1024ull;
  }();
  return budget;
}

int grouped_graph_mode() {
  static const int mode = [] {
    const char *value = std::getenv("LIZARD_GGML_GROUPED_GRAPHS");
    if (value == nullptr || *value == '\0' || std::strcmp(value, "1") == 0) {
      // Accepted Phase 0c base. Set 0 for a same-binary Phase 0b rollback.
      return 1;
    }
    if (std::strcmp(value, "0") == 0) {
      return 0;
    }
    return -1;
  }();
  return mode;
}

int fused_ffn_mode() {
  static const int mode = [] {
    const char *value = std::getenv("LIZARD_GGML_FUSED_FFN");
    if (value == nullptr || *value == '\0' || std::strcmp(value, "0") == 0) {
      return 0;
    }
    if (std::strcmp(value, "1") == 0) {
      return 1;
    }
    return -1;
  }();
  return mode;
}

const std::string &normalized_backend_request() {
  static const std::string request = [] {
    const char *value = std::getenv("LIZARD_NATIVE_CPU_BACKEND");
    if (value == nullptr || *value == '\0') {
#ifdef LIZARD_HAVE_GGML_CPU_BACKEND
      return std::string("ggml");
#else
      return std::string("native");
#endif
    }
    std::string normalized(value);
    std::transform(normalized.begin(), normalized.end(), normalized.begin(), [](unsigned char item) {
      return static_cast<char>(std::tolower(item));
    });
    return normalized;
  }();
  return request;
}

#ifdef LIZARD_HAVE_GGML_CPU_BACKEND

class PinnedGgmlCpuState {
public:
  ~PinnedGgmlCpuState() {
    if (threadpool_ != nullptr) {
      ggml_threadpool_free(threadpool_);
    }
  }

  void reset() {
    std::lock_guard<std::mutex> guard(mutex_);
    plans_.clear();
    multi_plans_.clear();
    ffn_plans_.clear();
  }

  bool matvec(
    std::uint32_t type_value,
    std::uint64_t row_count,
    std::uint64_t row_width,
    const std::uint8_t *weight_data,
    std::size_t weight_bytes,
    const float *input,
    std::size_t input_count,
    std::uint64_t thread_count,
    std::vector<float> *output,
    std::string *error) {
    std::lock_guard<std::mutex> guard(mutex_);
    if (weight_data == nullptr || input == nullptr || output == nullptr) {
      return fail("ggml CPU matvec received a null input or output.", error);
    }
    if (row_count == 0 || row_width == 0 || input_count != row_width) {
      return fail("ggml CPU matvec received an invalid matrix shape.", error);
    }
    if (row_count > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())
      || row_width > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
      return fail("ggml CPU matvec shape exceeds ggml's signed dimension range.", error);
    }

    const ggml_type type = static_cast<ggml_type>(type_value);
    if (type < GGML_TYPE_F32 || type >= GGML_TYPE_COUNT) {
      return fail("ggml CPU matvec does not recognize the tensor type.", error);
    }
    const std::size_t row_bytes = ggml_row_size(type, static_cast<std::int64_t>(row_width));
    if (row_bytes == 0
      || row_count > std::numeric_limits<std::size_t>::max() / row_bytes
      || row_bytes * static_cast<std::size_t>(row_count) != weight_bytes) {
      return fail("ggml CPU matvec tensor bytes do not match the advertised shape and type.", error);
    }

    const int threads = static_cast<int>(std::clamp<std::uint64_t>(thread_count, 1, 64));
    if (!ensure_threadpool(threads, error)) {
      return false;
    }

    const PlanKey key {weight_data, type_value, row_count, row_width, weight_bytes};
    auto found = plans_.find(key);
    if (found == plans_.end()) {
      std::unique_ptr<CachedPlan> created = create_plan(key, type, threads, error);
      if (!created) {
        return false;
      }
      found = plans_.emplace(key, std::move(created)).first;
      g_plan_misses.fetch_add(1, std::memory_order_relaxed);
    } else {
      g_plan_hits.fetch_add(1, std::memory_order_relaxed);
    }
    CachedPlan &cached = *found->second;
    output->assign(static_cast<std::size_t>(row_count), 0.0f);
    if (cached.weight_buffer == nullptr) {
      cached.weights->data = const_cast<std::uint8_t *>(weight_data);
    }
    cached.activation->data = const_cast<float *>(input);
    cached.result->data = output->data();
    const ggml_status status = ggml_graph_compute(cached.graph, &cached.plan);
    if (status != GGML_STATUS_SUCCESS) {
      return fail("ggml_graph_compute failed with status " + std::to_string(static_cast<int>(status)) + ".", error);
    }
    for (float value : *output) {
      if (!std::isfinite(value)) {
        return fail("ggml CPU matvec produced a non-finite result.", error);
      }
    }
    return true;
  }

  bool multi_matvec(
    const GgmlCpuMatvecOperation *operations,
    std::size_t operation_count,
    const float *input,
    std::size_t input_count,
    std::uint64_t thread_count,
    std::string *error) {
    std::lock_guard<std::mutex> guard(mutex_);
    if (operations == nullptr || input == nullptr || operation_count < 2 || operation_count > 4) {
      return fail("ggml grouped CPU matvec requires two through four operations and a non-null input.", error);
    }

    MultiPlanKey key;
    key.operation_count = operation_count;
    for (std::size_t index = 0; index < operation_count; index += 1) {
      const GgmlCpuMatvecOperation &operation = operations[index];
      if (operation.weight_data == nullptr || operation.output == nullptr) {
        return fail("ggml grouped CPU matvec received a null weight or output.", error);
      }
      if (operation.row_count == 0 || operation.row_width == 0 || operation.row_width != input_count) {
        return fail("ggml grouped CPU matvec received an invalid or non-shared input shape.", error);
      }
      if (operation.row_count > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())
        || operation.row_width > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
        return fail("ggml grouped CPU matvec shape exceeds ggml's signed dimension range.", error);
      }
      const ggml_type type = static_cast<ggml_type>(operation.ggml_type);
      if (type < GGML_TYPE_F32 || type >= GGML_TYPE_COUNT) {
        return fail("ggml grouped CPU matvec does not recognize a tensor type.", error);
      }
      const std::size_t row_bytes = ggml_row_size(type, static_cast<std::int64_t>(operation.row_width));
      if (row_bytes == 0
        || operation.row_count > std::numeric_limits<std::size_t>::max() / row_bytes
        || row_bytes * static_cast<std::size_t>(operation.row_count) != operation.weight_bytes) {
        return fail("ggml grouped CPU matvec tensor bytes do not match a shape and type.", error);
      }
      key.operations[index] = PlanKey {
        operation.weight_data,
        operation.ggml_type,
        operation.row_count,
        operation.row_width,
        operation.weight_bytes,
      };
    }

    const int threads = static_cast<int>(std::clamp<std::uint64_t>(thread_count, 1, 64));
    if (!ensure_threadpool(threads, error)) {
      return false;
    }
    auto found = multi_plans_.find(key);
    if (found == multi_plans_.end()) {
      std::unique_ptr<CachedMultiPlan> created = create_multi_plan(key, threads, error);
      if (!created) {
        return false;
      }
      found = multi_plans_.emplace(key, std::move(created)).first;
      g_plan_misses.fetch_add(operation_count, std::memory_order_relaxed);
    } else {
      g_plan_hits.fetch_add(operation_count, std::memory_order_relaxed);
    }

    CachedMultiPlan &cached = *found->second;
    cached.activation->data = const_cast<float *>(input);
    for (std::size_t index = 0; index < operation_count; index += 1) {
      const GgmlCpuMatvecOperation &operation = operations[index];
      operation.output->assign(static_cast<std::size_t>(operation.row_count), 0.0f);
      if (cached.weight_buffers[index] == nullptr) {
        cached.weights[index]->data = const_cast<std::uint8_t *>(operation.weight_data);
      }
      cached.results[index]->data = operation.output->data();
    }
    const ggml_status status = ggml_graph_compute(cached.graph, &cached.plan);
    if (status != GGML_STATUS_SUCCESS) {
      return fail("ggml grouped graph compute failed with status "
        + std::to_string(static_cast<int>(status)) + ".", error);
    }
    for (std::size_t index = 0; index < operation_count; index += 1) {
      for (float value : *operations[index].output) {
        if (!std::isfinite(value)) {
          return fail("ggml grouped CPU matvec produced a non-finite result.", error);
        }
      }
    }
    return true;
  }

  bool ffn_swiglu(
    const GgmlCpuWeight &gate,
    const GgmlCpuWeight &up,
    const GgmlCpuWeight &down,
    const float *input,
    std::size_t input_count,
    std::uint64_t thread_count,
    std::vector<float> *output,
    std::string *error) {
    std::lock_guard<std::mutex> guard(mutex_);
    if (input == nullptr || output == nullptr) {
      return fail("ggml fused FFN received a null input or output.", error);
    }
    const auto validate_weight = [&](const GgmlCpuWeight &weight, std::uint64_t expected_width) {
      if (weight.weight_data == nullptr
        || weight.row_count == 0
        || weight.row_width != expected_width
        || weight.row_count > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())
        || weight.row_width > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
        return false;
      }
      const ggml_type type = static_cast<ggml_type>(weight.ggml_type);
      if (type < GGML_TYPE_F32 || type >= GGML_TYPE_COUNT) {
        return false;
      }
      const std::size_t row_bytes = ggml_row_size(type, static_cast<std::int64_t>(weight.row_width));
      return row_bytes > 0
        && weight.row_count <= std::numeric_limits<std::size_t>::max() / row_bytes
        && row_bytes * static_cast<std::size_t>(weight.row_count) == weight.weight_bytes;
    };
    if (!validate_weight(gate, input_count)
      || !validate_weight(up, input_count)
      || gate.row_count != up.row_count
      || !validate_weight(down, gate.row_count)) {
      return fail("ggml fused FFN tensor shapes or bytes are inconsistent.", error);
    }

    MultiPlanKey key;
    key.operation_count = 3;
    const GgmlCpuWeight weights[3] = {gate, up, down};
    for (std::size_t index = 0; index < 3; index += 1) {
      key.operations[index] = PlanKey {
        weights[index].weight_data,
        weights[index].ggml_type,
        weights[index].row_count,
        weights[index].row_width,
        weights[index].weight_bytes,
      };
    }
    const int threads = static_cast<int>(std::clamp<std::uint64_t>(thread_count, 1, 64));
    if (!ensure_threadpool(threads, error)) {
      return false;
    }
    auto found = ffn_plans_.find(key);
    if (found == ffn_plans_.end()) {
      std::unique_ptr<CachedFfnPlan> created = create_ffn_plan(key, threads, error);
      if (!created) {
        return false;
      }
      found = ffn_plans_.emplace(key, std::move(created)).first;
      g_plan_misses.fetch_add(3, std::memory_order_relaxed);
    } else {
      g_plan_hits.fetch_add(3, std::memory_order_relaxed);
    }
    CachedFfnPlan &cached = *found->second;
    cached.activation->data = const_cast<float *>(input);
    for (std::size_t index = 0; index < 3; index += 1) {
      if (cached.weight_buffers[index] == nullptr) {
        cached.weights[index]->data = const_cast<std::uint8_t *>(weights[index].weight_data);
      }
    }
    cached.gate_result->data = cached.gate_scratch.data();
    cached.up_result->data = cached.up_scratch.data();
    cached.gate_activated->data = cached.gate_scratch.data();
    cached.product->data = cached.gate_scratch.data();
    output->assign(static_cast<std::size_t>(down.row_count), 0.0f);
    cached.result->data = output->data();
    const ggml_status status = ggml_graph_compute(cached.graph, &cached.plan);
    if (status != GGML_STATUS_SUCCESS) {
      return fail("ggml fused FFN graph compute failed with status "
        + std::to_string(static_cast<int>(status)) + ".", error);
    }
    for (float value : *output) {
      if (!std::isfinite(value)) {
        return fail("ggml fused FFN produced a non-finite result.", error);
      }
    }
    return true;
  }

private:
  struct PlanKey {
    const std::uint8_t *weight_data = nullptr;
    std::uint32_t type = 0;
    std::uint64_t row_count = 0;
    std::uint64_t row_width = 0;
    std::size_t weight_bytes = 0;

    bool operator==(const PlanKey &other) const {
      return weight_data == other.weight_data
        && type == other.type
        && row_count == other.row_count
        && row_width == other.row_width
        && weight_bytes == other.weight_bytes;
    }
  };

  struct PlanKeyHash {
    std::size_t operator()(const PlanKey &key) const {
      std::size_t hash = std::hash<const void *>{}(key.weight_data);
      hash ^= std::hash<std::uint64_t>{}(key.row_count + 0x9e3779b97f4a7c15ull + (hash << 6u) + (hash >> 2u));
      hash ^= std::hash<std::uint64_t>{}(key.row_width + (static_cast<std::uint64_t>(key.type) << 32u));
      hash ^= std::hash<std::size_t>{}(key.weight_bytes);
      return hash;
    }
  };

  struct MultiPlanKey {
    std::array<PlanKey, 4> operations {};
    std::size_t operation_count = 0;

    bool operator==(const MultiPlanKey &other) const {
      return operation_count == other.operation_count && operations == other.operations;
    }
  };

  struct MultiPlanKeyHash {
    std::size_t operator()(const MultiPlanKey &key) const {
      std::size_t hash = key.operation_count;
      const PlanKeyHash operation_hash;
      for (std::size_t index = 0; index < key.operation_count; index += 1) {
        hash ^= operation_hash(key.operations[index])
          + 0x9e3779b97f4a7c15ull + (hash << 6u) + (hash >> 2u);
      }
      return hash;
    }
  };

  struct CachedPlan {
    ~CachedPlan() {
      if (weight_buffer != nullptr) {
        ggml_backend_buffer_free(weight_buffer);
      }
      if (context != nullptr) {
        ggml_free(context);
      }
    }
    std::vector<std::uint8_t> metadata;
    ggml_context *context = nullptr;
    ggml_tensor *weights = nullptr;
    ggml_tensor *activation = nullptr;
    ggml_tensor *result = nullptr;
    ggml_backend_buffer_t weight_buffer = nullptr;
    ggml_cgraph *graph = nullptr;
    ggml_cplan plan {};
    std::vector<std::uint8_t> work;
  };

  struct CachedMultiPlan {
    ~CachedMultiPlan() {
      for (ggml_backend_buffer_t buffer : weight_buffers) {
        if (buffer != nullptr) {
          ggml_backend_buffer_free(buffer);
        }
      }
      if (context != nullptr) {
        ggml_free(context);
      }
    }
    std::vector<std::uint8_t> metadata;
    ggml_context *context = nullptr;
    std::vector<ggml_tensor *> weights;
    ggml_tensor *activation = nullptr;
    std::vector<ggml_tensor *> results;
    std::vector<ggml_backend_buffer_t> weight_buffers;
    ggml_cgraph *graph = nullptr;
    ggml_cplan plan {};
    std::vector<std::uint8_t> work;
  };

  struct CachedFfnPlan {
    ~CachedFfnPlan() {
      for (ggml_backend_buffer_t buffer : weight_buffers) {
        if (buffer != nullptr) {
          ggml_backend_buffer_free(buffer);
        }
      }
      if (context != nullptr) {
        ggml_free(context);
      }
    }
    std::vector<std::uint8_t> metadata;
    ggml_context *context = nullptr;
    std::array<ggml_tensor *, 3> weights {};
    std::array<ggml_backend_buffer_t, 3> weight_buffers {};
    ggml_tensor *activation = nullptr;
    ggml_tensor *gate_result = nullptr;
    ggml_tensor *up_result = nullptr;
    ggml_tensor *gate_activated = nullptr;
    ggml_tensor *product = nullptr;
    ggml_tensor *result = nullptr;
    std::vector<float> gate_scratch;
    std::vector<float> up_scratch;
    ggml_cgraph *graph = nullptr;
    ggml_cplan plan {};
    std::vector<std::uint8_t> work;
  };

  std::unique_ptr<CachedPlan> create_plan(
    const PlanKey &key,
    ggml_type type,
    int threads,
    std::string *error) {
    auto cached = std::make_unique<CachedPlan>();
    const std::size_t metadata_bytes = std::max<std::size_t>(
      64u * 1024u,
      ggml_tensor_overhead() * 4u + ggml_graph_overhead_custom(8, false) + 4096u);
    cached->metadata.resize(metadata_bytes);
    ggml_init_params params {};
    params.mem_size = cached->metadata.size();
    params.mem_buffer = cached->metadata.data();
    params.no_alloc = true;
    cached->context = ggml_init(params);
    if (cached->context == nullptr) {
      fail("ggml_init failed for a cached CPU matvec plan.", error);
      return nullptr;
    }
    cached->weights = ggml_new_tensor_2d(
      cached->context,
      type,
      static_cast<std::int64_t>(key.row_width),
      static_cast<std::int64_t>(key.row_count));
    cached->activation = ggml_new_tensor_1d(
      cached->context,
      GGML_TYPE_F32,
      static_cast<std::int64_t>(key.row_width));
    if (cached->weights == nullptr || cached->activation == nullptr) {
      fail("ggml failed to construct cached CPU matvec input tensors.", error);
      return nullptr;
    }
    cached->result = ggml_mul_mat(cached->context, cached->weights, cached->activation);
    cached->graph = ggml_new_graph_custom(cached->context, 8, false);
    if (cached->result == nullptr || cached->graph == nullptr) {
      fail("ggml failed to construct a cached CPU matvec graph.", error);
      return nullptr;
    }
    const std::uint64_t budget = repack_budget_bytes();
    if (budget == std::numeric_limits<std::uint64_t>::max()) {
      fail("LIZARD_GGML_REPACK_MB must be an integer from 0 through 4096.", error);
      return nullptr;
    }
    const std::uint64_t already_repacked = g_repacked_bytes.load(std::memory_order_relaxed);
    const bool repack_eligible = budget > 0
      && type == GGML_TYPE_Q4_K
      && ggml_cpu_has_avx2()
      && key.row_count % 8u == 0
      && key.weight_bytes <= budget - std::min(budget, already_repacked);
    if (repack_eligible) {
      ggml_backend_buffer_type_t repack_type = ggml_backend_cpu_repack_buffer_type();
      const std::size_t allocation_bytes = ggml_backend_buft_get_alloc_size(repack_type, cached->weights);
      cached->weight_buffer = ggml_backend_buft_alloc_buffer(repack_type, allocation_bytes);
      if (cached->weight_buffer == nullptr
        || ggml_backend_tensor_alloc(
          cached->weight_buffer,
          cached->weights,
          ggml_backend_buffer_get_base(cached->weight_buffer)) != GGML_STATUS_SUCCESS) {
        g_repack_failures.fetch_add(1, std::memory_order_relaxed);
        fail("The pinned ggml Q4_K repack buffer could not be allocated.", error);
        return nullptr;
      }
      ggml_backend_buffer_set_usage(cached->weight_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
      ggml_backend_tensor_set(cached->weights, key.weight_data, 0, key.weight_bytes);
      g_repacked_bytes.fetch_add(key.weight_bytes, std::memory_order_relaxed);
      g_repacked_tensors.fetch_add(1, std::memory_order_relaxed);
    } else {
      cached->weights->data = const_cast<std::uint8_t *>(key.weight_data);
    }
    ggml_build_forward_expand(cached->graph, cached->result);
    cached->plan = ggml_graph_plan(cached->graph, threads, threadpool_);
    if (cached->plan.work_size > 0) {
      cached->work.resize(cached->plan.work_size + 64u);
      const std::uintptr_t raw = reinterpret_cast<std::uintptr_t>(cached->work.data());
      cached->plan.work_data = reinterpret_cast<std::uint8_t *>((raw + 63u) & ~std::uintptr_t(63u));
    }
    return cached;
  }

  std::unique_ptr<CachedMultiPlan> create_multi_plan(
    const MultiPlanKey &key,
    int threads,
    std::string *error) {
    auto cached = std::make_unique<CachedMultiPlan>();
    const std::size_t graph_nodes = key.operation_count * 2u + 4u;
    const std::size_t metadata_bytes = std::max<std::size_t>(
      128u * 1024u,
      ggml_tensor_overhead() * graph_nodes
        + ggml_graph_overhead_custom(graph_nodes * 2u, false)
        + 8192u);
    cached->metadata.resize(metadata_bytes);
    ggml_init_params params {};
    params.mem_size = cached->metadata.size();
    params.mem_buffer = cached->metadata.data();
    params.no_alloc = true;
    cached->context = ggml_init(params);
    if (cached->context == nullptr) {
      fail("ggml_init failed for a cached grouped CPU matvec plan.", error);
      return nullptr;
    }
    cached->activation = ggml_new_tensor_1d(
      cached->context,
      GGML_TYPE_F32,
      static_cast<std::int64_t>(key.operations.front().row_width));
    cached->graph = ggml_new_graph_custom(cached->context, graph_nodes * 2u, false);
    if (cached->activation == nullptr || cached->graph == nullptr) {
      fail("ggml failed to construct grouped CPU matvec shared state.", error);
      return nullptr;
    }

    cached->weights.reserve(key.operation_count);
    cached->results.reserve(key.operation_count);
    cached->weight_buffers.assign(key.operation_count, nullptr);
    for (std::size_t index = 0; index < key.operation_count; index += 1) {
      const PlanKey &operation = key.operations[index];
      const ggml_type type = static_cast<ggml_type>(operation.type);
      ggml_tensor *weights = ggml_new_tensor_2d(
        cached->context,
        type,
        static_cast<std::int64_t>(operation.row_width),
        static_cast<std::int64_t>(operation.row_count));
      ggml_tensor *result = weights == nullptr
        ? nullptr
        : ggml_mul_mat(cached->context, weights, cached->activation);
      if (weights == nullptr || result == nullptr) {
        fail("ggml failed to construct an operation in a grouped CPU matvec plan.", error);
        return nullptr;
      }
      cached->weights.push_back(weights);
      cached->results.push_back(result);

      const std::uint64_t budget = repack_budget_bytes();
      if (budget == std::numeric_limits<std::uint64_t>::max()) {
        fail("LIZARD_GGML_REPACK_MB must be an integer from 0 through 4096.", error);
        return nullptr;
      }
      const std::uint64_t already_repacked = g_repacked_bytes.load(std::memory_order_relaxed);
      const bool repack_eligible = budget > 0
        && type == GGML_TYPE_Q4_K
        && ggml_cpu_has_avx2()
        && operation.row_count % 8u == 0
        && operation.weight_bytes <= budget - std::min(budget, already_repacked);
      if (repack_eligible) {
        ggml_backend_buffer_type_t repack_type = ggml_backend_cpu_repack_buffer_type();
        const std::size_t allocation_bytes = ggml_backend_buft_get_alloc_size(repack_type, weights);
        ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(repack_type, allocation_bytes);
        cached->weight_buffers[index] = buffer;
        if (buffer == nullptr
          || ggml_backend_tensor_alloc(buffer, weights, ggml_backend_buffer_get_base(buffer))
            != GGML_STATUS_SUCCESS) {
          g_repack_failures.fetch_add(1, std::memory_order_relaxed);
          fail("A grouped pinned ggml Q4_K repack buffer could not be allocated.", error);
          return nullptr;
        }
        ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        ggml_backend_tensor_set(weights, operation.weight_data, 0, operation.weight_bytes);
        g_repacked_bytes.fetch_add(operation.weight_bytes, std::memory_order_relaxed);
        g_repacked_tensors.fetch_add(1, std::memory_order_relaxed);
      } else {
        weights->data = const_cast<std::uint8_t *>(operation.weight_data);
      }
      ggml_build_forward_expand(cached->graph, result);
    }

    cached->plan = ggml_graph_plan(cached->graph, threads, threadpool_);
    if (cached->plan.work_size > 0) {
      cached->work.resize(cached->plan.work_size + 64u);
      const std::uintptr_t raw = reinterpret_cast<std::uintptr_t>(cached->work.data());
      cached->plan.work_data = reinterpret_cast<std::uint8_t *>((raw + 63u) & ~std::uintptr_t(63u));
    }
    return cached;
  }

  std::unique_ptr<CachedFfnPlan> create_ffn_plan(
    const MultiPlanKey &key,
    int threads,
    std::string *error) {
    auto cached = std::make_unique<CachedFfnPlan>();
    constexpr std::size_t graph_nodes = 24;
    const std::size_t metadata_bytes = std::max<std::size_t>(
      128u * 1024u,
      ggml_tensor_overhead() * 16u
        + ggml_graph_overhead_custom(graph_nodes, false)
        + 8192u);
    cached->metadata.resize(metadata_bytes);
    ggml_init_params params {};
    params.mem_size = cached->metadata.size();
    params.mem_buffer = cached->metadata.data();
    params.no_alloc = true;
    cached->context = ggml_init(params);
    if (cached->context == nullptr) {
      fail("ggml_init failed for a cached fused FFN plan.", error);
      return nullptr;
    }
    cached->activation = ggml_new_tensor_1d(
      cached->context,
      GGML_TYPE_F32,
      static_cast<std::int64_t>(key.operations[0].row_width));
    cached->graph = ggml_new_graph_custom(cached->context, graph_nodes, false);
    if (cached->activation == nullptr || cached->graph == nullptr) {
      fail("ggml failed to construct fused FFN shared state.", error);
      return nullptr;
    }

    for (std::size_t index = 0; index < 3; index += 1) {
      const PlanKey &operation = key.operations[index];
      const ggml_type type = static_cast<ggml_type>(operation.type);
      ggml_tensor *weights = ggml_new_tensor_2d(
        cached->context,
        type,
        static_cast<std::int64_t>(operation.row_width),
        static_cast<std::int64_t>(operation.row_count));
      if (weights == nullptr) {
        fail("ggml failed to construct a fused FFN weight tensor.", error);
        return nullptr;
      }
      cached->weights[index] = weights;

      const std::uint64_t budget = repack_budget_bytes();
      const std::uint64_t already_repacked = g_repacked_bytes.load(std::memory_order_relaxed);
      const bool repack_eligible = budget > 0
        && budget != std::numeric_limits<std::uint64_t>::max()
        && type == GGML_TYPE_Q4_K
        && ggml_cpu_has_avx2()
        && operation.row_count % 8u == 0
        && operation.weight_bytes <= budget - std::min(budget, already_repacked);
      if (repack_eligible) {
        ggml_backend_buffer_type_t repack_type = ggml_backend_cpu_repack_buffer_type();
        const std::size_t allocation_bytes = ggml_backend_buft_get_alloc_size(repack_type, weights);
        ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(repack_type, allocation_bytes);
        cached->weight_buffers[index] = buffer;
        if (buffer == nullptr
          || ggml_backend_tensor_alloc(buffer, weights, ggml_backend_buffer_get_base(buffer))
            != GGML_STATUS_SUCCESS) {
          g_repack_failures.fetch_add(1, std::memory_order_relaxed);
          fail("A fused FFN pinned ggml Q4_K repack buffer could not be allocated.", error);
          return nullptr;
        }
        ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        ggml_backend_tensor_set(weights, operation.weight_data, 0, operation.weight_bytes);
        g_repacked_bytes.fetch_add(operation.weight_bytes, std::memory_order_relaxed);
        g_repacked_tensors.fetch_add(1, std::memory_order_relaxed);
      } else {
        weights->data = const_cast<std::uint8_t *>(operation.weight_data);
      }
    }

    cached->gate_result = ggml_mul_mat(cached->context, cached->weights[0], cached->activation);
    cached->up_result = ggml_mul_mat(cached->context, cached->weights[1], cached->activation);
    cached->gate_activated = cached->gate_result == nullptr
      ? nullptr
      : ggml_silu_inplace(cached->context, cached->gate_result);
    cached->product = cached->gate_activated == nullptr || cached->up_result == nullptr
      ? nullptr
      : ggml_mul_inplace(cached->context, cached->gate_activated, cached->up_result);
    cached->result = cached->product == nullptr
      ? nullptr
      : ggml_mul_mat(cached->context, cached->weights[2], cached->product);
    if (cached->gate_result == nullptr
      || cached->up_result == nullptr
      || cached->gate_activated == nullptr
      || cached->product == nullptr
      || cached->result == nullptr) {
      fail("ggml failed to construct the fused SwiGLU FFN graph.", error);
      return nullptr;
    }
    cached->gate_scratch.assign(static_cast<std::size_t>(key.operations[0].row_count), 0.0f);
    cached->up_scratch.assign(static_cast<std::size_t>(key.operations[1].row_count), 0.0f);
    ggml_build_forward_expand(cached->graph, cached->result);
    cached->plan = ggml_graph_plan(cached->graph, threads, threadpool_);
    if (cached->plan.work_size > 0) {
      cached->work.resize(cached->plan.work_size + 64u);
      const std::uintptr_t raw = reinterpret_cast<std::uintptr_t>(cached->work.data());
      cached->plan.work_data = reinterpret_cast<std::uint8_t *>((raw + 63u) & ~std::uintptr_t(63u));
    }
    return cached;
  }

  bool ensure_threadpool(int threads, std::string *error) {
    if (threadpool_ != nullptr && thread_count_ == threads) {
      return true;
    }
    if (threadpool_ != nullptr) {
      plans_.clear();
      multi_plans_.clear();
      ffn_plans_.clear();
      ggml_threadpool_free(threadpool_);
      threadpool_ = nullptr;
      thread_count_ = 0;
    }
    ggml_threadpool_params params = ggml_threadpool_params_default(threads);
    if (const char *poll = std::getenv("LIZARD_GGML_POLL"); poll != nullptr && *poll != '\0') {
      char *end = nullptr;
      const unsigned long parsed = std::strtoul(poll, &end, 10);
      if (end == poll || *end != '\0' || parsed > 100ul) {
        return fail("LIZARD_GGML_POLL must be an integer from 0 through 100.", error);
      }
      params.poll = static_cast<std::uint32_t>(parsed);
    }
    threadpool_ = ggml_threadpool_new(&params);
    if (threadpool_ == nullptr) {
      return fail("ggml_threadpool_new failed for the pinned CPU backend.", error);
    }
    thread_count_ = threads;
    return true;
  }

  static bool fail(const std::string &message, std::string *error) {
    if (error != nullptr) {
      *error = message;
    }
    return false;
  }

  std::mutex mutex_;
  ggml_threadpool *threadpool_ = nullptr;
  int thread_count_ = 0;
  std::unordered_map<PlanKey, std::unique_ptr<CachedPlan>, PlanKeyHash> plans_;
  std::unordered_map<MultiPlanKey, std::unique_ptr<CachedMultiPlan>, MultiPlanKeyHash> multi_plans_;
  std::unordered_map<MultiPlanKey, std::unique_ptr<CachedFfnPlan>, MultiPlanKeyHash> ffn_plans_;
};

PinnedGgmlCpuState &backend_state() {
  static PinnedGgmlCpuState state;
  return state;
}

#endif

}  // namespace

bool ggml_cpu_backend_compiled() {
#ifdef LIZARD_HAVE_GGML_CPU_BACKEND
  return true;
#else
  return false;
#endif
}

bool ggml_cpu_backend_selected() {
  return normalized_backend_request() == "ggml";
}

const char *ggml_cpu_backend_effective_name() {
  return ggml_cpu_backend_selected() ? "ggml-cpu-pinned" : "lizard-native-cpu";
}

std::string ggml_cpu_backend_selection_error() {
  const std::string requested = normalized_backend_request();
  if (requested != "native" && requested != "ggml") {
    return "LIZARD_NATIVE_CPU_BACKEND must be either 'ggml' or 'native'.";
  }
  if (requested == "ggml" && !ggml_cpu_backend_compiled()) {
    return "The pinned ggml CPU backend was requested but this runner was built without it.";
  }
  if (repack_budget_bytes() == std::numeric_limits<std::uint64_t>::max()) {
    return "LIZARD_GGML_REPACK_MB must be an integer from 0 through 4096.";
  }
  if (grouped_graph_mode() < 0) {
    return "LIZARD_GGML_GROUPED_GRAPHS must be either 0 or 1.";
  }
  if (fused_ffn_mode() < 0) {
    return "LIZARD_GGML_FUSED_FFN must be either 0 or 1.";
  }
  return {};
}

std::uint64_t ggml_cpu_backend_matvec_calls() {
  return g_matvec_calls.load(std::memory_order_relaxed);
}

std::uint64_t ggml_cpu_backend_matvec_failures() {
  return g_matvec_failures.load(std::memory_order_relaxed);
}

std::uint64_t ggml_cpu_backend_plan_hits() {
  return g_plan_hits.load(std::memory_order_relaxed);
}

std::uint64_t ggml_cpu_backend_plan_misses() {
  return g_plan_misses.load(std::memory_order_relaxed);
}

std::uint64_t ggml_cpu_backend_matvec_nanoseconds() {
  return g_matvec_nanoseconds.load(std::memory_order_relaxed);
}

std::uint64_t ggml_cpu_backend_weight_bytes() {
  return g_weight_bytes.load(std::memory_order_relaxed);
}

std::uint64_t ggml_cpu_backend_repack_budget_bytes() {
  return repack_budget_bytes();
}

std::uint64_t ggml_cpu_backend_repacked_bytes() {
  return g_repacked_bytes.load(std::memory_order_relaxed);
}

std::uint64_t ggml_cpu_backend_repacked_tensors() {
  return g_repacked_tensors.load(std::memory_order_relaxed);
}

std::uint64_t ggml_cpu_backend_repack_failures() {
  return g_repack_failures.load(std::memory_order_relaxed);
}

bool ggml_cpu_backend_grouped_graphs_enabled() {
  return grouped_graph_mode() == 1;
}

std::uint64_t ggml_cpu_backend_grouped_graph_calls() {
  return g_grouped_graph_calls.load(std::memory_order_relaxed);
}

std::uint64_t ggml_cpu_backend_grouped_graph_operations() {
  return g_grouped_graph_operations.load(std::memory_order_relaxed);
}

bool ggml_cpu_backend_fused_ffn_enabled() {
  return fused_ffn_mode() == 1;
}

std::uint64_t ggml_cpu_backend_fused_ffn_calls() {
  return g_fused_ffn_calls.load(std::memory_order_relaxed);
}

void ggml_cpu_backend_reset() {
#ifdef LIZARD_HAVE_GGML_CPU_BACKEND
  backend_state().reset();
#endif
  g_matvec_calls.store(0, std::memory_order_relaxed);
  g_matvec_failures.store(0, std::memory_order_relaxed);
  g_plan_hits.store(0, std::memory_order_relaxed);
  g_plan_misses.store(0, std::memory_order_relaxed);
  g_matvec_nanoseconds.store(0, std::memory_order_relaxed);
  g_weight_bytes.store(0, std::memory_order_relaxed);
  g_repacked_bytes.store(0, std::memory_order_relaxed);
  g_repacked_tensors.store(0, std::memory_order_relaxed);
  g_repack_failures.store(0, std::memory_order_relaxed);
  g_grouped_graph_calls.store(0, std::memory_order_relaxed);
  g_grouped_graph_operations.store(0, std::memory_order_relaxed);
  g_fused_ffn_calls.store(0, std::memory_order_relaxed);
}

bool ggml_cpu_backend_matvec(
  std::uint32_t ggml_type,
  std::uint64_t row_count,
  std::uint64_t row_width,
  const std::uint8_t *weight_data,
  std::size_t weight_bytes,
  const float *input,
  std::size_t input_count,
  std::uint64_t thread_count,
  std::vector<float> *output,
  std::string *error) {
  g_matvec_calls.fetch_add(1, std::memory_order_relaxed);
  g_weight_bytes.fetch_add(weight_bytes, std::memory_order_relaxed);
  const auto started_at = std::chrono::steady_clock::now();
#ifdef LIZARD_HAVE_GGML_CPU_BACKEND
  if (backend_state().matvec(
      ggml_type,
      row_count,
      row_width,
      weight_data,
      weight_bytes,
      input,
      input_count,
      thread_count,
      output,
      error)) {
    g_matvec_nanoseconds.fetch_add(
      static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - started_at).count()),
      std::memory_order_relaxed);
    return true;
  }
#else
  if (error != nullptr) {
    *error = "The pinned ggml CPU backend is not compiled into this runner.";
  }
#endif
  g_matvec_nanoseconds.fetch_add(
    static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now() - started_at).count()),
    std::memory_order_relaxed);
  g_matvec_failures.fetch_add(1, std::memory_order_relaxed);
  return false;
}

bool ggml_cpu_backend_multi_matvec(
  const GgmlCpuMatvecOperation *operations,
  std::size_t operation_count,
  const float *input,
  std::size_t input_count,
  std::uint64_t thread_count,
  std::string *error) {
  if (operations == nullptr || operation_count < 2 || operation_count > 4) {
    if (error != nullptr) {
      *error = "Pinned ggml grouped graphs require two through four operations.";
    }
    return false;
  }
  if (!ggml_cpu_backend_grouped_graphs_enabled()) {
    if (error != nullptr) {
      *error = "Pinned ggml grouped graphs are disabled.";
    }
    return false;
  }
  g_matvec_calls.fetch_add(operation_count, std::memory_order_relaxed);
  std::uint64_t weight_bytes = 0;
  for (std::size_t index = 0; index < operation_count; index += 1) {
    weight_bytes += operations[index].weight_bytes;
  }
  g_weight_bytes.fetch_add(weight_bytes, std::memory_order_relaxed);
  g_grouped_graph_calls.fetch_add(1, std::memory_order_relaxed);
  g_grouped_graph_operations.fetch_add(operation_count, std::memory_order_relaxed);
  const auto started_at = std::chrono::steady_clock::now();
#ifdef LIZARD_HAVE_GGML_CPU_BACKEND
  if (backend_state().multi_matvec(
      operations,
      operation_count,
      input,
      input_count,
      thread_count,
      error)) {
    g_matvec_nanoseconds.fetch_add(
      static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - started_at).count()),
      std::memory_order_relaxed);
    return true;
  }
#else
  if (error != nullptr) {
    *error = "The pinned ggml CPU backend is not compiled into this runner.";
  }
#endif
  g_matvec_nanoseconds.fetch_add(
    static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now() - started_at).count()),
    std::memory_order_relaxed);
  g_matvec_failures.fetch_add(operation_count, std::memory_order_relaxed);
  return false;
}

bool ggml_cpu_backend_ffn_swiglu(
  const GgmlCpuWeight &gate,
  const GgmlCpuWeight &up,
  const GgmlCpuWeight &down,
  const float *input,
  std::size_t input_count,
  std::uint64_t thread_count,
  std::vector<float> *output,
  std::string *error) {
  if (!ggml_cpu_backend_fused_ffn_enabled()) {
    if (error != nullptr) {
      *error = "Pinned ggml fused FFN is disabled.";
    }
    return false;
  }
  if (input == nullptr || output == nullptr) {
    if (error != nullptr) {
      *error = "Pinned ggml fused FFN requires non-null input and output.";
    }
    return false;
  }
  const std::uint64_t weight_bytes = gate.weight_bytes + up.weight_bytes + down.weight_bytes;
  g_matvec_calls.fetch_add(3, std::memory_order_relaxed);
  g_weight_bytes.fetch_add(weight_bytes, std::memory_order_relaxed);
  g_fused_ffn_calls.fetch_add(1, std::memory_order_relaxed);
  const auto started_at = std::chrono::steady_clock::now();
#ifdef LIZARD_HAVE_GGML_CPU_BACKEND
  if (backend_state().ffn_swiglu(
      gate,
      up,
      down,
      input,
      input_count,
      thread_count,
      output,
      error)) {
    g_matvec_nanoseconds.fetch_add(
      static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - started_at).count()),
      std::memory_order_relaxed);
    return true;
  }
#else
  if (error != nullptr) {
    *error = "The pinned ggml CPU backend is not compiled into this runner.";
  }
#endif
  g_matvec_nanoseconds.fetch_add(
    static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now() - started_at).count()),
    std::memory_order_relaxed);
  g_matvec_failures.fetch_add(3, std::memory_order_relaxed);
  return false;
}

bool ggml_cpu_backend_selftest(std::string *error) {
  const std::vector<float> weights = {
    1.0f, 2.0f, 3.0f, 4.0f,
    -1.0f, 0.5f, 2.0f, -0.25f,
  };
  const std::vector<float> input = {0.25f, -0.5f, 1.5f, 2.0f};
  std::vector<float> output;
  if (!ggml_cpu_backend_matvec(
      0,
      2,
      4,
      reinterpret_cast<const std::uint8_t *>(weights.data()),
      weights.size() * sizeof(float),
      input.data(),
      input.size(),
      2,
      &output,
      error)) {
    return false;
  }
  const float expected0 = 11.75f;
  const float expected1 = 2.0f;
  if (output.size() != 2
    || std::abs(output[0] - expected0) > 1e-5f
    || std::abs(output[1] - expected1) > 1e-5f) {
    if (error != nullptr) {
      *error = "The pinned ggml CPU backend self-test produced incorrect F32 results.";
    }
    return false;
  }
  if (ggml_cpu_backend_grouped_graphs_enabled()) {
    const std::vector<float> second_weights = {2.0f, 0.0f, -1.0f, 0.5f};
    std::vector<float> grouped_first;
    std::vector<float> grouped_second;
    const GgmlCpuMatvecOperation operations[2] = {
      {
        0,
        2,
        4,
        reinterpret_cast<const std::uint8_t *>(weights.data()),
        weights.size() * sizeof(float),
        &grouped_first,
      },
      {
        0,
        1,
        4,
        reinterpret_cast<const std::uint8_t *>(second_weights.data()),
        second_weights.size() * sizeof(float),
        &grouped_second,
      },
    };
    if (!ggml_cpu_backend_multi_matvec(
        operations,
        2,
        input.data(),
        input.size(),
        2,
        error)) {
      return false;
    }
    if (grouped_first.size() != 2
      || grouped_second.size() != 1
      || std::abs(grouped_first[0] - expected0) > 1e-5f
      || std::abs(grouped_first[1] - expected1) > 1e-5f
      || std::abs(grouped_second[0]) > 1e-5f) {
      if (error != nullptr) {
        *error = "The pinned ggml grouped-graph self-test produced incorrect F32 results.";
      }
      return false;
    }
  }
  if (ggml_cpu_backend_fused_ffn_enabled()) {
    const std::vector<float> up_weights = {
      0.5f, -1.0f, 2.0f, 0.0f,
      1.0f, 1.0f, 1.0f, 1.0f,
    };
    const std::vector<float> down_weights = {1.25f, -0.75f};
    const GgmlCpuWeight gate {
      0, 2, 4,
      reinterpret_cast<const std::uint8_t *>(weights.data()),
      weights.size() * sizeof(float),
    };
    const GgmlCpuWeight up {
      0, 2, 4,
      reinterpret_cast<const std::uint8_t *>(up_weights.data()),
      up_weights.size() * sizeof(float),
    };
    const GgmlCpuWeight down {
      0, 1, 2,
      reinterpret_cast<const std::uint8_t *>(down_weights.data()),
      down_weights.size() * sizeof(float),
    };
    std::vector<float> fused;
    if (!ggml_cpu_backend_ffn_swiglu(
        gate, up, down, input.data(), input.size(), 2, &fused, error)) {
      return false;
    }
    const auto silu_value = [](float value) {
      return value / (1.0f + std::exp(-value));
    };
    const float expected_fused = 1.25f * silu_value(expected0) * 3.625f
      - 0.75f * silu_value(expected1) * 3.25f;
    if (fused.size() != 1 || std::abs(fused[0] - expected_fused) > 1e-4f) {
      if (error != nullptr) {
        *error = "The pinned ggml fused FFN self-test produced an incorrect result.";
      }
      return false;
    }
  }
  return true;
}

}  // namespace lizard_native
