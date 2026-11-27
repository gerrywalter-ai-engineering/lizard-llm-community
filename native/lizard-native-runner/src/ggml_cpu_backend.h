#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace lizard_native {

inline constexpr const char *k_lizard_ggml_commit =
  "ddd4ec1428a6201e18975ea52b07c71e0f9aef26";

bool ggml_cpu_backend_compiled();
bool ggml_cpu_backend_selected();
const char *ggml_cpu_backend_effective_name();
std::string ggml_cpu_backend_selection_error();
std::uint64_t ggml_cpu_backend_matvec_calls();
std::uint64_t ggml_cpu_backend_matvec_failures();
std::uint64_t ggml_cpu_backend_plan_hits();
std::uint64_t ggml_cpu_backend_plan_misses();
std::uint64_t ggml_cpu_backend_matvec_nanoseconds();
std::uint64_t ggml_cpu_backend_weight_bytes();
std::uint64_t ggml_cpu_backend_repack_budget_bytes();
std::uint64_t ggml_cpu_backend_repacked_bytes();
std::uint64_t ggml_cpu_backend_repacked_tensors();
std::uint64_t ggml_cpu_backend_repack_failures();
bool ggml_cpu_backend_grouped_graphs_enabled();
std::uint64_t ggml_cpu_backend_grouped_graph_calls();
std::uint64_t ggml_cpu_backend_grouped_graph_operations();
bool ggml_cpu_backend_fused_ffn_enabled();
std::uint64_t ggml_cpu_backend_fused_ffn_calls();
void ggml_cpu_backend_reset();

struct GgmlCpuWeight {
  std::uint32_t ggml_type = 0;
  std::uint64_t row_count = 0;
  std::uint64_t row_width = 0;
  const std::uint8_t *weight_data = nullptr;
  std::size_t weight_bytes = 0;
};

struct GgmlCpuMatvecOperation {
  std::uint32_t ggml_type = 0;
  std::uint64_t row_count = 0;
  std::uint64_t row_width = 0;
  const std::uint8_t *weight_data = nullptr;
  std::size_t weight_bytes = 0;
  std::vector<float> *output = nullptr;
};

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
  std::string *error);

bool ggml_cpu_backend_multi_matvec(
  const GgmlCpuMatvecOperation *operations,
  std::size_t operation_count,
  const float *input,
  std::size_t input_count,
  std::uint64_t thread_count,
  std::string *error);

bool ggml_cpu_backend_ffn_swiglu(
  const GgmlCpuWeight &gate,
  const GgmlCpuWeight &up,
  const GgmlCpuWeight &down,
  const float *input,
  std::size_t input_count,
  std::uint64_t thread_count,
  std::vector<float> *output,
  std::string *error);

bool ggml_cpu_backend_selftest(std::string *error);

}  // namespace lizard_native
