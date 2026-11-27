#pragma once

#include "graph.h"
#include "plan.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace cat {

// Static op-to-backend scheduler (Phase 3). Decided once per compiled plan,
// never per token. The only inputs are the fit decision (how many layers are
// GPU-resident) and per-weight GPU kernel availability; contiguous
// same-backend nodes coalesce into spans inside Plan::compile.
struct SchedulerInput {
    // GGML tensor type per quantized matmul weight TensorId. Float matmul
    // weights (bound as plain floats) need no entry.
    std::unordered_map<TensorId, std::uint32_t> weight_types;
    std::size_t resident_layer_count{};
    std::size_t total_layer_count{};
    bool gpu_available{false};
};

struct SchedulerResult {
    std::vector<SpanBackend> node_backends;
    std::size_t gpu_node_count{};
    std::size_t cpu_node_count{};
    // Non-empty when the scheduler had to demote the whole plan to CPU.
    std::string cpu_reason;
};

// True when the D3D12 backend has an in-shader dequant kernel for this GGML
// tensor type (resident staged blocks): F32=0, F16=1, Q4_0=2, Q8_0=8,
// Q4_K=12, Q6_K=14.
[[nodiscard]] bool gpu_weight_type_supported(std::uint32_t ggml_type);

// Layer index a node belongs to, derived from its destination tensor name
// ("blk.N."-prefixed activations); nodes outside any layer (embedding scale,
// final norm, logits) return nullopt and follow the all-resident decision.
[[nodiscard]] std::optional<std::size_t> node_layer(const Graph& graph, const Node& node);

[[nodiscard]] SchedulerResult schedule_decode_backends(const Graph& graph, const SchedulerInput& input);

}  // namespace cat
