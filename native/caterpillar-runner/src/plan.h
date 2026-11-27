#pragma once

#include "alloc.h"
#include "graph.h"
#include "tensor_data.h"

#include <array>
#include <cstddef>
#include <span>
#include <unordered_map>
#include <vector>

namespace cat {

using Bindings = std::unordered_map<TensorId, std::vector<float>>;
using BindingViews = std::unordered_map<TensorId, std::span<const float>>;
using QuantizedBindings = std::unordered_map<TensorId, QuantizedTensorView>;

enum class SpanBackend {
    cpu,
    gpu,
};

struct PlanSpan {
    std::size_t first_node{};
    std::size_t node_count{};
    SpanBackend backend{SpanBackend::cpu};
};

// Replay-time dynamic state for compile-once decode plans. A graph compiled
// with capacity-sized past-KV shapes replays with the live counts here; the
// sentinel value keeps the compile-time node parameters (the Phase 1/2
// behavior, where every replay matches the graph shape exactly).
struct ReplayDynamics {
    static constexpr std::size_t use_node_params = static_cast<std::size_t>(-1);
    std::size_t past_tokens{use_node_params};
    std::size_t position_offset{use_node_params};
};

class Plan;

// Executes the GPU spans of a compiled plan. Implemented by the D3D12
// backend; plan replay calls it for every gpu span in graph order. The
// executor reads dynamic inputs from and writes span outputs back into the
// replay's CPU arena/bindings so CPU and GPU spans interoperate.
class GpuExecutor {
public:
    virtual ~GpuExecutor() = default;
    virtual void execute_span(
        const Plan& plan,
        const PlanSpan& span,
        const BindingViews& external,
        const ReplayDynamics& dynamics,
        std::span<float> arena
    ) = 0;
};

class Plan {
public:
    static Plan compile(const Graph& graph);
    static Plan compile(const Graph& graph, const std::vector<SpanBackend>& node_backends);

    [[nodiscard]] Bindings replay(const Bindings& external) const;
    [[nodiscard]] Bindings replay(const Bindings& external, const QuantizedBindings& quantized) const;
    [[nodiscard]] Bindings replay_views(
        const BindingViews& external,
        const QuantizedBindings& quantized = {},
        const ReplayDynamics& dynamics = {},
        GpuExecutor* gpu = nullptr
    ) const;
    [[nodiscard]] const Graph& graph() const noexcept { return graph_; }
    [[nodiscard]] const AllocationPlan& allocation() const noexcept { return allocation_; }
    [[nodiscard]] const std::vector<PlanSpan>& spans() const noexcept { return spans_; }
    [[nodiscard]] std::vector<PlanSpan> cpu_spans() const;
    [[nodiscard]] std::size_t gpu_span_count() const;

private:
    Graph graph_;
    AllocationPlan allocation_;
    std::vector<PlanSpan> spans_;
    mutable std::vector<float> arena_;
};

// Executes one node on the CPU. Shared by plan replay, the imperative
// reference walk, and the GPU backend's kernel selftests (which compare
// device results against exactly this implementation).
void execute_node_reference(
    const Graph& graph,
    const Node& node,
    const std::array<const float*, 5>& src,
    float* dst,
    const ReplayDynamics& dynamics
);

[[nodiscard]] Bindings imperative_reference(const Graph& graph, const Bindings& external);
[[nodiscard]] bool bit_equal(const std::vector<float>& left, const std::vector<float>& right);

}  // namespace cat
