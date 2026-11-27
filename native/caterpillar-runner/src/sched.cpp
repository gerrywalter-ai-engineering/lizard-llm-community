// Static backend scheduler for Caterpillar Phase 3. Assigns each graph node
// to CPU or D3D12 once per compiled plan from the pinned fit decision and
// kernel availability. Deliberately conservative: a decode plan is either a
// stable GPU prefix + CPU tail (matching the fit ladder) or entirely CPU —
// there is no per-token rerouting and no mid-layer backend flip, which is
// what keeps decode at one-or-two submissions per token.

#include "sched.h"

#include <charconv>
#include <string_view>

namespace cat {
namespace {

std::optional<std::size_t> layer_of_name(std::string_view name) {
    constexpr std::string_view prefix = "blk.";
    if (!name.starts_with(prefix)) return std::nullopt;
    const auto first = name.data() + prefix.size();
    const auto last = name.data() + name.size();
    std::size_t layer{};
    const auto parsed = std::from_chars(first, last, layer);
    if (parsed.ec != std::errc{} || parsed.ptr == first || parsed.ptr == last || *parsed.ptr != '.') {
        return std::nullopt;
    }
    return layer;
}

}  // namespace

bool gpu_weight_type_supported(std::uint32_t ggml_type) {
    switch (ggml_type) {
        case 0:   // F32
        case 1:   // F16 (widened to F32 at staging; raw 2-byte loads hang Xe)
        case 2:   // Q4_0 (18-byte fp16-scale staged blocks)
        case 8:   // Q8_0
        case 12:  // Q4_K
        case 13:  // Q5_K
        case 14:  // Q6_K
        case 20:  // IQ4_NL
        case 23:  // IQ4_XS
            return true;
        default:
            return false;
    }
}

std::optional<std::size_t> node_layer(const Graph& graph, const Node& node) {
    return layer_of_name(graph.tensor(node.dst).name);
}

SchedulerResult schedule_decode_backends(const Graph& graph, const SchedulerInput& input) {
    SchedulerResult result;
    const auto& nodes = graph.nodes();
    result.node_backends.assign(nodes.size(), SpanBackend::cpu);
    result.cpu_node_count = nodes.size();

    if (!input.gpu_available) {
        result.cpu_reason = "GPU backend unavailable";
        return result;
    }
    if (input.resident_layer_count == 0) {
        result.cpu_reason = "fit pinned zero GPU-resident layers";
        return result;
    }
    for (const auto& [tensor_id, ggml_type] : input.weight_types) {
        if (!gpu_weight_type_supported(ggml_type)) {
            result.cpu_reason = "weight tensor " + graph.tensor(tensor_id).name
                + " has no GPU kernel for GGML type " + std::to_string(ggml_type);
            return result;
        }
    }

    const bool all_resident = input.resident_layer_count >= input.total_layer_count;
    result.gpu_node_count = 0;
    for (std::size_t index = 0; index < nodes.size(); ++index) {
        const auto layer = node_layer(graph, nodes[index]);
        const bool resident = layer.has_value()
            ? *layer < input.resident_layer_count
            : all_resident;
        if (resident) {
            result.node_backends[index] = SpanBackend::gpu;
            ++result.gpu_node_count;
        }
    }
    result.cpu_node_count = nodes.size() - result.gpu_node_count;

    // A stable split must stay a single GPU prefix followed by a single CPU
    // tail. Global nodes before the first layer (the Gemma embedding scale
    // lives outside blk.*) would otherwise splinter the prefix into extra
    // spans, so pull leading global nodes onto the GPU when a prefix exists.
    if (!all_resident && result.gpu_node_count != 0) {
        for (std::size_t index = 0; index < nodes.size(); ++index) {
            if (node_layer(graph, nodes[index]).has_value()) break;
            if (result.node_backends[index] != SpanBackend::gpu) {
                result.node_backends[index] = SpanBackend::gpu;
                ++result.gpu_node_count;
                --result.cpu_node_count;
            }
        }
    }
    return result;
}

}  // namespace cat
