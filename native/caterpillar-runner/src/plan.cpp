// Replayable graph plan. The plan owns one activation arena and resolves
// every tensor offset at compile time; replay performs no graph construction.
// Nodes are partitioned into contiguous same-backend spans: CPU spans execute
// inline, GPU spans are delegated to a registered GpuExecutor (the D3D12
// backend) which reads inputs from and writes results back into the same
// arena. Clean-room implementation based on published graph-plan concepts.

#include "plan.h"

#include "attention.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace cat {
namespace {

// Run 04 (2026-08-04) iteration 14: per-op stage profile.
//
// The matmul profile in tensor_data.cpp only counts tensor_matrix_*, which
// measured 47.6% of decode time on llama-3.2-3B. The other 52.4% had no
// instrumentation at all, and two attempts to reduce it by inspection (the
// iteration-12/13 parameter caches) both measured exactly zero - which is the
// argument for measuring it rather than reasoning about it.
//
// CATERPILLAR_CPU_STAGE_PROFILE=1 accumulates wall time per graph op across
// every replay and prints the table at process exit. Off by default: the
// clock read per node is cheap but not free, and this is a diagnostic.
constexpr std::size_t k_op_count = 10;

const char* op_name(std::size_t op) noexcept {
    switch (op) {
        case 0: return "rms_norm";
        case 1: return "rms_norm_heads";
        case 2: return "rms_norm_heads_unweighted";
        case 3: return "matmul";
        case 4: return "rope";
        case 5: return "causal_attention";
        case 6: return "silu_mul";
        case 7: return "gelu_mul";
        case 8: return "add";
        case 9: return "scale";
        default: return "unknown";
    }
}

struct StageTotals {
    std::array<std::atomic<std::uint64_t>, k_op_count> nanoseconds{};
    std::array<std::atomic<std::uint64_t>, k_op_count> calls{};

    ~StageTotals() {
        std::uint64_t total = 0;
        for (const auto& entry : nanoseconds) total += entry.load(std::memory_order_relaxed);
        if (total == 0) return;
        std::cerr << "caterpillar CPU stage profile (ms/op, summed over worker threads):\n";
        for (std::size_t op = 0; op < k_op_count; ++op) {
            const auto value = nanoseconds[op].load(std::memory_order_relaxed);
            if (value == 0) continue;
            std::cerr << "  " << op_name(op) << "="
                      << static_cast<double>(value) / 1.0e6 << "ms/"
                      << calls[op].load(std::memory_order_relaxed) << " ("
                      << (100.0 * static_cast<double>(value) / static_cast<double>(total))
                      << "%)\n";
        }
        std::cerr << "  stage total=" << static_cast<double>(total) / 1.0e6 << "ms\n";
    }
};

StageTotals& stage_totals() {
    static StageTotals totals;
    return totals;
}

bool stage_profile_enabled() noexcept {
    static const bool enabled = [] {
        const char* configured = std::getenv("CATERPILLAR_CPU_STAGE_PROFILE");
        return configured != nullptr && configured[0] == '1';
    }();
    return enabled;
}

void record_stage(std::size_t op, std::chrono::steady_clock::time_point started) noexcept {
    if (op >= k_op_count) return;
    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - started).count();
    stage_totals().nanoseconds[op].fetch_add(
        static_cast<std::uint64_t>(elapsed), std::memory_order_relaxed);
    stage_totals().calls[op].fetch_add(1, std::memory_order_relaxed);
}

void require_binding_size(const Tensor& tensor, const std::vector<float>& values) {
    if (values.size() != tensor.elements()) {
        throw std::invalid_argument("binding size mismatch for tensor: " + tensor.name);
    }
}

}  // namespace

void execute_node_reference(
    const Graph& graph,
    const Node& node,
    const std::array<const float*, 5>& src,
    float* dst,
    const ReplayDynamics& dynamics
) {
    const auto& output = graph.tensor(node.dst);
    const auto total = output.elements();
    const auto width = output.shape.back();
    switch (node.op) {
        case Op::rms_norm: {
            const auto rows = total / width;
            for (std::size_t row = 0; row < rows; ++row) {
                double square_sum = 0.0;
                for (std::size_t column = 0; column < width; ++column) {
                    const double value = src[0][row * width + column];
                    square_sum += value * value;
                }
                const float inverse_rms = static_cast<float>(
                    1.0 / std::sqrt(square_sum / static_cast<double>(width) + static_cast<double>(node.scalar)));
                for (std::size_t column = 0; column < width; ++column) {
                    dst[row * width + column] = src[0][row * width + column] * inverse_rms * src[1][column];
                }
            }
            break;
        }
        case Op::rms_norm_heads:
        case Op::rms_norm_heads_unweighted: {
            const auto heads = node.params[0];
            const auto head_dim = node.params[1];
            const auto batches = total / (heads * head_dim);
            for (std::size_t batch = 0; batch < batches; ++batch) {
                for (std::size_t head = 0; head < heads; ++head) {
                    const auto base = (batch * heads + head) * head_dim;
                    double square_sum = 0.0;
                    for (std::size_t column = 0; column < head_dim; ++column) {
                        const double value = src[0][base + column];
                        square_sum += value * value;
                    }
                    const float inverse_rms = static_cast<float>(1.0 / std::sqrt(square_sum / static_cast<double>(head_dim) + node.scalar));
                    for (std::size_t column = 0; column < head_dim; ++column) {
                        const float weight = node.op == Op::rms_norm_heads ? src[1][column] : 1.0F;
                        dst[base + column] = src[0][base + column] * inverse_rms * weight;
                    }
                }
            }
            break;
        }
        case Op::matmul: {
            const auto& weight = graph.tensor(node.src[1]);
            const auto rows = weight.shape[0];
            const auto columns = weight.shape[1];
            const auto batches = graph.tensor(node.src[0]).elements() / columns;
            for (std::size_t batch = 0; batch < batches; ++batch) {
                for (std::size_t row = 0; row < rows; ++row) {
                    double sum = 0.0;
                    for (std::size_t column = 0; column < columns; ++column) {
                        sum += static_cast<double>(src[0][batch * columns + column])
                            * static_cast<double>(src[1][row * columns + column]);
                    }
                    dst[batch * rows + row] = static_cast<float>(sum);
                }
            }
            break;
        }
        case Op::rope: {
            const auto heads = node.params[0];
            const auto head_dim = node.params[1];
            const auto position_offset = dynamics.position_offset != ReplayDynamics::use_node_params
                ? dynamics.position_offset
                : node.params[2];
            const auto tokens = total / (heads * head_dim);
            const float frequency_base = node.scalar;
            const bool neox = node.params.size() >= 4 && node.params[3] != 0;
            const bool independent_rows =
                node.params.size() >= 5 && node.params[4] != 0;
            const auto half_dim = head_dim / 2;
            for (std::size_t token = 0; token < tokens; ++token) {
                const float position = static_cast<float>(
                    position_offset + (independent_rows ? 0 : token)
                ) * node.secondary_scalar;
                for (std::size_t head = 0; head < heads; ++head) {
                    const auto base = (token * heads + head) * head_dim;
                    for (std::size_t pair = 0; pair < half_dim; ++pair) {
                        const auto left_index = neox ? pair : pair * 2;
                        const auto right_index = neox ? pair + half_dim : pair * 2 + 1;
                        const float exponent = static_cast<float>(pair * 2) / static_cast<float>(head_dim);
                        float inverse_frequency = 1.0F / std::pow(frequency_base, exponent);
                        if (node.src.size() == 2 && src[1][pair] > 0.0F) inverse_frequency /= src[1][pair];
                        const float angle = position * inverse_frequency;
                        const float cosine = std::cos(angle);
                        const float sine = std::sin(angle);
                        const float left = src[0][base + left_index];
                        const float right = src[0][base + right_index];
                        dst[base + left_index] = left * cosine - right * sine;
                        dst[base + right_index] = left * sine + right * cosine;
                    }
                }
            }
            break;
        }
        case Op::causal_attention: {
            const auto heads = node.params[0];
            const auto kv_heads = node.params[1];
            const auto head_dim = node.params[2];
            const auto tokens = output.shape[0];
            const float scale = node.scalar > 0.0F ? node.scalar : 1.0F / std::sqrt(static_cast<float>(head_dim));
            const auto window = node.params.size() >= 4 ? node.params[3] : 0;
            const auto past_tokens = node.src.size() == 5
                ? (dynamics.past_tokens != ReplayDynamics::use_node_params ? dynamics.past_tokens : node.params[4])
                : 0;
            const bool independent_rows =
                node.params.size() >= 6 && node.params[5] != 0;
            streaming_attention_f32(
                {src[0], tokens * heads * head_dim},
                {src[1], tokens * kv_heads * head_dim},
                {src[2], tokens * kv_heads * head_dim},
                node.src.size() == 5 ? std::span<const float>(src[3], graph.tensor(node.src[3]).elements()) : std::span<const float>{},
                node.src.size() == 5 ? std::span<const float>(src[4], graph.tensor(node.src[4]).elements()) : std::span<const float>{},
                {dst, total},
                tokens,
                heads,
                kv_heads,
                head_dim,
                past_tokens,
                window,
                scale,
                independent_rows);
            break;
        }
        case Op::silu_mul:
            for (std::size_t index = 0; index < total; ++index) {
                const float gate = src[0][index];
                dst[index] = (gate / (1.0F + std::exp(-gate))) * src[1][index];
            }
            break;
        case Op::gelu_mul:
            for (std::size_t index = 0; index < total; ++index) {
                const float gate = src[0][index];
                const float cubic = gate * gate * gate;
                const float activated = 0.5F * gate * (1.0F + std::tanh(0.7978845608028654F * (gate + 0.044715F * cubic)));
                dst[index] = activated * src[1][index];
            }
            break;
        case Op::add: {
            const auto rhs_elements = graph.tensor(node.src[1]).elements();
            for (std::size_t index = 0; index < total; ++index) {
                dst[index] = src[0][index] + src[1][index % rhs_elements];
            }
            break;
        }
        case Op::scale:
            for (std::size_t index = 0; index < total; ++index) {
                dst[index] = src[0][index] * (node.src.size() == 2 ? src[1][0] : node.scalar);
            }
            break;
    }
}

Plan Plan::compile(const Graph& graph) {
    return compile(graph, {});
}

Plan Plan::compile(const Graph& graph, const std::vector<SpanBackend>& node_backends) {
    graph.validate();
    if (!node_backends.empty() && node_backends.size() != graph.nodes().size()) {
        throw std::invalid_argument("node backend assignment does not cover the graph");
    }
    Plan plan;
    plan.graph_ = graph;
    plan.allocation_ = plan_activations(graph);
    const auto node_count = graph.nodes().size();
    if (node_backends.empty()) {
        plan.spans_.push_back(PlanSpan{0, node_count, SpanBackend::cpu});
    } else {
        std::size_t begin = 0;
        while (begin < node_count) {
            std::size_t end = begin + 1;
            while (end < node_count && node_backends[end] == node_backends[begin]) ++end;
            plan.spans_.push_back(PlanSpan{begin, end - begin, node_backends[begin]});
            begin = end;
        }
    }
    plan.arena_.resize((plan.allocation_.arena_bytes + sizeof(float) - 1) / sizeof(float));
    return plan;
}

std::vector<PlanSpan> Plan::cpu_spans() const {
    std::vector<PlanSpan> result;
    for (const auto& span : spans_) {
        if (span.backend == SpanBackend::cpu) result.push_back(span);
    }
    return result;
}

std::size_t Plan::gpu_span_count() const {
    std::size_t count = 0;
    for (const auto& span : spans_) {
        if (span.backend == SpanBackend::gpu) ++count;
    }
    return count;
}

Bindings Plan::replay(const Bindings& external) const {
    return replay(external, {});
}

Bindings Plan::replay(const Bindings& external, const QuantizedBindings& quantized) const {
    BindingViews views;
    views.reserve(external.size());
    for (const auto& [id, values] : external) views.emplace(id, values);
    return replay_views(views, quantized);
}

Bindings Plan::replay_views(
    const BindingViews& external,
    const QuantizedBindings& quantized,
    const ReplayDynamics& dynamics,
    GpuExecutor* gpu
) const {
    const auto pointer_for = [&](TensorId id) -> const float* {
        const auto& tensor = graph_.tensor(id);
        if (tensor.external) {
            const auto found = external.find(id);
            if (found == external.end()) {
                if (quantized.contains(id)) return nullptr;
                throw std::invalid_argument("missing external tensor: " + tensor.name);
            }
            if (found->second.size() != tensor.elements()) {
                throw std::invalid_argument("binding size mismatch for tensor: " + tensor.name);
            }
            return found->second.data();
        }
        const auto found = allocation_.tensors.find(id);
        if (found == allocation_.tensors.end()) throw std::logic_error("planned tensor has no activation allocation");
        return reinterpret_cast<const float*>(reinterpret_cast<const std::byte*>(arena_.data()) + found->second.offset);
    };
    const auto mutable_pointer_for = [&](TensorId id) -> float* {
        const auto found = allocation_.tensors.find(id);
        if (found == allocation_.tensors.end()) throw std::logic_error("planned destination has no activation allocation");
        return reinterpret_cast<float*>(reinterpret_cast<std::byte*>(arena_.data()) + found->second.offset);
    };

    for (const auto& span : spans_) {
        if (span.backend == SpanBackend::gpu) {
            if (gpu == nullptr) throw std::logic_error("plan contains GPU spans but no GPU executor was provided");
            gpu->execute_span(*this, span, external, dynamics, std::span<float>(arena_.data(), arena_.size()));
            continue;
        }
        const bool profile_stages = stage_profile_enabled();
        for (std::size_t index = span.first_node; index < span.first_node + span.node_count; ++index) {
            const auto& node = graph_.nodes()[index];
            const auto stage_started = profile_stages
                ? std::chrono::steady_clock::now()
                : std::chrono::steady_clock::time_point{};
            std::array<const float*, 5> sources{nullptr, nullptr, nullptr, nullptr, nullptr};
            for (std::size_t source_index = 0; source_index < node.src.size(); ++source_index) {
                sources[source_index] = pointer_for(node.src[source_index]);
            }
            if (node.op == Op::matmul && sources[1] == nullptr) {
                const auto found = quantized.find(node.src[1]);
                if (found == quantized.end()) throw std::logic_error("missing quantized matmul binding");
                const auto& weight = graph_.tensor(node.src[1]);
                const auto columns = weight.shape[1];
                const auto batches = graph_.tensor(node.src[0]).elements() / columns;
                float* destination = mutable_pointer_for(node.dst);
                const auto multiplied = tensor_matrix_multiply(
                    found->second,
                    std::span<const float>(sources[0], batches * columns),
                    batches
                );
                std::copy(multiplied.begin(), multiplied.end(), destination);
            } else {
                for (std::size_t source_index = 0; source_index < node.src.size(); ++source_index) {
                    if (sources[source_index] == nullptr) {
                        throw std::invalid_argument("quantized binding is only valid for a matmul weight");
                    }
                }
                execute_node_reference(graph_, node, sources, mutable_pointer_for(node.dst), dynamics);
            }
            if (profile_stages) {
                record_stage(static_cast<std::size_t>(node.op), stage_started);
            }
        }
    }

    Bindings outputs;
    for (const auto output : graph_.outputs()) {
        const float* pointer = pointer_for(output);
        outputs.emplace(output, std::vector<float>(pointer, pointer + graph_.tensor(output).elements()));
    }
    return outputs;
}

Bindings imperative_reference(const Graph& graph, const Bindings& external) {
    graph.validate();
    Bindings values = external;
    for (const auto& tensor : graph.tensors()) {
        if (tensor.external) {
            const auto found = values.find(tensor.id);
            if (found == values.end()) throw std::invalid_argument("missing reference binding: " + tensor.name);
            require_binding_size(tensor, found->second);
        }
    }
    for (const auto& node : graph.nodes()) {
        std::array<const float*, 5> sources{nullptr, nullptr, nullptr, nullptr, nullptr};
        for (std::size_t source_index = 0; source_index < node.src.size(); ++source_index) {
            sources[source_index] = values.at(node.src[source_index]).data();
        }
        auto& destination = values[node.dst];
        destination.resize(graph.tensor(node.dst).elements());
        execute_node_reference(graph, node, sources, destination.data(), {});
    }
    Bindings outputs;
    for (const auto output : graph.outputs()) outputs.emplace(output, values.at(output));
    return outputs;
}

bool bit_equal(const std::vector<float>& left, const std::vector<float>& right) {
    return left.size() == right.size()
        && std::memcmp(left.data(), right.data(), left.size() * sizeof(float)) == 0;
}

}  // namespace cat
