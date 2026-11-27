// Compute-graph shape inspired by public graph-runtime architecture patterns.
// Clean-room re-implementation; no upstream source was copied.

#include "graph.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <unordered_set>

namespace cat {

std::size_t Tensor::elements() const {
    if (shape.empty()) return 0;
    std::size_t count = 1;
    for (const auto dimension : shape) {
        if (dimension == 0 || count > std::numeric_limits<std::size_t>::max() / dimension) {
            throw std::overflow_error("invalid or overflowing tensor shape: " + name);
        }
        count *= dimension;
    }
    return count;
}

std::size_t Tensor::bytes() const {
    const auto count = elements();
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
        throw std::overflow_error("tensor byte size overflow: " + name);
    }
    return count * sizeof(float);
}

TensorId Graph::add_tensor(std::string name, std::vector<std::size_t> shape, bool external) {
    const TensorId id = tensors_.size();
    tensors_.push_back(Tensor{id, std::move(name), std::move(shape), external});
    static_cast<void>(tensors_.back().bytes());
    return id;
}

void Graph::add_node(
    Op op,
    std::vector<TensorId> src,
    TensorId dst,
    float scalar,
    std::vector<std::size_t> params,
    float secondary_scalar
) {
    nodes_.push_back(Node{op, std::move(src), dst, scalar, std::move(params), secondary_scalar});
}

void Graph::set_outputs(std::vector<TensorId> outputs) {
    outputs_ = std::move(outputs);
}

const Tensor& Graph::tensor(TensorId id) const {
    if (id >= tensors_.size()) throw std::out_of_range("unknown tensor id");
    return tensors_[id];
}

void Graph::validate() const {
    if (tensors_.empty()) throw std::invalid_argument("graph has no tensors");
    if (nodes_.empty()) throw std::invalid_argument("graph has no nodes");
    if (outputs_.empty()) throw std::invalid_argument("graph has no outputs");

    std::unordered_set<TensorId> available;
    for (const auto& item : tensors_) {
        static_cast<void>(item.bytes());
        if (item.external) available.insert(item.id);
    }

    for (const auto& node : nodes_) {
        if (node.dst >= tensors_.size()) throw std::invalid_argument("node destination is unknown");
        if (tensor(node.dst).external) throw std::invalid_argument("node cannot write an external tensor");
        if (available.contains(node.dst)) throw std::invalid_argument("tensor has multiple producers");
        for (const auto source : node.src) {
            if (source >= tensors_.size() || !available.contains(source)) {
                throw std::invalid_argument("graph is not topologically ordered or references an unknown source");
            }
        }

        const auto dst_elements = tensor(node.dst).elements();
        switch (node.op) {
            case Op::rms_norm:
                if (node.src.size() != 2 || tensor(node.src[0]).elements() != dst_elements) {
                    throw std::invalid_argument("RMS norm requires data + scale and a shape-preserving output");
                }
                if (tensor(node.src[1]).shape.size() != 1
                    || tensor(node.src[1]).elements() != tensor(node.dst).shape.back()) {
                    throw std::invalid_argument("RMS norm scale must match the final dimension");
                }
                break;
            case Op::rms_norm_heads:
                if (node.src.size() != 2 || tensor(node.src[0]).elements() != dst_elements
                    || node.params.size() != 2 || tensor(node.src[1]).shape != std::vector<std::size_t>{node.params[1]}) {
                    throw std::invalid_argument("head RMS norm requires data, per-head scale, heads, and head dimension");
                }
                if (node.params[0] * node.params[1] != tensor(node.dst).shape.back()) {
                    throw std::invalid_argument("head RMS norm dimensions do not match the tensor");
                }
                break;
            case Op::rms_norm_heads_unweighted:
                if (node.src.size() != 1 || tensor(node.src[0]).elements() != dst_elements
                    || node.params.size() != 2 || node.params[0] * node.params[1] != tensor(node.dst).shape.back()) {
                    throw std::invalid_argument("unweighted head RMS norm dimensions do not match the tensor");
                }
                break;
            case Op::matmul: {
                if (node.src.size() != 2 || tensor(node.src[1]).shape.size() != 2) {
                    throw std::invalid_argument("matmul requires an activation tensor and a 2D weight tensor");
                }
                const auto& input = tensor(node.src[0]);
                const auto& weight = tensor(node.src[1]);
                const auto columns = weight.shape[1];
                const auto rows = weight.shape[0];
                if (input.shape.back() != columns || dst_elements != (input.elements() / columns) * rows) {
                    throw std::invalid_argument("matmul tensor shapes do not agree");
                }
                break;
            }
            case Op::rope: {
                if ((node.src.size() != 1 && node.src.size() != 2) || tensor(node.src[0]).elements() != dst_elements
                    || (node.params.size() < 3 || node.params.size() > 5)) {
                    throw std::invalid_argument("RoPE requires a shape-preserving source and heads/head-dim/position parameters");
                }
                const auto heads = node.params[0];
                const auto head_dim = node.params[1];
                if (heads == 0 || head_dim == 0 || head_dim % 2 != 0 || heads * head_dim != tensor(node.dst).shape.back()) {
                    throw std::invalid_argument("RoPE head shape does not match the final tensor dimension");
                }
                if (!(node.secondary_scalar > 0.0F)) throw std::invalid_argument("RoPE frequency scale must be positive");
                if (node.src.size() == 2 && tensor(node.src[1]).shape != std::vector<std::size_t>{head_dim / 2}) {
                    throw std::invalid_argument("RoPE frequency factors must contain half one head dimension");
                }
                break;
            }
            case Op::causal_attention: {
                const bool cached = node.src.size() == 5
                    && (node.params.size() == 5 || node.params.size() == 6);
                const bool uncached = node.src.size() == 3 && (node.params.size() >= 3 && node.params.size() <= 5);
                if ((!cached && !uncached) || tensor(node.dst).shape.size() != 2) {
                    throw std::invalid_argument("causal attention requires Q/K/V plus heads/kv-heads/head-dim parameters");
                }
                const auto heads = node.params[0];
                const auto kv_heads = node.params[1];
                const auto head_dim = node.params[2];
                const auto tokens = tensor(node.dst).shape[0];
                if (heads == 0 || kv_heads == 0 || head_dim == 0 || heads % kv_heads != 0
                    || tensor(node.dst).shape[1] != heads * head_dim
                    || tensor(node.src[0]).shape != tensor(node.dst).shape
                    || tensor(node.src[1]).shape != std::vector<std::size_t>{tokens, kv_heads * head_dim}
                    || tensor(node.src[2]).shape != tensor(node.src[1]).shape) {
                    throw std::invalid_argument("causal attention tensor/head shapes do not agree");
                }
                if (cached) {
                    const auto past_tokens = node.params[4];
                    const bool independent_rows =
                        node.params.size() == 6 && node.params[5] != 0;
                    const auto past_shape = std::vector<std::size_t>{
                        past_tokens * (independent_rows ? tokens : 1),
                        kv_heads * head_dim
                    };
                    if (past_tokens == 0 || tensor(node.src[3]).shape != past_shape
                        || tensor(node.src[4]).shape != past_shape) {
                        throw std::invalid_argument("cached attention past K/V shapes do not agree");
                    }
                }
                break;
            }
            case Op::silu_mul:
            case Op::gelu_mul:
                if (node.src.size() != 2 || tensor(node.src[0]).elements() != dst_elements
                    || tensor(node.src[1]).elements() != dst_elements) {
                    throw std::invalid_argument("SiLU multiply requires equal shapes");
                }
                break;
            case Op::scale:
                if ((node.src.size() != 1 && node.src.size() != 2) || tensor(node.src[0]).elements() != dst_elements
                    || (node.src.size() == 2 && tensor(node.src[1]).elements() != 1)) {
                    throw std::invalid_argument("scale requires one shape-preserving source and an optional scalar tensor");
                }
                break;
            case Op::add: {
                if (node.src.size() != 2 || tensor(node.src[0]).elements() != dst_elements) {
                    throw std::invalid_argument("add requires a shape-preserving first input");
                }
                const auto rhs = tensor(node.src[1]).elements();
                if (rhs != dst_elements && rhs != tensor(node.dst).shape.back()) {
                    throw std::invalid_argument("add supports equal shapes or final-dimension broadcast");
                }
                break;
            }
        }
        available.insert(node.dst);
    }
    for (const auto output : outputs_) {
        if (!available.contains(output)) throw std::invalid_argument("graph output is unavailable");
    }
}

bool op_can_inplace(Op op, std::size_t source_index) {
    return source_index == 0 && (op == Op::rms_norm || op == Op::rms_norm_heads
        || op == Op::rms_norm_heads_unweighted || op == Op::rope || op == Op::silu_mul
        || op == Op::gelu_mul || op == Op::add || op == Op::scale);
}

}  // namespace cat
