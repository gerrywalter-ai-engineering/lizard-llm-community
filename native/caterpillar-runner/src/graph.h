#pragma once

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

namespace cat {

using TensorId = std::size_t;

enum class Op {
    rms_norm,
    rms_norm_heads,
    rms_norm_heads_unweighted,
    matmul,
    rope,
    causal_attention,
    silu_mul,
    gelu_mul,
    add,
    scale,
};

struct Tensor {
    TensorId id{};
    std::string name;
    std::vector<std::size_t> shape;
    bool external{false};

    [[nodiscard]] std::size_t elements() const;
    [[nodiscard]] std::size_t bytes() const;
};

struct Node {
    Op op{Op::add};
    std::vector<TensorId> src;
    TensorId dst{};
    float scalar{1.0e-5F};
    std::vector<std::size_t> params;
    float secondary_scalar{1.0F};
};

class Graph {
public:
    TensorId add_tensor(std::string name, std::vector<std::size_t> shape, bool external = false);
    void add_node(
        Op op,
        std::vector<TensorId> src,
        TensorId dst,
        float scalar = 1.0e-5F,
        std::vector<std::size_t> params = {},
        float secondary_scalar = 1.0F
    );
    void set_outputs(std::vector<TensorId> outputs);

    [[nodiscard]] const Tensor& tensor(TensorId id) const;
    [[nodiscard]] const std::vector<Tensor>& tensors() const noexcept { return tensors_; }
    [[nodiscard]] const std::vector<Node>& nodes() const noexcept { return nodes_; }
    [[nodiscard]] const std::vector<TensorId>& outputs() const noexcept { return outputs_; }
    void validate() const;

private:
    std::vector<Tensor> tensors_;
    std::vector<Node> nodes_;
    std::vector<TensorId> outputs_;
};

[[nodiscard]] bool op_can_inplace(Op op, std::size_t source_index);

}  // namespace cat
