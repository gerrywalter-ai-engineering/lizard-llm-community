#include "selftest.h"

#include "graph.h"
#include "plan.h"

#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace cat {
namespace {

struct TestGraph {
    Graph graph;
    TensorId input{};
    TensorId norm_scale{};
    TensorId weight{};
    TensorId bias{};
    TensorId output{};
};

TestGraph build_test_graph(std::size_t batch) {
    TestGraph test;
    test.input = test.graph.add_tensor("input", {batch, 4}, true);
    test.norm_scale = test.graph.add_tensor("norm_scale", {4}, true);
    test.weight = test.graph.add_tensor("projection_weight", {3, 4}, true);
    test.bias = test.graph.add_tensor("bias", {3}, true);
    const auto normalized = test.graph.add_tensor("normalized", {batch, 4});
    const auto projected = test.graph.add_tensor("projected", {batch, 3});
    test.output = test.graph.add_tensor("logits", {batch, 3});
    test.graph.add_node(Op::rms_norm, {test.input, test.norm_scale}, normalized, 1.0e-5F);
    test.graph.add_node(Op::matmul, {normalized, test.weight}, projected);
    test.graph.add_node(Op::add, {projected, test.bias}, test.output);
    test.graph.set_outputs({test.output});
    return test;
}

Bindings test_bindings(const TestGraph& test, std::size_t batch, float offset) {
    std::vector<float> input;
    input.reserve(batch * 4);
    for (std::size_t index = 0; index < batch * 4; ++index) {
        input.push_back(static_cast<float>(index + 1) * 0.125F + offset);
    }
    return {
        {test.input, std::move(input)},
        {test.norm_scale, {1.0F, 0.5F, 1.5F, 0.75F}},
        {test.weight, {
            0.25F, -0.5F, 0.75F, 1.0F,
            -0.25F, 0.125F, 0.5F, -0.75F,
            1.0F, 0.5F, -0.5F, 0.25F,
        }},
        {test.bias, {0.125F, -0.25F, 0.5F}},
    };
}

void verify_shape(std::string_view label, std::size_t batch) {
    const auto test = build_test_graph(batch);
    const auto plan = Plan::compile(test.graph);
    if (plan.cpu_spans().size() != 1 || plan.cpu_spans()[0].node_count != 3) {
        throw std::runtime_error(std::string(label) + ": CPU nodes were not coalesced into one plan span");
    }
    if (plan.allocation().arena_bytes >= plan.allocation().naive_bytes) {
        throw std::runtime_error(std::string(label) + ": activation lifetime/in-place reuse did not reduce the arena");
    }
    for (const float offset : {0.0F, 0.375F}) {
        const auto bindings = test_bindings(test, batch, offset);
        const auto reference = imperative_reference(test.graph, bindings);
        const auto replay = plan.replay(bindings);
        if (!bit_equal(reference.at(test.output), replay.at(test.output))) {
            throw std::runtime_error(std::string(label) + ": plan replay differs from imperative reference");
        }
    }
}

void verify_quantized_replay() {
    Graph graph;
    const auto input_id = graph.add_tensor("quant_input", {1, 32}, true);
    const auto weight_id = graph.add_tensor("quant_weight", {2, 32}, true);
    const auto output_id = graph.add_tensor("quant_output", {1, 2});
    graph.add_node(Op::matmul, {input_id, weight_id}, output_id);
    graph.set_outputs({output_id});
    const auto plan = Plan::compile(graph);

    GgufTensorInfo tensor{"quant_weight", {32, 2}, 2, 0};
    std::vector<std::byte> bytes(36, std::byte{0});
    for (std::size_t row = 0; row < 2; ++row) {
        bytes[row * 18] = std::byte{0x00};
        bytes[row * 18 + 1] = std::byte{0x3c};
        for (std::size_t index = 0; index < 16; ++index) {
            bytes[row * 18 + 2 + index] = row == 0 ? std::byte{0x98} : std::byte{0x76};
        }
    }
    std::vector<float> input(32);
    for (std::size_t index = 0; index < input.size(); ++index) input[index] = static_cast<float>(index + 1) / 32.0F;
    const Bindings float_bindings{{input_id, input}, {weight_id, dequantize_tensor(tensor, bytes)}};
    const Bindings quant_inputs{{input_id, input}};
    const QuantizedBindings quant_bindings{{weight_id, QuantizedTensorView{&tensor, bytes}}};
    const auto float_output = plan.replay(float_bindings).at(output_id);
    const auto quant_output = plan.replay(quant_inputs, quant_bindings).at(output_id);
    if (!bit_equal(float_output, quant_output)) throw std::runtime_error("quantized matmul replay differs from dequantized reference");
}

}  // namespace

int run_graph_selftest() {
    try {
        verify_shape("decode-shape", 1);
        verify_shape("prefill-shape", 3);
        verify_quantized_replay();
        std::cout << "caterpillar graph replay selftest: ok (decode + prefill + quantized matmul bit-exact)\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "caterpillar graph replay selftest: failed: " << error.what() << '\n';
        return 1;
    }
}

}  // namespace cat
