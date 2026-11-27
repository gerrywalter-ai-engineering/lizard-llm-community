#pragma once

#include "gguf_loader.h"
#include "graph.h"

#include <cstddef>
#include <string>
#include <vector>

namespace cat {

struct LayerSpec {
    std::size_t head_dim{};
    std::size_t query_hidden{};
    std::size_t kv_hidden{};
    std::size_t attention_window{};
    float rope_frequency_base{10000.0F};
    std::size_t kv_source_layer{};
};

struct FamilySpec {
    std::string architecture;
    std::string family;
    std::size_t layers{};
    std::size_t hidden_size{};
    std::size_t feed_forward_size{};
    std::size_t heads{};
    std::size_t kv_heads{};
    std::size_t head_dim{};
    float rms_epsilon{1.0e-5F};
    float rope_frequency_base{10000.0F};
    float rope_frequency_scale{1.0F};
    bool rope_neox{false};
    bool rope_frequency_factors{false};
    bool attention_biases{false};
    bool gemma4_extended{false};
    std::size_t per_layer_input_size{};
    std::vector<LayerSpec> layer_specs;

    void validate() const;
};

[[nodiscard]] FamilySpec resolve_family_spec(const GgufModelInfo& model);

struct DecoderGraph {
    struct KvBinding {
        std::size_t layer{};
        TensorId key{};
        TensorId value{};
    };

    Graph graph;
    TensorId input{};
    TensorId output{};
    std::vector<TensorId> parameters;
    std::vector<TensorId> runtime_inputs;
    std::vector<KvBinding> kv_inputs;
    std::vector<KvBinding> kv_outputs;
};

[[nodiscard]] DecoderGraph build_decoder_graph(
    const FamilySpec& family,
    std::size_t tokens,
    std::size_t position_offset = 0,
    std::size_t past_tokens = 0,
    bool capture_kv = false,
    bool independent_rows = false
);

int run_family_graph_selftest();

}  // namespace cat
