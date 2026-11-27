// FamilySpec-driven decoder graph construction for the initial
// Llama/Gemma/Qwen2 families. Clean-room implementation.

#include "family_spec.h"

#include "plan.h"

#include <cmath>
#include <algorithm>
#include <iostream>
#include <stdexcept>

namespace cat {
namespace {

TensorId parameter(
    DecoderGraph& decoder,
    std::string name,
    std::vector<std::size_t> shape
) {
    const auto id = decoder.graph.add_tensor(std::move(name), std::move(shape), true);
    decoder.parameters.push_back(id);
    return id;
}

const GgufTensorInfo& required_tensor(const GgufModelInfo& model, const std::string& name) {
    const auto found = std::find_if(model.tensors.begin(), model.tensors.end(), [&](const GgufTensorInfo& tensor) {
        return tensor.name == name;
    });
    if (found == model.tensors.end()) throw std::invalid_argument("FamilySpec requires GGUF tensor: " + name);
    return *found;
}

}  // namespace

void FamilySpec::validate() const {
    const bool supported_family = family == "llama" || family == "gemma" || family == "qwen2";
    if (!supported_family) {
        throw std::invalid_argument("Caterpillar Phase 1 supports llama, gemma, and qwen2 decoder families");
    }
    if (layers == 0 || hidden_size == 0 || feed_forward_size == 0 || heads == 0 || kv_heads == 0) {
        throw std::invalid_argument("FamilySpec is missing required decoder dimensions");
    }
    if (heads % kv_heads != 0 || head_dim == 0 || head_dim % 2 != 0) {
        throw std::invalid_argument("FamilySpec attention dimensions are inconsistent");
    }
    if (!(rms_epsilon > 0.0F) || !(rope_frequency_base > 1.0F)) {
        throw std::invalid_argument("FamilySpec numeric parameters are invalid");
    }
}

FamilySpec resolve_family_spec(const GgufModelInfo& model) {
    FamilySpec spec;
    spec.architecture = model.architecture;
    spec.family = model.family;
    if (spec.family == "mistral") spec.family = "llama";
    spec.layers = static_cast<std::size_t>(model.block_count);
    spec.hidden_size = static_cast<std::size_t>(model.embedding_length);
    spec.feed_forward_size = static_cast<std::size_t>(model.feed_forward_length);
    spec.heads = static_cast<std::size_t>(model.head_count);
    spec.kv_heads = static_cast<std::size_t>(model.head_count_kv == 0 ? model.head_count : model.head_count_kv);
    // hidden/heads is only the head dimension when the architecture keeps them
    // in step. Gemma 3 states attention.key_length 256 against a 2560 hidden
    // size over 8 heads, so deriving 320 built attn_q at 2560x2560 against a
    // 2560x2048 tensor and load failed on the shape. An explicit key_length
    // wins; rope.dimension_count remains the older signal.
    spec.head_dim = spec.heads == 0 ? 0 : spec.hidden_size / spec.heads;
    if (model.key_length != 0) spec.head_dim = static_cast<std::size_t>(model.key_length);
    else if (model.rope_dimension_count != 0) spec.head_dim = static_cast<std::size_t>(model.rope_dimension_count);
    spec.rms_epsilon = model.rms_epsilon;
    spec.rope_frequency_base = model.rope_frequency_base;
    spec.rope_frequency_scale = model.rope_frequency_scale;
    spec.rope_neox = spec.family == "qwen2" || spec.family == "gemma";
    spec.rope_frequency_factors = std::any_of(
        model.tensors.begin(),
        model.tensors.end(),
        [](const GgufTensorInfo& tensor) { return tensor.name == "rope_freqs.weight"; });
    spec.attention_biases = model.family == "qwen2";
    spec.gemma4_extended = model.architecture == "gemma4";
    spec.per_layer_input_size = static_cast<std::size_t>(model.embedding_length_per_layer_input);
    if (spec.gemma4_extended) {
        spec.layer_specs.reserve(spec.layers);
        const auto shared = static_cast<std::size_t>(model.attention_shared_kv_layers);
        const auto first_shared = shared == 0 ? spec.layers : spec.layers - shared;
        for (std::size_t layer = 0; layer < spec.layers; ++layer) {
            const auto prefix = "blk." + std::to_string(layer) + ".";
            const auto query_name = prefix + "attn_q.weight";
            const auto key_name = prefix + "attn_k.weight";
            const auto& query = required_tensor(model, query_name);
            const auto& key = required_tensor(model, key_name);
            if (query.dimensions.size() != 2 || key.dimensions.size() != 2
                || query.dimensions[1] % spec.heads != 0 || key.dimensions[1] % spec.kv_heads != 0) {
                throw std::invalid_argument("Gemma4 attention projection shapes are invalid at layer " + std::to_string(layer));
            }
            const auto layer_head_dim = static_cast<std::size_t>(query.dimensions[1] / spec.heads);
            if (key.dimensions[1] / spec.kv_heads != layer_head_dim) {
                throw std::invalid_argument("Gemma4 Q/K head dimensions differ at layer " + std::to_string(layer));
            }
            const bool sliding = model.attention_key_length_swa != 0
                && layer_head_dim == static_cast<std::size_t>(model.attention_key_length_swa);
            std::size_t source = layer;
            if (shared != 0 && layer >= first_shared) {
                const auto distance = sliding ? std::size_t{2} : std::size_t{1};
                if (first_shared < distance) throw std::invalid_argument("Gemma4 shared KV source is invalid");
                source = first_shared - distance;
            }
            spec.layer_specs.push_back(LayerSpec{
                .head_dim = layer_head_dim,
                .query_hidden = static_cast<std::size_t>(query.dimensions[1]),
                .kv_hidden = static_cast<std::size_t>(key.dimensions[1]),
                .attention_window = sliding ? static_cast<std::size_t>(model.attention_sliding_window) : 0,
                .rope_frequency_base = sliding ? model.rope_frequency_base_swa : model.rope_frequency_base,
                .kv_source_layer = source,
            });
        }
    }
    spec.validate();
    return spec;
}

DecoderGraph build_decoder_graph(
    const FamilySpec& family,
    std::size_t tokens,
    std::size_t position_offset,
    std::size_t past_tokens,
    bool capture_kv,
    bool independent_rows
) {
    family.validate();
    if (tokens == 0) throw std::invalid_argument("decoder graph requires at least one token");
    DecoderGraph decoder;
    const auto hidden = family.hidden_size;
    decoder.input = decoder.graph.add_tensor("token_state", {tokens, hidden}, true);
    TensorId state = decoder.input;
    std::vector<TensorId> kv_keys(family.layers);
    std::vector<TensorId> kv_values(family.layers);
    std::vector<TensorId> past_kv_keys(family.layers);
    std::vector<TensorId> past_kv_values(family.layers);

    for (std::size_t layer = 0; layer < family.layers; ++layer) {
        const auto prefix = "blk." + std::to_string(layer) + ".";
        const auto layer_spec = family.layer_specs.empty()
            ? LayerSpec{family.head_dim, family.heads * family.head_dim, family.kv_heads * family.head_dim, 0, family.rope_frequency_base, layer}
            : family.layer_specs.at(layer);
        const auto query_hidden = layer_spec.query_hidden;
        const auto kv_hidden = layer_spec.kv_hidden;
        const auto head_dim = layer_spec.head_dim;
        const auto attention_norm = parameter(decoder, prefix + "attn_norm.weight", {hidden});
        const auto q_weight = parameter(decoder, prefix + "attn_q.weight", {query_hidden, hidden});
        const auto o_weight = parameter(decoder, prefix + "attn_output.weight", {hidden, query_hidden});
        const auto ffn_norm = parameter(decoder, prefix + "ffn_norm.weight", {hidden});
        const auto gate_weight = parameter(decoder, prefix + "ffn_gate.weight", {family.feed_forward_size, hidden});
        const auto up_weight = parameter(decoder, prefix + "ffn_up.weight", {family.feed_forward_size, hidden});
        const auto down_weight = parameter(decoder, prefix + "ffn_down.weight", {hidden, family.feed_forward_size});

        const auto normalized = decoder.graph.add_tensor(prefix + "attn_norm", {tokens, hidden});
        auto query = decoder.graph.add_tensor(prefix + "query", {tokens, query_hidden});
        decoder.graph.add_node(Op::rms_norm, {state, attention_norm}, normalized, family.rms_epsilon);
        decoder.graph.add_node(Op::matmul, {normalized, q_weight}, query);
        TensorId key{};
        TensorId value{};
        const bool owns_kv = layer_spec.kv_source_layer == layer;
        if (owns_kv) {
            const auto k_weight = parameter(decoder, prefix + "attn_k.weight", {kv_hidden, hidden});
            const auto v_weight = parameter(decoder, prefix + "attn_v.weight", {kv_hidden, hidden});
            key = decoder.graph.add_tensor(prefix + "key", {tokens, kv_hidden});
            value = decoder.graph.add_tensor(prefix + "value", {tokens, kv_hidden});
            decoder.graph.add_node(Op::matmul, {normalized, k_weight}, key);
            decoder.graph.add_node(Op::matmul, {normalized, v_weight}, value);
        }

        if (family.attention_biases) {
            const auto q_bias = parameter(decoder, prefix + "attn_q.bias", {query_hidden});
            const auto biased_query = decoder.graph.add_tensor(prefix + "biased_query", {tokens, query_hidden});
            decoder.graph.add_node(Op::add, {query, q_bias}, biased_query);
            query = biased_query;
            if (owns_kv) {
                const auto k_bias = parameter(decoder, prefix + "attn_k.bias", {kv_hidden});
                const auto v_bias = parameter(decoder, prefix + "attn_v.bias", {kv_hidden});
                const auto biased_key = decoder.graph.add_tensor(prefix + "biased_key", {tokens, kv_hidden});
                const auto biased_value = decoder.graph.add_tensor(prefix + "biased_value", {tokens, kv_hidden});
                decoder.graph.add_node(Op::add, {key, k_bias}, biased_key);
                decoder.graph.add_node(Op::add, {value, v_bias}, biased_value);
                key = biased_key;
                value = biased_value;
            }
        }

        if (family.gemma4_extended) {
            const auto query_norm_weight = parameter(decoder, prefix + "attn_q_norm.weight", {head_dim});
            const auto normalized_query = decoder.graph.add_tensor(prefix + "normalized_query", {tokens, query_hidden});
            decoder.graph.add_node(Op::rms_norm_heads, {query, query_norm_weight}, normalized_query, family.rms_epsilon, {family.heads, head_dim});
            query = normalized_query;
            if (owns_kv) {
                const auto key_norm_weight = parameter(decoder, prefix + "attn_k_norm.weight", {head_dim});
                const auto normalized_key = decoder.graph.add_tensor(prefix + "normalized_key", {tokens, kv_hidden});
                const auto normalized_value = decoder.graph.add_tensor(prefix + "normalized_value", {tokens, kv_hidden});
                decoder.graph.add_node(Op::rms_norm_heads, {key, key_norm_weight}, normalized_key, family.rms_epsilon, {family.kv_heads, head_dim});
                decoder.graph.add_node(Op::rms_norm_heads_unweighted, {value}, normalized_value, family.rms_epsilon, {family.kv_heads, head_dim});
                key = normalized_key;
                value = normalized_value;
            }
        }

        const auto rotated_query = decoder.graph.add_tensor(prefix + "rotated_query", {tokens, query_hidden});
        std::vector<TensorId> query_rope_sources{query};
        std::vector<TensorId> key_rope_sources;
        if (owns_kv) key_rope_sources.push_back(key);
        if (family.rope_frequency_factors
            && (!family.gemma4_extended || layer_spec.attention_window == 0)) {
            const auto rope_factors = parameter(decoder, "rope_freqs.weight", {head_dim / 2});
            query_rope_sources.push_back(rope_factors);
            if (owns_kv) key_rope_sources.push_back(rope_factors);
        }
        std::vector<std::size_t> query_rope_params{
            family.heads, head_dim, position_offset, family.rope_neox ? 1U : 0U
        };
        if (independent_rows) query_rope_params.push_back(1);
        decoder.graph.add_node(
            Op::rope,
            query_rope_sources,
            rotated_query,
            layer_spec.rope_frequency_base,
            std::move(query_rope_params),
            family.rope_frequency_scale
        );
        TensorId rotated_key{};
        if (owns_kv) {
            rotated_key = decoder.graph.add_tensor(prefix + "rotated_key", {tokens, kv_hidden});
            std::vector<std::size_t> key_rope_params{
                family.kv_heads, head_dim, position_offset, family.rope_neox ? 1U : 0U
            };
            if (independent_rows) key_rope_params.push_back(1);
            decoder.graph.add_node(
                Op::rope,
                key_rope_sources,
                rotated_key,
                layer_spec.rope_frequency_base,
                std::move(key_rope_params),
                family.rope_frequency_scale
            );
            kv_keys[layer] = rotated_key;
            kv_values[layer] = value;
            if (past_tokens != 0) {
                past_kv_keys[layer] = decoder.graph.add_tensor(
                    "runtime.kv." + std::to_string(layer) + ".key",
                    {past_tokens * (independent_rows ? tokens : 1), kv_hidden},
                    true
                );
                past_kv_values[layer] = decoder.graph.add_tensor(
                    "runtime.kv." + std::to_string(layer) + ".value",
                    {past_tokens * (independent_rows ? tokens : 1), kv_hidden},
                    true
                );
                decoder.kv_inputs.push_back({layer, past_kv_keys[layer], past_kv_values[layer]});
            }
            if (capture_kv) decoder.kv_outputs.push_back({layer, rotated_key, value});
        } else {
            if (layer_spec.kv_source_layer >= layer) throw std::invalid_argument("shared KV source must precede its consumer");
            rotated_key = kv_keys.at(layer_spec.kv_source_layer);
            value = kv_values.at(layer_spec.kv_source_layer);
        }
        const auto attended = decoder.graph.add_tensor(prefix + "attention", {tokens, query_hidden});
        std::vector<TensorId> attention_sources{rotated_query, rotated_key, value};
        if (past_tokens != 0) {
            const auto source_layer = layer_spec.kv_source_layer;
            attention_sources.push_back(past_kv_keys.at(source_layer));
            attention_sources.push_back(past_kv_values.at(source_layer));
        }
        std::vector<std::size_t> attention_params{
            family.heads, family.kv_heads, head_dim, layer_spec.attention_window, past_tokens
        };
        if (independent_rows) attention_params.push_back(1);
        decoder.graph.add_node(
            Op::causal_attention,
            std::move(attention_sources),
            attended,
            family.gemma4_extended ? 1.0F : 0.0F,
            std::move(attention_params)
        );
        const auto attention_projection = decoder.graph.add_tensor(prefix + "attention_projection", {tokens, hidden});
        decoder.graph.add_node(Op::matmul, {attended, o_weight}, attention_projection);
        TensorId attention_branch = attention_projection;
        if (family.gemma4_extended) {
            const auto post_attention_norm = parameter(decoder, prefix + "post_attention_norm.weight", {hidden});
            const auto normalized_attention = decoder.graph.add_tensor(prefix + "post_attention_norm", {tokens, hidden});
            decoder.graph.add_node(Op::rms_norm, {attention_projection, post_attention_norm}, normalized_attention, family.rms_epsilon);
            attention_branch = normalized_attention;
        }
        const auto attention_residual = decoder.graph.add_tensor(prefix + "attention_residual", {tokens, hidden});
        decoder.graph.add_node(Op::add, {attention_branch, state}, attention_residual);

        const auto ffn_input = decoder.graph.add_tensor(prefix + "ffn_norm", {tokens, hidden});
        const auto gate = decoder.graph.add_tensor(prefix + "gate", {tokens, family.feed_forward_size});
        const auto up = decoder.graph.add_tensor(prefix + "up", {tokens, family.feed_forward_size});
        const auto activated = decoder.graph.add_tensor(prefix + "activated", {tokens, family.feed_forward_size});
        const auto down = decoder.graph.add_tensor(prefix + "down", {tokens, hidden});
        decoder.graph.add_node(Op::rms_norm, {attention_residual, ffn_norm}, ffn_input, family.rms_epsilon);
        decoder.graph.add_node(Op::matmul, {ffn_input, gate_weight}, gate);
        decoder.graph.add_node(Op::matmul, {ffn_input, up_weight}, up);
        decoder.graph.add_node(family.gemma4_extended ? Op::gelu_mul : Op::silu_mul, {gate, up}, activated);
        decoder.graph.add_node(Op::matmul, {activated, down_weight}, down);
        TensorId ffn_branch = down;
        if (family.gemma4_extended) {
            const auto post_ffw_norm = parameter(decoder, prefix + "post_ffw_norm.weight", {hidden});
            const auto normalized_down = decoder.graph.add_tensor(prefix + "post_ffw_norm", {tokens, hidden});
            decoder.graph.add_node(Op::rms_norm, {down, post_ffw_norm}, normalized_down, family.rms_epsilon);
            ffn_branch = normalized_down;
        }
        const auto layer_output = decoder.graph.add_tensor(prefix + "output", {tokens, hidden});
        decoder.graph.add_node(Op::add, {ffn_branch, attention_residual}, layer_output);
        state = layer_output;

        if (family.gemma4_extended && family.per_layer_input_size != 0) {
            const auto ple_input = decoder.graph.add_tensor("runtime.ple." + std::to_string(layer), {tokens, family.per_layer_input_size}, true);
            decoder.runtime_inputs.push_back(ple_input);
            const auto input_gate_weight = parameter(decoder, prefix + "inp_gate.weight", {family.per_layer_input_size, hidden});
            const auto projection_weight = parameter(decoder, prefix + "proj.weight", {hidden, family.per_layer_input_size});
            const auto post_norm_weight = parameter(decoder, prefix + "post_norm.weight", {hidden});
            const auto output_scale_weight = parameter(decoder, prefix + "layer_output_scale.weight", {1});
            const auto ple_gate = decoder.graph.add_tensor(prefix + "ple_gate", {tokens, family.per_layer_input_size});
            const auto ple_activated = decoder.graph.add_tensor(prefix + "ple_activated", {tokens, family.per_layer_input_size});
            const auto ple_projection = decoder.graph.add_tensor(prefix + "ple_projection", {tokens, hidden});
            const auto ple_normalized = decoder.graph.add_tensor(prefix + "ple_post_norm", {tokens, hidden});
            const auto ple_residual = decoder.graph.add_tensor(prefix + "ple_residual", {tokens, hidden});
            const auto scaled_output = decoder.graph.add_tensor(prefix + "scaled_output", {tokens, hidden});
            decoder.graph.add_node(Op::matmul, {state, input_gate_weight}, ple_gate);
            decoder.graph.add_node(Op::gelu_mul, {ple_gate, ple_input}, ple_activated);
            decoder.graph.add_node(Op::matmul, {ple_activated, projection_weight}, ple_projection);
            decoder.graph.add_node(Op::rms_norm, {ple_projection, post_norm_weight}, ple_normalized, family.rms_epsilon);
            decoder.graph.add_node(Op::add, {state, ple_normalized}, ple_residual);
            decoder.graph.add_node(Op::scale, {ple_residual, output_scale_weight}, scaled_output);
            state = scaled_output;
        }
    }

    const auto final_norm = parameter(decoder, "output_norm.weight", {hidden});
    decoder.output = decoder.graph.add_tensor("output_state", {tokens, hidden});
    decoder.graph.add_node(Op::rms_norm, {state, final_norm}, decoder.output, family.rms_epsilon);
    std::vector<TensorId> outputs{decoder.output};
    for (const auto& kv : decoder.kv_outputs) {
        outputs.push_back(kv.key);
        outputs.push_back(kv.value);
    }
    decoder.graph.set_outputs(std::move(outputs));
    decoder.graph.validate();
    return decoder;
}

int run_family_graph_selftest() {
    try {
        const auto bind_parameters = [](const DecoderGraph& decoder, Bindings& bindings) {
            for (const auto parameter_id : decoder.parameters) {
                auto& values = bindings[parameter_id];
                values.resize(decoder.graph.tensor(parameter_id).elements());
                for (std::size_t index = 0; index < values.size(); ++index) {
                    values[index] = 0.02F * static_cast<float>(1 + (index % 11));
                }
            }
        };
        // Proves an N-token full replay is bit-identical to an (N-1)-token
        // prefill plus one cached incremental token, wiring the prefill KV
        // outputs into the incremental graph exactly as the server does.
        const auto verify_incremental = [&bind_parameters](
            const FamilySpec& family, std::size_t total_tokens, const std::string& label) {
            const auto prefill_tokens = total_tokens - 1;
            const auto full = build_decoder_graph(family, total_tokens, 0, 0, true);
            Bindings full_bindings;
            auto& full_input = full_bindings[full.input];
            full_input.resize(full.graph.tensor(full.input).elements());
            for (std::size_t index = 0; index < full_input.size(); ++index) {
                full_input[index] = 0.05F * static_cast<float>(index + 1);
            }
            bind_parameters(full, full_bindings);
            const auto full_result = Plan::compile(full.graph).replay(full_bindings);

            const auto prefill = build_decoder_graph(family, prefill_tokens, 0, 0, true);
            Bindings prefill_bindings;
            prefill_bindings[prefill.input] = std::vector<float>(
                full_input.begin(),
                full_input.begin() + static_cast<std::ptrdiff_t>(prefill_tokens * family.hidden_size)
            );
            bind_parameters(prefill, prefill_bindings);
            const auto prefill_result = Plan::compile(prefill.graph).replay(prefill_bindings);

            const auto incremental = build_decoder_graph(family, 1, prefill_tokens, prefill_tokens, true);
            Bindings incremental_bindings;
            incremental_bindings[incremental.input] = std::vector<float>(
                full_input.begin() + static_cast<std::ptrdiff_t>(prefill_tokens * family.hidden_size),
                full_input.end()
            );
            bind_parameters(incremental, incremental_bindings);
            for (const auto& input : incremental.kv_inputs) {
                const auto output = std::find_if(prefill.kv_outputs.begin(), prefill.kv_outputs.end(), [&](const DecoderGraph::KvBinding& kv) {
                    return kv.layer == input.layer;
                });
                if (output == prefill.kv_outputs.end()) throw std::runtime_error(label + " prefill KV output is missing");
                incremental_bindings[input.key] = prefill_result.at(output->key);
                incremental_bindings[input.value] = prefill_result.at(output->value);
            }
            const auto incremental_result = Plan::compile(incremental.graph).replay(incremental_bindings);
            const auto& complete = full_result.at(full.output);
            const std::vector<float> expected_last(
                complete.end() - static_cast<std::ptrdiff_t>(family.hidden_size),
                complete.end()
            );
            if (!bit_equal(expected_last, incremental_result.at(incremental.output))) {
                throw std::runtime_error(label + " incremental KV replay differs from full replay");
            }
        };

        for (const auto& [family_name, biases] : std::vector<std::pair<std::string, bool>>{
                 {"llama", false}, {"gemma", false}, {"qwen2", true}}) {
            FamilySpec family{
                .architecture = family_name,
                .family = family_name,
                .layers = 2,
                .hidden_size = 4,
                .feed_forward_size = 6,
                .heads = 2,
                .kv_heads = 1,
                .head_dim = 2,
                .rms_epsilon = 1.0e-5F,
                .rope_frequency_base = 10000.0F,
                .rope_frequency_scale = 1.0F,
                .rope_neox = family_name == "qwen2" || family_name == "gemma",
                .rope_frequency_factors = false,
                .attention_biases = biases,
                .layer_specs = {},
            };
            for (const std::size_t tokens : {std::size_t{1}, std::size_t{3}}) {
                const auto decoder = build_decoder_graph(family, tokens);
                const auto plan = Plan::compile(decoder.graph);
                Bindings bindings;
                bindings[decoder.input].resize(decoder.graph.tensor(decoder.input).elements());
                for (std::size_t index = 0; index < bindings[decoder.input].size(); ++index) {
                    bindings[decoder.input][index] = 0.05F * static_cast<float>(index + 1);
                }
                for (const auto parameter_id : decoder.parameters) {
                    auto& values = bindings[parameter_id];
                    values.resize(decoder.graph.tensor(parameter_id).elements());
                    for (std::size_t index = 0; index < values.size(); ++index) {
                        values[index] = 0.02F * static_cast<float>(1 + (index % 11));
                    }
                }
                const auto reference = imperative_reference(decoder.graph, bindings);
                const auto replay = plan.replay(bindings);
                if (!bit_equal(reference.at(decoder.output), replay.at(decoder.output))) {
                    throw std::runtime_error(family_name + " decoder replay differs from its imperative reference");
                }
                if (plan.cpu_spans().size() != 1) {
                    throw std::runtime_error(family_name + " decoder was not coalesced into one CPU span");
                }
            }

            verify_incremental(family, 3, family_name);
        }

        // Cached attention with a sliding window plus Gemma4-style shared KV
        // routing: layers 0/1 own a global and a windowed cache, layers 2/3
        // consume them through kv_source_layer. Four total tokens make the
        // two-token window actually clip during both prefill and the cached
        // incremental step.
        const FamilySpec shared_window{
            .architecture = "gemma4-selftest",
            .family = "gemma",
            .layers = 4,
            .hidden_size = 4,
            .feed_forward_size = 6,
            .heads = 2,
            .kv_heads = 1,
            .head_dim = 2,
            .rms_epsilon = 1.0e-5F,
            .rope_frequency_base = 10000.0F,
            .rope_frequency_scale = 1.0F,
            .rope_neox = true,
            .rope_frequency_factors = false,
            .attention_biases = false,
            .gemma4_extended = false,
            .per_layer_input_size = 0,
            .layer_specs = {
                LayerSpec{2, 4, 2, 0, 10000.0F, 0},
                LayerSpec{2, 4, 2, 2, 40000.0F, 1},
                LayerSpec{2, 4, 2, 0, 10000.0F, 0},
                LayerSpec{2, 4, 2, 2, 40000.0F, 1},
            },
        };
        verify_incremental(shared_window, 4, "gemma shared+windowed");
        std::cout << "caterpillar FamilySpec decoder selftest: ok (llama + gemma + qwen2 + shared/windowed KV, full + incremental replay)\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "caterpillar FamilySpec decoder selftest: failed: " << error.what() << '\n';
        return 1;
    }
}

}  // namespace cat
