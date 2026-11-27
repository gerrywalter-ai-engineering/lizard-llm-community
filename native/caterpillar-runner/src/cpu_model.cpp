#include "cpu_model.h"

#include "family_spec.h"
#include "plan.h"
#include "tensor_data.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <span>
#include <stdexcept>

namespace cat {
namespace {

void verify_parameter_shape(const Tensor& graph_tensor, const GgufTensorInfo& gguf_tensor) {
    if (graph_tensor.shape.size() != gguf_tensor.dimensions.size()) {
        throw std::invalid_argument("GGUF rank differs from graph parameter: " + graph_tensor.name);
    }
    if (graph_tensor.shape.size() == 1) {
        if (graph_tensor.shape[0] != gguf_tensor.dimensions[0]) {
            throw std::invalid_argument("GGUF vector shape differs from graph parameter: " + graph_tensor.name);
        }
    } else if (graph_tensor.shape.size() == 2) {
        if (graph_tensor.shape[0] != gguf_tensor.dimensions[1]
            || graph_tensor.shape[1] != gguf_tensor.dimensions[0]) {
            throw std::invalid_argument("GGUF matrix shape differs from graph parameter: " + graph_tensor.name);
        }
    }
}

std::vector<float> matrix_vector(const QuantizedTensorView& matrix, const std::vector<float>& input) {
    if (matrix.tensor == nullptr || matrix.tensor->dimensions.size() != 2) {
        throw std::invalid_argument("matrix-vector input is not rank two");
    }
    return tensor_matrix_vector(matrix, input);
}

void scale_in_place(std::vector<float>& values, float scale) {
    for (auto& value : values) value *= scale;
}

}  // namespace

std::size_t CpuSessionState::bytes() const noexcept {
    std::size_t total{};
    for (const auto& layer : layers) {
        total += (layer.keys.size() + layer.values.size()) * sizeof(float);
    }
    return total;
}

// Run 04 iteration 13: the family spec is a pure function of the model
// metadata, and infer() resolved it again on every token. Resolve once here -
// this constructor already called it purely to validate the model, so the
// result was being computed and thrown away.
CpuModel::CpuModel(const std::filesystem::path& path)
    : model_(path), tokenizer_(model_.info()),
      base_family_(resolve_family_spec(model_.info())) {}

std::vector<std::vector<float>> CpuModel::build_ple_inputs(
    const std::vector<std::uint32_t>& tokens,
    const std::vector<float>& scaled_embeddings
) const {
    const auto& info = model_.info();
    const auto width = static_cast<std::size_t>(info.embedding_length_per_layer_input);
    if (width == 0) return {};
    const auto hidden = static_cast<std::size_t>(info.embedding_length);
    const auto layers = static_cast<std::size_t>(info.block_count);
    const auto packed = width * layers;
    if (scaled_embeddings.size() != tokens.size() * hidden) throw std::invalid_argument("PLE embedding batch has the wrong shape");
    const auto token_embedding = model_.view("per_layer_token_embd.weight");
    const auto projection = model_.view("per_layer_model_proj.weight");
    const auto projection_norm_view = model_.view("per_layer_proj_norm.weight");
    const auto projection_norm = dequantize_tensor(*projection_norm_view.tensor, projection_norm_view.bytes);
    if (projection_norm.size() != width) throw std::invalid_argument("PLE projection norm has the wrong shape");

    std::vector<std::vector<float>> layer_inputs(layers, std::vector<float>(tokens.size() * width));
    constexpr float combine_scale = 0.7071067811865475244F;
    for (std::size_t token_index = 0; token_index < tokens.size(); ++token_index) {
        auto token_values = tensor_row_f32(token_embedding, tokens[token_index]);
        scale_in_place(token_values, std::sqrt(static_cast<float>(width)));
        std::vector<float> hidden_values(
            scaled_embeddings.begin() + static_cast<std::ptrdiff_t>(token_index * hidden),
            scaled_embeddings.begin() + static_cast<std::ptrdiff_t>((token_index + 1) * hidden));
        auto projected = matrix_vector(projection, hidden_values);
        if (projected.size() != packed || token_values.size() != packed) throw std::invalid_argument("PLE packed projection has the wrong shape");
        scale_in_place(projected, 1.0F / std::sqrt(static_cast<float>(hidden)));
        for (std::size_t layer = 0; layer < layers; ++layer) {
            const auto offset = layer * width;
            double square_sum = 0.0;
            for (std::size_t column = 0; column < width; ++column) {
                const double value = projected[offset + column];
                square_sum += value * value;
            }
            const float inverse_rms = static_cast<float>(1.0 / std::sqrt(square_sum / static_cast<double>(width) + info.rms_epsilon));
            for (std::size_t column = 0; column < width; ++column) {
                const auto packed_index = offset + column;
                layer_inputs[layer][token_index * width + column] =
                    (projected[packed_index] * inverse_rms * projection_norm[column] + token_values[packed_index]) * combine_scale;
            }
        }
    }
    return layer_inputs;
}

CpuModel::InferenceResult CpuModel::infer(
    const std::vector<std::uint32_t>& tokens,
    const CpuSessionState* past,
    std::size_t layer_limit,
    const EmbeddingOverrides* embedding_overrides,
    std::span<CpuSessionState* const> independent_past
) const {
    if (tokens.empty()) throw std::invalid_argument("CPU model requires at least one token");
    auto family = base_family_;
    if (layer_limit != 0) {
        if (layer_limit > family.layers) throw std::invalid_argument("CPU layer limit exceeds the model layer count");
        family.layers = layer_limit;
        if (!family.layer_specs.empty()) family.layer_specs.resize(layer_limit);
    }
    const bool independent_rows = !independent_past.empty();
    if (independent_rows && independent_past.size() != tokens.size()) {
        throw std::invalid_argument(
            "independent decode requires one populated session per token");
    }
    const CpuSessionState* first_past =
        independent_rows ? independent_past.front() : past;
    const auto past_tokens = first_past == nullptr ? 0 : first_past->cached_tokens;
    if (independent_rows) {
        for (const auto* session : independent_past) {
            if (session == nullptr || session->cached_tokens != past_tokens
                || session->layers.empty()) {
                throw std::invalid_argument(
                    "independent decode sessions must have equal populated KV lengths");
            }
        }
    }
    const auto decoder = build_decoder_graph(
        family,
        tokens.size(),
        past_tokens,
        past_tokens,
        true,
        independent_rows);
    const auto plan = Plan::compile(decoder.graph);
    Bindings floats;
    QuantizedBindings quantized;

    const auto embeddings_view = model_.view("token_embd.weight");
    auto& embeddings = floats[decoder.input];
    embeddings.reserve(tokens.size() * family.hidden_size);
    for (std::size_t position = 0; position < tokens.size(); ++position) {
        const auto* supplied = embedding_overrides == nullptr
            ? nullptr
            : embedding_overrides->row_for(position);
        if (supplied != nullptr) {
            // Used as given: no gemma4 scale. A supplied row is already a
            // hidden state, not a token embedding waiting to be scaled.
            // Width is checked here because a short row would otherwise
            // silently shift every later position in this flat buffer.
            if (supplied->size() != family.hidden_size) {
                throw std::invalid_argument(
                    "supplied prompt embedding width does not match the model hidden size");
            }
            embeddings.insert(embeddings.end(), supplied->begin(), supplied->end());
            continue;
        }
        auto row = tensor_row_f32(embeddings_view, tokens[position]);
        if (family.gemma4_extended) scale_in_place(row, std::sqrt(static_cast<float>(family.hidden_size)));
        embeddings.insert(embeddings.end(), row.begin(), row.end());
    }

    // Run 04 (2026-08-04) iteration 12: this loop runs on every token. The
    // rank-1 parameters used to be dequantized here, once per parameter per
    // token, for values that never change - MappedGgufModel now converts them
    // at load and this hands out the cached copy.
    for (const auto id : decoder.parameters) {
        const auto& graph_tensor = decoder.graph.tensor(id);
        const auto view = model_.view(graph_tensor.name);
        verify_parameter_shape(graph_tensor, *view.tensor);
        if (graph_tensor.shape.size() == 2) {
            quantized.emplace(id, view);
            continue;
        }
        if (const auto* cached = model_.dequantized_parameter(graph_tensor.name);
            cached != nullptr) {
            floats.emplace(id, *cached);
            continue;
        }
        floats.emplace(id, dequantize_tensor(*view.tensor, view.bytes));
    }
    const auto ple = build_ple_inputs(tokens, embeddings);
    if (ple.size() != decoder.runtime_inputs.size()) {
        if (!ple.empty() || !decoder.runtime_inputs.empty()) throw std::logic_error("PLE graph/runtime input count mismatch");
    }
    for (std::size_t layer = 0; layer < ple.size(); ++layer) floats.emplace(decoder.runtime_inputs[layer], ple[layer]);

    if (independent_rows) {
        for (const auto& input : decoder.kv_inputs) {
            const auto width = decoder.graph.tensor(input.key).shape[1];
            std::vector<float> keys;
            std::vector<float> values;
            keys.reserve(independent_past.size() * past_tokens * width);
            values.reserve(independent_past.size() * past_tokens * width);
            for (const auto* session : independent_past) {
                const auto found = std::find_if(
                    session->layers.begin(),
                    session->layers.end(),
                    [&](const CpuKvLayer& layer) { return layer.layer == input.layer; });
                if (found == session->layers.end() || found->width != width
                    || found->keys.size() != past_tokens * width
                    || found->values.size() != past_tokens * width) {
                    throw std::invalid_argument(
                        "independent decode KV cache has the wrong shape");
                }
                keys.insert(keys.end(), found->keys.begin(), found->keys.end());
                values.insert(values.end(), found->values.begin(), found->values.end());
            }
            floats.emplace(input.key, std::move(keys));
            floats.emplace(input.value, std::move(values));
        }
    }

    BindingViews views;
    views.reserve(floats.size() + decoder.kv_inputs.size());
    for (const auto& [id, values] : floats) views.emplace(id, values);
    if (past != nullptr && !independent_rows) {
        for (const auto& input : decoder.kv_inputs) {
            const auto found = std::find_if(past->layers.begin(), past->layers.end(), [&](const CpuKvLayer& layer) {
                return layer.layer == input.layer;
            });
            if (found == past->layers.end()) throw std::invalid_argument("incremental KV cache is missing a layer");
            const auto width = decoder.graph.tensor(input.key).shape[1];
            if (found->width != width || found->keys.size() != past_tokens * width
                || found->values.size() != past_tokens * width) {
                throw std::invalid_argument("incremental KV cache has the wrong shape");
            }
            views.emplace(input.key, found->keys);
            views.emplace(input.value, found->values);
        }
    }

    auto outputs = plan.replay_views(views, quantized);
    InferenceResult result;
    result.hidden = std::move(outputs.at(decoder.output));
    result.layers.reserve(decoder.kv_outputs.size());
    for (const auto& output : decoder.kv_outputs) {
        const auto width = decoder.graph.tensor(output.key).shape[1];
        result.layers.push_back(CpuKvLayer{
            .layer = output.layer,
            .width = width,
            .keys = std::move(outputs.at(output.key)),
            .values = std::move(outputs.at(output.value)),
        });
    }
    return result;
}

std::vector<float> CpuModel::hidden_states(
    const std::vector<std::uint32_t>& tokens,
    std::size_t layer_limit
) const {
    return infer(tokens, nullptr, layer_limit).hidden;
}

std::uint32_t CpuModel::greedy_from_hidden(std::span<const float> hidden) const {
    const auto width = static_cast<std::size_t>(model_.info().embedding_length);
    if (hidden.size() < width) throw std::logic_error("decoder output is shorter than one hidden state");
    const auto last = hidden.last(width);
    const auto output = model_.view(model_.has_tensor("output.weight") ? "output.weight" : "token_embd.weight");
    const auto vocabulary = static_cast<std::size_t>(output.tensor->dimensions[1]);
    const auto logits = tensor_matrix_vector(output, last);
    std::uint32_t best = 0;
    float best_logit = -std::numeric_limits<float>::infinity();
    for (std::size_t token = 0; token < vocabulary; ++token) {
        const float logit = logits[token];
        if (logit > best_logit) {
            best_logit = logit;
            best = static_cast<std::uint32_t>(token);
        }
    }
    return best;
}

std::uint32_t CpuModel::greedy_next(
    const std::vector<std::uint32_t>& tokens,
    std::size_t layer_limit
) const {
    return greedy_from_hidden(hidden_states(tokens, layer_limit));
}

CpuSessionState CpuModel::prefill(const std::vector<std::uint32_t>& tokens) const {
    auto inference = infer(tokens, nullptr);
    CpuSessionState session;
    session.cached_tokens = tokens.size();
    session.layers = std::move(inference.layers);
    session.next_token = greedy_from_hidden(inference.hidden);
    session.next_token_valid = true;
    return session;
}

std::uint32_t CpuModel::advance(CpuSessionState& session, std::uint32_t token) const {
    if (session.cached_tokens == 0 || session.layers.empty()) {
        throw std::invalid_argument("incremental decode requires a populated prefill cache");
    }
    auto inference = infer({token}, &session);
    for (auto& current : inference.layers) {
        const auto found = std::find_if(session.layers.begin(), session.layers.end(), [&](const CpuKvLayer& layer) {
            return layer.layer == current.layer;
        });
        if (found == session.layers.end() || found->width != current.width
            || current.keys.size() != current.width || current.values.size() != current.width) {
            throw std::logic_error("incremental decode produced an incompatible KV layer");
        }
        found->keys.insert(found->keys.end(), current.keys.begin(), current.keys.end());
        found->values.insert(found->values.end(), current.values.begin(), current.values.end());
    }
    ++session.cached_tokens;
    session.next_token = greedy_from_hidden(inference.hidden);
    session.next_token_valid = true;
    return session.next_token;
}

std::vector<std::uint32_t> CpuModel::advance_batch(
    std::span<CpuSessionState*> sessions,
    std::span<const std::uint32_t> tokens
) const {
    if (sessions.empty() || sessions.size() != tokens.size()) {
        throw std::invalid_argument(
            "batched decode requires one token per non-empty session");
    }
    const std::vector<std::uint32_t> input_tokens(tokens.begin(), tokens.end());
    auto inference = infer(input_tokens, nullptr, 0, nullptr, sessions);
    const auto hidden = static_cast<std::size_t>(model_.info().embedding_length);
    if (inference.hidden.size() != sessions.size() * hidden) {
        throw std::logic_error(
            "batched decode produced an invalid hidden-state batch");
    }

    const auto output = model_.view(
        model_.has_tensor("output.weight") ? "output.weight" : "token_embd.weight");
    const auto vocabulary = static_cast<std::size_t>(output.tensor->dimensions[1]);
    const auto logits =
        tensor_matrix_multiply(output, inference.hidden, sessions.size());
    std::vector<std::uint32_t> next_tokens(sessions.size());
    for (std::size_t batch = 0; batch < sessions.size(); ++batch) {
        std::uint32_t best = 0;
        float best_logit = -std::numeric_limits<float>::infinity();
        for (std::size_t token = 0; token < vocabulary; ++token) {
            const float logit = logits[batch * vocabulary + token];
            if (logit > best_logit) {
                best_logit = logit;
                best = static_cast<std::uint32_t>(token);
            }
        }
        next_tokens[batch] = best;
    }

    for (auto& current : inference.layers) {
        if (current.keys.size() != sessions.size() * current.width
            || current.values.size() != sessions.size() * current.width) {
            throw std::logic_error(
                "batched decode produced an incompatible KV layer");
        }
        for (std::size_t batch = 0; batch < sessions.size(); ++batch) {
            auto& session = *sessions[batch];
            const auto found = std::find_if(
                session.layers.begin(),
                session.layers.end(),
                [&](const CpuKvLayer& layer) {
                    return layer.layer == current.layer;
                });
            if (found == session.layers.end() || found->width != current.width) {
                throw std::logic_error(
                    "batched decode session KV layer is incompatible");
            }
            const auto begin = static_cast<std::ptrdiff_t>(batch * current.width);
            const auto end = begin + static_cast<std::ptrdiff_t>(current.width);
            found->keys.insert(
                found->keys.end(),
                current.keys.begin() + begin,
                current.keys.begin() + end);
            found->values.insert(
                found->values.end(),
                current.values.begin() + begin,
                current.values.begin() + end);
        }
    }
    for (std::size_t batch = 0; batch < sessions.size(); ++batch) {
        auto& session = *sessions[batch];
        ++session.cached_tokens;
        session.next_token = next_tokens[batch];
        session.next_token_valid = true;
    }
    return next_tokens;
}

CpuDraftVerification CpuModel::verify_draft(
    const CpuSessionState& session,
    const std::vector<std::uint32_t>& input_tokens
) const {
    if (session.cached_tokens == 0 || session.layers.empty()) {
        throw std::invalid_argument("draft verification requires a populated KV cache");
    }
    if (input_tokens.empty()) throw std::invalid_argument("draft verification requires input tokens");
    auto inference = infer(input_tokens, &session);
    const auto hidden = static_cast<std::size_t>(model_.info().embedding_length);
    if (inference.hidden.size() != input_tokens.size() * hidden) {
        throw std::logic_error("draft verification produced an invalid hidden-state batch");
    }
    const auto output = model_.view(model_.has_tensor("output.weight") ? "output.weight" : "token_embd.weight");
    const auto vocabulary = static_cast<std::size_t>(output.tensor->dimensions[1]);
    const auto logits = tensor_matrix_multiply(output, inference.hidden, input_tokens.size());
    CpuDraftVerification result;
    result.next_tokens.resize(input_tokens.size());
    result.layers = std::move(inference.layers);
    for (std::size_t batch = 0; batch < input_tokens.size(); ++batch) {
        std::uint32_t best = 0;
        float best_logit = -std::numeric_limits<float>::infinity();
        for (std::size_t token = 0; token < vocabulary; ++token) {
            const float logit = logits[batch * vocabulary + token];
            if (logit > best_logit) {
                best_logit = logit;
                best = static_cast<std::uint32_t>(token);
            }
        }
        result.next_tokens[batch] = best;
    }
    return result;
}

void CpuModel::commit_draft(
    CpuSessionState& session,
    const CpuDraftVerification& verification,
    std::size_t committed_inputs
) const {
    if (committed_inputs == 0 || committed_inputs > verification.next_tokens.size()) {
        throw std::invalid_argument("draft commit count is outside the verified batch");
    }
    if (session.layers.size() != verification.layers.size()) {
        throw std::logic_error("draft commit layer count mismatch");
    }
    for (const auto& fresh : verification.layers) {
        const auto found = std::find_if(session.layers.begin(), session.layers.end(),
            [&](const CpuKvLayer& layer) { return layer.layer == fresh.layer; });
        if (found == session.layers.end() || found->width != fresh.width) {
            throw std::logic_error("draft commit produced an incompatible KV layer");
        }
        const auto committed_values = committed_inputs * fresh.width;
        if (fresh.keys.size() < committed_values || fresh.values.size() < committed_values) {
            throw std::logic_error("draft commit verification KV is too short");
        }
        found->keys.insert(found->keys.end(),
            fresh.keys.begin(),
            fresh.keys.begin() + static_cast<std::ptrdiff_t>(committed_values));
        found->values.insert(found->values.end(),
            fresh.values.begin(),
            fresh.values.begin() + static_cast<std::ptrdiff_t>(committed_values));
    }
    session.cached_tokens += committed_inputs;
    session.next_token = verification.next_tokens[committed_inputs - 1];
    session.next_token_valid = true;
}

CpuGenerationResult CpuModel::generate(std::string_view prompt, std::size_t max_tokens) const {
    CpuGenerationResult result;
    result.prompt_tokens = tokenizer_.encode(prompt);
    auto session = prefill(result.prompt_tokens);
    for (std::size_t index = 0; index < max_tokens; ++index) {
        if (!session.next_token_valid) throw std::logic_error("CPU session has no pending token");
        const auto token = session.next_token;
        session.next_token_valid = false;
        result.generated_tokens.push_back(token);
        if (token == model_.info().eos_token_id) break;
        if (index + 1 < max_tokens) static_cast<void>(advance(session, token));
    }
    result.text = tokenizer_.decode(result.generated_tokens);
    return result;
}

}  // namespace cat
