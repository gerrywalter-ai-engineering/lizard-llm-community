#include "gpu_model.h"

#include "tensor_data.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string_view>

namespace cat {
namespace {

bool integer_dot_requested() {
    const char* value = std::getenv("CATERPILLAR_GPU_DP4A");
    return value != nullptr && std::string_view(value) == "1";
}

bool unqualified_integer_dot_requested() {
    const char* value = std::getenv("CATERPILLAR_GPU_DP4A_ALL");
    return value != nullptr && std::string_view(value) == "1";
}

void verify_parameter_shape(const Tensor& graph_tensor, const GgufTensorInfo& tensor) {
    // GGUF dimension order is [row width, row count]; graph shapes are
    // {rows, columns} for matrices and {elements} for vectors.
    if (graph_tensor.shape.size() == 2) {
        if (tensor.dimensions.size() != 2
            || tensor.dimensions[0] != graph_tensor.shape[1]
            || tensor.dimensions[1] != graph_tensor.shape[0]) {
            throw std::invalid_argument("GPU parameter shape mismatch: " + graph_tensor.name);
        }
        return;
    }
    if (tensor_element_count(tensor) != graph_tensor.elements()) {
        throw std::invalid_argument("GPU parameter element mismatch: " + graph_tensor.name);
    }
}

}  // namespace

std::unique_ptr<GpuModel> GpuModel::try_create(
    const CpuModel& cpu,
    const FitPlan& fit,
    std::size_t kv_capacity_tokens,
    std::string* reason
) {
    try {
        std::string availability;
        if (!gpu_backend_available(&availability)) {
            if (reason != nullptr) *reason = availability;
            return nullptr;
        }
        if (fit.decision == FitDecision::cpu_only || fit.resident_layer_count == 0) {
            if (reason != nullptr) *reason = "fit decision pinned the session CPU-only";
            return nullptr;
        }
        auto model = std::unique_ptr<GpuModel>(new GpuModel(cpu, fit, kv_capacity_tokens));
        if (model->schedule_.gpu_node_count == 0) {
            if (reason != nullptr) *reason = model->schedule_.cpu_reason;
            return nullptr;
        }
        return model;
    } catch (const std::exception& error) {
        if (reason != nullptr) *reason = error.what();
        return nullptr;
    }
}

GpuModel::GpuModel(const CpuModel& cpu, const FitPlan& fit, std::size_t kv_capacity_tokens)
    : cpu_(cpu), family_(resolve_family_spec(cpu.info())), capacity_(kv_capacity_tokens) {
    if (capacity_ == 0) throw std::invalid_argument("GPU decode requires a nonzero KV capacity");

    // Capacity-shaped decode graph: one token, past inputs sized to the
    // session capacity. The live counts flow through ReplayDynamics.
    decoder_ = build_decoder_graph(family_, 1, 0, capacity_, true);

    // Extend the graph with the logits matmul so greedy decode needs no
    // separate CPU pass over the (large) output head.
    const auto& info = cpu_.info();
    const auto vocabulary = static_cast<std::size_t>(info.vocabulary_size);
    const auto hidden = family_.hidden_size;
    const std::string head_name = cpu_.mapped().has_tensor("output.weight")
        ? "output.weight"
        : "token_embd.weight";
    logits_weight_id_ = decoder_.graph.add_tensor(head_name, {vocabulary, hidden}, true);
    logits_id_ = decoder_.graph.add_tensor("logits", {1, vocabulary});
    decoder_.graph.add_node(Op::matmul, {decoder_.output, logits_weight_id_}, logits_id_);
    // Bindings: 2-D parameters stay quantized (GPU-resident + CPU-span
    // capable); 1-D parameters are dequantized once and treated as
    // immutable externals.
    SchedulerInput scheduler_input;
    scheduler_input.gpu_available = true;
    scheduler_input.resident_layer_count = fit.resident_layer_count;
    scheduler_input.total_layer_count = fit.total_layer_count;
    for (const auto id : decoder_.parameters) {
        const auto& graph_tensor = decoder_.graph.tensor(id);
        const auto view = cpu_.mapped().view(graph_tensor.name);
        verify_parameter_shape(graph_tensor, *view.tensor);
        if (graph_tensor.shape.size() == 2) {
            quantized_.emplace(id, view);
            scheduler_input.weight_types.emplace(id, view.tensor->type);
        } else {
            immutable_.emplace(id, dequantize_tensor(*view.tensor, view.bytes));
        }
    }
    {
        const auto view = cpu_.mapped().view(head_name);
        verify_parameter_shape(decoder_.graph.tensor(logits_weight_id_), *view.tensor);
        quantized_.emplace(logits_weight_id_, view);
        scheduler_input.weight_types.emplace(logits_weight_id_, view.tensor->type);
    }

    schedule_ = schedule_decode_backends(decoder_.graph, scheduler_input);
    if (schedule_.gpu_node_count == 0) return;  // try_create reports the reason

    // Fully resident decode returns exactly one graph value: logits. A stable
    // GPU-prefix/CPU-tail plan additionally returns K/V rows so its required
    // f32 CPU mirror can advance without changing the established fallback.
    std::vector<TensorId> outputs{logits_id_};
    if (schedule_.cpu_node_count != 0) {
        for (const auto& kv : decoder_.kv_outputs) {
            outputs.push_back(kv.key);
            outputs.push_back(kv.value);
        }
    }
    decoder_.graph.set_outputs(std::move(outputs));

    plan_ = Plan::compile(decoder_.graph, schedule_.node_backends);

    dynamic_ids_.push_back(decoder_.input);
    for (const auto id : decoder_.runtime_inputs) dynamic_ids_.push_back(id);

    gpu_ = std::make_unique<GpuPlanExecution>(
        plan_, quantized_, immutable_, dynamic_ids_, decoder_.kv_inputs,
        GpuPlanExecution::Config{
            capacity_, true, integer_dot_requested(), unqualified_integer_dot_requested(),
            fit.kv_cache_mode, schedule_.cpu_node_count != 0});

    std::size_t kv_values_per_token = 0;
    for (const auto& kv : decoder_.kv_inputs) {
        kv_values_per_token += 2 * decoder_.graph.tensor(kv.key).shape[1];
    }
    per_token_kv_bytes_ = static_cast<std::size_t>(gpu_->telemetry().kv_resident_bytes / capacity_);
    if (gpu_->has_cpu_kv_mirror()) per_token_kv_bytes_ += kv_values_per_token * sizeof(float);
}

GpuSessionState GpuModel::prefill(
    const std::vector<std::uint32_t>& tokens,
    bool retain_verifier_state
) {
    if (tokens.empty()) throw std::invalid_argument("GPU prefill requires at least one token");
    if (tokens.size() >= capacity_) {
        throw std::invalid_argument("prompt exceeds the GPU session KV capacity");
    }
    // CPU prefill seeds the packed device cache once. Incremental rows never
    // return to the CPU on a fully resident plan.
    auto cpu_session = cpu_.prefill(tokens);
    for (const auto& layer : cpu_session.layers) {
        gpu_->seed_kv(layer.layer, layer.keys, layer.values, cpu_session.cached_tokens);
    }
    if (retain_verifier_state) {
        // The verifier must see exactly the values consumed by device
        // attention. Normalize the f32 prefill cache through the selected
        // packed KV representation before retaining it.
        for (auto& layer : cpu_session.layers) {
            gpu_->read_kv_rows(
                layer.layer, 0, cpu_session.cached_tokens,
                layer.keys, layer.values);
        }
    }
    GpuSessionState session;
    session.cached_tokens = cpu_session.cached_tokens;
    session.next_token = cpu_session.next_token;
    session.next_token_valid = cpu_session.next_token_valid;
    if (retain_verifier_state) session.verifier_state = std::move(cpu_session);
    return session;
}

std::uint32_t GpuModel::advance(GpuSessionState& session, std::uint32_t token) {
    if (session.cached_tokens == 0) {
        throw std::invalid_argument("GPU incremental decode requires a prefilled session");
    }
    if (session.cached_tokens + 1 >= capacity_) {
        throw std::invalid_argument("GPU session reached its KV capacity");
    }

    // Token embedding row (plus the Gemma sqrt(hidden) scale) on the CPU;
    // this is the graph's dynamic input.
    const auto embeddings_view = cpu_.mapped().view("token_embd.weight");
    auto embedding = tensor_row_f32(embeddings_view, token);
    if (family_.gemma4_extended) {
        const float scale = std::sqrt(static_cast<float>(family_.hidden_size));
        for (auto& value : embedding) value *= scale;
    }
    const auto ple = cpu_.build_ple_inputs({token}, embedding);
    if (ple.size() != decoder_.runtime_inputs.size()) {
        throw std::logic_error("PLE input count mismatch in GPU decode");
    }

    BindingViews views;
    views.reserve(immutable_.size() + ple.size() + decoder_.kv_inputs.size() + 1);
    for (const auto& [id, values] : immutable_) views.emplace(id, values);
    views.emplace(decoder_.input, embedding);
    for (std::size_t index = 0; index < ple.size(); ++index) {
        views.emplace(decoder_.runtime_inputs[index], ple[index]);
    }
    // Only stable-split CPU tails bind the f32 mirror. Fully resident GPU
    // attention reads its packed cache directly through the backend.
    if (gpu_->has_cpu_kv_mirror()) {
        for (const auto& kv : decoder_.kv_inputs) {
            views.emplace(kv.key, gpu_->kv_keys(kv.layer));
            views.emplace(kv.value, gpu_->kv_values(kv.layer));
        }
    }

    ReplayDynamics dynamics;
    dynamics.past_tokens = session.cached_tokens;
    dynamics.position_offset = session.cached_tokens;

    auto outputs = plan_.replay_views(views, quantized_, dynamics, gpu_.get());

    // Stable-split only: advance the CPU tail's compatibility mirror. The
    // packed GPU cache was already appended inside the recorded command list.
    if (gpu_->has_cpu_kv_mirror()) {
        for (const auto& kv : decoder_.kv_outputs) {
            const auto width = decoder_.graph.tensor(kv.key).shape[1];
            const auto& key_row = outputs.at(kv.key);
            const auto& value_row = outputs.at(kv.value);
            auto keys = gpu_->kv_keys(kv.layer);
            auto values = gpu_->kv_values(kv.layer);
            std::copy(key_row.begin(), key_row.end(), keys.begin() + static_cast<std::ptrdiff_t>(session.cached_tokens * width));
            std::copy(value_row.begin(), value_row.end(), values.begin() + static_cast<std::ptrdiff_t>(session.cached_tokens * width));
        }
    }
    const auto appended_token = session.cached_tokens;
    ++session.cached_tokens;

    // Greedy argmax on the CPU, first-max-wins — identical tie-break to the
    // Phase 2 engine.
    const auto& logits = outputs.at(logits_id_);
    std::uint32_t best = 0;
    float best_logit = -std::numeric_limits<float>::infinity();
    for (std::size_t index = 0; index < logits.size(); ++index) {
        if (logits[index] > best_logit) {
            best_logit = logits[index];
            best = static_cast<std::uint32_t>(index);
        }
    }
    session.next_token = best;
    session.next_token_valid = true;
    if (session.verifier_state) {
        auto& verifier = *session.verifier_state;
        for (auto& layer : verifier.layers) {
            std::vector<float> key_row(layer.width);
            std::vector<float> value_row(layer.width);
            gpu_->read_kv_rows(layer.layer, appended_token, 1, key_row, value_row);
            layer.keys.insert(layer.keys.end(), key_row.begin(), key_row.end());
            layer.values.insert(layer.values.end(), value_row.begin(), value_row.end());
        }
        ++verifier.cached_tokens;
        verifier.next_token = best;
        verifier.next_token_valid = true;
    }
    return best;
}

CpuDraftVerification GpuModel::verify_draft(
    const GpuSessionState& session,
    const std::vector<std::uint32_t>& input_tokens
) const {
    if (!session.verifier_state) throw std::logic_error("GPU session has no speculative verifier state");
    return cpu_.verify_draft(*session.verifier_state, input_tokens);
}

void GpuModel::commit_draft(
    GpuSessionState& session,
    const CpuDraftVerification& verification,
    std::size_t input_count
) {
    if (!session.verifier_state) throw std::logic_error("GPU session has no speculative verifier state");
    if (input_count == 0 || input_count > verification.next_tokens.size()) {
        throw std::invalid_argument("speculative commit input count is invalid");
    }
    if (session.cached_tokens > capacity_ || input_count >= capacity_ - session.cached_tokens) {
        throw std::invalid_argument("speculative commit exceeds GPU KV capacity");
    }
    auto& verifier = *session.verifier_state;
    for (const auto& current : verification.layers) {
        const auto found = std::find_if(verifier.layers.begin(), verifier.layers.end(), [&](const CpuKvLayer& layer) {
            return layer.layer == current.layer;
        });
        if (found == verifier.layers.end() || found->width != current.width
            || current.keys.size() != verification.next_tokens.size() * current.width
            || current.values.size() != verification.next_tokens.size() * current.width) {
            throw std::logic_error("speculative verification produced incompatible KV rows");
        }
        const auto elements = input_count * current.width;
        gpu_->write_kv_rows(
            current.layer, session.cached_tokens,
            std::span<const float>(current.keys).first(elements),
            std::span<const float>(current.values).first(elements));
        std::vector<float> normalized_keys(elements);
        std::vector<float> normalized_values(elements);
        gpu_->read_kv_rows(
            current.layer, session.cached_tokens, input_count,
            normalized_keys, normalized_values);
        found->keys.insert(found->keys.end(), normalized_keys.begin(), normalized_keys.end());
        found->values.insert(found->values.end(), normalized_values.begin(), normalized_values.end());
    }
    session.cached_tokens += input_count;
    verifier.cached_tokens += input_count;
    session.next_token = verification.next_tokens[input_count - 1];
    session.next_token_valid = true;
    verifier.next_token = session.next_token;
    verifier.next_token_valid = true;
}

}  // namespace cat
