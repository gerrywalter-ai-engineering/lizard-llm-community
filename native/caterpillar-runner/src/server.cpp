#include "server.h"

#include "alloc.h"
#include "backend_d3d12.h"
#include "cpu_model.h"
#include "family_spec.h"
#include "fit.h"
#include "gpu_model.h"
#include "spec.h"
#include "tensor_data.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace cat {
namespace {

constexpr std::size_t kMaxRpcLineBytes = 16U * 1024U * 1024U;

enum class RpcLineStatus { line, oversized, end };

RpcLineStatus read_rpc_line(std::istream& input, std::string& output) {
    output.clear();
    output.reserve(4096);
    bool oversized = false;
    auto* buffer = input.rdbuf();
    while (true) {
        const auto next = buffer->sbumpc();
        if (next == std::char_traits<char>::eof()) {
            if (output.empty() && !oversized) return RpcLineStatus::end;
            return oversized ? RpcLineStatus::oversized : RpcLineStatus::line;
        }
        const auto character = static_cast<char>(next);
        if (character == '\n') {
            return oversized ? RpcLineStatus::oversized : RpcLineStatus::line;
        }
        if (!oversized) {
            if (output.size() >= kMaxRpcLineBytes) {
                oversized = true;
            } else {
                output.push_back(character);
            }
        }
    }
}

struct Session {
    std::vector<std::uint32_t> sequence;
    std::optional<CpuSessionState> cpu_state;
    std::optional<GpuSessionState> gpu_state;
    double prefill_ms{};
    bool cancelled{false};
    FitPlan fit;
    // End-of-turn markers supplied per request. The loop already stops on the
    // GGUF EOS, which a base model never emits, so without these every answer
    // decodes the full token budget and then gets truncated mid-word.
    std::vector<std::string> stop_sequences;
    std::size_t max_tokens{64};
    std::size_t generated_tokens{};
    std::string batch_stop_scan_tail;
    // Decode submissions per generated token, averaged over the session's
    // most recent generate call (0 on the CPU path).
    double submissions_per_token{};
    bool speculation_enabled{};
    std::string speculation_reason{"not-requested"};
    std::size_t speculation_max_draft{4};
    std::size_t speculation_min_match_tokens{4};
    // Adaptive controller (Step 02 tuning): draft length grows on accepted
    // batches and shrinks on rejected ones; a losing streak pauses drafting
    // for a stretch of tokens instead of permanently disabling speculation,
    // because n-gram acceptance is bursty (repetition appears mid-answer).
    std::size_t speculation_draft_length{4};
    std::size_t speculation_losing_streak{};
    std::size_t speculation_resume_after_tokens{};
    std::uint64_t speculation_proposed{};
    std::uint64_t speculation_accepted{};
    std::uint64_t speculation_batches{};
    std::uint64_t speculation_saved_gpu_steps{};
    double speculation_verify_ms{};
    double gpu_advance_ms{};
    std::uint64_t gpu_advance_samples{};
    // Roofline telemetry (2026-07-21 campaign, Phase 1): weight bytes the most
    // recent generate call streamed per emitted token, and the source of that
    // number ("measured-cpu" from the live counter, "derived-gpu-resident"
    // when resident GPU matvecs re-read the staged set each target step).
    double weight_bytes_per_token{};
    std::string weight_bytes_source{"unmeasured"};
};

// Host memory-bandwidth denominator for rooflineFraction. Deliberately has no
// built-in default: it is host-specific (measure with tools/bw_probe on any
// machine — laptop iGPU, desktop dGPU, server), so an unset value reports 0
// and the fraction is omitted from interpretation rather than being wrong.
double roofline_gbs() {
    static const double value = [] {
        const char* configured = std::getenv("CATERPILLAR_ROOFLINE_GBS");
        if (configured == nullptr || *configured == '\0') return 0.0;
        char* end{};
        const double parsed = std::strtod(configured, &end);
        return (end != configured && parsed > 0.0) ? parsed : 0.0;
    }();
    return value;
}

struct State {
    std::unique_ptr<CpuModel> model;
    std::unique_ptr<GpuModel> gpu_model;
    std::string gpu_reason;
    std::string model_id;
    double load_ms{};
    FitOptions fit_options;
    FitPlan fit;
    std::map<std::string, Session> sessions;

    // Resident prompt cache (2026-07-24). Prefill runs the whole prompt through
    // the CPU engine to seed the packed device KV - measured at 46s for a
    // 103-token chat prompt, against 2s of GPU decode for the answer itself.
    // A chat re-sends the whole conversation every turn, so nearly all of that
    // work is a repeat of the previous turn's.
    //
    // These fields remember which token sequence currently occupies the device
    // KV, across sessions (the packed cache lives on GpuModel, so exactly one
    // sequence is resident at a time). When the next prompt EXTENDS that
    // sequence, only the new tail is ingested, through the resident decode
    // graph. Anything else falls back to a full prefill, so correctness never
    // depends on the cache being warm.
    std::vector<std::uint32_t> resident_sequence;
    std::size_t resident_cached_tokens{};
    std::uint32_t resident_next_token{};
    bool resident_next_token_valid{false};
    bool resident_valid{false};
};

// How many leading tokens the resident KV and the incoming prompt agree on.
//
// Deliberately a COMMON PREFIX rather than a strict "resident extends" test:
// the resident sequence also holds the tokens the model generated last turn,
// and a chat re-sends only the cleaned answer text, which rarely re-tokenizes
// to the exact same tail. Matching the shared head and discarding the rest
// still reuses almost the whole conversation, and needs no agreement between
// the client's text and the runner's token history.
// Byte-level BPE encodes space as U+0120 ("Ġ") and newline as U+010A ("Ċ").
// Raw vocab text carries those, so a text stop marker like "\n### " never
// matches the stream without decoding them back. Printable ASCII maps to
// itself, so nothing else needs translating for stop matching.
std::string decode_bpe_whitespace(const std::string& text) {
    std::string out;
    out.reserve(text.size());
    for (std::size_t index = 0; index < text.size(); ++index) {
        if (index + 1 < text.size() && static_cast<unsigned char>(text[index]) == 0xC4) {
            const unsigned char next = static_cast<unsigned char>(text[index + 1]);
            if (next == 0xA0) { out.push_back(' '); ++index; continue; }
            if (next == 0x8A) { out.push_back('\n'); ++index; continue; }
        }
        out.push_back(text[index]);
    }
    return out;
}

std::vector<std::string> json_string_array_field(std::string_view line, std::string_view name) {
    std::vector<std::string> items;
    const std::string needle = "\"" + std::string(name) + "\":[";
    const auto start = line.find(needle);
    if (start == std::string_view::npos) return items;
    auto cursor = start + needle.size();
    while (cursor < line.size() && line[cursor] != ']') {
        if (line[cursor] != '"') { ++cursor; continue; }
        ++cursor;
        std::string value;
        while (cursor < line.size() && line[cursor] != '"') {
            if (line[cursor] == '\\' && cursor + 1 < line.size()) {
                ++cursor;
                switch (line[cursor]) {
                    case 'n': value.push_back('\n'); break;
                    case 't': value.push_back('\t'); break;
                    case 'r': value.push_back('\r'); break;
                    case '"': value.push_back('"'); break;
                    case '\\': value.push_back('\\'); break;
                    default: value.push_back(line[cursor]); break;
                }
            } else {
                value.push_back(line[cursor]);
            }
            ++cursor;
        }
        if (!value.empty()) items.push_back(std::move(value));
        if (cursor < line.size()) ++cursor;
    }
    return items;
}

std::size_t common_prefix_tokens(
    const std::vector<std::uint32_t>& resident,
    const std::vector<std::uint32_t>& tokens
) {
    const std::size_t limit = std::min(resident.size(), tokens.size());
    std::size_t shared = 0;
    while (shared < limit && resident[shared] == tokens[shared]) ++shared;
    return shared;
}

std::size_t gpu_kv_capacity(const FitOptions& options) {
    std::size_t capacity = options.context_tokens;
    if (const char* configured = std::getenv("CATERPILLAR_GPU_KV_CAPACITY");
        configured != nullptr && *configured != '\0') {
        char* end{};
        const auto parsed = std::strtoull(configured, &end, 10);
        if (end != configured && *end == '\0' && parsed > 1) {
            capacity = std::min<std::size_t>(options.context_tokens, static_cast<std::size_t>(parsed));
        }
    }
    return capacity;
}

std::string speculative_mode() {
    const char* value = std::getenv("CATERPILLAR_SPECULATIVE");
    if (value == nullptr || *value == '\0') return "auto";
    std::string mode(value);
    std::transform(mode.begin(), mode.end(), mode.begin(), [](unsigned char character) {
        return static_cast<char>(std::tolower(character));
    });
    return mode;
}

std::size_t speculative_max_draft() {
    const char* value = std::getenv("CATERPILLAR_SPEC_MAX_DRAFT");
    if (value == nullptr || *value == '\0') return 4;
    char* end{};
    const auto parsed = std::strtoull(value, &end, 10);
    if (end == value || *end != '\0') return 4;
    return std::clamp<std::size_t>(static_cast<std::size_t>(parsed), 1, 8);
}

std::string json_escape(std::string_view value) {
    std::ostringstream output;
    for (const unsigned char character : value) {
        switch (character) {
            case '"': output << "\\\""; break;
            case '\\': output << "\\\\"; break;
            case '\b': output << "\\b"; break;
            case '\f': output << "\\f"; break;
            case '\n': output << "\\n"; break;
            case '\r': output << "\\r"; break;
            case '\t': output << "\\t"; break;
            default:
                if (character < 0x20U) {
                    constexpr char digits[] = "0123456789abcdef";
                    output << "\\u00" << digits[character >> 4U] << digits[character & 0x0FU];
                } else {
                    output << static_cast<char>(character);
                }
        }
    }
    return output.str();
}

std::optional<std::string> parse_json_string(const std::string& input, std::size_t quote) {
    if (quote >= input.size() || input[quote] != '"') return std::nullopt;
    std::string output;
    for (std::size_t cursor = quote + 1; cursor < input.size(); ++cursor) {
        const char value = input[cursor];
        if (value == '"') return output;
        if (value != '\\') {
            output.push_back(value);
            continue;
        }
        if (++cursor >= input.size()) return std::nullopt;
        switch (input[cursor]) {
            case '"': output.push_back('"'); break;
            case '\\': output.push_back('\\'); break;
            case '/': output.push_back('/'); break;
            case 'b': output.push_back('\b'); break;
            case 'f': output.push_back('\f'); break;
            case 'n': output.push_back('\n'); break;
            case 'r': output.push_back('\r'); break;
            case 't': output.push_back('\t'); break;
            default: return std::nullopt;
        }
    }
    return std::nullopt;
}

std::optional<std::string> json_string_field(const std::string& input, std::string_view field) {
    const std::string marker = "\"" + std::string(field) + "\"";
    auto cursor = input.find(marker);
    if (cursor == std::string::npos) return std::nullopt;
    cursor = input.find(':', cursor + marker.size());
    if (cursor == std::string::npos) return std::nullopt;
    cursor = input.find('"', cursor + 1);
    return cursor == std::string::npos ? std::nullopt : parse_json_string(input, cursor);
}

std::optional<double> json_number_field(
    const std::string& input,
    std::string_view field,
    bool prefer_last = false) {
    const std::string marker = "\"" + std::string(field) + "\"";
    auto cursor = prefer_last ? input.rfind(marker) : input.find(marker);
    if (cursor == std::string::npos) return std::nullopt;
    cursor = input.find(':', cursor + marker.size());
    if (cursor == std::string::npos) return std::nullopt;
    ++cursor;
    while (cursor < input.size() && std::isspace(static_cast<unsigned char>(input[cursor]))) ++cursor;
    std::size_t consumed{};
    try {
        const double value = std::stod(input.substr(cursor), &consumed);
        return consumed == 0 ? std::nullopt : std::optional<double>(value);
    } catch (...) {
        return std::nullopt;
    }
}

std::optional<bool> json_boolean_field(const std::string& input, std::string_view field) {
    const std::string marker = "\"" + std::string(field) + "\"";
    auto cursor = input.find(marker);
    if (cursor == std::string::npos) return std::nullopt;
    cursor = input.find(':', cursor + marker.size());
    if (cursor == std::string::npos) return std::nullopt;
    ++cursor;
    while (cursor < input.size()
        && std::isspace(static_cast<unsigned char>(input[cursor]))) {
        ++cursor;
    }
    if (input.compare(cursor, 4, "true") == 0) return true;
    if (input.compare(cursor, 5, "false") == 0) return false;
    return std::nullopt;
}

std::vector<std::uint32_t> json_uint_array_field(const std::string& input, std::string_view field) {
    const std::string marker = "\"" + std::string(field) + "\"";
    auto cursor = input.find(marker);
    if (cursor == std::string::npos) return {};
    cursor = input.find('[', cursor + marker.size());
    if (cursor == std::string::npos) return {};
    std::vector<std::uint32_t> output;
    for (++cursor; cursor < input.size();) {
        while (cursor < input.size() && (std::isspace(static_cast<unsigned char>(input[cursor])) || input[cursor] == ',')) ++cursor;
        if (cursor >= input.size() || input[cursor] == ']') return output;
        std::size_t consumed{};
        unsigned long value{};
        try {
            value = std::stoul(input.substr(cursor), &consumed);
        } catch (...) {
            return {};
        }
        if (consumed == 0 || value > std::numeric_limits<std::uint32_t>::max()) return {};
        output.push_back(static_cast<std::uint32_t>(value));
        cursor += consumed;
    }
    return {};
}

std::string json_id(const std::string& input) {
    const auto marker = input.find("\"id\"");
    if (marker == std::string::npos) return "null";
    auto cursor = input.find(':', marker + 4);
    if (cursor == std::string::npos) return "null";
    ++cursor;
    while (cursor < input.size() && std::isspace(static_cast<unsigned char>(input[cursor]))) ++cursor;
    if (cursor >= input.size()) return "null";
    if (input[cursor] == '"') {
        const auto value = parse_json_string(input, cursor);
        return value ? "\"" + json_escape(*value) + "\"" : "null";
    }
    auto end = cursor;
    while (end < input.size() && input[end] != ',' && input[end] != '}') ++end;
    return input.substr(cursor, end - cursor);
}

void respond_ok(const std::string& id, std::string_view result) {
    std::cout << "{\"id\":" << id << ",\"ok\":true,\"result\":" << result << "}\n" << std::flush;
}

void respond_error(const std::string& id, std::string_view code, std::string_view message) {
    std::cout << "{\"id\":" << id << ",\"ok\":false,\"error\":{\"code\":\""
              << json_escape(code) << "\",\"message\":\"" << json_escape(message) << "\"}}\n" << std::flush;
}

std::string token_array_json(const std::vector<std::uint32_t>& tokens) {
    std::ostringstream output;
    output << '[';
    for (std::size_t index = 0; index < tokens.size(); ++index) {
        if (index != 0) output << ',';
        output << tokens[index];
    }
    output << ']';
    return output.str();
}

std::string fit_metrics_json(const FitPlan& fit) {
    std::ostringstream output;
    output << "\"fitDecision\":\"" << fit_decision_name(fit.decision) << "\""
           << ",\"fitReason\":\"" << json_escape(fit.reason) << "\""
           << ",\"fitStable\":" << (fit.stable ? "true" : "false")
           << ",\"fitBudgetLive\":" << (fit.live_budget ? "true" : "false")
           << ",\"fitBudgetSource\":\"" << json_escape(fit.budget_source) << "\""
           << ",\"fitAdapterName\":\"" << json_escape(fit.adapter_name) << "\""
           << ",\"residentLayerCount\":" << fit.resident_layer_count
           << ",\"totalLayerCount\":" << fit.total_layer_count
           << ",\"budgetBytes\":" << fit.budget_bytes
           << ",\"currentGpuUsageBytes\":" << fit.current_usage_bytes
           << ",\"availableFitBytes\":" << fit.available_bytes
           << ",\"fitActivationBytes\":" << fit.activation_bytes
           << ",\"fitContextTokens\":" << fit.context_tokens
           << ",\"fitKvCacheBytes\":" << fit.kv_cache_bytes
           << ",\"fitKvCpuMirrorBytes\":" << fit.kv_cpu_mirror_bytes
           << ",\"fitKvCacheMode\":\"" << kv_cache_mode_name(fit.kv_cache_mode) << "\""
           << ",\"fitWorkingSetBytes\":" << fit.working_set_bytes
           << ",\"fitReserveBytes\":" << fit.reserve_bytes
           << ",\"globalStagedWeightBytes\":" << fit.global_weight_bytes
           << ",\"totalStagedWeightBytes\":" << fit.total_staged_weight_bytes
           << ",\"gpuResidentWeightBytes\":" << fit.resident_weight_bytes
           << ",\"gpuWeightCacheEvictions\":" << fit.gpu_weight_cache_evictions
           << ",\"q4_0Fp16ScaleStaging\":" << (fit.q4_0_fp16_scale_staging ? "true" : "false")
           << ",\"fitHostEnvelope\":\"" << host_envelope_name(fit.host_envelope) << "\""
           << ",\"fitHostEnvelopeReason\":\"" << json_escape(fit.host_envelope_reason) << "\""
           << ",\"fitHostTotalRamBytes\":" << fit.host_total_ram_bytes
           << ",\"fitHostAvailableRamBytes\":" << fit.host_available_ram_bytes
           << ",\"fitHostWorkingSetBytes\":" << fit.host_working_set_bytes;
    return output.str();
}

std::string gpu_metrics_json(const State& state, const Session* session) {
    std::ostringstream output;
    output << "\"gpuExecution\":" << (state.gpu_model ? "true" : "false")
           << ",\"gpuExecutionReason\":\"" << json_escape(state.gpu_reason) << "\"";
    if (state.gpu_model) {
        const auto& telemetry = state.gpu_model->telemetry();
        output << ",\"gpuAdapterName\":\"" << json_escape(state.gpu_model->adapter_name()) << "\""
               << ",\"gpuSubmissionCount\":" << telemetry.submissions
               << ",\"gpuDynamicUploadBytes\":" << telemetry.dynamic_upload_bytes
               << ",\"gpuReadbackBytes\":" << telemetry.readback_bytes
               << ",\"gpuReadbackCount\":" << telemetry.readback_count
               << ",\"planGpuSpans\":" << state.gpu_model->plan_gpu_spans()
               << ",\"planCpuSpans\":" << state.gpu_model->plan_cpu_spans()
               << ",\"scheduledGpuNodes\":" << state.gpu_model->scheduled_gpu_nodes()
               << ",\"scheduledCpuNodes\":" << state.gpu_model->scheduled_cpu_nodes()
               << ",\"gpuKvCapacityTokens\":" << state.gpu_model->kv_capacity()
               << ",\"gpuResidentWeightBytesActual\":" << telemetry.resident_weight_bytes
               << ",\"gpuIntegerDotAvailable\":" << (telemetry.integer_dot_available ? "true" : "false")
               << ",\"gpuIntegerDotActive\":" << (telemetry.integer_dot_active ? "true" : "false")
               << ",\"gpuIntegerDotReason\":\"" << json_escape(telemetry.integer_dot_reason) << "\""
               << ",\"gpuQ8_1QuantizationCount\":" << telemetry.q8_1_quantizations
               << ",\"gpuIntegerDotMatvecCount\":" << telemetry.integer_dot_matvecs
               << ",\"gpuKvCacheMode\":\"" << kv_cache_mode_name(telemetry.kv_cache_mode) << "\""
               << ",\"gpuKvResident\":" << (telemetry.kv_gpu_resident ? "true" : "false")
               << ",\"gpuKvResidentBytes\":" << telemetry.kv_resident_bytes
               << ",\"gpuKvDeviceAppendCount\":" << telemetry.kv_device_appends
               << ",\"gpuNodeTimingAvailable\":" << (telemetry.node_timing_available ? "true" : "false")
               << ",\"gpuNodeMs\":{\"matmulLarge\":" << telemetry.node_class_ms[0]
               << ",\"matmulMid\":" << telemetry.node_class_ms[6]
               << ",\"matmulSmall\":" << telemetry.node_class_ms[7]
               << ",\"attention\":" << telemetry.node_class_ms[1]
               << ",\"rms\":" << telemetry.node_class_ms[2]
               << ",\"rope\":" << telemetry.node_class_ms[3]
               << ",\"elementwise\":" << telemetry.node_class_ms[4]
               << ",\"other\":" << telemetry.node_class_ms[5] << "}"
               << ",\"gpuNodeDispatches\":{\"matmulLarge\":" << telemetry.node_class_dispatches[0]
               << ",\"matmulMid\":" << telemetry.node_class_dispatches[6]
               << ",\"matmulSmall\":" << telemetry.node_class_dispatches[7]
               << ",\"attention\":" << telemetry.node_class_dispatches[1]
               << ",\"rms\":" << telemetry.node_class_dispatches[2]
               << ",\"rope\":" << telemetry.node_class_dispatches[3]
               << ",\"elementwise\":" << telemetry.node_class_dispatches[4]
               << ",\"other\":" << telemetry.node_class_dispatches[5] << "}";
    }
    if (session != nullptr) {
        output << ",\"submissionsPerToken\":" << session->submissions_per_token;
    }
    return output.str();
}

std::string session_metrics_json(const State& state, const Session& session, std::size_t generated = 0) {
    const bool gpu = session.gpu_state.has_value();
    const auto cached_tokens = gpu
        ? session.gpu_state->cached_tokens
        : (session.cpu_state ? session.cpu_state->cached_tokens : 0);
    const auto cache_bytes = gpu
        ? session.gpu_state->bytes(state.gpu_model ? state.gpu_model->per_token_kv_bytes() : 0)
        : (session.cpu_state ? session.cpu_state->bytes() : 0);
    std::ostringstream output;
    output << "{\"cachedTokenCount\":" << cached_tokens
           << ",\"sequenceTokenCount\":" << session.sequence.size()
           << ",\"generatedTokenCount\":" << generated
           << ",\"kvCacheBytes\":" << cache_bytes
           << ",\"speculationEnabled\":" << (session.speculation_enabled ? "true" : "false")
           << ",\"speculationReason\":\"" << json_escape(session.speculation_reason) << "\""
           << ",\"speculationProposedTokens\":" << session.speculation_proposed
           << ",\"speculationAcceptedTokens\":" << session.speculation_accepted
           << ",\"speculationVerificationBatches\":" << session.speculation_batches
           << ",\"speculationSavedGpuSteps\":" << session.speculation_saved_gpu_steps
           << ",\"speculationVerifyMs\":" << session.speculation_verify_ms
           // Phase 6 acceptance arithmetic, first-class: A = accepted tokens
           // per verification weight pass, c = verify-pass cost over a single
           // advance. Effective speedup ~ A/c; a run that cannot state A/c
           // has not measured anything.
           << ",\"speculationAcceptedTokensPerPass\":"
           << (session.speculation_batches != 0
               ? static_cast<double>(session.speculation_saved_gpu_steps)
                   / static_cast<double>(session.speculation_batches)
               : 0.0)
           << ",\"speculationVerifyCostMultiplier\":"
           << (session.speculation_batches != 0 && session.gpu_advance_samples != 0
                   && session.gpu_advance_ms > 0.0
               ? (session.speculation_verify_ms
                   / static_cast<double>(session.speculation_batches))
                   / (session.gpu_advance_ms
                       / static_cast<double>(session.gpu_advance_samples))
               : 0.0)
           << ",\"weightBytesPerToken\":" << session.weight_bytes_per_token
           << ",\"weightBytesSource\":\"" << json_escape(session.weight_bytes_source) << "\""
           << ",\"nativeBackend\":\"" << (gpu ? "directx-graph" : "cpu") << "\",\"cpuKernel\":\""
           << cpu_dot_kernel_name() << "\"," << gpu_metrics_json(state, &session)
           << ',' << fit_metrics_json(session.fit) << '}';
    return output.str();
}

std::string load_result_json(const State& state) {
    const auto& info = state.model->info();
    const bool gpu = state.gpu_model != nullptr;
    std::ostringstream output;
    output << "{\"runtimeProvider\":\"caterpillar\",\"caterpillarPhase\":7"
           << ",\"nativeBackend\":\"" << (gpu ? "directx-graph" : "cpu")
           << "\",\"cpuKernel\":\"" << cpu_dot_kernel_name()
           << "\",\"nativeModelStatus\":\"loaded\",\"executionReady\":true"
           << ",\"loadMs\":" << state.load_ms
           << ",\"model\":{\"id\":\"" << json_escape(state.model_id)
           << "\",\"architecture\":\"" << json_escape(info.architecture)
           << "\",\"quantization\":\"" << json_escape(info.quantization)
           << "\",\"vocabularySize\":" << info.vocabulary_size << "}"
           << ",\"compatibility\":{\"architecture\":{\"status\":\"supported\"},"
              "\"quantization\":{\"status\":\"supported\"},\"executor\":{\"status\":\"supported\"}}"
           << ",\"hybridBackends\":[" << (gpu ? "\"directx-graph\"" : "") << "],"
           << gpu_metrics_json(state, nullptr) << ',' << fit_metrics_json(state.fit)
           << ",\"gpuWeightBudgetBytes\":" << state.fit.budget_bytes
           << ",\"gpuResidentWeightBytes\":" << state.fit.resident_weight_bytes
           << ",\"gpuPrestageMs\":0,\"gpuMemory\":{\"effectiveWeightBudgetBytes\":"
           << state.fit.budget_bytes << "}}";
    return output.str();
}

}  // namespace

int run_rpc_server() {
    State state;
    std::string line;
    while (true) {
        const auto line_status = read_rpc_line(std::cin, line);
        if (line_status == RpcLineStatus::end) break;
        const auto id = json_id(line);
        if (line_status == RpcLineStatus::oversized) {
            respond_error(
                id,
                "CATERPILLAR_RPC_LINE_TOO_LARGE",
                "Caterpillar worker RPC line exceeds the 16777216-byte limit");
            continue;
        }
        const auto method = json_string_field(line, "method").value_or("");
        try {
            const auto rpc_version = json_number_field(line, "rpcVersion");
            if (!rpc_version || *rpc_version != 1.0) {
                respond_error(
                    id,
                    "CATERPILLAR_RPC_VERSION_UNSUPPORTED",
                    "Caterpillar worker requires rpcVersion 1");
                continue;
            }
            if (method == "handshake") {
                respond_ok(id,
                    "{\"worker\":\"caterpillar-runner\",\"rpcVersion\":1,\"rpcMinVersion\":1,\"rpcMaxVersion\":1,"
                    "\"runtimeProvider\":\"caterpillar\",\"phase\":7,\"cpuBuildProfile\":\"portable-reference\","
                    "\"speculativeDecoding\":true,"
                    "\"quantizations\":[\"f32\",\"f16\",\"bf16\",\"q4_0\",\"q8_0\",\"q4_k_m\",\"q5_k_m\",\"q6_k\",\"iq4_nl\",\"iq4_xs\"],"
                    "\"methods\":[\"handshake\",\"probe\",\"load_model\",\"unload_model\","
                    "\"session_open\",\"session_clone\",\"session_close\",\"prefill\",\"decode_batch\",\"generate\",\"cancel\",\"metrics\",\"shutdown\"]}");
            } else if (method == "probe") {
                respond_ok(id, "{\"runtimeProvider\":\"caterpillar\",\"phase\":7,\"backend\":\"graph-plan\",\"experimental\":true,\"speculativeDecoding\":true,\"qualityQuants\":[\"IQ4_NL\",\"IQ4_XS\"]}");
            } else if (method == "load_model") {
                const auto model_path = json_string_field(line, "modelPath");
                if (!model_path || model_path->empty()) throw std::invalid_argument("load_model requires modelArtifact.modelPath");
                const auto started = std::chrono::steady_clock::now();
                auto model = std::make_unique<CpuModel>(*model_path);
                state.load_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
                state.model_id = json_string_field(line, "modelId").value_or(model->info().name);
                state.model = std::move(model);
                const auto family = resolve_family_spec(state.model->info());
                const auto decoder = build_decoder_graph(family, 1);
                const auto activation = plan_activations(decoder.graph).arena_bytes;
                const auto requested_context = static_cast<std::size_t>(std::max(
                    1.0,
                    json_number_field(line, "contextSize").value_or(4096.0)
                ));
                const auto model_context = static_cast<std::size_t>(state.model->info().context_length);
                std::string gpu_reason;
                const bool device_available = gpu_backend_available(&gpu_reason);
                // Caterpillar's decode_batch executor owns independent CPU KV
                // state per session. Its current GPU graph owns one
                // device-global KV allocation, so it cannot safely represent
                // B=8/B=16 concurrent sessions. A product profile that opts
                // into batched decode therefore loads the measured CPU graph
                // lane (28.786 aggregate tok/s at B=16 on the qualification
                // host) even when a compatible GPU is present.
                const bool batched_decode_requested =
                    json_boolean_field(line, "batchedDecode").value_or(false);
                // Host-RAM envelope (Phase 2): probe what is actually free at
                // load time so the fit plan can refuse in-RAM-oversized models
                // instead of thrashing (adapter- and vendor-neutral: this is a
                // host property, not a GPU property).
                const auto host_memory = query_host_memory_info();
                state.fit_options = FitOptions{
                    .activation_bytes = activation,
                    .context_tokens = model_context == 0 ? requested_context : std::min(requested_context, model_context),
                    .gpu_backend_available =
                        device_available && !batched_decode_requested,
                    .video_memory = std::nullopt,
                    .requested_budget_bytes = std::nullopt,
                    .reserve_bytes = std::nullopt,
                    .kv_cache_mode = device_available
                        ? requested_kv_cache_mode()
                        : KvCacheMode::f32,
                    .host_total_ram_bytes = host_memory.live ? host_memory.total_bytes : 0,
                    .host_available_ram_bytes = host_memory.live ? host_memory.available_bytes : 0,
                };
                state.fit = plan_model_fit(state.model->info(), state.fit_options);
                state.gpu_model.reset();
                state.gpu_reason = batched_decode_requested
                    ? "cpu-batch-policy"
                    : gpu_reason;
                // Run 04 (2026-08-04) iteration 08: route on measured
                // throughput, not just on whether a device exists.
                //
                // The fit planner answers "do the weights fit on the device",
                // which is not the same question as "is the device faster".
                // Measured on llama-3.2-3B Q4_K_M, i7-1165G7 / Iris Xe, the
                // answer flips on whether the CPU integer-dot sidecar is live:
                //
                //   sidecar absent (float CPU path)   GPU 4.18 vs CPU 2.49  -> GPU
                //   sidecar live   (integer kernels)  GPU 4.13 vs CPU 8.18  -> CPU
                //
                // On a UMA adapter both lanes stream the same DRAM, so the iGPU
                // has no bandwidth advantage to trade for its dispatch overhead
                // once the CPU kernels are fast. Decline the GPU lane when the
                // sidecar is live. CATERPILLAR_ROUTE=gpu forces the device lane
                // back on, =cpu pins the CPU lane, unset applies this rule.
                const char* route = std::getenv("CATERPILLAR_ROUTE");
                const bool route_forces_gpu = route != nullptr
                    && std::string_view(route) == "gpu";
                const bool route_forces_cpu = route != nullptr
                    && std::string_view(route) == "cpu";
                const bool cpu_lane_is_fast = state.model->mapped().has_integer_sidecar();
                if (route_forces_cpu) {
                    state.gpu_reason = "route-pinned-cpu";
                } else if (cpu_lane_is_fast && !route_forces_gpu) {
                    state.gpu_reason = "cpu-lane-faster-integer-sidecar";
                }
                if (device_available && !batched_decode_requested
                    && state.fit.decision != FitDecision::cpu_only
                    && !route_forces_cpu
                    && (route_forces_gpu || !cpu_lane_is_fast)) {
                    state.gpu_model = GpuModel::try_create(
                        *state.model, state.fit, gpu_kv_capacity(state.fit_options), &state.gpu_reason);
                }
                if (!state.gpu_model) {
                    // Keep fit telemetry truthful: the CPU cache is f32 and
                    // no packed device cache exists without an executable
                    // GPU plan.
                    state.fit_options.gpu_backend_available = false;
                    state.fit_options.kv_cache_mode = KvCacheMode::f32;
                    state.fit = plan_model_fit(state.model->info(), state.fit_options);
                }
                if (state.gpu_model) state.gpu_reason = "gpu-decode-active";
                state.sessions.clear();
                // A freshly loaded model has an empty device KV; nothing from a
                // previous model may be treated as resident.
                state.resident_valid = false;
                state.resident_sequence.clear();
                respond_ok(id, load_result_json(state));
            } else if (method == "unload_model") {
                state.sessions.clear();
                state.gpu_model.reset();
                state.model.reset();
                state.resident_valid = false;
                state.resident_sequence.clear();
                respond_ok(id, "{\"nativeModelStatus\":\"unloaded\"}");
            } else if (method == "session_open") {
                if (!state.model) throw std::runtime_error("no Caterpillar model is loaded");
                const auto session_id = json_string_field(line, "sessionId");
                if (!session_id || session_id->empty()) throw std::invalid_argument("session_open requires sessionId");
                // With a pinned GPU plan the load-time decision IS the truth:
                // the weights are already resident, and a live re-query would
                // double-count our own staged bytes as foreign usage. CPU-only
                // models keep the Phase 2 behavior of re-checking the budget
                // at each sequence boundary.
                if (!state.gpu_model) {
                    state.fit = plan_model_fit(state.model->info(), state.fit_options);
                }
                Session session;
                session.fit = state.fit;
                session.speculation_max_draft = speculative_max_draft();
                session.speculation_draft_length = std::min<std::size_t>(4, session.speculation_max_draft);
                const auto mode = speculative_mode();
                const bool forced = mode == "1" || mode == "on" || mode == "true";
                // Host-envelope gate (Phase 2): speculation trades memory for
                // speed (draft batches, verification activations), so a tight
                // or oversized host disables the automatic mode. Forced mode
                // stays available as the explicit qualification override.
                const bool envelope_comfortable =
                    session.fit.host_envelope == HostEnvelope::comfortable;
                const bool automatic = mode == "auto" && envelope_comfortable;
                // Draft verification is a CPU batch GEMM (one weight pass for
                // the whole draft), so auto mode still requires the AVX2
                // integer-dot kernels to make that batch cheap; without them a
                // verify batch replays the scalar path and always loses.
                // Phase 6 (2026-07-21 campaign): CPU sessions speculate too -
                // CpuModel::verify_draft batches the whole draft through the
                // R8 tile GEMM, so one weight pass can emit several tokens.
                if (forced || (automatic && cpu_integer_dot_enabled())) {
                    session.speculation_enabled = true;
                    // Forced mode remains the broad qualification surface.
                    // Automatic mode skips weak two- and three-token suffixes:
                    // on normal prose they rarely accept enough tokens to pay
                    // for a target verification batch.
                    session.speculation_min_match_tokens = forced ? 2 : 4;
                    session.speculation_reason = forced ? "forced" : "auto-adaptive";
                } else if (mode == "0" || mode == "off" || mode == "false") {
                    session.speculation_reason = "disabled-by-config";
                } else if (mode == "auto" && !envelope_comfortable) {
                    session.speculation_reason = "disabled-host-envelope-"
                        + host_envelope_name(session.fit.host_envelope);
                } else {
                    session.speculation_reason = "auto-requires-cpu-integer-dot";
                }
                state.sessions[*session_id] = std::move(session);
                respond_ok(id, "{\"sessionId\":\"" + json_escape(*session_id)
                    + "\",\"nativeModelStatus\":\"loaded\"," + fit_metrics_json(state.fit) + "}");
            } else if (method == "session_clone") {
                const auto source_session_id =
                    json_string_field(line, "sourceSessionId").value_or("");
                const auto session_id = json_string_field(line, "sessionId").value_or("");
                if (source_session_id.empty() || session_id.empty()) {
                    throw std::invalid_argument(
                        "session_clone requires sourceSessionId and sessionId");
                }
                if (source_session_id == session_id) {
                    throw std::invalid_argument(
                        "session_clone source and target must differ");
                }
                const auto source = state.sessions.find(source_session_id);
                if (source == state.sessions.end()) {
                    throw std::invalid_argument(
                        "session_clone received an unknown source session");
                }
                // GpuSessionState contains counters for one device-global KV
                // allocation; copying those counters would make two sessions
                // alias mutable device KV. Advertise snapshot cloning for the
                // CPU execution class only until GpuModel owns per-session KV.
                if (!source->second.cpu_state
                    || source->second.cpu_state->cached_tokens == 0
                    || !source->second.cpu_state->next_token_valid) {
                    throw std::invalid_argument(
                        "session_clone requires a completed CPU prefill");
                }

                const auto started = std::chrono::steady_clock::now();
                Session cloned = source->second;
                cloned.gpu_state.reset();
                cloned.cancelled = false;
                cloned.stop_sequences = json_string_array_field(line, "stopSequences");
                cloned.max_tokens = static_cast<std::size_t>(std::max(
                    0.0,
                    json_number_field(line, "maxTokens", true).value_or(
                        static_cast<double>(cloned.max_tokens))));
                cloned.generated_tokens = 0;
                cloned.batch_stop_scan_tail.clear();
                cloned.submissions_per_token = 0.0;
                cloned.speculation_losing_streak = 0;
                cloned.speculation_resume_after_tokens = 0;
                cloned.speculation_proposed = 0;
                cloned.speculation_accepted = 0;
                cloned.speculation_batches = 0;
                cloned.speculation_saved_gpu_steps = 0;
                cloned.speculation_verify_ms = 0.0;
                cloned.gpu_advance_ms = 0.0;
                cloned.gpu_advance_samples = 0;
                cloned.weight_bytes_per_token = 0.0;
                cloned.weight_bytes_source = "snapshot-clone";
                cloned.prefill_ms = std::max(
                    0.001,
                    std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - started).count());
                state.sessions[session_id] = std::move(cloned);
                const auto& target = state.sessions.at(session_id);

                std::ostringstream result;
                result << "{\"sessionId\":\"" << json_escape(session_id)
                       << "\",\"sourceSessionId\":\"" << json_escape(source_session_id)
                       << "\",\"nativeBackend\":\"cpu\""
                       << ",\"prefillMs\":" << target.prefill_ms
                       << ",\"kvCacheBytes\":" << target.cpu_state->bytes()
                       << ",\"promptTokenIds\":[";
                for (std::size_t index = 0; index < target.sequence.size(); ++index) {
                    if (index != 0) result << ',';
                    result << target.sequence[index];
                }
                result << "],\"promptTokenCount\":" << target.sequence.size()
                       << ",\"cachedTokenCount\":" << target.cpu_state->cached_tokens
                       << ",\"promptTokenMode\":\"exact\""
                       << ",\"kvPrefixReusedTokens\":" << target.sequence.size()
                       << ",\"prefillSnapshotHit\":true"
                       << ",\"sessionMetrics\":"
                       << session_metrics_json(state, target) << '}';
                respond_ok(id, result.str());
            } else if (method == "session_close") {
                const auto session_id = json_string_field(line, "sessionId").value_or("");
                const bool closed = state.sessions.erase(session_id) != 0;
                respond_ok(id, "{\"sessionId\":\"" + json_escape(session_id) + "\",\"closed\":" + (closed ? "true" : "false") + "}");
            } else if (method == "prefill") {
                if (!state.model) throw std::runtime_error("no Caterpillar model is loaded");
                const auto session_id = json_string_field(line, "sessionId").value_or("");
                auto found = state.sessions.find(session_id);
                if (found == state.sessions.end()) throw std::invalid_argument("unknown Caterpillar session");
                const auto started = std::chrono::steady_clock::now();
                auto tokens = json_uint_array_field(line, "tokenIds");
                if (tokens.empty()) {
                    const auto prompt = json_string_field(line, "prompt").value_or("");
                    tokens = state.model->tokenizer().encode(prompt);
                }
                if (tokens.empty()) throw std::invalid_argument("prefill requires prompt token IDs");
                found->second.cpu_state.reset();
                found->second.gpu_state.reset();
                const bool use_gpu = state.gpu_model != nullptr
                    && found->second.fit.decision != FitDecision::cpu_only
                    && tokens.size() + 1 < state.gpu_model->kv_capacity();
                // Incremental prefill: the device KV already holds a prefix of
                // this prompt (the previous turn of the same conversation), so
                // ingest only the new tail through the resident decode graph
                // instead of re-running the entire prompt on the CPU engine.
                // Speculation retains a CPU verifier mirror that this path does
                // not rebuild, so it is only taken when speculation is off.
                // Opt-in, and chat is the only caller that opts in. Benchmarks
                // must keep the original CPU-seeded prefill: a resident cache
                // would make a repeated benchmark prompt look instant and break
                // comparability with recorded runs.
                const bool prompt_cache_allowed =
                    line.find("\"promptCache\":true") != std::string::npos;
                bool reused_resident = false;
                std::size_t reused_prefix_tokens = 0;
                if (prompt_cache_allowed && use_gpu && state.resident_valid
                    && !found->second.speculation_enabled) {
                    const auto shared = std::min(
                        common_prefix_tokens(state.resident_sequence, tokens),
                        state.resident_cached_tokens);
                    // At least one token must remain to ingest: the tail is what
                    // produces the logits the first generated token comes from.
                    if (shared > 0 && shared < tokens.size()) {
                        reused_prefix_tokens = shared;
                        GpuSessionState resumed;
                        // Attention only reads [0, cached_tokens), and advance()
                        // overwrites from there, so rewinding the count is all a
                        // truncation needs - the stale rows beyond it are
                        // re-written by the ingest below.
                        resumed.cached_tokens = shared;
                        resumed.next_token = 0;
                        resumed.next_token_valid = false;
                        for (std::size_t index = shared; index < tokens.size(); ++index) {
                            static_cast<void>(state.gpu_model->advance(resumed, tokens[index]));
                        }
                        found->second.gpu_state = std::move(resumed);
                        reused_resident = true;
                    }
                }
                if (reused_resident) {
                    // nothing further: the KV is already positioned
                } else if (prompt_cache_allowed && use_gpu
                    && !found->second.speculation_enabled && tokens.size() > 1) {
                    // Cold prompt, nothing resident to extend. The CPU engine
                    // still has to seed the packed cache - advance() needs a
                    // non-empty KV to append to - but only for the first token.
                    // The rest goes through the resident decode graph, which is
                    // several times cheaper per token than the CPU prefill it
                    // replaces (measured ~86ms vs ~400ms on this model).
                    const std::vector<std::uint32_t> seed(tokens.begin(), tokens.begin() + 1);
                    auto seeded = state.gpu_model->prefill(seed, false);
                    for (std::size_t index = 1; index < tokens.size(); ++index) {
                        static_cast<void>(state.gpu_model->advance(seeded, tokens[index]));
                    }
                    found->second.gpu_state = std::move(seeded);
                } else if (use_gpu) {
                    found->second.gpu_state = state.gpu_model->prefill(
                        tokens, found->second.speculation_enabled);
                } else {
                    // Phase 6: CPU sessions keep their speculation setting -
                    // the CPU batch verifier is exactly the route with
                    // headroom above the memory roofline.
                    found->second.cpu_state = state.model->prefill(tokens);
                }
                found->second.stop_sequences = json_string_array_field(line, "stopSequences");
                found->second.max_tokens = static_cast<std::size_t>(std::max(
                    0.0,
                    json_number_field(line, "maxTokens", true).value_or(64.0)));
                found->second.generated_tokens = 0;
                found->second.batch_stop_scan_tail.clear();
                found->second.sequence = std::move(tokens);
                found->second.cancelled = false;
                // Record what now occupies the device KV so the next turn can
                // extend it. A CPU prefill leaves the device cache untouched by
                // this prompt, so the cache is dropped rather than lied about.
                if (found->second.gpu_state) {
                    const auto cached = found->second.gpu_state->cached_tokens;
                    if (cached > 0 && cached <= found->second.sequence.size()) {
                        state.resident_sequence.assign(
                            found->second.sequence.begin(),
                            found->second.sequence.begin() + static_cast<std::ptrdiff_t>(cached));
                        state.resident_cached_tokens = cached;
                        state.resident_next_token = found->second.gpu_state->next_token;
                        state.resident_next_token_valid = found->second.gpu_state->next_token_valid;
                        state.resident_valid = true;
                    } else {
                        state.resident_valid = false;
                        state.resident_sequence.clear();
                    }
                } else {
                    state.resident_valid = false;
                    state.resident_sequence.clear();
                }
                found->second.prefill_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
                std::ostringstream result;
                result << "{\"prefillMs\":" << found->second.prefill_ms
                       << ",\"nativeBackend\":\"" << (use_gpu ? "directx-graph" : "cpu")
                       << "\",\"promptCacheReused\":" << (reused_resident ? "true" : "false")
                       << ",\"promptCacheReusedTokens\":" << reused_prefix_tokens
                       << ",\"nativeModelStatus\":\"loaded\",\"sessionMetrics\":"
                       << session_metrics_json(state, found->second) << '}';
                respond_ok(id, result.str());
            } else if (method == "decode_batch") {
                if (!state.model) {
                    throw std::runtime_error(
                        "decode_batch requires a loaded Caterpillar model");
                }
                const auto session_ids =
                    json_string_array_field(line, "sessionIds");
                if (session_ids.empty()) {
                    throw std::invalid_argument(
                        "decode_batch requires a non-empty sessionIds array");
                }

                struct Emission {
                    std::string session_id;
                    Session* session{};
                    std::uint32_t token{};
                    std::string text;
                    bool finished{};
                    std::string finish_reason;
                };
                std::vector<Emission> emitted;
                emitted.reserve(session_ids.size());
                std::map<std::size_t, std::vector<std::size_t>> advance_groups;
                constexpr std::size_t kStopScanTailMax = 64;

                for (const auto& session_id : session_ids) {
                    const auto found = state.sessions.find(session_id);
                    if (found == state.sessions.end()) {
                        throw std::invalid_argument(
                            "decode_batch received an unknown session: " + session_id);
                    }
                    auto& session = found->second;
                    if (!session.cpu_state || session.gpu_state) {
                        throw std::invalid_argument(
                            "decode_batch currently requires CPU Caterpillar sessions");
                    }
                    if (!session.cpu_state->next_token_valid) {
                        throw std::invalid_argument(
                            "decode_batch session has no pending token: " + session_id);
                    }

                    Emission item;
                    item.session_id = session_id;
                    item.session = &session;
                    item.token = session.cpu_state->next_token;
                    item.text = state.model->tokenizer().token_text(item.token);
                    session.cpu_state->next_token_valid = false;
                    session.sequence.push_back(item.token);
                    ++session.generated_tokens;

                    if (session.cancelled) {
                        item.finished = true;
                        item.finish_reason = "cancelled";
                    } else if (item.token == state.model->info().eos_token_id) {
                        item.finished = true;
                        item.finish_reason = "eos";
                    } else {
                        session.batch_stop_scan_tail +=
                            decode_bpe_whitespace(item.text);
                        if (session.batch_stop_scan_tail.size() > kStopScanTailMax) {
                            session.batch_stop_scan_tail.erase(
                                0,
                                session.batch_stop_scan_tail.size()
                                    - kStopScanTailMax);
                        }
                        for (const auto& marker : session.stop_sequences) {
                            if (!marker.empty()
                                && session.batch_stop_scan_tail.find(marker)
                                    != std::string::npos) {
                                item.finished = true;
                                item.finish_reason = "stop-sequence";
                                break;
                            }
                        }
                    }
                    if (!item.finished && session.max_tokens != 0
                        && session.generated_tokens >= session.max_tokens) {
                        item.finished = true;
                        item.finish_reason = "max-tokens";
                    }
                    emitted.push_back(std::move(item));
                    if (!emitted.back().finished) {
                        advance_groups[session.cpu_state->cached_tokens].push_back(
                            emitted.size() - 1);
                    }
                }

                const auto weight_bytes_before = cpu_weight_bytes_streamed();
                const auto started = std::chrono::steady_clock::now();
                std::size_t effective_rows = 0;
                std::size_t coalesced_sessions = 0;
                for (const auto& [cached_tokens, indices] : advance_groups) {
                    static_cast<void>(cached_tokens);
                    // Snapshot clones receiving deterministic requests remain
                    // byte-identical: same sequence, same KV, same pending
                    // token. Advancing every duplicate row repeats attention
                    // and activation work for no change in result. Advance one
                    // representative per unique sequence, then deep-copy its
                    // newly appended CPU KV into the equivalent sessions.
                    std::vector<std::vector<std::size_t>> equivalent;
                    for (const auto index : indices) {
                        const auto group = std::find_if(
                            equivalent.begin(),
                            equivalent.end(),
                            [&](const std::vector<std::size_t>& candidate) {
                                return emitted[candidate.front()].session->sequence
                                    == emitted[index].session->sequence;
                            });
                        if (group == equivalent.end()) {
                            equivalent.push_back({index});
                        } else {
                            group->push_back(index);
                        }
                    }

                    std::vector<CpuSessionState*> sessions;
                    std::vector<std::uint32_t> tokens;
                    sessions.reserve(equivalent.size());
                    tokens.reserve(equivalent.size());
                    for (const auto& group : equivalent) {
                        const auto index = group.front();
                        sessions.push_back(
                            &emitted[index].session->cpu_state.value());
                        tokens.push_back(emitted[index].token);
                    }
                    effective_rows += sessions.size();
                    coalesced_sessions += indices.size() - sessions.size();
                    if (sessions.size() == 1) {
                        static_cast<void>(
                            state.model->advance(*sessions.front(), tokens.front()));
                    } else {
                        static_cast<void>(
                            state.model->advance_batch(sessions, tokens));
                    }
                    for (const auto& group : equivalent) {
                        const auto& representative =
                            *emitted[group.front()].session->cpu_state;
                        for (std::size_t member = 1;
                            member < group.size();
                            ++member) {
                            *emitted[group[member]].session->cpu_state =
                                representative;
                        }
                    }
                }
                const auto elapsed_ms =
                    std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - started).count();
                const auto weight_bytes =
                    cpu_weight_bytes_streamed() - weight_bytes_before;

                std::ostringstream result;
                result << "{\"batchSize\":" << session_ids.size()
                       << ",\"stepMs\":" << elapsed_ms
                       << ",\"backend\":\"cpu-batched-graph\""
                       << ",\"effectiveRows\":" << effective_rows
                       << ",\"coalescedSessions\":" << coalesced_sessions
                       << ",\"weightBytesPerStep\":" << weight_bytes
                       << ",\"tokens\":[";
                for (std::size_t index = 0; index < emitted.size(); ++index) {
                    if (index != 0) result << ',';
                    const auto& item = emitted[index];
                    result << "{\"sessionId\":\""
                           << json_escape(item.session_id)
                           << "\",\"tokenId\":" << item.token
                           << ",\"token\":\"" << json_escape(item.text)
                           << "\",\"suppressed\":false"
                           << ",\"cachedTokenCount\":"
                           << item.session->cpu_state->cached_tokens
                           << ",\"generatedTokenCount\":"
                           << item.session->generated_tokens
                           << ",\"finished\":"
                           << (item.finished ? "true" : "false")
                           << ",\"finishReason\":";
                    if (item.finish_reason.empty()) {
                        result << "null";
                    } else {
                        result << '"' << json_escape(item.finish_reason) << '"';
                    }
                    result << '}';
                }
                result << "]}";
                respond_ok(id, result.str());
            } else if (method == "generate") {
                if (!state.model) throw std::runtime_error("no Caterpillar model is loaded");
                const auto session_id = json_string_field(line, "sessionId").value_or("");
                auto found = state.sessions.find(session_id);
                if (found == state.sessions.end()) throw std::invalid_argument("unknown Caterpillar session");
                const double temperature = json_number_field(line, "temperature").value_or(0.0);
                if (temperature > 0.0) throw std::invalid_argument("Caterpillar currently accepts deterministic temperature=0 sampling");
                // maxTokens can also occur in the nested runtime profile. The
                // request-level generation limit is serialized after profile,
                // so select the final occurrence instead of silently using a
                // profile default (often 160) for a one-token request.
                const auto max_tokens = static_cast<std::size_t>(
                    std::max(0.0, json_number_field(line, "maxTokens", true).value_or(64.0)));
                const auto started = std::chrono::steady_clock::now();
                std::vector<std::uint32_t> generated;
                generated.reserve(max_tokens);
                const bool gpu = found->second.gpu_state.has_value();
                if (!gpu && !found->second.cpu_state) throw std::logic_error("Caterpillar session has not been prefilled");
                const auto submissions_before = state.gpu_model ? state.gpu_model->telemetry().submissions : 0;
                const auto weight_bytes_before = cpu_weight_bytes_streamed();
                const auto speculative_steps_before = found->second.speculation_saved_gpu_steps;
                std::size_t advance_steps = 0;
                const auto emit_token = [&](std::uint32_t token) {
                    generated.push_back(token);
                    found->second.sequence.push_back(token);
                    // sessionId is what routes this to the caller waiting on it:
                    // the client registers its listener under the session it opened,
                    // and a notification without one falls through to the no-session
                    // listener - which is null in exactly that case, so every token was
                    // dropped and chat only ever saw the final generate result.
                    std::cout << "{\"jsonrpc\":\"2.0\",\"method\":\"generate.token\",\"params\":{"
                              << "\"sessionId\":\"" << json_escape(session_id) << "\","
                              << "\"index\":" << (generated.size() - 1)
                              << ",\"tokenId\":" << token
                              << ",\"token\":\"" << json_escape(state.model->tokenizer().token_text(token)) << "\"}}\n" << std::flush;
                };
                bool reached_eos = false;
                bool stopped_by_sequence = false;
                // Rolling decoded tail for stop-marker matching. Markers span
                // several tokens, so matching one token at a time would miss
                // them; only the last few dozen characters can contain one.
                std::string stop_scan_tail;
                const std::size_t kStopScanTailMax = 64;
                const auto hits_stop_sequence = [&](std::uint32_t token) {
                    if (found->second.stop_sequences.empty()) return false;
                    stop_scan_tail += decode_bpe_whitespace(
                        state.model->tokenizer().token_text(token));
                    if (stop_scan_tail.size() > kStopScanTailMax) {
                        stop_scan_tail.erase(0, stop_scan_tail.size() - kStopScanTailMax);
                    }
                    for (const auto& marker : found->second.stop_sequences) {
                        if (!marker.empty() && stop_scan_tail.find(marker) != std::string::npos) {
                            return true;
                        }
                    }
                    return false;
                };
                while (generated.size() < max_tokens && !found->second.cancelled && !reached_eos) {
                    std::uint32_t token{};
                    if (gpu) {
                        auto& gpu_state = *found->second.gpu_state;
                        if (!gpu_state.next_token_valid) {
                            if (found->second.sequence.empty()) throw std::logic_error("incremental session has no sequence tail");
                            const auto advance_started = std::chrono::steady_clock::now();
                            static_cast<void>(state.gpu_model->advance(gpu_state, found->second.sequence.back()));
                            found->second.gpu_advance_ms += std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - advance_started).count();
                            ++found->second.gpu_advance_samples;
                            ++advance_steps;
                        }
                        token = gpu_state.next_token;
                        gpu_state.next_token_valid = false;
                    } else {
                        auto& cpu_state = *found->second.cpu_state;
                        if (!cpu_state.next_token_valid) {
                            if (found->second.sequence.empty()) throw std::logic_error("incremental session has no sequence tail");
                            const auto advance_started = std::chrono::steady_clock::now();
                            static_cast<void>(state.model->advance(cpu_state, found->second.sequence.back()));
                            found->second.gpu_advance_ms += std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - advance_started).count();
                            ++found->second.gpu_advance_samples;
                            ++advance_steps;
                        }
                        token = cpu_state.next_token;
                        cpu_state.next_token_valid = false;
                    }
                    emit_token(token);
                    if (token == state.model->info().eos_token_id) {
                        reached_eos = true;
                        break;
                    }
                    // The turn ended at a chat/instruction marker. The marker's
                    // own tokens are already emitted; the streaming filter on
                    // the client trims them from what the user sees.
                    if (hits_stop_sequence(token)) {
                        stopped_by_sequence = true;
                        reached_eos = true;
                        break;
                    }

                    if (gpu && found->second.speculation_enabled
                        && found->second.gpu_advance_samples != 0
                        && generated.size() < max_tokens
                        && generated.size() >= found->second.speculation_resume_after_tokens) {
                        const auto remaining = max_tokens - generated.size();
                        const auto draft = draft_from_ngram(
                            found->second.sequence,
                            std::min(found->second.speculation_draft_length, remaining),
                            found->second.speculation_min_match_tokens);
                        if (!draft.tokens.empty()) {
                            std::vector<std::uint32_t> verifier_inputs;
                            verifier_inputs.reserve(draft.tokens.size() + 1);
                            verifier_inputs.push_back(token);
                            verifier_inputs.insert(
                                verifier_inputs.end(), draft.tokens.begin(), draft.tokens.end());
                            const auto verify_started = std::chrono::steady_clock::now();
                            const auto verification = state.gpu_model->verify_draft(
                                *found->second.gpu_state, verifier_inputs);
                            const auto verify_ms = std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - verify_started).count();
                            std::size_t accepted = 0;
                            while (accepted < draft.tokens.size()
                                && verification.next_tokens[accepted] == draft.tokens[accepted]) {
                                ++accepted;
                                if (draft.tokens[accepted - 1] == state.model->info().eos_token_id) break;
                            }
                            const auto committed_inputs = accepted + 1;
                            state.gpu_model->commit_draft(
                                *found->second.gpu_state, verification, committed_inputs);
                            ++found->second.speculation_batches;
                            found->second.speculation_proposed += draft.tokens.size();
                            found->second.speculation_accepted += accepted;
                            found->second.speculation_saved_gpu_steps += committed_inputs;
                            found->second.speculation_verify_ms += verify_ms;
                            for (std::size_t index = 0; index < accepted && generated.size() < max_tokens; ++index) {
                                emit_token(draft.tokens[index]);
                                if (draft.tokens[index] == state.model->info().eos_token_id) {
                                    reached_eos = true;
                                    break;
                                }
                                if (hits_stop_sequence(draft.tokens[index])) {
                                    stopped_by_sequence = true;
                                    reached_eos = true;
                                    break;
                                }
                            }

                            // Adaptive session controller (Step 02 tuning).
                            // Economics per batch: the verify replay reads the
                            // weights once for the whole draft, so a batch wins
                            // whenever its per-committed-input cost beats one
                            // plain GPU step. Wins grow the next draft (more
                            // tokens verified per weight pass); losses shrink
                            // it, and a losing streak pauses drafting for a
                            // stretch of tokens instead of disabling forever —
                            // n-gram acceptance is bursty and typically starts
                            // winning once the answer develops repetition.
                            const double gpu_ms = found->second.gpu_advance_ms
                                / static_cast<double>(found->second.gpu_advance_samples);
                            const bool batch_won =
                                verify_ms / static_cast<double>(committed_inputs) < gpu_ms * 0.98;
                            if (batch_won) {
                                found->second.speculation_losing_streak = 0;
                                found->second.speculation_draft_length = std::min(
                                    found->second.speculation_max_draft,
                                    found->second.speculation_draft_length * 2);
                                found->second.speculation_reason = "a/b-faster";
                            } else {
                                found->second.speculation_draft_length = std::max<std::size_t>(
                                    2, found->second.speculation_draft_length / 2);
                                if (++found->second.speculation_losing_streak >= 3) {
                                    found->second.speculation_losing_streak = 0;
                                    found->second.speculation_resume_after_tokens =
                                        generated.size() + 24;
                                    found->second.speculation_reason = "a/b-paused-losing-streak";
                                } else {
                                    found->second.speculation_reason = "a/b-lost-batch";
                                }
                            }
                        }
                    }

                    // Phase 6 (2026-07-21 campaign): CPU speculative batches.
                    // One CpuModel::verify_draft call runs [token, draft...]
                    // through the batched R8 tile GEMM - the weight stream is
                    // read ONCE for the whole batch, so every accepted draft
                    // token is a token emitted without a weight pass. The
                    // iteration-006 confidence gate, exact-prefix acceptance,
                    // and adaptive economics mirror the GPU controller.
                    if (!gpu && found->second.speculation_enabled
                        && found->second.gpu_advance_samples != 0
                        && generated.size() < max_tokens
                        && generated.size() >= found->second.speculation_resume_after_tokens) {
                        const auto remaining = max_tokens - generated.size();
                        const auto draft = draft_from_ngram(
                            found->second.sequence,
                            std::min(found->second.speculation_draft_length, remaining),
                            found->second.speculation_min_match_tokens);
                        if (!draft.tokens.empty()) {
                            auto& cpu_state = *found->second.cpu_state;
                            std::vector<std::uint32_t> verifier_inputs;
                            verifier_inputs.reserve(draft.tokens.size() + 1);
                            verifier_inputs.push_back(token);
                            verifier_inputs.insert(
                                verifier_inputs.end(), draft.tokens.begin(), draft.tokens.end());
                            const auto verify_started = std::chrono::steady_clock::now();
                            const auto verification = state.model->verify_draft(
                                cpu_state, verifier_inputs);
                            const auto verify_ms = std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - verify_started).count();
                            std::size_t accepted = 0;
                            while (accepted < draft.tokens.size()
                                && verification.next_tokens[accepted] == draft.tokens[accepted]) {
                                ++accepted;
                                if (draft.tokens[accepted - 1] == state.model->info().eos_token_id) break;
                            }
                            const auto committed_inputs = accepted + 1;
                            state.model->commit_draft(cpu_state, verification, committed_inputs);
                            ++found->second.speculation_batches;
                            found->second.speculation_proposed += draft.tokens.size();
                            found->second.speculation_accepted += accepted;
                            found->second.speculation_saved_gpu_steps += committed_inputs;
                            found->second.speculation_verify_ms += verify_ms;
                            for (std::size_t index = 0; index < accepted && generated.size() < max_tokens; ++index) {
                                emit_token(draft.tokens[index]);
                                if (draft.tokens[index] == state.model->info().eos_token_id) {
                                    reached_eos = true;
                                    break;
                                }
                                if (hits_stop_sequence(draft.tokens[index])) {
                                    stopped_by_sequence = true;
                                    reached_eos = true;
                                    break;
                                }
                            }
                            const double advance_ms = found->second.gpu_advance_ms
                                / static_cast<double>(found->second.gpu_advance_samples);
                            const bool batch_won =
                                verify_ms / static_cast<double>(committed_inputs) < advance_ms * 0.98;
                            if (batch_won) {
                                found->second.speculation_losing_streak = 0;
                                found->second.speculation_draft_length = std::min(
                                    found->second.speculation_max_draft,
                                    found->second.speculation_draft_length * 2);
                                found->second.speculation_reason = "a/b-faster";
                            } else {
                                found->second.speculation_draft_length = std::max<std::size_t>(
                                    2, found->second.speculation_draft_length / 2);
                                if (++found->second.speculation_losing_streak >= 3) {
                                    found->second.speculation_losing_streak = 0;
                                    found->second.speculation_resume_after_tokens =
                                        generated.size() + 24;
                                    found->second.speculation_reason = "a/b-paused-losing-streak";
                                } else {
                                    found->second.speculation_reason = "a/b-lost-batch";
                                }
                            }
                        }
                    }
                }
                const double decode_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
                const double tokens_per_second = decode_ms > 0.0 ? static_cast<double>(generated.size()) * 1000.0 / decode_ms : 0.0;
                const auto speculative_steps = found->second.speculation_saved_gpu_steps - speculative_steps_before;
                const auto target_steps = advance_steps + speculative_steps;
                if (gpu && state.gpu_model && target_steps != 0) {
                    const auto submissions_after = state.gpu_model->telemetry().submissions;
                    found->second.submissions_per_token =
                        static_cast<double>(submissions_after - submissions_before) / static_cast<double>(target_steps);
                }
                const auto cache_bytes = gpu
                    ? found->second.gpu_state->bytes(state.gpu_model ? state.gpu_model->per_token_kv_bytes() : 0)
                    : found->second.cpu_state->bytes();
                // Roofline telemetry: CPU weight traffic is measured exactly by
                // the tensor entry points; GPU-resident decode re-reads the
                // staged weight set once per target step, so that side is
                // derived from the fit plan's staged byte total. Both terms sum
                // so hybrid plans report the whole stream.
                auto weight_bytes = cpu_weight_bytes_streamed() - weight_bytes_before;
                found->second.weight_bytes_source = weight_bytes != 0 ? "measured-cpu" : "unmeasured";
                if (gpu && target_steps != 0) {
                    weight_bytes += found->second.fit.total_staged_weight_bytes
                        * static_cast<std::uint64_t>(target_steps);
                    found->second.weight_bytes_source = "derived-gpu-resident";
                }
                found->second.weight_bytes_per_token = generated.empty()
                    ? 0.0
                    : static_cast<double>(weight_bytes) / static_cast<double>(generated.size());
                const double effective_weight_gbs =
                    found->second.weight_bytes_per_token * tokens_per_second / 1.0e9;
                const double roofline = roofline_gbs();
                // Decode appended every generated token to the device KV, so the
                // resident cache now covers prompt + answer. Re-point it at the
                // session, which is what the next turn of this conversation will
                // extend. A cancelled or CPU run leaves it invalid instead.
                if (gpu && found->second.gpu_state) {
                    // The decode loop appends a token to `sequence` before the
                    // next iteration advances it into the KV, so cached_tokens
                    // trails sequence by one. Record exactly the tokens the KV
                    // actually holds - sequence[0, cached_tokens) - rather than
                    // the whole sequence, or the next turn's prefix match is
                    // compared against a token the device never saw.
                    const auto cached = found->second.gpu_state->cached_tokens;
                    if (cached > 0 && cached <= found->second.sequence.size()) {
                        state.resident_sequence.assign(
                            found->second.sequence.begin(),
                            found->second.sequence.begin() + static_cast<std::ptrdiff_t>(cached));
                        state.resident_cached_tokens = cached;
                        state.resident_next_token = found->second.gpu_state->next_token;
                        state.resident_next_token_valid = found->second.gpu_state->next_token_valid;
                        state.resident_valid = true;
                    } else {
                        state.resident_valid = false;
                        state.resident_sequence.clear();
                    }
                } else {
                    state.resident_valid = false;
                    state.resident_sequence.clear();
                }
                const auto text = state.model->tokenizer().decode(generated);
                const std::string finish_reason = found->second.cancelled
                    ? "cancelled"
                    : stopped_by_sequence
                        ? "stop-sequence"
                        : reached_eos
                            ? "eos"
                            : generated.size() >= max_tokens ? "max-tokens" : "unknown";
                std::ostringstream result;
                result << "{\"generatedTokenIds\":" << token_array_json(generated)
                       << ",\"output\":\"" << json_escape(text) << "\",\"chunks\":[\"" << json_escape(text) << "\"]"
                       << ",\"finishReason\":\"" << finish_reason << "\""
                       << ",\"nativeBackend\":\"" << (gpu ? "directx-graph" : "cpu")
                       << "\",\"nativeModelStatus\":\"loaded\",\"loadMs\":" << state.load_ms
                       << ",\"prefillMs\":" << found->second.prefill_ms << ",\"decodeMs\":" << decode_ms
                       << ",\"tokensPerSecond\":" << tokens_per_second
                       << ",\"weightBytesPerToken\":" << found->second.weight_bytes_per_token
                       << ",\"weightBytesSource\":\"" << json_escape(found->second.weight_bytes_source) << "\""
                       << ",\"effectiveWeightGBs\":" << effective_weight_gbs
                       << ",\"rooflineGBs\":" << roofline
                       << ",\"rooflineFraction\":" << (roofline > 0.0 ? effective_weight_gbs / roofline : 0.0)
                       << ",\"kvCacheBytes\":" << cache_bytes
                       << ",\"compatibility\":{\"executor\":{\"status\":\"supported\"}},\"sessionMetrics\":"
                       << session_metrics_json(state, found->second, generated.size()) << '}';
                respond_ok(id, result.str());
            } else if (method == "cancel") {
                const auto session_id = json_string_field(line, "sessionId").value_or("");
                bool cancelled = false;
                if (auto found = state.sessions.find(session_id); found != state.sessions.end()) {
                    found->second.cancelled = true;
                    cancelled = true;
                }
                respond_ok(id, std::string("{\"cancelled\":") + (cancelled ? "true" : "false") + "}");
            } else if (method == "metrics") {
                std::ostringstream result;
                result << "{\"runtimeProvider\":\"caterpillar\",\"caterpillarPhase\":7,\"nativeBackend\":\""
                       << (state.gpu_model ? "directx-graph" : "cpu")
                       << "\",\"nativeModelStatus\":\"" << (state.model ? "loaded" : "unloaded")
                       << "\",\"loadMs\":" << state.load_ms << ",\"activeSessions\":" << state.sessions.size();
                if (state.model) {
                    result << ',' << gpu_metrics_json(state, nullptr) << ',' << fit_metrics_json(state.fit);
                }
                result << '}';
                respond_ok(id, result.str());
            } else if (method == "tensor_benchmark") {
                respond_error(id, "CATERPILLAR_CPU_ONLY", "Caterpillar Phase 2 has no GPU calibration lane");
            } else if (method == "shutdown") {
                respond_ok(id, "{\"shutdown\":true}");
                return EXIT_SUCCESS;
            } else {
                respond_error(id, "CATERPILLAR_METHOD_NOT_FOUND", "Unsupported Caterpillar RPC method: " + method);
            }
        } catch (const std::exception& error) {
            respond_error(id, "CATERPILLAR_WORKER_ERROR", error.what());
        }
    }
    return EXIT_SUCCESS;
}

}  // namespace cat
