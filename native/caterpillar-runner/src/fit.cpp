// Stable model-to-VRAM fit planning for Caterpillar Phase 2.
// Idea sources: Microsoft DXGI budget documentation and the public
// auto-fit-to-VRAM concept recorded in .Agent/caterpillar/03. This is a
// clean-room implementation; no llama.cpp/ggml/ik_llama.cpp code is copied.

#include "fit.h"

#include "family_spec.h"
#include "tensor_data.h"

#include <algorithm>
#include <charconv>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>
#include <string_view>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <dxgi1_4.h>
#include <d3d12.h>
#endif

namespace cat {
namespace {

constexpr std::uint64_t mib = 1024ULL * 1024ULL;
constexpr std::uint64_t default_emergency_reserve = 256ULL * mib;

std::uint64_t checked_add(std::uint64_t left, std::uint64_t right, std::string_view label) {
    if (right > std::numeric_limits<std::uint64_t>::max() - left) {
        throw std::overflow_error(std::string(label) + " byte count overflowed");
    }
    return left + right;
}

std::optional<std::size_t> tensor_layer(std::string_view name) {
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

std::optional<std::uint64_t> budget_from_environment() {
    const char* value = std::getenv("LIZARD_NATIVE_GPU_WEIGHT_BUDGET_MB");
    if (value == nullptr || *value == '\0') return std::nullopt;
    char* end{};
    const auto parsed = std::strtoull(value, &end, 10);
    if (end == value || *end != '\0' || parsed == 0
        || parsed > std::numeric_limits<std::uint64_t>::max() / mib) {
        return std::nullopt;
    }
    return parsed * mib;
}

std::uint64_t default_reserve(std::uint64_t budget) {
    // DXGI already accounts for the desktop/display working set when it gives
    // this process a budget. Keep an additional 5% (at least 256 MiB) as the
    // Caterpillar emergency margin so a plan never intentionally saturates it.
    return std::max(default_emergency_reserve, budget / 20ULL);
}

#ifdef _WIN32
struct ComReleaser {
    template <typename T>
    void operator()(T* value) const noexcept {
        if (value != nullptr) value->Release();
    }
};

template <typename T>
using ComOwner = std::unique_ptr<T, ComReleaser>;

bool parse_requested_luid(LUID& output) {
    const char* value = std::getenv("LIZARD_NATIVE_GPU_ADAPTER_LUID");
    if (value == nullptr || *value == '\0') return false;
    char* separator{};
    const auto high = std::strtoll(value, &separator, 10);
    if (separator == value || separator == nullptr || *separator != ':') return false;
    char* end{};
    const auto low = std::strtoull(separator + 1, &end, 10);
    if (end == separator + 1 || *end != '\0'
        || high < std::numeric_limits<LONG>::min() || high > std::numeric_limits<LONG>::max()
        || low > std::numeric_limits<DWORD>::max()) {
        return false;
    }
    output.HighPart = static_cast<LONG>(high);
    output.LowPart = static_cast<DWORD>(low);
    return true;
}

bool luid_equal(const LUID& left, const LUID& right) {
    return left.HighPart == right.HighPart && left.LowPart == right.LowPart;
}

std::string narrow_adapter_name(const wchar_t* value) {
    if (value == nullptr || *value == L'\0') return {};
    const int length = WideCharToMultiByte(CP_UTF8, 0, value, -1, nullptr, 0, nullptr, nullptr);
    if (length <= 1) return {};
    std::string output(static_cast<std::size_t>(length), '\0');
    WideCharToMultiByte(CP_UTF8, 0, value, -1, output.data(), length, nullptr, nullptr);
    output.pop_back();
    return output;
}
#endif

}  // namespace

std::string fit_decision_name(FitDecision decision) {
    switch (decision) {
        case FitDecision::all_resident: return "all-resident";
        case FitDecision::stable_split: return "stable-split";
        case FitDecision::cpu_only: return "cpu-only";
    }
    return "cpu-only";
}

std::string host_envelope_name(HostEnvelope envelope) {
    switch (envelope) {
        case HostEnvelope::comfortable: return "comfortable";
        case HostEnvelope::tight: return "tight";
        case HostEnvelope::oversized: return "oversized";
    }
    return "oversized";
}

HostMemoryInfo query_host_memory_info() {
    HostMemoryInfo result;
#ifdef _WIN32
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    if (GlobalMemoryStatusEx(&status) != 0) {
        result.live = true;
        result.total_bytes = status.ullTotalPhys;
        result.available_bytes = status.ullAvailPhys;
    }
#endif
    return result;
}

std::uint64_t staged_tensor_bytes(const GgufTensorInfo& tensor) {
    // GGML type 2 is Q4_0: one fp16 scale plus 16 packed bytes for 32
    // weights. Caterpillar preserves that 18-byte representation instead of
    // widening the scale to f32 (20 bytes) during GPU staging.
    if (tensor.type == 2) {
        const auto elements = tensor_element_count(tensor);
        if (elements % 32 != 0) throw std::invalid_argument("Q4_0 tensor has a partial block");
        return static_cast<std::uint64_t>(elements / 32) * 18ULL;
    }
    return static_cast<std::uint64_t>(tensor_byte_length(tensor));
}

std::uint64_t estimate_kv_cache_bytes(
    const GgufModelInfo& model,
    std::size_t context_tokens,
    KvCacheMode mode
) {
    if (context_tokens == 0 || model.block_count == 0) return 0;
    const auto family = resolve_family_spec(model);
    std::uint64_t values_per_token{};
    if (!family.layer_specs.empty()) {
        std::set<std::size_t> counted_sources;
        for (const auto& layer : family.layer_specs) {
            if (counted_sources.insert(layer.kv_source_layer).second) {
                values_per_token = checked_add(
                    values_per_token,
                    static_cast<std::uint64_t>(layer.kv_hidden) * 2ULL,
                    "KV cache"
                );
            }
        }
    } else {
        // attention_key_length/attention_value_length are PER-HEAD dimensions
        // in GGUF metadata. The runtime caches full kv_hidden rows per token
        // (kv_heads * head dim for K and for V), so per-head values must be
        // scaled by kv_heads before they are comparable to the fallback.
        const auto key_head = model.attention_key_length != 0
            ? model.attention_key_length
            : static_cast<std::uint64_t>(family.head_dim);
        const auto value_head = model.attention_value_length != 0
            ? model.attention_value_length
            : key_head;
        values_per_token = checked_add(key_head, value_head, "KV cache")
            * static_cast<std::uint64_t>(family.kv_heads) * model.block_count;
    }
    // The selected row layout is the sizing contract: scale/header overhead
    // and live cache allocation must remain identical.
    if (values_per_token == 0) return 0;
    return kv_cache_storage_bytes(mode, values_per_token, context_tokens);
}

VideoMemoryInfo query_video_memory_info() {
    VideoMemoryInfo result;
#ifdef _WIN32
    IDXGIFactory1* factory_raw{};
    if (FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory1), reinterpret_cast<void**>(&factory_raw)))) {
        result.source = "dxgi-factory-unavailable";
        return result;
    }
    ComOwner<IDXGIFactory1> factory(factory_raw);
    LUID requested{};
    const bool has_requested = parse_requested_luid(requested);
    ComOwner<IDXGIAdapter3> best_adapter;
    DXGI_ADAPTER_DESC1 best_description{};
    std::uint64_t best_dedicated_bytes{};
    for (UINT index = 0;; ++index) {
        IDXGIAdapter1* adapter_raw{};
        const auto enumerated = factory->EnumAdapters1(index, &adapter_raw);
        if (enumerated == DXGI_ERROR_NOT_FOUND) break;
        if (FAILED(enumerated)) continue;
        ComOwner<IDXGIAdapter1> adapter(adapter_raw);
        DXGI_ADAPTER_DESC1 description{};
        if (FAILED(adapter->GetDesc1(&description))
            || (description.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) != 0
            || (has_requested && !luid_equal(description.AdapterLuid, requested))) {
            continue;
        }
        // Fit must describe the same class of adapter that the execution
        // backend can actually select. Caterpillar's current mapped L0 heaps
        // require a D3D12 FL 11_0 cache-coherent UMA device.
        ID3D12Device* probe_device_raw{};
        if (FAILED(D3D12CreateDevice(
                adapter.get(), D3D_FEATURE_LEVEL_11_0,
                __uuidof(ID3D12Device), reinterpret_cast<void**>(&probe_device_raw)))) {
            if (has_requested) break;
            continue;
        }
        ComOwner<ID3D12Device> probe_device(probe_device_raw);
        D3D12_FEATURE_DATA_ARCHITECTURE architecture{};
        if (FAILED(probe_device->CheckFeatureSupport(
                D3D12_FEATURE_ARCHITECTURE, &architecture, sizeof(architecture)))
            || architecture.UMA == 0
            || architecture.CacheCoherentUMA == 0) {
            if (has_requested) break;
            continue;
        }
        IDXGIAdapter3* adapter3_raw{};
        if (FAILED(adapter->QueryInterface(__uuidof(IDXGIAdapter3), reinterpret_cast<void**>(&adapter3_raw)))) {
            if (has_requested) break;
            continue;
        }
        ComOwner<IDXGIAdapter3> adapter3(adapter3_raw);
        if (!has_requested && best_adapter
            && description.DedicatedVideoMemory <= best_dedicated_bytes) {
            continue;
        }
        best_dedicated_bytes = description.DedicatedVideoMemory;
        best_description = description;
        best_adapter = std::move(adapter3);
        if (has_requested) break;
    }
    if (best_adapter) {
        DXGI_QUERY_VIDEO_MEMORY_INFO memory{};
        if (FAILED(best_adapter->QueryVideoMemoryInfo(
                0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &memory))) {
            result.source = has_requested
                ? "dxgi-requested-adapter-unavailable"
                : "dxgi-adapter-unavailable";
            return result;
        }
        result.live = true;
        result.source = "dxgi-local";
        result.adapter_name = narrow_adapter_name(best_description.Description);
        result.budget_bytes = memory.Budget;
        result.current_usage_bytes = memory.CurrentUsage;
        result.available_for_reservation_bytes = memory.AvailableForReservation;
        result.current_reservation_bytes = memory.CurrentReservation;
        return result;
    }
    result.source = has_requested ? "dxgi-requested-adapter-unavailable" : "dxgi-adapter-unavailable";
#else
    result.source = "non-windows";
#endif
    return result;
}

FitPlan plan_model_fit(const GgufModelInfo& model, const FitOptions& options) {
    FitPlan result;
    result.total_layer_count = static_cast<std::size_t>(model.block_count);
    result.layer_staged_bytes.resize(result.total_layer_count, 0);
    for (const auto& tensor : model.tensors) {
        const auto bytes = staged_tensor_bytes(tensor);
        result.total_staged_weight_bytes = checked_add(result.total_staged_weight_bytes, bytes, "model staging");
        const auto layer = tensor_layer(tensor.name);
        if (layer && *layer < result.layer_staged_bytes.size()) {
            result.layer_staged_bytes[*layer] = checked_add(result.layer_staged_bytes[*layer], bytes, "layer staging");
        } else {
            result.global_weight_bytes = checked_add(result.global_weight_bytes, bytes, "global staging");
        }
    }
    result.activation_bytes = static_cast<std::uint64_t>(options.activation_bytes);
    result.context_tokens = static_cast<std::uint64_t>(options.context_tokens);
    result.kv_cache_mode = options.kv_cache_mode;
    result.kv_cache_bytes = estimate_kv_cache_bytes(model, options.context_tokens, options.kv_cache_mode);
    result.working_set_bytes = checked_add(result.activation_bytes, result.kv_cache_bytes, "working set");

    // Host-RAM envelope (Phase 2, 2026-07-21 campaign). The whole host cost of
    // running this model: staged weights + KV + activations, judged against
    // the RAM that is actually free. Unprobed hosts skip classification with
    // an explicit reason instead of misclassifying.
    result.host_total_ram_bytes = options.host_total_ram_bytes;
    result.host_available_ram_bytes = options.host_available_ram_bytes;
    result.host_working_set_bytes = checked_add(
        result.total_staged_weight_bytes, result.working_set_bytes, "host working set");
    if (options.host_total_ram_bytes != 0 && options.host_available_ram_bytes != 0) {
        const auto ws = result.host_working_set_bytes;
        const auto available = options.host_available_ram_bytes;
        const auto total = options.host_total_ram_bytes;
        if (ws * 10 <= available * 6 && ws * 2 <= total) {
            result.host_envelope = HostEnvelope::comfortable;
            result.host_envelope_reason =
                "working set fits available RAM with headroom (<=60% available, <=50% total)";
        } else if (ws * 10 <= available * 9) {
            result.host_envelope = HostEnvelope::tight;
            result.host_envelope_reason =
                "working set fits available RAM without headroom (<=90% available); "
                "memory-for-speed features (speculation, GPU co-residency) stay off";
        } else {
            result.host_envelope = HostEnvelope::oversized;
            result.host_envelope_reason =
                "working set exceeds 90% of available host RAM; resident GPU plans are "
                "refused and the session degrades to the documented CPU/stream path";
        }
    }

    const auto memory = options.video_memory.value_or(query_video_memory_info());
    result.live_budget = memory.live;
    result.budget_source = memory.source;
    result.adapter_name = memory.adapter_name;
    result.current_usage_bytes = memory.current_usage_bytes;
    result.budget_bytes = memory.budget_bytes;
    const auto requested = options.requested_budget_bytes.has_value()
        ? options.requested_budget_bytes
        : budget_from_environment();
    if (requested && *requested > 0) {
        if (result.budget_bytes == 0 || *requested < result.budget_bytes) result.budget_bytes = *requested;
        if (!result.live_budget) result.budget_source = "configured-ceiling";
    }
    result.reserve_bytes = options.reserve_bytes.value_or(default_reserve(result.budget_bytes));
    if (result.budget_bytes > result.current_usage_bytes) {
        result.available_bytes = result.budget_bytes - result.current_usage_bytes;
    }

    if (!options.gpu_backend_available) {
        result.reason = "Caterpillar Phase 2 has no executable D3D12 graph spans; the session is pinned CPU-only.";
        return result;
    }
    if (result.host_envelope == HostEnvelope::oversized) {
        // On UMA hosts GPU staging IS host RAM: keeping an oversized model
        // resident double-commits memory and thrashes (the known lizard-era
        // failure class). Refuse the resident plan and say why, so the
        // dashboard shows the reason instead of a silently slow session.
        result.reason = "Host RAM envelope is oversized; resident GPU staging is refused: "
            + result.host_envelope_reason;
        return result;
    }
    if (result.budget_bytes == 0 || result.current_usage_bytes >= result.budget_bytes) {
        result.reason = "No usable live or configured GPU memory budget is available; the session is pinned CPU-only.";
        return result;
    }
    const auto fixed = checked_add(result.working_set_bytes, result.reserve_bytes, "fit fixed cost");
    if (fixed >= result.available_bytes || result.global_weight_bytes > result.available_bytes - fixed) {
        result.reason = "GPU budget cannot hold the fixed working set, reserves, and global weights; the session is pinned CPU-only.";
        return result;
    }
    const auto weight_capacity = result.available_bytes - fixed;
    if (result.total_staged_weight_bytes <= weight_capacity) {
        result.decision = FitDecision::all_resident;
        result.resident_layer_count = result.total_layer_count;
        result.resident_weight_bytes = result.total_staged_weight_bytes;
        result.reason = "All staged weights and the complete working set fit with the required reserve.";
        return result;
    }

    // A stable GPU-prefix/CPU-tail plan keeps the packed device cache plus a
    // decoded f32 mirror for CPU attention. Account for that mirror before
    // choosing the resident prefix so fit telemetry equals live allocation.
    result.kv_cpu_mirror_bytes = estimate_kv_cache_bytes(model, options.context_tokens, KvCacheMode::f32);
    result.kv_cache_bytes = checked_add(result.kv_cache_bytes, result.kv_cpu_mirror_bytes, "stable-split KV cache");
    result.working_set_bytes = checked_add(result.activation_bytes, result.kv_cache_bytes, "stable-split working set");
    const auto stable_fixed = checked_add(result.working_set_bytes, result.reserve_bytes, "stable-split fixed cost");
    if (stable_fixed >= result.available_bytes || result.global_weight_bytes > result.available_bytes - stable_fixed) {
        result.reason = "GPU budget cannot hold the packed KV cache, CPU-tail mirror, reserves, and global weights; the session is pinned CPU-only.";
        return result;
    }
    const auto stable_weight_capacity = result.available_bytes - stable_fixed;

    std::uint64_t resident = result.global_weight_bytes;
    std::size_t layers{};
    for (const auto layer_bytes : result.layer_staged_bytes) {
        if (layer_bytes > stable_weight_capacity - resident) break;
        resident += layer_bytes;
        ++layers;
    }
    if (layers == 0) {
        result.reason = "GPU budget cannot hold one complete layer in addition to global weights and working-set headroom; the session is pinned CPU-only.";
        return result;
    }
    result.decision = FitDecision::stable_split;
    result.resident_layer_count = layers;
    result.resident_weight_bytes = resident;
    result.reason = "A stable prefix of complete layers fits; all remaining layers are pinned to CPU for the session.";
    return result;
}

int run_fit_selftest() {
    try {
        GgufModelInfo model;
        model.family = "llama";
        model.architecture = "llama";
        model.block_count = 4;
        model.embedding_length = 32;
        model.feed_forward_length = 64;
        model.head_count = 4;
        model.head_count_kv = 2;
        model.context_length = 128;
        model.tensors.push_back(GgufTensorInfo{"token_embd.weight", {32, 32}, 1, 0});
        for (std::size_t layer = 0; layer < 4; ++layer) {
            model.tensors.push_back(GgufTensorInfo{"blk." + std::to_string(layer) + ".attn_q.weight", {32, 32}, 2, 0});
        }
        if (staged_tensor_bytes(model.tensors[1]) != 576) {
            throw std::runtime_error("Q4_0 fp16-scale staging must remain 18 bytes per 32 weights");
        }
        // The CPU session cache stores K/V as float; the planner must reserve
        // the same bytes. 4 layers * (16 K + 16 V) values/token * 4 bytes *
        // 16 tokens = 8192. A 2-byte fp16 estimate here would silently halve
        // the reserved working set (the Phase 2 handoff defect).
        if (estimate_kv_cache_bytes(model, 16) != 8192) {
            throw std::runtime_error("KV estimate must reserve 4-byte float values matching CpuSessionState::bytes");
        }
        if (estimate_kv_cache_bytes(model, 16, KvCacheMode::q8_0) != 2304) {
            throw std::runtime_error("Q8_0 KV estimate must include one f32 scale per 32 values");
        }
        // GGUF attention.key_length/value_length are per-head: 8 per head *
        // 2 kv_heads * (K + V) * 4 layers = 128 values/token, the same width
        // the fallback derives. Treating them as full-layer widths (the
        // Phase 2 llama-3.2 defect) would report 4096 here instead of 8192.
        model.attention_key_length = 8;
        model.attention_value_length = 8;
        if (estimate_kv_cache_bytes(model, 16) != 8192) {
            throw std::runtime_error("per-head key/value lengths must scale by kv_heads in the KV estimate");
        }
        const VideoMemoryInfo large{true, "selftest", "synthetic", 16 * mib, 0, 0, 0};
        auto all = plan_model_fit(model, FitOptions{
            .activation_bytes = 4096,
            .context_tokens = 16,
            .gpu_backend_available = true,
            .video_memory = large,
            .requested_budget_bytes = std::nullopt,
            .reserve_bytes = 4096,
            .kv_cache_mode = KvCacheMode::f32,
        });
        if (all.decision != FitDecision::all_resident || all.resident_layer_count != 4
            || all.gpu_weight_cache_evictions != 0 || !all.stable
            || all.kv_cache_bytes != 8192) {
            throw std::runtime_error("all-resident fit decision mismatch");
        }
        auto split_base = plan_model_fit(model, FitOptions{
            .activation_bytes = 4096,
            .context_tokens = 1,
            .gpu_backend_available = true,
            .video_memory = large,
            .requested_budget_bytes = std::nullopt,
            .reserve_bytes = 4096,
            .kv_cache_mode = KvCacheMode::f32,
        });
        const auto split_budget = split_base.current_usage_bytes + split_base.working_set_bytes
            + estimate_kv_cache_bytes(model, 1, KvCacheMode::f32) + split_base.reserve_bytes
            + split_base.global_weight_bytes + split_base.layer_staged_bytes[0] + split_base.layer_staged_bytes[1];
        auto split_memory = large;
        split_memory.budget_bytes = split_budget;
        auto split = plan_model_fit(model, FitOptions{
            .activation_bytes = 4096,
            .context_tokens = 1,
            .gpu_backend_available = true,
            .video_memory = split_memory,
            .requested_budget_bytes = std::nullopt,
            .reserve_bytes = 4096,
            .kv_cache_mode = KvCacheMode::f32,
        });
        if (split.decision != FitDecision::stable_split || split.resident_layer_count != 2
            || split.resident_weight_bytes > split.available_bytes - split.working_set_bytes - split.reserve_bytes) {
            throw std::runtime_error("stable-split fit decision mismatch");
        }
        auto cpu = plan_model_fit(model, FitOptions{
            .activation_bytes = 4096,
            .context_tokens = 16,
            .gpu_backend_available = false,
            .video_memory = large,
            .requested_budget_bytes = std::nullopt,
            .reserve_bytes = 4096,
            .kv_cache_mode = KvCacheMode::f32,
        });
        if (cpu.decision != FitDecision::cpu_only || cpu.resident_layer_count != 0
            || cpu.resident_weight_bytes != 0 || cpu.gpu_weight_cache_evictions != 0) {
            throw std::runtime_error("CPU-only fit decision mismatch");
        }
        // Host-RAM envelope (Phase 2, 2026-07-21 campaign). The synthetic
        // model's host working set is fixed: 4352 staged weight bytes + 8192
        // KV + 4096 activations = 16640. Boundaries are exercised exactly.
        if (cpu.host_envelope != HostEnvelope::comfortable
            || cpu.host_envelope_reason != "host-memory-unprobed") {
            throw std::runtime_error("unprobed host RAM must skip envelope classification");
        }
        const auto envelope_of = [&](std::uint64_t total, std::uint64_t available,
                                     bool gpu_available = true) {
            return plan_model_fit(model, FitOptions{
                .activation_bytes = 4096,
                .context_tokens = 16,
                .gpu_backend_available = gpu_available,
                .video_memory = large,
                .requested_budget_bytes = std::nullopt,
                .reserve_bytes = 4096,
                .kv_cache_mode = KvCacheMode::f32,
                .host_total_ram_bytes = total,
                .host_available_ram_bytes = available,
            });
        };
        const std::uint64_t ws = 16640;
        if (envelope_of(2 * ws, 33000).host_working_set_bytes != ws) {
            throw std::runtime_error("host working set must be staged weights + KV + activations");
        }
        if (envelope_of(2 * ws, 33000).host_envelope != HostEnvelope::comfortable) {
            throw std::runtime_error("comfortable classification failed at the exact total boundary");
        }
        if (envelope_of(2 * ws - 1, 33000).host_envelope != HostEnvelope::tight) {
            throw std::runtime_error("one byte over 50% of total RAM must demote to tight");
        }
        if (envelope_of(1ULL << 36, 18489).host_envelope != HostEnvelope::tight) {
            throw std::runtime_error("tight classification failed at the exact 90%-available boundary");
        }
        const auto oversized = envelope_of(1ULL << 36, 18488);
        if (oversized.host_envelope != HostEnvelope::oversized) {
            throw std::runtime_error("one byte past 90% of available RAM must classify oversized");
        }
        if (oversized.decision != FitDecision::cpu_only || oversized.host_envelope_reason.empty()
            || oversized.reason.find("oversized") == std::string::npos) {
            throw std::runtime_error("oversized envelope must refuse resident GPU plans with a reason");
        }
        std::cout << "caterpillar fit selftest: ok (all-resident + stable-split + cpu-only + q4_0-18b + host-envelope)\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "caterpillar fit selftest: failed: " << error.what() << '\n';
        return 1;
    }
}

}  // namespace cat
