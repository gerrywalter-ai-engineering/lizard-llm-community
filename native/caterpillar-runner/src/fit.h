#pragma once

#include "gguf_loader.h"
#include "kv_cache.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace cat {

// Clean-room VRAM-fit planner. The API shape follows the documented
// Caterpillar fit discipline; no upstream inference-engine code is used.
struct VideoMemoryInfo {
    bool live{false};
    std::string source{"unavailable"};
    std::string adapter_name;
    std::uint64_t budget_bytes{};
    std::uint64_t current_usage_bytes{};
    std::uint64_t available_for_reservation_bytes{};
    std::uint64_t current_reservation_bytes{};
};

enum class FitDecision {
    all_resident,
    stable_split,
    cpu_only,
};

// Host-RAM operating envelope (2026-07-21 campaign, Phase 2). Caterpillar's
// tuned route is claimed only for models that comfortably fit in the RAM that
// is actually free; outside that envelope the plan degrades honestly instead
// of thrashing. Keyed on AVAILABLE RAM, not installed: only ~4.5 GB was free
// on the 16 GB reference host during iteration 012, and that is the number
// that decides whether the GGUF stays in page cache.
enum class HostEnvelope {
    comfortable,  // working set <= 60% of available AND <= 50% of total RAM
    tight,        // working set <= 90% of available RAM
    oversized,    // anything larger - tuned/resident paths must refuse
};

struct HostMemoryInfo {
    bool live{false};
    std::uint64_t total_bytes{};
    std::uint64_t available_bytes{};
};

struct FitOptions {
    std::size_t activation_bytes{};
    std::size_t context_tokens{};
    bool gpu_backend_available{false};
    std::optional<VideoMemoryInfo> video_memory;
    std::optional<std::uint64_t> requested_budget_bytes;
    std::optional<std::uint64_t> reserve_bytes;
    KvCacheMode kv_cache_mode{KvCacheMode::q8_0};
    // Zero means "not probed": classification is skipped (comfortable with an
    // explicit unprobed reason) rather than misclassifying as oversized.
    std::uint64_t host_total_ram_bytes{};
    std::uint64_t host_available_ram_bytes{};
};

struct FitPlan {
    FitDecision decision{FitDecision::cpu_only};
    std::string reason;
    bool stable{true};
    bool live_budget{false};
    std::string budget_source{"unavailable"};
    std::string adapter_name;
    std::size_t total_layer_count{};
    std::size_t resident_layer_count{};
    std::uint64_t budget_bytes{};
    std::uint64_t current_usage_bytes{};
    std::uint64_t available_bytes{};
    std::uint64_t activation_bytes{};
    std::uint64_t context_tokens{};
    std::uint64_t kv_cache_bytes{};
    std::uint64_t kv_cpu_mirror_bytes{};
    KvCacheMode kv_cache_mode{KvCacheMode::q8_0};
    std::uint64_t working_set_bytes{};
    std::uint64_t reserve_bytes{};
    std::uint64_t global_weight_bytes{};
    std::uint64_t total_staged_weight_bytes{};
    std::uint64_t resident_weight_bytes{};
    std::uint64_t gpu_weight_cache_evictions{};
    bool q4_0_fp16_scale_staging{true};
    std::vector<std::uint64_t> layer_staged_bytes;
    HostEnvelope host_envelope{HostEnvelope::comfortable};
    std::string host_envelope_reason{"host-memory-unprobed"};
    std::uint64_t host_total_ram_bytes{};
    std::uint64_t host_available_ram_bytes{};
    std::uint64_t host_working_set_bytes{};
};

[[nodiscard]] std::string fit_decision_name(FitDecision decision);
[[nodiscard]] std::string host_envelope_name(HostEnvelope envelope);
[[nodiscard]] HostMemoryInfo query_host_memory_info();
[[nodiscard]] std::uint64_t staged_tensor_bytes(const GgufTensorInfo& tensor);
[[nodiscard]] std::uint64_t estimate_kv_cache_bytes(
    const GgufModelInfo& model,
    std::size_t context_tokens,
    KvCacheMode mode = KvCacheMode::f32
);
[[nodiscard]] VideoMemoryInfo query_video_memory_info();
[[nodiscard]] FitPlan plan_model_fit(const GgufModelInfo& model, const FitOptions& options);
int run_fit_selftest();

}  // namespace cat
