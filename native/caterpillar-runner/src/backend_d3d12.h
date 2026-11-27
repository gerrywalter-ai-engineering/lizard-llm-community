#pragma once

// Caterpillar Phase 4 D3D12 backend: executes the GPU spans of a compiled
// graph plan as command lists recorded ONCE at prepare time and replayed for
// every decoded token. Requires probe-confirmed cache-coherent UMA (a
// vendor-neutral D3D12 capability, currently proven on Iris Xe); on any
// other hardware the backend reports unavailable and the session stays on
// the Phase 2 CPU path. Clean-room implementation informed by the
// first-party lizard-native D3D12 lane (proven Iris Xe access patterns:
// clamp-before-load, no raw 16-bit loads, tanh clamps, batch-one wide-row
// matvec groups, tree-reduction norms, online-softmax attention).

#include "family_spec.h"
#include "gguf_loader.h"
#include "kv_cache.h"
#include "plan.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace cat {

struct GpuUnavailable : std::runtime_error {
    using std::runtime_error::runtime_error;
};

struct GpuTelemetry {
    std::uint64_t submissions{};
    std::uint64_t span_replays{};
    std::uint64_t dynamic_upload_bytes{};
    std::uint64_t readback_bytes{};
    std::uint64_t readback_count{};
    std::uint64_t resident_weight_bytes{};
    std::size_t gpu_spans{};
    std::size_t cpu_spans{};
    bool integer_dot_available{};
    bool integer_dot_active{};
    std::string integer_dot_reason;
    std::uint64_t q8_1_quantizations{};
    std::uint64_t integer_dot_matvecs{};
    std::uint64_t kv_device_appends{};
    std::uint64_t kv_resident_bytes{};
    KvCacheMode kv_cache_mode{KvCacheMode::q8_0};
    bool kv_gpu_resident{};
    // Wall-clock split of replayed GPU work by node class, measured with
    // in-list timestamp query pairs around every recorded dispatch (Step 07
    // tuning instrumentation). Classes: 0 matmul-large (rows >= 8192),
    // 1 attention, 2 rms, 3 rope, 4 elementwise (add/scale/silu/gelu),
    // 5 other, 6 matmul-mid (2048 <= rows < 8192), 7 matmul-small (< 2048).
    static constexpr std::size_t node_class_count = 8;
    std::array<double, node_class_count> node_class_ms{};
    std::array<std::uint64_t, node_class_count> node_class_dispatches{};
    bool node_timing_available{};
};

// One prepared, replayable GPU execution of a compiled plan. Owns the
// device, the recorded command lists, the resident weights, the mapped
// state buffer, and the persistent packed KV cache. Implements GpuExecutor so
// Plan::replay_views can hand it the GPU spans.
class GpuPlanExecution final : public GpuExecutor {
public:
    struct Config {
        // Capacity (tokens) of the persistent KV cache; attention reads
        // the live count from the per-token session constants.
        std::size_t kv_capacity_tokens{};
        // Optional DXGI adapter LUID filter "high:low" (matches the fit
        // planner's LIZARD_NATIVE_GPU_ADAPTER_LUID semantics).
        bool require_cache_coherent_uma{true};
        // Phase 5 is opt-in until the model-level golden and performance
        // gates pass. Unsupported builds/devices transparently use float.
        bool enable_integer_dot{false};
        // Correctness tests exercise every Phase 5 format. Production keeps
        // formats that have not independently beaten float on the fallback.
        bool enable_unqualified_integer_dot_formats{false};
        KvCacheMode kv_cache_mode{KvCacheMode::q8_0};
        // A stable GPU-prefix/CPU-tail plan still needs decoded f32 rows for
        // the reference CPU attention. Fully resident plans leave this off.
        bool cpu_kv_mirror{false};
    };

    // Throws GpuUnavailable when no suitable adapter/device exists.
    GpuPlanExecution(
        const Plan& plan,
        const QuantizedBindings& quantized,
        const Bindings& immutable_externals,
        const std::vector<TensorId>& dynamic_externals,
        const std::vector<DecoderGraph::KvBinding>& kv_inputs,
        const Config& config
    );
    ~GpuPlanExecution() override;

    GpuPlanExecution(const GpuPlanExecution&) = delete;
    GpuPlanExecution& operator=(const GpuPlanExecution&) = delete;

    void execute_span(
        const Plan& plan,
        const PlanSpan& span,
        const BindingViews& external,
        const ReplayDynamics& dynamics,
        std::span<float> arena
    ) override;

    // Seeds the device-resident cache from CPU prefill output. The one-time
    // prefill transfer is allowed; incremental rows append on the GPU.
    void seed_kv(
        std::size_t owning_layer,
        std::span<const float> keys,
        std::span<const float> values,
        std::size_t token_count
    );
    void write_kv_rows(
        std::size_t owning_layer,
        std::size_t first_token,
        std::span<const float> keys,
        std::span<const float> values
    );
    void read_kv_rows(
        std::size_t owning_layer,
        std::size_t first_token,
        std::size_t token_count,
        std::span<float> keys,
        std::span<float> values
    );

    // Stable-split-only f32 mirror access for CPU tail attention.
    [[nodiscard]] std::span<float> kv_keys(std::size_t owning_layer);
    [[nodiscard]] std::span<float> kv_values(std::size_t owning_layer);
    [[nodiscard]] bool has_cpu_kv_mirror() const noexcept;

    [[nodiscard]] const GpuTelemetry& telemetry() const noexcept { return telemetry_; }
    [[nodiscard]] std::string adapter_name() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    GpuTelemetry telemetry_;
};

// True when a Phase 4 capable device exists (probe-confirmed cache-coherent
// UMA compute device on the fit planner's adapter). Cheap enough to call at
// load.
[[nodiscard]] bool gpu_backend_available(std::string* reason = nullptr);

int run_gpu_selftest();

}  // namespace cat
