#pragma once

// Caterpillar Phase 4 GPU decode model: one capacity-shaped decode graph
// (single token, past-KV inputs sized to the session capacity, logits matmul
// in-graph) compiled ONCE into a backend-partitioned plan whose GPU spans
// are recorded command lists. Decode = replay. Prefill runs on the Phase 2
// CPU engine and seeds the persistent packed KV cache once.

#include "backend_d3d12.h"
#include "cpu_model.h"
#include "family_spec.h"
#include "fit.h"
#include "plan.h"
#include "sched.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace cat {

struct GpuSessionState {
    std::size_t cached_tokens{};
    std::uint32_t next_token{};
    bool next_token_valid{false};
    std::optional<CpuSessionState> verifier_state;

    [[nodiscard]] std::size_t bytes(std::size_t per_token_bytes) const noexcept {
        return cached_tokens * per_token_bytes;
    }
};

class GpuModel {
public:
    // Returns nullptr (with a reason) when the model cannot run its decode
    // graph on the GPU: no capable device, unsupported weight format, or a
    // fit decision without resident layers. The caller then stays on the
    // Phase 2 CPU path — never a silent substitution.
    static std::unique_ptr<GpuModel> try_create(
        const CpuModel& cpu,
        const FitPlan& fit,
        std::size_t kv_capacity_tokens,
        std::string* reason
    );

    [[nodiscard]] GpuSessionState prefill(
        const std::vector<std::uint32_t>& tokens,
        bool retain_verifier_state = false
    );
    [[nodiscard]] std::uint32_t advance(GpuSessionState& session, std::uint32_t token);
    [[nodiscard]] CpuDraftVerification verify_draft(
        const GpuSessionState& session,
        const std::vector<std::uint32_t>& input_tokens
    ) const;
    void commit_draft(
        GpuSessionState& session,
        const CpuDraftVerification& verification,
        std::size_t input_count
    );

    [[nodiscard]] const GpuTelemetry& telemetry() const noexcept { return gpu_->telemetry(); }
    [[nodiscard]] std::size_t kv_capacity() const noexcept { return capacity_; }
    [[nodiscard]] std::size_t per_token_kv_bytes() const noexcept { return per_token_kv_bytes_; }
    [[nodiscard]] std::size_t plan_gpu_spans() const noexcept { return plan_.gpu_span_count(); }
    [[nodiscard]] std::size_t plan_cpu_spans() const noexcept { return plan_.spans().size() - plan_.gpu_span_count(); }
    [[nodiscard]] std::size_t scheduled_gpu_nodes() const noexcept { return schedule_.gpu_node_count; }
    [[nodiscard]] std::size_t scheduled_cpu_nodes() const noexcept { return schedule_.cpu_node_count; }
    [[nodiscard]] std::string adapter_name() const { return gpu_->adapter_name(); }

private:
    GpuModel(const CpuModel& cpu, const FitPlan& fit, std::size_t kv_capacity_tokens);

    const CpuModel& cpu_;
    FamilySpec family_;
    std::size_t capacity_{};
    std::size_t per_token_kv_bytes_{};
    DecoderGraph decoder_;
    TensorId logits_id_{};
    TensorId logits_weight_id_{};
    SchedulerResult schedule_;
    Plan plan_;
    QuantizedBindings quantized_;
    Bindings immutable_;
    std::vector<TensorId> dynamic_ids_;
    std::unique_ptr<GpuPlanExecution> gpu_;
};

}  // namespace cat
