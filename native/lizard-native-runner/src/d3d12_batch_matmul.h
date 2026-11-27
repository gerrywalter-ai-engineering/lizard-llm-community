#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace lizard_native {

struct D3d12BatchMatmulCapabilities {
  bool supported = false;
  std::string backend_name = "cpu";
  std::string adapter_name;
  std::string reason;
  // Selected adapter identity (Phase 1 Finding 1): the stable DXGI LUID plus
  // vendor/device ids and native architecture facts, so the planner and the
  // worker can be proven to address the same physical GPU in telemetry.
  bool adapter_identity_valid = false;
  std::int64_t adapter_luid_high = 0;
  std::uint64_t adapter_luid_low = 0;
  std::uint32_t adapter_vendor_id = 0;
  std::uint32_t adapter_device_id = 0;
  bool adapter_uma = false;
  bool adapter_cache_coherent_uma = false;
  std::string adapter_selection_reason;
};

struct D3d12RuntimeTelemetry {
  std::uint64_t submission_count = 0;
  std::uint64_t fence_wait_count = 0;
  std::uint64_t token_graph_submission_count = 0;
  std::uint64_t weight_cache_hits = 0;
  std::uint64_t weight_cache_misses = 0;
  std::uint64_t weight_cache_evictions = 0;
  std::uint64_t weight_upload_bytes = 0;
  std::uint64_t input_upload_bytes = 0;
  std::uint64_t output_readback_bytes = 0;
};

const D3d12BatchMatmulCapabilities &d3d12_batch_matmul_capabilities();
// Like d3d12_batch_matmul_capabilities() but NEVER initializes the D3D12
// context: returns null while the GPU lane is untouched. Telemetry that only
// wants to report an already-selected adapter must use this - initializing
// the whole GPU stack (device + every PSO compile) for a CPU-only model
// load is expensive and wrong.
const D3d12BatchMatmulCapabilities *d3d12_batch_matmul_capabilities_if_initialized();

// Live GPU memory accounting (Phase 1 Finding 2). `live` is true when the
// numbers come from IDXGIAdapter3::QueryVideoMemoryInfo on the selected
// adapter; false means the static env/default budget is in effect and
// confidence is degraded.
struct D3d12GpuMemoryReport {
  bool live = false;
  std::uint64_t local_budget_bytes = 0;
  std::uint64_t local_usage_bytes = 0;
  std::uint64_t non_local_budget_bytes = 0;
  std::uint64_t non_local_usage_bytes = 0;
  std::uint64_t reserve_bytes = 0;
  std::uint64_t requested_weight_budget_bytes = 0;
  std::uint64_t effective_weight_budget_bytes = 0;
  std::uint64_t resident_weight_bytes = 0;
  std::string confidence = "static-fallback";
};

// Non-initializing: report.live stays false (and budgets zero) until the
// D3D12 context has been created by real GPU work.
D3d12GpuMemoryReport d3d12_gpu_memory_report();

// Force a live budget refresh (e.g. before model prestaging). No-op when
// the context is not initialized or DXGI memory queries are unavailable.
void d3d12_refresh_gpu_memory_budget();
const D3d12RuntimeTelemetry &d3d12_runtime_telemetry();
void d3d12_reset_runtime_telemetry();

// Shader Model 6.4 packed-integer-dot (DP4A) kernel availability. Requires a
// build with embedded DXIL kernels (LIZARD_HAVE_DXIL) plus driver support for
// SM 6.4; otherwise `supported` is false and `reason` says why. The float
// FXC kernels remain the execution path in that case.
struct D3d12Dp4aCapabilities {
  bool supported = false;
  std::string reason;
};

const D3d12Dp4aCapabilities &d3d12_dp4a_capabilities();

// Weight formats the GPU lane can consume directly from raw GGUF tensor bytes.
// Quantized formats are dequantized inside the compute shader, so weights are
// uploaded once in compressed form and stay resident on the device.
enum class D3d12WeightFormat : std::uint32_t {
  F32 = 0,
  F16 = 1,
  BF16 = 2,
  Q4_K = 3,
  Q6_K = 4,
  Q5_K = 5,
  Q8_0 = 6,
  Q4_0 = 7,
};

// Returns true when the GPU lane can execute this GGML tensor type directly
// from its raw bytes for the given row width.
bool d3d12_weight_format_supported(
  std::uint32_t ggml_type,
  std::uint64_t row_width,
  D3d12WeightFormat *format);

struct D3d12ResidentWeightRef {
  // Stable identity for the tensor payload; the mapped GGUF data pointer.
  const void *weight_key = nullptr;
  D3d12WeightFormat format = D3d12WeightFormat::F32;
  const std::uint8_t *weight_data = nullptr;
  std::size_t weight_bytes = 0;
  std::uint32_t row_count = 0;
  std::uint32_t row_width = 0;
};

// Quantizes `input` to Q8_1 on the GPU (blocks of 32, packed int8 plus f16
// d/s scales) and multiplies the resident Q4_K weights against it with
// dot4add_i8packed in one submission. Fails when DP4A is unavailable (see
// d3d12_dp4a_capabilities) or `op` is not a Q4_K matvec shape.
bool d3d12_dp4a_q4k_matvec(
  const D3d12ResidentWeightRef &op,
  const std::vector<float> &input,
  std::vector<float> *outputs,
  std::string *error_message = nullptr);

// Test-only: dispatches the cooperative subgroup-per-row Q4_K GEMV standalone
// so it can be verified against a CPU reference outside the token graph.
bool d3d12_coop_gemv_q4k_matvec(
  const D3d12ResidentWeightRef &op,
  const std::vector<float> &input,
  std::vector<float> *outputs,
  std::string *error_message = nullptr);

// Test-only: times a pure streaming read of `bytes` (no dequantisation, no
// arithmetic beyond an integer add) and returns milliseconds per iteration.
// This is the adapter's memory-bandwidth ceiling; compare it against the Q4_K
// matvec's achieved GB/s to tell a kernel that can be improved from one that is
// already at the hardware limit.
bool d3d12_batched_ab_benchmark(
  const D3d12ResidentWeightRef &op,
  const std::vector<float> &packed_inputs,
  std::uint32_t batch_count,
  int variant,
  int repeats,
  double *ms_per_iteration,
  std::string *error_message);

bool d3d12_fma_peak_benchmark(
  std::uint32_t iterations,
  std::uint32_t group_count,
  int repeats,
  double *ms_per_iteration,
  std::string *error_message);

bool d3d12_stream_read_benchmark(
  std::uint64_t bytes,
  int iterations,
  std::uint32_t group_count,
  // 0 = contiguous (the ceiling). Non-zero = each thread walks a "row" of this
  // many 16-byte quads, reproducing main_q4k's strided pattern so the two can
  // be compared with one kernel.
  std::uint32_t row_quads,
  double *ms_per_iteration,
  std::string *error_message = nullptr);

// Diagnostic only: same dispatch as the wide128 matvec but running the
// load-only twin kernel (identical loads, no dequantisation). Its output is
// meaningless; the timing separates memory cost from dequant ALU cost.
bool d3d12_loadonly_q4k_matvec(
  const D3d12ResidentWeightRef &op,
  const std::vector<float> &input,
  std::vector<float> *outputs,
  std::string *error_message = nullptr,
  // true addresses the same buffer with 16-byte quad interleaving instead of
  // 144-byte block interleaving, to price that layout change before making it.
  bool coalesced = false);

// Test-only: dispatches main_q4k compiled with 128 rows/group (instead of
// the default 64) standalone, so the wider-group hypothesis can be verified
// against a CPU reference and timed before considering it for real decode.
bool d3d12_wide128_q4k_matvec(
  const D3d12ResidentWeightRef &op,
  const std::vector<float> &input,
  std::vector<float> *outputs,
  std::string *error_message = nullptr);

// Test-only: dispatches main_q4k compiled with 256 rows/group, to check
// whether widening further than 128 keeps paying off.
bool d3d12_wide256_q4k_matvec(
  const D3d12ResidentWeightRef &op,
  const std::vector<float> &input,
  std::vector<float> *outputs,
  std::string *error_message = nullptr);

// Test-only: dispatches the DXC/SM6.0 WaveActiveSum cooperative Q4_K GEMV
// standalone so it can be verified against a CPU reference outside the graph.
bool d3d12_wave_gemv_q4k_matvec(
  const D3d12ResidentWeightRef &op,
  const std::vector<float> &input,
  std::vector<float> *outputs,
  std::string *error_message = nullptr);

struct D3d12GemmaAttentionConfig {
  std::uint64_t cache_id = 0;
  std::uint32_t position = 0;
  std::uint32_t max_positions = 0;
  std::uint32_t head_count = 0;
  std::uint32_t kv_head_count = 0;
  std::uint32_t head_dim = 0;
  std::uint32_t attention_window = 0;
  float rms_epsilon = 1e-5f;
  float rope_freq_base = 10000.0f;
  float rope_freq_scale = 1.0f;
};

// A K or V lane used to seed the device KV cache. Exactly one of f32/f16 is
// non-null. The device cache stores IEEE halves, so an f16 lane uploads with
// a straight copy while an f32 lane is converted during packing.
struct D3d12KvLaneView {
  const float *f32 = nullptr;
  const std::uint16_t *f16 = nullptr;
  std::size_t count = 0;
  // Distinguishes "descriptor never filled in" from a legitimately empty
  // lane (position 0 seeds an empty cache and both data pointers may be null).
  bool populated = false;

  std::size_t size() const {
    return count;
  }
  bool empty() const {
    return count == 0;
  }
};

struct D3d12GemmaTokenLayer {
  D3d12ResidentWeightRef query;
  D3d12ResidentWeightRef key;
  D3d12ResidentWeightRef value;
  D3d12ResidentWeightRef attn_out;
  D3d12ResidentWeightRef gate;
  D3d12ResidentWeightRef up;
  D3d12ResidentWeightRef down;
  D3d12ResidentWeightRef ple_gate;
  D3d12ResidentWeightRef ple_projection;
  const std::vector<float> *attention_norm = nullptr;
  const std::vector<float> *query_norm = nullptr;
  const std::vector<float> *key_norm = nullptr;
  const std::vector<float> *rope_freq_factors = nullptr;
  D3d12KvLaneView initial_keys;
  D3d12KvLaneView initial_values;
  const std::vector<float> *post_attention_norm = nullptr;
  const std::vector<float> *ffn_norm = nullptr;
  const std::vector<float> *post_ffw_norm = nullptr;
  const std::vector<float> *ple_input = nullptr;
  const std::vector<float> *ple_post_norm = nullptr;
  D3d12GemmaAttentionConfig attention;
  float output_scale = 1.0f;
  bool owns_kv_cache = true;
};

struct D3d12GemmaPleInputConfig {
  D3d12ResidentWeightRef model_projection;
  const std::vector<float> *token_inputs = nullptr;
  const std::vector<float> *projection_norm = nullptr;
  std::uint32_t layer_width = 0;
  float rms_epsilon = 1e-5f;
  float projection_scale = 1.0f;
};

// Records all transformer layers, final norm, logits, softcap, and argmax in
// one command list. Hidden state and every intermediate stay device-resident;
// the call performs exactly one queue submission and one fence wait after all
// required weights are resident.
bool d3d12_gemma_token_resident(
  const std::vector<float> &initial_hidden,
  const std::vector<D3d12GemmaTokenLayer> &layers,
  const D3d12GemmaPleInputConfig *ple_inputs,
  const std::vector<float> &output_norm,
  const D3d12ResidentWeightRef &output,
  float output_rms_epsilon,
  float final_logit_softcap,
  std::vector<float> *hidden_out,
  std::vector<float> *logits_out,
  std::uint32_t *argmax_token,
  std::string *error_message = nullptr);

// Executes Q/K/V projection, head normalization, NEOX RoPE, FP16 KV append,
// and online-softmax GQA attention in one command list. The KV cache stays on
// the device after the first call; initial_keys/initial_values seed it from
// prefill and are ignored after the cache is initialized. For shared-KV
// layers key/value may be null, in which case the existing cache entry is
// consumed without appending another position.
bool d3d12_gemma_attention_resident(
  const D3d12ResidentWeightRef &query,
  const D3d12ResidentWeightRef *key,
  const D3d12ResidentWeightRef *value,
  const std::vector<float> &normed_hidden,
  const std::vector<float> &query_norm,
  const std::vector<float> &key_norm,
  const std::vector<float> &rope_freq_factors,
  const D3d12KvLaneView &initial_keys,
  const D3d12KvLaneView &initial_values,
  const D3d12GemmaAttentionConfig &config,
  std::vector<float> *attention_context,
  std::string *error_message = nullptr);

// Executes the resident attention graph above and the complete Gemma
// attention/FFN/PLE layer tail in one command list and one fence wait.
bool d3d12_gemma_layer_resident(
  const D3d12ResidentWeightRef &query,
  const D3d12ResidentWeightRef *key,
  const D3d12ResidentWeightRef *value,
  const D3d12ResidentWeightRef &attn_out,
  const D3d12ResidentWeightRef &gate,
  const D3d12ResidentWeightRef &up,
  const D3d12ResidentWeightRef &down,
  const D3d12ResidentWeightRef *ple_gate,
  const D3d12ResidentWeightRef *ple_projection,
  const std::vector<float> &normed_hidden,
  const std::vector<float> &hidden_state,
  const std::vector<float> &query_norm,
  const std::vector<float> &key_norm,
  const std::vector<float> &rope_freq_factors,
  const D3d12KvLaneView &initial_keys,
  const D3d12KvLaneView &initial_values,
  const std::vector<float> &post_attention_norm,
  const std::vector<float> &ffn_norm,
  const std::vector<float> &post_ffw_norm,
  const std::vector<float> *ple_input,
  const std::vector<float> *ple_post_norm,
  const D3d12GemmaAttentionConfig &config,
  std::vector<float> *hidden_out,
  std::string *error_message = nullptr);

// Batched matmul against GPU-resident raw-format weights. Each op computes
// outputs[op][batch * row_count + row] = dot(weight_row, input_batch).
// Weights are uploaded to a persistent device buffer on first use and reused
// across calls; all ops share one packed input buffer and one submission.
bool d3d12_batch_matmul_resident(
  const D3d12ResidentWeightRef *ops,
  std::size_t op_count,
  const std::vector<float> &packed_inputs,
  std::uint32_t batch_count,
  const void *input_identity,
  std::vector<float> *outputs,
  std::string *error_message = nullptr);

// Uploads one tensor into the resident cache without dispatching compute.
// Used to pre-stage weights at model load. Returns false (without evicting)
// when the budget or per-tensor cap is reached; such tensors stage lazily on
// first use instead.
bool d3d12_stage_resident_weight(
  const D3d12ResidentWeightRef &op,
  std::string *error_message = nullptr);

// Brackets the load-time pre-stage pass. Inside it staging stops at the budget
// (the behavior documented above); outside it, decode-time staging evicts LRU
// weights instead of failing, so a model larger than the LIVE GPU budget still
// decodes on Direct3D 12.
void d3d12_set_prestaging(bool active);

// Releases the bulk staging upload buffer after a pre-stage pass.
void d3d12_shrink_staging_buffers();

// Runs a whole FFN block in ONE submission: gate and up matmuls into a
// device scratch buffer, an elementwise gelu(gate)*up shader, then the down
// matmul — no CPU round-trip between them. outputs receives
// batch_count * down.row_count floats (batch-major).
bool d3d12_ffn_gelu_glu_resident(
  const D3d12ResidentWeightRef &gate,
  const D3d12ResidentWeightRef &up,
  const D3d12ResidentWeightRef &down,
  const std::vector<float> &packed_inputs,
  std::uint32_t batch_count,
  const void *input_identity,
  std::vector<float> *outputs,
  std::string *error_message = nullptr);

// Same single-submission FFN chain with the llama-family SwiGLU activation:
// silu(gate) * up instead of gelu(gate) * up.
bool d3d12_ffn_silu_glu_resident(
  const D3d12ResidentWeightRef &gate,
  const D3d12ResidentWeightRef &up,
  const D3d12ResidentWeightRef &down,
  const std::vector<float> &packed_inputs,
  std::uint32_t batch_count,
  const void *input_identity,
  std::vector<float> *outputs,
  std::string *error_message = nullptr);

// Runs the whole Gemma decode layer tail in ONE submission: attention-output
// matmul, post-attention RMS-norm + residual add, FFN RMS-norm, gate/up
// matmuls, gelu, down matmul, post-FFW RMS-norm + residual add. Replaces two
// submissions plus all CPU elementwise work between them. Single-token
// (batch 1) only. hidden_out receives the layer's new hidden state.
bool d3d12_gemma_layer_tail_resident(
  const D3d12ResidentWeightRef &attn_out,
  const D3d12ResidentWeightRef &gate,
  const D3d12ResidentWeightRef &up,
  const D3d12ResidentWeightRef &down,
  const std::vector<float> &attention_context,
  const std::vector<float> &hidden_state,
  const std::vector<float> &post_attention_norm,
  const std::vector<float> &ffn_norm,
  const std::vector<float> &post_ffw_norm,
  float rms_epsilon,
  std::vector<float> *hidden_out,
  std::string *error_message = nullptr);

// Llama-family layer tail in ONE submission: attention-output matvec, residual
// add, FFN RMS-norm, gate/up matvecs, silu(gate)*up, down matvec, residual add.
// Same fused body as the Gemma tail above with that architecture's two extra
// norms omitted (Llama adds straight back into the hidden state at both points)
// and SwiGLU in place of GELU. Single-token (batch 1) only.
bool d3d12_llama_layer_tail_resident(
  const D3d12ResidentWeightRef &attn_out,
  const D3d12ResidentWeightRef &gate,
  const D3d12ResidentWeightRef &up,
  const D3d12ResidentWeightRef &down,
  const std::vector<float> &attention_context,
  const std::vector<float> &hidden_state,
  const std::vector<float> &ffn_norm,
  float rms_epsilon,
  std::vector<float> *hidden_out,
  std::string *error_message = nullptr);

// Batched Llama layer tail with the same topology as the single-token entry
// point above. Inputs and outputs are batch-major; one command submission
// serves every row.
bool d3d12_llama_layer_tail_batch_resident(
  const D3d12ResidentWeightRef &attn_out,
  const D3d12ResidentWeightRef &gate,
  const D3d12ResidentWeightRef &up,
  const D3d12ResidentWeightRef &down,
  const std::vector<float> &packed_attention_contexts,
  const std::vector<float> &packed_hidden_states,
  const std::vector<float> &ffn_norm,
  std::uint32_t batch_count,
  float rms_epsilon,
  std::vector<float> *packed_hidden_out,
  std::string *error_message = nullptr);

// Extended layer tail that also executes Gemma per-layer embedding (PLE):
// gate(hidden), gelu(gate)*PLE input, projection, post norm, and residual.
// The optional PLE block remains in the same command list and fence wait.
bool d3d12_gemma_layer_tail_ple_resident(
  const D3d12ResidentWeightRef &attn_out,
  const D3d12ResidentWeightRef &gate,
  const D3d12ResidentWeightRef &up,
  const D3d12ResidentWeightRef &down,
  const D3d12ResidentWeightRef &ple_gate,
  const D3d12ResidentWeightRef &ple_projection,
  const std::vector<float> &attention_context,
  const std::vector<float> &hidden_state,
  const std::vector<float> &post_attention_norm,
  const std::vector<float> &ffn_norm,
  const std::vector<float> &post_ffw_norm,
  const std::vector<float> &ple_input,
  const std::vector<float> &ple_post_norm,
  float rms_epsilon,
  std::vector<float> *hidden_out,
  std::string *error_message = nullptr);

// Drops all cached resident weight buffers (call when a model is unloaded or
// replaced so stale device copies cannot alias a new mapping).
void d3d12_reset_resident_weights();

// Total bytes currently held by resident weight buffers on the device.
std::uint64_t d3d12_resident_weight_bytes();

// Runs synthetic GPU matmul checks for every kernel/format against CPU
// reference dots and prints a report to stdout. Returns 0 when all pass.
// Used by the `--d3d12-selftest` CLI mode to isolate GPU faults from model
// execution.
int d3d12_matmul_selftest();

// Legacy immediate-mode matmul over caller-provided FP32 weights. Kept as the
// fallback lane for tensor types without a native GPU format.
bool d3d12_batch_matmul(
  const std::vector<float> &weights,
  std::uint32_t row_count,
  std::uint32_t row_width,
  const std::vector<float> &packed_inputs,
  std::uint32_t batch_count,
  const void *input_identity,
  std::vector<std::vector<float>> &outputs,
  std::string *error_message = nullptr);

}  // namespace lizard_native
