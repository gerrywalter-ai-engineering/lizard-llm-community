#include "d3d12_batch_matmul.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <memory>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#ifdef _WIN32

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <d3d12.h>
#include <d3dcompiler.h>
#include <dxgi1_6.h>

#ifdef LIZARD_HAVE_DXIL
// Offline-compiled Shader Model 6.0/6.4 DXC kernels (see CMakeLists.txt); each
// header defines one signed DXIL bytecode array.
#include "dp4a_quantize_q8_1.h"
#include "dp4a_q4k_matvec.h"
#include "dp4a_q4k_batched.h"
#include "dp4a_q4k_batched_nolds.h"
#include "dp4a_q6k_batched.h"
#include "dp4a_q4k_gemv_wave.h"
#endif

namespace lizard_native {

namespace {

constexpr std::uint64_t k_q4_k_block_bytes = 144;
constexpr std::uint64_t k_q5_k_block_bytes = 176;
constexpr std::uint64_t k_q6_k_block_bytes = 210;
// Legacy 32-element blocks. Raw layouts start with an fp16 scale, so the
// int8/nibble payload is not 4-byte aligned; staging widens the scale to F32
// and drops the payload on an aligned boundary for the ByteAddressBuffer path.
constexpr std::uint64_t k_legacy_block_elements = 32;
constexpr std::uint64_t k_q8_0_block_bytes = 34;         // fp16 d + 32 int8
constexpr std::uint64_t k_q8_0_staged_block_bytes = 36;  // f32 d + 32 int8
constexpr std::uint64_t k_q4_0_block_bytes = 18;         // fp16 d + 16 nibble bytes
constexpr std::uint64_t k_q4_0_staged_block_bytes = 20;  // f32 d + 16 nibble bytes
// Q6_K blocks are repacked to a 4-byte-aligned stride on upload so the shader
// can address every field with aligned dword loads.
constexpr std::uint64_t k_q6_k_staged_block_bytes = 212;
constexpr std::uint64_t k_quant_block_elements = 256;
constexpr std::uint64_t k_default_weight_budget_mb = 4096;
// Must match k_batch_tile in the shader source below. 16 was measured slower
// on Iris Xe: the doubled groupshared/accumulator footprint costs more in
// occupancy than the halved weight traffic saves.
constexpr std::uint32_t k_gpu_batch_tile = 8;
// Guard against absurd single-tensor uploads (e.g. per-layer token embedding
// tables that are gathered row-wise, never used as matmul weights).
constexpr std::uint64_t k_max_staged_tensor_bytes = 640ull * 1024ull * 1024ull;

// Opt-in gate for the SM 6.4 packed-integer-dot decode route inside the
// resident token graph. Default OFF until the quality and perf gates pass;
// the float kernels remain the default execution path.
bool dp4a_decode_enabled() {
  // Opt-in (LIZARD_NATIVE_GPU_DP4A=1). The SM 6.4 packed-integer route is
  // faster on the Q4_K matvecs but quantizes the activation vector to Q8_1,
  // an APPROXIMATION that can flip a greedy token (verified: the golden gate
  // drifts one case with it on), so it must not be the default path.
  static const bool enabled = [] {
    char value[8] = {};
    return GetEnvironmentVariableA("LIZARD_NATIVE_GPU_DP4A", value, sizeof(value)) != 0
      && value[0] == '1';
  }();
  return enabled;
}

// Cooperative one-workgroup-per-row Q4_K GEMV for the resident token graph.
// DEFAULT OFF: measured ~2x SLOWER than the one-thread-per-row batch-one kernel
// on Iris Xe (2.6 vs 5.0 tok/s) — the per-row barrier plus the serial lane-0
// reduction and the row_count-group launch overhead outweigh the coalescing
// win. Kept behind LIZARD_NATIVE_GPU_COOP_GEMV=1 and the selftest as the
// basis for a future DXC/SM6.4 WaveActiveSum reduction variant.
// Defined below; the format gates above it need the answer.
bool quad_interleave_enabled();

bool coop_gemv_enabled() {
  // Reads the resident buffer with block-order addressing, so it is
  // incompatible with the quad-interleaved layout and stands down when
  // that is active. main_q4k covers these shapes.
  if (quad_interleave_enabled()) return false;
  static const bool enabled = [] {
    char value[8] = {};
    return GetEnvironmentVariableA("LIZARD_NATIVE_GPU_COOP_GEMV", value, sizeof(value)) != 0
      && value[0] == '1';
  }();
  return enabled;
}

// DXC/SM6.0 WaveActiveSum cooperative one-workgroup-per-row Q4_K GEMV: the
// hardware-reduction counterpart to the FXC experiment above, built to test
// whether a barrier-free reduction would recover the coalescing win. DEFAULT
// OFF: measured 4.6x SLOWER (5.8-7.0 ms/iter vs 1.1-1.5 ms/iter float) on the
// 10240x2560 FFN-gate shape despite being bit-exact across all real Gemma
// shapes standalone (verified: a real bug — a missing input_base offset that
// silently re-read block 0's input range for every block — was found and
// fixed via per-lane partial dumps against a CPU reference; the corrected
// kernel is still this much slower standalone). The reduction method was
// never the bottleneck: both the barrier+serial-sum (FXC) and the hardware
// WaveActiveSum (DXC) variants lose to the simple one-thread-per-row kernel,
// pointing at per-dispatch group-count overhead (10240 groups vs 160 for
// batch-one) as the real cost on this adapter/driver, not the reduction.
// ADDITIONALLY, wiring this into the resident token graph currently produces
// DEGENERATE decode output (a further, uninvestigated bug distinct from the
// standalone-kernel correctness, which IS verified bit-exact by the
// selftest) — not debugged further because the kernel is already a
// conclusive performance loss even when correct. Kept opt-in behind
// LIZARD_NATIVE_GPU_WAVE_GEMV=1 purely for the selftest/reference value; DO
// NOT enable it for real decode.
bool wave_gemv_enabled() {
  // Reads the resident buffer with block-order addressing, so it is
  // incompatible with the quad-interleaved layout and stands down when
  // that is active. main_q4k covers these shapes.
  if (quad_interleave_enabled()) return false;
  static const bool enabled = [] {
    char value[8] = {};
    return GetEnvironmentVariableA("LIZARD_NATIVE_GPU_WAVE_GEMV", value, sizeof(value)) != 0
      && value[0] == '1';
  }();
  return enabled;
}

// 128-rows-per-group Q4_K decode kernel: main_q4k compiled with
// LIZARD_ROWS_PER_GROUP=128 instead of the default 64, amortizing the
// per-block load_input_tile+barrier pair over twice as many output rows per
// synchronization point. Architecturally distinct from the disproven
// cooperative-GEMV pattern above (FEWER, LARGER dispatches for the SAME
// one-thread-per-row math, not MORE/smaller ones). Default ON (2026-07-06):
// measured 1.11x faster standalone (10240x2560) and a real 3.4% direct
// decode speedup (200.4 -> 193.6 ms/token), verified bit-exact standalone
// (incl. a tail-row/non-multiple-of-128 boundary case) AND end-to-end via
// the golden greedy gate with it enabled in the real token graph - unlike
// the wave-reduction kernel, this integrates correctly. LIZARD_NATIVE_GPU_
// WIDE128=0 restores the default 64-row kernel.
bool wide128_enabled() {
  static const bool enabled = [] {
    char value[8] = {};
    return GetEnvironmentVariableA("LIZARD_NATIVE_GPU_WIDE128", value, sizeof(value)) == 0
      || value[0] != '0';
  }();
  return enabled;
}

// Opt-in (LIZARD_NATIVE_GPU_WIDE256=1): route Q4_K token-graph matvecs
// through the 256-rows-per-group pipeline instead of the 128 one. The
// kernel is selftest-verified bit-exact; standalone timings have disagreed
// between sessions (1.01x vs 1.36x relative to 128), so this gate exists to
// A/B it at real decode level before considering a default flip.
bool wide256_enabled() {
  static const bool enabled = [] {
    char value[8] = {};
    return GetEnvironmentVariableA("LIZARD_NATIVE_GPU_WIDE256", value, sizeof(value)) != 0
      && value[0] == '1';
  }();
  return enabled;
}

// Opt-in (LIZARD_NATIVE_GPU_DIRECT_INPUT=1): route plain batch-one Q4_K
// token-graph matvecs through main_q4k_direct - no groupshared input tile,
// no per-block barriers, direct broadcast reads served by the L1 cache.
// Bit-comparable math to main_q4k; gate exists for decode-level A/B before
// any default flip.
bool direct_input_enabled() {
  // Reads the resident buffer with block-order addressing, so it is
  // incompatible with the quad-interleaved layout and stands down when
  // that is active. main_q4k covers these shapes.
  if (quad_interleave_enabled()) return false;
  static const bool enabled = [] {
    char value[8] = {};
    return GetEnvironmentVariableA("LIZARD_NATIVE_GPU_DIRECT_INPUT", value, sizeof(value)) != 0
      && value[0] == '1';
  }();
  return enabled;
}

// Opt-in (LIZARD_NATIVE_GPU_BLOCK_PAIR=1): route plain batch-one Q4_K
// token-graph matvecs through main_q4k_pair - TWO 256-element blocks per
// tile-load/barrier pair instead of one, halving the group syncs per row
// and giving the scheduler two independent per-block load chains between
// barriers (the roadmap's B2 software-pipelining experiment). Bit-identical
// per-element math and accumulation order to main_q4k; gate exists for
// decode-level A/B before any default flip.
bool block_pair_enabled() {
  // Reads the resident buffer with block-order addressing, so it is
  // incompatible with the quad-interleaved layout and stands down when
  // that is active. main_q4k covers these shapes.
  if (quad_interleave_enabled()) return false;
  static const bool enabled = [] {
    char value[8] = {};
    return GetEnvironmentVariableA("LIZARD_NATIVE_GPU_BLOCK_PAIR", value, sizeof(value)) != 0
      && value[0] == '1';
  }();
  return enabled;
}

// Default ON (LIZARD_NATIVE_GPU_FUSED_GELU=0 opts out): fold the FFN
// gelu(gate)*up elementwise pass into the FFN-down matvec via the tile
// loader's existing input_transform=1 split-buffer path (same clamped tanh
// formula as main_gelu_glu, so values are identical). Removes one dispatch,
// one UAV barrier, and one full gate-width buffer write+read round-trip per
// layer per token (42x each on gemma4). Measured 2026-07-14 on the shipping
// benchmark lane (gemma4-e4b Q4_K_M, 64 tokens, adjacent runs): 4.90 ->
// 5.23 tok/s whole-run (+6.7%), identical outputs — the decode-level A/B
// the opt-in gate existed to collect.
bool fused_gelu_enabled() {
  static const bool enabled = [] {
    char value[8] = {};
    if (GetEnvironmentVariableA("LIZARD_NATIVE_GPU_FUSED_GELU", value, sizeof(value)) == 0) {
      return true;
    }
    return value[0] != '0';
  }();
  return enabled;
}

// Opt-in (LIZARD_NATIVE_GPU_DUAL_NORM=1): collapse the token graph's
// per-layer post-attention-residual + FFN-norm dispatch pair (recorder steps
// 2+3) into one main_rms_norm_dual dispatch. Byte-identical outputs by
// construction (same reduction structure, res mirrored in groupshared f32);
// removes one dispatch, one UAV barrier, and two scratch transitions per
// layer per token. Gate exists for decode-level A/B before any default flip.
bool dual_norm_enabled() {
  static const bool enabled = [] {
    char value[8] = {};
    return GetEnvironmentVariableA("LIZARD_NATIVE_GPU_DUAL_NORM", value, sizeof(value)) != 0
      && value[0] == '1';
  }();
  return enabled;
}

// Opt-in (LIZARD_NATIVE_GPU_KSPLIT=1): route narrow-output Q4_K token-graph
// matvecs (row_count <= 4096, row_width >= 2048: ffn_down, attn_out, QKV)
// through the k-split kernel pair. One thread per row caps occupancy at
// row_count; k-splitting multiplies it by S at the cost of one tiny reduce
// dispatch. 4e-3-gated (accumulation order changes). Gate exists for
// decode-level A/B before any default flip.
// Stage Q4_K weights interleaved at 16-byte quad granularity instead of
// 144-byte block granularity, and index them accordingly in main_q4k. Measured
// on Iris Xe: the quad addressing reaches 89% of the pure streaming ceiling
// against 49% for block addressing, and dequantisation costs ~2%, so this is
// the whole remaining gap. Opt-in until every Q4_K reader has been moved over -
// the layout is shared, so a kernel that still assumes block order would read
// the wrong bytes.
bool quad_interleave_enabled() {
  static const bool enabled = [] {
    char value[8] = {};
    return GetEnvironmentVariableA("LIZARD_NATIVE_GPU_QUAD_INTERLEAVE", value, sizeof(value)) != 0
      && value[0] == '1';
  }();
  return enabled;
}

bool ksplit_enabled() {
  // Reads the resident buffer with block-order addressing, so it is
  // incompatible with the quad-interleaved layout and stands down when
  // that is active. main_q4k covers these shapes.
  if (quad_interleave_enabled()) return false;
  static const bool enabled = [] {
    char value[8] = {};
    return GetEnvironmentVariableA("LIZARD_NATIVE_GPU_KSPLIT", value, sizeof(value)) != 0
      && value[0] == '1';
  }();
  return enabled;
}

// Opt-in (LIZARD_NATIVE_GPU_TIMESTAMPS=1): record GPU timestamps at the
// token graph's stage boundaries and append an aggregated per-stage summary
// to lizard-gpu-timestamps.log every 50 graphs (worker stderr is buffered
// by the host and only surfaced on crash). Diagnostics only - adds one
// ResolveQueryData + readback map per graph when enabled.
// Microseconds to spin on the fence before falling back to the blocking event
// wait. **Default 0 (off), because it was measured and it does not help.**
//
// The theory was sound: the phase profile puts a submission's fixed cost at
// 0.84 ms, decode issues 56 of them per token, and WaitForSingleObject pays a
// thread wake-up on each one while the CPU has nothing else to do. Clean A/B on
// this same binary via this variable, four repeats each:
//
//   SPIN_US=0     6.74 tok/s median
//   SPIN_US=4000  6.58 tok/s median   0.977x
//
// So the wake-up is not where the 0.84 ms goes. What remains inside a submission
// besides the GPU work: the input upload CopyBufferRegion, the command allocator
// reset and list re-record, the readback map, and resizing/memcpy of the output
// vectors. Those are the candidates - measure them individually before assuming
// which one it is, since this is now the third overhead theory to fail.
//
// Kept as an opt-in probe rather than deleted: it costs one branch when off, and
// a machine whose driver has a worse wake-up path may well benefit.
std::uint32_t fence_spin_micros() {
  static const std::uint32_t micros = [] {
    char value[16] = {};
    if (GetEnvironmentVariableA("LIZARD_NATIVE_GPU_FENCE_SPIN_US", value, sizeof(value)) == 0) {
      return 0u;
    }
    const long parsed = std::atol(value);
    return parsed < 0 ? 0u : static_cast<std::uint32_t>(std::min<long>(parsed, 50000));
  }();
  return micros;
}

// Output rows per thread for the batched Q4_K matmul (1 = off, use main_q4k).
//
// Prefill is compute-bound at 8.9% of this iGPU's peak - GEMV-shaped work on a
// GEMM problem - and widening the batch tile is already known to lose here
// (k_gpu_batch_tile). Rows-per-thread is the dimension that costs registers
// instead of groupshared, so it does not pay the occupancy penalty tile=16 paid.
//
// Measured on Iris Xe, unshared prompt prefixes so every request pays full
// prefill, five runs each, median TTFT:
//
//   R=1   5625 ms
//   R=2   4558 ms   1.23x
//   R=4   4087 ms   1.38x   <- default
//
// Greedy output is BIT-IDENTICAL across all three (sha256 of a 48-token
// completion), which is the expected result and the check that matters: this
// changes which rows a thread owns, not the order terms are summed within a row.
//
// Default 4 because that is what measured fastest on the target adapter. The
// risk it carries is register pressure - R * (8 scale + 8 min + 8 acc) floats -
// so on an adapter with a smaller register file a spill could make it slower
// rather than faster. Set this to 2 (still 1.23x) or 1 (original kernel) if that
// is ever suspected; the kernel also declines itself below 64*R rows.
// Batch columns per group for the PREFILL kernel only, independent of
// k_gpu_batch_tile which governs decode.
//
// The batched kernel dispatches ceil(batch / tile) groups in y, and every one of
// them covers all rows - so each y-group re-reads the whole weight matrix. At the
// tile of 8 that decode uses, a 48-token prefill reads the weights SIX times:
// 84 MB where 14 MB of unique weights exist. That redundancy, not arithmetic, is
// what makes prefill look compute-bound.
//
// k_gpu_batch_tile carries a note that 16 measured slower, and that result is
// real - but it was measured on decode, where batch is at most 16 and the y
// dimension is one group whatever the tile is. There a wider tile only grows the
// groupshared input tile and buys nothing. Prefill at batch 48 is the opposite
// regime: 8 -> 16 thirds the passes over the weights, 8 -> 32 halves them again.
// Groupshared cost is 256 * tile * 4 bytes, so 16 KB at tile 16 and 32 KB at
// tile 32, against the 64 KB a group may have.
std::uint32_t gpu_prefill_tile() {
  static const std::uint32_t tile = [] {
    char value[16] = {};
    // Default 8 - the value that is measured, compiles, and ships. Larger tiles
    // are the right idea on paper and are left reachable for whoever picks this
    // up, but see the unroll budget note in main_q4k_rowblock: FXC does not
    // finish compiling the wider bodies.
    if (GetEnvironmentVariableA("LIZARD_NATIVE_GPU_PREFILL_TILE", value, sizeof(value)) == 0) {
      return 8u;
    }
    const long parsed = std::atol(value);
    if (parsed >= 32) return 32u;
    if (parsed >= 16) return 16u;
    return 8u;
  }();
  return tile;
}

// Compile a second row-blocked pipeline at the other batch tile so the tile can
// be measured with the paired harness instead of across processes.
// Compile the alternate row-blocked pipeline with DOUBLE the rows per thread
// instead of a different tile, so the goal's own progression - 1x8, 2x8, 4x8 and
// now 8x8 - can be taken one step further and measured paired. Rows per thread is
// the dimension that costs registers rather than groupshared, so it is the one
// with headroom; R*tile past ~32 stalls FXC, which is why the lane loop has to be
// rolled for this comparison and why both halves of the pair get that treatment.
std::uint32_t gpu_rowblock_alt_rows() {
  static const std::uint32_t rows = [] {
    char value[8] = {};
    if (GetEnvironmentVariableA("LIZARD_NATIVE_GPU_ROWS_AB", value, sizeof(value)) == 0) {
      return 0u;
    }
    const long parsed = std::atol(value);
    return parsed >= 2 ? static_cast<std::uint32_t>(parsed) : 8u;
  }();
  return rows;
}

// Use the integer-dot GEMM for batched PREFILL matmuls.
//
// Separate from dp4a_decode_enabled because the trade is different. Both
// quantize activations to Q8_1, which that gate documents as able to flip a
// greedy token - so this cannot be a default either. What differs is the payoff:
// measured 313-376 GFLOP/s against the float row-blocked kernel's 180-235, so
// 1.3-1.5x on the phase that is 82% of time-to-first-token, where the decode
// gate was buying much less.
//
// Opt-in via LIZARD_NATIVE_GPU_DP4A_PREFILL=1, or inherited from
// LIZARD_NATIVE_GPU_DP4A=1 so the existing switch turns on both.
// Diagnostic input mode: activations chosen so Q8_1 quantisation is exact.
// Separates "the integer kernel is wrong" from "Q8_1 costs this much on
// cancelling dot products", which inspection alone could not.
// Repeat count for the isolated FFN-chain rate measurement. Off by default so
// the selftest stays fast; set to compare the chain at a real model shape
// against what prefill measures for the same op in situ.

// Rows per thread for the Q6_K row-blocked kernel, separate from gpu_row_block()
// because the Q6_K body expands to 4 FMAs per (row, batch) against Q4_K's 2:
// R=2 here is the straight-line size Q4_K reaches at R=4. Raising it walks into
// the FXC unroll cliff documented in main_q4k_rowblock.
// Batch tile for the Q6_K row-blocked kernel, separate from gpu_prefill_tile().
// The tile decides how many times the weights are re-streamed: at batch 269 a
// tile of 8 means 34 passes over the down projection's 20.6 MB per layer.
// Doubling it halves that traffic. The documented "tile 16 is 6.7x slower"
// result was measured at R=4, i.e. R*tile = 64, well past the FXC unroll cliff;
// Q6_K at R=2 x tile 16 is a different point on that curve and has to be
// measured rather than inferred from it.
std::uint32_t gpu_q6k_prefill_tile() {
  static const std::uint32_t tile = [] () -> std::uint32_t {
    const char *value = std::getenv("LIZARD_NATIVE_GPU_Q6K_TILE");
    if (value == nullptr) {
      return 0u;  // 0 = follow gpu_prefill_tile()
    }
    const long parsed = std::strtol(value, nullptr, 10);
    return parsed > 0 ? static_cast<std::uint32_t>(parsed) : 0u;
  }();
  return tile;
}

std::uint32_t gpu_q6k_row_block() {
  static const std::uint32_t rows = [] () -> std::uint32_t {
    const char *value = std::getenv("LIZARD_NATIVE_GPU_Q6K_ROWS");
    if (value == nullptr) {
      return 2u;
    }
    const long parsed = std::strtol(value, nullptr, 10);
    return parsed > 0 ? static_cast<std::uint32_t>(parsed) : 2u;
  }();
  return rows;
}

std::uint32_t ffn_chain_perf_repeats() {
  static const std::uint32_t repeats = [] () -> std::uint32_t {
    const char *value = std::getenv("LIZARD_NATIVE_FFN_CHAIN_PERF");
    if (value == nullptr) {
      return 0u;
    }
    const long parsed = std::strtol(value, nullptr, 10);
    return parsed > 0 ? static_cast<std::uint32_t>(parsed) : 0u;
  }();
  return repeats;
}

bool selftest_exact_q8_1_inputs() {
  static const bool enabled = [] {
    const char *value = std::getenv("LIZARD_NATIVE_SELFTEST_EXACT_Q8_1");
    return value != nullptr && value[0] == '1';
  }();
  return enabled;
}

bool gpu_dp4a_prefill_enabled() {
  static const bool enabled = [] {
    char value[8] = {};
    if (GetEnvironmentVariableA(
        "LIZARD_NATIVE_GPU_DP4A_PREFILL", value, sizeof(value)) != 0) {
      return value[0] == '1';
    }
    return dp4a_decode_enabled();
  }();
  return enabled;
}

bool gpu_rowblock_alt_tile_enabled() {
  static const bool enabled = [] {
    char value[8] = {};
    return GetEnvironmentVariableA("LIZARD_NATIVE_GPU_TILE_AB", value, sizeof(value)) != 0
      && value[0] == '1';
  }();
  return enabled;
}

std::uint32_t gpu_row_block() {
  static const std::uint32_t rows = [] {
    char value[16] = {};
    if (GetEnvironmentVariableA("LIZARD_NATIVE_GPU_ROW_BLOCK", value, sizeof(value)) == 0) {
      return 4u;
    }
    const long parsed = std::atol(value);
    if (parsed <= 1) return 1u;
    // Only powers of two up to 4: the row stride assumes k_rows_per_group lanes
    // per round, and beyond 4 the scale arrays alone exceed a sensible register
    // budget (R * 16 floats).
    return parsed >= 4 ? 4u : 2u;
  }();
  return rows;
}

bool gpu_timestamps_enabled() {
  static const bool enabled = [] {
    char value[8] = {};
    return GetEnvironmentVariableA("LIZARD_NATIVE_GPU_TIMESTAMPS", value, sizeof(value)) != 0
      && value[0] == '1';
  }();
  return enabled;
}

// Position-tiled online attention (LIZARD_NATIVE_GPU_ATTN_TILE=0 disables):
// the untiled kernel pays ~11 group barriers per attended position, which is
// why attention cost grows linearly with context. The tiled kernel processes
// 8 positions per barrier set with the identical floating-point order
// (bit-exact against the untiled kernel — verified by the selftest's
// bitwise comparison). Read fresh (not static-cached) so the selftest can
// exercise both kernels in one process.
bool gemma_attention_tiled_enabled() {
  char value[8] = {};
  if (GetEnvironmentVariableA("LIZARD_NATIVE_GPU_ATTN_TILE", value, sizeof(value)) == 0) {
    return true;
  }
  return value[0] != '0';
}


template<typename T>
class ComPtr {
 public:
  ComPtr() = default;
  ~ComPtr() {
    reset();
  }

  ComPtr(const ComPtr &) = delete;
  ComPtr &operator=(const ComPtr &) = delete;

  ComPtr(ComPtr &&other) noexcept
    : ptr_(other.ptr_) {
    other.ptr_ = nullptr;
  }

  ComPtr &operator=(ComPtr &&other) noexcept {
    if (this != &other) {
      reset();
      ptr_ = other.ptr_;
      other.ptr_ = nullptr;
    }
    return *this;
  }

  T *get() const {
    return ptr_;
  }

  T **put() {
    reset();
    return &ptr_;
  }

  T *operator->() const {
    return ptr_;
  }

  explicit operator bool() const {
    return ptr_ != nullptr;
  }

  void reset(T *ptr = nullptr) {
    if (ptr_ != nullptr) {
      ptr_->Release();
    }
    ptr_ = ptr;
  }

  T *detach() {
    T *ptr = ptr_;
    ptr_ = nullptr;
    return ptr;
  }

 private:
  T *ptr_ = nullptr;
};

std::string narrow_utf16(const wchar_t *value) {
  if (value == nullptr || *value == L'\0') {
    return "";
  }
  const int required = WideCharToMultiByte(CP_UTF8, 0, value, -1, nullptr, 0, nullptr, nullptr);
  if (required <= 1) {
    return "";
  }
  std::string output(static_cast<std::size_t>(required), '\0');
  WideCharToMultiByte(CP_UTF8, 0, value, -1, output.data(), required, nullptr, nullptr);
  if (!output.empty() && output.back() == '\0') {
    output.pop_back();
  }
  return output;
}

std::string hresult_message(const char *label, HRESULT hr) {
  std::ostringstream stream;
  stream << label << " failed with HRESULT 0x" << std::hex << static_cast<std::uint32_t>(hr);
  return stream.str();
}

void trace_d3d12(const std::string &message) {
  char enabled[8] = {};
  if (GetEnvironmentVariableA("LIZARD_NATIVE_D3D12_TRACE", enabled, sizeof(enabled)) == 0) {
    return;
  }
  char temp_path[MAX_PATH] = {};
  const DWORD length = GetTempPathA(MAX_PATH, temp_path);
  if (length == 0 || length >= MAX_PATH) {
    return;
  }
  std::ofstream output(std::string(temp_path) + "lizard-native-d3d12.log", std::ios::app);
  output << message << '\n';
}

// FXC compiles the float matmul kernels at OPTIMIZATION_LEVEL3 on every worker
// process start (~9 shaders). On slower/first-run integrated GPUs that front-end
// compile dominates cold model-load time (tens of seconds to over a minute),
// and because each benchmark row spawns a fresh worker, it is repaid every row -
// the reason a native/caterpillar cold load is far slower than llama.cpp's
// mmap-backed start. The compiled DXBC bytecode is device-independent, so it is
// cached on disk keyed by the shader source + entry point + macro variant; a
// later process reuses the blob and skips D3DCompile entirely. Corruption or a
// source change self-heals: a bad blob fails CreateComputePipelineState and the
// caller falls back to a fresh compile that overwrites the cache.
std::uint64_t fnv1a_hash(const void *data, std::size_t length) {
  const auto *bytes = static_cast<const unsigned char *>(data);
  std::uint64_t hash = 1469598103934665603ull;
  for (std::size_t index = 0; index < length; index += 1) {
    hash ^= static_cast<std::uint64_t>(bytes[index]);
    hash *= 1099511628211ull;
  }
  return hash;
}

bool shader_cache_disabled() {
  char value[8] = {};
  return GetEnvironmentVariableA("LIZARD_NATIVE_SHADER_CACHE", value, sizeof(value)) != 0
    && std::string(value) == "0";
}

std::filesystem::path shader_cache_path(std::uint64_t source_hash, const std::string &variant) {
  std::filesystem::path base;
  char override_dir[MAX_PATH] = {};
  char local_appdata[MAX_PATH] = {};
  if (GetEnvironmentVariableA("LIZARD_NATIVE_SHADER_CACHE_DIR", override_dir, sizeof(override_dir)) != 0) {
    base = std::filesystem::path(override_dir);
  } else if (GetEnvironmentVariableA("LOCALAPPDATA", local_appdata, sizeof(local_appdata)) != 0) {
    base = std::filesystem::path(local_appdata) / "lizard-llm" / "shader-cache";
  } else {
    char temp_path[MAX_PATH] = {};
    const DWORD length = GetTempPathA(MAX_PATH, temp_path);
    if (length == 0 || length >= MAX_PATH) {
      return {};
    }
    base = std::filesystem::path(temp_path) / "lizard-llm-shader-cache";
  }
  std::ostringstream name;
  name << "d3d12_cs50_" << std::hex << source_hash << "_" << variant << ".cso";
  return base / name.str();
}

std::vector<unsigned char> read_cached_shader_blob(const std::filesystem::path &path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) {
    return {};
  }
  const std::streamsize size = input.tellg();
  if (size <= 0) {
    return {};
  }
  input.seekg(0, std::ios::beg);
  std::vector<unsigned char> buffer(static_cast<std::size_t>(size));
  if (!input.read(reinterpret_cast<char *>(buffer.data()), size)) {
    return {};
  }
  return buffer;
}

void write_cached_shader_blob(const std::filesystem::path &path, const void *data, std::size_t length) {
  std::error_code ec;
  std::filesystem::create_directories(path.parent_path(), ec);
  // Write to a unique temp file then rename so a concurrent worker never reads a
  // half-written blob.
  const std::filesystem::path temp = path.string() + "." + std::to_string(GetCurrentProcessId()) + ".tmp";
  {
    std::ofstream output(temp, std::ios::binary | std::ios::trunc);
    if (!output) {
      return;
    }
    output.write(static_cast<const char *>(data), static_cast<std::streamsize>(length));
    if (!output) {
      output.close();
      std::filesystem::remove(temp, ec);
      return;
    }
  }
  std::filesystem::rename(temp, path, ec);
  if (ec) {
    std::filesystem::remove(temp, ec);
  }
}

std::uint64_t configured_weight_budget_bytes() {
  char value[32] = {};
  if (GetEnvironmentVariableA("LIZARD_NATIVE_GPU_WEIGHT_BUDGET_MB", value, sizeof(value)) != 0) {
    const std::uint64_t parsed = std::strtoull(value, nullptr, 10);
    if (parsed > 0) {
      return parsed * 1024ull * 1024ull;
    }
  }
  return k_default_weight_budget_mb * 1024ull * 1024ull;
}


// All matmul shaders share one dispatch layout: 64 rows per thread group on X,
// 8 batch entries per thread group on Y. Weights are read from a raw buffer in
// their storage format and dequantized in registers, so each weight byte is
// fetched once and applied to the whole batch tile from groupshared memory.
static constexpr char k_shader_source[] = R"(
cbuffer DispatchParams : register(b0) {
  uint row_width;
  uint row_count;
  uint batch_count;
  uint output_base;
  uint scalar_format;
  uint input_stride;
  uint weight_layout;  // 0 = row-major, 1 = column-major (block-interleaved)
  uint input_transform;  // 0 = identity, 1 = gelu(gate)*up over a split buffer
};

ByteAddressBuffer weights : register(t0);
StructuredBuffer<float> inputs : register(t1);
RWStructuredBuffer<float> outputs : register(u0);

#ifndef LIZARD_BATCH_TILE
#define LIZARD_BATCH_TILE 8
#endif
#ifndef LIZARD_ROWS_PER_GROUP
#define LIZARD_ROWS_PER_GROUP 64
#endif
#ifndef LIZARD_ROWS_PER_THREAD
#define LIZARD_ROWS_PER_THREAD 1
#endif
static const uint k_rows_per_group = LIZARD_ROWS_PER_GROUP;
static const uint k_batch_tile = LIZARD_BATCH_TILE;
// Output rows per thread. 1 reproduces main_q4k exactly; main_q4k_rowblock is
// the only kernel that reads it.
static const uint k_rows_per_thread = LIZARD_ROWS_PER_THREAD;
static const uint k_tile_elements = 256;

groupshared float input_tile[k_tile_elements * k_batch_tile];

uint byte_at(uint word, uint lane) {
  return (word >> (lane * 8u)) & 0xFFu;
}

uint buffer_byte(uint byte_address) {
  return (weights.Load(byte_address & ~3u) >> ((byte_address & 3u) * 8u)) & 0xFFu;
}

uint scale_k4(uint j, uint scales_byte_base) {
  return j < 4u
    ? (buffer_byte(scales_byte_base + j) & 63u)
    : ((buffer_byte(scales_byte_base + j + 4u) & 0x0Fu)
      | ((buffer_byte(scales_byte_base + j - 4u) >> 6u) << 4u));
}

uint min_k4(uint j, uint scales_byte_base) {
  return j < 4u
    ? (buffer_byte(scales_byte_base + j + 4u) & 63u)
    : ((buffer_byte(scales_byte_base + j + 4u) >> 4u)
      | ((buffer_byte(scales_byte_base + j) >> 6u) << 4u));
}

// scale_k4/min_k4 above each re-issue a full weights.Load() per single byte
// (via buffer_byte), so unpacking all 8 scale+8 min values costs 24 redundant
// 32-bit loads per block even though the bytes they touch all live in the
// same 12 contiguous bytes (3 dwords). These _words variants take that one
// Load3 result and extract bytes from the registers instead - identical bit
// layout and math, only the memory traffic changes (24 loads -> 1 per
// block). Used by the two production kernels (main_q4k, main_q5k); the
// opt-in reference kernels keep calling scale_k4/min_k4 unchanged since they
// exist only as a slower-but-simple correctness baseline.
uint scale_byte_at(uint3 words, uint byte_index) {
  const uint word = byte_index < 4u ? words.x : (byte_index < 8u ? words.y : words.z);
  return byte_at(word, byte_index & 3u);
}

uint scale_k4_words(uint j, uint3 scale_words) {
  return j < 4u
    ? (scale_byte_at(scale_words, j) & 63u)
    : ((scale_byte_at(scale_words, j + 4u) & 0x0Fu)
      | ((scale_byte_at(scale_words, j - 4u) >> 6u) << 4u));
}

uint min_k4_words(uint j, uint3 scale_words) {
  return j < 4u
    ? (scale_byte_at(scale_words, j + 4u) & 63u)
    : ((scale_byte_at(scale_words, j + 4u) >> 4u)
      | ((scale_byte_at(scale_words, j) >> 6u) << 4u));
}

int signed_scale16(uint4 words, uint index) {
  const uint word_index = index >> 2u;
  const uint word = word_index == 0u
    ? words.x
    : (word_index == 1u ? words.y : (word_index == 2u ? words.z : words.w));
  const uint value = (word >> ((index & 3u) * 8u)) & 0xFFu;
  return (int)(value << 24u) >> 24;
}

void load_input_tile(uint thread_index, uint batch_start, uint element_base) {
  [unroll]
  for (uint slot = 0; slot < (k_tile_elements * k_batch_tile) / k_rows_per_group; slot += 1u) {
    const uint flat = slot * k_rows_per_group + thread_index;
    const uint tile_batch = flat / k_tile_elements;
    const uint tile_element = flat % k_tile_elements;
    const uint batch = batch_start + tile_batch;
    const uint element = element_base + tile_element;
    // Clamp indices BEFORE any load: drivers may flatten these branches and
    // execute loads speculatively, and an out-of-bounds root-descriptor read
    // is undefined behavior (NaN results or a device hang). Only the stored
    // VALUE is masked by the real bounds check.
    const uint safe_batch = min(batch, batch_count - 1u);
    const uint safe_element = min(element, row_width - 1u);
    const uint base_index = safe_batch * input_stride + safe_element;
    const uint second_index = input_transform == 1u
      ? batch_count * input_stride + base_index
      : base_index;
    const float first_value = inputs[base_index];
    const float second_value = inputs[second_index];
    float value = first_value;
    if (input_transform == 1u) {
      // Split gate/up buffer: fuse gelu(gate)*up while staging the tile so
      // the FFN down matmul needs no separate elementwise pass. The tanh
      // argument is clamped because driver tanh is exp-based and returns
      // NaN (inf/inf) for |x| > ~44, while tanh(+-15) already rounds to
      // +-1.0f exactly.
      const float cubed = first_value * first_value * first_value;
      const float tanh_argument = clamp(
        0.7978845608028654f * (first_value + 0.044715f * cubed),
        -15.0f,
        15.0f);
      value = 0.5f * first_value * (1.0f + tanh(tanh_argument)) * second_value;
    }
    input_tile[flat] = (batch < batch_count && element < row_width) ? value : 0.0f;
  }
}

void store_outputs(uint row, uint batch_start, float acc[k_batch_tile]) {
  if (row >= row_count) {
    return;
  }
  [unroll]
  for (uint out_batch = 0; out_batch < k_batch_tile; out_batch += 1u) {
    const uint batch = batch_start + out_batch;
    if (batch < batch_count) {
      outputs[output_base + batch * row_count + row] = acc[out_batch];
    }
  }
}

[numthreads(k_rows_per_group, 1, 1)]
void main_q4k(uint3 group_id : SV_GroupID, uint thread_index : SV_GroupIndex) {
  const uint row = group_id.x * k_rows_per_group + thread_index;
  const uint batch_start = group_id.y * k_batch_tile;

  float acc[k_batch_tile];
  [unroll]
  for (uint init = 0; init < k_batch_tile; init += 1u) {
    acc[init] = 0.0f;
  }

  const uint blocks_per_row = row_width >> 8u;

  [loop]
  for (uint block = 0; block < blocks_per_row; block += 1u) {
    load_input_tile(thread_index, batch_start, block << 8u);
    GroupMemoryBarrierWithGroupSync();

    if (row < row_count) {
      // Blocks are stored column-major so 64 adjacent rows in this group
      // read 64 adjacent 144-byte blocks: one contiguous 9 KB stream.
      //
      // LIZARD_QUAD_INTERLEAVE goes finer: quad q of block b for every row is
      // stored together, so lane j's Load4 lands 16 bytes from lane j+1's
      // rather than 144. The block layout makes the wave's total span
      // contiguous but leaves each individual load scattered over 32 cache
      // lines; measured, the finer order reaches 89% of the streaming ceiling
      // against 49%. quad_base(q) is the byte offset of this row's quad q.
#ifdef LIZARD_QUAD_INTERLEAVE
      const uint block_quad0 = (block * 9u) * row_count + row;
      const uint block_byte = block_quad0 * 16u;
#define LIZARD_QUAD_BYTE(q) ((((block * 9u) + (q)) * row_count + row) * 16u)
#else
      const uint block_byte = (block * row_count + row) * 144u;
#define LIZARD_QUAD_BYTE(q) (block_byte + (q) * 16u)
#endif
      // The block header is d/dmin (4 bytes) plus 12 packed scale bytes: 16
      // bytes exactly. Under the quad-interleaved layout those 16 bytes are a
      // single quad, so one Load4 replaces a 4-byte Load and a 12-byte Load3.
      // Two narrow loads gate every block and neither can coalesce by width no
      // matter how the bytes are ordered - which is the leading suspect for why
      // interleaving the quant bytes alone did not move the kernel.
#ifdef LIZARD_QUAD_INTERLEAVE
      const uint4 header_words = weights.Load4(block_byte);
      const uint d_words = header_words.x;
      const uint3 scale_words = uint3(header_words.y, header_words.z, header_words.w);
#else
      const uint d_words = weights.Load(block_byte);
      // Hoist the eight sub-block scale/min pairs so the element loop only
      // touches quant bytes. Load the 12 packed scale bytes ONCE (Load3)
      // instead of the 24 redundant single-byte reloads scale_k4/min_k4
      // would otherwise issue - same bytes, same math.
      const uint3 scale_words = weights.Load3(block_byte + 4u);
#endif
      const float d = f16tof32(d_words & 0xFFFFu);
      const float dmin = f16tof32(d_words >> 16u);
      float scale_d[8];
      float scale_m[8];
      [unroll]
      for (uint j = 0; j < 8u; j += 1u) {
        scale_d[j] = d * (float)scale_k4_words(j, scale_words);
        scale_m[j] = dmin * (float)min_k4_words(j, scale_words);
      }

      // The 128 quant bytes are contiguous; byte index qi covers sub-chunk
      // chunk_id = qi/32, item = qi%32; its low nibble is element
      // chunk_id*64+item and its high nibble is that plus 32.
      // Load4 fetches 16 bytes (4 dwords) per call instead of 4: 8 wide loads
      // instead of 32 narrow ones for the same 128 bytes, raising per-thread
      // memory-level parallelism without changing dispatch geometry (measured
      // win on Iris Xe after two cooperative-dispatch kernels both proved the
      // bottleneck was dispatch/group-launch overhead, not per-byte work).
      [loop]
      for (uint quant_word4 = 0; quant_word4 < 8u; quant_word4 += 1u) {
        const uint4 packed_words = weights.Load4(LIZARD_QUAD_BYTE(1u + quant_word4));
        [unroll]
        for (uint sub_dword = 0; sub_dword < 4u; sub_dword += 1u) {
          const uint packed4 = sub_dword == 0u ? packed_words.x
            : sub_dword == 1u ? packed_words.y
            : sub_dword == 2u ? packed_words.z
            : packed_words.w;
          [unroll]
          for (uint lane = 0; lane < 4u; lane += 1u) {
            const uint quant_index = (quant_word4 * 4u + sub_dword) * 4u + lane;
            const uint chunk_id = quant_index >> 5u;
            const uint item = quant_index & 31u;
            const uint quant_byte = byte_at(packed4, lane);
            const uint low_element = chunk_id * 64u + item;
            const float weight_low = scale_d[chunk_id * 2u] * (float)(quant_byte & 0x0Fu)
              - scale_m[chunk_id * 2u];
            const float weight_high = scale_d[chunk_id * 2u + 1u] * (float)(quant_byte >> 4u)
              - scale_m[chunk_id * 2u + 1u];
            [unroll]
            for (uint b = 0; b < k_batch_tile; b += 1u) {
              const uint tile_base = b * k_tile_elements;
              acc[b] += weight_low * input_tile[tile_base + low_element]
                + weight_high * input_tile[tile_base + low_element + 32u];
            }
          }
        }
      }
    }
    GroupMemoryBarrierWithGroupSync();
  }

  store_outputs(row, batch_start, acc);
}
#undef LIZARD_QUAD_BYTE

// 2D register blocking: k_rows_per_thread output ROWS x k_batch_tile batch
// columns per thread, instead of main_q4k's 1 x k_batch_tile.
//
// Why rows and not a wider batch tile: prefill measured 142 GFLOP/s, 8.9% of this
// iGPU's ~1.6 TFLOP/s, and it is compute-bound (bandwidth predicts 2.1 ms per
// layer against 57.3 measured). That profile is a GEMV-shaped kernel on a GEMM
// problem. Widening the batch tile is the obvious response and is already known
// to LOSE here - see k_gpu_batch_tile, where 16 cost more occupancy through the
// doubled groupshared input tile than it saved in weight traffic.
//
// Rows-per-thread is the dimension that does not touch groupshared at all. The
// input tile is indexed by batch, so R rows reuse the SAME two LDS reads:
// main_q4k issues 2 LDS reads per 2 FMAs, this issues 2 per 2R FMAs. Cost is
// registers (R accumulator sets and R scale sets), which is the resource
// with headroom.
//
// Row assignment is strided by k_rows_per_group deliberately. Blocks are stored
// column-major, so the wave's 64 lanes must read 64 ADJACENT blocks to stay one
// contiguous stream; giving thread t rows {base+t, base+64+t, ...} keeps every
// one of the R load rounds coalesced, where a contiguous {base+R*t, ...}
// assignment would scatter each round across R*144-byte strides.
//
// Block-order addressing only: the quad-interleaved layout is opt-in and the
// host gate declines this kernel when it is active, exactly as the k-split lane
// does.
[numthreads(k_rows_per_group, 1, 1)]
void main_q4k_rowblock(uint3 group_id : SV_GroupID, uint thread_index : SV_GroupIndex) {
  const uint group_row_base = group_id.x * (k_rows_per_group * k_rows_per_thread);
  const uint batch_start = group_id.y * k_batch_tile;

  uint rows[k_rows_per_thread];
  [unroll]
  for (uint r_init = 0; r_init < k_rows_per_thread; r_init += 1u) {
    rows[r_init] = group_row_base + r_init * k_rows_per_group + thread_index;
  }

  float acc[k_rows_per_thread * k_batch_tile];
  [unroll]
  for (uint init = 0; init < k_rows_per_thread * k_batch_tile; init += 1u) {
    acc[init] = 0.0f;
  }

  const uint blocks_per_row = row_width >> 8u;

  [loop]
  for (uint block = 0; block < blocks_per_row; block += 1u) {
    load_input_tile(thread_index, batch_start, block << 8u);
    GroupMemoryBarrierWithGroupSync();

    float scale_d[k_rows_per_thread * 8];
    float scale_m[k_rows_per_thread * 8];
    uint row_block_byte[k_rows_per_thread];

    [unroll]
    for (uint rs = 0; rs < k_rows_per_thread; rs += 1u) {
      // Clamp before loading: an out-of-bounds root-descriptor read is undefined
      // behaviour even if the value is discarded, so inactive rows read row 0 and
      // are dropped at store time instead.
      const uint safe_row = min(rows[rs], row_count - 1u);
      const uint block_byte = (block * row_count + safe_row) * 144u;
      row_block_byte[rs] = block_byte;
      const uint d_words = weights.Load(block_byte);
      const uint3 scale_words = weights.Load3(block_byte + 4u);
      const float d = f16tof32(d_words & 0xFFFFu);
      const float dmin = f16tof32(d_words >> 16u);
      [unroll]
      for (uint j = 0; j < 8u; j += 1u) {
        scale_d[rs * 8u + j] = d * (float)scale_k4_words(j, scale_words);
        scale_m[rs * 8u + j] = dmin * (float)min_k4_words(j, scale_words);
      }
    }

    // Software-pipelined weight loads.
    //
    // The measured reason: this kernel runs at 3-19% of the ALU ceiling and 39%
    // of the memory bus, so it saturates neither and is therefore waiting. Loads
    // issued and consumed in the same iteration leave only k_rows_per_thread
    // requests in flight; fetching the NEXT iteration's quads before consuming
    // the current ones doubles that, which is what a latency-bound kernel needs
    // rather than more arithmetic per byte.
    uint4 packed_rows[k_rows_per_thread];
    [unroll]
    for (uint rp = 0; rp < k_rows_per_thread; rp += 1u) {
      packed_rows[rp] = weights.Load4(row_block_byte[rp] + 16u);
    }

    [loop]
    for (uint quant_word4 = 0; quant_word4 < 8u; quant_word4 += 1u) {
      // Issue the next iteration's loads before consuming this one's, so the
      // fetch overlaps the dequantise-and-accumulate below instead of following
      // it. The clamp keeps the final iteration's address in range; its result
      // is simply never used.
      uint4 prefetched[k_rows_per_thread];
      [unroll]
      for (uint rf = 0; rf < k_rows_per_thread; rf += 1u) {
        const uint next_word4 = min(quant_word4 + 1u, 7u);
        prefetched[rf] = weights.Load4(row_block_byte[rf] + (1u + next_word4) * 16u);
      }
      [unroll]
      for (uint sub_dword = 0; sub_dword < 4u; sub_dword += 1u) {
        // Unrolled, and it must stay that way: rolling this loop drops the
        // paired row-blocking gain from 1.578x to 1.274x. It is also what caps
        // the batch tile, because FXC's compile cost is the PRODUCT of the
        // unrolled loops - sub_dword 4 x lane 4 x rows R x tile - and past about
        // R*tile = 32 it does not finish at all (a run compiling R=4 with tile 16
        // wrote no cache blob in 5.5 minutes and printed nothing). That ceiling
        // costs nothing worth having: measured paired, with this loop rolled so
        // both tiles could compile, tile 16 ran 93.23 ms against tile 8's
        // 14.28 ms - 6.7x SLOWER. Set LIZARD_NATIVE_GPU_TILE_AB=1 to re-measure.
        // Unrolled, and this is the configuration the parameter space bottoms out
        // at. FXC's compile cost is the product of the unrolled loops -
        // sub_dword 4 x lane 4 x rows x tile - and R*tile past 32 both stops FXC
        // finishing AND falls off a performance cliff. Measured paired, against
        // the shipped 4x8:
        //
        //   4 x 16   0.149x   6.7x slower
        //   8 x 8    0.139x   7.2x slower
        //
        // Two different ways of reaching R*tile = 64, the same collapse, so it is
        // one cliff rather than two coincidences. 4x8 = 32 is the last product
        // that stays on the good side of it. Rolling this loop to get past the
        // compile limit costs the gain it was meant to buy (1.578x -> 1.274x), so
        // the ceiling is not worth working around.
        //
        // LIZARD_NATIVE_GPU_ROWS_AB and LIZARD_NATIVE_GPU_TILE_AB re-measure both
        // axes with the paired harness.
        [unroll]
        for (uint lane = 0; lane < 4u; lane += 1u) {
          const uint quant_index = (quant_word4 * 4u + sub_dword) * 4u + lane;
          const uint chunk_id = quant_index >> 5u;
          const uint item = quant_index & 31u;
          const uint low_element = chunk_id * 64u + item;

          float weight_low[k_rows_per_thread];
          float weight_high[k_rows_per_thread];
          [unroll]
          for (uint rw = 0; rw < k_rows_per_thread; rw += 1u) {
            const uint4 pw = packed_rows[rw];
            const uint packed4 = sub_dword == 0u ? pw.x
              : sub_dword == 1u ? pw.y
              : sub_dword == 2u ? pw.z
              : pw.w;
            const uint quant_byte = byte_at(packed4, lane);
            const uint scale_slot = rw * 8u + chunk_id * 2u;
            weight_low[rw] = scale_d[scale_slot] * (float)(quant_byte & 0x0Fu)
              - scale_m[scale_slot];
            weight_high[rw] = scale_d[scale_slot + 1u] * (float)(quant_byte >> 4u)
              - scale_m[scale_slot + 1u];
          }
          // The two LDS reads below are the whole point: they are hoisted out of
          // the row loop and reused across all R rows.
          //
          // Unrolled, and the product R * k_batch_tile is what has to stay in
          // budget. This body is multiplied by the unrolled sub_dword 4 and
          // lane 4 around it, so R=4 with tile 16 is 1024 straight-line
          // multiply-adds per quant_word4 iteration and FXC never finished
          // compiling it - the selftest produced no output because the pipeline
          // was still being built. Leaving it as a real loop compiles instantly
          // and costs 4x the runtime (55 GFLOP/s against 222), so the unroll is
          // not optional: keep R * tile near 32 instead.
          [unroll]
          for (uint b = 0; b < k_batch_tile; b += 1u) {
            const uint tile_base = b * k_tile_elements;
            const float in_low = input_tile[tile_base + low_element];
            const float in_high = input_tile[tile_base + low_element + 32u];
            [unroll]
            for (uint ra = 0; ra < k_rows_per_thread; ra += 1u) {
              acc[ra * k_batch_tile + b] += weight_low[ra] * in_low
                + weight_high[ra] * in_high;
            }
          }
        }
      }
      [unroll]
      for (uint rn = 0; rn < k_rows_per_thread; rn += 1u) {
        packed_rows[rn] = prefetched[rn];
      }
    }
    GroupMemoryBarrierWithGroupSync();
  }

  [unroll]
  for (uint ro = 0; ro < k_rows_per_thread; ro += 1u) {
    const uint row = rows[ro];
    if (row < row_count) {
      [unroll]
      for (uint out_batch = 0; out_batch < k_batch_tile; out_batch += 1u) {
        const uint batch = batch_start + out_batch;
        if (batch < batch_count) {
          outputs[output_base + batch * row_count + row] = acc[ro * k_batch_tile + out_batch];
        }
      }
    }
  }
}

// Load-only twin of main_q4k: identical addressing, identical Load/Load3/Load4
// instructions in the identical order, with the dequantisation arithmetic
// replaced by an integer accumulate. Diagnostic only - never used for real
// decode, and its output is meaningless.
//
// The gap between this and main_q4k is what dequantisation COSTS. The gap
// between this and the contiguous streaming ceiling is what main_q4k's memory
// access shape costs despite the column-major interleaving. Measuring both
// separates the two, which is the question left open after the interleaving was
// found to be already in place.
[numthreads(k_rows_per_group, 1, 1)]
void main_q4k_loadonly(uint3 group_id : SV_GroupID, uint thread_index : SV_GroupIndex) {
  const uint row = group_id.x * k_rows_per_group + thread_index;
  const uint blocks_per_row = row_width >> 8u;
  uint acc = 0u;
  for (uint block = 0; block < blocks_per_row; block += 1u) {
    const uint block_byte = (block * row_count + row) * 144u;
    acc += weights.Load(block_byte);
    const uint3 scale_words = weights.Load3(block_byte + 4u);
    acc += scale_words.x + scale_words.y + scale_words.z;
    [loop]
    for (uint quant_word4 = 0; quant_word4 < 8u; quant_word4 += 1u) {
      const uint4 packed_words = weights.Load4(block_byte + 16u + quant_word4 * 16u);
      acc += packed_words.x + packed_words.y + packed_words.z + packed_words.w;
    }
  }
  outputs[row] = asfloat(acc);
}

// Load-only twin again, but addressing the SAME buffer as if it were
// interleaved at 16-byte quad granularity instead of 144-byte block
// granularity: quad q of block b for every row stored together, so lane j's
// Load4 sits 16 bytes from lane j+1's.
//
// The data read is meaningless (the buffer is not actually in this order) but
// the ADDRESS PATTERN is exactly what a quad-interleaved layout would produce,
// and the byte volume is identical. That is enough to price the change before
// paying for it: if this reaches the contiguous ceiling, re-laying the weights
// is worth doing; if it lands with the block-interleaved twin, it is not.
[numthreads(k_rows_per_group, 1, 1)]
void main_q4k_loadonly_coalesced(uint3 group_id : SV_GroupID, uint thread_index : SV_GroupIndex) {
  const uint row = group_id.x * k_rows_per_group + thread_index;
  const uint blocks_per_row = row_width >> 8u;
  const uint quads_per_block = 9u;  // 144 bytes
  uint acc = 0u;
  for (uint block = 0; block < blocks_per_row; block += 1u) {
    [loop]
    for (uint quad = 0; quad < quads_per_block; quad += 1u) {
      const uint quad_index = (block * quads_per_block + quad) * row_count + row;
      const uint4 chunk = weights.Load4(quad_index * 16u);
      acc += chunk.x + chunk.y + chunk.z + chunk.w;
    }
  }
  outputs[row] = asfloat(acc);
}

// Batch-one Q4_K matvec with DIRECT input reads: no groupshared tile and no
// barriers. Decode matvecs broadcast the same input vector to every thread
// in the group, so the L1 cache serves the reads without staging; dropping
// the two group barriers per 256-element block lets each row stream
// unsynchronized. Same per-element math and accumulation order as main_q4k,
// so results are bit-comparable. input_transform is NOT supported (the
// token graph runs gelu as a dedicated pass); C++ routes only plain
// batch-one matvecs here.
[numthreads(k_rows_per_group, 1, 1)]
void main_q4k_direct(uint3 group_id : SV_GroupID, uint thread_index : SV_GroupIndex) {
  const uint row = group_id.x * k_rows_per_group + thread_index;
  if (row >= row_count) {
    return;
  }
  float acc = 0.0f;
  const uint blocks_per_row = row_width >> 8u;
  [loop]
  for (uint block = 0; block < blocks_per_row; block += 1u) {
    const uint input_base = block << 8u;
    const uint block_byte = (block * row_count + row) * 144u;
    const uint d_words = weights.Load(block_byte);
    const float d = f16tof32(d_words & 0xFFFFu);
    const float dmin = f16tof32(d_words >> 16u);
    const uint scales_byte_base = block_byte + 4u;
    // Same Load3-once scale unpack as main_q4k (iteration 25): one wide load
    // for the 12 packed scale bytes instead of 24 redundant byte reloads.
    const uint3 scale_words = weights.Load3(scales_byte_base);
    float scale_d[8];
    float scale_m[8];
    [unroll]
    for (uint j = 0; j < 8u; j += 1u) {
      scale_d[j] = d * (float)scale_k4_words(j, scale_words);
      scale_m[j] = dmin * (float)min_k4_words(j, scale_words);
    }
    [loop]
    for (uint quant_word4 = 0; quant_word4 < 8u; quant_word4 += 1u) {
      const uint4 packed_words = weights.Load4(block_byte + 16u + quant_word4 * 16u);
      [unroll]
      for (uint sub_dword = 0; sub_dword < 4u; sub_dword += 1u) {
        const uint packed4 = sub_dword == 0u ? packed_words.x
          : sub_dword == 1u ? packed_words.y
          : sub_dword == 2u ? packed_words.z
          : packed_words.w;
        [unroll]
        for (uint lane = 0; lane < 4u; lane += 1u) {
          const uint quant_index = (quant_word4 * 4u + sub_dword) * 4u + lane;
          const uint chunk_id = quant_index >> 5u;
          const uint item = quant_index & 31u;
          const uint quant_byte = byte_at(packed4, lane);
          const uint low_element = input_base + chunk_id * 64u + item;
          const float weight_low = scale_d[chunk_id * 2u] * (float)(quant_byte & 0x0Fu)
            - scale_m[chunk_id * 2u];
          const float weight_high = scale_d[chunk_id * 2u + 1u] * (float)(quant_byte >> 4u)
            - scale_m[chunk_id * 2u + 1u];
          acc += weight_low * inputs[low_element]
            + weight_high * inputs[low_element + 32u];
        }
      }
    }
  }
  outputs[output_base + row] = acc;
}

// K-split batch-one Q4_K matvec for NARROW outputs: one thread per row
// reaches occupancy = row_count, and Iris Xe has ~5.4-10.7k lanes, so a
// 2560-row matvec (ffn_down, attn_out) runs the machine at 25-50% fill while
// streaming ~40% of the per-token bytes. Split the k dimension across
// SV_GroupID.y (batch_count carries the split count S): split s accumulates
// blocks s, s+S, s+2S..., writing partials[output_base + s*row_count + row];
// main_ksplit_reduce then sums the S partials per row in fixed s order.
// Accumulation order differs from the single-thread kernel, so results are
// 4e-3-gated (answer parity), not byte-identical - the same trade llama.cpp
// makes in its k-split mul_mat_vec. Same per-block body as main_q4k_direct.
[numthreads(64, 1, 1)]
void main_q4k_ksplit(uint3 group_id : SV_GroupID, uint thread_index : SV_GroupIndex) {
  const uint row = group_id.x * 64u + thread_index;
  if (row >= row_count) {
    return;
  }
  const uint split = group_id.y;
  const uint split_count = batch_count;
  float acc = 0.0f;
  const uint blocks_per_row = row_width >> 8u;
  [loop]
  for (uint block = split; block < blocks_per_row; block += split_count) {
    const uint input_base = block << 8u;
    const uint block_byte = (block * row_count + row) * 144u;
    const uint d_words = weights.Load(block_byte);
    const float d = f16tof32(d_words & 0xFFFFu);
    const float dmin = f16tof32(d_words >> 16u);
    const uint scales_byte_base = block_byte + 4u;
    const uint3 scale_words = weights.Load3(scales_byte_base);
    float scale_d[8];
    float scale_m[8];
    [unroll]
    for (uint j = 0; j < 8u; j += 1u) {
      scale_d[j] = d * (float)scale_k4_words(j, scale_words);
      scale_m[j] = dmin * (float)min_k4_words(j, scale_words);
    }
    [loop]
    for (uint quant_word4 = 0; quant_word4 < 8u; quant_word4 += 1u) {
      const uint4 packed_words = weights.Load4(block_byte + 16u + quant_word4 * 16u);
      [unroll]
      for (uint sub_dword = 0; sub_dword < 4u; sub_dword += 1u) {
        const uint packed4 = sub_dword == 0u ? packed_words.x
          : sub_dword == 1u ? packed_words.y
          : sub_dword == 2u ? packed_words.z
          : packed_words.w;
        [unroll]
        for (uint lane = 0; lane < 4u; lane += 1u) {
          const uint quant_index = (quant_word4 * 4u + sub_dword) * 4u + lane;
          const uint chunk_id = quant_index >> 5u;
          const uint item = quant_index & 31u;
          const uint quant_byte = byte_at(packed4, lane);
          const uint low_element = input_base + chunk_id * 64u + item;
          const float weight_low = scale_d[chunk_id * 2u] * (float)(quant_byte & 0x0Fu)
            - scale_m[chunk_id * 2u];
          const float weight_high = scale_d[chunk_id * 2u + 1u] * (float)(quant_byte >> 4u)
            - scale_m[chunk_id * 2u + 1u];
          acc += weight_low * inputs[low_element]
            + weight_high * inputs[low_element + 32u];
        }
      }
    }
  }
  outputs[output_base + split * row_count + row] = acc;
}

// Sums the S k-split partials per row (fixed s order, deterministic).
// weight_layout carries the float offset of the partials inside t1.
[numthreads(64, 1, 1)]
void main_ksplit_reduce(uint3 group_id : SV_GroupID, uint thread_index : SV_GroupIndex) {
  const uint row = group_id.x * 64u + thread_index;
  if (row >= row_count) {
    return;
  }
  const uint split_count = batch_count;
  const uint partial_base = weight_layout;
  float acc = 0.0f;
  [loop]
  for (uint split = 0; split < split_count; split += 1u) {
    acc += inputs[partial_base + split * row_count + row];
  }
  outputs[output_base + row] = acc;
}

// Two-blocks-per-barrier batch-one Q4_K matvec (roadmap B2): stage 512
// input elements per cooperative load instead of 256, then let each thread
// consume BOTH blocks before the next sync. Halves the barrier count per
// row and interleaves two independent weight-load chains between barriers.
// Per-element math and accumulation order match main_q4k exactly (block N
// is fully accumulated before block N+1), so results are bit-comparable.
// Batch-one only; input_transform is NOT supported.
groupshared float input_pair_tile[512];

void load_input_pair(uint thread_index, uint element_base) {
  [unroll]
  for (uint slot = 0; slot < 512u / k_rows_per_group; slot += 1u) {
    const uint flat = slot * k_rows_per_group + thread_index;
    const uint element = element_base + flat;
    // Clamp BEFORE the load (speculative out-of-bounds root-descriptor reads
    // are undefined behavior); only the stored value is masked.
    const uint safe_element = min(element, row_width - 1u);
    const float value = inputs[safe_element];
    input_pair_tile[flat] = element < row_width ? value : 0.0f;
  }
}

[numthreads(k_rows_per_group, 1, 1)]
void main_q4k_pair(uint3 group_id : SV_GroupID, uint thread_index : SV_GroupIndex) {
  const uint row = group_id.x * k_rows_per_group + thread_index;
  float acc = 0.0f;
  const uint blocks_per_row = row_width >> 8u;

  [loop]
  for (uint block = 0; block < blocks_per_row; block += 2u) {
    load_input_pair(thread_index, block << 8u);
    GroupMemoryBarrierWithGroupSync();

    if (row < row_count) {
      const uint pair_count = min(2u, blocks_per_row - block);
      [loop]
      for (uint sub = 0; sub < pair_count; sub += 1u) {
        const uint tile_base = sub << 8u;
        const uint block_byte = ((block + sub) * row_count + row) * 144u;
        const uint d_words = weights.Load(block_byte);
        const float d = f16tof32(d_words & 0xFFFFu);
        const float dmin = f16tof32(d_words >> 16u);
        const uint scales_byte_base = block_byte + 4u;
        const uint3 scale_words = weights.Load3(scales_byte_base);
        float scale_d[8];
        float scale_m[8];
        [unroll]
        for (uint j = 0; j < 8u; j += 1u) {
          scale_d[j] = d * (float)scale_k4_words(j, scale_words);
          scale_m[j] = dmin * (float)min_k4_words(j, scale_words);
        }
        [loop]
        for (uint quant_word4 = 0; quant_word4 < 8u; quant_word4 += 1u) {
          const uint4 packed_words = weights.Load4(block_byte + 16u + quant_word4 * 16u);
          [unroll]
          for (uint sub_dword = 0; sub_dword < 4u; sub_dword += 1u) {
            const uint packed4 = sub_dword == 0u ? packed_words.x
              : sub_dword == 1u ? packed_words.y
              : sub_dword == 2u ? packed_words.z
              : packed_words.w;
            [unroll]
            for (uint lane = 0; lane < 4u; lane += 1u) {
              const uint quant_index = (quant_word4 * 4u + sub_dword) * 4u + lane;
              const uint chunk_id = quant_index >> 5u;
              const uint item = quant_index & 31u;
              const uint quant_byte = byte_at(packed4, lane);
              const uint low_element = tile_base + chunk_id * 64u + item;
              const float weight_low = scale_d[chunk_id * 2u] * (float)(quant_byte & 0x0Fu)
                - scale_m[chunk_id * 2u];
              const float weight_high = scale_d[chunk_id * 2u + 1u] * (float)(quant_byte >> 4u)
                - scale_m[chunk_id * 2u + 1u];
              acc += weight_low * input_pair_tile[low_element]
                + weight_high * input_pair_tile[low_element + 32u];
            }
          }
        }
      }
    }
    GroupMemoryBarrierWithGroupSync();
  }

  if (row < row_count) {
    outputs[output_base + row] = acc;
  }
}

// Batch-one Q6_K matvec with DIRECT input reads: the Q6_K counterpart of
// main_q4k_direct above - no groupshared input tile and no per-block
// barriers; decode matvecs broadcast the same input vector to every thread
// in the group, so L1 serves the reads without staging. Same per-element
// math and accumulation order as main_q6k, so results are bit-comparable.
// input_transform is NOT supported; C++ routes only plain batch-one matvecs
// here.
[numthreads(k_rows_per_group, 1, 1)]
void main_q6k_direct(uint3 group_id : SV_GroupID, uint thread_index : SV_GroupIndex) {
  const uint row = group_id.x * k_rows_per_group + thread_index;
  if (row >= row_count) {
    return;
  }
  float acc = 0.0f;
  const uint blocks_per_row = row_width >> 8u;
  [loop]
  for (uint block = 0; block < blocks_per_row; block += 1u) {
    const uint input_base = block << 8u;
    const uint block_byte = (block * row_count + row) * 212u;
    const float d = f16tof32(weights.Load(block_byte + 208u) & 0xFFFFu);
    const uint4 scale_words = weights.Load4(block_byte + 192u);

    [loop]
    for (uint half_index = 0; half_index < 2u; half_index += 1u) {
      const uint chunk = half_index * 128u;
      const uint ql_base = block_byte + chunk / 2u;
      const uint qh_base = block_byte + 128u + chunk / 4u;
      const uint scale_base = chunk / 16u;

      [loop]
      for (uint dword_group = 0; dword_group < 2u; dword_group += 1u) {
        const uint4 ql_low4 = weights.Load4(ql_base + dword_group * 16u);
        const uint4 ql_high4 = weights.Load4(ql_base + 32u + dword_group * 16u);
        const uint4 qh_word4 = weights.Load4(qh_base + dword_group * 16u);
        [unroll]
        for (uint sub = 0; sub < 4u; sub += 1u) {
          const uint dword_index = dword_group * 4u + sub;
          const uint ql_low = sub == 0u ? ql_low4.x : sub == 1u ? ql_low4.y : sub == 2u ? ql_low4.z : ql_low4.w;
          const uint ql_high = sub == 0u ? ql_high4.x : sub == 1u ? ql_high4.y : sub == 2u ? ql_high4.z : ql_high4.w;
          const uint qh_word = sub == 0u ? qh_word4.x : sub == 1u ? qh_word4.y : sub == 2u ? qh_word4.z : qh_word4.w;
          [unroll]
          for (uint lane = 0; lane < 4u; lane += 1u) {
            const uint item = dword_index * 4u + lane;
            const uint ql_low_byte = byte_at(ql_low, lane);
            const uint ql_high_byte = byte_at(ql_high, lane);
            const uint qh_byte = byte_at(qh_word, lane);
            const uint scale_group = item >> 4u;
            const int q1 = (int)((ql_low_byte & 0x0Fu) | (((qh_byte >> 0u) & 3u) << 4u)) - 32;
            const int q2 = (int)((ql_high_byte & 0x0Fu) | (((qh_byte >> 2u) & 3u) << 4u)) - 32;
            const int q3 = (int)((ql_low_byte >> 4u) | (((qh_byte >> 4u) & 3u) << 4u)) - 32;
            const int q4 = (int)((ql_high_byte >> 4u) | (((qh_byte >> 6u) & 3u) << 4u)) - 32;
            const float w1 = d * (float)signed_scale16(scale_words, scale_base + scale_group + 0u) * (float)q1;
            const float w2 = d * (float)signed_scale16(scale_words, scale_base + scale_group + 2u) * (float)q2;
            const float w3 = d * (float)signed_scale16(scale_words, scale_base + scale_group + 4u) * (float)q3;
            const float w4 = d * (float)signed_scale16(scale_words, scale_base + scale_group + 6u) * (float)q4;
            const uint element0 = input_base + chunk + item;
            acc += w1 * inputs[element0]
              + w2 * inputs[element0 + 32u]
              + w3 * inputs[element0 + 64u]
              + w4 * inputs[element0 + 96u];
          }
        }
      }
    }
  }
  outputs[output_base + row] = acc;
}

// Cooperative single-vector (batch-one) Q4_K GEMV: ONE workgroup of 64 lanes
// computes ONE output row. The lanes cooperatively read each 144-byte block
// (adjacent lanes -> adjacent bytes -> coalesced 64-byte transactions, unlike
// the one-thread-per-row kernel whose 64 threads stride 144 bytes apart), and
// a groupshared tree reduction sums the per-lane partials. Same dequant math
// as main_q4k, so results are bit-comparable. Dispatch row_count groups.
groupshared float gemv_partials[64];

[numthreads(64, 1, 1)]
void main_q4k_gemv(uint3 group_id : SV_GroupID, uint lane : SV_GroupIndex) {
  const uint row = group_id.x;
  const uint blocks_per_row = row_width >> 8u;
  float partial = 0.0f;
  if (row < row_count) {
    [loop]
    for (uint block = 0; block < blocks_per_row; block += 1u) {
      const uint block_byte = (block * row_count + row) * 144u;
      const uint d_words = weights.Load(block_byte);
      const float d = f16tof32(d_words & 0xFFFFu);
      const float dmin = f16tof32(d_words >> 16u);
      const uint scales_byte_base = block_byte + 4u;
      const uint input_base = block << 8u;
      float scale_d[8];
      float scale_m[8];
      [unroll]
      for (uint j = 0; j < 8u; j += 1u) {
        scale_d[j] = d * (float)scale_k4(j, scales_byte_base);
        scale_m[j] = dmin * (float)min_k4(j, scales_byte_base);
      }
      // Lane l handles quant bytes qi = l and qi = l + 64 (contiguous across
      // the 64 lanes -> coalesced 64-byte transactions). This loop MUST stay
      // [loop], not [unroll]: FXC miscompiles the unrolled two-iteration form
      // here (wrong per-row totals), verified by the coop selftest.
      [loop]
      for (uint half_index = 0; half_index < 2u; half_index += 1u) {
        const uint quant_index = lane + half_index * 64u;
        const uint chunk_id = quant_index >> 5u;
        const uint item = quant_index & 31u;
        const uint quant_byte = buffer_byte(block_byte + 16u + quant_index);
        const uint low_element = chunk_id * 64u + item;
        const float weight_low = scale_d[chunk_id * 2u] * (float)(quant_byte & 0x0Fu)
          - scale_m[chunk_id * 2u];
        const float weight_high = scale_d[chunk_id * 2u + 1u] * (float)(quant_byte >> 4u)
          - scale_m[chunk_id * 2u + 1u];
        partial += weight_low * inputs[input_base + low_element]
          + weight_high * inputs[input_base + low_element + 32u];
      }
    }
  }
  // Lane 0 sums the 64 partials serially. A groupshared tree reduction here
  // was miscompiled by FXC under this kernel's register pressure (verified:
  // per-lane partials summed correctly on the CPU but the tree gave a wrong
  // total); the serial sum over 64 elements is negligible next to the weight
  // reads and is bit-stable.
  gemv_partials[lane] = partial;
  GroupMemoryBarrierWithGroupSync();
  if (lane == 0u && row < row_count) {
    float sum = 0.0f;
    [loop]
    for (uint index = 0; index < 64u; index += 1u) {
      sum += gemv_partials[index];
    }
    outputs[output_base + row] = sum;
  }
}

[numthreads(k_rows_per_group, 1, 1)]
void main_q5k(uint3 group_id : SV_GroupID, uint thread_index : SV_GroupIndex) {
  const uint row = group_id.x * k_rows_per_group + thread_index;
  const uint batch_start = group_id.y * k_batch_tile;

  float acc[k_batch_tile];
  [unroll]
  for (uint init = 0; init < k_batch_tile; init += 1u) {
    acc[init] = 0.0f;
  }

  const uint blocks_per_row = row_width >> 8u;

  [loop]
  for (uint block = 0; block < blocks_per_row; block += 1u) {
    load_input_tile(thread_index, batch_start, block << 8u);
    GroupMemoryBarrierWithGroupSync();

    if (row < row_count) {
      // Q5_K uses the same 6-bit packed scale/min layout as Q4_K plus a
      // 32-byte high-bit plane. Blocks are 176 bytes and stored column-major
      // so 64 adjacent rows read one contiguous stream:
      //   [0..1] d, [2..3] dmin, [4..15] packed scales,
      //   [16..47] qh (one high bit per weight), [48..175] qs (low nibbles).
      const uint block_byte = (block * row_count + row) * 176u;
      const uint d_words = weights.Load(block_byte);
      const float d = f16tof32(d_words & 0xFFFFu);
      const float dmin = f16tof32(d_words >> 16u);
      const uint scales_byte_base = block_byte + 4u;
      const uint qh_byte_base = block_byte + 16u;
      const uint qs_byte_base = block_byte + 48u;

      // Same Load3-once optimization as main_q4k: the 12 packed scale bytes
      // sit at [4..15], identical layout to Q4_K's scale block.
      const uint3 scale_words = weights.Load3(scales_byte_base);
      float scale_d[8];
      float scale_m[8];
      [unroll]
      for (uint j = 0; j < 8u; j += 1u) {
        scale_d[j] = d * (float)scale_k4_words(j, scale_words);
        scale_m[j] = dmin * (float)min_k4_words(j, scale_words);
      }

      // qh_item_base below cycles through all 8 dword-aligned offsets of the
      // 32-byte qh region ONCE PER CHUNK (4 chunks/block), so the original
      // per-iteration weights.Load(qh_byte_base + qh_item_base) re-read the
      // entire qh region from scratch 4 times over. Load it ONCE (2 Load4
      // calls) into a small array indexed by dword instead.
      uint qh_words[8];
      {
        const uint4 qh_low4 = weights.Load4(qh_byte_base);
        const uint4 qh_high4 = weights.Load4(qh_byte_base + 16u);
        qh_words[0] = qh_low4.x;
        qh_words[1] = qh_low4.y;
        qh_words[2] = qh_low4.z;
        qh_words[3] = qh_low4.w;
        qh_words[4] = qh_high4.x;
        qh_words[5] = qh_high4.y;
        qh_words[6] = qh_high4.z;
        qh_words[7] = qh_high4.w;
      }

      // qs (128 bytes / 32 dwords) widened the same way as main_q4k: 8
      // Load4 calls instead of 32 scalar Loads for identical bytes.
      [loop]
      for (uint quant_word4 = 0; quant_word4 < 8u; quant_word4 += 1u) {
        const uint4 packed_words = weights.Load4(qs_byte_base + quant_word4 * 16u);
        [unroll]
        for (uint sub_dword = 0; sub_dword < 4u; sub_dword += 1u) {
          const uint quant_dword = quant_word4 * 4u + sub_dword;
          const uint packed4 = sub_dword == 0u ? packed_words.x
            : sub_dword == 1u ? packed_words.y
            : sub_dword == 2u ? packed_words.z
            : packed_words.w;
          // The four lanes of this dword share one aligned qh dword; item
          // and its high-bit byte are contiguous within a 32-wide sub-block.
          const uint qh_item_base = (quant_dword * 4u) & 31u;
          const uint qh_word = qh_words[qh_item_base >> 2u];
          [unroll]
          for (uint lane = 0; lane < 4u; lane += 1u) {
            const uint quant_index = quant_dword * 4u + lane;
            const uint chunk_id = quant_index >> 5u;   // group 0..3
            const uint item = quant_index & 31u;       // element within group
            const uint quant_byte = byte_at(packed4, lane);
            const uint qh_byte = byte_at(qh_word, lane);
            const uint high_low = (qh_byte >> (chunk_id * 2u)) & 1u;
            const uint high_high = (qh_byte >> (chunk_id * 2u + 1u)) & 1u;
            const uint low_element = chunk_id * 64u + item;
            const float weight_low = scale_d[chunk_id * 2u]
                * (float)((quant_byte & 0x0Fu) + (high_low << 4u))
              - scale_m[chunk_id * 2u];
            const float weight_high = scale_d[chunk_id * 2u + 1u]
                * (float)((quant_byte >> 4u) + (high_high << 4u))
              - scale_m[chunk_id * 2u + 1u];
            [unroll]
            for (uint b = 0; b < k_batch_tile; b += 1u) {
              const uint tile_base = b * k_tile_elements;
              acc[b] += weight_low * input_tile[tile_base + low_element]
                + weight_high * input_tile[tile_base + low_element + 32u];
            }
          }
        }
      }
    }
    GroupMemoryBarrierWithGroupSync();
  }

  store_outputs(row, batch_start, acc);
}

[numthreads(k_rows_per_group, 1, 1)]
void main_q6k(uint3 group_id : SV_GroupID, uint thread_index : SV_GroupIndex) {
  const uint row = group_id.x * k_rows_per_group + thread_index;
  const uint batch_start = group_id.y * k_batch_tile;

  float acc[k_batch_tile];
  [unroll]
  for (uint init = 0; init < k_batch_tile; init += 1u) {
    acc[init] = 0.0f;
  }

  const uint blocks_per_row = row_width >> 8u;

  [loop]
  for (uint block = 0; block < blocks_per_row; block += 1u) {
    load_input_tile(thread_index, batch_start, block << 8u);
    GroupMemoryBarrierWithGroupSync();

    if (row < row_count) {
      const uint block_byte = (block * row_count + row) * 212u;
      const float d = f16tof32(weights.Load(block_byte + 208u) & 0xFFFFu);
      // The 16 signed scale bytes are 4 contiguous dwords - one Load4 instead
      // of four separate 32-bit Loads (same bytes, same order).
      const uint4 scale_words = weights.Load4(block_byte + 192u);

      [loop]
      for (uint half_index = 0; half_index < 2u; half_index += 1u) {
        const uint chunk = half_index * 128u;
        const uint ql_base = block_byte + chunk / 2u;
        const uint qh_base = block_byte + 128u + chunk / 4u;
        const uint scale_base = chunk / 16u;

        // Each of the three 32-byte streams (ql_low, ql_high, qh) fits in two
        // Load4 (128-bit) calls instead of eight 32-bit Loads: same total
        // bytes and identical per-element math, fewer wider transactions.
        [loop]
        for (uint dword_group = 0; dword_group < 2u; dword_group += 1u) {
          const uint4 ql_low4 = weights.Load4(ql_base + dword_group * 16u);
          const uint4 ql_high4 = weights.Load4(ql_base + 32u + dword_group * 16u);
          const uint4 qh_word4 = weights.Load4(qh_base + dword_group * 16u);
          [unroll]
          for (uint sub = 0; sub < 4u; sub += 1u) {
            const uint dword_index = dword_group * 4u + sub;
            const uint ql_low = sub == 0u ? ql_low4.x : sub == 1u ? ql_low4.y : sub == 2u ? ql_low4.z : ql_low4.w;
            const uint ql_high = sub == 0u ? ql_high4.x : sub == 1u ? ql_high4.y : sub == 2u ? ql_high4.z : ql_high4.w;
            const uint qh_word = sub == 0u ? qh_word4.x : sub == 1u ? qh_word4.y : sub == 2u ? qh_word4.z : qh_word4.w;
            [unroll]
            for (uint lane = 0; lane < 4u; lane += 1u) {
              const uint item = dword_index * 4u + lane;
              const uint ql_low_byte = byte_at(ql_low, lane);
              const uint ql_high_byte = byte_at(ql_high, lane);
              const uint qh_byte = byte_at(qh_word, lane);
              const uint scale_group = item >> 4u;
              const int q1 = (int)((ql_low_byte & 0x0Fu) | (((qh_byte >> 0u) & 3u) << 4u)) - 32;
              const int q2 = (int)((ql_high_byte & 0x0Fu) | (((qh_byte >> 2u) & 3u) << 4u)) - 32;
              const int q3 = (int)((ql_low_byte >> 4u) | (((qh_byte >> 4u) & 3u) << 4u)) - 32;
              const int q4 = (int)((ql_high_byte >> 4u) | (((qh_byte >> 6u) & 3u) << 4u)) - 32;
              const float w1 = d * (float)signed_scale16(scale_words, scale_base + scale_group + 0u) * (float)q1;
              const float w2 = d * (float)signed_scale16(scale_words, scale_base + scale_group + 2u) * (float)q2;
              const float w3 = d * (float)signed_scale16(scale_words, scale_base + scale_group + 4u) * (float)q3;
              const float w4 = d * (float)signed_scale16(scale_words, scale_base + scale_group + 6u) * (float)q4;
              const uint element0 = chunk + item;
              [unroll]
              for (uint b = 0; b < k_batch_tile; b += 1u) {
                const uint tile_base = b * k_tile_elements;
                acc[b] += w1 * input_tile[tile_base + element0]
                  + w2 * input_tile[tile_base + element0 + 32u]
                  + w3 * input_tile[tile_base + element0 + 64u]
                  + w4 * input_tile[tile_base + element0 + 96u];
              }
            }
          }
        }
      }
    }
    GroupMemoryBarrierWithGroupSync();
  }

  store_outputs(row, batch_start, acc);
}

// 2D register blocking for Q6_K - the transform main_q4k_rowblock applies, on
// the format that never got it.
//
// Measured motivation: at llama-3.2-3b's FFN shape (3072x8192, batch 269) the
// Q6_K down projection runs 82-88 GFLOP/s against gate/up Q4_K's 202-229,
// because rowblock_pipeline_for declines every format except Q4_K. That single
// matmul is 65% of the FFN chain's time for 33% of its FLOPs, and the chain is
// ~48% of prefill.
//
// Same trick and same reason as the Q4_K kernel: main_q6k issues 4 LDS reads
// per 4 FMAs, this issues 4 per 4R FMAs by hoisting them out of the row loop.
// Rows are strided by k_rows_per_group so every load round stays one contiguous
// stream over the column-major blocks.
//
// The unroll budget is TIGHTER here than for Q4_K. Each lane iteration expands
// to 4 FMAs per (row, batch) instead of 2, so Q6_K at R * tile = 16 is already
// the straight-line size of Q4_K at its shipped R * tile = 32. R defaults to 2
// for that reason - see the cliff note in main_q4k_rowblock before raising it.
[numthreads(k_rows_per_group, 1, 1)]
void main_q6k_rowblock(uint3 group_id : SV_GroupID, uint thread_index : SV_GroupIndex) {
  const uint group_row_base = group_id.x * (k_rows_per_group * k_rows_per_thread);
  const uint batch_start = group_id.y * k_batch_tile;

  uint rows[k_rows_per_thread];
  [unroll]
  for (uint r_init = 0; r_init < k_rows_per_thread; r_init += 1u) {
    rows[r_init] = group_row_base + r_init * k_rows_per_group + thread_index;
  }

  float acc[k_rows_per_thread * k_batch_tile];
  [unroll]
  for (uint init = 0; init < k_rows_per_thread * k_batch_tile; init += 1u) {
    acc[init] = 0.0f;
  }

  const uint blocks_per_row = row_width >> 8u;

  [loop]
  for (uint block = 0; block < blocks_per_row; block += 1u) {
    load_input_tile(thread_index, batch_start, block << 8u);
    GroupMemoryBarrierWithGroupSync();

    uint row_block_byte[k_rows_per_thread];
    float row_d[k_rows_per_thread];
    uint4 row_scale_words[k_rows_per_thread];
    [unroll]
    for (uint rs = 0; rs < k_rows_per_thread; rs += 1u) {
      // Clamp before loading, as main_q4k_rowblock does: an out-of-range
      // root-descriptor read is undefined even when the value is discarded, so
      // inactive rows read row 0 and are dropped at store time.
      const uint safe_row = min(rows[rs], row_count - 1u);
      const uint block_byte = (block * row_count + safe_row) * 212u;
      row_block_byte[rs] = block_byte;
      row_d[rs] = f16tof32(weights.Load(block_byte + 208u) & 0xFFFFu);
      row_scale_words[rs] = weights.Load4(block_byte + 192u);
    }

    [loop]
    for (uint half_index = 0; half_index < 2u; half_index += 1u) {
      const uint chunk = half_index * 128u;
      const uint scale_base = chunk / 16u;

      [loop]
      for (uint dword_group = 0; dword_group < 2u; dword_group += 1u) {
        uint4 ql_low4[k_rows_per_thread];
        uint4 ql_high4[k_rows_per_thread];
        uint4 qh_word4[k_rows_per_thread];
        [unroll]
        for (uint rl = 0; rl < k_rows_per_thread; rl += 1u) {
          const uint ql_base = row_block_byte[rl] + chunk / 2u;
          const uint qh_base = row_block_byte[rl] + 128u + chunk / 4u;
          ql_low4[rl] = weights.Load4(ql_base + dword_group * 16u);
          ql_high4[rl] = weights.Load4(ql_base + 32u + dword_group * 16u);
          qh_word4[rl] = weights.Load4(qh_base + dword_group * 16u);
        }

        [unroll]
        for (uint sub = 0; sub < 4u; sub += 1u) {
          const uint dword_index = dword_group * 4u + sub;
          [unroll]
          for (uint lane = 0; lane < 4u; lane += 1u) {
            const uint item = dword_index * 4u + lane;
            const uint scale_group = item >> 4u;
            const uint element0 = chunk + item;

            float w1[k_rows_per_thread];
            float w2[k_rows_per_thread];
            float w3[k_rows_per_thread];
            float w4[k_rows_per_thread];
            [unroll]
            for (uint rw = 0; rw < k_rows_per_thread; rw += 1u) {
              const uint4 qlo = ql_low4[rw];
              const uint4 qhi = ql_high4[rw];
              const uint4 qhw = qh_word4[rw];
              const uint ql_low = sub == 0u ? qlo.x : sub == 1u ? qlo.y : sub == 2u ? qlo.z : qlo.w;
              const uint ql_high = sub == 0u ? qhi.x : sub == 1u ? qhi.y : sub == 2u ? qhi.z : qhi.w;
              const uint qh_word = sub == 0u ? qhw.x : sub == 1u ? qhw.y : sub == 2u ? qhw.z : qhw.w;
              const uint ql_low_byte = byte_at(ql_low, lane);
              const uint ql_high_byte = byte_at(ql_high, lane);
              const uint qh_byte = byte_at(qh_word, lane);
              const int q1 = (int)((ql_low_byte & 0x0Fu) | (((qh_byte >> 0u) & 3u) << 4u)) - 32;
              const int q2 = (int)((ql_high_byte & 0x0Fu) | (((qh_byte >> 2u) & 3u) << 4u)) - 32;
              const int q3 = (int)((ql_low_byte >> 4u) | (((qh_byte >> 4u) & 3u) << 4u)) - 32;
              const int q4 = (int)((ql_high_byte >> 4u) | (((qh_byte >> 6u) & 3u) << 4u)) - 32;
              const float d = row_d[rw];
              const uint4 sw = row_scale_words[rw];
              w1[rw] = d * (float)signed_scale16(sw, scale_base + scale_group + 0u) * (float)q1;
              w2[rw] = d * (float)signed_scale16(sw, scale_base + scale_group + 2u) * (float)q2;
              w3[rw] = d * (float)signed_scale16(sw, scale_base + scale_group + 4u) * (float)q3;
              w4[rw] = d * (float)signed_scale16(sw, scale_base + scale_group + 6u) * (float)q4;
            }

            // The four LDS reads are the whole point: hoisted out of the row
            // loop below and reused across all R rows.
            [unroll]
            for (uint b = 0; b < k_batch_tile; b += 1u) {
              const uint tile_base = b * k_tile_elements;
              const float in0 = input_tile[tile_base + element0];
              const float in1 = input_tile[tile_base + element0 + 32u];
              const float in2 = input_tile[tile_base + element0 + 64u];
              const float in3 = input_tile[tile_base + element0 + 96u];
              [unroll]
              for (uint ra = 0; ra < k_rows_per_thread; ra += 1u) {
                acc[ra * k_batch_tile + b] += w1[ra] * in0 + w2[ra] * in1
                  + w3[ra] * in2 + w4[ra] * in3;
              }
            }
          }
        }
      }
    }
    GroupMemoryBarrierWithGroupSync();
  }

  [unroll]
  for (uint ro = 0; ro < k_rows_per_thread; ro += 1u) {
    const uint row = rows[ro];
    if (row < row_count) {
      [unroll]
      for (uint out_batch = 0; out_batch < k_batch_tile; out_batch += 1u) {
        const uint batch = batch_start + out_batch;
        if (batch < batch_count) {
          outputs[output_base + batch * row_count + row] = acc[ro * k_batch_tile + out_batch];
        }
      }
    }
  }
}

[numthreads(k_rows_per_group, 1, 1)]
void main_q8_0(uint3 group_id : SV_GroupID, uint thread_index : SV_GroupIndex) {
  const uint row = group_id.x * k_rows_per_group + thread_index;
  const uint batch_start = group_id.y * k_batch_tile;

  float acc[k_batch_tile];
  [unroll]
  for (uint init = 0; init < k_batch_tile; init += 1u) {
    acc[init] = 0.0f;
  }

  // 32-element blocks staged to 36 bytes: [0..3] d as F32, [4..35] 32 int8.
  const uint blocks_per_row = row_width / 32u;
  const uint tile_count = (row_width + k_tile_elements - 1u) / k_tile_elements;

  [loop]
  for (uint tile = 0; tile < tile_count; tile += 1u) {
    load_input_tile(thread_index, batch_start, tile * k_tile_elements);
    GroupMemoryBarrierWithGroupSync();

    if (row < row_count) {
      const uint first_block = tile * 8u;  // 256 / 32
      const uint tile_blocks = min(8u, blocks_per_row - first_block);
      [loop]
      for (uint sub = 0; sub < tile_blocks; sub += 1u) {
        const uint block_column = first_block + sub;
        const uint block_byte = (block_column * row_count + row) * 36u;
        const float d = asfloat(weights.Load(block_byte));
        const uint tile_element_base = sub * 32u;
        // 32 quant bytes per block fit in two Load4 (128-bit) calls instead
        // of eight 32-bit Loads.
        [unroll]
        for (uint quad_group = 0; quad_group < 2u; quad_group += 1u) {
          const uint4 packed4 = weights.Load4(block_byte + 4u + quad_group * 16u);
          [unroll]
          for (uint sub_quad = 0; sub_quad < 4u; sub_quad += 1u) {
            const uint quad = quad_group * 4u + sub_quad;
            const uint packed = sub_quad == 0u ? packed4.x
              : sub_quad == 1u ? packed4.y
              : sub_quad == 2u ? packed4.z
              : packed4.w;
            [unroll]
            for (uint lane = 0; lane < 4u; lane += 1u) {
              const uint raw = byte_at(packed, lane);
              const int q = (int)(raw << 24u) >> 24;  // sign-extend int8
              const float weight_value = d * (float)q;
              const uint element = tile_element_base + quad * 4u + lane;
              [unroll]
              for (uint b = 0; b < k_batch_tile; b += 1u) {
                acc[b] += weight_value * input_tile[b * k_tile_elements + element];
              }
            }
          }
        }
      }
    }
    GroupMemoryBarrierWithGroupSync();
  }

  store_outputs(row, batch_start, acc);
}

[numthreads(k_rows_per_group, 1, 1)]
void main_q4_0(uint3 group_id : SV_GroupID, uint thread_index : SV_GroupIndex) {
  const uint row = group_id.x * k_rows_per_group + thread_index;
  const uint batch_start = group_id.y * k_batch_tile;

  float acc[k_batch_tile];
  [unroll]
  for (uint init = 0; init < k_batch_tile; init += 1u) {
    acc[init] = 0.0f;
  }

  // 32-element blocks staged to 20 bytes: [0..3] d as F32, [4..19] 16 nibble
  // bytes. Low nibble -> element j, high nibble -> element j+16, both minus 8.
  const uint blocks_per_row = row_width / 32u;
  const uint tile_count = (row_width + k_tile_elements - 1u) / k_tile_elements;

  [loop]
  for (uint tile = 0; tile < tile_count; tile += 1u) {
    load_input_tile(thread_index, batch_start, tile * k_tile_elements);
    GroupMemoryBarrierWithGroupSync();

    if (row < row_count) {
      const uint first_block = tile * 8u;
      const uint tile_blocks = min(8u, blocks_per_row - first_block);
      [loop]
      for (uint sub = 0; sub < tile_blocks; sub += 1u) {
        const uint block_column = first_block + sub;
        const uint block_byte = (block_column * row_count + row) * 20u;
        const float d = asfloat(weights.Load(block_byte));
        const uint tile_element_base = sub * 32u;
        // All 16 quant bytes per block fit in one Load4 (128-bit) call
        // instead of four 32-bit Loads.
        const uint4 packed4 = weights.Load4(block_byte + 4u);
        [unroll]
        for (uint quad = 0; quad < 4u; quad += 1u) {
          const uint packed = quad == 0u ? packed4.x
            : quad == 1u ? packed4.y
            : quad == 2u ? packed4.z
            : packed4.w;
          [unroll]
          for (uint lane = 0; lane < 4u; lane += 1u) {
            const uint raw = byte_at(packed, lane);
            const int x_low = (int)(raw & 0x0Fu) - 8;
            const int x_high = (int)(raw >> 4u) - 8;
            const uint element_low = tile_element_base + quad * 4u + lane;
            const float weight_low = d * (float)x_low;
            const float weight_high = d * (float)x_high;
            [unroll]
            for (uint b = 0; b < k_batch_tile; b += 1u) {
              const uint tile_base = b * k_tile_elements;
              acc[b] += weight_low * input_tile[tile_base + element_low]
                + weight_high * input_tile[tile_base + element_low + 16u];
            }
          }
        }
      }
    }
    GroupMemoryBarrierWithGroupSync();
  }

  store_outputs(row, batch_start, acc);
}

// Weighted RMS-norm over ONE vector with an optional residual add, matching
// the CPU path: out[i] = x[i] * rsqrt(mean(x^2) + eps) * w[i] (+ add[i]).
// Constants mapping for this entry point:
//   row_width      vector width
//   row_count      x offset in t1 (floats)
//   batch_count    add offset in t1 (floats; used when flags bit0 set)
//   output_base    output offset in u0 (floats)
//   scalar_format  flags: bit0 = residual add, bit1 = output scale,
//                         bit2 = skip normalization (pure residual add)
//   input_stride   rms epsilon as float bits
//   weight_layout  norm-weight offset in t0 (floats, raw F32)
//   input_transform output scale as float bits
// All indices are clamped BEFORE loads (speculative-load safety).
// Pure streaming read: the memory-bandwidth ceiling with no dequantisation,
// no scales and no per-element arithmetic beyond an integer add. Answers the
// question every kernel optimisation on this machine depends on - whether the
// Q4_K matvec's achieved GB/s is the kernel's limit or the hardware's.
// Constants: row_width = number of uint4s to read, row_count = group count.
// Each thread strides the whole buffer and writes one word so the reads cannot
// be optimised away.
// batch_count selects the access pattern:
//   0 = CONTIGUOUS. Adjacent threads read adjacent 16-byte quads. This is what
//       the streaming ceiling measures.
//   n = STRIDED by n quads per thread. Thread t starts at t*n and walks
//       forward, so the threads of a wave are n*16 bytes apart at every step -
//       exactly what main_q4k does when each thread owns a row (n = the row's
//       quad count). Measuring both with one kernel isolates the access pattern
//       from every other difference between the two shaders.
[numthreads(256, 1, 1)]
void main_stream_read(uint3 group_id : SV_GroupID, uint thread_index : SV_GroupIndex) {
  const uint total_quads = row_width;
  const uint groups = row_count;
  const uint row_quads = batch_count;
  uint acc = 0u;
  if (row_quads == 0u) {
    const uint stride = groups * 256u;
    for (uint index = group_id.x * 256u + thread_index; index < total_quads; index += stride) {
      const uint4 chunk = weights.Load4(index * 16u);
      acc += chunk.x + chunk.y + chunk.z + chunk.w;
    }
  } else {
    // One "row" per thread, walked front to back: the matvec's pattern.
    const uint rows = total_quads / row_quads;
    const uint thread_id = group_id.x * 256u + thread_index;
    const uint thread_count = groups * 256u;
    for (uint row = thread_id; row < rows; row += thread_count) {
      const uint base = row * row_quads;
      for (uint offset = 0u; offset < row_quads; offset += 1u) {
        const uint4 chunk = weights.Load4((base + offset) * 16u);
        acc += chunk.x + chunk.y + chunk.z + chunk.w;
      }
    }
  }
  outputs[group_id.x * 256u + thread_index] = asfloat(acc);
}

// Sustained ALU ceiling: fused multiply-adds out of registers, no memory in the
// loop at all. This exists to answer a question the ratios could not - "13% of
// peak" was 222 GFLOP/s against a DATASHEET 1.6 TFLOP/s, and this machine shares
// one power budget between CPU and iGPU (the High-performance power plan measured
// 33% SLOWER for exactly that reason). A denominator the hardware cannot reach
// under its own thermal and power limits makes every percentage meaningless.
//
// Eight independent accumulators so the FMA pipeline stays fed rather than
// serialising on latency, unrolled eight deep: 64 FMAs per loop iteration.
[numthreads(256, 1, 1)]
void main_fma_peak(uint3 group_id : SV_GroupID, uint thread_index : SV_GroupIndex) {
  const uint iterations = row_width;
  const float multiplier = 1.0000001f;
  const float addend = 0.0000001f;
  float a0 = (float)thread_index + 1.0f;
  float a1 = a0 + 1.0f;
  float a2 = a0 + 2.0f;
  float a3 = a0 + 3.0f;
  float a4 = a0 + 4.0f;
  float a5 = a0 + 5.0f;
  float a6 = a0 + 6.0f;
  float a7 = a0 + 7.0f;
  [loop]
  for (uint i = 0; i < iterations; i += 1u) {
    [unroll]
    for (uint u = 0; u < 8u; u += 1u) {
      a0 = mad(a0, multiplier, addend);
      a1 = mad(a1, multiplier, addend);
      a2 = mad(a2, multiplier, addend);
      a3 = mad(a3, multiplier, addend);
      a4 = mad(a4, multiplier, addend);
      a5 = mad(a5, multiplier, addend);
      a6 = mad(a6, multiplier, addend);
      a7 = mad(a7, multiplier, addend);
    }
  }
  outputs[group_id.x * 256u + thread_index] =
    a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
}

groupshared float norm_partials[256];

[numthreads(256, 1, 1)]
void main_rms_norm(uint3 group_id : SV_GroupID, uint thread_index : SV_GroupIndex) {
  const uint width = row_width;
  // One Y group owns one batch row. Existing single-vector callers dispatch
  // Y=1, so group_id.y=0 preserves their byte-for-byte addressing.
  const uint batch_offset = group_id.y * width;
  const uint x_offset = row_count + batch_offset;
  const uint add_offset = batch_count + batch_offset;
  const uint weight_offset = weight_layout;
  const float epsilon = asfloat(input_stride);
  const uint last = width - 1u;

  float sum_of_squares = 0.0f;
  for (uint index = thread_index; index < width; index += 256u) {
    const float value = inputs[x_offset + min(index, last)];
    sum_of_squares += value * value;
  }
  norm_partials[thread_index] = sum_of_squares;
  GroupMemoryBarrierWithGroupSync();
  [unroll]
  for (uint stride = 128u; stride > 0u; stride >>= 1u) {
    if (thread_index < stride) {
      norm_partials[thread_index] += norm_partials[thread_index + stride];
    }
    GroupMemoryBarrierWithGroupSync();
  }
  const float inverse_rms = rsqrt(norm_partials[0] / (float)width + epsilon);

  for (uint index = thread_index; index < width; index += 256u) {
    const uint safe_index = min(index, last);
    const float value = inputs[x_offset + safe_index];
    const float weight_value = asfloat(weights.Load((weight_offset + safe_index) * 4u));
    const float add_value = inputs[((scalar_format & 1u) != 0u ? add_offset : x_offset) + safe_index];
    // bit2 skips the norm entirely so this becomes a plain residual add.
    // Llama-family layers add the attention and FFN results straight back into
    // the hidden state with no normalization at those two points, while Gemma
    // normalizes first. Passing unit weights cannot express that: an RMS norm
    // with w=1 still divides by the RMS. Gemma paths never set bit2, so their
    // recorded graphs are unchanged bit for bit.
    float result = ((scalar_format & 4u) != 0u)
      ? value
      : value * inverse_rms * weight_value;
    if ((scalar_format & 1u) != 0u) {
      result += add_value;
    }
    if ((scalar_format & 2u) != 0u) {
      result *= asfloat(input_transform);
    }
    outputs[output_base + batch_offset + index] = result;
  }
}

// Two chained RMS norms in ONE dispatch: res = x*rms(x)*w1 + add, then
// f = res*rms(res)*w2 - the token graph's per-layer post-attention-residual
// and FFN-norm pair (recorder steps 2+3), which used to cost two dispatches
// and two UAV barriers per layer per token. res stays in groupshared f32
// between the halves, and both reductions replicate main_rms_norm's exact
// 256-partial + tree order, so BOTH outputs are byte-identical to the
// two-dispatch sequence. Constant aliasing (all 8 dwords):
//   row_width=width, row_count=x_offset, batch_count=add_offset,
//   weight_layout=w1_offset, scalar_format=w2_offset,
//   input_stride=asfloat(epsilon), output_base=res_offset,
//   input_transform=f_offset. Add is always enabled (step 2's semantics);
//   the width must fit the groupshared res mirror (4096).
groupshared float dual_norm_res[4096];

[numthreads(256, 1, 1)]
void main_rms_norm_dual(uint3 group_id : SV_GroupID, uint thread_index : SV_GroupIndex) {
  const uint width = row_width;
  const uint x_offset = row_count;
  const uint add_offset = batch_count;
  const uint w1_offset = weight_layout;
  const uint w2_offset = scalar_format;
  const uint res_offset = output_base;
  const uint f_offset = input_transform;
  const float epsilon = asfloat(input_stride);
  const uint last = width - 1u;

  float sum_of_squares = 0.0f;
  for (uint index = thread_index; index < width; index += 256u) {
    const float value = inputs[x_offset + min(index, last)];
    sum_of_squares += value * value;
  }
  norm_partials[thread_index] = sum_of_squares;
  GroupMemoryBarrierWithGroupSync();
  [unroll]
  for (uint stride = 128u; stride > 0u; stride >>= 1u) {
    if (thread_index < stride) {
      norm_partials[thread_index] += norm_partials[thread_index + stride];
    }
    GroupMemoryBarrierWithGroupSync();
  }
  const float inverse_rms = rsqrt(norm_partials[0] / (float)width + epsilon);

  for (uint index = thread_index; index < width; index += 256u) {
    const uint safe_index = min(index, last);
    const float value = inputs[x_offset + safe_index];
    const float weight_value = asfloat(weights.Load((w1_offset + safe_index) * 4u));
    const float add_value = inputs[add_offset + safe_index];
    const float result = value * inverse_rms * weight_value + add_value;
    if (index < width) {
      dual_norm_res[index] = result;
      outputs[res_offset + index] = result;
    }
  }
  GroupMemoryBarrierWithGroupSync();

  float sum_of_squares_res = 0.0f;
  for (uint index = thread_index; index < width; index += 256u) {
    const float value = dual_norm_res[min(index, last)];
    sum_of_squares_res += value * value;
  }
  norm_partials[thread_index] = sum_of_squares_res;
  GroupMemoryBarrierWithGroupSync();
  [unroll]
  for (uint stride = 128u; stride > 0u; stride >>= 1u) {
    if (thread_index < stride) {
      norm_partials[thread_index] += norm_partials[thread_index + stride];
    }
    GroupMemoryBarrierWithGroupSync();
  }
  const float inverse_rms_res = rsqrt(norm_partials[0] / (float)width + epsilon);

  for (uint index = thread_index; index < width; index += 256u) {
    const uint safe_index = min(index, last);
    const float value = dual_norm_res[safe_index];
    const float weight_value = asfloat(weights.Load((w2_offset + safe_index) * 4u));
    if (index < width) {
      outputs[f_offset + index] = value * inverse_rms_res * weight_value;
    }
  }
}

// Elementwise gelu(gate) * up over the FFN scratch buffer. t1 holds gate
// values at [0, row_count) and up values at [row_count, 2*row_count);
// row_count carries the total element count for this launch.
[numthreads(256, 1, 1)]
void main_gelu_glu(uint3 dispatch_id : SV_DispatchThreadID) {
  const uint index = dispatch_id.x;
  if (index >= row_count) {
    return;
  }
  const float gate_value = inputs[index];
  const float up_value = inputs[row_count + index];
  const float cubed = gate_value * gate_value * gate_value;
  const float tanh_argument = clamp(
    0.7978845608028654f * (gate_value + 0.044715f * cubed),
    -15.0f,
    15.0f);
  const float gelu = 0.5f * gate_value * (1.0f + tanh(tanh_argument));
  outputs[output_base + index] = gelu * up_value;
}

// Elementwise silu(gate) * up over the FFN scratch buffer — the llama-family
// SwiGLU activation. Same buffer layout as main_gelu_glu. The exp argument is
// clamped so the driver exp never overflows to inf (sigmoid is fully
// saturated beyond +-30, so the clamp is value-neutral).
[numthreads(256, 1, 1)]
void main_silu_glu(uint3 dispatch_id : SV_DispatchThreadID) {
  const uint index = dispatch_id.x;
  if (index >= row_count) {
    return;
  }
  const float gate_value = inputs[index];
  const float up_value = inputs[row_count + index];
  const float exp_argument = clamp(-gate_value, -30.0f, 30.0f);
  const float silu = gate_value / (1.0f + exp(exp_argument));
  outputs[output_base + index] = silu * up_value;
}

// Builds Gemma's persistent per-layer PLE inputs from one packed model
// projection. One group owns one layer slice. t0 contains the shared norm
// vector and scaled token-input slices; t1 contains projected slices.
// batch_count is projection scale as float bits, input_stride is epsilon,
// weight_layout is norm offset, and input_transform is token-input offset.
[numthreads(256, 1, 1)]
void main_ple_prepare(uint3 group_id : SV_GroupID, uint thread_index : SV_GroupIndex) {
  const uint width = row_width;
  const uint layer = group_id.x;
  if (layer >= row_count || width == 0u) {
    return;
  }
  const uint base = layer * width;
  const float projection_scale = asfloat(batch_count);
  float sum_of_squares = 0.0f;
  for (uint item = thread_index; item < width; item += 256u) {
    const float value = inputs[base + item] * projection_scale;
    sum_of_squares += value * value;
  }
  norm_partials[thread_index] = sum_of_squares;
  GroupMemoryBarrierWithGroupSync();
  [unroll]
  for (uint stride = 128u; stride > 0u; stride >>= 1u) {
    if (thread_index < stride) {
      norm_partials[thread_index] += norm_partials[thread_index + stride];
    }
    GroupMemoryBarrierWithGroupSync();
  }
  const float inverse_rms = rsqrt(
    norm_partials[0] / (float)width + asfloat(input_stride));
  for (uint item = thread_index; item < width; item += 256u) {
    const float projected = inputs[base + item] * projection_scale * inverse_rms
      * asfloat(weights.Load((weight_layout + item) * 4u));
    const float token_input =
      asfloat(weights.Load((input_transform + base + item) * 4u));
    outputs[base + item] = (projected + token_input) * 0.7071067811865475f;
  }
}

// Applies Gemma's optional final-logit softcap. input_stride carries the
// softcap as float bits and row_count carries the vocabulary size.
[numthreads(256, 1, 1)]
void main_logit_softcap(uint3 dispatch_id : SV_DispatchThreadID) {
  const uint index = dispatch_id.x;
  if (index >= row_count) {
    return;
  }
  const float cap = asfloat(input_stride);
  const float value = inputs[index];
  outputs[index] = cap > 0.0f ? tanh(value / cap) * cap : value;
}

groupshared float argmax_values[256];
groupshared uint argmax_indices[256];

// One group scans the logits and writes the selected token ID as uint bits in
// outputs[0]. Ties resolve to the lowest token ID, matching the CPU sampler.
[numthreads(256, 1, 1)]
void main_argmax(uint thread_index : SV_GroupIndex) {
  float best_value = -3.402823466e+38f;
  uint best_index = 0u;
  for (uint index = thread_index; index < row_count; index += 256u) {
    const float value = inputs[index];
    if (value > best_value || (value == best_value && index < best_index)) {
      best_value = value;
      best_index = index;
    }
  }
  argmax_values[thread_index] = best_value;
  argmax_indices[thread_index] = best_index;
  GroupMemoryBarrierWithGroupSync();
  [unroll]
  for (uint stride = 128u; stride > 0u; stride >>= 1u) {
    if (thread_index < stride) {
      const float other_value = argmax_values[thread_index + stride];
      const uint other_index = argmax_indices[thread_index + stride];
      if (other_value > argmax_values[thread_index]
        || (other_value == argmax_values[thread_index]
          && other_index < argmax_indices[thread_index])) {
        argmax_values[thread_index] = other_value;
        argmax_indices[thread_index] = other_index;
      }
    }
    GroupMemoryBarrierWithGroupSync();
  }
  if (thread_index == 0u) {
    outputs[0] = asfloat(argmax_indices[0]);
  }
}

// Head-wise RMS norm followed by optional NEOX RoPE. Query vectors are kept
// as F32; key/value vectors are packed to FP16 for the resident KV cache.
// Constants:
//   row_width      head dimension
//   row_count      head count
//   batch_count    source offset in t1 (F32 elements)
//   output_base    destination offset in u0 (F32 elements or packed words)
//   scalar_format  bit0 weighted norm, bit1 NEOX RoPE, bit2 FP16 pack
//   input_stride   token position
//   weight_layout  norm-weight offset in t0 (F32 elements)
//   input_transform rope-factor offset in t0 (F32 elements)
groupshared float head_partials[256];
groupshared float prepared_head[512];

[numthreads(256, 1, 1)]
void main_gemma_prepare_head(uint3 group_id : SV_GroupID, uint thread_index : SV_GroupIndex) {
  const uint head_dim = row_width;
  const uint head = group_id.x;
  if (head >= row_count || head_dim == 0u || head_dim > 512u) {
    return;
  }
  const uint source_base = batch_count + head * head_dim;
  float square_sum = 0.0f;
  for (uint item = thread_index; item < head_dim; item += 256u) {
    const float value = inputs[source_base + item];
    square_sum = value * value;
    if (item + 256u < head_dim) {
      const float second_value = inputs[source_base + item + 256u];
      square_sum += second_value * second_value;
      break;
    }
  }
  head_partials[thread_index] = square_sum;
  GroupMemoryBarrierWithGroupSync();
  [unroll]
  for (uint stride = 128u; stride > 0u; stride >>= 1u) {
    if (thread_index < stride) {
      head_partials[thread_index] += head_partials[thread_index + stride];
    }
    GroupMemoryBarrierWithGroupSync();
  }

  const float epsilon = asfloat(weights.Load(0u));
  const float inverse_rms = rsqrt(head_partials[0] / (float)head_dim + epsilon);
  for (uint item = thread_index; item < head_dim; item += 256u) {
    const float norm_weight = (scalar_format & 1u) != 0u
      ? asfloat(weights.Load((weight_layout + item) * 4u))
      : 1.0f;
    prepared_head[item] = inputs[source_base + item] * inverse_rms * norm_weight;
  }
  GroupMemoryBarrierWithGroupSync();

  if ((scalar_format & 2u) != 0u && thread_index < head_dim / 2u) {
    const uint first_index = thread_index;
    const uint second_index = thread_index + head_dim / 2u;
    const float rope_base = asfloat(weights.Load(4u));
    const float rope_scale = asfloat(weights.Load(8u));
    float inverse_frequency = pow(rope_base, -(2.0f * (float)thread_index) / (float)head_dim);
    const float factor = asfloat(weights.Load((input_transform + thread_index) * 4u));
    if (factor > 0.0f) {
      inverse_frequency /= factor;
    }
    const float angle = (float)input_stride * rope_scale * inverse_frequency;
    float sine_value;
    float cosine_value;
    sincos(angle, sine_value, cosine_value);
    const float first = prepared_head[first_index];
    const float second = prepared_head[second_index];
    prepared_head[first_index] = first * cosine_value - second * sine_value;
    prepared_head[second_index] = first * sine_value + second * cosine_value;
  }
  GroupMemoryBarrierWithGroupSync();

  if ((scalar_format & 4u) == 0u) {
    for (uint item = thread_index; item < head_dim; item += 256u) {
      outputs[output_base + head * head_dim + item] = prepared_head[item];
    }
  } else {
    const uint words_per_head = (head_dim + 1u) / 2u;
    for (uint word = thread_index; word < words_per_head; word += 256u) {
      const uint first_index = word * 2u;
      const uint second_index = min(first_index + 1u, head_dim - 1u);
      const uint packed = f32tof16(prepared_head[first_index])
        | (f32tof16(prepared_head[second_index]) << 16u);
      outputs[output_base + head * words_per_head + word] = asfloat(packed);
    }
  }
}

float load_fp16_element(uint packed_word_base, uint element_index) {
  const uint packed = weights.Load((packed_word_base + element_index / 2u) * 4u);
  const uint half_bits = (element_index & 1u) == 0u ? (packed & 0xFFFFu) : (packed >> 16u);
  return f16tof32(half_bits);
}

// One workgroup owns one query head and walks the attended positions with an
// online softmax, so scores are never materialized. t0 is the packed FP16 KV
// arena, t1 is the normalized/rotated F32 query, and u0 receives F32 context.
// Constants:
//   row_width      head dimension
//   row_count      query head count
//   batch_count    KV head count
//   output_base    context output offset
//   scalar_format  cached position count (including current token)
//   input_stride   first attended position
//   weight_layout  value-cache base in packed words
//   input_transform KV vector width
groupshared float attention_partials[256];
groupshared float online_alpha;
groupshared float online_beta;
groupshared float online_denominator;
groupshared float online_maximum;

[numthreads(256, 1, 1)]
void main_gemma_online_attention(uint3 group_id : SV_GroupID, uint thread_index : SV_GroupIndex) {
  const uint head_dim = row_width;
  const uint query_head = group_id.x;
  if (query_head >= row_count || head_dim == 0u || head_dim > 512u) {
    return;
  }
  const uint kv_head = query_head * batch_count / row_count;
  const uint query_base = query_head * head_dim;
  const uint kv_width = input_transform;
  float context_value = 0.0f;
  if (thread_index + 256u < head_dim) {
    prepared_head[thread_index + 256u] = 0.0f;
  }
  if (thread_index == 0u) {
    online_maximum = -3.402823466e+38f;
    online_denominator = 0.0f;
  }
  GroupMemoryBarrierWithGroupSync();

  [loop]
  for (uint source = input_stride; source < scalar_format; source += 1u) {
    float partial = 0.0f;
    for (uint item = thread_index; item < head_dim; item += 256u) {
      const uint kv_element = source * kv_width + kv_head * head_dim + item;
      partial += inputs[query_base + item] * load_fp16_element(0u, kv_element);
    }
    attention_partials[thread_index] = partial;
    GroupMemoryBarrierWithGroupSync();
    [unroll]
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
      if (thread_index < stride) {
        attention_partials[thread_index] += attention_partials[thread_index + stride];
      }
      GroupMemoryBarrierWithGroupSync();
    }
    if (thread_index == 0u) {
      // Gemma's QK-norm design uses the raw dot product as the attention
      // score: query/key head RMS-norms already bound the magnitude, and the
      // CPU reference path applies no 1/sqrt(head_dim) scaling. Scaling here
      // flattens the softmax and degrades decoding into repetition.
      const float score = attention_partials[0];
      const float next_maximum = max(online_maximum, score);
      online_alpha = exp(online_maximum - next_maximum);
      online_beta = exp(score - next_maximum);
      online_denominator = online_denominator * online_alpha + online_beta;
      online_maximum = next_maximum;
    }
    GroupMemoryBarrierWithGroupSync();
    for (uint item = thread_index; item < head_dim; item += 256u) {
      const uint kv_element = source * kv_width + kv_head * head_dim + item;
      const float value = load_fp16_element(weight_layout, kv_element);
      const uint local_slot = item / 256u;
      if (local_slot == 0u) {
        context_value = context_value * online_alpha + value * online_beta;
      } else {
        prepared_head[item] = prepared_head[item] * online_alpha + value * online_beta;
      }
    }
    GroupMemoryBarrierWithGroupSync();
  }

  for (uint item = thread_index; item < head_dim; item += 256u) {
    const float accumulated = item < 256u ? context_value : prepared_head[item];
    outputs[output_base + query_base + item] =
      accumulated / max(online_denominator, 1.175494351e-38f);
  }
}

// Position-tiled variant of the online attention above: processes up to 8
// attended positions per barrier set instead of paying ~11 barriers per
// position, and stages each tile's K/V rows into groupshared with one
// coalesced dword load per two elements (the per-element path loads a full
// dword per fp16 element, fetching every KV byte twice). Every
// floating-point operation still happens in the identical order as the
// untiled kernel (per-position partial dots over the same per-thread element
// ownership, the same pairwise tree reduction per score, the same sequential
// thread-0 softmax updates, and the same per-position context accumulation
// sequence per element), so the result is bit-exact against
// main_gemma_online_attention — enforced by the selftest's bitwise
// comparison. Same constants as the untiled kernel.
groupshared float attn_tile_partials[8][256];
groupshared float attn_tile_alpha[8];
groupshared float attn_tile_beta[8];
groupshared uint attn_tile_kv_words[8 * 256];

float tile_fp16_element(uint position_slot, uint element_index) {
  const uint packed = attn_tile_kv_words[position_slot * 256u + element_index / 2u];
  const uint half_bits = (element_index & 1u) == 0u ? (packed & 0xFFFFu) : (packed >> 16u);
  return f16tof32(half_bits);
}

[numthreads(256, 1, 1)]
void main_gemma_online_attention_tiled(uint3 group_id : SV_GroupID, uint thread_index : SV_GroupIndex) {
  const uint head_dim = row_width;
  const uint query_head = group_id.x;
  if (query_head >= row_count || head_dim == 0u || head_dim > 512u) {
    return;
  }
  const uint kv_head = query_head * batch_count / row_count;
  const uint query_base = query_head * head_dim;
  const uint kv_width = input_transform;
  const uint words_per_position = head_dim / 2u;
  float context_value = 0.0f;
  if (thread_index + 256u < head_dim) {
    prepared_head[thread_index + 256u] = 0.0f;
  }
  if (thread_index == 0u) {
    online_maximum = -3.402823466e+38f;
    online_denominator = 0.0f;
  }
  GroupMemoryBarrierWithGroupSync();

  [loop]
  for (uint tile_start = input_stride; tile_start < scalar_format; tile_start += 8u) {
    const uint tile_count = min(8u, scalar_format - tile_start);
    const uint tile_words = tile_count * words_per_position;
    // Stage the tile's key rows: one dword load per two fp16 elements.
    // kv_width and kv_head*head_dim are both even, so every position's
    // element base is dword-aligned in the packed arena.
    for (uint key_word = thread_index; key_word < tile_words; key_word += 256u) {
      const uint key_position = key_word / words_per_position;
      const uint word_in_position = key_word - key_position * words_per_position;
      const uint element_base =
        (tile_start + key_position) * kv_width + kv_head * head_dim;
      attn_tile_kv_words[key_position * 256u + word_in_position] =
        weights.Load((element_base / 2u + word_in_position) * 4u);
    }
    GroupMemoryBarrierWithGroupSync();
    for (uint dot_tile = 0u; dot_tile < tile_count; dot_tile += 1u) {
      float partial = 0.0f;
      for (uint item = thread_index; item < head_dim; item += 256u) {
        partial += inputs[query_base + item] * tile_fp16_element(dot_tile, item);
      }
      attn_tile_partials[dot_tile][thread_index] = partial;
    }
    GroupMemoryBarrierWithGroupSync();
    [unroll]
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
      if (thread_index < stride) {
        for (uint reduce_tile = 0u; reduce_tile < tile_count; reduce_tile += 1u) {
          attn_tile_partials[reduce_tile][thread_index] +=
            attn_tile_partials[reduce_tile][thread_index + stride];
        }
      }
      GroupMemoryBarrierWithGroupSync();
    }
    if (thread_index == 0u) {
      for (uint score_tile = 0u; score_tile < tile_count; score_tile += 1u) {
        // Raw dot as the score — same as the untiled kernel (Gemma QK-norm
        // design, no 1/sqrt(head_dim) scaling).
        const float score = attn_tile_partials[score_tile][0];
        const float next_maximum = max(online_maximum, score);
        attn_tile_alpha[score_tile] = exp(online_maximum - next_maximum);
        attn_tile_beta[score_tile] = exp(score - next_maximum);
        online_denominator = online_denominator * attn_tile_alpha[score_tile]
          + attn_tile_beta[score_tile];
        online_maximum = next_maximum;
      }
    }
    GroupMemoryBarrierWithGroupSync();
    // Stage the tile's value rows over the key words (scores are already
    // reduced into attn_tile_partials, so the key data is dead here).
    for (uint value_word = thread_index; value_word < tile_words; value_word += 256u) {
      const uint value_position = value_word / words_per_position;
      const uint word_in_position = value_word - value_position * words_per_position;
      const uint element_base =
        (tile_start + value_position) * kv_width + kv_head * head_dim;
      attn_tile_kv_words[value_position * 256u + word_in_position] =
        weights.Load((weight_layout + element_base / 2u + word_in_position) * 4u);
    }
    GroupMemoryBarrierWithGroupSync();
    for (uint value_tile = 0u; value_tile < tile_count; value_tile += 1u) {
      const float alpha_value = attn_tile_alpha[value_tile];
      const float beta_value = attn_tile_beta[value_tile];
      for (uint item = thread_index; item < head_dim; item += 256u) {
        const float value = tile_fp16_element(value_tile, item);
        const uint local_slot = item / 256u;
        if (local_slot == 0u) {
          context_value = context_value * alpha_value + value * beta_value;
        } else {
          prepared_head[item] = prepared_head[item] * alpha_value + value * beta_value;
        }
      }
    }
    GroupMemoryBarrierWithGroupSync();
  }

  for (uint item = thread_index; item < head_dim; item += 256u) {
    const float accumulated = item < 256u ? context_value : prepared_head[item];
    outputs[output_base + query_base + item] =
      accumulated / max(online_denominator, 1.175494351e-38f);
  }
}

[numthreads(k_rows_per_group, 1, 1)]
void main_scalar(uint3 group_id : SV_GroupID, uint thread_index : SV_GroupIndex) {
  const uint row = group_id.x * k_rows_per_group + thread_index;
  const uint batch_start = group_id.y * k_batch_tile;

  float acc[k_batch_tile];
  [unroll]
  for (uint init = 0; init < k_batch_tile; init += 1u) {
    acc[init] = 0.0f;
  }

  const uint tile_count = (row_width + k_tile_elements - 1u) / k_tile_elements;

  [loop]
  for (uint tile = 0; tile < tile_count; tile += 1u) {
    const uint element_base = tile * k_tile_elements;
    load_input_tile(thread_index, batch_start, element_base);
    GroupMemoryBarrierWithGroupSync();

    if (row < row_count) {
      const uint tile_width = min(k_tile_elements, row_width - element_base);
      [loop]
      for (uint element = 0; element < tile_width; element += 1u) {
        const uint index = weight_layout == 1u
          ? (element_base + element) * row_count + row
          : row * row_width + element_base + element;
        float weight_value;
        if (scalar_format == 0u) {
          weight_value = asfloat(weights.Load(index * 4u));
        } else {
          const uint byte_offset = index * 2u;
          const uint word = weights.Load(byte_offset & ~3u);
          const uint half_bits = ((byte_offset & 2u) != 0u) ? (word >> 16u) : (word & 0xFFFFu);
          weight_value = (scalar_format == 1u) ? f16tof32(half_bits) : asfloat(half_bits << 16u);
        }
        [unroll]
        for (uint b = 0; b < k_batch_tile; b += 1u) {
          acc[b] += weight_value * input_tile[b * k_tile_elements + element];
        }
      }
    }
    GroupMemoryBarrierWithGroupSync();
  }

  store_outputs(row, batch_start, acc);
}
)";

struct DispatchConstants {
  std::uint32_t row_width = 0;
  std::uint32_t row_count = 0;
  std::uint32_t batch_count = 0;
  std::uint32_t output_base = 0;
  std::uint32_t scalar_format = 0;
  std::uint32_t input_stride = 0;
  std::uint32_t weight_layout = 0;
  std::uint32_t input_transform = 0;
};

std::uint64_t staged_weight_bytes(D3d12WeightFormat format, std::uint64_t weight_bytes) {
  if (format == D3d12WeightFormat::Q6_K) {
    const std::uint64_t block_count = weight_bytes / k_q6_k_block_bytes;
    return block_count * k_q6_k_staged_block_bytes;
  }
  if (format == D3d12WeightFormat::Q8_0) {
    const std::uint64_t block_count = weight_bytes / k_q8_0_block_bytes;
    return block_count * k_q8_0_staged_block_bytes;
  }
  if (format == D3d12WeightFormat::Q4_0) {
    const std::uint64_t block_count = weight_bytes / k_q4_0_block_bytes;
    return block_count * k_q4_0_staged_block_bytes;
  }
  if (format == D3d12WeightFormat::F16 || format == D3d12WeightFormat::BF16) {
    // Half-precision tensors are widened to F32 at staging time: the raw
    // 2-byte load path in the scalar shader hangs the Intel Xe driver, while
    // the F32 lane is stable. The cost is 2x device memory for these tensors.
    return weight_bytes * 2;
  }
  return weight_bytes;
}

float staging_half_to_float(std::uint16_t half_bits) {
  const std::uint32_t sign = (half_bits & 0x8000u) << 16;
  const std::uint32_t exponent = (half_bits >> 10) & 0x1Fu;
  const std::uint32_t mantissa = half_bits & 0x3FFu;
  std::uint32_t bits;
  if (exponent == 0) {
    if (mantissa == 0) {
      bits = sign;
    } else {
      // Normalize the subnormal half.
      std::uint32_t shifted_mantissa = mantissa;
      std::int32_t adjusted_exponent = -1;
      while ((shifted_mantissa & 0x400u) == 0) {
        shifted_mantissa <<= 1;
        adjusted_exponent -= 1;
      }
      bits = sign
        | (static_cast<std::uint32_t>(113 + adjusted_exponent) << 23)
        | ((shifted_mantissa & 0x3FFu) << 13);
    }
  } else if (exponent == 31) {
    bits = sign | 0x7F800000u | (mantissa << 13);
  } else {
    bits = sign | ((exponent + 112u) << 23) | (mantissa << 13);
  }
  float value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

std::uint16_t staging_float_to_half(float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const std::uint32_t sign = (bits >> 16) & 0x8000u;
  const std::uint32_t exponent = (bits >> 23) & 0xFFu;
  const std::uint32_t mantissa = bits & 0x7FFFFFu;
  if (exponent == 0xFFu) {
    return static_cast<std::uint16_t>(
      sign | (mantissa == 0 ? 0x7C00u : (0x7C00u | std::max<std::uint32_t>(1, mantissa >> 13))));
  }
  const std::int32_t half_exponent = static_cast<std::int32_t>(exponent) - 127 + 15;
  if (half_exponent >= 31) {
    return static_cast<std::uint16_t>(sign | 0x7C00u);
  }
  if (half_exponent <= 0) {
    if (half_exponent < -10) {
      return static_cast<std::uint16_t>(sign);
    }
    const std::uint32_t normalized = mantissa | 0x800000u;
    const std::uint32_t shift = static_cast<std::uint32_t>(14 - half_exponent);
    const std::uint32_t rounded = (normalized + ((1u << (shift - 1u)) - 1u)
      + ((normalized >> shift) & 1u)) >> shift;
    return static_cast<std::uint16_t>(sign | rounded);
  }
  const std::uint32_t rounded_mantissa = mantissa + 0xFFFu + ((mantissa >> 13) & 1u);
  if ((rounded_mantissa & 0x800000u) != 0u) {
    if (half_exponent + 1 >= 31) {
      return static_cast<std::uint16_t>(sign | 0x7C00u);
    }
    return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(half_exponent + 1) << 10));
  }
  return static_cast<std::uint16_t>(
    sign | (static_cast<std::uint32_t>(half_exponent) << 10) | (rounded_mantissa >> 13));
}

// Writes a KV lane into the device cache's half layout. An f16 lane is
// already in that layout and copies straight through; an f32 lane converts
// element-wise (the historical path).
void pack_kv_lane_halves(std::uint16_t *destination, const D3d12KvLaneView &lane) {
  if (lane.count == 0) {
    return;
  }
  if (lane.f16 != nullptr) {
    std::memcpy(destination, lane.f16, lane.count * sizeof(std::uint16_t));
    return;
  }
  for (std::size_t item = 0; item < lane.count; item += 1) {
    destination[item] = staging_float_to_half(lane.f32[item]);
  }
}

// Repacks a row-major tensor into column-major (block-interleaved) order so
// adjacent GPU threads, which each own one row, read adjacent memory. Without
// this the whole-tensor dispatches are so bandwidth-starved that big shapes
// exceed the Windows GPU watchdog and hang the device.
void stage_weights_column_major(const D3d12ResidentWeightRef &op, std::uint8_t *staging) {
  const std::uint64_t row_count = op.row_count;
  const std::uint64_t row_width = op.row_width;
  switch (op.format) {
    case D3d12WeightFormat::Q4_K: {
      const std::uint64_t blocks_per_row = row_width / k_quant_block_elements;
      if (quad_interleave_enabled()) {
        // Finer interleaving: quad q of block b for every row stored together,
        // so lane j's 16-byte Load4 sits 16 bytes from lane j+1's instead of
        // 144. Block-granular interleaving makes a wave's TOTAL span contiguous
        // while every INDIVIDUAL load still scatters across 32 cache lines;
        // measured, this addressing reaches 89% of the adapter's streaming
        // ceiling against 49% for the block layout (--d3d12-selftest).
        constexpr std::uint64_t k_quads = k_q4_k_block_bytes / 16;  // 9
        for (std::uint64_t row = 0; row < row_count; row += 1) {
          const std::uint8_t *source_row =
            op.weight_data + row * blocks_per_row * k_q4_k_block_bytes;
          for (std::uint64_t column = 0; column < blocks_per_row; column += 1) {
            const std::uint8_t *source_block = source_row + column * k_q4_k_block_bytes;
            for (std::uint64_t quad = 0; quad < k_quads; quad += 1) {
              std::memcpy(
                staging + ((column * k_quads + quad) * row_count + row) * 16,
                source_block + quad * 16,
                16);
            }
          }
        }
        break;
      }
      for (std::uint64_t row = 0; row < row_count; row += 1) {
        const std::uint8_t *source_row = op.weight_data + row * blocks_per_row * k_q4_k_block_bytes;
        for (std::uint64_t column = 0; column < blocks_per_row; column += 1) {
          std::memcpy(
            staging + (column * row_count + row) * k_q4_k_block_bytes,
            source_row + column * k_q4_k_block_bytes,
            k_q4_k_block_bytes);
        }
      }
      break;
    }
    case D3d12WeightFormat::Q5_K: {
      const std::uint64_t blocks_per_row = row_width / k_quant_block_elements;
      for (std::uint64_t row = 0; row < row_count; row += 1) {
        const std::uint8_t *source_row = op.weight_data + row * blocks_per_row * k_q5_k_block_bytes;
        for (std::uint64_t column = 0; column < blocks_per_row; column += 1) {
          std::memcpy(
            staging + (column * row_count + row) * k_q5_k_block_bytes,
            source_row + column * k_q5_k_block_bytes,
            k_q5_k_block_bytes);
        }
      }
      break;
    }
    case D3d12WeightFormat::Q6_K: {
      const std::uint64_t blocks_per_row = row_width / k_quant_block_elements;
      for (std::uint64_t row = 0; row < row_count; row += 1) {
        const std::uint8_t *source_row = op.weight_data + row * blocks_per_row * k_q6_k_block_bytes;
        for (std::uint64_t column = 0; column < blocks_per_row; column += 1) {
          std::memcpy(
            staging + (column * row_count + row) * k_q6_k_staged_block_bytes,
            source_row + column * k_q6_k_block_bytes,
            k_q6_k_block_bytes);
        }
      }
      break;
    }
    case D3d12WeightFormat::Q8_0: {
      const std::uint64_t blocks_per_row = row_width / k_legacy_block_elements;
      for (std::uint64_t row = 0; row < row_count; row += 1) {
        const std::uint8_t *source_row = op.weight_data + row * blocks_per_row * k_q8_0_block_bytes;
        for (std::uint64_t column = 0; column < blocks_per_row; column += 1) {
          const std::uint8_t *src = source_row + column * k_q8_0_block_bytes;
          std::uint8_t *dst = staging + (column * row_count + row) * k_q8_0_staged_block_bytes;
          std::uint16_t d_half;
          std::memcpy(&d_half, src, sizeof(d_half));
          const float d = staging_half_to_float(d_half);
          std::memcpy(dst, &d, sizeof(d));
          std::memcpy(dst + 4, src + 2, k_legacy_block_elements);  // 32 int8
        }
      }
      break;
    }
    case D3d12WeightFormat::Q4_0: {
      const std::uint64_t blocks_per_row = row_width / k_legacy_block_elements;
      for (std::uint64_t row = 0; row < row_count; row += 1) {
        const std::uint8_t *source_row = op.weight_data + row * blocks_per_row * k_q4_0_block_bytes;
        for (std::uint64_t column = 0; column < blocks_per_row; column += 1) {
          const std::uint8_t *src = source_row + column * k_q4_0_block_bytes;
          std::uint8_t *dst = staging + (column * row_count + row) * k_q4_0_staged_block_bytes;
          std::uint16_t d_half;
          std::memcpy(&d_half, src, sizeof(d_half));
          const float d = staging_half_to_float(d_half);
          std::memcpy(dst, &d, sizeof(d));
          std::memcpy(dst + 4, src + 2, k_legacy_block_elements / 2);  // 16 nibble bytes
        }
      }
      break;
    }
    case D3d12WeightFormat::F32: {
      const float *source = reinterpret_cast<const float *>(op.weight_data);
      float *destination = reinterpret_cast<float *>(staging);
      for (std::uint64_t row = 0; row < row_count; row += 1) {
        for (std::uint64_t element = 0; element < row_width; element += 1) {
          destination[element * row_count + row] = source[row * row_width + element];
        }
      }
      break;
    }
    case D3d12WeightFormat::F16:
    case D3d12WeightFormat::BF16: {
      const std::uint16_t *source = reinterpret_cast<const std::uint16_t *>(op.weight_data);
      float *destination = reinterpret_cast<float *>(staging);
      const bool is_f16 = op.format == D3d12WeightFormat::F16;
      for (std::uint64_t row = 0; row < row_count; row += 1) {
        for (std::uint64_t element = 0; element < row_width; element += 1) {
          const std::uint16_t half_bits = source[row * row_width + element];
          float value;
          if (is_f16) {
            value = staging_half_to_float(half_bits);
          } else {
            const std::uint32_t bits = static_cast<std::uint32_t>(half_bits) << 16;
            std::memcpy(&value, &bits, sizeof(value));
          }
          destination[element * row_count + row] = value;
        }
      }
      break;
    }
  }
}

std::uint64_t expected_weight_bytes(
  D3d12WeightFormat format,
  std::uint64_t row_count,
  std::uint64_t row_width) {
  const std::uint64_t elements = row_count * row_width;
  switch (format) {
    case D3d12WeightFormat::F32:
      return elements * 4;
    case D3d12WeightFormat::F16:
    case D3d12WeightFormat::BF16:
      return elements * 2;
    case D3d12WeightFormat::Q4_K:
      return (elements / k_quant_block_elements) * k_q4_k_block_bytes;
    case D3d12WeightFormat::Q5_K:
      return (elements / k_quant_block_elements) * k_q5_k_block_bytes;
    case D3d12WeightFormat::Q6_K:
      return (elements / k_quant_block_elements) * k_q6_k_block_bytes;
    case D3d12WeightFormat::Q8_0:
      return (elements / k_legacy_block_elements) * k_q8_0_block_bytes;
    case D3d12WeightFormat::Q4_0:
      return (elements / k_legacy_block_elements) * k_q4_0_block_bytes;
  }
  return 0;
}

class D3d12MatmulContext {
 public:
  D3d12MatmulContext()
    : weight_budget_bytes_(configured_weight_budget_bytes()),
      requested_weight_budget_bytes_(configured_weight_budget_bytes()) {
  }

  void set_prestaging(bool active) { prestaging_ = active; }

  ~D3d12MatmulContext() {
    release_mapped_resource(layer_tail_upload_, mapped_layer_tail_upload_);
    release_mapped_resource(attention_readback_, mapped_attention_readback_);
    release_mapped_resource(output_default_, mapped_output_direct_);
    release_mapped_resource(output_readback_, mapped_output_readback_);
    release_mapped_resource(input_upload_, mapped_input_upload_);
    release_mapped_resource(weight_upload_, mapped_weight_upload_);
    if (fence_event_ != nullptr) {
      CloseHandle(fence_event_);
      fence_event_ = nullptr;
    }
  }

  bool initialize(std::string &error_message) {
    capabilities_.backend_name = "directx-hybrid";
    trace_d3d12("initialize: begin");

    IDXGIFactory1 *factory_raw = nullptr;
    HRESULT hr = CreateDXGIFactory1(IID_PPV_ARGS(&factory_raw));
    if (FAILED(hr)) {
      error_message = hresult_message("CreateDXGIFactory1", hr);
      capabilities_.reason = error_message;
      return false;
    }
    factory_.reset(factory_raw);

    if (!select_adapter(error_message)) {
      capabilities_.reason = error_message;
      return false;
    }
    // IDXGIAdapter3 unlocks live QueryVideoMemoryInfo budgets (Finding 2);
    // optional - without it the static requested budget stays in effect.
    {
      IDXGIAdapter3 *adapter3_raw = nullptr;
      if (SUCCEEDED(adapter_->QueryInterface(IID_PPV_ARGS(&adapter3_raw)))) {
        adapter3_.reset(adapter3_raw);
      }
    }

    ID3D12Device *device_raw = nullptr;
    hr = D3D12CreateDevice(adapter_.get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device_raw));
    if (FAILED(hr)) {
      error_message = hresult_message("D3D12CreateDevice", hr);
      capabilities_.reason = error_message;
      return false;
    }
    device_.reset(device_raw);
    // ID3D12Device1 unlocks SetResidencyPriority for weight-heap pinning
    // (Phase F2); optional - older runtimes simply skip the pinning.
    {
      ID3D12Device1 *device1_raw = nullptr;
      if (SUCCEEDED(device_->QueryInterface(IID_PPV_ARGS(&device1_raw)))) {
        device1_.reset(device1_raw);
      }
    }

    // On cache-coherent UMA hardware (integrated GPUs) the output UAV can
    // live in a CPU-visible custom heap and be read directly after the
    // fence, removing one barrier + one copy from every submission.
    D3D12_FEATURE_DATA_ARCHITECTURE architecture = {};
    if (SUCCEEDED(device_->CheckFeatureSupport(
      D3D12_FEATURE_ARCHITECTURE,
      &architecture,
      sizeof(architecture)))) {
      // Native architecture facts are authoritative (Phase 1): recorded in
      // the capabilities so telemetry reports the device's real memory
      // model, independent of vendor/name heuristics.
      capabilities_.adapter_uma = architecture.UMA != FALSE;
      capabilities_.adapter_cache_coherent_uma = architecture.CacheCoherentUMA != FALSE;
      if (architecture.CacheCoherentUMA) {
        char uma_env[8] = {};
        const bool env_disabled = GetEnvironmentVariableA(
          "LIZARD_NATIVE_GPU_UMA_OUTPUT", uma_env, sizeof(uma_env)) != 0
          && uma_env[0] == '0';
        uma_output_ = !env_disabled;
      }
    }

    D3D12_COMMAND_QUEUE_DESC queue_desc = {};
    queue_desc.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
    queue_desc.Priority = D3D12_COMMAND_QUEUE_PRIORITY_NORMAL;
    queue_desc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;

    ID3D12CommandQueue *queue_raw = nullptr;
    hr = device_->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue_raw));
    if (FAILED(hr)) {
      error_message = hresult_message("CreateCommandQueue", hr);
      capabilities_.reason = error_message;
      return false;
    }
    command_queue_.reset(queue_raw);

    ID3D12CommandAllocator *allocator_raw = nullptr;
    hr = device_->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&allocator_raw));
    if (FAILED(hr)) {
      error_message = hresult_message("CreateCommandAllocator", hr);
      capabilities_.reason = error_message;
      return false;
    }
    command_allocator_.reset(allocator_raw);

    ID3D12GraphicsCommandList *list_raw = nullptr;
    hr = device_->CreateCommandList(
      0,
      D3D12_COMMAND_LIST_TYPE_COMPUTE,
      command_allocator_.get(),
      nullptr,
      IID_PPV_ARGS(&list_raw));
    if (FAILED(hr)) {
      error_message = hresult_message("CreateCommandList", hr);
      capabilities_.reason = error_message;
      return false;
    }
    command_list_.reset(list_raw);
    command_list_->Close();

    ID3D12Fence *fence_raw = nullptr;
    hr = device_->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence_raw));
    if (FAILED(hr)) {
      error_message = hresult_message("CreateFence", hr);
      capabilities_.reason = error_message;
      return false;
    }
    fence_.reset(fence_raw);
    fence_event_ = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (fence_event_ == nullptr) {
      error_message = "CreateEventW failed for the D3D12 fence.";
      capabilities_.reason = error_message;
      return false;
    }

    if (!create_pipelines(error_message)) {
      capabilities_.reason = error_message;
      return false;
    }

    // DP4A (SM 6.4) kernels are strictly optional: any failure keeps the
    // float FXC pipelines as the execution path and only records the reason.
    initialize_dp4a_pipelines();

    // First live clamp: from here on LIZARD_NATIVE_GPU_WEIGHT_BUDGET_MB is a
    // request ceiling, never permission to exceed the adapter's DXGI budget.
    refresh_weight_budget(true);

    capabilities_.supported = true;
    capabilities_.reason =
      "Windows-native Direct3D 12 compute with resident quantized weights is available.";
    trace_d3d12("initialize: done");
    return true;
  }

  const D3d12Dp4aCapabilities &dp4a_capabilities() const {
    return dp4a_capabilities_;
  }

  const D3d12BatchMatmulCapabilities &capabilities() const {
    return capabilities_;
  }

  std::uint64_t resident_weight_bytes() const {
    return weight_cache_bytes_;
  }

  const D3d12RuntimeTelemetry &telemetry() const {
    return telemetry_;
  }

  void reset_telemetry() {
    telemetry_ = D3d12RuntimeTelemetry {};
  }

  void reset_resident_weights() {
    weight_cache_.clear();
    weight_cache_bytes_ = 0;
    gemma_kv_cache_.clear();
  }

  bool device_removed() const {
    return device_removed_;
  }

  bool compute_resident(
    const D3d12ResidentWeightRef *ops,
    std::size_t op_count,
    const std::vector<float> &packed_inputs,
    std::uint32_t batch_count,
    const void *input_identity,
    std::vector<float> *outputs,
    std::string &error_message) {
    if (!capabilities_.supported) {
      error_message = capabilities_.reason;
      return false;
    }
    const auto resident_entry_at = std::chrono::steady_clock::now();
    // Stamps in this lane too. `wait` above is 1.528 ms while 8.4 MB at the
    // kernels' measured 20.5 GB/s is only ~0.41 ms, and the two candidate
    // explanations demand opposite fixes: if the GPU really is busy that long,
    // these narrow matvecs (K and V are 1024 rows) are occupancy-limited and the
    // answer is to widen them into one dispatch; if it is not, the time is
    // submission latency and the answer is fewer submissions. Only a stamp
    // inside the list can tell them apart.
    bool resident_stamps = false;
    if (op_count == 0 || batch_count == 0 || packed_inputs.empty()) {
      error_message = "D3D12 resident matmul requires at least one op and a non-empty input batch.";
      return false;
    }

    const std::uint32_t input_width = ops[0].row_width;
    std::uint64_t total_output_floats = 0;
    std::uint64_t missing_staging_bytes = 0;
    std::uint64_t missing_resident_bytes = 0;
    for (std::size_t op_index = 0; op_index < op_count; op_index += 1) {
      const D3d12ResidentWeightRef &op = ops[op_index];
      if (op.weight_key == nullptr || op.weight_data == nullptr || op.row_count == 0 || op.row_width == 0) {
        error_message = "D3D12 resident matmul received an invalid weight reference.";
        return false;
      }
      if (op.row_width != input_width) {
        error_message = "D3D12 resident matmul ops must share one input width per submission.";
        return false;
      }
      if ((op.format == D3d12WeightFormat::Q4_K || op.format == D3d12WeightFormat::Q5_K || op.format == D3d12WeightFormat::Q6_K)
        && (op.row_width % k_quant_block_elements) != 0) {
        error_message = "D3D12 quantized matmul requires the row width to be a multiple of 256.";
        return false;
      }
      if ((op.format == D3d12WeightFormat::Q8_0 || op.format == D3d12WeightFormat::Q4_0)
        && (op.row_width % k_legacy_block_elements) != 0) {
        error_message = "D3D12 legacy-quant matmul requires the row width to be a multiple of 32.";
        return false;
      }
      const std::uint64_t expected_bytes = expected_weight_bytes(op.format, op.row_count, op.row_width);
      if (expected_bytes == 0 || op.weight_bytes != expected_bytes) {
        error_message = "D3D12 resident matmul weight payload does not match the requested shape.";
        return false;
      }
      total_output_floats += static_cast<std::uint64_t>(batch_count) * op.row_count;
      if (weight_cache_.find(op.weight_key) == weight_cache_.end()) {
        telemetry_.weight_cache_misses += 1;
        const std::uint64_t staged = staged_weight_bytes(op.format, op.weight_bytes);
        missing_staging_bytes += align_up(staged, 4);
        missing_resident_bytes += staged;
      } else {
        telemetry_.weight_cache_hits += 1;
      }
    }

    const std::uint64_t input_items = static_cast<std::uint64_t>(batch_count) * input_width;
    if (packed_inputs.size() != input_items) {
      error_message = "D3D12 resident matmul packed input payload size does not match the requested shape.";
      return false;
    }

    if (missing_resident_bytes > 0
      && !evict_for(missing_resident_bytes, ops, op_count, error_message)) {
      return false;
    }

    const std::uint64_t input_bytes = input_items * sizeof(float);
    const std::uint64_t output_bytes = total_output_floats * sizeof(float);
    const bool input_buffer_recreated = !input_default_ || input_default_bytes_ < input_bytes;

    if (missing_staging_bytes > 0
      && !ensure_persistent_upload_buffer(
        weight_upload_,
        missing_staging_bytes,
        weight_upload_bytes_,
        mapped_weight_upload_,
        error_message)) {
      return false;
    }
    if (!ensure_persistent_upload_buffer(
      input_upload_,
      input_bytes,
      input_upload_bytes_,
      mapped_input_upload_,
      error_message)) {
      return false;
    }
    if (!ensure_srv_buffer(input_default_, input_bytes, input_default_bytes_, input_state_, error_message)) {
      return false;
    }
    if (!ensure_output_buffer(output_default_, output_bytes, output_default_bytes_, output_state_, error_message)) {
      return false;
    }
    if (!uma_output_
      && !ensure_persistent_readback_buffer(
        output_readback_,
        output_bytes,
        output_readback_bytes_,
        mapped_output_readback_,
        error_message)) {
      return false;
    }

    // Stage missing weights into the upload buffer (Q6_K blocks repacked to a
    // 4-byte-aligned stride) and create their resident default-heap buffers.
    struct PendingUpload {
      const void *weight_key;
      std::uint64_t staging_offset;
      std::uint64_t staged_bytes;
    };
    std::vector<PendingUpload> pending_uploads;
    std::uint64_t staging_cursor = 0;
    for (std::size_t op_index = 0; op_index < op_count; op_index += 1) {
      const D3d12ResidentWeightRef &op = ops[op_index];
      if (weight_cache_.find(op.weight_key) != weight_cache_.end()) {
        continue;
      }
      const std::uint64_t staged = staged_weight_bytes(op.format, op.weight_bytes);
      // Repack in cacheable system memory first: the upload heap is
      // write-combined and strided writes into it are pathologically slow.
      staging_scratch_.resize(static_cast<std::size_t>(staged));
      stage_weights_column_major(op, staging_scratch_.data());
      std::memcpy(
        static_cast<std::uint8_t *>(mapped_weight_upload_) + staging_cursor,
        staging_scratch_.data(),
        static_cast<std::size_t>(staged));

      ResidentWeight resident;
      resident.bytes = staged;
      resident.state = D3D12_RESOURCE_STATE_COPY_DEST;
      std::uint64_t allocated = 0;
      if (!ensure_buffer(
        resident.buffer,
        staged,
        allocated,
        D3D12_HEAP_TYPE_DEFAULT,
        D3D12_RESOURCE_STATE_COPY_DEST,
        D3D12_RESOURCE_FLAG_NONE,
        error_message)) {
        return false;
      }
      raise_weight_residency_priority(resident.buffer.get());
      pending_uploads.push_back(PendingUpload {op.weight_key, staging_cursor, staged});
      weight_cache_bytes_ += staged;
      weight_cache_.emplace(op.weight_key, std::move(resident));
      staging_cursor += align_up(staged, 4);
    }

    // Until the copy below has executed on the GPU, newly registered weight
    // buffers hold garbage; drop them again if this call fails on any later
    // step so a future call cannot treat them as valid resident weights.
    const auto rollback_pending = [&]() {
      for (const PendingUpload &upload : pending_uploads) {
        auto iterator = weight_cache_.find(upload.weight_key);
        if (iterator != weight_cache_.end()) {
          weight_cache_bytes_ -= iterator->second.bytes;
          weight_cache_.erase(iterator);
        }
      }
    };

    const bool input_dirty = input_buffer_recreated
      || last_input_identity_ != input_identity
      || last_input_bytes_ != input_bytes;
    if (input_dirty
      && !copy_mapped_buffer(mapped_input_upload_, packed_inputs.data(), input_bytes, "input upload", error_message)) {
      rollback_pending();
      return false;
    }

    HRESULT hr = command_allocator_->Reset();
    if (FAILED(hr)) {
      error_message = hresult_message("ID3D12CommandAllocator::Reset", hr);
      rollback_pending();
      return false;
    }
    hr = command_list_->Reset(command_allocator_.get(), nullptr);
    if (FAILED(hr)) {
      error_message = hresult_message("ID3D12GraphicsCommandList::Reset", hr);
      rollback_pending();
      return false;
    }

    begin_stage_timestamps();
    resident_stamps = timestamps_active_;
    stamp_stage("resident-start");
    transition(output_default_.get(), output_state_, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    if (input_dirty) {
      transition(input_default_.get(), input_state_, D3D12_RESOURCE_STATE_COPY_DEST);
      command_list_->CopyBufferRegion(input_default_.get(), 0, input_upload_.get(), 0, input_bytes);
      transition(input_default_.get(), input_state_, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    }
    stamp_stage("resident-upload");
    for (const PendingUpload &upload : pending_uploads) {
      ResidentWeight &resident = weight_cache_.at(upload.weight_key);
      command_list_->CopyBufferRegion(
        resident.buffer.get(),
        0,
        weight_upload_.get(),
        upload.staging_offset,
        upload.staged_bytes);
      transition(resident.buffer.get(), resident.state, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    }

    command_list_->SetComputeRootSignature(root_signature_.get());
    // Prefill through the integer-dot GEMM when the opt-in gate is on and every
      // eligible op can use it. Activations are quantised ONCE here and shared by
    // all the ops in this call, which is what makes the trade worthwhile: the
    // quantisation is per input, the matmuls are per weight matrix.
    // Eligibility is PER OP, not all-or-nothing.
    //
    // It used to require every op in the call to qualify, which quietly cost the
    // most on the shape it matters for. Q4_K_M mixes formats within one fused
    // call: on llama-3.2-3b attn_v is Q6_K for 14 of 28 layers, and the integer
    // kernel only handles Q4_K, so a single ineligible V pushed Q and K onto the
    // float path too - and Q (3072 rows) plus K (1024) are 80% of the QKV FLOPs.
    // Half the model's QKV projections were getting no benefit from the gate at
    // all because of the other half of one call.
    //
    // The staged activations are per input and shared by every op here, so ops
    // can choose independently at no extra cost.
    bool prefill_integer = false;
    std::uint32_t prefill_activation_stride = 0;
    if (gpu_dp4a_prefill_enabled() && batch_count > 1) {
      for (std::size_t probe = 0; probe < op_count; probe += 1) {
        if (prefill_integer_eligible(ops[probe], batch_count, input_width)) {
          prefill_integer = true;
          break;
        }
      }
    }
    if (prefill_integer
      && !stage_prefill_activations(
        input_width, batch_count, prefill_activation_stride, error_message)) {
      // Staging failure is not fatal: fall back to the float kernels.
      prefill_integer = false;
    }
    std::uint64_t output_cursor = 0;
    use_counter_ += 1;
    for (std::size_t op_index = 0; op_index < op_count; op_index += 1) {
      const D3d12ResidentWeightRef &op = ops[op_index];
      ResidentWeight &resident = weight_cache_.at(op.weight_key);
      resident.last_use = use_counter_;

      DispatchConstants constants;
      constants.row_width = op.row_width;
      constants.row_count = op.row_count;
      constants.batch_count = batch_count;
      constants.output_base = static_cast<std::uint32_t>(output_cursor);
      constants.input_stride = input_width;
      constants.weight_layout = 1;
      const bool batch_one = batch_count == 1;
      ID3D12PipelineState *pipeline = batch_one ? scalar_b1_pipeline_.get() : scalar_pipeline_.get();
      switch (op.format) {
        case D3d12WeightFormat::Q4_K:
          pipeline = batch_one ? q4k_b1_pipeline_.get() : q4k_pipeline_.get();
          break;
        case D3d12WeightFormat::Q5_K:
          pipeline = batch_one ? q5k_b1_pipeline_.get() : q5k_pipeline_.get();
          break;
        case D3d12WeightFormat::Q6_K:
          pipeline = batch_one ? q6k_b1_pipeline_.get() : q6k_pipeline_.get();
          break;
        case D3d12WeightFormat::Q8_0:
          pipeline = batch_one ? q8_0_b1_pipeline_.get() : q8_0_pipeline_.get();
          break;
        case D3d12WeightFormat::Q4_0:
          pipeline = batch_one ? q4_0_b1_pipeline_.get() : q4_0_pipeline_.get();
          break;
        case D3d12WeightFormat::F32:
        case D3d12WeightFormat::F16:
        case D3d12WeightFormat::BF16:
          // Half formats are widened to F32 at staging; all three use the
          // stable F32 scalar lane.
          constants.scalar_format = 0;
          break;
      }

      if (prefill_integer
        && prefill_integer_eligible(op, batch_count, input_width)) {
        // Integer GEMM: 8 columns per group, one row per thread, activations
        // read from the staged Q8_1 buffer.
        DispatchConstants integer_constants = constants;
        integer_constants.input_stride = 0;
        command_list_->SetPipelineState(
          op.format == D3d12WeightFormat::Q6_K
            ? dp4a_q6k_batched_pipeline_.get()
            : dp4a_q4k_batched_pipeline_.get());
        command_list_->SetComputeRoot32BitConstants(0, 8, &integer_constants, 0);
        command_list_->SetComputeRootShaderResourceView(
          1, resident.buffer->GetGPUVirtualAddress());
        command_list_->SetComputeRootShaderResourceView(
          2, dp4a_activations_->GetGPUVirtualAddress());
        command_list_->SetComputeRootUnorderedAccessView(
          3, output_default_->GetGPUVirtualAddress());
        command_list_->Dispatch(
          static_cast<UINT>((op.row_count + 63u) / 64u),
          static_cast<UINT>((batch_count + 7u) / 8u),
          1);
        output_cursor += static_cast<std::uint64_t>(batch_count) * op.row_count;
        continue;
      }

      // Narrow-output k-split, ported from compute_layer_tail. This is the
      // measurement that motivated it: attn_out and Q are both 3072x3072 Q4_K,
      // and the tail runs attn_out at 14.4 GB/s while this lane ran Q at
      // 5.8 GB/s - same shape, same format, same bus, 2.5x apart. The only
      // difference was that the tail splits narrow-output matvecs four ways and
      // this lane did not. One thread per row caps occupancy at row_count, so a
      // 1-3k-row matvec leaves most of the machine idle; splitting k four ways
      // into ffn_activation_ and reducing gives the GPU four times the work to
      // schedule.
      //
      // ffn_activation_ is free here: nothing else in this lane touches it, and
      // the ksplit/reduce pair barriers around its own use.
      //
      // Gated to batch-one because the split kernel assumes a single input row,
      // and 4e-3-tolerance because the accumulation order changes.
      constexpr std::uint32_t k_split_count = 4;
      const bool use_ksplit = batch_one
        && ksplit_enabled()
        && op.format == D3d12WeightFormat::Q4_K
        && q4k_ksplit_pipeline_
        && ksplit_reduce_pipeline_
        && op.row_count <= 4096u
        && op.row_width >= 2048u
        && (op.row_width & 255u) == 0u
        && static_cast<std::uint64_t>(k_split_count) * op.row_count * sizeof(float)
          <= ffn_activation_bytes_;
      if (use_ksplit) {
        const auto uav_barrier = [&](ID3D12Resource *resource) {
          D3D12_RESOURCE_BARRIER barrier = {};
          barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
          barrier.UAV.pResource = resource;
          command_list_->ResourceBarrier(1, &barrier);
        };
        DispatchConstants split_constants = constants;
        split_constants.batch_count = k_split_count;
        split_constants.output_base = 0;
        transition(
          ffn_activation_.get(), ffn_activation_state_, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        command_list_->SetPipelineState(q4k_ksplit_pipeline_.get());
        command_list_->SetComputeRoot32BitConstants(0, 8, &split_constants, 0);
        command_list_->SetComputeRootShaderResourceView(1, resident.buffer->GetGPUVirtualAddress());
        command_list_->SetComputeRootShaderResourceView(2, input_default_->GetGPUVirtualAddress());
        command_list_->SetComputeRootUnorderedAccessView(3, ffn_activation_->GetGPUVirtualAddress());
        command_list_->Dispatch(static_cast<UINT>((op.row_count + 63u) / 64u), k_split_count, 1);
        uav_barrier(ffn_activation_.get());
        transition(
          ffn_activation_.get(),
          ffn_activation_state_,
          D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

        DispatchConstants reduce_constants;
        reduce_constants.row_width = op.row_width;
        reduce_constants.row_count = op.row_count;
        reduce_constants.batch_count = k_split_count;
        reduce_constants.output_base = static_cast<std::uint32_t>(output_cursor);
        reduce_constants.weight_layout = 0;
        command_list_->SetPipelineState(ksplit_reduce_pipeline_.get());
        command_list_->SetComputeRoot32BitConstants(0, 8, &reduce_constants, 0);
        command_list_->SetComputeRootShaderResourceView(1, resident.buffer->GetGPUVirtualAddress());
        command_list_->SetComputeRootShaderResourceView(2, ffn_activation_->GetGPUVirtualAddress());
        command_list_->SetComputeRootUnorderedAccessView(3, output_default_->GetGPUVirtualAddress());
        command_list_->Dispatch(static_cast<UINT>((op.row_count + 63u) / 64u), 1, 1);
      } else {
        // 128-rows-per-group kernel for batch-one, same as compute_layer_tail and
        // the token graph already use. Measured neutral on its own here (QKV
        // 1.295 -> 1.281 ms/call) because this lane's problem was occupancy, not
        // group width - kept because it is strictly the better kernel and it is
        // what runs for the shapes k-split declines.
        // Batched formats that row blocking declines fall back to the
        // 64-rows-per-group kernel: rowblock_pipeline_for takes only Q4_K, so the
        // Q6_K down projection runs the narrowest kernel available - measured 88
        // GFLOP/s against gate/up Q4_K's 229 at the same shape and batch, which is
        // 65% of the FFN chain's time for 33% of its FLOPs.
        //
        // wide128 canNOT be reused to close that: it stays batch-one on purpose.
        // Letting batched work take it was tried and the selftest rejected it
        // immediately (q4_k 1blk 128x256 b5: expected 109.07, actual 19.63, and
        // every batch-2 sample returning the same value) - its row indexing is
        // batch-one-specific, not merely a wider group. Closing this needs a
        // Q6_K row-blocked kernel, not a pipeline swap.
        ID3D12PipelineState *const wide128_pipeline =
          (batch_one && wide128_enabled()) ? wide128_pipeline_for(op.format) : nullptr;
        const bool use_wide128 = wide128_pipeline != nullptr;
        // Batched work here is prefill's QKV and attention-output projections
        // (374 ms and 220 ms of a 2197 ms prefill). Same row-blocking as the FFN
        // chain; declines itself for batch-one, so decode is untouched.
        ID3D12PipelineState *const rowblock = rowblock_pipeline_for(op, batch_one);
        const std::uint32_t row_block = rowblock != nullptr ? rowblock_rows_for(op.format) : 1u;
        command_list_->SetPipelineState(
          rowblock != nullptr ? rowblock : (use_wide128 ? wide128_pipeline : pipeline));
        command_list_->SetComputeRoot32BitConstants(0, 8, &constants, 0);
        command_list_->SetComputeRootShaderResourceView(1, resident.buffer->GetGPUVirtualAddress());
        command_list_->SetComputeRootShaderResourceView(2, input_default_->GetGPUVirtualAddress());
        command_list_->SetComputeRootUnorderedAccessView(3, output_default_->GetGPUVirtualAddress());
        const std::uint32_t dispatch_tile =
          rowblock != nullptr ? rowblock_tile_for(op.format) : k_gpu_batch_tile;
        command_list_->Dispatch(
          static_cast<UINT>(rowblock != nullptr
            ? (op.row_count + (64u * row_block) - 1u) / (64u * row_block)
            : use_wide128
              ? (op.row_count + 127u) / 128u
              : (op.row_count + 63u) / 64u),
          static_cast<UINT>(batch_one
            ? batch_count
            : (batch_count + dispatch_tile - 1u) / dispatch_tile),
          1);
      }

      output_cursor += static_cast<std::uint64_t>(batch_count) * op.row_count;
      // One stamp per op: on the QKV call this separates Q (3072 rows) from
      // K and V (1024 each), which is exactly the occupancy question.
      stamp_stage(op_index == 0 ? "resident-op0"
        : op_index == 1 ? "resident-op1"
        : "resident-op2plus");
    }

    if (!uma_output_) {
      transition(output_default_.get(), output_state_, D3D12_RESOURCE_STATE_COPY_SOURCE);
      command_list_->CopyBufferRegion(output_readback_.get(), 0, output_default_.get(), 0, output_bytes);
    }

    if (resident_stamps && timestamp_labels_.size() >= 2) {
      command_list_->ResolveQueryData(
        timestamp_heap_.get(),
        D3D12_QUERY_TYPE_TIMESTAMP,
        0,
        static_cast<UINT>(timestamp_labels_.size()),
        timestamp_readback_.get(),
        0);
    }
    hr = command_list_->Close();
    if (FAILED(hr)) {
      error_message = hresult_message("ID3D12GraphicsCommandList::Close", hr);
      rollback_pending();
      return false;
    }
    const auto resident_recorded_at = std::chrono::steady_clock::now();
    if (!submit_and_wait(error_message)) {
      rollback_pending();
      return false;
    }
    const auto resident_waited_at = std::chrono::steady_clock::now();
    telemetry_.weight_upload_bytes += missing_resident_bytes;
    telemetry_.input_upload_bytes += input_dirty ? input_bytes : 0;
    telemetry_.output_readback_bytes += output_bytes;

    const void *result_memory = uma_output_ ? mapped_output_direct_ : mapped_output_readback_;
    if (result_memory == nullptr) {
      error_message = "D3D12 output buffer is not mapped.";
      return false;
    }
    const float *output_data = static_cast<const float *>(result_memory);
    output_cursor = 0;
    for (std::size_t op_index = 0; op_index < op_count; op_index += 1) {
      const std::uint64_t op_floats = static_cast<std::uint64_t>(batch_count) * ops[op_index].row_count;
      outputs[op_index].resize(static_cast<std::size_t>(op_floats));
      std::memcpy(
        outputs[op_index].data(),
        output_data + output_cursor,
        static_cast<std::size_t>(op_floats) * sizeof(float));
      output_cursor += op_floats;
    }

    if (input_dirty) {
      last_input_identity_ = input_identity;
      last_input_bytes_ = input_bytes;
    }
    if (resident_stamps) {
      accumulate_timestamps();
    }
    if (gpu_timestamps_enabled()) {
      const auto resident_done_at = std::chrono::steady_clock::now();
      resident_record_ms_ += std::chrono::duration<double, std::milli>(
        resident_recorded_at - resident_entry_at).count();
      resident_wait_ms_ += std::chrono::duration<double, std::milli>(
        resident_waited_at - resident_recorded_at).count();
      resident_tail_ms_ += std::chrono::duration<double, std::milli>(
        resident_done_at - resident_waited_at).count();
      resident_phase_calls_ += 1;
    }
    return true;
  }

  bool compute_legacy(
    const std::vector<float> &weights,
    std::uint32_t row_count,
    std::uint32_t row_width,
    const std::vector<float> &packed_inputs,
    std::uint32_t batch_count,
    const void *input_identity,
    std::vector<std::vector<float>> &outputs,
    std::string &error_message) {
    if (!capabilities_.supported) {
      error_message = capabilities_.reason;
      return false;
    }
    if (row_count == 0 || row_width == 0 || batch_count == 0 || packed_inputs.empty()) {
      error_message = "D3D12 batch matmul requires non-empty weights and inputs.";
      return false;
    }
    const std::size_t expected_weight_items =
      static_cast<std::size_t>(row_count) * static_cast<std::size_t>(row_width);
    if (weights.size() != expected_weight_items) {
      error_message = "D3D12 batch matmul weight payload size does not match the requested shape.";
      return false;
    }
    const std::size_t input_items = static_cast<std::size_t>(batch_count) * static_cast<std::size_t>(row_width);
    if (packed_inputs.size() != input_items) {
      error_message = "D3D12 batch matmul packed input payload size does not match the requested shape.";
      return false;
    }

    const std::uint64_t weight_bytes = static_cast<std::uint64_t>(weights.size()) * sizeof(float);
    const std::uint64_t input_bytes = static_cast<std::uint64_t>(input_items) * sizeof(float);
    const std::uint64_t output_floats =
      static_cast<std::uint64_t>(batch_count) * static_cast<std::uint64_t>(row_count);
    const std::uint64_t output_bytes = output_floats * sizeof(float);
    const bool input_buffer_recreated = !input_default_ || input_default_bytes_ < input_bytes;

    if (!ensure_persistent_upload_buffer(
      weight_upload_, weight_bytes, weight_upload_bytes_, mapped_weight_upload_, error_message)) {
      return false;
    }
    if (!ensure_persistent_upload_buffer(
      input_upload_, input_bytes, input_upload_bytes_, mapped_input_upload_, error_message)) {
      return false;
    }
    if (!ensure_srv_buffer(
      legacy_weight_default_, weight_bytes, legacy_weight_default_bytes_, legacy_weight_state_, error_message)) {
      return false;
    }
    if (!ensure_srv_buffer(input_default_, input_bytes, input_default_bytes_, input_state_, error_message)) {
      return false;
    }
    if (!ensure_output_buffer(output_default_, output_bytes, output_default_bytes_, output_state_, error_message)) {
      return false;
    }
    if (!uma_output_
      && !ensure_persistent_readback_buffer(
        output_readback_, output_bytes, output_readback_bytes_, mapped_output_readback_, error_message)) {
      return false;
    }

    if (!copy_mapped_buffer(mapped_weight_upload_, weights.data(), weight_bytes, "weight upload", error_message)) {
      return false;
    }
    const bool input_dirty = input_buffer_recreated
      || last_input_identity_ != input_identity
      || last_input_bytes_ != input_bytes;
    if (input_dirty
      && !copy_mapped_buffer(mapped_input_upload_, packed_inputs.data(), input_bytes, "input upload", error_message)) {
      return false;
    }

    HRESULT hr = command_allocator_->Reset();
    if (FAILED(hr)) {
      error_message = hresult_message("ID3D12CommandAllocator::Reset", hr);
      return false;
    }
    ID3D12PipelineState *legacy_pipeline = batch_count == 1
      ? scalar_b1_pipeline_.get()
      : scalar_pipeline_.get();
    hr = command_list_->Reset(command_allocator_.get(), legacy_pipeline);
    if (FAILED(hr)) {
      error_message = hresult_message("ID3D12GraphicsCommandList::Reset", hr);
      return false;
    }

    transition(output_default_.get(), output_state_, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    transition(legacy_weight_default_.get(), legacy_weight_state_, D3D12_RESOURCE_STATE_COPY_DEST);
    command_list_->CopyBufferRegion(legacy_weight_default_.get(), 0, weight_upload_.get(), 0, weight_bytes);
    transition(legacy_weight_default_.get(), legacy_weight_state_, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    if (input_dirty) {
      transition(input_default_.get(), input_state_, D3D12_RESOURCE_STATE_COPY_DEST);
      command_list_->CopyBufferRegion(input_default_.get(), 0, input_upload_.get(), 0, input_bytes);
      transition(input_default_.get(), input_state_, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    }

    DispatchConstants constants;
    constants.row_width = row_width;
    constants.row_count = row_count;
    constants.batch_count = batch_count;
    constants.output_base = 0;
    constants.scalar_format = 0;
    constants.input_stride = row_width;
    constants.weight_layout = 0;

    command_list_->SetComputeRootSignature(root_signature_.get());
    command_list_->SetPipelineState(legacy_pipeline);
    command_list_->SetComputeRoot32BitConstants(0, 8, &constants, 0);
    command_list_->SetComputeRootShaderResourceView(1, legacy_weight_default_->GetGPUVirtualAddress());
    command_list_->SetComputeRootShaderResourceView(2, input_default_->GetGPUVirtualAddress());
    command_list_->SetComputeRootUnorderedAccessView(3, output_default_->GetGPUVirtualAddress());
    command_list_->Dispatch(
      static_cast<UINT>((row_count + 63u) / 64u),
      static_cast<UINT>(batch_count == 1
        ? batch_count
        : (batch_count + k_gpu_batch_tile - 1u) / k_gpu_batch_tile),
      1);

    if (!uma_output_) {
      transition(output_default_.get(), output_state_, D3D12_RESOURCE_STATE_COPY_SOURCE);
      command_list_->CopyBufferRegion(output_readback_.get(), 0, output_default_.get(), 0, output_bytes);
    }

    hr = command_list_->Close();
    if (FAILED(hr)) {
      error_message = hresult_message("ID3D12GraphicsCommandList::Close", hr);
      return false;
    }
    if (!submit_and_wait(error_message)) {
      return false;
    }

    const void *result_memory = uma_output_ ? mapped_output_direct_ : mapped_output_readback_;
    if (result_memory == nullptr) {
      error_message = "D3D12 output buffer is not mapped.";
      return false;
    }
    outputs.assign(batch_count, std::vector<float>(row_count, 0.0f));
    const float *output_data = static_cast<const float *>(result_memory);
    for (std::uint32_t batch_index = 0; batch_index < batch_count; batch_index += 1) {
      std::memcpy(
        outputs[batch_index].data(),
        output_data + static_cast<std::size_t>(batch_index) * static_cast<std::size_t>(row_count),
        static_cast<std::size_t>(row_count) * sizeof(float));
    }
    if (input_dirty) {
      last_input_identity_ = input_identity;
      last_input_bytes_ = input_bytes;
    }
    return true;
  }

  // Uploads one tensor into the resident cache without dispatching any
  // compute. Used to pre-stage weights at model load so the first request
  // does not pay the cold staging cost. Refuses to evict: when the budget is
  // full the remaining tensors simply stage lazily on first use.
  // Phase F2: pin resident weight heaps at high residency priority so the
  // OS does not demote them when another process (measured with Ollama's
  // multi-GB working set) pressures the shared UMA pool - demotion showed
  // up as a 4.1 -> 3.7 tok/s decode drop right after an Ollama run.
  // Best-effort: default ON, LIZARD_NATIVE_GPU_RESIDENCY_PRIORITY=0
  // disables; silently skipped when ID3D12Device1 is unavailable.
  void raise_weight_residency_priority(ID3D12Resource *resource) {
    static const bool enabled = [] {
      char value[8] = {};
      return GetEnvironmentVariableA(
          "LIZARD_NATIVE_GPU_RESIDENCY_PRIORITY", value, sizeof(value)) == 0
        || value[0] != '0';
    }();
    if (!enabled || resource == nullptr || !device1_) {
      return;
    }
    ID3D12Pageable *pageable = resource;
    const D3D12_RESIDENCY_PRIORITY priority = D3D12_RESIDENCY_PRIORITY_HIGH;
    device1_->SetResidencyPriority(1, &pageable, &priority);
  }

  bool stage_resident(const D3d12ResidentWeightRef &op, std::string &error_message) {
    if (!capabilities_.supported) {
      error_message = capabilities_.reason;
      return false;
    }
    if (op.weight_key == nullptr || op.weight_data == nullptr || op.row_count == 0 || op.row_width == 0) {
      error_message = "D3D12 resident staging received an invalid weight reference.";
      return false;
    }
    if (weight_cache_.find(op.weight_key) != weight_cache_.end()) {
      telemetry_.weight_cache_hits += 1;
      return true;
    }
    telemetry_.weight_cache_misses += 1;
    if ((op.format == D3d12WeightFormat::Q4_K || op.format == D3d12WeightFormat::Q5_K || op.format == D3d12WeightFormat::Q6_K)
      && (op.row_width % k_quant_block_elements) != 0) {
      error_message = "D3D12 quantized staging requires the row width to be a multiple of 256.";
      return false;
    }
    if ((op.format == D3d12WeightFormat::Q8_0 || op.format == D3d12WeightFormat::Q4_0)
      && (op.row_width % k_legacy_block_elements) != 0) {
      error_message = "D3D12 legacy-quant staging requires the row width to be a multiple of 32.";
      return false;
    }
    const std::uint64_t expected_bytes = expected_weight_bytes(op.format, op.row_count, op.row_width);
    if (expected_bytes == 0 || op.weight_bytes != expected_bytes) {
      error_message = "D3D12 resident staging payload does not match the requested shape.";
      return false;
    }
    const std::uint64_t staged = staged_weight_bytes(op.format, op.weight_bytes);
    if (staged > k_max_staged_tensor_bytes) {
      error_message = "D3D12 resident staging skipped a tensor above the per-tensor size cap.";
      return false;
    }
    if (weight_cache_bytes_ + staged > weight_budget_bytes_) {
      // The budget is live: another application may have released memory
      // since the last refresh (or the static request may now be safe to
      // grant). One refreshed re-check before giving up.
      refresh_weight_budget();
      if (weight_cache_bytes_ + staged > weight_budget_bytes_) {
        // At DECODE time, evict the least-recently-used resident weights to make
        // room and keep decode on the D3D12 lane - by then the GPU lane is
        // already committed, so evicting is the only alternative to failing the
        // request. evict_for protects nothing extra here (op_count 0), but LRU
        // order means the just-staged siblings of the current layer are the
        // newest and are never chosen as victims.
        //
        // Only the explicit load-time pre-stage keeps the historical
        // "stage what fits, the rest stage lazily on first use" behavior, so it
        // never churns the cache before decode has even started. This must NOT
        // be tied to the plan's "streaming" classification: a model can sit
        // under the plan's static VRAM-derived budget yet still overflow the
        // much smaller LIVE DXGI budget at runtime (measured: a 0.75 GB model
        // on a 2 GB card whose live budget was only 874 MB).
        if (!prestaging_) {
          if (!evict_for(staged, nullptr, 0, error_message)) {
            return false;
          }
        } else {
          error_message = "D3D12 resident staging stopped at the configured GPU weight budget.";
          return false;
        }
      }
    }

    if (!ensure_persistent_upload_buffer(
      weight_upload_, staged, weight_upload_bytes_, mapped_weight_upload_, error_message)) {
      return false;
    }
    staging_scratch_.resize(static_cast<std::size_t>(staged));
    stage_weights_column_major(op, staging_scratch_.data());
    std::memcpy(mapped_weight_upload_, staging_scratch_.data(), static_cast<std::size_t>(staged));

    ResidentWeight resident;
    resident.bytes = staged;
    resident.state = D3D12_RESOURCE_STATE_COPY_DEST;
    std::uint64_t allocated = 0;
    if (!ensure_buffer(
      resident.buffer,
      staged,
      allocated,
      D3D12_HEAP_TYPE_DEFAULT,
      D3D12_RESOURCE_STATE_COPY_DEST,
      D3D12_RESOURCE_FLAG_NONE,
      error_message)) {
      return false;
    }

    HRESULT hr = command_allocator_->Reset();
    if (FAILED(hr)) {
      error_message = hresult_message("ID3D12CommandAllocator::Reset", hr);
      return false;
    }
    hr = command_list_->Reset(command_allocator_.get(), nullptr);
    if (FAILED(hr)) {
      error_message = hresult_message("ID3D12GraphicsCommandList::Reset", hr);
      return false;
    }
    command_list_->CopyBufferRegion(resident.buffer.get(), 0, weight_upload_.get(), 0, staged);
    transition(resident.buffer.get(), resident.state, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    hr = command_list_->Close();
    if (FAILED(hr)) {
      error_message = hresult_message("ID3D12GraphicsCommandList::Close", hr);
      return false;
    }
    if (!submit_and_wait(error_message)) {
      return false;
    }

    use_counter_ += 1;
    resident.last_use = use_counter_;
    raise_weight_residency_priority(resident.buffer.get());
    weight_cache_bytes_ += staged;
    weight_cache_.emplace(op.weight_key, std::move(resident));
    telemetry_.weight_upload_bytes += staged;
    return true;
  }

  // Releases the (possibly very large) staging upload buffer after a bulk
  // pre-stage pass; it is recreated on demand at request-time sizes.
  void shrink_staging_buffers() {
    release_mapped_resource(weight_upload_, mapped_weight_upload_);
    weight_upload_bytes_ = 0;
  }

  // B6 diagnostics: GPU timestamp query heap + readback for the token
  // graph. Created lazily only when LIZARD_NATIVE_GPU_TIMESTAMPS=1.
  bool ensure_timestamp_resources() {
    if (timestamp_heap_ && timestamp_readback_) {
      return true;
    }
    const std::uint32_t capacity = 1024;
    D3D12_QUERY_HEAP_DESC heap_desc = {};
    heap_desc.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
    heap_desc.Count = capacity;
    ID3D12QueryHeap *heap_raw = nullptr;
    if (FAILED(device_->CreateQueryHeap(&heap_desc, IID_PPV_ARGS(&heap_raw)))) {
      return false;
    }
    timestamp_heap_.reset(heap_raw);
    D3D12_HEAP_PROPERTIES heap_properties = {};
    heap_properties.Type = D3D12_HEAP_TYPE_READBACK;
    D3D12_RESOURCE_DESC buffer_desc = {};
    buffer_desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    buffer_desc.Width = static_cast<std::uint64_t>(capacity) * sizeof(std::uint64_t);
    buffer_desc.Height = 1;
    buffer_desc.DepthOrArraySize = 1;
    buffer_desc.MipLevels = 1;
    buffer_desc.SampleDesc.Count = 1;
    buffer_desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    ID3D12Resource *readback_raw = nullptr;
    if (FAILED(device_->CreateCommittedResource(
        &heap_properties,
        D3D12_HEAP_FLAG_NONE,
        &buffer_desc,
        D3D12_RESOURCE_STATE_COPY_DEST,
        nullptr,
        IID_PPV_ARGS(&readback_raw)))) {
      timestamp_heap_.reset();
      return false;
    }
    timestamp_readback_.reset(readback_raw);
    timestamp_capacity_ = capacity;
    return true;
  }

  // Reads back the stamps recorded for one graph, accumulates per-label
  // Stage stamping for command lists recorded OUTSIDE the token graph. The
  // token graph has its own local `stamp` lambda; the fused layer tail - which
  // is what actually records decode on the Llama path - had no stamping at all,
  // so enabling LIZARD_NATIVE_GPU_TIMESTAMPS produced an empty file and no way
  // to tell which dispatch inside the tail owns the time. Same query heap, same
  // label convention (the label names the stage ENDING at that stamp).
  bool timestamps_active_ = false;

  void begin_stage_timestamps() {
    timestamps_active_ = gpu_timestamps_enabled() && ensure_timestamp_resources();
    timestamp_labels_.clear();
  }

  void stamp_stage(const char *label) {
    if (!timestamps_active_ || timestamp_labels_.size() >= timestamp_capacity_) {
      return;
    }
    command_list_->EndQuery(
      timestamp_heap_.get(),
      D3D12_QUERY_TYPE_TIMESTAMP,
      static_cast<UINT>(timestamp_labels_.size()));
    timestamp_labels_.push_back(label);
  }

  // GPU milliseconds, and appends a summary to lizard-gpu-timestamps.log
  // every 50 graphs.
  void accumulate_timestamps() {
    if (timestamp_labels_.size() < 2 || !timestamp_readback_) {
      return;
    }
    std::uint64_t frequency = 0;
    if (FAILED(command_queue_->GetTimestampFrequency(&frequency)) || frequency == 0) {
      return;
    }
    void *mapped = nullptr;
    const D3D12_RANGE read_range = {
      0, timestamp_labels_.size() * sizeof(std::uint64_t)};
    if (FAILED(timestamp_readback_->Map(0, &read_range, &mapped))) {
      return;
    }
    const std::uint64_t *ticks = static_cast<const std::uint64_t *>(mapped);
    static std::map<std::string, double> totals;
    static std::uint64_t graph_count = 0;
    for (std::size_t index = 1; index < timestamp_labels_.size(); index += 1) {
      const double milliseconds = ticks[index] >= ticks[index - 1]
        ? static_cast<double>(ticks[index] - ticks[index - 1]) * 1000.0
          / static_cast<double>(frequency)
        : 0.0;
      totals[timestamp_labels_[index]] += milliseconds;
    }
    const D3D12_RANGE empty_range = {0, 0};
    timestamp_readback_->Unmap(0, &empty_range);
    graph_count += 1;
    // Every 10, plus the first. At 50 a short run - a 20-token completion is
    // well under 50 graphs - wrote nothing, and no output reads as "the GPU
    // path never ran" rather than "the run was shorter than the reporting
    // interval". The CPU-side profile had exactly this defect and it cost a
    // full measurement cycle; same fix, same reason.
    if (graph_count == 1 || graph_count % 10 == 0) {
      // Absolute path, beside the CPU profile. A relative name lands in
      // whatever cwd the host happened to spawn the worker with, which is not
      // reliably anywhere the caller thinks to look.
      std::string log_path = "lizard-gpu-timestamps.log";
      if (const char *const home = std::getenv("USERPROFILE")) {
        log_path = std::string(home) + "\\.lizard-llm\\lizard-gpu-timestamps.log";
      }
      if (std::FILE *log = std::fopen(log_path.c_str(), "a")) {
        std::fprintf(log, "[gpu-timestamps] graphs=%llu\n",
          static_cast<unsigned long long>(graph_count));
        double total = 0.0;
        for (const auto &entry : totals) {
          total += entry.second;
        }
        for (const auto &entry : totals) {
          std::fprintf(log, "  %-18s %10.1f ms total  %7.2f ms/graph  %5.1f%%\n",
            entry.first.c_str(),
            entry.second,
            entry.second / static_cast<double>(graph_count),
            total > 0.0 ? entry.second * 100.0 / total : 0.0);
        }
        if (resident_phase_calls_ > 0) {
          std::fprintf(log,
            "  [resident] record %.3f ms  wait %.3f ms  readback %.3f ms  (%llu calls)\n",
            resident_record_ms_ / static_cast<double>(resident_phase_calls_),
            resident_wait_ms_ / static_cast<double>(resident_phase_calls_),
            resident_tail_ms_ / static_cast<double>(resident_phase_calls_),
            static_cast<unsigned long long>(resident_phase_calls_));
        }
        if (cpu_phase_graphs_ > 0) {
          std::fprintf(log,
            "  [cpu] record %.2f ms/graph  wait %.2f ms/graph  tail %.2f ms/graph  (%llu graphs)\n",
            cpu_record_ms_ / static_cast<double>(cpu_phase_graphs_),
            cpu_wait_ms_ / static_cast<double>(cpu_phase_graphs_),
            cpu_tail_ms_ / static_cast<double>(cpu_phase_graphs_),
            static_cast<unsigned long long>(cpu_phase_graphs_));
        }
        std::fclose(log);
      }
    }
  }

  // CPU-side wall time of the token graph's phases (record = everything
  // before Close, wait = Close+Execute+fence, tail = readbacks/bookkeeping).
  // Attribution for the gap between GPU timestamp totals and wall
  // ms/token; accumulated only when LIZARD_NATIVE_GPU_TIMESTAMPS=1.
  double cpu_record_ms_ = 0.0;
  double cpu_wait_ms_ = 0.0;
  double cpu_tail_ms_ = 0.0;
  // Same three-way split for compute_resident, the lane the Llama QKV
  // projection goes through. Its 1.250 ms/call carries 0.84 ms of fixed cost
  // and the next thing to establish is which of record / wait / readback owns
  // it - separate counters because averaging it in with the layer tail's would
  // hide exactly the difference being looked for.
  double resident_record_ms_ = 0.0;
  double resident_wait_ms_ = 0.0;
  double resident_tail_ms_ = 0.0;
  std::uint64_t resident_phase_calls_ = 0;
  std::chrono::steady_clock::time_point cpu_tail_pending_ {};
  std::uint64_t cpu_phase_graphs_ = 0;

  bool compute_gemma_token(
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
    std::string &error_message) {
    if (!capabilities_.supported) {
      error_message = capabilities_.reason;
      return false;
    }
    const auto cpu_phase_entry = std::chrono::steady_clock::now();
    const std::uint64_t submission_count_at_entry = telemetry_.submission_count;
    if (!ensure_gemma_attention_pipelines(error_message)) {
      return false;
    }
    const std::uint32_t hidden_width = static_cast<std::uint32_t>(initial_hidden.size());
    const std::uint32_t vocab_size = output.row_count;
    const bool has_global_ple = ple_inputs != nullptr;
    const std::uint64_t expected_ple_items = has_global_ple
      ? static_cast<std::uint64_t>(ple_inputs->layer_width) * layers.size()
      : 0;
    // logits_out may be null: greedy sampling only needs the GPU argmax, so
    // callers skip the vocab-sized logits readback (about 1 MB per token).
    if (hidden_width == 0
      || layers.empty()
      || output.row_width != hidden_width
      || output_norm.size() != hidden_width
      || hidden_out == nullptr
      || argmax_token == nullptr
      || (has_global_ple
        && (ple_inputs->token_inputs == nullptr
          || ple_inputs->projection_norm == nullptr
          || ple_inputs->layer_width == 0
          || ple_inputs->model_projection.row_width != hidden_width
          || ple_inputs->model_projection.row_count != expected_ple_items
          || ple_inputs->token_inputs->size() != expected_ple_items
          || ple_inputs->projection_norm->size() != ple_inputs->layer_width))) {
      error_message = "D3D12 Gemma token graph received inconsistent output shapes.";
      return false;
    }

    struct PackedLayer {
      std::uint32_t attention_norm = 0;
      std::uint32_t post_attention_norm = 0;
      std::uint32_t ffn_norm = 0;
      std::uint32_t post_ffw_norm = 0;
      std::uint32_t ple_input = 0;
      std::uint32_t ple_post_norm = 0;
      std::uint64_t attention_config_upload = 0;
      std::uint64_t attention_config_bytes = 0;
      std::uint64_t initial_kv_upload = 0;
      std::uint64_t initial_kv_bytes = 0;
      DeviceKvCache *cache = nullptr;
      bool new_cache = false;
    };
    std::vector<PackedLayer> packed_layers(layers.size());
    std::vector<float> packed_constants;
    packed_constants.reserve(
      static_cast<std::size_t>(layers.size()) * hidden_width * 5u + output_norm.size());
    const auto append_constants = [&](const std::vector<float> &values) {
      const std::uint32_t offset = static_cast<std::uint32_t>(packed_constants.size());
      packed_constants.insert(packed_constants.end(), values.begin(), values.end());
      return offset;
    };
    std::uint32_t ple_projection_norm_offset = 0;
    std::uint32_t ple_token_inputs_offset = 0;
    if (has_global_ple) {
      if (!stage_resident(ple_inputs->model_projection, error_message)) {
        return false;
      }
      ple_projection_norm_offset =
        append_constants(*ple_inputs->projection_norm);
      ple_token_inputs_offset = append_constants(*ple_inputs->token_inputs);
    }

    std::uint32_t max_query_size = 0;
    std::uint32_t max_kv_size = 0;
    std::uint32_t max_ffn_width = 0;
    std::uint32_t max_ple_width = 0;
    std::uint32_t max_attention_config_floats = 0;
    std::unordered_set<std::uint64_t> caches_appended;
    std::vector<std::vector<float>> attention_configs(layers.size());

    if (!stage_resident(output, error_message)) {
      return false;
    }
    for (std::size_t index = 0; index < layers.size(); index += 1) {
      const D3d12GemmaTokenLayer &layer = layers[index];
      const bool has_ple = layer.ple_gate.weight_key != nullptr
        && layer.ple_projection.weight_key != nullptr
        && layer.ple_post_norm != nullptr;
      const std::uint32_t query_size =
        layer.attention.head_count * layer.attention.head_dim;
      const std::uint32_t kv_size =
        layer.attention.kv_head_count * layer.attention.head_dim;
      if (layer.attention_norm == nullptr
        || layer.query_norm == nullptr
        || layer.key_norm == nullptr
        || layer.rope_freq_factors == nullptr
        || !layer.initial_keys.populated
        || !layer.initial_values.populated
        || layer.post_attention_norm == nullptr
        || layer.ffn_norm == nullptr
        || layer.post_ffw_norm == nullptr
        || layer.attention_norm->size() != hidden_width
        || layer.query_norm->size() != layer.attention.head_dim
        || layer.key_norm->size() != layer.attention.head_dim
        || layer.post_attention_norm->size() != hidden_width
        || layer.ffn_norm->size() != hidden_width
        || layer.post_ffw_norm->size() != hidden_width
        || layer.attention.cache_id == 0
        || layer.attention.max_positions == 0
        || layer.attention.position >= layer.attention.max_positions
        || layer.attention.head_dim == 0
        || layer.attention.head_dim > 512
        || (layer.attention.head_dim & 1u) != 0u
        || layer.query.row_width != hidden_width
        || layer.query.row_count != query_size
        || (layer.owns_kv_cache
          && (layer.key.row_width != hidden_width
            || layer.value.row_width != hidden_width
            || layer.key.row_count != kv_size
            || layer.value.row_count != kv_size))
        || layer.attn_out.row_width != query_size
        || layer.attn_out.row_count != hidden_width
        || layer.gate.row_width != hidden_width
        || layer.up.row_width != hidden_width
        || layer.gate.row_count != layer.up.row_count
        || layer.down.row_width != layer.gate.row_count
        || layer.down.row_count != hidden_width
        || ((layer.ple_gate.weight_key != nullptr) != has_ple)
        || ((layer.ple_projection.weight_key != nullptr) != has_ple)
        || ((layer.ple_post_norm != nullptr) != has_ple)
        || (has_ple != has_global_ple)
        || (has_ple
          && (layer.ple_gate.row_width != hidden_width
            || layer.ple_projection.row_width != layer.ple_gate.row_count
            || layer.ple_projection.row_count != hidden_width
            || layer.ple_gate.row_count != ple_inputs->layer_width
            || layer.ple_post_norm->size() != hidden_width))) {
        error_message = "D3D12 Gemma token graph layer shapes are inconsistent.";
        return false;
      }
      if (!stage_resident(layer.query, error_message)
        || (layer.owns_kv_cache
          && (!stage_resident(layer.key, error_message)
            || !stage_resident(layer.value, error_message)))
        || !stage_resident(layer.attn_out, error_message)
        || !stage_resident(layer.gate, error_message)
        || !stage_resident(layer.up, error_message)
        || !stage_resident(layer.down, error_message)
        || (has_ple
          && (!stage_resident(layer.ple_gate, error_message)
            || !stage_resident(layer.ple_projection, error_message)))) {
        return false;
      }

      PackedLayer &packed = packed_layers[index];
      packed.attention_norm = append_constants(*layer.attention_norm);
      packed.post_attention_norm = append_constants(*layer.post_attention_norm);
      packed.ffn_norm = append_constants(*layer.ffn_norm);
      packed.post_ffw_norm = append_constants(*layer.post_ffw_norm);
      if (has_ple) {
        packed.ple_post_norm = append_constants(*layer.ple_post_norm);
      }

      std::vector<float> &attention_config = attention_configs[index];
      attention_config.reserve(3u + layer.attention.head_dim * 5u / 2u);
      attention_config.push_back(layer.attention.rms_epsilon);
      attention_config.push_back(layer.attention.rope_freq_base);
      attention_config.push_back(layer.attention.rope_freq_scale);
      attention_config.insert(
        attention_config.end(), layer.query_norm->begin(), layer.query_norm->end());
      attention_config.insert(
        attention_config.end(), layer.key_norm->begin(), layer.key_norm->end());
      for (std::uint32_t factor = 0; factor < layer.attention.head_dim / 2u; factor += 1) {
        attention_config.push_back(
          factor < layer.rope_freq_factors->size()
            && (*layer.rope_freq_factors)[factor] > 0.0f
          ? (*layer.rope_freq_factors)[factor]
          : 1.0f);
      }
      max_attention_config_floats = std::max<std::uint32_t>(
        max_attention_config_floats,
        static_cast<std::uint32_t>(attention_config.size()));

      const std::uint32_t words_per_position = (kv_size + 1u) / 2u;
      const std::uint64_t cache_words =
        static_cast<std::uint64_t>(layer.attention.max_positions) * words_per_position * 2u;
      auto cache_iterator = gemma_kv_cache_.find(layer.attention.cache_id);
      packed.new_cache = cache_iterator == gemma_kv_cache_.end();
      if (packed.new_cache) {
        if (layer.initial_keys.size()
            != static_cast<std::size_t>(layer.attention.position) * kv_size
          || layer.initial_values.size()
            != static_cast<std::size_t>(layer.attention.position) * kv_size) {
          error_message = "D3D12 Gemma token graph initial KV payload is inconsistent.";
          return false;
        }
        DeviceKvCache cache;
        cache.max_positions = layer.attention.max_positions;
        cache.kv_size = kv_size;
        cache.words_per_position = words_per_position;
        cache.positions = layer.attention.position;
        std::uint64_t allocated = 0;
        if (!ensure_buffer(
          cache.buffer,
          cache_words * sizeof(std::uint32_t),
          allocated,
          D3D12_HEAP_TYPE_DEFAULT,
          D3D12_RESOURCE_STATE_COPY_DEST,
          D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
          error_message)) {
          return false;
        }
        cache.state = D3D12_RESOURCE_STATE_COPY_DEST;
        cache_iterator =
          gemma_kv_cache_.emplace(layer.attention.cache_id, std::move(cache)).first;
      }
      DeviceKvCache &cache = cache_iterator->second;
      if (cache.max_positions != layer.attention.max_positions
        || cache.kv_size != kv_size
        || cache.words_per_position != words_per_position) {
        error_message = "D3D12 Gemma token graph KV cache shape changed.";
        return false;
      }
      const bool owner_already_recorded =
        caches_appended.find(layer.attention.cache_id) != caches_appended.end();
      const std::uint32_t expected_positions = layer.owns_kv_cache
        ? layer.attention.position
        : layer.attention.position + (owner_already_recorded ? 0u : 1u);
      if (cache.positions != expected_positions) {
        error_message = "D3D12 Gemma token graph KV cache position is inconsistent.";
        return false;
      }
      if (layer.owns_kv_cache) {
        caches_appended.insert(layer.attention.cache_id);
      }
      packed.cache = &cache;
      max_query_size = std::max(max_query_size, query_size);
      max_kv_size = std::max(max_kv_size, kv_size);
      max_ffn_width = std::max(max_ffn_width, layer.gate.row_count);
      max_ple_width = std::max(
        max_ple_width, has_ple ? layer.ple_gate.row_count : 0u);
    }
    const std::uint32_t output_norm_offset = append_constants(output_norm);

    const std::uint64_t hidden_bytes =
      static_cast<std::uint64_t>(hidden_width) * sizeof(float);
    const std::uint64_t constants_bytes =
      static_cast<std::uint64_t>(packed_constants.size()) * sizeof(float);
    const std::uint64_t hidden_upload_offset = 0;
    const std::uint64_t constants_upload_offset = align_up(hidden_bytes, 256);
    std::uint64_t next_upload_offset =
      align_up(constants_upload_offset + constants_bytes, 256);
    for (std::size_t index = 0; index < layers.size(); index += 1) {
      PackedLayer &packed = packed_layers[index];
      packed.attention_config_upload = next_upload_offset;
      packed.attention_config_bytes =
        attention_configs[index].size() * sizeof(float);
      next_upload_offset =
        align_up(next_upload_offset + packed.attention_config_bytes, 256);
      if (packed.new_cache && layers[index].attention.position > 0) {
        const std::uint64_t populated_halves =
          static_cast<std::uint64_t>(layers[index].attention.position)
          * packed.cache->kv_size * 2u;
        packed.initial_kv_upload = next_upload_offset;
        packed.initial_kv_bytes =
          ((populated_halves + 1u) / 2u) * sizeof(std::uint32_t);
        next_upload_offset =
          align_up(next_upload_offset + packed.initial_kv_bytes, 256);
      }
    }
    const std::uint64_t upload_bytes = std::max<std::uint64_t>(next_upload_offset, 4);
    const std::uint64_t qkv_bytes =
      static_cast<std::uint64_t>(max_query_size + max_kv_size * 2u) * sizeof(float);
    const std::uint64_t query_bytes =
      static_cast<std::uint64_t>(max_query_size) * sizeof(float);
    const std::uint64_t scratch_ab_bytes = hidden_bytes * 2u;
    const std::uint64_t ffn_scratch_bytes = std::max<std::uint64_t>(
      std::max<std::uint64_t>(
        static_cast<std::uint64_t>(std::max(max_ffn_width, max_ple_width))
          * 2u * sizeof(float),
        static_cast<std::uint64_t>(vocab_size) * sizeof(float)),
      expected_ple_items * sizeof(float));
    const std::uint64_t activation_bytes =
      static_cast<std::uint64_t>(std::max(max_ffn_width, max_ple_width)) * sizeof(float);
    const std::uint64_t output_bytes =
      static_cast<std::uint64_t>(std::max(hidden_width, vocab_size)) * sizeof(float);
    const std::uint64_t readback_bytes =
      hidden_bytes + static_cast<std::uint64_t>(vocab_size) * sizeof(float)
      + sizeof(std::uint32_t);

    const auto ensure_uav = [&](
      ComPtr<ID3D12Resource> &resource,
      std::uint64_t required,
      std::uint64_t &allocated,
      D3D12_RESOURCE_STATES &state) {
      const std::uint64_t previous = allocated;
      if (!ensure_buffer(
        resource,
        std::max<std::uint64_t>(required, 4),
        allocated,
        D3D12_HEAP_TYPE_DEFAULT,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        error_message)) {
        return false;
      }
      if (previous != allocated) {
        state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
      }
      return true;
    };
    // Optional DP4A route: activations quantized to Q8_1 once per stage, then
    // Q4_K matvecs run as packed integer dots. The scratch is sized for the
    // widest quantized input in this graph (hidden state or attention
    // context).
    const bool use_dp4a = dp4a_decode_enabled()
      && dp4a_capabilities_.supported
      && dp4a_quantize_pipeline_
      && dp4a_q4k_b1_pipeline_;
    std::uint64_t dp4a_quantize_width = hidden_width;
    if (use_dp4a) {
      for (const D3d12GemmaTokenLayer &quantize_layer : layers) {
        dp4a_quantize_width = std::max<std::uint64_t>(
          dp4a_quantize_width, quantize_layer.attn_out.row_width);
      }
    }
    const std::uint64_t dp4a_scratch_bytes =
      (dp4a_quantize_width / 4 + dp4a_quantize_width / 32 * 2) * sizeof(std::uint32_t);
    if (use_dp4a
      && !ensure_uav(
        dp4a_activations_, dp4a_scratch_bytes, dp4a_activations_bytes_, dp4a_activations_state_)) {
      return false;
    }
    const std::uint64_t previous_upload_allocation = input_upload_bytes_;
    const std::uint64_t previous_constants_allocation = input_default_bytes_;
    if (!ensure_persistent_upload_buffer(
        input_upload_, upload_bytes, input_upload_bytes_, mapped_input_upload_, error_message)
      || !ensure_srv_buffer(
        input_default_, constants_bytes, input_default_bytes_, input_state_, error_message)
      || !ensure_uav(
        attention_input_, hidden_bytes, attention_input_bytes_, attention_input_state_)
      || !ensure_srv_buffer(
        attention_config_,
        static_cast<std::uint64_t>(max_attention_config_floats) * sizeof(float),
        attention_config_bytes_,
        attention_config_state_,
        error_message)
      || !ensure_uav(attention_qkv_, qkv_bytes, attention_qkv_bytes_, attention_qkv_state_)
      || !ensure_uav(attention_query_, query_bytes, attention_query_bytes_, attention_query_state_)
      || !ensure_uav(attention_context_, query_bytes, attention_context_bytes_, attention_context_state_)
      || !ensure_uav(layer_scratch_a_, scratch_ab_bytes, layer_scratch_a_bytes_, layer_scratch_a_state_)
      || !ensure_uav(layer_scratch_b_, scratch_ab_bytes, layer_scratch_b_bytes_, layer_scratch_b_state_)
      || !ensure_uav(ffn_scratch_, ffn_scratch_bytes, ffn_scratch_bytes_, ffn_scratch_state_)
      || !ensure_uav(ffn_activation_, activation_bytes, ffn_activation_bytes_, ffn_activation_state_)
      || (has_global_ple
        && !ensure_uav(
          ple_input_default_,
          expected_ple_items * sizeof(float),
          ple_input_default_bytes_,
          ple_input_default_state_))
      || !ensure_output_buffer(
        output_default_, output_bytes, output_default_bytes_, output_state_, error_message)
      || !ensure_persistent_readback_buffer(
        output_readback_,
        readback_bytes,
        output_readback_bytes_,
        mapped_output_readback_,
        error_message)) {
      return false;
    }

    if (previous_upload_allocation != input_upload_bytes_
      || previous_constants_allocation != input_default_bytes_) {
      // Reallocated buffers lost their contents; the static upload cache is
      // stale.
      token_graph_upload_model_ = nullptr;
    }
    // The norm/rope/PLE constants and per-layer attention configs are
    // immutable across tokens; they are rewritten only when the model, the
    // upload layout (initial-KV regions shift every offset on the first
    // token), or the upload buffer itself changed. Skipping the rewrite
    // removes about 2 MB of CPU memcpy per decoded token.
    bool has_initial_kv_upload = false;
    for (const PackedLayer &packed : packed_layers) {
      if (packed.initial_kv_bytes > 0) {
        has_initial_kv_upload = true;
        break;
      }
    }
    const bool static_upload_valid = !has_initial_kv_upload
      && !token_graph_upload_had_kv_
      && token_graph_upload_model_ == output.weight_key
      && token_graph_upload_bytes_ == upload_bytes;
    std::uint8_t *upload = static_cast<std::uint8_t *>(mapped_input_upload_);
    std::memcpy(upload + hidden_upload_offset, initial_hidden.data(), hidden_bytes);
    // The PLE token inputs inside the constants block change every token even
    // when the rest of the block is cacheable.
    const std::uint64_t ple_token_inputs_bytes = has_global_ple
      ? static_cast<std::uint64_t>(ple_inputs->token_inputs->size()) * sizeof(float)
      : 0;
    if (static_upload_valid && has_global_ple) {
      std::memcpy(
        upload + constants_upload_offset
          + static_cast<std::uint64_t>(ple_token_inputs_offset) * sizeof(float),
        ple_inputs->token_inputs->data(),
        ple_token_inputs_bytes);
    }
    if (!static_upload_valid) {
      std::memcpy(
        upload + constants_upload_offset, packed_constants.data(), constants_bytes);
      for (std::size_t index = 0; index < layers.size(); index += 1) {
        const PackedLayer &packed = packed_layers[index];
        std::memcpy(
          upload + packed.attention_config_upload,
          attention_configs[index].data(),
          packed.attention_config_bytes);
        if (packed.initial_kv_bytes > 0) {
          std::uint16_t *destination = reinterpret_cast<std::uint16_t *>(
            upload + packed.initial_kv_upload);
          const D3d12KvLaneView &keys = layers[index].initial_keys;
          const D3d12KvLaneView &values = layers[index].initial_values;
          pack_kv_lane_halves(destination, keys);
          pack_kv_lane_halves(destination + keys.size(), values);
        }
      }
    }
    // Cache identity is recorded after submission: submit_and_wait clears it
    // for every path, so only a completed graph token re-validates it.

    HRESULT hr = command_allocator_->Reset();
    if (FAILED(hr)) {
      error_message = hresult_message("ID3D12CommandAllocator::Reset", hr);
      return false;
    }
    hr = command_list_->Reset(command_allocator_.get(), nullptr);
    if (FAILED(hr)) {
      error_message = hresult_message("ID3D12GraphicsCommandList::Reset", hr);
      return false;
    }
    command_list_->SetComputeRootSignature(root_signature_.get());
    // B6 diagnostics: interval label names the stage ENDING at that stamp.
    const bool timestamps_on = gpu_timestamps_enabled() && ensure_timestamp_resources();
    timestamp_labels_.clear();
    const auto stamp = [&](const char *label) {
      if (!timestamps_on
        || timestamp_labels_.size() >= timestamp_capacity_) {
        return;
      }
      command_list_->EndQuery(
        timestamp_heap_.get(),
        D3D12_QUERY_TYPE_TIMESTAMP,
        static_cast<UINT>(timestamp_labels_.size()));
      timestamp_labels_.push_back(label);
    };
    stamp("graph-start");

    if (!static_upload_valid) {
      transition(input_default_.get(), input_state_, D3D12_RESOURCE_STATE_COPY_DEST);
      command_list_->CopyBufferRegion(
        input_default_.get(), 0, input_upload_.get(), constants_upload_offset, constants_bytes);
      transition(input_default_.get(), input_state_, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    } else if (has_global_ple) {
      // Static block cached: refresh only the per-token PLE input region.
      transition(input_default_.get(), input_state_, D3D12_RESOURCE_STATE_COPY_DEST);
      command_list_->CopyBufferRegion(
        input_default_.get(),
        static_cast<std::uint64_t>(ple_token_inputs_offset) * sizeof(float),
        input_upload_.get(),
        constants_upload_offset
          + static_cast<std::uint64_t>(ple_token_inputs_offset) * sizeof(float),
        ple_token_inputs_bytes);
      transition(input_default_.get(), input_state_, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    }
    transition(output_default_.get(), output_state_, D3D12_RESOURCE_STATE_COPY_DEST);
    command_list_->CopyBufferRegion(
      output_default_.get(), 0, input_upload_.get(), hidden_upload_offset, hidden_bytes);

    const auto uav_barrier = [&](ID3D12Resource *resource) {
      D3D12_RESOURCE_BARRIER barrier = {};
      barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
      barrier.UAV.pResource = resource;
      command_list_->ResourceBarrier(1, &barrier);
    };
    const auto pipeline_for = [&](const D3d12ResidentWeightRef &op) {
      switch (op.format) {
        case D3d12WeightFormat::Q4_K: return q4k_b1_pipeline_.get();
        case D3d12WeightFormat::Q5_K: return q5k_b1_pipeline_.get();
        case D3d12WeightFormat::Q6_K: return q6k_b1_pipeline_.get();
        case D3d12WeightFormat::Q8_0: return q8_0_b1_pipeline_.get();
        case D3d12WeightFormat::Q4_0: return q4_0_b1_pipeline_.get();
        case D3d12WeightFormat::F32:
        case D3d12WeightFormat::F16:
        case D3d12WeightFormat::BF16:
          return scalar_b1_pipeline_.get();
      }
      return scalar_b1_pipeline_.get();
    };
    const auto dispatch_matvec = [&](
      const D3d12ResidentWeightRef &op,
      ID3D12Resource *input,
      std::uint32_t input_stride,
      ID3D12Resource *destination,
      std::uint32_t destination_offset,
      std::uint32_t input_transform = 0) {
      ResidentWeight &resident = weight_cache_.at(op.weight_key);
      resident.last_use = use_counter_;
      DispatchConstants constants;
      constants.row_width = op.row_width;
      constants.row_count = op.row_count;
      constants.batch_count = 1;
      constants.output_base = destination_offset;
      constants.input_stride = input_stride;
      constants.weight_layout = 1;
      constants.input_transform = input_transform;
      // Only the tile-loader kernels (base 64-row and wide128/wide256
      // main_* family) implement input_transform; the gemv/direct/pair
      // variants read inputs directly and must not see transformed calls.
      const bool plain_input = input_transform == 0;
      // Cooperative Q4_K GEMV variants: one 64-lane group per output row with
      // coalesced reads. Both measured SLOWER than the plain batch-one kernel
      // on this hardware (see coop_gemv_enabled/wave_gemv_enabled comments);
      // kept opt-in for reference. Only for plain (non-input-transform) Q4_K
      // matvecs, which is every token-graph projection.
      const bool use_wave_gemv = plain_input && wave_gemv_enabled()
        && op.format == D3d12WeightFormat::Q4_K
        && q4k_gemv_wave_pipeline_;
      const bool use_coop_gemv = plain_input && !use_wave_gemv
        && coop_gemv_enabled()
        && op.format == D3d12WeightFormat::Q4_K
        && q4k_gemv_pipeline_;
      // 128-rows-per-group kernels: FEWER, LARGER batch-one dispatches
      // (not cooperative) - measured 1.11x faster standalone on Q4_K and
      // extended to every quantized decode format (identical per-thread
      // math, only the group width changes). See wide128_enabled() for the
      // rationale and measurement.
      ID3D12PipelineState *const direct_pipeline =
        (plain_input && !use_wave_gemv && !use_coop_gemv && direct_input_enabled())
          ? (op.format == D3d12WeightFormat::Q4_K ? q4k_direct_pipeline_.get()
            : op.format == D3d12WeightFormat::Q6_K ? q6k_direct_pipeline_.get()
            : nullptr)
          : nullptr;
      const bool use_direct = direct_pipeline != nullptr;
      const bool use_pair = plain_input && !use_wave_gemv && !use_coop_gemv && !use_direct
        && block_pair_enabled()
        && op.format == D3d12WeightFormat::Q4_K
        && q4k_pair_pipeline_;
      const bool use_wide256 = !use_wave_gemv && !use_coop_gemv && !use_direct && !use_pair
        && wide256_enabled()
        && op.format == D3d12WeightFormat::Q4_K
        && q4k_wide256_pipeline_;
      ID3D12PipelineState *const wide128_pipeline =
        (!use_wave_gemv && !use_coop_gemv && !use_direct && !use_pair && !use_wide256
          && wide128_enabled())
          ? wide128_pipeline_for(op.format)
          : nullptr;
      const bool use_wide128 = wide128_pipeline != nullptr;
      const bool use_row_per_group = use_wave_gemv || use_coop_gemv;
      command_list_->SetPipelineState(
        use_wave_gemv ? q4k_gemv_wave_pipeline_.get()
        : use_coop_gemv ? q4k_gemv_pipeline_.get()
        : use_direct ? direct_pipeline
        : use_pair ? q4k_pair_pipeline_.get()
        : use_wide256 ? q4k_wide256_pipeline_.get()
        : use_wide128 ? wide128_pipeline
        : pipeline_for(op));
      command_list_->SetComputeRoot32BitConstants(0, 8, &constants, 0);
      command_list_->SetComputeRootShaderResourceView(
        1, resident.buffer->GetGPUVirtualAddress());
      command_list_->SetComputeRootShaderResourceView(
        2, input->GetGPUVirtualAddress());
      command_list_->SetComputeRootUnorderedAccessView(
        3, destination->GetGPUVirtualAddress());
      command_list_->Dispatch(
        use_row_per_group ? op.row_count
        : use_direct ? (op.row_count + 127u) / 128u
        : use_pair ? (op.row_count + 127u) / 128u
        : use_wide256 ? (op.row_count + 255u) / 256u
        : use_wide128 ? (op.row_count + 127u) / 128u
        : (op.row_count + 63u) / 64u, 1, 1);
    };
    // DP4A helpers: quantize a float stage input once into the shared Q8_1
    // scratch, then run Q4_K matvecs as packed integer dots against it. The
    // scratch transitions (UAV for the quantize write, SRV for the matvec
    // reads) also order reuse across stages within the single command list.
    const auto dp4a_eligible = [&](const D3d12ResidentWeightRef &op) {
      return use_dp4a
        && op.format == D3d12WeightFormat::Q4_K
        && (op.row_width % k_quant_block_elements) == 0;
    };
    const auto dispatch_quantize_q8_1 = [&](ID3D12Resource *input, std::uint32_t width) {
      transition(
        dp4a_activations_.get(), dp4a_activations_state_, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
      DispatchConstants constants;
      constants.row_width = width;
      constants.input_stride = 0;
      constants.output_base = 0;
      command_list_->SetPipelineState(dp4a_quantize_pipeline_.get());
      command_list_->SetComputeRoot32BitConstants(0, 8, &constants, 0);
      command_list_->SetComputeRootShaderResourceView(1, input->GetGPUVirtualAddress());
      command_list_->SetComputeRootShaderResourceView(2, input->GetGPUVirtualAddress());
      command_list_->SetComputeRootUnorderedAccessView(
        3, dp4a_activations_->GetGPUVirtualAddress());
      command_list_->Dispatch((width + 63u) / 64u, 1, 1);
      uav_barrier(dp4a_activations_.get());
      transition(
        dp4a_activations_.get(),
        dp4a_activations_state_,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    };
    const auto dispatch_matvec_dp4a = [&](
      const D3d12ResidentWeightRef &op,
      ID3D12Resource *destination,
      std::uint32_t destination_offset) {
      ResidentWeight &resident = weight_cache_.at(op.weight_key);
      resident.last_use = use_counter_;
      DispatchConstants constants;
      constants.row_width = op.row_width;
      constants.row_count = op.row_count;
      constants.input_stride = 0;
      constants.output_base = destination_offset;
      command_list_->SetPipelineState(dp4a_q4k_b1_pipeline_.get());
      command_list_->SetComputeRoot32BitConstants(0, 8, &constants, 0);
      command_list_->SetComputeRootShaderResourceView(
        1, resident.buffer->GetGPUVirtualAddress());
      command_list_->SetComputeRootShaderResourceView(
        2, dp4a_activations_->GetGPUVirtualAddress());
      command_list_->SetComputeRootUnorderedAccessView(
        3, destination->GetGPUVirtualAddress());
      command_list_->Dispatch((op.row_count + 63u) / 64u, 1, 1);
    };
    const auto dispatch_rms_norm = [&](
      ID3D12Resource *vector,
      std::uint32_t x_offset,
      std::uint32_t add_offset,
      bool add_enabled,
      std::uint32_t weight_offset,
      float epsilon,
      float scale,
      ID3D12Resource *destination,
      std::uint32_t destination_offset) {
      DispatchConstants constants;
      constants.row_width = hidden_width;
      constants.row_count = x_offset;
      constants.batch_count = add_offset;
      constants.output_base = destination_offset;
      constants.scalar_format = add_enabled ? 1u : 0u;
      if (scale != 1.0f) {
        constants.scalar_format |= 2u;
      }
      std::memcpy(&constants.input_stride, &epsilon, sizeof(float));
      constants.weight_layout = weight_offset;
      std::memcpy(&constants.input_transform, &scale, sizeof(float));
      command_list_->SetPipelineState(rms_norm_pipeline_.get());
      command_list_->SetComputeRoot32BitConstants(0, 8, &constants, 0);
      command_list_->SetComputeRootShaderResourceView(
        1, input_default_->GetGPUVirtualAddress());
      command_list_->SetComputeRootShaderResourceView(
        2, vector->GetGPUVirtualAddress());
      command_list_->SetComputeRootUnorderedAccessView(
        3, destination->GetGPUVirtualAddress());
      command_list_->Dispatch(1, 1, 1);
    };

    use_counter_ += 1;
    if (has_global_ple) {
      transition(output_default_.get(), output_state_, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
      transition(ffn_scratch_.get(), ffn_scratch_state_, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
      dispatch_matvec(
        ple_inputs->model_projection,
        output_default_.get(),
        hidden_width,
        ffn_scratch_.get(),
        0);
      uav_barrier(ffn_scratch_.get());
      transition(
        ffn_scratch_.get(),
        ffn_scratch_state_,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
      transition(
        ple_input_default_.get(),
        ple_input_default_state_,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
      DispatchConstants ple_prepare_constants;
      ple_prepare_constants.row_width = ple_inputs->layer_width;
      ple_prepare_constants.row_count = static_cast<std::uint32_t>(layers.size());
      std::memcpy(
        &ple_prepare_constants.batch_count,
        &ple_inputs->projection_scale,
        sizeof(float));
      std::memcpy(
        &ple_prepare_constants.input_stride,
        &ple_inputs->rms_epsilon,
        sizeof(float));
      ple_prepare_constants.weight_layout = ple_projection_norm_offset;
      ple_prepare_constants.input_transform = ple_token_inputs_offset;
      command_list_->SetPipelineState(ple_prepare_pipeline_.get());
      command_list_->SetComputeRoot32BitConstants(0, 8, &ple_prepare_constants, 0);
      command_list_->SetComputeRootShaderResourceView(
        1, input_default_->GetGPUVirtualAddress());
      command_list_->SetComputeRootShaderResourceView(
        2, ffn_scratch_->GetGPUVirtualAddress());
      command_list_->SetComputeRootUnorderedAccessView(
        3, ple_input_default_->GetGPUVirtualAddress());
      command_list_->Dispatch(static_cast<UINT>(layers.size()), 1, 1);
      uav_barrier(ple_input_default_.get());
      transition(
        ple_input_default_.get(),
        ple_input_default_state_,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    }
    std::unordered_set<std::uint64_t> appended_this_token;
    std::uint64_t uploaded_initial_kv_bytes = 0;
    stamp("uploads");
    for (std::size_t index = 0; index < layers.size(); index += 1) {
      const D3d12GemmaTokenLayer &layer = layers[index];
      const PackedLayer &packed = packed_layers[index];
      DeviceKvCache &cache = *packed.cache;
      const std::uint32_t query_size =
        layer.attention.head_count * layer.attention.head_dim;
      const std::uint32_t kv_size =
        layer.attention.kv_head_count * layer.attention.head_dim;
      const std::uint32_t words_per_position = cache.words_per_position;
      const bool has_ple = layer.ple_gate.weight_key != nullptr;

      transition(
        attention_config_.get(),
        attention_config_state_,
        D3D12_RESOURCE_STATE_COPY_DEST);
      command_list_->CopyBufferRegion(
        attention_config_.get(),
        0,
        input_upload_.get(),
        packed.attention_config_upload,
        packed.attention_config_bytes);
      transition(
        attention_config_.get(),
        attention_config_state_,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
      if (packed.initial_kv_bytes > 0) {
        const std::uint64_t half_bytes =
          static_cast<std::uint64_t>(layer.attention.position) * kv_size
          * sizeof(std::uint16_t);
        command_list_->CopyBufferRegion(
          cache.buffer.get(), 0, input_upload_.get(), packed.initial_kv_upload, half_bytes);
        command_list_->CopyBufferRegion(
          cache.buffer.get(),
          static_cast<std::uint64_t>(layer.attention.max_positions)
            * words_per_position * sizeof(std::uint32_t),
          input_upload_.get(),
          packed.initial_kv_upload + half_bytes,
          half_bytes);
        uploaded_initial_kv_bytes += packed.initial_kv_bytes;
      }

      transition(output_default_.get(), output_state_, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
      transition(attention_input_.get(), attention_input_state_, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
      dispatch_rms_norm(
        output_default_.get(),
        0,
        0,
        false,
        packed.attention_norm,
        layer.attention.rms_epsilon,
        1.0f,
        attention_input_.get(),
        0);
      uav_barrier(attention_input_.get());
      transition(
        attention_input_.get(),
        attention_input_state_,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

      const bool query_dp4a = dp4a_eligible(layer.query);
      const bool key_dp4a = layer.owns_kv_cache && dp4a_eligible(layer.key);
      const bool value_dp4a = layer.owns_kv_cache && dp4a_eligible(layer.value);
      if (query_dp4a || key_dp4a || value_dp4a) {
        // One shared normed-hidden quantization feeds all sibling projections.
        dispatch_quantize_q8_1(attention_input_.get(), hidden_width);
      }
      transition(attention_qkv_.get(), attention_qkv_state_, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
      if (query_dp4a) {
        dispatch_matvec_dp4a(layer.query, attention_qkv_.get(), 0);
      } else {
        dispatch_matvec(layer.query, attention_input_.get(), hidden_width, attention_qkv_.get(), 0);
      }
      if (layer.owns_kv_cache) {
        if (key_dp4a) {
          dispatch_matvec_dp4a(layer.key, attention_qkv_.get(), query_size);
        } else {
          dispatch_matvec(
            layer.key, attention_input_.get(), hidden_width, attention_qkv_.get(), query_size);
        }
        if (value_dp4a) {
          dispatch_matvec_dp4a(layer.value, attention_qkv_.get(), query_size + kv_size);
        } else {
          dispatch_matvec(
            layer.value,
            attention_input_.get(),
            hidden_width,
            attention_qkv_.get(),
            query_size + kv_size);
        }
      }
      uav_barrier(attention_qkv_.get());
      transition(
        attention_qkv_.get(),
        attention_qkv_state_,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

      const std::uint32_t query_norm_offset = 3u;
      const std::uint32_t key_norm_offset = query_norm_offset + layer.attention.head_dim;
      const std::uint32_t rope_offset = key_norm_offset + layer.attention.head_dim;
      const auto dispatch_prepare = [&](
        std::uint32_t source_offset,
        std::uint32_t destination_offset,
        std::uint32_t head_count,
        std::uint32_t norm_offset,
        std::uint32_t flags,
        ID3D12Resource *destination) {
        DispatchConstants constants;
        constants.row_width = layer.attention.head_dim;
        constants.row_count = head_count;
        constants.batch_count = source_offset;
        constants.output_base = destination_offset;
        constants.scalar_format = flags;
        constants.input_stride = layer.attention.position;
        constants.weight_layout = norm_offset;
        constants.input_transform = rope_offset;
        command_list_->SetPipelineState(gemma_prepare_head_pipeline_.get());
        command_list_->SetComputeRoot32BitConstants(0, 8, &constants, 0);
        command_list_->SetComputeRootShaderResourceView(
          1, attention_config_->GetGPUVirtualAddress());
        command_list_->SetComputeRootShaderResourceView(
          2, attention_qkv_->GetGPUVirtualAddress());
        command_list_->SetComputeRootUnorderedAccessView(
          3, destination->GetGPUVirtualAddress());
        command_list_->Dispatch(head_count, 1, 1);
      };
      transition(attention_query_.get(), attention_query_state_, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
      dispatch_prepare(
        0, 0, layer.attention.head_count, query_norm_offset, 3u, attention_query_.get());
      uav_barrier(attention_query_.get());
      if (layer.owns_kv_cache) {
        transition(cache.buffer.get(), cache.state, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        dispatch_prepare(
          query_size,
          layer.attention.position * words_per_position,
          layer.attention.kv_head_count,
          key_norm_offset,
          7u,
          cache.buffer.get());
        uav_barrier(cache.buffer.get());
        dispatch_prepare(
          query_size + kv_size,
          layer.attention.max_positions * words_per_position
            + layer.attention.position * words_per_position,
          layer.attention.kv_head_count,
          key_norm_offset,
          4u,
          cache.buffer.get());
        uav_barrier(cache.buffer.get());
        appended_this_token.insert(layer.attention.cache_id);
      }
      transition(
        attention_query_.get(),
        attention_query_state_,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
      transition(cache.buffer.get(), cache.state, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

      transition(
        attention_context_.get(),
        attention_context_state_,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
      DispatchConstants attention_constants;
      attention_constants.row_width = layer.attention.head_dim;
      attention_constants.row_count = layer.attention.head_count;
      attention_constants.batch_count = layer.attention.kv_head_count;
      attention_constants.scalar_format = layer.attention.position + 1u;
      attention_constants.input_stride =
        layer.attention.attention_window > 0
          && layer.attention.position + 1u > layer.attention.attention_window
        ? layer.attention.position + 1u - layer.attention.attention_window
        : 0u;
      attention_constants.weight_layout =
        layer.attention.max_positions * words_per_position;
      attention_constants.input_transform = kv_size;
      command_list_->SetPipelineState(gemma_attention_tiled_enabled()
        ? gemma_online_attention_tiled_pipeline_.get()
        : gemma_online_attention_pipeline_.get());
      command_list_->SetComputeRoot32BitConstants(0, 8, &attention_constants, 0);
      command_list_->SetComputeRootShaderResourceView(1, cache.buffer->GetGPUVirtualAddress());
      command_list_->SetComputeRootShaderResourceView(
        2, attention_query_->GetGPUVirtualAddress());
      command_list_->SetComputeRootUnorderedAccessView(
        3, attention_context_->GetGPUVirtualAddress());
      command_list_->Dispatch(layer.attention.head_count, 1, 1);
      uav_barrier(attention_context_.get());
      transition(
        attention_context_.get(),
        attention_context_state_,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

      const bool attn_out_dp4a = dp4a_eligible(layer.attn_out);
      if (attn_out_dp4a) {
        dispatch_quantize_q8_1(attention_context_.get(), query_size);
      }
      transition(layer_scratch_a_.get(), layer_scratch_a_state_, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
      if (attn_out_dp4a) {
        dispatch_matvec_dp4a(layer.attn_out, layer_scratch_a_.get(), 0);
      } else {
        dispatch_matvec(
          layer.attn_out,
          attention_context_.get(),
          query_size,
          layer_scratch_a_.get(),
          0);
      }
      uav_barrier(layer_scratch_a_.get());
      transition(output_default_.get(), output_state_, D3D12_RESOURCE_STATE_COPY_SOURCE);
      transition(layer_scratch_a_.get(), layer_scratch_a_state_, D3D12_RESOURCE_STATE_COPY_DEST);
      command_list_->CopyBufferRegion(
        layer_scratch_a_.get(), hidden_bytes, output_default_.get(), 0, hidden_bytes);
      transition(
        layer_scratch_a_.get(),
        layer_scratch_a_state_,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
      transition(layer_scratch_b_.get(), layer_scratch_b_state_, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
      dispatch_rms_norm(
        layer_scratch_a_.get(),
        0,
        hidden_width,
        true,
        packed.post_attention_norm,
        layer.attention.rms_epsilon,
        1.0f,
        layer_scratch_b_.get(),
        0);
      uav_barrier(layer_scratch_b_.get());
      transition(
        layer_scratch_b_.get(),
        layer_scratch_b_state_,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

      stamp("attention");
      transition(layer_scratch_a_.get(), layer_scratch_a_state_, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
      dispatch_rms_norm(
        layer_scratch_b_.get(),
        0,
        0,
        false,
        packed.ffn_norm,
        layer.attention.rms_epsilon,
        1.0f,
        layer_scratch_a_.get(),
        0);
      uav_barrier(layer_scratch_a_.get());
      transition(
        layer_scratch_a_.get(),
        layer_scratch_a_state_,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

      const bool gate_dp4a = dp4a_eligible(layer.gate);
      const bool up_dp4a = dp4a_eligible(layer.up);
      if (gate_dp4a || up_dp4a) {
        // One shared FFN-normed quantization feeds both gate and up.
        dispatch_quantize_q8_1(layer_scratch_a_.get(), hidden_width);
      }
      transition(ffn_scratch_.get(), ffn_scratch_state_, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
      if (gate_dp4a) {
        dispatch_matvec_dp4a(layer.gate, ffn_scratch_.get(), 0);
      } else {
        dispatch_matvec(layer.gate, layer_scratch_a_.get(), hidden_width, ffn_scratch_.get(), 0);
      }
      if (up_dp4a) {
        dispatch_matvec_dp4a(layer.up, ffn_scratch_.get(), layer.gate.row_count);
      } else {
        dispatch_matvec(
          layer.up,
          layer_scratch_a_.get(),
          hidden_width,
          ffn_scratch_.get(),
          layer.gate.row_count);
      }
      uav_barrier(ffn_scratch_.get());
      transition(ffn_scratch_.get(), ffn_scratch_state_, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
      if (fused_gelu_enabled()) {
        // Fused path: the down matvec's tile loader computes gelu(gate)*up
        // from the split gate/up halves of ffn_scratch while staging each
        // input tile (input_transform=1, second half at gate.row_count).
        // Skips the dedicated gelu dispatch, its UAV barrier, and the
        // ffn_activation write+read round-trip entirely.
        transition(layer_scratch_b_.get(), layer_scratch_b_state_, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        dispatch_matvec(
          layer.down,
          ffn_scratch_.get(),
          layer.gate.row_count,
          layer_scratch_b_.get(),
          hidden_width,
          1u);
      } else {
        transition(ffn_activation_.get(), ffn_activation_state_, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        DispatchConstants gelu_constants;
        gelu_constants.row_count = layer.gate.row_count;
        command_list_->SetPipelineState(gelu_pipeline_.get());
        command_list_->SetComputeRoot32BitConstants(0, 8, &gelu_constants, 0);
        command_list_->SetComputeRootShaderResourceView(1, ffn_scratch_->GetGPUVirtualAddress());
        command_list_->SetComputeRootShaderResourceView(2, ffn_scratch_->GetGPUVirtualAddress());
        command_list_->SetComputeRootUnorderedAccessView(
          3, ffn_activation_->GetGPUVirtualAddress());
        command_list_->Dispatch((layer.gate.row_count + 255u) / 256u, 1, 1);
        uav_barrier(ffn_activation_.get());
        transition(
          ffn_activation_.get(),
          ffn_activation_state_,
          D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

        transition(layer_scratch_b_.get(), layer_scratch_b_state_, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        dispatch_matvec(
          layer.down,
          ffn_activation_.get(),
          layer.gate.row_count,
          layer_scratch_b_.get(),
          hidden_width);
      }
      uav_barrier(layer_scratch_b_.get());
      transition(
        layer_scratch_b_.get(),
        layer_scratch_b_state_,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
      transition(output_default_.get(), output_state_, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
      dispatch_rms_norm(
        layer_scratch_b_.get(),
        hidden_width,
        0,
        true,
        packed.post_ffw_norm,
        layer.attention.rms_epsilon,
        has_ple ? 1.0f : layer.output_scale,
        output_default_.get(),
        0);
      uav_barrier(output_default_.get());

      if (has_ple) {
        transition(
          output_default_.get(),
          output_state_,
          D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        transition(ffn_scratch_.get(), ffn_scratch_state_, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        dispatch_matvec(
          layer.ple_gate,
          output_default_.get(),
          hidden_width,
          ffn_scratch_.get(),
          0);
        uav_barrier(ffn_scratch_.get());
        transition(ffn_scratch_.get(), ffn_scratch_state_, D3D12_RESOURCE_STATE_COPY_DEST);
        transition(
          ple_input_default_.get(),
          ple_input_default_state_,
          D3D12_RESOURCE_STATE_COPY_SOURCE);
        command_list_->CopyBufferRegion(
          ffn_scratch_.get(),
          static_cast<std::uint64_t>(layer.ple_gate.row_count) * sizeof(float),
          ple_input_default_.get(),
          static_cast<std::uint64_t>(index) * layer.ple_gate.row_count * sizeof(float),
          static_cast<std::uint64_t>(layer.ple_gate.row_count) * sizeof(float));
        transition(
          ple_input_default_.get(),
          ple_input_default_state_,
          D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        transition(
          ffn_scratch_.get(),
          ffn_scratch_state_,
          D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        transition(ffn_activation_.get(), ffn_activation_state_, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        DispatchConstants ple_gelu_constants;
        ple_gelu_constants.row_count = layer.ple_gate.row_count;
        command_list_->SetPipelineState(gelu_pipeline_.get());
        command_list_->SetComputeRoot32BitConstants(0, 8, &ple_gelu_constants, 0);
        command_list_->SetComputeRootShaderResourceView(1, ffn_scratch_->GetGPUVirtualAddress());
        command_list_->SetComputeRootShaderResourceView(2, ffn_scratch_->GetGPUVirtualAddress());
        command_list_->SetComputeRootUnorderedAccessView(
          3, ffn_activation_->GetGPUVirtualAddress());
        command_list_->Dispatch((layer.ple_gate.row_count + 255u) / 256u, 1, 1);
        uav_barrier(ffn_activation_.get());
        transition(
          ffn_activation_.get(),
          ffn_activation_state_,
          D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        transition(layer_scratch_a_.get(), layer_scratch_a_state_, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        dispatch_matvec(
          layer.ple_projection,
          ffn_activation_.get(),
          layer.ple_gate.row_count,
          layer_scratch_a_.get(),
          0);
        uav_barrier(layer_scratch_a_.get());
        transition(output_default_.get(), output_state_, D3D12_RESOURCE_STATE_COPY_SOURCE);
        transition(layer_scratch_a_.get(), layer_scratch_a_state_, D3D12_RESOURCE_STATE_COPY_DEST);
        command_list_->CopyBufferRegion(
          layer_scratch_a_.get(), hidden_bytes, output_default_.get(), 0, hidden_bytes);
        transition(
          layer_scratch_a_.get(),
          layer_scratch_a_state_,
          D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        transition(output_default_.get(), output_state_, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        dispatch_rms_norm(
          layer_scratch_a_.get(),
          0,
          hidden_width,
          true,
          packed.ple_post_norm,
          layer.attention.rms_epsilon,
          layer.output_scale,
          output_default_.get(),
          0);
        uav_barrier(output_default_.get());
      }
      stamp("ffn+ple");
    }

    transition(output_default_.get(), output_state_, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    transition(layer_scratch_a_.get(), layer_scratch_a_state_, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    dispatch_rms_norm(
      output_default_.get(),
      0,
      0,
      false,
      output_norm_offset,
      output_rms_epsilon,
      1.0f,
      layer_scratch_a_.get(),
      0);
    uav_barrier(layer_scratch_a_.get());
    transition(
      layer_scratch_a_.get(),
      layer_scratch_a_state_,
      D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    const bool lm_head_dp4a = dp4a_eligible(output);
    if (lm_head_dp4a) {
      dispatch_quantize_q8_1(layer_scratch_a_.get(), hidden_width);
    }
    transition(output_default_.get(), output_state_, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    if (lm_head_dp4a) {
      dispatch_matvec_dp4a(output, output_default_.get(), 0);
    } else {
      dispatch_matvec(output, layer_scratch_a_.get(), hidden_width, output_default_.get(), 0);
    }
    uav_barrier(output_default_.get());
    transition(output_default_.get(), output_state_, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    stamp("lm-head");

    ID3D12Resource *final_logits = output_default_.get();
    if (final_logit_softcap > 0.0f) {
      transition(ffn_scratch_.get(), ffn_scratch_state_, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
      DispatchConstants softcap_constants;
      softcap_constants.row_count = vocab_size;
      std::memcpy(
        &softcap_constants.input_stride, &final_logit_softcap, sizeof(float));
      command_list_->SetPipelineState(logit_softcap_pipeline_.get());
      command_list_->SetComputeRoot32BitConstants(0, 8, &softcap_constants, 0);
      command_list_->SetComputeRootShaderResourceView(1, output_default_->GetGPUVirtualAddress());
      command_list_->SetComputeRootShaderResourceView(2, output_default_->GetGPUVirtualAddress());
      command_list_->SetComputeRootUnorderedAccessView(3, ffn_scratch_->GetGPUVirtualAddress());
      command_list_->Dispatch((vocab_size + 255u) / 256u, 1, 1);
      uav_barrier(ffn_scratch_.get());
      transition(
        ffn_scratch_.get(),
        ffn_scratch_state_,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
      final_logits = ffn_scratch_.get();
    }

    transition(attention_qkv_.get(), attention_qkv_state_, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    DispatchConstants argmax_constants;
    argmax_constants.row_count = vocab_size;
    command_list_->SetPipelineState(argmax_pipeline_.get());
    command_list_->SetComputeRoot32BitConstants(0, 8, &argmax_constants, 0);
    command_list_->SetComputeRootShaderResourceView(1, final_logits->GetGPUVirtualAddress());
    command_list_->SetComputeRootShaderResourceView(2, final_logits->GetGPUVirtualAddress());
    command_list_->SetComputeRootUnorderedAccessView(3, attention_qkv_->GetGPUVirtualAddress());
    command_list_->Dispatch(1, 1, 1);
    uav_barrier(attention_qkv_.get());
    stamp("softcap+argmax");

    transition(layer_scratch_a_.get(), layer_scratch_a_state_, D3D12_RESOURCE_STATE_COPY_SOURCE);
    if (final_logits == output_default_.get()) {
      transition(output_default_.get(), output_state_, D3D12_RESOURCE_STATE_COPY_SOURCE);
    } else {
      transition(ffn_scratch_.get(), ffn_scratch_state_, D3D12_RESOURCE_STATE_COPY_SOURCE);
    }
    transition(attention_qkv_.get(), attention_qkv_state_, D3D12_RESOURCE_STATE_COPY_SOURCE);
    command_list_->CopyBufferRegion(
      output_readback_.get(), 0, layer_scratch_a_.get(), 0, hidden_bytes);
    if (logits_out != nullptr) {
      command_list_->CopyBufferRegion(
        output_readback_.get(),
        hidden_bytes,
        final_logits,
        0,
        static_cast<std::uint64_t>(vocab_size) * sizeof(float));
    }
    command_list_->CopyBufferRegion(
      output_readback_.get(),
      hidden_bytes + static_cast<std::uint64_t>(vocab_size) * sizeof(float),
      attention_qkv_.get(),
      0,
      sizeof(std::uint32_t));
    stamp("readback-copies");
    if (timestamps_on && timestamp_labels_.size() >= 2) {
      command_list_->ResolveQueryData(
        timestamp_heap_.get(),
        D3D12_QUERY_TYPE_TIMESTAMP,
        0,
        static_cast<UINT>(timestamp_labels_.size()),
        timestamp_readback_.get(),
        0);
    }

    const auto cpu_phase_recorded = std::chrono::steady_clock::now();
    hr = command_list_->Close();
    if (FAILED(hr)) {
      error_message = hresult_message("ID3D12GraphicsCommandList::Close", hr);
      return false;
    }
    if (!submit_and_wait(error_message)) {
      return false;
    }
    const auto cpu_phase_waited = std::chrono::steady_clock::now();
    telemetry_.token_graph_submission_count += 1;
    if (timestamps_on) {
      cpu_record_ms_ += std::chrono::duration<double, std::milli>(
        cpu_phase_recorded - cpu_phase_entry).count();
      cpu_wait_ms_ += std::chrono::duration<double, std::milli>(
        cpu_phase_waited - cpu_phase_recorded).count();
      cpu_tail_pending_ = cpu_phase_waited;
      cpu_phase_graphs_ += 1;
      accumulate_timestamps();
    }
    // Re-validate the static upload cache for the next token; submit_and_wait
    // cleared it (any other execution path may clobber the shared buffers).
    token_graph_upload_model_ = output.weight_key;
    token_graph_upload_bytes_ = upload_bytes;
    token_graph_upload_had_kv_ = has_initial_kv_upload;
    {
      std::ostringstream trace;
      trace
        << "gemma token graph: submissions="
        << (telemetry_.submission_count - submission_count_at_entry)
        << " total=" << telemetry_.submission_count;
      trace_d3d12(trace.str());
    }
    for (const std::uint64_t cache_id : appended_this_token) {
      gemma_kv_cache_.at(cache_id).positions += 1;
    }

    const std::uint8_t *readback =
      static_cast<const std::uint8_t *>(mapped_output_readback_);
    hidden_out->resize(hidden_width);
    std::memcpy(hidden_out->data(), readback, hidden_bytes);
    if (logits_out != nullptr) {
      logits_out->resize(vocab_size);
      std::memcpy(
        logits_out->data(),
        readback + hidden_bytes,
        static_cast<std::uint64_t>(vocab_size) * sizeof(float));
    }
    std::memcpy(
      argmax_token,
      readback + hidden_bytes + static_cast<std::uint64_t>(vocab_size) * sizeof(float),
      sizeof(std::uint32_t));
    telemetry_.input_upload_bytes += hidden_bytes
      + uploaded_initial_kv_bytes
      + (static_upload_valid
        ? ple_token_inputs_bytes
        : constants_bytes
          + static_cast<std::uint64_t>(max_attention_config_floats) * sizeof(float));
    telemetry_.output_readback_bytes += hidden_bytes + sizeof(std::uint32_t)
      + (logits_out != nullptr
        ? static_cast<std::uint64_t>(vocab_size) * sizeof(float)
        : 0);
    last_input_identity_ = nullptr;
    last_input_bytes_ = 0;
    if (timestamps_on) {
      // Completes the tail measurement started right after submit_and_wait.
      // The every-50-graph flush misses only the in-flight graph's tail.
      cpu_tail_ms_ += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - cpu_tail_pending_).count();
    }
    return true;
  }

  bool compute_gemma_attention(
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
    std::string &error_message,
    bool defer_submission = false) {
    if (!capabilities_.supported) {
      error_message = capabilities_.reason;
      return false;
    }
    if (!ensure_gemma_attention_pipelines(error_message)) {
      return false;
    }
    const bool appends_kv = key != nullptr && value != nullptr;
    const std::uint32_t query_size = config.head_count * config.head_dim;
    const std::uint32_t kv_size = config.kv_head_count * config.head_dim;
    if ((!defer_submission && attention_context == nullptr)
      || deferred_attention_active_
      || config.cache_id == 0
      || config.max_positions == 0
      || config.position >= config.max_positions
      || config.head_count == 0
      || config.kv_head_count == 0
      || config.head_dim == 0
      || config.head_dim > 512
      || (config.head_dim & 1u) != 0u
      || query.row_width != normed_hidden.size()
      || query.row_count != query_size
      || query_norm.size() != config.head_dim
      || key_norm.size() != config.head_dim
      || (appends_kv
        && (key->row_width != normed_hidden.size()
          || value->row_width != normed_hidden.size()
          || key->row_count != kv_size
          || value->row_count != kv_size))
      || ((key == nullptr) != (value == nullptr))) {
      std::ostringstream detail;
      detail
        << "D3D12 Gemma attention shapes are inconsistent"
        << " (hidden=" << normed_hidden.size()
        << ", q=" << query.row_count << "x" << query.row_width
        << ", expectedQ=" << query_size
        << ", qNorm=" << query_norm.size()
        << ", kNorm=" << key_norm.size()
        << ", headDim=" << config.head_dim
        << ", heads=" << config.head_count
        << ", kvHeads=" << config.kv_head_count;
      if (appends_kv) {
        detail
          << ", k=" << key->row_count << "x" << key->row_width
          << ", v=" << value->row_count << "x" << value->row_width
          << ", expectedKV=" << kv_size;
      }
      detail << ").";
      error_message = detail.str();
      return false;
    }
    if (!stage_resident(query, error_message)
      || (appends_kv
        && (!stage_resident(*key, error_message) || !stage_resident(*value, error_message)))) {
      return false;
    }

    const std::uint32_t words_per_position = (kv_size + 1u) / 2u;
    const std::uint64_t cache_words =
      static_cast<std::uint64_t>(config.max_positions) * words_per_position * 2u;
    const std::uint64_t cache_bytes = cache_words * sizeof(std::uint32_t);
    auto cache_iterator = gemma_kv_cache_.find(config.cache_id);
    const bool new_cache = cache_iterator == gemma_kv_cache_.end();
    if (new_cache) {
      if (initial_keys.size() != static_cast<std::size_t>(config.position) * kv_size
        || initial_values.size() != static_cast<std::size_t>(config.position) * kv_size) {
        error_message = "D3D12 Gemma attention initial KV payload does not match the token position.";
        return false;
      }
      DeviceKvCache cache;
      cache.max_positions = config.max_positions;
      cache.kv_size = kv_size;
      cache.words_per_position = words_per_position;
      cache.positions = config.position;
      std::uint64_t allocated_bytes = 0;
      if (!ensure_buffer(
        cache.buffer,
        cache_bytes,
        allocated_bytes,
        D3D12_HEAP_TYPE_DEFAULT,
        D3D12_RESOURCE_STATE_COPY_DEST,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        error_message)) {
        return false;
      }
      cache.state = D3D12_RESOURCE_STATE_COPY_DEST;
      cache_iterator = gemma_kv_cache_.emplace(config.cache_id, std::move(cache)).first;
    }
    DeviceKvCache &cache = cache_iterator->second;
    if (cache.max_positions != config.max_positions
      || cache.kv_size != kv_size
      || cache.words_per_position != words_per_position) {
      error_message = "D3D12 Gemma attention cache shape changed within a session.";
      return false;
    }
    const std::uint32_t expected_positions = appends_kv ? config.position : config.position + 1u;
    if (cache.positions != expected_positions) {
      error_message = "D3D12 Gemma attention cache position is inconsistent with the decode request.";
      return false;
    }

    const std::uint32_t qkv_floats = query_size + (appends_kv ? kv_size * 2u : 0u);
    const std::uint64_t hidden_bytes = normed_hidden.size() * sizeof(float);
    const std::uint32_t query_norm_offset = 3u;
    const std::uint32_t key_norm_offset = query_norm_offset + config.head_dim;
    const std::uint32_t rope_factor_offset = key_norm_offset + config.head_dim;
    const std::uint32_t rope_factor_count = config.head_dim / 2u;
    const std::uint32_t config_floats = rope_factor_offset + rope_factor_count;
    const std::uint64_t config_bytes = static_cast<std::uint64_t>(config_floats) * sizeof(float);
    const std::uint64_t initial_half_count =
      static_cast<std::uint64_t>(config.position) * kv_size * 2u;
    const std::uint64_t initial_kv_bytes = new_cache
      ? ((initial_half_count + 1u) / 2u) * sizeof(std::uint32_t)
      : 0;
    const std::uint64_t hidden_upload_offset = 0;
    const std::uint64_t config_upload_offset = align_up(hidden_bytes, 256);
    const std::uint64_t kv_upload_offset = align_up(config_upload_offset + config_bytes, 256);
    const std::uint64_t upload_bytes = kv_upload_offset + initial_kv_bytes;
    const std::uint64_t qkv_bytes = static_cast<std::uint64_t>(qkv_floats) * sizeof(float);
    const std::uint64_t query_bytes = static_cast<std::uint64_t>(query_size) * sizeof(float);

    if (!ensure_persistent_upload_buffer(
      input_upload_, upload_bytes, input_upload_bytes_, mapped_input_upload_, error_message)
      || !ensure_srv_buffer(
        attention_input_, hidden_bytes, attention_input_bytes_, attention_input_state_, error_message)
      || !ensure_srv_buffer(
        attention_config_, config_bytes, attention_config_bytes_, attention_config_state_, error_message)) {
      return false;
    }
    const auto ensure_uav = [&](
      ComPtr<ID3D12Resource> &resource,
      std::uint64_t required,
      std::uint64_t &allocated,
      D3D12_RESOURCE_STATES &state) {
      const std::uint64_t previous = allocated;
      if (!ensure_buffer(
        resource,
        required,
        allocated,
        D3D12_HEAP_TYPE_DEFAULT,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        error_message)) {
        return false;
      }
      if (previous != allocated) {
        state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
      }
      return true;
    };
    if (!ensure_uav(attention_qkv_, qkv_bytes, attention_qkv_bytes_, attention_qkv_state_)
      || !ensure_uav(attention_query_, query_bytes, attention_query_bytes_, attention_query_state_)
      || !ensure_uav(attention_context_, query_bytes, attention_context_bytes_, attention_context_state_)
      || (!defer_submission
        && !ensure_persistent_readback_buffer(
          attention_readback_,
          query_bytes,
          attention_readback_bytes_,
          mapped_attention_readback_,
          error_message))) {
      return false;
    }

    std::uint8_t *upload = static_cast<std::uint8_t *>(mapped_input_upload_);
    std::memcpy(upload + hidden_upload_offset, normed_hidden.data(), hidden_bytes);
    float *config_upload = reinterpret_cast<float *>(upload + config_upload_offset);
    config_upload[0] = config.rms_epsilon;
    config_upload[1] = config.rope_freq_base;
    config_upload[2] = config.rope_freq_scale;
    std::memcpy(
      config_upload + query_norm_offset,
      query_norm.data(),
      query_norm.size() * sizeof(float));
    std::memcpy(
      config_upload + key_norm_offset,
      key_norm.data(),
      key_norm.size() * sizeof(float));
    for (std::uint32_t index = 0; index < rope_factor_count; index += 1) {
      config_upload[rope_factor_offset + index] =
        index < rope_freq_factors.size() && rope_freq_factors[index] > 0.0f
        ? rope_freq_factors[index]
        : 1.0f;
    }
    if (new_cache && initial_half_count > 0) {
      std::uint16_t *packed = reinterpret_cast<std::uint16_t *>(upload + kv_upload_offset);
      const std::uint64_t key_capacity_halves =
        static_cast<std::uint64_t>(config.max_positions) * kv_size;
      pack_kv_lane_halves(packed, initial_keys);
      const std::uint64_t value_upload_half_offset =
        static_cast<std::uint64_t>(config.position) * kv_size;
      pack_kv_lane_halves(packed + value_upload_half_offset, initial_values);
      (void)key_capacity_halves;
    }

    HRESULT hr = command_allocator_->Reset();
    if (FAILED(hr)) {
      error_message = hresult_message("ID3D12CommandAllocator::Reset", hr);
      return false;
    }
    hr = command_list_->Reset(command_allocator_.get(), nullptr);
    if (FAILED(hr)) {
      error_message = hresult_message("ID3D12GraphicsCommandList::Reset", hr);
      return false;
    }
    transition(attention_input_.get(), attention_input_state_, D3D12_RESOURCE_STATE_COPY_DEST);
    command_list_->CopyBufferRegion(
      attention_input_.get(), 0, input_upload_.get(), hidden_upload_offset, hidden_bytes);
    transition(attention_input_.get(), attention_input_state_, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    transition(attention_config_.get(), attention_config_state_, D3D12_RESOURCE_STATE_COPY_DEST);
    command_list_->CopyBufferRegion(
      attention_config_.get(), 0, input_upload_.get(), config_upload_offset, config_bytes);
    transition(attention_config_.get(), attention_config_state_, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    if (new_cache && initial_kv_bytes > 0) {
      const std::uint64_t populated_half_bytes =
        static_cast<std::uint64_t>(config.position) * kv_size * sizeof(std::uint16_t);
      command_list_->CopyBufferRegion(
        cache.buffer.get(), 0, input_upload_.get(), kv_upload_offset, populated_half_bytes);
      const std::uint64_t value_cache_offset =
        static_cast<std::uint64_t>(config.max_positions) * words_per_position * sizeof(std::uint32_t);
      command_list_->CopyBufferRegion(
        cache.buffer.get(),
        value_cache_offset,
        input_upload_.get(),
        kv_upload_offset + populated_half_bytes,
        populated_half_bytes);
    }

    transition(attention_qkv_.get(), attention_qkv_state_, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    command_list_->SetComputeRootSignature(root_signature_.get());
    use_counter_ += 1;
    const auto dispatch_matvec = [&](
      const D3d12ResidentWeightRef &op,
      std::uint32_t output_offset) {
      ResidentWeight &resident = weight_cache_.at(op.weight_key);
      resident.last_use = use_counter_;
      DispatchConstants constants;
      constants.row_width = op.row_width;
      constants.row_count = op.row_count;
      constants.batch_count = 1;
      constants.output_base = output_offset;
      constants.input_stride = op.row_width;
      constants.weight_layout = 1;
      ID3D12PipelineState *pipeline = scalar_b1_pipeline_.get();
      switch (op.format) {
        case D3d12WeightFormat::Q4_K: pipeline = q4k_b1_pipeline_.get(); break;
        case D3d12WeightFormat::Q5_K: pipeline = q5k_b1_pipeline_.get(); break;
        case D3d12WeightFormat::Q6_K: pipeline = q6k_b1_pipeline_.get(); break;
        case D3d12WeightFormat::Q8_0: pipeline = q8_0_b1_pipeline_.get(); break;
        case D3d12WeightFormat::Q4_0: pipeline = q4_0_b1_pipeline_.get(); break;
        case D3d12WeightFormat::F32:
        case D3d12WeightFormat::F16:
        case D3d12WeightFormat::BF16:
          constants.scalar_format = 0;
          break;
      }
      command_list_->SetPipelineState(pipeline);
      command_list_->SetComputeRoot32BitConstants(0, 8, &constants, 0);
      command_list_->SetComputeRootShaderResourceView(1, resident.buffer->GetGPUVirtualAddress());
      command_list_->SetComputeRootShaderResourceView(2, attention_input_->GetGPUVirtualAddress());
      command_list_->SetComputeRootUnorderedAccessView(3, attention_qkv_->GetGPUVirtualAddress());
      command_list_->Dispatch(static_cast<UINT>((op.row_count + 63u) / 64u), 1, 1);
    };
    dispatch_matvec(query, 0);
    if (appends_kv) {
      dispatch_matvec(*key, query_size);
      dispatch_matvec(*value, query_size + kv_size);
    }
    D3D12_RESOURCE_BARRIER qkv_barrier = {};
    qkv_barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    qkv_barrier.UAV.pResource = attention_qkv_.get();
    command_list_->ResourceBarrier(1, &qkv_barrier);
    transition(attention_qkv_.get(), attention_qkv_state_, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

    const auto dispatch_prepare = [&](
      std::uint32_t source_offset,
      std::uint32_t destination_offset,
      std::uint32_t head_count,
      std::uint32_t norm_offset,
      std::uint32_t flags,
      ID3D12Resource *destination) {
      DispatchConstants constants;
      constants.row_width = config.head_dim;
      constants.row_count = head_count;
      constants.batch_count = source_offset;
      constants.output_base = destination_offset;
      constants.scalar_format = flags;
      constants.input_stride = config.position;
      constants.weight_layout = norm_offset;
      constants.input_transform = rope_factor_offset;
      command_list_->SetPipelineState(gemma_prepare_head_pipeline_.get());
      command_list_->SetComputeRoot32BitConstants(0, 8, &constants, 0);
      command_list_->SetComputeRootShaderResourceView(1, attention_config_->GetGPUVirtualAddress());
      command_list_->SetComputeRootShaderResourceView(2, attention_qkv_->GetGPUVirtualAddress());
      command_list_->SetComputeRootUnorderedAccessView(3, destination->GetGPUVirtualAddress());
      command_list_->Dispatch(head_count, 1, 1);
    };
    transition(attention_query_.get(), attention_query_state_, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    dispatch_prepare(0, 0, config.head_count, query_norm_offset, 3u, attention_query_.get());
    D3D12_RESOURCE_BARRIER query_barrier = {};
    query_barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    query_barrier.UAV.pResource = attention_query_.get();
    command_list_->ResourceBarrier(1, &query_barrier);
    if (appends_kv) {
      transition(cache.buffer.get(), cache.state, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
      const std::uint32_t key_word_offset = config.position * words_per_position;
      const std::uint32_t value_word_offset =
        config.max_positions * words_per_position + key_word_offset;
      dispatch_prepare(
        query_size,
        key_word_offset,
        config.kv_head_count,
        key_norm_offset,
        7u,
        cache.buffer.get());
      D3D12_RESOURCE_BARRIER cache_barrier = {};
      cache_barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
      cache_barrier.UAV.pResource = cache.buffer.get();
      command_list_->ResourceBarrier(1, &cache_barrier);
      dispatch_prepare(
        query_size + kv_size,
        value_word_offset,
        config.kv_head_count,
        key_norm_offset,
        4u,
        cache.buffer.get());
      command_list_->ResourceBarrier(1, &cache_barrier);
    }

    transition(attention_query_.get(), attention_query_state_, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    transition(cache.buffer.get(), cache.state, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    transition(attention_context_.get(), attention_context_state_, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    const std::uint32_t cached_positions = config.position + 1u;
    const std::uint32_t first_source = config.attention_window > 0
      && cached_positions > config.attention_window
      ? cached_positions - config.attention_window
      : 0u;
    DispatchConstants attention_constants;
    attention_constants.row_width = config.head_dim;
    attention_constants.row_count = config.head_count;
    attention_constants.batch_count = config.kv_head_count;
    attention_constants.output_base = 0;
    attention_constants.scalar_format = cached_positions;
    attention_constants.input_stride = first_source;
    attention_constants.weight_layout = config.max_positions * words_per_position;
    attention_constants.input_transform = kv_size;
    command_list_->SetPipelineState(gemma_attention_tiled_enabled()
      ? gemma_online_attention_tiled_pipeline_.get()
      : gemma_online_attention_pipeline_.get());
    command_list_->SetComputeRoot32BitConstants(0, 8, &attention_constants, 0);
    command_list_->SetComputeRootShaderResourceView(1, cache.buffer->GetGPUVirtualAddress());
    command_list_->SetComputeRootShaderResourceView(2, attention_query_->GetGPUVirtualAddress());
    command_list_->SetComputeRootUnorderedAccessView(3, attention_context_->GetGPUVirtualAddress());
    command_list_->Dispatch(config.head_count, 1, 1);
    D3D12_RESOURCE_BARRIER context_barrier = {};
    context_barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    context_barrier.UAV.pResource = attention_context_.get();
    command_list_->ResourceBarrier(1, &context_barrier);
    if (defer_submission) {
      transition(
        attention_context_.get(),
        attention_context_state_,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
      deferred_attention_active_ = true;
      deferred_attention_cache_id_ = config.cache_id;
      deferred_attention_appends_kv_ = appends_kv;
      telemetry_.input_upload_bytes += hidden_bytes + config_bytes + initial_kv_bytes;
      return true;
    }
    transition(attention_context_.get(), attention_context_state_, D3D12_RESOURCE_STATE_COPY_SOURCE);
    command_list_->CopyBufferRegion(
      attention_readback_.get(), 0, attention_context_.get(), 0, query_bytes);

    hr = command_list_->Close();
    if (FAILED(hr)) {
      error_message = hresult_message("ID3D12GraphicsCommandList::Close", hr);
      return false;
    }
    if (!submit_and_wait(error_message)) {
      return false;
    }
    if (appends_kv) {
      cache.positions += 1;
    }
    telemetry_.input_upload_bytes += hidden_bytes + config_bytes + initial_kv_bytes;
    telemetry_.output_readback_bytes += query_bytes;
    attention_context->resize(query_size);
    std::memcpy(attention_context->data(), mapped_attention_readback_, query_bytes);
    last_input_identity_ = nullptr;
    last_input_bytes_ = 0;
    return true;
  }

  bool compute_gemma_layer(
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
    std::string &error_message) {
    const bool has_ple = ple_gate != nullptr
      && ple_projection != nullptr
      && ple_input != nullptr
      && ple_post_norm != nullptr;
    if ((ple_gate == nullptr) != (ple_projection == nullptr)
      || (ple_input == nullptr) != (ple_post_norm == nullptr)
      || ((ple_gate == nullptr) != (ple_input == nullptr))) {
      error_message = "D3D12 Gemma layer PLE arguments are incomplete.";
      return false;
    }
    if (!stage_resident(attn_out, error_message)
      || !stage_resident(gate, error_message)
      || !stage_resident(up, error_message)
      || !stage_resident(down, error_message)
      || (has_ple
        && (!stage_resident(*ple_gate, error_message)
          || !stage_resident(*ple_projection, error_message)))) {
      return false;
    }
    if (!compute_gemma_attention(
      query,
      key,
      value,
      normed_hidden,
      query_norm,
      key_norm,
      rope_freq_factors,
      initial_keys,
      initial_values,
      config,
      nullptr,
      error_message,
      true)) {
      return false;
    }
    if (!compute_layer_tail(
      attn_out,
      gate,
      up,
      down,
      ple_gate,
      ple_projection,
      {},
      hidden_state,
      post_attention_norm,
      ffn_norm,
      post_ffw_norm,
      ple_input,
      ple_post_norm,
      config.rms_epsilon,
      hidden_out,
      error_message,
      true)) {
      if (deferred_attention_active_) {
        command_list_->Close();
        deferred_attention_active_ = false;
        deferred_attention_cache_id_ = 0;
        deferred_attention_appends_kv_ = false;
      }
      return false;
    }
    return true;
  }

  // Whole Gemma decode layer tail in ONE submission (batch 1):
  //   t   = attn_out x context                     (matmul -> A[0,h))
  //   res = hidden + rmsnorm(t, post_attn_w)       (norm+add -> B[0,h))
  //   f   = rmsnorm(res, ffn_w)                    (norm -> A[0,h))
  //   g,u = gate x f, up x f                       (matmuls -> ffn_scratch_)
  //   a   = gelu(g)*u                              (gelu -> ffn_activation_)
  //   d   = down x a                               (matmul -> B[h,2h))
  //   out = res + rmsnorm(d, post_ffw_w)           (norm+add -> output[0,h))
  // The packed input upload carries [context, hidden, three norm vectors].
  bool compute_layer_tail(
    const D3d12ResidentWeightRef &attn_out,
    const D3d12ResidentWeightRef &gate,
    const D3d12ResidentWeightRef &up,
    const D3d12ResidentWeightRef &down,
    const D3d12ResidentWeightRef *ple_gate,
    const D3d12ResidentWeightRef *ple_projection,
    const std::vector<float> &attention_context,
    const std::vector<float> &hidden_state,
    const std::vector<float> &post_attention_norm,
    const std::vector<float> &ffn_norm,
    const std::vector<float> &post_ffw_norm,
    const std::vector<float> *ple_input,
    const std::vector<float> *ple_post_norm,
    float rms_epsilon,
    std::vector<float> *hidden_out,
    std::string &error_message,
    bool append_to_deferred_attention = false,
    // Llama-family topology: no post-attention or post-FFW norm (those two
    // points are plain residual adds) and SwiGLU instead of GELU. Gemma callers
    // leave these false and their recorded command list is unchanged.
    bool skip_post_attention_norm = false,
    bool skip_post_ffw_norm = false,
    bool use_silu = false) {
    if (!capabilities_.supported) {
      error_message = capabilities_.reason;
      return false;
    }
    const std::uint32_t context_width = append_to_deferred_attention
      ? attn_out.row_width
      : static_cast<std::uint32_t>(attention_context.size());
    const std::uint32_t hidden_width = static_cast<std::uint32_t>(hidden_state.size());
    const std::uint32_t ffn_width = gate.row_count;
    const bool has_ple = ple_gate != nullptr
      && ple_projection != nullptr
      && ple_input != nullptr
      && ple_post_norm != nullptr;
    const std::uint32_t ple_width = has_ple ? ple_gate->row_count : 0;
    if (hidden_out == nullptr
      || context_width == 0
      || (!append_to_deferred_attention && attention_context.size() != context_width)
      || hidden_width == 0
      || attn_out.row_width != context_width
      || attn_out.row_count != hidden_width
      || gate.row_width != hidden_width
      || up.row_width != hidden_width
      || up.row_count != ffn_width
      || down.row_width != ffn_width
      || down.row_count != hidden_width
      || (!skip_post_attention_norm && post_attention_norm.size() != hidden_width)
      || ffn_norm.size() != hidden_width
      || (!skip_post_ffw_norm && post_ffw_norm.size() != hidden_width)
      || (has_ple
        && (ple_gate->row_width != hidden_width
          || ple_projection->row_width != ple_width
          || ple_projection->row_count != hidden_width
          || ple_input->size() != ple_width
          || ple_post_norm->size() != hidden_width))) {
      error_message = "D3D12 layer tail shapes are inconsistent.";
      return false;
    }

    if (!stage_resident(attn_out, error_message)
      || !stage_resident(gate, error_message)
      || !stage_resident(up, error_message)
      || !stage_resident(down, error_message)
      || (has_ple
        && (!stage_resident(*ple_gate, error_message)
          || !stage_resident(*ple_projection, error_message)))) {
      return false;
    }

    // Packed input: [context | hidden | post_attn_w | ffn_w | post_ffw_w].
    const std::uint32_t hidden_offset = context_width;
    const std::uint32_t post_attn_offset = hidden_offset + hidden_width;
    const std::uint32_t ffn_norm_offset = post_attn_offset + hidden_width;
    const std::uint32_t post_ffw_offset = ffn_norm_offset + hidden_width;
    const std::uint32_t ple_input_offset = post_ffw_offset + hidden_width;
    const std::uint32_t ple_post_norm_offset = ple_input_offset + ple_width;
    const std::uint64_t input_floats = has_ple
      ? static_cast<std::uint64_t>(ple_post_norm_offset) + hidden_width
      : static_cast<std::uint64_t>(post_ffw_offset) + hidden_width;
    const std::uint64_t input_bytes = input_floats * sizeof(float);
    // 3x hidden: [0,h) res, [h,2h) down output / hidden copy, [2h,3h) the
    // dual-norm kernel's FFN-norm output (see main_rms_norm_dual).
    const std::uint64_t scratch_ab_bytes = static_cast<std::uint64_t>(hidden_width) * 3 * sizeof(float);
    const std::uint64_t ffn_scratch_needed =
      static_cast<std::uint64_t>(std::max(ffn_width, ple_width)) * 2 * sizeof(float);
    const std::uint64_t activation_needed =
      static_cast<std::uint64_t>(std::max(ffn_width, ple_width)) * sizeof(float);
    const std::uint64_t output_bytes = static_cast<std::uint64_t>(hidden_width) * sizeof(float);

    if (append_to_deferred_attention != deferred_attention_active_) {
      error_message = "D3D12 layer tail deferred-attention state is inconsistent.";
      return false;
    }
    if (!ensure_persistent_upload_buffer(
      layer_tail_upload_,
      input_bytes,
      layer_tail_upload_bytes_,
      mapped_layer_tail_upload_,
      error_message)) {
      return false;
    }
    if (!ensure_srv_buffer(input_default_, input_bytes, input_default_bytes_, input_state_, error_message)) {
      return false;
    }
    const auto ensure_uav_scratch = [&](
      ComPtr<ID3D12Resource> &resource,
      std::uint64_t required,
      std::uint64_t &allocated,
      D3D12_RESOURCE_STATES &state) {
      const std::uint64_t previous = allocated;
      if (!ensure_buffer(
        resource,
        required,
        allocated,
        D3D12_HEAP_TYPE_DEFAULT,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        error_message)) {
        return false;
      }
      if (previous != allocated) {
        state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
      }
      return true;
    };
    if (!ensure_uav_scratch(layer_scratch_a_, scratch_ab_bytes, layer_scratch_a_bytes_, layer_scratch_a_state_)
      || !ensure_uav_scratch(layer_scratch_b_, scratch_ab_bytes, layer_scratch_b_bytes_, layer_scratch_b_state_)
      || !ensure_uav_scratch(ffn_scratch_, ffn_scratch_needed, ffn_scratch_bytes_, ffn_scratch_state_)
      || !ensure_uav_scratch(ffn_activation_, activation_needed, ffn_activation_bytes_, ffn_activation_state_)) {
      return false;
    }
    if (!ensure_output_buffer(output_default_, output_bytes, output_default_bytes_, output_state_, error_message)) {
      return false;
    }
    if (!uma_output_
      && !ensure_persistent_readback_buffer(
        output_readback_, output_bytes, output_readback_bytes_, mapped_output_readback_, error_message)) {
      return false;
    }

    float *staged_inputs = static_cast<float *>(mapped_layer_tail_upload_);
    if (!append_to_deferred_attention) {
      std::memcpy(staged_inputs, attention_context.data(), context_width * sizeof(float));
    }
    std::memcpy(staged_inputs + hidden_offset, hidden_state.data(), hidden_width * sizeof(float));
    // A skipped norm has no weights to stage; the kernel ignores the slot, but
    // the region is zeroed rather than left holding a previous token's values
    // so a mis-set flag fails loudly instead of reading plausible garbage.
    if (skip_post_attention_norm) {
      std::memset(staged_inputs + post_attn_offset, 0, hidden_width * sizeof(float));
    } else {
      std::memcpy(staged_inputs + post_attn_offset, post_attention_norm.data(), hidden_width * sizeof(float));
    }
    std::memcpy(staged_inputs + ffn_norm_offset, ffn_norm.data(), hidden_width * sizeof(float));
    if (skip_post_ffw_norm) {
      std::memset(staged_inputs + post_ffw_offset, 0, hidden_width * sizeof(float));
    } else {
      std::memcpy(staged_inputs + post_ffw_offset, post_ffw_norm.data(), hidden_width * sizeof(float));
    }
    if (has_ple) {
      std::memcpy(staged_inputs + ple_input_offset, ple_input->data(), ple_width * sizeof(float));
      std::memcpy(staged_inputs + ple_post_norm_offset, ple_post_norm->data(), hidden_width * sizeof(float));
    }

    HRESULT hr = S_OK;
    if (!append_to_deferred_attention) {
      hr = command_allocator_->Reset();
      if (FAILED(hr)) {
        error_message = hresult_message("ID3D12CommandAllocator::Reset", hr);
        return false;
      }
      hr = command_list_->Reset(command_allocator_.get(), nullptr);
      if (FAILED(hr)) {
        error_message = hresult_message("ID3D12GraphicsCommandList::Reset", hr);
        return false;
      }
    }

    transition(output_default_.get(), output_state_, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    transition(layer_scratch_a_.get(), layer_scratch_a_state_, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    transition(layer_scratch_b_.get(), layer_scratch_b_state_, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    transition(ffn_scratch_.get(), ffn_scratch_state_, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    transition(ffn_activation_.get(), ffn_activation_state_, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    transition(input_default_.get(), input_state_, D3D12_RESOURCE_STATE_COPY_DEST);
    command_list_->CopyBufferRegion(input_default_.get(), 0, layer_tail_upload_.get(), 0, input_bytes);
    transition(input_default_.get(), input_state_, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

    use_counter_ += 1;
    ResidentWeight &attn_resident = weight_cache_.at(attn_out.weight_key);
    ResidentWeight &gate_resident = weight_cache_.at(gate.weight_key);
    ResidentWeight &up_resident = weight_cache_.at(up.weight_key);
    ResidentWeight &down_resident = weight_cache_.at(down.weight_key);
    ResidentWeight *ple_gate_resident = has_ple ? &weight_cache_.at(ple_gate->weight_key) : nullptr;
    ResidentWeight *ple_projection_resident = has_ple
      ? &weight_cache_.at(ple_projection->weight_key)
      : nullptr;
    attn_resident.last_use = use_counter_;
    gate_resident.last_use = use_counter_;
    up_resident.last_use = use_counter_;
    down_resident.last_use = use_counter_;
    if (has_ple) {
      ple_gate_resident->last_use = use_counter_;
      ple_projection_resident->last_use = use_counter_;
    }

    command_list_->SetComputeRootSignature(root_signature_.get());

    const auto uav_barrier = [&](ID3D12Resource *resource) {
      D3D12_RESOURCE_BARRIER barrier = {};
      barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
      barrier.UAV.pResource = resource;
      command_list_->ResourceBarrier(1, &barrier);
    };

    const auto dispatch_matvec = [&](
      const D3d12ResidentWeightRef &op,
      ID3D12Resource *weight_buffer,
      ID3D12Resource *input_buffer,
      std::uint32_t input_offset,
      std::uint32_t input_stride,
      ID3D12Resource *result_buffer,
      std::uint32_t result_offset) {
      DispatchConstants constants;
      constants.row_width = op.row_width;
      constants.row_count = op.row_count;
      constants.batch_count = 1;
      constants.output_base = result_offset;
      constants.input_stride = input_stride;
      constants.weight_layout = 1;
      ID3D12PipelineState *pipeline = scalar_b1_pipeline_.get();
      switch (op.format) {
        case D3d12WeightFormat::Q4_K:
          pipeline = q4k_b1_pipeline_.get();
          break;
        case D3d12WeightFormat::Q5_K:
          pipeline = q5k_b1_pipeline_.get();
          break;
        case D3d12WeightFormat::Q6_K:
          pipeline = q6k_b1_pipeline_.get();
          break;
        case D3d12WeightFormat::Q8_0:
          pipeline = q8_0_b1_pipeline_.get();
          break;
        case D3d12WeightFormat::Q4_0:
          pipeline = q4_0_b1_pipeline_.get();
          break;
        case D3d12WeightFormat::F32:
        case D3d12WeightFormat::F16:
        case D3d12WeightFormat::BF16:
          constants.scalar_format = 0;
          break;
      }
      // Narrow-output k-split lane: one thread per row caps occupancy at
      // row_count, so 1-2.5k-row matvecs (ffn_down, attn_out, K/V) run the
      // machine at a fraction of full fill while streaming ~40% of the
      // per-token bytes. Split k four ways into ffn_activation_ (free at
      // every narrow-matvec site: fused gelu vacated it in the FFN chain,
      // and the PLE writes it only after its own barrier), then reduce into
      // the real destination. 4e-3-gated: accumulation order changes.
      constexpr std::uint32_t k_split_count = 4;
      if (ksplit_enabled()
        && op.format == D3d12WeightFormat::Q4_K
        && q4k_ksplit_pipeline_
        && ksplit_reduce_pipeline_
        && op.row_count <= 4096u
        && op.row_width >= 2048u
        && (op.row_width & 255u) == 0u
        && static_cast<std::uint64_t>(k_split_count) * op.row_count * sizeof(float) <= ffn_activation_bytes_) {
        transition(ffn_activation_.get(), ffn_activation_state_, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        constants.batch_count = k_split_count;
        constants.output_base = 0;
        command_list_->SetPipelineState(q4k_ksplit_pipeline_.get());
        command_list_->SetComputeRoot32BitConstants(0, 8, &constants, 0);
        command_list_->SetComputeRootShaderResourceView(1, weight_buffer->GetGPUVirtualAddress());
        command_list_->SetComputeRootShaderResourceView(
          2,
          input_buffer->GetGPUVirtualAddress() + static_cast<UINT64>(input_offset) * sizeof(float));
        command_list_->SetComputeRootUnorderedAccessView(3, ffn_activation_->GetGPUVirtualAddress());
        command_list_->Dispatch(static_cast<UINT>((op.row_count + 63u) / 64u), k_split_count, 1);
        uav_barrier(ffn_activation_.get());
        transition(ffn_activation_.get(), ffn_activation_state_, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

        DispatchConstants reduce_constants;
        reduce_constants.row_width = op.row_width;
        reduce_constants.row_count = op.row_count;
        reduce_constants.batch_count = k_split_count;
        reduce_constants.output_base = result_offset;
        reduce_constants.weight_layout = 0;
        command_list_->SetPipelineState(ksplit_reduce_pipeline_.get());
        command_list_->SetComputeRoot32BitConstants(0, 8, &reduce_constants, 0);
        command_list_->SetComputeRootShaderResourceView(1, weight_buffer->GetGPUVirtualAddress());
        command_list_->SetComputeRootShaderResourceView(2, ffn_activation_->GetGPUVirtualAddress());
        command_list_->SetComputeRootUnorderedAccessView(3, result_buffer->GetGPUVirtualAddress());
        command_list_->Dispatch(static_cast<UINT>((op.row_count + 63u) / 64u), 1, 1);
        return;
      }

      // The token graph already chooses among the wide/pair/direct variants for
      // exactly these shapes; this lane only ever ran the 64-row kernel, so the
      // layer tail left the faster kernel on the table. The selftest measures
      // wide128 at 1.03 ms/iter against 1.12 for 64-rows on a 10240x2560 Q4_K
      // matvec (1.09x), and the wide matvecs here (gate, up, down) are the bulk
      // of a decoded token's weight traffic. Same kernel and same accumulation
      // order as the token graph's wide128 lane, so results are unchanged;
      // wide128_enabled() remains the A/B and emergency-disable switch.
      ID3D12PipelineState *const wide128_pipeline =
        wide128_enabled() ? wide128_pipeline_for(op.format) : nullptr;
      const bool use_wide128 = wide128_pipeline != nullptr;
      command_list_->SetPipelineState(use_wide128 ? wide128_pipeline : pipeline);
      command_list_->SetComputeRoot32BitConstants(0, 8, &constants, 0);
      command_list_->SetComputeRootShaderResourceView(1, weight_buffer->GetGPUVirtualAddress());
      // Root descriptors accept a byte offset directly; input_offset is in
      // floats so callers can address a region of a shared scratch buffer.
      command_list_->SetComputeRootShaderResourceView(
        2,
        input_buffer->GetGPUVirtualAddress() + static_cast<UINT64>(input_offset) * sizeof(float));
      command_list_->SetComputeRootUnorderedAccessView(3, result_buffer->GetGPUVirtualAddress());
      command_list_->Dispatch(
        static_cast<UINT>(use_wide128
          ? (op.row_count + 127u) / 128u
          : (op.row_count + 63u) / 64u),
        1,
        1);
    };

    const auto dispatch_rms_norm = [&](
      ID3D12Resource *vector_buffer,
      std::uint32_t x_offset,
      std::uint32_t add_offset,
      bool add_enabled,
      std::uint32_t weight_offset,
      ID3D12Resource *result_buffer,
      std::uint32_t result_offset,
      // Llama adds the attention and FFN results back into the hidden state
      // with no normalization at those points; Gemma normalizes first. bit2
      // makes the kernel a plain residual add for exactly those two steps.
      bool skip_norm = false) {
      DispatchConstants constants;
      constants.row_width = hidden_width;
      constants.row_count = x_offset;
      constants.batch_count = add_offset;
      constants.output_base = result_offset;
      constants.scalar_format = add_enabled ? 1 : 0;
      if (skip_norm) {
        constants.scalar_format |= 4u;
      }
      std::memcpy(&constants.input_stride, &rms_epsilon, sizeof(float));
      constants.weight_layout = weight_offset;
      command_list_->SetPipelineState(rms_norm_pipeline_.get());
      command_list_->SetComputeRoot32BitConstants(0, 8, &constants, 0);
      command_list_->SetComputeRootShaderResourceView(1, input_default_->GetGPUVirtualAddress());
      command_list_->SetComputeRootShaderResourceView(2, vector_buffer->GetGPUVirtualAddress());
      command_list_->SetComputeRootUnorderedAccessView(3, result_buffer->GetGPUVirtualAddress());
      command_list_->Dispatch(1, 1, 1);
    };

    // Steps 2+3 in one dispatch (see main_rms_norm_dual): res and f land in
    // the same result buffer, so one barrier covers both consumers.
    const auto dispatch_rms_norm_dual = [&](
      ID3D12Resource *vector_buffer,
      std::uint32_t x_offset,
      std::uint32_t add_offset,
      std::uint32_t first_weight_offset,
      std::uint32_t second_weight_offset,
      ID3D12Resource *result_buffer,
      std::uint32_t res_offset,
      std::uint32_t f_offset) {
      DispatchConstants constants;
      constants.row_width = hidden_width;
      constants.row_count = x_offset;
      constants.batch_count = add_offset;
      constants.output_base = res_offset;
      constants.scalar_format = second_weight_offset;
      std::memcpy(&constants.input_stride, &rms_epsilon, sizeof(float));
      constants.weight_layout = first_weight_offset;
      constants.input_transform = f_offset;
      command_list_->SetPipelineState(rms_norm_dual_pipeline_.get());
      command_list_->SetComputeRoot32BitConstants(0, 8, &constants, 0);
      command_list_->SetComputeRootShaderResourceView(1, input_default_->GetGPUVirtualAddress());
      command_list_->SetComputeRootShaderResourceView(2, vector_buffer->GetGPUVirtualAddress());
      command_list_->SetComputeRootUnorderedAccessView(3, result_buffer->GetGPUVirtualAddress());
      command_list_->Dispatch(1, 1, 1);
    };

    begin_stage_timestamps();
    // CPU-side phase split for this submission. The GPU stage stamps show only
    // what the GPU was busy with; per-layer submission cost is invisible to
    // them, and on this path it is 28 submissions per token.
    const auto cpu_phase_entry = std::chrono::steady_clock::now();
    stamp_stage("tail-start");

    // 1. t = attn_out x context -> A[0,h)
    dispatch_matvec(
      attn_out,
      attn_resident.buffer.get(),
      append_to_deferred_attention ? attention_context_.get() : input_default_.get(),
      0,
      context_width,
      layer_scratch_a_.get(),
      0);
    uav_barrier(layer_scratch_a_.get());
    stamp_stage("attn_out-matvec");

    // 2. res = hidden + rmsnorm(t, post_attn_w) -> B[0,h). The add source
    // (hidden) lives in the packed input buffer, so it is passed through the
    // norm-weight buffer binding (t0 raw) — but the kernel reads add from t1.
    // Instead, hidden is copied into A[h,2h) so x and add share t1.
    // (Copy recorded before the barrier chain below.)
    transition(layer_scratch_a_.get(), layer_scratch_a_state_, D3D12_RESOURCE_STATE_COPY_DEST);
    command_list_->CopyBufferRegion(
      layer_scratch_a_.get(),
      static_cast<std::uint64_t>(hidden_width) * sizeof(float),
      layer_tail_upload_.get(),
      static_cast<std::uint64_t>(hidden_offset) * sizeof(float),
      static_cast<std::uint64_t>(hidden_width) * sizeof(float));
    transition(layer_scratch_a_.get(), layer_scratch_a_state_, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    // The dual kernel fuses "norm+residual then norm" and spends scalar_format
    // on its second weight offset, so it has no room for the skip-norm flag.
    // A Llama layer's first step is a plain residual add, not a norm, so it
    // takes the split path; Gemma is unaffected.
    if (dual_norm_enabled() && hidden_width <= 4096 && !skip_post_attention_norm) {
      // 2+3 fused: one dispatch writes res -> B[0,h) AND f -> B[2h,3h)
      // (byte-identical to the two-dispatch sequence), so the pair costs one
      // barrier and scratch_a stays read-only for the rest of the FFN chain.
      transition(layer_scratch_b_.get(), layer_scratch_b_state_, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
      dispatch_rms_norm_dual(
        layer_scratch_a_.get(), 0, hidden_width,
        post_attn_offset, ffn_norm_offset,
        layer_scratch_b_.get(), 0, hidden_width * 2u);
      uav_barrier(layer_scratch_b_.get());
      transition(layer_scratch_b_.get(), layer_scratch_b_state_, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

      // 4. gate/up matmuls read f directly from B[2h,3h) -> ffn_scratch_
      dispatch_matvec(gate, gate_resident.buffer.get(), layer_scratch_b_.get(), hidden_width * 2u, hidden_width, ffn_scratch_.get(), 0);
      dispatch_matvec(up, up_resident.buffer.get(), layer_scratch_b_.get(), hidden_width * 2u, hidden_width, ffn_scratch_.get(), ffn_width);
    } else {
      dispatch_rms_norm(
        layer_scratch_a_.get(), 0, hidden_width, true, post_attn_offset,
        layer_scratch_b_.get(), 0, skip_post_attention_norm);
      uav_barrier(layer_scratch_b_.get());
      transition(layer_scratch_b_.get(), layer_scratch_b_state_, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

      // 3. f = rmsnorm(res, ffn_w) -> A[0,h)
      transition(layer_scratch_a_.get(), layer_scratch_a_state_, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
      dispatch_rms_norm(layer_scratch_b_.get(), 0, 0, false, ffn_norm_offset, layer_scratch_a_.get(), 0);
      uav_barrier(layer_scratch_a_.get());
      transition(layer_scratch_a_.get(), layer_scratch_a_state_, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

      // 4. gate/up matmuls -> ffn_scratch_
      dispatch_matvec(gate, gate_resident.buffer.get(), layer_scratch_a_.get(), 0, hidden_width, ffn_scratch_.get(), 0);
      dispatch_matvec(up, up_resident.buffer.get(), layer_scratch_a_.get(), 0, hidden_width, ffn_scratch_.get(), ffn_width);
    }
    uav_barrier(ffn_scratch_.get());
    transition(ffn_scratch_.get(), ffn_scratch_state_, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    // Covers the norm(s) and both wide matvecs: on a 3B Llama these two are
    // 8192x3072 each, the largest weight traffic in the layer.
    stamp_stage("norm+gate+up");

    // 5. gelu -> ffn_activation_
    {
      DispatchConstants gelu_constants;
      gelu_constants.row_count = ffn_width;
      gelu_constants.output_base = 0;
      // Identical buffer layout for both activations (see main_silu_glu);
      // only the elementwise function differs.
      command_list_->SetPipelineState(
        use_silu ? silu_glu_pipeline_.get() : gelu_pipeline_.get());
      command_list_->SetComputeRoot32BitConstants(0, 8, &gelu_constants, 0);
      command_list_->SetComputeRootShaderResourceView(1, ffn_scratch_->GetGPUVirtualAddress());
      command_list_->SetComputeRootShaderResourceView(2, ffn_scratch_->GetGPUVirtualAddress());
      command_list_->SetComputeRootUnorderedAccessView(3, ffn_activation_->GetGPUVirtualAddress());
      command_list_->Dispatch(static_cast<UINT>((ffn_width + 255u) / 256u), 1, 1);
    }
    uav_barrier(ffn_activation_.get());
    transition(ffn_activation_.get(), ffn_activation_state_, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    stamp_stage("swiglu");

    // 6. d = down x a -> B[h,2h)
    transition(layer_scratch_b_.get(), layer_scratch_b_state_, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    dispatch_matvec(down, down_resident.buffer.get(), ffn_activation_.get(), 0, ffn_width, layer_scratch_b_.get(), hidden_width);
    uav_barrier(layer_scratch_b_.get());
    transition(layer_scratch_b_.get(), layer_scratch_b_state_, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    stamp_stage("down-matvec");

    // 7. out = res + rmsnorm(d, post_ffw_w) -> output[0,h)
    dispatch_rms_norm(
      layer_scratch_b_.get(), hidden_width, 0, true, post_ffw_offset,
      output_default_.get(), 0, skip_post_ffw_norm);
    uav_barrier(output_default_.get());
    stamp_stage("final-residual");

    if (has_ple) {
      // 8. PLE gate(hidden) -> scratch[0,p), then copy the per-layer input
      // into scratch[p,2p) so the existing gelu-multiply kernel can consume
      // both vectors without a CPU round-trip.
      transition(output_default_.get(), output_state_, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
      transition(ffn_scratch_.get(), ffn_scratch_state_, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
      dispatch_matvec(
        *ple_gate,
        ple_gate_resident->buffer.get(),
        output_default_.get(),
        0,
        hidden_width,
        ffn_scratch_.get(),
        0);
      uav_barrier(ffn_scratch_.get());
      transition(ffn_scratch_.get(), ffn_scratch_state_, D3D12_RESOURCE_STATE_COPY_DEST);
      command_list_->CopyBufferRegion(
        ffn_scratch_.get(),
        static_cast<std::uint64_t>(ple_width) * sizeof(float),
        layer_tail_upload_.get(),
        static_cast<std::uint64_t>(ple_input_offset) * sizeof(float),
        static_cast<std::uint64_t>(ple_width) * sizeof(float));
      transition(ffn_scratch_.get(), ffn_scratch_state_, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

      // 9. gelu(gate) * PLE input -> activation, then project to hidden width.
      transition(ffn_activation_.get(), ffn_activation_state_, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
      DispatchConstants ple_gelu_constants;
      ple_gelu_constants.row_count = ple_width;
      command_list_->SetPipelineState(gelu_pipeline_.get());
      command_list_->SetComputeRoot32BitConstants(0, 8, &ple_gelu_constants, 0);
      command_list_->SetComputeRootShaderResourceView(1, ffn_scratch_->GetGPUVirtualAddress());
      command_list_->SetComputeRootShaderResourceView(2, ffn_scratch_->GetGPUVirtualAddress());
      command_list_->SetComputeRootUnorderedAccessView(3, ffn_activation_->GetGPUVirtualAddress());
      command_list_->Dispatch(static_cast<UINT>((ple_width + 255u) / 256u), 1, 1);
      uav_barrier(ffn_activation_.get());
      transition(ffn_activation_.get(), ffn_activation_state_, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

      transition(layer_scratch_a_.get(), layer_scratch_a_state_, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
      dispatch_matvec(
        *ple_projection,
        ple_projection_resident->buffer.get(),
        ffn_activation_.get(),
        0,
        ple_width,
        layer_scratch_a_.get(),
        0);
      uav_barrier(layer_scratch_a_.get());

      // 10. hidden + rmsnorm(PLE projection, post_norm) -> output.
      transition(output_default_.get(), output_state_, D3D12_RESOURCE_STATE_COPY_SOURCE);
      transition(layer_scratch_a_.get(), layer_scratch_a_state_, D3D12_RESOURCE_STATE_COPY_DEST);
      command_list_->CopyBufferRegion(
        layer_scratch_a_.get(),
        static_cast<std::uint64_t>(hidden_width) * sizeof(float),
        output_default_.get(),
        0,
        output_bytes);
      transition(layer_scratch_a_.get(), layer_scratch_a_state_, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
      transition(output_default_.get(), output_state_, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
      dispatch_rms_norm(
        layer_scratch_a_.get(),
        0,
        hidden_width,
        true,
        ple_post_norm_offset,
        output_default_.get(),
        0);
      uav_barrier(output_default_.get());
    }

    if (!uma_output_) {
      transition(output_default_.get(), output_state_, D3D12_RESOURCE_STATE_COPY_SOURCE);
      command_list_->CopyBufferRegion(output_readback_.get(), 0, output_default_.get(), 0, output_bytes);
    }

    // Resolve the stage queries into the readback buffer while the list is
    // still open; accumulate_timestamps() reads them after the fence.
    if (timestamps_active_ && timestamp_labels_.size() >= 2) {
      command_list_->ResolveQueryData(
        timestamp_heap_.get(),
        D3D12_QUERY_TYPE_TIMESTAMP,
        0,
        static_cast<UINT>(timestamp_labels_.size()),
        timestamp_readback_.get(),
        0);
    }

    hr = command_list_->Close();
    if (FAILED(hr)) {
      deferred_attention_active_ = false;
      deferred_attention_cache_id_ = 0;
      deferred_attention_appends_kv_ = false;
      error_message = hresult_message("ID3D12GraphicsCommandList::Close", hr);
      return false;
    }
    const auto cpu_phase_recorded = std::chrono::steady_clock::now();
    if (!submit_and_wait(error_message)) {
      deferred_attention_active_ = false;
      deferred_attention_cache_id_ = 0;
      deferred_attention_appends_kv_ = false;
      return false;
    }
    const auto cpu_phase_waited = std::chrono::steady_clock::now();
    cpu_record_ms_ += std::chrono::duration<double, std::milli>(
      cpu_phase_recorded - cpu_phase_entry).count();
    cpu_wait_ms_ += std::chrono::duration<double, std::milli>(
      cpu_phase_waited - cpu_phase_recorded).count();
    cpu_tail_pending_ = cpu_phase_waited;
    cpu_phase_graphs_ += 1;
    // After the fence: the query results are only valid once the GPU has
    // finished the list that wrote them.
    accumulate_timestamps();
    if (append_to_deferred_attention && deferred_attention_appends_kv_) {
      auto cache_iterator = gemma_kv_cache_.find(deferred_attention_cache_id_);
      if (cache_iterator == gemma_kv_cache_.end()) {
        deferred_attention_active_ = false;
        deferred_attention_cache_id_ = 0;
        deferred_attention_appends_kv_ = false;
        error_message = "D3D12 deferred attention cache disappeared before layer completion.";
        return false;
      }
      cache_iterator->second.positions += 1;
    }
    deferred_attention_active_ = false;
    deferred_attention_cache_id_ = 0;
    deferred_attention_appends_kv_ = false;
    telemetry_.input_upload_bytes += input_bytes;
    telemetry_.output_readback_bytes += output_bytes;

    const void *result_memory = uma_output_ ? mapped_output_direct_ : mapped_output_readback_;
    if (result_memory == nullptr) {
      error_message = "D3D12 output buffer is not mapped.";
      return false;
    }
    hidden_out->resize(hidden_width);
    std::memcpy(hidden_out->data(), result_memory, output_bytes);
    // Closes the third phase: everything after the fence (readback map, copy
    // out) that still belongs to this submission.
    cpu_tail_ms_ += std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - cpu_tail_pending_).count();
    // The input identity cache must not treat the next matmul call's input
    // as already uploaded: this call overwrote the shared upload buffer.
    last_input_identity_ = nullptr;
    last_input_bytes_ = 0;
    return true;
  }

  // Batched Llama layer tail in one submission. This is deliberately separate
  // from compute_layer_tail: Gemma's single-row PLE/post-norm topology is much
  // richer, while the Llama batch needs only two residual adds and one FFN
  // norm. Keeping the batch-major buffer layout explicit makes the B=1 Gemma
  // recording path completely untouched.
  bool compute_llama_layer_tail_batch(
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
    std::string &error_message) {
    if (!capabilities_.supported) {
      error_message = capabilities_.reason;
      return false;
    }
    const std::uint32_t context_width = attn_out.row_width;
    const std::uint32_t hidden_width = attn_out.row_count;
    const std::uint32_t ffn_width = gate.row_count;
    const std::uint64_t context_items =
      static_cast<std::uint64_t>(batch_count) * context_width;
    const std::uint64_t hidden_items =
      static_cast<std::uint64_t>(batch_count) * hidden_width;
    const std::uint64_t activation_items =
      static_cast<std::uint64_t>(batch_count) * ffn_width;
    if (batch_count < 2
      || packed_hidden_out == nullptr
      || context_width == 0
      || hidden_width == 0
      || packed_attention_contexts.size() != context_items
      || packed_hidden_states.size() != hidden_items
      || ffn_norm.size() != hidden_width
      || gate.row_width != hidden_width
      || up.row_width != hidden_width
      || up.row_count != ffn_width
      || down.row_width != ffn_width
      || down.row_count != hidden_width) {
      error_message = "D3D12 batched Llama layer tail shapes are inconsistent.";
      return false;
    }
    if (!stage_resident(attn_out, error_message)
      || !stage_resident(gate, error_message)
      || !stage_resident(up, error_message)
      || !stage_resident(down, error_message)) {
      return false;
    }

    // Upload [all contexts | all hidden rows | one shared FFN norm vector].
    const std::uint64_t hidden_offset = context_items;
    const std::uint64_t ffn_norm_offset = hidden_offset + hidden_items;
    const std::uint64_t input_floats = ffn_norm_offset + hidden_width;
    const std::uint64_t input_bytes = input_floats * sizeof(float);
    const std::uint64_t scratch_ab_bytes = hidden_items * 2 * sizeof(float);
    const std::uint64_t ffn_scratch_needed = activation_items * 2 * sizeof(float);
    const std::uint64_t activation_bytes = activation_items * sizeof(float);
    const std::uint64_t output_bytes = hidden_items * sizeof(float);

    if (!ensure_persistent_upload_buffer(
      layer_tail_upload_,
      input_bytes,
      layer_tail_upload_bytes_,
      mapped_layer_tail_upload_,
      error_message)) {
      return false;
    }
    if (!ensure_srv_buffer(input_default_, input_bytes, input_default_bytes_, input_state_, error_message)) {
      return false;
    }
    const auto ensure_uav_scratch = [&](
      ComPtr<ID3D12Resource> &resource,
      std::uint64_t required,
      std::uint64_t &allocated,
      D3D12_RESOURCE_STATES &state) {
      const std::uint64_t previous = allocated;
      if (!ensure_buffer(
        resource,
        required,
        allocated,
        D3D12_HEAP_TYPE_DEFAULT,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        error_message)) {
        return false;
      }
      if (previous != allocated) {
        state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
      }
      return true;
    };
    if (!ensure_uav_scratch(
        layer_scratch_a_, scratch_ab_bytes, layer_scratch_a_bytes_, layer_scratch_a_state_)
      || !ensure_uav_scratch(
        layer_scratch_b_, scratch_ab_bytes, layer_scratch_b_bytes_, layer_scratch_b_state_)
      || !ensure_uav_scratch(
        ffn_scratch_, ffn_scratch_needed, ffn_scratch_bytes_, ffn_scratch_state_)
      || !ensure_uav_scratch(
        ffn_activation_, activation_bytes, ffn_activation_bytes_, ffn_activation_state_)) {
      return false;
    }
    if (!ensure_output_buffer(
      output_default_, output_bytes, output_default_bytes_, output_state_, error_message)) {
      return false;
    }
    if (!uma_output_
      && !ensure_persistent_readback_buffer(
        output_readback_,
        output_bytes,
        output_readback_bytes_,
        mapped_output_readback_,
        error_message)) {
      return false;
    }

    float *staged = static_cast<float *>(mapped_layer_tail_upload_);
    std::memcpy(
      staged,
      packed_attention_contexts.data(),
      static_cast<std::size_t>(context_items * sizeof(float)));
    std::memcpy(
      staged + hidden_offset,
      packed_hidden_states.data(),
      static_cast<std::size_t>(hidden_items * sizeof(float)));
    std::memcpy(
      staged + ffn_norm_offset,
      ffn_norm.data(),
      static_cast<std::size_t>(hidden_width) * sizeof(float));

    HRESULT hr = command_allocator_->Reset();
    if (FAILED(hr)) {
      error_message = hresult_message("ID3D12CommandAllocator::Reset", hr);
      return false;
    }
    hr = command_list_->Reset(command_allocator_.get(), nullptr);
    if (FAILED(hr)) {
      error_message = hresult_message("ID3D12GraphicsCommandList::Reset", hr);
      return false;
    }

    transition(output_default_.get(), output_state_, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    transition(layer_scratch_a_.get(), layer_scratch_a_state_, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    transition(layer_scratch_b_.get(), layer_scratch_b_state_, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    transition(ffn_scratch_.get(), ffn_scratch_state_, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    transition(ffn_activation_.get(), ffn_activation_state_, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    transition(input_default_.get(), input_state_, D3D12_RESOURCE_STATE_COPY_DEST);
    command_list_->CopyBufferRegion(
      input_default_.get(), 0, layer_tail_upload_.get(), 0, input_bytes);
    transition(input_default_.get(), input_state_, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

    use_counter_ += 1;
    ResidentWeight &attn_resident = weight_cache_.at(attn_out.weight_key);
    ResidentWeight &gate_resident = weight_cache_.at(gate.weight_key);
    ResidentWeight &up_resident = weight_cache_.at(up.weight_key);
    ResidentWeight &down_resident = weight_cache_.at(down.weight_key);
    attn_resident.last_use = use_counter_;
    gate_resident.last_use = use_counter_;
    up_resident.last_use = use_counter_;
    down_resident.last_use = use_counter_;

    command_list_->SetComputeRootSignature(root_signature_.get());
    const auto uav_barrier = [&](ID3D12Resource *resource) {
      D3D12_RESOURCE_BARRIER barrier = {};
      barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
      barrier.UAV.pResource = resource;
      command_list_->ResourceBarrier(1, &barrier);
    };
    const auto dispatch_matmul = [&](
      const D3d12ResidentWeightRef &op,
      ID3D12Resource *weight_buffer,
      ID3D12Resource *input_buffer,
      std::uint32_t input_offset,
      std::uint32_t input_stride,
      ID3D12Resource *result_buffer,
      std::uint32_t output_base) {
      DispatchConstants constants;
      constants.row_width = op.row_width;
      constants.row_count = op.row_count;
      constants.batch_count = batch_count;
      constants.output_base = output_base;
      constants.input_stride = input_stride;
      constants.weight_layout = 1;
      ID3D12PipelineState *pipeline = scalar_pipeline_.get();
      switch (op.format) {
        case D3d12WeightFormat::Q4_K: pipeline = q4k_pipeline_.get(); break;
        case D3d12WeightFormat::Q5_K: pipeline = q5k_pipeline_.get(); break;
        case D3d12WeightFormat::Q6_K: pipeline = q6k_pipeline_.get(); break;
        case D3d12WeightFormat::Q8_0: pipeline = q8_0_pipeline_.get(); break;
        case D3d12WeightFormat::Q4_0: pipeline = q4_0_pipeline_.get(); break;
        case D3d12WeightFormat::F32:
        case D3d12WeightFormat::F16:
        case D3d12WeightFormat::BF16:
          constants.scalar_format = 0;
          break;
      }
      command_list_->SetPipelineState(pipeline);
      command_list_->SetComputeRoot32BitConstants(0, 8, &constants, 0);
      command_list_->SetComputeRootShaderResourceView(1, weight_buffer->GetGPUVirtualAddress());
      command_list_->SetComputeRootShaderResourceView(
        2,
        input_buffer->GetGPUVirtualAddress()
          + static_cast<UINT64>(input_offset) * sizeof(float));
      command_list_->SetComputeRootUnorderedAccessView(3, result_buffer->GetGPUVirtualAddress());
      command_list_->Dispatch(
        static_cast<UINT>((op.row_count + 63u) / 64u),
        static_cast<UINT>((batch_count + k_gpu_batch_tile - 1u) / k_gpu_batch_tile),
        1);
    };
    const auto dispatch_rms_norm = [&](
      ID3D12Resource *vector_buffer,
      std::uint32_t x_offset,
      std::uint32_t add_offset,
      bool add_enabled,
      std::uint32_t weight_offset,
      ID3D12Resource *result_buffer,
      std::uint32_t result_offset,
      bool skip_norm) {
      DispatchConstants constants;
      constants.row_width = hidden_width;
      constants.row_count = x_offset;
      constants.batch_count = add_offset;
      constants.output_base = result_offset;
      constants.scalar_format = add_enabled ? 1u : 0u;
      if (skip_norm) {
        constants.scalar_format |= 4u;
      }
      std::memcpy(&constants.input_stride, &rms_epsilon, sizeof(float));
      constants.weight_layout = weight_offset;
      command_list_->SetPipelineState(rms_norm_pipeline_.get());
      command_list_->SetComputeRoot32BitConstants(0, 8, &constants, 0);
      command_list_->SetComputeRootShaderResourceView(1, input_default_->GetGPUVirtualAddress());
      command_list_->SetComputeRootShaderResourceView(2, vector_buffer->GetGPUVirtualAddress());
      command_list_->SetComputeRootUnorderedAccessView(3, result_buffer->GetGPUVirtualAddress());
      command_list_->Dispatch(1, batch_count, 1);
    };

    // 1. Project all attention contexts into A[0,B*H).
    dispatch_matmul(
      attn_out,
      attn_resident.buffer.get(),
      input_default_.get(),
      0,
      context_width,
      layer_scratch_a_.get(),
      0);
    uav_barrier(layer_scratch_a_.get());

    // Copy the matching hidden rows beside the projected rows so one batched
    // residual dispatch can read both from the same SRV.
    transition(layer_scratch_a_.get(), layer_scratch_a_state_, D3D12_RESOURCE_STATE_COPY_DEST);
    command_list_->CopyBufferRegion(
      layer_scratch_a_.get(),
      hidden_items * sizeof(float),
      layer_tail_upload_.get(),
      hidden_offset * sizeof(float),
      hidden_items * sizeof(float));
    transition(layer_scratch_a_.get(), layer_scratch_a_state_, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

    // 2. res = hidden + attention_output -> B[0,B*H).
    dispatch_rms_norm(
      layer_scratch_a_.get(),
      0,
      static_cast<std::uint32_t>(hidden_items),
      true,
      static_cast<std::uint32_t>(ffn_norm_offset),
      layer_scratch_b_.get(),
      0,
      true);
    uav_barrier(layer_scratch_b_.get());
    transition(layer_scratch_b_.get(), layer_scratch_b_state_, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

    // 3. f = rmsnorm(res, ffn_norm) -> A[0,B*H).
    transition(layer_scratch_a_.get(), layer_scratch_a_state_, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    dispatch_rms_norm(
      layer_scratch_b_.get(),
      0,
      0,
      false,
      static_cast<std::uint32_t>(ffn_norm_offset),
      layer_scratch_a_.get(),
      0,
      false);
    uav_barrier(layer_scratch_a_.get());
    transition(layer_scratch_a_.get(), layer_scratch_a_state_, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

    // 4. Batched gate/up, SwiGLU, and down projection.
    dispatch_matmul(
      gate, gate_resident.buffer.get(), layer_scratch_a_.get(), 0, hidden_width,
      ffn_scratch_.get(), 0);
    dispatch_matmul(
      up, up_resident.buffer.get(), layer_scratch_a_.get(), 0, hidden_width,
      ffn_scratch_.get(), static_cast<std::uint32_t>(activation_items));
    uav_barrier(ffn_scratch_.get());
    transition(ffn_scratch_.get(), ffn_scratch_state_, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

    DispatchConstants silu_constants;
    silu_constants.row_count = static_cast<std::uint32_t>(activation_items);
    command_list_->SetPipelineState(silu_glu_pipeline_.get());
    command_list_->SetComputeRoot32BitConstants(0, 8, &silu_constants, 0);
    command_list_->SetComputeRootShaderResourceView(1, ffn_scratch_->GetGPUVirtualAddress());
    command_list_->SetComputeRootShaderResourceView(2, ffn_scratch_->GetGPUVirtualAddress());
    command_list_->SetComputeRootUnorderedAccessView(3, ffn_activation_->GetGPUVirtualAddress());
    command_list_->Dispatch(
      static_cast<UINT>((activation_items + 255u) / 256u), 1, 1);
    uav_barrier(ffn_activation_.get());
    transition(ffn_activation_.get(), ffn_activation_state_, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

    transition(layer_scratch_b_.get(), layer_scratch_b_state_, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    dispatch_matmul(
      down,
      down_resident.buffer.get(),
      ffn_activation_.get(),
      0,
      ffn_width,
      layer_scratch_b_.get(),
      static_cast<std::uint32_t>(hidden_items));
    uav_barrier(layer_scratch_b_.get());
    transition(layer_scratch_b_.get(), layer_scratch_b_state_, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

    // 5. out = res + down -> output[0,B*H).
    dispatch_rms_norm(
      layer_scratch_b_.get(),
      static_cast<std::uint32_t>(hidden_items),
      0,
      true,
      static_cast<std::uint32_t>(ffn_norm_offset),
      output_default_.get(),
      0,
      true);
    uav_barrier(output_default_.get());

    if (!uma_output_) {
      transition(output_default_.get(), output_state_, D3D12_RESOURCE_STATE_COPY_SOURCE);
      command_list_->CopyBufferRegion(
        output_readback_.get(), 0, output_default_.get(), 0, output_bytes);
    }
    hr = command_list_->Close();
    if (FAILED(hr)) {
      error_message = hresult_message("ID3D12GraphicsCommandList::Close", hr);
      return false;
    }
    if (!submit_and_wait(error_message)) {
      return false;
    }
    telemetry_.input_upload_bytes += input_bytes;
    telemetry_.output_readback_bytes += output_bytes;

    const void *result_memory = uma_output_ ? mapped_output_direct_ : mapped_output_readback_;
    if (result_memory == nullptr) {
      error_message = "D3D12 output buffer is not mapped.";
      return false;
    }
    packed_hidden_out->resize(static_cast<std::size_t>(hidden_items));
    std::memcpy(
      packed_hidden_out->data(),
      result_memory,
      static_cast<std::size_t>(output_bytes));
    last_input_identity_ = nullptr;
    last_input_bytes_ = 0;
    return true;
  }

  // Whole FFN block in one submission: gate and up matmuls write a device
  // scratch buffer, a gelu(gate)*up shader produces the activation, and the
  // down matmul consumes it — intermediates never leave the device and only
  // one fence round-trip is paid for the three projections.
  bool compute_ffn(
    const D3d12ResidentWeightRef &gate,
    const D3d12ResidentWeightRef &up,
    const D3d12ResidentWeightRef &down,
    const std::vector<float> &packed_inputs,
    std::uint32_t batch_count,
    const void *input_identity,
    bool use_silu,
    std::vector<float> *outputs,
    std::string &error_message) {
    if (!capabilities_.supported) {
      error_message = capabilities_.reason;
      return false;
    }
    if (batch_count == 0 || packed_inputs.empty() || outputs == nullptr) {
      error_message = "D3D12 FFN chain requires a non-empty input batch.";
      return false;
    }
    if (gate.row_width != up.row_width
      || gate.row_count != up.row_count
      || down.row_width != gate.row_count) {
      error_message = "D3D12 FFN chain shapes are inconsistent.";
      return false;
    }
    const std::uint64_t input_items = static_cast<std::uint64_t>(batch_count) * gate.row_width;
    if (packed_inputs.size() != input_items) {
      error_message = "D3D12 FFN chain packed input payload size does not match the requested shape.";
      return false;
    }

    if (!stage_resident(gate, error_message)
      || !stage_resident(up, error_message)
      || !stage_resident(down, error_message)) {
      return false;
    }

    const std::uint64_t activation_floats = static_cast<std::uint64_t>(batch_count) * gate.row_count;
    const std::uint64_t scratch_bytes = activation_floats * 2 * sizeof(float);
    const std::uint64_t output_floats = static_cast<std::uint64_t>(batch_count) * down.row_count;
    const std::uint64_t output_bytes = output_floats * sizeof(float);
    const std::uint64_t input_bytes = input_items * sizeof(float);
    const bool input_buffer_recreated = !input_default_ || input_default_bytes_ < input_bytes;

    if (!ensure_persistent_upload_buffer(
      input_upload_, input_bytes, input_upload_bytes_, mapped_input_upload_, error_message)) {
      return false;
    }
    if (!ensure_srv_buffer(input_default_, input_bytes, input_default_bytes_, input_state_, error_message)) {
      return false;
    }
    const std::uint64_t previous_scratch_bytes = ffn_scratch_bytes_;
    if (!ensure_buffer(
      ffn_scratch_,
      scratch_bytes,
      ffn_scratch_bytes_,
      D3D12_HEAP_TYPE_DEFAULT,
      D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
      D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
      error_message)) {
      return false;
    }
    if (previous_scratch_bytes != ffn_scratch_bytes_) {
      ffn_scratch_state_ = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    }
    const std::uint64_t activation_bytes = activation_floats * sizeof(float);
    const std::uint64_t previous_activation_bytes = ffn_activation_bytes_;
    if (!ensure_buffer(
      ffn_activation_,
      activation_bytes,
      ffn_activation_bytes_,
      D3D12_HEAP_TYPE_DEFAULT,
      D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
      D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
      error_message)) {
      return false;
    }
    if (previous_activation_bytes != ffn_activation_bytes_) {
      ffn_activation_state_ = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    }
    if (!ensure_output_buffer(output_default_, output_bytes, output_default_bytes_, output_state_, error_message)) {
      return false;
    }
    if (!uma_output_
      && !ensure_persistent_readback_buffer(
        output_readback_, output_bytes, output_readback_bytes_, mapped_output_readback_, error_message)) {
      return false;
    }

    const bool input_dirty = input_buffer_recreated
      || last_input_identity_ != input_identity
      || last_input_bytes_ != input_bytes;
    if (input_dirty
      && !copy_mapped_buffer(mapped_input_upload_, packed_inputs.data(), input_bytes, "input upload", error_message)) {
      return false;
    }

    HRESULT hr = command_allocator_->Reset();
    if (FAILED(hr)) {
      error_message = hresult_message("ID3D12CommandAllocator::Reset", hr);
      return false;
    }
    hr = command_list_->Reset(command_allocator_.get(), nullptr);
    if (FAILED(hr)) {
      error_message = hresult_message("ID3D12GraphicsCommandList::Reset", hr);
      return false;
    }

    transition(output_default_.get(), output_state_, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    transition(ffn_scratch_.get(), ffn_scratch_state_, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    transition(ffn_activation_.get(), ffn_activation_state_, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    if (input_dirty) {
      transition(input_default_.get(), input_state_, D3D12_RESOURCE_STATE_COPY_DEST);
      command_list_->CopyBufferRegion(input_default_.get(), 0, input_upload_.get(), 0, input_bytes);
      transition(input_default_.get(), input_state_, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    }

    use_counter_ += 1;
    ResidentWeight &gate_resident = weight_cache_.at(gate.weight_key);
    ResidentWeight &up_resident = weight_cache_.at(up.weight_key);
    ResidentWeight &down_resident = weight_cache_.at(down.weight_key);
    gate_resident.last_use = use_counter_;
    up_resident.last_use = use_counter_;
    down_resident.last_use = use_counter_;

    command_list_->SetComputeRootSignature(root_signature_.get());

    // Integer GEMM for gate/up only.
    //
    // The gate was reaching qkv and attn-output - 20.6% of prefill - and stopping
    // there, because this chain has its own dispatch and never consulted integer
    // eligibility. FFN is 64.5% of prefill, so that was most of the win sitting
    // unused. gate and up both read input_default_, which is exactly the buffer
    // stage_prefill_activations quantises, so they need no new machinery.
    //
    // down cannot join them: its input is ffn_activation_, the on-GPU SiLU
    // output, and the Q8_1 stager quantises from a host-uploaded buffer. That
    // one needs a GPU-side quantiser.
    bool ffn_prefill_integer = false;
    const auto dispatch_matmul = [&](
      const D3d12ResidentWeightRef &op,
      ID3D12Resource *weight_buffer,
      D3D12_GPU_VIRTUAL_ADDRESS input_address,
      std::uint32_t input_stride,
      ID3D12Resource *result_buffer,
      std::uint32_t output_base,
      std::uint32_t input_transform,
      bool allow_integer = false) {
      DispatchConstants constants;
      constants.row_width = op.row_width;
      constants.row_count = op.row_count;
      constants.batch_count = batch_count;
      constants.output_base = output_base;
      constants.input_stride = input_stride;
      constants.weight_layout = 1;
      constants.input_transform = input_transform;
      const bool batch_one = batch_count == 1;
      if (allow_integer && ffn_prefill_integer) {
        // Same binding as compute_resident's integer path: activations come from
        // the staged Q8_1 buffer, so input_stride is 0.
        DispatchConstants integer_constants = constants;
        integer_constants.input_stride = 0;
        command_list_->SetPipelineState(dp4a_q4k_batched_pipeline_.get());
        command_list_->SetComputeRoot32BitConstants(0, 8, &integer_constants, 0);
        command_list_->SetComputeRootShaderResourceView(
          1, weight_buffer->GetGPUVirtualAddress());
        command_list_->SetComputeRootShaderResourceView(
          2, dp4a_activations_->GetGPUVirtualAddress());
        command_list_->SetComputeRootUnorderedAccessView(
          3, result_buffer->GetGPUVirtualAddress());
        command_list_->Dispatch(
          static_cast<UINT>((op.row_count + 63u) / 64u),
          static_cast<UINT>((batch_count + 7u) / 8u),
          1);
        return;
      }
      ID3D12PipelineState *pipeline = batch_one ? scalar_b1_pipeline_.get() : scalar_pipeline_.get();
      switch (op.format) {
        case D3d12WeightFormat::Q4_K:
          pipeline = batch_one ? q4k_b1_pipeline_.get() : q4k_pipeline_.get();
          break;
        case D3d12WeightFormat::Q5_K:
          pipeline = batch_one ? q5k_b1_pipeline_.get() : q5k_pipeline_.get();
          break;
        case D3d12WeightFormat::Q6_K:
          pipeline = batch_one ? q6k_b1_pipeline_.get() : q6k_pipeline_.get();
          break;
        case D3d12WeightFormat::Q8_0:
          pipeline = batch_one ? q8_0_b1_pipeline_.get() : q8_0_pipeline_.get();
          break;
        case D3d12WeightFormat::Q4_0:
          pipeline = batch_one ? q4_0_b1_pipeline_.get() : q4_0_pipeline_.get();
          break;
        case D3d12WeightFormat::F32:
        case D3d12WeightFormat::F16:
        case D3d12WeightFormat::BF16:
          constants.scalar_format = 0;
          break;
      }
      // Prefill's FFN chain is where the time is: measured 1603 ms of a 2197 ms
      // prefill, 73%, at 8.9% of this iGPU's arithmetic peak. The row-blocked
      // kernel computes several output rows per thread so each groupshared input
      // read feeds that many FMAs instead of one.
      ID3D12PipelineState *const rowblock = rowblock_pipeline_for(op, batch_one);
      const std::uint32_t row_block = rowblock != nullptr ? rowblock_rows_for(op.format) : 1u;
      command_list_->SetPipelineState(rowblock != nullptr ? rowblock : pipeline);
      command_list_->SetComputeRoot32BitConstants(0, 8, &constants, 0);
      command_list_->SetComputeRootShaderResourceView(1, weight_buffer->GetGPUVirtualAddress());
      command_list_->SetComputeRootShaderResourceView(2, input_address);
      command_list_->SetComputeRootUnorderedAccessView(3, result_buffer->GetGPUVirtualAddress());
      const std::uint32_t dispatch_tile = rowblock != nullptr ? rowblock_tile_for(op.format) : 8u;
      command_list_->Dispatch(
        static_cast<UINT>((op.row_count + (64u * row_block) - 1u) / (64u * row_block)),
        static_cast<UINT>(batch_one
          ? batch_count
          : (batch_count + dispatch_tile - 1u) / dispatch_tile),
        1);
    };

    // Decided here rather than at the lambda so it happens after the input upload
    // and before the first dispatch that reads the staged buffer. A staging
    // failure is not fatal - the float kernels below are still correct.
    std::uint32_t ffn_activation_stride = 0;
    ffn_prefill_integer = gpu_dp4a_prefill_enabled()
      && batch_count > 1
      && prefill_integer_eligible(gate, batch_count, gate.row_width)
      && prefill_integer_eligible(up, batch_count, gate.row_width);
    if (ffn_prefill_integer
      && !stage_prefill_activations(
        gate.row_width, batch_count, ffn_activation_stride, error_message)) {
      ffn_prefill_integer = false;
    }

    dispatch_matmul(
      gate,
      gate_resident.buffer.get(),
      input_default_->GetGPUVirtualAddress(),
      gate.row_width,
      ffn_scratch_.get(),
      0,
      0,
      true);
    dispatch_matmul(
      up,
      up_resident.buffer.get(),
      input_default_->GetGPUVirtualAddress(),
      gate.row_width,
      ffn_scratch_.get(),
      static_cast<std::uint32_t>(activation_floats),
      0,
      true);

    D3D12_RESOURCE_BARRIER scratch_uav_barrier = {};
    scratch_uav_barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    scratch_uav_barrier.UAV.pResource = ffn_scratch_.get();
    command_list_->ResourceBarrier(1, &scratch_uav_barrier);
    transition(ffn_scratch_.get(), ffn_scratch_state_, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

    // Dedicated gelu pass: computes gelu(gate)*up exactly once per element
    // (doing it in the down matmul's tile loader recomputes tanh once per
    // thread group and measurably slows decode).
    DispatchConstants gelu_constants;
    gelu_constants.row_count = static_cast<std::uint32_t>(activation_floats);
    gelu_constants.output_base = 0;
    command_list_->SetPipelineState(use_silu ? silu_glu_pipeline_.get() : gelu_pipeline_.get());
    command_list_->SetComputeRoot32BitConstants(0, 8, &gelu_constants, 0);
    command_list_->SetComputeRootShaderResourceView(1, ffn_scratch_->GetGPUVirtualAddress());
    command_list_->SetComputeRootShaderResourceView(2, ffn_scratch_->GetGPUVirtualAddress());
    command_list_->SetComputeRootUnorderedAccessView(3, ffn_activation_->GetGPUVirtualAddress());
    command_list_->Dispatch(
      static_cast<UINT>((activation_floats + 255u) / 256u),
      1,
      1);

    D3D12_RESOURCE_BARRIER activation_uav_barrier = {};
    activation_uav_barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    activation_uav_barrier.UAV.pResource = ffn_activation_.get();
    command_list_->ResourceBarrier(1, &activation_uav_barrier);
    transition(ffn_activation_.get(), ffn_activation_state_, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

    // down stays on the float kernel. Two independent reasons, both measured:
    //
    // Cost: routing it through the integer path needs a re-stage over
    // ffn_activation_ (its input is the on-GPU SiLU output, not the host upload).
    // That measured slower every time - 13.6-18.2 s with the per-column quantiser,
    // 12.0-12.6 s after batching it, and 13.8-16.2 s even with the Q6_K integer
    // kernel making the matmul itself 4.5x faster. The staging work at width 8192
    // is simply larger than the matmul it enables.
    //
    // Correctness: stage_prefill_activations calls ensure_buffer, and re-staging
    // at a LARGER width reallocates dp4a_activations_ while the gate/up dispatches
    // recorded above still reference the old GPU address. The chain selftest
    // caught it immediately (512x256 b3: expected -1.55e9, got 1.30e13). Any
    // future attempt has to stage both widths up front, or size the buffer for the
    // larger one before either dispatch is recorded.
    dispatch_matmul(
      down,
      down_resident.buffer.get(),
      ffn_activation_->GetGPUVirtualAddress(),
      static_cast<std::uint32_t>(gate.row_count),
      output_default_.get(),
      0,
      0);

    if (!uma_output_) {
      transition(output_default_.get(), output_state_, D3D12_RESOURCE_STATE_COPY_SOURCE);
      command_list_->CopyBufferRegion(output_readback_.get(), 0, output_default_.get(), 0, output_bytes);
    }

    hr = command_list_->Close();
    if (FAILED(hr)) {
      error_message = hresult_message("ID3D12GraphicsCommandList::Close", hr);
      return false;
    }
    if (!submit_and_wait(error_message)) {
      return false;
    }
    telemetry_.input_upload_bytes += input_dirty ? input_bytes : 0;
    telemetry_.output_readback_bytes += output_bytes;

    const void *result_memory = uma_output_ ? mapped_output_direct_ : mapped_output_readback_;
    if (result_memory == nullptr) {
      error_message = "D3D12 output buffer is not mapped.";
      return false;
    }
    outputs->resize(static_cast<std::size_t>(output_floats));
    std::memcpy(
      outputs->data(),
      result_memory,
      static_cast<std::size_t>(output_bytes));

    if (ffn_debug_enabled()) {
      debug_dump_ffn_scratch(activation_floats, batch_count, gate.row_count, outputs->data(), output_floats);
    }

    if (input_dirty) {
      last_input_identity_ = input_identity;
      last_input_bytes_ = input_bytes;
    }
    return true;
  }

  static bool ffn_debug_enabled() {
    static const bool enabled = [] {
      char value[8] = {};
      return GetEnvironmentVariableA("LIZARD_NATIVE_GPU_FFN_DEBUG", value, sizeof(value)) != 0
        && value[0] == '1';
    }();
    return enabled;
  }

  // Reads the gate/up scratch back to the CPU after a chain submission and
  // logs non-finite statistics per region to the trace file, isolating
  // whether corruption enters at the gate/up matmuls or the down pass.
  void debug_dump_ffn_scratch(
    std::uint64_t activation_floats,
    std::uint32_t batch_count,
    std::uint32_t gate_rows,
    const float *down_output,
    std::uint64_t output_floats) {
    const std::uint64_t scratch_floats = activation_floats * 2;
    const std::uint64_t scratch_bytes = scratch_floats * sizeof(float);
    ComPtr<ID3D12Resource> debug_readback;
    std::uint64_t allocated = 0;
    std::string error_message;
    if (!ensure_buffer(
      debug_readback,
      scratch_bytes,
      allocated,
      D3D12_HEAP_TYPE_READBACK,
      D3D12_RESOURCE_STATE_COPY_DEST,
      D3D12_RESOURCE_FLAG_NONE,
      error_message)) {
      trace_d3d12("ffn debug: readback alloc failed: " + error_message);
      return;
    }
    if (FAILED(command_allocator_->Reset())
      || FAILED(command_list_->Reset(command_allocator_.get(), nullptr))) {
      return;
    }
    transition(ffn_scratch_.get(), ffn_scratch_state_, D3D12_RESOURCE_STATE_COPY_SOURCE);
    command_list_->CopyBufferRegion(debug_readback.get(), 0, ffn_scratch_.get(), 0, scratch_bytes);
    if (FAILED(command_list_->Close()) || !submit_and_wait(error_message)) {
      trace_d3d12("ffn debug: scratch copy failed: " + error_message);
      return;
    }
    void *mapped = nullptr;
    D3D12_RANGE range = {0, static_cast<SIZE_T>(scratch_bytes)};
    if (FAILED(debug_readback->Map(0, &range, &mapped))) {
      return;
    }
    const float *scratch = static_cast<const float *>(mapped);
    std::ostringstream report;
    report << "ffn debug: batches=" << batch_count << " gate_rows=" << gate_rows;
    for (std::uint32_t batch = 0; batch < batch_count; batch += 1) {
      std::uint64_t gate_bad = 0;
      std::uint64_t up_bad = 0;
      std::uint64_t down_bad = 0;
      for (std::uint32_t row = 0; row < gate_rows; row += 1) {
        const std::uint64_t index = static_cast<std::uint64_t>(batch) * gate_rows + row;
        if (!std::isfinite(scratch[index])) {
          gate_bad += 1;
        }
        if (!std::isfinite(scratch[activation_floats + index])) {
          up_bad += 1;
        }
      }
      const std::uint64_t down_rows = output_floats / batch_count;
      for (std::uint64_t row = 0; row < down_rows; row += 1) {
        if (!std::isfinite(down_output[batch * down_rows + row])) {
          down_bad += 1;
        }
      }
      if (gate_bad > 0 || up_bad > 0 || down_bad > 0) {
        report << " | b" << batch
          << " gateBad=" << gate_bad
          << " upBad=" << up_bad
          << " downBad=" << down_bad
          << " gate0=" << scratch[static_cast<std::uint64_t>(batch) * gate_rows]
          << " up0=" << scratch[activation_floats + static_cast<std::uint64_t>(batch) * gate_rows];
      }
    }
    trace_d3d12(report.str());
    D3D12_RANGE written = {0, 0};
    debug_readback->Unmap(0, &written);
  }

 private:
  struct ResidentWeight {
    ComPtr<ID3D12Resource> buffer;
    std::uint64_t bytes = 0;
    std::uint64_t last_use = 0;
    D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_COPY_DEST;
  };

  struct DeviceKvCache {
    ComPtr<ID3D12Resource> buffer;
    std::uint32_t max_positions = 0;
    std::uint32_t kv_size = 0;
    std::uint32_t words_per_position = 0;
    std::uint32_t positions = 0;
    D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_COPY_DEST;
  };

  static std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
  }

  void transition(ID3D12Resource *resource, D3D12_RESOURCE_STATES &state, D3D12_RESOURCE_STATES target) {
    if (state == target) {
      return;
    }
    D3D12_RESOURCE_BARRIER barrier = {};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = resource;
    barrier.Transition.StateBefore = state;
    barrier.Transition.StateAfter = target;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    command_list_->ResourceBarrier(1, &barrier);
    state = target;
  }

  bool submit_and_wait(std::string &error_message) {
    // Any submission may have rewritten the shared upload/config buffers; the
    // token graph re-validates its static upload cache after its own submit.
    token_graph_upload_model_ = nullptr;
    ID3D12CommandList *command_lists[] = {command_list_.get()};
    command_queue_->ExecuteCommandLists(1, command_lists);
    telemetry_.submission_count += 1;

    fence_value_ += 1;
    HRESULT hr = command_queue_->Signal(fence_.get(), fence_value_);
    if (FAILED(hr)) {
      error_message = hresult_message("ID3D12CommandQueue::Signal", hr);
      return false;
    }
    if (fence_->GetCompletedValue() < fence_value_) {
      telemetry_.fence_wait_count += 1;
      // Bounded spin before the kernel event. WaitForSingleObject deschedules
      // this thread and pays a wake-up on every completion; the phase profile
      // puts the fixed cost of a submission at 0.84 ms against GPU payloads of
      // 0.4-2.4 ms, and decode issues 56 submissions per token, so that wake-up
      // is charged 56 times for every token the user sees.
      //
      // Spinning is affordable specifically here: this thread has nothing else
      // to do until the fence signals - the CPU is idle waiting on the GPU for
      // most of decode - and the wait is bounded so a hung GPU still falls
      // through to the blocking path rather than burning a core forever.
      if (fence_spin_micros() > 0) {
        const auto spin_deadline = std::chrono::steady_clock::now()
          + std::chrono::microseconds(fence_spin_micros());
        while (fence_->GetCompletedValue() < fence_value_
          && std::chrono::steady_clock::now() < spin_deadline) {
          YieldProcessor();
        }
      }
      if (fence_->GetCompletedValue() < fence_value_) {
        hr = fence_->SetEventOnCompletion(fence_value_, fence_event_);
        if (FAILED(hr)) {
          error_message = hresult_message("ID3D12Fence::SetEventOnCompletion", hr);
          return false;
        }
        WaitForSingleObject(fence_event_, INFINITE);
      }
    }

    // A removed device completes fences immediately and readback contains
    // garbage. Detect that here so a faulted dispatch can never be consumed
    // as a successful result. The context is marked dead and recreated on a
    // later call, because the Intel Xe driver removes the device sporadically
    // and a long-running server must be able to regain the GPU lane.
    const HRESULT removed_reason = device_->GetDeviceRemovedReason();
    if (FAILED(removed_reason)) {
      error_message = hresult_message("D3D12 device removed during compute", removed_reason);
      capabilities_.supported = false;
      capabilities_.reason = error_message;
      device_removed_ = true;
      reset_resident_weights();
      return false;
    }
    return true;
  }

 public:
  // Clamp the resident-weight budget to the selected adapter's CURRENT
  // DXGI budget (Finding 2). Usable-for-weights adds our own resident bytes
  // back to (Budget - CurrentUsage) - our staged weights are part of
  // CurrentUsage, and without the correction the budget would shrink as we
  // stage, double-counting ourselves. Reserves keep headroom for the
  // OS/display, transient workspace/upload/readback, GPU KV state, and an
  // emergency margin. Throttled: refreshing on every staging call would put
  // a syscall in the hot path.
  void refresh_weight_budget(bool force = false) {
    if (!adapter3_) {
      weight_budget_bytes_ = requested_weight_budget_bytes_;
      live_budget_available_ = false;
      return;
    }
    const auto now = std::chrono::steady_clock::now();
    if (!force && live_budget_available_
      && now - last_budget_refresh_at_ < std::chrono::milliseconds(250)) {
      return;
    }
    last_budget_refresh_at_ = now;

    DXGI_QUERY_VIDEO_MEMORY_INFO local_info = {};
    if (FAILED(adapter3_->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &local_info))) {
      weight_budget_bytes_ = requested_weight_budget_bytes_;
      live_budget_available_ = false;
      return;
    }
    DXGI_QUERY_VIDEO_MEMORY_INFO non_local_info = {};
    if (FAILED(adapter3_->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, &non_local_info))) {
      non_local_info = {};
    }
    live_local_budget_bytes_ = local_info.Budget;
    live_local_usage_bytes_ = local_info.CurrentUsage;
    live_non_local_budget_bytes_ = non_local_info.Budget;
    live_non_local_usage_bytes_ = non_local_info.CurrentUsage;

    const std::uint64_t used_by_others = local_info.CurrentUsage > weight_cache_bytes_
      ? local_info.CurrentUsage - weight_cache_bytes_
      : 0;
    const std::uint64_t available_for_weights = local_info.Budget > used_by_others
      ? local_info.Budget - used_by_others
      : 0;
    constexpr std::uint64_t k_mb = 1024ull * 1024ull;
    const std::uint64_t display_reserve = std::max<std::uint64_t>(384 * k_mb, local_info.Budget / 10);
    const std::uint64_t workspace_reserve = 256 * k_mb;  // upload/readback/command/transients
    const std::uint64_t kv_reserve = 128 * k_mb;
    const std::uint64_t emergency_reserve = 64 * k_mb;
    std::uint64_t reserves = display_reserve + workspace_reserve + kv_reserve + emergency_reserve;
    // These reserves are sized for a card whose live budget is multiple GB. A
    // discrete GPU's LIVE DXGI budget can be far smaller than its physical VRAM
    // (hybrid laptops where the panel renders on the iGPU, or VRAM pressure), and
    // the fixed ~832 MB total then swallows nearly the whole budget: measured
    // 832 MB reserved out of an 874 MB live budget, leaving 28 MB - far too
    // little for a single 207 MB LM-head tensor, which hard-failed the required
    // D3D12 lane. Cap the reserve at 40% of the live budget so weights always
    // keep the majority. Safe to do: the workspace/KV/staging this guards is
    // already counted in CurrentUsage (and therefore already subtracted via
    // used_by_others), so the fixed figures were double-spending it.
    const std::uint64_t reserve_cap = (local_info.Budget / 5) * 2;
    if (reserve_cap > 0 && reserves > reserve_cap) {
      reserves = reserve_cap;
    }
    memory_reserve_bytes_ = reserves;
    const std::uint64_t safe_available = available_for_weights > memory_reserve_bytes_
      ? available_for_weights - memory_reserve_bytes_
      : 0;
    const std::uint64_t previous = weight_budget_bytes_;
    weight_budget_bytes_ = std::min<std::uint64_t>(requested_weight_budget_bytes_, safe_available);
    live_budget_available_ = true;
    if (weight_budget_bytes_ != previous) {
      trace_d3d12("refresh_weight_budget: effective "
        + std::to_string(weight_budget_bytes_ / k_mb) + " MiB (requested "
        + std::to_string(requested_weight_budget_bytes_ / k_mb) + ", safe available "
        + std::to_string(safe_available / k_mb) + ", reserves "
        + std::to_string(memory_reserve_bytes_ / k_mb) + ")");
    }
  }

  D3d12GpuMemoryReport memory_report() const {
    D3d12GpuMemoryReport report;
    report.live = live_budget_available_;
    report.local_budget_bytes = live_local_budget_bytes_;
    report.local_usage_bytes = live_local_usage_bytes_;
    report.non_local_budget_bytes = live_non_local_budget_bytes_;
    report.non_local_usage_bytes = live_non_local_usage_bytes_;
    report.reserve_bytes = memory_reserve_bytes_;
    report.requested_weight_budget_bytes = requested_weight_budget_bytes_;
    report.effective_weight_budget_bytes = weight_budget_bytes_;
    report.resident_weight_bytes = weight_cache_bytes_;
    report.confidence = live_budget_available_ ? "live-dxgi" : "static-fallback";
    return report;
  }

 private:
  bool evict_for(
    std::uint64_t incoming_bytes,
    const D3d12ResidentWeightRef *ops,
    std::size_t op_count,
    std::string &error_message) {
    // Current pressure, not just the configured cache size, decides how
    // much may stay resident (doc Finding 2 implementation work item 7).
    refresh_weight_budget();
    if (incoming_bytes > weight_budget_bytes_) {
      const std::uint64_t k_mb = 1024ull * 1024ull;
      error_message = "D3D12 resident weight upload (" + std::to_string(incoming_bytes / k_mb)
        + " MiB) exceeds the effective GPU weight budget (" + std::to_string(weight_budget_bytes_ / k_mb)
        + " MiB; live budget " + std::to_string(live_local_budget_bytes_ / k_mb)
        + " MiB, usage " + std::to_string(live_local_usage_bytes_ / k_mb)
        + " MiB, reserves " + std::to_string(memory_reserve_bytes_ / k_mb) + " MiB).";
      return false;
    }
    while (weight_cache_bytes_ + incoming_bytes > weight_budget_bytes_ && !weight_cache_.empty()) {
      auto victim = weight_cache_.end();
      for (auto iterator = weight_cache_.begin(); iterator != weight_cache_.end(); ++iterator) {
        bool needed_now = false;
        for (std::size_t op_index = 0; op_index < op_count; op_index += 1) {
          if (ops[op_index].weight_key == iterator->first) {
            needed_now = true;
            break;
          }
        }
        if (needed_now) {
          continue;
        }
        if (victim == weight_cache_.end() || iterator->second.last_use < victim->second.last_use) {
          victim = iterator;
        }
      }
      if (victim == weight_cache_.end()) {
        error_message = "D3D12 resident weight cache cannot free enough budget for this dispatch.";
        return false;
      }
      weight_cache_bytes_ -= victim->second.bytes;
      weight_cache_.erase(victim);
      telemetry_.weight_cache_evictions += 1;
    }
    return true;
  }

  // Requested adapter identity from the execution plan (Phase 1 Finding 1):
  // "luidHigh:luidLow". Returns false when the variable is absent/malformed.
  static bool requested_adapter_luid(std::int64_t &luid_high, std::uint64_t &luid_low) {
    const char *value = std::getenv("LIZARD_NATIVE_GPU_ADAPTER_LUID");
    if (value == nullptr || value[0] == '\0') {
      return false;
    }
    char *separator = nullptr;
    const long long high = std::strtoll(value, &separator, 10);
    if (separator == nullptr || *separator != ':') {
      return false;
    }
    char *end = nullptr;
    const unsigned long long low = std::strtoull(separator + 1, &end, 10);
    if (end == nullptr || *end != '\0') {
      return false;
    }
    luid_high = static_cast<std::int64_t>(high);
    luid_low = static_cast<std::uint64_t>(low);
    return true;
  }

  void record_adapter_identity(const DXGI_ADAPTER_DESC1 &desc, const char *reason) {
    capabilities_.adapter_name = narrow_utf16(desc.Description);
    capabilities_.adapter_identity_valid = true;
    capabilities_.adapter_luid_high = static_cast<std::int64_t>(desc.AdapterLuid.HighPart);
    capabilities_.adapter_luid_low = static_cast<std::uint64_t>(desc.AdapterLuid.LowPart);
    capabilities_.adapter_vendor_id = desc.VendorId;
    capabilities_.adapter_device_id = desc.DeviceId;
    capabilities_.adapter_selection_reason = reason;
  }

  bool select_adapter(std::string &error_message) {
    std::int64_t requested_high = 0;
    std::uint64_t requested_low = 0;
    const bool luid_requested = requested_adapter_luid(requested_high, requested_low);

    IDXGIAdapter1 *best_adapter = nullptr;
    DXGI_ADAPTER_DESC1 best_desc = {};
    std::uint64_t best_dedicated = 0;
    bool any_hardware_adapter = false;

    for (UINT index = 0;; index += 1) {
      IDXGIAdapter1 *adapter_raw = nullptr;
      HRESULT hr = factory_->EnumAdapters1(index, &adapter_raw);
      if (hr == DXGI_ERROR_NOT_FOUND) {
        break;
      }
      if (FAILED(hr)) {
        error_message = hresult_message("EnumAdapters1", hr);
        return false;
      }
      ComPtr<IDXGIAdapter1> adapter;
      adapter.reset(adapter_raw);

      DXGI_ADAPTER_DESC1 desc = {};
      adapter->GetDesc1(&desc);

      if (luid_requested) {
        // Explicit plan identity: only the exact LUID is acceptable. A plan
        // calculated for one adapter must never silently execute on another.
        if (static_cast<std::int64_t>(desc.AdapterLuid.HighPart) != requested_high
          || static_cast<std::uint64_t>(desc.AdapterLuid.LowPart) != requested_low) {
          continue;
        }
        ID3D12Device *probe_device = nullptr;
        hr = D3D12CreateDevice(adapter.get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&probe_device));
        if (FAILED(hr)) {
          error_message = "The requested Direct3D 12 adapter (LUID "
            + std::to_string(requested_high) + ":" + std::to_string(requested_low)
            + ", " + narrow_utf16(desc.Description)
            + ") could not create a device; refusing to substitute another adapter. "
            + hresult_message("D3D12CreateDevice", hr);
          return false;
        }
        probe_device->Release();
        adapter_.reset(adapter.detach());
        record_adapter_identity(desc, "requested LUID matched");
        return true;
      }

      if ((desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) != 0) {
        continue;
      }
      any_hardware_adapter = true;

      ID3D12Device *probe_device = nullptr;
      hr = D3D12CreateDevice(adapter.get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&probe_device));
      if (FAILED(hr)) {
        continue;
      }
      probe_device->Release();

      // Deterministic fallback policy when no identity was requested:
      // largest dedicated VRAM wins (matches the JS selectPreferredAdapter
      // default), first-enumerated wins ties. Enumeration order alone can
      // no longer decide between an iGPU and a dGPU.
      if (best_adapter == nullptr || desc.DedicatedVideoMemory > best_dedicated) {
        best_desc = desc;
        best_dedicated = desc.DedicatedVideoMemory;
        ComPtr<IDXGIAdapter1> previous;
        previous.reset(best_adapter);
        best_adapter = adapter.detach();
      }
    }

    if (luid_requested) {
      error_message = "The requested Direct3D 12 adapter (LUID "
        + std::to_string(requested_high) + ":" + std::to_string(requested_low)
        + ") was not found; refusing to substitute another adapter. "
        + "Re-plan against the current adapter inventory or fall back to CPU.";
      return false;
    }
    if (best_adapter == nullptr) {
      error_message = any_hardware_adapter
        ? "No hardware Direct3D 12 adapter could create a feature-level 11.0 device."
        : "No hardware Direct3D 12 adapter was available for the native hybrid backend.";
      return false;
    }

    adapter_.reset(best_adapter);
    record_adapter_identity(best_desc, "largest dedicated VRAM (no identity requested)");
    return true;
  }

  // The row-blocked Q4_K kernel, or nullptr when it does not apply. Batched
  // Q4_K only: batch-one has no batch dimension to amortise LDS reads across,
  // and the other formats have their own unpack shapes.
  // Rows per thread for whichever row-blocked kernel `op` will take. Q6_K runs
  // its own value; a mismatch here would make the dispatch grid disagree with
  // the shader and silently drop or double-cover rows.
  std::uint32_t rowblock_rows_for(D3d12WeightFormat format) const {
    return format == D3d12WeightFormat::Q6_K ? gpu_q6k_row_block() : gpu_row_block();
  }

  // Must match whatever tile the pipeline was compiled with, or the dispatch
  // grid covers the wrong number of batch columns.
  std::uint32_t rowblock_tile_for(D3d12WeightFormat format) const {
    if (format == D3d12WeightFormat::Q6_K && gpu_q6k_prefill_tile() > 0u) {
      return gpu_q6k_prefill_tile();
    }
    return gpu_prefill_tile();
  }

  ID3D12PipelineState *rowblock_pipeline_for(
    const D3d12ResidentWeightRef &op,
    bool batch_one) const {
    if (batch_one) {
      return nullptr;
    }
    ID3D12PipelineState *selected = nullptr;
    if (op.format == D3d12WeightFormat::Q4_K && q4k_rowblock_pipeline_) {
      selected = q4k_rowblock_pipeline_.get();
    } else if (op.format == D3d12WeightFormat::Q6_K && q6k_rowblock_pipeline_) {
      selected = q6k_rowblock_pipeline_.get();
    }
    if (selected == nullptr) {
      return nullptr;
    }
    const std::uint32_t rows = rowblock_rows_for(op.format);
    if (rows <= 1) {
      return nullptr;
    }
    // Each group now covers k_rows_per_group * rows outputs. Below that there is
    // only one partially-filled group and the blocking buys nothing.
    if (op.row_count < 64u * rows) {
      return nullptr;
    }
    return selected;
  }

  bool compile_pipeline(
    const char *entry_point,
    ComPtr<ID3D12PipelineState> &pipeline,
    std::string &error_message,
    bool batch_one = false,
    const char *rows_per_group = nullptr,
    const char *rows_per_thread = nullptr,
    std::uint32_t tile_override = 0u) {
    // main_q4k reads the resident buffer, so it must agree with whatever
    // stage_weights_column_major wrote. The flag is global and the pipelines
    // are built once at init, so consulting it here keeps the shader and the
    // staging in lockstep without threading a parameter through every caller.
    const bool quad = quad_interleave_enabled();
    const D3D_SHADER_MACRO batch_one_macros_quad[] = {
      {"LIZARD_BATCH_TILE", "1"},
      {"LIZARD_QUAD_INTERLEAVE", "1"},
      {nullptr, nullptr},
    };
    const D3D_SHADER_MACRO batch_one_macros_plain[] = {
      {"LIZARD_BATCH_TILE", "1"},
      {nullptr, nullptr},
    };
    const D3D_SHADER_MACRO rows_per_group_macros_quad[] = {
      {"LIZARD_BATCH_TILE", "1"},
      {"LIZARD_ROWS_PER_GROUP", rows_per_group ? rows_per_group : "64"},
      {"LIZARD_QUAD_INTERLEAVE", "1"},
      {nullptr, nullptr},
    };
    const D3D_SHADER_MACRO rows_per_group_macros_plain[] = {
      {"LIZARD_BATCH_TILE", "1"},
      {"LIZARD_ROWS_PER_GROUP", rows_per_group ? rows_per_group : "64"},
      {nullptr, nullptr},
    };
    const D3D_SHADER_MACRO *const batch_one_macros =
      quad ? batch_one_macros_quad : batch_one_macros_plain;
    const D3D_SHADER_MACRO *const rows_per_group_macros =
      quad ? rows_per_group_macros_quad : rows_per_group_macros_plain;
    // The default (batched, no rows-per-group) variant passes no macros at all,
    // so it silently missed the layout define and read quad-staged weights with
    // block-order offsets - the b5 selftest failures that survived the cache fix.
    const D3D_SHADER_MACRO default_macros_quad[] = {
      {"LIZARD_QUAD_INTERLEAVE", "1"},
      {nullptr, nullptr},
    };
    // Row-blocked variant: batched tile, block-order addressing only (the host
    // gate declines it under quad interleave), plus the rows-per-thread factor.
    const std::string prefill_tile_value =
      std::to_string(tile_override != 0u ? tile_override : gpu_prefill_tile());
    const D3D_SHADER_MACRO rows_per_thread_macros[] = {
      {"LIZARD_ROWS_PER_THREAD", rows_per_thread ? rows_per_thread : "1"},
      // Prefill gets its own batch tile; decode keeps k_gpu_batch_tile.
      {"LIZARD_BATCH_TILE", prefill_tile_value.c_str()},
      {nullptr, nullptr},
    };

    // Cache key: shader source identity + entry point + macro variant. The
    // source hash is computed once; a shader edit changes it and invalidates
    // every cached blob.
    static const std::uint64_t source_hash = fnv1a_hash(k_shader_source, sizeof(k_shader_source) - 1);
    // The weight layout is part of the variant. Without it a blob compiled for
    // block-order addressing is served to a run staging quad-order weights, and
    // the kernel reads the right bytes from the wrong offsets - which is
    // exactly how the first quad-interleave run failed the selftest while the
    // source was already correct.
    const std::string variant = std::string(entry_point) + "_"
      + (rows_per_thread
        ? (std::string("rpt") + rows_per_thread + "_bt" + prefill_tile_value)
        : rows_per_group ? (std::string("rpg") + rows_per_group)
        : (batch_one ? std::string("b1") : std::string("def")))
      + (quad ? "_quad3" : "");  // bump when the macro set for a variant changes
    const bool cache_enabled = !shader_cache_disabled();
    const std::filesystem::path cache_file = cache_enabled
      ? shader_cache_path(source_hash, variant)
      : std::filesystem::path();

    const auto create_from_bytecode = [&](const void *bytecode, std::size_t length) -> bool {
      D3D12_COMPUTE_PIPELINE_STATE_DESC pipeline_desc = {};
      pipeline_desc.pRootSignature = root_signature_.get();
      pipeline_desc.CS.pShaderBytecode = bytecode;
      pipeline_desc.CS.BytecodeLength = length;
      pipeline_desc.Flags = D3D12_PIPELINE_STATE_FLAG_NONE;
      ID3D12PipelineState *pipeline_raw = nullptr;
      const HRESULT create_hr =
        device_->CreateComputePipelineState(&pipeline_desc, IID_PPV_ARGS(&pipeline_raw));
      if (FAILED(create_hr)) {
        return false;
      }
      pipeline.reset(pipeline_raw);
      return true;
    };

    // Fast path: a previously compiled DXBC blob skips the FXC front-end.
    if (cache_enabled && !cache_file.empty()) {
      const std::vector<unsigned char> cached = read_cached_shader_blob(cache_file);
      if (!cached.empty() && create_from_bytecode(cached.data(), cached.size())) {
        trace_d3d12("compile_pipeline: reused cached shader blob for " + variant);
        return true;
      }
    }

    ID3DBlob *shader_blob_raw = nullptr;
    ID3DBlob *error_blob_raw = nullptr;
    HRESULT hr = D3DCompile(
      k_shader_source,
      sizeof(k_shader_source) - 1,
      "lizard_native_matmul",
      rows_per_thread ? rows_per_thread_macros
        : rows_per_group ? rows_per_group_macros
        : (batch_one ? batch_one_macros : (quad ? default_macros_quad : nullptr)),
      nullptr,
      entry_point,
      "cs_5_0",
      D3DCOMPILE_OPTIMIZATION_LEVEL3,
      0,
      &shader_blob_raw,
      &error_blob_raw);
    ComPtr<ID3DBlob> shader_blob;
    ComPtr<ID3DBlob> error_blob;
    shader_blob.reset(shader_blob_raw);
    error_blob.reset(error_blob_raw);
    if (FAILED(hr)) {
      error_message = error_blob
        ? std::string(static_cast<const char *>(error_blob->GetBufferPointer()), error_blob->GetBufferSize())
        : hresult_message("D3DCompile", hr);
      return false;
    }

    if (!create_from_bytecode(shader_blob->GetBufferPointer(), shader_blob->GetBufferSize())) {
      error_message = hresult_message("CreateComputePipelineState", hr);
      return false;
    }
    // Persist the freshly compiled blob so the next worker process reuses it.
    if (cache_enabled && !cache_file.empty()) {
      write_cached_shader_blob(cache_file, shader_blob->GetBufferPointer(), shader_blob->GetBufferSize());
    }
    return true;
  }

  // Probes Shader Model 6.4 support and creates the embedded DXIL DP4A
  // pipelines. Strictly best-effort: every failure path only records a
  // reason and leaves the float FXC kernels as the execution path.
  void initialize_dp4a_pipelines() {
#ifndef LIZARD_HAVE_DXIL
    dp4a_capabilities_.supported = false;
    dp4a_capabilities_.reason =
      "Build has no embedded DXIL kernels (dxc.exe was not found at build time).";
#else
    // Raw values keep this robust across MinGW header vintages:
    // D3D12_FEATURE_SHADER_MODEL == 7, D3D_SHADER_MODEL_6_4 == 0x64.
    struct FeatureDataShaderModel {
      UINT highest_shader_model;
    } shader_model = {0x64};
    const HRESULT probe_hr = device_->CheckFeatureSupport(
      static_cast<D3D12_FEATURE>(7), &shader_model, sizeof(shader_model));
    if (FAILED(probe_hr) || shader_model.highest_shader_model < 0x64) {
      dp4a_capabilities_.supported = false;
      dp4a_capabilities_.reason = FAILED(probe_hr)
        ? "The driver rejected the Shader Model feature probe."
        : "The device driver reports a shader model below 6.4.";
      return;
    }
    const auto create_dxil_pipeline = [&](
      const unsigned char *bytecode,
      std::size_t bytecode_length,
      ComPtr<ID3D12PipelineState> &pipeline,
      std::string &error_message) {
      D3D12_COMPUTE_PIPELINE_STATE_DESC pipeline_desc = {};
      pipeline_desc.pRootSignature = root_signature_.get();
      pipeline_desc.CS.pShaderBytecode = bytecode;
      pipeline_desc.CS.BytecodeLength = bytecode_length;
      pipeline_desc.Flags = D3D12_PIPELINE_STATE_FLAG_NONE;
      ID3D12PipelineState *pipeline_raw = nullptr;
      const HRESULT hr =
        device_->CreateComputePipelineState(&pipeline_desc, IID_PPV_ARGS(&pipeline_raw));
      if (FAILED(hr)) {
        error_message = hresult_message("CreateComputePipelineState(DXIL)", hr);
        return false;
      }
      pipeline.reset(pipeline_raw);
      return true;
    };
    std::string dxil_error;
    if (!create_dxil_pipeline(
        g_dxil_quantize_q8_1, sizeof(g_dxil_quantize_q8_1), dp4a_quantize_pipeline_, dxil_error)
      || !create_dxil_pipeline(
        g_dxil_q4k_q8_1_b1, sizeof(g_dxil_q4k_q8_1_b1), dp4a_q4k_b1_pipeline_, dxil_error)
      || !create_dxil_pipeline(
        g_dxil_q4k_gemv_wave, sizeof(g_dxil_q4k_gemv_wave), q4k_gemv_wave_pipeline_, dxil_error)
      || !create_dxil_pipeline(
        g_dxil_q4k_q8_1_batched, sizeof(g_dxil_q4k_q8_1_batched),
        dp4a_q4k_batched_pipeline_, dxil_error)
      || !create_dxil_pipeline(
        g_dxil_q4k_q8_1_batched_nolds, sizeof(g_dxil_q4k_q8_1_batched_nolds),
        dp4a_q4k_batched_nolds_pipeline_, dxil_error)
      || !create_dxil_pipeline(
        g_dxil_q6k_q8_1_batched, sizeof(g_dxil_q6k_q8_1_batched),
        dp4a_q6k_batched_pipeline_, dxil_error)) {
      dp4a_quantize_pipeline_.reset();
      dp4a_q4k_b1_pipeline_.reset();
      q4k_gemv_wave_pipeline_.reset();
      dp4a_q4k_batched_pipeline_.reset();
      dp4a_q4k_batched_nolds_pipeline_.reset();
      dp4a_q6k_batched_pipeline_.reset();
      dp4a_capabilities_.supported = false;
      dp4a_capabilities_.reason = "DXIL pipeline creation failed: " + dxil_error;
      return;
    }
    dp4a_capabilities_.supported = true;
    dp4a_capabilities_.reason =
      "Shader Model 6.4 packed integer dot kernels are available.";
#endif
  }

 public:
  // Test-only: dispatch the cooperative subgroup-per-row Q4_K GEMV standalone
  // (float input straight from the upload heap into the matvec) so the kernel
  // can be verified against a CPU reference outside the token graph.
  bool coop_gemv_q4k_matvec(
    const D3d12ResidentWeightRef &op,
    const std::vector<float> &input,
    std::vector<float> *outputs,
    std::string &error_message) {
    // This kernel addresses the resident buffer in block order. Under the
    // quad-interleaved layout it would read the right bytes from the wrong
    // offsets, so it reports unavailable rather than returning wrong numbers -
    // the runtime gate cannot cover this entry because the selftest calls it
    // directly.
    if (quad_interleave_enabled()) {
      error_message = "Cooperative Q4_K GEMV does not support the quad-interleaved layout.";
      return false;
    }
    if (!q4k_gemv_pipeline_) {
      error_message = "Cooperative Q4_K GEMV pipeline is unavailable.";
      return false;
    }
    if (op.format != D3d12WeightFormat::Q4_K
      || op.row_width == 0
      || op.row_count == 0
      || (op.row_width % k_quant_block_elements) != 0
      || input.size() != op.row_width
      || expected_weight_bytes(op.format, op.row_count, op.row_width) != op.weight_bytes) {
      error_message = "Cooperative GEMV received an invalid Q4_K matvec shape.";
      return false;
    }
    if (weight_cache_.find(op.weight_key) == weight_cache_.end()
      && !stage_resident(op, error_message)) {
      return false;
    }
    use_counter_ += 1;
    ResidentWeight &resident = weight_cache_.at(op.weight_key);
    resident.last_use = use_counter_;

    const std::uint64_t input_bytes =
      static_cast<std::uint64_t>(input.size()) * sizeof(float);
    const std::uint64_t output_bytes =
      static_cast<std::uint64_t>(op.row_count) * sizeof(float);
    const std::uint64_t previous_output_bytes = dp4a_output_bytes_;
    if (!ensure_persistent_upload_buffer(
        dp4a_upload_, input_bytes, dp4a_upload_bytes_, mapped_dp4a_upload_, error_message)
      || !ensure_buffer(
        dp4a_output_,
        output_bytes,
        dp4a_output_bytes_,
        D3D12_HEAP_TYPE_DEFAULT,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        error_message)
      || !ensure_persistent_readback_buffer(
        dp4a_readback_, output_bytes, dp4a_readback_bytes_, mapped_dp4a_readback_, error_message)) {
      return false;
    }
    if (previous_output_bytes != dp4a_output_bytes_) {
      dp4a_output_state_ = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    }
    std::memcpy(mapped_dp4a_upload_, input.data(), static_cast<std::size_t>(input_bytes));

    HRESULT hr = command_allocator_->Reset();
    if (FAILED(hr)) {
      error_message = hresult_message("ID3D12CommandAllocator::Reset", hr);
      return false;
    }
    hr = command_list_->Reset(command_allocator_.get(), nullptr);
    if (FAILED(hr)) {
      error_message = hresult_message("ID3D12GraphicsCommandList::Reset", hr);
      return false;
    }
    command_list_->SetComputeRootSignature(root_signature_.get());
    transition(dp4a_output_.get(), dp4a_output_state_, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    DispatchConstants constants;
    constants.row_width = op.row_width;
    constants.row_count = op.row_count;
    constants.output_base = 0;
    constants.input_stride = op.row_width;
    constants.weight_layout = 1;
    command_list_->SetPipelineState(q4k_gemv_pipeline_.get());
    command_list_->SetComputeRoot32BitConstants(0, 8, &constants, 0);
    command_list_->SetComputeRootShaderResourceView(1, resident.buffer->GetGPUVirtualAddress());
    command_list_->SetComputeRootShaderResourceView(2, dp4a_upload_->GetGPUVirtualAddress());
    command_list_->SetComputeRootUnorderedAccessView(3, dp4a_output_->GetGPUVirtualAddress());
    command_list_->Dispatch(op.row_count, 1, 1);
    transition(dp4a_output_.get(), dp4a_output_state_, D3D12_RESOURCE_STATE_COPY_SOURCE);
    command_list_->CopyBufferRegion(dp4a_readback_.get(), 0, dp4a_output_.get(), 0, output_bytes);
    hr = command_list_->Close();
    if (FAILED(hr)) {
      error_message = hresult_message("ID3D12GraphicsCommandList::Close", hr);
      return false;
    }
    if (!submit_and_wait(error_message)) {
      return false;
    }
    outputs->resize(static_cast<std::size_t>(op.row_count));
    std::memcpy(outputs->data(), mapped_dp4a_readback_, static_cast<std::size_t>(output_bytes));
    return true;
  }

  // Test-only: sweep a resident buffer with the arithmetic-free streaming
  // shader. The caller times repeated invocations to establish the adapter's
  // raw sequential-read ceiling independently of quantization and matvec math.
  bool stream_read(
    const D3d12ResidentWeightRef &op,
    std::uint32_t group_count,
    std::uint64_t *checksum,
    std::string &error_message) {
    if (!stream_read_pipeline_) {
      error_message = "Streaming-read pipeline is unavailable.";
      return false;
    }
    if (group_count == 0
      || op.weight_bytes == 0
      || (op.weight_bytes % 16u) != 0
      || op.weight_bytes / 16u > std::numeric_limits<std::uint32_t>::max()) {
      error_message = "Streaming read received an invalid buffer shape.";
      return false;
    }
    if (weight_cache_.find(op.weight_key) == weight_cache_.end()
      && !stage_resident(op, error_message)) {
      return false;
    }
    use_counter_ += 1;
    ResidentWeight &resident = weight_cache_.at(op.weight_key);
    resident.last_use = use_counter_;

    const std::uint64_t output_count =
      static_cast<std::uint64_t>(group_count) * 256u;
    const std::uint64_t output_bytes = output_count * sizeof(std::uint32_t);
    const std::uint64_t previous_output_bytes = dp4a_output_bytes_;
    if (!ensure_buffer(
        dp4a_output_,
        output_bytes,
        dp4a_output_bytes_,
        D3D12_HEAP_TYPE_DEFAULT,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        error_message)
      || !ensure_persistent_readback_buffer(
        dp4a_readback_,
        output_bytes,
        dp4a_readback_bytes_,
        mapped_dp4a_readback_,
        error_message)) {
      return false;
    }
    if (previous_output_bytes != dp4a_output_bytes_) {
      dp4a_output_state_ = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    }

    HRESULT hr = command_allocator_->Reset();
    if (FAILED(hr)) {
      error_message = hresult_message("ID3D12CommandAllocator::Reset", hr);
      return false;
    }
    hr = command_list_->Reset(command_allocator_.get(), nullptr);
    if (FAILED(hr)) {
      error_message = hresult_message("ID3D12GraphicsCommandList::Reset", hr);
      return false;
    }
    command_list_->SetComputeRootSignature(root_signature_.get());
    transition(dp4a_output_.get(), dp4a_output_state_, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    DispatchConstants constants;
    constants.row_width = static_cast<std::uint32_t>(op.weight_bytes / 16u);
    constants.row_count = group_count;
    command_list_->SetPipelineState(stream_read_pipeline_.get());
    command_list_->SetComputeRoot32BitConstants(0, 8, &constants, 0);
    command_list_->SetComputeRootShaderResourceView(1, resident.buffer->GetGPUVirtualAddress());
    command_list_->SetComputeRootUnorderedAccessView(3, dp4a_output_->GetGPUVirtualAddress());
    command_list_->Dispatch(group_count, 1, 1);
    transition(dp4a_output_.get(), dp4a_output_state_, D3D12_RESOURCE_STATE_COPY_SOURCE);
    command_list_->CopyBufferRegion(dp4a_readback_.get(), 0, dp4a_output_.get(), 0, output_bytes);
    hr = command_list_->Close();
    if (FAILED(hr)) {
      error_message = hresult_message("ID3D12GraphicsCommandList::Close", hr);
      return false;
    }
    if (!submit_and_wait(error_message)) {
      return false;
    }

    if (checksum != nullptr) {
      const auto *words = static_cast<const std::uint32_t *>(mapped_dp4a_readback_);
      std::uint64_t sum = 0;
      for (std::uint64_t index = 0; index < output_count; index += 1) {
        sum += words[index];
      }
      *checksum = sum;
    }
    return true;
  }

  // Per-format 128-rows-per-group batch-one pipelines. Returns null for
  // formats without a wide variant (floats keep the default kernels).
  ID3D12PipelineState *wide128_pipeline_for(D3d12WeightFormat format) const {
    switch (format) {
      case D3d12WeightFormat::Q4_K: return q4k_wide128_pipeline_.get();
      case D3d12WeightFormat::Q5_K: return q5k_wide128_pipeline_.get();
      case D3d12WeightFormat::Q6_K: return q6k_wide128_pipeline_.get();
      case D3d12WeightFormat::Q8_0: return q8_0_wide128_pipeline_.get();
      case D3d12WeightFormat::Q4_0: return q4_0_wide128_pipeline_.get();
      default: return nullptr;
    }
  }

  // Test-only: dispatch a quant kernel compiled with 128 rows/group (instead
  // of the default 64) standalone, so the wider-group variant can be
  // verified against a CPU reference and timed before considering it for
  // real decode. Same one-thread-per-row math as the default b1 kernel,
  // just twice as many rows sharing each load_input_tile+barrier pair.
  bool loadonly_q4k_matvec(
    const D3d12ResidentWeightRef &op,
    const std::vector<float> &input,
    std::vector<float> *outputs,
    std::string &error_message,
    bool coalesced = false) {
    ID3D12PipelineState *const pipeline = coalesced
      ? q4k_loadonly_coalesced_pipeline_.get()
      : q4k_loadonly_pipeline_.get();
    if (!pipeline) {
      error_message = "load-only diagnostic pipeline is unavailable.";
      return false;
    }
    return wide128_q4k_matvec(op, input, outputs, error_message, pipeline);
  }

  bool wide128_q4k_matvec(
    const D3d12ResidentWeightRef &op,
    const std::vector<float> &input,
    std::vector<float> *outputs,
    std::string &error_message,
    // Diagnostic override: run the identical dispatch with a different
    // pipeline (the load-only twin) so the two can be timed under exactly the
    // same buffer, upload and readback path.
    ID3D12PipelineState *pipeline_override = nullptr) {
    if (!pipeline_override && !wide128_pipeline_for(op.format)) {
      error_message = "128-rows-per-group pipeline is unavailable for this format.";
      return false;
    }
    if (op.row_width == 0
      || op.row_count == 0
      || (op.row_width % k_quant_block_elements) != 0
      || input.size() != op.row_width
      || expected_weight_bytes(op.format, op.row_count, op.row_width) != op.weight_bytes) {
      error_message = "wide128 matvec received an invalid Q4_K matvec shape.";
      return false;
    }
    if (weight_cache_.find(op.weight_key) == weight_cache_.end()
      && !stage_resident(op, error_message)) {
      return false;
    }
    use_counter_ += 1;
    ResidentWeight &resident = weight_cache_.at(op.weight_key);
    resident.last_use = use_counter_;

    const std::uint64_t input_bytes =
      static_cast<std::uint64_t>(input.size()) * sizeof(float);
    const std::uint64_t output_bytes =
      static_cast<std::uint64_t>(op.row_count) * sizeof(float);
    const std::uint64_t previous_output_bytes = dp4a_output_bytes_;
    if (!ensure_persistent_upload_buffer(
        dp4a_upload_, input_bytes, dp4a_upload_bytes_, mapped_dp4a_upload_, error_message)
      || !ensure_buffer(
        dp4a_output_,
        output_bytes,
        dp4a_output_bytes_,
        D3D12_HEAP_TYPE_DEFAULT,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        error_message)
      || !ensure_persistent_readback_buffer(
        dp4a_readback_, output_bytes, dp4a_readback_bytes_, mapped_dp4a_readback_, error_message)) {
      return false;
    }
    if (previous_output_bytes != dp4a_output_bytes_) {
      dp4a_output_state_ = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    }
    std::memcpy(mapped_dp4a_upload_, input.data(), static_cast<std::size_t>(input_bytes));

    HRESULT hr = command_allocator_->Reset();
    if (FAILED(hr)) {
      error_message = hresult_message("ID3D12CommandAllocator::Reset", hr);
      return false;
    }
    hr = command_list_->Reset(command_allocator_.get(), nullptr);
    if (FAILED(hr)) {
      error_message = hresult_message("ID3D12GraphicsCommandList::Reset", hr);
      return false;
    }
    command_list_->SetComputeRootSignature(root_signature_.get());
    transition(dp4a_output_.get(), dp4a_output_state_, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    DispatchConstants constants;
    constants.row_width = op.row_width;
    constants.row_count = op.row_count;
    constants.batch_count = 1;
    constants.output_base = 0;
    constants.input_stride = op.row_width;
    constants.weight_layout = 1;
    command_list_->SetPipelineState(
      pipeline_override ? pipeline_override : wide128_pipeline_for(op.format));
    command_list_->SetComputeRoot32BitConstants(0, 8, &constants, 0);
    command_list_->SetComputeRootShaderResourceView(1, resident.buffer->GetGPUVirtualAddress());
    command_list_->SetComputeRootShaderResourceView(2, dp4a_upload_->GetGPUVirtualAddress());
    command_list_->SetComputeRootUnorderedAccessView(3, dp4a_output_->GetGPUVirtualAddress());
    command_list_->Dispatch((op.row_count + 127u) / 128u, 1, 1);
    transition(dp4a_output_.get(), dp4a_output_state_, D3D12_RESOURCE_STATE_COPY_SOURCE);
    command_list_->CopyBufferRegion(dp4a_readback_.get(), 0, dp4a_output_.get(), 0, output_bytes);
    hr = command_list_->Close();
    if (FAILED(hr)) {
      error_message = hresult_message("ID3D12GraphicsCommandList::Close", hr);
      return false;
    }
    if (!submit_and_wait(error_message)) {
      return false;
    }
    outputs->resize(static_cast<std::size_t>(op.row_count));
    std::memcpy(outputs->data(), mapped_dp4a_readback_, static_cast<std::size_t>(output_bytes));
    return true;
  }

  // Test-only: dispatch main_q4k compiled with 256 rows/group, to check
  // whether widening further than 128 keeps paying off. Not wired into the
  // token graph unless proven better than the 128 variant.
  bool wide256_q4k_matvec(
    const D3d12ResidentWeightRef &op,
    const std::vector<float> &input,
    std::vector<float> *outputs,
    std::string &error_message) {
    if (!q4k_wide256_pipeline_) {
      error_message = "256-rows-per-group Q4_K pipeline is unavailable.";
      return false;
    }
    if (op.format != D3d12WeightFormat::Q4_K
      || op.row_width == 0
      || op.row_count == 0
      || (op.row_width % k_quant_block_elements) != 0
      || input.size() != op.row_width
      || expected_weight_bytes(op.format, op.row_count, op.row_width) != op.weight_bytes) {
      error_message = "wide256 matvec received an invalid Q4_K matvec shape.";
      return false;
    }
    if (weight_cache_.find(op.weight_key) == weight_cache_.end()
      && !stage_resident(op, error_message)) {
      return false;
    }
    use_counter_ += 1;
    ResidentWeight &resident = weight_cache_.at(op.weight_key);
    resident.last_use = use_counter_;

    const std::uint64_t input_bytes =
      static_cast<std::uint64_t>(input.size()) * sizeof(float);
    const std::uint64_t output_bytes =
      static_cast<std::uint64_t>(op.row_count) * sizeof(float);
    const std::uint64_t previous_output_bytes = dp4a_output_bytes_;
    if (!ensure_persistent_upload_buffer(
        dp4a_upload_, input_bytes, dp4a_upload_bytes_, mapped_dp4a_upload_, error_message)
      || !ensure_buffer(
        dp4a_output_,
        output_bytes,
        dp4a_output_bytes_,
        D3D12_HEAP_TYPE_DEFAULT,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        error_message)
      || !ensure_persistent_readback_buffer(
        dp4a_readback_, output_bytes, dp4a_readback_bytes_, mapped_dp4a_readback_, error_message)) {
      return false;
    }
    if (previous_output_bytes != dp4a_output_bytes_) {
      dp4a_output_state_ = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    }
    std::memcpy(mapped_dp4a_upload_, input.data(), static_cast<std::size_t>(input_bytes));

    HRESULT hr = command_allocator_->Reset();
    if (FAILED(hr)) {
      error_message = hresult_message("ID3D12CommandAllocator::Reset", hr);
      return false;
    }
    hr = command_list_->Reset(command_allocator_.get(), nullptr);
    if (FAILED(hr)) {
      error_message = hresult_message("ID3D12GraphicsCommandList::Reset", hr);
      return false;
    }
    command_list_->SetComputeRootSignature(root_signature_.get());
    transition(dp4a_output_.get(), dp4a_output_state_, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    DispatchConstants constants;
    constants.row_width = op.row_width;
    constants.row_count = op.row_count;
    constants.batch_count = 1;
    constants.output_base = 0;
    constants.input_stride = op.row_width;
    constants.weight_layout = 1;
    command_list_->SetPipelineState(q4k_wide256_pipeline_.get());
    command_list_->SetComputeRoot32BitConstants(0, 8, &constants, 0);
    command_list_->SetComputeRootShaderResourceView(1, resident.buffer->GetGPUVirtualAddress());
    command_list_->SetComputeRootShaderResourceView(2, dp4a_upload_->GetGPUVirtualAddress());
    command_list_->SetComputeRootUnorderedAccessView(3, dp4a_output_->GetGPUVirtualAddress());
    command_list_->Dispatch((op.row_count + 255u) / 256u, 1, 1);
    transition(dp4a_output_.get(), dp4a_output_state_, D3D12_RESOURCE_STATE_COPY_SOURCE);
    command_list_->CopyBufferRegion(dp4a_readback_.get(), 0, dp4a_output_.get(), 0, output_bytes);
    hr = command_list_->Close();
    if (FAILED(hr)) {
      error_message = hresult_message("ID3D12GraphicsCommandList::Close", hr);
      return false;
    }
    if (!submit_and_wait(error_message)) {
      return false;
    }
    outputs->resize(static_cast<std::size_t>(op.row_count));
    std::memcpy(outputs->data(), mapped_dp4a_readback_, static_cast<std::size_t>(output_bytes));
    return true;
  }

  // Test-only: dispatch the DXC/SM6.0 WaveActiveSum cooperative Q4_K GEMV
  // standalone, mirroring coop_gemv_q4k_matvec above but on the wave-reduction
  // pipeline, so it can be verified against a CPU reference outside the graph.
  bool wave_gemv_q4k_matvec(
    const D3d12ResidentWeightRef &op,
    const std::vector<float> &input,
    std::vector<float> *outputs,
    std::string &error_message) {
    // Block-order addressing; declines under the quad layout for the same
    // reason the cooperative GEMV does.
    if (quad_interleave_enabled()) {
      error_message = "Wave Q4_K GEMV does not support the quad-interleaved layout.";
      return false;
    }
    if (!q4k_gemv_wave_pipeline_) {
      error_message = "Wave-reduction Q4_K GEMV pipeline is unavailable.";
      return false;
    }
    if (op.format != D3d12WeightFormat::Q4_K
      || op.row_width == 0
      || op.row_count == 0
      || (op.row_width % k_quant_block_elements) != 0
      || input.size() != op.row_width
      || expected_weight_bytes(op.format, op.row_count, op.row_width) != op.weight_bytes) {
      error_message = "Wave GEMV received an invalid Q4_K matvec shape.";
      return false;
    }
    if (weight_cache_.find(op.weight_key) == weight_cache_.end()
      && !stage_resident(op, error_message)) {
      return false;
    }
    use_counter_ += 1;
    ResidentWeight &resident = weight_cache_.at(op.weight_key);
    resident.last_use = use_counter_;

    const std::uint64_t input_bytes =
      static_cast<std::uint64_t>(input.size()) * sizeof(float);
    const std::uint64_t output_bytes =
      static_cast<std::uint64_t>(op.row_count) * sizeof(float);
    const std::uint64_t previous_output_bytes = dp4a_output_bytes_;
    if (!ensure_persistent_upload_buffer(
        dp4a_upload_, input_bytes, dp4a_upload_bytes_, mapped_dp4a_upload_, error_message)
      || !ensure_buffer(
        dp4a_output_,
        output_bytes,
        dp4a_output_bytes_,
        D3D12_HEAP_TYPE_DEFAULT,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        error_message)
      || !ensure_persistent_readback_buffer(
        dp4a_readback_, output_bytes, dp4a_readback_bytes_, mapped_dp4a_readback_, error_message)) {
      return false;
    }
    if (previous_output_bytes != dp4a_output_bytes_) {
      dp4a_output_state_ = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    }
    std::memcpy(mapped_dp4a_upload_, input.data(), static_cast<std::size_t>(input_bytes));

    HRESULT hr = command_allocator_->Reset();
    if (FAILED(hr)) {
      error_message = hresult_message("ID3D12CommandAllocator::Reset", hr);
      return false;
    }
    hr = command_list_->Reset(command_allocator_.get(), nullptr);
    if (FAILED(hr)) {
      error_message = hresult_message("ID3D12GraphicsCommandList::Reset", hr);
      return false;
    }
    command_list_->SetComputeRootSignature(root_signature_.get());
    transition(dp4a_output_.get(), dp4a_output_state_, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    DispatchConstants constants;
    constants.row_width = op.row_width;
    constants.row_count = op.row_count;
    constants.output_base = 0;
    constants.input_stride = 0;
    constants.weight_layout = 1;
    command_list_->SetPipelineState(q4k_gemv_wave_pipeline_.get());
    command_list_->SetComputeRoot32BitConstants(0, 8, &constants, 0);
    command_list_->SetComputeRootShaderResourceView(1, resident.buffer->GetGPUVirtualAddress());
    command_list_->SetComputeRootShaderResourceView(2, dp4a_upload_->GetGPUVirtualAddress());
    command_list_->SetComputeRootUnorderedAccessView(3, dp4a_output_->GetGPUVirtualAddress());
    command_list_->Dispatch(op.row_count, 1, 1);
    transition(dp4a_output_.get(), dp4a_output_state_, D3D12_RESOURCE_STATE_COPY_SOURCE);
    command_list_->CopyBufferRegion(dp4a_readback_.get(), 0, dp4a_output_.get(), 0, output_bytes);
    hr = command_list_->Close();
    if (FAILED(hr)) {
      error_message = hresult_message("ID3D12GraphicsCommandList::Close", hr);
      return false;
    }
    if (!submit_and_wait(error_message)) {
      return false;
    }
    outputs->resize(static_cast<std::size_t>(op.row_count));
    std::memcpy(outputs->data(), mapped_dp4a_readback_, static_cast<std::size_t>(output_bytes));
    return true;
  }

  // Quantizes `input` to Q8_1 on the GPU, then multiplies the resident Q4_K
  // weights against it with dot4add_i8packed. One submission: quantize pass,
  // barrier, matvec, readback copy.
  bool dp4a_q4k_matvec(
    const D3d12ResidentWeightRef &op,
    const std::vector<float> &input,
    std::vector<float> *outputs,
    std::string &error_message) {
    if (!dp4a_capabilities_.supported) {
      error_message = dp4a_capabilities_.reason;
      return false;
    }
    if (op.format != D3d12WeightFormat::Q4_K) {
      error_message = "DP4A matvec currently supports Q4_K weights only.";
      return false;
    }
    if (op.row_width == 0
      || op.row_count == 0
      || (op.row_width % k_quant_block_elements) != 0
      || input.size() != op.row_width
      || expected_weight_bytes(op.format, op.row_count, op.row_width) != op.weight_bytes) {
      error_message = "DP4A matvec received an invalid Q4_K matvec shape.";
      return false;
    }
    if (weight_cache_.find(op.weight_key) == weight_cache_.end()
      && !stage_resident(op, error_message)) {
      return false;
    }
    use_counter_ += 1;
    ResidentWeight &resident = weight_cache_.at(op.weight_key);
    resident.last_use = use_counter_;

    const std::uint64_t input_bytes =
      static_cast<std::uint64_t>(input.size()) * sizeof(float);
    const std::uint64_t activation_words =
      static_cast<std::uint64_t>(op.row_width) / 4
      + static_cast<std::uint64_t>(op.row_width) / 32 * 2;
    const std::uint64_t activation_bytes = activation_words * sizeof(std::uint32_t);
    const std::uint64_t output_bytes =
      static_cast<std::uint64_t>(op.row_count) * sizeof(float);
    const std::uint64_t previous_activation_bytes = dp4a_activations_bytes_;
    const std::uint64_t previous_output_bytes = dp4a_output_bytes_;
    if (!ensure_persistent_upload_buffer(
        dp4a_upload_, input_bytes, dp4a_upload_bytes_, mapped_dp4a_upload_, error_message)
      || !ensure_buffer(
        dp4a_activations_,
        activation_bytes,
        dp4a_activations_bytes_,
        D3D12_HEAP_TYPE_DEFAULT,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        error_message)
      || !ensure_buffer(
        dp4a_output_,
        output_bytes,
        dp4a_output_bytes_,
        D3D12_HEAP_TYPE_DEFAULT,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        error_message)
      || !ensure_persistent_readback_buffer(
        dp4a_readback_, output_bytes, dp4a_readback_bytes_, mapped_dp4a_readback_, error_message)) {
      return false;
    }
    if (previous_activation_bytes != dp4a_activations_bytes_) {
      dp4a_activations_state_ = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    }
    if (previous_output_bytes != dp4a_output_bytes_) {
      dp4a_output_state_ = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    }
    std::memcpy(mapped_dp4a_upload_, input.data(), static_cast<std::size_t>(input_bytes));

    HRESULT hr = command_allocator_->Reset();
    if (FAILED(hr)) {
      error_message = hresult_message("ID3D12CommandAllocator::Reset", hr);
      return false;
    }
    hr = command_list_->Reset(command_allocator_.get(), nullptr);
    if (FAILED(hr)) {
      error_message = hresult_message("ID3D12GraphicsCommandList::Reset", hr);
      return false;
    }
    command_list_->SetComputeRootSignature(root_signature_.get());

    // Quantize pass: t1 = raw floats straight from the upload heap.
    transition(dp4a_activations_.get(), dp4a_activations_state_, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    DispatchConstants quantize_constants;
    quantize_constants.row_width = op.row_width;
    quantize_constants.input_stride = 0;
    quantize_constants.output_base = 0;
    command_list_->SetPipelineState(dp4a_quantize_pipeline_.get());
    command_list_->SetComputeRoot32BitConstants(0, 8, &quantize_constants, 0);
    command_list_->SetComputeRootShaderResourceView(1, resident.buffer->GetGPUVirtualAddress());
    command_list_->SetComputeRootShaderResourceView(2, dp4a_upload_->GetGPUVirtualAddress());
    command_list_->SetComputeRootUnorderedAccessView(3, dp4a_activations_->GetGPUVirtualAddress());
    command_list_->Dispatch(static_cast<UINT>((op.row_width + 63u) / 64u), 1, 1);

    D3D12_RESOURCE_BARRIER activation_barrier = {};
    activation_barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    activation_barrier.UAV.pResource = dp4a_activations_.get();
    command_list_->ResourceBarrier(1, &activation_barrier);
    transition(
      dp4a_activations_.get(),
      dp4a_activations_state_,
      D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

    // Integer matvec pass against the resident Q4_K arena.
    transition(dp4a_output_.get(), dp4a_output_state_, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    DispatchConstants matvec_constants;
    matvec_constants.row_width = op.row_width;
    matvec_constants.row_count = op.row_count;
    matvec_constants.input_stride = 0;
    matvec_constants.output_base = 0;
    command_list_->SetPipelineState(dp4a_q4k_b1_pipeline_.get());
    command_list_->SetComputeRoot32BitConstants(0, 8, &matvec_constants, 0);
    command_list_->SetComputeRootShaderResourceView(1, resident.buffer->GetGPUVirtualAddress());
    command_list_->SetComputeRootShaderResourceView(2, dp4a_activations_->GetGPUVirtualAddress());
    command_list_->SetComputeRootUnorderedAccessView(3, dp4a_output_->GetGPUVirtualAddress());
    command_list_->Dispatch(static_cast<UINT>((op.row_count + 63u) / 64u), 1, 1);

    transition(dp4a_output_.get(), dp4a_output_state_, D3D12_RESOURCE_STATE_COPY_SOURCE);
    command_list_->CopyBufferRegion(dp4a_readback_.get(), 0, dp4a_output_.get(), 0, output_bytes);

    hr = command_list_->Close();
    if (FAILED(hr)) {
      error_message = hresult_message("ID3D12GraphicsCommandList::Close", hr);
      return false;
    }
    if (!submit_and_wait(error_message)) {
      return false;
    }
    telemetry_.input_upload_bytes += input_bytes;
    telemetry_.output_readback_bytes += output_bytes;
    outputs->resize(static_cast<std::size_t>(op.row_count));
    std::memcpy(outputs->data(), mapped_dp4a_readback_, static_cast<std::size_t>(output_bytes));
    return true;
  }

 private:

  bool create_pipelines(std::string &error_message) {
    D3D12_ROOT_PARAMETER parameters[4] = {};
    parameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    parameters[0].Constants.ShaderRegister = 0;
    parameters[0].Constants.RegisterSpace = 0;
    parameters[0].Constants.Num32BitValues = 8;
    parameters[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    parameters[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
    parameters[1].Descriptor.ShaderRegister = 0;
    parameters[1].Descriptor.RegisterSpace = 0;
    parameters[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    parameters[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
    parameters[2].Descriptor.ShaderRegister = 1;
    parameters[2].Descriptor.RegisterSpace = 0;
    parameters[2].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    parameters[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    parameters[3].Descriptor.ShaderRegister = 0;
    parameters[3].Descriptor.RegisterSpace = 0;
    parameters[3].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    D3D12_ROOT_SIGNATURE_DESC root_signature_desc = {};
    root_signature_desc.NumParameters = 4;
    root_signature_desc.pParameters = parameters;
    root_signature_desc.NumStaticSamplers = 0;
    root_signature_desc.pStaticSamplers = nullptr;
    root_signature_desc.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;

    ID3DBlob *root_blob_raw = nullptr;
    ID3DBlob *root_error_blob_raw = nullptr;
    HRESULT hr = D3D12SerializeRootSignature(
      &root_signature_desc,
      D3D_ROOT_SIGNATURE_VERSION_1,
      &root_blob_raw,
      &root_error_blob_raw);
    ComPtr<ID3DBlob> root_blob;
    ComPtr<ID3DBlob> root_error_blob;
    root_blob.reset(root_blob_raw);
    root_error_blob.reset(root_error_blob_raw);
    if (FAILED(hr)) {
      error_message = root_error_blob
        ? std::string(static_cast<const char *>(root_error_blob->GetBufferPointer()), root_error_blob->GetBufferSize())
        : hresult_message("D3D12SerializeRootSignature", hr);
      return false;
    }

    ID3D12RootSignature *root_signature_raw = nullptr;
    hr = device_->CreateRootSignature(
      0,
      root_blob->GetBufferPointer(),
      root_blob->GetBufferSize(),
      IID_PPV_ARGS(&root_signature_raw));
    if (FAILED(hr)) {
      error_message = hresult_message("CreateRootSignature", hr);
      return false;
    }
    root_signature_.reset(root_signature_raw);

    // Row-blocked prefill kernel, compiled only when asked for and never under
    // the quad-interleaved layout (it addresses blocks in block order).
    const std::uint32_t row_block = gpu_row_block();
    const bool want_row_block = row_block > 1 && !quad_interleave_enabled();
    const std::string row_block_value = std::to_string(row_block);
    if (want_row_block
      && !compile_pipeline(
        "main_q4k_rowblock",
        q4k_rowblock_pipeline_,
        error_message,
        false,
        nullptr,
        row_block_value.c_str())) {
      return false;
    }
    // Q6_K gets its own rows-per-thread: its inner body is twice the Q4_K one,
    // so it hits the FXC unroll cliff at half the R.
    const std::string q6k_row_block_value = std::to_string(gpu_q6k_row_block());
    if (want_row_block
      && gpu_q6k_row_block() > 1u
      && !compile_pipeline(
        "main_q6k_rowblock",
        q6k_rowblock_pipeline_,
        error_message,
        false,
        nullptr,
        q6k_row_block_value.c_str(),
        gpu_q6k_prefill_tile())) {
      return false;
    }
    // The alternate tile is opt-in: it doubles row-blocked pipeline compilation,
    // and a first run after any shader edit already recompiles everything through
    // FXC with no output for minutes.
    if (want_row_block
      && (gpu_rowblock_alt_tile_enabled() || gpu_rowblock_alt_rows() > 0u)) {
      // Rows A/B takes precedence when both are set: it is the open question.
      const std::uint32_t alt_rows = gpu_rowblock_alt_rows();
      const std::string alt_rows_value =
        std::to_string(alt_rows > 0u ? alt_rows : row_block);
      rowblock_alt_tile_ = alt_rows > 0u
        ? gpu_prefill_tile()
        : (gpu_prefill_tile() == 8u ? 16u : 8u);
      rowblock_alt_rows_ = alt_rows > 0u ? alt_rows : row_block;
      if (!compile_pipeline(
          "main_q4k_rowblock",
          q4k_rowblock_alt_pipeline_,
          error_message,
          false,
          nullptr,
          alt_rows_value.c_str(),
          rowblock_alt_tile_)) {
        return false;
      }
    }

    return compile_pipeline("main_q4k", q4k_pipeline_, error_message)
      && compile_pipeline("main_q5k", q5k_pipeline_, error_message)
      && compile_pipeline("main_q6k", q6k_pipeline_, error_message)
      && compile_pipeline("main_q8_0", q8_0_pipeline_, error_message)
      && compile_pipeline("main_q4_0", q4_0_pipeline_, error_message)
      && compile_pipeline("main_scalar", scalar_pipeline_, error_message)
      && compile_pipeline("main_q4k", q4k_b1_pipeline_, error_message, true)
      && compile_pipeline("main_q4k", q4k_wide128_pipeline_, error_message, false, "128")
      && compile_pipeline("main_q5k", q5k_wide128_pipeline_, error_message, false, "128")
      && compile_pipeline("main_q6k", q6k_wide128_pipeline_, error_message, false, "128")
      && compile_pipeline("main_q8_0", q8_0_wide128_pipeline_, error_message, false, "128")
      && compile_pipeline("main_q4_0", q4_0_wide128_pipeline_, error_message, false, "128")
      && compile_pipeline("main_q4k", q4k_wide256_pipeline_, error_message, false, "256")
      && compile_pipeline("main_q4k_direct", q4k_direct_pipeline_, error_message, false, "128")
      && compile_pipeline("main_q6k_direct", q6k_direct_pipeline_, error_message, false, "128")
      && compile_pipeline("main_q4k_pair", q4k_pair_pipeline_, error_message, false, "128")
      && compile_pipeline("main_q4k_gemv", q4k_gemv_pipeline_, error_message)
      && compile_pipeline("main_q5k", q5k_b1_pipeline_, error_message, true)
      && compile_pipeline("main_q6k", q6k_b1_pipeline_, error_message, true)
      && compile_pipeline("main_q8_0", q8_0_b1_pipeline_, error_message, true)
      && compile_pipeline("main_q4_0", q4_0_b1_pipeline_, error_message, true)
      && compile_pipeline("main_scalar", scalar_b1_pipeline_, error_message, true)
      && compile_pipeline("main_q4k_loadonly", q4k_loadonly_pipeline_, error_message, false, "128")
      && compile_pipeline("main_q4k_loadonly_coalesced", q4k_loadonly_coalesced_pipeline_, error_message, false, "128")
      && compile_pipeline("main_stream_read", stream_read_pipeline_, error_message)
      && compile_pipeline("main_fma_peak", fma_peak_pipeline_, error_message)
      && compile_pipeline("main_gelu_glu", gelu_pipeline_, error_message)
      && compile_pipeline("main_silu_glu", silu_glu_pipeline_, error_message)
      && compile_pipeline("main_stream_read", stream_read_pipeline_, error_message)
      && compile_pipeline("main_rms_norm", rms_norm_pipeline_, error_message)
      && compile_pipeline("main_rms_norm_dual", rms_norm_dual_pipeline_, error_message)
      && compile_pipeline("main_q4k_ksplit", q4k_ksplit_pipeline_, error_message)
      && compile_pipeline("main_ksplit_reduce", ksplit_reduce_pipeline_, error_message);
  }

  bool ensure_gemma_attention_pipelines(std::string &error_message) {
    if (gemma_prepare_head_pipeline_
      && gemma_online_attention_pipeline_
      && gemma_online_attention_tiled_pipeline_
      && ple_prepare_pipeline_
      && logit_softcap_pipeline_
      && argmax_pipeline_) {
      return true;
    }
    gemma_prepare_head_pipeline_.reset();
    gemma_online_attention_pipeline_.reset();
    gemma_online_attention_tiled_pipeline_.reset();
    ple_prepare_pipeline_.reset();
    logit_softcap_pipeline_.reset();
    argmax_pipeline_.reset();
    return compile_pipeline(
      "main_gemma_prepare_head",
      gemma_prepare_head_pipeline_,
      error_message)
      && compile_pipeline(
        "main_gemma_online_attention",
        gemma_online_attention_pipeline_,
        error_message)
      && compile_pipeline(
        "main_gemma_online_attention_tiled",
        gemma_online_attention_tiled_pipeline_,
        error_message)
      && compile_pipeline(
        "main_ple_prepare",
        ple_prepare_pipeline_,
        error_message)
      && compile_pipeline(
        "main_logit_softcap",
        logit_softcap_pipeline_,
        error_message)
      && compile_pipeline(
        "main_argmax",
        argmax_pipeline_,
        error_message);
  }

  bool ensure_persistent_upload_buffer(
    ComPtr<ID3D12Resource> &resource,
    std::uint64_t required_bytes,
    std::uint64_t &allocated_bytes,
    void *&mapped,
    std::string &error_message) {
    if (resource && allocated_bytes >= required_bytes && mapped != nullptr) {
      return true;
    }
    release_mapped_resource(resource, mapped);
    allocated_bytes = 0;
    if (!ensure_buffer(
      resource,
      required_bytes,
      allocated_bytes,
      D3D12_HEAP_TYPE_UPLOAD,
      D3D12_RESOURCE_STATE_GENERIC_READ,
      D3D12_RESOURCE_FLAG_NONE,
      error_message)) {
      return false;
    }
    return map_persistent_resource(resource.get(), mapped, "ID3D12Resource::Map(upload)", error_message);
  }

  bool ensure_persistent_readback_buffer(
    ComPtr<ID3D12Resource> &resource,
    std::uint64_t required_bytes,
    std::uint64_t &allocated_bytes,
    void *&mapped,
    std::string &error_message) {
    if (resource && allocated_bytes >= required_bytes && mapped != nullptr) {
      return true;
    }
    release_mapped_resource(resource, mapped);
    allocated_bytes = 0;
    if (!ensure_buffer(
      resource,
      required_bytes,
      allocated_bytes,
      D3D12_HEAP_TYPE_READBACK,
      D3D12_RESOURCE_STATE_COPY_DEST,
      D3D12_RESOURCE_FLAG_NONE,
      error_message)) {
      return false;
    }
    return map_persistent_resource(resource.get(), mapped, "ID3D12Resource::Map(readback)", error_message);
  }

  bool ensure_output_buffer(
    ComPtr<ID3D12Resource> &resource,
    std::uint64_t required_bytes,
    std::uint64_t &allocated_bytes,
    D3D12_RESOURCE_STATES &state,
    std::string &error_message) {
    if (uma_output_) {
      if (resource && allocated_bytes >= required_bytes && mapped_output_direct_ != nullptr) {
        return true;
      }
      release_mapped_resource(resource, mapped_output_direct_);
      allocated_bytes = 0;

      D3D12_HEAP_PROPERTIES heap_properties = {};
      heap_properties.Type = D3D12_HEAP_TYPE_CUSTOM;
      heap_properties.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_WRITE_BACK;
      heap_properties.MemoryPoolPreference = D3D12_MEMORY_POOL_L0;
      heap_properties.CreationNodeMask = 1;
      heap_properties.VisibleNodeMask = 1;

      D3D12_RESOURCE_DESC resource_desc = {};
      resource_desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
      resource_desc.Width = std::max<std::uint64_t>(required_bytes, 4u);
      resource_desc.Height = 1;
      resource_desc.DepthOrArraySize = 1;
      resource_desc.MipLevels = 1;
      resource_desc.Format = DXGI_FORMAT_UNKNOWN;
      resource_desc.SampleDesc.Count = 1;
      resource_desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
      resource_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

      ID3D12Resource *resource_raw = nullptr;
      const HRESULT hr = device_->CreateCommittedResource(
        &heap_properties,
        D3D12_HEAP_FLAG_NONE,
        &resource_desc,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        nullptr,
        IID_PPV_ARGS(&resource_raw));
      if (SUCCEEDED(hr)) {
        resource.reset(resource_raw);
        allocated_bytes = resource_desc.Width;
        state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        if (map_persistent_resource(resource.get(), mapped_output_direct_, "ID3D12Resource::Map(uma output)", error_message)) {
          return true;
        }
        resource.reset();
        allocated_bytes = 0;
      }
      // Fall back to the default-heap + readback path for this process if
      // the driver rejects a CPU-visible UAV heap.
      uma_output_ = false;
      trace_d3d12("ensure_output_buffer: uma output unavailable, using readback path");
    }

    const std::uint64_t previous_bytes = allocated_bytes;
    if (!ensure_buffer(
      resource,
      required_bytes,
      allocated_bytes,
      D3D12_HEAP_TYPE_DEFAULT,
      D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
      D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
      error_message)) {
      return false;
    }
    if (previous_bytes != allocated_bytes) {
      state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    }
    return true;
  }

  bool ensure_srv_buffer(
    ComPtr<ID3D12Resource> &resource,
    std::uint64_t required_bytes,
    std::uint64_t &allocated_bytes,
    D3D12_RESOURCE_STATES &state,
    std::string &error_message) {
    const std::uint64_t previous_bytes = allocated_bytes;
    if (!ensure_buffer(
      resource,
      required_bytes,
      allocated_bytes,
      D3D12_HEAP_TYPE_DEFAULT,
      D3D12_RESOURCE_STATE_COPY_DEST,
      D3D12_RESOURCE_FLAG_NONE,
      error_message)) {
      return false;
    }
    if (previous_bytes != allocated_bytes) {
      state = D3D12_RESOURCE_STATE_COPY_DEST;
    }
    return true;
  }

 public:
  // Times a pure streaming read of `bytes` with no dequantisation and no
  // arithmetic beyond an integer add: the memory-bandwidth ceiling this adapter
  // can reach at all. Compare it against the Q4_K matvec's achieved GB/s - if
  // they are close, the matvec is already at the hardware limit and no kernel
  // rewrite will close the remaining gap to the performance goal.
  // Sustained ALU rate, measured rather than taken from a datasheet. Reuses the
  // stream-read output buffer: this kernel touches memory only to store its
  // result, so the number it produces is the arithmetic ceiling this machine
  // actually reaches under its own power and thermal limits.
  // Paired A/B for the batched Q4_K kernels.
  //
  // Comparing separate processes on this machine cannot resolve anything under
  // about 25%: prefill samples spread 184-236 GFLOP/s on one binary, and the
  // pure-FMA kernel spread 6415-9267. The variance is between processes, because
  // CPU and iGPU share a power budget and clocks move. Four hypotheses in this
  // campaign were judged against that noise and every one of them produced a
  // confident number that led nowhere.
  //
  // The fix is to run both variants back to back inside ONE process, alternating,
  // and report the paired difference - which cancels whatever the clocks are
  // doing, because both halves of each pair see the same conditions.
  bool run_batched_ab(
    const D3d12ResidentWeightRef &op,
    const std::vector<float> &packed_inputs,
    std::uint32_t batch_count,
    int variant,   // 0 = plain, 1 = row-blocked, 2 = row-blocked at the alt tile
    int repeats,
    double *ms_per_iteration,
    std::string &error_message) {
    if (!capabilities_.supported) {
      error_message = capabilities_.reason;
      return false;
    }
    if (ms_per_iteration == nullptr || repeats <= 0) {
      error_message = "run_batched_ab needs an output pointer and one repeat.";
      return false;
    }
    if (variant == 4 && !dp4a_q4k_batched_nolds_pipeline_) {
      error_message = "ablation pipeline is unavailable (needs SM 6.4 and a DXIL build).";
      return false;
    }
    if (variant == 3
      && (!dp4a_q4k_batched_pipeline_ || !dp4a_quantize_pipeline_)) {
      error_message = "integer-dot batched pipeline is unavailable (needs SM 6.4 and a DXIL build).";
      return false;
    }
    ID3D12PipelineState *const pipeline = variant == 2
      ? q4k_rowblock_alt_pipeline_.get()
      : variant == 1
        ? q4k_rowblock_pipeline_.get()
        : q4k_pipeline_.get();
    if (pipeline == nullptr) {
      error_message = variant == 2
        ? "alternate-tile pipeline is unavailable (set LIZARD_NATIVE_GPU_TILE_AB=1)."
        : variant == 1
          ? "row-blocked pipeline is unavailable (set LIZARD_NATIVE_GPU_ROW_BLOCK>1)."
          : "batched q4k pipeline is unavailable.";
      return false;
    }
    // The rowblock kernel covers k_rows_per_group * rows outputs per group; the
    // plain one covers k_rows_per_group. Getting this wrong silently computes a
    // fraction of the rows and reports a flattering time, so it is derived from
    // the same helper the production dispatch uses.
    const std::uint32_t row_block = variant == 0
      ? 1u
      : variant == 2
        ? (rowblock_alt_rows_ > 0u ? rowblock_alt_rows_ : gpu_row_block())
        : gpu_row_block();
    const std::uint32_t tile = variant == 2
      ? rowblock_alt_tile_
      : variant == 1
        ? gpu_prefill_tile()
        : k_gpu_batch_tile;

    std::vector<std::vector<float>> outputs(1);
    std::string local_error;
    // One staging pass so resident weights and the input upload are warm and are
    // not part of either half of the pair.
    if (!compute_resident(
        &op, 1, packed_inputs, batch_count, packed_inputs.data(),
        outputs.data(), local_error)) {
      error_message = local_error;
      return false;
    }

    const auto dispatch_once = [&]() {
      HRESULT hr = command_allocator_->Reset();
      if (FAILED(hr)) {
        error_message = hresult_message("ID3D12CommandAllocator::Reset", hr);
        return false;
      }
      hr = command_list_->Reset(command_allocator_.get(), nullptr);
      if (FAILED(hr)) {
        error_message = hresult_message("ID3D12GraphicsCommandList::Reset", hr);
        return false;
      }
      command_list_->SetComputeRootSignature(root_signature_.get());
      ResidentWeight &resident = weight_cache_.at(op.weight_key);
      DispatchConstants constants;
      constants.row_width = op.row_width;
      constants.row_count = op.row_count;
      constants.batch_count = batch_count;
      constants.output_base = 0;
      constants.input_stride = op.row_width;
      constants.weight_layout = 1;
      constants.scalar_format = 1;
      if (variant == 4) {
        // Ablation: identical arithmetic, no activation input at all. Any
        // difference from variant 3 is the cost of the activation path.
        command_list_->SetPipelineState(dp4a_q4k_batched_nolds_pipeline_.get());
        command_list_->SetComputeRoot32BitConstants(0, 8, &constants, 0);
        command_list_->SetComputeRootShaderResourceView(
          1, resident.buffer->GetGPUVirtualAddress());
        command_list_->SetComputeRootShaderResourceView(
          2, input_default_->GetGPUVirtualAddress());
        command_list_->SetComputeRootUnorderedAccessView(
          3, output_default_->GetGPUVirtualAddress());
        // The ablation twin keeps 64-thread groups and one row per thread, so it
        // still isolates arithmetic plus weight loads from the activation path.
        command_list_->Dispatch(
          static_cast<UINT>((op.row_count + 63u) / 64u),
          static_cast<UINT>((batch_count + 7u) / 8u),
          1);
        if (FAILED(command_list_->Close())) {
          error_message = "ablation command list could not be closed.";
          return false;
        }
        return submit_and_wait(error_message);
      }
      if (variant == 3) {
        // Integer-dot path: quantise every column to Q8_1 first, each into its own
        // region, then one GEMM pass that reuses each row's nibbles across the
        // k_dp4a_tile columns of its group.
        const std::uint32_t quant_words_per_column =
          op.row_width / 4u + (op.row_width / 32u) * 2u;
        const std::uint64_t activation_bytes =
          static_cast<std::uint64_t>(quant_words_per_column) * batch_count * sizeof(std::uint32_t);
        if (!ensure_buffer(
            dp4a_activations_, activation_bytes, dp4a_activations_bytes_,
            D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, error_message)) {
          return false;
        }
        transition(
          dp4a_activations_.get(), dp4a_activations_state_,
          D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        for (std::uint32_t column = 0; column < batch_count; column += 1) {
          DispatchConstants quantize_constants;
          quantize_constants.row_width = op.row_width;
          quantize_constants.input_stride = column * op.row_width;
          quantize_constants.output_base = column * quant_words_per_column;
          command_list_->SetPipelineState(dp4a_quantize_pipeline_.get());
          command_list_->SetComputeRoot32BitConstants(0, 8, &quantize_constants, 0);
          // main_quantize_q8_1 reads quantize_inputs at register(t1), which is
          // root parameter 2 - not 1. Binding the source to parameter 1 left t1
          // pointing at whatever the previous dispatch had there and hung the
          // device (DXGI_ERROR_DEVICE_HUNG, 0x887a0006). t0 is bound as well so
          // no declared slot is left stale.
          command_list_->SetComputeRootShaderResourceView(
            1, input_default_->GetGPUVirtualAddress());
          command_list_->SetComputeRootShaderResourceView(
            2, input_default_->GetGPUVirtualAddress());
          command_list_->SetComputeRootUnorderedAccessView(
            3, dp4a_activations_->GetGPUVirtualAddress());
          command_list_->Dispatch((op.row_width + 63u) / 64u, 1, 1);
        }
        D3D12_RESOURCE_BARRIER activation_barrier = {};
        activation_barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        activation_barrier.UAV.pResource = dp4a_activations_.get();
        command_list_->ResourceBarrier(1, &activation_barrier);
        transition(
          dp4a_activations_.get(), dp4a_activations_state_,
          D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

        constants.input_stride = 0;
        command_list_->SetPipelineState(dp4a_q4k_batched_pipeline_.get());
        command_list_->SetComputeRoot32BitConstants(0, 8, &constants, 0);
        command_list_->SetComputeRootShaderResourceView(
          1, resident.buffer->GetGPUVirtualAddress());
        command_list_->SetComputeRootShaderResourceView(
          2, dp4a_activations_->GetGPUVirtualAddress());
        command_list_->SetComputeRootUnorderedAccessView(
          3, output_default_->GetGPUVirtualAddress());
        // Must match LIZARD_DP4A_ROWS and LIZARD_DP4A_TILE in dp4a_q4k.hlsl:
        // a group covers 64 * rows outputs and tile columns. Getting either
        // wrong computes a fraction of the problem and reports a flatteringly
        // short time, which is the easiest way to fake a speedup here.
        // Mirror LIZARD_DP4A_GROUP / ROWS / TILE in dp4a_q4k.hlsl.
        const std::uint32_t dp4a_group = 64u;
        const std::uint32_t dp4a_rows = 1u;
        const std::uint32_t dp4a_tile = 8u;
        command_list_->Dispatch(
          static_cast<UINT>((op.row_count + (dp4a_group * dp4a_rows) - 1u)
            / (dp4a_group * dp4a_rows)),
          static_cast<UINT>((batch_count + dp4a_tile - 1u) / dp4a_tile),
          1);
        if (FAILED(command_list_->Close())) {
          error_message = "integer-dot A/B command list could not be closed.";
          return false;
        }
        return submit_and_wait(error_message);
      }
      command_list_->SetPipelineState(pipeline);
      command_list_->SetComputeRoot32BitConstants(0, 8, &constants, 0);
      command_list_->SetComputeRootShaderResourceView(
        1, resident.buffer->GetGPUVirtualAddress());
      command_list_->SetComputeRootShaderResourceView(
        2, input_default_->GetGPUVirtualAddress());
      command_list_->SetComputeRootUnorderedAccessView(
        3, output_default_->GetGPUVirtualAddress());
      command_list_->Dispatch(
        static_cast<UINT>((op.row_count + (64u * row_block) - 1u) / (64u * row_block)),
        static_cast<UINT>((batch_count + tile - 1u) / tile),
        1);
      if (FAILED(command_list_->Close())) {
        error_message = "A/B command list could not be closed.";
        return false;
      }
      return submit_and_wait(error_message);
    };

    if (!dispatch_once()) return false;
    const auto started = std::chrono::steady_clock::now();
    for (int repeat = 0; repeat < repeats; repeat += 1) {
      if (!dispatch_once()) return false;
    }
    const auto finished = std::chrono::steady_clock::now();
    *ms_per_iteration = std::chrono::duration_cast<std::chrono::microseconds>(
      finished - started).count() / 1000.0 / repeats;
    return true;
  }

  // Stages Q8_1 activations for every column of the current input into
  // dp4a_activations_, ready for the integer GEMM. Records the per-column
  // stride so the kernel and the host agree on the layout.
  // `source` defaults to the host-uploaded input, but any f32 buffer laid out as
  // batch_count columns of `width` contiguous floats works - the quantiser only
  // addresses it as column * width, and the resident matmuls write
  // outputs[base + batch * row_count + row], which is that layout.
  bool stage_prefill_activations(
    std::uint32_t width,
    std::uint32_t batch_count,
    std::uint32_t &stride_out,
    std::string &error_message,
    ID3D12Resource *source = nullptr) {
    const std::uint32_t stride = width / 4u + (width / 32u) * 2u;
    stride_out = stride;
    const std::uint64_t bytes =
      static_cast<std::uint64_t>(stride) * batch_count * sizeof(std::uint32_t);
    if (!ensure_buffer(
        dp4a_activations_, bytes, dp4a_activations_bytes_,
        D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, error_message)) {
      return false;
    }
    transition(
      dp4a_activations_.get(), dp4a_activations_state_,
      D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    // One dispatch for every column, not one per column.
    //
    // The host loop here was batch_count dispatches per staging call - 269 per
    // layer on a 269-token prefill, 7532 across the model - and it is the reason
    // extending the integer path to the FFN chain's down projection measured 1.8x
    // SLOWER than leaving it on the float kernels: a second staging pass at width
    // 8192 cost more than the faster matmul saved. The kernel now takes the column
    // from group_id.y, so the same work is one 2D dispatch.
    DispatchConstants quantize_constants;
    quantize_constants.row_width = width;
    quantize_constants.input_stride = 0;
    quantize_constants.output_base = 0;
    quantize_constants.input_transform = stride;
    command_list_->SetPipelineState(dp4a_quantize_pipeline_.get());
    command_list_->SetComputeRoot32BitConstants(0, 8, &quantize_constants, 0);
    // quantize_inputs is register(t1) = root parameter 2. Binding it to 1
    // hangs the device rather than returning garbage.
    ID3D12Resource *const source_buffer =
      source != nullptr ? source : input_default_.get();
    command_list_->SetComputeRootShaderResourceView(
      1, source_buffer->GetGPUVirtualAddress());
    command_list_->SetComputeRootShaderResourceView(
      2, source_buffer->GetGPUVirtualAddress());
    command_list_->SetComputeRootUnorderedAccessView(
      3, dp4a_activations_->GetGPUVirtualAddress());
    command_list_->Dispatch((width + 63u) / 64u, batch_count, 1);
    D3D12_RESOURCE_BARRIER activation_barrier = {};
    activation_barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    activation_barrier.UAV.pResource = dp4a_activations_.get();
    command_list_->ResourceBarrier(1, &activation_barrier);
    transition(
      dp4a_activations_.get(), dp4a_activations_state_,
      D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    return true;
  }

  // True when this op should run through the integer GEMM.
  bool prefill_integer_eligible(
    const D3d12ResidentWeightRef &op,
    std::uint32_t batch_count,
    std::uint32_t input_width) const {
    return gpu_dp4a_prefill_enabled()
      && dp4a_capabilities_.supported
      && dp4a_quantize_pipeline_
      && batch_count > 1
      && ((op.format == D3d12WeightFormat::Q4_K && dp4a_q4k_batched_pipeline_)
        || (op.format == D3d12WeightFormat::Q6_K && dp4a_q6k_batched_pipeline_))
      && (op.row_width % 256u) == 0u
      // The activations are staged with a per-column stride derived from
      // input_width, and the kernel recomputes that stride from row_width. If
      // they disagree every column address after the first is wrong - which is
      // the leading suspect for the 9.2% error, since a shifted column reads
      // another column's scales and sums rather than garbage, so the result
      // stays plausible instead of exploding.
      && op.row_width == input_width;
  }

  bool run_fma_peak(
    std::uint32_t iterations,
    std::uint32_t group_count,
    int repeats,
    double *ms_per_iteration,
    std::string &error_message) {
    if (!capabilities_.supported) {
      error_message = capabilities_.reason;
      return false;
    }
    if (!fma_peak_pipeline_) {
      error_message = "fma-peak pipeline is unavailable.";
      return false;
    }
    if (iterations == 0 || group_count == 0 || repeats <= 0) {
      error_message = "fma-peak benchmark needs non-zero iterations, groups and repeats.";
      return false;
    }
    const std::uint64_t output_bytes =
      static_cast<std::uint64_t>(group_count) * 256u * sizeof(float);
    if (!ensure_buffer(
        stream_read_output_, output_bytes, stream_read_output_bytes_,
        D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, error_message)) {
      return false;
    }
    const auto dispatch_once = [&]() {
      HRESULT hr = command_allocator_->Reset();
      if (FAILED(hr)) {
        error_message = hresult_message("ID3D12CommandAllocator::Reset", hr);
        return false;
      }
      hr = command_list_->Reset(command_allocator_.get(), nullptr);
      if (FAILED(hr)) {
        error_message = hresult_message("ID3D12GraphicsCommandList::Reset", hr);
        return false;
      }
      command_list_->SetComputeRootSignature(root_signature_.get());
      DispatchConstants constants;
      constants.row_width = iterations;
      constants.row_count = group_count;
      constants.batch_count = 1u;
      command_list_->SetPipelineState(fma_peak_pipeline_.get());
      command_list_->SetComputeRoot32BitConstants(0, 8, &constants, 0);
      command_list_->SetComputeRootShaderResourceView(
        1, stream_read_output_->GetGPUVirtualAddress());
      command_list_->SetComputeRootShaderResourceView(
        2, stream_read_output_->GetGPUVirtualAddress());
      command_list_->SetComputeRootUnorderedAccessView(
        3, stream_read_output_->GetGPUVirtualAddress());
      command_list_->Dispatch(group_count, 1, 1);
      if (FAILED(command_list_->Close())) {
        error_message = "fma-peak command list could not be closed.";
        return false;
      }
      return submit_and_wait(error_message);
    };
    if (!dispatch_once()) return false;
    const auto started = std::chrono::steady_clock::now();
    for (int repeat = 0; repeat < repeats; repeat += 1) {
      if (!dispatch_once()) return false;
    }
    const auto finished = std::chrono::steady_clock::now();
    *ms_per_iteration = std::chrono::duration_cast<std::chrono::microseconds>(
      finished - started).count() / 1000.0 / repeats;
    return true;
  }

  bool measure_stream_read(
    std::uint64_t bytes,
    int iterations,
    std::uint32_t group_count,
    std::uint32_t row_quads,
    double *ms_per_iteration,
    std::string &error_message) {
    if (!capabilities_.supported) {
      error_message = capabilities_.reason;
      return false;
    }
    if (!stream_read_pipeline_) {
      error_message = "stream-read pipeline is unavailable.";
      return false;
    }
    const std::uint64_t quads = bytes / 16u;
    if (quads == 0 || iterations <= 0) {
      error_message = "stream-read benchmark needs at least one 16-byte quad and one iteration.";
      return false;
    }
    // Group count is swept by the caller: the ceiling must be shown to be a
    // property of the memory system rather than of one dispatch shape before it
    // is used to judge the matvec, which dispatches a different shape.
    const std::uint32_t k_groups = group_count > 0 ? group_count : 64u;
    const std::uint64_t output_bytes = static_cast<std::uint64_t>(k_groups) * 256u * sizeof(float);
    if (!ensure_buffer(
        stream_read_buffer_, bytes, stream_read_buffer_bytes_,
        D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_FLAG_NONE, error_message)
      || !ensure_buffer(
        stream_read_output_, output_bytes, stream_read_output_bytes_,
        D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, error_message)) {
      return false;
    }

    const auto dispatch_once = [&]() {
      HRESULT hr = command_allocator_->Reset();
      if (FAILED(hr)) {
        error_message = hresult_message("ID3D12CommandAllocator::Reset", hr);
        return false;
      }
      hr = command_list_->Reset(command_allocator_.get(), nullptr);
      if (FAILED(hr)) {
        error_message = hresult_message("ID3D12GraphicsCommandList::Reset", hr);
        return false;
      }
      command_list_->SetComputeRootSignature(root_signature_.get());
      DispatchConstants constants;
      constants.row_width = static_cast<std::uint32_t>(quads);
      constants.row_count = k_groups;
      constants.batch_count = row_quads;
      command_list_->SetPipelineState(stream_read_pipeline_.get());
      command_list_->SetComputeRoot32BitConstants(0, 8, &constants, 0);
      command_list_->SetComputeRootShaderResourceView(
        1, stream_read_buffer_->GetGPUVirtualAddress());
      command_list_->SetComputeRootShaderResourceView(
        2, stream_read_buffer_->GetGPUVirtualAddress());
      command_list_->SetComputeRootUnorderedAccessView(
        3, stream_read_output_->GetGPUVirtualAddress());
      command_list_->Dispatch(k_groups, 1, 1);
      if (FAILED(command_list_->Close())) {
        error_message = "stream-read command list could not be closed.";
        return false;
      }
      return submit_and_wait(error_message);
    };

    if (!dispatch_once()) return false;  // warm the caches and the pipeline
    const auto started = std::chrono::steady_clock::now();
    for (int iteration = 0; iteration < iterations; iteration += 1) {
      if (!dispatch_once()) return false;
    }
    const auto ended = std::chrono::steady_clock::now();
    *ms_per_iteration = std::chrono::duration_cast<std::chrono::microseconds>(
      ended - started).count() / 1000.0 / iterations;
    return true;
  }

 private:
  bool ensure_buffer(
    ComPtr<ID3D12Resource> &resource,
    std::uint64_t required_bytes,
    std::uint64_t &allocated_bytes,
    D3D12_HEAP_TYPE heap_type,
    D3D12_RESOURCE_STATES initial_state,
    D3D12_RESOURCE_FLAGS flags,
    std::string &error_message) {
    if (resource && allocated_bytes >= required_bytes) {
      return true;
    }

    resource.reset();
    allocated_bytes = 0;
    D3D12_HEAP_PROPERTIES heap_properties = {};
    heap_properties.Type = heap_type;
    heap_properties.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    heap_properties.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    heap_properties.CreationNodeMask = 1;
    heap_properties.VisibleNodeMask = 1;

    D3D12_RESOURCE_DESC resource_desc = {};
    resource_desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    resource_desc.Alignment = 0;
    resource_desc.Width = std::max<std::uint64_t>(required_bytes, 4u);
    resource_desc.Height = 1;
    resource_desc.DepthOrArraySize = 1;
    resource_desc.MipLevels = 1;
    resource_desc.Format = DXGI_FORMAT_UNKNOWN;
    resource_desc.SampleDesc.Count = 1;
    resource_desc.SampleDesc.Quality = 0;
    resource_desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    resource_desc.Flags = flags;

    ID3D12Resource *resource_raw = nullptr;
    HRESULT hr = device_->CreateCommittedResource(
      &heap_properties,
      D3D12_HEAP_FLAG_NONE,
      &resource_desc,
      initial_state,
      nullptr,
      IID_PPV_ARGS(&resource_raw));
    if (FAILED(hr)) {
      error_message = hresult_message("CreateCommittedResource", hr);
      return false;
    }
    resource.reset(resource_raw);
    allocated_bytes = resource_desc.Width;
    return true;
  }

  bool map_persistent_resource(
    ID3D12Resource *resource,
    void *&mapped,
    const char *map_label,
    std::string &error_message) {
    mapped = nullptr;
    D3D12_RANGE range = {0, 0};
    const HRESULT hr = resource->Map(0, &range, &mapped);
    if (FAILED(hr)) {
      error_message = hresult_message(map_label, hr);
      return false;
    }
    return true;
  }

  void release_mapped_resource(ComPtr<ID3D12Resource> &resource, void *&mapped) {
    if (resource && mapped != nullptr) {
      D3D12_RANGE written_range = {0, 0};
      resource->Unmap(0, &written_range);
      mapped = nullptr;
    }
    resource.reset();
  }

  bool copy_mapped_buffer(
    void *mapped,
    const void *source,
    std::uint64_t bytes,
    const char *label,
    std::string &error_message) {
    if (mapped == nullptr) {
      error_message = std::string(label) + " buffer is not mapped.";
      return false;
    }
    std::memcpy(mapped, source, static_cast<std::size_t>(bytes));
    return true;
  }

  D3d12BatchMatmulCapabilities capabilities_;
  ComPtr<IDXGIFactory1> factory_;
  ComPtr<IDXGIAdapter1> adapter_;
  ComPtr<ID3D12Device> device_;
  ComPtr<ID3D12Device1> device1_;
  ComPtr<ID3D12CommandQueue> command_queue_;
  ComPtr<ID3D12CommandAllocator> command_allocator_;
  ComPtr<ID3D12GraphicsCommandList> command_list_;
  ComPtr<ID3D12Fence> fence_;
  HANDLE fence_event_ = nullptr;
  std::uint64_t fence_value_ = 0;
  bool device_removed_ = false;
  // Cache-coherent UMA lets the output UAV be mapped for direct CPU reads.
  bool uma_output_ = false;
  void *mapped_output_direct_ = nullptr;

  ComPtr<ID3D12RootSignature> root_signature_;
  ComPtr<ID3D12PipelineState> q4k_pipeline_;
  ComPtr<ID3D12PipelineState> q4k_rowblock_pipeline_;
  // Same kernel, different batch tile, compiled so the tile can be A/B'd inside
  // one process. "Tile 16 loses" was concluded from between-process runs that
  // could not resolve the difference; this is how to settle it.
  ComPtr<ID3D12PipelineState> q4k_rowblock_alt_pipeline_;
  ComPtr<ID3D12PipelineState> q6k_rowblock_pipeline_;
  // Batched Q4_K x Q8_1 integer-dot kernel (prefill). Shader Model 6.4, so it
  // lives with the other DXIL pipelines rather than the FXC ones.
  ComPtr<ID3D12PipelineState> dp4a_q4k_batched_pipeline_;
  ComPtr<ID3D12PipelineState> dp4a_q6k_batched_pipeline_;
  // Ablation twin: same arithmetic, no activation path. Diagnostic only.
  ComPtr<ID3D12PipelineState> dp4a_q4k_batched_nolds_pipeline_;
  std::uint32_t rowblock_alt_tile_ = 0u;
  std::uint32_t rowblock_alt_rows_ = 0u;
  ComPtr<ID3D12PipelineState> fma_peak_pipeline_;
  ComPtr<ID3D12PipelineState> q5k_pipeline_;
  ComPtr<ID3D12PipelineState> q6k_pipeline_;
  ComPtr<ID3D12PipelineState> q8_0_pipeline_;
  ComPtr<ID3D12PipelineState> q4_0_pipeline_;
  ComPtr<ID3D12PipelineState> scalar_pipeline_;
  ComPtr<ID3D12PipelineState> q4k_b1_pipeline_;
  // Experimental: main_q4k compiled with 128 rows/group instead of 64,
  // amortizing the per-block load_input_tile+barrier pair over twice as
  // many output rows per synchronization point. Opt-in
  // (LIZARD_NATIVE_GPU_WIDE_GROUP=1); see wide_group_enabled() for the
  // rationale (fewer, LARGER dispatches - not more/smaller - unlike the
  // disproven cooperative-GEMV pattern).
  ComPtr<ID3D12PipelineState> q4k_wide128_pipeline_;
  // Same 128-rows-per-group variant for the remaining quantized decode
  // formats. The macro only changes numthreads/k_rows_per_group; every quant
  // kernel derives row identically (group_id.x * k_rows_per_group + index).
  ComPtr<ID3D12PipelineState> q5k_wide128_pipeline_;
  ComPtr<ID3D12PipelineState> q6k_wide128_pipeline_;
  ComPtr<ID3D12PipelineState> q8_0_wide128_pipeline_;
  ComPtr<ID3D12PipelineState> q4_0_wide128_pipeline_;
  // Experiment: does widening further (256 rows/group) help more? Test-only,
  // not wired into the token graph unless proven better than 128.
  ComPtr<ID3D12PipelineState> q4k_wide256_pipeline_;
  // Barrier-free direct-input batch-one Q4_K variant (see
  // direct_input_enabled()).
  ComPtr<ID3D12PipelineState> q4k_direct_pipeline_;
  ComPtr<ID3D12PipelineState> q6k_direct_pipeline_;
  ComPtr<ID3D12PipelineState> q4k_pair_pipeline_;
  // B6 diagnostics (LIZARD_NATIVE_GPU_TIMESTAMPS=1): per-stage GPU timing
  // inside the token graph.
  ComPtr<ID3D12QueryHeap> timestamp_heap_;
  ComPtr<ID3D12Resource> timestamp_readback_;
  std::uint32_t timestamp_capacity_ = 0;
  std::vector<const char *> timestamp_labels_;
  ComPtr<ID3D12PipelineState> q4k_gemv_pipeline_;
  ComPtr<ID3D12PipelineState> q5k_b1_pipeline_;
  ComPtr<ID3D12PipelineState> q6k_b1_pipeline_;
  ComPtr<ID3D12PipelineState> q8_0_b1_pipeline_;
  ComPtr<ID3D12PipelineState> q4_0_b1_pipeline_;
  ComPtr<ID3D12PipelineState> scalar_b1_pipeline_;
  ComPtr<ID3D12PipelineState> dp4a_quantize_pipeline_;
  ComPtr<ID3D12PipelineState> dp4a_q4k_b1_pipeline_;
  ComPtr<ID3D12PipelineState> q4k_gemv_wave_pipeline_;
  D3d12Dp4aCapabilities dp4a_capabilities_;
  ComPtr<ID3D12Resource> dp4a_upload_;
  std::uint64_t dp4a_upload_bytes_ = 0;
  void *mapped_dp4a_upload_ = nullptr;
  ComPtr<ID3D12Resource> dp4a_activations_;
  std::uint64_t dp4a_activations_bytes_ = 0;
  D3D12_RESOURCE_STATES dp4a_activations_state_ = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
  ComPtr<ID3D12Resource> dp4a_output_;
  std::uint64_t dp4a_output_bytes_ = 0;
  D3D12_RESOURCE_STATES dp4a_output_state_ = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
  ComPtr<ID3D12Resource> dp4a_readback_;
  std::uint64_t dp4a_readback_bytes_ = 0;
  void *mapped_dp4a_readback_ = nullptr;
  // Static token-graph upload cache: identifies the model + upload layout the
  // persistent upload buffer currently holds, so immutable norm/config data
  // is not rewritten every decoded token. Null model pointer = invalid.
  const void *token_graph_upload_model_ = nullptr;
  std::uint64_t token_graph_upload_bytes_ = 0;
  bool token_graph_upload_had_kv_ = true;
  ComPtr<ID3D12PipelineState> gelu_pipeline_;
  ComPtr<ID3D12PipelineState> silu_glu_pipeline_;
  ComPtr<ID3D12PipelineState> q4k_loadonly_pipeline_;
  ComPtr<ID3D12PipelineState> q4k_loadonly_coalesced_pipeline_;
  ComPtr<ID3D12PipelineState> stream_read_pipeline_;
  ComPtr<ID3D12Resource> stream_read_buffer_;
  std::uint64_t stream_read_buffer_bytes_ = 0;
  ComPtr<ID3D12Resource> stream_read_output_;
  std::uint64_t stream_read_output_bytes_ = 0;
  ComPtr<ID3D12PipelineState> rms_norm_pipeline_;
  ComPtr<ID3D12PipelineState> rms_norm_dual_pipeline_;
  ComPtr<ID3D12PipelineState> q4k_ksplit_pipeline_;
  ComPtr<ID3D12PipelineState> ksplit_reduce_pipeline_;
  ComPtr<ID3D12PipelineState> gemma_prepare_head_pipeline_;
  ComPtr<ID3D12PipelineState> gemma_online_attention_pipeline_;
  ComPtr<ID3D12PipelineState> gemma_online_attention_tiled_pipeline_;
  ComPtr<ID3D12PipelineState> ple_prepare_pipeline_;
  ComPtr<ID3D12PipelineState> logit_softcap_pipeline_;
  ComPtr<ID3D12PipelineState> argmax_pipeline_;

  // Device-only intermediates for the fused FFN chain (gate/up results and
  // the gelu(gate)*up activations); contents never cross the PCIe/UMA bus.
  ComPtr<ID3D12Resource> ffn_scratch_;
  ComPtr<ID3D12Resource> ffn_activation_;
  std::uint64_t ffn_scratch_bytes_ = 0;
  std::uint64_t ffn_activation_bytes_ = 0;
  D3D12_RESOURCE_STATES ffn_scratch_state_ = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
  D3D12_RESOURCE_STATES ffn_activation_state_ = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;

  // Ping-pong vector scratch for the fused decode layer tail.
  ComPtr<ID3D12Resource> layer_scratch_a_;
  ComPtr<ID3D12Resource> layer_scratch_b_;
  std::uint64_t layer_scratch_a_bytes_ = 0;
  std::uint64_t layer_scratch_b_bytes_ = 0;
  D3D12_RESOURCE_STATES layer_scratch_a_state_ = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
  D3D12_RESOURCE_STATES layer_scratch_b_state_ = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;

  std::unordered_map<const void *, ResidentWeight> weight_cache_;
  std::unordered_map<std::uint64_t, DeviceKvCache> gemma_kv_cache_;
  std::vector<std::uint8_t> staging_scratch_;
  std::uint64_t weight_cache_bytes_ = 0;
  std::uint64_t weight_budget_bytes_ = 0;
  // True only while the load-time pre-stage loop is filling the cache. Decode
  // (prestaging_ == false) evicts LRU weights instead of failing at the budget.
  bool prestaging_ = false;
  // Live DXGI budget clamping (Phase 1 Finding 2). The env variable is a
  // REQUEST ceiling; the effective budget is clamped to the selected
  // adapter's current safe budget whenever IDXGIAdapter3 is available.
  ComPtr<IDXGIAdapter3> adapter3_;
  std::uint64_t requested_weight_budget_bytes_ = 0;
  bool live_budget_available_ = false;
  std::chrono::steady_clock::time_point last_budget_refresh_at_ {};
  std::uint64_t live_local_budget_bytes_ = 0;
  std::uint64_t live_local_usage_bytes_ = 0;
  std::uint64_t live_non_local_budget_bytes_ = 0;
  std::uint64_t live_non_local_usage_bytes_ = 0;
  std::uint64_t memory_reserve_bytes_ = 0;
  std::uint64_t use_counter_ = 0;
  D3d12RuntimeTelemetry telemetry_;

  ComPtr<ID3D12Resource> weight_upload_;
  ComPtr<ID3D12Resource> input_upload_;
  ComPtr<ID3D12Resource> legacy_weight_default_;
  ComPtr<ID3D12Resource> input_default_;
  ComPtr<ID3D12Resource> output_default_;
  ComPtr<ID3D12Resource> output_readback_;
  std::uint64_t weight_upload_bytes_ = 0;
  std::uint64_t input_upload_bytes_ = 0;
  std::uint64_t legacy_weight_default_bytes_ = 0;
  std::uint64_t input_default_bytes_ = 0;
  std::uint64_t output_default_bytes_ = 0;
  std::uint64_t output_readback_bytes_ = 0;
  D3D12_RESOURCE_STATES legacy_weight_state_ = D3D12_RESOURCE_STATE_COPY_DEST;
  D3D12_RESOURCE_STATES input_state_ = D3D12_RESOURCE_STATE_COPY_DEST;
  D3D12_RESOURCE_STATES output_state_ = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
  void *mapped_weight_upload_ = nullptr;
  void *mapped_input_upload_ = nullptr;
  void *mapped_output_readback_ = nullptr;
  ComPtr<ID3D12Resource> attention_input_;
  ComPtr<ID3D12Resource> attention_config_;
  ComPtr<ID3D12Resource> attention_qkv_;
  ComPtr<ID3D12Resource> attention_query_;
  ComPtr<ID3D12Resource> attention_context_;
  ComPtr<ID3D12Resource> attention_readback_;
  ComPtr<ID3D12Resource> ple_input_default_;
  ComPtr<ID3D12Resource> layer_tail_upload_;
  std::uint64_t attention_input_bytes_ = 0;
  std::uint64_t attention_config_bytes_ = 0;
  std::uint64_t attention_qkv_bytes_ = 0;
  std::uint64_t attention_query_bytes_ = 0;
  std::uint64_t attention_context_bytes_ = 0;
  std::uint64_t attention_readback_bytes_ = 0;
  std::uint64_t ple_input_default_bytes_ = 0;
  std::uint64_t layer_tail_upload_bytes_ = 0;
  D3D12_RESOURCE_STATES attention_input_state_ = D3D12_RESOURCE_STATE_COPY_DEST;
  D3D12_RESOURCE_STATES attention_config_state_ = D3D12_RESOURCE_STATE_COPY_DEST;
  D3D12_RESOURCE_STATES attention_qkv_state_ = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
  D3D12_RESOURCE_STATES attention_query_state_ = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
  D3D12_RESOURCE_STATES attention_context_state_ = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
  D3D12_RESOURCE_STATES ple_input_default_state_ = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
  void *mapped_attention_readback_ = nullptr;
  void *mapped_layer_tail_upload_ = nullptr;
  bool deferred_attention_active_ = false;
  std::uint64_t deferred_attention_cache_id_ = 0;
  bool deferred_attention_appends_kv_ = false;
  const void *last_input_identity_ = nullptr;
  std::uint64_t last_input_bytes_ = 0;
};

struct GlobalContextState {
  std::unique_ptr<D3d12MatmulContext> context;
  bool initialized = false;
  std::chrono::steady_clock::time_point last_reinit_at {};
};

GlobalContextState &global_context_state() {
  static GlobalContextState state;
  return state;
}

D3d12MatmulContext *existing_context() {
  return global_context_state().context.get();
}

D3d12MatmulContext *global_context() {
  GlobalContextState &state = global_context_state();

  // Recover from sporadic driver device removal: tear the dead context down
  // and rebuild it (resident weights re-stage lazily), with a cooldown so a
  // persistently failing device does not cause an init storm.
  if (state.context && state.context->device_removed()) {
    const auto now = std::chrono::steady_clock::now();
    if (now - state.last_reinit_at >= std::chrono::seconds(5)) {
      state.last_reinit_at = now;
      state.context.reset();
      state.initialized = false;
      trace_d3d12("global_context: reinitializing after device removal");
    }
  }

  if (!state.initialized) {
    state.initialized = true;
    state.context = std::make_unique<D3d12MatmulContext>();
    std::string error_message;
    if (!state.context->initialize(error_message)) {
      state.context.reset();
    }
  }
  return state.context.get();
}

D3d12BatchMatmulCapabilities &fallback_capabilities() {
  static D3d12BatchMatmulCapabilities capabilities = [] {
    D3d12BatchMatmulCapabilities value;
    value.supported = false;
    value.backend_name = "cpu";
    value.reason = "Windows-native Direct3D 12 compute is unavailable.";
    return value;
  }();
  return capabilities;
}

}  // namespace

const D3d12BatchMatmulCapabilities &d3d12_batch_matmul_capabilities() {
  D3d12MatmulContext *context = global_context();
  if (context != nullptr) {
    return context->capabilities();
  }
  return fallback_capabilities();
}

const D3d12BatchMatmulCapabilities *d3d12_batch_matmul_capabilities_if_initialized() {
  D3d12MatmulContext *context = existing_context();
  return context != nullptr ? &context->capabilities() : nullptr;
}

D3d12GpuMemoryReport d3d12_gpu_memory_report() {
  D3d12MatmulContext *context = existing_context();
  if (context == nullptr) {
    return D3d12GpuMemoryReport {};
  }
  return context->memory_report();
}

void d3d12_refresh_gpu_memory_budget() {
  D3d12MatmulContext *context = existing_context();
  if (context != nullptr) {
    context->refresh_weight_budget(true);
  }
}

const D3d12Dp4aCapabilities &d3d12_dp4a_capabilities() {
  D3d12MatmulContext *context = global_context();
  if (context != nullptr) {
    return context->dp4a_capabilities();
  }
  static const D3d12Dp4aCapabilities unavailable = {
    false,
    "Direct3D 12 device is unavailable on this host.",
  };
  return unavailable;
}

bool d3d12_dp4a_q4k_matvec(
  const D3d12ResidentWeightRef &op,
  const std::vector<float> &input,
  std::vector<float> *outputs,
  std::string *error_message) {
  D3d12MatmulContext *context = global_context();
  if (context == nullptr) {
    if (error_message != nullptr) {
      *error_message = fallback_capabilities().reason;
    }
    return false;
  }
  if (quad_interleave_enabled()) {
    if (error_message != nullptr) {
      *error_message = "DP4A Q4_K does not support the quad-interleaved layout.";
    }
    return false;
  }
  std::string local_error;
  const bool success = context->dp4a_q4k_matvec(op, input, outputs, local_error);
  if (!success && error_message != nullptr) {
    *error_message = local_error;
  }
  return success;
}

bool d3d12_coop_gemv_q4k_matvec(
  const D3d12ResidentWeightRef &op,
  const std::vector<float> &input,
  std::vector<float> *outputs,
  std::string *error_message) {
  D3d12MatmulContext *context = global_context();
  if (context == nullptr) {
    if (error_message != nullptr) {
      *error_message = fallback_capabilities().reason;
    }
    return false;
  }
  std::string local_error;
  const bool success = context->coop_gemv_q4k_matvec(op, input, outputs, local_error);
  if (!success && error_message != nullptr) {
    *error_message = local_error;
  }
  return success;
}

// Times the pure-FMA kernel so the achievable ALU rate can be measured instead
// of assumed. Returns milliseconds per iteration for `group_count` groups of 256
// threads each running `iterations` * 64 fused multiply-adds.
// Times the batched Q4_K matmul with an explicit kernel choice, for paired A/B
// measurement inside one process. See run_batched_ab for why paired matters here.
bool d3d12_batched_ab_benchmark(
  const D3d12ResidentWeightRef &op,
  const std::vector<float> &packed_inputs,
  std::uint32_t batch_count,
  int variant,
  int repeats,
  double *ms_per_iteration,
  std::string *error_message) {
  D3d12MatmulContext *context = global_context();
  if (context == nullptr || ms_per_iteration == nullptr) {
    if (error_message != nullptr) {
      *error_message = context == nullptr
        ? fallback_capabilities().reason
        : std::string("ms_per_iteration is required.");
    }
    return false;
  }
  std::string local_error;
  const bool ok = context->run_batched_ab(
    op, packed_inputs, batch_count, variant, repeats,
    ms_per_iteration, local_error);
  if (!ok && error_message != nullptr) {
    *error_message = local_error;
  }
  return ok;
}

bool d3d12_fma_peak_benchmark(
  std::uint32_t iterations,
  std::uint32_t group_count,
  int repeats,
  double *ms_per_iteration,
  std::string *error_message) {
  D3d12MatmulContext *context = global_context();
  if (context == nullptr || ms_per_iteration == nullptr) {
    if (error_message != nullptr) {
      *error_message = context == nullptr
        ? fallback_capabilities().reason
        : std::string("ms_per_iteration is required.");
    }
    return false;
  }
  std::string local_error;
  const bool ok = context->run_fma_peak(
    iterations, group_count, repeats, ms_per_iteration, local_error);
  if (!ok && error_message != nullptr) {
    *error_message = local_error;
  }
  return ok;
}

bool d3d12_stream_read_benchmark(
  std::uint64_t bytes,
  int iterations,
  std::uint32_t group_count,
  std::uint32_t row_quads,
  double *ms_per_iteration,
  std::string *error_message) {
  D3d12MatmulContext *context = global_context();
  if (context == nullptr || ms_per_iteration == nullptr) {
    if (error_message != nullptr) {
      *error_message = context == nullptr
        ? fallback_capabilities().reason
        : std::string("ms_per_iteration is required.");
    }
    return false;
  }
  std::string local_error;
  const bool ok = context->measure_stream_read(bytes, iterations, group_count, row_quads, ms_per_iteration, local_error);
  if (!ok && error_message != nullptr) {
    *error_message = local_error;
  }
  return ok;
}

bool d3d12_loadonly_q4k_matvec(
  const D3d12ResidentWeightRef &op,
  const std::vector<float> &input,
  std::vector<float> *outputs,
  std::string *error_message,
  bool coalesced) {
  D3d12MatmulContext *context = global_context();
  if (context == nullptr) {
    if (error_message != nullptr) *error_message = fallback_capabilities().reason;
    return false;
  }
  std::string local_error;
  const bool ok = context->loadonly_q4k_matvec(op, input, outputs, local_error, coalesced);
  if (!ok && error_message != nullptr) *error_message = local_error;
  return ok;
}

bool d3d12_wide128_q4k_matvec(
  const D3d12ResidentWeightRef &op,
  const std::vector<float> &input,
  std::vector<float> *outputs,
  std::string *error_message) {
  D3d12MatmulContext *context = global_context();
  if (context == nullptr) {
    if (error_message != nullptr) {
      *error_message = fallback_capabilities().reason;
    }
    return false;
  }
  std::string local_error;
  const bool success = context->wide128_q4k_matvec(op, input, outputs, local_error);
  if (!success && error_message != nullptr) {
    *error_message = local_error;
  }
  return success;
}

bool d3d12_wide256_q4k_matvec(
  const D3d12ResidentWeightRef &op,
  const std::vector<float> &input,
  std::vector<float> *outputs,
  std::string *error_message) {
  D3d12MatmulContext *context = global_context();
  if (context == nullptr) {
    if (error_message != nullptr) {
      *error_message = fallback_capabilities().reason;
    }
    return false;
  }
  std::string local_error;
  const bool success = context->wide256_q4k_matvec(op, input, outputs, local_error);
  if (!success && error_message != nullptr) {
    *error_message = local_error;
  }
  return success;
}

bool d3d12_stream_read(
  const D3d12ResidentWeightRef &op,
  std::uint32_t group_count,
  std::uint64_t *checksum,
  std::string *error_message) {
  D3d12MatmulContext *context = global_context();
  if (context == nullptr) {
    if (error_message != nullptr) {
      *error_message = fallback_capabilities().reason;
    }
    return false;
  }
  std::string local_error;
  const bool success = context->stream_read(op, group_count, checksum, local_error);
  if (!success && error_message != nullptr) {
    *error_message = local_error;
  }
  return success;
}

bool d3d12_wave_gemv_q4k_matvec(
  const D3d12ResidentWeightRef &op,
  const std::vector<float> &input,
  std::vector<float> *outputs,
  std::string *error_message) {
  D3d12MatmulContext *context = global_context();
  if (context == nullptr) {
    if (error_message != nullptr) {
      *error_message = fallback_capabilities().reason;
    }
    return false;
  }
  std::string local_error;
  const bool success = context->wave_gemv_q4k_matvec(op, input, outputs, local_error);
  if (!success && error_message != nullptr) {
    *error_message = local_error;
  }
  return success;
}

const D3d12RuntimeTelemetry &d3d12_runtime_telemetry() {
  D3d12MatmulContext *context = existing_context();
  if (context != nullptr) {
    return context->telemetry();
  }
  static const D3d12RuntimeTelemetry telemetry;
  return telemetry;
}

void d3d12_reset_runtime_telemetry() {
  D3d12MatmulContext *context = existing_context();
  if (context != nullptr) {
    context->reset_telemetry();
  }
}

bool d3d12_weight_format_supported(
  std::uint32_t ggml_type,
  std::uint64_t row_width,
  D3d12WeightFormat *format) {
  D3d12WeightFormat resolved;
  switch (ggml_type) {
    case 0:
      resolved = D3d12WeightFormat::F32;
      break;
    case 1:
      resolved = D3d12WeightFormat::F16;
      break;
    case 30:
      resolved = D3d12WeightFormat::BF16;
      break;
    case 12:
      resolved = D3d12WeightFormat::Q4_K;
      break;
    case 13:
      resolved = D3d12WeightFormat::Q5_K;
      break;
    case 14:
      resolved = D3d12WeightFormat::Q6_K;
      break;
    case 8:
      resolved = D3d12WeightFormat::Q8_0;
      break;
    case 2:
      resolved = D3d12WeightFormat::Q4_0;
      break;
    default:
      return false;
  }
  if ((resolved == D3d12WeightFormat::Q4_K
      || resolved == D3d12WeightFormat::Q5_K
      || resolved == D3d12WeightFormat::Q6_K)
    && (row_width % k_quant_block_elements) != 0) {
    return false;
  }
  if ((resolved == D3d12WeightFormat::Q8_0 || resolved == D3d12WeightFormat::Q4_0)
    && (row_width % k_legacy_block_elements) != 0) {
    return false;
  }
  if (format != nullptr) {
    *format = resolved;
  }
  return true;
}

bool d3d12_batch_matmul_resident(
  const D3d12ResidentWeightRef *ops,
  std::size_t op_count,
  const std::vector<float> &packed_inputs,
  std::uint32_t batch_count,
  const void *input_identity,
  std::vector<float> *outputs,
  std::string *error_message) {
  D3d12MatmulContext *context = global_context();
  if (context == nullptr) {
    if (error_message != nullptr) {
      *error_message = fallback_capabilities().reason;
    }
    return false;
  }
  std::string local_error;
  const bool success = context->compute_resident(
    ops,
    op_count,
    packed_inputs,
    batch_count,
    input_identity,
    outputs,
    local_error);
  if (!success && error_message != nullptr) {
    *error_message = local_error;
  }
  return success;
}

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
  std::string *error_message) {
  D3d12MatmulContext *context = global_context();
  if (context == nullptr) {
    if (error_message != nullptr) {
      *error_message = fallback_capabilities().reason;
    }
    return false;
  }
  std::string local_error;
  const bool success = context->compute_gemma_token(
    initial_hidden,
    layers,
    ple_inputs,
    output_norm,
    output,
    output_rms_epsilon,
    final_logit_softcap,
    hidden_out,
    logits_out,
    argmax_token,
    local_error);
  if (!success && error_message != nullptr) {
    *error_message = local_error;
  }
  return success;
}

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
  std::string *error_message) {
  D3d12MatmulContext *context = global_context();
  if (context == nullptr) {
    if (error_message != nullptr) {
      *error_message = fallback_capabilities().reason;
    }
    return false;
  }
  std::string local_error;
  const bool success = context->compute_gemma_attention(
    query,
    key,
    value,
    normed_hidden,
    query_norm,
    key_norm,
    rope_freq_factors,
    initial_keys,
    initial_values,
    config,
    attention_context,
    local_error);
  if (!success && error_message != nullptr) {
    *error_message = local_error;
  }
  return success;
}

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
  std::string *error_message) {
  D3d12MatmulContext *context = global_context();
  if (context == nullptr) {
    if (error_message != nullptr) {
      *error_message = fallback_capabilities().reason;
    }
    return false;
  }
  std::string local_error;
  const bool success = context->compute_gemma_layer(
    query,
    key,
    value,
    attn_out,
    gate,
    up,
    down,
    ple_gate,
    ple_projection,
    normed_hidden,
    hidden_state,
    query_norm,
    key_norm,
    rope_freq_factors,
    initial_keys,
    initial_values,
    post_attention_norm,
    ffn_norm,
    post_ffw_norm,
    ple_input,
    ple_post_norm,
    config,
    hidden_out,
    local_error);
  if (!success && error_message != nullptr) {
    *error_message = local_error;
  }
  return success;
}

bool d3d12_ffn_gelu_glu_resident(
  const D3d12ResidentWeightRef &gate,
  const D3d12ResidentWeightRef &up,
  const D3d12ResidentWeightRef &down,
  const std::vector<float> &packed_inputs,
  std::uint32_t batch_count,
  const void *input_identity,
  std::vector<float> *outputs,
  std::string *error_message) {
  D3d12MatmulContext *context = global_context();
  if (context == nullptr) {
    if (error_message != nullptr) {
      *error_message = fallback_capabilities().reason;
    }
    return false;
  }
  std::string local_error;
  const bool success = context->compute_ffn(
    gate,
    up,
    down,
    packed_inputs,
    batch_count,
    input_identity,
    false,
    outputs,
    local_error);
  if (!success && error_message != nullptr) {
    *error_message = local_error;
  }
  return success;
}

bool d3d12_ffn_silu_glu_resident(
  const D3d12ResidentWeightRef &gate,
  const D3d12ResidentWeightRef &up,
  const D3d12ResidentWeightRef &down,
  const std::vector<float> &packed_inputs,
  std::uint32_t batch_count,
  const void *input_identity,
  std::vector<float> *outputs,
  std::string *error_message) {
  D3d12MatmulContext *context = global_context();
  if (context == nullptr) {
    if (error_message != nullptr) {
      *error_message = fallback_capabilities().reason;
    }
    return false;
  }
  std::string local_error;
  const bool success = context->compute_ffn(
    gate,
    up,
    down,
    packed_inputs,
    batch_count,
    input_identity,
    true,
    outputs,
    local_error);
  if (!success && error_message != nullptr) {
    *error_message = local_error;
  }
  return success;
}

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
  std::string *error_message) {
  D3d12MatmulContext *context = global_context();
  if (context == nullptr) {
    if (error_message != nullptr) {
      *error_message = fallback_capabilities().reason;
    }
    return false;
  }
  static const std::vector<float> k_no_norm;
  std::string local_error;
  const bool success = context->compute_layer_tail(
    attn_out,
    gate,
    up,
    down,
    nullptr,
    nullptr,
    attention_context,
    hidden_state,
    k_no_norm,
    ffn_norm,
    k_no_norm,
    nullptr,
    nullptr,
    rms_epsilon,
    hidden_out,
    local_error,
    false,
    true,   // skip post-attention norm: Llama adds the residual directly
    true,   // skip post-FFW norm: same
    true);  // SwiGLU
  if (!success && error_message != nullptr) {
    *error_message = local_error;
  }
  return success;
}

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
  std::string *error_message) {
  D3d12MatmulContext *context = global_context();
  if (context == nullptr) {
    if (error_message != nullptr) {
      *error_message = fallback_capabilities().reason;
    }
    return false;
  }
  std::string local_error;
  const bool success = context->compute_llama_layer_tail_batch(
    attn_out,
    gate,
    up,
    down,
    packed_attention_contexts,
    packed_hidden_states,
    ffn_norm,
    batch_count,
    rms_epsilon,
    packed_hidden_out,
    local_error);
  if (!success && error_message != nullptr) {
    *error_message = local_error;
  }
  return success;
}

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
  std::string *error_message) {
  D3d12MatmulContext *context = global_context();
  if (context == nullptr) {
    if (error_message != nullptr) {
      *error_message = fallback_capabilities().reason;
    }
    return false;
  }
  std::string local_error;
  const bool success = context->compute_layer_tail(
    attn_out,
    gate,
    up,
    down,
    nullptr,
    nullptr,
    attention_context,
    hidden_state,
    post_attention_norm,
    ffn_norm,
    post_ffw_norm,
    nullptr,
    nullptr,
    rms_epsilon,
    hidden_out,
    local_error);
  if (!success && error_message != nullptr) {
    *error_message = local_error;
  }
  return success;
}

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
  std::string *error_message) {
  D3d12MatmulContext *context = global_context();
  if (context == nullptr) {
    if (error_message != nullptr) {
      *error_message = fallback_capabilities().reason;
    }
    return false;
  }
  std::string local_error;
  const bool success = context->compute_layer_tail(
    attn_out,
    gate,
    up,
    down,
    &ple_gate,
    &ple_projection,
    attention_context,
    hidden_state,
    post_attention_norm,
    ffn_norm,
    post_ffw_norm,
    &ple_input,
    &ple_post_norm,
    rms_epsilon,
    hidden_out,
    local_error);
  if (!success && error_message != nullptr) {
    *error_message = local_error;
  }
  return success;
}

void d3d12_reset_resident_weights() {
  D3d12MatmulContext *context = existing_context();
  if (context != nullptr) {
    context->reset_resident_weights();
  }
}

std::uint64_t d3d12_resident_weight_bytes() {
  D3d12MatmulContext *context = existing_context();
  return context != nullptr ? context->resident_weight_bytes() : 0;
}

// Marks the load-time pre-stage window. Inside it, staging that would exceed the
// weight budget simply stops (the remaining tensors stage lazily on first use)
// instead of evicting what was just staged. Outside it (decode), staging evicts.
void d3d12_set_prestaging(bool active) {
  D3d12MatmulContext *context = existing_context();
  if (context != nullptr) {
    context->set_prestaging(active);
  }
}

bool d3d12_stage_resident_weight(const D3d12ResidentWeightRef &op, std::string *error_message) {
  D3d12MatmulContext *context = global_context();
  if (context == nullptr) {
    if (error_message != nullptr) {
      *error_message = fallback_capabilities().reason;
    }
    return false;
  }
  std::string local_error;
  const bool success = context->stage_resident(op, local_error);
  if (!success && error_message != nullptr) {
    *error_message = local_error;
  }
  return success;
}

void d3d12_shrink_staging_buffers() {
  D3d12MatmulContext *context = existing_context();
  if (context != nullptr) {
    context->shrink_staging_buffers();
  }
}

bool d3d12_batch_matmul(
  const std::vector<float> &weights,
  std::uint32_t row_count,
  std::uint32_t row_width,
  const std::vector<float> &packed_inputs,
  std::uint32_t batch_count,
  const void *input_identity,
  std::vector<std::vector<float>> &outputs,
  std::string *error_message) {
  D3d12MatmulContext *context = global_context();
  if (context == nullptr) {
    if (error_message != nullptr) {
      *error_message = fallback_capabilities().reason;
    }
    return false;
  }
  std::string local_error;
  const bool success = context->compute_legacy(
    weights,
    row_count,
    row_width,
    packed_inputs,
    batch_count,
    input_identity,
    outputs,
    local_error);
  if (!success && error_message != nullptr) {
    *error_message = local_error;
  }
  return success;
}

namespace {

std::uint32_t selftest_rng(std::uint32_t &state) {
  state ^= state << 13;
  state ^= state >> 17;
  state ^= state << 5;
  return state;
}

float selftest_half_to_float(std::uint16_t half_bits) {
  const std::uint32_t sign = (half_bits & 0x8000u) << 16;
  const std::uint32_t exponent = (half_bits >> 10) & 0x1Fu;
  const std::uint32_t mantissa = half_bits & 0x3FFu;
  std::uint32_t bits;
  if (exponent == 0) {
    if (mantissa == 0) {
      bits = sign;
    } else {
      // Subnormal halves are irrelevant for synthetic scales; flush to zero.
      bits = sign;
    }
  } else if (exponent == 31) {
    bits = sign | 0x7F800000u | (mantissa << 13);
  } else {
    bits = sign | ((exponent + 112u) << 23) | (mantissa << 13);
  }
  float value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

// CPU reference dot for one row of row-major raw weights in any GPU format.
double selftest_reference_row_dot(
  D3d12WeightFormat format,
  const std::vector<std::uint8_t> &weights,
  std::uint64_t row,
  std::uint64_t row_width,
  const float *input) {
  double sum = 0.0;
  if (format == D3d12WeightFormat::F32) {
    const float *row_data = reinterpret_cast<const float *>(weights.data()) + row * row_width;
    for (std::uint64_t element = 0; element < row_width; element += 1) {
      sum += static_cast<double>(row_data[element]) * input[element];
    }
    return sum;
  }
  if (format == D3d12WeightFormat::F16 || format == D3d12WeightFormat::BF16) {
    const std::uint16_t *row_data = reinterpret_cast<const std::uint16_t *>(weights.data()) + row * row_width;
    for (std::uint64_t element = 0; element < row_width; element += 1) {
      float value;
      if (format == D3d12WeightFormat::F16) {
        value = selftest_half_to_float(row_data[element]);
      } else {
        const std::uint32_t bits = static_cast<std::uint32_t>(row_data[element]) << 16;
        std::memcpy(&value, &bits, sizeof(value));
      }
      sum += static_cast<double>(value) * input[element];
    }
    return sum;
  }

  if (format == D3d12WeightFormat::Q8_0) {
    const std::uint64_t blocks_per_row = row_width / k_legacy_block_elements;
    for (std::uint64_t block = 0; block < blocks_per_row; block += 1) {
      const std::uint8_t *block_ptr = weights.data() + (row * blocks_per_row + block) * k_q8_0_block_bytes;
      const float *block_input = input + block * k_legacy_block_elements;
      std::uint16_t d_bits;
      std::memcpy(&d_bits, block_ptr, sizeof(d_bits));
      const float d = selftest_half_to_float(d_bits);
      const std::int8_t *qs = reinterpret_cast<const std::int8_t *>(block_ptr + 2);
      for (std::uint64_t k = 0; k < k_legacy_block_elements; k += 1) {
        sum += static_cast<double>(d * static_cast<float>(qs[k])) * block_input[k];
      }
    }
    return sum;
  }

  if (format == D3d12WeightFormat::Q4_0) {
    const std::uint64_t blocks_per_row = row_width / k_legacy_block_elements;
    for (std::uint64_t block = 0; block < blocks_per_row; block += 1) {
      const std::uint8_t *block_ptr = weights.data() + (row * blocks_per_row + block) * k_q4_0_block_bytes;
      const float *block_input = input + block * k_legacy_block_elements;
      std::uint16_t d_bits;
      std::memcpy(&d_bits, block_ptr, sizeof(d_bits));
      const float d = selftest_half_to_float(d_bits);
      const std::uint8_t *qs = block_ptr + 2;
      for (std::uint64_t j = 0; j < k_legacy_block_elements / 2; j += 1) {
        const int x0 = static_cast<int>(qs[j] & 0x0F) - 8;
        const int x1 = static_cast<int>(qs[j] >> 4) - 8;
        sum += static_cast<double>(d * static_cast<float>(x0)) * block_input[j];
        sum += static_cast<double>(d * static_cast<float>(x1)) * block_input[j + 16];
      }
    }
    return sum;
  }

  const auto get_scale_min_k4_ref = [](int j, const std::uint8_t *packed, std::uint8_t *scale, std::uint8_t *minimum) {
    if (j < 4) {
      *scale = packed[j] & 63;
      *minimum = packed[j + 4] & 63;
    } else {
      *scale = static_cast<std::uint8_t>((packed[j + 4] & 0x0F) | ((packed[j - 4] >> 6) << 4));
      *minimum = static_cast<std::uint8_t>((packed[j + 4] >> 4) | ((packed[j] >> 6) << 4));
    }
  };

  if (format == D3d12WeightFormat::Q5_K) {
    const std::uint64_t blocks_per_row_q5 = row_width / k_quant_block_elements;
    for (std::uint64_t block = 0; block < blocks_per_row_q5; block += 1) {
      const std::uint8_t *block_ptr = weights.data() + (row * blocks_per_row_q5 + block) * k_q5_k_block_bytes;
      const float *block_input = input + block * k_quant_block_elements;
      std::uint16_t d_bits;
      std::uint16_t dmin_bits;
      std::memcpy(&d_bits, block_ptr, sizeof(d_bits));
      std::memcpy(&dmin_bits, block_ptr + 2, sizeof(dmin_bits));
      const float d = selftest_half_to_float(d_bits);
      const float dmin = selftest_half_to_float(dmin_bits);
      const std::uint8_t *packed_scales = block_ptr + 4;
      const std::uint8_t *qh = block_ptr + 16;
      const std::uint8_t *qs = block_ptr + 48;
      int scale_index = 0;
      std::uint8_t u1 = 1;
      std::uint8_t u2 = 2;
      for (std::uint64_t j = 0; j < k_quant_block_elements; j += 64) {
        std::uint8_t sc0 = 0;
        std::uint8_t m0 = 0;
        std::uint8_t sc1 = 0;
        std::uint8_t m1 = 0;
        get_scale_min_k4_ref(scale_index + 0, packed_scales, &sc0, &m0);
        get_scale_min_k4_ref(scale_index + 1, packed_scales, &sc1, &m1);
        const float d0 = d * static_cast<float>(sc0);
        const float mm0 = dmin * static_cast<float>(m0);
        const float d1 = d * static_cast<float>(sc1);
        const float mm1 = dmin * static_cast<float>(m1);
        const std::uint8_t *ql = qs + (j / 64) * 32;
        for (std::uint64_t l = 0; l < 32; l += 1) {
          const float w_low = d0 * static_cast<float>((ql[l] & 0x0F) + ((qh[l] & u1) ? 16 : 0)) - mm0;
          const float w_high = d1 * static_cast<float>((ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0)) - mm1;
          sum += static_cast<double>(w_low) * block_input[j + l];
          sum += static_cast<double>(w_high) * block_input[j + l + 32];
        }
        scale_index += 2;
        u1 = static_cast<std::uint8_t>(u1 << 2);
        u2 = static_cast<std::uint8_t>(u2 << 2);
      }
    }
    return sum;
  }

  const std::uint64_t blocks_per_row = row_width / k_quant_block_elements;
  const std::uint64_t block_bytes = format == D3d12WeightFormat::Q4_K
    ? k_q4_k_block_bytes
    : k_q6_k_block_bytes;
  const auto get_scale_min_k4 = [](int j, const std::uint8_t *packed, std::uint8_t *scale, std::uint8_t *minimum) {
    if (j < 4) {
      *scale = packed[j] & 63;
      *minimum = packed[j + 4] & 63;
    } else {
      *scale = static_cast<std::uint8_t>((packed[j + 4] & 0x0F) | ((packed[j - 4] >> 6) << 4));
      *minimum = static_cast<std::uint8_t>((packed[j + 4] >> 4) | ((packed[j] >> 6) << 4));
    }
  };

  for (std::uint64_t block = 0; block < blocks_per_row; block += 1) {
    const std::uint8_t *block_ptr = weights.data() + (row * blocks_per_row + block) * block_bytes;
    const float *block_input = input + block * k_quant_block_elements;
    if (format == D3d12WeightFormat::Q4_K) {
      std::uint16_t d_bits;
      std::uint16_t dmin_bits;
      std::memcpy(&d_bits, block_ptr, sizeof(d_bits));
      std::memcpy(&dmin_bits, block_ptr + 2, sizeof(dmin_bits));
      const float d = selftest_half_to_float(d_bits);
      const float dmin = selftest_half_to_float(dmin_bits);
      const std::uint8_t *packed_scales = block_ptr + 4;
      const std::uint8_t *quants = packed_scales + 12;
      int packed_index = 0;
      for (std::uint64_t chunk = 0; chunk < k_quant_block_elements; chunk += 64) {
        std::uint8_t scale0 = 0;
        std::uint8_t min0 = 0;
        std::uint8_t scale1 = 0;
        std::uint8_t min1 = 0;
        get_scale_min_k4(packed_index + 0, packed_scales, &scale0, &min0);
        get_scale_min_k4(packed_index + 1, packed_scales, &scale1, &min1);
        const float d0 = d * static_cast<float>(scale0);
        const float m0 = dmin * static_cast<float>(min0);
        const float d1 = d * static_cast<float>(scale1);
        const float m1 = dmin * static_cast<float>(min1);
        const std::uint8_t *chunk_ptr = quants + (chunk / 2);
        for (std::uint64_t item = 0; item < 32; item += 1) {
          const std::uint8_t packed = chunk_ptr[item];
          sum += static_cast<double>(d0 * static_cast<float>(packed & 0x0F) - m0)
            * block_input[chunk + item];
          sum += static_cast<double>(d1 * static_cast<float>(packed >> 4) - m1)
            * block_input[chunk + item + 32];
        }
        packed_index += 2;
      }
    } else {
      const std::uint8_t *ql = block_ptr;
      const std::uint8_t *qh = ql + (k_quant_block_elements / 2);
      const std::int8_t *scales = reinterpret_cast<const std::int8_t *>(qh + (k_quant_block_elements / 4));
      std::uint16_t d_bits;
      std::memcpy(&d_bits, block_ptr + k_q6_k_block_bytes - 2, sizeof(d_bits));
      const float d = selftest_half_to_float(d_bits);
      for (std::uint64_t chunk = 0; chunk < k_quant_block_elements; chunk += 128) {
        const std::uint8_t *chunk_ql = ql + (chunk / 2);
        const std::uint8_t *chunk_qh = qh + (chunk / 4);
        const std::int8_t *chunk_scales = scales + (chunk / 16);
        for (std::uint64_t item = 0; item < 32; item += 1) {
          const int scale_index = static_cast<int>(item / 16);
          const int q1 = static_cast<int>(
            static_cast<std::int8_t>((chunk_ql[item] & 0x0F) | (((chunk_qh[item] >> 0) & 0x03) << 4)) - 32);
          const int q2 = static_cast<int>(
            static_cast<std::int8_t>((chunk_ql[item + 32] & 0x0F) | (((chunk_qh[item] >> 2) & 0x03) << 4)) - 32);
          const int q3 = static_cast<int>(
            static_cast<std::int8_t>((chunk_ql[item] >> 4) | (((chunk_qh[item] >> 4) & 0x03) << 4)) - 32);
          const int q4 = static_cast<int>(
            static_cast<std::int8_t>((chunk_ql[item + 32] >> 4) | (((chunk_qh[item] >> 6) & 0x03) << 4)) - 32);
          sum += static_cast<double>(d * chunk_scales[scale_index + 0] * q1) * block_input[chunk + item];
          sum += static_cast<double>(d * chunk_scales[scale_index + 2] * q2) * block_input[chunk + item + 32];
          sum += static_cast<double>(d * chunk_scales[scale_index + 4] * q3) * block_input[chunk + item + 64];
          sum += static_cast<double>(d * chunk_scales[scale_index + 6] * q4) * block_input[chunk + item + 96];
        }
      }
    }
  }
  return sum;
}

bool run_one_selftest(
  const char *label,
  D3d12WeightFormat format,
  std::uint32_t row_count,
  std::uint32_t row_width,
  std::uint32_t batch_count,
  int q4k_scale_variant = 0) {
  // Weight vectors are freed between cases, so the heap can hand the next
  // case the same base address and alias a stale resident-cache key.
  d3d12_reset_resident_weights();
  std::uint32_t rng = 0x12345678u
    ^ (row_count * 2654435761u)
    ^ (row_width * 40503u)
    ^ static_cast<std::uint32_t>(format);

  const std::uint64_t elements = static_cast<std::uint64_t>(row_count) * row_width;
  std::uint64_t weight_bytes = 0;
  switch (format) {
    case D3d12WeightFormat::F32:
      weight_bytes = elements * 4;
      break;
    case D3d12WeightFormat::F16:
    case D3d12WeightFormat::BF16:
      weight_bytes = elements * 2;
      break;
    case D3d12WeightFormat::Q4_K:
      weight_bytes = elements / k_quant_block_elements * k_q4_k_block_bytes;
      break;
    case D3d12WeightFormat::Q5_K:
      weight_bytes = elements / k_quant_block_elements * k_q5_k_block_bytes;
      break;
    case D3d12WeightFormat::Q6_K:
      weight_bytes = elements / k_quant_block_elements * k_q6_k_block_bytes;
      break;
    case D3d12WeightFormat::Q8_0:
      weight_bytes = elements / k_legacy_block_elements * k_q8_0_block_bytes;
      break;
    case D3d12WeightFormat::Q4_0:
      weight_bytes = elements / k_legacy_block_elements * k_q4_0_block_bytes;
      break;
  }

  std::vector<std::uint8_t> weights(static_cast<std::size_t>(weight_bytes));
  if (format == D3d12WeightFormat::F32) {
    float *values = reinterpret_cast<float *>(weights.data());
    for (std::uint64_t index = 0; index < elements; index += 1) {
      values[index] = static_cast<float>(static_cast<std::int32_t>(selftest_rng(rng) % 2001) - 1000) / 1000.0f;
    }
  } else if (format == D3d12WeightFormat::F16) {
    std::uint16_t *values = reinterpret_cast<std::uint16_t *>(weights.data());
    for (std::uint64_t index = 0; index < elements; index += 1) {
      // Normal-range halves only: exponent 11..20, random sign/mantissa.
      const std::uint32_t random = selftest_rng(rng);
      values[index] = static_cast<std::uint16_t>(
        ((random & 1u) << 15) | (((11u + (random >> 1) % 10u) & 0x1Fu) << 10) | ((random >> 8) & 0x3FFu));
    }
  } else if (format == D3d12WeightFormat::BF16) {
    std::uint16_t *values = reinterpret_cast<std::uint16_t *>(weights.data());
    for (std::uint64_t index = 0; index < elements; index += 1) {
      // bf16 with exponent near 127 so magnitudes stay ~[0.5, 2).
      const std::uint32_t random = selftest_rng(rng);
      values[index] = static_cast<std::uint16_t>(
        ((random & 1u) << 15) | ((125u + ((random >> 1) % 3u)) << 7) | ((random >> 8) & 0x7Fu));
    }
  } else {
    for (std::uint64_t index = 0; index < weight_bytes; index += 1) {
      weights[index] = static_cast<std::uint8_t>(selftest_rng(rng) & 0xFFu);
    }
    // Overwrite the fp16 block scales with small exact values so reference
    // and shader math stay in a well-conditioned range.
    if (format == D3d12WeightFormat::Q8_0 || format == D3d12WeightFormat::Q4_0) {
      // Legacy 32-element blocks carry only an fp16 scale header; overwrite it
      // per block and leave the random int8/nibble payload intact.
      const std::uint64_t legacy_block_bytes = format == D3d12WeightFormat::Q8_0
        ? k_q8_0_block_bytes
        : k_q4_0_block_bytes;
      const std::uint64_t legacy_block_count = weight_bytes / legacy_block_bytes;
      const std::uint16_t legacy_d_bits = format == D3d12WeightFormat::Q8_0 ? 0x2066 : 0x2E66;
      for (std::uint64_t block = 0; block < legacy_block_count; block += 1) {
        std::memcpy(weights.data() + block * legacy_block_bytes, &legacy_d_bits, sizeof(legacy_d_bits));
      }
    } else {
    const std::uint64_t block_bytes = format == D3d12WeightFormat::Q4_K
      ? k_q4_k_block_bytes
      : (format == D3d12WeightFormat::Q5_K ? k_q5_k_block_bytes : k_q6_k_block_bytes);
    const std::uint64_t block_count = weight_bytes / block_bytes;
    for (std::uint64_t block = 0; block < block_count; block += 1) {
      std::uint8_t *block_ptr = weights.data() + block * block_bytes;
      const std::uint16_t d_bits = 0x2E66;  // ~0.1 in fp16
      const std::uint16_t dmin_bits = 0x2A66;  // ~0.05 in fp16
      if (format == D3d12WeightFormat::Q5_K) {
        // Q5_K shares Q4_K's d/dmin header; keep well-conditioned scales and
        // leave the random qh/qs quant bytes untouched.
        std::memcpy(block_ptr, &d_bits, sizeof(d_bits));
        std::memcpy(block_ptr + 2, &dmin_bits, sizeof(dmin_bits));
      } else if (format == D3d12WeightFormat::Q4_K) {
        const std::uint16_t zero_bits = 0;
        std::memcpy(block_ptr, q4k_scale_variant == 4 ? &zero_bits : &d_bits, sizeof(d_bits));
        std::memcpy(block_ptr + 2, q4k_scale_variant == 3 ? &zero_bits : &dmin_bits, sizeof(dmin_bits));
        std::uint8_t *packed_scales = block_ptr + 4;
        if (q4k_scale_variant == 1) {
          // Keep only j<4 scale/min data: zero bytes 8..11 and the high two
          // bits of bytes 0..7 so chunks 2 and 3 dequantize to exactly zero.
          for (int index = 0; index < 8; index += 1) {
            packed_scales[index] &= 0x3F;
          }
          for (int index = 8; index < 12; index += 1) {
            packed_scales[index] = 0;
          }
        } else if (q4k_scale_variant == 2) {
          // Keep only j>=4 scale/min data: keep the high two bits of bytes
          // 0..7 and bytes 8..11 so chunks 0 and 1 dequantize to zero.
          for (int index = 0; index < 8; index += 1) {
            packed_scales[index] &= 0xC0;
          }
        }
      } else {
        std::memcpy(block_ptr + k_q6_k_block_bytes - 2, &d_bits, sizeof(d_bits));
      }
    }
    }
  }

  std::vector<float> packed_inputs(static_cast<std::size_t>(batch_count) * row_width);
  for (std::size_t index = 0; index < packed_inputs.size(); index += 1) {
    packed_inputs[index] = static_cast<float>(static_cast<std::int32_t>(selftest_rng(rng) % 2001) - 1000) / 1000.0f;
  }
  // Diagnostic: make every activation exactly representable in Q8_1, so the
  // integer prefill path's quantisation becomes lossless and any error it still
  // shows is its own bug rather than the approximation. Each 32-element block
  // gets a 1.0 so absmax is exactly 1 and d is exactly 1/127; every other value
  // is k/127, so value/d == k with no rounding. Off unless asked for.
  if (selftest_exact_q8_1_inputs()) {
    for (std::size_t index = 0; index < packed_inputs.size(); index += 1) {
      if (index % 32u == 0) {
        packed_inputs[index] = 1.0f;
        continue;
      }
      const std::int32_t k =
        static_cast<std::int32_t>(selftest_rng(rng) % 255u) - 127;
      packed_inputs[index] = static_cast<float>(k) / 127.0f;
    }
  }

  D3d12ResidentWeightRef op;
  op.weight_key = weights.data();
  op.format = format;
  op.weight_data = weights.data();
  op.weight_bytes = static_cast<std::size_t>(weight_bytes);
  op.row_count = row_count;
  op.row_width = row_width;

  std::vector<float> outputs;
  std::string error_message;
  // Heap reuse can hand consecutive tests the same input address and byte
  // size, so a raw pointer identity would skip the input upload. Use a
  // process-unique counter instead.
  static std::uintptr_t selftest_input_epoch = 0;
  selftest_input_epoch += 1;
  const auto started_at = std::chrono::steady_clock::now();
  const bool success = d3d12_batch_matmul_resident(
    &op,
    1,
    packed_inputs,
    batch_count,
    reinterpret_cast<const void *>(selftest_input_epoch),
    &outputs,
    &error_message);
  const auto finished_at = std::chrono::steady_clock::now();
  const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(finished_at - started_at).count();

  if (!success) {
    std::printf("[FAIL] %s: %s\n", label, error_message.c_str());
    return false;
  }

  // Sample entries across the output and compare with CPU reference dots.
  double max_difference = 0.0;
  std::uint32_t sample_rng = 0x9E3779B9u;
  for (int sample = 0; sample < 64; sample += 1) {
    const std::uint64_t row = selftest_rng(sample_rng) % row_count;
    const std::uint32_t batch = selftest_rng(sample_rng) % batch_count;
    const double reference = selftest_reference_row_dot(
      format,
      weights,
      row,
      row_width,
      packed_inputs.data() + static_cast<std::size_t>(batch) * row_width);
    const double actual = outputs[static_cast<std::size_t>(batch) * row_count + row];
    const double difference = std::abs(actual - reference);
    const double tolerance = 1e-2 * std::max(1.0, std::abs(reference));
    if (difference > max_difference) {
      max_difference = difference;
    }
    if (difference > tolerance) {
      std::printf(
        "[FAIL] %s: row=%llu batch=%u expected=%f actual=%f\n",
        label,
        static_cast<unsigned long long>(row),
        batch,
        reference,
        actual);
      for (std::uint32_t probe_batch = 0; probe_batch < batch_count; probe_batch += 1) {
        const double batch_reference = selftest_reference_row_dot(
          format,
          weights,
          row,
          row_width,
          packed_inputs.data() + static_cast<std::size_t>(probe_batch) * row_width);
        std::printf(
          "  row %llu vs batch %u inputs: reference %f, gpu[batch=%u]=%f\n",
          static_cast<unsigned long long>(row),
          probe_batch,
          batch_reference,
          probe_batch,
          outputs[static_cast<std::size_t>(probe_batch) * row_count + row]);
      }
      return false;
    }
  }

  std::printf("[ ok ] %s: %lld ms, max sample diff %g\n", label, static_cast<long long>(elapsed_ms), max_difference);
  return true;
}

// Verifies the fused gate/up -> gelu -> down chain against a CPU reference.
bool run_ffn_chain_selftest(
  const char *label,
  std::uint32_t width,
  std::uint32_t hidden_rows,
  std::uint32_t batch_count,
  std::uint32_t verified_rows,
  bool interleave_matmul = false,
  float input_scale = 1.0f,
  bool use_silu = false) {
  d3d12_reset_resident_weights();

  std::uint32_t rng = 0xC0FFEE11u;
  const auto build_quant_weights = [&](D3d12WeightFormat format, std::uint32_t row_count, std::uint32_t row_width) {
    const std::uint64_t block_bytes = format == D3d12WeightFormat::Q4_K
      ? k_q4_k_block_bytes
      : k_q6_k_block_bytes;
    const std::uint64_t total_bytes =
      static_cast<std::uint64_t>(row_count) * row_width / k_quant_block_elements * block_bytes;
    std::vector<std::uint8_t> weights(static_cast<std::size_t>(total_bytes));
    for (std::size_t index = 0; index < weights.size(); index += 1) {
      weights[index] = static_cast<std::uint8_t>(selftest_rng(rng) & 0xFFu);
    }
    const std::uint16_t d_bits = 0x2E66;
    const std::uint16_t dmin_bits = 0x2A66;
    const std::uint64_t block_count = total_bytes / block_bytes;
    for (std::uint64_t block = 0; block < block_count; block += 1) {
      std::uint8_t *block_ptr = weights.data() + block * block_bytes;
      if (format == D3d12WeightFormat::Q4_K) {
        std::memcpy(block_ptr, &d_bits, sizeof(d_bits));
        std::memcpy(block_ptr + 2, &dmin_bits, sizeof(dmin_bits));
      } else {
        std::memcpy(block_ptr + k_q6_k_block_bytes - 2, &d_bits, sizeof(d_bits));
      }
    }
    return weights;
  };

  const std::vector<std::uint8_t> gate_weights = build_quant_weights(D3d12WeightFormat::Q4_K, hidden_rows, width);
  const std::vector<std::uint8_t> up_weights = build_quant_weights(D3d12WeightFormat::Q4_K, hidden_rows, width);
  const std::vector<std::uint8_t> down_weights = build_quant_weights(D3d12WeightFormat::Q6_K, width, hidden_rows);

  std::vector<float> packed_inputs(static_cast<std::size_t>(batch_count) * width);
  for (std::size_t index = 0; index < packed_inputs.size(); index += 1) {
    packed_inputs[index] = input_scale
      * static_cast<float>(static_cast<std::int32_t>(selftest_rng(rng) % 2001) - 1000) / 1000.0f;
    // Same exactly-representable-in-Q8_1 mode the other selftests use, so the
    // chain's integer gate/up path can be checked for arithmetic bugs without the
    // quantisation loss - which compounds through silu and down - looking like one.
    if (selftest_exact_q8_1_inputs()) {
      packed_inputs[index] = index % 32u == 0
        ? input_scale
        : input_scale
          * static_cast<float>(static_cast<std::int32_t>(selftest_rng(rng) % 255u) - 127)
          / 127.0f;
    }
  }

  // Mimics the real graph: a resident matmul submission runs between chain
  // calls so shared buffer/state interactions are exercised.
  if (interleave_matmul) {
    D3d12ResidentWeightRef interleaved;
    interleaved.weight_key = gate_weights.data();
    interleaved.format = D3d12WeightFormat::Q4_K;
    interleaved.weight_data = gate_weights.data();
    interleaved.weight_bytes = gate_weights.size();
    interleaved.row_count = hidden_rows;
    interleaved.row_width = width;
    std::vector<float> interleaved_output;
    static std::uintptr_t interleave_epoch = 0x61000000;
    interleave_epoch += 1;
    std::string interleave_error;
    if (!d3d12_batch_matmul_resident(
      &interleaved,
      1,
      packed_inputs,
      batch_count,
      reinterpret_cast<const void *>(interleave_epoch),
      &interleaved_output,
      &interleave_error)) {
      std::printf("[FAIL] %s (interleave matmul): %s\n", label, interleave_error.c_str());
      return false;
    }
  }

  const auto make_ref = [](const std::vector<std::uint8_t> &weights, D3d12WeightFormat format, std::uint32_t rows, std::uint32_t row_width) {
    D3d12ResidentWeightRef ref;
    ref.weight_key = weights.data();
    ref.format = format;
    ref.weight_data = weights.data();
    ref.weight_bytes = weights.size();
    ref.row_count = rows;
    ref.row_width = row_width;
    return ref;
  };

  std::vector<float> outputs;
  std::string error_message;
  static std::uintptr_t chain_epoch = 0x51000000;
  chain_epoch += 1;
  const auto started_at = std::chrono::steady_clock::now();
  const auto chain_call = use_silu ? d3d12_ffn_silu_glu_resident : d3d12_ffn_gelu_glu_resident;
  if (!chain_call(
    make_ref(gate_weights, D3d12WeightFormat::Q4_K, hidden_rows, width),
    make_ref(up_weights, D3d12WeightFormat::Q4_K, hidden_rows, width),
    make_ref(down_weights, D3d12WeightFormat::Q6_K, width, hidden_rows),
    packed_inputs,
    batch_count,
    reinterpret_cast<const void *>(chain_epoch),
    &outputs,
    &error_message)) {
    std::printf("[FAIL] %s: %s\n", label, error_message.c_str());
    return false;
  }
  const auto finished_at = std::chrono::steady_clock::now();
  const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(finished_at - started_at).count();

  // Isolated rate at the caller's exact shape and batch.
  //
  // In-situ prefill measures FFN at ~170 GFLOP/s while the projection kernel
  // reaches 310 on a 10240x2560 b48 microbenchmark, and ~3.0 s of a 269-token
  // prefill sits in that gap. Copies cannot explain it (280 MB up + 341 MB back
  // for the WHOLE prefill), so either the chain is simply slower at THIS shape -
  // in which case the gap is not a gap at all and the lever is kernel work at
  // 8192x3072 - or the loss is around the call. Running the same chain in
  // isolation here separates those without needing GPU timestamps.
  if (ffn_chain_perf_repeats() > 0) {
    const auto perf_started = std::chrono::steady_clock::now();
    for (std::uint32_t iteration = 0; iteration < ffn_chain_perf_repeats(); iteration += 1) {
      chain_epoch += 1;
      std::vector<float> perf_outputs;
      std::string perf_error;
      if (!chain_call(
        make_ref(gate_weights, D3d12WeightFormat::Q4_K, hidden_rows, width),
        make_ref(up_weights, D3d12WeightFormat::Q4_K, hidden_rows, width),
        make_ref(down_weights, D3d12WeightFormat::Q6_K, width, hidden_rows),
        packed_inputs,
        batch_count,
        reinterpret_cast<const void *>(chain_epoch),
        &perf_outputs,
        &perf_error)) {
        std::printf("[FAIL] %s perf: %s\n", label, perf_error.c_str());
        return false;
      }
    }
    const double perf_ms =
      std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - perf_started).count()
      / static_cast<double>(ffn_chain_perf_repeats());
    // gate + up (hidden_rows x width) and down (width x hidden_rows): three
    // matmuls of the same 2*width*hidden_rows*batch shape.
    const double flops =
      3.0 * 2.0 * static_cast<double>(width) * hidden_rows * batch_count;
    std::printf(
      "[perf] %s ISOLATED: %.2f ms/iter, %.0f GFLOP/s (batch %u, %ux%u)\n",
      label, perf_ms, flops / (perf_ms / 1000.0) / 1e9, batch_count, hidden_rows, width);

    // Split the chain by format. rowblock_pipeline_for returns nullptr unless
    // the format is Q4_K, so gate/up (Q4_K) are row-blocked and down (Q6_K)
    // never is - which would make one third of the chain's FLOPs run on the
    // unblocked kernel. Timing the two shapes separately says whether that is
    // actually where the chain's ~165 GFLOP/s comes from, instead of inferring
    // it from the combined number.
    struct FormatProbe {
      const char *name;
      D3d12WeightFormat format;
      const std::vector<std::uint8_t> *weights;
      std::uint32_t rows;
      std::uint32_t row_width;
    };
    const FormatProbe probes[2] = {
      {"gate/up Q4_K", D3d12WeightFormat::Q4_K, &gate_weights, hidden_rows, width},
      {"down   Q6_K", D3d12WeightFormat::Q6_K, &down_weights, width, hidden_rows},
    };
    for (const FormatProbe &probe : probes) {
      std::vector<float> probe_inputs(
        static_cast<std::size_t>(batch_count) * probe.row_width, 0.25f);
      D3d12ResidentWeightRef probe_op =
        make_ref(*probe.weights, probe.format, probe.rows, probe.row_width);
      std::vector<std::vector<float>> probe_outputs(1);
      std::string probe_error;
      const auto probe_started = std::chrono::steady_clock::now();
      bool probe_ok = true;
      for (std::uint32_t iteration = 0; iteration < ffn_chain_perf_repeats(); iteration += 1) {
        if (!d3d12_batch_matmul_resident(
          &probe_op, 1, probe_inputs, batch_count,
          probe_inputs.data(), probe_outputs.data(), &probe_error)) {
          std::printf("[skip] %s %s: %s\n", label, probe.name, probe_error.c_str());
          probe_ok = false;
          break;
        }
      }
      if (!probe_ok) {
        continue;
      }
      const double probe_ms =
        std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - probe_started).count()
        / static_cast<double>(ffn_chain_perf_repeats());
      const double probe_flops =
        2.0 * static_cast<double>(probe.rows) * probe.row_width * batch_count;
      std::printf(
        "[perf]   %s %ux%u: %.2f ms/iter, %.0f GFLOP/s\n",
        probe.name, probe.rows, probe.row_width, probe_ms,
        probe_flops / (probe_ms / 1000.0) / 1e9);
    }
  }

  double max_difference = 0.0;
  const std::uint32_t verified_batches = std::min<std::uint32_t>(batch_count, 2);
  for (std::uint32_t batch = 0; batch < verified_batches; batch += 1) {
    const float *batch_input = packed_inputs.data() + static_cast<std::size_t>(batch) * width;
    std::vector<float> activation(hidden_rows, 0.0f);
    for (std::uint32_t row = 0; row < hidden_rows; row += 1) {
      const double gate_value = selftest_reference_row_dot(
        D3d12WeightFormat::Q4_K, gate_weights, row, width, batch_input);
      const double up_value = selftest_reference_row_dot(
        D3d12WeightFormat::Q4_K, up_weights, row, width, batch_input);
      if (use_silu) {
        const double silu = gate_value / (1.0 + std::exp(-gate_value));
        activation[row] = static_cast<float>(silu * up_value);
      } else {
        const double cubed = gate_value * gate_value * gate_value;
        const double gelu = 0.5 * gate_value
          * (1.0 + std::tanh(0.7978845608028654 * (gate_value + 0.044715 * cubed)));
        activation[row] = static_cast<float>(gelu * up_value);
      }
    }
    const std::uint32_t rows_to_verify = std::min<std::uint32_t>(width, verified_rows);
    for (std::uint32_t row = 0; row < rows_to_verify; row += 1) {
      const double reference = selftest_reference_row_dot(
        D3d12WeightFormat::Q6_K, down_weights, row, hidden_rows, activation.data());
      const double actual = outputs[static_cast<std::size_t>(batch) * width + row];
      const double difference = std::abs(actual - reference);
      const double tolerance = 5e-2 * std::max(1.0, std::abs(reference));
      max_difference = std::max(max_difference, difference);
      if (difference > tolerance) {
        std::printf(
          "[FAIL] %s: batch=%u row=%u expected=%f actual=%f\n",
          label,
          batch,
          row,
          reference,
          actual);
        return false;
      }
    }
  }
  std::printf(
    "[ ok ] %s: %lld ms, max sample diff %g\n",
    label,
    static_cast<long long>(elapsed_ms),
    max_difference);
  return true;
}

bool run_one_gemma_attention_case(
  const char *label,
  std::uint32_t hidden_width,
  std::uint32_t head_count,
  std::uint32_t kv_head_count,
  std::uint32_t head_dim,
  std::uint32_t position,
  std::uint32_t attention_window,
  float rope_base,
  float rope_scale,
  bool quantized_qkv = false,
  std::vector<float> *actual_out = nullptr) {
  const std::uint32_t query_size = head_count * head_dim;
  const std::uint32_t kv_size = kv_head_count * head_dim;
  constexpr float epsilon = 1e-5f;

  std::vector<float> query_weights(query_size * hidden_width);
  std::vector<float> key_weights(kv_size * hidden_width);
  std::vector<float> value_weights(kv_size * hidden_width);
  std::vector<float> hidden(hidden_width);
  std::vector<float> query_norm(head_dim);
  std::vector<float> key_norm(head_dim);
  std::vector<float> rope_factors(head_dim / 2);
  std::vector<float> initial_keys(position * kv_size);
  std::vector<float> initial_values(position * kv_size);
  for (std::size_t index = 0; index < query_weights.size(); index += 1) {
    query_weights[index] = 0.025f * std::sin(static_cast<float>(index + 3) * 0.071f);
  }
  for (std::size_t index = 0; index < key_weights.size(); index += 1) {
    key_weights[index] = 0.03f * std::cos(static_cast<float>(index + 5) * 0.053f);
    value_weights[index] = 0.02f * std::sin(static_cast<float>(index + 7) * 0.047f);
  }
  for (std::size_t index = 0; index < hidden.size(); index += 1) {
    hidden[index] = 0.4f * std::sin(static_cast<float>(index + 1) * 0.19f);
  }
  for (std::size_t index = 0; index < query_norm.size(); index += 1) {
    query_norm[index] = 0.9f + 0.02f * static_cast<float>(index);
    key_norm[index] = 1.05f - 0.015f * static_cast<float>(index);
  }
  for (std::size_t index = 0; index < rope_factors.size(); index += 1) {
    rope_factors[index] = 1.0f + 0.1f * static_cast<float>(index);
  }
  for (std::size_t index = 0; index < initial_keys.size(); index += 1) {
    initial_keys[index] = 0.3f * std::sin(static_cast<float>(index + 11) * 0.13f);
    initial_values[index] = 0.25f * std::cos(static_cast<float>(index + 17) * 0.11f);
  }

  // Optional Q4_K projections: exercises the quantized batch-one matvec route
  // used by the real decode graph instead of the F32 scalar lane.
  std::uint32_t qkv_rng = 0x51E57A1u;
  const auto build_q4k_weights = [&](std::uint32_t rows) {
    std::vector<std::uint8_t> bytes(
      static_cast<std::size_t>(rows) * hidden_width / 256u * 144u);
    for (std::size_t index = 0; index < bytes.size(); index += 1) {
      bytes[index] = static_cast<std::uint8_t>(selftest_rng(qkv_rng) & 0xFFu);
    }
    const std::uint16_t d_bits = 0x2E66;
    const std::uint16_t dmin_bits = 0x2A66;
    for (std::size_t block = 0; block * 144u < bytes.size(); block += 1) {
      std::memcpy(bytes.data() + block * 144u, &d_bits, sizeof(d_bits));
      std::memcpy(bytes.data() + block * 144u + 2u, &dmin_bits, sizeof(dmin_bits));
    }
    return bytes;
  };
  std::vector<std::uint8_t> query_qbytes;
  std::vector<std::uint8_t> key_qbytes;
  std::vector<std::uint8_t> value_qbytes;
  if (quantized_qkv) {
    query_qbytes = build_q4k_weights(query_size);
    key_qbytes = build_q4k_weights(kv_size);
    value_qbytes = build_q4k_weights(kv_size);
  }
  const auto matvec_q4k = [&](
    const std::vector<std::uint8_t> &bytes,
    std::uint32_t rows,
    const std::vector<float> &input) {
    std::vector<float> output(rows, 0.0f);
    for (std::uint32_t row = 0; row < rows; row += 1) {
      output[row] = static_cast<float>(selftest_reference_row_dot(
        D3d12WeightFormat::Q4_K, bytes, row, hidden_width, input.data()));
    }
    return output;
  };
  const auto make_q4k_ref = [&](const std::vector<std::uint8_t> &bytes, std::uint32_t rows) {
    D3d12ResidentWeightRef ref;
    ref.weight_key = bytes.data();
    ref.format = D3d12WeightFormat::Q4_K;
    ref.weight_data = bytes.data();
    ref.weight_bytes = bytes.size();
    ref.row_count = rows;
    ref.row_width = hidden_width;
    return ref;
  };

  const auto make_ref = [](
    const std::vector<float> &weights,
    std::uint32_t rows,
    std::uint32_t width) {
    D3d12ResidentWeightRef ref;
    ref.weight_key = weights.data();
    ref.format = D3d12WeightFormat::F32;
    ref.weight_data = reinterpret_cast<const std::uint8_t *>(weights.data());
    ref.weight_bytes = weights.size() * sizeof(float);
    ref.row_count = rows;
    ref.row_width = width;
    return ref;
  };
  const auto matvec = [](
    const std::vector<float> &weights,
    std::uint32_t rows,
    const std::vector<float> &input) {
    std::vector<float> output(rows, 0.0f);
    for (std::uint32_t row = 0; row < rows; row += 1) {
      double sum = 0.0;
      for (std::size_t item = 0; item < input.size(); item += 1) {
        sum += static_cast<double>(weights[static_cast<std::size_t>(row) * input.size() + item])
          * input[item];
      }
      output[row] = static_cast<float>(sum);
    }
    return output;
  };
  const auto prepare_heads = [&](
    std::vector<float> values,
    std::uint32_t heads,
    const std::vector<float> *norm,
    bool rope) {
    for (std::uint32_t head = 0; head < heads; head += 1) {
      const std::size_t base = static_cast<std::size_t>(head) * head_dim;
      double square_sum = 0.0;
      for (std::uint32_t item = 0; item < head_dim; item += 1) {
        square_sum += static_cast<double>(values[base + item]) * values[base + item];
      }
      const float inverse_rms = static_cast<float>(
        1.0 / std::sqrt(square_sum / head_dim + epsilon));
      for (std::uint32_t item = 0; item < head_dim; item += 1) {
        values[base + item] *= inverse_rms * (norm == nullptr ? 1.0f : (*norm)[item]);
      }
      if (rope) {
        for (std::uint32_t pair = 0; pair < head_dim / 2; pair += 1) {
          const float inverse_frequency = std::pow(
            rope_base,
            -(2.0f * static_cast<float>(pair)) / static_cast<float>(head_dim))
            / rope_factors[pair];
          const float angle = static_cast<float>(position) * rope_scale * inverse_frequency;
          const float cosine = std::cos(angle);
          const float sine = std::sin(angle);
          const std::size_t first = base + pair;
          const std::size_t second = first + head_dim / 2;
          const float first_value = values[first];
          const float second_value = values[second];
          values[first] = first_value * cosine - second_value * sine;
          values[second] = first_value * sine + second_value * cosine;
        }
      }
    }
    return values;
  };
  const auto round_fp16 = [](std::vector<float> values) {
    for (float &value : values) {
      value = staging_half_to_float(staging_float_to_half(value));
    }
    return values;
  };

  const std::vector<float> query = prepare_heads(
    quantized_qkv
      ? matvec_q4k(query_qbytes, query_size, hidden)
      : matvec(query_weights, query_size, hidden),
    head_count, &query_norm, true);
  const std::vector<float> current_key = round_fp16(prepare_heads(
    quantized_qkv
      ? matvec_q4k(key_qbytes, kv_size, hidden)
      : matvec(key_weights, kv_size, hidden),
    kv_head_count, &key_norm, true));
  const std::vector<float> current_value = round_fp16(prepare_heads(
    quantized_qkv
      ? matvec_q4k(value_qbytes, kv_size, hidden)
      : matvec(value_weights, kv_size, hidden),
    kv_head_count, nullptr, false));
  std::vector<float> all_keys = round_fp16(initial_keys);
  std::vector<float> all_values = round_fp16(initial_values);
  all_keys.insert(all_keys.end(), current_key.begin(), current_key.end());
  all_values.insert(all_values.end(), current_value.begin(), current_value.end());
  const std::uint32_t first_source =
    attention_window > 0 && position + 1 > attention_window
    ? position + 1 - attention_window
    : 0;
  std::vector<float> reference(query_size, 0.0f);
  for (std::uint32_t head = 0; head < head_count; head += 1) {
    const std::uint32_t kv_head = head * kv_head_count / head_count;
    std::vector<float> scores(position + 1 - first_source, 0.0f);
    for (std::uint32_t source = first_source; source <= position; source += 1) {
      double dot = 0.0;
      for (std::uint32_t item = 0; item < head_dim; item += 1) {
        dot += static_cast<double>(query[head * head_dim + item])
          * all_keys[source * kv_size + kv_head * head_dim + item];
      }
      // Raw dot as the score, matching the production CPU path (QK-norm
      // Gemma applies no 1/sqrt(head_dim) attention scaling).
      scores[source - first_source] = static_cast<float>(dot);
    }
    const float maximum = *std::max_element(scores.begin(), scores.end());
    double denominator = 0.0;
    for (float &score : scores) {
      score = std::exp(score - maximum);
      denominator += score;
    }
    for (std::uint32_t source = first_source; source <= position; source += 1) {
      const float weight = static_cast<float>(scores[source - first_source] / denominator);
      for (std::uint32_t item = 0; item < head_dim; item += 1) {
        reference[head * head_dim + item] +=
          weight * all_values[source * kv_size + kv_head * head_dim + item];
      }
    }
  }

  D3d12GemmaAttentionConfig config;
  static std::uint64_t attention_case_counter = 0;
  attention_case_counter += 1;
  config.cache_id = 0xA771E001ull + (attention_case_counter << 40);
  config.position = position;
  config.max_positions = std::max<std::uint32_t>(position + 2, 16);
  config.head_count = head_count;
  config.kv_head_count = kv_head_count;
  config.head_dim = head_dim;
  config.attention_window = attention_window;
  config.rms_epsilon = epsilon;
  config.rope_freq_base = rope_base;
  config.rope_freq_scale = rope_scale;
  std::vector<float> actual;
  std::string error_message;
  const D3d12ResidentWeightRef query_ref = quantized_qkv
    ? make_q4k_ref(query_qbytes, query_size)
    : make_ref(query_weights, query_size, hidden_width);
  const D3d12ResidentWeightRef key_ref = quantized_qkv
    ? make_q4k_ref(key_qbytes, kv_size)
    : make_ref(key_weights, kv_size, hidden_width);
  const D3d12ResidentWeightRef value_ref = quantized_qkv
    ? make_q4k_ref(value_qbytes, kv_size)
    : make_ref(value_weights, kv_size, hidden_width);
  const D3d12KvLaneView initial_keys_view {initial_keys.data(), nullptr, initial_keys.size(), true};
  const D3d12KvLaneView initial_values_view {initial_values.data(), nullptr, initial_values.size(), true};
  if (!d3d12_gemma_attention_resident(
    query_ref,
    &key_ref,
    &value_ref,
    hidden,
    query_norm,
    key_norm,
    rope_factors,
    initial_keys_view,
    initial_values_view,
    config,
    &actual,
    &error_message)) {
    std::printf("[FAIL] %s: %s\n", label, error_message.c_str());
    return false;
  }
  double maximum_difference = 0.0;
  for (std::size_t index = 0; index < reference.size(); index += 1) {
    const double difference = std::abs(static_cast<double>(actual[index]) - reference[index]);
    maximum_difference = std::max(maximum_difference, difference);
    if (difference > 4e-3 * std::max(1.0, std::abs(static_cast<double>(reference[index])))) {
      std::printf(
        "[FAIL] %s: index=%zu expected=%f actual=%f\n",
        label,
        index,
        reference[index],
        actual[index]);
      return false;
    }
  }
  std::printf("[ ok ] %s: max diff %g\n", label, maximum_difference);
  if (actual_out != nullptr) {
    *actual_out = actual;
  }
  return true;
}

// Runs one attention shape through BOTH kernels (untiled and position-tiled)
// and requires bitwise-identical context outputs. The tiled kernel claims
// exact floating-point-order equivalence; this is the check that holds it to
// that claim on real hardware instead of trusting the derivation.
bool run_attention_tile_parity_case(
  const char *label,
  std::uint32_t hidden_width,
  std::uint32_t head_count,
  std::uint32_t kv_head_count,
  std::uint32_t head_dim,
  std::uint32_t position,
  std::uint32_t attention_window,
  float rope_base,
  float rope_scale) {
  std::vector<float> untiled_output;
  std::vector<float> tiled_output;
  // Same-shaped weight vectors are allocated/freed back to back across the
  // parity invocations, so the allocator reuses addresses and the
  // pointer-keyed resident-weight cache would serve stale weights
  // nondeterministically (garbage/NaN in EITHER kernel). Reset between
  // cases exactly like run_ffn_chain_selftest does.
  d3d12_reset_resident_weights();
  SetEnvironmentVariableA("LIZARD_NATIVE_GPU_ATTN_TILE", "0");
  bool ok = run_one_gemma_attention_case(
    label, hidden_width, head_count, kv_head_count, head_dim,
    position, attention_window, rope_base, rope_scale, false, &untiled_output);
  d3d12_reset_resident_weights();
  SetEnvironmentVariableA("LIZARD_NATIVE_GPU_ATTN_TILE", "1");
  ok = ok && run_one_gemma_attention_case(
    label, hidden_width, head_count, kv_head_count, head_dim,
    position, attention_window, rope_base, rope_scale, false, &tiled_output);
  SetEnvironmentVariableA("LIZARD_NATIVE_GPU_ATTN_TILE", nullptr);
  if (!ok) {
    return false;
  }
  if (untiled_output.size() != tiled_output.size()) {
    std::printf("[FAIL] %s (tile parity): output sizes differ\n", label);
    return false;
  }
  for (std::size_t index = 0; index < untiled_output.size(); index += 1) {
    if (std::memcmp(&untiled_output[index], &tiled_output[index], sizeof(float)) != 0) {
      std::printf(
        "[FAIL] %s (tile parity): index=%zu untiled=%.9g tiled=%.9g\n",
        label,
        index,
        untiled_output[index],
        tiled_output[index]);
      return false;
    }
  }
  std::printf("[ ok ] %s (tile parity): bit-identical over %zu values\n", label, tiled_output.size());
  return true;
}

bool run_attention_tile_parity_selftest() {
  return run_attention_tile_parity_case(
      "attention tile parity global 20pos", 2560, 8, 2, 256, 19, 0, 1000000.0f, 1.0f)
    && run_attention_tile_parity_case(
      "attention tile parity local windowed", 2560, 8, 2, 256, 19, 512, 10000.0f, 1.0f)
    && run_attention_tile_parity_case(
      "attention tile parity tight window", 2560, 8, 2, 256, 19, 8, 10000.0f, 1.0f)
    && run_attention_tile_parity_case(
      "attention tile parity exact 32pos", 2560, 8, 2, 256, 31, 0, 1000000.0f, 1.0f)
    && run_attention_tile_parity_case(
      "attention tile parity sub-tile 3pos", 32, 4, 2, 8, 2, 0, 10000.0f, 1.0f)
    && run_attention_tile_parity_case(
      "attention tile parity probe base1e4 w0 20pos", 2560, 8, 2, 256, 19, 0, 10000.0f, 1.0f)
    && run_attention_tile_parity_case(
      "attention tile parity head-dim 512", 2560, 8, 2, 512, 19, 0, 1000000.0f, 1.0f);
}

bool run_gemma_attention_selftest() {
  // Tiny legacy case plus real Gemma-3n decode shapes: global layer (1M rope
  // base), local sliding-window layer (10K base), a window that truncates the
  // attended range, and a scaled-rope variant.
  return run_one_gemma_attention_case(
      "Gemma resident FP16 online attention", 32, 4, 2, 8, 2, 0, 10000.0f, 1.0f)
    && run_one_gemma_attention_case(
      "Gemma attention real-dims global", 2560, 8, 2, 256, 19, 0, 1000000.0f, 1.0f)
    && run_one_gemma_attention_case(
      "Gemma attention real-dims local", 2560, 8, 2, 256, 19, 512, 10000.0f, 1.0f)
    && run_one_gemma_attention_case(
      "Gemma attention real-dims windowed", 2560, 8, 2, 256, 19, 8, 10000.0f, 1.0f)
    && run_one_gemma_attention_case(
      "Gemma attention real-dims rope-scale", 2560, 8, 2, 256, 19, 0, 1000000.0f, 0.125f)
    && run_one_gemma_attention_case(
      "Gemma attention real-dims q4_k", 2560, 8, 2, 256, 19, 0, 1000000.0f, 1.0f, true);
}

// Verifies the fused decode layer tail (attn-out matmul, norms, residuals,
// FFN chain) against a CPU reference at real Gemma shapes.
bool run_layer_tail_selftest() {
  d3d12_reset_resident_weights();
  const std::uint32_t context_width = 2048;
  const std::uint32_t hidden_width = 2560;
  const std::uint32_t ffn_width = 10240;
  const std::uint32_t ple_width = 2688;

  std::uint32_t rng = 0xFEEDF00Du;
  const auto build_quant_weights = [&](D3d12WeightFormat format, std::uint32_t row_count, std::uint32_t row_width) {
    const std::uint64_t block_bytes = format == D3d12WeightFormat::Q4_K
      ? k_q4_k_block_bytes
      : k_q6_k_block_bytes;
    const std::uint64_t total_bytes =
      static_cast<std::uint64_t>(row_count) * row_width / k_quant_block_elements * block_bytes;
    std::vector<std::uint8_t> weights(static_cast<std::size_t>(total_bytes));
    for (std::size_t index = 0; index < weights.size(); index += 1) {
      weights[index] = static_cast<std::uint8_t>(selftest_rng(rng) & 0xFFu);
    }
    const std::uint16_t d_bits = 0x2266;   // ~0.025 in fp16: keeps chain values sane
    const std::uint16_t dmin_bits = 0x1E66;
    const std::uint64_t block_count = total_bytes / block_bytes;
    for (std::uint64_t block = 0; block < block_count; block += 1) {
      std::uint8_t *block_ptr = weights.data() + block * block_bytes;
      if (format == D3d12WeightFormat::Q4_K) {
        std::memcpy(block_ptr, &d_bits, sizeof(d_bits));
        std::memcpy(block_ptr + 2, &dmin_bits, sizeof(dmin_bits));
      } else {
        std::memcpy(block_ptr + k_q6_k_block_bytes - 2, &d_bits, sizeof(d_bits));
      }
    }
    return weights;
  };
  const auto random_vector = [&](std::uint32_t count, float scale, float bias) {
    std::vector<float> values(count);
    for (std::uint32_t index = 0; index < count; index += 1) {
      values[index] = bias + scale
        * static_cast<float>(static_cast<std::int32_t>(selftest_rng(rng) % 2001) - 1000) / 1000.0f;
    }
    return values;
  };
  const auto build_bf16_weights = [&](std::uint32_t row_count, std::uint32_t row_width) {
    std::vector<std::uint8_t> weights(
      static_cast<std::size_t>(row_count) * row_width * sizeof(std::uint16_t));
    auto *values = reinterpret_cast<std::uint16_t *>(weights.data());
    const std::size_t value_count = weights.size() / sizeof(std::uint16_t);
    for (std::size_t index = 0; index < value_count; index += 1) {
      // +/- 0.0078125 keeps the two additional projections numerically stable.
      values[index] = (selftest_rng(rng) & 1u) == 0 ? 0x3C00u : 0xBC00u;
    }
    return weights;
  };

  const std::vector<std::uint8_t> attn_weights = build_quant_weights(D3d12WeightFormat::Q4_K, hidden_width, context_width);
  const std::vector<std::uint8_t> gate_weights = build_quant_weights(D3d12WeightFormat::Q4_K, ffn_width, hidden_width);
  const std::vector<std::uint8_t> up_weights = build_quant_weights(D3d12WeightFormat::Q4_K, ffn_width, hidden_width);
  const std::vector<std::uint8_t> down_weights = build_quant_weights(D3d12WeightFormat::Q6_K, hidden_width, ffn_width);
  const std::vector<float> context = random_vector(context_width, 1.0f, 0.0f);
  const std::vector<float> hidden = random_vector(hidden_width, 1.0f, 0.0f);
  const std::vector<float> post_attn_norm = random_vector(hidden_width, 0.1f, 1.0f);
  const std::vector<float> ffn_norm = random_vector(hidden_width, 0.1f, 1.0f);
  const std::vector<float> post_ffw_norm = random_vector(hidden_width, 0.1f, 1.0f);
  const std::vector<std::uint8_t> ple_gate_weights = build_bf16_weights(ple_width, hidden_width);
  const std::vector<std::uint8_t> ple_projection_weights = build_bf16_weights(hidden_width, ple_width);
  const std::vector<float> ple_input = random_vector(ple_width, 0.25f, 0.0f);
  const std::vector<float> ple_post_norm = random_vector(hidden_width, 0.1f, 1.0f);
  const float epsilon = 1e-6f;

  const auto make_ref = [](const std::vector<std::uint8_t> &weights, D3d12WeightFormat format, std::uint32_t rows, std::uint32_t row_width) {
    D3d12ResidentWeightRef ref;
    ref.weight_key = weights.data();
    ref.format = format;
    ref.weight_data = weights.data();
    ref.weight_bytes = weights.size();
    ref.row_count = rows;
    ref.row_width = row_width;
    return ref;
  };

  std::vector<float> gpu_output;
  std::string error_message;
  const auto started_at = std::chrono::steady_clock::now();
  if (!d3d12_gemma_layer_tail_resident(
    make_ref(attn_weights, D3d12WeightFormat::Q4_K, hidden_width, context_width),
    make_ref(gate_weights, D3d12WeightFormat::Q4_K, ffn_width, hidden_width),
    make_ref(up_weights, D3d12WeightFormat::Q4_K, ffn_width, hidden_width),
    make_ref(down_weights, D3d12WeightFormat::Q6_K, hidden_width, ffn_width),
    context,
    hidden,
    post_attn_norm,
    ffn_norm,
    post_ffw_norm,
    epsilon,
    &gpu_output,
    &error_message)) {
    std::printf("[FAIL] layer tail 2560/10240: %s\n", error_message.c_str());
    return false;
  }
  const auto finished_at = std::chrono::steady_clock::now();
  const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(finished_at - started_at).count();

  // CPU reference of the identical sequence.
  const auto cpu_rms_norm = [&](const std::vector<float> &input, const std::vector<float> &norm_weights) {
    double sum = 0.0;
    for (const float value : input) {
      sum += static_cast<double>(value) * value;
    }
    const double inverse = 1.0 / std::sqrt(sum / input.size() + epsilon);
    std::vector<float> output(input.size());
    for (std::size_t index = 0; index < input.size(); index += 1) {
      output[index] = static_cast<float>(input[index] * inverse * norm_weights[index]);
    }
    return output;
  };
  std::vector<float> attn_out(hidden_width);
  for (std::uint32_t row = 0; row < hidden_width; row += 1) {
    attn_out[row] = static_cast<float>(selftest_reference_row_dot(
      D3d12WeightFormat::Q4_K, attn_weights, row, context_width, context.data()));
  }
  std::vector<float> normed_attention = cpu_rms_norm(attn_out, post_attn_norm);
  std::vector<float> residual(hidden_width);
  for (std::uint32_t index = 0; index < hidden_width; index += 1) {
    residual[index] = hidden[index] + normed_attention[index];
  }
  const std::vector<float> ffn_in = cpu_rms_norm(residual, ffn_norm);
  std::vector<float> activation(ffn_width);
  for (std::uint32_t row = 0; row < ffn_width; row += 1) {
    const double gate_value = selftest_reference_row_dot(
      D3d12WeightFormat::Q4_K, gate_weights, row, hidden_width, ffn_in.data());
    const double up_value = selftest_reference_row_dot(
      D3d12WeightFormat::Q4_K, up_weights, row, hidden_width, ffn_in.data());
    const double cubed = gate_value * gate_value * gate_value;
    const double gelu = 0.5 * gate_value
      * (1.0 + std::tanh(0.7978845608028654 * (gate_value + 0.044715 * cubed)));
    activation[row] = static_cast<float>(gelu * up_value);
  }
  std::vector<float> down_out(hidden_width);
  for (std::uint32_t row = 0; row < hidden_width; row += 1) {
    down_out[row] = static_cast<float>(selftest_reference_row_dot(
      D3d12WeightFormat::Q6_K, down_weights, row, ffn_width, activation.data()));
  }
  const std::vector<float> normed_down = cpu_rms_norm(down_out, post_ffw_norm);

  double max_difference = 0.0;
  std::vector<float> tail_reference(hidden_width);
  for (std::uint32_t index = 0; index < hidden_width; index += 1) {
    const double reference = residual[index] + normed_down[index];
    tail_reference[index] = static_cast<float>(reference);
    const double actual = gpu_output[index];
    const double difference = std::abs(actual - reference);
    const double tolerance = 5e-2 * std::max(1.0, std::abs(reference));
    max_difference = std::max(max_difference, difference);
    if (difference > tolerance) {
      std::printf(
        "[FAIL] layer tail 2560/10240: index=%u expected=%f actual=%f\n",
        index,
        reference,
        actual);
      return false;
    }
  }

  // The batch path must preserve the established single-row Llama tail. Use
  // three distinct rows so offset/stride errors cannot hide behind identical
  // data, and compare against the production B=1 kernel rather than a second
  // reimplementation of SwiGLU.
  constexpr std::uint32_t llama_batch_count = 3;
  std::vector<float> packed_llama_contexts;
  std::vector<float> packed_llama_hidden;
  std::vector<std::vector<float>> llama_context_rows;
  std::vector<std::vector<float>> llama_hidden_rows;
  for (std::uint32_t row = 0; row < llama_batch_count; row += 1) {
    llama_context_rows.push_back(random_vector(context_width, 1.0f, 0.05f * row));
    llama_hidden_rows.push_back(random_vector(hidden_width, 1.0f, -0.03f * row));
    packed_llama_contexts.insert(
      packed_llama_contexts.end(),
      llama_context_rows.back().begin(),
      llama_context_rows.back().end());
    packed_llama_hidden.insert(
      packed_llama_hidden.end(),
      llama_hidden_rows.back().begin(),
      llama_hidden_rows.back().end());
  }
  std::vector<float> llama_batch_output;
  error_message.clear();
  if (!d3d12_llama_layer_tail_batch_resident(
    make_ref(attn_weights, D3d12WeightFormat::Q4_K, hidden_width, context_width),
    make_ref(gate_weights, D3d12WeightFormat::Q4_K, ffn_width, hidden_width),
    make_ref(up_weights, D3d12WeightFormat::Q4_K, ffn_width, hidden_width),
    make_ref(down_weights, D3d12WeightFormat::Q6_K, hidden_width, ffn_width),
    packed_llama_contexts,
    packed_llama_hidden,
    ffn_norm,
    llama_batch_count,
    1e-5f,
    &llama_batch_output,
    &error_message)) {
    std::printf("[FAIL] Llama batch layer tail b3: %s\n", error_message.c_str());
    return false;
  }
  double llama_batch_max_difference = 0.0;
  for (std::uint32_t row = 0; row < llama_batch_count; row += 1) {
    std::vector<float> single_output;
    error_message.clear();
    if (!d3d12_llama_layer_tail_resident(
      make_ref(attn_weights, D3d12WeightFormat::Q4_K, hidden_width, context_width),
      make_ref(gate_weights, D3d12WeightFormat::Q4_K, ffn_width, hidden_width),
      make_ref(up_weights, D3d12WeightFormat::Q4_K, ffn_width, hidden_width),
      make_ref(down_weights, D3d12WeightFormat::Q6_K, hidden_width, ffn_width),
      llama_context_rows[row],
      llama_hidden_rows[row],
      ffn_norm,
      1e-5f,
      &single_output,
      &error_message)) {
      std::printf("[FAIL] Llama batch layer tail b3 single oracle: %s\n", error_message.c_str());
      return false;
    }
    for (std::uint32_t index = 0; index < hidden_width; index += 1) {
      const double expected = single_output[index];
      const double actual =
        llama_batch_output[static_cast<std::size_t>(row) * hidden_width + index];
      const double difference = std::abs(actual - expected);
      const double tolerance = 5e-2 * std::max(1.0, std::abs(expected));
      llama_batch_max_difference = std::max(llama_batch_max_difference, difference);
      if (difference > tolerance) {
        std::printf(
          "[FAIL] Llama batch layer tail b3: row=%u index=%u expected=%f actual=%f\n",
          row,
          index,
          expected,
          actual);
        return false;
      }
    }
  }

  std::vector<float> ple_gpu_output;
  error_message.clear();
  if (!d3d12_gemma_layer_tail_ple_resident(
    make_ref(attn_weights, D3d12WeightFormat::Q4_K, hidden_width, context_width),
    make_ref(gate_weights, D3d12WeightFormat::Q4_K, ffn_width, hidden_width),
    make_ref(up_weights, D3d12WeightFormat::Q4_K, ffn_width, hidden_width),
    make_ref(down_weights, D3d12WeightFormat::Q6_K, hidden_width, ffn_width),
    make_ref(ple_gate_weights, D3d12WeightFormat::BF16, ple_width, hidden_width),
    make_ref(ple_projection_weights, D3d12WeightFormat::BF16, hidden_width, ple_width),
    context,
    hidden,
    post_attn_norm,
    ffn_norm,
    post_ffw_norm,
    ple_input,
    ple_post_norm,
    epsilon,
    &ple_gpu_output,
    &error_message)) {
    std::printf("[FAIL] layer tail PLE 2560/10240/2688: %s\n", error_message.c_str());
    return false;
  }

  std::vector<float> ple_activation(ple_width);
  for (std::uint32_t row = 0; row < ple_width; row += 1) {
    const double gate_value = selftest_reference_row_dot(
      D3d12WeightFormat::BF16, ple_gate_weights, row, hidden_width, tail_reference.data());
    const double cubed = gate_value * gate_value * gate_value;
    const double gelu = 0.5 * gate_value
      * (1.0 + std::tanh(0.7978845608028654 * (gate_value + 0.044715 * cubed)));
    ple_activation[row] = static_cast<float>(gelu * ple_input[row]);
  }
  std::vector<float> ple_projection(hidden_width);
  for (std::uint32_t row = 0; row < hidden_width; row += 1) {
    ple_projection[row] = static_cast<float>(selftest_reference_row_dot(
      D3d12WeightFormat::BF16, ple_projection_weights, row, ple_width, ple_activation.data()));
  }
  const std::vector<float> normed_ple = cpu_rms_norm(ple_projection, ple_post_norm);
  double ple_max_difference = 0.0;
  for (std::uint32_t index = 0; index < hidden_width; index += 1) {
    const double reference = tail_reference[index] + normed_ple[index];
    const double actual = ple_gpu_output[index];
    const double difference = std::abs(actual - reference);
    const double tolerance = 5e-2 * std::max(1.0, std::abs(reference));
    ple_max_difference = std::max(ple_max_difference, difference);
    if (difference > tolerance) {
      std::printf(
        "[FAIL] layer tail PLE 2560/10240/2688: index=%u expected=%f actual=%f\n",
        index,
        reference,
        actual);
      return false;
    }
  }
  std::printf(
    "[ ok ] layer tail 2560/10240: %lld ms, max sample diff %g\n",
    static_cast<long long>(elapsed_ms),
    max_difference);
  std::printf(
    "[ ok ] layer tail PLE 2560/10240/2688: max sample diff %g\n",
    ple_max_difference);
  std::printf(
    "[ ok ] Llama batch layer tail b3: max sample diff %g\n",
    llama_batch_max_difference);
  return true;
}

}  // namespace

// Verifies the DP4A Q4_K x Q8_1 route against a CPU reference that replicates
// the GPU quantization exactly, then compares its speed with the float
// batch-one kernel on the largest shape. Skips (successfully) when DP4A is
// unavailable on this build or device.
bool run_dp4a_selftest() {
  const D3d12Dp4aCapabilities &dp4a = d3d12_dp4a_capabilities();
  if (!dp4a.supported) {
    std::printf("[skip] dp4a q4k: %s\n", dp4a.reason.c_str());
    return true;
  }
  const struct Dp4aCase {
    const char *label;
    std::uint32_t rows;
    std::uint32_t width;
  } cases[] = {
    {"dp4a q4k small 128x512", 128, 512},
    {"dp4a q4k gemma-q 2048x2560", 2048, 2560},
    {"dp4a q4k gemma-gate 10240x2560", 10240, 2560},
  };
  const auto get_scale_min = [](int j, const std::uint8_t *packed, std::uint8_t *scale, std::uint8_t *minimum) {
    if (j < 4) {
      *scale = packed[j] & 63;
      *minimum = packed[j + 4] & 63;
    } else {
      *scale = static_cast<std::uint8_t>((packed[j + 4] & 0x0F) | ((packed[j - 4] >> 6) << 4));
      *minimum = static_cast<std::uint8_t>((packed[j + 4] >> 4) | ((packed[j] >> 6) << 4));
    }
  };

  std::vector<std::uint8_t> timing_weights;
  D3d12ResidentWeightRef timing_op;
  std::vector<float> timing_input;

  if (quad_interleave_enabled()) {
    std::printf("[skip] dp4a q4k: not supported under the quad-interleaved layout\n");
    return true;
  }
  for (const Dp4aCase &test_case : cases) {
    d3d12_reset_resident_weights();
    std::uint32_t rng = 0xD94AD94Au
      ^ (test_case.rows * 2654435761u)
      ^ (test_case.width * 40503u);
    const std::uint64_t block_count =
      static_cast<std::uint64_t>(test_case.rows) * test_case.width / 256u;
    std::vector<std::uint8_t> weights(static_cast<std::size_t>(block_count) * 144u);
    for (std::size_t index = 0; index < weights.size(); index += 1) {
      weights[index] = static_cast<std::uint8_t>(selftest_rng(rng) & 0xFFu);
    }
    const std::uint16_t d_bits = 0x2E66;   // ~0.1
    const std::uint16_t dmin_bits = 0x2A66;  // ~0.05
    for (std::uint64_t block = 0; block < block_count; block += 1) {
      std::memcpy(weights.data() + block * 144u, &d_bits, sizeof(d_bits));
      std::memcpy(weights.data() + block * 144u + 2u, &dmin_bits, sizeof(dmin_bits));
    }
    std::vector<float> input(test_case.width);
    for (std::size_t index = 0; index < input.size(); index += 1) {
      input[index] =
        static_cast<float>(static_cast<std::int32_t>(selftest_rng(rng) % 2001) - 1000) / 1000.0f;
    }

    D3d12ResidentWeightRef op;
    op.weight_key = weights.data();
    op.format = D3d12WeightFormat::Q4_K;
    op.weight_data = weights.data();
    op.weight_bytes = weights.size();
    op.row_count = test_case.rows;
    op.row_width = test_case.width;

    std::vector<float> actual;
    std::string error_message;
    const auto started_at = std::chrono::steady_clock::now();
    if (!d3d12_dp4a_q4k_matvec(op, input, &actual, &error_message)) {
      std::printf("[FAIL] %s: %s\n", test_case.label, error_message.c_str());
      return false;
    }
    const auto finished_at = std::chrono::steady_clock::now();
    const auto elapsed_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(finished_at - started_at).count();

    // CPU reference with the exact GPU quantization semantics.
    const std::uint32_t activation_blocks = test_case.width / 32u;
    std::vector<int> quantized(test_case.width, 0);
    std::vector<float> dx(activation_blocks, 0.0f);
    std::vector<float> sx(activation_blocks, 0.0f);
    for (std::uint32_t block = 0; block < activation_blocks; block += 1) {
      float absmax = 0.0f;
      for (std::uint32_t item = 0; item < 32u; item += 1) {
        absmax = std::max(absmax, std::abs(input[block * 32u + item]));
      }
      const float d = absmax / 127.0f;
      const float inverse_d = d > 0.0f ? 1.0f / d : 0.0f;
      int sum = 0;
      for (std::uint32_t item = 0; item < 32u; item += 1) {
        // HLSL round() lowers to round-to-nearest-even; std::nearbyint under
        // the default rounding mode matches that tie behavior exactly.
        const int q = std::max(-127, std::min(127,
          static_cast<int>(std::nearbyint(input[block * 32u + item] * inverse_d))));
        quantized[block * 32u + item] = q;
        sum += q;
      }
      dx[block] = d;
      sx[block] = d * static_cast<float>(sum);
    }
    const float weight_d = selftest_half_to_float(d_bits);
    const float weight_dmin = selftest_half_to_float(dmin_bits);
    double maximum_difference = 0.0;
    const std::uint32_t blocks_per_row = test_case.width / 256u;
    for (std::uint32_t row = 0; row < test_case.rows; row += 1) {
      double reference = 0.0;
      for (std::uint32_t block = 0; block < blocks_per_row; block += 1) {
        const std::uint8_t *block_ptr =
          weights.data() + (static_cast<std::uint64_t>(row) * blocks_per_row + block) * 144u;
        const std::uint8_t *packed_scales = block_ptr + 4;
        const std::uint8_t *qs = block_ptr + 16;
        for (std::uint32_t j = 0; j < 8u; j += 1) {
          std::uint8_t scale = 0;
          std::uint8_t minimum = 0;
          get_scale_min(static_cast<int>(j), packed_scales, &scale, &minimum);
          const std::uint8_t *chunk = qs + (j >> 1u) * 32u;
          const bool high = (j & 1u) != 0u;
          int int_dot = 0;
          for (std::uint32_t item = 0; item < 32u; item += 1) {
            const int nibble = high ? (chunk[item] >> 4) : (chunk[item] & 0x0F);
            int_dot += nibble * quantized[block * 256u + j * 32u + item];
          }
          const std::uint32_t activation_block = block * 8u + j;
          reference += static_cast<double>(weight_d) * scale * dx[activation_block] * int_dot
            - static_cast<double>(weight_dmin) * minimum * sx[activation_block];
        }
      }
      const double difference = std::abs(static_cast<double>(actual[row]) - reference);
      maximum_difference = std::max(maximum_difference, difference);
      if (difference > 5e-3 * std::max(1.0, std::abs(reference))) {
        std::printf(
          "[FAIL] %s: row=%u expected=%f actual=%f\n",
          test_case.label,
          row,
          reference,
          actual[row]);
        return false;
      }
    }
    std::printf(
      "[ ok ] %s: %lld ms, max sample diff %g\n",
      test_case.label,
      static_cast<long long>(elapsed_ms),
      maximum_difference);

    if (test_case.rows == 10240u) {
      timing_weights = std::move(weights);
      timing_op = op;
      timing_op.weight_key = timing_weights.data();
      timing_op.weight_data = timing_weights.data();
      timing_input = std::move(input);
    }
  }

  // Throughput comparison on the FFN-gate shape: integer DP4A route versus
  // the float batch-one kernel, weights resident for both.
  static std::uintptr_t dp4a_identity_epoch = 0x9000000;
  constexpr int k_iterations = 10;
  std::vector<float> scratch;
  std::string timing_error;
  for (int warm = 0; warm < 2; warm += 1) {
    if (!d3d12_dp4a_q4k_matvec(timing_op, timing_input, &scratch, &timing_error)) {
      std::printf("[FAIL] dp4a timing warmup: %s\n", timing_error.c_str());
      return false;
    }
  }
  const auto dp4a_start = std::chrono::steady_clock::now();
  for (int iteration = 0; iteration < k_iterations; iteration += 1) {
    if (!d3d12_dp4a_q4k_matvec(timing_op, timing_input, &scratch, &timing_error)) {
      std::printf("[FAIL] dp4a timing: %s\n", timing_error.c_str());
      return false;
    }
  }
  const auto dp4a_end = std::chrono::steady_clock::now();
  for (int warm = 0; warm < 2; warm += 1) {
    dp4a_identity_epoch += 1;
    if (!d3d12_batch_matmul_resident(
      &timing_op, 1, timing_input, 1,
      reinterpret_cast<const void *>(dp4a_identity_epoch), &scratch, &timing_error)) {
      std::printf("[FAIL] float timing warmup: %s\n", timing_error.c_str());
      return false;
    }
  }
  const auto float_start = std::chrono::steady_clock::now();
  for (int iteration = 0; iteration < k_iterations; iteration += 1) {
    dp4a_identity_epoch += 1;
    if (!d3d12_batch_matmul_resident(
      &timing_op, 1, timing_input, 1,
      reinterpret_cast<const void *>(dp4a_identity_epoch), &scratch, &timing_error)) {
      std::printf("[FAIL] float timing: %s\n", timing_error.c_str());
      return false;
    }
  }
  const auto float_end = std::chrono::steady_clock::now();
  const double dp4a_ms = std::chrono::duration_cast<std::chrono::microseconds>(
    dp4a_end - dp4a_start).count() / 1000.0 / k_iterations;
  const double float_ms = std::chrono::duration_cast<std::chrono::microseconds>(
    float_end - float_start).count() / 1000.0 / k_iterations;
  std::printf(
    "[perf] dp4a q4k 10240x2560 b1: %.2f ms/iter vs float %.2f ms/iter (%.2fx)\n",
    dp4a_ms,
    float_ms,
    dp4a_ms > 0.0 ? float_ms / dp4a_ms : 0.0);
  return true;
}

// Verifies the cooperative subgroup-per-row Q4_K GEMV against the exact float
// reference dot at real Gemma decode shapes.
bool run_coop_gemv_selftest() {
  const struct CoopCase {
    const char *label;
    std::uint32_t rows;
    std::uint32_t width;
  } cases[] = {
    {"coop q4k small 128x512", 128, 512},
    {"coop q4k gemma-q 2048x2560", 2048, 2560},
    {"coop q4k gemma-gate 10240x2560", 10240, 2560},
    {"coop q4k gemma-down 2560x10240", 2560, 10240},
  };
  // The cooperative GEMV addresses weights in block order and declines under
  // the quad-interleaved layout. Skipping is the correct outcome there: a
  // kernel that refuses a layout it cannot read is behaving, not failing.
  if (quad_interleave_enabled()) {
    std::printf("[skip] coop q4k: not supported under the quad-interleaved layout\n");
    return true;
  }
  for (const CoopCase &test_case : cases) {
    d3d12_reset_resident_weights();
    std::uint32_t rng = 0xC007EDEDu
      ^ (test_case.rows * 2654435761u)
      ^ (test_case.width * 40503u);
    const std::uint64_t block_count =
      static_cast<std::uint64_t>(test_case.rows) * test_case.width / 256u;
    std::vector<std::uint8_t> weights(static_cast<std::size_t>(block_count) * 144u);
    for (std::size_t index = 0; index < weights.size(); index += 1) {
      weights[index] = static_cast<std::uint8_t>(selftest_rng(rng) & 0xFFu);
    }
    const std::uint16_t d_bits = 0x2E66;
    const std::uint16_t dmin_bits = 0x2A66;
    for (std::uint64_t block = 0; block < block_count; block += 1) {
      std::memcpy(weights.data() + block * 144u, &d_bits, sizeof(d_bits));
      std::memcpy(weights.data() + block * 144u + 2u, &dmin_bits, sizeof(dmin_bits));
    }
    std::vector<float> input(test_case.width);
    for (std::size_t index = 0; index < input.size(); index += 1) {
      input[index] =
        static_cast<float>(static_cast<std::int32_t>(selftest_rng(rng) % 2001) - 1000) / 1000.0f;
    }
    D3d12ResidentWeightRef op;
    op.weight_key = weights.data();
    op.format = D3d12WeightFormat::Q4_K;
    op.weight_data = weights.data();
    op.weight_bytes = weights.size();
    op.row_count = test_case.rows;
    op.row_width = test_case.width;

    std::vector<float> actual;
    std::string error_message;
    const auto started_at = std::chrono::steady_clock::now();
    if (!d3d12_coop_gemv_q4k_matvec(op, input, &actual, &error_message)) {
      std::printf("[FAIL] %s: %s\n", test_case.label, error_message.c_str());
      return false;
    }
    const auto finished_at = std::chrono::steady_clock::now();
    const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
      finished_at - started_at).count();
    double maximum_difference = 0.0;
    for (std::uint32_t row = 0; row < test_case.rows; row += 1) {
      const double reference = selftest_reference_row_dot(
        D3d12WeightFormat::Q4_K, weights, row, test_case.width, input.data());
      const double difference = std::abs(static_cast<double>(actual[row]) - reference);
      maximum_difference = std::max(maximum_difference, difference);
      if (difference > 5e-3 * std::max(1.0, std::abs(reference))) {
        std::printf(
          "[FAIL] %s: row=%u expected=%f actual=%f\n",
          test_case.label, row, reference, actual[row]);
        return false;
      }
    }
    std::printf(
      "[ ok ] %s: %lld ms, max sample diff %g\n",
      test_case.label, static_cast<long long>(elapsed_ms), maximum_difference);
  }
  return true;
}

// Measures the adapter's raw sequential-read ceiling with a buffer large
// enough to defeat the CPU/GPU cache hierarchy. Unlike the Q4_K timing below,
// this shader performs no dequantization or floating-point dot products, so
// the result tells us whether further kernel/layout work has enough headroom
// to reach the single-stream throughput gate.
bool run_stream_read_selftest() {
  constexpr std::uint64_t k_stream_bytes = 128ull * 1024ull * 1024ull;
  constexpr std::uint32_t k_group_count = 256;
  constexpr int k_iterations = 10;
  d3d12_reset_resident_weights();

  std::vector<std::uint32_t> words(
    static_cast<std::size_t>(k_stream_bytes / sizeof(std::uint32_t)),
    0x01010101u);
  D3d12ResidentWeightRef op;
  op.weight_key = words.data();
  op.format = D3d12WeightFormat::F32;
  op.weight_data = reinterpret_cast<const std::uint8_t *>(words.data());
  op.weight_bytes = k_stream_bytes;
  op.row_count = 1;
  op.row_width = static_cast<std::uint32_t>(words.size());

  std::string error_message;
  std::uint64_t checksum = 0;
  for (int warm = 0; warm < 2; warm += 1) {
    if (!d3d12_stream_read(op, k_group_count, &checksum, &error_message)) {
      std::printf("[FAIL] raw streaming-read warmup: %s\n", error_message.c_str());
      return false;
    }
  }
  if (checksum == 0) {
    std::printf("[FAIL] raw streaming-read checksum was zero\n");
    return false;
  }

  const auto started_at = std::chrono::steady_clock::now();
  for (int iteration = 0; iteration < k_iterations; iteration += 1) {
    if (!d3d12_stream_read(op, k_group_count, &checksum, &error_message)) {
      std::printf("[FAIL] raw streaming-read timing: %s\n", error_message.c_str());
      return false;
    }
  }
  const auto finished_at = std::chrono::steady_clock::now();
  const double elapsed_seconds =
    std::chrono::duration_cast<std::chrono::microseconds>(
      finished_at - started_at).count() / 1000000.0;
  const double gib_per_second = elapsed_seconds > 0.0
    ? (static_cast<double>(k_stream_bytes) * k_iterations / elapsed_seconds)
      / (1024.0 * 1024.0 * 1024.0)
    : 0.0;
  std::printf(
    "[perf] raw streaming-read ceiling: %.1f GB/s"
    " (128.0 MiB x %d, checksum=%llu)\n",
    gib_per_second,
    k_iterations,
    static_cast<unsigned long long>(checksum));
  return true;
}

// Verifies main_q4k compiled with 128 rows/group (instead of the default 64)
// against the exact float reference dot at real Gemma decode shapes, plus a
// row count that is NOT a multiple of 128 (100) to exercise the tail-row
// masking, then times it against the default 64-row kernel on the largest
// shape.
bool run_wide128_selftest() {
  const struct Wide128Case {
    const char *label;
    std::uint32_t rows;
    std::uint32_t width;
  } cases[] = {
    {"wide128 q4k small 128x512", 128, 512},
    {"wide128 q4k tail-rows 100x512", 100, 512},
    {"wide128 q4k gemma-q 2048x2560", 2048, 2560},
    {"wide128 q4k gemma-gate 10240x2560", 10240, 2560},
    {"wide128 q4k gemma-down 2560x10240", 2560, 10240},
  };
  std::vector<std::uint8_t> timing_weights;
  D3d12ResidentWeightRef timing_op;
  std::vector<float> timing_input;
  for (const Wide128Case &test_case : cases) {
    d3d12_reset_resident_weights();
    std::uint32_t rng = 0x00128000u
      ^ (test_case.rows * 2654435761u)
      ^ (test_case.width * 40503u);
    const std::uint64_t block_count =
      static_cast<std::uint64_t>(test_case.rows) * test_case.width / 256u;
    std::vector<std::uint8_t> weights(static_cast<std::size_t>(block_count) * 144u);
    for (std::size_t index = 0; index < weights.size(); index += 1) {
      weights[index] = static_cast<std::uint8_t>(selftest_rng(rng) & 0xFFu);
    }
    const std::uint16_t d_bits = 0x2E66;
    const std::uint16_t dmin_bits = 0x2A66;
    for (std::uint64_t block = 0; block < block_count; block += 1) {
      std::memcpy(weights.data() + block * 144u, &d_bits, sizeof(d_bits));
      std::memcpy(weights.data() + block * 144u + 2u, &dmin_bits, sizeof(dmin_bits));
    }
    std::vector<float> input(test_case.width);
    for (std::size_t index = 0; index < input.size(); index += 1) {
      input[index] =
        static_cast<float>(static_cast<std::int32_t>(selftest_rng(rng) % 2001) - 1000) / 1000.0f;
    }
    D3d12ResidentWeightRef op;
    op.weight_key = weights.data();
    op.format = D3d12WeightFormat::Q4_K;
    op.weight_data = weights.data();
    op.weight_bytes = weights.size();
    op.row_count = test_case.rows;
    op.row_width = test_case.width;

    std::vector<float> actual;
    std::string error_message;
    const auto started_at = std::chrono::steady_clock::now();
    if (!d3d12_wide128_q4k_matvec(op, input, &actual, &error_message)) {
      std::printf("[FAIL] %s: %s\n", test_case.label, error_message.c_str());
      return false;
    }
    const auto finished_at = std::chrono::steady_clock::now();
    const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
      finished_at - started_at).count();
    double maximum_difference = 0.0;
    for (std::uint32_t row = 0; row < test_case.rows; row += 1) {
      const double reference = selftest_reference_row_dot(
        D3d12WeightFormat::Q4_K, weights, row, test_case.width, input.data());
      const double difference = std::abs(static_cast<double>(actual[row]) - reference);
      maximum_difference = std::max(maximum_difference, difference);
      if (difference > 5e-3 * std::max(1.0, std::abs(reference))) {
        std::printf(
          "[FAIL] %s: row=%u expected=%f actual=%f\n",
          test_case.label, row, reference, actual[row]);
        return false;
      }
    }
    std::printf(
      "[ ok ] %s: %lld ms, max sample diff %g\n",
      test_case.label, static_cast<long long>(elapsed_ms), maximum_difference);
    if (test_case.rows == 10240u) {
      timing_weights = std::move(weights);
      timing_op = op;
      timing_op.weight_key = timing_weights.data();
      timing_op.weight_data = timing_weights.data();
      timing_input = std::move(input);
    }
  }

  // The remaining quantized decode formats share the identical
  // row-per-thread pattern, so the 128-rows-per-group variant applies to
  // them unchanged. Verify each against the CPU reference dot, including a
  // tail-row case for the row<row_count masking.
  const struct WideFormatCase {
    const char *label;
    D3d12WeightFormat format;
    std::uint32_t rows;
    std::uint32_t width;
  } format_cases[] = {
    {"wide128 q5k tail-rows 100x512", D3d12WeightFormat::Q5_K, 100, 512},
    {"wide128 q5k gemma-q 2048x2560", D3d12WeightFormat::Q5_K, 2048, 2560},
    {"wide128 q6k tail-rows 100x512", D3d12WeightFormat::Q6_K, 100, 512},
    {"wide128 q6k gemma-down 2560x10240", D3d12WeightFormat::Q6_K, 2560, 10240},
    {"wide128 q8_0 tail-rows 100x512", D3d12WeightFormat::Q8_0, 100, 512},
    {"wide128 q8_0 gemma-q 2048x2560", D3d12WeightFormat::Q8_0, 2048, 2560},
    {"wide128 q4_0 tail-rows 100x512", D3d12WeightFormat::Q4_0, 100, 512},
    {"wide128 q4_0 gemma-q 2048x2560", D3d12WeightFormat::Q4_0, 2048, 2560},
  };
  for (const WideFormatCase &test_case : format_cases) {
    d3d12_reset_resident_weights();
    std::uint32_t rng = 0x00128001u
      ^ (test_case.rows * 2654435761u)
      ^ (test_case.width * 40503u)
      ^ static_cast<std::uint32_t>(test_case.format);
    const std::uint64_t elements =
      static_cast<std::uint64_t>(test_case.rows) * test_case.width;
    std::uint64_t block_bytes = 0;
    std::uint64_t block_elements = k_quant_block_elements;
    std::uint16_t d_bits = 0x2E66;
    switch (test_case.format) {
      case D3d12WeightFormat::Q5_K: block_bytes = k_q5_k_block_bytes; break;
      case D3d12WeightFormat::Q6_K: block_bytes = k_q6_k_block_bytes; break;
      case D3d12WeightFormat::Q8_0:
        block_bytes = k_q8_0_block_bytes;
        block_elements = k_legacy_block_elements;
        d_bits = 0x2066;
        break;
      case D3d12WeightFormat::Q4_0:
        block_bytes = k_q4_0_block_bytes;
        block_elements = k_legacy_block_elements;
        break;
      default:
        std::printf("[FAIL] %s: unsupported selftest format\n", test_case.label);
        return false;
    }
    const std::uint64_t block_count = elements / block_elements;
    std::vector<std::uint8_t> weights(static_cast<std::size_t>(block_count * block_bytes));
    for (std::size_t index = 0; index < weights.size(); index += 1) {
      weights[index] = static_cast<std::uint8_t>(selftest_rng(rng) & 0xFFu);
    }
    // Overwrite the fp16 scale headers with small exact values so reference
    // and shader math stay in a well-conditioned range (mirrors
    // run_one_selftest).
    const std::uint16_t dmin_bits = 0x2A66;
    for (std::uint64_t block = 0; block < block_count; block += 1) {
      std::uint8_t *block_ptr = weights.data() + block * block_bytes;
      if (test_case.format == D3d12WeightFormat::Q5_K) {
        std::memcpy(block_ptr, &d_bits, sizeof(d_bits));
        std::memcpy(block_ptr + 2, &dmin_bits, sizeof(dmin_bits));
      } else if (test_case.format == D3d12WeightFormat::Q6_K) {
        std::memcpy(block_ptr + k_q6_k_block_bytes - 2, &d_bits, sizeof(d_bits));
      } else {
        std::memcpy(block_ptr, &d_bits, sizeof(d_bits));
      }
    }
    std::vector<float> input(test_case.width);
    for (std::size_t index = 0; index < input.size(); index += 1) {
      input[index] =
        static_cast<float>(static_cast<std::int32_t>(selftest_rng(rng) % 2001) - 1000) / 1000.0f;
    }
    D3d12ResidentWeightRef op;
    op.weight_key = weights.data();
    op.format = test_case.format;
    op.weight_data = weights.data();
    op.weight_bytes = weights.size();
    op.row_count = test_case.rows;
    op.row_width = test_case.width;

    std::vector<float> actual;
    std::string error_message;
    if (!d3d12_wide128_q4k_matvec(op, input, &actual, &error_message)) {
      std::printf("[FAIL] %s: %s\n", test_case.label, error_message.c_str());
      return false;
    }
    double maximum_difference = 0.0;
    for (std::uint32_t row = 0; row < test_case.rows; row += 1) {
      const double reference = selftest_reference_row_dot(
        test_case.format, weights, row, test_case.width, input.data());
      const double difference = std::abs(static_cast<double>(actual[row]) - reference);
      maximum_difference = std::max(maximum_difference, difference);
      if (difference > 5e-3 * std::max(1.0, std::abs(reference))) {
        std::printf(
          "[FAIL] %s: row=%u expected=%f actual=%f\n",
          test_case.label, row, reference, actual[row]);
        return false;
      }
    }
    std::printf(
      "[ ok ] %s: max sample diff %g\n", test_case.label, maximum_difference);
  }

  // Throughput comparison on the FFN-gate shape: 128-rows-per-group versus
  // the default 64-rows-per-group kernel.
  static std::uintptr_t wide128_identity_epoch = 0xC000000;
  constexpr int k_iterations = 10;
  std::vector<float> scratch;
  std::string timing_error;
  for (int warm = 0; warm < 2; warm += 1) {
    if (!d3d12_wide128_q4k_matvec(timing_op, timing_input, &scratch, &timing_error)) {
      std::printf("[FAIL] wide128 timing warmup: %s\n", timing_error.c_str());
      return false;
    }
  }
  const auto wide128_start = std::chrono::steady_clock::now();
  for (int iteration = 0; iteration < k_iterations; iteration += 1) {
    if (!d3d12_wide128_q4k_matvec(timing_op, timing_input, &scratch, &timing_error)) {
      std::printf("[FAIL] wide128 timing: %s\n", timing_error.c_str());
      return false;
    }
  }
  const auto wide128_end = std::chrono::steady_clock::now();
  for (int warm = 0; warm < 2; warm += 1) {
    wide128_identity_epoch += 1;
    if (!d3d12_batch_matmul_resident(
      &timing_op, 1, timing_input, 1,
      reinterpret_cast<const void *>(wide128_identity_epoch), &scratch, &timing_error)) {
      std::printf("[FAIL] float64 timing warmup: %s\n", timing_error.c_str());
      return false;
    }
  }
  const auto float64_start = std::chrono::steady_clock::now();
  for (int iteration = 0; iteration < k_iterations; iteration += 1) {
    wide128_identity_epoch += 1;
    if (!d3d12_batch_matmul_resident(
      &timing_op, 1, timing_input, 1,
      reinterpret_cast<const void *>(wide128_identity_epoch), &scratch, &timing_error)) {
      std::printf("[FAIL] float64 timing: %s\n", timing_error.c_str());
      return false;
    }
  }
  const auto float64_end = std::chrono::steady_clock::now();
  const double wide128_ms = std::chrono::duration_cast<std::chrono::microseconds>(
    wide128_end - wide128_start).count() / 1000.0 / k_iterations;
  const double float64_ms = std::chrono::duration_cast<std::chrono::microseconds>(
    float64_end - float64_start).count() / 1000.0 / k_iterations;
  std::printf(
    "[perf] wide128 q4k 10240x2560 b1: %.2f ms/iter vs 64-rows %.2f ms/iter (%.2fx)\n",
    wide128_ms,
    float64_ms,
    wide128_ms > 0.0 ? float64_ms / wide128_ms : 0.0);

  // A B=1 matvec reads the whole weight tensor exactly once, so ms/iter is an
  // achieved-memory-bandwidth measurement in disguise. Printing it as GB/s is
  // what makes the number comparable to anything: the decode rate a user sees
  // is this figure divided by the model's byte size, and the campaign in
  // .Agent/enhancementRun_user_speed_01 reduces entirely to raising it. Quote
  // this rather than tok/s when judging a kernel change - it is independent of
  // model size and of the +/-4% end-to-end benchmark noise.
  const double timing_bytes = static_cast<double>(timing_op.weight_bytes);
  const auto to_gbs = [&](double ms) {
    return ms > 0.0 ? (timing_bytes / (ms / 1000.0)) / (1024.0 * 1024.0 * 1024.0) : 0.0;
  };
  std::printf(
    "[perf] q4k b1 achieved bandwidth: wide128 %.1f GB/s, 64-rows %.1f GB/s"
    " (%.1f MiB swept per iteration)\n",
    to_gbs(wide128_ms),
    to_gbs(float64_ms),
    timing_bytes / (1024.0 * 1024.0));

  // The ceiling: the same byte volume read with no dequantisation at all. If
  // this lands close to the matvec figure above, the kernel is already at the
  // hardware limit and no rewrite of it will close a performance gap; if it is
  // far above, the kernel is leaving bandwidth on the table and is worth work.
  double best_stream_ms = 0.0;
  std::string stream_error;
  for (const std::uint32_t groups : {16u, 32u, 64u, 128u, 256u}) {
    double stream_ms = 0.0;
    if (!d3d12_stream_read_benchmark(
        static_cast<std::uint64_t>(timing_bytes), k_iterations, groups, 0u, &stream_ms, &stream_error)) {
      std::printf("[perf] pure streaming read ceiling: unavailable (%s)\n", stream_error.c_str());
      best_stream_ms = 0.0;
      break;
    }
    std::printf("[perf]   stream read @ %3u groups: %.1f GB/s\n", groups, to_gbs(stream_ms));
    if (best_stream_ms == 0.0 || stream_ms < best_stream_ms) {
      best_stream_ms = stream_ms;
    }
  }
  // The sweep above uses the 14 MiB timing tensor, which partly fits in cache
  // and therefore overstates what a real decode can reach: a token sweeps the
  // whole model, ~2 GB, with no reuse at all. Re-measure at a working set far
  // past any cache so the number can be compared against the performance goal
  // honestly.
  {
    constexpr std::uint64_t k_large_bytes = 512ull * 1024ull * 1024ull;
    double large_ms = 0.0;
    std::string large_error;
    if (d3d12_stream_read_benchmark(k_large_bytes, 3, 64u, 0u, &large_ms, &large_error)) {
      const double large_gbs = large_ms > 0.0
        ? (static_cast<double>(k_large_bytes) / (large_ms / 1000.0)) / (1024.0 * 1024.0 * 1024.0)
        : 0.0;
      std::printf(
        "[perf] streaming ceiling, 512 MiB working set: %.1f GB/s"
        " (the honest ceiling for a full-model sweep)\n",
        large_gbs);
    } else {
      std::printf("[perf] 512 MiB ceiling: unavailable (%s)\n", large_error.c_str());
    }
  }

  if (best_stream_ms > 0.0) {
    const double stream_gbs = to_gbs(best_stream_ms);
    const double matvec_gbs = to_gbs(wide128_ms);
    std::printf(
      "[perf] pure streaming read ceiling: %.1f GB/s best of sweep —"
      " q4k matvec reaches %.0f%% of it\n",
      stream_gbs,
      stream_gbs > 0.0 ? (matvec_gbs / stream_gbs) * 100.0 : 0.0);

    // The decisive A/B: the SAME kernel over the SAME bytes, reading with the
    // matvec's row-per-thread stride instead of contiguously. Everything else
    // - dequantisation, scales, accumulation, occupancy - is identical and
    // absent from both. Whatever separates these two numbers is the access
    // pattern and nothing else.
    // Load-only twin: same addressing, same loads, no dequant arithmetic.
    // Its gap to the real matvec is the ALU cost; its gap to the contiguous
    // ceiling is what this access shape costs even when interleaved.
    {
      std::vector<float> loadonly_out;
      std::string loadonly_error;
      bool loadonly_ok = true;
      for (int warm = 0; warm < 2 && loadonly_ok; warm += 1) {
        loadonly_ok = d3d12_loadonly_q4k_matvec(
          timing_op, timing_input, &loadonly_out, &loadonly_error);
      }
      const auto loadonly_start = std::chrono::steady_clock::now();
      for (int iteration = 0; iteration < k_iterations && loadonly_ok; iteration += 1) {
        loadonly_ok = d3d12_loadonly_q4k_matvec(
          timing_op, timing_input, &loadonly_out, &loadonly_error);
      }
      const auto loadonly_end = std::chrono::steady_clock::now();
      if (loadonly_ok) {
        const double loadonly_ms = std::chrono::duration_cast<std::chrono::microseconds>(
          loadonly_end - loadonly_start).count() / 1000.0 / k_iterations;
        const double loadonly_gbs = to_gbs(loadonly_ms);
        std::printf(
          "[perf] q4k load-only (no dequant): %.1f GB/s — dequant costs %.0f%%,"
          " access shape costs %.0f%% of ceiling\n",
          loadonly_gbs,
          loadonly_gbs > 0.0 ? (1.0 - matvec_gbs / loadonly_gbs) * 100.0 : 0.0,
          stream_gbs > 0.0 ? (1.0 - loadonly_gbs / stream_gbs) * 100.0 : 0.0);
      } else {
        std::printf("[perf] q4k load-only: unavailable (%s)\n", loadonly_error.c_str());
      }

      // Same bytes, same kernel shape, addressed as if quad-interleaved.
      std::vector<float> coalesced_out;
      std::string coalesced_error;
      bool coalesced_ok = true;
      for (int warm = 0; warm < 2 && coalesced_ok; warm += 1) {
        coalesced_ok = d3d12_loadonly_q4k_matvec(
          timing_op, timing_input, &coalesced_out, &coalesced_error, true);
      }
      const auto coalesced_start = std::chrono::steady_clock::now();
      for (int iteration = 0; iteration < k_iterations && coalesced_ok; iteration += 1) {
        coalesced_ok = d3d12_loadonly_q4k_matvec(
          timing_op, timing_input, &coalesced_out, &coalesced_error, true);
      }
      const auto coalesced_end = std::chrono::steady_clock::now();
      if (coalesced_ok) {
        const double coalesced_ms = std::chrono::duration_cast<std::chrono::microseconds>(
          coalesced_end - coalesced_start).count() / 1000.0 / k_iterations;
        std::printf(
          "[perf] q4k load-only, quad-interleaved addressing: %.1f GB/s"
          " (%.0f%% of ceiling)\n",
          to_gbs(coalesced_ms),
          stream_gbs > 0.0 ? (to_gbs(coalesced_ms) / stream_gbs) * 100.0 : 0.0);
      }
    }

    const std::uint32_t row_quads = static_cast<std::uint32_t>(
      (timing_op.row_width / 256u) * 144u / 16u);
    double strided_ms = 0.0;
    std::string strided_error;
    if (row_quads > 0
      && d3d12_stream_read_benchmark(
        static_cast<std::uint64_t>(timing_bytes), k_iterations, 64u, row_quads,
        &strided_ms, &strided_error)) {
      const double strided_gbs = to_gbs(strided_ms);
      std::printf(
        "[perf] same kernel, matvec stride (%u quads/row): %.1f GB/s"
        " — %.0f%% of contiguous\n",
        row_quads,
        strided_gbs,
        stream_gbs > 0.0 ? (strided_gbs / stream_gbs) * 100.0 : 0.0);
    }
  }

  // Quick check: does widening further to 256 rows/group keep paying off?
  std::vector<float> wide256_actual;
  std::string wide256_error;
  if (!d3d12_wide256_q4k_matvec(timing_op, timing_input, &wide256_actual, &wide256_error)) {
    std::printf("[FAIL] wide256 q4k gemma-gate 10240x2560: %s\n", wide256_error.c_str());
    return false;
  }
  double wide256_max_diff = 0.0;
  for (std::uint32_t row = 0; row < timing_op.row_count; row += 1) {
    const double reference = selftest_reference_row_dot(
      D3d12WeightFormat::Q4_K, timing_weights, row, timing_op.row_width, timing_input.data());
    const double difference = std::abs(static_cast<double>(wide256_actual[row]) - reference);
    wide256_max_diff = std::max(wide256_max_diff, difference);
    if (difference > 5e-3 * std::max(1.0, std::abs(reference))) {
      std::printf(
        "[FAIL] wide256 q4k gemma-gate 10240x2560: row=%u expected=%f actual=%f\n",
        row, reference, wide256_actual[row]);
      return false;
    }
  }
  std::printf(
    "[ ok ] wide256 q4k gemma-gate 10240x2560: max sample diff %g\n", wide256_max_diff);

  // Sustained ALU ceiling first, so the prefill figure below can be quoted
  // against a rate this machine actually reaches instead of a datasheet one.
  double fma_peak_gflops = 0.0;
  {
    const std::uint32_t k_fma_iterations = 4096;
    const std::uint32_t k_fma_groups = 512;
    double fma_ms = 0.0;
    std::string fma_error;
    if (d3d12_fma_peak_benchmark(k_fma_iterations, k_fma_groups, 4, &fma_ms, &fma_error)) {
      const double flops = 2.0 * 64.0 * k_fma_iterations
        * static_cast<double>(k_fma_groups) * 256.0;
      fma_peak_gflops = flops / (fma_ms / 1000.0) / 1e9;
      std::printf("[perf] sustained ALU ceiling (pure FMA, no memory): %.0f GFLOP/s\n",
        fma_peak_gflops);
    } else {
      std::printf("[skip] sustained ALU ceiling: %s\n", fma_error.c_str());
    }
  }

  // Batched Q4_K at a prefill-shaped batch. Everything else in this selftest
  // measures batch-one matvecs, which is decode - but prefill is a GEMM, it is
  // 82% of time-to-first-token, and it had no kernel-level gate at all. Runs
  // through d3d12_batch_matmul_resident so whichever kernel the row-block gate
  // selects is what gets measured; set LIZARD_NATIVE_GPU_ROW_BLOCK to A/B it on
  // one binary without needing a model or the residency claim.
  {
    constexpr std::uint32_t k_prefill_batch = 48;
    const std::uint32_t width = timing_op.row_width;
    std::vector<float> batched_inputs(
      static_cast<std::size_t>(k_prefill_batch) * width, 0.0f);
    std::uint32_t input_rng = 0x51A7C0DEu;
    for (std::size_t index = 0; index < batched_inputs.size(); index += 1) {
      input_rng = input_rng * 1664525u + 1013904223u;
      batched_inputs[index] = static_cast<float>((input_rng >> 9) & 0x3FFu) / 1023.0f - 0.5f;
      // Same exactly-representable-in-Q8_1 mode the small cases use, so the
      // integer prefill path can be checked for arithmetic bugs without the
      // quantisation loss masquerading as one.
      if (selftest_exact_q8_1_inputs()) {
        batched_inputs[index] = index % 32u == 0
          ? 1.0f
          : static_cast<float>(static_cast<std::int32_t>((input_rng >> 7) % 255u) - 127) / 127.0f;
      }
    }
    std::vector<std::vector<float>> batched_outputs(1);
    std::string batched_error;
    const std::uint32_t row_block = gpu_row_block();
    if (!d3d12_batch_matmul_resident(
      &timing_op, 1, batched_inputs, k_prefill_batch,
      batched_inputs.data(), batched_outputs.data(), &batched_error)) {
      std::printf("[FAIL] batched q4k prefill 10240x2560 b%u: %s\n",
        k_prefill_batch, batched_error.c_str());
      return false;
    }
    // Correctness on a sampled set of (column, row) pairs against the scalar
    // reference. A row-blocked kernel changes which thread owns a row, so a
    // wrong row/stride mapping shows up here and nowhere else.
    double batched_max_diff = 0.0;
    for (std::uint32_t column = 0; column < k_prefill_batch; column += 7) {
      for (std::uint32_t row = 0; row < timing_op.row_count; row += 419) {
        const double reference = selftest_reference_row_dot(
          D3d12WeightFormat::Q4_K, timing_weights, row, width,
          batched_inputs.data() + static_cast<std::size_t>(column) * width);
        const double actual = static_cast<double>(
          batched_outputs[0][static_cast<std::size_t>(column) * timing_op.row_count + row]);
        const double difference = std::abs(actual - reference);
        batched_max_diff = std::max(batched_max_diff, difference);
        if (difference > 5e-3 * std::max(1.0, std::abs(reference))) {
          std::printf(
            "[FAIL] batched q4k prefill b%u rows=%u: column=%u row=%u expected=%f actual=%f\n",
            k_prefill_batch, row_block, column, row, reference, actual);
          return false;
        }
      }
    }
    std::printf(
      "[ ok ] batched q4k prefill 10240x2560 b%u (rows/thread %u): max sample diff %g\n",
      k_prefill_batch, row_block, batched_max_diff);


    for (int warm = 0; warm < 2; warm += 1) {
      if (!d3d12_batch_matmul_resident(
        &timing_op, 1, batched_inputs, k_prefill_batch,
        batched_inputs.data(), batched_outputs.data(), &batched_error)) {
        std::printf("[FAIL] batched q4k prefill timing warmup: %s\n", batched_error.c_str());
        return false;
      }
    }
    const auto batched_start = std::chrono::steady_clock::now();
    // 24, not 8. Run-to-run spread at 8 was 186-233 GFLOP/s on the same binary,
    // which is wider than any effect worth measuring here - this machine shares a
    // power budget between CPU and iGPU, so clocks move between runs. More
    // averaging per reported number is what makes a 10% change legible.
    constexpr int k_batched_iterations = 24;
    for (int iteration = 0; iteration < k_batched_iterations; iteration += 1) {
      if (!d3d12_batch_matmul_resident(
        &timing_op, 1, batched_inputs, k_prefill_batch,
        batched_inputs.data(), batched_outputs.data(), &batched_error)) {
        std::printf("[FAIL] batched q4k prefill timing: %s\n", batched_error.c_str());
        return false;
      }
    }
    const auto batched_end = std::chrono::steady_clock::now();
    const double batched_ms = std::chrono::duration_cast<std::chrono::microseconds>(
      batched_end - batched_start).count() / 1000.0 / k_batched_iterations;
    // 2 flops per multiply-accumulate, one per (row, column, element).
    const double gflops = (2.0 * timing_op.row_count * width * k_prefill_batch)
      / (batched_ms / 1000.0) / 1e9;
    std::printf(
      "[perf] batched q4k prefill b%u, rows/thread %u: %.2f ms/iter, %.0f GFLOP/s\n",
      k_prefill_batch, row_block, batched_ms, gflops);
    if (fma_peak_gflops > 0.0) {
      std::printf("[perf] prefill reaches %.0f%% of the MEASURED ALU ceiling (%.0f of %.0f)\n",
        gflops / fma_peak_gflops * 100.0, gflops, fma_peak_gflops);
    }

    // Paired A/B: row-blocked kernel against the plain one, alternating inside
    // this process so each pair sees the same clocks. Between-process comparison
    // cannot resolve anything under ~25% on this machine, which is how four
    // earlier hypotheses produced confident numbers that led nowhere.
    if (gpu_row_block() > 1) {
      const int k_pairs = 5;
      const int k_reps = 8;
      std::vector<double> deltas;
      double rowblock_total = 0.0;
      double plain_total = 0.0;
      bool ab_ok = true;
      for (int pair = 0; pair < k_pairs && ab_ok; pair += 1) {
        double rowblock_ms = 0.0;
        double plain_ms = 0.0;
        std::string ab_error;
        // Order alternates so a warming trend cannot favour one side.
        const bool rowblock_first = (pair % 2) == 0;
        for (int half = 0; half < 2 && ab_ok; half += 1) {
          const bool want_rowblock = (half == 0) == rowblock_first;
          double ms = 0.0;
          ab_ok = d3d12_batched_ab_benchmark(
            timing_op, batched_inputs, k_prefill_batch, want_rowblock ? 1 : 0,
            k_reps, &ms, &ab_error);
          if (!ab_ok) {
            std::printf("[skip] paired A/B: %s\n", ab_error.c_str());
            break;
          }
          if (want_rowblock) rowblock_ms = ms; else plain_ms = ms;
        }
        if (!ab_ok) break;
        rowblock_total += rowblock_ms;
        plain_total += plain_ms;
        deltas.push_back(plain_ms / rowblock_ms);
      }
      if (ab_ok && !deltas.empty()) {
        std::sort(deltas.begin(), deltas.end());
        const double median = deltas[deltas.size() / 2];
        std::printf(
          "[perf] paired A/B over %d pairs: rowblock %.2f ms vs plain %.2f ms"
          " (mean of pairs)\n",
          static_cast<int>(deltas.size()),
          rowblock_total / deltas.size(),
          plain_total / deltas.size());
        std::printf(
          "[perf] paired speedup from row blocking: median %.3fx, range %.3f-%.3f\n",
          median, deltas.front(), deltas.back());
      }

      // Same paired method applied to the batch tile. The earlier verdict that a
      // wider tile loses came from between-process runs whose spread was larger
      // than the difference being claimed, so it did not settle anything.
      if (gpu_rowblock_alt_tile_enabled() || gpu_rowblock_alt_rows() > 0u) {
        std::vector<double> tile_deltas;
        double primary_total = 0.0;
        double alt_total = 0.0;
        bool tile_ok = true;
        for (int pair = 0; pair < k_pairs && tile_ok; pair += 1) {
          double primary_ms = 0.0;
          double alt_ms = 0.0;
          std::string tile_error;
          const bool primary_first = (pair % 2) == 0;
          for (int half = 0; half < 2 && tile_ok; half += 1) {
            const bool want_primary = (half == 0) == primary_first;
            double ms = 0.0;
            tile_ok = d3d12_batched_ab_benchmark(
              timing_op, batched_inputs, k_prefill_batch, want_primary ? 1 : 2,
              k_reps, &ms, &tile_error);
            if (!tile_ok) {
              std::printf("[skip] paired tile A/B: %s\n", tile_error.c_str());
              break;
            }
            if (want_primary) primary_ms = ms; else alt_ms = ms;
          }
          if (!tile_ok) break;
          primary_total += primary_ms;
          alt_total += alt_ms;
          tile_deltas.push_back(primary_ms / alt_ms);
        }
        if (tile_ok && !tile_deltas.empty()) {
          std::sort(tile_deltas.begin(), tile_deltas.end());
          const double tile_median = tile_deltas[tile_deltas.size() / 2];
          const bool rows_ab = gpu_rowblock_alt_rows() > 0u;
          if (rows_ab) {
            std::printf(
              "[perf] paired ROWS A/B: rows %u %.2f ms vs rows %u %.2f ms\n",
              gpu_row_block(), primary_total / tile_deltas.size(),
              gpu_rowblock_alt_rows(), alt_total / tile_deltas.size());
          } else {
            std::printf(
              "[perf] paired TILE A/B: tile %u %.2f ms vs tile %u %.2f ms\n",
              gpu_prefill_tile(), primary_total / tile_deltas.size(),
              gpu_prefill_tile() == 8u ? 16u : 8u, alt_total / tile_deltas.size());
          }
          std::printf(
            "[perf] alt is %.3fx the primary (>1 means the ALT is faster),"
            " range %.3f-%.3f\n",
            tile_median, tile_deltas.front(), tile_deltas.back());
        }
      }

      // Integer-dot (dp4a) GEMM against the float row-blocked kernel. This is the
      // structural lever: dot4add_i8packed does four multiply-accumulates per
      // instruction, so it raises work per instruction rather than reshaping the
      // same scalar arithmetic - which is what every constant in the float kernel
      // has now been shown unable to do.
      {
        double dp4a_ms = 0.0;
        std::string dp4a_error;
        if (!d3d12_batched_ab_benchmark(
            timing_op, batched_inputs, k_prefill_batch, 3, 4, &dp4a_ms, &dp4a_error)) {
          std::printf("[skip] integer-dot GEMM: %s\n", dp4a_error.c_str());
        } else {
          // Correctness first: dp4a quantises the activations to int8, so this is
          // a lossy path and is judged on relative error, like the b1 dp4a cases.
          std::vector<std::vector<float>> dp4a_out(1);
          std::string run_error;
          double worst_relative = 0.0;
          if (d3d12_batched_ab_benchmark(
              timing_op, batched_inputs, k_prefill_batch, 3, 1, &dp4a_ms, &run_error)) {
            // Re-run through the normal path to read the outputs back.
            std::vector<std::vector<float>> reference_out(1);
            std::string ref_error;
            if (d3d12_batch_matmul_resident(
                &timing_op, 1, batched_inputs, k_prefill_batch,
                batched_inputs.data(), reference_out.data(), &ref_error)) {
              for (std::uint32_t column = 0; column < k_prefill_batch; column += 11) {
                for (std::uint32_t row = 0; row < timing_op.row_count; row += 631) {
                  const double expected = selftest_reference_row_dot(
                    D3d12WeightFormat::Q4_K, timing_weights, row, width,
                    batched_inputs.data() + static_cast<std::size_t>(column) * width);
                  const std::size_t index =
                    static_cast<std::size_t>(column) * timing_op.row_count + row;
                  const double actual = static_cast<double>(reference_out[0][index]);
                  const double denominator = std::max(1.0, std::abs(expected));
                  worst_relative = std::max(
                    worst_relative, std::abs(actual - expected) / denominator);
                }
              }
            }
          }
          const double dp4a_gflops =
            (2.0 * timing_op.row_count * width * k_prefill_batch) / (dp4a_ms / 1000.0) / 1e9;
          std::printf(
            "[perf] integer-dot GEMM b%u: %.2f ms/iter, %.0f GFLOP/s\n",
            k_prefill_batch, dp4a_ms, dp4a_gflops);
          // Ablation: same arithmetic, no activation path at all. The gap
          // between this and the real integer kernel IS the activation cost.
          double nolds_ms = 0.0;
          std::string nolds_error;
          if (d3d12_batched_ab_benchmark(
              timing_op, batched_inputs, k_prefill_batch, 4, 4, &nolds_ms, &nolds_error)) {
            const double nolds_gflops =
              (2.0 * timing_op.row_count * width * k_prefill_batch) / (nolds_ms / 1000.0) / 1e9;
            std::printf(
              "[perf] ABLATION (same dots, no activation path): %.2f ms, %.0f GFLOP/s\n",
              nolds_ms, nolds_gflops);
            std::printf(
              "[perf] activation path costs %.2f ms of %.2f ms (%.0f%% of the kernel)\n",
              dp4a_ms - nolds_ms, dp4a_ms, (dp4a_ms - nolds_ms) / dp4a_ms * 100.0);
          } else {
            std::printf("[skip] ablation: %s\n", nolds_error.c_str());
          }
          std::printf(
            "[perf] integer-dot vs float row-blocked: %.3fx"
            " (>1 means the integer path is faster)\n",
            (rowblock_total / std::max<std::size_t>(1, deltas.size())) / dp4a_ms);
        }
      }
    }
  }
  for (int warm = 0; warm < 2; warm += 1) {
    if (!d3d12_wide256_q4k_matvec(timing_op, timing_input, &scratch, &timing_error)) {
      std::printf("[FAIL] wide256 timing warmup: %s\n", timing_error.c_str());
      return false;
    }
  }
  const auto wide256_start = std::chrono::steady_clock::now();
  for (int iteration = 0; iteration < k_iterations; iteration += 1) {
    if (!d3d12_wide256_q4k_matvec(timing_op, timing_input, &scratch, &timing_error)) {
      std::printf("[FAIL] wide256 timing: %s\n", timing_error.c_str());
      return false;
    }
  }
  const auto wide256_end = std::chrono::steady_clock::now();
  const double wide256_ms = std::chrono::duration_cast<std::chrono::microseconds>(
    wide256_end - wide256_start).count() / 1000.0 / k_iterations;
  std::printf(
    "[perf] wide256 q4k 10240x2560 b1: %.2f ms/iter vs wide128 %.2f ms/iter (%.2fx)\n",
    wide256_ms,
    wide128_ms,
    wide256_ms > 0.0 ? wide128_ms / wide256_ms : 0.0);
  return true;
}

// Verifies the DXC/SM6.0 WaveActiveSum cooperative Q4_K GEMV against the exact
// float reference dot at real Gemma decode shapes, then times it against the
// float batch-one kernel on the largest shape. Skips (successfully) when DXIL
// kernels are unavailable on this build/device.
bool run_wave_gemv_selftest() {
  const D3d12Dp4aCapabilities &dp4a = d3d12_dp4a_capabilities();
  if (!dp4a.supported) {
    std::printf("[skip] wave gemv q4k: %s\n", dp4a.reason.c_str());
    return true;
  }
  const struct WaveCase {
    const char *label;
    std::uint32_t rows;
    std::uint32_t width;
  } cases[] = {
    {"wave q4k 1blk 128x256", 128, 256},
    {"wave q4k small 128x512", 128, 512},
    {"wave q4k gemma-q 2048x2560", 2048, 2560},
    {"wave q4k gemma-gate 10240x2560", 10240, 2560},
    {"wave q4k gemma-down 2560x10240", 2560, 10240},
  };
  std::vector<std::uint8_t> timing_weights;
  D3d12ResidentWeightRef timing_op;
  std::vector<float> timing_input;
  // Same reasoning as the cooperative GEMV: a kernel that refuses a layout
  // it cannot address is behaving correctly, not failing.
  if (quad_interleave_enabled()) {
    std::printf("[skip] wave q4k: not supported under the quad-interleaved layout\n");
    return true;
  }
  for (const WaveCase &test_case : cases) {
    d3d12_reset_resident_weights();
    std::uint32_t rng = 0x00A7E123u
      ^ (test_case.rows * 2654435761u)
      ^ (test_case.width * 40503u);
    const std::uint64_t block_count =
      static_cast<std::uint64_t>(test_case.rows) * test_case.width / 256u;
    std::vector<std::uint8_t> weights(static_cast<std::size_t>(block_count) * 144u);
    for (std::size_t index = 0; index < weights.size(); index += 1) {
      weights[index] = static_cast<std::uint8_t>(selftest_rng(rng) & 0xFFu);
    }
    const std::uint16_t d_bits = 0x2E66;
    const std::uint16_t dmin_bits = 0x2A66;
    for (std::uint64_t block = 0; block < block_count; block += 1) {
      std::memcpy(weights.data() + block * 144u, &d_bits, sizeof(d_bits));
      std::memcpy(weights.data() + block * 144u + 2u, &dmin_bits, sizeof(dmin_bits));
    }
    std::vector<float> input(test_case.width);
    for (std::size_t index = 0; index < input.size(); index += 1) {
      input[index] =
        static_cast<float>(static_cast<std::int32_t>(selftest_rng(rng) % 2001) - 1000) / 1000.0f;
    }
    D3d12ResidentWeightRef op;
    op.weight_key = weights.data();
    op.format = D3d12WeightFormat::Q4_K;
    op.weight_data = weights.data();
    op.weight_bytes = weights.size();
    op.row_count = test_case.rows;
    op.row_width = test_case.width;

    std::vector<float> actual;
    std::string error_message;
    const auto started_at = std::chrono::steady_clock::now();
    if (!d3d12_wave_gemv_q4k_matvec(op, input, &actual, &error_message)) {
      std::printf("[FAIL] %s: %s\n", test_case.label, error_message.c_str());
      return false;
    }
    const auto finished_at = std::chrono::steady_clock::now();
    const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
      finished_at - started_at).count();
    double maximum_difference = 0.0;
    for (std::uint32_t row = 0; row < test_case.rows; row += 1) {
      const double reference = selftest_reference_row_dot(
        D3d12WeightFormat::Q4_K, weights, row, test_case.width, input.data());
      const double difference = std::abs(static_cast<double>(actual[row]) - reference);
      maximum_difference = std::max(maximum_difference, difference);
      if (difference > 5e-3 * std::max(1.0, std::abs(reference))) {
        std::printf(
          "[FAIL] %s: row=%u expected=%f actual=%f\n",
          test_case.label, row, reference, actual[row]);
        return false;
      }
    }
    std::printf(
      "[ ok ] %s: %lld ms, max sample diff %g\n",
      test_case.label, static_cast<long long>(elapsed_ms), maximum_difference);
    if (test_case.rows == 10240u) {
      timing_weights = std::move(weights);
      timing_op = op;
      timing_op.weight_key = timing_weights.data();
      timing_op.weight_data = timing_weights.data();
      timing_input = std::move(input);
    }
  }

  // Throughput comparison on the FFN-gate shape: wave-reduction cooperative
  // GEMV versus the float one-thread-per-row batch-one kernel.
  static std::uintptr_t wave_identity_epoch = 0xA000000;
  constexpr int k_iterations = 10;
  std::vector<float> scratch;
  std::string timing_error;
  for (int warm = 0; warm < 2; warm += 1) {
    if (!d3d12_wave_gemv_q4k_matvec(timing_op, timing_input, &scratch, &timing_error)) {
      std::printf("[FAIL] wave timing warmup: %s\n", timing_error.c_str());
      return false;
    }
  }
  const auto wave_start = std::chrono::steady_clock::now();
  for (int iteration = 0; iteration < k_iterations; iteration += 1) {
    if (!d3d12_wave_gemv_q4k_matvec(timing_op, timing_input, &scratch, &timing_error)) {
      std::printf("[FAIL] wave timing: %s\n", timing_error.c_str());
      return false;
    }
  }
  const auto wave_end = std::chrono::steady_clock::now();
  for (int warm = 0; warm < 2; warm += 1) {
    wave_identity_epoch += 1;
    if (!d3d12_batch_matmul_resident(
      &timing_op, 1, timing_input, 1,
      reinterpret_cast<const void *>(wave_identity_epoch), &scratch, &timing_error)) {
      std::printf("[FAIL] float timing warmup: %s\n", timing_error.c_str());
      return false;
    }
  }
  const auto float_start = std::chrono::steady_clock::now();
  for (int iteration = 0; iteration < k_iterations; iteration += 1) {
    wave_identity_epoch += 1;
    if (!d3d12_batch_matmul_resident(
      &timing_op, 1, timing_input, 1,
      reinterpret_cast<const void *>(wave_identity_epoch), &scratch, &timing_error)) {
      std::printf("[FAIL] float timing: %s\n", timing_error.c_str());
      return false;
    }
  }
  const auto float_end = std::chrono::steady_clock::now();
  const double wave_ms = std::chrono::duration_cast<std::chrono::microseconds>(
    wave_end - wave_start).count() / 1000.0 / k_iterations;
  const double float_ms = std::chrono::duration_cast<std::chrono::microseconds>(
    float_end - float_start).count() / 1000.0 / k_iterations;
  std::printf(
    "[perf] wave gemv q4k 10240x2560 b1: %.2f ms/iter vs float %.2f ms/iter (%.2fx)\n",
    wave_ms,
    float_ms,
    wave_ms > 0.0 ? float_ms / wave_ms : 0.0);
  return true;
}

int d3d12_matmul_selftest() {
  const D3d12BatchMatmulCapabilities &capabilities = d3d12_batch_matmul_capabilities();
  std::printf(
    "D3D12 selftest: supported=%d adapter=%s\n",
    capabilities.supported ? 1 : 0,
    capabilities.adapter_name.c_str());
  const D3d12Dp4aCapabilities &dp4a = d3d12_dp4a_capabilities();
  std::printf(
    "D3D12 DP4A (SM 6.4): %s (%s)\n",
    dp4a.supported ? "available" : "unavailable",
    dp4a.reason.c_str());
  if (!capabilities.supported) {
    std::printf("reason: %s\n", capabilities.reason.c_str());
    return 1;
  }

  struct SelftestCase {
    const char *label;
    D3d12WeightFormat format;
    std::uint32_t row_count;
    std::uint32_t row_width;
    std::uint32_t batch_count;
    int q4k_scale_variant = 0;
  };
  const SelftestCase cases[] = {
    {"f32 small 128x512 b3", D3d12WeightFormat::F32, 128, 512, 3},
    {"f32 decode 128x512 b1", D3d12WeightFormat::F32, 128, 512, 1},
    {"f32 small 256x512 b5", D3d12WeightFormat::F32, 256, 512, 5},
    {"f16 small 256x512 b5", D3d12WeightFormat::F16, 256, 512, 5},
    {"bf16 decode 256x512 b1", D3d12WeightFormat::BF16, 256, 512, 1},
    {"bf16 small 256x512 b5", D3d12WeightFormat::BF16, 256, 512, 5},
    {"q4_k 1blk 128x256 b5", D3d12WeightFormat::Q4_K, 128, 256, 5},
    {"q4_k 2blk b1 128x512 b1", D3d12WeightFormat::Q4_K, 128, 512, 1},
    {"q4_k small 128x512 b5", D3d12WeightFormat::Q4_K, 128, 512, 5},
    {"q4_k lo-only 128x512 b5", D3d12WeightFormat::Q4_K, 128, 512, 5, 1},
    {"q4_k hi-only 128x512 b5", D3d12WeightFormat::Q4_K, 128, 512, 5, 2},
    {"q4_k d-only 128x512 b5", D3d12WeightFormat::Q4_K, 128, 512, 5, 3},
    {"q4_k min-only 128x512 b5", D3d12WeightFormat::Q4_K, 128, 512, 5, 4},
    {"q5_k 1blk 128x256 b5", D3d12WeightFormat::Q5_K, 128, 256, 5},
    {"q5_k 2blk 128x512 b1", D3d12WeightFormat::Q5_K, 128, 512, 1},
    {"q5_k small 128x512 b5", D3d12WeightFormat::Q5_K, 128, 512, 5},
    {"q6_k decode 128x512 b1", D3d12WeightFormat::Q6_K, 128, 512, 1},
    {"q6_k small 128x512 b5", D3d12WeightFormat::Q6_K, 128, 512, 5},
    {"q8_0 1blk 128x32 b5", D3d12WeightFormat::Q8_0, 128, 32, 5},
    {"q8_0 decode 128x512 b1", D3d12WeightFormat::Q8_0, 128, 512, 1},
    {"q8_0 small 128x512 b5", D3d12WeightFormat::Q8_0, 128, 512, 5},
    {"q8_0 tile-edge 128x288 b3", D3d12WeightFormat::Q8_0, 128, 288, 3},
    {"q4_0 1blk 128x32 b5", D3d12WeightFormat::Q4_0, 128, 32, 5},
    {"q4_0 decode 128x512 b1", D3d12WeightFormat::Q4_0, 128, 512, 1},
    {"q4_0 small 128x512 b5", D3d12WeightFormat::Q4_0, 128, 512, 5},
    {"q4_0 tile-edge 128x288 b3", D3d12WeightFormat::Q4_0, 128, 288, 3},
    {"q4_k gemma-q decode 2048x2560 b1", D3d12WeightFormat::Q4_K, 2048, 2560, 1},
    {"q4_k gemma-kv decode 512x2560 b1", D3d12WeightFormat::Q4_K, 512, 2560, 1},
    {"q4_k gemma-gate 10240x2560 b15", D3d12WeightFormat::Q4_K, 10240, 2560, 15},
    {"q5_k gemma-gate 10240x2560 b15", D3d12WeightFormat::Q5_K, 10240, 2560, 15},
    {"q8_0 gemma-gate 10240x2560 b15", D3d12WeightFormat::Q8_0, 10240, 2560, 15},
    {"q4_0 gemma-gate 10240x2560 b15", D3d12WeightFormat::Q4_0, 10240, 2560, 15},
    {"q6_k gemma-down 2560x10240 b15", D3d12WeightFormat::Q6_K, 2560, 10240, 15},
    {"q4_k gemma-q 2048x2560 b63", D3d12WeightFormat::Q4_K, 2048, 2560, 63},
    {"bf16 medium 2688x2560 b15", D3d12WeightFormat::BF16, 2688, 2560, 15},
    {"bf16 gemma-ple 10752x2560 b15", D3d12WeightFormat::BF16, 10752, 2560, 15},
  };

  int failures = 0;
  for (const SelftestCase &test_case : cases) {
    if (!run_one_selftest(
      test_case.label,
      test_case.format,
      test_case.row_count,
      test_case.row_width,
      test_case.batch_count,
      test_case.q4k_scale_variant)) {
      failures += 1;
      const D3d12BatchMatmulCapabilities &after = d3d12_batch_matmul_capabilities();
      if (!after.supported) {
        std::printf("device lost: %s\n", after.reason.c_str());
        break;
      }
    }
  }
  if (failures == 0 && !run_ffn_chain_selftest("ffn chain 512x256 b3", 256, 512, 3, 256)) {
    failures += 1;
  }
  if (failures == 0
    && !run_ffn_chain_selftest("ffn chain gemma 10240x2560 b15", 2560, 10240, 15, 64)) {
    failures += 1;
  }
  if (failures == 0
    && !run_ffn_chain_selftest("ffn chain gemma 10240x2560 b1", 2560, 10240, 1, 64)) {
    failures += 1;
  }
  if (failures == 0
    && !run_ffn_chain_selftest("ffn chain interleaved b15", 2560, 10240, 15, 64, true)) {
    failures += 1;
  }
  if (failures == 0
    && !run_ffn_chain_selftest("ffn chain big inputs b15", 2560, 10240, 15, 64, true, 40.0f)) {
    failures += 1;
  }
  if (failures == 0
    && !run_ffn_chain_selftest("ffn chain silu 512x256 b3", 256, 512, 3, 256, false, 1.0f, true)) {
    failures += 1;
  }
  // Same shape prefill actually runs (llama-3.2-3b FFN) at a prefill-sized
  // batch, so the isolated rate is directly comparable to the in-situ number.
  if (failures == 0
    && !run_ffn_chain_selftest(
      "ffn chain silu llama 8192x3072 b269", 3072, 8192, 269, 8, false, 1.0f, true)) {
    failures += 1;
  }
  if (failures == 0
    && !run_ffn_chain_selftest("ffn chain silu llama 8192x3072 b1", 3072, 8192, 1, 64, false, 1.0f, true)) {
    failures += 1;
  }
  if (failures == 0 && !run_gemma_attention_selftest()) {
    failures += 1;
  }
  if (failures == 0 && !run_attention_tile_parity_selftest()) {
    failures += 1;
  }
  if (failures == 0 && !run_layer_tail_selftest()) {
    failures += 1;
  }
  if (failures == 0 && !run_coop_gemv_selftest()) {
    failures += 1;
  }
  if (failures == 0 && !run_stream_read_selftest()) {
    failures += 1;
  }
  if (failures == 0 && !run_wide128_selftest()) {
    failures += 1;
  }
  if (failures == 0 && !run_wave_gemv_selftest()) {
    failures += 1;
  }
  if (failures == 0 && !run_dp4a_selftest()) {
    failures += 1;
  }
  std::printf(failures == 0 ? "selftest passed\n" : "selftest failed\n");
  return failures == 0 ? 0 : 1;
}

}  // namespace lizard_native

#else

namespace lizard_native {

const D3d12BatchMatmulCapabilities &d3d12_batch_matmul_capabilities() {
  static D3d12BatchMatmulCapabilities capabilities = {
    false,
    "cpu",
    "",
    "Windows-native Direct3D 12 compute is only available on Windows builds.",
  };
  return capabilities;
}

const D3d12BatchMatmulCapabilities *d3d12_batch_matmul_capabilities_if_initialized() {
  return nullptr;
}

D3d12GpuMemoryReport d3d12_gpu_memory_report() {
  return D3d12GpuMemoryReport {};
}

void d3d12_refresh_gpu_memory_budget() {
}

const D3d12Dp4aCapabilities &d3d12_dp4a_capabilities() {
  static const D3d12Dp4aCapabilities capabilities = {
    false,
    "Windows-native Direct3D 12 compute is only available on Windows builds.",
  };
  return capabilities;
}

bool d3d12_dp4a_q4k_matvec(
  const D3d12ResidentWeightRef &,
  const std::vector<float> &,
  std::vector<float> *,
  std::string *error_message) {
  if (error_message != nullptr) {
    *error_message = d3d12_dp4a_capabilities().reason;
  }
  return false;
}

const D3d12RuntimeTelemetry &d3d12_runtime_telemetry() {
  static const D3d12RuntimeTelemetry telemetry;
  return telemetry;
}

void d3d12_reset_runtime_telemetry() {
}

bool d3d12_weight_format_supported(std::uint32_t, std::uint64_t, D3d12WeightFormat *) {
  return false;
}

bool d3d12_batch_matmul_resident(
  const D3d12ResidentWeightRef *,
  std::size_t,
  const std::vector<float> &,
  std::uint32_t,
  const void *,
  std::vector<float> *,
  std::string *error_message) {
  if (error_message != nullptr) {
    *error_message = d3d12_batch_matmul_capabilities().reason;
  }
  return false;
}

bool d3d12_gemma_token_resident(
  const std::vector<float> &,
  const std::vector<D3d12GemmaTokenLayer> &,
  const D3d12GemmaPleInputConfig *,
  const std::vector<float> &,
  const D3d12ResidentWeightRef &,
  float,
  float,
  std::vector<float> *,
  std::vector<float> *,
  std::uint32_t *,
  std::string *error_message) {
  if (error_message != nullptr) {
    *error_message = "Direct3D 12 is only available on Windows.";
  }
  return false;
}

bool d3d12_gemma_attention_resident(
  const D3d12ResidentWeightRef &,
  const D3d12ResidentWeightRef *,
  const D3d12ResidentWeightRef *,
  const std::vector<float> &,
  const std::vector<float> &,
  const std::vector<float> &,
  const std::vector<float> &,
  const std::vector<float> &,
  const std::vector<float> &,
  const D3d12GemmaAttentionConfig &,
  std::vector<float> *,
  std::string *error_message) {
  if (error_message != nullptr) {
    *error_message = d3d12_batch_matmul_capabilities().reason;
  }
  return false;
}

bool d3d12_gemma_layer_resident(
  const D3d12ResidentWeightRef &,
  const D3d12ResidentWeightRef *,
  const D3d12ResidentWeightRef *,
  const D3d12ResidentWeightRef &,
  const D3d12ResidentWeightRef &,
  const D3d12ResidentWeightRef &,
  const D3d12ResidentWeightRef &,
  const D3d12ResidentWeightRef *,
  const D3d12ResidentWeightRef *,
  const std::vector<float> &,
  const std::vector<float> &,
  const std::vector<float> &,
  const std::vector<float> &,
  const std::vector<float> &,
  const std::vector<float> &,
  const std::vector<float> &,
  const std::vector<float> &,
  const std::vector<float> &,
  const std::vector<float> &,
  const std::vector<float> *,
  const std::vector<float> *,
  const D3d12GemmaAttentionConfig &,
  std::vector<float> *,
  std::string *error_message) {
  if (error_message != nullptr) {
    *error_message = d3d12_batch_matmul_capabilities().reason;
  }
  return false;
}

bool d3d12_ffn_gelu_glu_resident(
  const D3d12ResidentWeightRef &,
  const D3d12ResidentWeightRef &,
  const D3d12ResidentWeightRef &,
  const std::vector<float> &,
  std::uint32_t,
  const void *,
  std::vector<float> *,
  std::string *error_message) {
  if (error_message != nullptr) {
    *error_message = d3d12_batch_matmul_capabilities().reason;
  }
  return false;
}

bool d3d12_ffn_silu_glu_resident(
  const D3d12ResidentWeightRef &,
  const D3d12ResidentWeightRef &,
  const D3d12ResidentWeightRef &,
  const std::vector<float> &,
  std::uint32_t,
  const void *,
  std::vector<float> *,
  std::string *error_message) {
  if (error_message != nullptr) {
    *error_message = d3d12_batch_matmul_capabilities().reason;
  }
  return false;
}

bool d3d12_gemma_layer_tail_resident(
  const D3d12ResidentWeightRef &,
  const D3d12ResidentWeightRef &,
  const D3d12ResidentWeightRef &,
  const D3d12ResidentWeightRef &,
  const std::vector<float> &,
  const std::vector<float> &,
  const std::vector<float> &,
  const std::vector<float> &,
  const std::vector<float> &,
  float,
  std::vector<float> *,
  std::string *error_message) {
  if (error_message != nullptr) {
    *error_message = d3d12_batch_matmul_capabilities().reason;
  }
  return false;
}

bool d3d12_gemma_layer_tail_ple_resident(
  const D3d12ResidentWeightRef &,
  const D3d12ResidentWeightRef &,
  const D3d12ResidentWeightRef &,
  const D3d12ResidentWeightRef &,
  const D3d12ResidentWeightRef &,
  const D3d12ResidentWeightRef &,
  const std::vector<float> &,
  const std::vector<float> &,
  const std::vector<float> &,
  const std::vector<float> &,
  const std::vector<float> &,
  const std::vector<float> &,
  const std::vector<float> &,
  float,
  std::vector<float> *,
  std::string *error_message) {
  if (error_message != nullptr) {
    *error_message = d3d12_batch_matmul_capabilities().reason;
  }
  return false;
}

void d3d12_reset_resident_weights() {
}

std::uint64_t d3d12_resident_weight_bytes() {
  return 0;
}

bool d3d12_stage_resident_weight(const D3d12ResidentWeightRef &, std::string *error_message) {
  if (error_message != nullptr) {
    *error_message = d3d12_batch_matmul_capabilities().reason;
  }
  return false;
}

void d3d12_shrink_staging_buffers() {
}

int d3d12_matmul_selftest() {
  std::fputs("D3D12 selftest is only available on Windows builds.\n", stdout);
  return 1;
}

bool d3d12_batch_matmul(
  const std::vector<float> &,
  std::uint32_t,
  std::uint32_t,
  const std::vector<float> &,
  std::uint32_t,
  const void *,
  std::vector<std::vector<float>> &,
  std::string *error_message) {
  if (error_message != nullptr) {
    *error_message = d3d12_batch_matmul_capabilities().reason;
  }
  return false;
}

}  // namespace lizard_native

#endif
