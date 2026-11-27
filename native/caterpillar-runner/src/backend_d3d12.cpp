// Caterpillar Phase 4 D3D12 graph-plan backend.
//
// Design (see .Agent/caterpillar/01_Architecture_Spec.md §2-4):
// - One command list per contiguous GPU span, recorded ONCE at prepare time
//   and re-executed for every replay. All per-dispatch root bindings are
//   static; per-token values (cached position count, RoPE position) flow
//   through a tiny session buffer whose CONTENTS the CPU rewrites between
//   submissions.
// - Requires cache-coherent UMA: activations, weights, and the packed KV cache
//   live in CUSTOM write-back heaps mapped persistently, so span inputs and
//   outputs are plain memcpys (UMA zero-copy; no upload/readback lists).
// - Hard-won Iris Xe rules (first-party lizard-native lane): clamp indices
//   BEFORE loads (drivers execute guarded loads speculatively), never issue
//   raw 16-bit loads (extract halves from dword loads), clamp tanh args to
//   +-15, batch-one matvec = many rows per group walking a groupshared
//   input tile, check GetDeviceRemovedReason after every submit.

#include "backend_d3d12.h"

#include "graph.h"
#include "tensor_data.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d12.h>
#include <d3dcompiler.h>
#include <dxgi1_4.h>
#ifdef CATERPILLAR_HAVE_DXIL
#include "caterpillar_q4_0_q8_1.h"
#include "caterpillar_q4k_q8_1.h"
#include "caterpillar_q5k_q8_1.h"
#include "caterpillar_q6k_q8_1.h"
#include "caterpillar_q8_0_q8_1.h"
#include "caterpillar_quantize_q8_1.h"
#endif
#endif

namespace cat {

#ifndef _WIN32

struct GpuPlanExecution::Impl {};
GpuPlanExecution::GpuPlanExecution(const Plan&, const QuantizedBindings&, const Bindings&,
    const std::vector<TensorId>&, const std::vector<DecoderGraph::KvBinding>&, const Config&) {
    throw GpuUnavailable("D3D12 is only available on Windows");
}
GpuPlanExecution::~GpuPlanExecution() = default;
void GpuPlanExecution::execute_span(const Plan&, const PlanSpan&, const BindingViews&, const ReplayDynamics&, std::span<float>) {
    throw GpuUnavailable("D3D12 is only available on Windows");
}
void GpuPlanExecution::seed_kv(std::size_t, std::span<const float>, std::span<const float>, std::size_t) {}
void GpuPlanExecution::write_kv_rows(std::size_t, std::size_t, std::span<const float>, std::span<const float>) {}
void GpuPlanExecution::read_kv_rows(std::size_t, std::size_t, std::size_t, std::span<float>, std::span<float>) {}
std::span<float> GpuPlanExecution::kv_keys(std::size_t) { return {}; }
std::span<float> GpuPlanExecution::kv_values(std::size_t) { return {}; }
bool GpuPlanExecution::has_cpu_kv_mirror() const noexcept { return false; }
std::string GpuPlanExecution::adapter_name() const { return {}; }
bool gpu_backend_available(std::string* reason) {
    if (reason != nullptr) *reason = "D3D12 is only available on Windows";
    return false;
}
int run_gpu_selftest() {
    std::cout << "caterpillar gpu selftest: skipped (non-Windows)\n";
    return 0;
}

#else

namespace {

template <typename T>
struct ComOwner {
    T* value{};
    ComOwner() = default;
    explicit ComOwner(T* raw) : value(raw) {}
    ComOwner(const ComOwner&) = delete;
    ComOwner& operator=(const ComOwner&) = delete;
    ComOwner(ComOwner&& other) noexcept : value(other.value) { other.value = nullptr; }
    ComOwner& operator=(ComOwner&& other) noexcept {
        if (this != &other) {
            reset();
            value = other.value;
            other.value = nullptr;
        }
        return *this;
    }
    ~ComOwner() { reset(); }
    void reset() {
        if (value != nullptr) {
            value->Release();
            value = nullptr;
        }
    }
    T** put() {
        reset();
        return &value;
    }
    T* operator->() const { return value; }
    explicit operator bool() const { return value != nullptr; }
};

[[noreturn]] void throw_hresult(const char* label, HRESULT hr) {
    std::ostringstream message;
    message << label << " failed with HRESULT 0x" << std::hex << static_cast<unsigned long>(hr);
    throw GpuUnavailable(message.str());
}

// Step 08 tuning: opt-in cooperative Q4_K matvec (32 rows x 8 column slices
// per group). Off by default until the acceptance gate (oracle tolerance +
// identical greedy token stream + interleaved A/B win) promotes it.
bool coop_matvec_requested() {
    const char* value = std::getenv("CATERPILLAR_GPU_COOP_MATVEC");
    return value != nullptr && value[0] == '1' && value[1] == '\0';
}

// Step 09b (Route B): CPU fraction of each detected FFN gate/up pair, e.g.
// CATERPILLAR_HYBRID_FFN=0.4 gives the CPU 40% of the output rows while the
// GPU computes the rest concurrently. 0/unset disables the hybrid split.
double hybrid_ffn_cpu_fraction() {
    const char* value = std::getenv("CATERPILLAR_HYBRID_FFN");
    if (value == nullptr || *value == '\0') return 0.0;
    char* end{};
    const double parsed = std::strtod(value, &end);
    if (end == value || !(parsed > 0.0)) return 0.0;
    return std::min(parsed, 0.9);
}

void check_hresult(const char* label, HRESULT hr) {
    if (FAILED(hr)) throw_hresult(label, hr);
}

bool parse_requested_luid(LUID& output) {
    const char* value = std::getenv("LIZARD_NATIVE_GPU_ADAPTER_LUID");
    if (value == nullptr || *value == '\0') return false;
    char* separator{};
    const auto high = std::strtoll(value, &separator, 10);
    if (separator == value || separator == nullptr || *separator != ':') return false;
    char* end{};
    const auto low = std::strtoull(separator + 1, &end, 10);
    if (end == separator + 1 || *end != '\0') return false;
    output.HighPart = static_cast<LONG>(high);
    output.LowPart = static_cast<DWORD>(low);
    return true;
}

std::string narrow_name(const wchar_t* value) {
    if (value == nullptr || *value == L'\0') return {};
    const int length = WideCharToMultiByte(CP_UTF8, 0, value, -1, nullptr, 0, nullptr, nullptr);
    if (length <= 1) return {};
    std::string output(static_cast<std::size_t>(length), '\0');
    WideCharToMultiByte(CP_UTF8, 0, value, -1, output.data(), length, nullptr, nullptr);
    output.pop_back();
    return output;
}

// ---------------------------------------------------------------------------
// Kernels. cs_5_0 float path (integer-dot DP4A is Phase 5). Root layout:
//   b0  16 root constants (per-dispatch static parameters)
//   t0  weights      ByteAddressBuffer        (per-dispatch resident tensor)
//   t1  unused/dummy SRV (kept for Phase 5 embedded-DXIL compatibility)
//   t2  session      ByteAddressBuffer        (dword 0 = active past tokens,
//                                              dword 1 = rope position)
//   u0  state        RWStructuredBuffer<float> (activation arena + externals)
//   u1  kv_cache     RWByteAddressBuffer       (device-resident packed K/V)
// ---------------------------------------------------------------------------
constexpr char k_kernel_source[] = R"(
cbuffer Params : register(b0) {
  uint4 p0;
  uint4 p1;
  uint4 p2;
  uint4 p3;
};

ByteAddressBuffer weights : register(t0);
ByteAddressBuffer session : register(t2);
RWStructuredBuffer<float> state : register(u0);
RWByteAddressBuffer kv_cache : register(u1);

static const uint k_rows_per_group = 128;
static const uint k_tile = 256;

groupshared float input_tile[k_tile];
groupshared float reduce_partials[256];
groupshared float attn_context[512];
groupshared float attn_key[512];
groupshared float attn_value[512];
groupshared float attn_state[4];
groupshared float kv_block[32];

float load_fp16_at(uint byte_address) {
  const uint word = weights.Load(byte_address & ~3u);
  const uint half_bits = ((byte_address & 2u) != 0u) ? (word >> 16u) : (word & 0xFFFFu);
  return f16tof32(half_bits & 0xFFFFu);
}

uint byte_of_word(uint word, uint lane) {
  return (word >> (lane * 8u)) & 0xFFu;
}

uint kv_block_bytes(uint mode) {
  return mode == 2u ? 36u : (mode == 3u ? 20u : 16u);
}

float load_kv_element(uint lane_base, uint row_bytes, uint token, uint element, uint mode) {
  const uint row_base = lane_base + token * row_bytes;
  if (mode == 0u) {
    return asfloat(kv_cache.Load(row_base + element * 4u));
  }
  if (mode == 1u) {
    const uint address = row_base + element * 2u;
    const uint packed = kv_cache.Load(address & ~3u);
    const uint half_bits = (address & 2u) != 0u ? packed >> 16u : packed & 0xffffu;
    return f16tof32(half_bits);
  }
  const uint block = element >> 5u;
  const uint item = element & 31u;
  const uint base = row_base + block * kv_block_bytes(mode);
  const float scale = asfloat(kv_cache.Load(base));
  if (mode == 2u) {
    const uint packed = kv_cache.Load((base + 4u + item) & ~3u);
    const uint raw = byte_of_word(packed, (base + 4u + item) & 3u);
    return (float)((int)(raw << 24u) >> 24) * scale;
  }
  if (mode == 3u) {
    const uint address = base + 4u + (item >> 1u);
    const uint packed = kv_cache.Load(address & ~3u);
    const uint byte_value = byte_of_word(packed, address & 3u);
    const uint nibble = (item & 1u) == 0u ? byte_value & 15u : byte_value >> 4u;
    return (float)((int)nibble - 8) * scale;
  }
  float reconstructed = 0.0f;
  [unroll]
  for (uint rotated = 0u; rotated < 32u; rotated += 1u) {
    const uint bit = rotated * 3u;
    const uint address = base + 4u + (bit >> 3u);
    uint raw = byte_of_word(kv_cache.Load(address & ~3u), address & 3u);
    if ((bit & 7u) > 5u) {
      const uint next_address = address + 1u;
      raw |= byte_of_word(kv_cache.Load(next_address & ~3u), next_address & 3u) << 8u;
    }
    const int code = (int)((raw >> (bit & 7u)) & 7u) - 4;
    const float sign_value = (countbits(rotated & item) & 1u) == 0u ? 1.0f : -1.0f;
    reconstructed += sign_value * (float)code;
  }
  return reconstructed * scale * (1.0f / 32.0f);
}

// Fused matvec epilogue (Step 07 tuning). p2.x selects the folded elementwise
// consumer, p2.y is the second operand's element offset. Expressions are
// copied verbatim from cs_add / cs_silu_mul / cs_gelu_mul so a fused row
// stores the exact bits the standalone kernels would have produced (IEEE-754
// float addition is commutative; the activation expression trees are
// identical).
//   0: none                     1: acc + operand (residual add)
//   2: silu(operand) * acc      3: silu(acc) * operand
//   4: gelu(operand) * acc      5: gelu(acc) * operand
float epilogue_gelu(float gate) {
  const float cubic = gate * gate * gate;
  const float tanh_argument = clamp(0.7978845608028654f * (gate + 0.044715f * cubic), -15.0f, 15.0f);
  return 0.5f * gate * (1.0f + tanh(tanh_argument));
}

float apply_matvec_epilogue(float acc, uint row) {
  if (p2.x == 0u) {
    return acc;
  }
  if (p2.x == 1u) {
    return acc + state[p2.y + row];
  }
  if (p2.x == 2u) {
    const float gate = state[p2.y + row];
    return (gate / (1.0f + exp(-gate))) * acc;
  }
  if (p2.x == 3u) {
    return (acc / (1.0f + exp(-acc))) * state[p2.y + row];
  }
  if (p2.x == 4u) {
    return epilogue_gelu(state[p2.y + row]) * acc;
  }
  return epilogue_gelu(acc) * state[p2.y + row];
}

void load_tile(uint thread_index, uint input_offset, uint element_base, uint columns) {
  [unroll]
  for (uint slot = 0; slot < k_tile / k_rows_per_group; slot += 1u) {
    const uint element = element_base + slot * k_rows_per_group + thread_index;
    const uint safe_element = min(element, columns - 1u);
    const float value = state[input_offset + safe_element];
    input_tile[slot * k_rows_per_group + thread_index] = element < columns ? value : 0.0f;
  }
}

// --- quantized / float matvec (batch = 1 decode shapes) -------------------
// p0.x rows, p0.y columns, p0.z input_offset, p0.w output_offset,
// p1.x weight_base_bytes.

[numthreads(k_rows_per_group, 1, 1)]
void cs_matvec_f32(uint3 group_id : SV_GroupID, uint thread_index : SV_GroupIndex) {
  const uint row = group_id.x * k_rows_per_group + thread_index;
  // Tail-group threads must still compute IN-BOUNDS addresses: the driver
  // executes guarded loads speculatively, so the row used for address math
  // is clamped and only the store is guarded by the real row.
  const uint safe_row = min(row, p0.x - 1u);
  float acc = 0.0f;
  const uint tiles = (p0.y + k_tile - 1u) / k_tile;
  [loop]
  for (uint tile = 0; tile < tiles; tile += 1u) {
    load_tile(thread_index, p0.z, tile * k_tile, p0.y);
    GroupMemoryBarrierWithGroupSync();
    if (row < p0.x) {
      const uint columns_left = min(k_tile, p0.y - tile * k_tile);
      const uint row_base = p1.x + (safe_row * p0.y + tile * k_tile) * 4u;
      for (uint item4 = 0; item4 < k_tile / 4u; item4 += 1u) {
        if (item4 * 4u >= columns_left) { break; }
        const uint4 words = weights.Load4(row_base + item4 * 16u);
        const uint remaining = columns_left - item4 * 4u;
        acc += asfloat(words.x) * input_tile[item4 * 4u];
        if (remaining > 1u) { acc += asfloat(words.y) * input_tile[item4 * 4u + 1u]; }
        if (remaining > 2u) { acc += asfloat(words.z) * input_tile[item4 * 4u + 2u]; }
        if (remaining > 3u) { acc += asfloat(words.w) * input_tile[item4 * 4u + 3u]; }
      }
    }
    GroupMemoryBarrierWithGroupSync();
  }
  if (row < p0.x) {
    state[p0.w + row] = acc;
  }
}

// Q4_K: 144-byte super-blocks of 256, staged column-major:
// staged_byte = weight_base + (block * rows + row) * 144.
uint scale_byte_of(uint3 words, uint index) {
  const uint word = index < 4u ? words.x : (index < 8u ? words.y : words.z);
  return byte_of_word(word, index & 3u);
}

[numthreads(k_rows_per_group, 1, 1)]
void cs_matvec_q4k(uint3 group_id : SV_GroupID, uint thread_index : SV_GroupIndex) {
  const uint row = group_id.x * k_rows_per_group + thread_index;
  const uint safe_row = min(row, p0.x - 1u);
  // Route B row split (Step 09): a nonzero p2.z caps the rows this dispatch
  // computes ([0, p2.z)); the CPU produces the remainder concurrently.
  // Addressing still uses the full p0.x staged layout. p2.z == 0 (default
  // constants) keeps the historical full-row behavior at zero cost.
  const uint row_limit = p2.z != 0u ? min(p2.z, p0.x) : p0.x;
  float acc = 0.0f;
  const uint blocks = p0.y >> 8u;
  [loop]
  for (uint block = 0; block < blocks; block += 1u) {
    load_tile(thread_index, p0.z, block << 8u, p0.y);
    GroupMemoryBarrierWithGroupSync();
    if (row < row_limit) {
      const uint block_byte = p1.x + (block * p0.x + safe_row) * 144u;
      const uint d_words = weights.Load(block_byte);
      const float d = f16tof32(d_words & 0xFFFFu);
      const float dmin = f16tof32(d_words >> 16u);
      const uint3 scale_words = weights.Load3(block_byte + 4u);
      float scale_d[8];
      float scale_m[8];
      [unroll]
      for (uint j = 0; j < 8u; j += 1u) {
        uint sd;
        uint sm;
        if (j < 4u) {
          sd = scale_byte_of(scale_words, j) & 63u;
          sm = scale_byte_of(scale_words, j + 4u) & 63u;
        } else {
          sd = (scale_byte_of(scale_words, j + 4u) & 0x0Fu)
            | ((scale_byte_of(scale_words, j - 4u) >> 6u) << 4u);
          sm = (scale_byte_of(scale_words, j + 4u) >> 4u)
            | ((scale_byte_of(scale_words, j) >> 6u) << 4u);
        }
        scale_d[j] = d * (float)sd;
        scale_m[j] = dmin * (float)sm;
      }
      [loop]
      for (uint quad = 0; quad < 8u; quad += 1u) {
        const uint4 packed = weights.Load4(block_byte + 16u + quad * 16u);
        [unroll]
        for (uint sub = 0; sub < 4u; sub += 1u) {
          const uint packed4 = sub == 0u ? packed.x : (sub == 1u ? packed.y : (sub == 2u ? packed.z : packed.w));
          [unroll]
          for (uint lane = 0; lane < 4u; lane += 1u) {
            const uint quant_index = (quad * 4u + sub) * 4u + lane;
            const uint chunk = quant_index >> 5u;
            const uint item = quant_index & 31u;
            const uint quant_byte = byte_of_word(packed4, lane);
            const uint element = chunk * 64u + item;
            acc += (scale_d[chunk * 2u] * (float)(quant_byte & 0x0Fu) - scale_m[chunk * 2u]) * input_tile[element];
            acc += (scale_d[chunk * 2u + 1u] * (float)(quant_byte >> 4u) - scale_m[chunk * 2u + 1u]) * input_tile[element + 32u];
          }
        }
      }
    }
    GroupMemoryBarrierWithGroupSync();
  }
  if (row < row_limit) {
    state[p0.w + row] = apply_matvec_epilogue(acc, row);
  }
}

// Cooperative Q4_K matvec (Step 08 tuning, opt-in CATERPILLAR_GPU_COOP_MATVEC).
// The serial kernel runs ONE thread per output row - 3k-8k threads total on an
// iGPU that wants ~50k - and node timing showed matmul at 96% of decode time
// (~12 GB/s effective). This variant keeps the same column-major staging and
// warp-coalesced weight reads (32 CONSECUTIVE rows per 32-lane wave) but adds
// an 8-way column split: 256 threads = 32 rows x 8 block-slices, so every
// dispatch carries 8x the threads. Slice partials reduce through groupshared
// memory in fixed slice order - deterministic, but a DIFFERENT float summation
// order than the serial kernel, so acceptance uses the oracle-tolerance
// selftest plus an identical greedy token stream instead of the byte gate.
groupshared float coop_partials[256];

[numthreads(256, 1, 1)]
void cs_matvec_q4k_coop(uint3 group_id : SV_GroupID, uint thread_index : SV_GroupIndex) {
  const uint row_lane = thread_index & 31u;
  const uint slice = thread_index >> 5u;
  const uint row = group_id.x * 32u + row_lane;
  const uint safe_row = min(row, p0.x - 1u);
  float acc = 0.0f;
  const uint blocks = p0.y >> 8u;
  [loop]
  for (uint block = slice; block < blocks; block += 8u) {
    const uint input_base = p0.z + (block << 8u);
    const uint block_byte = p1.x + (block * p0.x + safe_row) * 144u;
    const uint d_words = weights.Load(block_byte);
    const float d = f16tof32(d_words & 0xFFFFu);
    const float dmin = f16tof32(d_words >> 16u);
    const uint3 scale_words = weights.Load3(block_byte + 4u);
    float scale_d[8];
    float scale_m[8];
    [unroll]
    for (uint j = 0; j < 8u; j += 1u) {
      uint sd;
      uint sm;
      if (j < 4u) {
        sd = scale_byte_of(scale_words, j) & 63u;
        sm = scale_byte_of(scale_words, j + 4u) & 63u;
      } else {
        sd = (scale_byte_of(scale_words, j + 4u) & 0x0Fu)
          | ((scale_byte_of(scale_words, j - 4u) >> 6u) << 4u);
        sm = (scale_byte_of(scale_words, j + 4u) >> 4u)
          | ((scale_byte_of(scale_words, j) >> 6u) << 4u);
      }
      scale_d[j] = d * (float)sd;
      scale_m[j] = dmin * (float)sm;
    }
    [loop]
    for (uint quad = 0; quad < 8u; quad += 1u) {
      const uint4 packed = weights.Load4(block_byte + 16u + quad * 16u);
      [unroll]
      for (uint sub = 0; sub < 4u; sub += 1u) {
        const uint packed4 = sub == 0u ? packed.x : (sub == 1u ? packed.y : (sub == 2u ? packed.z : packed.w));
        [unroll]
        for (uint lane = 0; lane < 4u; lane += 1u) {
          const uint quant_index = (quad * 4u + sub) * 4u + lane;
          const uint chunk = quant_index >> 5u;
          const uint item = quant_index & 31u;
          const uint quant_byte = byte_of_word(packed4, lane);
          const uint element = chunk * 64u + item;
          acc += (scale_d[chunk * 2u] * (float)(quant_byte & 0x0Fu) - scale_m[chunk * 2u]) * state[input_base + element];
          acc += (scale_d[chunk * 2u + 1u] * (float)(quant_byte >> 4u) - scale_m[chunk * 2u + 1u]) * state[input_base + element + 32u];
        }
      }
    }
  }
  coop_partials[thread_index] = acc;
  GroupMemoryBarrierWithGroupSync();
  if (slice == 0u && row < p0.x) {
    float total = coop_partials[row_lane];
    [unroll]
    for (uint s = 1u; s < 8u; s += 1u) {
      total += coop_partials[row_lane + s * 32u];
    }
    state[p0.w + row] = apply_matvec_epilogue(total, row);
  }
}

// Q5_K: 176-byte super-blocks of 256, staged column-major. Layout is Q4_K's
// d/dmin + packed scale/min fields, followed by a 32-byte high-bit plane and
// 128 low-nibble bytes.
[numthreads(k_rows_per_group, 1, 1)]
void cs_matvec_q5k(uint3 group_id : SV_GroupID, uint thread_index : SV_GroupIndex) {
  const uint row = group_id.x * k_rows_per_group + thread_index;
  const uint safe_row = min(row, p0.x - 1u);
  float acc = 0.0f;
  const uint blocks = p0.y >> 8u;
  [loop]
  for (uint block = 0; block < blocks; block += 1u) {
    load_tile(thread_index, p0.z, block << 8u, p0.y);
    GroupMemoryBarrierWithGroupSync();
    if (row < p0.x) {
      const uint block_byte = p1.x + (block * p0.x + safe_row) * 176u;
      const uint d_words = weights.Load(block_byte);
      const float d = f16tof32(d_words & 0xFFFFu);
      const float dmin = f16tof32(d_words >> 16u);
      const uint3 scale_words = weights.Load3(block_byte + 4u);
      [unroll]
      for (uint group = 0u; group < 8u; group += 1u) {
        uint scale;
        uint minimum;
        if (group < 4u) {
          scale = scale_byte_of(scale_words, group) & 63u;
          minimum = scale_byte_of(scale_words, group + 4u) & 63u;
        } else {
          scale = (scale_byte_of(scale_words, group + 4u) & 0x0Fu)
            | ((scale_byte_of(scale_words, group - 4u) >> 6u) << 4u);
          minimum = (scale_byte_of(scale_words, group + 4u) >> 4u)
            | ((scale_byte_of(scale_words, group) >> 6u) << 4u);
        }
        const uint quant_base = block_byte + 48u + (group >> 1u) * 32u;
        const bool high_nibble = (group & 1u) != 0u;
        [unroll]
        for (uint word_index = 0u; word_index < 8u; word_index += 1u) {
          const uint packed = weights.Load(quant_base + word_index * 4u);
          const uint high = weights.Load(block_byte + 16u + word_index * 4u);
          [unroll]
          for (uint lane = 0u; lane < 4u; lane += 1u) {
            const uint nibble = high_nibble
              ? (byte_of_word(packed, lane) >> 4u)
              : (byte_of_word(packed, lane) & 0x0Fu);
            const uint high_bit = (byte_of_word(high, lane) >> group) & 1u;
            const float value = d * (float)scale * (float)(nibble | (high_bit << 4u))
              - dmin * (float)minimum;
            acc += value * input_tile[group * 32u + word_index * 4u + lane];
          }
        }
      }
    }
    GroupMemoryBarrierWithGroupSync();
  }
  if (row < p0.x) state[p0.w + row] = acc;
}

// Q6_K: staged 212-byte blocks: [0..127] ql, [128..191] qh, [192..207]
// signed scales, [208..211] d widened to f32.
[numthreads(k_rows_per_group, 1, 1)]
void cs_matvec_q6k(uint3 group_id : SV_GroupID, uint thread_index : SV_GroupIndex) {
  const uint row = group_id.x * k_rows_per_group + thread_index;
  const uint safe_row = min(row, p0.x - 1u);
  float acc = 0.0f;
  const uint blocks = p0.y >> 8u;
  [loop]
  for (uint block = 0; block < blocks; block += 1u) {
    load_tile(thread_index, p0.z, block << 8u, p0.y);
    GroupMemoryBarrierWithGroupSync();
    if (row < p0.x) {
      const uint block_byte = p1.x + (block * p0.x + safe_row) * 212u;
      const float d = asfloat(weights.Load(block_byte + 208u));
      // 16 signed sub-block scales.
      float scales[16];
      [unroll]
      for (uint sw = 0; sw < 4u; sw += 1u) {
        const uint word = weights.Load(block_byte + 192u + sw * 4u);
        [unroll]
        for (uint lane = 0; lane < 4u; lane += 1u) {
          const uint raw = byte_of_word(word, lane);
          scales[sw * 4u + lane] = (float)((int)(raw << 24u) >> 24);
        }
      }
      // Walk the 256 elements in two 128-halves matching the CPU decoder:
      // for half h (0,1), element e in [0,128): value index = h*128 + e,
      // ql byte = h*64 + (e & 63), low/high nibble by e/64; qh byte =
      // h*32 + (e & 31), 2 bits selected by e/32; scale = value_index/16.
      [loop]
      for (uint half_index = 0; half_index < 2u; half_index += 1u) {
        const uint ql_base = block_byte + half_index * 64u;
        const uint qh_base = block_byte + 128u + half_index * 32u;
        for (uint pair = 0; pair < 32u; pair += 1u) {
          const uint qh_byte = byte_of_word(weights.Load((qh_base + pair) & ~3u), (qh_base + pair) & 3u);
          const uint ql_a = byte_of_word(weights.Load((ql_base + pair) & ~3u), (ql_base + pair) & 3u);
          const uint ql_b = byte_of_word(weights.Load((ql_base + 32u + pair) & ~3u), (ql_base + 32u + pair) & 3u);
          const uint element0 = half_index * 128u + pair;
          const uint element1 = element0 + 32u;
          const uint element2 = element0 + 64u;
          const uint element3 = element0 + 96u;
          const float q0 = (float)((int)((ql_a & 0x0Fu) | ((qh_byte & 0x03u) << 4u)) - 32);
          const float q1 = (float)((int)((ql_b & 0x0Fu) | (((qh_byte >> 2u) & 0x03u) << 4u)) - 32);
          const float q2 = (float)((int)((ql_a >> 4u) | (((qh_byte >> 4u) & 0x03u) << 4u)) - 32);
          const float q3 = (float)((int)((ql_b >> 4u) | (((qh_byte >> 6u) & 0x03u) << 4u)) - 32);
          acc += d * scales[element0 >> 4u] * q0 * input_tile[element0];
          acc += d * scales[element1 >> 4u] * q1 * input_tile[element1];
          acc += d * scales[element2 >> 4u] * q2 * input_tile[element2];
          acc += d * scales[element3 >> 4u] * q3 * input_tile[element3];
        }
      }
    }
    GroupMemoryBarrierWithGroupSync();
  }
  if (row < p0.x) {
    state[p0.w + row] = apply_matvec_epilogue(acc, row);
  }
}

// Q8_0: staged 36-byte blocks of 32: [0..3] f32 d, [4..35] int8 quants.
[numthreads(k_rows_per_group, 1, 1)]
void cs_matvec_q8_0(uint3 group_id : SV_GroupID, uint thread_index : SV_GroupIndex) {
  const uint row = group_id.x * k_rows_per_group + thread_index;
  const uint safe_row = min(row, p0.x - 1u);
  float acc = 0.0f;
  const uint tiles = (p0.y + k_tile - 1u) / k_tile;
  [loop]
  for (uint tile = 0; tile < tiles; tile += 1u) {
    load_tile(thread_index, p0.z, tile * k_tile, p0.y);
    GroupMemoryBarrierWithGroupSync();
    if (row < p0.x) {
      const uint blocks_left = min(8u, (p0.y - tile * k_tile) / 32u);
      for (uint sub = 0; sub < blocks_left; sub += 1u) {
        const uint block = tile * 8u + sub;
        const uint block_byte = p1.x + (block * p0.x + safe_row) * 36u;
        const float d = asfloat(weights.Load(block_byte));
        [unroll]
        for (uint word_index = 0; word_index < 8u; word_index += 1u) {
          const uint word = weights.Load(block_byte + 4u + word_index * 4u);
          [unroll]
          for (uint lane = 0; lane < 4u; lane += 1u) {
            const int quant = (int)(byte_of_word(word, lane) << 24u) >> 24;
            acc += d * (float)quant * input_tile[sub * 32u + word_index * 4u + lane];
          }
        }
      }
    }
    GroupMemoryBarrierWithGroupSync();
  }
  if (row < p0.x) {
    state[p0.w + row] = acc;
  }
}

// Q4_0: staged as the RAW 18-byte block (fp16 d + 16 nibble bytes) so the
// resident footprint matches the fit planner exactly; blocks are only
// 2-byte aligned, so d uses the paired-half load and quant bytes are
// extracted from dword loads.
[numthreads(k_rows_per_group, 1, 1)]
void cs_matvec_q4_0(uint3 group_id : SV_GroupID, uint thread_index : SV_GroupIndex) {
  const uint row = group_id.x * k_rows_per_group + thread_index;
  const uint safe_row = min(row, p0.x - 1u);
  float acc = 0.0f;
  const uint tiles = (p0.y + k_tile - 1u) / k_tile;
  [loop]
  for (uint tile = 0; tile < tiles; tile += 1u) {
    load_tile(thread_index, p0.z, tile * k_tile, p0.y);
    GroupMemoryBarrierWithGroupSync();
    if (row < p0.x) {
      const uint blocks_left = min(8u, (p0.y - tile * k_tile) / 32u);
      for (uint sub = 0; sub < blocks_left; sub += 1u) {
        const uint block = tile * 8u + sub;
        const uint block_byte = p1.x + (block * p0.x + safe_row) * 18u;
        const float d = load_fp16_at(block_byte);
        [unroll]
        for (uint item = 0; item < 16u; item += 1u) {
          const uint quant_address = block_byte + 2u + item;
          const uint quant_byte = byte_of_word(weights.Load(quant_address & ~3u), quant_address & 3u);
          const float low = d * ((float)(quant_byte & 0x0Fu) - 8.0f);
          const float high = d * ((float)(quant_byte >> 4u) - 8.0f);
          acc += low * input_tile[sub * 32u + item];
          acc += high * input_tile[sub * 32u + item + 16u];
        }
      }
    }
    GroupMemoryBarrierWithGroupSync();
  }
  if (row < p0.x) {
    state[p0.w + row] = acc;
  }
}

int iq4_nl_value(uint code) {
  code &= 15u;
  const uint group = code >> 2u;
  const uint packed = group == 0u ? 0xbfad9881u
    : (group == 1u ? 0xf6eaddcfu
    : (group == 2u ? 0x26190d01u : 0x71594535u));
  const uint raw = (packed >> ((code & 3u) * 8u)) & 255u;
  return (int)(raw << 24u) >> 24;
}

// IQ4_NL: raw 18-byte blocks of 32. Unlike Q4_0, each nibble indexes the
// fixed nonlinear 16-value codebook above.
[numthreads(k_rows_per_group, 1, 1)]
void cs_matvec_iq4_nl(uint3 group_id : SV_GroupID, uint thread_index : SV_GroupIndex) {
  const uint row = group_id.x * k_rows_per_group + thread_index;
  const uint safe_row = min(row, p0.x - 1u);
  float acc = 0.0f;
  const uint tiles = (p0.y + k_tile - 1u) / k_tile;
  [loop]
  for (uint tile = 0; tile < tiles; tile += 1u) {
    load_tile(thread_index, p0.z, tile * k_tile, p0.y);
    GroupMemoryBarrierWithGroupSync();
    if (row < p0.x) {
      const uint blocks_left = min(8u, (p0.y - tile * k_tile) / 32u);
      for (uint sub = 0u; sub < blocks_left; sub += 1u) {
        const uint block = tile * 8u + sub;
        const uint block_byte = p1.x + (block * p0.x + safe_row) * 18u;
        const float d = load_fp16_at(block_byte);
        [unroll]
        for (uint item = 0u; item < 16u; item += 1u) {
          const uint address = block_byte + 2u + item;
          const uint packed = byte_of_word(weights.Load(address & ~3u), address & 3u);
          acc += d * (float)iq4_nl_value(packed) * input_tile[sub * 32u + item];
          acc += d * (float)iq4_nl_value(packed >> 4u) * input_tile[sub * 32u + item + 16u];
        }
      }
    }
    GroupMemoryBarrierWithGroupSync();
  }
  if (row < p0.x) state[p0.w + row] = acc;
}

// IQ4_XS: 136-byte super-blocks of 256: fp16 d, sixteen high scale bits,
// four low-scale bytes, then eight independently packed IQ4_NL groups.
[numthreads(k_rows_per_group, 1, 1)]
void cs_matvec_iq4_xs(uint3 group_id : SV_GroupID, uint thread_index : SV_GroupIndex) {
  const uint row = group_id.x * k_rows_per_group + thread_index;
  const uint safe_row = min(row, p0.x - 1u);
  float acc = 0.0f;
  const uint blocks = p0.y >> 8u;
  [loop]
  for (uint block = 0u; block < blocks; block += 1u) {
    load_tile(thread_index, p0.z, block << 8u, p0.y);
    GroupMemoryBarrierWithGroupSync();
    if (row < p0.x) {
      const uint block_byte = p1.x + (block * p0.x + safe_row) * 136u;
      const uint header = weights.Load(block_byte);
      const float d = f16tof32(header & 0xffffu);
      const uint scales_h = header >> 16u;
      [unroll]
      for (uint sub = 0u; sub < 8u; sub += 1u) {
        const uint low_address = block_byte + 4u + (sub >> 1u);
        const uint low_packed = byte_of_word(weights.Load(low_address & ~3u), low_address & 3u);
        const uint low = (low_packed >> (4u * (sub & 1u))) & 15u;
        const uint high = (scales_h >> (2u * sub)) & 3u;
        const float dl = d * (float)((int)(low | (high << 4u)) - 32);
        const uint quant_base = block_byte + 8u + sub * 16u;
        [unroll]
        for (uint item = 0u; item < 16u; item += 1u) {
          const uint address = quant_base + item;
          const uint packed = byte_of_word(weights.Load(address & ~3u), address & 3u);
          acc += dl * (float)iq4_nl_value(packed) * input_tile[sub * 32u + item];
          acc += dl * (float)iq4_nl_value(packed >> 4u) * input_tile[sub * 32u + item + 16u];
        }
      }
    }
    GroupMemoryBarrierWithGroupSync();
  }
  if (row < p0.x) state[p0.w + row] = acc;
}

// --- rms_norm --------------------------------------------------------------
// One group per row. p0.x width, p0.y input_offset, p0.z weight_offset in
// state (0xFFFFFFFF = unweighted), p0.w output_offset, p1.x epsilon bits,
// p1.y rows.
[numthreads(256, 1, 1)]
void cs_rms_norm(uint3 group_id : SV_GroupID, uint thread_index : SV_GroupIndex) {
  const uint width = p0.x;
  const uint row = group_id.x;
  const uint input_base = p0.y + row * width;
  const uint output_base = p0.w + row * width;
  const uint last = width - 1u;
  float sum = 0.0f;
  for (uint index = thread_index; index < width; index += 256u) {
    const float value = state[input_base + min(index, last)];
    sum += value * value;
  }
  reduce_partials[thread_index] = sum;
  GroupMemoryBarrierWithGroupSync();
  [unroll]
  for (uint stride = 128u; stride > 0u; stride >>= 1u) {
    if (thread_index < stride) {
      reduce_partials[thread_index] += reduce_partials[thread_index + stride];
    }
    GroupMemoryBarrierWithGroupSync();
  }
  const float inverse_rms = rsqrt(reduce_partials[0] / (float)width + asfloat(p1.x));
  for (uint index = thread_index; index < width; index += 256u) {
    if (index < width) {
      // Clamp the ADDRESS before the load and mask the VALUE: the driver
      // flattens this select and executes both loads speculatively, so an
      // 0xFFFFFFFF sentinel in the address arithmetic hangs the device.
      const uint weight_index = (p0.z == 0xFFFFFFFFu ? 0u : p0.z) + index;
      const float weight_value = state[weight_index];
      const float weight = p0.z == 0xFFFFFFFFu ? 1.0f : weight_value;
      state[output_base + index] = state[input_base + index] * inverse_rms * weight;
    }
  }
}

// --- rope -------------------------------------------------------------------
// Threads over head*pair. p0.x heads, p0.y head_dim, p0.z input_offset,
// p0.w output_offset, p1.x frequency_base bits, p1.y frequency_scale bits,
// p1.z neox flag, p1.w factors_offset in state (0xFFFFFFFF = none).
// Position = session dword 1.
[numthreads(256, 1, 1)]
void cs_rope(uint3 dispatch_id : SV_DispatchThreadID) {
  const uint half_dim = p0.y / 2u;
  const uint total = p0.x * half_dim;
  const uint index = dispatch_id.x;
  if (index >= total) {
    return;
  }
  const uint head = index / half_dim;
  const uint pair = index % half_dim;
  const uint base = p0.z + head * p0.y;
  const uint out_base = p0.w + head * p0.y;
  const uint left_index = p1.z != 0u ? pair : pair * 2u;
  const uint right_index = p1.z != 0u ? pair + half_dim : pair * 2u + 1u;
  const float position = (float)session.Load(4u) * asfloat(p1.y);
  const float exponent = (float)(pair * 2u) / (float)p0.y;
  float inverse_frequency = 1.0f / pow(asfloat(p1.x), exponent);
  // Speculative-load-safe optional frequency factors: load from a clamped
  // address unconditionally, apply only when the factors really exist.
  const uint factor_index = (p1.w == 0xFFFFFFFFu ? 0u : p1.w) + pair;
  const float factor_value = state[factor_index];
  if (p1.w != 0xFFFFFFFFu && factor_value > 0.0f) {
    inverse_frequency /= factor_value;
  }
  const float angle = position * inverse_frequency;
  float sine_value;
  float cosine_value;
  sincos(angle, sine_value, cosine_value);
  const float left = state[base + left_index];
  const float right = state[base + right_index];
  state[out_base + left_index] = left * cosine_value - right * sine_value;
  state[out_base + right_index] = left * sine_value + right * cosine_value;
}

// --- device-resident KV append ----------------------------------------------
// One 32-thread group per block and K/V lane. The current token index comes
// from session dword 0, so this command list remains static across replay.
// Uses the same constants as cs_attention plus p3.x mode, p3.y KV width.
[numthreads(32, 1, 1)]
void cs_kv_append(uint3 group_id : SV_GroupID, uint thread_index : SV_GroupIndex) {
  const uint mode = p3.x;
  const uint width = p3.y;
  const uint element = group_id.x * 32u + thread_index;
  const bool active = element < width;
  const uint source_base = group_id.y == 0u ? p1.z : p1.w;
  const uint lane_base = group_id.y == 0u ? p2.y : p2.z;
  const uint row_base = lane_base + session.Load(0u) * p2.w;
  kv_block[thread_index] = active ? state[source_base + element] : 0.0f;
  GroupMemoryBarrierWithGroupSync();

  if (mode == 0u) {
    if (active) kv_cache.Store(row_base + element * 4u, asuint(kv_block[thread_index]));
    return;
  }
  if (mode == 1u) {
    if ((thread_index & 1u) == 0u && element + 1u < width) {
      const uint packed = f32tof16(kv_block[thread_index])
        | (f32tof16(kv_block[thread_index + 1u]) << 16u);
      kv_cache.Store(row_base + element * 2u, packed);
    }
    return;
  }

  if (mode == 4u) {
    [unroll]
    for (uint stride = 1u; stride < 32u; stride <<= 1u) {
      const uint partner = thread_index ^ stride;
      const float self_value = kv_block[thread_index];
      const float partner_value = kv_block[partner];
      GroupMemoryBarrierWithGroupSync();
      kv_block[thread_index] = (thread_index & stride) == 0u
        ? self_value + partner_value
        : partner_value - self_value;
      GroupMemoryBarrierWithGroupSync();
    }
  }

  reduce_partials[thread_index] = abs(kv_block[thread_index]);
  GroupMemoryBarrierWithGroupSync();
  [unroll]
  for (uint stride = 16u; stride > 0u; stride >>= 1u) {
    if (thread_index < stride) {
      reduce_partials[thread_index] = max(reduce_partials[thread_index], reduce_partials[thread_index + stride]);
    }
    GroupMemoryBarrierWithGroupSync();
  }
  const float range = mode == 2u ? 127.0f : (mode == 3u ? 7.0f : 3.5f);
  const float scale = reduce_partials[0] / range;
  const float inverse = scale > 0.0f ? 1.0f / scale : 0.0f;
  const uint block_base = row_base + group_id.x * kv_block_bytes(mode);
  if (thread_index == 0u) kv_cache.Store(block_base, asuint(scale));
  GroupMemoryBarrierWithGroupSync();

  if (mode == 2u && thread_index < 8u) {
    uint packed = 0u;
    [unroll]
    for (uint lane = 0u; lane < 4u; lane += 1u) {
      const float scaled = kv_block[thread_index * 4u + lane] * inverse;
      const int quant = (int)clamp(scaled >= 0.0f ? floor(scaled + 0.5f) : ceil(scaled - 0.5f), -127.0f, 127.0f);
      packed |= ((uint)quant & 255u) << (lane * 8u);
    }
    kv_cache.Store(block_base + 4u + thread_index * 4u, packed);
  } else if (mode == 3u && thread_index < 4u) {
    uint packed = 0u;
    [unroll]
    for (uint lane = 0u; lane < 8u; lane += 1u) {
      const float scaled = kv_block[thread_index * 8u + lane] * inverse;
      const int quant = (int)clamp(scaled >= 0.0f ? floor(scaled + 0.5f) : ceil(scaled - 0.5f), -8.0f, 7.0f);
      packed |= ((uint)(quant + 8) & 15u) << (lane * 4u);
    }
    kv_cache.Store(block_base + 4u + thread_index * 4u, packed);
  } else if (mode == 4u && thread_index == 0u) {
    uint packed0 = 0u;
    uint packed1 = 0u;
    uint packed2 = 0u;
    [unroll]
    for (uint item = 0u; item < 32u; item += 1u) {
      const float scaled = kv_block[item] * inverse;
      const int quant = (int)clamp(scaled >= 0.0f ? floor(scaled + 0.5f) : ceil(scaled - 0.5f), -4.0f, 3.0f);
      const uint code = (uint)(quant + 4) & 7u;
      const uint bit = item * 3u;
      if (bit < 32u) packed0 |= code << bit;
      else if (bit < 64u) packed1 |= code << (bit - 32u);
      else packed2 |= code << (bit - 64u);
      if ((bit & 31u) > 29u) {
        if (bit < 32u) packed1 |= code >> (32u - bit);
        else packed2 |= code >> (64u - bit);
      }
    }
    kv_cache.Store3(block_base + 4u, uint3(packed0, packed1, packed2));
  }
}

// --- causal attention (decode: one new token) -------------------------------
// One group per query head. Past K/V come from the packed resident cache,
// the current token's K/V from state. Replicates the CPU reference math exactly,
// including its per-step renormalizing division.
// p0.x heads, p0.y kv_heads, p0.z head_dim, p0.w window,
// p1.x scale bits (0 => rsqrt(head_dim)), p1.y query_offset,
// p1.z current_key_offset, p1.w current_value_offset,
// p2.x output_offset, p2.y cache_key_base bytes, p2.z cache_value_base bytes,
// p2.w cache row bytes; p3.x mode, p3.y kv_width. Active past = session dword 0.
[numthreads(256, 1, 1)]
void cs_attention(uint3 group_id : SV_GroupID, uint thread_index : SV_GroupIndex) {
  const uint head = group_id.x;
  const uint head_dim = p0.z;
  if (head >= p0.x || head_dim == 0u || head_dim > 512u) {
    return;
  }
  const uint groups = p0.x / p0.y;
  const uint kv_head = head / groups;
  const uint query_base = p1.y + head * head_dim;
  const uint past = session.Load(0u);
  const float raw_scale = asfloat(p1.x);
  const float scale = raw_scale > 0.0f ? raw_scale : rsqrt((float)head_dim);

  // Default q8_0 append is fused into the first query group for each KV
  // head. Current-token attention still reads state, so query groups do not
  // need a device-wide barrier; the command-list UAV barrier after this
  // dispatch publishes the row for the next replay.
  const uint query_groups_per_kv = p0.x / p0.y;
  if (p3.z != 0u && p3.x == 2u && (head % query_groups_per_kv) == 0u) {
    const uint blocks_per_head = head_dim / 32u;
    [loop]
    for (uint lane = 0u; lane < 2u; lane += 1u) {
      const uint source_base = lane == 0u ? p1.z : p1.w;
      const uint cache_base = lane == 0u ? p2.y : p2.z;
      [loop]
      for (uint block = 0u; block < blocks_per_head; block += 1u) {
        if (thread_index < 32u) {
          const uint source_element = kv_head * head_dim + block * 32u + thread_index;
          kv_block[thread_index] = state[source_base + source_element];
          reduce_partials[thread_index] = abs(kv_block[thread_index]);
        }
        GroupMemoryBarrierWithGroupSync();
        [unroll]
        for (uint stride = 16u; stride > 0u; stride >>= 1u) {
          if (thread_index < stride) {
            reduce_partials[thread_index] = max(reduce_partials[thread_index], reduce_partials[thread_index + stride]);
          }
          GroupMemoryBarrierWithGroupSync();
        }
        const float kv_scale = reduce_partials[0] / 127.0f;
        const float inverse = kv_scale > 0.0f ? 1.0f / kv_scale : 0.0f;
        const uint row_base = cache_base + past * p2.w;
        const uint packed_base = row_base + (kv_head * blocks_per_head + block) * 36u;
        if (thread_index == 0u) kv_cache.Store(packed_base, asuint(kv_scale));
        if (thread_index < 8u) {
          uint packed = 0u;
          [unroll]
          for (uint item = 0u; item < 4u; item += 1u) {
            const float scaled = kv_block[thread_index * 4u + item] * inverse;
            const int quant = (int)clamp(scaled >= 0.0f ? floor(scaled + 0.5f) : ceil(scaled - 0.5f), -127.0f, 127.0f);
            packed |= ((uint)quant & 255u) << (item * 8u);
          }
          kv_cache.Store(packed_base + 4u + thread_index * 4u, packed);
        }
        GroupMemoryBarrierWithGroupSync();
      }
    }
  }

  for (uint item = thread_index; item < head_dim; item += 256u) {
    attn_context[item] = 0.0f;
  }
  if (thread_index == 0u) {
    attn_state[0] = -3.402823466e+38f; // running max
    attn_state[1] = 0.0f;              // running sum
  }
  GroupMemoryBarrierWithGroupSync();

  const uint absolute_token = past;
  const uint first_key = (p0.w != 0u && absolute_token + 1u > p0.w) ? (absolute_token + 1u - p0.w) : 0u;

  [loop]
  for (uint key_token = first_key; key_token <= absolute_token; key_token += 1u) {
    const bool is_past = key_token < past;
    const uint local_key = is_past ? key_token : 0u;
    const uint kv_element = kv_head * head_dim;

    // q8_0 fast load: one packed dword load expands four adjacent values,
    // instead of issuing a scale + byte-address load for every scalar.
    if (is_past && p3.x == 2u) {
      const uint packed_count = (head_dim + 3u) / 4u;
      for (uint packed_index = thread_index; packed_index < packed_count; packed_index += 256u) {
        const uint element = kv_element + packed_index * 4u;
        const uint block = element >> 5u;
        const uint item = element & 31u;
        const uint key_base = p2.y + local_key * p2.w + block * 36u;
        const uint value_base = p2.z + local_key * p2.w + block * 36u;
        const float key_scale = asfloat(kv_cache.Load(key_base));
        const float value_scale = asfloat(kv_cache.Load(value_base));
        const uint key_packed = kv_cache.Load(key_base + 4u + item);
        const uint value_packed = kv_cache.Load(value_base + 4u + item);
        [unroll]
        for (uint lane = 0u; lane < 4u; lane += 1u) {
          const uint local_item = packed_index * 4u + lane;
          if (local_item < head_dim) {
            const uint key_raw = byte_of_word(key_packed, lane);
            const uint value_raw = byte_of_word(value_packed, lane);
            attn_key[local_item] = (float)((int)(key_raw << 24u) >> 24) * key_scale;
            attn_value[local_item] = (float)((int)(value_raw << 24u) >> 24) * value_scale;
          }
        }
      }
    }
    GroupMemoryBarrierWithGroupSync();

    float partial = 0.0f;
    for (uint item = thread_index; item < head_dim; item += 256u) {
      const float key_value = is_past
        ? (p3.x == 2u ? attn_key[item] : load_kv_element(p2.y, p2.w, local_key, kv_element + item, p3.x))
        : state[p1.z + kv_head * head_dim + item];
      partial += state[query_base + item] * key_value;
    }
    reduce_partials[thread_index] = partial;
    GroupMemoryBarrierWithGroupSync();
    [unroll]
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
      if (thread_index < stride) {
        reduce_partials[thread_index] += reduce_partials[thread_index + stride];
      }
      GroupMemoryBarrierWithGroupSync();
    }
    if (thread_index == 0u) {
      const float score = reduce_partials[0] * scale;
      const float running_max = attn_state[0];
      const float running_sum = attn_state[1];
      const float next_max = max(running_max, score);
      const float previous_scale = running_sum == 0.0f ? 0.0f : exp(running_max - next_max);
      const float score_scale = exp(score - next_max);
      const float next_sum = running_sum * previous_scale + score_scale;
      attn_state[0] = next_max;
      attn_state[2] = previous_scale * running_sum; // dst multiplier numerator
      attn_state[3] = score_scale;
      attn_state[1] = next_sum;
    }
    GroupMemoryBarrierWithGroupSync();
    const float dst_scale = attn_state[2];
    const float value_scale = attn_state[3];
    const float next_sum = attn_state[1];
    for (uint item = thread_index; item < head_dim; item += 256u) {
      const float value_value = is_past
        ? (p3.x == 2u ? attn_value[item] : load_kv_element(p2.z, p2.w, local_key, kv_element + item, p3.x))
        : state[p1.w + kv_head * head_dim + item];
      attn_context[item] = (attn_context[item] * dst_scale + value_value * value_scale) / next_sum;
    }
    GroupMemoryBarrierWithGroupSync();
  }

  for (uint item = thread_index; item < head_dim; item += 256u) {
    state[p2.x + head * head_dim + item] = attn_context[item];
  }
}

// --- elementwise ------------------------------------------------------------
// p0.x total, p0.y a_offset, p0.z b_offset, p0.w output_offset,
// p1.x rhs_elements (add) / scalar bits (scale), p1.y flags.
[numthreads(256, 1, 1)]
void cs_silu_mul(uint3 dispatch_id : SV_DispatchThreadID) {
  const uint index = dispatch_id.x;
  if (index >= p0.x) {
    return;
  }
  const float gate = state[p0.y + index];
  state[p0.w + index] = (gate / (1.0f + exp(-gate))) * state[p0.z + index];
}

[numthreads(256, 1, 1)]
void cs_gelu_mul(uint3 dispatch_id : SV_DispatchThreadID) {
  const uint index = dispatch_id.x;
  if (index >= p0.x) {
    return;
  }
  const float gate = state[p0.y + index];
  const float cubic = gate * gate * gate;
  // Driver tanh is exp-based and NaNs for |x| > ~44; tanh(+-15) is exactly
  // +-1 in float, so the clamp is value-preserving.
  const float tanh_argument = clamp(0.7978845608028654f * (gate + 0.044715f * cubic), -15.0f, 15.0f);
  const float activated = 0.5f * gate * (1.0f + tanh(tanh_argument));
  state[p0.w + index] = activated * state[p0.z + index];
}

[numthreads(256, 1, 1)]
void cs_add(uint3 dispatch_id : SV_DispatchThreadID) {
  const uint index = dispatch_id.x;
  if (index >= p0.x) {
    return;
  }
  state[p0.w + index] = state[p0.y + index] + state[p0.z + index % p1.x];
}

[numthreads(256, 1, 1)]
void cs_scale(uint3 dispatch_id : SV_DispatchThreadID) {
  const uint index = dispatch_id.x;
  if (index >= p0.x) {
    return;
  }
  const float factor = p1.y != 0u ? state[p0.z] : asfloat(p1.x);
  state[p0.w + index] = state[p0.y + index] * factor;
}

// Per-head RMS norm (Gemma4 q/k norms). One group per (batch*head).
// p0.x head_dim, p0.y heads, p0.z input_offset, p0.w weight_offset
// (0xFFFFFFFF = unweighted), p1.x output_offset, p1.y epsilon bits.
[numthreads(256, 1, 1)]
void cs_rms_norm_heads(uint3 group_id : SV_GroupID, uint thread_index : SV_GroupIndex) {
  const uint head_dim = p0.x;
  const uint base = p0.z + group_id.x * head_dim;
  const uint out_base = p1.x + group_id.x * head_dim;
  const uint last = head_dim - 1u;
  float sum = 0.0f;
  for (uint index = thread_index; index < head_dim; index += 256u) {
    const float value = state[base + min(index, last)];
    sum += value * value;
  }
  reduce_partials[thread_index] = sum;
  GroupMemoryBarrierWithGroupSync();
  [unroll]
  for (uint stride = 128u; stride > 0u; stride >>= 1u) {
    if (thread_index < stride) {
      reduce_partials[thread_index] += reduce_partials[thread_index + stride];
    }
    GroupMemoryBarrierWithGroupSync();
  }
  const float inverse_rms = rsqrt(reduce_partials[0] / (float)head_dim + asfloat(p1.y));
  for (uint index = thread_index; index < head_dim; index += 256u) {
    if (index < head_dim) {
      // Same speculative-load rule as cs_rms_norm: never let the sentinel
      // reach address arithmetic.
      const uint weight_index = (p0.w == 0xFFFFFFFFu ? 0u : p0.w) + index;
      const float weight_value = state[weight_index];
      const float weight = p0.w == 0xFFFFFFFFu ? 1.0f : weight_value;
      state[out_base + index] = state[base + index] * inverse_rms * weight;
    }
  }
}
)";

struct KernelSet {
    ComOwner<ID3D12PipelineState> matvec_f32;
    ComOwner<ID3D12PipelineState> matvec_q4k;
    ComOwner<ID3D12PipelineState> matvec_q4k_coop;
    ComOwner<ID3D12PipelineState> matvec_q5k;
    ComOwner<ID3D12PipelineState> matvec_q6k;
    ComOwner<ID3D12PipelineState> matvec_q8_0;
    ComOwner<ID3D12PipelineState> matvec_q4_0;
    ComOwner<ID3D12PipelineState> matvec_iq4_nl;
    ComOwner<ID3D12PipelineState> matvec_iq4_xs;
    ComOwner<ID3D12PipelineState> rms_norm;
    ComOwner<ID3D12PipelineState> rms_norm_heads;
    ComOwner<ID3D12PipelineState> rope;
    ComOwner<ID3D12PipelineState> kv_append;
    ComOwner<ID3D12PipelineState> attention;
    ComOwner<ID3D12PipelineState> silu_mul;
    ComOwner<ID3D12PipelineState> gelu_mul;
    ComOwner<ID3D12PipelineState> add;
    ComOwner<ID3D12PipelineState> scale;
    ComOwner<ID3D12PipelineState> quantize_q8_1;
    ComOwner<ID3D12PipelineState> matvec_q4k_q8_1;
    ComOwner<ID3D12PipelineState> matvec_q5k_q8_1;
    ComOwner<ID3D12PipelineState> matvec_q6k_q8_1;
    ComOwner<ID3D12PipelineState> matvec_q8_0_q8_1;
    ComOwner<ID3D12PipelineState> matvec_q4_0_q8_1;
};

struct DeviceBundle {
    ComOwner<IDXGIAdapter1> adapter;
    ComOwner<ID3D12Device> device;
    ComOwner<ID3D12CommandQueue> queue;
    ComOwner<ID3D12Fence> fence;
    HANDLE fence_event{};
    std::uint64_t fence_value{};
    ComOwner<ID3D12RootSignature> root_signature;
    KernelSet kernels;
    std::string adapter_name;
    bool cache_coherent_uma{};
    bool integer_dot_available{};
    std::string integer_dot_reason;

    ~DeviceBundle() {
        if (fence_event != nullptr) CloseHandle(fence_event);
    }
};

ComOwner<IDXGIAdapter1> select_adapter(
    std::string* adapter_name,
    bool require_coherent_uma
) {
    ComOwner<IDXGIFactory1> factory;
    check_hresult("CreateDXGIFactory1",
        CreateDXGIFactory1(__uuidof(IDXGIFactory1), reinterpret_cast<void**>(factory.put())));
    LUID requested{};
    const bool has_requested = parse_requested_luid(requested);
    ComOwner<IDXGIAdapter1> best;
    DXGI_ADAPTER_DESC1 best_description{};
    std::uint64_t best_dedicated_bytes{};
    bool matching_hardware_seen = false;
    bool d3d12_device_seen = false;
    for (UINT index = 0;; ++index) {
        ComOwner<IDXGIAdapter1> adapter;
        if (factory->EnumAdapters1(index, adapter.put()) == DXGI_ERROR_NOT_FOUND) break;
        DXGI_ADAPTER_DESC1 description{};
        if (FAILED(adapter->GetDesc1(&description))) continue;
        if ((description.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) != 0) continue;
        if (has_requested
            && (description.AdapterLuid.HighPart != requested.HighPart
                || description.AdapterLuid.LowPart != requested.LowPart)) {
            continue;
        }
        matching_hardware_seen = true;

        // Enumeration alone does not prove that the adapter can execute the
        // backend. Probe the exact FL 11_0 contract before selecting it so an
        // unusable first adapter cannot hide a later compatible one.
        ComOwner<ID3D12Device> probe_device;
        const auto device_result = D3D12CreateDevice(
            adapter.value, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(probe_device.put()));
        if (FAILED(device_result)) {
            if (has_requested) {
                throw GpuUnavailable(
                    "requested DXGI adapter cannot create a D3D12 FL 11_0 device: "
                    + narrow_name(description.Description));
            }
            continue;
        }
        d3d12_device_seen = true;

        D3D12_FEATURE_DATA_ARCHITECTURE architecture{};
        const bool coherent_uma = SUCCEEDED(probe_device->CheckFeatureSupport(
            D3D12_FEATURE_ARCHITECTURE, &architecture, sizeof(architecture)))
            && architecture.UMA != 0
            && architecture.CacheCoherentUMA != 0;
        if (require_coherent_uma && !coherent_uma) {
            if (has_requested) {
                throw GpuUnavailable(
                    "requested DXGI adapter is not cache-coherent UMA: "
                    + narrow_name(description.Description));
            }
            continue;
        }

        if (has_requested) {
            if (adapter_name != nullptr) *adapter_name = narrow_name(description.Description);
            return adapter;
        }
        // Same deterministic no-identity rule as Lizard: prefer the largest
        // dedicated-memory candidate and keep enumeration order for ties.
        if (!best || description.DedicatedVideoMemory > best_dedicated_bytes) {
            best_dedicated_bytes = description.DedicatedVideoMemory;
            best_description = description;
            best = std::move(adapter);
        }
    }
    if (best) {
        if (adapter_name != nullptr) *adapter_name = narrow_name(best_description.Description);
        return best;
    }
    if (has_requested) {
        throw GpuUnavailable("requested hardware DXGI adapter was not found");
    }
    if (!matching_hardware_seen) {
        throw GpuUnavailable("no hardware DXGI adapter was available");
    }
    if (!d3d12_device_seen) {
        throw GpuUnavailable("no hardware DXGI adapter could create a D3D12 FL 11_0 device");
    }
    if (require_coherent_uma) {
        throw GpuUnavailable("no D3D12 adapter reports cache-coherent UMA for Caterpillar's zero-copy backend");
    }
    throw GpuUnavailable("no hardware DXGI adapter matched the Caterpillar GPU selection");
}

// FXC compiles Caterpillar's 19 compute kernels at OPTIMIZATION_LEVEL3 on every
// worker process start. On slower/first-run integrated GPUs that front-end
// compile dominates cold model-load time (tens of seconds to over a minute),
// and because each benchmark row spawns a fresh worker it is repaid every row -
// the same problem lizard-native already solved. The compiled DXBC bytecode is
// device-independent, so it is cached on disk keyed by the shader source +
// entry point; a later process reuses the blob and skips D3DCompile entirely.
// The cache dir and on/off toggle are shared with lizard-native
// (LIZARD_NATIVE_SHADER_CACHE / LIZARD_NATIVE_SHADER_CACHE_DIR); a distinct file
// prefix plus the source hash keep the two runners' blobs from colliding.
// Corruption or a source change self-heals: a bad blob fails
// CreateComputePipelineState and the caller falls back to a fresh compile that
// overwrites the cache.
std::uint64_t caterpillar_fnv1a_hash(const void* data, std::size_t length) {
    const auto* bytes = static_cast<const unsigned char*>(data);
    std::uint64_t hash = 1469598103934665603ull;
    for (std::size_t index = 0; index < length; index += 1) {
        hash ^= static_cast<std::uint64_t>(bytes[index]);
        hash *= 1099511628211ull;
    }
    return hash;
}

bool caterpillar_shader_cache_disabled() {
    char value[8] = {};
    return GetEnvironmentVariableA("LIZARD_NATIVE_SHADER_CACHE", value, sizeof(value)) != 0
        && std::string(value) == "0";
}

std::filesystem::path caterpillar_shader_cache_path(std::uint64_t source_hash, const std::string& variant) {
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
    name << "caterpillar_cs50_" << std::hex << source_hash << "_" << variant << ".cso";
    return base / name.str();
}

std::vector<unsigned char> caterpillar_read_cached_shader_blob(const std::filesystem::path& path) {
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
    if (!input.read(reinterpret_cast<char*>(buffer.data()), size)) {
        return {};
    }
    return buffer;
}

void caterpillar_write_cached_shader_blob(const std::filesystem::path& path, const void* data, std::size_t length) {
    std::error_code ec;
    std::filesystem::create_directories(path.parent_path(), ec);
    // Write to a unique temp file then rename so a concurrent worker never reads
    // a half-written blob.
    const std::filesystem::path temp = path.string() + "." + std::to_string(GetCurrentProcessId()) + ".tmp";
    {
        std::ofstream output(temp, std::ios::binary | std::ios::trunc);
        if (!output) {
            return;
        }
        output.write(static_cast<const char*>(data), static_cast<std::streamsize>(length));
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

ComOwner<ID3D12PipelineState> compile_kernel(
    ID3D12Device* device,
    ID3D12RootSignature* root_signature,
    const char* entry
) {
    // Cache key: shader source identity + entry point. The source hash is
    // computed once; a shader edit changes it and invalidates every cached blob.
    static const std::uint64_t source_hash =
        caterpillar_fnv1a_hash(k_kernel_source, sizeof(k_kernel_source) - 1);
    const bool cache_enabled = !caterpillar_shader_cache_disabled();
    const std::filesystem::path cache_file = cache_enabled
        ? caterpillar_shader_cache_path(source_hash, std::string(entry))
        : std::filesystem::path();

    const auto create_from_bytecode = [&](const void* bytecode, std::size_t length,
                                          ComOwner<ID3D12PipelineState>& pipeline) -> bool {
        D3D12_COMPUTE_PIPELINE_STATE_DESC description{};
        description.pRootSignature = root_signature;
        description.CS.pShaderBytecode = bytecode;
        description.CS.BytecodeLength = length;
        return SUCCEEDED(device->CreateComputePipelineState(&description, IID_PPV_ARGS(pipeline.put())));
    };

    // Fast path: a previously compiled DXBC blob skips the FXC front-end.
    if (cache_enabled && !cache_file.empty()) {
        const std::vector<unsigned char> cached = caterpillar_read_cached_shader_blob(cache_file);
        ComOwner<ID3D12PipelineState> pipeline;
        if (!cached.empty() && create_from_bytecode(cached.data(), cached.size(), pipeline)) {
            return pipeline;
        }
    }

    ComOwner<ID3DBlob> shader;
    ComOwner<ID3DBlob> errors;
    const auto hr = D3DCompile(
        k_kernel_source,
        sizeof(k_kernel_source) - 1,
        "caterpillar_kernels",
        nullptr,
        nullptr,
        entry,
        "cs_5_0",
        D3DCOMPILE_OPTIMIZATION_LEVEL3,
        0,
        shader.put(),
        errors.put()
    );
    if (FAILED(hr)) {
        std::string message = "D3DCompile(" + std::string(entry) + ") failed";
        if (errors && errors->GetBufferPointer() != nullptr) {
            message += ": ";
            message += static_cast<const char*>(errors->GetBufferPointer());
        }
        throw GpuUnavailable(message);
    }
    D3D12_COMPUTE_PIPELINE_STATE_DESC description{};
    description.pRootSignature = root_signature;
    description.CS.pShaderBytecode = shader->GetBufferPointer();
    description.CS.BytecodeLength = shader->GetBufferSize();
    ComOwner<ID3D12PipelineState> pipeline;
    check_hresult("CreateComputePipelineState",
        device->CreateComputePipelineState(&description, IID_PPV_ARGS(pipeline.put())));
    // Persist the freshly compiled blob so the next worker process reuses it.
    if (cache_enabled && !cache_file.empty()) {
        caterpillar_write_cached_shader_blob(cache_file, shader->GetBufferPointer(), shader->GetBufferSize());
    }
    return pipeline;
}

ComOwner<ID3D12PipelineState> create_embedded_kernel(
    ID3D12Device* device,
    ID3D12RootSignature* root_signature,
    const void* bytecode,
    std::size_t bytecode_bytes
) {
    D3D12_COMPUTE_PIPELINE_STATE_DESC description{};
    description.pRootSignature = root_signature;
    description.CS.pShaderBytecode = bytecode;
    description.CS.BytecodeLength = bytecode_bytes;
    ComOwner<ID3D12PipelineState> pipeline;
    check_hresult("CreateComputePipelineState(DXIL)",
        device->CreateComputePipelineState(&description, IID_PPV_ARGS(pipeline.put())));
    return pipeline;
}

std::unique_ptr<DeviceBundle> create_device_bundle(bool require_coherent_uma) {
    auto bundle = std::make_unique<DeviceBundle>();
    bundle->adapter = select_adapter(&bundle->adapter_name, require_coherent_uma);
    check_hresult("D3D12CreateDevice", D3D12CreateDevice(
        bundle->adapter.value, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(bundle->device.put())));

    D3D12_FEATURE_DATA_ARCHITECTURE architecture{};
    if (SUCCEEDED(bundle->device->CheckFeatureSupport(
            D3D12_FEATURE_ARCHITECTURE, &architecture, sizeof(architecture)))) {
        bundle->cache_coherent_uma = architecture.UMA != 0 && architecture.CacheCoherentUMA != 0;
    }
    if (require_coherent_uma && !bundle->cache_coherent_uma) {
        throw GpuUnavailable(
            "Caterpillar Phase 3 requires a cache-coherent UMA adapter; "
            + bundle->adapter_name + " does not report CacheCoherentUMA");
    }

    D3D12_COMMAND_QUEUE_DESC queue_description{};
    queue_description.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
    check_hresult("CreateCommandQueue", bundle->device->CreateCommandQueue(
        &queue_description, IID_PPV_ARGS(bundle->queue.put())));
    check_hresult("CreateFence", bundle->device->CreateFence(
        0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(bundle->fence.put())));
    bundle->fence_event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (bundle->fence_event == nullptr) throw GpuUnavailable("CreateEvent failed");

    std::array<D3D12_ROOT_PARAMETER, 6> parameters{};
    parameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    parameters[0].Constants.ShaderRegister = 0;
    parameters[0].Constants.Num32BitValues = 16;
    parameters[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
    parameters[1].Descriptor.ShaderRegister = 0;
    parameters[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
    parameters[2].Descriptor.ShaderRegister = 1;
    parameters[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
    parameters[3].Descriptor.ShaderRegister = 2;
    parameters[4].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    parameters[4].Descriptor.ShaderRegister = 0;
    parameters[5].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    parameters[5].Descriptor.ShaderRegister = 1;
    for (auto& parameter : parameters) {
        parameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    }
    D3D12_ROOT_SIGNATURE_DESC signature_description{};
    signature_description.NumParameters = static_cast<UINT>(parameters.size());
    signature_description.pParameters = parameters.data();
    ComOwner<ID3DBlob> serialized;
    ComOwner<ID3DBlob> errors;
    const auto serialize_result = D3D12SerializeRootSignature(
        &signature_description, D3D_ROOT_SIGNATURE_VERSION_1, serialized.put(), errors.put());
    if (FAILED(serialize_result)) {
        std::string message = "D3D12SerializeRootSignature failed";
        if (errors && errors->GetBufferPointer() != nullptr) {
            message += ": ";
            message += static_cast<const char*>(errors->GetBufferPointer());
        }
        throw GpuUnavailable(message);
    }
    check_hresult("CreateRootSignature", bundle->device->CreateRootSignature(
        0, serialized->GetBufferPointer(), serialized->GetBufferSize(),
        IID_PPV_ARGS(bundle->root_signature.put())));

    auto& kernels = bundle->kernels;
    kernels.matvec_f32 = compile_kernel(bundle->device.value, bundle->root_signature.value, "cs_matvec_f32");
    kernels.matvec_q4k = compile_kernel(bundle->device.value, bundle->root_signature.value, "cs_matvec_q4k");
    kernels.matvec_q4k_coop = compile_kernel(bundle->device.value, bundle->root_signature.value, "cs_matvec_q4k_coop");
    kernels.matvec_q5k = compile_kernel(bundle->device.value, bundle->root_signature.value, "cs_matvec_q5k");
    kernels.matvec_q6k = compile_kernel(bundle->device.value, bundle->root_signature.value, "cs_matvec_q6k");
    kernels.matvec_q8_0 = compile_kernel(bundle->device.value, bundle->root_signature.value, "cs_matvec_q8_0");
    kernels.matvec_q4_0 = compile_kernel(bundle->device.value, bundle->root_signature.value, "cs_matvec_q4_0");
    kernels.matvec_iq4_nl = compile_kernel(bundle->device.value, bundle->root_signature.value, "cs_matvec_iq4_nl");
    kernels.matvec_iq4_xs = compile_kernel(bundle->device.value, bundle->root_signature.value, "cs_matvec_iq4_xs");
    kernels.rms_norm = compile_kernel(bundle->device.value, bundle->root_signature.value, "cs_rms_norm");
    kernels.rms_norm_heads = compile_kernel(bundle->device.value, bundle->root_signature.value, "cs_rms_norm_heads");
    kernels.rope = compile_kernel(bundle->device.value, bundle->root_signature.value, "cs_rope");
    kernels.kv_append = compile_kernel(bundle->device.value, bundle->root_signature.value, "cs_kv_append");
    kernels.attention = compile_kernel(bundle->device.value, bundle->root_signature.value, "cs_attention");
    kernels.silu_mul = compile_kernel(bundle->device.value, bundle->root_signature.value, "cs_silu_mul");
    kernels.gelu_mul = compile_kernel(bundle->device.value, bundle->root_signature.value, "cs_gelu_mul");
    kernels.add = compile_kernel(bundle->device.value, bundle->root_signature.value, "cs_add");
    kernels.scale = compile_kernel(bundle->device.value, bundle->root_signature.value, "cs_scale");

#ifdef CATERPILLAR_HAVE_DXIL
    // Raw feature values keep this compatible with older MinGW header sets:
    // D3D12_FEATURE_SHADER_MODEL = 7, D3D_SHADER_MODEL_6_4 = 0x64.
    struct ShaderModelProbe { UINT highest_shader_model; } shader_model{0x64};
    const auto shader_model_result = bundle->device->CheckFeatureSupport(
        static_cast<D3D12_FEATURE>(7), &shader_model, sizeof(shader_model));
    if (SUCCEEDED(shader_model_result) && shader_model.highest_shader_model >= 0x64) {
        try {
            kernels.quantize_q8_1 = create_embedded_kernel(
                bundle->device.value, bundle->root_signature.value,
                g_caterpillar_quantize_q8_1, sizeof(g_caterpillar_quantize_q8_1));
            kernels.matvec_q4k_q8_1 = create_embedded_kernel(
                bundle->device.value, bundle->root_signature.value,
                g_caterpillar_q4k_q8_1, sizeof(g_caterpillar_q4k_q8_1));
            kernels.matvec_q5k_q8_1 = create_embedded_kernel(
                bundle->device.value, bundle->root_signature.value,
                g_caterpillar_q5k_q8_1, sizeof(g_caterpillar_q5k_q8_1));
            kernels.matvec_q6k_q8_1 = create_embedded_kernel(
                bundle->device.value, bundle->root_signature.value,
                g_caterpillar_q6k_q8_1, sizeof(g_caterpillar_q6k_q8_1));
            kernels.matvec_q8_0_q8_1 = create_embedded_kernel(
                bundle->device.value, bundle->root_signature.value,
                g_caterpillar_q8_0_q8_1, sizeof(g_caterpillar_q8_0_q8_1));
            kernels.matvec_q4_0_q8_1 = create_embedded_kernel(
                bundle->device.value, bundle->root_signature.value,
                g_caterpillar_q4_0_q8_1, sizeof(g_caterpillar_q4_0_q8_1));
            bundle->integer_dot_available = true;
            bundle->integer_dot_reason = "Shader Model 6.4 packed integer dot kernels are available";
        } catch (const std::exception& error) {
            kernels.quantize_q8_1.reset();
            kernels.matvec_q4k_q8_1.reset();
            kernels.matvec_q5k_q8_1.reset();
            kernels.matvec_q6k_q8_1.reset();
            kernels.matvec_q8_0_q8_1.reset();
            kernels.matvec_q4_0_q8_1.reset();
            bundle->integer_dot_reason = std::string("DXIL pipeline creation failed: ") + error.what();
        }
    } else {
        bundle->integer_dot_reason = FAILED(shader_model_result)
            ? "driver rejected the Shader Model 6.4 feature probe"
            : "device reports a shader model below 6.4";
    }
#else
    bundle->integer_dot_reason = "build has no embedded DXIL kernels";
#endif
    return bundle;
}

ComOwner<ID3D12Resource> create_uma_buffer(
    ID3D12Device* device,
    std::uint64_t bytes,
    D3D12_RESOURCE_FLAGS flags,
    D3D12_RESOURCE_STATES initial_state
) {
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = D3D12_HEAP_TYPE_CUSTOM;
    heap.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_WRITE_BACK;
    heap.MemoryPoolPreference = D3D12_MEMORY_POOL_L0;
    D3D12_RESOURCE_DESC description{};
    description.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    description.Width = std::max<std::uint64_t>(bytes, 16);
    description.Height = 1;
    description.DepthOrArraySize = 1;
    description.MipLevels = 1;
    description.SampleDesc.Count = 1;
    description.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    description.Flags = flags;
    ComOwner<ID3D12Resource> resource;
    check_hresult("CreateCommittedResource", device->CreateCommittedResource(
        &heap, D3D12_HEAP_FLAG_NONE, &description, initial_state, nullptr,
        IID_PPV_ARGS(resource.put())));
    return resource;
}

// Step 08 tuning: immutable weight buffers use WRITE_COMBINE CPU pages instead
// of WRITE_BACK. On Intel UMA a WRITE_BACK (cache-coherent) resource makes
// EVERY GPU read snoop the CPU cache hierarchy, capping sustained streaming at
// ~12 GB/s on Tiger Lake - exactly the measured matmul ceiling that no kernel
// shape change could move. Weights are written once at staging (sequential CPU
// writes, which write-combining handles well), unmapped, and never read by the
// CPU again, so nothing needs coherence and the GPU streams at full rate.
ComOwner<ID3D12Resource> create_gpu_weight_buffer(
    ID3D12Device* device,
    std::uint64_t bytes,
    D3D12_RESOURCE_FLAGS flags,
    D3D12_RESOURCE_STATES initial_state
) {
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = D3D12_HEAP_TYPE_CUSTOM;
    heap.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_WRITE_COMBINE;
    heap.MemoryPoolPreference = D3D12_MEMORY_POOL_L0;
    D3D12_RESOURCE_DESC description{};
    description.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    description.Width = std::max<std::uint64_t>(bytes, 16);
    description.Height = 1;
    description.DepthOrArraySize = 1;
    description.MipLevels = 1;
    description.SampleDesc.Count = 1;
    description.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    description.Flags = flags;
    ComOwner<ID3D12Resource> resource;
    check_hresult("CreateCommittedResource(weights)", device->CreateCommittedResource(
        &heap, D3D12_HEAP_FLAG_NONE, &description, initial_state, nullptr,
        IID_PPV_ARGS(resource.put())));
    return resource;
}

void* map_whole(ID3D12Resource* resource) {
    void* pointer{};
    check_hresult("Map", resource->Map(0, nullptr, &pointer));
    return pointer;
}

// --- staging transforms -----------------------------------------------------
// Column-major block interleave: staged_offset(block, row) = (block * rows +
// row) * staged_block_bytes, so adjacent rows in a thread group stream
// adjacent blocks.

struct StagedWeight {
    std::uint64_t bytes{};
    std::uint32_t kind{};  // matches the kernel selector below
};

enum class WeightKind : std::uint32_t {
    f32 = 0,
    q4k = 1,
    q6k = 2,
    q8_0 = 3,
    q4_0 = 4,
    q5k = 5,
    iq4_nl = 6,
    iq4_xs = 7,
};

std::uint64_t staged_weight_bytes_for(const GgufTensorInfo& tensor) {
    const auto elements = tensor_element_count(tensor);
    switch (tensor.type) {
        case 0: return elements * 4ULL;
        case 1: return elements * 4ULL;                    // F16 widened to F32
        case 2: return (elements / 32ULL) * 18ULL;         // Q4_0 raw blocks
        case 8: return (elements / 32ULL) * 36ULL;         // Q8_0 f32-d blocks
        case 12: return (elements / 256ULL) * 144ULL;      // Q4_K raw blocks
        case 13: return (elements / 256ULL) * 176ULL;      // Q5_K raw blocks
        case 14: return (elements / 256ULL) * 212ULL;      // Q6_K f32-d blocks
        case 20: return (elements / 32ULL) * 18ULL;        // IQ4_NL raw blocks
        case 23: return (elements / 256ULL) * 136ULL;      // IQ4_XS raw blocks
        default:
            throw GpuUnavailable("unsupported GGML weight type for GPU staging: " + std::to_string(tensor.type));
    }
}

WeightKind weight_kind_for(std::uint32_t ggml_type) {
    switch (ggml_type) {
        case 0: case 1: return WeightKind::f32;
        case 2: return WeightKind::q4_0;
        case 8: return WeightKind::q8_0;
        case 12: return WeightKind::q4k;
        case 13: return WeightKind::q5k;
        case 14: return WeightKind::q6k;
        case 20: return WeightKind::iq4_nl;
        case 23: return WeightKind::iq4_xs;
        default:
            throw GpuUnavailable("unsupported GGML weight type for GPU kernels: " + std::to_string(ggml_type));
    }
}

float fp16_bits_to_float(std::uint16_t bits) {
    const std::uint32_t sign = (bits & 0x8000U) << 16U;
    const std::uint32_t exponent = (bits >> 10U) & 0x1FU;
    const std::uint32_t mantissa = bits & 0x3FFU;
    std::uint32_t result;
    if (exponent == 0U) {
        if (mantissa == 0U) {
            result = sign;
        } else {
            auto shifted_exponent = 127U - 15U + 1U;
            auto shifted_mantissa = mantissa;
            while ((shifted_mantissa & 0x400U) == 0U) {
                shifted_mantissa <<= 1U;
                --shifted_exponent;
            }
            result = sign | (shifted_exponent << 23U) | ((shifted_mantissa & 0x3FFU) << 13U);
        }
    } else if (exponent == 31U) {
        result = sign | 0x7F800000U | (mantissa << 13U);
    } else {
        result = sign | ((exponent - 15U + 127U) << 23U) | (mantissa << 13U);
    }
    float value;
    std::memcpy(&value, &result, sizeof(value));
    return value;
}

void stage_weight_tensor(
    const QuantizedTensorView& view,
    std::byte* destination
) {
    // GGUF dimension order is [ne0, ne1] = [row width, row count].
    const auto& tensor = *view.tensor;
    const auto columns = static_cast<std::size_t>(tensor.dimensions[0]);
    const auto rows = static_cast<std::size_t>(tensor.dimensions[1]);
    const auto* source = view.bytes.data();
    switch (tensor.type) {
        case 0: {  // F32 row-major copy
            std::memcpy(destination, source, rows * columns * 4);
            break;
        }
        case 1: {  // F16 -> F32 widen, row-major
            const auto* halves = reinterpret_cast<const std::uint16_t*>(source);
            auto* out = reinterpret_cast<float*>(destination);
            for (std::size_t index = 0; index < rows * columns; ++index) {
                out[index] = fp16_bits_to_float(halves[index]);
            }
            break;
        }
        case 2: {  // Q4_0: 18-byte raw blocks, column-major interleave
            const auto blocks_per_row = columns / 32;
            for (std::size_t row = 0; row < rows; ++row) {
                for (std::size_t block = 0; block < blocks_per_row; ++block) {
                    std::memcpy(
                        destination + (block * rows + row) * 18,
                        source + (row * blocks_per_row + block) * 18,
                        18);
                }
            }
            break;
        }
        case 8: {  // Q8_0: 34 -> 36 (f32 d + 32 int8), column-major
            const auto blocks_per_row = columns / 32;
            for (std::size_t row = 0; row < rows; ++row) {
                for (std::size_t block = 0; block < blocks_per_row; ++block) {
                    const auto* raw = source + (row * blocks_per_row + block) * 34;
                    auto* out = destination + (block * rows + row) * 36;
                    std::uint16_t d_bits;
                    std::memcpy(&d_bits, raw, 2);
                    const float d = fp16_bits_to_float(d_bits);
                    std::memcpy(out, &d, 4);
                    std::memcpy(out + 4, raw + 2, 32);
                }
            }
            break;
        }
        case 12: {  // Q4_K: raw 144-byte blocks, column-major
            const auto blocks_per_row = columns / 256;
            for (std::size_t row = 0; row < rows; ++row) {
                for (std::size_t block = 0; block < blocks_per_row; ++block) {
                    std::memcpy(
                        destination + (block * rows + row) * 144,
                        source + (row * blocks_per_row + block) * 144,
                        144);
                }
            }
            break;
        }
        case 13: {  // Q5_K: raw 176-byte blocks, column-major
            const auto blocks_per_row = columns / 256;
            for (std::size_t row = 0; row < rows; ++row) {
                for (std::size_t block = 0; block < blocks_per_row; ++block) {
                    std::memcpy(
                        destination + (block * rows + row) * 176,
                        source + (row * blocks_per_row + block) * 176,
                        176);
                }
            }
            break;
        }
        case 14: {  // Q6_K: 210 -> 212 ([0..207] payload, f32 d), column-major
            const auto blocks_per_row = columns / 256;
            for (std::size_t row = 0; row < rows; ++row) {
                for (std::size_t block = 0; block < blocks_per_row; ++block) {
                    const auto* raw = source + (row * blocks_per_row + block) * 210;
                    auto* out = destination + (block * rows + row) * 212;
                    std::memcpy(out, raw, 208);
                    std::uint16_t d_bits;
                    std::memcpy(&d_bits, raw + 208, 2);
                    const float d = fp16_bits_to_float(d_bits);
                    std::memcpy(out + 208, &d, 4);
                }
            }
            break;
        }
        case 20: {  // IQ4_NL: raw 18-byte blocks, column-major interleave
            const auto blocks_per_row = columns / 32;
            for (std::size_t row = 0; row < rows; ++row) {
                for (std::size_t block = 0; block < blocks_per_row; ++block) {
                    std::memcpy(
                        destination + (block * rows + row) * 18,
                        source + (row * blocks_per_row + block) * 18,
                        18);
                }
            }
            break;
        }
        case 23: {  // IQ4_XS: raw 136-byte blocks, column-major interleave
            const auto blocks_per_row = columns / 256;
            for (std::size_t row = 0; row < rows; ++row) {
                for (std::size_t block = 0; block < blocks_per_row; ++block) {
                    std::memcpy(
                        destination + (block * rows + row) * 136,
                        source + (row * blocks_per_row + block) * 136,
                        136);
                }
            }
            break;
        }
        default:
            throw GpuUnavailable("unsupported GGML weight type for GPU staging");
    }
}

constexpr std::size_t align_up(std::size_t value, std::size_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

}  // namespace

// ---------------------------------------------------------------------------
// GpuPlanExecution
// ---------------------------------------------------------------------------

struct GpuPlanExecution::Impl {
    std::unique_ptr<DeviceBundle> bundle;

    // State buffer: [0, arena_bytes) planned activations at their CPU
    // offsets, then externals (immutable + dynamic) at assigned offsets.
    ComOwner<ID3D12Resource> state_buffer;
    float* state_mapped{};
    std::size_t arena_floats{};
    std::size_t state_floats{};
    std::unordered_map<TensorId, std::size_t> external_offsets;  // float index
    bool integer_dot_enabled{};
    bool unqualified_integer_dot_formats{};
    std::size_t q8_1_scratch_offset{};  // dword index in state_buffer
    std::size_t q8_1_scratch_words{};

    // Dynamic externals uploaded per replay from the binding views.
    std::vector<TensorId> dynamic_externals;

    // Session constants: dword 0 = active past, dword 1 = position.
    ComOwner<ID3D12Resource> session_buffer;
    std::uint32_t* session_mapped{};

    // Persistent packed KV cache, one K and one V section per owning layer.
    ComOwner<ID3D12Resource> kv_buffer;
    std::byte* kv_mapped{};
    KvCacheMode kv_mode{KvCacheMode::q8_0};
    std::size_t kv_bytes{};
    std::size_t kv_capacity{};
    std::unordered_map<std::size_t, std::pair<std::size_t, std::size_t>> kv_layer_offsets;  // layer -> (k,v) byte offsets
    std::unordered_map<TensorId, std::size_t> kv_tensor_offsets;  // kv-input tensor -> byte offset
    std::unordered_map<std::size_t, std::size_t> kv_layer_widths;
    std::unordered_map<std::size_t, std::size_t> kv_layer_row_bytes;

    // Stable-split CPU tails need decoded float rows. Fully resident plans do
    // not allocate this mirror, which is what makes the q8_0 8k gate real.
    bool cpu_kv_mirror{};
    std::vector<float> kv_cpu_mapped;
    std::unordered_map<std::size_t, std::pair<std::size_t, std::size_t>> kv_cpu_layer_offsets;

    // Resident weights, one buffer per quantized matmul weight tensor.
    struct ResidentWeight {
        ComOwner<ID3D12Resource> buffer;
        WeightKind kind{};
    };
    std::unordered_map<TensorId, ResidentWeight> weights;

    // Recorded spans, keyed by first node index.
    // One replayable command list. A span records one segment normally; the
    // Step 09b hybrid FFN split cuts a span into several segments so the queue
    // can signal "FFN input ready" after the pre-FFN segment and wait for the
    // CPU row share before the segment that consumes the stitched vector.
    struct Segment {
        ComOwner<ID3D12CommandAllocator> allocator;
        ComOwner<ID3D12GraphicsCommandList> list;
        std::uint32_t query_first{};
        std::uint32_t query_count{};
        // After submitting this segment, signal hybrid_gpu_fence at
        // base + signal_input_ready (-1 = none).
        int signal_input_ready{-1};
        // Before submitting this segment, queue-wait hybrid_cpu_fence at
        // base + wait_cpu_share (-1 = none).
        int wait_cpu_share{-1};
    };
    // CPU share of one FFN gate/up pair: rows [first_row, rows) of both
    // matvecs, combined with SiLU straight into the silu destination of the
    // coherent mapped state buffer.
    struct HybridFfnTask {
        QuantizedTensorView gate_view;
        QuantizedTensorView up_view;
        std::size_t input_offset{};
        std::size_t output_offset{};
        std::size_t first_row{};
        std::size_t rows{};
    };
    struct RecordedSpan {
        std::vector<Segment> segments;
        std::vector<HybridFfnTask> hybrid_tasks;
        // Ranges (float offset, float count) to copy from the mapped state
        // buffer back into the CPU arena after the span completes.
        std::vector<std::pair<std::size_t, std::size_t>> copy_back;
        std::uint64_t q8_1_quantizations{};
        std::uint64_t integer_dot_matvecs{};
        std::uint64_t kv_device_appends{};
        // Node-class byte per recorded dispatch (timestamp pair order) and the
        // first query slot this span owns in the shared timestamp heap.
        std::vector<std::uint8_t> dispatch_classes;
        std::uint32_t query_base{};
    };
    std::map<std::size_t, RecordedSpan> recorded;

    // Hybrid FFN fences (Step 09b): the queue signals gpu_fence when the FFN
    // input vector is resident, the CPU signals cpu_fence when its row share
    // is written; queue Waits gate the consuming segment. Values are
    // monotonic across replays (base handed out per execute_span).
    ComOwner<ID3D12Fence> hybrid_gpu_fence;
    ComOwner<ID3D12Fence> hybrid_cpu_fence;
    std::uint64_t hybrid_value_next = 1;
    std::vector<float> hybrid_gate_scratch;
    std::vector<float> hybrid_up_scratch;

    // Per-dispatch timestamp instrumentation (Step 07 tuning). Zero capacity
    // means the queue rejected timestamps and recording skips the queries.
    ComOwner<ID3D12QueryHeap> timestamp_heap;
    ComOwner<ID3D12Resource> timestamp_readback;
    const std::uint64_t* timestamp_mapped{};
    std::uint64_t timestamp_frequency{};
    std::uint32_t timestamp_capacity{};

    std::uint64_t resident_weight_bytes{};

    void wait_for_gpu() {
        const auto value = ++bundle->fence_value;
        check_hresult("queue Signal", bundle->queue->Signal(bundle->fence.value, value));
        if (bundle->fence->GetCompletedValue() < value) {
            check_hresult("SetEventOnCompletion",
                bundle->fence->SetEventOnCompletion(value, bundle->fence_event));
            WaitForSingleObject(bundle->fence_event, INFINITE);
        }
        const auto removed = bundle->device->GetDeviceRemovedReason();
        if (removed != S_OK) throw_hresult("device removed during execution", removed);
    }
};

GpuPlanExecution::GpuPlanExecution(
    const Plan& plan,
    const QuantizedBindings& quantized,
    const Bindings& immutable_externals,
    const std::vector<TensorId>& dynamic_externals,
    const std::vector<DecoderGraph::KvBinding>& kv_inputs,
    const Config& config
) : impl_(std::make_unique<Impl>()) {
    auto& impl = *impl_;
    impl.bundle = create_device_bundle(config.require_cache_coherent_uma);
    impl.dynamic_externals = dynamic_externals;
    impl.kv_capacity = config.kv_capacity_tokens;
    impl.kv_mode = config.kv_cache_mode;
    impl.cpu_kv_mirror = config.cpu_kv_mirror;

    const auto& graph = plan.graph();

    telemetry_.integer_dot_available = impl.bundle->integer_dot_available;
    telemetry_.integer_dot_reason = impl.bundle->integer_dot_reason;
    impl.integer_dot_enabled = config.enable_integer_dot && impl.bundle->integer_dot_available;
    impl.unqualified_integer_dot_formats = config.enable_unqualified_integer_dot_formats;

    std::size_t widest_integer_dot_input = 0;
    if (impl.integer_dot_enabled) {
        for (const auto& plan_span : plan.spans()) {
            if (plan_span.backend != SpanBackend::gpu) continue;
            for (std::size_t node_index = plan_span.first_node;
                 node_index < plan_span.first_node + plan_span.node_count; ++node_index) {
                const auto& node = graph.nodes()[node_index];
                if (node.op != Op::matmul) continue;
                const auto weight = quantized.find(node.src[1]);
                if (weight == quantized.end()) continue;
                const auto type = weight->second.tensor->type;
                if (type != 12 && !(impl.unqualified_integer_dot_formats
                    && (type == 2 || type == 8 || type == 13 || type == 14))) continue;
                const auto columns = graph.tensor(node.src[1]).shape[1];
                const auto alignment = (type == 2 || type == 8) ? 32U : 256U;
                if ((columns % alignment) == 0) {
                    widest_integer_dot_input = std::max(widest_integer_dot_input, columns);
                }
            }
        }
        if (widest_integer_dot_input == 0) impl.integer_dot_enabled = false;
    }

    // --- state buffer layout -------------------------------------------------
    impl.arena_floats = (plan.allocation().arena_bytes + sizeof(float) - 1) / sizeof(float);
    std::size_t cursor = align_up(impl.arena_floats, 16);
    const auto assign_external = [&](TensorId id) {
        if (impl.external_offsets.contains(id)) return;
        impl.external_offsets.emplace(id, cursor);
        cursor = align_up(cursor + graph.tensor(id).elements(), 16);
    };
    for (const auto& [id, values] : immutable_externals) {
        static_cast<void>(values);
        assign_external(id);
    }
    for (const auto id : dynamic_externals) assign_external(id);
    if (impl.integer_dot_enabled) {
        impl.q8_1_scratch_words = widest_integer_dot_input / 4 + widest_integer_dot_input / 32 * 2;
        impl.q8_1_scratch_offset = align_up(cursor, 16);
        cursor = impl.q8_1_scratch_offset + impl.q8_1_scratch_words;
    }
    // +256-float tail pad: elementwise kernels dispatch whole 256-thread
    // groups and their early-return guards may not stop speculative loads
    // on the trailing partial group; the pad keeps those reads in-bounds.
    impl.state_floats = cursor + 256;

    impl.state_buffer = create_uma_buffer(
        impl.bundle->device.value,
        impl.state_floats * sizeof(float),
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    impl.state_mapped = static_cast<float*>(map_whole(impl.state_buffer.value));
    std::memset(impl.state_mapped, 0, impl.state_floats * sizeof(float));
    for (const auto& [id, values] : immutable_externals) {
        std::memcpy(impl.state_mapped + impl.external_offsets.at(id), values.data(), values.size() * sizeof(float));
    }
    // --- session constants ---------------------------------------------------
    impl.session_buffer = create_uma_buffer(
        impl.bundle->device.value, 16,
        D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    impl.session_mapped = static_cast<std::uint32_t*>(map_whole(impl.session_buffer.value));
    std::memset(impl.session_mapped, 0, 16);

    // --- device-resident packed KV -------------------------------------------
    std::size_t kv_bytes = 0;
    std::size_t kv_cpu_floats = 0;
    for (const auto& kv : kv_inputs) {
        const auto width = graph.tensor(kv.key).shape[1];
        const auto row_bytes = kv_cache_row_bytes(impl.kv_mode, width);
        impl.kv_layer_widths.emplace(kv.layer, width);
        impl.kv_layer_row_bytes.emplace(kv.layer, row_bytes);
        const auto key_base = kv_bytes;
        const auto value_base = key_base + impl.kv_capacity * row_bytes;
        impl.kv_layer_offsets.emplace(kv.layer, std::make_pair(key_base, value_base));
        impl.kv_tensor_offsets.emplace(kv.key, key_base);
        impl.kv_tensor_offsets.emplace(kv.value, value_base);
        kv_bytes = value_base + impl.kv_capacity * row_bytes;
        if (impl.cpu_kv_mirror) {
            const auto cpu_key_base = kv_cpu_floats;
            const auto cpu_value_base = cpu_key_base + impl.kv_capacity * width;
            impl.kv_cpu_layer_offsets.emplace(kv.layer, std::make_pair(cpu_key_base, cpu_value_base));
            kv_cpu_floats = cpu_value_base + impl.kv_capacity * width;
        }
    }
    impl.kv_bytes = kv_bytes;
    impl.kv_buffer = create_uma_buffer(
        impl.bundle->device.value,
        std::max<std::size_t>(kv_bytes, 16),
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    impl.kv_mapped = static_cast<std::byte*>(map_whole(impl.kv_buffer.value));
    std::memset(impl.kv_mapped, 0, std::max<std::size_t>(kv_bytes, 16));
    if (impl.cpu_kv_mirror) impl.kv_cpu_mapped.resize(kv_cpu_floats);
    telemetry_.kv_cache_mode = impl.kv_mode;
    telemetry_.kv_resident_bytes = kv_bytes;
    telemetry_.kv_gpu_resident = true;

    // --- resident weights ------------------------------------------------------
    // Stage ONLY the weights GPU spans actually consume: on a stable-split
    // plan the CPU tail keeps reading its quantized views from the mapped
    // model file, and staging the whole directory would burst the budget the
    // fit planner reserved for the resident prefix.
    std::set<TensorId> gpu_span_weights;
    for (const auto& plan_span : plan.spans()) {
        if (plan_span.backend != SpanBackend::gpu) continue;
        for (std::size_t node_index = plan_span.first_node;
             node_index < plan_span.first_node + plan_span.node_count; ++node_index) {
            const auto& node = graph.nodes()[node_index];
            if (node.op == Op::matmul && quantized.contains(node.src[1])) {
                gpu_span_weights.insert(node.src[1]);
            }
        }
    }
    for (const auto& [id, view] : quantized) {
        if (!gpu_span_weights.contains(id)) continue;
        const auto staged_bytes = staged_weight_bytes_for(*view.tensor);
        // +16 pad: the f32 matvec's Load4 tail may read up to 12 bytes past
        // the last row; Xe requires every issued load be inside the resource.
        auto buffer = create_gpu_weight_buffer(
            impl.bundle->device.value, staged_bytes + 16,
            D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        auto* mapped = static_cast<std::byte*>(map_whole(buffer.value));
        stage_weight_tensor(view, mapped);
        buffer->Unmap(0, nullptr);
        impl.resident_weight_bytes += staged_bytes;
        impl.weights.emplace(id, Impl::ResidentWeight{std::move(buffer), weight_kind_for(view.tensor->type)});
    }
    telemetry_.resident_weight_bytes = impl.resident_weight_bytes;

    // --- record one command list per GPU span ----------------------------------
    const auto element_offset_of = [&](TensorId id) -> std::size_t {
        const auto& tensor = graph.tensor(id);
        if (tensor.external) {
            const auto found = impl.external_offsets.find(id);
            if (found != impl.external_offsets.end()) return found->second;
            throw GpuUnavailable("GPU span sources unbound external tensor: " + tensor.name);
        }
        const auto found = plan.allocation().tensors.find(id);
        if (found == plan.allocation().tensors.end()) {
            throw GpuUnavailable("GPU span sources unplanned tensor: " + tensor.name);
        }
        return found->second.offset / sizeof(float);
    };

    // Which tensors must be copied back after each span: graph outputs plus
    // anything a later CPU span consumes.
    const auto& spans = plan.spans();
    std::set<TensorId> later_cpu_inputs;
    std::vector<std::set<TensorId>> cpu_inputs_after(spans.size());
    for (std::size_t index = spans.size(); index-- > 0;) {
        cpu_inputs_after[index] = later_cpu_inputs;
        if (spans[index].backend == SpanBackend::cpu) {
            for (std::size_t node_index = spans[index].first_node;
                 node_index < spans[index].first_node + spans[index].node_count; ++node_index) {
                for (const auto source : graph.nodes()[node_index].src) later_cpu_inputs.insert(source);
            }
        }
    }
    const std::set<TensorId> graph_outputs(graph.outputs().begin(), graph.outputs().end());

    // Timestamp instrumentation (Step 07 tuning): two query slots per recorded
    // dispatch, resolved at the end of each span into a persistently-mapped
    // readback buffer, so every replay attributes wall-clock GPU time to node
    // classes at effectively zero cost. Any failure simply disables timing.
    const auto node_class_of = [](Op op) -> std::uint8_t {
        switch (op) {
            case Op::matmul: return 0;
            case Op::causal_attention: return 1;
            case Op::rms_norm:
            case Op::rms_norm_heads:
            case Op::rms_norm_heads_unweighted: return 2;
            case Op::rope: return 3;
            case Op::add:
            case Op::scale:
            case Op::silu_mul:
            case Op::gelu_mul: return 4;
            default: return 5;
        }
    };
    if (impl.timestamp_capacity == 0) {
        std::uint32_t dispatch_upper_bound = 0;
        for (const auto& candidate : spans) {
            if (candidate.backend == SpanBackend::gpu) {
                dispatch_upper_bound += static_cast<std::uint32_t>(candidate.node_count);
            }
        }
        std::uint64_t frequency = 0;
        if (dispatch_upper_bound != 0
            && SUCCEEDED(impl.bundle->queue->GetTimestampFrequency(&frequency))
            && frequency != 0) {
            const std::uint32_t capacity = dispatch_upper_bound * 2U;
            D3D12_QUERY_HEAP_DESC heap_desc{};
            heap_desc.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
            heap_desc.Count = capacity;
            D3D12_HEAP_PROPERTIES readback_heap{};
            readback_heap.Type = D3D12_HEAP_TYPE_READBACK;
            D3D12_RESOURCE_DESC buffer_desc{};
            buffer_desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
            buffer_desc.Width = static_cast<std::uint64_t>(capacity) * sizeof(std::uint64_t);
            buffer_desc.Height = 1;
            buffer_desc.DepthOrArraySize = 1;
            buffer_desc.MipLevels = 1;
            buffer_desc.SampleDesc.Count = 1;
            buffer_desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
            void* mapped = nullptr;
            if (SUCCEEDED(impl.bundle->device->CreateQueryHeap(
                    &heap_desc, IID_PPV_ARGS(impl.timestamp_heap.put())))
                && SUCCEEDED(impl.bundle->device->CreateCommittedResource(
                    &readback_heap, D3D12_HEAP_FLAG_NONE, &buffer_desc,
                    D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                    IID_PPV_ARGS(impl.timestamp_readback.put())))
                && SUCCEEDED(impl.timestamp_readback->Map(0, nullptr, &mapped))) {
                impl.timestamp_mapped = static_cast<const std::uint64_t*>(mapped);
                impl.timestamp_frequency = frequency;
                impl.timestamp_capacity = capacity;
            } else {
                impl.timestamp_readback.reset();
                impl.timestamp_heap.reset();
                impl.timestamp_mapped = nullptr;
            }
        }
    }
    std::uint32_t timestamp_cursor = 0;

    // A GPU span may only consume externals, mirror KV, resident weights,
    // or tensors produced by EARLIER GPU spans (still live in the state
    // buffer). Values produced by a preceding CPU span exist only in the
    // replay's CPU arena, so such plans are rejected here and the caller
    // falls back to a CPU-only plan (the scheduler's GPU-prefix discipline
    // never produces them).
    {
        std::set<TensorId> produced_by_cpu;
        for (const auto& span : spans) {
            for (std::size_t node_index = span.first_node;
                 node_index < span.first_node + span.node_count; ++node_index) {
                const auto& node = graph.nodes()[node_index];
                if (span.backend == SpanBackend::gpu) {
                    for (const auto source : node.src) {
                        if (produced_by_cpu.contains(source)) {
                            throw GpuUnavailable(
                                "GPU span consumes a CPU-produced tensor: " + graph.tensor(source).name);
                        }
                    }
                } else {
                    produced_by_cpu.insert(node.dst);
                }
            }
        }
    }

    std::set<TensorId> appended_kv_sources;
    // Hybrid FFN split is incompatible with the DXIL integer-dot pipelines
    // (no row-limit constant) and the cooperative kernel; those combinations
    // simply record the historical single-segment span.
    const double hybrid_fraction = (impl.integer_dot_enabled || coop_matvec_requested())
        ? 0.0
        : hybrid_ffn_cpu_fraction();

    for (std::size_t span_index = 0; span_index < spans.size(); ++span_index) {
        const auto& span = spans[span_index];
        if (span.backend != SpanBackend::gpu) continue;
        Impl::RecordedSpan recorded;
        recorded.query_base = timestamp_cursor;
        ID3D12GraphicsCommandList* list = nullptr;
        const auto begin_segment = [&]() -> ID3D12GraphicsCommandList* {
            Impl::Segment segment;
            segment.query_first = timestamp_cursor;
            check_hresult("CreateCommandAllocator", impl.bundle->device->CreateCommandAllocator(
                D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(segment.allocator.put())));
            check_hresult("CreateCommandList", impl.bundle->device->CreateCommandList(
                0, D3D12_COMMAND_LIST_TYPE_COMPUTE, segment.allocator.value, nullptr,
                IID_PPV_ARGS(segment.list.put())));
            auto* fresh = segment.list.value;
            fresh->SetComputeRootSignature(impl.bundle->root_signature.value);
            fresh->SetComputeRootShaderResourceView(2, impl.session_buffer->GetGPUVirtualAddress());
            fresh->SetComputeRootShaderResourceView(3, impl.session_buffer->GetGPUVirtualAddress());
            fresh->SetComputeRootUnorderedAccessView(4, impl.state_buffer->GetGPUVirtualAddress());
            fresh->SetComputeRootUnorderedAccessView(5, impl.kv_buffer->GetGPUVirtualAddress());
            recorded.segments.push_back(std::move(segment));
            return fresh;
        };
        const auto end_segment = [&]() {
            auto& segment = recorded.segments.back();
            segment.query_count = timestamp_cursor - segment.query_first;
            if (segment.query_count != 0) {
                segment.list.value->ResolveQueryData(
                    impl.timestamp_heap.value,
                    D3D12_QUERY_TYPE_TIMESTAMP,
                    segment.query_first,
                    segment.query_count,
                    impl.timestamp_readback.value,
                    static_cast<std::uint64_t>(segment.query_first) * sizeof(std::uint64_t));
            }
            check_hresult("Close", segment.list.value->Close());
        };
        list = begin_segment();

        // Step 09b hybrid FFN state for this span. The fences are created on
        // first use and live for the execution's lifetime.
        std::size_t hybrid_gpu_rows = 0;
        int hybrid_limit_remaining = 0;
        if (hybrid_fraction > 0.0 && !impl.hybrid_gpu_fence) {
            check_hresult("CreateFence(hybrid gpu)", impl.bundle->device->CreateFence(
                0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(impl.hybrid_gpu_fence.put())));
            check_hresult("CreateFence(hybrid cpu)", impl.bundle->device->CreateFence(
                0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(impl.hybrid_cpu_fence.put())));
        }

        D3D12_RESOURCE_BARRIER uav_barrier{};
        uav_barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        uav_barrier.UAV.pResource = impl.state_buffer.value;
        D3D12_RESOURCE_BARRIER kv_uav_barrier{};
        kv_uav_barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        kv_uav_barrier.UAV.pResource = impl.kv_buffer.value;

        struct Constants {
            std::uint32_t values[16];
        };

        std::optional<std::pair<TensorId, std::uint32_t>> quantized_q8_1_source;
        const auto ensure_q8_1 = [&](TensorId source_id, std::uint32_t columns, std::uint32_t input_offset) {
            const auto source = std::make_pair(source_id, columns);
            if (quantized_q8_1_source == source) return;
            Constants quantize{};
            quantize.values[1] = columns;
            quantize.values[2] = input_offset;
            quantize.values[4] = static_cast<std::uint32_t>(impl.q8_1_scratch_offset);
            list->SetPipelineState(impl.bundle->kernels.quantize_q8_1.value);
            list->SetComputeRoot32BitConstants(0, 16, quantize.values, 0);
            list->SetComputeRootUnorderedAccessView(4, impl.state_buffer->GetGPUVirtualAddress());
            list->Dispatch((columns + 31U) / 32U, 1, 1);
            list->ResourceBarrier(1, &uav_barrier);
            quantized_q8_1_source = source;
            ++recorded.q8_1_quantizations;
        };

        for (std::size_t node_index = span.first_node;
             node_index < span.first_node + span.node_count; ++node_index) {
            const auto& node = graph.nodes()[node_index];
            const auto& output = graph.tensor(node.dst);
            const auto total = static_cast<std::uint32_t>(output.elements());

            // Step 09b: detect the FFN gate/up/silu triple and cut the span so
            // the CPU owns rows [G, R) of gate/up concurrently with the GPU.
            // Eligibility mirrors the fusion peephole (private intermediates)
            // plus hybrid requirements (q4_K resident weights on both, CPU
            // views available, same row count).
            if (hybrid_fraction > 0.0
                && hybrid_limit_remaining == 0
                && node.op == Op::matmul
                && node.src.size() == 2
                && node_index + 2 < span.first_node + span.node_count) {
                const auto& up_node = graph.nodes()[node_index + 1];
                const auto& silu_node = graph.nodes()[node_index + 2];
                bool eligible = up_node.op == Op::matmul
                    && silu_node.op == Op::silu_mul
                    && up_node.src.size() == 2
                    && silu_node.src.size() == 2
                    && up_node.src[0] == node.src[0]
                    && silu_node.src[0] == node.dst
                    && silu_node.src[1] == up_node.dst;
                const auto gate_weight = eligible ? impl.weights.find(node.src[1]) : impl.weights.end();
                const auto up_weight = eligible ? impl.weights.find(up_node.src[1]) : impl.weights.end();
                const auto gate_cpu_view = eligible ? quantized.find(node.src[1]) : quantized.end();
                const auto up_cpu_view = eligible ? quantized.find(up_node.src[1]) : quantized.end();
                eligible = eligible
                    && gate_weight != impl.weights.end() && gate_weight->second.kind == WeightKind::q4k
                    && up_weight != impl.weights.end() && up_weight->second.kind == WeightKind::q4k
                    && gate_cpu_view != quantized.end() && up_cpu_view != quantized.end();
                std::size_t ffn_rows = 0;
                if (eligible) {
                    ffn_rows = graph.tensor(node.src[1]).shape[0];
                    eligible = graph.tensor(up_node.src[1]).shape[0] == ffn_rows
                        && graph.tensor(node.dst).elements() == ffn_rows
                        && graph.tensor(up_node.dst).elements() == ffn_rows
                        && graph.tensor(silu_node.dst).elements() == ffn_rows
                        && ffn_rows >= 64;
                }
                if (eligible) {
                    std::size_t gate_uses = 0;
                    std::size_t up_uses = 0;
                    for (const auto& consumer : graph.nodes()) {
                        for (const auto source : consumer.src) {
                            if (source == node.dst) ++gate_uses;
                            if (source == up_node.dst) ++up_uses;
                        }
                    }
                    eligible = gate_uses == 1 && up_uses == 1
                        && !graph_outputs.contains(node.dst)
                        && !graph_outputs.contains(up_node.dst)
                        && !cpu_inputs_after[span_index].contains(node.dst)
                        && !cpu_inputs_after[span_index].contains(up_node.dst);
                }
                if (eligible) {
                    std::size_t cpu_rows = static_cast<std::size_t>(
                        static_cast<double>(ffn_rows) * hybrid_fraction);
                    cpu_rows = (cpu_rows / 32) * 32;
                    if (cpu_rows >= 32 && cpu_rows + 32 <= ffn_rows) {
                        const auto gpu_rows = ffn_rows - cpu_rows;
                        // Close the pre-FFN segment; its completion means the
                        // shared input vector x is resident for the CPU.
                        end_segment();
                        recorded.segments.back().signal_input_ready =
                            static_cast<int>(recorded.hybrid_tasks.size());
                        list = begin_segment();
                        Impl::HybridFfnTask task;
                        task.gate_view = gate_cpu_view->second;
                        task.up_view = up_cpu_view->second;
                        task.input_offset = element_offset_of(node.src[0]);
                        task.output_offset = element_offset_of(silu_node.dst);
                        task.first_row = gpu_rows;
                        task.rows = ffn_rows;
                        recorded.hybrid_tasks.push_back(task);
                        hybrid_gpu_rows = gpu_rows;
                        hybrid_limit_remaining = 2;
                    }
                }
            }
            Constants constants{};
            ID3D12PipelineState* pipeline{};
            std::uint32_t groups = 1;
            bool integer_dot_matvec = false;
            bool coop_matvec = false;
            bool fused_kv_append = false;
            bool fused_elementwise = false;
            TensorId fused_elementwise_dst{};
            std::uint8_t dispatch_class = node_class_of(node.op);
            // The weights root SRV must always be bound to something valid;
            // non-matvec kernels never read it.
            D3D12_GPU_VIRTUAL_ADDRESS weights_address = impl.session_buffer->GetGPUVirtualAddress();

            switch (node.op) {
                case Op::matmul: {
                    const auto& weight = graph.tensor(node.src[1]);
                    const auto rows = static_cast<std::uint32_t>(weight.shape[0]);
                    const auto columns = static_cast<std::uint32_t>(weight.shape[1]);
                    constants.values[0] = rows;
                    constants.values[1] = columns;
                    constants.values[2] = static_cast<std::uint32_t>(element_offset_of(node.src[0]));
                    constants.values[3] = static_cast<std::uint32_t>(element_offset_of(node.dst));
                    constants.values[4] = 0;
                    const auto resident = impl.weights.find(node.src[1]);
                    if (resident != impl.weights.end()) {
                        weights_address = resident->second.buffer->GetGPUVirtualAddress();
                        switch (resident->second.kind) {
                            case WeightKind::f32: pipeline = impl.bundle->kernels.matvec_f32.value; break;
                            case WeightKind::iq4_nl: pipeline = impl.bundle->kernels.matvec_iq4_nl.value; break;
                            case WeightKind::iq4_xs: pipeline = impl.bundle->kernels.matvec_iq4_xs.value; break;
                            case WeightKind::q4k: {
                                // The neutral Iris Xe gate qualifies Llama's
                                // 8192x3072 gate/up projection. Other shapes
                                // stay float unless the harness qualification
                                // override is explicit.
                                if (impl.integer_dot_enabled
                                    && (impl.unqualified_integer_dot_formats
                                        || (rows == 8192U && columns == 3072U))) {
                                    ensure_q8_1(node.src[0], columns, constants.values[2]);
                                    constants.values[4] = static_cast<std::uint32_t>(impl.q8_1_scratch_offset);
                                    pipeline = impl.bundle->kernels.matvec_q4k_q8_1.value;
                                    groups = (rows + 127U) / 128U;
                                    ++recorded.integer_dot_matvecs;
                                    telemetry_.integer_dot_active = true;
                                    integer_dot_matvec = true;
                                } else if (coop_matvec_requested() && (columns % 256U) == 0U) {
                                    pipeline = impl.bundle->kernels.matvec_q4k_coop.value;
                                    coop_matvec = true;
                                } else {
                                    pipeline = impl.bundle->kernels.matvec_q4k.value;
                                }
                                break;
                            }
                            case WeightKind::q5k:
                            case WeightKind::q6k:
                            case WeightKind::q8_0:
                            case WeightKind::q4_0: {
                                if (impl.integer_dot_enabled && impl.unqualified_integer_dot_formats) {
                                    ensure_q8_1(node.src[0], columns, constants.values[2]);
                                    constants.values[4] = static_cast<std::uint32_t>(impl.q8_1_scratch_offset);
                                    switch (resident->second.kind) {
                                        case WeightKind::q5k: pipeline = impl.bundle->kernels.matvec_q5k_q8_1.value; break;
                                        case WeightKind::q6k: pipeline = impl.bundle->kernels.matvec_q6k_q8_1.value; break;
                                        case WeightKind::q8_0: pipeline = impl.bundle->kernels.matvec_q8_0_q8_1.value; break;
                                        case WeightKind::q4_0: pipeline = impl.bundle->kernels.matvec_q4_0_q8_1.value; break;
                                        default: break;
                                    }
                                    groups = (rows + 63U) / 64U;
                                    ++recorded.integer_dot_matvecs;
                                    telemetry_.integer_dot_active = true;
                                    integer_dot_matvec = true;
                                } else {
                                    switch (resident->second.kind) {
                                        case WeightKind::q5k: pipeline = impl.bundle->kernels.matvec_q5k.value; break;
                                        case WeightKind::q6k: pipeline = impl.bundle->kernels.matvec_q6k.value; break;
                                        case WeightKind::q8_0: pipeline = impl.bundle->kernels.matvec_q8_0.value; break;
                                        case WeightKind::q4_0: pipeline = impl.bundle->kernels.matvec_q4_0.value; break;
                                        default: break;
                                    }
                                }
                                break;
                            }
                        }
                    } else {
                        throw GpuUnavailable("GPU matmul weight is not resident: " + weight.name);
                    }
                    const auto batches = graph.tensor(node.src[0]).elements() / columns;
                    if (batches != 1) {
                        throw GpuUnavailable("Phase 3 GPU matvec requires decode shapes (batch 1): " + output.name);
                    }
                    if (!integer_dot_matvec) {
                        // Serial FXC kernels: one thread per row, 128 rows/group.
                        // Cooperative Q4_K kernel: 32 rows x 8 slices per group.
                        groups = coop_matvec ? (rows + 31U) / 32U : (rows + 127U) / 128U;
                    }
                    // Shape-bucketed matmul timing: large shapes saturate the
                    // machine, small ones are latency-bound; the split shows
                    // which bucket eats the token budget.
                    dispatch_class = rows >= 8192U ? 0 : (rows >= 2048U ? 6 : 7);
                    if (hybrid_limit_remaining > 0) {
                        // Hybrid FFN row split: this dispatch computes only the
                        // GPU share [0, G); the CPU task produces the rest.
                        constants.values[10] = static_cast<std::uint32_t>(hybrid_gpu_rows);
                        groups = (static_cast<std::uint32_t>(hybrid_gpu_rows) + 127U) / 128U;
                    }
                    // Step 07 fusion: fold an immediately-following elementwise
                    // consumer (residual add, SiLU/GELU gate multiply) into this
                    // matvec's epilogue and skip its dispatch + UAV barrier.
                    // Constraints keep it bit-exact and private: the FXC q4_K /
                    // q6_K kernels carry the epilogue (integer-dot DXIL kernels
                    // do not), the matvec's output must feed exactly that one
                    // consumer and nothing else (not a graph output, not a CPU
                    // input), and every operand is full row width so the fused
                    // expression matches the standalone kernel element-for-
                    // element. Recording happens once per session shape, so the
                    // O(nodes) use scan is a one-time cost.
                    if (!integer_dot_matvec
                        && (resident->second.kind == WeightKind::q4k || resident->second.kind == WeightKind::q6k)
                        && node_index + 1 < span.first_node + span.node_count) {
                        const auto& next = graph.nodes()[node_index + 1];
                        const bool fusable_op = next.op == Op::add
                            || next.op == Op::silu_mul
                            || next.op == Op::gelu_mul;
                        if (fusable_op && next.src.size() == 2) {
                            const bool dst_is_first = next.src[0] == node.dst;
                            const bool dst_is_second = next.src[1] == node.dst;
                            if (dst_is_first != dst_is_second) {
                                const auto operand = dst_is_first ? next.src[1] : next.src[0];
                                std::size_t uses = 0;
                                for (const auto& consumer : graph.nodes()) {
                                    for (const auto source : consumer.src) {
                                        if (source == node.dst) ++uses;
                                    }
                                }
                                const bool intermediate_private = uses == 1
                                    && !graph_outputs.contains(node.dst)
                                    && !cpu_inputs_after[span_index].contains(node.dst);
                                const bool shapes_align =
                                    graph.tensor(next.dst).elements() == rows
                                    && graph.tensor(next.src[0]).elements() == rows
                                    && graph.tensor(next.src[1]).elements() == rows;
                                if (intermediate_private && shapes_align) {
                                    constants.values[3] = static_cast<std::uint32_t>(element_offset_of(next.dst));
                                    constants.values[9] = static_cast<std::uint32_t>(element_offset_of(operand));
                                    if (next.op == Op::add) {
                                        constants.values[8] = 1U;
                                    } else if (next.op == Op::silu_mul) {
                                        constants.values[8] = dst_is_second ? 2U : 3U;
                                    } else {
                                        constants.values[8] = dst_is_second ? 4U : 5U;
                                    }
                                    fused_elementwise = true;
                                    fused_elementwise_dst = next.dst;
                                    ++node_index;
                                }
                            }
                        }
                    }
                    break;
                }
                case Op::rms_norm: {
                    const auto width = static_cast<std::uint32_t>(output.shape.back());
                    const auto rows = total / width;
                    constants.values[0] = width;
                    constants.values[1] = static_cast<std::uint32_t>(element_offset_of(node.src[0]));
                    constants.values[2] = static_cast<std::uint32_t>(element_offset_of(node.src[1]));
                    constants.values[3] = static_cast<std::uint32_t>(element_offset_of(node.dst));
                    std::memcpy(&constants.values[4], &node.scalar, 4);
                    constants.values[5] = rows;
                    pipeline = impl.bundle->kernels.rms_norm.value;
                    groups = rows;
                    break;
                }
                case Op::rms_norm_heads:
                case Op::rms_norm_heads_unweighted: {
                    const auto heads = static_cast<std::uint32_t>(node.params[0]);
                    const auto head_dim = static_cast<std::uint32_t>(node.params[1]);
                    const auto batches = total / (heads * head_dim);
                    constants.values[0] = head_dim;
                    constants.values[1] = heads;
                    constants.values[2] = static_cast<std::uint32_t>(element_offset_of(node.src[0]));
                    constants.values[3] = node.op == Op::rms_norm_heads
                        ? static_cast<std::uint32_t>(element_offset_of(node.src[1]))
                        : 0xFFFFFFFFU;
                    constants.values[4] = static_cast<std::uint32_t>(element_offset_of(node.dst));
                    std::memcpy(&constants.values[5], &node.scalar, 4);
                    pipeline = impl.bundle->kernels.rms_norm_heads.value;
                    groups = static_cast<std::uint32_t>(batches * heads);
                    break;
                }
                case Op::rope: {
                    const auto heads = static_cast<std::uint32_t>(node.params[0]);
                    const auto head_dim = static_cast<std::uint32_t>(node.params[1]);
                    constants.values[0] = heads;
                    constants.values[1] = head_dim;
                    constants.values[2] = static_cast<std::uint32_t>(element_offset_of(node.src[0]));
                    constants.values[3] = static_cast<std::uint32_t>(element_offset_of(node.dst));
                    std::memcpy(&constants.values[4], &node.scalar, 4);
                    std::memcpy(&constants.values[5], &node.secondary_scalar, 4);
                    constants.values[6] = node.params.size() == 4 && node.params[3] != 0 ? 1U : 0U;
                    constants.values[7] = node.src.size() == 2
                        ? static_cast<std::uint32_t>(element_offset_of(node.src[1]))
                        : 0xFFFFFFFFU;
                    pipeline = impl.bundle->kernels.rope.value;
                    const auto tokens = total / (heads * head_dim);
                    if (tokens != 1) throw GpuUnavailable("Phase 3 GPU rope requires decode shapes");
                    groups = (heads * (head_dim / 2U) + 255U) / 256U;
                    break;
                }
                case Op::causal_attention: {
                    const auto heads = static_cast<std::uint32_t>(node.params[0]);
                    const auto kv_heads = static_cast<std::uint32_t>(node.params[1]);
                    const auto head_dim = static_cast<std::uint32_t>(node.params[2]);
                    const auto window = static_cast<std::uint32_t>(node.params.size() >= 4 ? node.params[3] : 0);
                    if (node.src.size() != 5) {
                        throw GpuUnavailable("Phase 3 GPU attention requires a past-KV decode graph");
                    }
                    if (head_dim > 512) throw GpuUnavailable("GPU attention head_dim exceeds 512");
                    if (output.shape[0] != 1) throw GpuUnavailable("Phase 3 GPU attention requires decode shapes");
                    const auto key_offset = impl.kv_tensor_offsets.find(node.src[3]);
                    const auto value_offset = impl.kv_tensor_offsets.find(node.src[4]);
                    if (key_offset == impl.kv_tensor_offsets.end() || value_offset == impl.kv_tensor_offsets.end()) {
                        throw GpuUnavailable("GPU attention past KV is not device-resident");
                    }
                    const auto kv_width = static_cast<std::uint32_t>(graph.tensor(node.src[3]).shape[1]);
                    constants.values[0] = heads;
                    constants.values[1] = kv_heads;
                    constants.values[2] = head_dim;
                    constants.values[3] = window;
                    std::memcpy(&constants.values[4], &node.scalar, 4);
                    constants.values[5] = static_cast<std::uint32_t>(element_offset_of(node.src[0]));
                    constants.values[6] = static_cast<std::uint32_t>(element_offset_of(node.src[1]));
                    constants.values[7] = static_cast<std::uint32_t>(element_offset_of(node.src[2]));
                    constants.values[8] = static_cast<std::uint32_t>(element_offset_of(node.dst));
                    constants.values[9] = static_cast<std::uint32_t>(key_offset->second);
                    constants.values[10] = static_cast<std::uint32_t>(value_offset->second);
                    constants.values[11] = static_cast<std::uint32_t>(kv_cache_row_bytes(impl.kv_mode, kv_width));
                    constants.values[12] = static_cast<std::uint32_t>(impl.kv_mode);
                    constants.values[13] = kv_width;
                    fused_kv_append = appended_kv_sources.insert(node.src[1]).second;
                    constants.values[14] = fused_kv_append ? 1U : 0U;
                    if (fused_kv_append && impl.kv_mode != KvCacheMode::q8_0) {
                        list->SetPipelineState(impl.bundle->kernels.kv_append.value);
                        list->SetComputeRoot32BitConstants(0, 16, constants.values, 0);
                        list->SetComputeRootUnorderedAccessView(4, impl.state_buffer->GetGPUVirtualAddress());
                        list->SetComputeRootUnorderedAccessView(5, impl.kv_buffer->GetGPUVirtualAddress());
                        if (impl.kv_mode != KvCacheMode::f32 && impl.kv_mode != KvCacheMode::f16
                            && (kv_width % 32U) != 0U) {
                            throw GpuUnavailable("GPU-resident KV width must be a multiple of 32");
                        }
                        list->Dispatch((kv_width + 31U) / 32U, 2, 1);
                        list->ResourceBarrier(1, &kv_uav_barrier);
                    }
                    if (fused_kv_append) ++recorded.kv_device_appends;
                    pipeline = impl.bundle->kernels.attention.value;
                    groups = heads;
                    break;
                }
                case Op::silu_mul:
                case Op::gelu_mul: {
                    constants.values[0] = total;
                    constants.values[1] = static_cast<std::uint32_t>(element_offset_of(node.src[0]));
                    constants.values[2] = static_cast<std::uint32_t>(element_offset_of(node.src[1]));
                    constants.values[3] = static_cast<std::uint32_t>(element_offset_of(node.dst));
                    pipeline = node.op == Op::silu_mul
                        ? impl.bundle->kernels.silu_mul.value
                        : impl.bundle->kernels.gelu_mul.value;
                    groups = (total + 255U) / 256U;
                    break;
                }
                case Op::add: {
                    constants.values[0] = total;
                    constants.values[1] = static_cast<std::uint32_t>(element_offset_of(node.src[0]));
                    constants.values[2] = static_cast<std::uint32_t>(element_offset_of(node.src[1]));
                    constants.values[3] = static_cast<std::uint32_t>(element_offset_of(node.dst));
                    constants.values[4] = static_cast<std::uint32_t>(graph.tensor(node.src[1]).elements());
                    pipeline = impl.bundle->kernels.add.value;
                    groups = (total + 255U) / 256U;
                    break;
                }
                case Op::scale: {
                    constants.values[0] = total;
                    constants.values[1] = static_cast<std::uint32_t>(element_offset_of(node.src[0]));
                    constants.values[2] = node.src.size() == 2
                        ? static_cast<std::uint32_t>(element_offset_of(node.src[1]))
                        : 0U;
                    constants.values[3] = static_cast<std::uint32_t>(element_offset_of(node.dst));
                    std::memcpy(&constants.values[4], &node.scalar, 4);
                    constants.values[5] = node.src.size() == 2 ? 1U : 0U;
                    pipeline = impl.bundle->kernels.scale.value;
                    groups = (total + 255U) / 256U;
                    break;
                }
            }

            const bool timestamped = impl.timestamp_capacity != 0
                && timestamp_cursor + 2 <= impl.timestamp_capacity;
            if (timestamped) {
                list->EndQuery(impl.timestamp_heap.value, D3D12_QUERY_TYPE_TIMESTAMP, timestamp_cursor);
            }
            list->SetPipelineState(pipeline);
            list->SetComputeRoot32BitConstants(0, 16, constants.values, 0);
            list->SetComputeRootShaderResourceView(1, weights_address);
            list->SetComputeRootShaderResourceView(2, impl.session_buffer->GetGPUVirtualAddress());
            list->SetComputeRootUnorderedAccessView(4, impl.state_buffer->GetGPUVirtualAddress());
            list->SetComputeRootUnorderedAccessView(5, impl.kv_buffer->GetGPUVirtualAddress());
            list->Dispatch(groups, 1, 1);
            list->ResourceBarrier(1, &uav_barrier);
            if (fused_kv_append) list->ResourceBarrier(1, &kv_uav_barrier);
            if (timestamped) {
                // A fused matvec attributes its folded elementwise tail to the
                // matmul class - the fused dispatch IS one matvec pass.
                list->EndQuery(impl.timestamp_heap.value, D3D12_QUERY_TYPE_TIMESTAMP, timestamp_cursor + 1);
                recorded.dispatch_classes.push_back(dispatch_class);
                timestamp_cursor += 2;
            }

            if (hybrid_limit_remaining > 0 && node.op == Op::matmul) {
                if (--hybrid_limit_remaining == 0) {
                    // Both split matvecs are recorded (gate, then up with the
                    // fused SiLU epilogue). The next segment consumes the
                    // stitched vector, so it queue-waits for the CPU share.
                    end_segment();
                    list = begin_segment();
                    recorded.segments.back().wait_cpu_share =
                        static_cast<int>(recorded.hybrid_tasks.size()) - 1;
                }
            }

            // A fused matvec wrote the folded consumer's tensor, so copy-back
            // eligibility follows that destination instead of the matvec's own.
            const auto effective_dst = fused_elementwise ? fused_elementwise_dst : node.dst;
            const bool needed_later = graph_outputs.contains(effective_dst)
                || cpu_inputs_after[span_index].contains(effective_dst);
            if (needed_later) {
                const auto offset = element_offset_of(effective_dst);
                recorded.copy_back.emplace_back(offset, graph.tensor(effective_dst).elements());
            }
        }
        end_segment();
        impl.recorded.emplace(span.first_node, std::move(recorded));
    }

    telemetry_.gpu_spans = plan.gpu_span_count();
    telemetry_.cpu_spans = plan.spans().size() - telemetry_.gpu_spans;
}

GpuPlanExecution::~GpuPlanExecution() = default;

void GpuPlanExecution::execute_span(
    const Plan& plan,
    const PlanSpan& span,
    const BindingViews& external,
    const ReplayDynamics& dynamics,
    std::span<float> arena
) {
    static_cast<void>(plan);
    auto& impl = *impl_;
    const auto recorded = impl.recorded.find(span.first_node);
    if (recorded == impl.recorded.end()) {
        throw std::logic_error("GPU span was not recorded at prepare time");
    }

    // Per-token dynamics: the recorded lists read these buffer contents.
    impl.session_mapped[0] = dynamics.past_tokens == ReplayDynamics::use_node_params
        ? 0U
        : static_cast<std::uint32_t>(dynamics.past_tokens);
    impl.session_mapped[1] = dynamics.position_offset == ReplayDynamics::use_node_params
        ? 0U
        : static_cast<std::uint32_t>(dynamics.position_offset);

    for (const auto id : impl.dynamic_externals) {
        const auto found = external.find(id);
        if (found == external.end()) {
            throw std::invalid_argument("GPU replay is missing a dynamic external binding");
        }
        std::memcpy(
            impl.state_mapped + impl.external_offsets.at(id),
            found->second.data(),
            found->second.size() * sizeof(float));
        telemetry_.dynamic_upload_bytes += found->second.size() * sizeof(float);
    }

    // Submit every segment up front. Queue-level Signals/Waits express the
    // hybrid FFN dependencies, so the CPU thread below can compute its row
    // shares while the GPU pipeline drains ahead of and behind each Wait.
    const auto& rec = recorded->second;
    const auto task_count = rec.hybrid_tasks.size();
    const std::uint64_t hybrid_base = impl.hybrid_value_next;
    impl.hybrid_value_next += task_count;
    for (const auto& segment : rec.segments) {
        if (segment.wait_cpu_share >= 0) {
            check_hresult("queue Wait(hybrid cpu)", impl.bundle->queue->Wait(
                impl.hybrid_cpu_fence.value,
                hybrid_base + static_cast<std::uint64_t>(segment.wait_cpu_share)));
        }
        ID3D12CommandList* lists[] = {segment.list.value};
        impl.bundle->queue->ExecuteCommandLists(1, lists);
        ++telemetry_.submissions;
        if (segment.signal_input_ready >= 0) {
            check_hresult("queue Signal(hybrid gpu)", impl.bundle->queue->Signal(
                impl.hybrid_gpu_fence.value,
                hybrid_base + static_cast<std::uint64_t>(segment.signal_input_ready)));
        }
    }
    ++telemetry_.span_replays;
    telemetry_.q8_1_quantizations += rec.q8_1_quantizations;
    telemetry_.integer_dot_matvecs += rec.integer_dot_matvecs;
    telemetry_.kv_device_appends += rec.kv_device_appends;

    // CPU row shares, in dependency order. Every task ALWAYS signals its fence
    // value - even on failure - so the queue can never deadlock; a captured
    // error surfaces after the GPU drains.
    std::exception_ptr hybrid_error;
    for (std::size_t task_index = 0; task_index < task_count; ++task_index) {
        const auto ready_value = hybrid_base + static_cast<std::uint64_t>(task_index);
        if (impl.hybrid_gpu_fence->GetCompletedValue() < ready_value) {
            check_hresult("SetEventOnCompletion(hybrid)", impl.hybrid_gpu_fence->SetEventOnCompletion(
                ready_value, impl.bundle->fence_event));
            WaitForSingleObject(impl.bundle->fence_event, INFINITE);
        }
        if (!hybrid_error) {
            try {
                const auto& task = rec.hybrid_tasks[task_index];
                const auto columns = static_cast<std::size_t>(task.gate_view.tensor->dimensions[0]);
                const std::span<const float> x{impl.state_mapped + task.input_offset, columns};
                const auto count = task.rows - task.first_row;
                impl.hybrid_gate_scratch.resize(count);
                impl.hybrid_up_scratch.resize(count);
                tensor_matvec_row_range(
                    task.gate_view, x, task.first_row, task.rows, impl.hybrid_gate_scratch);
                tensor_matvec_row_range(
                    task.up_view, x, task.first_row, task.rows, impl.hybrid_up_scratch);
                float* destination = impl.state_mapped + task.output_offset + task.first_row;
                for (std::size_t index = 0; index < count; ++index) {
                    const float gate = impl.hybrid_gate_scratch[index];
                    destination[index] = (gate / (1.0F + std::exp(-gate))) * impl.hybrid_up_scratch[index];
                }
            } catch (...) {
                hybrid_error = std::current_exception();
            }
        }
        check_hresult("Signal(hybrid cpu)", impl.hybrid_cpu_fence->Signal(ready_value));
    }

    impl.wait_for_gpu();
    if (hybrid_error) std::rethrow_exception(hybrid_error);

    if (impl.timestamp_mapped != nullptr
        && impl.timestamp_frequency != 0
        && !recorded->second.dispatch_classes.empty()) {
        const auto* ticks = impl.timestamp_mapped + recorded->second.query_base;
        const double ticks_to_ms = 1000.0 / static_cast<double>(impl.timestamp_frequency);
        for (std::size_t index = 0; index < recorded->second.dispatch_classes.size(); ++index) {
            const auto begin = ticks[index * 2];
            const auto end = ticks[index * 2 + 1];
            if (end <= begin) continue;
            const auto node_class = std::min<std::size_t>(
                recorded->second.dispatch_classes[index], GpuTelemetry::node_class_count - 1);
            telemetry_.node_class_ms[node_class] += static_cast<double>(end - begin) * ticks_to_ms;
            ++telemetry_.node_class_dispatches[node_class];
        }
        telemetry_.node_timing_available = true;
    }

    for (const auto& [offset, count] : recorded->second.copy_back) {
        std::memcpy(arena.data() + offset, impl.state_mapped + offset, count * sizeof(float));
        telemetry_.readback_bytes += count * sizeof(float);
        ++telemetry_.readback_count;
    }
}

void GpuPlanExecution::seed_kv(
    std::size_t owning_layer,
    std::span<const float> keys,
    std::span<const float> values,
    std::size_t token_count
) {
    auto& impl = *impl_;
    const auto found = impl.kv_layer_offsets.find(owning_layer);
    if (found == impl.kv_layer_offsets.end()) throw std::invalid_argument("unknown GPU KV layer");
    const auto width = impl.kv_layer_widths.at(owning_layer);
    const auto row_bytes = impl.kv_layer_row_bytes.at(owning_layer);
    if (token_count > impl.kv_capacity || keys.size() != token_count * width || values.size() != token_count * width) {
        throw std::invalid_argument("GPU KV seed has the wrong shape");
    }
    for (std::size_t token = 0; token < token_count; ++token) {
        encode_kv_cache_row(
            impl.kv_mode,
            keys.subspan(token * width, width),
            {impl.kv_mapped + found->second.first + token * row_bytes, row_bytes});
        encode_kv_cache_row(
            impl.kv_mode,
            values.subspan(token * width, width),
            {impl.kv_mapped + found->second.second + token * row_bytes, row_bytes});
    }
    if (impl.cpu_kv_mirror) {
        auto key_mirror = kv_keys(owning_layer);
        auto value_mirror = kv_values(owning_layer);
        std::copy(keys.begin(), keys.end(), key_mirror.begin());
        std::copy(values.begin(), values.end(), value_mirror.begin());
    }
}

void GpuPlanExecution::write_kv_rows(
    std::size_t owning_layer,
    std::size_t first_token,
    std::span<const float> keys,
    std::span<const float> values
) {
    auto& impl = *impl_;
    const auto found = impl.kv_layer_offsets.find(owning_layer);
    if (found == impl.kv_layer_offsets.end()) throw std::invalid_argument("unknown GPU KV layer");
    const auto width = impl.kv_layer_widths.at(owning_layer);
    const auto row_bytes = impl.kv_layer_row_bytes.at(owning_layer);
    if (keys.size() != values.size() || keys.size() % width != 0) {
        throw std::invalid_argument("GPU KV row write has the wrong shape");
    }
    const auto token_count = keys.size() / width;
    if (first_token > impl.kv_capacity || token_count > impl.kv_capacity - first_token) {
        throw std::invalid_argument("GPU KV row write exceeds cache capacity");
    }
    for (std::size_t token = 0; token < token_count; ++token) {
        const auto row = first_token + token;
        encode_kv_cache_row(
            impl.kv_mode,
            keys.subspan(token * width, width),
            {impl.kv_mapped + found->second.first + row * row_bytes, row_bytes});
        encode_kv_cache_row(
            impl.kv_mode,
            values.subspan(token * width, width),
            {impl.kv_mapped + found->second.second + row * row_bytes, row_bytes});
    }
    if (impl.cpu_kv_mirror) {
        auto key_mirror = kv_keys(owning_layer).subspan(first_token * width, token_count * width);
        auto value_mirror = kv_values(owning_layer).subspan(first_token * width, token_count * width);
        std::copy(keys.begin(), keys.end(), key_mirror.begin());
        std::copy(values.begin(), values.end(), value_mirror.begin());
    }
}

void GpuPlanExecution::read_kv_rows(
    std::size_t owning_layer,
    std::size_t first_token,
    std::size_t token_count,
    std::span<float> keys,
    std::span<float> values
) {
    auto& impl = *impl_;
    const auto found = impl.kv_layer_offsets.find(owning_layer);
    if (found == impl.kv_layer_offsets.end()) throw std::invalid_argument("unknown GPU KV layer");
    const auto width = impl.kv_layer_widths.at(owning_layer);
    const auto row_bytes = impl.kv_layer_row_bytes.at(owning_layer);
    if (first_token > impl.kv_capacity || token_count > impl.kv_capacity - first_token
        || keys.size() != token_count * width || values.size() != token_count * width) {
        throw std::invalid_argument("GPU KV row read has the wrong shape");
    }
    for (std::size_t token = 0; token < token_count; ++token) {
        const auto row = first_token + token;
        decode_kv_cache_row(
            impl.kv_mode,
            {impl.kv_mapped + found->second.first + row * row_bytes, row_bytes},
            keys.subspan(token * width, width));
        decode_kv_cache_row(
            impl.kv_mode,
            {impl.kv_mapped + found->second.second + row * row_bytes, row_bytes},
            values.subspan(token * width, width));
    }
}

std::span<float> GpuPlanExecution::kv_keys(std::size_t owning_layer) {
    auto& impl = *impl_;
    if (!impl.cpu_kv_mirror) return {};
    const auto found = impl.kv_cpu_layer_offsets.find(owning_layer);
    if (found == impl.kv_cpu_layer_offsets.end()) return {};
    const auto width = impl.kv_layer_widths.at(owning_layer);
    return {impl.kv_cpu_mapped.data() + found->second.first, impl.kv_capacity * width};
}

std::span<float> GpuPlanExecution::kv_values(std::size_t owning_layer) {
    auto& impl = *impl_;
    if (!impl.cpu_kv_mirror) return {};
    const auto found = impl.kv_cpu_layer_offsets.find(owning_layer);
    if (found == impl.kv_cpu_layer_offsets.end()) return {};
    const auto width = impl.kv_layer_widths.at(owning_layer);
    return {impl.kv_cpu_mapped.data() + found->second.second, impl.kv_capacity * width};
}

bool GpuPlanExecution::has_cpu_kv_mirror() const noexcept {
    return impl_ && impl_->cpu_kv_mirror;
}

std::string GpuPlanExecution::adapter_name() const {
    return impl_->bundle ? impl_->bundle->adapter_name : std::string{};
}

bool gpu_backend_available(std::string* reason) {
    if (const char* disabled = std::getenv("CATERPILLAR_GPU");
        disabled != nullptr && std::string_view(disabled) == "0") {
        if (reason != nullptr) *reason = "disabled via CATERPILLAR_GPU=0";
        return false;
    }
    try {
        static_cast<void>(create_device_bundle(true));
        return true;
    } catch (const std::exception& error) {
        if (reason != nullptr) *reason = error.what();
        return false;
    }
}

// ---------------------------------------------------------------------------
// Selftest
// ---------------------------------------------------------------------------
namespace {

std::uint16_t float_to_fp16(float value) {
    // Round-to-nearest-even float32 -> float16, sufficient for the sane
    // synthetic scale values used below (no subnormal/overflow cases).
    std::uint32_t bits;
    std::memcpy(&bits, &value, 4);
    const std::uint32_t sign = (bits >> 16U) & 0x8000U;
    const std::int32_t exponent = static_cast<std::int32_t>((bits >> 23U) & 0xFFU) - 127 + 15;
    std::uint32_t mantissa = bits & 0x7FFFFFU;
    if (exponent <= 0) return static_cast<std::uint16_t>(sign);
    if (exponent >= 31) return static_cast<std::uint16_t>(sign | 0x7C00U);
    const std::uint32_t rounded = mantissa + 0x1000U + (((mantissa >> 13U) & 1U) == 0U && (mantissa & 0x1FFFU) == 0x1000U ? 0U : 0U);
    mantissa = rounded;
    std::uint32_t half = sign | (static_cast<std::uint32_t>(exponent) << 10U) | ((mantissa >> 13U) & 0x3FFU);
    if ((mantissa & 0x800000U) != 0U) half += 1U << 10U;
    return static_cast<std::uint16_t>(half);
}

struct SyntheticWeight {
    GgufTensorInfo info;
    std::vector<std::byte> bytes;
};

SyntheticWeight make_synthetic_weight(std::string name, std::uint32_t type, std::size_t rows, std::size_t columns) {
    SyntheticWeight weight;
    // GGUF dimension order: [row width, row count].
    weight.info = GgufTensorInfo{std::move(name), {columns, rows}, type, 0};
    const auto pattern_byte = [](std::size_t index) {
        return static_cast<std::byte>((index * 37U + 11U) & 0xFFU);
    };
    switch (type) {
        case 0: {
            weight.bytes.resize(rows * columns * 4);
            auto* values = reinterpret_cast<float*>(weight.bytes.data());
            for (std::size_t index = 0; index < rows * columns; ++index) {
                values[index] = 0.01F * static_cast<float>((index % 41) + 1) * ((index % 3) == 0 ? -1.0F : 1.0F);
            }
            break;
        }
        case 1: {
            weight.bytes.resize(rows * columns * 2);
            auto* halves = reinterpret_cast<std::uint16_t*>(weight.bytes.data());
            for (std::size_t index = 0; index < rows * columns; ++index) {
                halves[index] = float_to_fp16(0.02F * static_cast<float>((index % 29) + 1) * ((index % 5) == 0 ? -1.0F : 1.0F));
            }
            break;
        }
        case 2: {  // Q4_0: fp16 d + 16 quant bytes
            const auto blocks = rows * columns / 32;
            weight.bytes.resize(blocks * 18);
            for (std::size_t block = 0; block < blocks; ++block) {
                auto* base = weight.bytes.data() + block * 18;
                const auto d = float_to_fp16(0.02F + 0.001F * static_cast<float>(block % 7));
                std::memcpy(base, &d, 2);
                for (std::size_t index = 0; index < 16; ++index) base[2 + index] = pattern_byte(block * 16 + index);
            }
            break;
        }
        case 8: {  // Q8_0: fp16 d + 32 int8
            const auto blocks = rows * columns / 32;
            weight.bytes.resize(blocks * 34);
            for (std::size_t block = 0; block < blocks; ++block) {
                auto* base = weight.bytes.data() + block * 34;
                const auto d = float_to_fp16(0.015F + 0.001F * static_cast<float>(block % 5));
                std::memcpy(base, &d, 2);
                for (std::size_t index = 0; index < 32; ++index) base[2 + index] = pattern_byte(block * 32 + index);
            }
            break;
        }
        case 12: {  // Q4_K: d/dmin fp16, 12 scale bytes, 128 quant bytes
            const auto blocks = rows * columns / 256;
            weight.bytes.resize(blocks * 144);
            for (std::size_t block = 0; block < blocks; ++block) {
                auto* base = weight.bytes.data() + block * 144;
                const auto d = float_to_fp16(0.02F + 0.0005F * static_cast<float>(block % 9));
                const auto dmin = float_to_fp16(0.01F);
                std::memcpy(base, &d, 2);
                std::memcpy(base + 2, &dmin, 2);
                for (std::size_t index = 0; index < 12; ++index) base[4 + index] = pattern_byte(block * 12 + index);
                for (std::size_t index = 0; index < 128; ++index) base[16 + index] = pattern_byte(block * 128 + index + 3);
            }
            break;
        }
        case 13: {  // Q5_K: Q4_K fields + 32 high-bit bytes
            const auto blocks = rows * columns / 256;
            weight.bytes.resize(blocks * 176);
            for (std::size_t block = 0; block < blocks; ++block) {
                auto* base = weight.bytes.data() + block * 176;
                const auto d = float_to_fp16(0.018F + 0.0005F * static_cast<float>(block % 9));
                const auto dmin = float_to_fp16(0.009F);
                std::memcpy(base, &d, 2);
                std::memcpy(base + 2, &dmin, 2);
                for (std::size_t index = 0; index < 12; ++index) base[4 + index] = pattern_byte(block * 12 + index);
                for (std::size_t index = 0; index < 32; ++index) base[16 + index] = pattern_byte(block * 32 + index + 5);
                for (std::size_t index = 0; index < 128; ++index) base[48 + index] = pattern_byte(block * 128 + index + 3);
            }
            break;
        }
        case 14: {  // Q6_K: ql[128] qh[64] scales[16] d fp16
            const auto blocks = rows * columns / 256;
            weight.bytes.resize(blocks * 210);
            for (std::size_t block = 0; block < blocks; ++block) {
                auto* base = weight.bytes.data() + block * 210;
                for (std::size_t index = 0; index < 208; ++index) base[index] = pattern_byte(block * 208 + index);
                const auto d = float_to_fp16(0.005F + 0.0002F * static_cast<float>(block % 11));
                std::memcpy(base + 208, &d, 2);
            }
            break;
        }
        case 20: {  // IQ4_NL: fp16 d + 16 nonlinear-codebook nibbles
            const auto blocks = rows * columns / 32;
            weight.bytes.resize(blocks * 18);
            for (std::size_t block = 0; block < blocks; ++block) {
                auto* base = weight.bytes.data() + block * 18;
                const auto d = float_to_fp16(0.006F + 0.0002F * static_cast<float>(block % 7));
                std::memcpy(base, &d, 2);
                for (std::size_t index = 0; index < 16; ++index) {
                    base[2 + index] = pattern_byte(block * 16 + index);
                }
            }
            break;
        }
        case 23: {  // IQ4_XS: fp16 d, packed six-bit scales, IQ4_NL nibbles
            const auto blocks = rows * columns / 256;
            weight.bytes.resize(blocks * 136);
            for (std::size_t block = 0; block < blocks; ++block) {
                auto* base = weight.bytes.data() + block * 136;
                const auto d = float_to_fp16(0.001F + 0.00005F * static_cast<float>(block % 5));
                std::memcpy(base, &d, 2);
                std::uint16_t scales_h = 0;
                std::array<std::uint8_t, 4> scales_l{};
                for (std::size_t group = 0; group < 8; ++group) {
                    const auto encoded = static_cast<std::uint8_t>(24U + ((block + group * 7U) % 17U));
                    scales_l[group / 2] |= static_cast<std::uint8_t>(
                        (encoded & 0x0fU) << (4U * (group % 2)));
                    scales_h |= static_cast<std::uint16_t>(
                        ((encoded >> 4U) & 0x03U) << (2U * group));
                }
                std::memcpy(base + 2, &scales_h, 2);
                std::memcpy(base + 4, scales_l.data(), scales_l.size());
                for (std::size_t index = 0; index < 128; ++index) {
                    base[8 + index] = pattern_byte(block * 128 + index + 17);
                }
            }
            break;
        }
        default:
            throw std::runtime_error("unsupported synthetic weight type");
    }
    return weight;
}

float max_absolute_difference(const std::vector<float>& left, const std::vector<float>& right) {
    if (left.size() != right.size()) return std::numeric_limits<float>::infinity();
    float worst = 0.0F;
    for (std::size_t index = 0; index < left.size(); ++index) {
        worst = std::max(worst, std::abs(left[index] - right[index]));
    }
    return worst;
}

void verify_close(const char* label, const std::vector<float>& reference, const std::vector<float>& actual, float tolerance) {
    const auto difference = max_absolute_difference(reference, actual);
    if (!(difference <= tolerance)) {
        throw std::runtime_error(std::string(label) + ": GPU result diverges from CPU reference (max diff "
            + std::to_string(difference) + ", tolerance " + std::to_string(tolerance) + ")");
    }
}

std::vector<float> quantize_q8_1_dequantize(std::span<const float> input) {
    if ((input.size() % 32) != 0) {
        throw std::invalid_argument("Q8_1 reference input must be a multiple of 32");
    }
    std::vector<float> output(input.size());
    for (std::size_t block = 0; block < input.size(); block += 32) {
        float absolute_max = 0.0F;
        for (std::size_t index = 0; index < 32; ++index) {
            absolute_max = std::max(absolute_max, std::abs(input[block + index]));
        }
        const float d = absolute_max / 127.0F;
        const float inverse_d = d > 0.0F ? 1.0F / d : 0.0F;
        for (std::size_t index = 0; index < 32; ++index) {
            // HLSL round() lowers to round-to-nearest-even. nearbyint uses
            // the default FE_TONEAREST mode and deliberately differs from
            // std::round's halfway-away-from-zero rule.
            const auto quantized = std::clamp(
                static_cast<int>(std::nearbyint(input[block + index] * inverse_d)),
                -127, 127);
            output[block + index] = d * static_cast<float>(quantized);
        }
    }
    return output;
}

}  // namespace

int run_gpu_selftest() {
    std::string reason;
    if (!gpu_backend_available(&reason)) {
        std::cout << "caterpillar gpu selftest: skipped (" << reason << ")\n";
        return 0;
    }
    try {
        // --- Case 1: quantized/float matvec per staged format ----------------
        {
            constexpr std::size_t rows = 48;
            constexpr std::size_t columns = 512;
            Graph graph;
            const auto input_id = graph.add_tensor("input", {1, columns}, true);
            std::vector<SyntheticWeight> weights;
            std::vector<TensorId> weight_ids;
            std::vector<TensorId> output_ids;
            std::vector<std::uint32_t> types{0, 1, 2, 8, 12, 13, 14, 20, 23};
            // Bisection aid for device-hang debugging: comma-separated GGML
            // type list, e.g. CATERPILLAR_GPU_SELFTEST_TYPES=12,14.
            if (const char* filter = std::getenv("CATERPILLAR_GPU_SELFTEST_TYPES");
                filter != nullptr && *filter != '\0') {
                types.clear();
                std::string token;
                std::istringstream stream(filter);
                while (std::getline(stream, token, ',')) {
                    if (!token.empty()) types.push_back(static_cast<std::uint32_t>(std::stoul(token)));
                }
            }
            for (const auto type : types) {
                weights.push_back(make_synthetic_weight("w" + std::to_string(type), type, rows, columns));
            }
            for (std::size_t index = 0; index < weights.size(); ++index) {
                const auto weight_id = graph.add_tensor(weights[index].info.name, {rows, columns}, true);
                const auto output_id = graph.add_tensor("out" + std::to_string(index), {1, rows});
                graph.add_node(Op::matmul, {input_id, weight_id}, output_id);
                weight_ids.push_back(weight_id);
                output_ids.push_back(output_id);
            }
            graph.set_outputs(output_ids);

            QuantizedBindings quantized;
            for (std::size_t index = 0; index < weights.size(); ++index) {
                quantized.emplace(weight_ids[index], QuantizedTensorView{&weights[index].info, weights[index].bytes});
            }
            Bindings inputs;
            auto& input = inputs[input_id];
            input.resize(columns);
            for (std::size_t index = 0; index < columns; ++index) {
                input[index] = 0.05F * static_cast<float>((index % 23) + 1) * ((index % 4) == 0 ? -1.0F : 1.0F);
            }

            const auto cpu_plan = Plan::compile(graph);
            const auto cpu_outputs = cpu_plan.replay(inputs, quantized);

            Bindings q8_1_inputs = inputs;
            q8_1_inputs[input_id] = quantize_q8_1_dequantize(inputs.at(input_id));
            const auto q8_1_cpu_outputs = cpu_plan.replay(q8_1_inputs, quantized);

            const auto gpu_plan = Plan::compile(graph, std::vector<SpanBackend>(graph.nodes().size(), SpanBackend::gpu));
            GpuPlanExecution gpu(
                gpu_plan, quantized, {}, {input_id}, {},
                GpuPlanExecution::Config{4, true, true, true});
            BindingViews views;
            views.emplace(input_id, input);
            const auto gpu_outputs = gpu_plan.replay_views(views, quantized, {}, &gpu);
            const auto gpu_again = gpu_plan.replay_views(views, quantized, {}, &gpu);
            for (std::size_t index = 0; index < output_ids.size(); ++index) {
                const bool quantized_type = types[index] == 2 || types[index] == 8
                    || types[index] == 12 || types[index] == 13 || types[index] == 14;
                const auto& expected = quantized_type && gpu.telemetry().integer_dot_active
                    ? q8_1_cpu_outputs.at(output_ids[index])
                    : cpu_outputs.at(output_ids[index]);
                verify_close(
                    ("matvec type " + std::to_string(types[index])).c_str(),
                    expected,
                    gpu_outputs.at(output_ids[index]),
                    2.0e-3F);
                if (!bit_equal(gpu_outputs.at(output_ids[index]), gpu_again.at(output_ids[index]))) {
                    throw std::runtime_error("recorded matvec replay is not deterministic");
                }
            }
            if (gpu.telemetry().integer_dot_available) {
                const auto integer_format_count = static_cast<std::uint64_t>(std::count_if(
                    types.begin(), types.end(), [](std::uint32_t type) {
                        return type == 2 || type == 8 || type == 12 || type == 13 || type == 14;
                    }));
                const bool telemetry_matches = integer_format_count == 0
                    ? !gpu.telemetry().integer_dot_active
                        && gpu.telemetry().q8_1_quantizations == 0
                        && gpu.telemetry().integer_dot_matvecs == 0
                    : gpu.telemetry().integer_dot_active
                        && gpu.telemetry().q8_1_quantizations == 2
                        && gpu.telemetry().integer_dot_matvecs == integer_format_count * 2;
                if (!telemetry_matches) {
                    throw std::runtime_error("DP4A telemetry did not match the selected quantized formats");
                }
            }
        }

        if (const char* matvec_only = std::getenv("CATERPILLAR_GPU_SELFTEST_MATVEC_ONLY");
            matvec_only != nullptr && std::string_view(matvec_only) == "1") {
            std::cout << "caterpillar gpu matvec selftest: ok\n";
            return 0;
        }

        // Neutral, opt-in Phase 5 A/B gate. The two executions use the same
        // graph, weights, input, submission count, and interleaved ordering;
        // only the float-vs-DP4A kernel selection differs.
        if (const char* performance = std::getenv("CATERPILLAR_GPU_SELFTEST_PERF");
            performance != nullptr && std::string_view(performance) == "1") {
            constexpr std::size_t rows = 8192;
            constexpr std::size_t columns = 3072;
            Graph graph;
            const auto input_id = graph.add_tensor("perf.input", {1, columns}, true);
            auto gate_weight = make_synthetic_weight("perf.gate.q4k", 12, rows, columns);
            auto up_weight = make_synthetic_weight("perf.up.q4k", 12, rows, columns);
            const auto gate_weight_id = graph.add_tensor(gate_weight.info.name, {rows, columns}, true);
            const auto up_weight_id = graph.add_tensor(up_weight.info.name, {rows, columns}, true);
            const auto gate_output_id = graph.add_tensor("perf.gate", {1, rows});
            const auto up_output_id = graph.add_tensor("perf.up", {1, rows});
            graph.add_node(Op::matmul, {input_id, gate_weight_id}, gate_output_id);
            graph.add_node(Op::matmul, {input_id, up_weight_id}, up_output_id);
            graph.set_outputs({gate_output_id, up_output_id});
            QuantizedBindings quantized{
                {gate_weight_id, QuantizedTensorView{&gate_weight.info, gate_weight.bytes}},
                {up_weight_id, QuantizedTensorView{&up_weight.info, up_weight.bytes}},
            };
            std::vector<float> input(columns);
            for (std::size_t index = 0; index < columns; ++index) {
                input[index] = 0.04F * static_cast<float>((index % 31) + 1)
                    * ((index % 5) == 0 ? -1.0F : 1.0F);
            }
            BindingViews views{{input_id, input}};
            const auto plan = Plan::compile(
                graph, std::vector<SpanBackend>(graph.nodes().size(), SpanBackend::gpu));
            GpuPlanExecution float_execution(
                plan, quantized, {}, {input_id}, {}, GpuPlanExecution::Config{4, true, false});
            GpuPlanExecution integer_execution(
                plan, quantized, {}, {input_id}, {}, GpuPlanExecution::Config{4, true, true});
            if (!integer_execution.telemetry().integer_dot_active) {
                throw std::runtime_error(
                    "DP4A performance gate requested but integer-dot execution is unavailable: "
                    + integer_execution.telemetry().integer_dot_reason);
            }
            for (int warmup = 0; warmup < 3; ++warmup) {
                static_cast<void>(plan.replay_views(views, quantized, {}, &float_execution));
                static_cast<void>(plan.replay_views(views, quantized, {}, &integer_execution));
            }
            std::chrono::nanoseconds float_time{};
            std::chrono::nanoseconds integer_time{};
            constexpr int iterations = 16;
            const auto measure = [&](GpuPlanExecution& execution, std::chrono::nanoseconds& elapsed) {
                const auto started = std::chrono::steady_clock::now();
                static_cast<void>(plan.replay_views(views, quantized, {}, &execution));
                elapsed += std::chrono::steady_clock::now() - started;
            };
            for (int iteration = 0; iteration < iterations; ++iteration) {
                if ((iteration & 1) == 0) {
                    measure(float_execution, float_time);
                    measure(integer_execution, integer_time);
                } else {
                    measure(integer_execution, integer_time);
                    measure(float_execution, float_time);
                }
            }
            const double float_ms = std::chrono::duration<double, std::milli>(float_time).count() / iterations;
            const double integer_ms = std::chrono::duration<double, std::milli>(integer_time).count() / iterations;
            const double speedup = integer_ms > 0.0 ? float_ms / integer_ms : 0.0;
            std::cout << "caterpillar DP4A A/B: shared-Q8_1 q4_k gate+up 2x8192x3072 float=" << float_ms
                      << " ms integer=" << integer_ms << " ms speedup=" << speedup << "x\n";
            if (!(integer_ms < float_ms)) {
                throw std::runtime_error("DP4A A/B gate failed: integer-dot was not faster than float");
            }
        }

        // --- Case 2: decoder op micro-graph with dynamic past -----------------
        if (std::getenv("CATERPILLAR_GPU_SELFTEST_TYPES") == nullptr) {
            constexpr std::size_t hidden = 32;
            constexpr std::size_t heads = 1;
            constexpr std::size_t kv_heads = 1;
            constexpr std::size_t head_dim = 32;
            constexpr std::size_t capacity = 4;
            Graph graph;
            const auto x_id = graph.add_tensor("token_state", {1, hidden}, true);
            const auto norm_w_id = graph.add_tensor("attn_norm.weight", {hidden}, true);
            const auto normed_id = graph.add_tensor("normed", {1, hidden});
            graph.add_node(Op::rms_norm, {x_id, norm_w_id}, normed_id, 1.0e-5F);
            const auto q_w = make_synthetic_weight("wq", 0, heads * head_dim, hidden);
            const auto k_w = make_synthetic_weight("wk", 0, kv_heads * head_dim, hidden);
            const auto v_w = make_synthetic_weight("wv", 0, kv_heads * head_dim, hidden);
            const auto q_w_id = graph.add_tensor("wq", {heads * head_dim, hidden}, true);
            const auto k_w_id = graph.add_tensor("wk", {kv_heads * head_dim, hidden}, true);
            const auto v_w_id = graph.add_tensor("wv", {kv_heads * head_dim, hidden}, true);
            const auto query_id = graph.add_tensor("query", {1, heads * head_dim});
            const auto key_id = graph.add_tensor("key", {1, kv_heads * head_dim});
            const auto value_id = graph.add_tensor("value", {1, kv_heads * head_dim});
            graph.add_node(Op::matmul, {normed_id, q_w_id}, query_id);
            graph.add_node(Op::matmul, {normed_id, k_w_id}, key_id);
            graph.add_node(Op::matmul, {normed_id, v_w_id}, value_id);
            const auto q_normed_id = graph.add_tensor("query_normed", {1, heads * head_dim});
            graph.add_node(Op::rms_norm_heads_unweighted, {query_id}, q_normed_id, 1.0e-5F, {heads, head_dim});
            const auto q_rot_id = graph.add_tensor("query_rot", {1, heads * head_dim});
            const auto k_rot_id = graph.add_tensor("key_rot", {1, kv_heads * head_dim});
            graph.add_node(Op::rope, {q_normed_id}, q_rot_id, 10000.0F, {heads, head_dim, 0, 1}, 1.0F);
            graph.add_node(Op::rope, {key_id}, k_rot_id, 10000.0F, {kv_heads, head_dim, 0, 1}, 1.0F);
            const auto past_k_id = graph.add_tensor("runtime.kv.0.key", {capacity, kv_heads * head_dim}, true);
            const auto past_v_id = graph.add_tensor("runtime.kv.0.value", {capacity, kv_heads * head_dim}, true);
            const auto attended_id = graph.add_tensor("attended", {1, heads * head_dim});
            graph.add_node(
                Op::causal_attention,
                {q_rot_id, k_rot_id, value_id, past_k_id, past_v_id},
                attended_id,
                0.0F,
                {heads, kv_heads, head_dim, 3, capacity});
            const auto gate_id = graph.add_tensor("gate", {1, hidden});
            const auto up_id = graph.add_tensor("up", {1, hidden});
            const auto o_w = make_synthetic_weight("wo", 0, hidden, heads * head_dim);
            const auto o_w_id = graph.add_tensor("wo", {hidden, heads * head_dim}, true);
            graph.add_node(Op::matmul, {attended_id, o_w_id}, gate_id);
            graph.add_node(Op::add, {gate_id, x_id}, up_id);
            const auto silu_id = graph.add_tensor("silu", {1, hidden});
            const auto gelu_id = graph.add_tensor("gelu", {1, hidden});
            graph.add_node(Op::silu_mul, {gate_id, up_id}, silu_id);
            graph.add_node(Op::gelu_mul, {gate_id, up_id}, gelu_id);
            const auto scale_w_id = graph.add_tensor("output_scale", {1}, true);
            const auto scaled_id = graph.add_tensor("scaled", {1, hidden});
            graph.add_node(Op::scale, {silu_id, scale_w_id}, scaled_id);
            const auto sum_id = graph.add_tensor("sum", {1, hidden});
            graph.add_node(Op::add, {scaled_id, gelu_id}, sum_id);
            graph.set_outputs({sum_id, attended_id, k_rot_id, value_id});

            QuantizedBindings quantized;
            quantized.emplace(q_w_id, QuantizedTensorView{&q_w.info, q_w.bytes});
            quantized.emplace(k_w_id, QuantizedTensorView{&k_w.info, k_w.bytes});
            quantized.emplace(v_w_id, QuantizedTensorView{&v_w.info, v_w.bytes});
            quantized.emplace(o_w_id, QuantizedTensorView{&o_w.info, o_w.bytes});

            Bindings immutable;
            auto& norm_w = immutable[norm_w_id];
            norm_w.resize(hidden);
            for (std::size_t index = 0; index < hidden; ++index) norm_w[index] = 1.0F + 0.05F * static_cast<float>(index);
            immutable[scale_w_id] = {0.75F};

            std::vector<float> token(hidden);
            for (std::size_t index = 0; index < hidden; ++index) {
                token[index] = 0.1F * static_cast<float>(index + 1) * ((index % 2) == 0 ? 1.0F : -0.5F);
            }
            std::vector<float> past_keys(capacity * kv_heads * head_dim);
            std::vector<float> past_values(capacity * kv_heads * head_dim);
            for (std::size_t index = 0; index < past_keys.size(); ++index) {
                past_keys[index] = 0.03F * static_cast<float>((index % 13) + 1);
                past_values[index] = 0.02F * static_cast<float>((index % 17) + 1) * ((index % 3) == 0 ? -1.0F : 1.0F);
            }

            auto backends = std::vector<SpanBackend>(graph.nodes().size(), SpanBackend::gpu);
            // Bisection aid: run only the first N nodes on the GPU.
            if (const char* prefix = std::getenv("CATERPILLAR_GPU_SELFTEST_CASE2_NODES");
                prefix != nullptr && *prefix != '\0') {
                const auto count = static_cast<std::size_t>(std::stoul(prefix));
                for (std::size_t index = 0; index < backends.size(); ++index) {
                    backends[index] = index < count ? SpanBackend::gpu : SpanBackend::cpu;
                }
            }
            const auto gpu_plan = Plan::compile(graph, backends);
            const auto cpu_plan = Plan::compile(graph);
            for (const auto mode : {
                KvCacheMode::f32, KvCacheMode::f16, KvCacheMode::q8_0,
                KvCacheMode::q4_0, KvCacheMode::q3_wht}) {
                if (const char* filter = std::getenv("CATERPILLAR_GPU_SELFTEST_KV");
                    filter != nullptr && *filter != '\0' && std::string_view(filter) != kv_cache_mode_name(mode)) {
                    continue;
                }
                GpuPlanExecution gpu(
                    gpu_plan, quantized, immutable, {x_id},
                    {DecoderGraph::KvBinding{0, past_k_id, past_v_id}},
                    GpuPlanExecution::Config{capacity, true, false, false, mode});
                std::vector<float> reference_keys(past_keys.size());
                std::vector<float> reference_values(past_values.size());
                const auto width = kv_heads * head_dim;
                const auto encoded_bytes = kv_cache_row_bytes(mode, width);
                std::vector<std::byte> encoded(encoded_bytes);
                for (std::size_t row = 0; row < capacity; ++row) {
                    encode_kv_cache_row(mode, std::span<const float>(past_keys).subspan(row * width, width), encoded);
                    decode_kv_cache_row(mode, encoded, std::span<float>(reference_keys).subspan(row * width, width));
                    encode_kv_cache_row(mode, std::span<const float>(past_values).subspan(row * width, width), encoded);
                    decode_kv_cache_row(mode, encoded, std::span<float>(reference_values).subspan(row * width, width));
                }
                for (const std::size_t past : {std::size_t{0}, std::size_t{2}, std::size_t{4}}) {
                    gpu.seed_kv(0, past_keys, past_values, capacity);
                    ReplayDynamics dynamics;
                    dynamics.past_tokens = past;
                    dynamics.position_offset = past;
                    Bindings cpu_inputs = immutable;
                    cpu_inputs[x_id] = token;
                    cpu_inputs[past_k_id] = reference_keys;
                    cpu_inputs[past_v_id] = reference_values;
                    BindingViews cpu_views;
                    for (const auto& [id, values] : cpu_inputs) cpu_views.emplace(id, values);
                    const auto cpu_outputs = cpu_plan.replay_views(cpu_views, quantized, dynamics, nullptr);

                    BindingViews gpu_views;
                    for (const auto& [id, values] : immutable) gpu_views.emplace(id, values);
                    gpu_views.emplace(x_id, token);
                    const auto gpu_outputs = gpu_plan.replay_views(gpu_views, quantized, dynamics, &gpu);
                    const auto label = std::string(kv_cache_mode_name(mode)) + " past=" + std::to_string(past);
                    verify_close(("decoder ops " + label).c_str(), cpu_outputs.at(sum_id), gpu_outputs.at(sum_id), 8.0e-4F);
                    verify_close(("attention " + label).c_str(), cpu_outputs.at(attended_id), gpu_outputs.at(attended_id), 8.0e-4F);
                    if (past == 2) {
                        // Consume the row that the preceding replay appended
                        // on-device. This proves append packing as well as
                        // seed/dequant correctness for every cache mode.
                        auto appended_keys = reference_keys;
                        auto appended_values = reference_values;
                        encode_kv_cache_row(mode, cpu_outputs.at(k_rot_id), encoded);
                        decode_kv_cache_row(
                            mode, encoded,
                            std::span<float>(appended_keys).subspan(past * width, width));
                        encode_kv_cache_row(mode, cpu_outputs.at(value_id), encoded);
                        decode_kv_cache_row(
                            mode, encoded,
                            std::span<float>(appended_values).subspan(past * width, width));
                        ReplayDynamics appended_dynamics;
                        appended_dynamics.past_tokens = past + 1;
                        appended_dynamics.position_offset = past + 1;
                        Bindings appended_cpu_inputs = immutable;
                        appended_cpu_inputs[x_id] = token;
                        appended_cpu_inputs[past_k_id] = appended_keys;
                        appended_cpu_inputs[past_v_id] = appended_values;
                        BindingViews appended_cpu_views;
                        for (const auto& [id, values] : appended_cpu_inputs) {
                            appended_cpu_views.emplace(id, values);
                        }
                        const auto appended_cpu = cpu_plan.replay_views(
                            appended_cpu_views, quantized, appended_dynamics, nullptr);
                        const auto appended_gpu = gpu_plan.replay_views(
                            gpu_views, quantized, appended_dynamics, &gpu);
                        const float append_tolerance = mode == KvCacheMode::f16 ? 2.0e-3F : 8.0e-4F;
                        verify_close(
                            ("device append " + label).c_str(),
                            appended_cpu.at(attended_id), appended_gpu.at(attended_id), append_tolerance);
                    }
                }
                if (gpu.telemetry().kv_device_appends == 0) {
                    throw std::runtime_error(std::string(kv_cache_mode_name(mode)) + " did not execute device-resident KV append");
                }
            }
        }

        std::cout << "caterpillar gpu selftest: ok (matvec formats + DP4A + resident-KV f32/f16/q8_0/q4_0/q3_wht + dynamic past, recorded-replay deterministic)\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "caterpillar gpu selftest: failed: " << error.what() << '\n';
        return 1;
    }
}

#endif  // _WIN32

}  // namespace cat
