// DP4A decode kernels (Shader Model 6.4, compiled offline with DXC to DXIL
// and embedded at build time; see CMakeLists.txt).
//
// Activations are quantized once per projection group to Q8_1 blocks of 32
// (packed int8 quants + fp16 d + fp16 s where s = d * sum(q)), then Q4_K
// weights are multiplied against the quantized activations with
// dot4add_i8packed integer dot products, following the llama.cpp MMQ scheme.
// Weights are consumed straight from the existing resident column-major
// Q4_K arena; no weight restaging is required.
//
// Root signature (shared with the FXC pipelines):
//   b0: 8 root constants (DispatchConstants)
//   t0: weights (raw Q4_K blocks, column-major interleaved)
//   t1: inputs
//   u0: outputs
//
// Q8_1 activation buffer layout for a row_width-wide vector:
//   uint[row_width / 4]        packed int8 quants (4 per uint, little-endian)
//   uint[row_width / 32 * 2]   per-block scales as full F32 bit patterns:
//                              word 0 = d, word 1 = s (f32tof16 is
//                              round-toward-zero in DXIL, which biases the
//                              Q4_K min-correction term; full floats keep the
//                              GPU math bit-comparable with the CPU path)

cbuffer DispatchParams : register(b0) {
  uint row_width;
  uint row_count;
  uint batch_count;
  uint output_base;
  uint scalar_format;
  uint input_stride;
  uint weight_layout;
  uint input_transform;
};

ByteAddressBuffer weights : register(t0);

// ---------------------------------------------------------------------------
// Activation quantization: float vector -> Q8_1 blocks of 32.
// t1 holds the source floats; u0 receives the packed layout above.
// Constants: row_width = vector width (multiple of 32);
//            input_stride = source offset in floats;
//            output_base = destination offset in uints.
// One 64-thread group quantizes two 32-element blocks.
// ---------------------------------------------------------------------------
StructuredBuffer<float> quantize_inputs : register(t1);
RWStructuredBuffer<uint> quantize_outputs : register(u0);

groupshared float quantize_partials[64];
groupshared uint quantize_bytes[64];

[numthreads(64, 1, 1)]
void main_quantize_q8_1(uint3 group_id : SV_GroupID, uint thread_index : SV_GroupIndex) {
  // group_id.y selects the column, so one dispatch covers the whole batch
  // instead of the host issuing one dispatch per column. Callers that quantize a
  // single vector dispatch y = 1, which makes column 0 and both offsets below
  // collapse to input_stride / output_base exactly as before.
  const uint column = group_id.y;
  const uint source_offset = input_stride + column * row_width;
  const uint dest_offset = output_base + column * input_transform;
  const uint element = group_id.x * 64u + thread_index;
  // Clamp BEFORE the load (speculative-load safety on Intel Xe); mask the
  // stored value with the real bounds check.
  const uint safe_element = min(element, row_width - 1u);
  const float loaded = quantize_inputs[source_offset + safe_element];
  const float value = element < row_width ? loaded : 0.0f;

  const uint block_lane = thread_index & 31u;
  const uint block_base = thread_index & 32u;

  // Per-32-block absmax reduction.
  quantize_partials[thread_index] = abs(value);
  GroupMemoryBarrierWithGroupSync();
  [unroll]
  for (uint stride = 16u; stride > 0u; stride >>= 1u) {
    if (block_lane < stride) {
      quantize_partials[thread_index] =
        max(quantize_partials[thread_index], quantize_partials[thread_index + stride]);
    }
    GroupMemoryBarrierWithGroupSync();
  }
  const float block_absmax = quantize_partials[block_base];
  const float d = block_absmax / 127.0f;
  const float inverse_d = d > 0.0f ? 1.0f / d : 0.0f;
  const int quantized = clamp((int)round(value * inverse_d), -127, 127);

  // Pack four int8 lanes per uint.
  quantize_bytes[thread_index] = (uint)(quantized & 0xFF);
  GroupMemoryBarrierWithGroupSync();
  if ((thread_index & 3u) == 0u && element < row_width) {
    quantize_outputs[dest_offset + element / 4u] =
      quantize_bytes[thread_index]
      | (quantize_bytes[thread_index + 1u] << 8u)
      | (quantize_bytes[thread_index + 2u] << 16u)
      | (quantize_bytes[thread_index + 3u] << 24u);
  }

  // Per-32-block sum of quantized values for the Q4_K min correction.
  quantize_partials[thread_index] = (float)quantized;
  GroupMemoryBarrierWithGroupSync();
  [unroll]
  for (uint sum_stride = 16u; sum_stride > 0u; sum_stride >>= 1u) {
    if (block_lane < sum_stride) {
      quantize_partials[thread_index] += quantize_partials[thread_index + sum_stride];
    }
    GroupMemoryBarrierWithGroupSync();
  }
  if (block_lane == 0u && element < row_width) {
    const float s = d * quantize_partials[block_base];
    const uint scale_base = dest_offset + row_width / 4u + (element / 32u) * 2u;
    quantize_outputs[scale_base] = asuint(d);
    quantize_outputs[scale_base + 1u] = asuint(s);
  }
}

// ---------------------------------------------------------------------------
// Q4_K x Q8_1 batch-one matvec with packed integer dot products.
// t0: resident Q4_K blocks, column-major ((block * row_count + row) * 144).
// t1: Q8_1 activation buffer (layout above).
// u0: float outputs.
// Constants: row_width, row_count as usual; input_stride = activation base
//            offset in uints; output_base = output offset in floats.
// ---------------------------------------------------------------------------
StructuredBuffer<uint> packed_activations : register(t1);
RWStructuredBuffer<float> matvec_outputs : register(u0);

uint dp4a_buffer_byte(uint byte_address) {
  return (weights.Load(byte_address & ~3u) >> ((byte_address & 3u) * 8u)) & 0xFFu;
}

uint dp4a_scale_k4(uint j, uint scales_byte_base) {
  return j < 4u
    ? (dp4a_buffer_byte(scales_byte_base + j) & 63u)
    : ((dp4a_buffer_byte(scales_byte_base + j + 4u) & 0x0Fu)
      | ((dp4a_buffer_byte(scales_byte_base + j - 4u) >> 6u) << 4u));
}

uint dp4a_min_k4(uint j, uint scales_byte_base) {
  return j < 4u
    ? (dp4a_buffer_byte(scales_byte_base + j + 4u) & 63u)
    : ((dp4a_buffer_byte(scales_byte_base + j + 4u) >> 4u)
      | ((dp4a_buffer_byte(scales_byte_base + j) >> 6u) << 4u));
}

[numthreads(64, 1, 1)]
void main_q4k_q8_1_b1(uint3 group_id : SV_GroupID, uint thread_index : SV_GroupIndex) {
  const uint row = group_id.x * 64u + thread_index;
  if (row >= row_count) {
    return;
  }
  const uint blocks_per_row = row_width >> 8u;
  const uint scale_words_base = row_width / 4u;

  float total = 0.0f;
  [loop]
  for (uint block = 0; block < blocks_per_row; block += 1u) {
    // Column-major blocks: 64 adjacent rows read 64 adjacent 144-byte blocks.
    const uint block_byte = (block * row_count + row) * 144u;
    const uint d_words = weights.Load(block_byte);
    const float d = f16tof32(d_words & 0xFFFFu);
    const float dmin = f16tof32(d_words >> 16u);
    const uint scales_byte_base = block_byte + 4u;
    const uint qs_base = block_byte + 16u;

    // Eight 32-element sub-blocks per superblock. Sub-block j reads the low
    // (j even) or high (j odd) nibbles of qs bytes (j >> 1) * 32 .. +31,
    // covering elements block * 256 + j * 32 .. +31, which is exactly Q8_1
    // activation block (block * 8 + j).
    [unroll]
    for (uint j = 0; j < 8u; j += 1u) {
      const uint activation_block = block * 8u + j;
      const uint scale_base =
        input_stride + scale_words_base + activation_block * 2u;
      const float dx = asfloat(packed_activations[scale_base]);
      const float sx = asfloat(packed_activations[scale_base + 1u]);
      const float sub_scale = d * (float)dp4a_scale_k4(j, scales_byte_base);
      const float sub_min = dmin * (float)dp4a_min_k4(j, scales_byte_base);

      const uint chunk_bytes = qs_base + (j >> 1u) * 32u;
      const bool high_nibbles = (j & 1u) != 0u;
      int int_dot = 0;
      [unroll]
      for (uint k = 0; k < 8u; k += 1u) {
        const uint qs_word = weights.Load(chunk_bytes + k * 4u);
        const uint nibbles = high_nibbles
          ? ((qs_word >> 4u) & 0x0F0F0F0Fu)
          : (qs_word & 0x0F0F0F0Fu);
        const uint x_word =
          packed_activations[input_stride + activation_block * 8u + k];
        // Q4_K quants are 0..15: valid positive signed int8 lanes, so the
        // signed packed dot handles unsigned nibbles times signed int8.
        int_dot = dot4add_i8packed(nibbles, x_word, int_dot);
      }
      total += sub_scale * dx * (float)int_dot - sub_min * sx;
    }
  }
  matvec_outputs[output_base + row] = total;
}

// ---------------------------------------------------------------------------
// Q4_K x Q8_1 BATCHED matmul (prefill), packed integer dots.
//
// The batch-one kernel above reads a row's nibbles and consumes them for a single
// activation vector. Prefill has many vectors, so this loads each row's nibbles
// ONCE per sub-block and accumulates them against k_dp4a_tile columns - the same
// amortisation the float kernel gets from its batch tile, but with
// dot4add_i8packed doing four multiply-accumulates per instruction instead of one.
//
// That instruction density is the reason to try this at all: the float row-blocked
// kernel sits at 11-15% of this iGPU's arithmetic peak and every constant in it has
// now been measured in both directions, so the remaining factor has to come from
// doing more per instruction rather than from a better shape.
//
// Activation layout, per column, in uints:
//   [0, row_width/4)                packed int8 quants, four per uint
//   [row_width/4, +2 per 32-block)  (scale, sum) float pairs
// Columns are consecutive regions of that size; input_stride is the base.
//
// t0: resident Q4_K blocks, column-major ((block * row_count + row) * 144)
// t1: Q8_1 activations for every column
// u0: float outputs, column-major (column * row_count + row)
// ---------------------------------------------------------------------------
#ifndef LIZARD_DP4A_TILE
#define LIZARD_DP4A_TILE 8
#endif
// Shape: 8 columns x 1 row per thread, in groups of LIZARD_DP4A_GROUP threads.
//
// 4 columns x 2 rows was tried and lost: activation cost stayed at 4.97 ms
// against 4.77 while the rest doubled (3.44 -> 6.94 ms) with the doubled weight
// loads. Halving the per-thread LDS reads changed nothing, which says the
// activation cost is NOT read throughput - it is the staging pass itself.
//
// And the staging is redundant across groups: the activation tile for a set of
// columns is identical for every row-group, yet each of the ~160 row-groups
// stages it again from the buffer. A larger group covers more rows per staging
// pass, which amortises it without touching per-thread registers - the resource
// that killed row blocking.
//
// Measured, and it LOSES: a 256-thread group ran 14.05 ms against 64's 8.21, and
// the activation cost ROSE to 10.46 ms from 4.77. Fewer, larger groups mean fewer
// staging passes but also fewer groups resident to hide latency, and a barrier
// across 256 threads costs more than one across 64. Both theories about the
// activation path - read throughput, then staging redundancy - are now refuted,
// and what remains is the barrier pair itself plus whatever the staging loads
// cost at this occupancy. 8 columns x 1 row in 64-thread groups is the best
// measured shape.
//
// The ablation measured the activation path at 58% of this kernel (4.77 of
// 8.21 ms), and the cause is the reuse imbalance: at 8 columns x 1 row each
// weight is used 8 times while each activation word is used ONCE, so a thread
// pulls 512 activation words per block from groupshared against 64 weight
// loads - eight times the traffic. Halving the columns and doubling the rows
// keeps the output count and the unrolled body identical while cutting
// activation reads to 256 and raising weight loads to 128, which is a good
// trade at 8:1. It is not the same as the earlier row-blocking attempt, which
// held columns constant and so multiplied the body instead of trading.
// Output rows per thread for the integer kernel. The float kernel gets 1.51x from
// this and the integer one had none of it; the staged activation tile is shared
// across rows, so each extra row costs registers and weight loads but no extra
// groupshared traffic. Unlike the float kernel this is compiled by DXC, so the FXC
// unroll ceiling that capped rows x tile at 32 there does not apply here, and the
// 2.5 KB tile leaves far more occupancy headroom than the float kernel's 8 KB.
// 
// Measured anyway, and it LOSES here: R=4 ran 34.99 ms against R=1's 11.18 ms,
// a 3.1x regression. The unrolled body is rows x 8 sub-blocks x 8 columns x 8
// dot4adds, so R=4 is 2048 straight-line integer dots and the register pressure
// swamps the sharing it was meant to buy. R=1 is the setting.
#ifndef LIZARD_DP4A_GROUP
#define LIZARD_DP4A_GROUP 64
#endif
#ifndef LIZARD_DP4A_ROWS
#define LIZARD_DP4A_ROWS 1
#endif
static const uint k_dp4a_tile = LIZARD_DP4A_TILE;
static const uint k_dp4a_rows = LIZARD_DP4A_ROWS;
static const uint k_dp4a_group = LIZARD_DP4A_GROUP;

// Groupshared staging: 8 columns x 8 sub-blocks x 8 quant words = 512 uints,
// plus 8 x 8 (scale, sum) pairs = 128 uints. 2.5 KB, staged once per 256-element
// block, so the barrier cost is two per block rather than two per sub-block.
//
// This is the piece the first version lacked. Reading activations straight from
// the buffer meant 64 buffer loads per block per thread against the float
// kernel's 16 groupshared reads, and measured 0.31x - four times denser
// arithmetic behind a four times worse memory path. llama.cpp's MMQ kernels
// stage the tile for exactly this reason.
groupshared uint dp4a_tile_quants[LIZARD_DP4A_TILE * 64];
groupshared float dp4a_tile_scale[LIZARD_DP4A_TILE * 8];
groupshared float dp4a_tile_sum[LIZARD_DP4A_TILE * 8];

[numthreads(LIZARD_DP4A_GROUP, 1, 1)]
void main_q4k_q8_1_batched(uint3 group_id : SV_GroupID, uint thread_index : SV_GroupIndex) {
  const uint group_row_base = group_id.x * (k_dp4a_group * k_dp4a_rows);
  const uint column_base = group_id.y * k_dp4a_tile;
  // No early return: every thread has to reach the barriers below, so an
  // out-of-range row computes and is discarded at store time instead.
  //
  // Strided by 64 like the float kernel: blocks are column-major, so the wave's
  // 64 lanes must read 64 adjacent blocks for each load round to stay one
  // contiguous stream.
  uint rows[k_dp4a_rows];
  uint safe_rows[k_dp4a_rows];
  [unroll]
  for (uint r_init = 0; r_init < k_dp4a_rows; r_init += 1u) {
    rows[r_init] = group_row_base + r_init * k_dp4a_group + thread_index;
    safe_rows[r_init] = min(rows[r_init], row_count - 1u);
  }
  const uint blocks_per_row = row_width >> 8u;
  const uint scale_words_base = row_width / 4u;
  const uint activation_stride = scale_words_base + (row_width / 32u) * 2u;

  float totals[k_dp4a_rows * k_dp4a_tile];
  [unroll]
  for (uint init = 0; init < k_dp4a_rows * k_dp4a_tile; init += 1u) {
    totals[init] = 0.0f;
  }

  [loop]
  for (uint block = 0; block < blocks_per_row; block += 1u) {
    // ---- stage this block's activations for every column in the tile ----
    GroupMemoryBarrierWithGroupSync();
    {
      const uint quant_slots = k_dp4a_tile * 64u;
      [loop]
      for (uint slot = thread_index; slot < quant_slots; slot += k_dp4a_group) {
        const uint column_index = slot / 64u;
        const uint within = slot % 64u;          // sub-block j * 8 + word k
        const uint j = within >> 3u;
        const uint k = within & 7u;
        const uint column = min(column_base + column_index, batch_count - 1u);
        const uint column_words = input_stride + column * activation_stride;
        dp4a_tile_quants[slot] =
          packed_activations[column_words + (block * 8u + j) * 8u + k];
      }
      const uint scale_slots = k_dp4a_tile * 8u;
      [loop]
      for (uint slot = thread_index; slot < scale_slots; slot += k_dp4a_group) {
        const uint column_index = slot >> 3u;
        const uint j = slot & 7u;
        const uint column = min(column_base + column_index, batch_count - 1u);
        const uint column_words = input_stride + column * activation_stride;
        const uint scale_word =
          column_words + scale_words_base + (block * 8u + j) * 2u;
        dp4a_tile_scale[slot] = asfloat(packed_activations[scale_word]);
        dp4a_tile_sum[slot] = asfloat(packed_activations[scale_word + 1u]);
      }
    }
    GroupMemoryBarrierWithGroupSync();

    [unroll]
    for (uint r = 0; r < k_dp4a_rows; r += 1u) {
      const uint block_byte = (block * row_count + safe_rows[r]) * 144u;
      const uint d_words = weights.Load(block_byte);
      const float d = f16tof32(d_words & 0xFFFFu);
      const float dmin = f16tof32(d_words >> 16u);
      const uint scales_byte_base = block_byte + 4u;
      const uint qs_base = block_byte + 16u;

      [unroll]
      for (uint j = 0; j < 8u; j += 1u) {
        const float sub_scale = d * (float)dp4a_scale_k4(j, scales_byte_base);
        const float sub_min = dmin * (float)dp4a_min_k4(j, scales_byte_base);
        const uint chunk_bytes = qs_base + (j >> 1u) * 32u;
        const bool high_nibbles = (j & 1u) != 0u;

        // Loaded once per (row, sub-block), reused across every column below.
        uint nibble_words[8];
        [unroll]
        for (uint k = 0; k < 8u; k += 1u) {
          const uint qs_word = weights.Load(chunk_bytes + k * 4u);
          nibble_words[k] = high_nibbles
            ? ((qs_word >> 4u) & 0x0F0F0F0Fu)
            : (qs_word & 0x0F0F0F0Fu);
        }

        [unroll]
        for (uint c = 0; c < k_dp4a_tile; c += 1u) {
          const uint quant_slot = c * 64u + j * 8u;
          const uint scale_slot = c * 8u + j;
          int int_dot = 0;
          [unroll]
          for (uint k2 = 0; k2 < 8u; k2 += 1u) {
            int_dot = dot4add_i8packed(
              nibble_words[k2], dp4a_tile_quants[quant_slot + k2], int_dot);
          }
          totals[r * k_dp4a_tile + c] +=
            sub_scale * dp4a_tile_scale[scale_slot] * (float)int_dot
            - sub_min * dp4a_tile_sum[scale_slot];
        }
      }
    }
  }

  [unroll]
  for (uint rs = 0; rs < k_dp4a_rows; rs += 1u) {
    if (rows[rs] >= row_count) {
      continue;
    }
    [unroll]
    for (uint store = 0; store < k_dp4a_tile; store += 1u) {
      const uint column = column_base + store;
      if (column < batch_count) {
        matvec_outputs[output_base + column * row_count + rows[rs]] =
          totals[rs * k_dp4a_tile + store];
      }
    }
  }
}


// ---------------------------------------------------------------------------
// ABLATION TWIN of main_q4k_q8_1_batched. Diagnostic only - its output is
// meaningless, exactly like main_q4k_loadonly in the FXC source.
//
// Identical weight loads, identical nibble extraction, identical count of
// dot4add_i8packed instructions, and NO activation path at all: no staging, no
// barriers, no groupshared reads. The activation operand comes from a register.
//
// The question it answers: the float and integer kernels reach the same
// throughput (223 vs 225 GFLOP/s) despite the integer one doing four times the
// arithmetic per instruction, so neither is arithmetic-bound, and both read their
// activations from groupshared. If removing that path collapses the time, LDS
// throughput is the shared limiter. If it does not, the limiter is the weight
// loads and nothing about the activation side can fix it.
// ---------------------------------------------------------------------------
[numthreads(64, 1, 1)]
void main_q4k_q8_1_batched_nolds(uint3 group_id : SV_GroupID, uint thread_index : SV_GroupIndex) {
  const uint row = group_id.x * 64u + thread_index;
  const uint safe_row = min(row, row_count - 1u);
  const uint blocks_per_row = row_width >> 8u;

  // Stands in for every activation word. Derived from thread_index so the
  // compiler cannot fold the dots away as constant.
  const uint fake_activation = 0x01010101u * (thread_index & 3u) + 0x01020304u;
  const float fake_scale = 1.0f / 128.0f;

  float totals[k_dp4a_tile];
  [unroll]
  for (uint init = 0; init < k_dp4a_tile; init += 1u) {
    totals[init] = 0.0f;
  }

  [loop]
  for (uint block = 0; block < blocks_per_row; block += 1u) {
    const uint block_byte = (block * row_count + safe_row) * 144u;
    const uint d_words = weights.Load(block_byte);
    const float d = f16tof32(d_words & 0xFFFFu);
    const float dmin = f16tof32(d_words >> 16u);
    const uint scales_byte_base = block_byte + 4u;
    const uint qs_base = block_byte + 16u;

    [unroll]
    for (uint j = 0; j < 8u; j += 1u) {
      const float sub_scale = d * (float)dp4a_scale_k4(j, scales_byte_base);
      const float sub_min = dmin * (float)dp4a_min_k4(j, scales_byte_base);
      const uint chunk_bytes = qs_base + (j >> 1u) * 32u;
      const bool high_nibbles = (j & 1u) != 0u;

      uint nibble_words[8];
      [unroll]
      for (uint k = 0; k < 8u; k += 1u) {
        const uint qs_word = weights.Load(chunk_bytes + k * 4u);
        nibble_words[k] = high_nibbles
          ? ((qs_word >> 4u) & 0x0F0F0F0Fu)
          : (qs_word & 0x0F0F0F0Fu);
      }

      [unroll]
      for (uint c = 0; c < k_dp4a_tile; c += 1u) {
        int int_dot = 0;
        [unroll]
        for (uint k2 = 0; k2 < 8u; k2 += 1u) {
          int_dot = dot4add_i8packed(nibble_words[k2], fake_activation, int_dot);
        }
        totals[c] += sub_scale * fake_scale * (float)int_dot - sub_min * fake_scale;
      }
    }
  }

  if (row < row_count) {
    [unroll]
    for (uint store = 0; store < k_dp4a_tile; store += 1u) {
      const uint column = group_id.y * k_dp4a_tile + store;
      if (column < batch_count) {
        matvec_outputs[output_base + column * row_count + row] = totals[store];
      }
    }
  }
}


// ---------------------------------------------------------------------------
// Cooperative one-workgroup-per-row Q4_K GEMV using WaveActiveSum (SM 6.0+)
// for the cross-lane reduction instead of a groupshared tree or a barrier per
// reduction step. This is the DXC counterpart to the FXC coop experiment
// (main_q4k_gemv in the FXC source), which measured 2x SLOWER because its
// per-block-barrier + serial 64-element reduction dominated: here the
// reduction is a single hardware wave op per wave plus ONE barrier for the
// (small) cross-wave combine, so the coalesced-read win is not swallowed by
// reduction overhead.
//
// t1/u0 reuse the plain-float bindings declared above (quantize_inputs,
// matvec_outputs) since this kernel takes an unquantized float input vector,
// unlike main_q4k_q8_1_b1 which reads the Q8_1 activation buffer.
// ---------------------------------------------------------------------------
groupshared float wave_gemv_sums[64];

[numthreads(64, 1, 1)]
void main_q4k_gemv_wave(uint3 group_id : SV_GroupID, uint lane : SV_GroupIndex) {
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
      // Lane l reads quant bytes qi = l and qi = l + 64 (contiguous across the
      // 64 lanes -> coalesced 64-byte transactions per half).
      [loop]
      for (uint half_index = 0; half_index < 2u; half_index += 1u) {
        const uint quant_index = lane + half_index * 64u;
        const uint chunk_id = quant_index >> 5u;
        const uint item = quant_index & 31u;
        const uint quant_byte = dp4a_buffer_byte(block_byte + 16u + quant_index);
        const uint low_element = chunk_id * 64u + item;
        const float scale_d_low = d * (float)dp4a_scale_k4(chunk_id * 2u, scales_byte_base);
        const float scale_m_low = dmin * (float)dp4a_min_k4(chunk_id * 2u, scales_byte_base);
        const float scale_d_high = d * (float)dp4a_scale_k4(chunk_id * 2u + 1u, scales_byte_base);
        const float scale_m_high = dmin * (float)dp4a_min_k4(chunk_id * 2u + 1u, scales_byte_base);
        const float weight_low = scale_d_low * (float)(quant_byte & 0x0Fu) - scale_m_low;
        const float weight_high = scale_d_high * (float)(quant_byte >> 4u) - scale_m_high;
        partial += weight_low * quantize_inputs[input_stride + input_base + low_element]
          + weight_high * quantize_inputs[input_stride + input_base + low_element + 32u];
      }
    }
  }
  // Hardware reduction within each wave (no barrier), then one barrier to
  // combine the handful of per-wave totals (wave count = 64 / lane count,
  // e.g. 2 waves at width 32, 4 at width 16 - always small on real hardware).
  const float wave_total = WaveActiveSum(partial);
  const uint lane_count = WaveGetLaneCount();
  const uint wave_index = lane / lane_count;
  if (WaveIsFirstLane()) {
    wave_gemv_sums[wave_index] = wave_total;
  }
  GroupMemoryBarrierWithGroupSync();
  if (lane == 0u && row < row_count) {
    const uint wave_count = 64u / lane_count;
    float total = 0.0f;
    [loop]
    for (uint w = 0; w < wave_count; w += 1u) {
      total += wave_gemv_sums[w];
    }
    matvec_outputs[output_base + row] = total;
  }
}

// ---------------------------------------------------------------------------
// Q6_K x Q8_1 batched GEMM with packed integer dot products.
//
// Q6_K blocks are 212 bytes here: ql[128] at 0, qh[64] at 128, sixteen int8
// sub-block scales at 192, d (f16) at 208. 256 elements, 16 sub-blocks of 16,
// w = d * sc[sub] * (q - 32) with q in [0, 63].
//
// Element order is not byte order. Within a 128-element chunk:
//   elements chunk+  0.. 31  <- ql[ 0..31] low nibble,  qh[0..31] bits 0-1
//   elements chunk+ 32.. 63  <- ql[32..63] low nibble,  qh[0..31] bits 2-3
//   elements chunk+ 64.. 95  <- ql[ 0..31] high nibble, qh[0..31] bits 4-5
//   elements chunk+ 96..127  <- ql[32..63] high nibble, qh[0..31] bits 6-7
// so sub-block j needs 16 CONSECUTIVE ql bytes and 16 consecutive qh bytes.
// Iterating per (chunk, sub_half) loads 4 uints of ql-low, 4 of ql-high and 4
// of qh and produces four sub-blocks from them - 12 loads per 64 elements,
// against the Q4_K kernel's 16. The six-bit format costs FEWER weight loads
// here than four-bit does, the opposite of the float kernels where Q6_K's
// per-element unpack costs about twice Q4_K's.
//
// The -32 is handled with a per-16-element activation sum rather than by
// biasing the quants, because a per-byte subtract of 32 from values in [0,63]
// borrows across lanes. sum(w*a) = d*sc*dx*(int_dot - 32*sum(x_int)), and
// sum(x_int) depends only on the column and sub-block - never on the weight
// row - so it is computed once per group into LDS and the inner loop stays
// pure dp4a. Computing it per row instead would double the dp4a count.
// ---------------------------------------------------------------------------

// Sign-extended int8 scale from the 16-byte scale block held in a uint4.
int dp4a_signed_scale16(uint4 words, uint index) {
  const uint word = index < 4u ? words.x : index < 8u ? words.y : index < 12u ? words.z : words.w;
  const uint byte_value = (word >> ((index & 3u) * 8u)) & 0xFFu;
  return (int)(byte_value << 24u) >> 24;
}

groupshared int dp4a_tile_sum16[LIZARD_DP4A_TILE * 16];

[numthreads(LIZARD_DP4A_GROUP, 1, 1)]
void main_q6k_q8_1_batched(uint3 group_id : SV_GroupID, uint thread_index : SV_GroupIndex) {
  const uint group_row_base = group_id.x * (k_dp4a_group * k_dp4a_rows);
  const uint column_base = group_id.y * k_dp4a_tile;

  uint rows[k_dp4a_rows];
  uint safe_rows[k_dp4a_rows];
  [unroll]
  for (uint r_init = 0; r_init < k_dp4a_rows; r_init += 1u) {
    rows[r_init] = group_row_base + r_init * k_dp4a_group + thread_index;
    safe_rows[r_init] = min(rows[r_init], row_count - 1u);
  }
  const uint blocks_per_row = row_width >> 8u;
  const uint scale_words_base = row_width / 4u;
  const uint activation_stride = scale_words_base + (row_width / 32u) * 2u;

  float totals[k_dp4a_rows * k_dp4a_tile];
  [unroll]
  for (uint init = 0; init < k_dp4a_rows * k_dp4a_tile; init += 1u) {
    totals[init] = 0.0f;
  }

  [loop]
  for (uint block = 0; block < blocks_per_row; block += 1u) {
    GroupMemoryBarrierWithGroupSync();
    {
      const uint quant_slots = k_dp4a_tile * 64u;
      [loop]
      for (uint slot = thread_index; slot < quant_slots; slot += k_dp4a_group) {
        const uint column_index = slot / 64u;
        const uint within = slot % 64u;
        const uint column = min(column_base + column_index, batch_count - 1u);
        const uint column_words = input_stride + column * activation_stride;
        dp4a_tile_quants[slot] = packed_activations[column_words + block * 64u + within];
      }
      const uint scale_slots = k_dp4a_tile * 8u;
      [loop]
      for (uint slot = thread_index; slot < scale_slots; slot += k_dp4a_group) {
        const uint column_index = slot >> 3u;
        const uint j = slot & 7u;
        const uint column = min(column_base + column_index, batch_count - 1u);
        const uint column_words = input_stride + column * activation_stride;
        const uint scale_word = column_words + scale_words_base + (block * 8u + j) * 2u;
        dp4a_tile_scale[slot] = asfloat(packed_activations[scale_word]);
      }
    }
    GroupMemoryBarrierWithGroupSync();
    {
      // Per-16-element activation sums: once per (column, sub-block) for the
      // whole group, reused by every row it covers.
      const uint sum_slots = k_dp4a_tile * 16u;
      [loop]
      for (uint slot = thread_index; slot < sum_slots; slot += k_dp4a_group) {
        const uint column_index = slot >> 4u;
        const uint sub = slot & 15u;
        int lane_sum = 0;
        [unroll]
        for (uint t = 0; t < 4u; t += 1u) {
          lane_sum = dot4add_i8packed(
            0x01010101u, dp4a_tile_quants[column_index * 64u + sub * 4u + t], lane_sum);
        }
        dp4a_tile_sum16[slot] = lane_sum;
      }
    }
    GroupMemoryBarrierWithGroupSync();

    [unroll]
    for (uint r = 0; r < k_dp4a_rows; r += 1u) {
      const uint block_byte = (block * row_count + safe_rows[r]) * 212u;
      const float d = f16tof32(weights.Load(block_byte + 208u) & 0xFFFFu);
      const uint4 scale_words = weights.Load4(block_byte + 192u);

      [unroll]
      for (uint half_index = 0; half_index < 2u; half_index += 1u) {
        const uint ql_base = block_byte + half_index * 64u;
        const uint qh_base = block_byte + 128u + half_index * 32u;
        [unroll]
        for (uint sub_half = 0; sub_half < 2u; sub_half += 1u) {
          const uint4 ql_low4 = weights.Load4(ql_base + sub_half * 16u);
          const uint4 ql_high4 = weights.Load4(ql_base + 32u + sub_half * 16u);
          const uint4 qh_word4 = weights.Load4(qh_base + sub_half * 16u);
          [unroll]
          for (uint group_index = 0; group_index < 4u; group_index += 1u) {
            const uint j = half_index * 8u + group_index * 2u + sub_half;
            const float sub_scale = d * (float)dp4a_signed_scale16(scale_words, j);
            const uint nibble_shift = group_index >= 2u ? 4u : 0u;
            const uint qh_shift = group_index * 2u;
            const uint4 ql_src = (group_index & 1u) == 0u ? ql_low4 : ql_high4;
            uint packed_q[4];
            [unroll]
            for (uint t = 0; t < 4u; t += 1u) {
              const uint ql_word = t == 0u ? ql_src.x : t == 1u ? ql_src.y : t == 2u ? ql_src.z : ql_src.w;
              const uint qh_word = t == 0u ? qh_word4.x : t == 1u ? qh_word4.y : t == 2u ? qh_word4.z : qh_word4.w;
              packed_q[t] = ((ql_word >> nibble_shift) & 0x0F0F0F0Fu)
                | (((qh_word >> qh_shift) & 0x03030303u) << 4u);
            }
            [unroll]
            for (uint c = 0; c < k_dp4a_tile; c += 1u) {
              int int_dot = 0;
              [unroll]
              for (uint t2 = 0; t2 < 4u; t2 += 1u) {
                int_dot = dot4add_i8packed(
                  packed_q[t2], dp4a_tile_quants[c * 64u + j * 4u + t2], int_dot);
              }
              totals[r * k_dp4a_tile + c] +=
                sub_scale * dp4a_tile_scale[c * 8u + (j >> 1u)]
                * (float)(int_dot - 32 * dp4a_tile_sum16[c * 16u + j]);
            }
          }
        }
      }
    }
    GroupMemoryBarrierWithGroupSync();
  }

  [unroll]
  for (uint rs = 0; rs < k_dp4a_rows; rs += 1u) {
    if (rows[rs] >= row_count) {
      continue;
    }
    [unroll]
    for (uint store = 0; store < k_dp4a_tile; store += 1u) {
      const uint column = column_base + store;
      if (column < batch_count) {
        matvec_outputs[output_base + column * row_count + rows[rs]] =
          totals[rs * k_dp4a_tile + store];
      }
    }
  }
}
