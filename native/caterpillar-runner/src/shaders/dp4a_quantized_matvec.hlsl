// Caterpillar Phase 5 packed-integer-dot kernels.
//
// These entry points remain compatible with the extended Phase 4 root signature:
//   b0: 16 root constants
//   t0: resident, column-major quantized weights
//   t1: compatibility SRV (unused here)
//   t2: replay dynamics (unused here)
//   u0: activation arena, Q8_1 scratch, and outputs
//
// Q8_1 scratch stores four signed int8 values per dword followed by two
// full-float dwords per 32-element block (d and s=d*sum(q)). Full floats are
// deliberate: DXIL f32tof16 rounds toward zero and biases Q4_K's minimum
// correction. HLSL round() is round-to-nearest-even; the CPU oracle must use
// nearbyint rather than std::round.

cbuffer Params : register(b0) {
  uint4 p0;
  uint4 p1;
  uint4 p2;
  uint4 p3;
};

ByteAddressBuffer weights : register(t0);
ByteAddressBuffer inputs : register(t1);
ByteAddressBuffer session : register(t2);
RWByteAddressBuffer outputs : register(u0);

groupshared float quantize_wave_maxima[4];
groupshared float quantize_wave_sums[4];

// p0.y = width, p0.z = input float offset, p1.x = scratch dword offset.
[numthreads(32, 1, 1)]
void cs_quantize_q8_1(uint3 group_id : SV_GroupID, uint lane : SV_GroupIndex) {
  const uint width = p0.y;
  const uint element = group_id.x * 32u + lane;
  const uint safe_element = min(element, width - 1u);
  const float loaded = asfloat(outputs.Load((p0.z + safe_element) * 4u));
  const float value = element < width ? loaded : 0.0f;
  const uint wave_lane = WaveGetLaneIndex();
  const uint wave_size = WaveGetLaneCount();
  const uint wave = lane / wave_size;
  const uint wave_count = 32u / wave_size;
  const float wave_maximum = WaveActiveMax(abs(value));
  if (wave_lane == 0u) {
    quantize_wave_maxima[wave] = wave_maximum;
  }
  GroupMemoryBarrierWithGroupSync();
  float absolute_max = 0.0f;
  [unroll]
  for (uint wave_index = 0u; wave_index < 4u; wave_index += 1u) {
    if (wave_index < wave_count) absolute_max = max(absolute_max, quantize_wave_maxima[wave_index]);
  }
  const float d = absolute_max / 127.0f;
  const float inverse_d = d > 0.0f ? 1.0f / d : 0.0f;
  const int quantized = clamp((int)round(value * inverse_d), -127, 127);
  if ((wave_lane & 3u) == 0u && element < width) {
    const uint packed = (uint)(quantized & 0xFF)
      | ((uint)(WaveReadLaneAt(quantized, wave_lane + 1u) & 0xFF) << 8u)
      | ((uint)(WaveReadLaneAt(quantized, wave_lane + 2u) & 0xFF) << 16u)
      | ((uint)(WaveReadLaneAt(quantized, wave_lane + 3u) & 0xFF) << 24u);
    outputs.Store((p1.x + element / 4u) * 4u, packed);
  }
  const float wave_sum = WaveActiveSum((float)quantized);
  if (wave_lane == 0u) {
    quantize_wave_sums[wave] = wave_sum;
  }
  GroupMemoryBarrierWithGroupSync();
  if (lane == 0u && element < width) {
    float quantized_sum = 0.0f;
    [unroll]
    for (uint wave_index = 0u; wave_index < 4u; wave_index += 1u) {
      if (wave_index < wave_count) quantized_sum += quantize_wave_sums[wave_index];
    }
    const uint scale_base = p1.x + width / 4u + group_id.x * 2u;
    outputs.Store(scale_base * 4u, asuint(d));
    outputs.Store((scale_base + 1u) * 4u, asuint(d * quantized_sum));
  }
}

uint weight_byte(uint byte_address) {
  return (weights.Load(byte_address & ~3u) >> ((byte_address & 3u) * 8u)) & 0xFFu;
}

uint q4k_scale(uint group, uint scales_base) {
  return group < 4u
    ? (weight_byte(scales_base + group) & 63u)
    : ((weight_byte(scales_base + group + 4u) & 0x0Fu)
      | ((weight_byte(scales_base + group - 4u) >> 6u) << 4u));
}

uint q4k_min(uint group, uint scales_base) {
  return group < 4u
    ? (weight_byte(scales_base + group + 4u) & 63u)
    : ((weight_byte(scales_base + group + 4u) >> 4u)
      | ((weight_byte(scales_base + group) >> 6u) << 4u));
}

uint scale_byte_words(uint3 words, uint index) {
  const uint word = index < 4u ? words.x : (index < 8u ? words.y : words.z);
  return (word >> ((index & 3u) * 8u)) & 0xFFu;
}

uint q4k_scale_words(uint group, uint3 words) {
  return group < 4u
    ? (scale_byte_words(words, group) & 63u)
    : ((scale_byte_words(words, group + 4u) & 0x0Fu)
      | ((scale_byte_words(words, group - 4u) >> 6u) << 4u));
}

uint q4k_min_words(uint group, uint3 words) {
  return group < 4u
    ? (scale_byte_words(words, group + 4u) & 63u)
    : ((scale_byte_words(words, group + 4u) >> 4u)
      | ((scale_byte_words(words, group) >> 6u) << 4u));
}

// Q4_K x Q8_1. p0.x=rows, p0.y=columns, p0.w=output float offset,
// p1.x=scratch dword offset. Weights are raw 144-byte blocks staged as
// (block * rows + row), matching the existing float kernel.
groupshared uint q4k_q8_tile[240]; // three blocks: 64 packed quants + 16 d/s words each

[numthreads(128, 1, 1)]
void cs_matvec_q4k_q8_1(uint3 group_id : SV_GroupID, uint lane : SV_GroupIndex) {
  const uint row = group_id.x * 128u + lane;
  const uint safe_row = min(row, p0.x - 1u);
  const uint blocks = p0.y >> 8u;
  const uint scale_words_base = p0.y / 4u;
  float total = 0.0f;

  [loop]
  for (uint block = 0u; block < blocks; block += 3u) {
    const uint pair_count = min(3u, blocks - block);
    [unroll]
    for (uint sub = 0u; sub < 3u; sub += 1u) {
      if (sub < pair_count) {
        const uint tile_base = sub * 80u;
        const uint source_block = block + sub;
        if (lane < 64u) {
          q4k_q8_tile[tile_base + lane] = outputs.Load(
            (p1.x + source_block * 64u + lane) * 4u);
        } else if (lane < 80u) {
          q4k_q8_tile[tile_base + lane] = outputs.Load(
            (p1.x + scale_words_base + source_block * 16u + lane - 64u) * 4u);
        }
      }
    }
    GroupMemoryBarrierWithGroupSync();
    [unroll]
    for (uint sub = 0u; sub < 3u; sub += 1u) {
      if (sub < pair_count) {
        const uint tile_base = sub * 80u;
        const uint source_block = block + sub;
        const uint block_byte = (source_block * p0.x + safe_row) * 144u;
        const uint d_words = weights.Load(block_byte);
        const float d = f16tof32(d_words & 0xFFFFu);
        const float dmin = f16tof32(d_words >> 16u);
        const uint3 scale_words = weights.Load3(block_byte + 4u);
        const uint qs_base = block_byte + 16u;
        [unroll]
        for (uint group = 0u; group < 8u; group += 1u) {
          const float dx = asfloat(q4k_q8_tile[tile_base + 64u + group * 2u]);
          const float sx = asfloat(q4k_q8_tile[tile_base + 64u + group * 2u + 1u]);
          const float sub_scale = d * (float)q4k_scale_words(group, scale_words);
          const float sub_min = dmin * (float)q4k_min_words(group, scale_words);
          const uint chunk_bytes = qs_base + (group >> 1u) * 32u;
          const bool high_nibbles = (group & 1u) != 0u;
          const uint4 packed_low = weights.Load4(chunk_bytes);
          const uint4 packed_high = weights.Load4(chunk_bytes + 16u);
          int integer_dot = 0;
          [unroll]
          for (uint word_index = 0u; word_index < 8u; word_index += 1u) {
            const uint4 selected = word_index < 4u ? packed_low : packed_high;
            const uint selected_index = word_index & 3u;
            const uint packed_weight = selected_index == 0u ? selected.x
              : selected_index == 1u ? selected.y
              : selected_index == 2u ? selected.z
              : selected.w;
            const uint nibbles = high_nibbles
              ? ((packed_weight >> 4u) & 0x0F0F0F0Fu)
              : (packed_weight & 0x0F0F0F0Fu);
            const uint packed_activation = q4k_q8_tile[tile_base + group * 8u + word_index];
            integer_dot = dot4add_i8packed(nibbles, packed_activation, integer_dot);
          }
          total += sub_scale * dx * (float)integer_dot - sub_min * sx;
        }
      }
    }
    GroupMemoryBarrierWithGroupSync();
  }
  if (row < p0.x) outputs.Store((p0.w + row) * 4u, asuint(total));
}

uint pack_i8x4(int x0, int x1, int x2, int x3) {
  return (uint)(x0 & 255)
    | ((uint)(x1 & 255) << 8u)
    | ((uint)(x2 & 255) << 16u)
    | ((uint)(x3 & 255) << 24u);
}

int signed_weight_byte(uint value) {
  return (int)(value << 24u) >> 24;
}

// Q5_K uses Q4_K's scale/min fields plus a high-bit plane.
[numthreads(64, 1, 1)]
void cs_matvec_q5k_q8_1(uint3 group_id : SV_GroupID, uint lane : SV_GroupIndex) {
  const uint row = group_id.x * 64u + lane;
  if (row >= p0.x) return;
  const uint blocks = p0.y >> 8u;
  const uint scale_words_base = p0.y / 4u;
  float total = 0.0f;
  [loop]
  for (uint block = 0u; block < blocks; block += 1u) {
    const uint block_byte = (block * p0.x + row) * 176u;
    const uint d_words = weights.Load(block_byte);
    const float d = f16tof32(d_words & 0xFFFFu);
    const float dmin = f16tof32(d_words >> 16u);
    const uint scales_base = block_byte + 4u;
    const uint3 scale_words = weights.Load3(scales_base);
    [unroll]
    for (uint group = 0u; group < 8u; group += 1u) {
      const uint activation_block = block * 8u + group;
      const uint activation_scale = p1.x + scale_words_base + activation_block * 2u;
      const float dx = asfloat(outputs.Load(activation_scale * 4u));
      const float sx = asfloat(outputs.Load((activation_scale + 1u) * 4u));
      const uint quant_base = block_byte + 48u + (group >> 1u) * 32u;
      const bool high_nibble = (group & 1u) != 0u;
      int integer_dot = 0;
      [unroll]
      for (uint word_index = 0u; word_index < 8u; word_index += 1u) {
        const uint packed = weights.Load(quant_base + word_index * 4u);
        const uint high = weights.Load(block_byte + 16u + word_index * 4u);
        int q[4];
        [unroll]
        for (uint item = 0u; item < 4u; item += 1u) {
          const uint raw = (packed >> (item * 8u)) & 0xFFu;
          const uint nibble = high_nibble ? (raw >> 4u) : (raw & 0x0Fu);
          const uint high_bit = (((high >> (item * 8u)) & 0xFFu) >> group) & 1u;
          q[item] = (int)(nibble | (high_bit << 4u));
        }
        const uint packed_q = pack_i8x4(q[0], q[1], q[2], q[3]);
        const uint packed_x = outputs.Load((p1.x + activation_block * 8u + word_index) * 4u);
        integer_dot = dot4add_i8packed(packed_q, packed_x, integer_dot);
      }
      total += d * (float)q4k_scale_words(group, scale_words) * dx * (float)integer_dot
        - dmin * (float)q4k_min_words(group, scale_words) * sx;
    }
  }
  outputs.Store((p0.w + row) * 4u, asuint(total));
}

// Q8_0 staged blocks contain a full-float weight scale followed by 32 int8.
[numthreads(64, 1, 1)]
void cs_matvec_q8_0_q8_1(uint3 group_id : SV_GroupID, uint lane : SV_GroupIndex) {
  const uint row = group_id.x * 64u + lane;
  if (row >= p0.x) return;
  const uint blocks = p0.y >> 5u;
  const uint scale_words_base = p0.y / 4u;
  float total = 0.0f;
  [loop]
  for (uint block = 0u; block < blocks; block += 1u) {
    const uint block_byte = (block * p0.x + row) * 36u;
    const float d = asfloat(weights.Load(block_byte));
    const uint activation_scale = p1.x + scale_words_base + block * 2u;
    const float dx = asfloat(outputs.Load(activation_scale * 4u));
    int integer_dot = 0;
    [unroll]
    for (uint word_index = 0u; word_index < 8u; word_index += 1u) {
      integer_dot = dot4add_i8packed(
        weights.Load(block_byte + 4u + word_index * 4u),
        outputs.Load((p1.x + block * 8u + word_index) * 4u),
        integer_dot);
    }
    total += d * dx * (float)integer_dot;
  }
  outputs.Store((p0.w + row) * 4u, asuint(total));
}

// Q4_0 raw blocks contain fp16 d plus 16 bytes whose low/high nibbles map
// to the first/second 16 values. Subtracting 8 before packing makes them
// signed int8 operands for dot4add_i8packed.
[numthreads(64, 1, 1)]
void cs_matvec_q4_0_q8_1(uint3 group_id : SV_GroupID, uint lane : SV_GroupIndex) {
  const uint row = group_id.x * 64u + lane;
  if (row >= p0.x) return;
  const uint blocks = p0.y >> 5u;
  const uint scale_words_base = p0.y / 4u;
  float total = 0.0f;
  [loop]
  for (uint block = 0u; block < blocks; block += 1u) {
    const uint block_byte = (block * p0.x + row) * 18u;
    const uint d_word = weights.Load(block_byte & ~3u);
    const uint d_bits = (block_byte & 2u) != 0u ? (d_word >> 16u) : (d_word & 0xFFFFu);
    const float d = f16tof32(d_bits);
    const uint activation_scale = p1.x + scale_words_base + block * 2u;
    const float dx = asfloat(outputs.Load(activation_scale * 4u));
    int integer_dot = 0;
    [unroll]
    for (uint half = 0u; half < 2u; half += 1u) {
      [unroll]
      for (uint word_index = 0u; word_index < 4u; word_index += 1u) {
        const uint byte_base = block_byte + 2u + word_index * 4u;
        int q[4];
        [unroll]
        for (uint item = 0u; item < 4u; item += 1u) {
          const uint packed = weight_byte(byte_base + item);
          q[item] = (int)(half == 0u ? (packed & 0x0Fu) : (packed >> 4u)) - 8;
        }
        const uint packed_q = pack_i8x4(q[0], q[1], q[2], q[3]);
        const uint activation_word = block * 8u + half * 4u + word_index;
        integer_dot = dot4add_i8packed(
          packed_q, outputs.Load((p1.x + activation_word) * 4u), integer_dot);
      }
    }
    total += d * dx * (float)integer_dot;
  }
  outputs.Store((p0.w + row) * 4u, asuint(total));
}

int q6_value(uint block_byte, uint element) {
  const uint half_index = element >> 7u;
  const uint local = element & 127u;
  const uint quarter = local >> 5u;
  const uint item = local & 31u;
  const uint low_address = block_byte + half_index * 64u
    + ((quarter & 1u) != 0u ? 32u : 0u) + item;
  const uint low_byte = weight_byte(low_address);
  const uint low_nibble = quarter >= 2u ? (low_byte >> 4u) : (low_byte & 0x0Fu);
  const uint high_byte = weight_byte(block_byte + 128u + half_index * 32u + item);
  return (int)(low_nibble | (((high_byte >> (quarter * 2u)) & 3u) << 4u)) - 32;
}

// Q6_K: 16 signed scales per 256-value superblock and signed 6-bit quants.
[numthreads(64, 1, 1)]
void cs_matvec_q6k_q8_1(uint3 group_id : SV_GroupID, uint lane : SV_GroupIndex) {
  const uint row = group_id.x * 64u + lane;
  if (row >= p0.x) return;
  const uint blocks = p0.y >> 8u;
  const uint scale_words_base = p0.y / 4u;
  float total = 0.0f;
  [loop]
  for (uint block = 0u; block < blocks; block += 1u) {
    const uint block_byte = (block * p0.x + row) * 212u;
    const float d = asfloat(weights.Load(block_byte + 208u));
    [unroll]
    for (uint activation_group = 0u; activation_group < 8u; activation_group += 1u) {
      const uint activation_block = block * 8u + activation_group;
      const uint activation_scale = p1.x + scale_words_base + activation_block * 2u;
      const float dx = asfloat(outputs.Load(activation_scale * 4u));
      [unroll]
      for (uint half = 0u; half < 2u; half += 1u) {
        const uint element_base = activation_group * 32u + half * 16u;
        int integer_dot = 0;
        [unroll]
        for (uint word_index = 0u; word_index < 4u; word_index += 1u) {
          const uint element = element_base + word_index * 4u;
          const uint packed_q = pack_i8x4(
            q6_value(block_byte, element), q6_value(block_byte, element + 1u),
            q6_value(block_byte, element + 2u), q6_value(block_byte, element + 3u));
          const uint activation_word = activation_block * 8u + half * 4u + word_index;
          integer_dot = dot4add_i8packed(
            packed_q, outputs.Load((p1.x + activation_word) * 4u), integer_dot);
        }
        const int scale = signed_weight_byte(weight_byte(
          block_byte + 192u + activation_group * 2u + half));
        total += d * (float)scale * dx * (float)integer_dot;
      }
    }
  }
  outputs.Store((p0.w + row) * 4u, asuint(total));
}
