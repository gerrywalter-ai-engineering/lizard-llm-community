#pragma once

#include <cstddef>
#include <span>

namespace cat {

// Exact online-softmax attention. Keys are streamed in fixed-size tiles that
// remain L1/L2 friendly on CPU; the running max/sum keeps memory O(head_dim)
// and preserves key order, so tiling does not change the reference result.
void streaming_attention_f32(
    std::span<const float> query,
    std::span<const float> current_keys,
    std::span<const float> current_values,
    std::span<const float> past_keys,
    std::span<const float> past_values,
    std::span<float> output,
    std::size_t tokens,
    std::size_t heads,
    std::size_t kv_heads,
    std::size_t head_dim,
    std::size_t past_tokens,
    std::size_t window,
    float scale,
    bool independent_rows = false
);

}  // namespace cat
