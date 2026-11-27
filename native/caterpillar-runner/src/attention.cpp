#include "attention.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace cat {

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
    bool independent_rows
) {
    if (heads == 0 || kv_heads == 0 || head_dim == 0 || heads % kv_heads != 0) {
        throw std::invalid_argument("streaming attention has an invalid head shape");
    }
    if (query.size() < tokens * heads * head_dim
        || current_keys.size() < tokens * kv_heads * head_dim
        || current_values.size() < tokens * kv_heads * head_dim
        || past_keys.size() < (independent_rows ? tokens : 1) * past_tokens * kv_heads * head_dim
        || past_values.size() < (independent_rows ? tokens : 1) * past_tokens * kv_heads * head_dim
        || output.size() < tokens * heads * head_dim) {
        throw std::invalid_argument("streaming attention binding is smaller than its live shape");
    }
    constexpr std::size_t key_tile = 32;
    const auto grouped_queries = heads / kv_heads;
    for (std::size_t token = 0; token < tokens; ++token) {
        for (std::size_t head = 0; head < heads; ++head) {
            const auto kv_head = head / grouped_queries;
            const auto query_base = (token * heads + head) * head_dim;
            float running_max = -std::numeric_limits<float>::infinity();
            float running_sum = 0.0F;
            std::fill_n(output.begin() + static_cast<std::ptrdiff_t>(query_base), head_dim, 0.0F);
            const auto absolute_token = past_tokens + (independent_rows ? 0 : token);
            const auto first_key = window != 0 && absolute_token + 1 > window
                ? absolute_token + 1 - window
                : 0;
            for (std::size_t tile_begin = first_key; tile_begin <= absolute_token; tile_begin += key_tile) {
                const auto tile_end = std::min(absolute_token + 1, tile_begin + key_tile);
                for (std::size_t key_token = tile_begin; key_token < tile_end; ++key_token) {
                    const bool past = key_token < past_tokens;
                    const auto local_key = past
                        ? key_token
                        : (independent_rows ? token : key_token - past_tokens);
                    const auto past_row_offset =
                        independent_rows && past ? token * past_tokens : 0;
                    const auto kv_base =
                        ((past_row_offset + local_key) * kv_heads + kv_head) * head_dim;
                    const auto key_values = past ? past_keys : current_keys;
                    const auto value_values = past ? past_values : current_values;
                    float score = 0.0F;
                    for (std::size_t column = 0; column < head_dim; ++column) {
                        score += query[query_base + column] * key_values[kv_base + column];
                    }
                    score *= scale;
                    const float next_max = std::max(running_max, score);
                    const float previous_scale = running_sum == 0.0F ? 0.0F : std::exp(running_max - next_max);
                    const float score_scale = std::exp(score - next_max);
                    const float next_sum = running_sum * previous_scale + score_scale;
                    for (std::size_t column = 0; column < head_dim; ++column) {
                        output[query_base + column] =
                            (output[query_base + column] * running_sum * previous_scale
                                + value_values[kv_base + column] * score_scale) / next_sum;
                    }
                    running_max = next_max;
                    running_sum = next_sum;
                }
            }
        }
    }
}

}  // namespace cat
