#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

namespace cat {

enum class KvCacheMode : std::uint32_t {
    f32 = 0,
    f16 = 1,
    q8_0 = 2,
    q4_0 = 3,
    q3_wht = 4,
};

[[nodiscard]] KvCacheMode parse_kv_cache_mode(std::string_view value);
[[nodiscard]] KvCacheMode requested_kv_cache_mode();
[[nodiscard]] const char* kv_cache_mode_name(KvCacheMode mode) noexcept;

// Device rows are independently addressable. Quantized rows use 32-value
// blocks with an f32 scale followed by packed values, so an attention shader
// can decode any element without scanning preceding tokens.
[[nodiscard]] std::size_t kv_cache_row_bytes(KvCacheMode mode, std::size_t width);
[[nodiscard]] std::uint64_t kv_cache_storage_bytes(
    KvCacheMode mode,
    std::uint64_t values_per_token,
    std::size_t context_tokens
);

void encode_kv_cache_row(
    KvCacheMode mode,
    std::span<const float> source,
    std::span<std::byte> destination
);
void decode_kv_cache_row(
    KvCacheMode mode,
    std::span<const std::byte> source,
    std::span<float> destination
);

int run_kv_cache_selftest();

}  // namespace cat
