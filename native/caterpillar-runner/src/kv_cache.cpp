#include "kv_cache.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace cat {
namespace {

constexpr std::size_t block_size = 32;

std::uint16_t float_to_half(float value) {
    std::uint32_t bits{};
    std::memcpy(&bits, &value, sizeof(bits));
    const auto sign = static_cast<std::uint16_t>((bits >> 16U) & 0x8000U);
    const auto exponent = (bits >> 23U) & 0xffU;
    const auto mantissa = bits & 0x7fffffU;
    if (exponent == 0xffU) {
        return static_cast<std::uint16_t>(sign | 0x7c00U | (mantissa == 0 ? 0 : 0x0200U));
    }
    const auto unbiased = static_cast<std::int32_t>(exponent) - 127;
    if (unbiased > 15) return static_cast<std::uint16_t>(sign | 0x7c00U);
    if (unbiased >= -14) {
        auto half_mantissa = mantissa >> 13U;
        const auto remainder = mantissa & 0x1fffU;
        if (remainder > 0x1000U || (remainder == 0x1000U && (half_mantissa & 1U) != 0U)) {
            ++half_mantissa;
            if (half_mantissa == 0x400U) {
                if (unbiased == 15) return static_cast<std::uint16_t>(sign | 0x7c00U);
                return static_cast<std::uint16_t>(sign | ((unbiased + 16) << 10U));
            }
        }
        return static_cast<std::uint16_t>(sign | ((unbiased + 15) << 10U) | half_mantissa);
    }
    if (unbiased >= -24) {
        const auto full_mantissa = mantissa | 0x800000U;
        const auto shift = static_cast<std::uint32_t>(-unbiased - 1);
        auto half_mantissa = full_mantissa >> shift;
        const auto mask = (1U << shift) - 1U;
        const auto remainder = full_mantissa & mask;
        const auto halfway = 1U << (shift - 1U);
        if (remainder > halfway || (remainder == halfway && (half_mantissa & 1U) != 0U)) {
            ++half_mantissa;
        }
        return static_cast<std::uint16_t>(sign | half_mantissa);
    }
    return sign;
}

float half_to_float(std::uint16_t half) {
    const std::uint32_t sign = static_cast<std::uint32_t>(half & 0x8000U) << 16U;
    const std::uint32_t exponent = (half >> 10U) & 0x1fU;
    const std::uint32_t mantissa = half & 0x03ffU;
    std::uint32_t bits{};
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            auto normalized = mantissa;
            std::int32_t shift = -1;
            do {
                ++shift;
                normalized <<= 1U;
            } while ((normalized & 0x0400U) == 0U);
            normalized &= 0x03ffU;
            bits = sign | static_cast<std::uint32_t>(127 - 14 - shift) << 23U | normalized << 13U;
        }
    } else if (exponent == 0x1fU) {
        bits = sign | 0x7f800000U | mantissa << 13U;
    } else {
        bits = sign | (exponent + 112U) << 23U | mantissa << 13U;
    }
    float value{};
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

void hadamard(std::array<float, block_size>& values) {
    for (std::size_t stride = 1; stride < block_size; stride <<= 1U) {
        for (std::size_t base = 0; base < block_size; base += stride * 2) {
            for (std::size_t item = 0; item < stride; ++item) {
                const float left = values[base + item];
                const float right = values[base + stride + item];
                values[base + item] = left + right;
                values[base + stride + item] = left - right;
            }
        }
    }
}

template <typename T>
void store_scalar(std::span<std::byte> destination, std::size_t offset, T value) {
    std::memcpy(destination.data() + offset, &value, sizeof(value));
}

template <typename T>
T load_scalar(std::span<const std::byte> source, std::size_t offset) {
    T value{};
    std::memcpy(&value, source.data() + offset, sizeof(value));
    return value;
}

std::int32_t rounded_clamped(float value, std::int32_t minimum, std::int32_t maximum) {
    const float rounded = value >= 0.0F ? value + 0.5F : value - 0.5F;
    return static_cast<std::int32_t>(std::clamp(rounded, static_cast<float>(minimum), static_cast<float>(maximum)));
}

void require_block_width(std::size_t width) {
    if (width == 0 || width % block_size != 0) {
        throw std::invalid_argument("quantized KV row width must be a nonzero multiple of 32");
    }
}

}  // namespace

KvCacheMode parse_kv_cache_mode(std::string_view value) {
    std::string normalized(value);
    std::transform(normalized.begin(), normalized.end(), normalized.begin(), [](unsigned char character) {
        return static_cast<char>(std::tolower(character));
    });
    if (normalized == "f32" || normalized == "fp32") return KvCacheMode::f32;
    if (normalized == "f16" || normalized == "fp16" || normalized == "half") return KvCacheMode::f16;
    if (normalized == "q8_0" || normalized == "q8" || normalized == "int8" || normalized.empty()) return KvCacheMode::q8_0;
    if (normalized == "q4_0" || normalized == "q4") return KvCacheMode::q4_0;
    if (normalized == "q3_wht" || normalized == "q3" || normalized == "turbo3") return KvCacheMode::q3_wht;
    throw std::invalid_argument("unsupported Caterpillar KV cache mode: " + normalized);
}

KvCacheMode requested_kv_cache_mode() {
    const char* value = std::getenv("CATERPILLAR_KV_CACHE");
    return parse_kv_cache_mode(value == nullptr ? std::string_view{} : std::string_view(value));
}

const char* kv_cache_mode_name(KvCacheMode mode) noexcept {
    switch (mode) {
        case KvCacheMode::f32: return "f32";
        case KvCacheMode::f16: return "f16";
        case KvCacheMode::q8_0: return "q8_0";
        case KvCacheMode::q4_0: return "q4_0";
        case KvCacheMode::q3_wht: return "q3_wht";
    }
    return "unknown";
}

std::size_t kv_cache_row_bytes(KvCacheMode mode, std::size_t width) {
    if (width == 0) return 0;
    switch (mode) {
        case KvCacheMode::f32: return width * sizeof(float);
        case KvCacheMode::f16: return width * sizeof(std::uint16_t);
        case KvCacheMode::q8_0:
            require_block_width(width);
            return width / block_size * 36;
        case KvCacheMode::q4_0:
            require_block_width(width);
            return width / block_size * 20;
        case KvCacheMode::q3_wht:
            require_block_width(width);
            return width / block_size * 16;
    }
    throw std::invalid_argument("invalid KV cache mode");
}

std::uint64_t kv_cache_storage_bytes(
    KvCacheMode mode,
    std::uint64_t values_per_token,
    std::size_t context_tokens
) {
    if (values_per_token == 0 || context_tokens == 0) return 0;
    if (values_per_token > std::numeric_limits<std::size_t>::max()) {
        throw std::overflow_error("KV row width exceeds addressable memory");
    }
    const auto row_bytes = kv_cache_row_bytes(mode, static_cast<std::size_t>(values_per_token));
    if (context_tokens > std::numeric_limits<std::uint64_t>::max() / row_bytes) {
        throw std::overflow_error("KV cache byte count overflowed");
    }
    return static_cast<std::uint64_t>(row_bytes) * context_tokens;
}

void encode_kv_cache_row(
    KvCacheMode mode,
    std::span<const float> source,
    std::span<std::byte> destination
) {
    const auto required = kv_cache_row_bytes(mode, source.size());
    if (destination.size() != required) throw std::invalid_argument("KV encoded row has the wrong byte size");
    if (mode == KvCacheMode::f32) {
        std::memcpy(destination.data(), source.data(), required);
        return;
    }
    if (mode == KvCacheMode::f16) {
        for (std::size_t item = 0; item < source.size(); ++item) {
            store_scalar(destination, item * 2, float_to_half(source[item]));
        }
        return;
    }

    const auto block_bytes = required / (source.size() / block_size);
    for (std::size_t block = 0; block < source.size() / block_size; ++block) {
        const auto base = block * block_bytes;
        std::array<float, block_size> values{};
        std::copy_n(source.begin() + static_cast<std::ptrdiff_t>(block * block_size), block_size, values.begin());
        if (mode == KvCacheMode::q3_wht) hadamard(values);
        float absolute_max = 0.0F;
        for (const auto value : values) absolute_max = std::max(absolute_max, std::abs(value));
        const float range = mode == KvCacheMode::q8_0 ? 127.0F : (mode == KvCacheMode::q4_0 ? 7.0F : 3.5F);
        const float scale = absolute_max / range;
        const float inverse = scale > 0.0F ? 1.0F / scale : 0.0F;
        store_scalar(destination, base, scale);
        if (mode == KvCacheMode::q8_0) {
            for (std::size_t item = 0; item < block_size; ++item) {
                const auto quant = static_cast<std::int8_t>(rounded_clamped(values[item] * inverse, -127, 127));
                store_scalar(destination, base + 4 + item, quant);
            }
        } else if (mode == KvCacheMode::q4_0) {
            for (std::size_t item = 0; item < block_size; item += 2) {
                const auto low = static_cast<std::uint8_t>(rounded_clamped(values[item] * inverse, -8, 7) + 8);
                const auto high = static_cast<std::uint8_t>(rounded_clamped(values[item + 1] * inverse, -8, 7) + 8);
                store_scalar(destination, base + 4 + item / 2, static_cast<std::uint8_t>(low | (high << 4U)));
            }
        } else {
            std::array<std::uint8_t, 12> packed{};
            for (std::size_t item = 0; item < block_size; ++item) {
                const auto code = static_cast<std::uint32_t>(rounded_clamped(values[item] * inverse, -4, 3) + 4);
                const auto bit = item * 3;
                packed[bit / 8] = static_cast<std::uint8_t>(packed[bit / 8] | (code << (bit % 8)));
                if (bit % 8 > 5) packed[bit / 8 + 1] = static_cast<std::uint8_t>(packed[bit / 8 + 1] | (code >> (8 - bit % 8)));
            }
            std::memcpy(destination.data() + base + 4, packed.data(), packed.size());
        }
    }
}

void decode_kv_cache_row(
    KvCacheMode mode,
    std::span<const std::byte> source,
    std::span<float> destination
) {
    const auto required = kv_cache_row_bytes(mode, destination.size());
    if (source.size() != required) throw std::invalid_argument("KV encoded row has the wrong byte size");
    if (mode == KvCacheMode::f32) {
        std::memcpy(destination.data(), source.data(), required);
        return;
    }
    if (mode == KvCacheMode::f16) {
        for (std::size_t item = 0; item < destination.size(); ++item) {
            destination[item] = half_to_float(load_scalar<std::uint16_t>(source, item * 2));
        }
        return;
    }
    const auto block_bytes = required / (destination.size() / block_size);
    for (std::size_t block = 0; block < destination.size() / block_size; ++block) {
        const auto base = block * block_bytes;
        const float scale = load_scalar<float>(source, base);
        std::array<float, block_size> values{};
        if (mode == KvCacheMode::q8_0) {
            for (std::size_t item = 0; item < block_size; ++item) {
                values[item] = static_cast<float>(load_scalar<std::int8_t>(source, base + 4 + item)) * scale;
            }
        } else if (mode == KvCacheMode::q4_0) {
            for (std::size_t item = 0; item < block_size; ++item) {
                const auto packed = load_scalar<std::uint8_t>(source, base + 4 + item / 2);
                const auto nibble = item % 2 == 0 ? packed & 0x0fU : packed >> 4U;
                values[item] = static_cast<float>(static_cast<std::int32_t>(nibble) - 8) * scale;
            }
        } else {
            for (std::size_t item = 0; item < block_size; ++item) {
                const auto bit = item * 3;
                auto raw = static_cast<std::uint32_t>(load_scalar<std::uint8_t>(source, base + 4 + bit / 8));
                if (bit % 8 > 5) raw |= static_cast<std::uint32_t>(load_scalar<std::uint8_t>(source, base + 5 + bit / 8)) << 8U;
                values[item] = static_cast<float>(static_cast<std::int32_t>((raw >> (bit % 8)) & 7U) - 4) * scale;
            }
            hadamard(values);
            for (auto& value : values) value /= static_cast<float>(block_size);
        }
        std::copy(values.begin(), values.end(), destination.begin() + static_cast<std::ptrdiff_t>(block * block_size));
    }
}

int run_kv_cache_selftest() {
    try {
        std::array<float, 64> source{};
        for (std::size_t index = 0; index < source.size(); ++index) {
            source[index] = std::sin(static_cast<float>(index) * 0.37F) * (1.0F + static_cast<float>(index % 7));
        }
        for (const auto mode : {KvCacheMode::f32, KvCacheMode::f16, KvCacheMode::q8_0, KvCacheMode::q4_0, KvCacheMode::q3_wht}) {
            std::vector<std::byte> encoded(kv_cache_row_bytes(mode, source.size()));
            std::array<float, 64> decoded{};
            encode_kv_cache_row(mode, source, encoded);
            decode_kv_cache_row(mode, encoded, decoded);
            float maximum_error = 0.0F;
            for (std::size_t index = 0; index < source.size(); ++index) {
                maximum_error = std::max(maximum_error, std::abs(source[index] - decoded[index]));
            }
            const float limit = mode == KvCacheMode::f32 ? 0.0F
                : (mode == KvCacheMode::f16 ? 0.01F : (mode == KvCacheMode::q8_0 ? 0.08F : 3.0F));
            if (maximum_error > limit) {
                throw std::runtime_error(std::string(kv_cache_mode_name(mode)) + " KV round-trip error exceeds its precision contract");
            }
        }
        if (kv_cache_row_bytes(KvCacheMode::q8_0, 64) != 72
            || kv_cache_row_bytes(KvCacheMode::q4_0, 64) != 40
            || kv_cache_row_bytes(KvCacheMode::q3_wht, 64) != 32) {
            throw std::runtime_error("KV compact row byte contracts changed");
        }
        std::cout << "caterpillar KV cache selftest: ok (f32/f16/q8_0/q4_0/q3_wht)\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "caterpillar KV cache selftest: failed: " << error.what() << '\n';
        return 1;
    }
}

}  // namespace cat
