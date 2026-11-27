#pragma once

#include "gguf_loader.h"

#include <cstddef>
#include <filesystem>
#include <span>
#include <vector>

namespace cat {

struct TensorLayout {
    std::size_t block_elements{};
    std::size_t block_bytes{};
};

struct QuantizedTensorView {
    const GgufTensorInfo* tensor{};
    std::span<const std::byte> bytes;
    std::span<const std::byte> r4_bytes;
    // Interleave width of r4_bytes: 4 = classic R4 sidecar, 8 = the Phase 3
    // (2026-07-21 campaign) native-byte R8 layout for Q4_K. Carried on the
    // view so kernels and the repacked buffer can never disagree.
    std::uint8_t r4_lanes{4};

    QuantizedTensorView() = default;
    QuantizedTensorView(
        const GgufTensorInfo* tensor_entry,
        std::span<const std::byte> raw_bytes,
        std::span<const std::byte> interleaved_bytes = {},
        std::uint8_t interleave_lanes = 4
    ) : tensor(tensor_entry), bytes(raw_bytes), r4_bytes(interleaved_bytes),
        r4_lanes(interleave_lanes) {}
};

[[nodiscard]] TensorLayout tensor_layout(std::uint32_t type);
[[nodiscard]] const char* cpu_dot_kernel_name() noexcept;
// Process-wide count of weight bytes streamed by the CPU matmul/matvec entry
// points (R4 sidecar bytes when the integer path reads them, raw GGUF bytes
// otherwise). Roofline telemetry (Phase 1 of the 2026-07-21 campaign) samples
// this around a generate call to report weightBytesPerToken.
[[nodiscard]] std::uint64_t cpu_weight_bytes_streamed() noexcept;
// Phase 3 (2026-07-21 campaign): 8-row native-byte Q4_K interleave. Default
// on; CATERPILLAR_CPU_ROWS8=0 restores the 192-byte R4 layout for same-binary
// A/B. Process-stable (cached on first read) because the repacked buffer and
// the kernels must agree for the process lifetime.
[[nodiscard]] bool cpu_rows8_enabled() noexcept;
// Lane width repack_tensor_r4 will use for this tensor type in this process.
[[nodiscard]] std::uint8_t cpu_repack_lanes(std::uint32_t type) noexcept;
[[nodiscard]] bool cpu_integer_dot_enabled() noexcept;
// forced_lanes: 0 = the process default (cpu_repack_lanes), 4 = classic R4,
// 8 = Phase 3 R8 (Q4_K only). Non-zero values exist for selftests that must
// exercise both layouts in one process; production callers pass 0.
[[nodiscard]] std::vector<std::byte> repack_tensor_r4(
    const GgufTensorInfo& tensor,
    std::span<const std::byte> bytes,
    std::uint8_t forced_lanes = 0
);
// Exact byte count repack_tensor_r4 would allocate for this tensor, without
// allocating it. The integer-dot sidecar is a second, expanded copy of the
// weights held on the heap next to the mmap, so the loader has to be able to
// price the whole set before it commits to building any of it. Returns 0 for
// tensors repack_tensor_r4 does not handle.
[[nodiscard]] std::size_t cpu_r4_sidecar_bytes(const GgufTensorInfo& tensor) noexcept;
[[nodiscard]] std::size_t tensor_element_count(const GgufTensorInfo& tensor);
[[nodiscard]] std::size_t tensor_byte_length(const GgufTensorInfo& tensor);
[[nodiscard]] std::vector<float> dequantize_tensor(
    const GgufTensorInfo& tensor,
    std::span<const std::byte> bytes
);
[[nodiscard]] std::vector<float> read_tensor_f32(
    const std::filesystem::path& path,
    const GgufModelInfo& model,
    std::string_view tensor_name
);
[[nodiscard]] float tensor_row_dot(
    const QuantizedTensorView& view,
    std::size_t row,
    std::span<const float> input
);
[[nodiscard]] std::vector<float> tensor_matrix_multiply(
    const QuantizedTensorView& view,
    std::span<const float> input,
    std::size_t batches
);
// Route B (Step 09) hybrid decode primitive: computes output rows
// [first_row, last_row) of `view · input` into `output[0..count)` using the
// same per-row kernels as tensor_matrix_multiply (bit-identical per row), so
// the CPU can own a row share of a matvec while the GPU computes the rest.
void tensor_matvec_row_range(
    const QuantizedTensorView& view,
    std::span<const float> input,
    std::size_t first_row,
    std::size_t last_row,
    std::span<float> output
);
[[nodiscard]] std::vector<float> tensor_matrix_vector(
    const QuantizedTensorView& view,
    std::span<const float> input
);
[[nodiscard]] std::vector<float> tensor_row_f32(
    const QuantizedTensorView& view,
    std::size_t row
);
int run_tensor_data_selftest();

}  // namespace cat
