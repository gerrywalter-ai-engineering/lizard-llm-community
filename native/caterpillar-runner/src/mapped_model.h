#pragma once

#include "gguf_loader.h"
#include "tensor_data.h"

#include <cstddef>
#include <filesystem>
#include <memory>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace cat {

class MappedFile {
public:
    explicit MappedFile(const std::filesystem::path& path);
    ~MappedFile();
    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;
    MappedFile(MappedFile&&) noexcept;
    MappedFile& operator=(MappedFile&&) noexcept = delete;

    [[nodiscard]] const std::byte* data() const noexcept;
    [[nodiscard]] std::size_t size() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class MappedGgufModel {
public:
    explicit MappedGgufModel(const std::filesystem::path& path);

    [[nodiscard]] const GgufModelInfo& info() const noexcept { return info_; }
    [[nodiscard]] bool has_tensor(std::string_view name) const noexcept;
    [[nodiscard]] const GgufTensorInfo& tensor(std::string_view name) const;
    [[nodiscard]] QuantizedTensorView view(std::string_view name) const;
    // True when the integer-dot sidecar was actually built for this model.
    // The route decision depends on it: the CPU lane is only the fast lane
    // when its integer kernels have a sidecar to read.
    [[nodiscard]] bool has_integer_sidecar() const noexcept {
        return !r4_tensors_.empty();
    }
    // Rank-1 parameters (the norm weights) dequantized once at load. Returns
    // nullptr for anything not pre-converted. See mapped_model.cpp.
    [[nodiscard]] const std::vector<float>* dequantized_parameter(
        std::string_view name) const noexcept;

private:
    // Prices the whole integer-dot sidecar set against free physical RAM and
    // declines it when it would not fit with headroom. Kernels fall back per
    // tensor on an empty r4_bytes span, so this is a throughput decision only.
    [[nodiscard]] bool integer_dot_sidecar_fits() const;

    GgufModelInfo info_;
    MappedFile file_;
    std::unordered_map<std::string, std::vector<std::byte>> r4_tensors_;
    // Both built in the constructor and never mutated afterwards, so shared
    // sessions read them without locking.
    std::unordered_map<std::string, std::size_t> index_;
    std::unordered_map<std::string, std::vector<float>> dequantized_;
};

int run_mapped_model_selftest();

}  // namespace cat
