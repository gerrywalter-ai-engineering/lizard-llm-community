#include "mapped_model.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace cat {

struct MappedFile::Impl {
    const std::byte* data{};
    std::size_t size{};
#ifdef _WIN32
    HANDLE file{INVALID_HANDLE_VALUE};
    HANDLE mapping{};
#else
    int file{-1};
#endif
};

MappedFile::MappedFile(const std::filesystem::path& path) : impl_(std::make_unique<Impl>()) {
#ifdef _WIN32
    impl_->file = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (impl_->file == INVALID_HANDLE_VALUE) throw std::runtime_error("unable to open GGUF mapping");
    LARGE_INTEGER length{};
    if (!GetFileSizeEx(impl_->file, &length) || length.QuadPart <= 0
        || static_cast<unsigned long long>(length.QuadPart) > std::numeric_limits<std::size_t>::max()) {
        CloseHandle(impl_->file);
        impl_->file = INVALID_HANDLE_VALUE;
        throw std::runtime_error("invalid GGUF mapping length");
    }
    impl_->size = static_cast<std::size_t>(length.QuadPart);
    impl_->mapping = CreateFileMappingW(impl_->file, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (impl_->mapping == nullptr) {
        CloseHandle(impl_->file);
        impl_->file = INVALID_HANDLE_VALUE;
        throw std::runtime_error("unable to create GGUF file mapping");
    }
    impl_->data = static_cast<const std::byte*>(MapViewOfFile(impl_->mapping, FILE_MAP_READ, 0, 0, 0));
    if (impl_->data == nullptr) {
        CloseHandle(impl_->mapping);
        CloseHandle(impl_->file);
        impl_->mapping = nullptr;
        impl_->file = INVALID_HANDLE_VALUE;
        throw std::runtime_error("unable to map GGUF file view");
    }
#else
    impl_->file = ::open(path.c_str(), O_RDONLY);
    if (impl_->file < 0) throw std::runtime_error("unable to open GGUF mapping");
    struct stat status {};
    if (::fstat(impl_->file, &status) != 0 || status.st_size <= 0
        || static_cast<unsigned long long>(status.st_size) > std::numeric_limits<std::size_t>::max()) {
        ::close(impl_->file);
        impl_->file = -1;
        throw std::runtime_error("invalid GGUF mapping length");
    }
    impl_->size = static_cast<std::size_t>(status.st_size);
    const void* address = ::mmap(nullptr, impl_->size, PROT_READ, MAP_PRIVATE, impl_->file, 0);
    if (address == MAP_FAILED) {
        ::close(impl_->file);
        impl_->file = -1;
        throw std::runtime_error("unable to map GGUF file view");
    }
    impl_->data = static_cast<const std::byte*>(address);
#endif
}

MappedFile::~MappedFile() {
    if (!impl_) return;
#ifdef _WIN32
    if (impl_->data != nullptr) UnmapViewOfFile(impl_->data);
    if (impl_->mapping != nullptr) CloseHandle(impl_->mapping);
    if (impl_->file != INVALID_HANDLE_VALUE) CloseHandle(impl_->file);
#else
    if (impl_->data != nullptr) ::munmap(const_cast<std::byte*>(impl_->data), impl_->size);
    if (impl_->file >= 0) ::close(impl_->file);
#endif
}

MappedFile::MappedFile(MappedFile&&) noexcept = default;
const std::byte* MappedFile::data() const noexcept { return impl_->data; }
std::size_t MappedFile::size() const noexcept { return impl_->size; }

MappedGgufModel::MappedGgufModel(const std::filesystem::path& path)
    : info_(inspect_gguf(path)), file_(path) {
    for (const auto& entry : info_.tensors) {
        const auto length = tensor_byte_length(entry);
        const auto absolute = info_.tensor_data_offset + entry.offset;
        if (absolute > file_.size() || length > file_.size() - static_cast<std::size_t>(absolute)) {
            throw std::runtime_error("GGUF tensor extends outside mapped file: " + entry.name);
        }
    }
    // Name -> position index, so tensor()/view()/has_tensor() are O(1). Built
    // before anything else uses them.
    index_.reserve(info_.tensors.size() * 2);
    for (std::size_t position = 0; position < info_.tensors.size(); ++position) {
        index_.emplace(info_.tensors[position].name, position);
    }
    // Dequantize the rank-1 parameters once. These are the norm weights; they
    // are small and were being converted per token by CpuModel::infer().
    for (const auto& entry : info_.tensors) {
        if (entry.dimensions.size() != 1) continue;
        const auto absolute = static_cast<std::size_t>(info_.tensor_data_offset + entry.offset);
        const auto source = std::span<const std::byte>(
            file_.data() + absolute, tensor_byte_length(entry));
        dequantized_.emplace(entry.name, dequantize_tensor(entry, source));
    }
    if (cpu_integer_dot_enabled() && integer_dot_sidecar_fits()) {
        for (const auto& entry : info_.tensors) {
            if (entry.dimensions.size() != 2
                || (entry.type != 2 && entry.type != 8 && entry.type != 12
                    && entry.type != 13 && entry.type != 14)) {
                continue;
            }
            const auto absolute = static_cast<std::size_t>(info_.tensor_data_offset + entry.offset);
            const auto source = std::span<const std::byte>(
                file_.data() + absolute, tensor_byte_length(entry));
            r4_tensors_.emplace(entry.name, repack_tensor_r4(entry, source));
        }
    }
}

// Run 04 (2026-08-04) iteration 07.
//
// The integer-dot sidecar is a second, expanded copy of every quantised weight,
// held on the heap next to the mmap: ~2.25 GB for llama-3.2-3B Q4_K_M on top of
// the ~2.0 GB mapped file. Iteration 01 promoted integer dot to default-on for
// its 1.9x throughput win, which silently made that allocation unconditional -
// and on a 16 GB host with four concurrent sessions the HTTP chat smoke
// degraded to "working set exceeds 90% of available host RAM".
//
// Price the whole sidecar set before building any of it and decline when it
// would not fit with headroom. Every kernel already falls back per tensor on an
// empty r4_bytes span, so declining is a throughput decision, never a
// correctness one. CATERPILLAR_CPU_SIDECAR=0 forces it off, =1 forces it on
// (accepting the memory cost), unset applies the budget.
bool MappedGgufModel::integer_dot_sidecar_fits() const {
    if (const char* configured = std::getenv("CATERPILLAR_CPU_SIDECAR");
        configured != nullptr && configured[0] != '\0') {
        return configured[0] != '0';
    }
    std::uint64_t required = 0;
    for (const auto& entry : info_.tensors) {
        required += cpu_r4_sidecar_bytes(entry);
    }
    if (required == 0) return true;

    std::uint64_t available = 0;
#ifdef _WIN32
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    if (GlobalMemoryStatusEx(&status) != 0) {
        available = status.ullAvailPhys;
    }
#endif
    // Unknown availability: keep the historical behaviour rather than guess.
    if (available == 0) return true;

    // Headroom, not a fraction.
    //
    // This was two thirds of free RAM, which was far too aggressive: declining
    // the sidecar drops decode onto the float path and costs ~6x (measured 9.9
    // -> 1.29 tok/s). A benchmark that runs a llama.cpp server alongside pushes
    // free RAM under the threshold, so the comparison silently measured
    // Lizard's FALLBACK against llama.cpp's fast path - a meaningless result
    // that looked like a catastrophic regression.
    //
    // The sidecar is not additive with the mapped file in the way that rule
    // assumed: both hold the same weights, and once the sidecar exists the
    // mmap's pages are cold and reclaimable. So reserve a fixed working margin
    // for the KV cache and activations and spend the rest.
    constexpr std::uint64_t working_margin = 768ull * 1024ull * 1024ull;
    const auto budget = available > working_margin ? available - working_margin : 0;
    if (required <= budget) return true;
    std::cerr << "caterpillar: integer-dot sidecar declined ("
              << (required >> 20) << " MiB needed, " << (budget >> 20)
              << " MiB budget of " << (available >> 20)
              << " MiB available); decode falls back to the float path\n";
    return false;
}

// Run 04 iteration 13: A/B switch for the iteration-12 per-token caches.
// CATERPILLAR_CPU_PARAM_CACHE=0 restores the linear tensor scan and the
// per-token dequantization so the win can be measured in one binary, which is
// the only way to separate it from this host's thermal drift.
bool cpu_param_cache_enabled() noexcept {
    static const bool enabled = [] {
        const char* configured = std::getenv("CATERPILLAR_CPU_PARAM_CACHE");
        return configured == nullptr || configured[0] != '0';
    }();
    return enabled;
}

const GgufTensorInfo& MappedGgufModel::tensor(std::string_view name) const {
    // Run 04 (2026-08-04) iteration 12: O(1) instead of a linear scan.
    // CpuModel::infer() resolves every decoder parameter by name on every
    // token - roughly 250 lookups against roughly 250 tensors - so the scan
    // cost ~60k string comparisons per token before this index existed.
    if (cpu_param_cache_enabled()) {
        if (const auto indexed = index_.find(std::string(name)); indexed != index_.end()) {
            return info_.tensors[indexed->second];
        }
    }
    const auto found = std::find_if(info_.tensors.begin(), info_.tensors.end(), [&](const GgufTensorInfo& entry) {
        return entry.name == name;
    });
    if (found == info_.tensors.end()) throw std::invalid_argument("GGUF tensor not found: " + std::string(name));
    return *found;
}

bool MappedGgufModel::has_tensor(std::string_view name) const noexcept {
    if (!index_.empty()) return index_.find(std::string(name)) != index_.end();
    return std::any_of(info_.tensors.begin(), info_.tensors.end(), [&](const GgufTensorInfo& entry) {
        return entry.name == name;
    });
}

// Run 04 iteration 12: the decoder's 1-D parameters (every RMSNorm weight, and
// anything else stored rank-1) were dequantized fresh on every token - an
// allocation plus a conversion per parameter per token, for values that never
// change. They are tiny next to the weight matrices (a few hundred KB for a 3B
// model), so they are converted once at load and handed out as a span.
// Built eagerly in the constructor and never mutated, so this needs no locking
// even though sessions share one model.
const std::vector<float>* MappedGgufModel::dequantized_parameter(
    std::string_view name
) const noexcept {
    if (!cpu_param_cache_enabled()) return nullptr;
    const auto found = dequantized_.find(std::string(name));
    return found == dequantized_.end() ? nullptr : &found->second;
}

QuantizedTensorView MappedGgufModel::view(std::string_view name) const {
    const auto& entry = tensor(name);
    const auto absolute = static_cast<std::size_t>(info_.tensor_data_offset + entry.offset);
    const auto repacked = r4_tensors_.find(entry.name);
    return QuantizedTensorView{
        &entry,
        std::span<const std::byte>(file_.data() + absolute, tensor_byte_length(entry)),
        repacked == r4_tensors_.end()
            ? std::span<const std::byte>{}
            : std::span<const std::byte>(repacked->second),
        // The lane width the repack actually used for this type (8 = Phase 3
        // R8 for Q4_K when enabled); kernels key off the view, never the env.
        repacked == r4_tensors_.end() ? std::uint8_t{4} : cpu_repack_lanes(entry.type),
    };
}

int run_mapped_model_selftest() {
    const auto path = std::filesystem::temp_directory_path() / "caterpillar-mapped-selftest.bin";
    try {
        {
            std::ofstream output(path, std::ios::binary | std::ios::trunc);
            const std::string contents = "caterpillar-map";
            output.write(contents.data(), static_cast<std::streamsize>(contents.size()));
        }
        {
            MappedFile mapped(path);
            if (mapped.size() != 15 || std::to_integer<char>(mapped.data()[0]) != 'c') {
                throw std::runtime_error("mapped file contents mismatch");
            }
        }
        std::filesystem::remove(path);
        std::cout << "caterpillar mapped-file selftest: ok\n";
        return 0;
    } catch (const std::exception& error) {
        std::filesystem::remove(path);
        std::cerr << "caterpillar mapped-file selftest: failed: " << error.what() << '\n';
        return 1;
    }
}

}  // namespace cat
