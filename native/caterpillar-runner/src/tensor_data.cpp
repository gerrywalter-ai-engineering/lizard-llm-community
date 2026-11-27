// GGML tensor block decoding for Caterpillar's CPU reference lane.
// Clean-room implementation from the published block layouts; no upstream
// kernel or source file was copied.

#include "tensor_data.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

#if (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#define CAT_HAS_AVX2_DISPATCH 1
#else
#define CAT_HAS_AVX2_DISPATCH 0
#endif

namespace cat {
namespace {

struct CpuMatmulProfile {
    std::array<std::atomic<std::uint64_t>, 32> nanoseconds{};
    std::array<std::atomic<std::uint64_t>, 32> calls{};

    ~CpuMatmulProfile() {
        const char* enabled = std::getenv("CATERPILLAR_CPU_PROFILE");
        if (enabled == nullptr || enabled[0] != '1') return;
        std::cerr << "caterpillar CPU matmul profile:";
        for (std::size_t type = 0; type < nanoseconds.size(); ++type) {
            const auto count = calls[type].load(std::memory_order_relaxed);
            if (count == 0) continue;
            std::cerr << " type" << type << '='
                      << static_cast<double>(nanoseconds[type].load(std::memory_order_relaxed)) / 1.0e6
                      << "ms/" << count;
        }
        std::cerr << '\n';
    }
};

CpuMatmulProfile& cpu_matmul_profile() {
    static CpuMatmulProfile profile;
    return profile;
}

class CpuMatmulProfileScope {
public:
    explicit CpuMatmulProfileScope(std::uint32_t type)
        : type_(type), enabled_(std::getenv("CATERPILLAR_CPU_PROFILE") != nullptr) {
        if (enabled_) start_ = std::chrono::steady_clock::now();
    }
    ~CpuMatmulProfileScope() {
        if (!enabled_ || type_ >= 32) return;
        const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - start_).count();
        auto& profile = cpu_matmul_profile();
        profile.nanoseconds[type_].fetch_add(
            static_cast<std::uint64_t>(elapsed), std::memory_order_relaxed);
        profile.calls[type_].fetch_add(1, std::memory_order_relaxed);
    }
private:
    std::uint32_t type_{};
    bool enabled_{};
    std::chrono::steady_clock::time_point start_{};
};

inline void cpu_relax() noexcept {
#if CAT_HAS_AVX2_DISPATCH
    _mm_pause();
#else
    std::this_thread::yield();
#endif
}

// Bounded spin count per poll unit. CATERPILLAR_CPU_POLL follows the
// llama.cpp thread-pool "poll" concept (upstream default 50): higher = spin
// longer before sleeping, 0 = pure condvar. The spin is bounded because this
// class of hardware is package-power-limited (TUNING_LOG wall #2): burning
// cores in a barrier steals the budget the kernels need.
std::size_t cpu_poll_setting() noexcept {
    static const std::size_t value = [] {
        const char* configured = std::getenv("CATERPILLAR_CPU_POLL");
        if (configured == nullptr || *configured == '\0') return std::size_t{50};
        char* end{};
        const auto parsed = std::strtoull(configured, &end, 10);
        if (end == configured || *end != '\0') return std::size_t{50};
        return std::min<std::size_t>(parsed, 10000);
    }();
    return value;
}

// Physical core count: the Phase 0 roofline peaked at 4 threads (33.05 GB/s)
// on this 4C/8T host and llama.cpp's own tg was 12% faster at 4T than 8T, so
// the default worker count is physical cores, not logical. SMT never helped a
// memory-bound GEMV here; CATERPILLAR_CPU_THREADS still overrides.
std::size_t physical_core_count() noexcept {
#ifdef _WIN32
    DWORD length = 0;
    GetLogicalProcessorInformation(nullptr, &length);
    if (GetLastError() == ERROR_INSUFFICIENT_BUFFER && length != 0) {
        std::vector<SYSTEM_LOGICAL_PROCESSOR_INFORMATION> buffer(
            length / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION));
        if (GetLogicalProcessorInformation(buffer.data(), &length) != 0) {
            std::size_t cores = 0;
            for (const auto& entry : buffer) {
                if (entry.Relationship == RelationProcessorCore) ++cores;
            }
            if (cores != 0) return cores;
        }
    }
#endif
    return std::max(1U, std::thread::hardware_concurrency());
}

// Phase 5 (2026-07-21 campaign) executor. Differences from the original:
// a spin-then-sleep barrier instead of a mutex+condvar handshake per matmul
// (a 3B model runs ~200 matmuls per token), a raw function-pointer range API
// instead of one std::function indirect call per row, and cache-line-padded
// counters (next_/remaining_/generation_ were previously false-sharing one
// line under writes from every worker).
class RowExecutor {
public:
    using RangeFunction = void (*)(void* context, std::size_t begin, std::size_t end);

    RowExecutor() {
        std::size_t thread_count = physical_core_count();
        if (const char* configured = std::getenv("CATERPILLAR_CPU_THREADS"); configured != nullptr && *configured != '\0') {
            char* end{};
            const auto parsed = std::strtoull(configured, &end, 10);
            if (end != configured && *end == '\0' && parsed > 0) {
                thread_count = std::min<std::size_t>(parsed, 64);
            }
        }
        // Run 04 round 8 (2026-08-04) iteration 31: pin workers to distinct
        // physical cores.
        //
        // First source: llama.cpp's threadpool calls
        // ggml_thread_apply_affinity(workers[ith].cpumask) for every worker
        // (ggml/src/ggml-cpu/ggml-cpu.c). Caterpillar pinned nothing, so on an
        // SMT part the scheduler is free to place several workers on sibling
        // logical processors of the same physical core - they then contend for
        // one set of execution ports and one L1, which looks exactly like the
        // sublinear scaling measured in iterations 26-28.
        //
        // Worker k takes logical processor k * stride, where stride is the SMT
        // ratio, so four workers land on four distinct physical cores.
        //
        // MEASURED AND REJECTED in this form (iteration 31). Absolute 4-thread
        // time got WORSE - 0.332/0.383 ms pinned against 0.323/0.352 unpinned -
        // even though the 1T-to-4T ratio improved, because pinning made the
        // single-thread case slower by more than it helped the parallel one.
        //
        // The likely defect is visible in llama.cpp's own source, which carries
        // the comment "Place the main thread last (towards the higher numbered
        // CPU cores)". This code pins the calling thread to CPU 0 - precisely
        // the placement llama.cpp avoids, because CPU 0 absorbs interrupt and
        // DPC work on Windows. A correct retry must enumerate physical cores via
        // GetLogicalProcessorInformation relations rather than assuming a fixed
        // SMT stride, and must place the calling thread on the highest core, not
        // the lowest. CATERPILLAR_CPU_AFFINITY=1 re-arms it for that work.
        // REJECTED as a default in this form; see the note below.
        const bool pin = [] {
            const char* configured = std::getenv("CATERPILLAR_CPU_AFFINITY");
            return configured != nullptr && configured[0] == '1';
        }();
        const auto logical = std::max<std::size_t>(1, std::thread::hardware_concurrency());
        const auto physical = std::max<std::size_t>(1, physical_core_count());
        const auto stride = std::max<std::size_t>(1, logical / physical);
        for (std::size_t index = 1; index < thread_count; ++index) {
            workers_.emplace_back([this] { worker_loop(); });
#ifdef _WIN32
            if (pin && logical <= 64) {
                const auto cpu = (index * stride) % logical;
                const DWORD_PTR mask = static_cast<DWORD_PTR>(1) << cpu;
                SetThreadAffinityMask(
                    reinterpret_cast<HANDLE>(workers_.back().native_handle()), mask);
            }
#else
            (void) pin; (void) stride;
#endif
        }
#ifdef _WIN32
        // The calling thread is worker 0 and runs drain_rows too; pin it to the
        // first physical core so it does not migrate onto a worker's sibling.
        if (pin && logical <= 64) {
            SetThreadAffinityMask(GetCurrentThread(), static_cast<DWORD_PTR>(1));
        }
#endif
    }

    ~RowExecutor() {
        {
            std::lock_guard lock(mutex_);
            stopping_.store(true, std::memory_order_release);
            generation_.fetch_add(1, std::memory_order_release);
        }
        work_ready_.notify_all();
        for (auto& worker : workers_) worker.join();
    }

    RowExecutor(const RowExecutor&) = delete;
    RowExecutor& operator=(const RowExecutor&) = delete;

    // Typed adapter: captures stay on the caller's stack, no allocation, no
    // per-index indirect call. The callable receives [begin, end).
    template <typename Callable>
    void run(std::size_t count, Callable&& callable) {
        run_ranges(count, [](void* context, std::size_t begin, std::size_t end) {
            (*static_cast<std::remove_reference_t<Callable>*>(context))(begin, end);
        }, std::addressof(callable));
    }

    void run_ranges(std::size_t count, RangeFunction function, void* context) {
        if (count == 0) return;
        if (workers_.empty() || count < 32) {
            function(context, 0, count);
            return;
        }
        function_ = function;
        context_ = context;
        count_ = count;
        claim_size_ = adaptive_claim_size(count);
        next_.store(0, std::memory_order_relaxed);
        remaining_.store(workers_.size(), std::memory_order_relaxed);
        generation_.fetch_add(1, std::memory_order_release);
        if (sleeping_.load(std::memory_order_acquire) != 0) {
            std::lock_guard lock(mutex_);
            work_ready_.notify_all();
        }
        drain_rows();
        // Completion: bounded spin, then sleep.
        const auto spin_limit = cpu_poll_setting() * 1024;
        for (std::size_t spin = 0; spin < spin_limit; ++spin) {
            if (remaining_.load(std::memory_order_acquire) == 0) return;
            cpu_relax();
        }
        caller_waiting_.store(true, std::memory_order_seq_cst);
        if (remaining_.load(std::memory_order_acquire) != 0) {
            std::unique_lock lock(mutex_);
            work_done_.wait(lock, [&] {
                return remaining_.load(std::memory_order_acquire) == 0;
            });
        }
        caller_waiting_.store(false, std::memory_order_relaxed);
    }

private:
    // About four claims per worker, clamped to [1, 64]. The old fixed 64 is
    // the upper bound, so large tensors keep their previous behaviour exactly.
    [[nodiscard]] std::size_t adaptive_claim_size(std::size_t count) const noexcept {
        static const std::size_t pinned = [] () -> std::size_t {
            const char* configured = std::getenv("CATERPILLAR_CPU_CLAIM");
            if (configured == nullptr || *configured == ' ') return 0;
            const auto requested = std::strtoull(configured, nullptr, 10);
            return (requested >= 1 && requested <= 4096) ? static_cast<std::size_t>(requested) : 0;
        }();
        if (pinned != 0) return pinned;
        const auto lanes = workers_.size() + 1;
        const auto target = count / (lanes * 4);
        return std::clamp<std::size_t>(target, 1, 64);
    }

    void drain_rows() {
        // Run 04 round 7 (2026-08-04) iteration 29: adaptive claim size.
        //
        // This was a fixed 64 tiles per fetch_add, which starves real decode
        // matmuls. A 1024-row Q4_K matmul has 128 row-tiles = only TWO claims
        // for four threads, so two threads idled and the call measured 1.48x
        // instead of ~3.2x. The campaign never saw it because every scaling
        // measurement used a 92 MB tensor (8192 claims), where 64 is fine.
        //
        // Size the claim so every worker gets about four of them: enough to
        // balance a ragged tail, few enough that the atomic is not contended.
        // CATERPILLAR_CPU_CLAIM=<n> pins a fixed size for A/B.
        const auto row_tile = claim_size_;
        const auto function = function_;
        auto* context = context_;
        const auto count = count_;
        while (true) {
            const auto begin = next_.fetch_add(row_tile, std::memory_order_relaxed);
            if (begin >= count) return;
            function(context, begin, std::min(count, begin + row_tile));
        }
    }

    void worker_loop() {
        std::size_t observed_generation{};
        const auto spin_limit = cpu_poll_setting() * 1024;
        while (true) {
            bool ready = false;
            for (std::size_t spin = 0; spin < spin_limit; ++spin) {
                if (generation_.load(std::memory_order_acquire) != observed_generation) {
                    ready = true;
                    break;
                }
                cpu_relax();
            }
            if (!ready) {
                std::unique_lock lock(mutex_);
                sleeping_.fetch_add(1, std::memory_order_release);
                work_ready_.wait(lock, [&] {
                    return stopping_.load(std::memory_order_acquire)
                        || generation_.load(std::memory_order_acquire) != observed_generation;
                });
                sleeping_.fetch_sub(1, std::memory_order_release);
            }
            if (stopping_.load(std::memory_order_acquire)) return;
            observed_generation = generation_.load(std::memory_order_acquire);
            drain_rows();
            if (remaining_.fetch_sub(1, std::memory_order_acq_rel) == 1
                && caller_waiting_.load(std::memory_order_seq_cst)) {
                std::lock_guard lock(mutex_);
                work_done_.notify_one();
            }
        }
    }

    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable work_ready_;
    std::condition_variable work_done_;
    RangeFunction function_{};
    void* context_{};
    std::size_t count_{};
    std::size_t claim_size_{64};
    alignas(64) std::atomic<std::size_t> next_{};
    alignas(64) std::atomic<std::size_t> remaining_{};
    alignas(64) std::atomic<std::size_t> generation_{};
    alignas(64) std::atomic<std::size_t> sleeping_{};
    alignas(64) std::atomic<bool> caller_waiting_{false};
    std::atomic<bool> stopping_{false};
};

RowExecutor& row_executor() {
    static RowExecutor executor;
    return executor;
}

// Weight bytes streamed by the CPU matmul entries. One relaxed add per matmul
// call (~200/token), so contention is irrelevant. See cpu_weight_bytes_streamed.
std::atomic<std::uint64_t> g_weight_bytes_streamed{};

void add_weight_bytes_streamed(std::uint64_t bytes) noexcept {
    g_weight_bytes_streamed.fetch_add(bytes, std::memory_order_relaxed);
}

std::uint16_t load_u16(const std::byte* data);
float fp16_to_float(std::uint16_t value);
void decode_scale_min(const std::byte* packed, std::size_t group, std::uint8_t& scale, std::uint8_t& minimum);
// Defined below, next to the R8 layout constants it governs; the ragged
// per-row fallback needs it before that point.
bool cpu_rows8_lanewise_enabled() noexcept;
// Same: the Q6_K repack and the generic block dot both precede its definition.
bool cpu_q6k_biased_enabled() noexcept;

bool avx2_dot_enabled() noexcept {
#if CAT_HAS_AVX2_DISPATCH
    const char* configured = std::getenv("CATERPILLAR_CPU_SIMD");
    static const bool supported = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
    if (configured == nullptr || configured[0] == '\0') return supported;
    return configured[0] == '1' && supported;
#else
    return false;
#endif
}

// Run 04 (2026-08-04), iteration 01 - PROMOTED: this used to require an
// explicit CATERPILLAR_CPU_INTEGER_DOT=1, which meant the entire integer
// kernel family the 2026-07-21 roofline campaign built (R4, R8, rows4, VNNI)
// and the speculative decode that depends on it were unreachable in every
// shipped configuration; the default CPU lane silently ran the float
// avx2-quant-dot fallback. Measured on a quiet host, llama-3.2-3B Q4_K_M:
// 4.18 -> 8.00 tok/s (1.91x) against a 10.33 tok/s llama.cpp 4T bar.
// CATERPILLAR_CPU_INTEGER_DOT=0 restores the float path for same-binary A/B.
bool cpu_integer_dot_requested() noexcept {
    const char* configured = std::getenv("CATERPILLAR_CPU_INTEGER_DOT");
    if (configured != nullptr && configured[0] != '\0') {
        return configured[0] == '1' && avx2_dot_enabled();
    }
    return avx2_dot_enabled();
}

bool read_cpu_rows4_enabled() noexcept {
    const char* configured = std::getenv("CATERPILLAR_CPU_ROWS4");
    return configured == nullptr || configured[0] != '0';
}

bool cpu_selector_cache_enabled() noexcept {
    static const bool enabled = [] {
        const char* configured = std::getenv("CATERPILLAR_CPU_CACHE_SELECTORS");
        return configured == nullptr || configured[0] != '0';
    }();
    return enabled;
}

bool cpu_rows4_enabled() noexcept {
    static const bool enabled = read_cpu_rows4_enabled();
    return cpu_selector_cache_enabled() ? enabled : read_cpu_rows4_enabled();
}

// Unlike the rows4 selector this is ALWAYS process-cached: it selects the
// repacked buffer layout at model load, and kernels address that buffer, so
// flipping mid-process would read garbage. Same-binary A/B restarts the
// process with CATERPILLAR_CPU_ROWS8=0/1.
bool read_cpu_rows8_enabled_uncached() noexcept {
#if CAT_HAS_AVX2_DISPATCH
    const char* configured = std::getenv("CATERPILLAR_CPU_ROWS8");
    return configured == nullptr || configured[0] != '0';
#else
    return false;
#endif
}

bool avx512_vnni_available() noexcept {
#if CAT_HAS_AVX2_DISPATCH
    // Re-qualified 2026-07-21 (campaign Phases 0/4): MinGW GCC 16.1 executes
    // AVX-512 VNNI correctly when entry points carry force_align_arg_pointer
    // and vectors never cross a non-inlined call boundary. Iteration 005's
    // 0xC0000005 was a stack-alignment ABI defect, not a toolchain gap.
    // CATERPILLAR_CPU_VNNI=1 forces on, =0 forces off; unset follows the
    // Phase 4 A/B default. CPUID gates it on any vendor (Intel Ice Lake+,
    // AMD Zen 4+).
    static const bool enabled = [] {
        constexpr bool default_on = false;  // flips only on a Phase 4 A/B win
        const char* configured = std::getenv("CATERPILLAR_CPU_VNNI");
        const bool requested = (configured == nullptr || configured[0] == '\0')
            ? default_on
            : configured[0] == '1';
        return requested
            && __builtin_cpu_supports("avx512vnni")
            && __builtin_cpu_supports("avx512vl");
    }();
    return enabled;
#else
    return false;
#endif
}

struct Q8KActivationBlock {
    alignas(32) std::array<std::int8_t, 256> values{};
    std::array<std::int16_t, 16> sums{};
    float d{};
};

std::vector<Q8KActivationBlock> quantize_q8_k_activations(std::span<const float> input) {
    if ((input.size() % 256) != 0) {
        throw std::invalid_argument("integer-dot activation width is not block aligned");
    }
    std::vector<Q8KActivationBlock> output(input.size() / 256);
    for (std::size_t block = 0; block < output.size(); ++block) {
        const auto base = block * 256;
        float absolute_max = 0.0F;
        for (std::size_t index = 0; index < 256; ++index) {
            absolute_max = std::max(absolute_max, std::abs(input[base + index]));
        }
        auto& quantized = output[block];
        quantized.d = absolute_max / 127.0F;
        const float inverse = quantized.d > 0.0F ? 1.0F / quantized.d : 0.0F;
        for (std::size_t index = 0; index < 256; ++index) {
            const auto value = std::clamp(
                static_cast<int>(std::nearbyint(input[base + index] * inverse)), -127, 127);
            quantized.values[index] = static_cast<std::int8_t>(value);
            quantized.sums[index / 16] = static_cast<std::int16_t>(
                static_cast<int>(quantized.sums[index / 16]) + value);
        }
    }
    return output;
}

#if CAT_HAS_AVX2_DISPATCH
__attribute__((target("avx2"), always_inline))
inline int horizontal_sum_i32(__m256i value) {
    auto sum = _mm_add_epi32(
        _mm256_castsi256_si128(value),
        _mm256_extracti128_si256(value, 1));
    sum = _mm_hadd_epi32(sum, sum);
    sum = _mm_hadd_epi32(sum, sum);
    return _mm_cvtsi128_si32(sum);
}

__attribute__((target("avx2")))
int dot_u8_s8_avx2(__m256i unsigned_weights, const std::int8_t* activations) {
    const auto signed_activations = _mm256_loadu_si256(
        reinterpret_cast<const __m256i*>(activations));
    const auto pairs = _mm256_maddubs_epi16(unsigned_weights, signed_activations);
    return horizontal_sum_i32(_mm256_madd_epi16(pairs, _mm256_set1_epi16(1)));
}

__attribute__((target("avx2"), always_inline))
inline int dot_u8_s8_vectors_avx2(__m256i unsigned_weights, __m256i signed_activations) {
    const auto pairs = _mm256_maddubs_epi16(unsigned_weights, signed_activations);
    return horizontal_sum_i32(_mm256_madd_epi16(pairs, _mm256_set1_epi16(1)));
}

__attribute__((target("avx2"), always_inline))
inline __m128i dot_u8_s8_batch4_avx2(
    __m256i weights,
    __m256i input0,
    __m256i input1,
    __m256i input2,
    __m256i input3
) {
    const auto ones = _mm256_set1_epi16(1);
    const auto dot0 = _mm256_madd_epi16(_mm256_maddubs_epi16(weights, input0), ones);
    const auto dot1 = _mm256_madd_epi16(_mm256_maddubs_epi16(weights, input1), ones);
    const auto dot2 = _mm256_madd_epi16(_mm256_maddubs_epi16(weights, input2), ones);
    const auto dot3 = _mm256_madd_epi16(_mm256_maddubs_epi16(weights, input3), ones);
    const auto pair01 = _mm256_hadd_epi32(dot0, dot1);
    const auto pair23 = _mm256_hadd_epi32(dot2, dot3);
    const auto quarters = _mm256_hadd_epi32(pair01, pair23);
    return _mm_add_epi32(
        _mm256_castsi256_si128(quarters),
        _mm256_extracti128_si256(quarters, 1));
}

// Four output rows share one Q8_K activation vector during decode. Reduce the
// four independent AVX2 dot products together so each 32-value group pays for
// one horizontal-reduction tree instead of four scalar reduction trees.
__attribute__((target("avx2"), always_inline))
inline __m128i dot_u8_s8_rows4_avx2(
    __m256i weights0,
    __m256i weights1,
    __m256i weights2,
    __m256i weights3,
    __m256i input
) {
    const auto ones = _mm256_set1_epi16(1);
    const auto dot0 = _mm256_madd_epi16(_mm256_maddubs_epi16(weights0, input), ones);
    const auto dot1 = _mm256_madd_epi16(_mm256_maddubs_epi16(weights1, input), ones);
    const auto dot2 = _mm256_madd_epi16(_mm256_maddubs_epi16(weights2, input), ones);
    const auto dot3 = _mm256_madd_epi16(_mm256_maddubs_epi16(weights3, input), ones);
    const auto pair01 = _mm256_hadd_epi32(dot0, dot1);
    const auto pair23 = _mm256_hadd_epi32(dot2, dot3);
    const auto quarters = _mm256_hadd_epi32(pair01, pair23);
    return _mm_add_epi32(
        _mm256_castsi256_si128(quarters),
        _mm256_extracti128_si256(quarters, 1));
}

// VNNI twin of dot_u8_s8_rows4_avx2: one _mm512-free _mm256_dpbusd_epi32
// replaces each maddubs+madd pair; the reduction tree is identical. Q4/Q5
// weights (u8 <= 31) times |s8| <= 127 activations cannot saturate maddubs's
// i16 lanes, so both forms produce the same integer dots bit-for-bit.
// always_inline into same-target callers only - never called cross-target
// (the iteration-005 ABI landmine, anti-pattern A1).
__attribute__((target("avx512vnni,avx512vl,avx2"), always_inline))
inline __m128i dot_u8_s8_rows4_vnni(
    __m256i weights0,
    __m256i weights1,
    __m256i weights2,
    __m256i weights3,
    __m256i input
) {
    const auto zero = _mm256_setzero_si256();
    const auto dot0 = _mm256_dpbusd_epi32(zero, weights0, input);
    const auto dot1 = _mm256_dpbusd_epi32(zero, weights1, input);
    const auto dot2 = _mm256_dpbusd_epi32(zero, weights2, input);
    const auto dot3 = _mm256_dpbusd_epi32(zero, weights3, input);
    const auto pair01 = _mm256_hadd_epi32(dot0, dot1);
    const auto pair23 = _mm256_hadd_epi32(dot2, dot3);
    const auto quarters = _mm256_hadd_epi32(pair01, pair23);
    return _mm_add_epi32(
        _mm256_castsi256_si128(quarters),
        _mm256_extracti128_si256(quarters, 1));
}

// Phase 4 (2026-07-21 campaign): the old per-dot VNNI dispatch that lived
// here was removed. Calling an avx512-target helper from an avx2-target
// function cannot inline, so every dot marshalled two __m256i through stack
// memory - the measured iteration-005 regression AND a live segfault (win64
// vector-arg ABI across the target boundary). VNNI now engages only inside
// same-target kernels (integer_q4k_r8_tile_dot_vnni); every generic dot is
// pure AVX2.
__attribute__((target("avx2")))
int dot_u8_s8(__m256i unsigned_weights, const std::int8_t* activations) {
    return dot_u8_s8_avx2(unsigned_weights, activations);
}

__attribute__((target("avx2"), always_inline))
inline int dot_s8_s8(__m256i signed_weights, const std::int8_t* activations) {
    const auto signed_activations = _mm256_loadu_si256(
        reinterpret_cast<const __m256i*>(activations));
    const auto unsigned_weights = _mm256_abs_epi8(signed_weights);
    const auto adjusted_activations = _mm256_sign_epi8(signed_activations, signed_weights);
    const auto pairs = _mm256_maddubs_epi16(unsigned_weights, adjusted_activations);
    return horizontal_sum_i32(_mm256_madd_epi16(pairs, _mm256_set1_epi16(1)));
}

__attribute__((target("avx2")))
int dot_s8_s8_16(const std::int8_t* signed_weights, const std::int8_t* activations) {
    const auto weights = _mm_loadu_si128(reinterpret_cast<const __m128i*>(signed_weights));
    const auto input = _mm_loadu_si128(reinterpret_cast<const __m128i*>(activations));
    const auto unsigned_weights = _mm_abs_epi8(weights);
    const auto adjusted_input = _mm_sign_epi8(input, weights);
    const auto pairs = _mm_maddubs_epi16(unsigned_weights, adjusted_input);
    const auto quads = _mm_madd_epi16(pairs, _mm_set1_epi16(1));
    auto sum = _mm_add_epi32(quads, _mm_shuffle_epi32(quads, _MM_SHUFFLE(1, 0, 3, 2)));
    sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm_cvtsi128_si32(sum);
}

__attribute__((target("avx2"), always_inline))
inline int dot_s8_s8_16_vectors(__m128i weights, __m128i input) {
    const auto unsigned_weights = _mm_abs_epi8(weights);
    const auto adjusted_input = _mm_sign_epi8(input, weights);
    const auto pairs = _mm_maddubs_epi16(unsigned_weights, adjusted_input);
    const auto quads = _mm_madd_epi16(pairs, _mm_set1_epi16(1));
    auto sum = _mm_add_epi32(quads, _mm_shuffle_epi32(quads, _MM_SHUFFLE(1, 0, 3, 2)));
    sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm_cvtsi128_si32(sum);
}

// Run 04 (2026-08-04) iteration 15: Q6_K weights stored biased to u8.
//
// dot_s8_s8_16_quads below pays _mm_abs_epi8 + _mm_sign_epi8 on every call to
// turn signed weights into the u8 x s8 form maddubs needs - 2 extra operations
// per lane per group, 128 per 256-weight block across 4 lanes and 16 groups.
// Q6_K's native values are a 6-bit unsigned field that the repack was
// subtracting 32 from; storing the raw field instead makes them u8 in [0, 63]
// and the dot becomes maddubs directly, with no abs and no sign.
//
// The -32 is then a per-group constant times the activation sum, which the
// Q8_K activation block already carries: sums[g] is the sum of exactly the 16
// activations of group g. So
//     true_dot(g) = biased_dot(g) - 32 * sums[g]
// which is the same bsum correction Q4_K already uses for its minima.
//
// Ranges: maddubs lanes reach 2*63*128 = 16128 (int16 safe); the corrected dot
// reaches ~194k, times a |scale| <= 127 over 16 groups reaches ~395M (int32
// safe). CATERPILLAR_CPU_Q6K_BIASED=0 restores the signed payload for A/B.
__attribute__((target("avx2"), always_inline))
inline __m128i dot_u8_s8_16_quads(__m128i weights, __m128i input) {
    return _mm_madd_epi16(_mm_maddubs_epi16(weights, input), _mm_set1_epi16(1));
}

__attribute__((target("avx2"), always_inline))
inline __m128i dot_s8_s8_16_quads(__m128i weights, __m128i input) {
    const auto unsigned_weights = _mm_abs_epi8(weights);
    const auto adjusted_input = _mm_sign_epi8(input, weights);
    return _mm_madd_epi16(
        _mm_maddubs_epi16(unsigned_weights, adjusted_input), _mm_set1_epi16(1));
}

__attribute__((target("avx2"), always_inline))
inline __m128i dot_s8_s8_16_batch4(
    __m128i weights,
    __m128i input0,
    __m128i input1,
    __m128i input2,
    __m128i input3
) {
    const auto pair01 = _mm_hadd_epi32(
        dot_s8_s8_16_quads(weights, input0), dot_s8_s8_16_quads(weights, input1));
    const auto pair23 = _mm_hadd_epi32(
        dot_s8_s8_16_quads(weights, input2), dot_s8_s8_16_quads(weights, input3));
    return _mm_hadd_epi32(pair01, pair23);
}

__attribute__((target("avx2")))
int dot_q4_nibbles(const std::byte* packed, bool high, const std::int8_t* activations) {
    auto weights = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(packed));
    if (high) weights = _mm256_srli_epi16(weights, 4);
    weights = _mm256_and_si256(weights, _mm256_set1_epi8(0x0f));
    return dot_u8_s8(weights, activations);
}

__attribute__((target("avx2")))
__m256i unpack_q4_nibbles(const std::byte* packed, bool high) {
    auto weights = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(packed));
    if (high) weights = _mm256_srli_epi16(weights, 4);
    return _mm256_and_si256(weights, _mm256_set1_epi8(0x0f));
}

__attribute__((target("avx2")))
int dot_q5_group(
    const std::byte* packed,
    const std::byte* high_bits,
    std::size_t group,
    const std::int8_t* activations
) {
    auto weights = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(packed));
    if ((group & 1U) != 0) weights = _mm256_srli_epi16(weights, 4);
    weights = _mm256_and_si256(weights, _mm256_set1_epi8(0x0f));
    const auto high = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(high_bits));
    const auto mask = _mm256_set1_epi8(static_cast<char>(1U << group));
    const auto selected = _mm256_cmpeq_epi8(_mm256_and_si256(high, mask), mask);
    weights = _mm256_or_si256(weights, _mm256_and_si256(selected, _mm256_set1_epi8(16)));
    return dot_u8_s8(weights, activations);
}

int q6_value_at(const std::byte* block, std::size_t element) {
    const auto half = element / 128;
    const auto local = element % 128;
    const auto quarter = local / 32;
    const auto item = local % 32;
    const auto low_index = half * 64 + ((quarter & 1U) != 0 ? 32 : 0) + item;
    const auto low_byte = std::to_integer<std::uint8_t>(block[low_index]);
    const auto low_nibble = quarter >= 2 ? (low_byte >> 4U) : (low_byte & 0x0FU);
    const auto high_byte = std::to_integer<std::uint8_t>(block[128 + half * 32 + item]);
    return static_cast<int>(low_nibble | (((high_byte >> (quarter * 2)) & 3U) << 4U)) - 32;
}

// Run 04 (2026-08-04) iteration 04: Q6_K R4 block layout.
//
// The old block spent 64 bytes on 16 f32 scales that were only ever
// `d * raw_scale` of a native int8 scale, then 256 bytes of expanded s8
// weights = 320 bytes per 256 weights, against a 210-byte native Q6_K block.
// Q6_K carries output.weight plus the Q4_K_M mixed tensors and is 39% of
// decode matmul time on llama-3.2-3B, so those 64 bytes are the single most
// expensive header in the engine.
//
// Keeping the int8 scales native shrinks the block to 288 bytes (-10%) AND
// lets the kernel accumulate all 16 groups in int32, paying one float
// multiply per block instead of 16. Padded to 32 so blocks stay 32-byte
// aligned and the weight payload starts on a 32-byte boundary.
//   offset  0: f32 d              (the superblock scale, hoisted out of the loop)
//   offset  4: s8 raw_scales[16]  (native 6-bit-derived group scales)
//   offset 20: padding to 32
//   offset 32: s8 weights[256]    (expanded, so the dot stays a plain s8xs8)
// Run 04 round 2 (2026-08-04) iteration 17: the Q6_K four-row TILE.
//
// The old layout stored four independent 288-byte blocks, one per lane, so the
// four lanes of a group were 288 bytes apart and every dot had to be issued
// per lane against a 128-bit register. This tile puts the four lanes' weights
// for one group contiguous, so a single 256-bit load covers two rows and one
// maddubs does the work of two. Scales become group-major, collapsing a 4-way
// strided _mm_set_epi32 into one 4-byte load.
//
//   @   0  f32 d[4]
//   @  16  s8  scales[16][4]        group-major: scales[g * 4 + lane]
//   @  80  padding to 96
//   @  96  s8  weights[16][4][16]   weights[g * 64 + lane * 16 + i]
//
// 1120 bytes per 4-row tile = 280 bytes per 256-weight row, down from 288.
// CATERPILLAR_CPU_Q6K_TILE=0 restores the per-lane blocks for same-binary A/B.
constexpr std::size_t q6k_tile_d_offset = 0;
constexpr std::size_t q6k_tile_scale_offset = 16;
constexpr std::size_t q6k_tile_weight_offset = 96;
constexpr std::size_t q6k_tile_bytes = q6k_tile_weight_offset + 16 * 4 * 16;
static_assert(q6k_tile_bytes == 1120, "Q6_K tile must stay 1120 bytes");

constexpr std::size_t q6k_r4_d_offset = 0;
constexpr std::size_t q6k_r4_scale_offset = 4;
constexpr std::size_t q6k_r4_weight_offset = 32;
constexpr std::size_t q6k_r4_block_bytes = q6k_r4_weight_offset + 256;
static_assert(q6k_r4_block_bytes == 288, "Q6_K R4 block must stay 288 bytes");

std::size_t cpu_r4_block_bytes(std::uint32_t type) {
    switch (type) {
        case 2: return 36;   // f32 d + 32 expanded s8 weights
        case 8: return 36;   // f32 d + 32 s8 weights
        case 12: return 192; // 8 f32 scales + 8 f32 minima + 128 packed q4 bytes
        case 13: return 320; // 8 f32 scales + 8 f32 minima + 256 expanded u5 weights
        case 14: return q6k_r4_block_bytes;
        default: throw std::invalid_argument("unsupported CPU R4 tensor type");
    }
}

// Phase 3 (2026-07-21 campaign) R8 layout for Q4_K: a tile is 8 rows; per
// (tile, superblock) the sidecar stores one group-major SoA header followed
// by the 8 native quant payloads, all contiguous:
//   offset   0: f32 d[8 lanes]
//   offset  32: f32 dmin[8 lanes]
//   offset  64: u8 scale[8 groups][8 lanes]   (6-bit values, pre-unpacked)
//   offset 128: u8 minimum[8 groups][8 lanes]
//   offset 192: u8 quant[8 lanes][128]        (native GGUF nibble payload)
// = 1184 bytes per tile-block = 148 bytes per 256-weight block: -22.9% vs the
// 192-byte R4 expansion (native is 144; the +8 keeps the 6-bit scale unpack
// out of the decode hot loop entirely - the first pure-native attempt decoded
// headers per superblock per token and measured -19% on the 3B model).
// The hot loop reads ready lane-vectors: no scalar stores, no per-group
// decode - the anti-pattern-A2 combination iteration 002 lacked.
// Run 04 (2026-08-04) iteration 11: d/dmin are carried as f16, the same width
// the GGUF stores them in. They were widened to f32 at repack time, which cost
// 32 bytes per tile block for range nothing needs - and at 63% of the memory
// roofline the engine is bandwidth-bound, so header bytes are throughput.
// 1216 -> 1184 bytes per tile = 152 -> 148 bytes per 256-weight block.
// One _mm256_cvtph_ps per superblock replaces the widening.
constexpr std::size_t r8_d_offset = 0;
constexpr std::size_t r8_dmin_offset = 16;
constexpr std::size_t r8_scale_offset = 32;
constexpr std::size_t r8_minimum_offset = 96;
constexpr std::size_t r8_quant_offset = 160;
constexpr std::size_t r8_quant_bytes = 128;
constexpr std::size_t r8_tile_block_bytes = r8_quant_offset + 8 * r8_quant_bytes;

void repack_cpu_block(std::uint32_t type, const std::byte* source, std::byte* destination) {
    if (type == 2) {
        const float d = fp16_to_float(load_u16(source));
        std::memcpy(destination, &d, sizeof(d));
        auto* weights = reinterpret_cast<std::int8_t*>(destination + 4);
        for (std::size_t index = 0; index < 16; ++index) {
            const auto packed = std::to_integer<std::uint8_t>(source[2 + index]);
            weights[index] = static_cast<std::int8_t>(static_cast<int>(packed & 0x0fU) - 8);
            weights[index + 16] = static_cast<std::int8_t>(static_cast<int>(packed >> 4U) - 8);
        }
        return;
    }
    if (type == 8) {
        const float d = fp16_to_float(load_u16(source));
        std::memcpy(destination, &d, sizeof(d));
        std::memcpy(destination + 4, source + 2, 32);
        return;
    }
    if (type == 12 || type == 13) {
        const float d = fp16_to_float(load_u16(source));
        const float dmin = fp16_to_float(load_u16(source + 2));
        auto* scales = reinterpret_cast<float*>(destination);
        auto* minima = scales + 8;
        for (std::size_t group = 0; group < 8; ++group) {
            std::uint8_t scale{};
            std::uint8_t minimum{};
            decode_scale_min(source + 4, group, scale, minimum);
            scales[group] = d * static_cast<float>(scale);
            minima[group] = dmin * static_cast<float>(minimum);
        }
        if (type == 12) {
            std::memcpy(destination + 64, source + 16, 128);
        } else {
            const auto* high_bits = source + 16;
            const auto* packed = source + 48;
            auto* weights = reinterpret_cast<std::uint8_t*>(destination + 64);
            for (std::size_t group = 0; group < 8; ++group) {
                for (std::size_t item = 0; item < 32; ++item) {
                    const auto byte = std::to_integer<std::uint8_t>(
                        packed[(group / 2) * 32 + item]);
                    const auto low = (group & 1U) == 0 ? (byte & 0x0fU) : (byte >> 4U);
                    const auto high = (std::to_integer<std::uint8_t>(high_bits[item]) >> group) & 1U;
                    weights[group * 32 + item] = static_cast<std::uint8_t>(low | (high << 4U));
                }
            }
        }
        return;
    }
    if (type == 14) {
        const float d = fp16_to_float(load_u16(source + 208));
        std::memcpy(destination + q6k_r4_d_offset, &d, sizeof(d));
        // Carry the native int8 group scales through unchanged: folding d into
        // them cost 64 bytes per block and forced a float multiply per group.
        std::memcpy(destination + q6k_r4_scale_offset, source + 192, 16);
        // q6_value_at returns the field already centred (value - 32). Store
        // the raw unsigned field when biased so the kernel can use maddubs
        // with no abs/sign; the -32 becomes a bsum correction per group.
        const bool biased = cpu_q6k_biased_enabled();
        auto* weights = reinterpret_cast<std::uint8_t*>(
            destination + q6k_r4_weight_offset);
        for (std::size_t item = 0; item < 256; ++item) {
            const auto centred = q6_value_at(source, item);
            weights[item] = biased
                ? static_cast<std::uint8_t>(centred + 32)
                : static_cast<std::uint8_t>(static_cast<std::int8_t>(centred));
        }
        return;
    }
    throw std::invalid_argument("unsupported CPU R4 tensor type");
}

__attribute__((target("avx2")))
float integer_quantized_block_dot(
    std::uint32_t type,
    const std::byte* block,
    std::size_t block_index,
    std::span<const Q8KActivationBlock> activations
) {
    float total = 0.0F;
    if (type == 12 || type == 13) {
            const auto& activation = activations[block_index];
            const float d = fp16_to_float(load_u16(block));
            const float dmin = fp16_to_float(load_u16(block + 2));
            const auto* scales = block + 4;
            const auto* high_bits = type == 13 ? block + 16 : nullptr;
            const auto* packed = type == 13 ? block + 48 : block + 16;
            for (std::size_t group = 0; group < 8; ++group) {
                std::uint8_t scale{};
                std::uint8_t minimum{};
                decode_scale_min(scales, group, scale, minimum);
                const auto* values = activation.values.data() + group * 32;
                const int integer_dot = type == 13
                    ? dot_q5_group(packed + (group / 2) * 32, high_bits, group, values)
                    : dot_q4_nibbles(
                        packed + (group / 2) * 32, (group & 1U) != 0, values);
                const auto activation_sum = static_cast<int>(activation.sums[group * 2])
                    + static_cast<int>(activation.sums[group * 2 + 1]);
                total += activation.d * (
                    d * static_cast<float>(scale) * static_cast<float>(integer_dot)
                    - dmin * static_cast<float>(minimum) * static_cast<float>(activation_sum));
            }
    } else if (type == 8) {
            const auto& activation = activations[block_index / 8];
            const auto* values = activation.values.data() + (block_index % 8) * 32;
            const float d = fp16_to_float(load_u16(block));
            const auto weights = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(block + 2));
            total += d * activation.d
                * static_cast<float>(dot_s8_s8(weights, values));
    } else if (type == 2) {
            const auto& activation = activations[block_index / 8];
            const auto* values = activation.values.data() + (block_index % 8) * 32;
            alignas(32) std::int8_t weights[32];
            for (std::size_t index = 0; index < 16; ++index) {
                const auto packed = std::to_integer<std::uint8_t>(block[2 + index]);
                weights[index] = static_cast<std::int8_t>(static_cast<int>(packed & 0x0FU) - 8);
                weights[index + 16] = static_cast<std::int8_t>(static_cast<int>(packed >> 4U) - 8);
            }
            total += fp16_to_float(load_u16(block)) * activation.d
                * static_cast<float>(dot_s8_s8(
                    _mm256_loadu_si256(reinterpret_cast<const __m256i*>(weights)),
                    values));
    } else if (type == 14) {
            const auto& activation = activations[block_index];
            const float d = fp16_to_float(load_u16(block + 208));
            const auto* scales = reinterpret_cast<const std::int8_t*>(block + 192);
            for (std::size_t group = 0; group < 8; ++group) {
                for (std::size_t half = 0; half < 2; ++half) {
                    int integer_dot = 0;
                    for (std::size_t item = 0; item < 16; ++item) {
                        integer_dot += q6_value_at(block, group * 32 + half * 16 + item)
                            * static_cast<int>(activation.values[group * 32 + half * 16 + item]);
                    }
                    total += d * static_cast<float>(scales[group * 2 + half])
                        * activation.d * static_cast<float>(integer_dot);
                }
            }
    }
    return total;
}

__attribute__((target("avx2")))
float integer_quantized_r4_block_dot(
    std::uint32_t type,
    const std::byte* block,
    std::size_t block_index,
    std::span<const Q8KActivationBlock> activations
) {
    float total = 0.0F;
    if (type == 12 || type == 13) {
        const auto& activation = activations[block_index];
        const auto* scales = reinterpret_cast<const float*>(block);
        const auto* minima = scales + 8;
        if (type == 12) {
            const auto* packed = block + 64;
            for (std::size_t group = 0; group < 8; ++group) {
                const auto integer_dot = dot_q4_nibbles(
                    packed + (group / 2) * 32,
                    (group & 1U) != 0,
                    activation.values.data() + group * 32);
                const auto activation_sum = static_cast<int>(activation.sums[group * 2])
                    + static_cast<int>(activation.sums[group * 2 + 1]);
                total += activation.d * (
                    scales[group] * static_cast<float>(integer_dot)
                    - minima[group] * static_cast<float>(activation_sum));
            }
        } else {
            const auto* weights = reinterpret_cast<const std::uint8_t*>(block + 64);
            for (std::size_t group = 0; group < 8; ++group) {
                const auto packed_weights = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(weights + group * 32));
                const auto integer_dot = dot_u8_s8(
                    packed_weights, activation.values.data() + group * 32);
                const auto activation_sum = static_cast<int>(activation.sums[group * 2])
                    + static_cast<int>(activation.sums[group * 2 + 1]);
                total += activation.d * (
                    scales[group] * static_cast<float>(integer_dot)
                    - minima[group] * static_cast<float>(activation_sum));
            }
        }
    } else if (type == 2 || type == 8) {
        const auto& activation = activations[block_index / 8];
        const auto group = block_index % 8;
        float scale;
        std::memcpy(&scale, block, sizeof(scale));
        const auto weights = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(block + 4));
        total = scale * activation.d * static_cast<float>(dot_s8_s8(
            weights, activation.values.data() + group * 32));
    } else if (type == 14) {
        // Integer-domain accumulation across all 16 groups, mirroring what
        // llama.cpp's Q6_K kernel does: the group scale is a native int8 and
        // the per-group dot is at most 16*127*127, so scale*dot stays inside
        // int32 with room for all 16 groups (<=524M against a 2.1B ceiling).
        // One float multiply per block instead of sixteen.
        const auto& activation = activations[block_index];
        float d{};
        std::memcpy(&d, block + q6k_r4_d_offset, sizeof(d));
        const auto* scales = reinterpret_cast<const std::int8_t*>(
            block + q6k_r4_scale_offset);
        const auto* weights = reinterpret_cast<const std::int8_t*>(
            block + q6k_r4_weight_offset);
        const bool biased = cpu_q6k_biased_enabled();
        std::int32_t accumulated = 0;
        for (std::size_t group = 0; group < 16; ++group) {
            const auto* group_weights = weights + group * 16;
            const auto* group_input = activation.values.data() + group * 16;
            int group_dot = 0;
            if (biased) {
                // Weights are the raw u8 6-bit field; reduce the four lanes of
                // the maddubs result and undo the +32 with the group's bsum.
                const auto quads = dot_u8_s8_16_quads(
                    _mm_loadu_si128(reinterpret_cast<const __m128i*>(group_weights)),
                    _mm_loadu_si128(reinterpret_cast<const __m128i*>(group_input)));
                auto folded = _mm_add_epi32(
                    quads, _mm_shuffle_epi32(quads, _MM_SHUFFLE(1, 0, 3, 2)));
                folded = _mm_add_epi32(
                    folded, _mm_shuffle_epi32(folded, _MM_SHUFFLE(2, 3, 0, 1)));
                group_dot = _mm_cvtsi128_si32(folded)
                    - 32 * static_cast<int>(activation.sums[group]);
            } else {
                group_dot = dot_s8_s8_16(group_weights, group_input);
            }
            accumulated += static_cast<std::int32_t>(scales[group]) * group_dot;
        }
        total += d * activation.d * static_cast<float>(accumulated);
    }
    return total;
}

__attribute__((target("avx2")))
float integer_quantized_row_dot(
    const QuantizedTensorView& view,
    std::size_t row,
    std::span<const Q8KActivationBlock> activations
) {
    const auto type = view.tensor->type;
    const auto columns = static_cast<std::size_t>(view.tensor->dimensions[0]);
    const auto layout = tensor_layout(type);
    const auto blocks_per_row = columns / layout.block_elements;
    float total = 0.0F;
    if (!view.r4_bytes.empty() && type == 12 && view.r4_lanes == 8) {
        // R8 ragged/per-row fallback: read this lane's pre-decoded header
        // scalars, then run the classic nibble dot on its native payload.
        const auto tile = row / 8;
        const auto lane = row % 8;
        // Iteration 09 made the quant payload lane-interleaved and iteration 11
        // made the header f16, so this fallback can no longer point straight at
        // a contiguous row. Gather the row into a local buffer when the payload
        // is lane-wise; this is the ragged path, where correctness is the only
        // thing that matters.
        const bool lanewise = cpu_rows8_lanewise_enabled();
        std::array<std::byte, r8_quant_bytes> gathered{};
        for (std::size_t block_index = 0; block_index < blocks_per_row; ++block_index) {
            const auto* tile_block = view.r4_bytes.data()
                + (tile * blocks_per_row + block_index) * r8_tile_block_bytes;
            const auto d = fp16_to_float(load_u16(tile_block + r8_d_offset + lane * 2));
            const auto dmin = fp16_to_float(load_u16(tile_block + r8_dmin_offset + lane * 2));
            const auto* scale_rows = tile_block + r8_scale_offset;
            const auto* minimum_rows = tile_block + r8_minimum_offset;
            const std::byte* quant = nullptr;
            if (lanewise) {
                const auto* payload = tile_block + r8_quant_offset;
                for (std::size_t chunk = 0; chunk < 32; ++chunk) {
                    std::memcpy(
                        gathered.data() + chunk * 4,
                        payload + chunk * 32 + lane * 4,
                        4);
                }
                quant = gathered.data();
            } else {
                quant = tile_block + r8_quant_offset + lane * r8_quant_bytes;
            }
            const auto& activation = activations[block_index];
            for (std::size_t group = 0; group < 8; ++group) {
                const auto scale = std::to_integer<std::uint8_t>(scale_rows[group * 8 + lane]);
                const auto minimum = std::to_integer<std::uint8_t>(minimum_rows[group * 8 + lane]);
                const auto integer_dot = dot_q4_nibbles(
                    quant + (group / 2) * 32,
                    (group & 1U) != 0,
                    activation.values.data() + group * 32);
                const auto activation_sum = static_cast<int>(activation.sums[group * 2])
                    + static_cast<int>(activation.sums[group * 2 + 1]);
                total += activation.d * (
                    d * static_cast<float>(scale) * static_cast<float>(integer_dot)
                    - dmin * static_cast<float>(minimum) * static_cast<float>(activation_sum));
            }
        }
        return total;
    }
    for (std::size_t block_index = 0; block_index < blocks_per_row; ++block_index) {
        const std::byte* block;
        if (view.r4_bytes.empty()) {
            block = view.bytes.data() + (row * blocks_per_row + block_index) * layout.block_bytes;
            total += integer_quantized_block_dot(type, block, block_index, activations);
        } else {
            const auto tile = row / 4;
            const auto lane = row % 4;
            const auto r4_block_bytes = cpu_r4_block_bytes(type);
            block = view.r4_bytes.data()
                + ((tile * blocks_per_row + block_index) * 4 + lane) * r4_block_bytes;
            total += integer_quantized_r4_block_dot(type, block, block_index, activations);
        }
    }
    return total;
}

// Run 04 (2026-08-04) iteration 02: streaming prefetch. The weight sidecar is
// read once per token and never reused - 2.35 GB streamed per token through a
// 12 MB L3 - so every superblock arrives cold from DRAM and the kernel stalls
// on the demand miss. Pull the next tile-block's lines in while the current
// one is being consumed. CATERPILLAR_CPU_PREFETCH=0 disables for A/B.
bool cpu_prefetch_enabled() noexcept {
    static const bool enabled = [] {
        const char* configured = std::getenv("CATERPILLAR_CPU_PREFETCH");
        return configured == nullptr || configured[0] != '0';
    }();
    return enabled;
}

// Run 04 round 5 (2026-08-04) iteration 23: prefetch distance. Iteration 02
// fixed it at one tile-block ahead and never tuned it. A tile-block is 1184
// bytes and decode streams 2.2 GB/token, so one block of lead time may be too
// little to cover a DRAM miss. CATERPILLAR_CPU_PREFETCH_DISTANCE sets how many
// blocks ahead to pull (default 1).
std::size_t cpu_prefetch_distance() noexcept {
    static const std::size_t distance = [] () -> std::size_t {
        const char* configured = std::getenv("CATERPILLAR_CPU_PREFETCH_DISTANCE");
        if (configured == nullptr || *configured == ' ') return 1;
        const auto requested = std::strtoul(configured, nullptr, 10);
        return (requested >= 1 && requested <= 8) ? static_cast<std::size_t>(requested) : 1;
    }();
    return distance;
}

// The same hint on the generic R4 path (which carries Q6_K) did NOT reproduce
// the R8 win: with it on, the measured prefetch-on/off ratio fell from +10.4%
// to +3.5% in an interleaved A/B, i.e. it costs more than it saves - a 320-byte
// Q6_K block is 20 lines per tile and the extra issue pressure outweighs the
// miss it hides. DEFERRED, default off; CATERPILLAR_CPU_PREFETCH_R4=1 re-arms
// it for a future re-test on a different host class.
bool cpu_prefetch_r4_enabled() noexcept {
    static const bool enabled = [] {
        const char* configured = std::getenv("CATERPILLAR_CPU_PREFETCH_R4");
        return configured != nullptr && configured[0] == '1';
    }();
    return enabled;
}

// Run 04 (2026-08-04) iteration 09: lane-wise R8 quant ordering.
//
// R8 stores each row's 128 quant bytes contiguously, so a row's dot product
// lives spread across one 256-bit register and has to be collapsed by a
// horizontal reduction. dot_u8_s8_rows4_avx2 pays 3 hadd_epi32 plus an extract
// per four rows, twice per 32-element group: 48 horizontal reductions per
// superblock. Iteration 06 removed the per-group float pipeline and gained only
// 2.6%, which is the measurement that proves the reduction tree - not the float
// work - is what the kernel spends its time on.
//
// R8L reorders only the quant payload so that consecutive 4-byte chunks belong
// to consecutive rows:
//     quant[(b / 4) * 32 + lane * 4 + (b % 4)] = row `lane` source byte b
// A 32-byte load then holds 4 bytes of each of the 8 rows, one row per 32-bit
// lane, so maddubs against a broadcast activation quad accumulates each row in
// its own lane and the reduction disappears entirely. This is the same idea as
// llama.cpp's block_q4_Kx8 (ggml/src/ggml-cpu/repack.cpp), at 4-byte rather
// than 8-byte granularity so no cross-lane blend/shuffle dance is needed.
//
// The header (f32 d/dmin, pre-unpacked u8 scales/minima) is untouched, so the
// block stays 152 bytes and this isolates the compute change from bandwidth.
// CATERPILLAR_CPU_R8L=0 restores the row-major payload for same-binary A/B.
// Process-stable: the repacked buffer and the kernels must agree for the
// process lifetime.
bool cpu_rows8_lanewise_enabled() noexcept {
    static const bool enabled = [] {
        const char* configured = std::getenv("CATERPILLAR_CPU_R8L");
        return configured == nullptr || configured[0] != '0';
    }();
    return enabled;
}

// Iteration 19: broadcast activations with permutevar8x32 from a register
// instead of set1_epi32 from memory. CATERPILLAR_CPU_R8L_PERM=0 restores the
// memory broadcasts.
// REJECTED as a default. Measured 9.39 tok/s against 9.84 for the memory
// broadcasts - a 4.6% loss, outside noise at sd 0.34. The hypothesis was that
// llama.cpp's ~24 memory operations per superblock against this kernel's 96
// explained its 1.4x bandwidth advantage. It does not: the Q8_K activation
// block is 256 bytes and stays hot in L1, so vpbroadcastd is effectively free,
// while vpermd carries 3-cycle latency and contends on port 5 with the mask
// and shift work already there. Trading a cheap load for a contended shuffle
// is a loss. Kept behind CATERPILLAR_CPU_R8L_PERM=1 as documented evidence.
// Iteration 22: two accumulator chains per sub-block to halve the add_epi16
// dependency chain. CATERPILLAR_CPU_R8L_ILP=0 restores the single chain.
// REJECTED as a default. Screened at 400 samples, twice: 0.516/0.541 ms with
// two chains against 0.531/0.533 with one - inconsistent in sign and far under
// the ~10% microbenchmark threshold that iteration 21 calibrated as the floor
// for mattering end-to-end. The four add_epi16 are not the critical path; the
// maddubs throughput already covers them. Kept behind
// CATERPILLAR_CPU_R8L_ILP=1 as evidence, default off.
bool cpu_r8l_ilp_enabled() noexcept {
    static const bool enabled = [] {
        const char* configured = std::getenv("CATERPILLAR_CPU_R8L_ILP");
        return configured != nullptr && configured[0] == '1';
    }();
    return enabled;
}

bool cpu_r8l_permute_enabled() noexcept {
    static const bool enabled = [] {
        const char* configured = std::getenv("CATERPILLAR_CPU_R8L_PERM");
        return configured != nullptr && configured[0] == '1';
    }();
    return enabled;
}

// Run 04 iteration 15: Q6_K weights stored biased to u8 (see
// dot_u8_s8_16_quads). CATERPILLAR_CPU_Q6K_BIASED=0 restores the signed
// payload. Process-stable: the repacked buffer and the kernels must agree.
// DEFERRED, default OFF. The theory is sound - it removes 128 abs/sign
// operations per 256-weight block - but the interleaved A/B did not support it:
// 7.65 tok/s signed vs 6.48 biased, on a host that had degraded to sd 1.29/1.14
// after hours of benchmarking, so the result is inconclusive and leans
// negative. A likely cause is this very flag: selecting the dot with a runtime
// ternary inside the group loop blocks the hoisting the signed path gets for
// free. Re-test as a compile-time choice on a cold machine before promoting.
bool cpu_q6k_biased_enabled() noexcept {
    static const bool enabled = [] {
        const char* configured = std::getenv("CATERPILLAR_CPU_Q6K_BIASED");
        return configured != nullptr && configured[0] == '1';
    }();
    return enabled;
}

// Run 04 round 2 iteration 17: the Q6_K four-row tile (see q6k_tile_* above).
// Requires the rows4 kernel, which is the only consumer of the tile layout.
// CATERPILLAR_CPU_Q6K_TILE=0 restores the per-lane 288-byte blocks.
bool cpu_q6k_tile_enabled() noexcept {
    static const bool enabled = [] {
        const char* configured = std::getenv("CATERPILLAR_CPU_Q6K_TILE");
        const bool requested = configured == nullptr || configured[0] != '0';
        return requested && read_cpu_rows4_enabled();
    }();
    return enabled;
}

// Run 04 iteration 16: defer the Q6_K lane reduction to once per block.
// CATERPILLAR_CPU_Q6K_DEFER=0 restores the per-group reduction for A/B.
// DEFERRED, default OFF. Removing 45 of the 48 horizontal reductions per block
// bought nothing: means 8.07 (per-group) vs 8.08 (deferred) over 4 interleaved
// rounds. Scaling four per-lane partials costs four broadcasts and four
// mullo_epi32 per group, which is exactly what the three hadds cost. The hadd
// count was never the Q6_K bottleneck - unlike Q4_K, where iteration 09's
// lane-wise payload removed the reductions *without* adding per-lane work.
// The lesson: a reduction is only worth removing if nothing replaces it.
bool cpu_q6k_deferred_reduction_enabled() noexcept {
    static const bool enabled = [] {
        const char* configured = std::getenv("CATERPILLAR_CPU_Q6K_DEFER");
        return configured != nullptr && configured[0] == '1';
    }();
    return enabled;
}

// Run 04 iteration 05: the Q6_K four-row decode kernel.
// CATERPILLAR_CPU_Q6K_ROWS4=0 restores the generic per-lane loop for A/B.
bool cpu_q6k_rows4_enabled() noexcept {
    static const bool enabled = [] {
        const char* configured = std::getenv("CATERPILLAR_CPU_Q6K_ROWS4");
        return configured == nullptr || configured[0] != '0';
    }();
    return enabled;
}

// Issue one T0 prefetch per 64-byte line of the block starting at `ahead`.
// Called once per superblock, one superblock ahead of the read stream.
inline void prefetch_block(const std::byte* ahead, std::size_t bytes) noexcept {
    for (std::size_t line = 0; line < bytes; line += 64) {
        _mm_prefetch(reinterpret_cast<const char*>(ahead + line), _MM_HINT_T0);
    }
}

// Load one group's eight-lane scale (or minimum) row: 8 pre-unpacked u8
// values -> f32 vector, multiplied by the per-lane d (or dmin) vector. Pure
// register work; the 6-bit unpack happened once at repack time.
// Load 8 pre-unpacked u8 scales (or minima) and duplicate each into both int16
// halves of its 32-bit lane, so madd_epi16 against a pair-accumulator computes
// (lo + hi) * scale per row in one instruction.
__attribute__((target("avx2"), always_inline))
inline __m256i duplicate_u8_lanes_epi16(const std::byte* row) {
    const auto widened = _mm256_cvtepu8_epi32(
        _mm_loadl_epi64(reinterpret_cast<const __m128i*>(row)));
    return _mm256_or_si256(widened, _mm256_slli_epi32(widened, 16));
}

__attribute__((target("avx2"), always_inline))
inline __m256 r8_group_row(const std::byte* row, __m256 d_vector) {
    const auto bytes = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(row));
    return _mm256_mul_ps(d_vector, _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(bytes)));
}

// Iteration 09: the lane-wise R8 decode kernel. Each 32-bit lane owns one
// output row for the whole superblock, so there is no horizontal reduction at
// all - the 48 hadds per superblock that the row-major kernel below pays are
// simply gone.
//
// Q4_K packs sub-block 2p in the low nibbles and 2p+1 in the high nibbles of
// source bytes [32p, 32p+32), so one loaded chunk feeds two sub-blocks. The
// int16 accumulators take four 4-element steps before being drained: each
// maddubs lane is bounded by 2*15*128 = 3840 and four of them reach 15360,
// inside int16. Draining through madd_epi16 against a scale duplicated into
// both halves of each lane folds the pair-add and the scale multiply into one
// instruction, and lands the result in int32 - the trick llama.cpp's
// ggml_gemv_q4_K_8x8_q8_K uses.
__attribute__((target("avx2,f16c")))
void integer_q4k_r8l_tile_dot(
    const QuantizedTensorView& view,
    std::size_t first_row,
    std::span<const Q8KActivationBlock> activations,
    std::span<float> output
) {
    const auto rows = static_cast<std::size_t>(view.tensor->dimensions[1]);
    const auto columns = static_cast<std::size_t>(view.tensor->dimensions[0]);
    const auto blocks_per_row = columns / 256;
    const auto lanes = std::min<std::size_t>(8, rows - first_row);
    const auto tile = first_row / 8;
    const bool prefetch = cpu_prefetch_enabled();
    const auto prefetch_distance = cpu_prefetch_distance();
    const bool permute_activations = cpu_r8l_permute_enabled();
    const bool interleave_chains = cpu_r8l_ilp_enabled();
    const auto low_mask = _mm256_set1_epi8(0x0F);
    auto totals = _mm256_setzero_ps();
    for (std::size_t block_index = 0; block_index < blocks_per_row; ++block_index) {
        const auto* tile_block = view.r4_bytes.data()
            + (tile * blocks_per_row + block_index) * r8_tile_block_bytes;
        if (prefetch && block_index + prefetch_distance < blocks_per_row) {
            prefetch_block(
                tile_block + prefetch_distance * r8_tile_block_bytes,
                r8_tile_block_bytes);
        }
        const auto d_vector = _mm256_cvtph_ps(_mm_loadu_si128(
            reinterpret_cast<const __m128i*>(tile_block + r8_d_offset)));
        const auto dmin_vector = _mm256_cvtph_ps(_mm_loadu_si128(
            reinterpret_cast<const __m128i*>(tile_block + r8_dmin_offset)));
        const auto* quant = tile_block + r8_quant_offset;
        const auto& block_activation = activations[block_index];
        const auto* values = block_activation.values.data();
        auto accumulated = _mm256_setzero_si256();
        auto accumulated_minimum = _mm256_setzero_si256();
        for (std::size_t pair = 0; pair < 4; ++pair) {
            const auto low_group = pair * 2;
            const auto high_group = pair * 2 + 1;
            const auto scale_low = duplicate_u8_lanes_epi16(
                tile_block + r8_scale_offset + low_group * 8);
            const auto scale_high = duplicate_u8_lanes_epi16(
                tile_block + r8_scale_offset + high_group * 8);
            // Run 04 round 3 (2026-08-04) iteration 19: broadcast activations
            // from a register, not from memory.
            //
            // Comparing this kernel to llama.cpp's ggml_gemv_q4_K_8x8_q8_K, the
            // two run a similar ~5:1 total overhead per maddubs - but llama.cpp
            // issues ~24 memory operations per superblock against this kernel's
            // 96 (32 weight loads + 64 activation broadcasts), because its
            // supporting work is blends and shuffles on registers already held.
            // Loads are port-limited at 2-3 per cycle; register shuffles
            // dual-issue. That is the likeliest source of the measured 1.4x
            // bandwidth difference.
            //
            // This tests the hypothesis without touching the weight layout:
            // load each sub-block's 8 activation dwords once and select the one
            // this step needs with permutevar8x32, taking activation memory
            // operations per superblock from 64 to 8.
            // CATERPILLAR_CPU_R8L_PERM=0 restores the memory broadcasts.
            const auto low_activations = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(values + low_group * 32));
            const auto high_activations = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(values + high_group * 32));
            for (std::size_t half = 0; half < 2; ++half) {
                // Run 04 round 4 iteration 22: two accumulator chains per
                // sub-block instead of one. The four add_epi16 in this loop
                // form a serial dependency; alternating between two
                // accumulators halves the chain and lets the maddubs latency
                // overlap. Folded once at the end of the half.
                // CATERPILLAR_CPU_R8L_ILP=0 restores the single chain.
                auto pairs_low = _mm256_setzero_si256();
                auto pairs_high = _mm256_setzero_si256();
                auto pairs_low_b = _mm256_setzero_si256();
                auto pairs_high_b = _mm256_setzero_si256();
                for (std::size_t step = 0; step < 4; ++step) {
                    const auto index = half * 4 + step;
                    const auto weights = _mm256_loadu_si256(
                        reinterpret_cast<const __m256i*>(
                            quant + (pair * 8 + index) * 32));
                    __m256i low_broadcast{};
                    __m256i high_broadcast{};
                    if (permute_activations) {
                        const auto selector = _mm256_set1_epi32(static_cast<int>(index));
                        low_broadcast = _mm256_permutevar8x32_epi32(low_activations, selector);
                        high_broadcast = _mm256_permutevar8x32_epi32(high_activations, selector);
                    } else {
                        std::int32_t low_quad{};
                        std::int32_t high_quad{};
                        std::memcpy(&low_quad, values + low_group * 32 + index * 4, 4);
                        std::memcpy(&high_quad, values + high_group * 32 + index * 4, 4);
                        low_broadcast = _mm256_set1_epi32(low_quad);
                        high_broadcast = _mm256_set1_epi32(high_quad);
                    }
                    const auto low_product = _mm256_maddubs_epi16(
                        _mm256_and_si256(weights, low_mask), low_broadcast);
                    const auto high_product = _mm256_maddubs_epi16(
                        _mm256_and_si256(
                            _mm256_srli_epi16(weights, 4), low_mask),
                        high_broadcast);
                    if (interleave_chains && (step & 1U) != 0) {
                        pairs_low_b = _mm256_add_epi16(pairs_low_b, low_product);
                        pairs_high_b = _mm256_add_epi16(pairs_high_b, high_product);
                    } else {
                        pairs_low = _mm256_add_epi16(pairs_low, low_product);
                        pairs_high = _mm256_add_epi16(pairs_high, high_product);
                    }
                }
                if (interleave_chains) {
                    pairs_low = _mm256_add_epi16(pairs_low, pairs_low_b);
                    pairs_high = _mm256_add_epi16(pairs_high, pairs_high_b);
                }
                accumulated = _mm256_add_epi32(
                    accumulated, _mm256_madd_epi16(pairs_low, scale_low));
                accumulated = _mm256_add_epi32(
                    accumulated, _mm256_madd_epi16(pairs_high, scale_high));
            }
            accumulated_minimum = _mm256_add_epi32(
                accumulated_minimum,
                _mm256_mullo_epi32(
                    _mm256_set1_epi32(
                        static_cast<int>(block_activation.sums[low_group * 2])
                        + static_cast<int>(block_activation.sums[low_group * 2 + 1])),
                    _mm256_cvtepu8_epi32(_mm_loadl_epi64(
                        reinterpret_cast<const __m128i*>(
                            tile_block + r8_minimum_offset + low_group * 8)))));
            accumulated_minimum = _mm256_add_epi32(
                accumulated_minimum,
                _mm256_mullo_epi32(
                    _mm256_set1_epi32(
                        static_cast<int>(block_activation.sums[high_group * 2])
                        + static_cast<int>(block_activation.sums[high_group * 2 + 1])),
                    _mm256_cvtepu8_epi32(_mm_loadl_epi64(
                        reinterpret_cast<const __m128i*>(
                            tile_block + r8_minimum_offset + high_group * 8)))));
        }
        totals = _mm256_add_ps(
            totals,
            _mm256_mul_ps(
                _mm256_set1_ps(block_activation.d),
                _mm256_sub_ps(
                    _mm256_mul_ps(d_vector, _mm256_cvtepi32_ps(accumulated)),
                    _mm256_mul_ps(
                        dmin_vector,
                        _mm256_cvtepi32_ps(accumulated_minimum)))));
    }
    alignas(32) std::array<float, 8> reduced{};
    _mm256_store_ps(reduced.data(), totals);
    std::copy_n(reduced.begin(), lanes, output.begin());
}

// Eight-output-row Q4_K decode kernel over the R8 layout.
//
// Run 04 iteration 06 changed the accumulation domain. This kernel used to
// mirror the rows4 kernel's float arithmetic exactly - one cvtepi32_ps, two
// f32 scale-vector builds, a sub and two muls per 32-element group, eight
// times per superblock - which made it bit-identical to R4 but cost ~112
// float-domain operations per superblock. It now accumulates the scaled dot
// and the minima correction in int32 across the whole superblock and converts
// once, which is what llama.cpp's ggml_gemv_q4_K_8x8_q8_K does and is the
// mechanism behind its higher roofline fraction.
//
// Consequence: R8 is no longer bit-identical to R4. The integer form is the
// more accurate of the two (exact until the single conversion), and the
// selftest holds them to a numerical tolerance instead of bit equality.
__attribute__((target("avx2,f16c")))
void integer_q4k_r8_tile_dot(
    const QuantizedTensorView& view,
    std::size_t first_row,
    std::span<const Q8KActivationBlock> activations,
    std::span<float> output
) {
    const auto rows = static_cast<std::size_t>(view.tensor->dimensions[1]);
    const auto columns = static_cast<std::size_t>(view.tensor->dimensions[0]);
    const auto blocks_per_row = columns / 256;
    const auto lanes = std::min<std::size_t>(8, rows - first_row);
    const auto tile = first_row / 8;
    const bool prefetch = cpu_prefetch_enabled();
    const auto prefetch_distance = cpu_prefetch_distance();
    auto totals = _mm256_setzero_ps();
    for (std::size_t block_index = 0; block_index < blocks_per_row; ++block_index) {
        const auto* tile_block = view.r4_bytes.data()
            + (tile * blocks_per_row + block_index) * r8_tile_block_bytes;
        if (prefetch && block_index + prefetch_distance < blocks_per_row) {
            prefetch_block(
                tile_block + prefetch_distance * r8_tile_block_bytes,
                r8_tile_block_bytes);
        }
        const auto d_vector = _mm256_cvtph_ps(_mm_loadu_si128(
            reinterpret_cast<const __m128i*>(tile_block + r8_d_offset)));
        const auto dmin_vector = _mm256_cvtph_ps(_mm_loadu_si128(
            reinterpret_cast<const __m128i*>(tile_block + r8_dmin_offset)));
        const auto* quant = tile_block + r8_quant_offset;
        const auto& block_activation = activations[block_index];
        // Run 04 iteration 06: the scaled dot and the minima correction both
        // accumulate in int32 across the whole superblock, so the kernel pays
        // one float conversion per superblock instead of one per group.
        // Bounds: |dot| <= 32*15*128 = 61440, scale is a 6-bit u8 <= 63, and 8
        // groups reach 31M; the minima term reaches 2.1M. Both sit far inside
        // int32.
        auto accumulated = _mm256_setzero_si256();
        auto accumulated_minimum = _mm256_setzero_si256();
        for (std::size_t group = 0; group < 8; ++group) {
            const auto input = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
                block_activation.values.data() + group * 32));
            const auto quant_offset = (group / 2) * 32;
            const bool high = (group & 1U) != 0;
            const auto dots_low = dot_u8_s8_rows4_avx2(
                unpack_q4_nibbles(quant + 0 * r8_quant_bytes + quant_offset, high),
                unpack_q4_nibbles(quant + 1 * r8_quant_bytes + quant_offset, high),
                unpack_q4_nibbles(quant + 2 * r8_quant_bytes + quant_offset, high),
                unpack_q4_nibbles(quant + 3 * r8_quant_bytes + quant_offset, high),
                input);
            const auto dots_high = dot_u8_s8_rows4_avx2(
                unpack_q4_nibbles(quant + 4 * r8_quant_bytes + quant_offset, high),
                unpack_q4_nibbles(quant + 5 * r8_quant_bytes + quant_offset, high),
                unpack_q4_nibbles(quant + 6 * r8_quant_bytes + quant_offset, high),
                unpack_q4_nibbles(quant + 7 * r8_quant_bytes + quant_offset, high),
                input);
            const auto dots = _mm256_set_m128i(dots_high, dots_low);
            const auto scale_lanes = _mm256_cvtepu8_epi32(_mm_loadl_epi64(
                reinterpret_cast<const __m128i*>(
                    tile_block + r8_scale_offset + group * 8)));
            const auto minimum_lanes = _mm256_cvtepu8_epi32(_mm_loadl_epi64(
                reinterpret_cast<const __m128i*>(
                    tile_block + r8_minimum_offset + group * 8)));
            const auto activation_sum = _mm256_set1_epi32(
                static_cast<int>(block_activation.sums[group * 2])
                + static_cast<int>(block_activation.sums[group * 2 + 1]));
            accumulated = _mm256_add_epi32(
                accumulated, _mm256_mullo_epi32(dots, scale_lanes));
            accumulated_minimum = _mm256_add_epi32(
                accumulated_minimum,
                _mm256_mullo_epi32(activation_sum, minimum_lanes));
        }
        // mul+add rather than fmadd: these kernels carry target("avx2") only,
        // and _mm256_fmadd_ps needs target("fma"). One extra operation per
        // superblock is not worth widening the target attribute of a kernel
        // that other always_inline avx2 helpers are inlined into.
        totals = _mm256_add_ps(
            totals,
            _mm256_mul_ps(
                _mm256_set1_ps(block_activation.d),
                _mm256_sub_ps(
                    _mm256_mul_ps(d_vector, _mm256_cvtepi32_ps(accumulated)),
                    _mm256_mul_ps(
                        dmin_vector,
                        _mm256_cvtepi32_ps(accumulated_minimum)))));
    }
    alignas(32) std::array<float, 8> reduced{};
    _mm256_store_ps(reduced.data(), totals);
    std::copy_n(reduced.begin(), lanes, output.begin());
}

// Phase 4 (2026-07-21 campaign): the VNNI build of the R8 decode kernel. The
// body is a deliberate textual twin of integer_q4k_r8_tile_dot with only the
// dot helper swapped - GCC cannot inline across mismatched target attributes,
// and passing __m256i through a non-inlined call boundary marshals vectors
// through memory (the measured iteration-005 regression, anti-pattern A1), so
// the whole kernel is compiled once per ISA and dispatch happens per tile.
// force_align_arg_pointer guards the AVX-512 entry ABI (the Phase 0 probe
// proved the old "MinGW VNNI faults" note was this, not a toolchain gap).
__attribute__((target("avx512vnni,avx512vl,avx2,f16c"), force_align_arg_pointer))
void integer_q4k_r8_tile_dot_vnni(
    const QuantizedTensorView& view,
    std::size_t first_row,
    std::span<const Q8KActivationBlock> activations,
    std::span<float> output
) {
    const auto rows = static_cast<std::size_t>(view.tensor->dimensions[1]);
    const auto columns = static_cast<std::size_t>(view.tensor->dimensions[0]);
    const auto blocks_per_row = columns / 256;
    const auto lanes = std::min<std::size_t>(8, rows - first_row);
    const auto tile = first_row / 8;
    const bool prefetch = cpu_prefetch_enabled();
    const auto prefetch_distance = cpu_prefetch_distance();
    auto totals = _mm256_setzero_ps();
    for (std::size_t block_index = 0; block_index < blocks_per_row; ++block_index) {
        const auto* tile_block = view.r4_bytes.data()
            + (tile * blocks_per_row + block_index) * r8_tile_block_bytes;
        if (prefetch && block_index + prefetch_distance < blocks_per_row) {
            prefetch_block(
                tile_block + prefetch_distance * r8_tile_block_bytes,
                r8_tile_block_bytes);
        }
        const auto d_vector = _mm256_cvtph_ps(_mm_loadu_si128(
            reinterpret_cast<const __m128i*>(tile_block + r8_d_offset)));
        const auto dmin_vector = _mm256_cvtph_ps(_mm_loadu_si128(
            reinterpret_cast<const __m128i*>(tile_block + r8_dmin_offset)));
        const auto* quant = tile_block + r8_quant_offset;
        const auto& block_activation = activations[block_index];
        // Run 04 iteration 06: the scaled dot and the minima correction both
        // accumulate in int32 across the whole superblock, so the kernel pays
        // one float conversion per superblock instead of one per group.
        // Bounds: |dot| <= 32*15*128 = 61440, scale is a 6-bit u8 <= 63, and 8
        // groups reach 31M; the minima term reaches 2.1M. Both sit far inside
        // int32.
        auto accumulated = _mm256_setzero_si256();
        auto accumulated_minimum = _mm256_setzero_si256();
        for (std::size_t group = 0; group < 8; ++group) {
            const auto input = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
                block_activation.values.data() + group * 32));
            const auto quant_offset = (group / 2) * 32;
            const bool high = (group & 1U) != 0;
            const auto dots_low = dot_u8_s8_rows4_vnni(
                unpack_q4_nibbles(quant + 0 * r8_quant_bytes + quant_offset, high),
                unpack_q4_nibbles(quant + 1 * r8_quant_bytes + quant_offset, high),
                unpack_q4_nibbles(quant + 2 * r8_quant_bytes + quant_offset, high),
                unpack_q4_nibbles(quant + 3 * r8_quant_bytes + quant_offset, high),
                input);
            const auto dots_high = dot_u8_s8_rows4_vnni(
                unpack_q4_nibbles(quant + 4 * r8_quant_bytes + quant_offset, high),
                unpack_q4_nibbles(quant + 5 * r8_quant_bytes + quant_offset, high),
                unpack_q4_nibbles(quant + 6 * r8_quant_bytes + quant_offset, high),
                unpack_q4_nibbles(quant + 7 * r8_quant_bytes + quant_offset, high),
                input);
            const auto dots = _mm256_set_m128i(dots_high, dots_low);
            const auto scale_lanes = _mm256_cvtepu8_epi32(_mm_loadl_epi64(
                reinterpret_cast<const __m128i*>(
                    tile_block + r8_scale_offset + group * 8)));
            const auto minimum_lanes = _mm256_cvtepu8_epi32(_mm_loadl_epi64(
                reinterpret_cast<const __m128i*>(
                    tile_block + r8_minimum_offset + group * 8)));
            const auto activation_sum = _mm256_set1_epi32(
                static_cast<int>(block_activation.sums[group * 2])
                + static_cast<int>(block_activation.sums[group * 2 + 1]));
            accumulated = _mm256_add_epi32(
                accumulated, _mm256_mullo_epi32(dots, scale_lanes));
            accumulated_minimum = _mm256_add_epi32(
                accumulated_minimum,
                _mm256_mullo_epi32(activation_sum, minimum_lanes));
        }
        // mul+add rather than fmadd: these kernels carry target("avx2") only,
        // and _mm256_fmadd_ps needs target("fma"). One extra operation per
        // superblock is not worth widening the target attribute of a kernel
        // that other always_inline avx2 helpers are inlined into.
        totals = _mm256_add_ps(
            totals,
            _mm256_mul_ps(
                _mm256_set1_ps(block_activation.d),
                _mm256_sub_ps(
                    _mm256_mul_ps(d_vector, _mm256_cvtepi32_ps(accumulated)),
                    _mm256_mul_ps(
                        dmin_vector,
                        _mm256_cvtepi32_ps(accumulated_minimum)))));
    }
    alignas(32) std::array<float, 8> reduced{};
    _mm256_store_ps(reduced.data(), totals);
    std::copy_n(reduced.begin(), lanes, output.begin());
}

// R8 batched verifier GEMM: one weight-tile unpack serves every batch, so
// weight bytes streamed per verification pass are independent of the batch
// count - the invariant Phase 6 builds on.
__attribute__((target("avx2,f16c")))
void integer_q4k_r8_tile_gemm(
    const QuantizedTensorView& view,
    std::size_t first_row,
    std::span<const Q8KActivationBlock> activations,
    std::size_t activation_blocks_per_batch,
    std::size_t batches,
    std::span<float> output
) {
    const auto rows = static_cast<std::size_t>(view.tensor->dimensions[1]);
    const auto columns = static_cast<std::size_t>(view.tensor->dimensions[0]);
    const auto blocks_per_row = columns / 256;
    const auto lanes = std::min<std::size_t>(8, rows - first_row);
    const auto tile = first_row / 8;
    // Accumulators live in a plain float buffer (8 per batch), loaded and
    // stored per group. Never hold __m256 values in std::vector: its member
    // functions instantiate without the avx2 target attribute, which is the
    // iteration-005 cross-boundary ABI landmine (anti-pattern A1).
    thread_local std::vector<float> batch_totals;
    batch_totals.assign(batches * 8, 0.0F);
    for (std::size_t block_index = 0; block_index < blocks_per_row; ++block_index) {
        const auto* tile_block = view.r4_bytes.data()
            + (tile * blocks_per_row + block_index) * r8_tile_block_bytes;
        const auto d_vector = _mm256_cvtph_ps(_mm_loadu_si128(
            reinterpret_cast<const __m128i*>(tile_block + r8_d_offset)));
        const auto dmin_vector = _mm256_cvtph_ps(_mm_loadu_si128(
            reinterpret_cast<const __m128i*>(tile_block + r8_dmin_offset)));
        const auto* quant = tile_block + r8_quant_offset;
        for (std::size_t group = 0; group < 8; ++group) {
            const auto quant_offset = (group / 2) * 32;
            const bool high = (group & 1U) != 0;
            __m256i weights[8];
            for (std::size_t lane = 0; lane < 8; ++lane) {
                weights[lane] = unpack_q4_nibbles(
                    quant + lane * r8_quant_bytes + quant_offset, high);
            }
            const auto scale_vector = r8_group_row(
                tile_block + r8_scale_offset + group * 8, d_vector);
            const auto minimum_vector = r8_group_row(
                tile_block + r8_minimum_offset + group * 8, dmin_vector);
            for (std::size_t batch = 0; batch < batches; ++batch) {
                const auto& block_activation = activations[
                    batch * activation_blocks_per_batch + block_index];
                const auto input = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
                    block_activation.values.data() + group * 32));
                const auto dots_low = dot_u8_s8_rows4_avx2(
                    weights[0], weights[1], weights[2], weights[3], input);
                const auto dots_high = dot_u8_s8_rows4_avx2(
                    weights[4], weights[5], weights[6], weights[7], input);
                const auto dots = _mm256_cvtepi32_ps(
                    _mm256_set_m128i(dots_high, dots_low));
                const auto activation_sum = static_cast<float>(
                    static_cast<int>(block_activation.sums[group * 2])
                    + static_cast<int>(block_activation.sums[group * 2 + 1]));
                const auto correction = _mm256_sub_ps(
                    _mm256_mul_ps(dots, scale_vector),
                    _mm256_mul_ps(_mm256_set1_ps(activation_sum), minimum_vector));
                auto* totals = batch_totals.data() + batch * 8;
                _mm256_storeu_ps(totals, _mm256_add_ps(
                    _mm256_loadu_ps(totals),
                    _mm256_mul_ps(_mm256_set1_ps(block_activation.d), correction)));
            }
        }
    }
    for (std::size_t batch = 0; batch < batches; ++batch) {
        std::copy_n(
            batch_totals.begin() + static_cast<std::ptrdiff_t>(batch * 8), lanes,
            output.begin() + batch * rows + first_row);
    }
}

// Run 04 (2026-08-04) iteration 05: the Q6_K four-row decode kernel.
//
// Q6_K had no rows4 kernel at all - integer_quantized_r4_tile_dot special-cased
// Q4_K and dropped Q6_K onto the generic per-lane loop, which calls
// dot_s8_s8_16 once per lane per group: 4 lanes x 16 groups = 64 separate
// horizontal reduction trees per 256-weight block, each ending in a
// cvtsi128_si32 that stalls the vector pipe. Q6_K is 39% of decode matmul time
// on llama-3.2-3B Q4_K_M, so this was the largest un-vectorised surface left.
//
// Here the four lanes share one reduction: dot_s8_s8_16_quads keeps the
// per-lane partials in vector form and three hadds transpose-reduce all four
// rows at once. The group scale is applied in int32 (native s8 scale x a dot
// bounded by 16*127*127, so 16 groups stay under 524M against a 2.1B ceiling),
// leaving exactly one float multiply per block per lane instead of sixteen.
// Iteration 17: the Q6_K tile kernel. Two rows per maddubs, group-major scales,
// and the reduction recombined with one unpacklo. See q6k_tile_* for layout.
__attribute__((target("avx2")))
void integer_q6k_tile_dot(
    const QuantizedTensorView& view,
    std::size_t first_row,
    std::span<const Q8KActivationBlock> activations,
    std::span<float> output
) {
    const auto rows = static_cast<std::size_t>(view.tensor->dimensions[1]);
    const auto columns = static_cast<std::size_t>(view.tensor->dimensions[0]);
    const auto blocks_per_row = columns / 256;
    const auto lanes = std::min<std::size_t>(4, rows - first_row);
    const auto tile = first_row / 4;
    const auto ones = _mm256_set1_epi16(1);
    auto totals = _mm_setzero_ps();
    for (std::size_t block_index = 0; block_index < blocks_per_row; ++block_index) {
        const auto* tile_block = view.r4_bytes.data()
            + (tile * blocks_per_row + block_index) * q6k_tile_bytes;
        const auto& block_activation = activations[block_index];
        const auto* scale_base = tile_block + q6k_tile_scale_offset;
        const auto* weight_base = tile_block + q6k_tile_weight_offset;
        auto accumulated = _mm_setzero_si128();
        for (std::size_t group = 0; group < 16; ++group) {
            const auto input = _mm_loadu_si128(reinterpret_cast<const __m128i*>(
                block_activation.values.data() + group * 16));
            const auto wide = _mm256_broadcastsi128_si256(input);
            const auto w01 = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(weight_base + group * 64));
            const auto w23 = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(weight_base + group * 64 + 32));
            const auto d01 = _mm256_madd_epi16(_mm256_maddubs_epi16(
                _mm256_abs_epi8(w01), _mm256_sign_epi8(wide, w01)), ones);
            const auto d23 = _mm256_madd_epi16(_mm256_maddubs_epi16(
                _mm256_abs_epi8(w23), _mm256_sign_epi8(wide, w23)), ones);
            // hadd works within each 128-bit half, so two of them leave
            // [A,C,A,C | B,D,B,D]; unpacklo against the extracted high half
            // recombines the four row totals into [A,B,C,D].
            const auto pairs = _mm256_hadd_epi32(d01, d23);
            const auto folded = _mm256_hadd_epi32(pairs, pairs);
            const auto dots = _mm_unpacklo_epi32(
                _mm256_castsi256_si128(folded),
                _mm256_extracti128_si256(folded, 1));
            std::int32_t packed_scales{};
            std::memcpy(&packed_scales, scale_base + group * 4, sizeof(packed_scales));
            const auto scales = _mm_cvtepi8_epi32(_mm_cvtsi32_si128(packed_scales));
            accumulated = _mm_add_epi32(accumulated, _mm_mullo_epi32(dots, scales));
        }
        totals = _mm_add_ps(
            totals,
            _mm_mul_ps(
                _mm_mul_ps(
                    _mm_loadu_ps(reinterpret_cast<const float*>(
                        tile_block + q6k_tile_d_offset)),
                    _mm_set1_ps(block_activation.d)),
                _mm_cvtepi32_ps(accumulated)));
    }
    alignas(16) std::array<float, 4> reduced{};
    _mm_store_ps(reduced.data(), totals);
    std::copy_n(reduced.begin(), lanes, output.begin());
}

__attribute__((target("avx2")))
void integer_q6k_r4_tile_dot(
    const QuantizedTensorView& view,
    std::size_t first_row,
    std::span<const Q8KActivationBlock> activations,
    std::span<float> output
) {
    const auto rows = static_cast<std::size_t>(view.tensor->dimensions[1]);
    const auto columns = static_cast<std::size_t>(view.tensor->dimensions[0]);
    const auto blocks_per_row = columns / 256;
    const auto lanes = std::min<std::size_t>(4, rows - first_row);
    const auto tile = first_row / 4;
    constexpr std::size_t tile_bytes = 4 * q6k_r4_block_bytes;
    auto totals = _mm_setzero_ps();
    for (std::size_t block_index = 0; block_index < blocks_per_row; ++block_index) {
        const auto* interleaved = view.r4_bytes.data()
            + (tile * blocks_per_row + block_index) * tile_bytes;
        const auto& block_activation = activations[block_index];
        // Trailing lanes of the last tile address zero-filled sidecar bytes, so
        // they contribute a zero dot and a zero d - the same contract the Q4_K
        // rows4 kernel relies on. Only `lanes` results are copied out.
        alignas(16) std::array<float, 4> d_lanes{};
        for (std::size_t lane = 0; lane < 4; ++lane) {
            std::memcpy(
                &d_lanes[lane],
                interleaved + lane * q6k_r4_block_bytes + q6k_r4_d_offset,
                sizeof(float));
        }
        const auto* scale_base = interleaved + q6k_r4_scale_offset;
        const auto* weight_base = interleaved + q6k_r4_weight_offset;
        const bool biased = cpu_q6k_biased_enabled();
        const bool deferred_reduction = cpu_q6k_deferred_reduction_enabled();
        auto accumulated = _mm_setzero_si128();
        auto partial0 = _mm_setzero_si128();
        auto partial1 = _mm_setzero_si128();
        auto partial2 = _mm_setzero_si128();
        auto partial3 = _mm_setzero_si128();
        for (std::size_t group = 0; group < 16; ++group) {
            const auto input = _mm_loadu_si128(reinterpret_cast<const __m128i*>(
                block_activation.values.data() + group * 16));
            const auto w0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(
                weight_base + 0 * q6k_r4_block_bytes + group * 16));
            const auto w1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(
                weight_base + 1 * q6k_r4_block_bytes + group * 16));
            const auto w2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(
                weight_base + 2 * q6k_r4_block_bytes + group * 16));
            const auto w3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(
                weight_base + 3 * q6k_r4_block_bytes + group * 16));
            const auto quads0 = biased ? dot_u8_s8_16_quads(w0, input)
                                       : dot_s8_s8_16_quads(w0, input);
            const auto quads1 = biased ? dot_u8_s8_16_quads(w1, input)
                                       : dot_s8_s8_16_quads(w1, input);
            const auto quads2 = biased ? dot_u8_s8_16_quads(w2, input)
                                       : dot_s8_s8_16_quads(w2, input);
            const auto quads3 = biased ? dot_u8_s8_16_quads(w3, input)
                                       : dot_s8_s8_16_quads(w3, input);
            if (deferred_reduction && !biased) {
                // Run 04 iteration 16: scale the pre-reduction partials and
                // reduce once per block instead of once per group. The three
                // hadd_epi32 that transpose four lanes into four row totals
                // cost 48 horizontal reductions per 256-weight block; applying
                // the group scale to the partials lets all 16 groups share one
                // reduction, leaving 3. No layout change - the partials are
                // already per-lane, only the accumulator moves.
                // Range: a partial is <= 4*32*127 = 16256, times |scale| <= 127
                // is 2.06M, times 16 groups is 33M - inside int32.
                partial0 = _mm_add_epi32(partial0, _mm_mullo_epi32(quads0,
                    _mm_set1_epi32(static_cast<int>(static_cast<std::int8_t>(
                        scale_base[0 * q6k_r4_block_bytes + group])))));
                partial1 = _mm_add_epi32(partial1, _mm_mullo_epi32(quads1,
                    _mm_set1_epi32(static_cast<int>(static_cast<std::int8_t>(
                        scale_base[1 * q6k_r4_block_bytes + group])))));
                partial2 = _mm_add_epi32(partial2, _mm_mullo_epi32(quads2,
                    _mm_set1_epi32(static_cast<int>(static_cast<std::int8_t>(
                        scale_base[2 * q6k_r4_block_bytes + group])))));
                partial3 = _mm_add_epi32(partial3, _mm_mullo_epi32(quads3,
                    _mm_set1_epi32(static_cast<int>(static_cast<std::int8_t>(
                        scale_base[3 * q6k_r4_block_bytes + group])))));
                continue;
            }
            auto dots = _mm_hadd_epi32(
                _mm_hadd_epi32(quads0, quads1), _mm_hadd_epi32(quads2, quads3));
            if (biased) {
                // Undo the +32 bias: 32 * (sum of this group's activations).
                dots = _mm_sub_epi32(dots, _mm_set1_epi32(
                    32 * static_cast<int>(block_activation.sums[group])));
            }
            const auto scales = _mm_set_epi32(
                static_cast<int>(static_cast<std::int8_t>(
                    scale_base[3 * q6k_r4_block_bytes + group])),
                static_cast<int>(static_cast<std::int8_t>(
                    scale_base[2 * q6k_r4_block_bytes + group])),
                static_cast<int>(static_cast<std::int8_t>(
                    scale_base[1 * q6k_r4_block_bytes + group])),
                static_cast<int>(static_cast<std::int8_t>(
                    scale_base[0 * q6k_r4_block_bytes + group])));
            accumulated = _mm_add_epi32(
                accumulated, _mm_mullo_epi32(dots, scales));
        }
        if (deferred_reduction && !biased) {
            accumulated = _mm_hadd_epi32(
                _mm_hadd_epi32(partial0, partial1),
                _mm_hadd_epi32(partial2, partial3));
        }
        totals = _mm_add_ps(
            totals,
            _mm_mul_ps(
                _mm_mul_ps(_mm_loadu_ps(d_lanes.data()),
                           _mm_set1_ps(block_activation.d)),
                _mm_cvtepi32_ps(accumulated)));
    }
    alignas(16) std::array<float, 4> reduced{};
    _mm_store_ps(reduced.data(), totals);
    std::copy_n(reduced.begin(), lanes, output.begin());
}

__attribute__((target("avx2")))
void integer_quantized_r4_tile_dot(
    const QuantizedTensorView& view,
    std::size_t first_row,
    std::span<const Q8KActivationBlock> activations,
    std::span<float> output,
    int rows4_mode
) {
    if (view.tensor->type == 12 && view.r4_lanes == 8) {
        // Iteration 09: the lane-wise payload has no horizontal reduction, so
        // it wins outright when enabled and there is no VNNI variant to pick
        // between - dpbusd would land in int32 and lose the free pair-add that
        // madd_epi16 gives against the duplicated scale.
        if (cpu_rows8_lanewise_enabled()) {
            integer_q4k_r8l_tile_dot(view, first_row, activations, output);
            return;
        }
        // One selector, highest ISA first: vnni-rows8 -> avx2-rows8.
        if (avx512_vnni_available()) {
            integer_q4k_r8_tile_dot_vnni(view, first_row, activations, output);
        } else {
            integer_q4k_r8_tile_dot(view, first_row, activations, output);
        }
        return;
    }
    const auto type = view.tensor->type;
    const auto rows = static_cast<std::size_t>(view.tensor->dimensions[1]);
    const auto columns = static_cast<std::size_t>(view.tensor->dimensions[0]);
    const auto layout = tensor_layout(type);
    const auto r4_block_bytes = cpu_r4_block_bytes(type);
    const auto blocks_per_row = columns / layout.block_elements;
    const auto lanes = std::min<std::size_t>(4, rows - first_row);
    const bool use_rows4 = rows4_mode < 0
        ? read_cpu_rows4_enabled()
        : rows4_mode != 0;
    // Phase 4 fix: this used to require !avx512_vnni_available(), so enabling
    // VNNI silently dropped the R4 layout onto the slow generic per-lane loop
    // (part of iteration 005's regression). The rows4 kernel is the right R4
    // path regardless of the VNNI gate.
    if (type == 14 && use_rows4 && cpu_q6k_rows4_enabled()) {
        if (cpu_q6k_tile_enabled()) {
            integer_q6k_tile_dot(view, first_row, activations, output);
        } else {
            integer_q6k_r4_tile_dot(view, first_row, activations, output);
        }
        return;
    }
    const bool prefetch = cpu_prefetch_r4_enabled();
    if (type == 12 && use_rows4) {
        const auto tile = first_row / 4;
        constexpr std::size_t r4_q4k_tile_bytes = 4 * 192;
        auto totals = _mm_setzero_ps();
        for (std::size_t block_index = 0; block_index < blocks_per_row; ++block_index) {
            const auto* interleaved = view.r4_bytes.data()
                + (tile * blocks_per_row + block_index) * r4_q4k_tile_bytes;
            if (prefetch && block_index + 1 < blocks_per_row) {
                prefetch_block(interleaved + r4_q4k_tile_bytes, r4_q4k_tile_bytes);
            }
            const auto& block_activation = activations[block_index];
            for (std::size_t group = 0; group < 8; ++group) {
                __m256i weights[4]{};
                std::array<float, 4> scales{};
                std::array<float, 4> minima{};
                for (std::size_t lane = 0; lane < 4; ++lane) {
                    const auto* block = interleaved + lane * 192;
                    const auto* block_scales = reinterpret_cast<const float*>(block);
                    weights[lane] = unpack_q4_nibbles(
                        block + 64 + (group / 2) * 32, (group & 1U) != 0);
                    scales[lane] = block_scales[group];
                    minima[lane] = block_scales[8 + group];
                }
                const auto input = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
                    block_activation.values.data() + group * 32));
                const auto integer_dots = dot_u8_s8_rows4_avx2(
                    weights[0], weights[1], weights[2], weights[3], input);
                const auto activation_sum = static_cast<float>(
                    static_cast<int>(block_activation.sums[group * 2])
                    + static_cast<int>(block_activation.sums[group * 2 + 1]));
                const auto correction = _mm_sub_ps(
                    _mm_mul_ps(_mm_cvtepi32_ps(integer_dots), _mm_loadu_ps(scales.data())),
                    _mm_mul_ps(_mm_set1_ps(activation_sum), _mm_loadu_ps(minima.data())));
                totals = _mm_add_ps(
                    totals, _mm_mul_ps(_mm_set1_ps(block_activation.d), correction));
            }
        }
        alignas(16) std::array<float, 4> reduced{};
        _mm_store_ps(reduced.data(), totals);
        std::copy_n(reduced.begin(), lanes, output.begin());
        return;
    }
    // The generic R4 path carries Q6_K, which is 39% of decode matmul time on
    // llama-3.2-3B Q4_K_M (output.weight plus the Q4_K_M mixed tensors) and
    // streams a 320-byte-per-block sidecar. It needs the same next-tile
    // prefetch the Q4_K R8 kernels got in iteration 02.
    std::array<float, 4> totals{};
    const auto tile = first_row / 4;
    const auto r4_tile_bytes = 4 * r4_block_bytes;
    for (std::size_t block_index = 0; block_index < blocks_per_row; ++block_index) {
        const auto* interleaved = view.r4_bytes.data()
            + (tile * blocks_per_row + block_index) * r4_tile_bytes;
        if (prefetch && block_index + 1 < blocks_per_row) {
            prefetch_block(interleaved + r4_tile_bytes, r4_tile_bytes);
        }
        for (std::size_t lane = 0; lane < lanes; ++lane) {
            totals[lane] += integer_quantized_r4_block_dot(
                type, interleaved + lane * r4_block_bytes, block_index, activations);
        }
    }
    std::copy_n(totals.begin(), lanes, output.begin());
}

__attribute__((target("avx2")))
void integer_q4k_r4_tile_gemm(
    const QuantizedTensorView& view,
    std::size_t first_row,
    std::span<const Q8KActivationBlock> activations,
    std::size_t activation_blocks_per_batch,
    std::size_t batches,
    std::span<float> output
) {
    const auto rows = static_cast<std::size_t>(view.tensor->dimensions[1]);
    const auto columns = static_cast<std::size_t>(view.tensor->dimensions[0]);
    const auto blocks_per_row = columns / 256;
    const auto lanes = std::min<std::size_t>(4, rows - first_row);
    const auto tile = first_row / 4;
    thread_local std::vector<float> totals;
    totals.assign(batches * lanes, 0.0F);
    for (std::size_t block_index = 0; block_index < blocks_per_row; ++block_index) {
        const auto* interleaved = view.r4_bytes.data()
            + (tile * blocks_per_row + block_index) * 4 * 192;
        for (std::size_t group = 0; group < 8; ++group) {
            __m256i weights[4]{};
            std::array<float, 4> scales{};
            std::array<float, 4> minima{};
            for (std::size_t lane = 0; lane < lanes; ++lane) {
                const auto* block = interleaved + lane * 192;
                const auto* block_scales = reinterpret_cast<const float*>(block);
                weights[lane] = unpack_q4_nibbles(
                    block + 64 + (group / 2) * 32, (group & 1U) != 0);
                scales[lane] = block_scales[group];
                minima[lane] = block_scales[8 + group];
            }
            std::size_t batch = 0;
            if (!avx512_vnni_available()) {
                for (; batch + 4 <= batches; batch += 4) {
                    const auto& activation0 = activations[
                        batch * activation_blocks_per_batch + block_index];
                    const auto& activation1 = activations[
                        (batch + 1) * activation_blocks_per_batch + block_index];
                    const auto& activation2 = activations[
                        (batch + 2) * activation_blocks_per_batch + block_index];
                    const auto& activation3 = activations[
                        (batch + 3) * activation_blocks_per_batch + block_index];
                    const auto input0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
                        activation0.values.data() + group * 32));
                    const auto input1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
                        activation1.values.data() + group * 32));
                    const auto input2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
                        activation2.values.data() + group * 32));
                    const auto input3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
                        activation3.values.data() + group * 32));
                    const auto activation_scales = _mm_setr_ps(
                        activation0.d, activation1.d, activation2.d, activation3.d);
                    const auto activation_sums = _mm_setr_ps(
                        static_cast<float>(static_cast<int>(activation0.sums[group * 2])
                            + static_cast<int>(activation0.sums[group * 2 + 1])),
                        static_cast<float>(static_cast<int>(activation1.sums[group * 2])
                            + static_cast<int>(activation1.sums[group * 2 + 1])),
                        static_cast<float>(static_cast<int>(activation2.sums[group * 2])
                            + static_cast<int>(activation2.sums[group * 2 + 1])),
                        static_cast<float>(static_cast<int>(activation3.sums[group * 2])
                            + static_cast<int>(activation3.sums[group * 2 + 1])));
                    for (std::size_t lane = 0; lane < lanes; ++lane) {
                        const auto integer_dots = dot_u8_s8_batch4_avx2(
                            weights[lane], input0, input1, input2, input3);
                        const auto correction = _mm_sub_ps(
                            _mm_mul_ps(_mm_cvtepi32_ps(integer_dots), _mm_set1_ps(scales[lane])),
                            _mm_mul_ps(activation_sums, _mm_set1_ps(minima[lane])));
                        auto accumulated = _mm_loadu_ps(totals.data() + lane * batches + batch);
                        accumulated = _mm_add_ps(
                            accumulated, _mm_mul_ps(activation_scales, correction));
                        _mm_storeu_ps(totals.data() + lane * batches + batch, accumulated);
                    }
                }
            }
            for (; batch < batches; ++batch) {
                const auto& activation = activations[
                    batch * activation_blocks_per_batch + block_index];
                const auto input = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
                    activation.values.data() + group * 32));
                const auto activation_sum = static_cast<int>(activation.sums[group * 2])
                    + static_cast<int>(activation.sums[group * 2 + 1]);
                for (std::size_t lane = 0; lane < lanes; ++lane) {
                    // Pure AVX2: the cross-target VNNI dispatch that was here
                    // segfaulted (vector args across the target ABI boundary).
                    const auto integer_dot = dot_u8_s8_vectors_avx2(weights[lane], input);
                    totals[lane * batches + batch] += activation.d * (
                        scales[lane] * static_cast<float>(integer_dot)
                        - minima[lane] * static_cast<float>(activation_sum));
                }
            }
        }
    }
    for (std::size_t batch = 0; batch < batches; ++batch) {
        for (std::size_t lane = 0; lane < lanes; ++lane) {
            output[batch * rows + first_row + lane] = totals[lane * batches + batch];
        }
    }
}

__attribute__((target("avx2")))
void integer_q6k_r4_tile_gemm(
    const QuantizedTensorView& view,
    std::size_t first_row,
    std::span<const Q8KActivationBlock> activations,
    std::size_t activation_blocks_per_batch,
    std::size_t batches,
    std::span<float> output
) {
    const auto rows = static_cast<std::size_t>(view.tensor->dimensions[1]);
    const auto columns = static_cast<std::size_t>(view.tensor->dimensions[0]);
    const auto blocks_per_row = columns / 256;
    const auto lanes = std::min<std::size_t>(4, rows - first_row);
    const auto tile = first_row / 4;
    thread_local std::vector<float> totals;
    totals.assign(batches * lanes, 0.0F);
    for (std::size_t block_index = 0; block_index < blocks_per_row; ++block_index) {
        const auto* interleaved = view.r4_bytes.data()
            + (tile * blocks_per_row + block_index) * 4 * q6k_r4_block_bytes;
        for (std::size_t group = 0; group < 16; ++group) {
            __m128i weights[4]{};
            std::array<float, 4> scales{};
            for (std::size_t lane = 0; lane < lanes; ++lane) {
                const auto* block = interleaved + lane * q6k_r4_block_bytes;
                // Iteration 04 stores the native int8 group scale plus a
                // per-block f32 d; reconstitute the old fused scale here so
                // the batched verifier keeps its existing arithmetic order.
                float block_d{};
                std::memcpy(&block_d, block + q6k_r4_d_offset, sizeof(block_d));
                const auto* raw_scales = reinterpret_cast<const std::int8_t*>(
                    block + q6k_r4_scale_offset);
                weights[lane] = _mm_loadu_si128(reinterpret_cast<const __m128i*>(
                    block + q6k_r4_weight_offset + group * 16));
                scales[lane] = block_d * static_cast<float>(raw_scales[group]);
            }
            std::size_t batch = 0;
            for (; batch + 4 <= batches; batch += 4) {
                const auto& activation0 = activations[
                    batch * activation_blocks_per_batch + block_index];
                const auto& activation1 = activations[
                    (batch + 1) * activation_blocks_per_batch + block_index];
                const auto& activation2 = activations[
                    (batch + 2) * activation_blocks_per_batch + block_index];
                const auto& activation3 = activations[
                    (batch + 3) * activation_blocks_per_batch + block_index];
                const auto input0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(
                    activation0.values.data() + group * 16));
                const auto input1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(
                    activation1.values.data() + group * 16));
                const auto input2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(
                    activation2.values.data() + group * 16));
                const auto input3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(
                    activation3.values.data() + group * 16));
                const auto activation_scales = _mm_setr_ps(
                    activation0.d, activation1.d, activation2.d, activation3.d);
                for (std::size_t lane = 0; lane < lanes; ++lane) {
                    const auto dots = dot_s8_s8_16_batch4(
                        weights[lane], input0, input1, input2, input3);
                    auto accumulated = _mm_loadu_ps(totals.data() + lane * batches + batch);
                    accumulated = _mm_add_ps(
                        accumulated,
                        _mm_mul_ps(
                            activation_scales,
                            _mm_mul_ps(_mm_cvtepi32_ps(dots), _mm_set1_ps(scales[lane]))));
                    _mm_storeu_ps(totals.data() + lane * batches + batch, accumulated);
                }
            }
            for (; batch < batches; ++batch) {
                const auto& activation = activations[
                    batch * activation_blocks_per_batch + block_index];
                const auto input = _mm_loadu_si128(reinterpret_cast<const __m128i*>(
                    activation.values.data() + group * 16));
                for (std::size_t lane = 0; lane < lanes; ++lane) {
                    totals[lane * batches + batch] += scales[lane] * activation.d
                        * static_cast<float>(dot_s8_s8_16_vectors(weights[lane], input));
                }
            }
        }
    }
    for (std::size_t batch = 0; batch < batches; ++batch) {
        for (std::size_t lane = 0; lane < lanes; ++lane) {
            output[batch * rows + first_row + lane] = totals[lane * batches + batch];
        }
    }
}

__attribute__((target("avx2,fma")))
float horizontal_sum(__m256 accumulated) {
    alignas(32) float lanes[8];
    _mm256_store_ps(lanes, accumulated);
    return lanes[0] + lanes[1] + lanes[2] + lanes[3]
        + lanes[4] + lanes[5] + lanes[6] + lanes[7];
}

__attribute__((target("avx2,fma")))
__m256 accumulate_u4_group(
    __m256 accumulated,
    const std::byte* packed,
    const float* input,
    bool high,
    float scale,
    float minimum
) {
    const auto nibble_mask = _mm_set1_epi8(0x0f);
    const auto scale_values = _mm256_set1_ps(scale);
    const auto minimum_values = _mm256_set1_ps(minimum);
    for (std::size_t offset = 0; offset < 32; offset += 8) {
        auto bytes = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(packed + offset));
        if (high) bytes = _mm_srli_epi16(bytes, 4);
        bytes = _mm_and_si128(bytes, nibble_mask);
        const auto integers = _mm256_cvtepu8_epi32(bytes);
        const auto values = _mm256_fmsub_ps(
            _mm256_cvtepi32_ps(integers), scale_values, minimum_values);
        accumulated = _mm256_fmadd_ps(
            _mm256_loadu_ps(input + offset), values, accumulated);
    }
    return accumulated;
}

__attribute__((target("avx2,fma")))
__m256 avx2_q4_k_block_accumulate(const std::byte* block, const float* input, __m256 accumulated) {
    const float super_scale = fp16_to_float(load_u16(block));
    const float super_minimum = fp16_to_float(load_u16(block + 2));
    const auto* scales = block + 4;
    const auto* quant = block + 16;
    for (std::size_t group = 0; group < 8; ++group) {
        std::uint8_t scale{};
        std::uint8_t minimum{};
        decode_scale_min(scales, group, scale, minimum);
        accumulated = accumulate_u4_group(
            accumulated,
            quant + (group / 2) * 32,
            input + group * 32,
            (group & 1U) != 0,
            super_scale * static_cast<float>(scale),
            super_minimum * static_cast<float>(minimum)
        );
    }
    return accumulated;
}

__attribute__((target("avx2,fma")))
__m256 avx2_q4_0_block_accumulate(const std::byte* block, const float* input, __m256 accumulated) {
    const float scale = fp16_to_float(load_u16(block));
    const auto* quant = block + 2;
    const auto scale_values = _mm256_set1_ps(scale);
    const auto bias = _mm256_set1_ps(8.0F);
    // Q4_0 owns exactly 16 packed bytes. Each byte stores one low-nibble
    // value for columns 0..15 and one high-nibble value for columns 16..31.
    for (std::size_t half = 0; half < 2; ++half) {
        const auto offset = half * 8;
        const auto packed8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(quant + offset));
        const auto low = _mm_and_si128(packed8, _mm_set1_epi8(0x0f));
        const auto high = _mm_and_si128(_mm_srli_epi16(packed8, 4), _mm_set1_epi8(0x0f));
        const auto low_values = _mm256_mul_ps(
            _mm256_sub_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(low)), bias),
            scale_values
        );
        const auto high_values = _mm256_mul_ps(
            _mm256_sub_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(high)), bias),
            scale_values
        );
        accumulated = _mm256_fmadd_ps(
            _mm256_loadu_ps(input + offset), low_values, accumulated);
        accumulated = _mm256_fmadd_ps(
            _mm256_loadu_ps(input + 16 + offset), high_values, accumulated);
    }
    return accumulated;
}

__attribute__((target("avx2,fma")))
__m256 accumulate_q6_values(
    __m256 accumulated,
    __m128i quantized,
    const float* input,
    float scale
) {
    const auto integers = _mm256_sub_epi32(
        _mm256_cvtepu8_epi32(quantized),
        _mm256_set1_epi32(32)
    );
    const auto values = _mm256_mul_ps(
        _mm256_cvtepi32_ps(integers),
        _mm256_set1_ps(scale)
    );
    return _mm256_fmadd_ps(_mm256_loadu_ps(input), values, accumulated);
}

__attribute__((target("avx2,fma")))
__m256 avx2_q6_k_block_accumulate(const std::byte* block, const float* input, __m256 accumulated) {
    const auto* low = block;
    const auto* high = block + 128;
    const auto* scales = reinterpret_cast<const std::int8_t*>(block + 192);
    const float super_scale = fp16_to_float(load_u16(block + 208));
    const auto low_mask = _mm_set1_epi8(0x0f);
    const auto high_mask = _mm_set1_epi8(0x03);
    for (std::size_t chunk = 0; chunk < 256; chunk += 128) {
        const auto* chunk_low = low + chunk / 2;
        const auto* chunk_high = high + chunk / 4;
        const auto* chunk_scales = scales + chunk / 16;
        for (std::size_t item = 0; item < 32; item += 8) {
            const auto scale_index = item / 16;
            const auto low_first = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(chunk_low + item));
            const auto low_second = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(chunk_low + 32 + item));
            const auto upper = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(chunk_high + item));
            const auto q1 = _mm_or_si128(
                _mm_and_si128(low_first, low_mask),
                _mm_slli_epi16(_mm_and_si128(upper, high_mask), 4)
            );
            const auto q2 = _mm_or_si128(
                _mm_and_si128(low_second, low_mask),
                _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(upper, 2), high_mask), 4)
            );
            const auto q3 = _mm_or_si128(
                _mm_and_si128(_mm_srli_epi16(low_first, 4), low_mask),
                _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(upper, 4), high_mask), 4)
            );
            const auto q4 = _mm_or_si128(
                _mm_and_si128(_mm_srli_epi16(low_second, 4), low_mask),
                _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(upper, 6), high_mask), 4)
            );
            accumulated = accumulate_q6_values(
                accumulated, q1, input + chunk + item,
                super_scale * static_cast<float>(chunk_scales[scale_index]));
            accumulated = accumulate_q6_values(
                accumulated, q2, input + chunk + 32 + item,
                super_scale * static_cast<float>(chunk_scales[scale_index + 2]));
            accumulated = accumulate_q6_values(
                accumulated, q3, input + chunk + 64 + item,
                super_scale * static_cast<float>(chunk_scales[scale_index + 4]));
            accumulated = accumulate_q6_values(
                accumulated, q4, input + chunk + 96 + item,
                super_scale * static_cast<float>(chunk_scales[scale_index + 6]));
        }
    }
    return accumulated;
}

__attribute__((target("avx2,fma")))
float avx2_quantized_row_dot(
    std::uint32_t type,
    const std::byte* row,
    std::size_t blocks,
    std::size_t block_bytes,
    std::size_t block_elements,
    const float* input
) {
    __m256 accumulated = _mm256_setzero_ps();
    for (std::size_t block_index = 0; block_index < blocks; ++block_index) {
        const auto* block = row + block_index * block_bytes;
        const auto* block_input = input + block_index * block_elements;
        if (type == 12) {
            accumulated = avx2_q4_k_block_accumulate(block, block_input, accumulated);
        } else if (type == 14) {
            accumulated = avx2_q6_k_block_accumulate(block, block_input, accumulated);
        } else {
            accumulated = avx2_q4_0_block_accumulate(block, block_input, accumulated);
        }
    }
    return horizontal_sum(accumulated);
}

#endif

TensorLayout layout_for(std::uint32_t type) {
    switch (type) {
        case 0: return {1, 4};       // F32
        case 1: return {1, 2};       // F16
        case 2: return {32, 18};     // Q4_0
        case 8: return {32, 34};     // Q8_0
        case 12: return {256, 144};  // Q4_K
        case 13: return {256, 176};  // Q5_K
        case 14: return {256, 210};  // Q6_K
        case 20: return {32, 18};    // IQ4_NL
        case 23: return {256, 136};  // IQ4_XS
        case 30: return {1, 2};      // BF16
        default: throw std::invalid_argument("unsupported Caterpillar CPU tensor type: " + std::to_string(type));
    }
}

std::uint16_t load_u16(const std::byte* data) {
    return static_cast<std::uint16_t>(std::to_integer<std::uint8_t>(data[0]))
        | (static_cast<std::uint16_t>(std::to_integer<std::uint8_t>(data[1])) << 8U);
}

float fp16_to_float(std::uint16_t value) {
    const std::uint32_t sign = static_cast<std::uint32_t>(value & 0x8000U) << 16U;
    std::uint32_t exponent = (value >> 10U) & 0x1FU;
    std::uint32_t mantissa = value & 0x03FFU;
    std::uint32_t output{};
    if (exponent == 0) {
        if (mantissa == 0) {
            output = sign;
        } else {
            int shift = 0;
            while ((mantissa & 0x0400U) == 0) {
                mantissa <<= 1U;
                ++shift;
            }
            mantissa &= 0x03FFU;
            const auto adjusted = static_cast<std::uint32_t>(127 - 14 - shift);
            output = sign | (adjusted << 23U) | (mantissa << 13U);
        }
    } else if (exponent == 0x1FU) {
        output = sign | 0x7F800000U | (mantissa << 13U);
    } else {
        exponent += 127U - 15U;
        output = sign | (exponent << 23U) | (mantissa << 13U);
    }
    return std::bit_cast<float>(output);
}

float bf16_to_float(std::uint16_t value) {
    return std::bit_cast<float>(static_cast<std::uint32_t>(value) << 16U);
}

void decode_scale_min(const std::byte* packed, std::size_t group, std::uint8_t& scale, std::uint8_t& minimum) {
    const auto at = [&](std::size_t index) { return std::to_integer<std::uint8_t>(packed[index]); };
    if (group < 4) {
        scale = at(group) & 0x3FU;
        minimum = at(group + 4) & 0x3FU;
    } else {
        scale = static_cast<std::uint8_t>((at(group + 4) & 0x0FU) | ((at(group - 4) >> 6U) << 4U));
        minimum = static_cast<std::uint8_t>((at(group + 4) >> 4U) | ((at(group) >> 6U) << 4U));
    }
}

void dequantize_q4_0(const std::byte* block, float* output) {
    const float scale = fp16_to_float(load_u16(block));
    const auto* quant = block + 2;
    for (std::size_t index = 0; index < 16; ++index) {
        const auto packed = std::to_integer<std::uint8_t>(quant[index]);
        output[index] = scale * static_cast<float>(static_cast<int>(packed & 0x0FU) - 8);
        output[index + 16] = scale * static_cast<float>(static_cast<int>(packed >> 4U) - 8);
    }
}

void dequantize_q8_0(const std::byte* block, float* output) {
    const float scale = fp16_to_float(load_u16(block));
    for (std::size_t index = 0; index < 32; ++index) {
        const auto value = static_cast<std::int8_t>(std::to_integer<std::uint8_t>(block[2 + index]));
        output[index] = scale * static_cast<float>(value);
    }
}

void dequantize_q4_k(const std::byte* block, float* output) {
    const float super_scale = fp16_to_float(load_u16(block));
    const float super_minimum = fp16_to_float(load_u16(block + 2));
    const auto* scales = block + 4;
    const auto* quant = block + 16;
    for (std::size_t group = 0; group < 8; ++group) {
        std::uint8_t scale{};
        std::uint8_t minimum{};
        decode_scale_min(scales, group, scale, minimum);
        const std::size_t pair = group / 2;
        const bool high = (group % 2) != 0;
        for (std::size_t index = 0; index < 32; ++index) {
            const auto packed = std::to_integer<std::uint8_t>(quant[pair * 32 + index]);
            const auto value = high ? (packed >> 4U) : (packed & 0x0FU);
            output[group * 32 + index] = super_scale * static_cast<float>(scale) * static_cast<float>(value)
                - super_minimum * static_cast<float>(minimum);
        }
    }
}

void dequantize_q5_k(const std::byte* block, float* output) {
    const float super_scale = fp16_to_float(load_u16(block));
    const float super_minimum = fp16_to_float(load_u16(block + 2));
    const auto* scales = block + 4;
    const auto* high_bits = block + 16;
    const auto* quant = block + 48;
    for (std::size_t group = 0; group < 8; ++group) {
        std::uint8_t scale{};
        std::uint8_t minimum{};
        decode_scale_min(scales, group, scale, minimum);
        const auto* packed_group = quant + (group / 2) * 32;
        const bool high_nibble = (group & 1U) != 0;
        for (std::size_t index = 0; index < 32; ++index) {
            const auto packed = std::to_integer<std::uint8_t>(packed_group[index]);
            const auto low_four = high_nibble ? (packed >> 4U) : (packed & 0x0FU);
            const auto high_one = (std::to_integer<std::uint8_t>(high_bits[index]) >> group) & 1U;
            const auto value = low_four | (high_one << 4U);
            output[group * 32 + index] =
                super_scale * static_cast<float>(scale) * static_cast<float>(value)
                - super_minimum * static_cast<float>(minimum);
        }
    }
}

void dequantize_q6_k(const std::byte* block, float* output) {
    const auto* low = block;
    const auto* high = block + 128;
    const auto* scales = block + 192;
    const float scale = fp16_to_float(load_u16(block + 208));
    for (std::size_t half = 0; half < 2; ++half) {
        const auto* low_half = low + half * 64;
        const auto* high_half = high + half * 32;
        const auto* scale_half = scales + half * 8;
        for (std::size_t index = 0; index < 32; ++index) {
            const auto lo0 = std::to_integer<std::uint8_t>(low_half[index]);
            const auto lo1 = std::to_integer<std::uint8_t>(low_half[index + 32]);
            const auto hi = std::to_integer<std::uint8_t>(high_half[index]);
            const int values[4] = {
                static_cast<int>((lo0 & 0x0FU) | ((hi & 0x03U) << 4U)) - 32,
                static_cast<int>((lo1 & 0x0FU) | (((hi >> 2U) & 0x03U) << 4U)) - 32,
                static_cast<int>((lo0 >> 4U) | (((hi >> 4U) & 0x03U) << 4U)) - 32,
                static_cast<int>((lo1 >> 4U) | (((hi >> 6U) & 0x03U) << 4U)) - 32,
            };
            for (std::size_t quarter = 0; quarter < 4; ++quarter) {
                const auto signed_scale = static_cast<std::int8_t>(
                    std::to_integer<std::uint8_t>(scale_half[(index / 16) + quarter * 2])
                );
                output[half * 128 + quarter * 32 + index] =
                    scale * static_cast<float>(signed_scale) * static_cast<float>(values[quarter]);
            }
        }
    }
}

constexpr std::array<std::int8_t, 16> k_iq4_nl_values{
    -127, -104, -83, -65, -49, -35, -22, -10,
       1,   13,  25,  38,  53,  69,  89, 113,
};

void dequantize_iq4_nl(const std::byte* block, float* output) {
    const float scale = fp16_to_float(load_u16(block));
    for (std::size_t index = 0; index < 16; ++index) {
        const auto packed = std::to_integer<std::uint8_t>(block[2 + index]);
        output[index] = scale * static_cast<float>(k_iq4_nl_values[packed & 0x0fU]);
        output[index + 16] = scale * static_cast<float>(k_iq4_nl_values[packed >> 4U]);
    }
}

void dequantize_iq4_xs(const std::byte* block, float* output) {
    const float super_scale = fp16_to_float(load_u16(block));
    const auto high_scale_bits = load_u16(block + 2);
    const auto* low_scale_bits = block + 4;
    const auto* quant = block + 8;
    for (std::size_t group = 0; group < 8; ++group) {
        const auto packed_low = std::to_integer<std::uint8_t>(low_scale_bits[group / 2]);
        const auto low = (packed_low >> (4U * (group % 2))) & 0x0fU;
        const auto high = (high_scale_bits >> (2U * group)) & 0x03U;
        const int signed_scale = static_cast<int>(low | (high << 4U)) - 32;
        const float scale = super_scale * static_cast<float>(signed_scale);
        const auto* group_quant = quant + group * 16;
        for (std::size_t index = 0; index < 16; ++index) {
            const auto packed = std::to_integer<std::uint8_t>(group_quant[index]);
            output[group * 32 + index] =
                scale * static_cast<float>(k_iq4_nl_values[packed & 0x0fU]);
            output[group * 32 + index + 16] =
                scale * static_cast<float>(k_iq4_nl_values[packed >> 4U]);
        }
    }
}

void decode_block(std::uint32_t type, const std::byte* block, float* destination) {
    switch (type) {
        case 0: std::memcpy(destination, block, sizeof(float)); break;
        case 1: destination[0] = fp16_to_float(load_u16(block)); break;
        case 2: dequantize_q4_0(block, destination); break;
        case 8: dequantize_q8_0(block, destination); break;
        case 12: dequantize_q4_k(block, destination); break;
        case 13: dequantize_q5_k(block, destination); break;
        case 14: dequantize_q6_k(block, destination); break;
        case 20: dequantize_iq4_nl(block, destination); break;
        case 23: dequantize_iq4_xs(block, destination); break;
        case 30: destination[0] = bf16_to_float(load_u16(block)); break;
        default: throw std::logic_error("unreachable tensor type");
    }
}

}  // namespace

std::uint64_t cpu_weight_bytes_streamed() noexcept {
    return g_weight_bytes_streamed.load(std::memory_order_relaxed);
}

bool cpu_rows8_enabled() noexcept {
    static const bool enabled = read_cpu_rows8_enabled_uncached();
    return enabled;
}

std::uint8_t cpu_repack_lanes(std::uint32_t type) noexcept {
    return (type == 12 && cpu_rows8_enabled()) ? 8 : 4;
}

std::size_t cpu_r4_sidecar_bytes(const GgufTensorInfo& tensor) noexcept {
    if (tensor.dimensions.size() != 2) return 0;
    const auto type = tensor.type;
    if (type != 2 && type != 8 && type != 12 && type != 13 && type != 14) return 0;
    std::size_t block_elements = 0;
    switch (type) {
        case 2: case 8: block_elements = 32; break;
        default: block_elements = 256; break;
    }
    const auto columns = static_cast<std::size_t>(tensor.dimensions[0]);
    const auto rows = static_cast<std::size_t>(tensor.dimensions[1]);
    if (columns == 0 || rows == 0 || columns % block_elements != 0) return 0;
    const auto blocks_per_row = columns / block_elements;
    if (cpu_repack_lanes(type) == 8) {
        return ((rows + 7) / 8) * blocks_per_row * r8_tile_block_bytes;
    }
    std::size_t block_bytes = 0;
    switch (type) {
        case 2: case 8: block_bytes = 36; break;
        case 12: block_bytes = 192; break;
        case 13: block_bytes = 320; break;
        case 14:
            if (cpu_q6k_tile_enabled()) {
                return ((rows + 3) / 4) * blocks_per_row * q6k_tile_bytes;
            }
            block_bytes = q6k_r4_block_bytes;
            break;
        default: return 0;
    }
    return ((rows + 3) / 4) * blocks_per_row * 4 * block_bytes;
}

std::size_t tensor_element_count(const GgufTensorInfo& tensor) {
    std::size_t elements = 1;
    for (const auto dimension : tensor.dimensions) {
        if (dimension == 0 || dimension > std::numeric_limits<std::size_t>::max()
            || elements > std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(dimension)) {
            throw std::overflow_error("invalid tensor dimensions: " + tensor.name);
        }
        elements *= static_cast<std::size_t>(dimension);
    }
    return elements;
}

std::size_t tensor_byte_length(const GgufTensorInfo& tensor) {
    const auto layout = layout_for(tensor.type);
    const auto elements = tensor_element_count(tensor);
    if (elements % layout.block_elements != 0) throw std::invalid_argument("tensor is not aligned to its quant block: " + tensor.name);
    const auto blocks = elements / layout.block_elements;
    if (blocks > std::numeric_limits<std::size_t>::max() / layout.block_bytes) throw std::overflow_error("tensor byte length overflow");
    return blocks * layout.block_bytes;
}

std::vector<float> dequantize_tensor(const GgufTensorInfo& tensor, std::span<const std::byte> bytes) {
    const auto expected = tensor_byte_length(tensor);
    if (bytes.size() != expected) throw std::invalid_argument("tensor byte span does not match the GGUF layout");
    const auto elements = tensor_element_count(tensor);
    std::vector<float> output(elements);
    const auto layout = layout_for(tensor.type);
    for (std::size_t block_index = 0; block_index < elements / layout.block_elements; ++block_index) {
        const auto* block = bytes.data() + block_index * layout.block_bytes;
        float* destination = output.data() + block_index * layout.block_elements;
        decode_block(tensor.type, block, destination);
    }
    return output;
}

TensorLayout tensor_layout(std::uint32_t type) {
    return layout_for(type);
}

bool cpu_integer_dot_enabled() noexcept {
    return cpu_integer_dot_requested();
}

std::vector<std::byte> repack_tensor_r4(
    const GgufTensorInfo& tensor,
    std::span<const std::byte> bytes,
    std::uint8_t forced_lanes
) {
    if (tensor.dimensions.size() != 2) {
        throw std::invalid_argument("R4 repack requires a rank-two tensor: " + tensor.name);
    }
    if (bytes.size() != tensor_byte_length(tensor)) {
        throw std::invalid_argument("R4 repack byte span mismatch: " + tensor.name);
    }
    const auto layout = layout_for(tensor.type);
    const auto columns = static_cast<std::size_t>(tensor.dimensions[0]);
    const auto rows = static_cast<std::size_t>(tensor.dimensions[1]);
    if (columns % layout.block_elements != 0) {
        throw std::invalid_argument("R4 repack row is not block aligned: " + tensor.name);
    }
    const auto blocks_per_row = columns / layout.block_elements;
    const auto lanes = forced_lanes != 0 ? forced_lanes : cpu_repack_lanes(tensor.type);
    if (lanes == 8 && tensor.type != 12) {
        throw std::invalid_argument("R8 repack is Q4_K-only: " + tensor.name);
    }
    if (lanes == 8) {
        // Phase 3 R8 layout (Q4_K only): group-major SoA header (f32 d/dmin
        // per lane + pre-unpacked 6-bit scales/minima as u8) + native quant
        // payloads. 152 bytes per 256-weight block: -20.8% vs the R4
        // expansion, and the decode hot loop never touches packed scales.
        static_assert(r8_tile_block_bytes == 8 * 148,
            "Q4_K R8 must average exactly 148 bytes per block");
        // Iteration 09: same bytes, different payload order. Read once here so
        // the whole buffer is built in one ordering.
        const bool lanewise = cpu_rows8_lanewise_enabled();
        const auto row_tiles = (rows + 7) / 8;
        std::vector<std::byte> output(
            row_tiles * blocks_per_row * r8_tile_block_bytes, std::byte{0});
        for (std::size_t tile = 0; tile < row_tiles; ++tile) {
            for (std::size_t block = 0; block < blocks_per_row; ++block) {
                auto* destination = output.data()
                    + (tile * blocks_per_row + block) * r8_tile_block_bytes;
                for (std::size_t lane = 0; lane < 8; ++lane) {
                    const auto row = tile * 8 + lane;
                    if (row >= rows) continue;
                    const auto* source = bytes.data()
                        + (row * blocks_per_row + block) * layout.block_bytes;
                    // Carry the GGUF's own f16 d/dmin through untouched; the
                    // kernel widens them once per superblock with cvtph_ps.
                    // Trailing lanes of the last tile keep the zero fill, which
                    // is f16 +0.0 and contributes nothing - the same contract
                    // the f32 header relied on.
                    std::memcpy(destination + r8_d_offset + lane * 2, source, 2);
                    std::memcpy(destination + r8_dmin_offset + lane * 2, source + 2, 2);
                    for (std::size_t group = 0; group < 8; ++group) {
                        std::uint8_t scale{};
                        std::uint8_t minimum{};
                        decode_scale_min(source + 4, group, scale, minimum);
                        destination[r8_scale_offset + group * 8 + lane] =
                            static_cast<std::byte>(scale);
                        destination[r8_minimum_offset + group * 8 + lane] =
                            static_cast<std::byte>(minimum);
                    }
                    if (lanewise) {
                        // Scatter row `lane`'s payload into 4-byte chunks so a
                        // 32-byte load lands one row per 32-bit lane.
                        for (std::size_t chunk = 0; chunk < 32; ++chunk) {
                            std::memcpy(
                                destination + r8_quant_offset + chunk * 32 + lane * 4,
                                source + 16 + chunk * 4,
                                4);
                        }
                    } else {
                        std::memcpy(destination + r8_quant_offset + lane * r8_quant_bytes,
                                    source + 16, r8_quant_bytes);
                    }
                }
            }
        }
        return output;
    }
    if (tensor.type == 14 && forced_lanes == 0 && cpu_q6k_tile_enabled()) {
        // Iteration 17 Q6_K tile: group-major scales plus lane-contiguous
        // weights, so one 256-bit load covers two rows.
        const auto q6_row_tiles = (rows + 3) / 4;
        std::vector<std::byte> output(
            q6_row_tiles * blocks_per_row * q6k_tile_bytes, std::byte{0});
        for (std::size_t tile = 0; tile < q6_row_tiles; ++tile) {
            for (std::size_t block = 0; block < blocks_per_row; ++block) {
                auto* destination = output.data()
                    + (tile * blocks_per_row + block) * q6k_tile_bytes;
                for (std::size_t lane = 0; lane < 4; ++lane) {
                    const auto row = tile * 4 + lane;
                    if (row >= rows) continue;
                    const auto* source = bytes.data()
                        + (row * blocks_per_row + block) * layout.block_bytes;
                    const auto d = fp16_to_float(load_u16(source + 208));
                    std::memcpy(
                        destination + q6k_tile_d_offset + lane * sizeof(float),
                        &d, sizeof(d));
                    const auto* raw_scales =
                        reinterpret_cast<const std::int8_t*>(source + 192);
                    auto* weights = reinterpret_cast<std::int8_t*>(
                        destination + q6k_tile_weight_offset);
                    for (std::size_t group = 0; group < 16; ++group) {
                        destination[q6k_tile_scale_offset + group * 4 + lane] =
                            static_cast<std::byte>(raw_scales[group]);
                        for (std::size_t item = 0; item < 16; ++item) {
                            weights[group * 64 + lane * 16 + item] =
                                static_cast<std::int8_t>(
                                    q6_value_at(source, group * 16 + item));
                        }
                    }
                }
            }
        }
        return output;
    }
    const auto row_tiles = (rows + 3) / 4;
    const auto r4_block_bytes = cpu_r4_block_bytes(tensor.type);
    std::vector<std::byte> output(
        row_tiles * blocks_per_row * 4 * r4_block_bytes, std::byte{0});
    for (std::size_t tile = 0; tile < row_tiles; ++tile) {
        for (std::size_t block = 0; block < blocks_per_row; ++block) {
            for (std::size_t lane = 0; lane < 4; ++lane) {
                const auto row = tile * 4 + lane;
                if (row >= rows) continue;
                const auto source_offset = (row * blocks_per_row + block) * layout.block_bytes;
                const auto destination_offset =
                    ((tile * blocks_per_row + block) * 4 + lane) * r4_block_bytes;
                repack_cpu_block(
                    tensor.type,
                    bytes.data() + source_offset,
                    output.data() + destination_offset);
            }
        }
    }
    return output;
}

const char* cpu_dot_kernel_name() noexcept {
    if (cpu_integer_dot_requested()) {
        if (cpu_rows8_enabled()) {
            // R8L has no VNNI variant: dpbusd lands in int32 and would lose the
            // free pair-add that madd_epi16 gives against the duplicated scale.
            if (cpu_rows8_lanewise_enabled()) return "avx2-integer-q8_k-r8l-lanewise";
            return avx512_vnni_available()
                ? "avx512-vnni-q8_k-r8-rows8"
                : "avx2-integer-q8_k-r8-rows8";
        }
        // The R4 layout is pure AVX2 since Phase 4 removed the per-dot VNNI
        // dispatch; VNNI only exists as the same-target R8 kernel above.
        if (!cpu_rows4_enabled()) return "avx2-integer-q8_k-r4";
        return cpu_selector_cache_enabled()
            ? "avx2-integer-q8_k-r4-rows4-cachedcfg"
            : "avx2-integer-q8_k-r4-rows4";
    }
    return avx2_dot_enabled() ? "avx2-quant-dot" : "portable-scalar-dot";
}

float tensor_row_dot_impl(
    const QuantizedTensorView& view,
    std::size_t row,
    std::span<const float> input,
    bool use_avx2
) {
    if (view.tensor == nullptr) throw std::invalid_argument("quantized tensor view has no directory entry");
    const auto& tensor = *view.tensor;
    if (tensor.dimensions.size() != 2) throw std::invalid_argument("row dot requires a rank-two GGUF tensor: " + tensor.name);
    const auto columns = static_cast<std::size_t>(tensor.dimensions[0]);
    const auto rows = static_cast<std::size_t>(tensor.dimensions[1]);
    if (row >= rows || input.size() != columns) throw std::invalid_argument("row dot shape mismatch: " + tensor.name);
    if (view.bytes.size() != tensor_byte_length(tensor)) throw std::invalid_argument("row dot tensor byte span mismatch: " + tensor.name);
    const auto layout = layout_for(tensor.type);
    if (columns % layout.block_elements != 0) throw std::invalid_argument("matrix row is not quant-block aligned: " + tensor.name);
    std::array<float, 256> decoded{};
    if (layout.block_elements > decoded.size()) throw std::logic_error("tensor block exceeds CPU reference scratch");
    double sum = 0.0;
    const auto blocks_per_row = columns / layout.block_elements;
    if (use_avx2 && (tensor.type == 2 || tensor.type == 12 || tensor.type == 14)) {
#if CAT_HAS_AVX2_DISPATCH
        return avx2_quantized_row_dot(
            tensor.type,
            view.bytes.data() + row * blocks_per_row * layout.block_bytes,
            blocks_per_row,
            layout.block_bytes,
            layout.block_elements,
            input.data()
        );
#endif
    }
    for (std::size_t block_index = 0; block_index < blocks_per_row; ++block_index) {
        const auto byte_offset = (row * blocks_per_row + block_index) * layout.block_bytes;
        const auto* block = view.bytes.data() + byte_offset;
        const auto input_offset = block_index * layout.block_elements;
        const auto accumulate = [&](std::size_t column, float value) {
            sum += static_cast<double>(input[input_offset + column]) * static_cast<double>(value);
        };
        if (tensor.type == 12) {
            const float super_scale = fp16_to_float(load_u16(block));
            const float super_minimum = fp16_to_float(load_u16(block + 2));
            const auto* scales = block + 4;
            const auto* quant = block + 16;
            for (std::size_t chunk = 0; chunk < 256; chunk += 64) {
                const auto group = chunk / 32;
                std::uint8_t scale0{};
                std::uint8_t minimum0{};
                std::uint8_t scale1{};
                std::uint8_t minimum1{};
                decode_scale_min(scales, group, scale0, minimum0);
                decode_scale_min(scales, group + 1, scale1, minimum1);
                const float d0 = super_scale * static_cast<float>(scale0);
                const float m0 = super_minimum * static_cast<float>(minimum0);
                const float d1 = super_scale * static_cast<float>(scale1);
                const float m1 = super_minimum * static_cast<float>(minimum1);
                const auto* chunk_quant = quant + chunk / 2;
                for (std::size_t item = 0; item < 32; ++item) {
                    const auto packed = std::to_integer<std::uint8_t>(chunk_quant[item]);
                    accumulate(chunk + item, d0 * static_cast<float>(packed & 0x0FU) - m0);
                    accumulate(chunk + 32 + item, d1 * static_cast<float>(packed >> 4U) - m1);
                }
            }
        } else if (tensor.type == 14) {
            const auto* low = block;
            const auto* high = block + 128;
            const auto* scales = reinterpret_cast<const std::int8_t*>(block + 192);
            const float d = fp16_to_float(load_u16(block + 208));
            for (std::size_t chunk = 0; chunk < 256; chunk += 128) {
                const auto* chunk_low = low + chunk / 2;
                const auto* chunk_high = high + chunk / 4;
                const auto* chunk_scales = scales + chunk / 16;
                for (std::size_t item = 0; item < 32; ++item) {
                    const auto scale_index = item / 16;
                    const int q1 = static_cast<int>((std::to_integer<std::uint8_t>(chunk_low[item]) & 0x0FU)
                        | ((std::to_integer<std::uint8_t>(chunk_high[item]) & 0x03U) << 4U)) - 32;
                    const int q2 = static_cast<int>((std::to_integer<std::uint8_t>(chunk_low[item + 32]) & 0x0FU)
                        | (((std::to_integer<std::uint8_t>(chunk_high[item]) >> 2U) & 0x03U) << 4U)) - 32;
                    const int q3 = static_cast<int>((std::to_integer<std::uint8_t>(chunk_low[item]) >> 4U)
                        | (((std::to_integer<std::uint8_t>(chunk_high[item]) >> 4U) & 0x03U) << 4U)) - 32;
                    const int q4 = static_cast<int>((std::to_integer<std::uint8_t>(chunk_low[item + 32]) >> 4U)
                        | (((std::to_integer<std::uint8_t>(chunk_high[item]) >> 6U) & 0x03U) << 4U)) - 32;
                    accumulate(chunk + item, d * static_cast<float>(chunk_scales[scale_index]) * static_cast<float>(q1));
                    accumulate(chunk + 32 + item, d * static_cast<float>(chunk_scales[scale_index + 2]) * static_cast<float>(q2));
                    accumulate(chunk + 64 + item, d * static_cast<float>(chunk_scales[scale_index + 4]) * static_cast<float>(q3));
                    accumulate(chunk + 96 + item, d * static_cast<float>(chunk_scales[scale_index + 6]) * static_cast<float>(q4));
                }
            }
        } else if (tensor.type == 2) {
            const float d = fp16_to_float(load_u16(block));
            for (std::size_t item = 0; item < 16; ++item) {
                const auto packed = std::to_integer<std::uint8_t>(block[2 + item]);
                accumulate(item, d * static_cast<float>(static_cast<int>(packed & 0x0FU) - 8));
                accumulate(item + 16, d * static_cast<float>(static_cast<int>(packed >> 4U) - 8));
            }
        } else {
            decode_block(tensor.type, block, decoded.data());
            for (std::size_t column = 0; column < layout.block_elements; ++column) {
                accumulate(column, decoded[column]);
            }
        }
    }
    return static_cast<float>(sum);
}

float tensor_row_dot(const QuantizedTensorView& view, std::size_t row, std::span<const float> input) {
    return tensor_row_dot_impl(view, row, input, avx2_dot_enabled());
}

std::vector<float> tensor_matrix_multiply(
    const QuantizedTensorView& view,
    std::span<const float> input,
    std::size_t batches
) {
    if (view.tensor == nullptr || view.tensor->dimensions.size() != 2) {
        throw std::invalid_argument("matrix multiply requires a rank-two GGUF tensor");
    }
    const auto columns = static_cast<std::size_t>(view.tensor->dimensions[0]);
    const auto rows = static_cast<std::size_t>(view.tensor->dimensions[1]);
    CpuMatmulProfileScope profile_scope(view.tensor->type);
    if (batches == 0 || input.size() != batches * columns) {
        throw std::invalid_argument("matrix multiply input shape mismatch: " + view.tensor->name);
    }
    std::vector<float> output(batches * rows);
    const auto type = view.tensor->type;
    const bool integer_dot = cpu_integer_dot_requested()
        && (type == 2 || type == 8 || type == 12 || type == 13 || type == 14)
        && (columns % 256) == 0;
    if (integer_dot) {
#if CAT_HAS_AVX2_DISPATCH
        // Quantize every activation row once to Q8_K, then consume four-row
        // interleaved weight blocks when the model loader supplied the R4
        // sidecar. Raw GGUF bytes remain the portable/GPU fallback.
        const auto activations = quantize_q8_k_activations(input);
        const auto activation_blocks_per_batch = columns / 256;
        // Phase 3: Q4_K tensors repacked R8 execute in 8-row tiles; every
        // other layout keeps the classic 4-row tiling.
        const std::size_t tile_rows =
            (!view.r4_bytes.empty() && type == 12 && view.r4_lanes == 8) ? 8 : 4;
        const auto row_tiles = (rows + tile_rows - 1) / tile_rows;
        const int rows4_mode = cpu_selector_cache_enabled()
            ? (cpu_rows4_enabled() ? 1 : 0)
            : -1;
        if (!view.r4_bytes.empty() && (type == 12 || type == 14) && batches > 1) {
            // Tile GEMM streams the weight tile set once for all batches.
            add_weight_bytes_streamed(view.r4_bytes.size());
            row_executor().run(row_tiles, [&](std::size_t begin, std::size_t end) {
                for (auto tile_index = begin; tile_index < end; ++tile_index) {
                    if (type == 12 && tile_rows == 8 && cpu_rows8_lanewise_enabled()) {
                        // The batched verifier reads the row-major payload
                        // directly, so it cannot run against the iteration-09
                        // lane-wise ordering. Drive the lane-wise matvec once
                        // per batch instead: correct, and it re-reads the
                        // weight tile per batch rather than once for all of
                        // them. Only the speculative decode path takes
                        // batches > 1, and that path is currently unreachable
                        // anyway (auto-requires-cpu-integer-dot + CPU route);
                        // a lane-wise GEMM is Phase 5 work.
                        for (std::size_t batch = 0; batch < batches; ++batch) {
                            integer_q4k_r8l_tile_dot(
                                view,
                                tile_index * 8,
                                std::span<const Q8KActivationBlock>(
                                    activations.data() + batch * activation_blocks_per_batch,
                                    activation_blocks_per_batch),
                                std::span<float>(output).subspan(
                                    batch * rows + tile_index * 8));
                        }
                    } else if (type == 12 && tile_rows == 8) {
                        integer_q4k_r8_tile_gemm(
                            view,
                            tile_index * 8,
                            activations,
                            activation_blocks_per_batch,
                            batches,
                            output);
                    } else if (type == 12) {
                        integer_q4k_r4_tile_gemm(
                            view,
                            tile_index * 4,
                            activations,
                            activation_blocks_per_batch,
                            batches,
                            output);
                    } else if (cpu_q6k_tile_enabled()) {
                        // The batched Q6_K verifier reads the per-lane block
                        // layout and cannot run against the iteration-17 tile.
                        // Drive the tile matvec once per batch instead:
                        // correct, at the cost of re-reading the weight tile
                        // per batch. Only speculative decode takes batches > 1
                        // and that path is currently unreachable; a tile GEMM
                        // is Phase D work.
                        for (std::size_t batch = 0; batch < batches; ++batch) {
                            integer_q6k_tile_dot(
                                view,
                                tile_index * 4,
                                std::span<const Q8KActivationBlock>(
                                    activations.data() + batch * activation_blocks_per_batch,
                                    activation_blocks_per_batch),
                                std::span<float>(output).subspan(
                                    batch * rows + tile_index * 4));
                        }
                    } else {
                        integer_q6k_r4_tile_gemm(
                            view,
                            tile_index * 4,
                            activations,
                            activation_blocks_per_batch,
                            batches,
                            output);
                    }
                }
            });
            return output;
        }
        // Per-batch dot path re-reads the weight stream once per batch.
        add_weight_bytes_streamed(
            static_cast<std::uint64_t>(
                view.r4_bytes.empty() ? view.bytes.size() : view.r4_bytes.size())
            * batches);
        row_executor().run(batches * row_tiles, [&](std::size_t begin, std::size_t end) {
            for (auto tile_index = begin; tile_index < end; ++tile_index) {
                const auto batch = tile_index / row_tiles;
                const auto first_row = (tile_index % row_tiles) * tile_rows;
                const auto batch_activations = std::span<const Q8KActivationBlock>(
                    activations.data() + batch * activation_blocks_per_batch,
                    activation_blocks_per_batch);
                if (!view.r4_bytes.empty()) {
                    integer_quantized_r4_tile_dot(
                        view,
                        first_row,
                        batch_activations,
                        std::span<float>(output).subspan(batch * rows + first_row),
                        rows4_mode);
                } else {
                    for (std::size_t lane = 0; lane < tile_rows && first_row + lane < rows; ++lane) {
                        output[batch * rows + first_row + lane] = integer_quantized_row_dot(
                            view, first_row + lane, batch_activations);
                    }
                }
            }
        });
        return output;
#endif
    }
    const bool use_avx2 = avx2_dot_enabled();
    add_weight_bytes_streamed(static_cast<std::uint64_t>(view.bytes.size()) * batches);
    row_executor().run(output.size(), [&](std::size_t begin, std::size_t end) {
        for (auto index = begin; index < end; ++index) {
            const auto batch = index / rows;
            const auto row = index % rows;
            output[index] = tensor_row_dot_impl(
                view,
                row,
                input.subspan(batch * columns, columns),
                use_avx2
            );
        }
    });
    return output;
}

std::vector<float> tensor_matrix_vector(const QuantizedTensorView& view, std::span<const float> input) {
    return tensor_matrix_multiply(view, input, 1);
}

void tensor_matvec_row_range(
    const QuantizedTensorView& view,
    std::span<const float> input,
    std::size_t first_row,
    std::size_t last_row,
    std::span<float> output
) {
    if (view.tensor == nullptr || view.tensor->dimensions.size() != 2) {
        throw std::invalid_argument("row-range matvec requires a rank-two GGUF tensor");
    }
    const auto columns = static_cast<std::size_t>(view.tensor->dimensions[0]);
    const auto rows = static_cast<std::size_t>(view.tensor->dimensions[1]);
    if (first_row > last_row || last_row > rows) {
        throw std::invalid_argument("row-range matvec bounds are invalid: " + view.tensor->name);
    }
    if (input.size() != columns) {
        throw std::invalid_argument("row-range matvec input width mismatch: " + view.tensor->name);
    }
    const auto count = last_row - first_row;
    if (output.size() < count) {
        throw std::invalid_argument("row-range matvec output span is too small: " + view.tensor->name);
    }
    if (count == 0) return;
    CpuMatmulProfileScope profile_scope(view.tensor->type);
    const auto type = view.tensor->type;
    const bool integer_dot = cpu_integer_dot_requested()
        && (type == 2 || type == 8 || type == 12 || type == 13 || type == 14)
        && (columns % 256) == 0;
    {
        // Row-range matvec streams only its share of the weight rows.
        const bool reads_r4 = integer_dot && !view.r4_bytes.empty();
        const auto pass_bytes = reads_r4 ? view.r4_bytes.size() : view.bytes.size();
        add_weight_bytes_streamed(static_cast<std::uint64_t>(pass_bytes) * count / rows);
    }
    if (integer_dot) {
#if CAT_HAS_AVX2_DISPATCH
        const auto activations = quantize_q8_k_activations(input);
        // Tile-width R4/R8 spans when the range start is tile aligned (the
        // hybrid executor rounds splits to 32 rows, a multiple of both);
        // ragged edges and unaligned starts use the per-row integer dot,
        // which produces the same bits.
        const std::size_t tile_rows =
            (!view.r4_bytes.empty() && type == 12 && view.r4_lanes == 8) ? 8 : 4;
        if (!view.r4_bytes.empty() && (type == 12 || type == 14)
            && (first_row % tile_rows) == 0) {
            const auto tiles = count / tile_rows;
            const int rows4_mode = cpu_selector_cache_enabled()
                ? (cpu_rows4_enabled() ? 1 : 0)
                : -1;
            row_executor().run(tiles, [&](std::size_t begin, std::size_t end) {
                for (auto tile = begin; tile < end; ++tile) {
                    integer_quantized_r4_tile_dot(
                        view, first_row + tile * tile_rows, activations,
                        output.subspan(tile * tile_rows), rows4_mode);
                }
            });
            for (std::size_t row = first_row + tiles * tile_rows; row < last_row; ++row) {
                output[row - first_row] = integer_quantized_row_dot(view, row, activations);
            }
        } else {
            row_executor().run(count, [&](std::size_t begin, std::size_t end) {
                for (auto index = begin; index < end; ++index) {
                    output[index] = integer_quantized_row_dot(view, first_row + index, activations);
                }
            });
        }
        return;
#endif
    }
    const bool use_avx2 = avx2_dot_enabled();
    row_executor().run(count, [&](std::size_t begin, std::size_t end) {
        for (auto index = begin; index < end; ++index) {
            output[index] = tensor_row_dot_impl(view, first_row + index, input, use_avx2);
        }
    });
}

std::vector<float> tensor_row_f32(const QuantizedTensorView& view, std::size_t row) {
    if (view.tensor == nullptr) throw std::invalid_argument("quantized tensor view has no directory entry");
    const auto& tensor = *view.tensor;
    if (tensor.dimensions.size() != 2) throw std::invalid_argument("row read requires a rank-two GGUF tensor: " + tensor.name);
    const auto columns = static_cast<std::size_t>(tensor.dimensions[0]);
    const auto rows = static_cast<std::size_t>(tensor.dimensions[1]);
    if (row >= rows) throw std::out_of_range("tensor row is outside the matrix: " + tensor.name);
    if (view.bytes.size() != tensor_byte_length(tensor)) throw std::invalid_argument("row read tensor byte span mismatch: " + tensor.name);
    const auto layout = layout_for(tensor.type);
    if (columns % layout.block_elements != 0) throw std::invalid_argument("matrix row is not quant-block aligned: " + tensor.name);
    std::vector<float> output(columns);
    const auto blocks_per_row = columns / layout.block_elements;
    for (std::size_t block_index = 0; block_index < blocks_per_row; ++block_index) {
        const auto byte_offset = (row * blocks_per_row + block_index) * layout.block_bytes;
        decode_block(tensor.type, view.bytes.data() + byte_offset, output.data() + block_index * layout.block_elements);
    }
    return output;
}

std::vector<float> read_tensor_f32(
    const std::filesystem::path& path,
    const GgufModelInfo& model,
    std::string_view tensor_name
) {
    const auto found = std::find_if(model.tensors.begin(), model.tensors.end(), [&](const GgufTensorInfo& tensor) {
        return tensor.name == tensor_name;
    });
    if (found == model.tensors.end()) throw std::invalid_argument("GGUF tensor not found: " + std::string(tensor_name));
    const auto length = tensor_byte_length(*found);
    std::vector<std::byte> bytes(length);
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("unable to open GGUF tensor data");
    const auto absolute_offset = model.tensor_data_offset + found->offset;
    stream.seekg(static_cast<std::streamoff>(absolute_offset));
    stream.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!stream) throw std::runtime_error("unable to read complete GGUF tensor: " + found->name);
    return dequantize_tensor(*found, bytes);
}

int run_tensor_data_selftest() {
    try {
        GgufTensorInfo f16_subnormal{"f16_subnormal", {1}, 1, 0};
        const std::array<std::byte, 2> f16_subnormal_bytes{
            std::byte{0x01}, std::byte{0x00},
        };
        const auto f16_subnormal_values = dequantize_tensor(f16_subnormal, f16_subnormal_bytes);
        if (f16_subnormal_values.size() != 1
            || f16_subnormal_values[0] != std::ldexp(1.0F, -24)) {
            throw std::runtime_error("F16 subnormal decode mismatch");
        }

        GgufTensorInfo q4_0{"q4_0", {32}, 2, 0};
        std::vector<std::byte> q4_bytes(18, std::byte{0});
        q4_bytes[0] = std::byte{0x00}; q4_bytes[1] = std::byte{0x3c};  // fp16 1.0
        for (std::size_t index = 0; index < 16; ++index) q4_bytes[2 + index] = std::byte{0x98};
        const auto q4_values = dequantize_tensor(q4_0, q4_bytes);
        for (std::size_t index = 0; index < 16; ++index) {
            if (q4_values[index] != 0.0F || q4_values[index + 16] != 1.0F) throw std::runtime_error("Q4_0 decode mismatch");
        }

        GgufTensorInfo q4_k{"q4_k", {256}, 12, 0};
        std::vector<std::byte> q4k_bytes(144, std::byte{0});
        q4k_bytes[0] = std::byte{0x00}; q4k_bytes[1] = std::byte{0x3c};
        for (std::size_t index = 0; index < 4; ++index) q4k_bytes[4 + index] = std::byte{0x01};
        for (std::size_t index = 0; index < 4; ++index) q4k_bytes[12 + index] = std::byte{0x01};
        for (std::size_t index = 0; index < 128; ++index) q4k_bytes[16 + index] = std::byte{0x21};
        const auto q4k_values = dequantize_tensor(q4_k, q4k_bytes);
        for (std::size_t group = 0; group < 8; ++group) {
            const float expected = group % 2 == 0 ? 1.0F : 2.0F;
            if (q4k_values[group * 32] != expected) throw std::runtime_error("Q4_K decode mismatch");
        }

        GgufTensorInfo q5_k{"q5_k", {256}, 13, 0};
        std::vector<std::byte> q5k_bytes(176, std::byte{0});
        q5k_bytes[0] = std::byte{0x00}; q5k_bytes[1] = std::byte{0x3c};
        for (std::size_t index = 0; index < 4; ++index) q5k_bytes[4 + index] = std::byte{0x01};
        for (std::size_t index = 0; index < 4; ++index) q5k_bytes[12 + index] = std::byte{0x01};
        for (std::size_t index = 0; index < 32; ++index) q5k_bytes[16 + index] = std::byte{0x03};
        for (std::size_t index = 0; index < 128; ++index) q5k_bytes[48 + index] = std::byte{0x21};
        const auto q5k_values = dequantize_tensor(q5_k, q5k_bytes);
        for (std::size_t group = 0; group < 8; ++group) {
            const float nibble = group % 2 == 0 ? 1.0F : 2.0F;
            const float high = group < 2 ? 16.0F : 0.0F;
            if (q5k_values[group * 32] != nibble + high) {
                throw std::runtime_error("Q5_K decode mismatch");
            }
        }

        GgufTensorInfo q6_k{"q6_k", {256}, 14, 0};
        std::vector<std::byte> q6k_bytes(210, std::byte{0});
        for (std::size_t index = 128; index < 192; ++index) q6k_bytes[index] = std::byte{0xaa};
        for (std::size_t index = 192; index < 208; ++index) q6k_bytes[index] = std::byte{0x01};
        q6k_bytes[208] = std::byte{0x00}; q6k_bytes[209] = std::byte{0x3c};
        const auto q6k_values = dequantize_tensor(q6_k, q6k_bytes);
        for (const float value : q6k_values) if (value != 0.0F) throw std::runtime_error("Q6_K zero decode mismatch");

        GgufTensorInfo iq4_nl{"iq4_nl", {32}, 20, 0};
        std::vector<std::byte> iq4nl_bytes(18, std::byte{0});
        iq4nl_bytes[0] = std::byte{0x00}; iq4nl_bytes[1] = std::byte{0x3c};
        for (std::size_t index = 0; index < 16; ++index) iq4nl_bytes[2 + index] = std::byte{0xf0};
        const auto iq4nl_values = dequantize_tensor(iq4_nl, iq4nl_bytes);
        for (std::size_t index = 0; index < 16; ++index) {
            if (iq4nl_values[index] != -127.0F || iq4nl_values[index + 16] != 113.0F) {
                throw std::runtime_error("IQ4_NL decode mismatch");
            }
        }

        GgufTensorInfo iq4_xs{"iq4_xs", {256}, 23, 0};
        std::vector<std::byte> iq4xs_bytes(136, std::byte{0});
        iq4xs_bytes[0] = std::byte{0x00}; iq4xs_bytes[1] = std::byte{0x3c};
        iq4xs_bytes[2] = std::byte{0xaa}; iq4xs_bytes[3] = std::byte{0xaa};
        for (std::size_t index = 0; index < 4; ++index) iq4xs_bytes[4 + index] = std::byte{0x11};
        for (std::size_t index = 0; index < 128; ++index) iq4xs_bytes[8 + index] = std::byte{0xf0};
        const auto iq4xs_values = dequantize_tensor(iq4_xs, iq4xs_bytes);
        for (std::size_t group = 0; group < 8; ++group) {
            for (std::size_t index = 0; index < 16; ++index) {
                if (iq4xs_values[group * 32 + index] != -127.0F
                    || iq4xs_values[group * 32 + index + 16] != 113.0F) {
                    throw std::runtime_error("IQ4_XS decode mismatch");
                }
            }
        }

        GgufTensorInfo q4_matrix{"q4_matrix", {256, 64}, 2, 0};
        std::vector<std::byte> q4_matrix_bytes(64 * 8 * 18);
        for (std::size_t row = 0; row < 64; ++row) {
            for (std::size_t block = 0; block < 8; ++block) {
                const auto base = (row * 8 + block) * 18;
                q4_matrix_bytes[base] = std::byte{0x00};
                q4_matrix_bytes[base + 1] = std::byte{0x3c};  // fp16 1.0
                for (std::size_t item = 0; item < 16; ++item) {
                    q4_matrix_bytes[base + 2 + item] = std::byte{0x98};
                }
            }
        }
        std::array<float, 256> dot_input{};
        for (std::size_t index = 0; index < dot_input.size(); ++index) {
            dot_input[index] = static_cast<float>(index + 1) / 32.0F;
        }
        GgufTensorInfo q6_matrix{"q6_matrix", {256, 64}, 14, 0};
        std::vector<std::byte> q6_matrix_bytes(64 * 210);
        std::array<std::byte, 210> q6_block{};
        for (std::size_t index = 0; index < 128; ++index) {
            q6_block[index] = std::byte{static_cast<std::uint8_t>(index * 13U + 7U)};
        }
        for (std::size_t index = 0; index < 64; ++index) {
            q6_block[128 + index] = std::byte{static_cast<std::uint8_t>(index * 5U + 3U)};
        }
        for (std::size_t index = 0; index < 16; ++index) {
            const auto scale = static_cast<std::int8_t>(static_cast<int>(index % 7) - 3);
            q6_block[192 + index] = std::byte{static_cast<std::uint8_t>(scale)};
        }
        q6_block[208] = std::byte{0x00}; q6_block[209] = std::byte{0x38};  // fp16 0.5
        for (std::size_t row = 0; row < 64; ++row) {
            std::copy(q6_block.begin(), q6_block.end(), q6_matrix_bytes.begin() + row * q6_block.size());
        }
        std::array<float, 256> q6_dot_input{};
        for (std::size_t index = 0; index < q6_dot_input.size(); ++index) {
            q6_dot_input[index] = static_cast<float>(static_cast<int>(index % 19) - 9) / 19.0F;
        }
        // This block compares the scalar float path against the AVX2 float
        // path, so it must hold the integer-dot selector constant. Run 04
        // promoted CATERPILLAR_CPU_INTEGER_DOT to default-on, and
        // cpu_integer_dot_requested() gates on avx2_dot_enabled() - so
        // toggling CATERPILLAR_CPU_SIMD alone would compare scalar-float
        // against the lossy Q8_K integer path and fail a 1e-5 tolerance for
        // the wrong reason. Integer-path parity is covered separately below.
#ifdef _WIN32
        _putenv_s("CATERPILLAR_CPU_INTEGER_DOT", "0");
#else
        setenv("CATERPILLAR_CPU_INTEGER_DOT", "0", 1);
#endif
#ifdef _WIN32
        _putenv_s("CATERPILLAR_CPU_SIMD", "0");
#else
        setenv("CATERPILLAR_CPU_SIMD", "0", 1);
#endif
        const auto scalar_rows = tensor_matrix_vector(
            QuantizedTensorView{&q4_matrix, q4_matrix_bytes},
            dot_input
        );
        const auto scalar_q6_rows = tensor_matrix_vector(
            QuantizedTensorView{&q6_matrix, q6_matrix_bytes},
            q6_dot_input
        );
#ifdef _WIN32
        _putenv_s("CATERPILLAR_CPU_SIMD", "1");
#else
        setenv("CATERPILLAR_CPU_SIMD", "1", 1);
#endif
        const auto dispatched_rows = tensor_matrix_vector(
            QuantizedTensorView{&q4_matrix, q4_matrix_bytes},
            dot_input
        );
        const auto dispatched_q6_rows = tensor_matrix_vector(
            QuantizedTensorView{&q6_matrix, q6_matrix_bytes},
            q6_dot_input
        );
        if (scalar_rows.size() != dispatched_rows.size()) throw std::runtime_error("parallel row count mismatch");
        for (std::size_t index = 0; index < scalar_rows.size(); ++index) {
            if (std::abs(scalar_rows[index] - dispatched_rows[index]) > 1.0e-5F) {
                throw std::runtime_error("AVX2 row-dot parity mismatch");
            }
        }
        if (scalar_q6_rows.size() != dispatched_q6_rows.size()) throw std::runtime_error("Q6_K parallel row count mismatch");
        for (std::size_t index = 0; index < scalar_q6_rows.size(); ++index) {
            if (std::abs(scalar_q6_rows[index] - dispatched_q6_rows[index]) > 1.0e-3F) {
                throw std::runtime_error("Q6_K AVX2 row-dot parity mismatch");
            }
        }
        // Release the float-only pin so the integer-path checks below run
        // against the shipped default (empty removes the variable on Windows).
#ifdef _WIN32
        _putenv_s("CATERPILLAR_CPU_INTEGER_DOT", "");
#else
        unsetenv("CATERPILLAR_CPU_INTEGER_DOT");
#endif
        const auto q8_dequantize = [](std::span<const float> values) {
            const auto blocks = quantize_q8_k_activations(values);
            std::vector<float> output(values.size());
            for (std::size_t block = 0; block < blocks.size(); ++block) {
                for (std::size_t item = 0; item < 256; ++item) {
                    output[block * 256 + item] = blocks[block].d
                        * static_cast<float>(blocks[block].values[item]);
                }
            }
            return output;
        };
        const auto q4_q8_input = q8_dequantize(dot_input);
        const auto q6_q8_input = q8_dequantize(q6_dot_input);
#ifdef _WIN32
        _putenv_s("CATERPILLAR_CPU_SIMD", "0");
#else
        setenv("CATERPILLAR_CPU_SIMD", "0", 1);
#endif
        const auto q4_q8_reference = tensor_matrix_vector(
            QuantizedTensorView{&q4_matrix, q4_matrix_bytes}, q4_q8_input);
        const auto q6_q8_reference = tensor_matrix_vector(
            QuantizedTensorView{&q6_matrix, q6_matrix_bytes}, q6_q8_input);
#ifdef _WIN32
        _putenv_s("CATERPILLAR_CPU_SIMD", "1");
        _putenv_s("CATERPILLAR_CPU_INTEGER_DOT", "1");
#else
        setenv("CATERPILLAR_CPU_SIMD", "1", 1);
        setenv("CATERPILLAR_CPU_INTEGER_DOT", "1", 1);
#endif
        const auto q4_r4 = repack_tensor_r4(q4_matrix, q4_matrix_bytes);
        const auto q6_r4 = repack_tensor_r4(q6_matrix, q6_matrix_bytes);
        const auto q4_integer_rows = tensor_matrix_vector(
            QuantizedTensorView{&q4_matrix, q4_matrix_bytes, q4_r4}, dot_input);
        const auto q6_integer_rows = tensor_matrix_vector(
            QuantizedTensorView{&q6_matrix, q6_matrix_bytes, q6_r4}, q6_dot_input);
        for (std::size_t index = 0; index < q4_q8_reference.size(); ++index) {
            if (std::abs(q4_q8_reference[index] - q4_integer_rows[index]) > 1.0e-4F) {
                throw std::runtime_error("Q4_0 integer-dot Q8_K reference mismatch");
            }
        }
        for (std::size_t index = 0; index < q6_q8_reference.size(); ++index) {
            if (std::abs(q6_q8_reference[index] - q6_integer_rows[index]) > 2.0e-3F) {
                throw std::runtime_error("Q6_K integer-dot Q8_K reference mismatch");
            }
        }

        GgufTensorInfo q4k_matrix{"q4k_matrix", {256, 64}, 12, 0};
        GgufTensorInfo q5k_matrix{"q5k_matrix", {256, 64}, 13, 0};
        GgufTensorInfo q8_matrix{"q8_matrix", {256, 64}, 8, 0};
        std::vector<std::byte> q4k_matrix_bytes(64 * q4k_bytes.size());
        std::vector<std::byte> q5k_matrix_bytes(64 * q5k_bytes.size());
        std::vector<std::byte> q8_matrix_bytes(64 * 8 * 34, std::byte{0});
        for (std::size_t row = 0; row < 64; ++row) {
            std::copy(q4k_bytes.begin(), q4k_bytes.end(),
                q4k_matrix_bytes.begin() + static_cast<std::ptrdiff_t>(row * q4k_bytes.size()));
            std::copy(q5k_bytes.begin(), q5k_bytes.end(),
                q5k_matrix_bytes.begin() + static_cast<std::ptrdiff_t>(row * q5k_bytes.size()));
            for (std::size_t block = 0; block < 8; ++block) {
                const auto base = (row * 8 + block) * 34;
                q8_matrix_bytes[base] = std::byte{0x00};
                q8_matrix_bytes[base + 1] = std::byte{0x38};  // fp16 0.5
                for (std::size_t item = 0; item < 32; ++item) {
                    q8_matrix_bytes[base + 2 + item] = std::byte{
                        static_cast<std::uint8_t>(static_cast<std::int8_t>(static_cast<int>(item) - 16))};
                }
            }
        }
#ifdef _WIN32
        _putenv_s("CATERPILLAR_CPU_SIMD", "0");
        _putenv_s("CATERPILLAR_CPU_INTEGER_DOT", "");
#else
        setenv("CATERPILLAR_CPU_SIMD", "0", 1);
        unsetenv("CATERPILLAR_CPU_INTEGER_DOT");
#endif
        const std::array<const GgufTensorInfo*, 3> remaining_infos{
            &q4k_matrix, &q5k_matrix, &q8_matrix};
        const std::array<std::span<const std::byte>, 3> remaining_bytes{
            q4k_matrix_bytes, q5k_matrix_bytes, q8_matrix_bytes};
        std::array<std::vector<float>, 3> remaining_references;
        for (std::size_t format = 0; format < remaining_infos.size(); ++format) {
            remaining_references[format] = tensor_matrix_vector(
                QuantizedTensorView{remaining_infos[format], remaining_bytes[format]}, q6_q8_input);
        }
#ifdef _WIN32
        _putenv_s("CATERPILLAR_CPU_SIMD", "1");
        _putenv_s("CATERPILLAR_CPU_INTEGER_DOT", "1");
#else
        setenv("CATERPILLAR_CPU_SIMD", "1", 1);
        setenv("CATERPILLAR_CPU_INTEGER_DOT", "1", 1);
#endif
        for (std::size_t format = 0; format < remaining_infos.size(); ++format) {
            const auto r4 = repack_tensor_r4(*remaining_infos[format], remaining_bytes[format]);
            const auto actual = tensor_matrix_vector(
                QuantizedTensorView{remaining_infos[format], remaining_bytes[format], r4,
                    cpu_repack_lanes(remaining_infos[format]->type)}, q6_dot_input);
            for (std::size_t row = 0; row < actual.size(); ++row) {
                if (std::abs(remaining_references[format][row] - actual[row]) > 3.0e-3F) {
                    throw std::runtime_error(
                        remaining_infos[format]->name + " integer-dot Q8_K reference mismatch");
                }
            }
        }

        // Phase 3 (2026-07-21 campaign): R8 native-byte layout parity. Both
        // interleaves are built explicitly in this one process.
        //
        // Run 04 iteration 06 moved the R8 *matvec* to int32 accumulation with
        // a single float conversion per superblock, so it no longer shares R4's
        // per-group float arithmetic order and the matvec and the row-range
        // fallback that reuses it are held to a numerical tolerance rather than
        // bit equality. The integer form is the more accurate of the two. The
        // batched GEMM still runs the shared float order and must stay
        // bit-identical - that check is deliberately left exact, so a future
        // change to the GEMM cannot silently drift.
        constexpr float r8_matvec_tolerance = 1.0e-3F;
        {
            const auto q4k_r4 = repack_tensor_r4(q4k_matrix, q4k_matrix_bytes, 4);
            const auto q4k_r8 = repack_tensor_r4(q4k_matrix, q4k_matrix_bytes, 8);
            if (q4k_r8.size() != 64 / 8 * 8 * 148) {
                throw std::runtime_error("Q4_K R8 sidecar must be exactly 148 bytes per block");
            }
            const QuantizedTensorView r4_view{&q4k_matrix, q4k_matrix_bytes, q4k_r4, 4};
            const QuantizedTensorView r8_view{&q4k_matrix, q4k_matrix_bytes, q4k_r8, 8};
            const auto r4_out = tensor_matrix_vector(r4_view, q6_dot_input);
            const auto r8_out = tensor_matrix_vector(r8_view, q6_dot_input);
            for (std::size_t row = 0; row < r4_out.size(); ++row) {
                if (std::abs(r4_out[row] - r8_out[row]) > r8_matvec_tolerance) {
                    throw std::runtime_error("Q4_K R8 matvec diverges from R4");
                }
            }
            std::vector<float> gemm_input(4 * 256);
            for (std::size_t index = 0; index < gemm_input.size(); ++index) {
                gemm_input[index] = static_cast<float>(static_cast<int>(index % 53) - 26) / 53.0F;
            }
            const auto r4_gemm = tensor_matrix_multiply(r4_view, gemm_input, 4);
            const auto r8_gemm = tensor_matrix_multiply(r8_view, gemm_input, 4);
            // Under the iteration-09 lane-wise payload the batched path routes
            // through the lane-wise matvec, which accumulates in int32, so it
            // is held to the same tolerance as the matvec. With the row-major
            // payload the shared float order still applies and bit-identity is
            // still required.
            const bool gemm_shares_float_order = !cpu_rows8_lanewise_enabled();
            for (std::size_t index = 0; index < r4_gemm.size(); ++index) {
                const bool diverged = gemm_shares_float_order
                    ? r4_gemm[index] != r8_gemm[index]
                    : std::abs(r4_gemm[index] - r8_gemm[index]) > r8_matvec_tolerance;
                if (diverged) {
                    throw std::runtime_error("Q4_K R8 batched GEMM diverges from R4");
                }
            }
            // Ragged fallback: an unaligned row range forces the per-row path.
            std::vector<float> r4_ranged(5, -1.0F);
            std::vector<float> r8_ranged(5, -1.0F);
            const std::vector<float> range_input(q6_dot_input.begin(), q6_dot_input.begin() + 256);
            tensor_matvec_row_range(r4_view, range_input, 3, 8, r4_ranged);
            tensor_matvec_row_range(r8_view, range_input, 3, 8, r8_ranged);
            for (std::size_t index = 0; index < r4_ranged.size(); ++index) {
                if (std::abs(r4_ranged[index] - r8_ranged[index]) > r8_matvec_tolerance) {
                    throw std::runtime_error("Q4_K R8 row-range fallback diverges from R4");
                }
            }
        }
        if (const char* perf = std::getenv("CATERPILLAR_CPU_SELFTEST_PERF");
            perf != nullptr && perf[0] == '1') {
            constexpr std::size_t perf_columns = 2560;
            constexpr std::size_t perf_batches = 4;
            // Run 04 round 5 iteration 25: the default 4096 rows is ~6 MB of
            // Q4_K sidecar, which fits this host's 12 MB L3 - so the benchmark
            // never touches DRAM and cannot see bandwidth effects (the
            // limitation iteration 23 hit). CATERPILLAR_CPU_SELFTEST_PERF_ROWS
            // raises it past L3 to measure the kernel's DRAM-resident rate,
            // which is the regime decode actually runs in.
            std::size_t perf_rows = 4096;
            if (const char* configured = std::getenv("CATERPILLAR_CPU_SELFTEST_PERF_ROWS");
                configured != nullptr && *configured != ' ') {
                const auto requested = std::strtoul(configured, nullptr, 10);
                if (requested >= 1024 && requested <= 262144) {
                    perf_rows = static_cast<std::size_t>(requested);
                }
            }
            GgufTensorInfo perf_info{"q4k_r4_perf", {perf_columns, perf_rows}, 12, 0};
            const auto blocks_per_row = perf_columns / 256;
            std::vector<std::byte> perf_bytes(perf_rows * blocks_per_row * q4k_bytes.size());
            for (std::size_t block = 0; block < perf_rows * blocks_per_row; ++block) {
                std::copy(q4k_bytes.begin(), q4k_bytes.end(),
                    perf_bytes.begin() + static_cast<std::ptrdiff_t>(block * q4k_bytes.size()));
            }
            const auto perf_r4 = repack_tensor_r4(perf_info, perf_bytes);
            const auto perf_lanes = cpu_repack_lanes(perf_info.type);
            std::vector<float> perf_input(perf_batches * perf_columns);
            for (std::size_t index = 0; index < perf_input.size(); ++index) {
                perf_input[index] = static_cast<float>(static_cast<int>(index % 67) - 33) / 67.0F;
            }
            const auto perf_q8_input = q8_dequantize(perf_input);
            const auto raw_view = QuantizedTensorView{&perf_info, perf_bytes};
            const auto r4_view = QuantizedTensorView{&perf_info, perf_bytes, perf_r4, perf_lanes};
            const auto set_integer_mode = [](bool enabled) {
#ifdef _WIN32
                _putenv_s("CATERPILLAR_CPU_SIMD", enabled ? "1" : "0");
                _putenv_s("CATERPILLAR_CPU_INTEGER_DOT", enabled ? "1" : "");
#else
                setenv("CATERPILLAR_CPU_SIMD", enabled ? "1" : "0", 1);
                if (enabled) setenv("CATERPILLAR_CPU_INTEGER_DOT", "1", 1);
                else unsetenv("CATERPILLAR_CPU_INTEGER_DOT");
#endif
            };
            set_integer_mode(false);
            const auto scalar_reference = tensor_matrix_multiply(raw_view, perf_q8_input, perf_batches);
            set_integer_mode(true);
            const auto integer_reference = tensor_matrix_multiply(r4_view, perf_input, perf_batches);
            for (std::size_t index = 0; index < scalar_reference.size(); ++index) {
                if (std::abs(scalar_reference[index] - integer_reference[index]) > 1.0e-2F) {
                    throw std::runtime_error("CPU integer perf graph differs from scalar Q8_K oracle");
                }
            }
            // Run 04 round 4 (2026-08-04) iteration 21: make this a usable
            // instrument.
            //
            // At three samples with no warm-up, repeated process launches of
            // this benchmark swung 0.36-0.60 ms - a 40% spread, wider than any
            // kernel effect worth measuring, which is why the campaign never
            // used it. The integer pass costs ~0.5 ms, so hundreds of samples
            // cost under a second: take them, discard a warm-up phase, and
            // report the minimum alongside the median. The minimum is the
            // honest figure for a kernel microbenchmark - it is the run least
            // disturbed by scheduling and thermal noise, and kernel work is
            // exactly the case where the undisturbed run is what changed.
            //
            // CATERPILLAR_CPU_SELFTEST_PERF_ITERS sets the sample count
            // (default 3, so the existing release gate is unchanged).
            std::size_t perf_iterations = 3;
            if (const char* configured = std::getenv("CATERPILLAR_CPU_SELFTEST_PERF_ITERS");
                configured != nullptr && *configured != '\0') {
                const auto requested = std::strtoul(configured, nullptr, 10);
                if (requested >= 3 && requested <= 100000) {
                    perf_iterations = static_cast<std::size_t>(requested);
                }
            }
            std::vector<double> scalar_ms(perf_iterations, 0.0);
            std::vector<double> integer_ms(perf_iterations, 0.0);
            volatile float checksum = 0.0F;
            // Warm-up: first touch of the repacked buffer costs page faults the
            // steady state never pays again.
            set_integer_mode(true);
            for (std::size_t warm = 0; warm < 3; ++warm) {
                auto warmed = tensor_matrix_multiply(r4_view, perf_input, perf_batches);
                checksum = checksum + warmed[warm % warmed.size()];
            }
            for (std::size_t iteration = 0; iteration < perf_iterations; ++iteration) {
                set_integer_mode(false);
                auto start = std::chrono::steady_clock::now();
                auto scalar = tensor_matrix_multiply(raw_view, perf_q8_input, perf_batches);
                auto end = std::chrono::steady_clock::now();
                scalar_ms[iteration] = std::chrono::duration<double, std::milli>(end - start).count();
                checksum = checksum + scalar[iteration % scalar.size()];

                set_integer_mode(true);
                start = std::chrono::steady_clock::now();
                auto integer = tensor_matrix_multiply(r4_view, perf_input, perf_batches);
                end = std::chrono::steady_clock::now();
                integer_ms[iteration] = std::chrono::duration<double, std::milli>(end - start).count();
                checksum = checksum + integer[iteration % integer.size()];
            }
            std::sort(scalar_ms.begin(), scalar_ms.end());
            std::sort(integer_ms.begin(), integer_ms.end());
            const auto middle = perf_iterations / 2;
            const auto speedup = scalar_ms[middle] / integer_ms[middle];
            std::cout << "caterpillar CPU integer-dot perf: scalar=" << scalar_ms[middle]
                      << " ms, integer=" << integer_ms[middle] << " ms, " << speedup
                      << "x, checksum=" << checksum
                      << " (n=" << perf_iterations
                      << " integerMin=" << integer_ms.front()
                      << " integerP10=" << integer_ms[perf_iterations / 10] << ")\n";
            if (speedup < 5.0) {
                throw std::runtime_error("CPU integer-dot gate requires at least 5x scalar-float speedup");
            }
        }
#ifdef _WIN32
        _putenv_s("CATERPILLAR_CPU_INTEGER_DOT", "");
#else
        unsetenv("CATERPILLAR_CPU_INTEGER_DOT");
#endif

        // Route B (Step 09) primitive: stitched row ranges must reproduce the
        // full matvec bit-for-bit (each row runs the same per-row kernel).
        {
            constexpr std::size_t range_rows = 8;
            constexpr std::size_t range_columns = 256;
            GgufTensorInfo range_tensor{"q4_k_range", {range_columns, range_rows}, 12, 0};
            std::vector<std::byte> range_bytes(range_rows * 144, std::byte{0});
            for (std::size_t row = 0; row < range_rows; ++row) {
                auto* block = range_bytes.data() + row * 144;
                block[0] = std::byte{0x00};
                block[1] = std::byte{0x3c};  // d = 1.0
                for (std::size_t index = 0; index < 4; ++index) {
                    block[4 + index] = static_cast<std::byte>(row + 1);
                }
                for (std::size_t index = 0; index < 128; ++index) {
                    block[16 + index] = std::byte{0x21};
                }
            }
            const QuantizedTensorView range_view{&range_tensor, range_bytes};
            std::vector<float> range_input(range_columns);
            for (std::size_t index = 0; index < range_columns; ++index) {
                range_input[index] = 0.25F * static_cast<float>((index % 7) + 1);
            }
            const auto full = tensor_matrix_multiply(range_view, range_input, 1);
            std::vector<float> ranged(range_rows, -1.0F);
            tensor_matvec_row_range(range_view, range_input, 0, 3, std::span<float>(ranged).first(3));
            tensor_matvec_row_range(range_view, range_input, 3, range_rows, std::span<float>(ranged).subspan(3));
            for (std::size_t row = 0; row < range_rows; ++row) {
                if (ranged[row] != full[row]) {
                    throw std::runtime_error("row-range matvec differs from the full matvec");
                }
            }
        }

        std::cout << "caterpillar tensor decoder selftest: ok (Q4_0/Q4_K/Q5_K/Q6_K/Q8_0/IQ4_NL/IQ4_XS + AVX2/VNNI Q8_K + R4 parity + R8 matvec/row-range within tolerance + R8 GEMM bit-parity)\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "caterpillar tensor decoder selftest: failed: " << error.what() << '\n';
        return 1;
    }
}

}  // namespace cat
