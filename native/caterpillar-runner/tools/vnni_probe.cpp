// Feasibility probe: can MinGW GCC 16.1 run AVX-512 VNNI 512-bit dots
// with in-register __m512i locals, given an explicit stack realign?
#include <immintrin.h>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <chrono>
#include <cstring>

__attribute__((target("avx512f,avx512bw,avx512vl,avx512vnni"), noinline, force_align_arg_pointer))
static int dot512_vnni(const uint8_t* w, const int8_t* a, int groups) {
    __m512i acc = _mm512_setzero_si512();
    for (int g = 0; g < groups; ++g) {
        __m512i wv = _mm512_loadu_si512((const void*)(w + g * 64));
        __m512i av = _mm512_loadu_si512((const void*)(a + g * 64));
        acc = _mm512_dpbusd_epi32(acc, wv, av);
    }
    return _mm512_reduce_add_epi32(acc);
}

__attribute__((target("avx2"), noinline))
static int dot256_avx2(const uint8_t* w, const int8_t* a, int groups) {
    __m256i acc = _mm256_setzero_si256();
    const __m256i ones = _mm256_set1_epi16(1);
    for (int g = 0; g < groups * 2; ++g) {
        __m256i wv = _mm256_loadu_si256((const __m256i*)(w + g * 32));
        __m256i av = _mm256_loadu_si256((const __m256i*)(a + g * 32));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(wv, av), ones));
    }
    __m128i lo = _mm256_castsi256_si128(acc);
    __m128i hi = _mm256_extracti128_si256(acc, 1);
    __m128i s = _mm_add_epi32(lo, hi);
    s = _mm_hadd_epi32(s, s);
    s = _mm_hadd_epi32(s, s);
    return _mm_cvtsi128_si32(s);
}

int main() {
    printf("cpu avx512vnni=%d avx512vl=%d avx512f=%d avx2=%d\n",
           __builtin_cpu_supports("avx512vnni"),
           __builtin_cpu_supports("avx512vl"),
           __builtin_cpu_supports("avx512f"),
           __builtin_cpu_supports("avx2"));
    fflush(stdout);

    const int groups = 1024;                 // 64 KiB of weights, L2-resident
    std::vector<uint8_t> w(groups * 64);
    std::vector<int8_t>  a(groups * 64);
    for (int i = 0; i < groups * 64; ++i) { w[i] = (uint8_t)(i * 7 & 15); a[i] = (int8_t)((i * 3 & 31) - 16); }

    int ref = dot256_avx2(w.data(), a.data(), groups);
    printf("avx2 ok, ref=%d\n", ref); fflush(stdout);

    int v = dot512_vnni(w.data(), a.data(), groups);
    printf("vnni ok, val=%d, match=%d\n", v, v == ref); fflush(stdout);

    // throughput
    const int iters = 20000;
    volatile int sink = 0;
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) sink += dot256_avx2(w.data(), a.data(), groups);
    auto t1 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) sink += dot512_vnni(w.data(), a.data(), groups);
    auto t2 = std::chrono::steady_clock::now();
    double ms_avx2 = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double ms_vnni = std::chrono::duration<double, std::milli>(t2 - t1).count();
    double bytes = (double)iters * groups * 64.0;
    printf("avx2  %.2f ms  %.2f GB/s-equiv\n", ms_avx2, bytes / (ms_avx2 * 1e6));
    printf("vnni  %.2f ms  %.2f GB/s-equiv\n", ms_vnni, bytes / (ms_vnni * 1e6));
    printf("vnni speedup over avx2: %.3fx\n", ms_avx2 / ms_vnni);
    return sink == 0 ? 1 : 0;
}
