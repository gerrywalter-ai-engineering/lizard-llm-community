// Roofline probe: sustained DRAM read bandwidth for a GEMV-like streaming
// read over a buffer far larger than L3 (i7-1165G7 has 12 MB L3).
// Also measures the strided 4-lane pattern Caterpillar's R4 kernel uses.
#include <immintrin.h>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <thread>
#include <chrono>
#include <atomic>

static const size_t BYTES = 2048ull * 1024 * 1024 / 4 * 4; // 2 GiB ~ model size

__attribute__((target("avx2"), noinline))
static uint64_t stream_read(const uint8_t* p, size_t n) {
    __m256i acc = _mm256_setzero_si256();
    for (size_t i = 0; i + 256 <= n; i += 256) {
        acc = _mm256_add_epi64(acc, _mm256_loadu_si256((const __m256i*)(p + i)));
        acc = _mm256_add_epi64(acc, _mm256_loadu_si256((const __m256i*)(p + i + 32)));
        acc = _mm256_add_epi64(acc, _mm256_loadu_si256((const __m256i*)(p + i + 64)));
        acc = _mm256_add_epi64(acc, _mm256_loadu_si256((const __m256i*)(p + i + 96)));
        acc = _mm256_add_epi64(acc, _mm256_loadu_si256((const __m256i*)(p + i + 128)));
        acc = _mm256_add_epi64(acc, _mm256_loadu_si256((const __m256i*)(p + i + 160)));
        acc = _mm256_add_epi64(acc, _mm256_loadu_si256((const __m256i*)(p + i + 192)));
        acc = _mm256_add_epi64(acc, _mm256_loadu_si256((const __m256i*)(p + i + 224)));
    }
    alignas(32) uint64_t out[4];
    _mm256_store_si256((__m256i*)out, acc);
    return out[0] + out[1] + out[2] + out[3];
}

// Caterpillar R4 pattern: 4 lanes strided 192 B apart, 32 B read per lane per group.
__attribute__((target("avx2"), noinline))
static uint64_t strided_r4_read(const uint8_t* p, size_t n) {
    __m256i acc = _mm256_setzero_si256();
    const size_t tile = 4 * 192;
    for (size_t base = 0; base + tile <= n; base += tile) {
        for (int group = 0; group < 4; ++group) {
            for (int lane = 0; lane < 4; ++lane) {
                const uint8_t* b = p + base + lane * 192;
                acc = _mm256_add_epi64(acc, _mm256_loadu_si256((const __m256i*)(b + 64 + group * 32)));
                acc = _mm256_add_epi64(acc, _mm256_loadu_si256((const __m256i*)(b)));
            }
        }
    }
    alignas(32) uint64_t out[4];
    _mm256_store_si256((__m256i*)out, acc);
    return out[0] + out[1] + out[2] + out[3];
}

static void run(const char* label, const uint8_t* buf, size_t bytes, int threads,
                uint64_t (*fn)(const uint8_t*, size_t)) {
    std::atomic<uint64_t> sink{0};
    auto t0 = std::chrono::steady_clock::now();
    std::vector<std::thread> ts;
    size_t chunk = bytes / threads;
    chunk &= ~(size_t)4095;
    for (int t = 0; t < threads; ++t)
        ts.emplace_back([&, t] { sink += fn(buf + (size_t)t * chunk, chunk); });
    for (auto& th : ts) th.join();
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double gbs = (double)(chunk * threads) / (ms * 1e6);
    printf("%-22s threads=%d  %8.1f ms  %6.2f GB/s  (sink=%llu)\n",
           label, threads, ms, gbs, (unsigned long long)sink.load());
    fflush(stdout);
}

// ---------------------------------------------------------------------------
// Phase 7 (2026-07-21 campaign) concurrent CPU+GPU bandwidth probe.
// Question: can the iGPU reach DRAM bandwidth the CPU cores cannot, so that
// concurrent streaming aggregates above the CPU-only ceiling? Kill condition
// (recorded in the plan): concurrent total < 36 GB/s => the aggregation route
// is closed permanently on this host class. UMA-specific hypothesis - on a
// discrete GPU this comparison is meaningless (its VRAM is a separate
// controller) and the probe reports that instead.
// ---------------------------------------------------------------------------
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_4.h>
#include <d3dcompiler.h>

static const char* kShader = R"(
StructuredBuffer<uint4> Input : register(t0);
RWStructuredBuffer<uint> Output : register(u0);
cbuffer Params : register(b0) { uint elements_per_thread; uint total_elements; }
[numthreads(256, 1, 1)]
void cs_stream(uint3 id : SV_DispatchThreadID) {
    uint base = id.x * elements_per_thread;
    uint sum = 0;
    for (uint i = 0; i < elements_per_thread; ++i) {
        uint index = base + i;
        if (index < total_elements) {
            uint4 v = Input[index];
            sum += v.x + v.y + v.z + v.w;
        }
    }
    if (sum == 0xFFFFFFFFu) Output[0] = sum;  // never true; defeats DCE
}
)";

struct GpuStream {
    ID3D12Device* device{};
    ID3D12CommandQueue* queue{};
    ID3D12CommandAllocator* allocator{};
    ID3D12GraphicsCommandList* list{};
    ID3D12Fence* fence{};
    HANDLE event{};
    UINT64 fence_value{};
    ID3D12Resource* big{};
    ID3D12Resource* out{};
    ID3D12RootSignature* root{};
    ID3D12PipelineState* pso{};
    size_t bytes{};
    UINT groups{};
    bool uma{};
    char adapter_name[128]{};

    bool init(size_t gpu_bytes) {
        IDXGIFactory1* factory{};
        if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) return false;
        IDXGIAdapter1* adapter{};
        for (UINT i = 0; factory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; ++i) {
            DXGI_ADAPTER_DESC1 desc{};
            adapter->GetDesc1(&desc);
            if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) { adapter->Release(); continue; }
            if (SUCCEEDED(D3D12CreateDevice(adapter, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device)))) {
                WideCharToMultiByte(CP_UTF8, 0, desc.Description, -1,
                                    adapter_name, sizeof(adapter_name), nullptr, nullptr);
                adapter->Release();
                break;
            }
            adapter->Release();
        }
        factory->Release();
        if (!device) return false;
        D3D12_FEATURE_DATA_ARCHITECTURE arch{};
        device->CheckFeatureSupport(D3D12_FEATURE_ARCHITECTURE, &arch, sizeof(arch));
        uma = arch.UMA != 0;

        D3D12_COMMAND_QUEUE_DESC qd{};
        qd.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
        if (FAILED(device->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue)))) return false;
        device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&allocator));
        device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, allocator, nullptr, IID_PPV_ARGS(&list));
        list->Close();
        device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));
        event = CreateEventW(nullptr, FALSE, FALSE, nullptr);

        bytes = gpu_bytes;
        D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_DEFAULT};
        D3D12_RESOURCE_DESC rd{};
        rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        rd.Width = bytes;
        rd.Height = 1; rd.DepthOrArraySize = 1; rd.MipLevels = 1;
        rd.SampleDesc.Count = 1;
        rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        if (FAILED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &rd,
                D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&big)))) return false;
        rd.Width = 4;
        rd.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &rd,
                D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&out));

        ID3DBlob* cs{};
        ID3DBlob* errors{};
        if (FAILED(D3DCompile(kShader, strlen(kShader), nullptr, nullptr, nullptr,
                "cs_stream", "cs_5_0", 0, 0, &cs, &errors))) {
            if (errors) printf("shader: %s\n", (const char*)errors->GetBufferPointer());
            return false;
        }
        D3D12_ROOT_PARAMETER params[3]{};
        params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
        params[0].Descriptor.ShaderRegister = 0;
        params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
        params[1].Descriptor.ShaderRegister = 0;
        params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        params[2].Constants.ShaderRegister = 0;
        params[2].Constants.Num32BitValues = 2;
        D3D12_ROOT_SIGNATURE_DESC rs{};
        rs.NumParameters = 3;
        rs.pParameters = params;
        ID3DBlob* sig{};
        D3D12SerializeRootSignature(&rs, D3D_ROOT_SIGNATURE_VERSION_1, &sig, nullptr);
        device->CreateRootSignature(0, sig->GetBufferPointer(), sig->GetBufferSize(), IID_PPV_ARGS(&root));
        sig->Release();
        D3D12_COMPUTE_PIPELINE_STATE_DESC pd{};
        pd.pRootSignature = root;
        pd.CS = {cs->GetBufferPointer(), cs->GetBufferSize()};
        if (FAILED(device->CreateComputePipelineState(&pd, IID_PPV_ARGS(&pso)))) return false;
        cs->Release();

        // 64 threads-worth per thread keeps groups low; each thread reads
        // elements_per_thread uint4 (16 B) values sequentially.
        const UINT total_elements = (UINT)(bytes / 16);
        const UINT elements_per_thread = 64;
        groups = (total_elements + 256 * elements_per_thread - 1) / (256 * elements_per_thread);
        return true;
    }

    // One full-buffer read pass; returns wall ms.
    double pass() {
        const UINT total_elements = (UINT)(bytes / 16);
        const UINT elements_per_thread = 64;
        allocator->Reset();
        list->Reset(allocator, pso);
        list->SetComputeRootSignature(root);
        list->SetComputeRootShaderResourceView(0, big->GetGPUVirtualAddress());
        list->SetComputeRootUnorderedAccessView(1, out->GetGPUVirtualAddress());
        UINT constants[2] = {elements_per_thread, total_elements};
        list->SetComputeRoot32BitConstants(2, 2, constants, 0);
        list->Dispatch(groups, 1, 1);
        list->Close();
        auto t0 = std::chrono::steady_clock::now();
        ID3D12CommandList* lists[] = {list};
        queue->ExecuteCommandLists(1, lists);
        queue->Signal(fence, ++fence_value);
        fence->SetEventOnCompletion(fence_value, event);
        WaitForSingleObject(event, 10000);
        auto t1 = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
};

static void concurrent_probe(const uint8_t* cpu_buf, size_t cpu_bytes) {
    GpuStream gpu;
    const size_t gpu_bytes = 512ull * 1024 * 1024;
    if (!gpu.init(gpu_bytes)) {
        printf("\n-- gpu probe: no usable D3D12 adapter --\n");
        return;
    }
    printf("\n-- Phase 7 concurrent CPU+GPU probe: %s (UMA=%d) --\n",
           gpu.adapter_name, gpu.uma ? 1 : 0);
    if (!gpu.uma) {
        printf("discrete adapter: separate VRAM controller - the shared-DRAM\n"
               "aggregation hypothesis does not apply; all-resident on the\n"
               "adapter is the fast path on this host. Probe ends here.\n");
        return;
    }
    // Warm the GPU buffer (first pass faults/pages it).
    gpu.pass();

    // GPU-only: sustained over 8 passes.
    double gpu_ms = 0;
    for (int i = 0; i < 8; ++i) gpu_ms += gpu.pass();
    const double gpu_only_gbs = (8.0 * gpu_bytes) / (gpu_ms * 1e6);
    printf("gpu-only               %8.1f ms  %6.2f GB/s\n", gpu_ms, gpu_only_gbs);

    // CPU-only reference (4 threads, matches the campaign default).
    run("cpu-only-4T", cpu_buf, cpu_bytes, 4, stream_read);

    // Concurrent: CPU threads stream while the GPU loops passes. Each thread
    // counts its own completed passes; bytes are computed after the join so
    // no shared counter is touched in the loop.
    std::atomic<bool> stop{false};
    const int threads = 4;
    uint64_t passes_done[4] = {0, 0, 0, 0};
    uint64_t sinks[4] = {0, 0, 0, 0};
    std::vector<std::thread> cpu_threads;
    size_t chunk = (cpu_bytes / threads) & ~(size_t)4095;
    auto t0 = std::chrono::steady_clock::now();
    for (int t = 0; t < threads; ++t) {
        cpu_threads.emplace_back([&, t] {
            uint64_t local_passes = 0;
            uint64_t local_sink = 0;
            while (!stop.load(std::memory_order_relaxed)) {
                local_sink += stream_read(cpu_buf + (size_t)t * chunk, chunk);
                ++local_passes;
            }
            passes_done[t] = local_passes;
            sinks[t] = local_sink;
        });
    }
    double gpu_conc_ms = 0;
    int gpu_passes = 0;
    while (gpu_passes < 16) { gpu_conc_ms += gpu.pass(); ++gpu_passes; }
    stop.store(true);
    for (auto& th : cpu_threads) th.join();
    auto t1 = std::chrono::steady_clock::now();
    const double wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    uint64_t total_cpu_passes = 0;
    uint64_t sink_total = 0;
    for (int t = 0; t < threads; ++t) { total_cpu_passes += passes_done[t]; sink_total += sinks[t]; }
    const double gpu_conc_gbs = ((double)gpu_passes * gpu_bytes) / (gpu_conc_ms * 1e6);
    const double cpu_conc_gbs =
        ((double)total_cpu_passes * (double)chunk) / (wall_ms * 1e6);
    static_cast<void>(sink_total);
    const double total = gpu_conc_gbs + cpu_conc_gbs;
    printf("concurrent gpu side    %8.1f ms  %6.2f GB/s\n", gpu_conc_ms, gpu_conc_gbs);
    printf("concurrent cpu side    %8.1f ms  %6.2f GB/s\n", wall_ms, cpu_conc_gbs);
    printf("CONCURRENT TOTAL                  %6.2f GB/s  (kill condition: < 36)\n", total);
    printf("verdict: %s\n", total >= 36.0
        ? "aggregation headroom EXISTS - Phase 7 implementation may proceed"
        : "KILL - concurrent total under 36 GB/s; aggregation route closed on this host");
    fflush(stdout);
}
#endif  // _WIN32

int main() {
    printf("allocating %.2f GiB...\n", BYTES / 1073741824.0); fflush(stdout);
    std::vector<uint8_t> buf(BYTES);
    for (size_t i = 0; i < BYTES; i += 4096) buf[i] = (uint8_t)i;  // fault pages in
    printf("touched.\n"); fflush(stdout);

    for (int t : {1, 2, 4, 8}) run("sequential-read", buf.data(), BYTES, t, stream_read);
    for (int t : {1, 4, 8})    run("r4-strided-4lane", buf.data(), BYTES, t, strided_r4_read);

    // What decode rate does each bandwidth imply?
    printf("\n-- implied decode rate for llama-3.2-3b Q4_K_M --\n");
    printf("native GGUF weights   : 2.011 GB/token\n");
    printf("caterpillar R8 152B   : 2.376 GB/token measured (R4 was 2.746)\n");
    printf("llama.cpp bar 12.79 tok/s @4T => %.2f GB/s of native weights\n", 2.011 * 12.79);
    printf("target   14.07 tok/s  => %.2f GB/s native / %.2f GB/s at the R8 layout\n",
           2.011 * 14.07, 2.376 * 14.07);

#ifdef _WIN32
    concurrent_probe(buf.data(), BYTES);
#endif
    return 0;
}
