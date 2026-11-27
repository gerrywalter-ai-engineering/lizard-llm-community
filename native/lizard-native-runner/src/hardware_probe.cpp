// Hardware probe for `--probe-hardware` (First Big Refactor Phase 1,
// Detection stages A-C). Self-contained on purpose: it must keep working
// when the main D3D12 executor cannot initialize, because "the GPU lane is
// broken" is exactly the fact it exists to report.
//
// The probe never throws past its own boundary and never returns a failing
// exit code for a hardware limitation: every adapter failure is recorded in
// the JSON so the JavaScript side can mark the adapter unavailable while
// CPU-only operation continues.

#include "hardware_probe.h"

#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <windows.h>

#include <d3d12.h>
#include <d3dcompiler.h>
#include <dxgi1_4.h>

#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <cpuid.h>
#include <immintrin.h>
#endif
#endif

namespace lizard_native {
namespace {

// ---- minimal JSON writer -------------------------------------------------

void json_escape_into(std::string &out, const std::string &value) {
  for (const char raw : value) {
    const unsigned char c = static_cast<unsigned char>(raw);
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (c < 0x20) {
          char buffer[8];
          std::snprintf(buffer, sizeof(buffer), "\\u%04x", c);
          out += buffer;
        } else {
          out += static_cast<char>(c);
        }
    }
  }
}

class JsonWriter {
 public:
  void begin_object() { separator(); out_ += '{'; fresh_ = true; }
  void end_object() { out_ += '}'; fresh_ = false; }
  void begin_array() { separator(); out_ += '['; fresh_ = true; }
  void end_array() { out_ += ']'; fresh_ = false; }

  void key(const char *name) {
    separator();
    out_ += '"';
    out_ += name;
    out_ += "\":";
    fresh_ = true;
  }

  void value(const std::string &text) {
    separator();
    out_ += '"';
    json_escape_into(out_, text);
    out_ += '"';
    fresh_ = false;
  }
  void value(const char *text) { value(std::string(text)); }
  void value(bool flag) { separator(); out_ += flag ? "true" : "false"; fresh_ = false; }
  void value(std::uint64_t number) { separator(); out_ += std::to_string(number); fresh_ = false; }
  void value(std::int64_t number) { separator(); out_ += std::to_string(number); fresh_ = false; }
  void value(int number) { separator(); out_ += std::to_string(number); fresh_ = false; }
  void value_null() { separator(); out_ += "null"; fresh_ = false; }

  const std::string &str() const { return out_; }

 private:
  void separator() {
    if (!fresh_ && !out_.empty()) {
      const char last = out_.back();
      if (last != '{' && last != '[' && last != ':') {
        out_ += ',';
      }
    }
    fresh_ = false;
  }

  std::string out_;
  bool fresh_ = true;
};

#if defined(_WIN32)

std::string narrow_utf16(const wchar_t *wide) {
  if (wide == nullptr || wide[0] == L'\0') {
    return "";
  }
  const int required = WideCharToMultiByte(CP_UTF8, 0, wide, -1, nullptr, 0, nullptr, nullptr);
  if (required <= 1) {
    return "";
  }
  std::string narrow(static_cast<std::size_t>(required - 1), '\0');
  WideCharToMultiByte(CP_UTF8, 0, wide, -1, narrow.data(), required, nullptr, nullptr);
  return narrow;
}

// ---- Detection stage A: CPU via CPUID + OS-enabled extended state --------

struct CpuIdRegs {
  std::uint32_t eax = 0;
  std::uint32_t ebx = 0;
  std::uint32_t ecx = 0;
  std::uint32_t edx = 0;
};

bool cpuid_supported() {
#if defined(_M_X64) || defined(_M_IX86) || defined(__x86_64__) || defined(__i386__)
  return true;
#else
  return false;
#endif
}

CpuIdRegs cpuid(std::uint32_t leaf, std::uint32_t subleaf) {
  CpuIdRegs regs;
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
  int raw[4] = {};
  __cpuidex(raw, static_cast<int>(leaf), static_cast<int>(subleaf));
  regs.eax = static_cast<std::uint32_t>(raw[0]);
  regs.ebx = static_cast<std::uint32_t>(raw[1]);
  regs.ecx = static_cast<std::uint32_t>(raw[2]);
  regs.edx = static_cast<std::uint32_t>(raw[3]);
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
  unsigned int eax = 0;
  unsigned int ebx = 0;
  unsigned int ecx = 0;
  unsigned int edx = 0;
  __get_cpuid_count(leaf, subleaf, &eax, &ebx, &ecx, &edx);
  regs.eax = eax;
  regs.ebx = ebx;
  regs.ecx = ecx;
  regs.edx = edx;
#else
  (void)leaf;
  (void)subleaf;
#endif
  return regs;
}

std::uint64_t read_xcr0() {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
  return _xgetbv(0);
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
  std::uint32_t eax = 0;
  std::uint32_t edx = 0;
  __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
  return (static_cast<std::uint64_t>(edx) << 32) | eax;
#else
  return 0;
#endif
}

struct CpuFeature {
  bool cpu = false;
  bool os = false;
};

struct CpuReport {
  std::string architecture;
  std::string vendor;
  std::string brand;
  int family = 0;
  int model = 0;
  int stepping = 0;
  std::uint32_t logical_processors = 0;
  std::uint32_t processor_groups = 0;
  std::uint32_t physical_cores = 0;
  bool cpuid_available = false;
  CpuFeature sse2;
  CpuFeature sse41;
  CpuFeature sse42;
  CpuFeature avx;
  CpuFeature f16c;
  CpuFeature fma;
  CpuFeature avx2;
  CpuFeature avx512f;
  CpuFeature avx512bw;
  CpuFeature avx512vl;
  CpuFeature avx512_vnni;
  CpuFeature avx_vnni;
};

std::uint32_t count_physical_cores() {
  DWORD length = 0;
  GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &length);
  if (GetLastError() != ERROR_INSUFFICIENT_BUFFER || length == 0) {
    return 0;
  }
  std::vector<char> buffer(length);
  if (!GetLogicalProcessorInformationEx(
        RelationProcessorCore,
        reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data()),
        &length)) {
    return 0;
  }
  std::uint32_t cores = 0;
  for (DWORD offset = 0; offset < length;) {
    const auto *info =
      reinterpret_cast<const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *>(buffer.data() + offset);
    if (info->Relationship == RelationProcessorCore) {
      cores += 1;
    }
    offset += info->Size;
  }
  return cores;
}

CpuReport probe_cpu() {
  CpuReport report;
#if defined(_M_X64) || defined(__x86_64__)
  report.architecture = "x86-64";
#elif defined(_M_ARM64) || defined(__aarch64__)
  report.architecture = "arm64";
#else
  report.architecture = "unknown";
#endif
  report.logical_processors = GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
  report.processor_groups = GetActiveProcessorGroupCount();
  report.physical_cores = count_physical_cores();

  if (!cpuid_supported()) {
    return report;
  }
  report.cpuid_available = true;

  const CpuIdRegs leaf0 = cpuid(0, 0);
  char vendor[13] = {};
  std::memcpy(vendor, &leaf0.ebx, 4);
  std::memcpy(vendor + 4, &leaf0.edx, 4);
  std::memcpy(vendor + 8, &leaf0.ecx, 4);
  report.vendor = vendor;

  const CpuIdRegs leaf1 = cpuid(1, 0);
  report.stepping = static_cast<int>(leaf1.eax & 0xF);
  int family = static_cast<int>((leaf1.eax >> 8) & 0xF);
  int model = static_cast<int>((leaf1.eax >> 4) & 0xF);
  const int ext_family = static_cast<int>((leaf1.eax >> 20) & 0xFF);
  const int ext_model = static_cast<int>((leaf1.eax >> 16) & 0xF);
  if (family == 0xF) {
    family += ext_family;
  }
  if (family == 0x6 || family == 0xF) {
    model += ext_model << 4;
  }
  report.family = family;
  report.model = model;

  const bool osxsave = (leaf1.ecx & (1u << 27)) != 0;
  const std::uint64_t xcr0 = osxsave ? read_xcr0() : 0;
  // XCR0 bit 1 = SSE state, bit 2 = AVX (YMM) state, bits 5-7 = AVX-512
  // opmask/ZMM state. A feature is OS-enabled only when the OS saves the
  // registers the feature writes - CPUID alone must never enable dispatch.
  const bool sse_os = (xcr0 & 0x2) != 0;
  const bool avx_os = osxsave && (xcr0 & 0x6) == 0x6;
  const bool avx512_os = osxsave && (xcr0 & 0xE6) == 0xE6;

  report.sse2 = { (leaf1.edx & (1u << 26)) != 0, true };  // baseline register state on x86-64
  report.sse41 = { (leaf1.ecx & (1u << 19)) != 0, true };
  report.sse42 = { (leaf1.ecx & (1u << 20)) != 0, true };
  report.avx = { (leaf1.ecx & (1u << 28)) != 0, avx_os };
  report.f16c = { (leaf1.ecx & (1u << 29)) != 0, avx_os };
  report.fma = { (leaf1.ecx & (1u << 12)) != 0, avx_os };

  if (leaf0.eax >= 7) {
    const CpuIdRegs leaf7 = cpuid(7, 0);
    report.avx2 = { (leaf7.ebx & (1u << 5)) != 0, avx_os };
    report.avx512f = { (leaf7.ebx & (1u << 16)) != 0, avx512_os };
    report.avx512bw = { (leaf7.ebx & (1u << 30)) != 0, avx512_os };
    report.avx512vl = { (leaf7.ebx & (1u << 31)) != 0, avx512_os };
    report.avx512_vnni = { (leaf7.ecx & (1u << 11)) != 0, avx512_os };
    if (leaf7.eax >= 1) {
      const CpuIdRegs leaf7_1 = cpuid(7, 1);
      report.avx_vnni = { (leaf7_1.eax & (1u << 4)) != 0, avx_os };
    }
  }
  (void)sse_os;

  const CpuIdRegs ext0 = cpuid(0x80000000u, 0);
  if (ext0.eax >= 0x80000004u) {
    char brand[49] = {};
    for (std::uint32_t part = 0; part < 3; part += 1) {
      const CpuIdRegs regs = cpuid(0x80000002u + part, 0);
      std::memcpy(brand + part * 16 + 0, &regs.eax, 4);
      std::memcpy(brand + part * 16 + 4, &regs.ebx, 4);
      std::memcpy(brand + part * 16 + 8, &regs.ecx, 4);
      std::memcpy(brand + part * 16 + 12, &regs.edx, 4);
    }
    std::string trimmed(brand);
    const std::size_t start = trimmed.find_first_not_of(' ');
    report.brand = start == std::string::npos ? "" : trimmed.substr(start);
  }
  return report;
}

void write_cpu_feature(JsonWriter &json, const char *name, const CpuFeature &feature) {
  json.key(name);
  json.begin_object();
  json.key("cpu");
  json.value(feature.cpu);
  json.key("os");
  json.value(feature.os);
  json.end_object();
}

// ---- Detection stage B/C: DXGI adapter inventory + D3D12 probe -----------

template <typename T>
struct ComPtr {
  T *ptr = nullptr;
  ~ComPtr() { reset(); }
  void reset(T *next = nullptr) {
    if (ptr != nullptr) {
      ptr->Release();
    }
    ptr = next;
  }
  T *get() const { return ptr; }
  T **put() { reset(); return &ptr; }
  T *operator->() const { return ptr; }
  explicit operator bool() const { return ptr != nullptr; }
};

std::string hresult_text(const char *operation, HRESULT hr) {
  char buffer[96];
  std::snprintf(buffer, sizeof(buffer), "%s failed (0x%08lx)", operation, static_cast<unsigned long>(hr));
  return buffer;
}

// Minimal end-to-end compute self-test: allocation, dispatch, fence,
// readback, verify. Proves the driver can actually execute a shader, not
// merely create a device (Detection stage C requirement).
const char k_selftest_shader[] =
  "RWByteAddressBuffer output_buffer : register(u0);\n"
  "[numthreads(64, 1, 1)]\n"
  "void main(uint3 tid : SV_DispatchThreadID) {\n"
  "  output_buffer.Store(tid.x * 4u, tid.x * 2u + 1u);\n"
  "}\n";

bool run_compute_selftest(ID3D12Device *device, std::string &failure) {
  ComPtr<ID3DBlob> shader;
  {
    ID3DBlob *shader_raw = nullptr;
    ID3DBlob *errors_raw = nullptr;
    const HRESULT hr = D3DCompile(
      k_selftest_shader, sizeof(k_selftest_shader) - 1, "lizard_probe_selftest",
      nullptr, nullptr, "main", "cs_5_0", 0, 0, &shader_raw, &errors_raw);
    ComPtr<ID3DBlob> errors;
    errors.reset(errors_raw);
    if (FAILED(hr)) {
      failure = hresult_text("D3DCompile", hr);
      return false;
    }
    shader.reset(shader_raw);
  }

  D3D12_ROOT_PARAMETER root_param = {};
  root_param.ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
  root_param.Descriptor.ShaderRegister = 0;
  root_param.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
  D3D12_ROOT_SIGNATURE_DESC root_desc = {};
  root_desc.NumParameters = 1;
  root_desc.pParameters = &root_param;

  ComPtr<ID3DBlob> root_blob;
  {
    ID3DBlob *root_raw = nullptr;
    ID3DBlob *errors_raw = nullptr;
    const HRESULT hr = D3D12SerializeRootSignature(
      &root_desc, D3D_ROOT_SIGNATURE_VERSION_1, &root_raw, &errors_raw);
    ComPtr<ID3DBlob> errors;
    errors.reset(errors_raw);
    if (FAILED(hr)) {
      failure = hresult_text("D3D12SerializeRootSignature", hr);
      return false;
    }
    root_blob.reset(root_raw);
  }

  ComPtr<ID3D12RootSignature> root_signature;
  HRESULT hr = device->CreateRootSignature(
    0, root_blob->GetBufferPointer(), root_blob->GetBufferSize(),
    IID_PPV_ARGS(root_signature.put()));
  if (FAILED(hr)) {
    failure = hresult_text("CreateRootSignature", hr);
    return false;
  }

  D3D12_COMPUTE_PIPELINE_STATE_DESC pso_desc = {};
  pso_desc.pRootSignature = root_signature.get();
  pso_desc.CS = { shader->GetBufferPointer(), shader->GetBufferSize() };
  ComPtr<ID3D12PipelineState> pipeline;
  hr = device->CreateComputePipelineState(&pso_desc, IID_PPV_ARGS(pipeline.put()));
  if (FAILED(hr)) {
    failure = hresult_text("CreateComputePipelineState", hr);
    return false;
  }

  constexpr UINT64 k_buffer_bytes = 64 * 4;
  D3D12_HEAP_PROPERTIES default_heap = {};
  default_heap.Type = D3D12_HEAP_TYPE_DEFAULT;
  D3D12_HEAP_PROPERTIES readback_heap = {};
  readback_heap.Type = D3D12_HEAP_TYPE_READBACK;
  D3D12_RESOURCE_DESC buffer_desc = {};
  buffer_desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
  buffer_desc.Width = k_buffer_bytes;
  buffer_desc.Height = 1;
  buffer_desc.DepthOrArraySize = 1;
  buffer_desc.MipLevels = 1;
  buffer_desc.SampleDesc.Count = 1;
  buffer_desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
  buffer_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

  ComPtr<ID3D12Resource> output;
  hr = device->CreateCommittedResource(
    &default_heap, D3D12_HEAP_FLAG_NONE, &buffer_desc,
    D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(output.put()));
  if (FAILED(hr)) {
    failure = hresult_text("CreateCommittedResource(output)", hr);
    return false;
  }

  buffer_desc.Flags = D3D12_RESOURCE_FLAG_NONE;
  ComPtr<ID3D12Resource> readback;
  hr = device->CreateCommittedResource(
    &readback_heap, D3D12_HEAP_FLAG_NONE, &buffer_desc,
    D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(readback.put()));
  if (FAILED(hr)) {
    failure = hresult_text("CreateCommittedResource(readback)", hr);
    return false;
  }

  D3D12_COMMAND_QUEUE_DESC queue_desc = {};
  queue_desc.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
  ComPtr<ID3D12CommandQueue> queue;
  hr = device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(queue.put()));
  if (FAILED(hr)) {
    failure = hresult_text("CreateCommandQueue", hr);
    return false;
  }
  ComPtr<ID3D12CommandAllocator> allocator;
  hr = device->CreateCommandAllocator(
    D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(allocator.put()));
  if (FAILED(hr)) {
    failure = hresult_text("CreateCommandAllocator", hr);
    return false;
  }
  ComPtr<ID3D12GraphicsCommandList> list;
  hr = device->CreateCommandList(
    0, D3D12_COMMAND_LIST_TYPE_COMPUTE, allocator.get(), nullptr, IID_PPV_ARGS(list.put()));
  if (FAILED(hr)) {
    failure = hresult_text("CreateCommandList", hr);
    return false;
  }

  list->SetPipelineState(pipeline.get());
  list->SetComputeRootSignature(root_signature.get());
  list->SetComputeRootUnorderedAccessView(0, output->GetGPUVirtualAddress());
  list->Dispatch(1, 1, 1);

  D3D12_RESOURCE_BARRIER barrier = {};
  barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
  barrier.Transition.pResource = output.get();
  barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
  barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
  barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
  list->ResourceBarrier(1, &barrier);
  list->CopyResource(readback.get(), output.get());
  hr = list->Close();
  if (FAILED(hr)) {
    failure = hresult_text("CommandList::Close", hr);
    return false;
  }

  ComPtr<ID3D12Fence> fence;
  hr = device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(fence.put()));
  if (FAILED(hr)) {
    failure = hresult_text("CreateFence", hr);
    return false;
  }
  ID3D12CommandList *lists[] = { list.get() };
  queue->ExecuteCommandLists(1, lists);
  hr = queue->Signal(fence.get(), 1);
  if (FAILED(hr)) {
    failure = hresult_text("Queue::Signal", hr);
    return false;
  }
  const HANDLE event_handle = CreateEventW(nullptr, FALSE, FALSE, nullptr);
  if (event_handle == nullptr) {
    failure = "CreateEvent failed";
    return false;
  }
  fence->SetEventOnCompletion(1, event_handle);
  const DWORD wait_result = WaitForSingleObject(event_handle, 5000);
  CloseHandle(event_handle);
  if (wait_result != WAIT_OBJECT_0) {
    failure = "GPU fence wait timed out (5 s)";
    return false;
  }
  if (device->GetDeviceRemovedReason() != S_OK) {
    failure = "device removed during self-test";
    return false;
  }

  void *mapped = nullptr;
  const D3D12_RANGE read_range = { 0, k_buffer_bytes };
  hr = readback->Map(0, &read_range, &mapped);
  if (FAILED(hr) || mapped == nullptr) {
    failure = hresult_text("Readback Map", hr);
    return false;
  }
  const std::uint32_t *values = static_cast<const std::uint32_t *>(mapped);
  bool correct = true;
  for (std::uint32_t index = 0; index < 64; index += 1) {
    if (values[index] != index * 2u + 1u) {
      correct = false;
      break;
    }
  }
  readback->Unmap(0, nullptr);
  if (!correct) {
    failure = "readback values did not match the dispatched computation";
    return false;
  }
  return true;
}

std::string shader_model_label(UINT value) {
  char buffer[16];
  std::snprintf(buffer, sizeof(buffer), "%u.%u", (value >> 4) & 0xF, value & 0xF);
  return buffer;
}

std::string dxgi_driver_version(const LARGE_INTEGER &version) {
  char buffer[48];
  std::snprintf(
    buffer,
    sizeof(buffer),
    "%u.%u.%u.%u",
    static_cast<unsigned int>(HIWORD(version.HighPart)),
    static_cast<unsigned int>(LOWORD(version.HighPart)),
    static_cast<unsigned int>(HIWORD(version.LowPart)),
    static_cast<unsigned int>(LOWORD(version.LowPart)));
  return buffer;
}

void write_memory_segment(JsonWriter &json, const char *name, const DXGI_QUERY_VIDEO_MEMORY_INFO &info, bool available) {
  json.key(name);
  if (!available) {
    json.value_null();
    return;
  }
  json.begin_object();
  json.key("budgetBytes");
  json.value(static_cast<std::uint64_t>(info.Budget));
  json.key("currentUsageBytes");
  json.value(static_cast<std::uint64_t>(info.CurrentUsage));
  json.key("availableForReservationBytes");
  json.value(static_cast<std::uint64_t>(info.AvailableForReservation));
  json.end_object();
}

void probe_adapters(JsonWriter &json) {
  json.key("adapters");
  json.begin_array();

  ComPtr<IDXGIFactory1> factory;
  {
    IDXGIFactory1 *factory_raw = nullptr;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory_raw)))) {
      json.end_array();
      return;
    }
    factory.reset(factory_raw);
  }

  for (UINT index = 0;; index += 1) {
    IDXGIAdapter1 *adapter_raw = nullptr;
    const HRESULT enum_hr = factory->EnumAdapters1(index, &adapter_raw);
    if (enum_hr == DXGI_ERROR_NOT_FOUND || FAILED(enum_hr)) {
      break;
    }
    ComPtr<IDXGIAdapter1> adapter;
    adapter.reset(adapter_raw);

    DXGI_ADAPTER_DESC1 desc = {};
    adapter->GetDesc1(&desc);
    const bool is_software = (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) != 0;

    json.begin_object();
    json.key("luidHigh");
    json.value(static_cast<std::int64_t>(desc.AdapterLuid.HighPart));
    json.key("luidLow");
    json.value(static_cast<std::uint64_t>(desc.AdapterLuid.LowPart));
    json.key("vendorId");
    json.value(static_cast<std::uint64_t>(desc.VendorId));
    json.key("deviceId");
    json.value(static_cast<std::uint64_t>(desc.DeviceId));
    json.key("name");
    json.value(narrow_utf16(desc.Description));
    json.key("dedicatedVideoMemoryBytes");
    json.value(static_cast<std::uint64_t>(desc.DedicatedVideoMemory));
    json.key("sharedSystemMemoryBytes");
    json.value(static_cast<std::uint64_t>(desc.SharedSystemMemory));
    json.key("isSoftware");
    json.value(is_software);
    LARGE_INTEGER driver_version = {};
    json.key("driverVersion");
    // Microsoft documents IDXGIDevice as the driver-version query GUID.
    // D3D11+ interface GUIDs return DXGI_ERROR_UNSUPPORTED here.
    if (SUCCEEDED(adapter->CheckInterfaceSupport(__uuidof(IDXGIDevice), &driver_version))) {
      json.value(dxgi_driver_version(driver_version));
    } else {
      json.value_null();
    }

    // Live memory budgets come from DXGI, not from the D3D12 device, so
    // report them even when device creation fails.
    {
      IDXGIAdapter3 *adapter3_raw = nullptr;
      const bool has_adapter3 =
        SUCCEEDED(adapter->QueryInterface(IID_PPV_ARGS(&adapter3_raw)));
      ComPtr<IDXGIAdapter3> adapter3;
      adapter3.reset(adapter3_raw);
      DXGI_QUERY_VIDEO_MEMORY_INFO local_info = {};
      DXGI_QUERY_VIDEO_MEMORY_INFO non_local_info = {};
      const bool local_ok = has_adapter3
        && SUCCEEDED(adapter3->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &local_info));
      const bool non_local_ok = has_adapter3
        && SUCCEEDED(adapter3->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, &non_local_info));
      json.key("videoMemory");
      json.begin_object();
      write_memory_segment(json, "local", local_info, local_ok);
      write_memory_segment(json, "nonLocal", non_local_info, non_local_ok);
      json.end_object();
    }

    json.key("d3d12");
    json.begin_object();
    ComPtr<ID3D12Device> device;
    {
      ID3D12Device *device_raw = nullptr;
      const HRESULT hr = D3D12CreateDevice(
        adapter.get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device_raw));
      if (SUCCEEDED(hr)) {
        device.reset(device_raw);
      }
      json.key("deviceCreated");
      json.value(static_cast<bool>(device));
      if (!device) {
        json.key("state");
        json.value("device-unsupported");
        json.key("detail");
        json.value(hresult_text("D3D12CreateDevice(FL 11_0)", hr));
      }
    }
    if (device) {
      D3D12_FEATURE_DATA_ARCHITECTURE architecture = {};
      if (SUCCEEDED(device->CheckFeatureSupport(
            D3D12_FEATURE_ARCHITECTURE, &architecture, sizeof(architecture)))) {
        json.key("uma");
        json.value(architecture.UMA != FALSE);
        json.key("cacheCoherentUma");
        json.value(architecture.CacheCoherentUMA != FALSE);
      }

      // Highest shader model: the query fails outright for enum values the
      // installed runtime does not know, so walk downward.
      static const UINT k_shader_models[] = { 0x66, 0x65, 0x64, 0x63, 0x62, 0x61, 0x60, 0x51, 0x50 };
      for (const UINT candidate : k_shader_models) {
        D3D12_FEATURE_DATA_SHADER_MODEL shader_model = {};
        shader_model.HighestShaderModel = static_cast<D3D_SHADER_MODEL>(candidate);
        if (SUCCEEDED(device->CheckFeatureSupport(
              D3D12_FEATURE_SHADER_MODEL, &shader_model, sizeof(shader_model)))) {
          json.key("highestShaderModel");
          json.value(shader_model_label(static_cast<UINT>(shader_model.HighestShaderModel)));
          break;
        }
      }

      D3D12_FEATURE_DATA_D3D12_OPTIONS1 options1 = {};
      if (SUCCEEDED(device->CheckFeatureSupport(
            D3D12_FEATURE_D3D12_OPTIONS1, &options1, sizeof(options1)))) {
        json.key("waveOps");
        json.value(options1.WaveOps != FALSE);
        json.key("waveLaneCountMin");
        json.value(static_cast<std::uint64_t>(options1.WaveLaneCountMin));
      }

      std::string failure;
      const bool selftest_ok = run_compute_selftest(device.get(), failure);
      json.key("selfTest");
      json.value(selftest_ok ? std::string("passed") : ("failed: " + failure));
      json.key("state");
      // Vocabulary from src/lib/hardware-contracts.js BACKEND_PROBE_STATES.
      json.value(selftest_ok ? "usable" : "feature-limited");
    }
    json.end_object();

    json.end_object();
  }
  json.end_array();
}

#endif  // _WIN32

}  // namespace

int run_hardware_probe_main() {
  JsonWriter json;
  json.begin_object();
  json.key("schemaVersion");
  json.value(1);
#if defined(_WIN32)
  json.key("platform");
  json.value("windows");

  const CpuReport cpu = probe_cpu();
  json.key("cpu");
  json.begin_object();
  json.key("architecture");
  json.value(cpu.architecture);
  json.key("cpuidAvailable");
  json.value(cpu.cpuid_available);
  json.key("vendor");
  json.value(cpu.vendor);
  json.key("brand");
  json.value(cpu.brand);
  json.key("family");
  json.value(cpu.family);
  json.key("model");
  json.value(cpu.model);
  json.key("stepping");
  json.value(cpu.stepping);
  json.key("logicalProcessors");
  json.value(static_cast<std::uint64_t>(cpu.logical_processors));
  json.key("physicalCores");
  json.value(static_cast<std::uint64_t>(cpu.physical_cores));
  json.key("processorGroups");
  json.value(static_cast<std::uint64_t>(cpu.processor_groups));
  json.key("features");
  json.begin_object();
  write_cpu_feature(json, "sse2", cpu.sse2);
  write_cpu_feature(json, "sse41", cpu.sse41);
  write_cpu_feature(json, "sse42", cpu.sse42);
  write_cpu_feature(json, "avx", cpu.avx);
  write_cpu_feature(json, "f16c", cpu.f16c);
  write_cpu_feature(json, "fma", cpu.fma);
  write_cpu_feature(json, "avx2", cpu.avx2);
  write_cpu_feature(json, "avx512f", cpu.avx512f);
  write_cpu_feature(json, "avx512bw", cpu.avx512bw);
  write_cpu_feature(json, "avx512vl", cpu.avx512vl);
  write_cpu_feature(json, "avx512Vnni", cpu.avx512_vnni);
  write_cpu_feature(json, "avxVnni", cpu.avx_vnni);
  json.end_object();
  json.end_object();

  MEMORYSTATUSEX memory_status = {};
  memory_status.dwLength = sizeof(memory_status);
  json.key("memory");
  if (GlobalMemoryStatusEx(&memory_status)) {
    json.begin_object();
    json.key("totalBytes");
    json.value(static_cast<std::uint64_t>(memory_status.ullTotalPhys));
    json.key("availableBytes");
    json.value(static_cast<std::uint64_t>(memory_status.ullAvailPhys));
    json.key("commitLimitBytes");
    json.value(static_cast<std::uint64_t>(memory_status.ullTotalPageFile));
    json.key("memoryLoadPercent");
    json.value(static_cast<std::uint64_t>(memory_status.dwMemoryLoad));
    json.end_object();
  } else {
    json.value_null();
  }

  probe_adapters(json);
#else
  json.key("platform");
  json.value("unsupported");
  json.key("adapters");
  json.begin_array();
  json.end_array();
#endif
  json.end_object();

  std::cout << json.str() << '\n';
  std::cout.flush();
  return 0;
}

}  // namespace lizard_native
