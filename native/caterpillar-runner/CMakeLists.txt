cmake_minimum_required(VERSION 3.20)
project(caterpillar_runner VERSION 0.1.0 LANGUAGES CXX)

add_executable(caterpillar-runner
  src/main.cpp
  src/attention.cpp
  src/graph.cpp
  src/alloc.cpp
  src/plan.cpp
  src/gguf_loader.cpp
  src/tensor_data.cpp
  src/mapped_model.cpp
  src/cpu_model.cpp
  src/server.cpp
  src/tokenizer.cpp
  src/family_spec.cpp
  src/fit.cpp
  src/sched.cpp
  src/backend_d3d12.cpp
  src/gpu_model.cpp
  src/kv_cache.cpp
  src/spec.cpp
  src/selftest.cpp
)
target_include_directories(caterpillar-runner PRIVATE src)
target_compile_features(caterpillar-runner PRIVATE cxx_std_20)

if(MSVC)
  target_compile_options(caterpillar-runner PRIVATE /W4 /permissive- /EHsc)
else()
  target_compile_options(caterpillar-runner PRIVATE -Wall -Wextra -Wpedantic)
endif()

# On GCC/MinGW, link the runtime statically so the produced executable is
# self-contained and does not depend on libstdc++/libgcc/winpthread DLLs
# (Node spawns it without the MinGW bin directory on PATH; a dynamic link
# fails with Windows loader error 127). Mirrors lizard-native-runner.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  target_link_options(caterpillar-runner PRIVATE
    -static -static-libgcc -static-libstdc++)
endif()

# Host-calibration probes (2026-07-21 roofline campaign, Phase 1). Built on
# demand only: `cmake --build build --target caterpillar-probes`. bw_probe
# measures the host's sustained CPU memory bandwidth — the denominator to pass
# as CATERPILLAR_ROOFLINE_GBS on ANY machine (laptop iGPU, desktop dGPU,
# server); vnni_probe qualifies AVX-512 VNNI on the local toolchain/CPU.
add_executable(caterpillar-bw-probe EXCLUDE_FROM_ALL tools/bw_probe.cpp)
add_executable(caterpillar-vnni-probe EXCLUDE_FROM_ALL tools/vnni_probe.cpp)
add_custom_target(caterpillar-probes
  DEPENDS caterpillar-bw-probe caterpillar-vnni-probe)
if(WIN32)
  # bw_probe's Phase 7 section streams through a D3D12 compute shader to
  # measure concurrent CPU+GPU DRAM bandwidth (UMA hosts only).
  target_link_libraries(caterpillar-bw-probe PRIVATE dxgi d3d12 d3dcompiler)
endif()
foreach(probe caterpillar-bw-probe caterpillar-vnni-probe)
  target_compile_features(${probe} PRIVATE cxx_std_20)
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_link_options(${probe} PRIVATE -static -static-libgcc -static-libstdc++)
  endif()
endforeach()

if(WIN32)
  target_link_libraries(caterpillar-runner PRIVATE dxgi d3d12 d3dcompiler)

  # Phase 5 packed-int8 dot kernels require DXC / Shader Model 6.4. Build
  # them into signed DXIL headers when dxc.exe is available; otherwise the
  # runner remains fully buildable and uses the Phase 3 FXC float kernels.
  set(CATERPILLAR_DXC "" CACHE FILEPATH "Path to dxc.exe for SM 6.4 kernel compilation")
  if(NOT CATERPILLAR_DXC)
    file(GLOB _caterpillar_dxc_candidates
      "$ENV{LOCALAPPDATA}/Microsoft/WinGet/Packages/Microsoft.DirectX.ShaderCompiler_*/bin/x64/dxc.exe")
    if(_caterpillar_dxc_candidates)
      list(GET _caterpillar_dxc_candidates 0 CATERPILLAR_DXC)
    endif()
  endif()
  if(CATERPILLAR_DXC AND EXISTS "${CATERPILLAR_DXC}")
    message(STATUS "Caterpillar DP4A kernels enabled via DXC: ${CATERPILLAR_DXC}")
    set(_caterpillar_dxil_dir "${CMAKE_BINARY_DIR}/dxil")
    file(MAKE_DIRECTORY "${_caterpillar_dxil_dir}")
    set(_caterpillar_dp4a_hlsl
      "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/dp4a_quantized_matvec.hlsl")
    add_custom_command(
      OUTPUT "${_caterpillar_dxil_dir}/caterpillar_quantize_q8_1.h"
      COMMAND "${CATERPILLAR_DXC}" -T cs_6_4 -E cs_quantize_q8_1 -O3
        -Fh "${_caterpillar_dxil_dir}/caterpillar_quantize_q8_1.h"
        -Vn g_caterpillar_quantize_q8_1 "${_caterpillar_dp4a_hlsl}"
      DEPENDS "${_caterpillar_dp4a_hlsl}"
      COMMENT "DXC cs_quantize_q8_1 -> signed DXIL header")
    add_custom_command(
      OUTPUT "${_caterpillar_dxil_dir}/caterpillar_q4k_q8_1.h"
      COMMAND "${CATERPILLAR_DXC}" -T cs_6_4 -E cs_matvec_q4k_q8_1 -O3
        -Fh "${_caterpillar_dxil_dir}/caterpillar_q4k_q8_1.h"
        -Vn g_caterpillar_q4k_q8_1 "${_caterpillar_dp4a_hlsl}"
      DEPENDS "${_caterpillar_dp4a_hlsl}"
      COMMENT "DXC cs_matvec_q4k_q8_1 -> signed DXIL header")
    add_custom_command(
      OUTPUT "${_caterpillar_dxil_dir}/caterpillar_q5k_q8_1.h"
      COMMAND "${CATERPILLAR_DXC}" -T cs_6_4 -E cs_matvec_q5k_q8_1 -O3
        -Fh "${_caterpillar_dxil_dir}/caterpillar_q5k_q8_1.h"
        -Vn g_caterpillar_q5k_q8_1 "${_caterpillar_dp4a_hlsl}"
      DEPENDS "${_caterpillar_dp4a_hlsl}")
    add_custom_command(
      OUTPUT "${_caterpillar_dxil_dir}/caterpillar_q6k_q8_1.h"
      COMMAND "${CATERPILLAR_DXC}" -T cs_6_4 -E cs_matvec_q6k_q8_1 -O3
        -Fh "${_caterpillar_dxil_dir}/caterpillar_q6k_q8_1.h"
        -Vn g_caterpillar_q6k_q8_1 "${_caterpillar_dp4a_hlsl}"
      DEPENDS "${_caterpillar_dp4a_hlsl}")
    add_custom_command(
      OUTPUT "${_caterpillar_dxil_dir}/caterpillar_q8_0_q8_1.h"
      COMMAND "${CATERPILLAR_DXC}" -T cs_6_4 -E cs_matvec_q8_0_q8_1 -O3
        -Fh "${_caterpillar_dxil_dir}/caterpillar_q8_0_q8_1.h"
        -Vn g_caterpillar_q8_0_q8_1 "${_caterpillar_dp4a_hlsl}"
      DEPENDS "${_caterpillar_dp4a_hlsl}")
    add_custom_command(
      OUTPUT "${_caterpillar_dxil_dir}/caterpillar_q4_0_q8_1.h"
      COMMAND "${CATERPILLAR_DXC}" -T cs_6_4 -E cs_matvec_q4_0_q8_1 -O3
        -Fh "${_caterpillar_dxil_dir}/caterpillar_q4_0_q8_1.h"
        -Vn g_caterpillar_q4_0_q8_1 "${_caterpillar_dp4a_hlsl}"
      DEPENDS "${_caterpillar_dp4a_hlsl}")
    add_custom_target(caterpillar-dxil-headers DEPENDS
      "${_caterpillar_dxil_dir}/caterpillar_quantize_q8_1.h"
      "${_caterpillar_dxil_dir}/caterpillar_q4k_q8_1.h"
      "${_caterpillar_dxil_dir}/caterpillar_q5k_q8_1.h"
      "${_caterpillar_dxil_dir}/caterpillar_q6k_q8_1.h"
      "${_caterpillar_dxil_dir}/caterpillar_q8_0_q8_1.h"
      "${_caterpillar_dxil_dir}/caterpillar_q4_0_q8_1.h")
    add_dependencies(caterpillar-runner caterpillar-dxil-headers)
    target_include_directories(caterpillar-runner PRIVATE "${_caterpillar_dxil_dir}")
    target_compile_definitions(caterpillar-runner PRIVATE CATERPILLAR_HAVE_DXIL=1)
  else()
    message(STATUS "Caterpillar DP4A kernels disabled: dxc.exe not found")
  endif()
endif()

enable_testing()
add_test(NAME caterpillar_runner_selftest COMMAND caterpillar-runner --selftest)
add_test(NAME caterpillar_graph_replay_selftest COMMAND caterpillar-runner --graph-selftest)
add_test(NAME caterpillar_gguf_loader_selftest COMMAND caterpillar-runner --gguf-selftest)
add_test(NAME caterpillar_tensor_data_selftest COMMAND caterpillar-runner --tensor-selftest)
add_test(NAME caterpillar_mapped_model_selftest COMMAND caterpillar-runner --mapped-model-selftest)
add_test(NAME caterpillar_tokenizer_selftest COMMAND caterpillar-runner --tokenizer-selftest)
add_test(NAME caterpillar_family_graph_selftest COMMAND caterpillar-runner --family-graph-selftest)
add_test(NAME caterpillar_fit_selftest COMMAND caterpillar-runner --fit-selftest)
add_test(NAME caterpillar_gpu_selftest COMMAND caterpillar-runner --gpu-selftest)
add_test(NAME caterpillar_kv_cache_selftest COMMAND caterpillar-runner --kv-cache-selftest)
add_test(NAME caterpillar_spec_selftest COMMAND caterpillar-runner --spec-selftest)
