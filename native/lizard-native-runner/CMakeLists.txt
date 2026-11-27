cmake_minimum_required(VERSION 3.20)
project(lizard-native-runner LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(lizard-native-runner
  src/d3d12_batch_matmul.cpp
  src/engine_loop.cpp
  src/ggml_cpu_backend.cpp
  src/kv_blocks.cpp
  src/prefix_cache.cpp
  src/hardware_probe.cpp
  src/main.cpp
)

if(WIN32)
  target_link_libraries(lizard-native-runner PRIVATE psapi)
endif()

# Release/installable binaries must run on CPUs other than the build host.
# Keep host-specific code generation opt-in: -march=native and /arch:AVX2 can
# make the worker fail at process startup with STATUS_ILLEGAL_INSTRUCTION on
# an older x86-64 CPU, before runtime capability detection can do anything.
option(LIZARD_NATIVE_OPTIMIZE_FOR_HOST "Use the build host CPU instruction set (development benchmarks only)" OFF)
set(LIZARD_NATIVE_CPU_VARIANT "baseline" CACHE STRING "Release CPU variant: baseline, avx2, or avx512")
set_property(CACHE LIZARD_NATIVE_CPU_VARIANT PROPERTY STRINGS baseline avx2 avx512)

# Enhancement Run 06 foundation: a source- and checksum-pinned ggml CPU lane.
# Lizard retains its GGUF/session/service runtime; only CPU tensor operations
# are delegated. The exact source can be supplied locally for offline/release
# builds. The supported build script hydrates that exact commit into its local
# dependency cache before configuring CMake.
option(LIZARD_ENABLE_GGML_CPU_BACKEND "Build the pinned ggml CPU backend" ON)
set(LIZARD_GGML_SOURCE_DIR "" CACHE PATH "Optional local llama.cpp checkout at the pinned commit")
set(LIZARD_GGML_COMMIT "ddd4ec1428a6201e18975ea52b07c71e0f9aef26")

if(LIZARD_NATIVE_OPTIMIZE_FOR_HOST)
  target_compile_definitions(lizard-native-runner PRIVATE LIZARD_NATIVE_HOST_OPTIMIZED=1)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(lizard-native-runner PRIVATE
      $<$<CONFIG:Release>:-march=native>
      $<$<CONFIG:Release>:-mtune=native>)
  elseif(MSVC)
    target_compile_options(lizard-native-runner PRIVATE
      $<$<CONFIG:Release>:/arch:AVX2>)
  endif()
elseif(LIZARD_NATIVE_CPU_VARIANT STREQUAL "avx2")
  target_compile_definitions(lizard-native-runner PRIVATE LIZARD_NATIVE_AVX2=1)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(lizard-native-runner PRIVATE
      $<$<CONFIG:Release>:-march=x86-64>
      $<$<CONFIG:Release>:-mavx2>
      $<$<CONFIG:Release>:-mfma>
      $<$<CONFIG:Release>:-mf16c>
      $<$<CONFIG:Release>:-mtune=generic>)
  elseif(MSVC)
    target_compile_options(lizard-native-runner PRIVATE
      $<$<CONFIG:Release>:/arch:AVX2>)
  endif()
elseif(LIZARD_NATIVE_CPU_VARIANT STREQUAL "avx512")
  # AVX-512 F/BW/VL baseline (Skylake-X / Ice Lake+ / Zen 4+). Implies the
  # AVX2+FMA+F16C code paths, so LIZARD_NATIVE_AVX2 stays defined and the
  # existing intrinsics keep compiling; LIZARD_NATIVE_AVX512 marks the wider
  # variant for capability reporting and future 512-bit kernels.
  target_compile_definitions(lizard-native-runner PRIVATE
    LIZARD_NATIVE_AVX2=1
    LIZARD_NATIVE_AVX512=1)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(lizard-native-runner PRIVATE
      $<$<CONFIG:Release>:-march=x86-64>
      $<$<CONFIG:Release>:-mavx2>
      $<$<CONFIG:Release>:-mfma>
      $<$<CONFIG:Release>:-mf16c>
      $<$<CONFIG:Release>:-mavx512f>
      $<$<CONFIG:Release>:-mavx512bw>
      $<$<CONFIG:Release>:-mavx512vl>
      $<$<CONFIG:Release>:-mtune=generic>)
  elseif(MSVC)
    target_compile_options(lizard-native-runner PRIVATE
      $<$<CONFIG:Release>:/arch:AVX512>)
  endif()
elseif(LIZARD_NATIVE_CPU_VARIANT STREQUAL "baseline")
  target_compile_definitions(lizard-native-runner PRIVATE LIZARD_NATIVE_PORTABLE_CPU=1)
  # A 64-bit Windows installation guarantees the x86-64/SSE2 baseline. Pin
  # GCC/Clang to that baseline so a toolchain default or reused CMake cache
  # cannot silently inherit AVX/AVX2/FMA instructions from the build PC.
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang"
      AND CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64|amd64|x86_64|x64")
    target_compile_options(lizard-native-runner PRIVATE
      $<$<CONFIG:Release>:-march=x86-64>
      $<$<CONFIG:Release>:-mtune=generic>)
  endif()
else()
  message(FATAL_ERROR "Unsupported LIZARD_NATIVE_CPU_VARIANT=${LIZARD_NATIVE_CPU_VARIANT}")
endif()

if(LIZARD_ENABLE_GGML_CPU_BACKEND)
  set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
  set(GGML_STATIC ON CACHE BOOL "" FORCE)
  set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
  set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
  set(GGML_BLAS OFF CACHE BOOL "" FORCE)
  set(GGML_LLAMAFILE OFF CACHE BOOL "" FORCE)
  set(GGML_CPU_REPACK ON CACHE BOOL "" FORCE)
  set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(GGML_CCACHE OFF CACHE BOOL "" FORCE)

  # The dependency is built to the same release-safe ISA tier as its Lizard
  # executable. Do not let ggml silently use the build host's instruction set.
  if(LIZARD_NATIVE_CPU_VARIANT STREQUAL "avx2"
      OR LIZARD_NATIVE_CPU_VARIANT STREQUAL "avx512"
      OR LIZARD_NATIVE_OPTIMIZE_FOR_HOST)
    set(GGML_SSE42 ON CACHE BOOL "" FORCE)
    set(GGML_AVX ON CACHE BOOL "" FORCE)
    set(GGML_AVX2 ON CACHE BOOL "" FORCE)
    set(GGML_FMA ON CACHE BOOL "" FORCE)
    set(GGML_F16C ON CACHE BOOL "" FORCE)
  else()
    set(GGML_SSE42 OFF CACHE BOOL "" FORCE)
    set(GGML_AVX OFF CACHE BOOL "" FORCE)
    set(GGML_AVX2 OFF CACHE BOOL "" FORCE)
    set(GGML_FMA OFF CACHE BOOL "" FORCE)
    set(GGML_F16C OFF CACHE BOOL "" FORCE)
  endif()
  set(GGML_BMI2 OFF CACHE BOOL "" FORCE)
  set(GGML_AVX_VNNI OFF CACHE BOOL "" FORCE)
  if(LIZARD_NATIVE_CPU_VARIANT STREQUAL "avx512")
    set(GGML_AVX512 ON CACHE BOOL "" FORCE)
  else()
    set(GGML_AVX512 OFF CACHE BOOL "" FORCE)
  endif()
  set(GGML_AVX512_VBMI OFF CACHE BOOL "" FORCE)
  if(LIZARD_NATIVE_CPU_VARIANT STREQUAL "avx512")
    set(GGML_AVX512_VNNI ON CACHE BOOL "" FORCE)
  else()
    set(GGML_AVX512_VNNI OFF CACHE BOOL "" FORCE)
  endif()
  set(GGML_AVX512_BF16 OFF CACHE BOOL "" FORCE)

  if(NOT LIZARD_GGML_SOURCE_DIR)
    message(FATAL_ERROR
      "The pinned ggml backend requires LIZARD_GGML_SOURCE_DIR. Use npm run build:native; the build script fetches and verifies ${LIZARD_GGML_COMMIT}.")
  endif()
  if(NOT EXISTS "${LIZARD_GGML_SOURCE_DIR}/ggml/CMakeLists.txt")
    message(FATAL_ERROR "LIZARD_GGML_SOURCE_DIR must contain ggml/CMakeLists.txt")
  endif()
  find_package(Git REQUIRED)
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" -C "${LIZARD_GGML_SOURCE_DIR}" rev-parse HEAD
    RESULT_VARIABLE _lizard_ggml_git_status
    OUTPUT_VARIABLE _lizard_ggml_local_commit
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(NOT _lizard_ggml_git_status EQUAL 0
      OR NOT "${_lizard_ggml_local_commit}" STREQUAL "${LIZARD_GGML_COMMIT}")
    message(FATAL_ERROR
      "LIZARD_GGML_SOURCE_DIR must be Git commit ${LIZARD_GGML_COMMIT}; got '${_lizard_ggml_local_commit}'")
  endif()
  add_subdirectory(
    "${LIZARD_GGML_SOURCE_DIR}/ggml"
    "${CMAKE_BINARY_DIR}/_deps/lizard-ggml-build"
    EXCLUDE_FROM_ALL)
  target_include_directories(lizard-native-runner PRIVATE
    "${LIZARD_GGML_SOURCE_DIR}/ggml/src"
    "${LIZARD_GGML_SOURCE_DIR}/ggml/src/ggml-cpu")

  target_compile_definitions(lizard-native-runner PRIVATE
    LIZARD_HAVE_GGML_CPU_BACKEND=1
    LIZARD_GGML_COMMIT="${LIZARD_GGML_COMMIT}")
  target_link_libraries(lizard-native-runner PRIVATE ggml)
endif()

# On GCC/MinGW, link the runtime statically so the produced executable is
# self-contained and does not depend on libstdc++/libgcc/winpthread DLLs.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  target_link_options(lizard-native-runner PRIVATE
    -static -static-libgcc -static-libstdc++)
endif()

if(WIN32)
  target_link_libraries(lizard-native-runner PRIVATE
    d3d12
    dxgi
    d3dcompiler_47
    ole32
    uuid)

  # Optional Shader Model 6.4 DP4A kernels: compiled offline with DXC to DXIL
  # headers and embedded. When no DXC is found the build proceeds without
  # them and the runtime keeps the FXC float kernels.
  set(LIZARD_DXC "" CACHE FILEPATH "Path to dxc.exe for SM 6.4 kernel compilation")
  if(NOT LIZARD_DXC)
    file(GLOB _lizard_dxc_candidates
      "$ENV{LOCALAPPDATA}/Microsoft/WinGet/Packages/Microsoft.DirectX.ShaderCompiler_*/bin/x64/dxc.exe")
    if(_lizard_dxc_candidates)
      list(GET _lizard_dxc_candidates 0 LIZARD_DXC)
    endif()
  endif()
  if(LIZARD_DXC AND EXISTS "${LIZARD_DXC}")
    message(STATUS "DP4A kernels enabled via DXC: ${LIZARD_DXC}")
    set(_lizard_dxil_dir "${CMAKE_BINARY_DIR}/dxil")
    file(MAKE_DIRECTORY "${_lizard_dxil_dir}")
    set(_lizard_dp4a_hlsl "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/dp4a_q4k.hlsl")
    add_custom_command(
      OUTPUT "${_lizard_dxil_dir}/dp4a_quantize_q8_1.h"
      COMMAND "${LIZARD_DXC}" -T cs_6_4 -E main_quantize_q8_1
        -Fh "${_lizard_dxil_dir}/dp4a_quantize_q8_1.h" -Vn g_dxil_quantize_q8_1
        "${_lizard_dp4a_hlsl}"
      DEPENDS "${_lizard_dp4a_hlsl}"
      COMMENT "DXC main_quantize_q8_1 -> DXIL header")
    add_custom_command(
      OUTPUT "${_lizard_dxil_dir}/dp4a_q4k_matvec.h"
      COMMAND "${LIZARD_DXC}" -T cs_6_4 -E main_q4k_q8_1_b1
        -Fh "${_lizard_dxil_dir}/dp4a_q4k_matvec.h" -Vn g_dxil_q4k_q8_1_b1
        "${_lizard_dp4a_hlsl}"
      DEPENDS "${_lizard_dp4a_hlsl}"
      COMMENT "DXC main_q4k_q8_1_b1 -> DXIL header")
    add_custom_command(
      OUTPUT "${_lizard_dxil_dir}/dp4a_q4k_gemv_wave.h"
      COMMAND "${LIZARD_DXC}" -T cs_6_0 -E main_q4k_gemv_wave
        -Fh "${_lizard_dxil_dir}/dp4a_q4k_gemv_wave.h" -Vn g_dxil_q4k_gemv_wave
        "${_lizard_dp4a_hlsl}"
      DEPENDS "${_lizard_dp4a_hlsl}"
      COMMENT "DXC main_q4k_gemv_wave -> DXIL header")
    add_custom_command(
      OUTPUT "${_lizard_dxil_dir}/dp4a_q4k_batched.h"
      COMMAND "${LIZARD_DXC}" -T cs_6_4 -E main_q4k_q8_1_batched
        -Fh "${_lizard_dxil_dir}/dp4a_q4k_batched.h" -Vn g_dxil_q4k_q8_1_batched
        "${_lizard_dp4a_hlsl}"
      DEPENDS "${_lizard_dp4a_hlsl}"
      COMMENT "DXC main_q4k_q8_1_batched -> DXIL header")
    add_custom_command(
      OUTPUT "${_lizard_dxil_dir}/dp4a_q4k_batched_nolds.h"
      COMMAND "${LIZARD_DXC}" -T cs_6_4 -E main_q4k_q8_1_batched_nolds
        -Fh "${_lizard_dxil_dir}/dp4a_q4k_batched_nolds.h"
        -Vn g_dxil_q4k_q8_1_batched_nolds
        "${_lizard_dp4a_hlsl}"
      DEPENDS "${_lizard_dp4a_hlsl}"
      COMMENT "DXC main_q4k_q8_1_batched_nolds -> DXIL header")
    add_custom_command(
      OUTPUT "${_lizard_dxil_dir}/dp4a_q6k_batched.h"
      COMMAND "${LIZARD_DXC}" -T cs_6_4 -E main_q6k_q8_1_batched
        -Fh "${_lizard_dxil_dir}/dp4a_q6k_batched.h" -Vn g_dxil_q6k_q8_1_batched
        "${_lizard_dp4a_hlsl}"
      DEPENDS "${_lizard_dp4a_hlsl}"
      COMMENT "DXC main_q6k_q8_1_batched -> DXIL header")
    add_custom_target(lizard-dxil-headers DEPENDS
      "${_lizard_dxil_dir}/dp4a_quantize_q8_1.h"
      "${_lizard_dxil_dir}/dp4a_q4k_matvec.h"
      "${_lizard_dxil_dir}/dp4a_q4k_gemv_wave.h"
      "${_lizard_dxil_dir}/dp4a_q4k_batched.h"
      "${_lizard_dxil_dir}/dp4a_q4k_batched_nolds.h"
      "${_lizard_dxil_dir}/dp4a_q6k_batched.h")
    add_dependencies(lizard-native-runner lizard-dxil-headers)
    target_include_directories(lizard-native-runner PRIVATE "${_lizard_dxil_dir}")
    target_compile_definitions(lizard-native-runner PRIVATE LIZARD_HAVE_DXIL=1)
  else()
    message(STATUS "DP4A kernels disabled: dxc.exe not found (set LIZARD_DXC to enable)")
  endif()
endif()

include(CTest)
if(BUILD_TESTING)
  if(LIZARD_ENABLE_GGML_CPU_BACKEND)
    add_test(
      NAME lizard-pinned-ggml-cpu-selftest
      COMMAND $<TARGET_FILE:lizard-native-runner> --ggml-cpu-selftest)
    add_test(
      NAME lizard-pinned-ggml-grouped-graph-selftest
      COMMAND "${CMAKE_COMMAND}" -E env LIZARD_GGML_GROUPED_GRAPHS=1
        $<TARGET_FILE:lizard-native-runner> --ggml-cpu-selftest)
    add_test(
      NAME lizard-pinned-ggml-phase0b-rollback-selftest
      COMMAND "${CMAKE_COMMAND}" -E env LIZARD_GGML_GROUPED_GRAPHS=0
        $<TARGET_FILE:lizard-native-runner> --ggml-cpu-selftest)
    add_test(
      NAME lizard-pinned-ggml-fused-ffn-selftest
      COMMAND "${CMAKE_COMMAND}" -E env LIZARD_GGML_FUSED_FFN=1
        $<TARGET_FILE:lizard-native-runner> --ggml-cpu-selftest)
  endif()
  add_test(
    NAME lizard-native-cpu-rollback-selftest
    COMMAND "${CMAKE_COMMAND}" -E env LIZARD_NATIVE_CPU_BACKEND=native
      $<TARGET_FILE:lizard-native-runner> --cpu-selftest)
endif()
