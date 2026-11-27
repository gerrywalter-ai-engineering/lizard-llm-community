import assert from "node:assert/strict";
import fs from "node:fs";
import test from "node:test";
import {
  LIZARD_GGML_COMMIT,
  nativeCpuCmakeOption,
  nativeCpuVariantCmakeOption,
  parseArgs,
} from "../scripts/build-native-runner.mjs";

test("native runner builds use the portable CPU profile by default", () => {
  assert.equal(parseArgs([]).optimizeForHost, false);
  assert.deepEqual(parseArgs([]).variants, ["baseline", "avx2", "avx512"]);
  assert.equal(nativeCpuCmakeOption(false), "-DLIZARD_NATIVE_OPTIMIZE_FOR_HOST=OFF");
  assert.equal(nativeCpuVariantCmakeOption("avx2"), "-DLIZARD_NATIVE_CPU_VARIANT=avx2");
  assert.equal(nativeCpuVariantCmakeOption("avx512"), "-DLIZARD_NATIVE_CPU_VARIANT=avx512");
});

test("host CPU optimization requires an explicit development-only flag", () => {
  assert.equal(parseArgs(["--optimize-for-host"]).optimizeForHost, true);
  assert.deepEqual(parseArgs(["--optimize-for-host"]).variants, ["host"]);
  assert.equal(nativeCpuCmakeOption(true), "-DLIZARD_NATIVE_OPTIMIZE_FOR_HOST=ON");
});

test("CMake also defaults host-specific CPU optimization off", () => {
  const cmake = fs.readFileSync("native/lizard-native-runner/CMakeLists.txt", "utf8");
  assert.match(
    cmake,
    /option\(LIZARD_NATIVE_OPTIMIZE_FOR_HOST[^\n]*OFF\)/,
    "a direct CMake release build must be portable unless explicitly overridden",
  );
  assert.match(cmake, /-march=x86-64/);
  assert.match(cmake, /-mtune=generic/);
  assert.match(cmake, /-mavx2/);
  assert.match(cmake, /-mavx512f/);
  assert.match(cmake, /-mavx512bw/);
  assert.match(cmake, /-mavx512vl/);
  assert.doesNotMatch(cmake, /x86-64-v3/);
});

test("optimized CPU startup self-test covers qualified Q_K x Q8_K kernels", () => {
  const source = fs.readFileSync("native/lizard-native-runner/src/main.cpp", "utf8");
  assert.match(source, /static bool q6_k_q8_k_kernel_selftest\(\)/);
  assert.match(source, /if \(!q6_k_q8_k_kernel_selftest\(\)\)/);
  assert.match(source, /LIZARD_NATIVE_Q6_Q8_FASTPATH/);
  assert.match(source, /static bool q4_k_q8_k_kernel_selftest\(\)/);
  assert.match(source, /if \(!q4_k_q8_k_kernel_selftest\(\)\)/);
  assert.match(source, /LIZARD_NATIVE_Q4_Q8_FASTPATH/);
});

test("the default ggml CPU backend is source-pinned, verified, attributed, and reversible", () => {
  assert.equal(LIZARD_GGML_COMMIT, "ddd4ec1428a6201e18975ea52b07c71e0f9aef26");
  const cmake = fs.readFileSync("native/lizard-native-runner/CMakeLists.txt", "utf8");
  const build = fs.readFileSync("scripts/build-native-runner.mjs", "utf8");
  const adapter = fs.readFileSync("native/lizard-native-runner/src/ggml_cpu_backend.cpp", "utf8");
  const notices = fs.readFileSync("THIRD-PARTY-NOTICES.txt", "utf8");

  assert.match(cmake, /option\(LIZARD_ENABLE_GGML_CPU_BACKEND[^\n]*ON\)/);
  assert.match(cmake, new RegExp(LIZARD_GGML_COMMIT));
  assert.match(cmake, /rev-parse HEAD/);
  assert.match(build, /fetch", "--depth", "1", "origin", LIZARD_GGML_COMMIT/);
  assert.match(build, /Pinned ggml checkout verification failed/);
  assert.match(adapter, /LIZARD_NATIVE_CPU_BACKEND/);
  assert.match(adapter, /return std::string\("ggml"\)/, "compiled builds default to the pinned backend");
  assert.match(adapter, /"ggml-cpu-pinned" : "lizard-native-cpu"/, "the native rollback remains explicit");
  assert.match(notices, /ggml-cpu \(llama\.cpp\).*ddd4ec1428a6201e18975ea52b07c71e0f9aef26/);
  assert.match(notices, /Copyright \(c\) 2023-2026 The ggml authors/);
});
