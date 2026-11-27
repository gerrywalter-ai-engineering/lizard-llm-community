import test from "node:test";
import assert from "node:assert/strict";
import {
  CATERPILLAR_BALANCED_PROFILE,
  CATERPILLAR_TTFT_PROFILE,
  applyCaterpillarPerformanceProfile,
  caterpillarPerformanceProfileFromOptions,
  caterpillarProfileCliArgs,
  isCaterpillarTtftProfile,
} from "../src/runtime/caterpillar-performance-profile.js";

test("cpil_ttft aliases resolve to the one canonical Caterpillar profile", () => {
  for (const options of [
    { cpil_ttft: true },
    { "cpil-ttft": true },
    { cpli_ttft: true },
    { caterpillarProfile: "b1" },
  ]) {
    assert.equal(caterpillarPerformanceProfileFromOptions(options), CATERPILLAR_TTFT_PROFILE);
  }
  assert.equal(caterpillarPerformanceProfileFromOptions({}), CATERPILLAR_BALANCED_PROFILE);
  assert.deepEqual(caterpillarProfileCliArgs(CATERPILLAR_TTFT_PROFILE), ["--cpil_ttft"]);
});

test("cpil_ttft applies a correctness-preserving B1 scheduling preference only to Caterpillar", () => {
  const base = {
    nativeRuntime: {
      batchedDecode: true,
      alwaysBatch: true,
      forceCpuOnlyPlan: true,
    },
  };
  const selected = applyCaterpillarPerformanceProfile(base, {
    runtimeProvider: "caterpillar",
    performanceProfile: CATERPILLAR_TTFT_PROFILE,
  });
  assert.equal(isCaterpillarTtftProfile(selected), true);
  assert.equal(selected.nativeRuntime.batchedDecode, false);
  assert.equal(selected.nativeRuntime.alwaysBatch, false);
  assert.equal(selected.nativeRuntime.batchTargetSize, 1);
  assert.equal(selected.nativeRuntime.forceCpuOnlyPlan, false);
  assert.equal(selected.nativeRuntime.gpuDecodePolicy, "allowed");
  assert.equal(selected.nativeRuntime.integerDot, undefined);

  const lizard = applyCaterpillarPerformanceProfile(base, {
    runtimeProvider: "lizard-native",
    performanceProfile: CATERPILLAR_TTFT_PROFILE,
  });
  assert.equal(lizard, base);
});
