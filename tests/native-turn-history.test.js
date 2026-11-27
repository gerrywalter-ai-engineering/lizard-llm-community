import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import {
  appendNativeTurnMetric,
  contextBucket,
  listNativeTurnMetrics,
  summarizeNativeTurnMetrics,
} from "../src/lib/native-turn-history.js";

test("native turn history keeps real per-user rates with path and concurrency", () => {
  const runtimeDirectory = fs.mkdtempSync(path.join(os.tmpdir(), "lizard-turn-history-"));
  const config = { runtimeDirectory };
  try {
    assert.equal(appendNativeTurnMetric(config, { tokensPerSecond: 0 }), null);
    appendNativeTurnMetric(config, {
      userId: "alice",
      modelId: "llama-3.2-3b-instruct",
      quant: "Q4_K_M",
      runtimeProvider: "lizard-native",
      backend: "directx-hybrid",
      servingPath: "batched",
      batched: true,
      concurrencyLevel: 8,
      promptTokens: 1200,
      decodeTokensPerSecond: 12,
      userVisibleTokensPerSecond: 10,
      endToEndMs: 2000,
      ttftMs: 500,
      latencyPerTokenMs: 100,
      transport: "http",
      tokensPerSecond: 10,
      evidenceMetrics: {
        normalizedCpuUtilizationPercent: 60,
        peakWorkingSetBytes: 2_500_000_000,
        peakPrivateBytes: 500_000_000,
        tokensPerSecondPerFullCpu: 20,
        measurementScope: "decode",
        sampleSource: "native-worker-process",
      },
    });
    appendNativeTurnMetric(config, {
      userId: "alice",
      modelId: "llama-3.2-3b-instruct",
      quant: "Q4_K_M",
      runtimeProvider: "caterpillar",
      optimizationProfile: "cpil_ttft",
      backend: "cpu",
      concurrencyLevel: 1,
      promptTokens: 100,
      tokensPerSecond: 6,
    });
    const cpil = listNativeTurnMetrics(config, {
      runtimeProvider: "caterpillar",
      optimizationProfile: "cpil_ttft",
    });
    assert.equal(cpil.length, 1);
    assert.equal(cpil[0].optimizationProfile, "cpil_ttft");
    appendNativeTurnMetric(config, {
      userId: "bob",
      tokensPerSecond: 100,
    });

    const alice = listNativeTurnMetrics(config, { userId: "alice" });
    assert.equal(alice.length, 2);
    assert.equal(alice[0].servingPath, "single-shot");
    assert.equal(alice[1].concurrencyLevel, 8);
    assert.equal(alice[1].contextBucket, "ctx-2048");
    assert.deepEqual(summarizeNativeTurnMetrics(alice), {
      sampleCount: 2,
      latestTokensPerSecond: 6,
      medianTokensPerSecond: 8,
      bestTokensPerSecond: 10,
      latestDecodeTokensPerSecond: 6,
      medianDecodeTokensPerSecond: 9,
      bestDecodeTokensPerSecond: 12,
      latestEndToEndMs: null,
      medianEndToEndMs: 2000,
      latestTtftMs: null,
      medianTtftMs: 500,
      latestLatencyPerTokenMs: null,
      medianLatencyPerTokenMs: 100,
      latestNormalizedCpuUtilizationPercent: null,
      medianNormalizedCpuUtilizationPercent: 60,
      latestWorkerWorkingSetBytes: null,
      medianWorkerWorkingSetBytes: 2_500_000_000,
      latestWorkerPrivateBytes: null,
      medianWorkerPrivateBytes: 500_000_000,
      latestTokensPerSecondPerFullCpu: null,
      medianTokensPerSecondPerFullCpu: 20,
      latestTransport: "direct",
      latestConcurrencyLevel: 1,
      latestServingPath: "single-shot",
      latestRuntimeProvider: "caterpillar",
      latestOptimizationProfile: "cpil_ttft",
      latestModelId: "llama-3.2-3b-instruct",
    });
    const exact = listNativeTurnMetrics(config, {
      userId: "alice",
      runtimeProvider: "lizard-native",
      modelId: "llama-3.2-3b-instruct",
      quant: "Q4_K_M",
      context: "ctx-2048",
    });
    assert.equal(exact.length, 1);
    assert.equal(exact[0].userVisibleTokensPerSecond, 10);
    assert.equal(exact[0].transport, "http");
  } finally {
    fs.rmSync(runtimeDirectory, { recursive: true, force: true });
  }
});

test("context buckets stay comparable across prompt sizes", () => {
  assert.equal(contextBucket(1), "ctx-1024");
  assert.equal(contextBucket(1025), "ctx-2048");
  assert.equal(contextBucket(9000), "ctx-16384");
  assert.equal(contextBucket(40000), "ctx-32768+");
});
