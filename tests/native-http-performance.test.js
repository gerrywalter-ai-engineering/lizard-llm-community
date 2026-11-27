import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import {
  findNativeHttpPerformance,
  recordNativeHttpPerformance,
  recordRepeatedNativeHttpPerformance,
} from "../src/lib/native-http-performance.js";

test("native HTTP performance profiles preserve measured B16/B8 selection", () => {
  const runtimeDirectory = fs.mkdtempSync(path.join(os.tmpdir(), "lizard-http-performance-"));
  const config = { runtimeDirectory };
  try {
    const profile = recordNativeHttpPerformance(config, {
      transport: "http",
      finishedAt: "2026-07-29T00:00:00.000Z",
      settings: { contextTokensPerRequest: 2048 },
      candidate: {
        provider: "lizard-native",
        model: "llama-3.2-3b-instruct",
        quant: "Q4_K_M",
        levels: {
          16: { failed: 0, completed: 16, aggregateTokensPerSecond: 38 },
          8: { failed: 0, completed: 8, aggregateTokensPerSecond: 41 },
          4: { failed: 0, completed: 4, aggregateTokensPerSecond: 80 },
        },
      },
    }, { artifactPath: "measured.json" });

    assert.equal(profile.selectedBatchSize, 8);
    assert.equal(profile.selectionRole, "primary");
    const restored = findNativeHttpPerformance(config, {
      runtimeProvider: "lizard-native",
      modelId: "llama-3.2-3b-instruct",
      quant: "q4_k_m",
      contextTokens: 2048,
    });
    assert.equal(restored.match, "exact");
    assert.equal(restored.levels["16"].aggregateTokensPerSecond, 38);
    assert.equal(restored.artifactPath, "measured.json");
  } finally {
    fs.rmSync(runtimeDirectory, { recursive: true, force: true });
  }
});

test("Caterpillar keeps balanced and cpil_ttft measurements in separate histories", () => {
  const runtimeDirectory = fs.mkdtempSync(path.join(os.tmpdir(), "caterpillar-http-performance-"));
  const config = { runtimeDirectory };
  const artifact = (optimizationProfile, tokensPerSecond) => ({
    transport: "http",
    finishedAt: "2026-08-03T00:00:00.000Z",
    settings: {
      contextTokensPerRequest: 2048,
      caterpillarPerformanceProfile: optimizationProfile,
    },
    candidate: {
      provider: "caterpillar",
      model: "llama-3.2-3b-instruct",
      quant: "Q4_K_M",
      caterpillarPerformanceProfile: optimizationProfile,
      levels: { 1: { failed: 0, completed: 1, aggregateTokensPerSecond: tokensPerSecond } },
    },
  });
  try {
    recordNativeHttpPerformance(config, artifact("balanced", 2.4));
    recordNativeHttpPerformance(config, artifact("cpil_ttft", 3.2));
    recordNativeHttpPerformance(config, artifact("cpil_ttft", 2.1));

    const balanced = findNativeHttpPerformance(config, {
      runtimeProvider: "caterpillar",
      modelId: "llama-3.2-3b-instruct",
      quant: "Q4_K_M",
      contextTokens: 2048,
      optimizationProfile: "balanced",
    });
    const ttft = findNativeHttpPerformance(config, {
      runtimeProvider: "caterpillar",
      modelId: "llama-3.2-3b-instruct",
      quant: "Q4_K_M",
      contextTokens: 2048,
      optimizationProfile: "cpil_ttft",
    });

    assert.equal(balanced.optimizationProfile, "balanced");
    assert.equal(balanced.aggregateTokensPerSecond, 2.4);
    assert.equal(ttft.optimizationProfile, "cpil_ttft");
    assert.equal(ttft.aggregateTokensPerSecond, 3.2);
    assert.equal(ttft.levels["1"].latestAggregateTokensPerSecond, 2.1);
    assert.equal(ttft.levels["1"].bestAggregateTokensPerSecond, 3.2);
    assert.notEqual(balanced.id, ttft.id);
  } finally {
    fs.rmSync(runtimeDirectory, { recursive: true, force: true });
  }
});

test("direct or failed HTTP results never become runtime tuning evidence", () => {
  const runtimeDirectory = fs.mkdtempSync(path.join(os.tmpdir(), "lizard-http-performance-"));
  const config = { runtimeDirectory };
  try {
    assert.equal(recordNativeHttpPerformance(config, {
      transport: "direct",
      candidate: { provider: "lizard-native", model: "model", levels: {} },
    }), null);
    assert.equal(recordNativeHttpPerformance(config, {
      transport: "http",
      settings: { contextTokensPerRequest: 2048 },
      candidate: {
        provider: "lizard-native",
        model: "model",
        levels: { 16: { failed: 1, aggregateTokensPerSecond: 99 } },
      },
    }), null);
    assert.equal(findNativeHttpPerformance(config, {
      runtimeProvider: "lizard-native",
      modelId: "model",
    }), null);
  } finally {
    fs.rmSync(runtimeDirectory, { recursive: true, force: true });
  }
});

test("stable repeated medians merge B16 and B8 and cannot be replaced by one run", () => {
  const runtimeDirectory = fs.mkdtempSync(path.join(os.tmpdir(), "lizard-http-performance-"));
  const config = { runtimeDirectory };
  const repeated = (batch, median, label) => ({
    schema: "lizard.smoke-bench.repeated-http.v1",
    status: "completed",
    label,
    finishedAt: "2026-07-29T00:00:00.000Z",
    runs: Array.from({ length: 10 }, () => ({
      report: {
        transport: "http",
        settings: { contextTokensPerRequest: 2048 },
        candidate: {
          provider: "lizard-native",
          model: "llama-3.2-3b-instruct",
          quant: "Q4_K_M",
        },
      },
    })),
    summary: {
      [batch]: {
        repetitions: 10,
        candidate: {
          failedRequests: { max: 0 },
          aggregateTokensPerSecond: {
            mean: median,
            median,
            min: median - 1,
            max: median + 1,
            stddev: 0.5,
            p90: median + 0.8,
            p95: median + 0.9,
            coefficientOfVariationPercent: 2,
            stability: "high",
          },
          streamingProbe: { ttftMs: { median: 250 } },
          resources: {},
        },
      },
    },
  });
  try {
    recordRepeatedNativeHttpPerformance(config, repeated(16, 38, "b16"));
    const merged = recordRepeatedNativeHttpPerformance(config, repeated(8, 41, "b8"));
    assert.equal(merged.selectedBatchSize, 8);
    assert.equal(merged.levels["16"].aggregateTokensPerSecond, 38);
    assert.equal(merged.levels["8"].aggregateTokensPerSecond, 41);
    assert.equal(merged.stable, true);

    recordNativeHttpPerformance(config, {
      transport: "http",
      settings: { contextTokensPerRequest: 2048 },
      candidate: {
        provider: "lizard-native",
        model: "llama-3.2-3b-instruct",
        quant: "Q4_K_M",
        levels: {
          8: { failed: 0, aggregateTokensPerSecond: 99 },
        },
      },
    });
    const restored = findNativeHttpPerformance(config, {
      runtimeProvider: "lizard-native",
      modelId: "llama-3.2-3b-instruct",
      quant: "Q4_K_M",
      contextTokens: 2048,
    });
    assert.equal(restored.selectedBatchSize, 8);
    assert.equal(restored.levels["8"].aggregateTokensPerSecond, 41);
    assert.equal(restored.levels["8"].evidenceType, "repeated");
  } finally {
    fs.rmSync(runtimeDirectory, { recursive: true, force: true });
  }
});
