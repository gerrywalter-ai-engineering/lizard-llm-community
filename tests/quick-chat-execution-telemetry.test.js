import test from "node:test";
import assert from "node:assert/strict";

import { buildQuickChatExecutionTelemetry } from "../src/lib/native-execution-telemetry.js";
import { calculateBatchedTokensPerSecond } from "../src/runtime/core/native-cpu-runtime.js";

test("batched completions report generated-token throughput", () => {
  assert.equal(calculateBatchedTokensPerSecond(16, 2000), 8);
  assert.equal(calculateBatchedTokensPerSecond(0, 2000), 0);
  assert.equal(calculateBatchedTokensPerSecond(16, 0), 0);
});

test("quick chat telemetry exposes adapter, placement, phase backends, and memory pressure", () => {
  const telemetry = buildQuickChatExecutionTelemetry({
    nativeBackend: "directx-hybrid",
    nativeExecution: {
      cpuBuildProfile: "avx2-x86-64",
      batched: true,
      servingPath: "batched",
      concurrencyLevel: 8,
      executionPlan: { mode: "hybrid-low-vram", selectedPlacement: "hybrid-bounded" },
      hardwareTuning: {
        recommendedDecodeBackend: "cpu",
        reason: "GPU calibration was slower.",
        speedup: 0.7,
        source: "bounded-native-tensor-calibration",
      },
      sessionMetrics: {
        gpuAdapter: {
          name: "NVIDIA GeForce GTX 950M",
          luidHigh: 0,
          luidLow: 42,
          vendorId: 0x10de,
          deviceId: 0x139a,
          uma: false,
          selectionReason: "requested DXGI LUID",
        },
        gpuMemory: {
          confidence: "live-dxgi",
          localBudgetBytes: 2_000_000_000,
          localUsageBytes: 500_000_000,
          reserveBytes: 800_000_000,
          requestedWeightBudgetBytes: 900_000_000,
          effectiveWeightBudgetBytes: 700_000_000,
          residentWeightBytes: 650_000_000,
        },
        prefillBackend: "directx-hybrid",
        decodeBackend: "cpu",
        decodeBackendDetail: "GPU decode rejected to avoid weight-cache thrashing.",
        gpuDecodePolicy: "allowed",
        gpuWeightCacheHits: 9,
        gpuWeightCacheMisses: 12,
        gpuWeightCacheEvictions: 3,
        gpuWeightUploadBytes: 600_000_000,
        gpuInputUploadBytes: 20_000,
        gpuOutputReadbackBytes: 30_000,
        gpuSubmissionCount: 15,
        evidenceMetrics: {
          measurementScope: "decode",
          sampleSource: "native-worker-process",
          tokensPerSecond: 9.5,
          normalizedCpuUtilizationPercent: 57.5,
          peakWorkingSetBytes: 2_500_000_000,
          peakPrivateBytes: 504_000_000,
          tokensPerSecondPerFullCpu: 16.52,
        },
      },
    },
  });

  assert.equal(telemetry.adapter.name, "NVIDIA GeForce GTX 950M");
  assert.equal(telemetry.cpuBuildProfile, "avx2-x86-64");
  assert.equal(telemetry.servingPath, "batched");
  assert.equal(telemetry.batched, true);
  assert.equal(telemetry.concurrencyLevel, 8);
  assert.equal(telemetry.adapter.luidLow, 42);
  assert.equal(telemetry.plannedPlacement, "hybrid-bounded");
  assert.equal(telemetry.prefillBackend, "directx-hybrid");
  assert.equal(telemetry.decodeBackend, "cpu");
  assert.equal(telemetry.gpuMemory.effectiveWeightBudgetBytes, 700_000_000);
  assert.equal(telemetry.gpuWeightCacheEvictions, 3);
  assert.match(telemetry.decodeDetail, /thrashing/);
  assert.equal(telemetry.hardwareTuning.recommendedDecodeBackend, "cpu");
  assert.equal(telemetry.hardwareTuning.speedup, 0.7);
  assert.equal(telemetry.evidenceMetrics.normalizedCpuUtilizationPercent, 57.5);
  assert.equal(telemetry.evidenceMetrics.tokensPerSecondPerFullCpu, 16.52);
});

test("quick chat telemetry remains additive and CPU-safe when native GPU metrics are absent", () => {
  const telemetry = buildQuickChatExecutionTelemetry({
    nativeBackend: "cpu",
    fallbackReason: "No validated GPU adapter.",
  });

  assert.equal(telemetry.adapter, null);
  assert.equal(telemetry.gpuMemory, null);
  assert.equal(telemetry.prefillBackend, "cpu");
  assert.equal(telemetry.decodeBackend, "cpu");
  assert.equal(telemetry.servingPath, "single-shot");
  assert.equal(telemetry.batched, false);
  assert.equal(telemetry.concurrencyLevel, 1);
  assert.equal(telemetry.fallbackReason, "No validated GPU adapter.");
  assert.equal(telemetry.evidenceMetrics, null);
});
