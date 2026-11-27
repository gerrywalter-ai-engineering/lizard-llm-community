import test from "node:test";
import assert from "node:assert/strict";
import { estimateConfiguration } from "../src/lib/estimator.js";

test("estimates memory budgets", () => {
  const estimate = estimateConfiguration({
    model: {
      paramsB: 7.6,
      totalLayers: 28,
      hiddenSize: 3584,
    },
    quant: "Q4_K_M",
    contextTokens: 8192,
    gpuLayers: 20,
    kvCache: "q8_0",
    hardware: {
      memory: { totalGB: 32 },
      primaryGpu: { vramGB: 12 },
    },
    config: {
      maxRamUsagePercent: 85,
      maxVramUsagePercent: 90,
    },
  });

  assert.equal(typeof estimate.ramRequiredGB, "number");
  assert.equal(typeof estimate.vramRequiredGB, "number");
  assert.equal(estimate.fitsRam, true);
});

test("does not charge VRAM when gpu offload is disabled", () => {
  const estimate = estimateConfiguration({
    model: {
      paramsB: 7.6,
      totalLayers: 28,
      hiddenSize: 3584,
    },
    quant: "Q4_K_M",
    contextTokens: 4096,
    gpuLayers: 0,
    kvCache: "q8_0",
    hardware: {
      memory: { totalGB: 32 },
      primaryGpu: { vramGB: 2 },
    },
    config: {
      maxRamUsagePercent: 85,
      maxVramUsagePercent: 90,
    },
  });

  assert.equal(estimate.vramRequiredGB, 0);
});

test("estimates ternary, importance, and full-precision GGUF profiles", () => {
  const base = {
    model: {
      paramsB: 3,
      totalLayers: 28,
      hiddenSize: 3072,
    },
    contextTokens: 2048,
    gpuLayers: 0,
    kvCache: "f16",
    hardware: {
      memory: { totalGB: 32 },
      primaryGpu: null,
    },
    config: {
      maxRamUsagePercent: 85,
      maxVramUsagePercent: 90,
    },
  };

  const tq1 = estimateConfiguration({ ...base, quant: "TQ1_0" });
  const iq4 = estimateConfiguration({ ...base, quant: "IQ4_XS" });
  const f16 = estimateConfiguration({ ...base, quant: "F16" });
  const f32 = estimateConfiguration({ ...base, quant: "F32" });

  assert.ok(tq1.weightGB < iq4.weightGB);
  assert.ok(iq4.weightGB < f16.weightGB);
  assert.ok(f16.weightGB < f32.weightGB);
});

test("reports an unsupported quantization instead of dereferencing it", () => {
  assert.throws(() => estimateConfiguration({
    model: { paramsB: 3, totalLayers: 28, hiddenSize: 3072 },
    quant: "NEW_QUANT",
    contextTokens: 2048,
    gpuLayers: 0,
    kvCache: "f16",
    hardware: { memory: { totalGB: 32 }, primaryGpu: null },
    config: { maxRamUsagePercent: 85, maxVramUsagePercent: 90 },
  }), /Unsupported quantization profile: NEW_QUANT/);
});
