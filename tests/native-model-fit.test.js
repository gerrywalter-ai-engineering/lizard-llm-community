import assert from "node:assert/strict";
import test from "node:test";

import {
  buildModelFitReport,
  rankPlacementCandidates,
  PLACEMENT_MODES,
} from "../src/runtime/native-model-fit.js";
import { buildNativeExecutionPlan } from "../src/runtime/native-execution-plan.js";

const GIB = 1024 * 1024 * 1024;

// gemma4-e4b Q8_0-like metadata (real catalog shape).
const GEMMA_GGUF = {
  fileSizeBytes: 8031240160,
  blockCount: 35,
  embeddingLength: 2560,
  contextLength: 32768,
  quant: "Q8_0",
};

test("the fit report combines weights, context-dependent KV, workspace, staging, and reserves", () => {
  const fit = buildModelFitReport({ gguf: GEMMA_GGUF, contextTokens: 2048 });
  assert.equal(fit.weights.bytes, 8031240160);
  // f32 KV: layers * hidden * tokens * 4 bytes * 2 (K+V).
  assert.equal(fit.kvCache.bytes, 35 * 2560 * 2048 * 4 * 2);
  assert.equal(fit.kvCache.estimated, false);
  assert.equal(fit.kvCache.contextTokens, 2048);
  assert.ok(fit.totalGpuResidentBytes > fit.weights.bytes + fit.kvCache.bytes);
  assert.ok(fit.staging.bytes > 0 && fit.safetyReserve.bytes > 0);
  assert.equal(fit.confidence, "measured");
});

test("missing KV metadata degrades confidence to estimated instead of failing", () => {
  const fit = buildModelFitReport({
    gguf: { fileSizeBytes: 2 * GIB },
    contextTokens: 4096,
  });
  assert.equal(fit.kvCache.estimated, true);
  assert.ok(fit.kvCache.bytes > 0, "a conservative stand-in is used");
  assert.equal(fit.confidence, "estimated");
});

test("a model larger than the GPU budget rejects full residency with an inspectable reason and selects hybrid", () => {
  const fit = buildModelFitReport({ gguf: GEMMA_GGUF, contextTokens: 2048 });
  const { candidates, selected } = rankPlacementCandidates({
    fit,
    budgets: { gpuBudgetBytes: 1.5 * GIB, cpuBudgetBytes: 12 * GIB },
    gpuPresent: true,
    gpuValidated: true,
    uma: false,
  });
  const [fullGpu, hybrid, cpu] = candidates;
  assert.equal(fullGpu.mode, PLACEMENT_MODES.GPU_FULL_RESIDENT);
  assert.equal(fullGpu.ok, false);
  assert.match(fullGpu.rejectionReason, /exceeds the budget/);
  assert.equal(hybrid.ok, true);
  assert.equal(cpu.ok, true);
  assert.equal(selected, PLACEMENT_MODES.HYBRID_BOUNDED);
});

test("a small model on a large GPU selects full residency", () => {
  const fit = buildModelFitReport({
    gguf: { fileSizeBytes: 2 * GIB, blockCount: 28, embeddingLength: 2048, contextLength: 8192 },
    contextTokens: 2048,
  });
  const { candidates, selected } = rankPlacementCandidates({
    fit,
    budgets: { gpuBudgetBytes: 18 * GIB, cpuBudgetBytes: 24 * GIB },
    gpuPresent: true,
    gpuValidated: true,
    uma: false,
  });
  assert.equal(selected, PLACEMENT_MODES.GPU_FULL_RESIDENT);
  assert.equal(candidates[0].rejectionReason, null);
});

test("no GPU resolves to CPU with reasons recorded on both GPU candidates", () => {
  const fit = buildModelFitReport({ gguf: GEMMA_GGUF });
  const { candidates, selected } = rankPlacementCandidates({
    fit,
    budgets: { cpuBudgetBytes: 12 * GIB },
    gpuPresent: false,
  });
  assert.equal(selected, PLACEMENT_MODES.CPU);
  assert.match(candidates[0].rejectionReason, /no usable GPU adapter/);
  assert.match(candidates[1].rejectionReason, /no usable GPU adapter/);
});

test("CPU stays selectable under memory pressure but carries a paging warning", () => {
  const fit = buildModelFitReport({ gguf: GEMMA_GGUF, contextTokens: 2048 });
  const { candidates, selected } = rankPlacementCandidates({
    fit,
    budgets: { cpuBudgetBytes: 4 * GIB },
    gpuPresent: false,
  });
  assert.equal(selected, PLACEMENT_MODES.CPU);
  const cpu = candidates[2];
  assert.equal(cpu.ok, true);
  assert.match(cpu.warnings[0], /expect paging/);
});

test("the execution plan embeds the fit and candidates when gguf metadata is available", () => {
  const plan = buildNativeExecutionPlan({
    capabilityRegistry: {
      os: "windows",
      memory: { availableGb: 12 },
      gpu: [{ name: "NVIDIA GeForce MX", vendor: "nvidia", vramGb: 2 }],
    },
    profile: { backend: "cpu", contextSize: 2048 },
    gguf: GEMMA_GGUF,
  });
  assert.ok(plan.fit);
  assert.equal(plan.fit.weights.bytes, GEMMA_GGUF.fileSizeBytes);
  assert.equal(plan.placementCandidates.length, 3);
  assert.equal(plan.selectedPlacement, PLACEMENT_MODES.HYBRID_BOUNDED);
  assert.match(plan.placementCandidates[0].rejectionReason, /exceeds the budget/);
  assert.equal(plan.execution.prefillStrategy, "auto-hybrid");
  assert.equal(plan.execution.decodeStrategy, "cpu-no-thrash");
  assert.equal(plan.execution.allowGpuPrefill, true);
  assert.equal(plan.execution.allowGpuDecode, false);
  assert.equal(plan.execution.allowCompleteTokenGraph, false);
  assert.match(plan.execution.decodeReason, /eviction and re-upload/);
});

test("a model whose complete state fits enables authoritative full GPU decode", () => {
  const plan = buildNativeExecutionPlan({
    capabilityRegistry: {
      os: "windows",
      memory: { availableGb: 24 },
      selectedAdapter: {
        identity: { name: "RTX", luidHigh: 0, luidLow: 7 },
        vendor: "nvidia",
        isUma: false,
        dedicatedVideoMemoryBytes: 12 * GIB,
      },
      gpu: [{ name: "RTX", vendor: "nvidia", vramGb: 12 }],
    },
    profile: { backend: "cpu", contextSize: 2048 },
    gguf: { ...GEMMA_GGUF, fileSizeBytes: 2 * GIB },
  });

  assert.equal(plan.selectedPlacement, PLACEMENT_MODES.GPU_FULL_RESIDENT);
  assert.equal(plan.execution.decodeStrategy, "gpu-full-resident");
  assert.equal(plan.execution.allowGpuDecode, true);
  assert.equal(plan.execution.allowCompleteTokenGraph, true);
});

test("legacy plans without model metadata keep fit null", () => {
  const plan = buildNativeExecutionPlan({
    capabilityRegistry: { os: "windows", memory: { availableGb: 12 }, gpu: [] },
    profile: { backend: "cpu" },
  });
  assert.equal(plan.fit, null);
  assert.equal(plan.placementCandidates, null);
  assert.equal(plan.selectedPlacement, null);
});
