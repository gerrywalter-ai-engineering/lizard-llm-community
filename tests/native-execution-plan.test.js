import test from "node:test";
import assert from "node:assert/strict";
import { buildNativeExecutionPlan, resolveStreamingDiscreteDecode } from "../src/runtime/native-execution-plan.js";

// A ~2 GB Q4_K_M model (bigger than a 2 GB card's resident decode budget).
const OVERSIZED_GGUF = {
  architecture: "llama",
  quant: "Q4_K_M",
  fileSizeBytes: 2_020_000_000,
  blockCount: 28,
  embeddingLength: 3072,
  contextLength: 8192,
};

function capabilityRegistry({
  availableGb = 12,
  os = "windows",
  gpu = null,
} = {}) {
  return {
    os,
    memory: { availableGb },
    gpu: gpu ? [gpu] : [],
  };
}

test("native execution plan stays cpu-only when no GPU is present", () => {
  const plan = buildNativeExecutionPlan({
    capabilityRegistry: capabilityRegistry(),
    profile: {
      backend: "cpu",
    },
    gguf: {
      architecture: "gemma4",
      quant: "Q4_K_M",
    },
    requestedBackend: "cpu",
  });

  assert.equal(plan.mode, "cpu-only");
  assert.equal(plan.execution.recommendedGpuLayers, 0);
  assert.match(plan.warnings.join(" "), /no gpu was detected/i);
});

test("native execution plan enables hybrid low-vram mode for Iris Xe on Windows", () => {
  const plan = buildNativeExecutionPlan({
    capabilityRegistry: capabilityRegistry({
      gpu: { name: "Intel Iris Xe Graphics", vendor: "intel", vramGb: 1 },
    }),
    profile: {
      backend: "cpu",
      nativeRuntime: {
        enableHybridGpuPrefill: true,
      },
    },
    gguf: {
      architecture: "gemma4",
      quant: "Q4_K_M",
    },
    requestedBackend: "cpu",
  });

  assert.equal(plan.mode, "hybrid-low-vram");
  assert.equal(plan.hardware.unifiedMemory, true);
  assert.equal(plan.placement.weights, "cpu-mmap-plus-staged-hot-layers");
  assert.equal(plan.execution.prefillChunkTokens, 32);
  assert.equal(plan.execution.minHybridGpuPrefillTokens, 12);
  assert.equal(plan.execution.recommendedGpuLayers, "auto");
  assert.doesNotMatch(plan.warnings.join(" "), /advisory only/i);
});

test("native execution plan treats a 2 GB NVIDIA GPU as low-VRAM hybrid hardware", () => {
  const plan = buildNativeExecutionPlan({
    capabilityRegistry: capabilityRegistry({
      availableGb: 12,
      gpu: { name: "NVIDIA GeForce MX", vendor: "nvidia", vramGb: 2 },
    }),
    profile: {
      backend: "cpu",
      nativeRuntime: { enableHybridGpuPrefill: true },
    },
    gguf: { architecture: "gemma4", quant: "Q4_K_M" },
    requestedBackend: "cpu",
  });

  assert.equal(plan.mode, "hybrid-low-vram");
  assert.equal(plan.hardware.unifiedMemory, false);
  assert.equal(plan.placement.weights, "cpu-mmap-plus-staged-hot-layers");
  assert.equal(plan.execution.microBatchTokens, 8);
  assert.equal(plan.execution.recommendedGpuLayers, 1);
  assert.match(plan.warnings.join(" "), /2 GB or smaller GPU mode must stay hybrid/i);
});

function discreteOversizedPlan(nativeRuntime = {}) {
  return buildNativeExecutionPlan({
    capabilityRegistry: capabilityRegistry({
      availableGb: 12,
      gpu: { name: "NVIDIA GeForce GTX 950M", vendor: "nvidia", vramGb: 2 },
    }),
    profile: { backend: "cpu", nativeRuntime: { enableHybridGpuPrefill: true, ...nativeRuntime } },
    gguf: OVERSIZED_GGUF,
    requestedBackend: "cpu",
  });
}

test("discrete GPU + oversized model defaults to no-thrash CPU decode (unchanged)", () => {
  const plan = discreteOversizedPlan();
  assert.equal(plan.hardware.unifiedMemory, false);
  assert.equal(plan.execution.allowGpuDecode, false);
  assert.equal(plan.execution.decodeStrategy, "cpu-no-thrash");
  assert.equal(plan.execution.streamingResidentDecode, false);
});

test("gpuDecodePolicy=required streams decode over D3D12 for a model larger than a discrete GPU", () => {
  const plan = discreteOversizedPlan({ gpuDecodePolicy: "required" });
  assert.equal(plan.execution.allowGpuDecode, true);
  assert.equal(plan.execution.decodeStrategy, "gpu-streaming-resident");
  assert.equal(plan.execution.streamingResidentDecode, true);
  // Streaming uses the per-layer resident lane, never the all-or-nothing
  // complete-token graph (which stays reserved for full-resident / UMA).
  assert.equal(plan.execution.allowCompleteTokenGraph, false);
  assert.match(plan.execution.decodeReason, /streams weights|resident lane/i);
});

test("allowStreamingGpuDecode opts into streaming decode without requiring GPU", () => {
  const plan = discreteOversizedPlan({ allowStreamingGpuDecode: true });
  assert.equal(plan.execution.decodeStrategy, "gpu-streaming-resident");
  assert.equal(plan.execution.streamingResidentDecode, true);
});

test("Iris/UMA never takes the discrete streaming lane", () => {
  const plan = buildNativeExecutionPlan({
    capabilityRegistry: capabilityRegistry({
      availableGb: 12,
      gpu: { name: "Intel Iris Xe Graphics", vendor: "intel", vramGb: 1 },
    }),
    profile: { backend: "cpu", nativeRuntime: { enableHybridGpuPrefill: true, gpuDecodePolicy: "required" } },
    gguf: OVERSIZED_GGUF,
    requestedBackend: "cpu",
  });
  assert.equal(plan.hardware.unifiedMemory, true);
  assert.equal(plan.execution.streamingResidentDecode, false);
  assert.notEqual(plan.execution.decodeStrategy, "gpu-streaming-resident");
});

test("resolveStreamingDiscreteDecode gates on discrete + oversized + opt-in", () => {
  const base = { allowGpuPrefill: true, oversizedForResidentDecode: true, uma: false, gpuDecodePolicy: "required" };
  assert.equal(resolveStreamingDiscreteDecode(base).stream, true);
  assert.equal(resolveStreamingDiscreteDecode({ ...base, gpuDecodePolicy: "allowed" }).stream, false);
  assert.equal(resolveStreamingDiscreteDecode({ ...base, gpuDecodePolicy: "allowed", allowStreamingGpuDecode: true }).stream, true);
  assert.equal(resolveStreamingDiscreteDecode({ ...base, uma: true }).stream, false);
  assert.equal(resolveStreamingDiscreteDecode({ ...base, oversizedForResidentDecode: false }).stream, false);
  assert.equal(resolveStreamingDiscreteDecode({ ...base, allowGpuPrefill: false }).stream, false);
});
