import assert from "node:assert/strict";
import test from "node:test";

import { buildNativeExecutionPlan } from "../src/runtime/native-execution-plan.js";
import { resolveNativeWorkerEnvironment } from "../src/runtime/worker/process-client.js";

// Hybrid laptop registry after the iteration 3 merge: the explicit selection
// is the NVIDIA dGPU even though the raw gpu list once led with the iGPU.
function hybridRegistry() {
  return {
    os: "windows",
    memory: { availableGb: 12 },
    gpu: [
      { name: "Intel(R) HD Graphics 530", vendor: "intel", vramGb: 0.125 },
      { name: "NVIDIA GeForce GTX 950M", vendor: "nvidia", vramGb: 2 },
    ],
    selectedAdapter: {
      identity: { luidHigh: 0, luidLow: 222, vendorId: 0x10de, deviceId: 0x139a, name: "NVIDIA GeForce GTX 950M", nvidiaUuid: null },
      vendor: "nvidia",
      isUma: false,
      dedicatedVideoMemoryBytes: 2 * 1024 * 1024 * 1024,
      reason: "preferred selectable adapter",
    },
  };
}

function validatedAdapter({
  luidLow,
  vendor,
  vendorId,
  name,
  dedicatedVideoMemoryBytes,
  isUma,
  isCacheCoherentUma,
}) {
  return {
    identity: { luidHigh: 0, luidLow, vendorId, deviceId: 1, name },
    vendor,
    memory: { dedicatedVideoMemoryBytes, isUma, isCacheCoherentUma },
    validated: true,
    isSoftware: false,
    probes: { d3d12: { state: "usable" } },
  };
}

function probedHybridRegistry() {
  const registry = hybridRegistry();
  registry.adapterInventory = [
    validatedAdapter({
      luidLow: 111,
      vendor: "intel",
      vendorId: 0x8086,
      name: "Intel(R) HD Graphics 530",
      dedicatedVideoMemoryBytes: 128 * 1024 * 1024,
      isUma: true,
      isCacheCoherentUma: true,
    }),
    validatedAdapter({
      luidLow: 222,
      vendor: "nvidia",
      vendorId: 0x10de,
      name: "NVIDIA GeForce GTX 950M",
      dedicatedVideoMemoryBytes: 2 * 1024 * 1024 * 1024,
      isUma: false,
      isCacheCoherentUma: false,
    }),
  ];
  return registry;
}

test("the plan takes device facts from the explicit selected adapter, not gpu[0]", () => {
  const plan = buildNativeExecutionPlan({
    capabilityRegistry: hybridRegistry(),
    profile: { backend: "cpu", nativeRuntime: { enableHybridGpuPrefill: true } },
  });
  assert.equal(plan.hardware.gpuName, "NVIDIA GeForce GTX 950M");
  assert.equal(plan.hardware.gpuVendor, "nvidia");
  assert.equal(plan.hardware.gpuVramGb, 2);
  // Authoritative architecture fact: NOT UMA, even though gpu[0] is an iGPU
  // the name heuristic would call UMA.
  assert.equal(plan.hardware.unifiedMemory, false);
  assert.deepEqual(plan.hardware.adapterIdentity, {
    luidHigh: 0,
    luidLow: 222,
    vendorId: 0x10de,
    deviceId: 0x139a,
    name: "NVIDIA GeForce GTX 950M",
    nvidiaUuid: null,
  });
  assert.equal(plan.mode, "hybrid-low-vram", "2 GB dedicated stays on the bounded hybrid plan");
});

test("without a selected adapter the plan keeps the legacy gpu[0] path and no identity", () => {
  const registry = hybridRegistry();
  delete registry.selectedAdapter;
  const plan = buildNativeExecutionPlan({
    capabilityRegistry: registry,
    profile: { backend: "cpu" },
  });
  assert.equal(plan.hardware.gpuName, "Intel(R) HD Graphics 530");
  assert.equal(plan.hardware.unifiedMemory, true, "legacy name/vendor UMA inference");
  assert.equal(plan.hardware.adapterIdentity, null);
});

test("the worker environment pins the planned adapter LUID", () => {
  const env = resolveNativeWorkerEnvironment({
    profile: { backend: "cpu", capabilityRegistry: hybridRegistry() },
    modelArtifact: { modelId: "m", modelPath: "E:\\demo.gguf" },
    env: {},
  });
  assert.equal(env.LIZARD_NATIVE_GPU_ADAPTER_LUID, "0:222");
});

test("Caterpillar pins a validated coherent-UMA adapter instead of the preferred dGPU", () => {
  const registry = probedHybridRegistry();
  const plan = buildNativeExecutionPlan({
    capabilityRegistry: registry,
    profile: { backend: "cpu" },
    runtimeProvider: "caterpillar",
  });
  assert.equal(plan.hardware.gpuName, "Intel(R) HD Graphics 530");
  assert.equal(plan.hardware.unifiedMemory, true);
  assert.equal(plan.hardware.adapterIdentity.luidLow, 111);

  const env = resolveNativeWorkerEnvironment({
    profile: { backend: "cpu", capabilityRegistry: registry },
    modelArtifact: { modelId: "m", modelPath: "E:\\demo.gguf" },
    runtimeProvider: "caterpillar",
    env: {},
  });
  assert.equal(env.LIZARD_NATIVE_GPU_ADAPTER_LUID, "0:111");
});

test("Caterpillar adapter selection is capability-based and accepts a validated AMD APU", () => {
  const registry = probedHybridRegistry();
  registry.adapterInventory[0] = validatedAdapter({
    luidLow: 333,
    vendor: "amd",
    vendorId: 0x1002,
    name: "AMD Radeon Integrated Graphics",
    dedicatedVideoMemoryBytes: 512 * 1024 * 1024,
    isUma: true,
    isCacheCoherentUma: true,
  });
  const plan = buildNativeExecutionPlan({
    capabilityRegistry: registry,
    profile: { backend: "cpu" },
    runtimeProvider: "caterpillar",
  });
  assert.equal(plan.hardware.gpuVendor, "amd");
  assert.equal(plan.hardware.adapterIdentity.luidLow, 333);
});

test("Caterpillar stays CPU-only when the native inventory has no compatible adapter", () => {
  const registry = probedHybridRegistry();
  registry.adapterInventory = registry.adapterInventory.filter((record) => !record.memory.isUma);
  const plan = buildNativeExecutionPlan({
    capabilityRegistry: registry,
    profile: { backend: "cpu" },
    runtimeProvider: "caterpillar",
  });
  assert.equal(plan.mode, "cpu-only");
  assert.equal(plan.hardware.adapterIdentity, null);

  const env = resolveNativeWorkerEnvironment({
    profile: { backend: "cpu", capabilityRegistry: registry },
    modelArtifact: { modelId: "m", modelPath: "E:\\demo.gguf" },
    runtimeProvider: "caterpillar",
    env: {},
  });
  assert.equal(env.LIZARD_NATIVE_GPU_ADAPTER_LUID, undefined);
  assert.equal(env.LIZARD_NATIVE_GPU_WEIGHT_BUDGET_MB, undefined);
});

test("an explicit adapter LUID in the environment is never overridden", () => {
  const env = resolveNativeWorkerEnvironment({
    profile: { backend: "cpu", capabilityRegistry: hybridRegistry() },
    modelArtifact: { modelId: "m", modelPath: "E:\\demo.gguf" },
    env: { LIZARD_NATIVE_GPU_ADAPTER_LUID: "0:111" },
  });
  assert.equal(env.LIZARD_NATIVE_GPU_ADAPTER_LUID, undefined);
});

test("cpu-only plans and prototype workers pin no adapter", () => {
  const cpuOnly = resolveNativeWorkerEnvironment({
    profile: {
      backend: "cpu",
      capabilityRegistry: hybridRegistry(),
      nativeRuntime: { forceCpuOnlyPlan: true },
    },
    modelArtifact: { modelId: "m", modelPath: "E:\\demo.gguf" },
    env: {},
  });
  assert.equal(cpuOnly.LIZARD_NATIVE_GPU_ADAPTER_LUID, undefined);

  const prototype = resolveNativeWorkerEnvironment({
    profile: { backend: "cpu", capabilityRegistry: hybridRegistry() },
    modelArtifact: {
      modelId: "tiny",
      modelPath: "E:\\demo.json",
      runtimePrototype: { format: "lizard-tiny-linear" },
    },
    env: {},
  });
  assert.deepEqual(prototype, {});
});
