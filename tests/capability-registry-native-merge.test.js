import assert from "node:assert/strict";
import test from "node:test";

import {
  buildBackendCapabilityRegistry,
  buildValidatedCapabilityRegistry,
} from "../src/lib/backend-capability-registry.js";
import { HARDWARE_VALIDATION_STATES } from "../src/lib/hardware-contracts.js";
import { normalizeHardwareProbeReport } from "../src/lib/native-hardware-probe.js";

// Hybrid-laptop scenario from the plan document (Finding 1): the UI scan
// lists the NVIDIA card first (nvidia-smi order), the native probe finds the
// Intel iGPU first (DXGI order). Both adapters validate.
const HYBRID_HARDWARE = {
  platform: "win32",
  arch: "x64",
  cpu: { model: "Intel(R) Core(TM) i7-6700HQ", physicalCores: 4, logicalThreads: 8 },
  memory: { totalGB: 16, availableGB: 12, plannerBudgetGB: 12 },
  gpus: [
    { vendor: "nvidia", name: "NVIDIA GeForce GTX 950M", vramGB: 2, driverVersion: "552.22", backends: { cuda: true } },
    { vendor: "intel", name: "Intel(R) HD Graphics 530", vramGB: 0.125, backends: { vulkan: true } },
  ],
  backends: { cuda: true, metal: false, vulkan: true, rocm: false, cpu: true },
};

const HYBRID_PROBE = normalizeHardwareProbeReport({
  schemaVersion: 1,
  platform: "windows",
  cpu: {
    architecture: "x86-64",
    brand: "Intel(R) Core(TM) i7-6700HQ",
    logicalProcessors: 8,
    physicalCores: 4,
    features: {
      sse2: { cpu: true, os: true },
      avx: { cpu: true, os: true },
      fma: { cpu: true, os: true },
      avx2: { cpu: true, os: true },
      avx512f: { cpu: false, os: false },
    },
  },
  memory: { totalBytes: 17179869184, availableBytes: 9663676416 },
  adapters: [
    {
      luidHigh: 0, luidLow: 111, vendorId: 0x8086, deviceId: 0x191b,
      name: "Intel(R) HD Graphics 530",
      dedicatedVideoMemoryBytes: 134217728, sharedSystemMemoryBytes: 8589934592,
      isSoftware: false,
      videoMemory: { local: { budgetBytes: 6871947673, currentUsageBytes: 100, availableForReservationBytes: 0 }, nonLocal: null },
      d3d12: { deviceCreated: true, uma: true, cacheCoherentUma: true, selfTest: "passed", state: "usable", highestShaderModel: "6.0" },
    },
    {
      luidHigh: 0, luidLow: 222, vendorId: 0x10de, deviceId: 0x139a,
      name: "NVIDIA GeForce GTX 950M",
      dedicatedVideoMemoryBytes: 2147483648, sharedSystemMemoryBytes: 8589934592,
      isSoftware: false,
      videoMemory: { local: { budgetBytes: 1932735283, currentUsageBytes: 50, availableForReservationBytes: 0 }, nonLocal: { budgetBytes: 6871947673, currentUsageBytes: 0, availableForReservationBytes: 0 } },
      d3d12: { deviceCreated: true, uma: false, cacheCoherentUma: false, selfTest: "passed", state: "usable", highestShaderModel: "6.4" },
    },
  ],
});

test("without a native probe the registry behaves exactly as before", () => {
  const registry = buildBackendCapabilityRegistry({ hardware: HYBRID_HARDWARE, config: {} });
  assert.equal(registry.gpu[0].name, "NVIDIA GeForce GTX 950M");
  assert.equal(registry.gpu[0].identity, undefined, "legacy entries gain no identity fields");
  assert.equal(registry.selectedAdapter, null);
  assert.equal(registry.hardwareValidation.state, HARDWARE_VALIDATION_STATES.DETECTING);
  assert.equal(registry.cpu.featureSource, "name-inference");
});

test("the merged registry selects one explicit adapter with a stable LUID", () => {
  const registry = buildBackendCapabilityRegistry({
    hardware: HYBRID_HARDWARE,
    config: {},
    nativeProbe: HYBRID_PROBE,
  });
  // Deterministic policy: dedicated-VRAM hardware beats the UMA iGPU.
  assert.equal(registry.selectedAdapter.identity.luidLow, 222);
  assert.equal(registry.selectedAdapter.vendor, "nvidia");
  assert.equal(registry.selectedAdapter.isUma, false);
  assert.equal(registry.hardwareValidation.state, HARDWARE_VALIDATION_STATES.READY);
  // gpu[0] mirrors the explicit selection so legacy consumers agree with it.
  assert.equal(registry.gpu[0].identity.luidLow, 222);
  assert.equal(registry.gpu[0].validated, true);
  // The nvidia-smi driver version survived the merge into the DXGI record.
  const inventoryNvidia = registry.adapterInventory.find((r) => r.vendor === "nvidia");
  assert.equal(inventoryNvidia.driver.version, "552.22");
  // Authoritative UMA facts from the native architecture query.
  const inventoryIntel = registry.adapterInventory.find((r) => r.vendor === "intel");
  assert.equal(inventoryIntel.memory.isUma, true);
});

test("probe CPU facts replace name-inferred features and pick the runner tier", () => {
  const registry = buildBackendCapabilityRegistry({
    hardware: HYBRID_HARDWARE,
    config: {},
    nativeProbe: HYBRID_PROBE,
  });
  assert.equal(registry.cpu.featureSource, "cpuid");
  assert.equal(registry.cpu.runnerTier, "avx2");
  assert.ok(registry.cpu.features.includes("avx2"));
  assert.ok(!registry.cpu.features.includes("avx512f"), "CPU-incapable feature is not listed");
  assert.equal(registry.cpu.cores, 4);
  assert.equal(registry.cpu.threads, 8);
});

test("no selectable adapter resolves to cpu-fallback-ready, not an error", () => {
  const probe = normalizeHardwareProbeReport({
    schemaVersion: 1,
    platform: "windows",
    cpu: { features: { sse2: { cpu: true, os: true } } },
    memory: null,
    adapters: [
      {
        luidHigh: 0, luidLow: 1, vendorId: 0x1414, deviceId: 0x8c,
        name: "Microsoft Basic Render Driver", isSoftware: true,
        d3d12: { deviceCreated: true, selfTest: "passed", state: "usable" },
      },
      {
        luidHigh: 0, luidLow: 2, vendorId: 0x10de, deviceId: 0x139a,
        name: "NVIDIA GeForce GTX 950M", isSoftware: false,
        d3d12: { deviceCreated: false, state: "device-unsupported", detail: "driver too old" },
      },
    ],
  });
  const registry = buildBackendCapabilityRegistry({ hardware: HYBRID_HARDWARE, config: {}, nativeProbe: probe });
  assert.equal(registry.selectedAdapter, null);
  assert.equal(registry.hardwareValidation.state, HARDWARE_VALIDATION_STATES.CPU_FALLBACK_READY);
  // The failed dGPU remains visible in the inventory with its reason.
  const nvidia = registry.adapterInventory.find((r) => r.vendor === "nvidia" && r.identity.luidLow === 2);
  assert.equal(nvidia.validated, false);
  assert.match(nvidia.probes.d3d12.detail, /driver too old/);
});

test("buildValidatedCapabilityRegistry accepts an injected probe and stays pure", () => {
  const registry = buildValidatedCapabilityRegistry({
    hardware: HYBRID_HARDWARE,
    config: {},
    probe: HYBRID_PROBE,
  });
  assert.equal(registry.selectedAdapter.identity.luidLow, 222);
  const again = buildValidatedCapabilityRegistry({
    hardware: HYBRID_HARDWARE,
    config: {},
    probe: HYBRID_PROBE,
  });
  assert.deepEqual(registry, again, "deterministic for identical inputs");
});
