import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import {
  buildBackendCapabilityRegistry,
  selectBackendForProfile,
} from "../src/lib/backend-capability-registry.js";

function tempConfig() {
  const runtimeDirectory = fs.mkdtempSync(path.join(os.tmpdir(), "lizard-backend-"));
  return {
    runtimeDirectory,
    engine: {
      cliBinary: process.execPath,
      serverBinary: process.execPath,
      preferredBinaryDirectories: [],
    },
  };
}

function hardware({
  platform = "linux",
  arch = "x64",
  ramGb = 16,
  gpu = null,
  backends = { cpu: true, cuda: false, metal: false, vulkan: false, rocm: false },
} = {}) {
  return {
    platform,
    arch,
    cpu: { model: "Test CPU", physicalCores: 8, logicalThreads: 16 },
    memory: { totalGB: ramGb, freeGB: ramGb - 2, availableGB: ramGb - 2, plannerBudgetGB: ramGb - 2 },
    gpus: gpu ? [gpu] : [],
    primaryGpu: gpu,
    backends,
  };
}

test("buildBackendCapabilityRegistry prefers CPU on CPU-only machines", () => {
  const registry = buildBackendCapabilityRegistry({
    hardware: hardware(),
    config: tempConfig(),
  });
  assert.equal(registry.selectedBackend, "cpu");
  assert.equal(registry.backends.cpu.supported, true);
  assert.equal(registry.backends.cuda.supported, false);
});

test("buildBackendCapabilityRegistry prefers CUDA on supported NVIDIA systems", () => {
  const registry = buildBackendCapabilityRegistry({
    hardware: hardware({
      gpu: { vendor: "nvidia", name: "RTX 4070", vramGB: 12, backends: { cuda: true, metal: false, vulkan: false, rocm: false } },
      backends: { cpu: true, cuda: true, metal: false, vulkan: false, rocm: false },
    }),
    config: tempConfig(),
  });
  assert.equal(registry.selectedBackend, "cuda");
  assert.equal(registry.backends.cuda.recommended, true);
});

test("buildBackendCapabilityRegistry prefers Metal on macOS Apple Silicon", () => {
  const registry = buildBackendCapabilityRegistry({
    hardware: hardware({
      platform: "darwin",
      arch: "arm64",
      gpu: { vendor: "apple", name: "Apple GPU", vramGB: null, backends: { cuda: false, metal: true, vulkan: false, rocm: false } },
      backends: { cpu: true, cuda: false, metal: true, vulkan: false, rocm: false },
    }),
    config: tempConfig(),
  });
  assert.equal(registry.selectedBackend, "metal");
  assert.equal(registry.backends.metal.recommended, true);
});

test("selectBackendForProfile falls back when a requested backend is unsupported", () => {
  const registry = buildBackendCapabilityRegistry({
    hardware: hardware(),
    config: tempConfig(),
  });
  const selection = selectBackendForProfile({
    capabilityRegistry: registry,
    requestedBackend: "cuda",
    estimate: { ramRequiredGB: 4 },
  });
  assert.equal(selection.backend, "cpu");
  assert.match(selection.warnings.join(" "), /requested backend cuda/i);
});

test("selectBackendForProfile reports high memory risk on low-RAM systems", () => {
  const registry = buildBackendCapabilityRegistry({
    hardware: hardware({ ramGb: 6 }),
    config: tempConfig(),
  });
  const selection = selectBackendForProfile({
    capabilityRegistry: registry,
    estimate: { ramRequiredGB: 7 },
  });
  assert.equal(selection.memoryRisk, "high");
});

test("buildBackendCapabilityRegistry is deterministic for the same input", () => {
  const inputHardware = hardware({
    gpu: { vendor: "nvidia", name: "RTX 4070", vramGB: 12, backends: { cuda: true, metal: false, vulkan: false, rocm: false } },
    backends: { cpu: true, cuda: true, metal: false, vulkan: false, rocm: false },
  });
  const config = tempConfig();
  const first = buildBackendCapabilityRegistry({ hardware: inputHardware, config });
  const second = buildBackendCapabilityRegistry({ hardware: inputHardware, config });
  assert.deepEqual(first, second);
});
