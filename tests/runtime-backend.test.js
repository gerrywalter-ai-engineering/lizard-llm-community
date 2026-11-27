import test from "node:test";
import assert from "node:assert/strict";
import {
  buildNativeBackendFeatureRegistry,
  buildNativeBackendTestMatrix,
  createCpuBackend,
  resolveNativeBackendRoute,
} from "../src/runtime/backend/index.js";
import { NativeCpuRuntime, generateWithRuntimeSession } from "../src/runtime/index.js";

function prototypeArtifact(modelPath = "E:\\demo-prototype.json", extra = {}) {
  return {
    modelId: "tiny-native",
    modelPath,
    runtimePrototype: {
      format: "lizard-tiny-linear",
      ...extra,
    },
  };
}

function prototypeProfile(overrides = {}) {
  return {
    backend: "cpu",
    contextSize: 64,
    timeoutMs: 1000,
    gguf: {
      blockCount: 1,
      embeddingLength: 2,
      tokenizer: "prototype",
    },
    capabilityRegistry: {
      backends: {
        cpu: { supported: true, reason: "CPU available." },
        cuda: { supported: true, reason: "CUDA hardware detected." },
        metal: { supported: false, reason: "Metal unavailable." },
        vulkan: { supported: false, reason: "Vulkan unavailable." },
      },
      memory: {
        availableGb: 1,
      },
    },
    ...overrides,
  };
}

test("native backend feature registry reports CPU as executable and GPU paths as planned", () => {
  const registry = buildNativeBackendFeatureRegistry({
    capabilityRegistry: prototypeProfile().capabilityRegistry,
    requestedBackend: "cuda",
  });

  assert.equal(registry.selectedBackend, "cuda");
  assert.equal(registry.backends.cpu.supported, true);
  assert.equal(registry.backends.cuda.supported, false);
  assert.equal(registry.backends.cuda.planned, true);
  assert.equal(registry.backends.cuda.detected, true);
  assert.equal(registry.fallbackPolicy.explicitOnly, true);
});

test("native backend test matrix declares active CPU coverage and planned GPU coverage", () => {
  const matrix = buildNativeBackendTestMatrix();
  assert.equal(matrix.find((entry) => entry.backend === "cpu")?.status, "active");
  assert.equal(matrix.find((entry) => entry.backend === "cuda")?.status, "planned");
  assert.ok(matrix.find((entry) => entry.backend === "metal")?.requiredCoverage.includes("fallback safety"));
});

test("resolveNativeBackendRoute requires explicit fallback for unsupported GPU backends", () => {
  assert.throws(() => resolveNativeBackendRoute({
    capabilityRegistry: prototypeProfile().capabilityRegistry,
    requestedBackend: "cuda",
    allowBackendFallback: false,
  }), /explicitly allow backend fallback/i);

  const route = resolveNativeBackendRoute({
    capabilityRegistry: prototypeProfile().capabilityRegistry,
    requestedBackend: "cuda",
    allowBackendFallback: true,
  });
  assert.equal(route.backend, "cpu");
  assert.equal(route.fallbackUsed, true);
  assert.match(route.fallbackReason, /falling back to CPU/i);
});

test("native runtime falls back through the existing backend only when backend fallback is explicitly allowed", async () => {
  let invoked = 0;
  const result = await generateWithRuntimeSession({
    requestedEngine: "lizard-native",
    env: {},
    recommendation: { selected: {} },
    modelArtifact: prototypeArtifact("E:\\fallback-prototype.json", {
      allowBackendFallback: true,
    }),
    profile: prototypeProfile({
      backend: "cuda",
    }),
    prompt: "hello",
    maxTokens: 2,
    config: {
      runtimeDirectory: "E:\\tmp",
      generation: { temperature: 0 },
    },
    generate: async () => {
      invoked += 1;
      return {
        status: "completed",
        runtime: "llama-cli",
        durationMs: 4,
        output: "fallback-output",
      };
    },
  });

  assert.equal(invoked, 1);
  assert.equal(result.runtime, "llama-cli");
  assert.equal(result.output, "fallback-output");
});

test("native CPU backend still performs numeric operations for the prototype executor", () => {
  const backend = createCpuBackend();
  assert.deepEqual(backend.matmulVectorMatrix([1, 2], [[3, 4], [5, 6]]), [13, 16]);
  assert.deepEqual(backend.addBias([13, 16], [1, -1]), [14, 15]);
  assert.equal(backend.argmax([1, 7, 3]), 1);
});
