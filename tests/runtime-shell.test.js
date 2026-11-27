import test from "node:test";
import assert from "node:assert/strict";
import { ExistingBackendRuntime } from "../src/runtime/core/existing-backend-runtime.js";
import { ModelSession } from "../src/runtime/core/model-session.js";
import {
  createInferenceRuntime,
  generateWithRuntimeSession,
  resolveModelRuntimeEngine,
  resolveRuntimeForCommand,
  resolveRuntimeEngine,
} from "../src/runtime/index.js";
import { summarizeAttentionValueGaps } from "../src/lib/gguf-metadata.js";

async function collect(stream) {
  const items = [];
  for await (const item of stream) {
    items.push(item);
  }
  return items;
}

test("ExistingBackendRuntime creates a session and emits a simple request lifecycle", async () => {
  const runtime = new ExistingBackendRuntime({
    generate: async ({ prompt, maxTokens }) => ({
      status: "completed",
      runtime: "llama-cli",
      durationMs: 42,
      output: `${prompt} :: ${maxTokens}`,
    }),
  });

  const session = await runtime.createSession({
    recommendation: { selected: {} },
    modelArtifact: { modelId: "demo-local", modelPath: "E:\\demo.gguf" },
    profile: null,
    config: { runtimeDirectory: "E:\\tmp" },
  });
  const events = await collect(session.generateStream({
    prompt: "hello",
    maxTokens: 16,
  }));

  assert.deepEqual(events.map((event) => event.type), ["start", "delta", "complete"]);
  assert.equal(events.at(-1).output, "hello :: 16");
  assert.equal(events.at(-1).metadata.scheduler.chunkCount, 1);
  assert.equal(session.state.activeRequestId, null);
  await session.close();
  assert.equal(session.state.closed, true);
});

test("ModelSession supports cancellation during an active request", async () => {
  let release;
  const ready = new Promise((resolve) => {
    release = resolve;
  });
  const session = new ModelSession({
    modelId: "demo-local",
    onGenerate: async () => {
      await ready;
      return {
        output: "late output",
        durationMs: 10,
      };
    },
  });

  const pending = collect(session.generateStream({
    requestId: "req-1",
    prompt: "cancel me",
  }));
  await Promise.resolve();
  assert.equal(session.cancelRequest("req-1"), true);
  release();
  const events = await pending;

  assert.deepEqual(events.map((event) => event.type), ["start", "cancelled"]);
});

test("resolveRuntimeEngine recognizes the experimental native gate", () => {
  assert.deepEqual(resolveRuntimeEngine({
    requestedEngine: "lizard-native",
    env: {},
  }), {
    engine: "lizard-native",
    experimental: true,
    enabledBy: "cli",
  });

  assert.deepEqual(resolveRuntimeEngine({
    env: { LIZARD_NATIVE_INFERENCE: "1" },
  }), {
    engine: "lizard-native",
    experimental: true,
    enabledBy: "env",
  });
});

test("createInferenceRuntime keeps the existing backend adapter behind the native flag", () => {
  const runtime = createInferenceRuntime({
    requestedEngine: "lizard-native",
    env: {},
    generate: async () => ({
      status: "completed",
      runtime: "llama-cli",
      durationMs: 1,
      output: "ok",
    }),
  });

  assert.equal(runtime.resolution.engine, "lizard-native");
  assert.equal(runtime.runtime.engine, "lizard-native");
  assert.equal(typeof runtime.runtime.createSession, "function");
});

test("resolveModelRuntimeEngine keeps single-wave GGUF families like phi3 on the native worker", () => {
  const resolution = resolveModelRuntimeEngine({
    requestedEngine: "lizard-native",
    env: {},
    modelArtifact: {
      modelId: "phi-3.5-mini-instruct-gguf-q4_k_m",
      modelPath: "E:\\phi.gguf",
      gguf: {
        architecture: "phi3",
        quant: "Q4_K_M",
      },
    },
    profile: {
      gguf: {
        architecture: "phi3",
        quant: "Q4_K_M",
      },
    },
  });

  assert.equal(resolution.engine, "lizard-native");
  assert.equal(resolution.experimental, true);
  assert.equal(resolution.fallbackReason ?? null, null);
});

test("resolveModelRuntimeEngine reports blocked GGUF families (deepseek2 MLA) explicitly", () => {
  const resolution = resolveModelRuntimeEngine({
    requestedEngine: "lizard-native",
    env: {},
    modelArtifact: {
      modelId: "deepseek-v2-lite-q4_k_m",
      modelPath: "E:\\deepseek2.gguf",
      gguf: {
        architecture: "deepseek2",
        quant: "Q4_K_M",
      },
    },
    profile: {
      gguf: {
        architecture: "deepseek2",
        quant: "Q4_K_M",
      },
    },
  });

  assert.equal(resolution.engine, "llama.cpp");
  assert.equal(resolution.experimental, false);
  assert.equal(resolution.experimentalRequested, true);
  assert.match(resolution.fallbackReason, /Multi-head Latent Attention/);
  assert.match(resolution.fallbackReason, /will be updated soon/i);
  assert.equal(resolution.nativeCompatibility.architecture.status, "blocked");
});

test("summarizeAttentionValueGaps flags Gemma-4-style cross-layer value sharing only", () => {
  const layers = 12;
  const names = [];
  for (let layer = 0; layer < layers; layer += 1) {
    names.push(`blk.${layer}.attn_q.weight`, `blk.${layer}.attn_k.weight`);
    if (layer % 6 !== 5) {
      names.push(`blk.${layer}.attn_v.weight`);
    }
  }
  const gaps = summarizeAttentionValueGaps(names, layers);
  assert.deepEqual(gaps.missingLayers, [5, 11]);
  assert.equal(gaps.missingLayerCount, 2);
  assert.equal(gaps.presentLayerCount, 10);
  assert.equal(gaps.totalLayers, 12);

  // Complete directories report no gaps.
  const complete = names.concat(["blk.5.attn_v.weight", "blk.11.attn_v.weight"]);
  assert.equal(summarizeAttentionValueGaps(complete, layers), null);

  // Fused-QKV layouts satisfy the value requirement per layer.
  const fused = [];
  for (let layer = 0; layer < 4; layer += 1) {
    fused.push(`blk.${layer}.attn_qkv.weight`);
  }
  assert.equal(summarizeAttentionValueGaps(fused, 4), null);

  // Fully V-less layouts belong to the architecture matrix, not this summary.
  assert.equal(summarizeAttentionValueGaps(["blk.0.attn_q.weight", "blk.1.attn_q.weight"], 2), null);
});

test("resolveModelRuntimeEngine routes cross-layer value-sharing GGUFs to llama.cpp before native load", () => {
  const gguf = {
    architecture: "gemma4",
    quant: "Q4_K_M",
    attentionValueGaps: {
      missingLayers: [5, 11, 17, 23, 29, 35, 41, 47],
      missingLayerCount: 8,
      presentLayerCount: 40,
      totalLayers: 48,
    },
  };
  const resolution = resolveModelRuntimeEngine({
    requestedEngine: "lizard-native",
    env: {},
    modelArtifact: {
      modelId: "gemma-4-12b-coder-q4_k_m",
      modelPath: "E:\\gemma4-coder.gguf",
      gguf,
    },
    profile: { gguf },
  });

  assert.equal(resolution.engine, "llama.cpp");
  assert.equal(resolution.experimental, false);
  assert.equal(resolution.experimentalRequested, true);
  assert.match(resolution.fallbackReason, /value projections across layers/i);
  assert.match(resolution.fallbackReason, /8 of 48 layers/);
  assert.equal(resolution.nativeCompatibility.tensors.status, "blocked");
  assert.equal(resolution.nativeCompatibility.architecture.status, "supported");
});

test("resolveModelRuntimeEngine keeps complete gemma4 directories (e4b-style) on the native worker", () => {
  const gguf = {
    architecture: "gemma4",
    quant: "Q4_K_M",
    attentionValueGaps: null,
  };
  const resolution = resolveModelRuntimeEngine({
    requestedEngine: "lizard-native",
    env: {},
    modelArtifact: {
      modelId: "gemma4-e4b-q4_k_m",
      modelPath: "E:\\gemma4-e4b.gguf",
      gguf,
    },
    profile: { gguf },
  });

  assert.equal(resolution.engine, "lizard-native");
  assert.equal(resolution.fallbackReason ?? null, null);
});

test("generateWithRuntimeSession uses the runtime shell when the experimental engine is selected", async () => {
  const result = await generateWithRuntimeSession({
    requestedEngine: "lizard-native",
    env: {},
    recommendation: { selected: {} },
    modelArtifact: { modelId: "demo-local", modelPath: "E:\\demo.gguf" },
    profile: null,
    prompt: "hello runtime",
    maxTokens: 24,
    config: {
      runtimeDirectory: "E:\\tmp",
      generation: { temperature: 0.2 },
    },
    allowOracleFallback: true,
    generate: async ({ prompt, maxTokens }) => ({
      status: "completed",
      runtime: "llama-cli",
      durationMs: 21,
      output: `${prompt}:${maxTokens}`,
    }),
  });

  assert.equal(result.engine, "lizard-native");
  assert.equal(result.experimentalEngine, true);
  assert.equal(result.runtime, "llama-cli");
  assert.equal(result.output, "hello runtime:24");
  assert.equal(result.promptArtifact.tokenCountMode, "estimate");
  assert.match(result.promptArtifact.renderedPrompt, /hello runtime/);
  assert.ok(result.kvCache.prefixKey);
  assert.equal(result.kvCache.simulated, true);
  assert.equal(result.kvCache.slotId, "kv-slot-1");
  assert.equal(result.runtimeProvider, "llama.cpp");
  assert.match(result.note, /oracle fallback/i);
});

test("generateWithRuntimeSession routes blocked GGUF families to llama.cpp before native worker load", async () => {
  let generateCalls = 0;
  const result = await generateWithRuntimeSession({
    requestedEngine: "lizard-native",
    env: {},
    recommendation: { selected: {} },
    modelArtifact: {
      modelId: "deepseek-v2-lite-q4_k_m",
      modelPath: "E:\\deepseek2.gguf",
      gguf: {
        architecture: "deepseek2",
        quant: "Q4_K_M",
      },
    },
    profile: {
      backend: "cpu",
      contextSize: 64,
      gguf: {
        architecture: "deepseek2",
        quant: "Q4_K_M",
        blockCount: 27,
        embeddingLength: 2048,
        tokenizer: "gpt2",
      },
      capabilityRegistry: {
        memory: {
          availableGb: 8,
        },
      },
    },
    prompt: "hello phi",
    maxTokens: 24,
    config: {
      runtimeDirectory: "E:\\tmp",
      generation: { temperature: 0.2 },
    },
    generate: async ({ prompt, maxTokens }) => {
      generateCalls += 1;
      return {
        status: "completed",
        runtime: "llama-cli",
        durationMs: 9,
        output: `${prompt}:${maxTokens}`,
      };
    },
  });

  assert.equal(generateCalls, 1);
  assert.equal(result.engine, "llama.cpp");
  assert.equal(result.experimentalEngine, false);
  assert.equal(result.runtime, "llama-cli");
  assert.equal(result.runtimeProvider, "llama.cpp");
  assert.equal(result.output, "hello phi:24");
  assert.match(result.note, /falling back to llama\.cpp/i);
});

test("experimental native runtime rejects unsupported native backend requests unless fallback is explicitly allowed", async () => {
  await assert.rejects(() => generateWithRuntimeSession({
    requestedEngine: "lizard-native",
    env: {},
    recommendation: { selected: {} },
    modelArtifact: {
      modelId: "tiny-native",
      modelPath: "E:\\demo-prototype.json",
      runtimePrototype: {
        format: "lizard-tiny-linear",
      },
    },
    profile: {
      backend: "cuda",
      contextSize: 64,
      capabilityRegistry: {
        backends: {
          cpu: { supported: true, reason: "CPU available." },
          cuda: { supported: true, reason: "CUDA hardware detected." },
        },
        memory: {
          availableGb: 1,
        },
      },
      gguf: {
        blockCount: 1,
        embeddingLength: 2,
        tokenizer: "prototype",
      },
    },
    prompt: "hello",
    maxTokens: 2,
    config: {
      runtimeDirectory: "E:\\tmp",
      generation: { temperature: 0 },
    },
    generate: async () => ({
      status: "completed",
      runtime: "llama-cli",
      durationMs: 4,
      output: "fallback",
    }),
  }), /not implemented yet/i);
});

test("resolveRuntimeForCommand allows native rollout for run and benchmark but still blocks unsupported subcommands", () => {
  assert.deepEqual(resolveRuntimeForCommand({
    command: "run",
    requestedEngine: "lizard-native",
    env: {},
  }), {
    engine: "lizard-native",
    experimental: true,
    enabledBy: "cli",
  });

  assert.deepEqual(resolveRuntimeForCommand({
    command: "benchmark",
    env: { LIZARD_NATIVE_INFERENCE: "1" },
  }), {
    engine: "lizard-native",
    experimental: true,
    enabledBy: "env",
  });

  assert.throws(() => resolveRuntimeForCommand({
    command: "benchmark pro",
    requestedEngine: "lizard-native",
    env: {},
  }), /supported experimental commands: generate, run, benchmark/i);
});
