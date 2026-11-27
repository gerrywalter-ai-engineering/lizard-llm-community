import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { createCpuPrototypeExecutor } from "../src/runtime/executor/index.js";
import { loadCpuPrototypeModel, CPU_PROTOTYPE_FORMAT } from "../src/runtime/loader/index.js";
import { generateWithRuntimeSession } from "../src/runtime/index.js";

function tempDir() {
  return fs.mkdtempSync(path.join(os.tmpdir(), "lizard-native-cpu-"));
}

function writePrototypeModel(root) {
  const modelPath = path.join(root, "tiny-native-model.json");
  fs.writeFileSync(modelPath, JSON.stringify({
    format: CPU_PROTOTYPE_FORMAT,
    architecture: "tiny-linear",
    quantization: "f32",
    vocabulary: ["hello", "world", "lizard"],
    embeddings: [
      [1, 0],
      [0, 1],
      [1, 1],
    ],
    outputWeights: [
      [0, 4, 1],
      [4, 0, 2],
    ],
    bias: [0, 0, 0],
    maxGenerationTokens: 4,
  }, null, 2));
  return modelPath;
}

function buildRuntimeArgs(modelPath, runtimeDirectory) {
  return {
    recommendation: { selected: {} },
    modelArtifact: {
      modelId: "tiny-native",
      modelPath,
      runtimePrototype: {
        format: CPU_PROTOTYPE_FORMAT,
      },
    },
    profile: {
      backend: "cpu",
      threads: 1,
      contextSize: 64,
      timeoutMs: 1000,
      gguf: {
        blockCount: 1,
        embeddingLength: 2,
        tokenizer: "prototype",
      },
      capabilityRegistry: {
        memory: {
          availableGb: 1,
        },
      },
    },
    prompt: "hello",
    maxTokens: 2,
    config: {
      runtimeDirectory,
      generation: {
        temperature: 0,
      },
    },
  };
}

test("CPU prototype executor generates deterministic tokens for a tiny linear model", () => {
  const root = tempDir();
  const modelPath = writePrototypeModel(root);
  const model = loadCpuPrototypeModel({
    modelPath,
  });
  const executor = createCpuPrototypeExecutor();
  const cases = [
    {
      prompt: "hello",
      maxTokens: 2,
      expectedSeedToken: "hello",
      expectedGeneratedTokens: ["world", "hello"],
      expectedOutput: "world hello",
    },
    {
      prompt: "HELLO!!!",
      maxTokens: 2,
      expectedSeedToken: "hello",
      expectedGeneratedTokens: ["world", "hello"],
      expectedOutput: "world hello",
    },
    {
      prompt: "unknown-token",
      maxTokens: 5,
      expectedSeedToken: "hello",
      expectedGeneratedTokens: ["world", "hello", "world", "hello"],
      expectedOutput: "world hello world hello",
    },
    {
      prompt: "world hello",
      maxTokens: 2,
      expectedSeedToken: "hello",
      expectedGeneratedTokens: ["world", "hello"],
      expectedOutput: "world hello",
    },
  ];

  for (const entry of cases) {
    const result = executor.generate({
      model,
      prompt: entry.prompt,
      maxTokens: entry.maxTokens,
    });

    assert.equal(result.seedToken, entry.expectedSeedToken);
    assert.deepEqual(result.generatedTokens, entry.expectedGeneratedTokens);
    assert.equal(result.output, entry.expectedOutput);
  }
});

test("native CPU runtime matches existing backend output for a prototype artifact", async () => {
  const runtimeDirectory = tempDir();
  const modelPath = writePrototypeModel(runtimeDirectory);
  const baseArgs = buildRuntimeArgs(modelPath, runtimeDirectory);

  const baseline = await generateWithRuntimeSession({
    ...baseArgs,
    requestedEngine: null,
    env: {},
    generate: async () => ({
      status: "completed",
      runtime: "llama-cli",
      durationMs: 8,
      output: "world hello",
    }),
  });

  const native = await generateWithRuntimeSession({
    ...baseArgs,
    requestedEngine: "lizard-native",
    env: {},
    generate: async () => {
      throw new Error("Native CPU prototype should not call the existing backend generator.");
    },
  });

  assert.equal(native.runtime, "lizard-native-cpu");
  assert.equal(native.output, baseline.output);
  assert.equal(native.promptArtifact.renderedPromptHash, baseline.promptArtifact.renderedPromptHash);
  assert.equal(native.kvCache.prefixKey, baseline.kvCache.prefixKey);
  assert.equal(native.kvCache.simulated, true);
  assert.equal(native.scheduler.chunkCount, 2);
  assert.deepEqual(native.nativeExecution.generatedTokenIds, [1, 0]);
  assert.equal(native.nativeExecution.mode, "worker-rpc");
  assert.equal(native.nativeExecution.backend, "cpu");
  assert.equal(native.nativeExecution.compatibility.supported, true);
  assert.equal(native.nativeExecution.compatibility.architecture.key, "tiny-linear");
  assert.equal(native.nativeExecution.compatibility.quantization.key, "f32");
  assert.equal(native.runtimeProvider, "lizard-native");
  assert.equal(native.nativeBackend, "cpu");
  assert.match(native.note, /native worker/i);
});
