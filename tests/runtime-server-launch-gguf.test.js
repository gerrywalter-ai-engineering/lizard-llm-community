import test from "node:test";
import assert from "node:assert/strict";
import { hydrateLaunchGguf } from "../src/runtime/server/launch-gguf.js";

test("native server re-inspects a GGUF when a stripped vocabulary has no compacted flag", () => {
  const inspected = {
    architecture: "gemma4",
    quant: "Q4_K_M",
    raw: {
      "tokenizer.ggml.tokens": Array.from({ length: 200_001 }, (_, index) => `token-${index}`),
    },
  };
  const calls = [];
  const result = hydrateLaunchGguf({
    modelArtifact: {
      modelPath: "C:\\models\\large-vocabulary.gguf",
      gguf: {
        architecture: "gemma4",
        quant: "Q4_K_M",
        raw: { "general.architecture": "gemma4" },
      },
    },
    inspect(modelPath, options) {
      calls.push({ modelPath, options });
      return inspected;
    },
  });

  assert.equal(result, inspected);
  assert.deepEqual(calls, [{
    modelPath: "C:\\models\\large-vocabulary.gguf",
    options: {
      fallbackQuant: "Q4_K_M",
      fallbackArchitecture: "gemma4",
    },
  }]);
  assert.equal(result.raw["tokenizer.ggml.tokens"].length, 200_001);
});

test("native server keeps an already complete launch vocabulary without reading the GGUF again", () => {
  const gguf = {
    architecture: "llama",
    raw: { "tokenizer.ggml.tokens": ["<s>", "hello"] },
  };
  const result = hydrateLaunchGguf({
    modelArtifact: { modelPath: "C:\\models\\llama.gguf", gguf },
    inspect() {
      throw new Error("complete metadata must not be re-inspected");
    },
  });
  assert.equal(result, gguf);
});

test("native server does not inspect declared runtime prototypes as GGUF files", () => {
  const ggufHints = {
    blockCount: 1,
    embeddingLength: 2,
    tokenizer: "prototype",
  };
  const result = hydrateLaunchGguf({
    modelArtifact: {
      modelPath: "C:\\models\\tiny-native-model.json",
      runtimePrototype: { format: "lizard-cpu-prototype-v1" },
    },
    profile: { gguf: ggufHints },
    inspect() {
      throw new Error("prototype models must not be parsed as GGUF");
    },
  });
  assert.equal(result, ggufHints);
});

test("native server uses profile GGUF hints when the model artifact metadata is absent", () => {
  const calls = [];
  const inspected = { raw: { "tokenizer.ggml.tokens": ["<s>", "ready"] } };
  const result = hydrateLaunchGguf({
    modelArtifact: { modelPath: "C:\\models\\profile-only.gguf" },
    profile: { gguf: { architecture: "qwen2", quant: "Q6_K", raw: {} } },
    inspect(modelPath, options) {
      calls.push({ modelPath, options });
      return inspected;
    },
  });
  assert.equal(result, inspected);
  assert.deepEqual(calls[0].options, {
    fallbackQuant: "Q6_K",
    fallbackArchitecture: "qwen2",
  });
});
