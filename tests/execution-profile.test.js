import test from "node:test";
import assert from "node:assert/strict";
import { buildLlamaCppArgs, resolveProfileThreads } from "../src/lib/execution-profile.js";

test("the qualified Run 06 GGUF resolves to five workers while explicit overrides win", () => {
  const selected = {
    family: "llama",
    totalLayers: 28,
    hiddenSize: 3072,
    quant: "Q5_K_M",
  };
  const hardware = { cpu: { logicalThreads: 8 } };
  assert.equal(resolveProfileThreads({
    selected,
    gguf: { quant: "Q4_K_M" },
    hardware,
  }), 5);
  assert.equal(resolveProfileThreads({
    overrides: { threads: 6 },
    selected,
    gguf: { quant: "Q4_K_M" },
    hardware,
  }), 6);
  assert.equal(resolveProfileThreads({ selected, gguf: { quant: "Q5_K_M" }, hardware }), 6);
});

test("buildLlamaCppArgs disables conversation mode when noConversationMode is true", () => {
  const args = buildLlamaCppArgs({
    profile: {
      modelPath: "E:\\demo.gguf",
      contextSize: 128,
      gpuLayers: 0,
      threads: 4,
      batchSize: 512,
      ubatchSize: 128,
      kvCache: "f16",
      mmap: true,
      mlock: false,
      noConversationMode: true,
    },
    prompt: "Hello",
    maxTokens: 1,
  });

  assert.ok(args.includes("--no-conversation"));
  assert.ok(!args.includes("-cnv"));
});

