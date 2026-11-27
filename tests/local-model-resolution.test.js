import test from "node:test";
import assert from "node:assert/strict";

import { syntheticHfModelFromGguf, findAllowlistedHfModel } from "../src/lib/hf-model-catalog.js";
import {
  buildCaterpillarGgufFamilyMatrix,
  caterpillarRunnerLoadsFamily,
} from "../src/runtime/index.js";

// A catalog entry as the download manager records one for a model imported
// through Ollama: real GGUF header, no hfRepo, source "ollama".
function ollamaEntry(overrides = {}) {
  return {
    id: "gemma3-4b",
    source: "ollama",
    hfRepo: undefined,
    importedPath: "C:/models/gemma3-4b.gguf",
    gguf: {
      architecture: "gemma3",
      quant: "Q4_K_M",
      blockCount: 34,
      embeddingLength: 2560,
      contextLength: 131072,
    },
    ...overrides,
  };
}

test("a model imported through Ollama still yields a planning shape", () => {
  const entry = ollamaEntry();
  // Not curated, so the allowlist cannot help - the GGUF header has to.
  assert.equal(findAllowlistedHfModel(entry.hfRepo), null);
  const model = syntheticHfModelFromGguf(entry);
  assert.ok(model, "no planning shape built from a complete GGUF header");
  assert.equal(model.id, "gemma3-4b");
  assert.equal(model.totalLayers, 34);
  assert.equal(model.hiddenSize, 2560);
  assert.deepEqual(model.quantizations, ["Q4_K_M"]);
  assert.equal(model.synthetic, true);
});

test("provenance does not decide whether a local model can be planned", () => {
  // The same header planned identically whatever route put it on disk. Gating
  // on source === "huggingface" rejected Ollama-imported models and dropped
  // them onto the static registry, which knows seven curated ids and answers
  // "Unknown model: <id>" for everything else - with the file sitting on disk.
  const fromHuggingFace = syntheticHfModelFromGguf(ollamaEntry({ source: "huggingface", hfRepo: "x/y" }));
  const fromOllama = syntheticHfModelFromGguf(ollamaEntry());
  const fromImport = syntheticHfModelFromGguf(ollamaEntry({ source: "import" }));
  for (const model of [fromHuggingFace, fromOllama, fromImport]) {
    assert.ok(model, "a complete GGUF header must always plan");
    assert.equal(model.totalLayers, 34);
    assert.equal(model.hiddenSize, 2560);
  }
});

test("an incomplete GGUF header still refuses to plan", () => {
  // The one honest reason to decline: without a layer/hidden shape the
  // footprint and GPU-layer maths are meaningless, so guessing would produce a
  // plan that cannot be trusted.
  assert.equal(syntheticHfModelFromGguf(ollamaEntry({ gguf: { architecture: "gemma3" } })), null);
  assert.equal(syntheticHfModelFromGguf(ollamaEntry({ gguf: { blockCount: 34 } })), null);
  assert.equal(syntheticHfModelFromGguf(null), null);
});

test("context length falls back rather than planning a zero-length window", () => {
  const model = syntheticHfModelFromGguf(ollamaEntry({
    gguf: { architecture: "llama", quant: "Q8_0", blockCount: 16, embeddingLength: 2048 },
  }));
  assert.equal(model.maxContext, 8192);
  assert.equal(model.defaultContext, 8192);
});

test("Caterpillar advertises only the decoder families its runner will load", () => {
  // FamilySpec::validate() accepts family llama, gemma or qwen2 and throws
  // otherwise; family_name() folds every gemma revision onto gemma and mistral
  // onto llama. The matrix used to be the native worker's list with the
  // provider relabelled, so ten families planned and then died on load with
  // "Caterpillar Phase 1 supports llama, gemma, and qwen2 decoder families".
  for (const family of ["llama", "mistral", "gemma", "gemma2", "gemma3", "gemma4", "qwen2"]) {
    assert.equal(caterpillarRunnerLoadsFamily(family), true, `${family} should load`);
  }
  for (const family of ["qwen3", "mixtral", "phi3", "deepseek", "glm", "starcoder2"]) {
    assert.equal(caterpillarRunnerLoadsFamily(family), false, `${family} should not load`);
  }

  const matrix = buildCaterpillarGgufFamilyMatrix();
  for (const entry of matrix) {
    if (caterpillarRunnerLoadsFamily(entry.family)) continue;
    assert.equal(entry.workerStatus, "blocked", `${entry.family} still advertised`);
    assert.match(String(entry.reason ?? ""), /llama, gemma and qwen2/);
  }
  // And the ones it does load are still offered, or the fix would have made
  // Caterpillar useless rather than honest.
  const loadable = matrix.filter((entry) => entry.workerStatus === "implemented").map((entry) => entry.family);
  assert.ok(loadable.includes("llama"));
  assert.ok(loadable.includes("gemma"));
  assert.ok(loadable.includes("qwen2"));
});
