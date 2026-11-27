import test from "node:test";
import assert from "node:assert/strict";
import { buildRecommendation, recommendedCpuThreads } from "../src/lib/recommendation.js";
import { findModel, findModelByOllamaTag, loadModelRegistry } from "../src/lib/model-registry.js";

test("selects a feasible coding model", () => {
  const recommendation = buildRecommendation({
    hardware: {
      cpu: { logicalThreads: 16 },
      memory: { totalGB: 32, availableGB: 24 },
      primaryGpu: { name: "RTX 4070", vramGB: 12, backends: { cuda: true } },
    },
    registry: loadModelRegistry(),
    config: {
      defaultMode: "balanced",
      maxRamUsagePercent: 85,
      maxVramUsagePercent: 90,
    },
    requestedMode: "balanced",
    requestedModelId: undefined,
    taskProfile: {
      taskType: "coding",
      difficulty: "medium",
      recommendedMode: "balanced",
      contextNeed: 8192,
      rationale: "test",
    },
  });

  assert.equal(recommendation.selected.mode, "balanced");
  assert.equal(typeof recommendation.selected.modelId, "string");
  assert.equal(recommendation.selected.estimate.fits, true);
});

test("includes runnable local-source models in the registry", () => {
  const registry = loadModelRegistry();
  const llama32 = registry.models.find((model) => model.id === "llama-3.2-3b-instruct");
  assert.equal(llama32.ollamaTag, "llama3.2:3b");
});

test("Run 06 uses five workers only for its qualified CPU/model/quant shape", () => {
  const qualifiedModel = {
    family: "llama",
    totalLayers: 28,
    hiddenSize: 3072,
  };
  assert.equal(recommendedCpuThreads({
    hardware: { cpu: { logicalThreads: 8 } },
    model: qualifiedModel,
    quant: "Q4_K_M",
  }), 5);
  assert.equal(recommendedCpuThreads({
    hardware: { cpu: { logicalThreads: 8 } },
    model: qualifiedModel,
    quant: "Q5_K_M",
  }), 6);
  assert.equal(recommendedCpuThreads({
    hardware: { cpu: { logicalThreads: 16 } },
    model: qualifiedModel,
    quant: "Q4_K_M",
  }), 12);
  assert.equal(recommendedCpuThreads({
    hardware: { cpu: { logicalThreads: 8 } },
    model: { ...qualifiedModel, totalLayers: 32 },
    quant: "Q4_K_M",
  }), 6);
});

test("maps gemma4 aliases and official Ollama tags to the canonical registry model", () => {
  const registry = loadModelRegistry();
  const gemma4 = findModel(registry, "gemma4");

  assert.equal(gemma4.id, "gemma4-e4b");
  assert.equal(findModel(registry, "gemma-2-9b-instruct")?.id, "gemma4-e4b");
  assert.equal(findModelByOllamaTag(registry, "gemma4:e4b")?.id, "gemma4-e4b");
  assert.equal(findModelByOllamaTag(registry, "gemma4:latest")?.id, "gemma4-e4b");
});
