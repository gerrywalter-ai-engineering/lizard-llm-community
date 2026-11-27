import test from "node:test";
import assert from "node:assert/strict";
import {
  detectQuantFromFilename,
  estimateParamsBFromText,
  pipelineCategory,
  parseNextCursor,
  quantBitsPerWeight,
  isDownloadableGguf,
  searchHubModels,
  getHubModelDetail,
} from "../src/lib/hf-hub.js";
import {
  classifyCompatibility,
  estimateQuantFootprint,
  approxModelShape,
} from "../src/lib/huggingface.js";
import { syntheticHfModelFromGguf, familyFromArchitecture } from "../src/lib/hf-model-catalog.js";

test("detectQuantFromFilename picks the most specific quant label", () => {
  assert.equal(detectQuantFromFilename("Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"), "Q4_K_M");
  assert.equal(detectQuantFromFilename("gemma-4-E4B-it-Q8_0.gguf"), "Q8_0");
  assert.equal(detectQuantFromFilename("model.IQ4_XS.gguf"), "IQ4_XS");
  assert.equal(detectQuantFromFilename("model-f16.gguf"), "F16");
  assert.equal(detectQuantFromFilename("plain-model.gguf"), null);
  assert.equal(detectQuantFromFilename(""), null);
});

test("estimateParamsBFromText parses B / M / MoE conventions", () => {
  assert.equal(estimateParamsBFromText("Qwen2.5-7B-Instruct"), 7);
  assert.equal(estimateParamsBFromText("Llama-3.2-1.5B"), 1.5);
  assert.equal(estimateParamsBFromText("SmolLM-360M"), 0.36);
  assert.equal(estimateParamsBFromText("Mixtral-8x7B"), 56);
  assert.equal(estimateParamsBFromText("no-size-here"), null);
});

test("pipelineCategory maps HF tags to onboarding filter buckets", () => {
  assert.equal(pipelineCategory("text-generation"), "text");
  assert.equal(pipelineCategory("feature-extraction"), "embeddings");
  assert.equal(pipelineCategory("image-text-to-text"), "multimodal");
  assert.equal(pipelineCategory("automatic-speech-recognition"), "audio");
  assert.equal(pipelineCategory("image-classification"), "vision");
  assert.equal(pipelineCategory(null), "other");
});

test("parseNextCursor extracts the cursor from a Link header", () => {
  assert.equal(
    parseNextCursor('<https://huggingface.co/api/models?filter=gguf&cursor=ABC123>; rel="next"'),
    "ABC123",
  );
  assert.equal(parseNextCursor(null), null);
  assert.equal(parseNextCursor("<https://x/api/models?foo=bar>; rel=\"prev\""), null);
});

test("quantBitsPerWeight covers the wider Hub quant set", () => {
  assert.equal(quantBitsPerWeight("TQ1_0"), 1.69);
  assert.equal(quantBitsPerWeight("Q4_K_M"), 4.5);
  assert.equal(quantBitsPerWeight("Q8_0"), 8);
  assert.equal(quantBitsPerWeight("IQ4_XS"), 4.25);
  assert.equal(quantBitsPerWeight("F16"), 16);
  assert.equal(quantBitsPerWeight("nonexistent"), null);
});

test("isDownloadableGguf excludes auxiliary and split-shard files", () => {
  assert.equal(isDownloadableGguf("Qwen2.5-7B-Instruct-Q4_K_M.gguf"), true);
  assert.equal(isDownloadableGguf("mmproj-model-f16.gguf"), false);
  assert.equal(isDownloadableGguf("model.projector.gguf"), false);
  assert.equal(isDownloadableGguf("Big-Model-Q6_K-00001-of-00003.gguf"), false);
  assert.equal(isDownloadableGguf("README.md"), false);
});

function hardware({ ramGB = 16, vramGB = 0 } = {}) {
  return {
    memory: { plannerBudgetGB: ramGB * 0.85, availableGB: ramGB * 0.85 },
    primaryGpu: vramGB ? { name: "Test GPU", vramGB } : null,
  };
}

test("classifyCompatibility returns rich labels across the fit spectrum", () => {
  const tiny = classifyCompatibility({ paramsB: 1, quant: "Q4_K_M", hardware: hardware({ ramGB: 32 }) });
  assert.ok(["Recommended", "Runs Great"].includes(tiny.label));
  assert.equal(tiny.tone, "good");

  const huge = classifyCompatibility({ paramsB: 70, quant: "Q8_0", hardware: hardware({ ramGB: 16 }) });
  assert.equal(huge.label, "Too Large");
  assert.equal(huge.tone, "bad");

  // A GPU that fits the weights earns the top "Recommended" label.
  const gpu = classifyCompatibility({ paramsB: 7, quant: "Q4_K_M", hardware: hardware({ ramGB: 32, vramGB: 24 }) });
  assert.equal(gpu.label, "Recommended");
  assert.ok(gpu.vramGB > 0);
  assert.ok(gpu.tokensPerSec > 0);
});

test("classifyCompatibility trusts a known GGUF file size for the disk figure", () => {
  const comp = classifyCompatibility({ paramsB: 8, quant: "Q4_K_M", hardware: hardware({ ramGB: 16 }), sizeGB: 4.9 });
  assert.equal(comp.diskGB, 4.9);
  assert.ok(comp.ramGB >= 4.9);
});

test("estimateQuantFootprint falls back to Hub bpw for non-curated quants", () => {
  const shape = approxModelShape(7);
  const iq = estimateQuantFootprint({ paramsB: 7, quant: "IQ4_XS", contextTokens: 4096, ...shape });
  assert.ok(iq.weightsGB > 0 && iq.totalGB > iq.weightsGB);
});

test("syntheticHfModelFromGguf reconstructs a runnable shape from parsed GGUF", () => {
  const model = syntheticHfModelFromGguf({
    id: "custom-model-q4_k_m",
    hfRepo: "someone/Custom-Model-GGUF",
    license: "apache-2.0",
    paramsB: 7,
    quant: "Q4_K_M",
    gguf: { architecture: "qwen2", blockCount: 28, embeddingLength: 3584, contextLength: 32768, quant: "Q4_K_M" },
  });
  assert.equal(model.family, "qwen");
  assert.equal(model.totalLayers, 28);
  assert.equal(model.hiddenSize, 3584);
  assert.equal(model.defaultContext, 8192);
  assert.equal(model.maxContext, 32768);
  assert.deepEqual(model.quantizations, ["Q4_K_M"]);
  assert.equal(model.synthetic, true);
});

test("syntheticHfModelFromGguf returns null without a usable layer/embedding shape", () => {
  assert.equal(syntheticHfModelFromGguf({ id: "x", gguf: {} }), null);
  assert.equal(syntheticHfModelFromGguf(null), null);
});

test("familyFromArchitecture normalizes versioned architectures", () => {
  assert.equal(familyFromArchitecture("qwen2"), "qwen");
  assert.equal(familyFromArchitecture("gemma2"), "gemma");
  assert.equal(familyFromArchitecture("llama"), "llama");
  assert.equal(familyFromArchitecture("phi3"), "phi");
  assert.equal(familyFromArchitecture(""), "unknown");
});

test("searchHubModels normalizes results and threads the cursor (mocked fetch)", async () => {
  const fetchImpl = async (url) => {
    assert.match(url, /filter=gguf/);
    assert.match(url, /sort=downloads/);
    return {
      ok: true,
      headers: { get: (name) => (name === "link" ? '<https://huggingface.co/api/models?cursor=NEXT99>; rel="next"' : null) },
      json: async () => [
        { id: "bartowski/Qwen2.5-7B-Instruct-GGUF", downloads: 1000, likes: 50, pipeline_tag: "text-generation", tags: ["gguf", "license:apache-2.0"], lastModified: "2025-01-01T00:00:00Z" },
      ],
    };
  };
  const { models, nextCursor } = await searchHubModels({ query: "qwen", fetchImpl });
  assert.equal(nextCursor, "NEXT99");
  assert.equal(models.length, 1);
  assert.equal(models[0].repo, "bartowski/Qwen2.5-7B-Instruct-GGUF");
  assert.equal(models[0].author, "bartowski");
  assert.equal(models[0].paramsB, 7);
  assert.equal(models[0].category, "text");
  assert.equal(models[0].license, "apache-2.0");
  assert.equal(models[0].gguf, true);
});

test("getHubModelDetail merges the model record with the GGUF file tree (mocked fetch)", async () => {
  const fetchImpl = async (url) => {
    if (url.endsWith("/tree/main?recursive=true")) {
      return {
        ok: true,
        json: async () => [
          { type: "file", path: "Custom-Q4_K_M.gguf", size: 4_800_000_000 },
          { type: "file", path: "Custom-Q8_0.gguf", size: 8_100_000_000 },
          { type: "file", path: "README.md", size: 1000 },
        ],
      };
    }
    return {
      ok: true,
      json: async () => ({
        id: "someone/Custom-7B-GGUF",
        downloads: 42,
        likes: 3,
        pipeline_tag: "text-generation",
        tags: ["gguf", "license:mit"],
        cardData: { license: "mit" },
        siblings: [],
      }),
    };
  };
  const detail = await getHubModelDetail("someone/Custom-7B-GGUF", { fetchImpl });
  assert.equal(detail.repo, "someone/Custom-7B-GGUF");
  assert.equal(detail.paramsB, 7);
  assert.equal(detail.ggufFiles.length, 2);
  // Ordered low-footprint first.
  assert.equal(detail.ggufFiles[0].quant, "Q4_K_M");
  assert.equal(detail.ggufFiles[0].sizeGB, 4.47);
  assert.deepEqual(detail.quantVariants, ["Q4_K_M", "Q8_0"]);
});
