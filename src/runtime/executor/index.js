import { createCpuBackend } from "../backend/index.js";

function normalizePromptToken(token) {
  return String(token ?? "")
    .trim()
    .toLowerCase()
    .replace(/^[^a-z0-9_-]+|[^a-z0-9_-]+$/g, "");
}

function resolveSeedTokenIndex({ model, prompt }) {
  const tokens = String(prompt ?? "")
    .split(/\s+/)
    .map((token) => normalizePromptToken(token))
    .filter(Boolean);
  for (let index = tokens.length - 1; index >= 0; index -= 1) {
    const resolved = model.vocabularyIndex.get(tokens[index]);
    if (resolved != null) {
      return resolved;
    }
  }
  return 0;
}

export function createCpuPrototypeExecutor({
  backend = createCpuBackend(),
} = {}) {
  return {
    backend,
    generate({
      model,
      prompt,
      maxTokens = null,
    }) {
      if (!model?.vocabulary?.length) {
        throw new Error("CPU prototype executor requires a loaded model vocabulary.");
      }

      const generationLimit = Number.isFinite(Number(maxTokens)) && Number(maxTokens) > 0
        ? Math.min(Math.floor(Number(maxTokens)), model.maxGenerationTokens)
        : 1;
      let currentTokenIndex = resolveSeedTokenIndex({
        model,
        prompt,
      });
      const seedToken = model.vocabulary[currentTokenIndex];
      const generatedTokens = [];
      const generatedTokenIds = [];
      let lastLogits = [];

      for (let step = 0; step < generationLimit; step += 1) {
        const hiddenState = model.embeddings[currentTokenIndex];
        const logits = backend.addBias(
          backend.matmulVectorMatrix(hiddenState, model.outputWeights),
          model.bias,
        );
        lastLogits = logits;
        currentTokenIndex = backend.argmax(logits);
        generatedTokenIds.push(currentTokenIndex);
        generatedTokens.push(model.vocabulary[currentTokenIndex]);
      }

      return {
        output: generatedTokens.join(" ").trim(),
        chunks: generatedTokens.map((token, index) => (
          index < generatedTokens.length - 1
            ? `${token} `
            : token
        )),
        seedToken,
        generatedTokens,
        generatedTokenIds,
        logits: [...lastLogits],
      };
    },
  };
}
