import fs from "node:fs";
import {
  estimatePrototypeModelBytes,
  getArchitectureCompatibility,
  getQuantizationCompatibility,
} from "../compatibility/index.js";

export const CPU_PROTOTYPE_FORMAT = "lizard-tiny-linear";

function normalizeVector(values, label) {
  if (!Array.isArray(values) || values.length === 0) {
    throw new Error(`${label} must be a non-empty numeric array.`);
  }
  return values.map((value, index) => {
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) {
      throw new Error(`${label}[${index}] must be a finite number.`);
    }
    return numeric;
  });
}

function normalizeMatrix(rows, label) {
  if (!Array.isArray(rows) || rows.length === 0) {
    throw new Error(`${label} must be a non-empty matrix.`);
  }
  const matrix = rows.map((row, index) => normalizeVector(row, `${label}[${index}]`));
  const width = matrix[0].length;
  if (!matrix.every((row) => row.length === width)) {
    throw new Error(`${label} rows must all have the same width.`);
  }
  return matrix;
}

export function canLoadCpuPrototypeModel(modelArtifact = {}) {
  return modelArtifact?.runtimePrototype?.format === CPU_PROTOTYPE_FORMAT;
}

export function resolveCpuPrototypeModelPath(modelArtifact = {}) {
  return modelArtifact?.runtimePrototype?.modelPath ?? modelArtifact?.modelPath ?? null;
}

export function loadCpuPrototypeModel({
  modelPath = null,
  modelArtifact = null,
} = {}) {
  const resolvedPath = modelPath ?? resolveCpuPrototypeModelPath(modelArtifact);
  if (!resolvedPath) {
    throw new Error("CPU prototype loader requires a model path.");
  }

  let parsed;
  try {
    parsed = JSON.parse(fs.readFileSync(resolvedPath, "utf8"));
  } catch (error) {
    throw new Error(`Failed to load CPU prototype model at ${resolvedPath}: ${error.message}`);
  }

  const format = parsed?.format ?? modelArtifact?.runtimePrototype?.format ?? CPU_PROTOTYPE_FORMAT;
  if (format !== CPU_PROTOTYPE_FORMAT) {
    throw new Error(`Unsupported CPU prototype format '${format}'.`);
  }

  const architecture = String(parsed?.architecture ?? "tiny-linear");
  const architectureCompatibility = getArchitectureCompatibility(architecture);
  if (architectureCompatibility.status !== "supported") {
    throw new Error(architectureCompatibility.reason ?? `Unsupported CPU prototype architecture '${architecture}'.`);
  }

  const quantization = String(parsed?.quantization ?? "f32");
  const quantizationCompatibility = getQuantizationCompatibility(quantization);
  if (quantizationCompatibility.status !== "supported") {
    throw new Error(quantizationCompatibility.reason ?? `Unsupported CPU prototype quantization '${quantization}'.`);
  }

  const vocabulary = Array.isArray(parsed?.vocabulary) && parsed.vocabulary.length > 0
    ? parsed.vocabulary.map((token, index) => {
      const normalized = String(token ?? "").trim();
      if (!normalized) {
        throw new Error(`vocabulary[${index}] must be a non-empty string.`);
      }
      return normalized;
    })
    : null;
  if (!vocabulary) {
    throw new Error("CPU prototype model requires a non-empty vocabulary.");
  }

  const embeddings = normalizeMatrix(parsed?.embeddings ?? parsed?.embedding, "embeddings");
  if (embeddings.length !== vocabulary.length) {
    throw new Error("CPU prototype embeddings height must match vocabulary size.");
  }

  const outputWeights = normalizeMatrix(parsed?.outputWeights ?? parsed?.output, "outputWeights");
  if (outputWeights.length !== embeddings[0].length) {
    throw new Error("CPU prototype output weights height must match embedding width.");
  }
  if (outputWeights[0].length !== vocabulary.length) {
    throw new Error("CPU prototype output weights width must match vocabulary size.");
  }

  const bias = parsed?.bias == null
    ? new Array(vocabulary.length).fill(0)
    : normalizeVector(parsed.bias, "bias");
  if (bias.length !== vocabulary.length) {
    throw new Error("CPU prototype bias width must match vocabulary size.");
  }

  const vocabularyIndex = new Map(vocabulary.map((token, index) => [token.toLowerCase(), index]));
  const maxGenerationTokens = Number.isFinite(Number(parsed?.maxGenerationTokens)) && Number(parsed.maxGenerationTokens) > 0
    ? Math.floor(Number(parsed.maxGenerationTokens))
    : 1;
  const maxContextTokens = Number.isFinite(Number(parsed?.maxContextTokens)) && Number(parsed.maxContextTokens) > 0
    ? Math.floor(Number(parsed.maxContextTokens))
    : 128;
  const estimatedModelBytes = estimatePrototypeModelBytes({
    embeddings,
    outputWeights,
    bias,
    bytesPerElement: quantizationCompatibility.bytesPerElement ?? 4,
  });

  return {
    format,
    architecture,
    quantization,
    modelPath: resolvedPath,
    vocabulary,
    vocabularyIndex,
    embeddings,
    outputWeights,
    bias,
    maxGenerationTokens,
    maxContextTokens,
    estimatedModelBytes,
    architectureCompatibility,
    quantizationCompatibility,
    metadata: parsed?.metadata ?? {},
  };
}
