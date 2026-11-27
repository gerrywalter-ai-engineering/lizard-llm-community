// Curated allowlist of corporate-safe Hugging Face GGUF repositories.
//
// We deliberately do NOT allow downloading arbitrary repositories. Every entry
// here carries registry-compatible planning metadata (paramsB, totalLayers,
// hiddenSize, defaultContext, maxContext, tasks, quantizations) so the existing
// `buildRecommendation` / estimator pipeline can plan, benchmark, and activate a
// downloaded GGUF exactly like a registry model.
//
// `files` maps a llama.cpp quantization label (matching `quantizationProfiles`
// in profiles.js) to the concrete GGUF filename inside the repo. sha256 is left
// null when we cannot pin it ahead of time; the download manager will fetch the
// expected hash from the Hugging Face LFS metadata when online and otherwise
// record the locally computed hash.

function quantFile(filename, { sizeGB = null, sha256 = null } = {}) {
  return { filename, sizeGB, sha256 };
}

export const HF_GGUF_ALLOWLIST = [
  {
    id: "gemma4-e4b",
    label: "Gemma 4 E4B Instruct",
    family: "gemma4",
    hfRepo: "ggml-org/gemma-4-E4B-it-GGUF",
    license: "gemma",
    paramsB: 8,
    totalLayers: 42,
    hiddenSize: 3584,
    defaultContext: 8192,
    maxContext: 131072,
    tasks: ["chat", "reasoning", "summarization"],
    // Lizard's native D3D12 resident-GPU decode path is validated against
    // this exact model (see .Agent/betterway/LizardD3D12_Performanceverhaul.md) -
    // it is the only allowlist entry with a proven native GPU speed record,
    // hence `featured`. BF16 is intentionally not offered here even though
    // the repo has it: quantizationProfiles has no 16-bit entry, so its
    // footprint estimate would silently fall back to Q4_K_M's bits-per-weight
    // and understate RAM/VRAM requirements.
    featured: true,
    nativeBackend: true,
    note: "Lizard's native GPU decode path - benchmarked at ~95% of Ollama's decode speed on this machine.",
    ollamaTag: "gemma4:e4b",
    ollamaTags: ["gemma4:latest"],
    quantizations: ["Q4_K_M", "Q8_0"],
    files: {
      Q4_K_M: quantFile("gemma-4-E4B-it-Q4_K_M.gguf", { sizeGB: 4.9 }),
      Q8_0: quantFile("gemma-4-E4B-it-Q8_0.gguf", { sizeGB: 8.5 }),
    },
  },
  {
    id: "qwen2.5-coder-7b-instruct-gguf",
    label: "Qwen2.5 Coder 7B Instruct",
    family: "qwen",
    hfRepo: "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
    license: "apache-2.0",
    paramsB: 7.6,
    totalLayers: 28,
    hiddenSize: 3584,
    defaultContext: 8192,
    maxContext: 32768,
    tasks: ["coding", "reasoning", "structured_output"],
    quantizations: ["Q3_K_M", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"],
    files: {
      Q3_K_M: quantFile("qwen2.5-coder-7b-instruct-q3_k_m.gguf", { sizeGB: 3.8 }),
      Q4_K_M: quantFile("qwen2.5-coder-7b-instruct-q4_k_m.gguf", { sizeGB: 4.7 }),
      Q5_K_M: quantFile("qwen2.5-coder-7b-instruct-q5_k_m.gguf", { sizeGB: 5.4 }),
      Q6_K: quantFile("qwen2.5-coder-7b-instruct-q6_k.gguf", { sizeGB: 6.3 }),
      Q8_0: quantFile("qwen2.5-coder-7b-instruct-q8_0.gguf", { sizeGB: 8.1 }),
    },
  },
  {
    id: "meta-llama-3.1-8b-instruct-gguf",
    label: "Llama 3.1 8B Instruct",
    family: "llama",
    hfRepo: "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
    license: "llama3.1",
    paramsB: 8,
    totalLayers: 32,
    hiddenSize: 4096,
    defaultContext: 8192,
    maxContext: 16384,
    tasks: ["chat", "reasoning", "summarization"],
    quantizations: ["Q3_K_M", "Q4_K_M", "Q5_K_M", "Q6_K"],
    files: {
      Q3_K_M: quantFile("Meta-Llama-3.1-8B-Instruct-Q3_K_M.gguf", { sizeGB: 4.0 }),
      Q4_K_M: quantFile("Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf", { sizeGB: 4.9 }),
      Q5_K_M: quantFile("Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf", { sizeGB: 5.7 }),
      Q6_K: quantFile("Meta-Llama-3.1-8B-Instruct-Q6_K.gguf", { sizeGB: 6.6 }),
    },
  },
  {
    id: "mistral-7b-instruct-v0.3-gguf",
    label: "Mistral 7B Instruct v0.3",
    family: "mistral",
    hfRepo: "bartowski/Mistral-7B-Instruct-v0.3-GGUF",
    license: "apache-2.0",
    paramsB: 7.2,
    totalLayers: 32,
    hiddenSize: 4096,
    defaultContext: 8192,
    maxContext: 32768,
    tasks: ["chat", "coding", "summarization"],
    quantizations: ["Q3_K_M", "Q4_K_M", "Q5_K_M", "Q6_K"],
    files: {
      Q3_K_M: quantFile("Mistral-7B-Instruct-v0.3-Q3_K_M.gguf", { sizeGB: 3.5 }),
      Q4_K_M: quantFile("Mistral-7B-Instruct-v0.3-Q4_K_M.gguf", { sizeGB: 4.4 }),
      Q5_K_M: quantFile("Mistral-7B-Instruct-v0.3-Q5_K_M.gguf", { sizeGB: 5.1 }),
      Q6_K: quantFile("Mistral-7B-Instruct-v0.3-Q6_K.gguf", { sizeGB: 5.9 }),
    },
  },
  {
    id: "llama-3.2-3b-instruct-gguf",
    label: "Llama 3.2 3B Instruct",
    family: "llama",
    hfRepo: "bartowski/Llama-3.2-3B-Instruct-GGUF",
    license: "llama3.2",
    paramsB: 3.2,
    totalLayers: 28,
    hiddenSize: 3072,
    defaultContext: 8192,
    maxContext: 131072,
    tasks: ["chat", "summarization", "reasoning"],
    quantizations: ["Q3_K_M", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"],
    files: {
      Q3_K_M: quantFile("Llama-3.2-3B-Instruct-Q3_K_M.gguf", { sizeGB: 1.7 }),
      Q4_K_M: quantFile("Llama-3.2-3B-Instruct-Q4_K_M.gguf", { sizeGB: 2.0 }),
      Q5_K_M: quantFile("Llama-3.2-3B-Instruct-Q5_K_M.gguf", { sizeGB: 2.3 }),
      Q6_K: quantFile("Llama-3.2-3B-Instruct-Q6_K.gguf", { sizeGB: 2.6 }),
      Q8_0: quantFile("Llama-3.2-3B-Instruct-Q8_0.gguf", { sizeGB: 3.4 }),
    },
  },
  {
    id: "qwen2.5-7b-instruct-gguf",
    label: "Qwen2.5 7B Instruct",
    family: "qwen",
    hfRepo: "bartowski/Qwen2.5-7B-Instruct-GGUF",
    license: "apache-2.0",
    paramsB: 7.6,
    totalLayers: 28,
    hiddenSize: 3584,
    defaultContext: 8192,
    maxContext: 32768,
    tasks: ["chat", "reasoning", "summarization", "structured_output"],
    quantizations: ["Q3_K_M", "Q4_K_M", "Q5_K_M", "Q6_K"],
    files: {
      Q3_K_M: quantFile("Qwen2.5-7B-Instruct-Q3_K_M.gguf", { sizeGB: 3.8 }),
      Q4_K_M: quantFile("Qwen2.5-7B-Instruct-Q4_K_M.gguf", { sizeGB: 4.7 }),
      Q5_K_M: quantFile("Qwen2.5-7B-Instruct-Q5_K_M.gguf", { sizeGB: 5.4 }),
      Q6_K: quantFile("Qwen2.5-7B-Instruct-Q6_K.gguf", { sizeGB: 6.3 }),
    },
  },
  {
    id: "phi-3.5-mini-instruct-gguf",
    label: "Phi 3.5 Mini Instruct",
    family: "phi",
    hfRepo: "bartowski/Phi-3.5-mini-instruct-GGUF",
    license: "mit",
    paramsB: 3.8,
    totalLayers: 32,
    hiddenSize: 3072,
    defaultContext: 4096,
    maxContext: 131072,
    tasks: ["chat", "structured_output", "summarization", "reasoning"],
    quantizations: ["Q3_K_M", "Q4_K_M", "Q5_K_M", "Q6_K"],
    files: {
      Q3_K_M: quantFile("Phi-3.5-mini-instruct-Q3_K_M.gguf", { sizeGB: 2.0 }),
      Q4_K_M: quantFile("Phi-3.5-mini-instruct-Q4_K_M.gguf", { sizeGB: 2.4 }),
      Q5_K_M: quantFile("Phi-3.5-mini-instruct-Q5_K_M.gguf", { sizeGB: 2.8 }),
      Q6_K: quantFile("Phi-3.5-mini-instruct-Q6_K.gguf", { sizeGB: 3.1 }),
    },
  },
  {
    id: "gemma-2-2b-it-gguf",
    label: "Gemma 2 2B Instruct",
    family: "gemma",
    hfRepo: "bartowski/gemma-2-2b-it-GGUF",
    license: "gemma",
    paramsB: 2.6,
    totalLayers: 26,
    hiddenSize: 2304,
    defaultContext: 4096,
    maxContext: 8192,
    tasks: ["chat", "summarization", "structured_output"],
    quantizations: ["Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"],
    files: {
      Q4_K_M: quantFile("gemma-2-2b-it-Q4_K_M.gguf", { sizeGB: 1.7 }),
      Q5_K_M: quantFile("gemma-2-2b-it-Q5_K_M.gguf", { sizeGB: 1.9 }),
      Q6_K: quantFile("gemma-2-2b-it-Q6_K.gguf", { sizeGB: 2.2 }),
      Q8_0: quantFile("gemma-2-2b-it-Q8_0.gguf", { sizeGB: 2.8 }),
    },
  },
  {
    id: "qwen2.5-1.5b-instruct-gguf",
    label: "Qwen2.5 1.5B Instruct",
    family: "qwen",
    hfRepo: "bartowski/Qwen2.5-1.5B-Instruct-GGUF",
    license: "apache-2.0",
    paramsB: 1.5,
    totalLayers: 28,
    hiddenSize: 1536,
    defaultContext: 8192,
    maxContext: 32768,
    tasks: ["chat", "coding", "summarization", "structured_output"],
    quantizations: ["Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"],
    files: {
      Q4_K_M: quantFile("Qwen2.5-1.5B-Instruct-Q4_K_M.gguf", { sizeGB: 1.1 }),
      Q5_K_M: quantFile("Qwen2.5-1.5B-Instruct-Q5_K_M.gguf", { sizeGB: 1.3 }),
      Q6_K: quantFile("Qwen2.5-1.5B-Instruct-Q6_K.gguf", { sizeGB: 1.4 }),
      Q8_0: quantFile("Qwen2.5-1.5B-Instruct-Q8_0.gguf", { sizeGB: 1.7 }),
    },
  },
];

export function listAllowlistedHfModels() {
  return HF_GGUF_ALLOWLIST.map((model) => ({ ...model }));
}

// Normalize a GGUF `general.architecture` value to the family label the rest of
// the pipeline uses (estimator, native runner, chat-template selection).
export function familyFromArchitecture(architecture) {
  const arch = String(architecture ?? "").toLowerCase();
  if (!arch) return "unknown";
  if (/gpt[-_]?oss|gptoss/.test(arch)) return "gpt-oss";
  if (arch.includes("kimi")) return "kimi";
  if (arch.includes("gemma3n")) return "gemma3n";
  if (arch.includes("gemma")) return arch.includes("gemma4") || arch.includes("gemma-4") ? "gemma4" : "gemma";
  if (arch.includes("qwen3vl") || arch.includes("qwen3-vl")) return "qwen3vl";
  if (arch.includes("qwen2vl") || arch.includes("qwen2-vl")) return "qwen2vl";
  if (arch.includes("qwen3")) return "qwen3";
  // Keep the catalog's stable public family label. Native compatibility still
  // receives the exact GGUF architecture and can distinguish qwen2/qwen3/VL.
  if (arch.includes("qwen")) return "qwen";
  if (arch.includes("llama")) return "llama";
  if (arch.includes("mistral") || arch.includes("mixtral")) return "mistral";
  if (arch.includes("phi")) return "phi";
  if (arch.includes("mamba")) return "mamba";
  if (arch.includes("rwkv")) return "rwkv";
  if (arch.includes("bert")) return "bert";
  if (arch.includes("command-r") || arch.includes("cohere")) return "cohere2";
  // Strip a trailing version digit block (Qwen handled above; keeps e.g.
  // "falcon", "starcoder" intact) so unknown-but-consistent families still group.
  return arch.replace(/[0-9._-]+$/, "") || arch;
}

// Build an allowlist-SHAPED model object from a downloaded catalog entry's
// parsed GGUF metadata, so a live-discovered (non-allowlist) Hugging Face model
// can flow through the SAME buildHfRecommendation / benchmark / serve pipeline
// as a curated entry. Every field the pipeline reads (paramsB, totalLayers,
// hiddenSize, defaultContext, maxContext, family, tasks, quantizations, label)
// is derived from the GGUF header the download step already parsed. Returns
// null when the entry lacks the minimum GGUF shape needed to plan a run.
export function syntheticHfModelFromGguf(entry) {
  if (!entry) return null;
  const gguf = entry.gguf ?? {};
  const totalLayers = Number.isFinite(gguf.blockCount) ? gguf.blockCount : null;
  const hiddenSize = Number.isFinite(gguf.embeddingLength) ? gguf.embeddingLength : null;
  if (!totalLayers || !hiddenSize) {
    // Without a real layer/hidden shape the footprint/gpu-layer math is
    // meaningless; let the caller decide how to degrade.
    return null;
  }
  const contextLength = Number.isFinite(gguf.contextLength) && gguf.contextLength > 0 ? gguf.contextLength : 8192;
  const paramsB = Number(entry.paramsB) || Number(gguf.parameterEstimateB) || null;
  const quant = gguf.quant ?? entry.quant ?? "Q4_K_M";
  return {
    id: entry.id,
    label: entry.id,
    family: familyFromArchitecture(gguf.architecture),
    hfRepo: entry.hfRepo,
    license: entry.license ?? "unknown",
    paramsB: paramsB ?? 7,
    totalLayers,
    hiddenSize,
    defaultContext: Math.min(contextLength, 8192),
    maxContext: contextLength,
    tasks: ["chat"],
    quantizations: [quant],
    files: {},
    synthetic: true,
  };
}

export function findAllowlistedHfModel(idOrRepo) {
  if (!idOrRepo) {
    return null;
  }
  const needle = String(idOrRepo).toLowerCase();
  return (
    HF_GGUF_ALLOWLIST.find((model) => model.id.toLowerCase() === needle) ??
    HF_GGUF_ALLOWLIST.find((model) => model.hfRepo.toLowerCase() === needle) ??
    null
  );
}

export function isAllowlistedRepo(repo) {
  if (!repo) {
    return false;
  }
  const needle = String(repo).toLowerCase();
  return HF_GGUF_ALLOWLIST.some((model) => model.hfRepo.toLowerCase() === needle);
}

export function findAllowlistedFile(repo, filename) {
  const model = HF_GGUF_ALLOWLIST.find((entry) => entry.hfRepo.toLowerCase() === String(repo).toLowerCase());
  if (!model) {
    return null;
  }
  const target = String(filename).toLowerCase();
  for (const [quant, file] of Object.entries(model.files)) {
    if (file.filename.toLowerCase() === target) {
      return { model, quant, file };
    }
  }
  return null;
}
