// UniversalModel (model wrapper Phase 1, spec §2/§9).
//
// One normalized description of a model regardless of where it came from -
// the curated registry (src/data/models.json), the local catalog
// (~/.lizard-llm/model-catalog.json), or an installed Ollama model - plus a
// per-runtime verdict computed by the SAME deciders the runtimes actually use
// (evaluateNativeGgufCompatibility), never by parallel rules.
//
// Design constraints, from the Phase 0 gap analysis:
// - The registry stays the source of truth for curated facts (family,
//   parameter count, context, Ollama tag linkage).
// - The catalog stays the source of truth for what is on disk (paths, parsed
//   GGUF metadata).
// - This module only NORMALIZES and MERGES; it never writes state and never
//   invents a verdict it cannot back with metadata. A model whose GGUF has
//   not been inspected gets native status "unknown", not a guess.

import { evaluateNativeGgufCompatibility } from "../runtime/compatibility/index.js";

// MoE-aware sizing (spec s12): Parameter Count != Actual Runtime Cost. For a
// sparse MoE model only expert_used_count of expert_count experts run per
// token, so active parameters (what determines per-token memory bandwidth) are
// far below total. Dense models have active == total. Returns null fields when
// the header does not carry the numbers rather than fabricating an estimate.
export function deriveParameterSizing(gguf, totalParameterCountB = null) {
  const total = Number.isFinite(totalParameterCountB)
    ? totalParameterCountB
    : (Number.isFinite(gguf?.parameterEstimateB) ? gguf.parameterEstimateB : null);
  const expertCount = Number(gguf?.expertCount ?? 0);
  const expertUsed = Number(gguf?.expertUsedCount ?? 0);
  const isMoe = Number.isFinite(expertCount) && expertCount > 1;

  if (!isMoe || total == null || !Number.isFinite(expertUsed) || expertUsed < 1) {
    return { totalParameterCountB: total, activeParameterCountB: total, isMoe };
  }
  // Approximation: attention + shared layers run every token; only the MoE FFN
  // experts are gated. Without a per-tensor breakdown the honest, widely-used
  // estimate is total scaled by the used/total expert ratio, floored at the
  // dense backbone - labelled as an estimate by returning `activeIsEstimate`.
  const ratio = expertUsed / expertCount;
  const active = Number((total * ratio).toFixed(2));
  return {
    totalParameterCountB: total,
    activeParameterCountB: active,
    isMoe: true,
    expertCount,
    expertUsedCount: expertUsed,
    activeIsEstimate: true,
  };
}

// Model capabilities (spec s10): DETECTED from GGUF metadata, not guessed.
// Kept separate from RUNTIME capability - a model may support tool calling
// while a given runtime does not, and the UI only enables a feature when both
// agree. Heuristic sources are labelled as such: tool-calling is inferred from
// the chat template mentioning tools, so it is a strong hint, not a contract.
export function deriveModelCapabilities(gguf, hints = {}) {
  const caps = {
    text: true,
    chat: false,
    toolCalling: false,
    vision: false,
    embeddings: false,
    moe: false,
    longContext: false,
    reasoning: false,
  };
  if (!gguf) return caps;

  const template = String(gguf.chatTemplate ?? "").toLowerCase();
  caps.chat = Boolean(gguf.chatTemplate);
  // Tool calling: the chat template renders tool/function scaffolding. This is
  // how llama.cpp itself decides a model is tool-capable, so it is the honest
  // signal available from the GGUF alone.
  caps.toolCalling = /tool|function|tool_call|<\|python_tag\|>/.test(template);
  // Reasoning: the template carries an explicit thinking channel.
  caps.reasoning = /<think>|<\|thinking\|>|reasoning|analysis<\|/.test(template)
    || Boolean(hints.reasoning);

  // MoE: the header reports experts.
  const expertCount = Number(gguf.expertCount ?? gguf.raw?.["llama.expert_count"] ?? 0);
  caps.moe = Number.isFinite(expertCount) && expertCount > 1;

  // Long context: 32k+ trained context.
  const ctx = Number(gguf.contextLength ?? 0);
  caps.longContext = Number.isFinite(ctx) && ctx >= 32768;

  // Vision: a multimodal projector is present (mmproj/clip tensors), or the
  // architecture is a known vision family.
  const tensorNames = Array.isArray(gguf.tensorsPreview)
    ? gguf.tensorsPreview.map((t) => String(t?.name ?? t).toLowerCase())
    : [];
  const arch = String(gguf.architecture ?? "").toLowerCase();
  caps.vision = tensorNames.some((n) => n.includes("mmproj") || n.includes("v.blk") || n.includes("clip"))
    || /vl|vision|llava|pixtral/.test(arch)
    || Boolean(hints.vision);

  // Embeddings: pooling-type metadata or an embedding-only architecture.
  caps.embeddings = Boolean(gguf.raw?.["llama.pooling_type"])
    || /bert|embed|gte|bge|e5/.test(arch)
    || Boolean(hints.embeddings);

  return caps;
}

export const RUNTIME_STATUS = Object.freeze({
  SUPPORTED: "supported",
  BLOCKED: "blocked",
  PULLABLE: "pullable",
  IMPORTABLE: "importable",
  UNKNOWN: "unknown",
});

function normalizeTag(tag) {
  const value = String(tag ?? "").trim().toLowerCase();
  if (!value) return null;
  // "llama3.2:3b" and "llama3.2:3b" with an implicit :latest must compare equal.
  return value.includes(":") ? value : `${value}:latest`;
}

function normalizeId(id) {
  return String(id ?? "").trim().toLowerCase() || null;
}

export function normalizeRegistryModel(model) {
  const tags = new Set(
    [model.ollamaTag, ...(Array.isArray(model.ollamaTags) ? model.ollamaTags : [])]
      .map(normalizeTag)
      .filter(Boolean),
  );
  return {
    id: model.id,
    name: model.id,
    family: model.family ?? null,
    architecture: null,
    parameterCountB: Number.isFinite(model.paramsB) ? model.paramsB : null,
    quantization: null,
    format: "gguf",
    contextLength: model.maxContext ?? model.defaultContext ?? null,
    license: model.license ?? null,
    tasks: Array.isArray(model.tasks) ? [...model.tasks] : [],
    quantizations: Array.isArray(model.quantizations) ? [...model.quantizations] : [],
    ollamaTag: model.ollamaTag ? normalizeTag(model.ollamaTag) : null,
    ollamaTags: [...tags],
    localPath: null,
    gguf: null,
    ollamaInstalled: false,
    sources: [{ kind: "registry", id: model.id }],
  };
}

export function normalizeCatalogModel(entry) {
  const gguf = entry.gguf ?? null;
  return {
    id: entry.id,
    name: entry.id,
    family: gguf?.architecture ?? null,
    architecture: gguf?.architecture ?? null,
    parameterCountB: Number.isFinite(gguf?.paramsB) ? gguf.paramsB : null,
    quantization: gguf?.quant ?? entry.quant ?? null,
    format: "gguf",
    contextLength: gguf?.contextLength ?? null,
    license: entry.license ?? null,
    tasks: [],
    quantizations: [],
    ollamaTag: entry.ollamaTag ? normalizeTag(entry.ollamaTag) : null,
    ollamaTags: entry.ollamaTag ? [normalizeTag(entry.ollamaTag)] : [],
    localPath: entry.importedPath ?? entry.path ?? null,
    gguf,
    ollamaInstalled: false,
    sources: [{
      kind: "local",
      id: entry.id,
      origin: entry.source ?? null,
      hfRepo: entry.hfRepo ?? null,
    }],
  };
}

// Safetensors detection (model wrapper Phase 7, spec s3). A Hugging Face repo
// (or local dir) that ships .safetensors but no GGUF is a real model the app
// should represent - and route honestly. It cannot run on the native GGUF
// worker; where the architecture is one Ollama can import, that is the path.
export function detectSafetensors(files = []) {
  const names = files.map((f) => String(f?.rfilename ?? f?.path ?? f?.filename ?? f).toLowerCase());
  const hasSafetensors = names.some((n) => n.endsWith(".safetensors"));
  const hasGguf = names.some((n) => n.endsWith(".gguf"));
  return { hasSafetensors, hasGguf, safetensorsOnly: hasSafetensors && !hasGguf };
}

export function normalizeSafetensorsModel(repo) {
  const arch = String(repo.architecture ?? "").toLowerCase();
  // Ollama can import Safetensors for a subset of architectures (llama,
  // mistral, gemma, qwen2, phi3 and a few more) via its create-from-safetensors
  // path. This is an availability hint, not a guarantee for a specific repo.
  const ollamaImportable = /llama|mistral|gemma|qwen2|qwen3|phi3|phi/.test(arch);
  return {
    id: repo.id ?? repo.repo ?? repo.name,
    name: repo.name ?? repo.id ?? repo.repo,
    family: repo.family ?? repo.architecture ?? null,
    architecture: repo.architecture ?? null,
    parameterCountB: Number.isFinite(repo.paramsB) ? repo.paramsB : null,
    quantization: null,
    format: "safetensors",
    contextLength: repo.contextLength ?? null,
    license: repo.license ?? null,
    tasks: [],
    quantizations: [],
    ollamaTag: null,
    ollamaTags: [],
    localPath: repo.localPath ?? null,
    gguf: null,
    ollamaInstalled: false,
    safetensors: { safetensorsOnly: true, ollamaImportable },
    sources: [{ kind: "safetensors", id: repo.id ?? repo.repo, hfRepo: repo.repo ?? repo.id ?? null }],
  };
}

export function normalizeOllamaModel(model) {
  const tag = normalizeTag(model.tag);
  return {
    id: tag ?? model.tag,
    name: model.tag,
    family: null,
    architecture: null,
    parameterCountB: null,
    quantization: null,
    format: "ollama",
    contextLength: null,
    license: null,
    tasks: [],
    quantizations: [],
    ollamaTag: tag,
    ollamaTags: tag ? [tag] : [],
    localPath: null,
    gguf: null,
    ollamaInstalled: true,
    sources: [{ kind: "ollama", tag: model.tag, sizeLabel: model.size ?? null }],
  };
}

// Merge two normalized entries describing the same logical model. Curated
// facts win where present; on-disk facts (paths, GGUF metadata) always
// survive; sources accumulate so the caller can see every origin.
function mergeModels(base, extra) {
  const tags = new Set([...(base.ollamaTags ?? []), ...(extra.ollamaTags ?? [])]);
  return {
    ...base,
    name: base.sources.some((s) => s.kind === "registry") ? base.name : extra.name ?? base.name,
    family: base.family ?? extra.family,
    architecture: base.architecture ?? extra.architecture,
    parameterCountB: base.parameterCountB ?? extra.parameterCountB,
    quantization: base.quantization ?? extra.quantization,
    format: base.format === "ollama" ? extra.format ?? base.format : base.format,
    contextLength: base.contextLength ?? extra.contextLength,
    license: base.license ?? extra.license,
    tasks: base.tasks.length ? base.tasks : extra.tasks,
    quantizations: base.quantizations.length ? base.quantizations : extra.quantizations,
    ollamaTag: base.ollamaTag ?? extra.ollamaTag,
    ollamaTags: [...tags],
    localPath: base.localPath ?? extra.localPath,
    gguf: base.gguf ?? extra.gguf,
    ollamaInstalled: base.ollamaInstalled || extra.ollamaInstalled,
    safetensors: base.safetensors ?? extra.safetensors,
    sources: [...base.sources, ...extra.sources],
  };
}

// Dedup (spec §9.8: "The user should see one model, not five duplicate
// entries"). Merge preference order:
//   1. shared Ollama tag linkage (the curated registry and the catalog both
//      record the tag an import came from), then
//   2. exact id equality.
export function dedupeUniversalModels(models) {
  const merged = [];
  const byTag = new Map();
  const byId = new Map();

  for (const model of models) {
    let target = null;
    for (const tag of model.ollamaTags ?? []) {
      if (byTag.has(tag)) {
        target = byTag.get(tag);
        break;
      }
    }
    if (!target) {
      const id = normalizeId(model.id);
      if (id && byId.has(id)) target = byId.get(id);
    }

    if (target) {
      const index = merged.indexOf(target);
      const combined = mergeModels(target, model);
      merged[index] = combined;
      for (const tag of combined.ollamaTags ?? []) byTag.set(tag, combined);
      const id = normalizeId(combined.id);
      if (id) byId.set(id, combined);
      // Re-point every alias that referenced the old object.
      for (const [key, value] of byTag) if (value === target) byTag.set(key, combined);
      for (const [key, value] of byId) if (value === target) byId.set(key, combined);
    } else {
      merged.push(model);
      for (const tag of model.ollamaTags ?? []) byTag.set(tag, model);
      const id = normalizeId(model.id);
      if (id) byId.set(id, model);
    }
  }
  return merged;
}

// Per-runtime verdicts for a merged model. Honest by construction:
// - native: only claimed when parsed GGUF metadata says so, via the same
//   evaluateNativeGgufCompatibility the runtime itself consults. An installed
//   Ollama model whose blob has not been inspected is "importable", not
//   "supported" (Phase 2 adds the blob probe).
// - llama.cpp: any local GGUF file.
// - Ollama: installed tag = supported; tag known from the registry but not
//   installed = pullable.
export function computeRuntimeVerdicts(model) {
  const runtimes = [];

  // Safetensors: the native GGUF worker cannot execute it, and there is no
  // local Safetensors runtime. Honest routing: Ollama import where the
  // architecture allows, otherwise convert to GGUF first.
  if (model.format === "safetensors" && !model.gguf) {
    runtimes.push({
      id: "native",
      status: RUNTIME_STATUS.BLOCKED,
      reason: "Safetensors weights cannot run on the native GGUF worker. Convert to GGUF, or import through Ollama if the architecture is supported.",
    });
    runtimes.push({
      id: "llama.cpp",
      status: RUNTIME_STATUS.BLOCKED,
      reason: "llama.cpp runs GGUF. This repo ships Safetensors only - convert it to GGUF first.",
    });
    runtimes.push(model.safetensors?.ollamaImportable
      ? { id: "ollama", status: RUNTIME_STATUS.IMPORTABLE, reason: "Ollama can import this architecture from Safetensors." }
      : { id: "ollama", status: RUNTIME_STATUS.BLOCKED, reason: "Ollama cannot import this Safetensors architecture." });
    return runtimes;
  }

  if (model.gguf) {
    const verdict = evaluateNativeGgufCompatibility({
      gguf: model.gguf,
      modelArtifact: { modelId: model.id, gguf: model.gguf },
    });
    if (!verdict.supported) {
      runtimes.push({ id: "native", status: RUNTIME_STATUS.BLOCKED, reason: verdict.reason });
    } else if (model.localPath) {
      runtimes.push({ id: "native", status: RUNTIME_STATUS.SUPPORTED, reason: null });
    } else {
      // Compatible metadata but no loadable file yet (a probed Ollama blob):
      // honest status is one-click-importable, not "supported".
      runtimes.push({
        id: "native",
        status: RUNTIME_STATUS.IMPORTABLE,
        reason: "Compatible with the native runtime. Import it to enable native execution - no re-download needed.",
      });
    }
  } else if (model.ollamaInstalled) {
    runtimes.push({
      id: "native",
      status: RUNTIME_STATUS.IMPORTABLE,
      reason: "Installed through Ollama; import it to inspect the underlying GGUF and unlock native execution.",
    });
  } else {
    runtimes.push({
      id: "native",
      status: RUNTIME_STATUS.UNKNOWN,
      reason: "No local GGUF metadata to inspect yet.",
    });
  }

  runtimes.push(model.localPath
    ? { id: "llama.cpp", status: RUNTIME_STATUS.SUPPORTED, reason: null }
    : { id: "llama.cpp", status: RUNTIME_STATUS.UNKNOWN, reason: "No local GGUF file yet." });

  if (model.ollamaInstalled) {
    runtimes.push({ id: "ollama", status: RUNTIME_STATUS.SUPPORTED, reason: null });
  } else if (model.ollamaTag) {
    runtimes.push({
      id: "ollama",
      status: RUNTIME_STATUS.PULLABLE,
      reason: `Not installed; available as "${model.ollamaTag}" via ollama pull.`,
    });
  } else {
    runtimes.push({ id: "ollama", status: RUNTIME_STATUS.UNKNOWN, reason: "No Ollama tag is known for this model." });
  }

  return runtimes;
}

function runtimeVerdict(model, id) {
  return (model.runtimes ?? []).find((runtime) => runtime.id === id) ?? null;
}

function alternativeScore(selected, candidate) {
  let score = 0;
  if (candidate.localPath) score += 50;
  if (runtimeVerdict(candidate, "native")?.status === RUNTIME_STATUS.SUPPORTED) score += 40;
  if (selected.capabilities?.chat && candidate.capabilities?.chat) score += 12;
  if (selected.capabilities?.reasoning && candidate.capabilities?.reasoning) score += 8;
  if (selected.capabilities?.toolCalling && candidate.capabilities?.toolCalling) score += 8;
  if (selected.capabilities?.longContext && candidate.capabilities?.longContext) score += 5;
  const selectedSize = Number(selected.activeParameterCountB ?? selected.parameterCountB);
  const candidateSize = Number(candidate.activeParameterCountB ?? candidate.parameterCountB);
  if (Number.isFinite(selectedSize) && Number.isFinite(candidateSize)) {
    score += Math.max(0, 20 - Math.abs(selectedSize - candidateSize) * 2);
  }
  if (selected.family && candidate.family === selected.family) score += 10;
  return score;
}

// Every blocked/unknown model carries a machine-readable route and a ranked
// list containing ONLY models the native compatibility evaluator accepted.
// Newly released families therefore degrade into an explained recommendation
// instead of an opaque worker-load failure. The selected model remains visible
// so the UI can explain why it is unavailable; incompatible entries are never
// offered as alternatives.
export function attachCompatibleAlternatives(models, { limit = 3 } = {}) {
  const candidates = models.filter((model) => {
    const status = runtimeVerdict(model, "native")?.status;
    return status === RUNTIME_STATUS.SUPPORTED || status === RUNTIME_STATUS.IMPORTABLE;
  });
  return models.map((model) => {
    const native = runtimeVerdict(model, "native");
    if (native?.status === RUNTIME_STATUS.SUPPORTED) {
      return {
        ...model,
        compatibilityRouting: {
          available: true,
          selectedRuntime: "native",
          reason: null,
          alternatives: [],
        },
      };
    }
    const fallback = ["llama.cpp", "ollama"]
      .map((id) => runtimeVerdict(model, id))
      .find((runtime) => [
        RUNTIME_STATUS.SUPPORTED,
        RUNTIME_STATUS.IMPORTABLE,
        RUNTIME_STATUS.PULLABLE,
      ].includes(runtime?.status)) ?? null;
    const alternatives = candidates
      .filter((candidate) => candidate.id !== model.id)
      .map((candidate) => ({ candidate, score: alternativeScore(model, candidate) }))
      .sort((left, right) => right.score - left.score || String(left.candidate.name).localeCompare(String(right.candidate.name)))
      .slice(0, Math.max(1, Math.min(Number(limit) || 3, 10)))
      .map(({ candidate }) => ({
        id: candidate.id,
        name: candidate.name,
        family: candidate.family,
        parameterCountB: candidate.parameterCountB,
        activeParameterCountB: candidate.activeParameterCountB,
        local: Boolean(candidate.localPath),
        nativeStatus: runtimeVerdict(candidate, "native")?.status ?? null,
        reason: candidate.localPath
          ? "Already installed and compatible with the native runtime."
          : "Compatible with the native runtime after import/download.",
      }));
    return {
      ...model,
      compatibilityRouting: {
        available: Boolean(fallback),
        selectedRuntime: fallback?.id ?? null,
        selectedRuntimeStatus: fallback?.status ?? null,
        reason: native?.reason ?? "The GGUF family or quantization is not supported by the native runtime.",
        alternatives,
      },
    };
  });
}

// Assemble the unified list from the three raw source listings. Pure -
// callers (model-sources.js, the UI server) supply the data.
export function buildUniversalModelList({ registryModels = [], catalogModels = [], ollamaModels = [], safetensorsModels = [] } = {}) {
  const normalized = [
    ...registryModels.map(normalizeRegistryModel),
    ...catalogModels.map(normalizeCatalogModel),
    ...ollamaModels.map(normalizeOllamaModel),
    ...safetensorsModels.map(normalizeSafetensorsModel),
  ];
  const models = dedupeUniversalModels(normalized).map((model) => ({
    ...model,
    ...deriveParameterSizing(model.gguf, model.parameterCountB),
    capabilities: deriveModelCapabilities(model.gguf, {
      reasoning: /reasoning|r1|think|qwq/i.test(String(model.id)),
      vision: /vl|vision|llava|pixtral/i.test(String(model.id)),
      embeddings: /embed|bge|gte|e5/i.test(String(model.id)),
    }),
    runtimes: computeRuntimeVerdicts(model),
  }));
  return attachCompatibleAlternatives(models);
}
