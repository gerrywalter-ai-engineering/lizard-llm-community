import fs from "node:fs";
import crypto from "node:crypto";
import { scanHardware } from "./hardware.js";
import {
  buildValidatedCapabilityRegistry,
  saveBackendCapabilitySnapshot,
  selectBackendForProfile,
} from "./backend-capability-registry.js";
import { classifyPrompt } from "./prompt-classifier.js";
import { buildRecommendation, recommendedCpuThreads } from "./recommendation.js";
import { executionModes } from "./profiles.js";
import { estimateConfiguration, estimateGpuLayers } from "./estimator.js";
import {
  ensureModelForRecommendation,
  resolveExplicitModel,
} from "./model-sources.js";
import { findModel, findModelByOllamaTag } from "./model-registry.js";
import { getActiveModel, findCatalogModel } from "./download-manager.js";
import { findAllowlistedHfModel, syntheticHfModelFromGguf } from "./hf-model-catalog.js";
import { deriveLoadingPolicy, inspectGgufFile } from "./gguf-metadata.js";
import { verifyGgufMagic } from "./ollama.js";

const DEFAULT_BATCH_SIZE = 512;
const DEFAULT_UBATCH_SIZE = 128;
const DEFAULT_TIMEOUT_MS = 120000;

function parseInteger(value, fallback) {
  if (value == null || value === "") {
    return fallback;
  }
  const parsed = Number.parseInt(value, 10);
  return Number.isFinite(parsed) ? parsed : fallback;
}

export function resolveProfileThreads({ overrides = {}, selected = {}, gguf = null, hardware } = {}) {
  const resolvedQuant = gguf?.quant ?? selected.quant ?? null;
  return parseInteger(overrides.threads, recommendedCpuThreads({
    hardware,
    model: {
      family: selected.family,
      totalLayers: selected.totalLayers,
      hiddenSize: selected.hiddenSize,
    },
    quant: resolvedQuant,
  }));
}

function normalizeBooleanFlag(value) {
  return value === true || value === "true" || value === "1";
}

function resolveRequestedModelId(optionValue, registry) {
  if (!optionValue || optionValue === "auto") {
    return undefined;
  }
  if (!String(optionValue).includes(":")) {
    return findModel(registry, optionValue)?.id ?? optionValue;
  }
  return findModelByOllamaTag(registry, optionValue)?.id;
}

function resolveActiveModelOption(modelOption, config) {
  if (modelOption !== "active") {
    return modelOption;
  }
  const active = getActiveModel(config);
  if (!active) {
    throw new Error("No active model is set. Run 'lizard-llm models activate --model ID' or 'lizard-llm onboard' first.");
  }
  return active.id;
}

export function buildHfRecommendation({ hfModel, quant, hardware, config, task, mode = "balanced" }) {
  const executionMode = executionModes[mode] ?? executionModes.balanced;
  const kvCache = executionMode.preferredKvCache;
  const contextTokens = Math.min(hfModel.defaultContext, executionMode.contextCap, hfModel.maxContext);
  const model = {
    id: hfModel.id,
    family: hfModel.family,
    paramsB: hfModel.paramsB,
    totalLayers: hfModel.totalLayers,
    hiddenSize: hfModel.hiddenSize,
    defaultContext: hfModel.defaultContext,
    maxContext: hfModel.maxContext,
    tasks: hfModel.tasks,
  };
  const gpuLayers = estimateGpuLayers({ model, quant, contextTokens, kvCache, hardware, config });
  const estimate = estimateConfiguration({ model, quant, contextTokens, gpuLayers, kvCache, hardware, config });

  return {
    hardwareSummary: {
      ramAvailableGB: hardware.memory.availableGB,
      gpu: hardware.primaryGpu?.name ?? "CPU only",
      vramGB: hardware.primaryGpu?.vramGB ?? 0,
    },
    selected: {
      modelId: hfModel.id,
      family: hfModel.family,
      paramsB: hfModel.paramsB,
      hiddenSize: hfModel.hiddenSize,
      quant,
      kvCache,
      contextTokens,
      gpuLayers,
      totalLayers: hfModel.totalLayers,
      threads: recommendedCpuThreads({ hardware, model, quant }),
      mode,
      estimate,
      elasticWarning: false,
    },
    reasons: [
      `${hfModel.label} (${hfModel.paramsB}B) ${quant} planned for ${mode} mode`,
      gpuLayers > 0
        ? `${gpuLayers}/${hfModel.totalLayers} layers can offload to ${hardware.primaryGpu?.name}`
        : "CPU-first plan (no usable GPU VRAM detected)",
    ],
    alternatives: [],
  };
}

function maybeResolveHfRunContext({ requestedModel, config, hardware, mode, taskProfile }) {
  if (!requestedModel || requestedModel === "auto" || String(requestedModel).includes(":")) {
    return null;
  }
  const entry = findCatalogModel(config, requestedModel);
  if (!entry) {
    return null;
  }
  // Prefer the curated allowlist; otherwise reconstruct the planning shape from
  // the downloaded GGUF header, so any model already on disk resolves for
  // chat/serve/run exactly like a curated one. Returns null only when the GGUF
  // lacked the layer/embedding shape needed to size a run.
  //
  // This used to require entry.source === "huggingface", which is provenance,
  // not capability: a model imported through Ollama carries the same GGUF
  // header and plans identically, but was rejected here and fell through to
  // the static registry - where anything not in the seven curated entries died
  // as "Unknown model: <id>" with the file sitting on disk. gemma3-4b is
  // catalogued as source "ollama" and hit exactly that.
  const hfModel = findAllowlistedHfModel(entry.hfRepo) ?? syntheticHfModelFromGguf(entry);
  if (!hfModel) {
    return null;
  }
  const quant = entry.gguf?.quant ?? entry.quant ?? hfModel.quantizations[0];
  const recommendation = buildHfRecommendation({
    hfModel,
    quant,
    hardware,
    config,
    task: taskProfile?.taskType,
    mode: mode ?? config.defaultMode,
  });
  return {
    recommendation,
    modelArtifact: {
      modelId: entry.id,
      modelPath: entry.importedPath,
      source: entry.source,
      ollamaTag: entry.ollamaTag ?? null,
      sha256: entry.sha256 ?? null,
      gguf: entry.gguf ?? null,
    },
    catalogEntry: entry,
  };
}

function collectUserOverrides(options = {}) {
  const overrides = {};
  const mappings = [
    ["ctx-size", "contextSize"],
    ["threads", "threads"],
    ["gpu-layers", "gpuLayers"],
    ["batch-size", "batchSize"],
    ["ubatch-size", "ubatchSize"],
    ["backend", "backend"],
    ["timeout", "timeoutMs"],
  ];
  for (const [optionKey, field] of mappings) {
    if (options[optionKey] != null && options[optionKey] !== false) {
      overrides[field] = options[optionKey];
    }
  }
  if (options["no-mmap"]) {
    overrides.mmap = false;
  }
  if (options.mlock) {
    overrides.mlock = true;
  }
  if (options["advanced-runtime"]) {
    overrides.advancedRuntime = true;
  }
  return overrides;
}

function validateProfile({ profile, modelArtifact, promptRequired, capabilityRegistry }) {
  if (profile.engine !== "ollama" && (!modelArtifact?.modelPath || !fs.existsSync(modelArtifact.modelPath))) {
    throw new Error(`Resolved model path does not exist: ${modelArtifact?.modelPath ?? "unknown"}`);
  }
  if (profile.engine !== "ollama" && !verifyGgufMagic(modelArtifact.modelPath)) {
    throw new Error(`Resolved model path is not a verified GGUF artifact: ${modelArtifact.modelPath}`);
  }
  if (capabilityRegistry?.backends?.[profile.backend]?.supported === false) {
    throw new Error(`Selected backend ${profile.backend} is not supported on this machine.`);
  }
  if (profile.backend !== "cpu" && profile.backend !== "unknown" && profile.gpuLayers !== "auto" && Number(profile.gpuLayers) <= 0) {
    throw new Error(`Backend ${profile.backend} requires GPU layers greater than 0.`);
  }
  if (profile.gpuLayers !== "auto" && Number(profile.gpuLayers) < 0) {
    throw new Error(`Invalid GPU layer override: ${profile.gpuLayers}`);
  }
  if (promptRequired && !profile.prompt) {
    throw new Error("Benchmark prompt is required for this execution profile.");
  }
}

function profileReason({ profile, hardware, recommendation, historyAvailable, backendSelection }) {
  const reasons = [];
  const backendLabel = profile.backend === "cpu" ? "CPU" : String(profile.backend).toUpperCase();
  reasons.push(`${backendLabel} backend selected`);
  if (backendSelection?.reason) {
    reasons.push(backendSelection.reason);
  }
  reasons.push(`context ${profile.contextSize}`);
  reasons.push(`threads ${profile.threads}`);
  if (profile.gpuLayers === "auto") {
    reasons.push("GPU layers auto-tuned");
  } else {
    reasons.push(`GPU layers ${profile.gpuLayers}`);
  }
  if (hardware?.primaryGpu?.name && profile.backend !== "cpu") {
    reasons.push(`hardware ${hardware.primaryGpu.name}`);
  } else {
    reasons.push("no compatible GPU backend was detected");
  }
  if (historyAvailable) {
    reasons.push("previous benchmark history is available");
  }
  if (recommendation?.reasons?.[0]) {
    reasons.push(recommendation.reasons[0]);
  }
  if (profile?.gguf?.architecture) {
    reasons.push(`GGUF architecture ${profile.gguf.architecture}`);
  }
  if (profile?.loadingPolicy?.memoryRisk) {
    reasons.push(`memory risk ${profile.loadingPolicy.memoryRisk}`);
  }
  if (profile?.advancedRuntime) {
    reasons.push("advanced runtime metrics enabled");
  }
  return `Lizard selected ${reasons.join(", ")}.`;
}

function normalizeTimeoutMs(value) {
  const parsed = parseInteger(value, DEFAULT_TIMEOUT_MS);
  return parsed < 1000 ? parsed * 1000 : parsed;
}

function normalizeProfileMode(command) {
  if (command === "run") {
    return "server";
  }
  if (command === "benchmark-pro" || command === "pro") {
    return "pro";
  }
  return command;
}

function resolveCatalogEntry(config, modelArtifact) {
  if (!modelArtifact?.modelId) {
    return null;
  }
  return findCatalogModel(config, modelArtifact.modelId) ?? null;
}

function resolveGgufMetadata({ catalogEntry, modelArtifact, recommendation }) {
  if (catalogEntry?.gguf) {
    return catalogEntry.gguf;
  }
  if (modelArtifact?.gguf) {
    return modelArtifact.gguf;
  }
  if (!modelArtifact?.modelPath || !fs.existsSync(modelArtifact.modelPath)) {
    return null;
  }
  return inspectGgufFile(modelArtifact.modelPath, {
    fallbackQuant: recommendation?.selected?.quant ?? null,
    fallbackArchitecture: recommendation?.selected?.family ?? null,
    fallbackParamsB: recommendation?.selected?.paramsB ?? null,
  });
}

function buildExecutionProfile({
  command,
  options,
  config,
  hardware,
  recommendation,
  modelArtifact,
  catalogEntry,
  historyAvailable,
  capabilityRegistry,
}) {
  const overrides = collectUserOverrides(options);
  const selected = recommendation.selected ?? {};
  const gguf = resolveGgufMetadata({ catalogEntry, modelArtifact, recommendation });
  const mode = normalizeProfileMode(command);
  const loadingPolicy = deriveLoadingPolicy({
    gguf,
    hardware,
    estimate: selected.estimate ?? null,
    requestedContextSize: parseInteger(overrides.contextSize, selected.contextTokens ?? config.benchmark?.ci?.contextTokens ?? 4096),
    requestedGpuLayers: overrides.gpuLayers === "auto"
      ? "auto"
      : parseInteger(overrides.gpuLayers, selected.gpuLayers ?? 0),
    mode,
  });
  const backendSelection = selectBackendForProfile({
    capabilityRegistry,
    requestedBackend: overrides.backend ?? null,
    estimate: selected.estimate ?? null,
    mode,
  });
  const backend = backendSelection.backend;
  const resolvedContextSize = parseInteger(
    overrides.contextSize,
    loadingPolicy.recommendedContextSize ?? selected.contextTokens ?? config.benchmark?.ci?.contextTokens ?? 4096,
  );
  const resolvedGpuLayers = overrides.gpuLayers === "auto"
    ? "auto"
    : parseInteger(overrides.gpuLayers, loadingPolicy.recommendedGpuLayers ?? selected.gpuLayers ?? 0);
  const backendRequiresGpuLayers = backend !== "cpu" && backend !== "unknown";
  const zeroGpuLayers = resolvedGpuLayers !== "auto" && Number(resolvedGpuLayers) <= 0;
  const effectiveBackend = backendRequiresGpuLayers && zeroGpuLayers
    ? "cpu"
    : backend;
  const adjustedBackendWarnings = [...(backendSelection.warnings ?? [])];
  if (effectiveBackend !== backend) {
    adjustedBackendWarnings.push(
      `Backend ${backend} was downgraded to CPU because the resolved GPU layer count is ${resolvedGpuLayers}.`,
    );
  }
  const resolvedQuant = gguf?.quant ?? selected.quant ?? null;
  const resolvedThreads = resolveProfileThreads({ overrides, selected, gguf, hardware });
  const profile = {
    schemaVersion: 1,
    profileId: crypto.randomUUID(),
    modelId: modelArtifact.modelId ?? selected.modelId,
    modelPath: modelArtifact.modelPath ?? null,
    quant: resolvedQuant,
    kvCache: selected.kvCache ?? "q8_0",
    gguf,
    loadingPolicy,
    capabilityRegistry,
    engine: modelArtifact.ollamaTag && !modelArtifact.modelPath ? "ollama" : "llama.cpp",
    backend: effectiveBackend,
    mode,
    contextSize: resolvedContextSize,
    threads: resolvedThreads,
    batchSize: parseInteger(overrides.batchSize, config.execution?.batchSize ?? DEFAULT_BATCH_SIZE),
    ubatchSize: parseInteger(overrides.ubatchSize, config.execution?.ubatchSize ?? DEFAULT_UBATCH_SIZE),
    gpuLayers: resolvedGpuLayers,
    mmap: overrides.mmap ?? loadingPolicy.mmap,
    mlock: overrides.mlock ?? loadingPolicy.mlock,
    noConversationMode: true,
    timeoutMs: normalizeTimeoutMs(overrides.timeoutMs ?? config.benchmark?.timeoutSeconds ?? DEFAULT_TIMEOUT_MS),
    promptCache: mode === "server",
    serverMode: mode === "server" || mode === "pro",
    advancedRuntime: Boolean(overrides.advancedRuntime),
    source: {
      hardwareScan: true,
      ggufMetadata: Boolean(modelArtifact.modelPath),
      benchmarkHistory: historyAvailable,
      userOverride: Object.keys(overrides).length > 0,
    },
    reason: profileReason({
      profile: {
        contextSize: resolvedContextSize,
        threads: resolvedThreads,
        gpuLayers: resolvedGpuLayers,
        backend: effectiveBackend,
        gguf,
        loadingPolicy,
        capabilityRegistry,
        advancedRuntime: Boolean(overrides.advancedRuntime),
      },
      hardware,
      recommendation,
      historyAvailable,
      backendSelection,
    }),
    backendWarnings: adjustedBackendWarnings,
    memoryRisk: backendSelection.memoryRisk,
    prompt: options.prompt ? String(options.prompt) : null,
    maxTokens: parseInteger(options["max-tokens"], config.generation?.maxTokens ?? config.benchmark?.generatedTokens ?? null),
  };
  return profile;
}

export async function resolveExecutionContext({
  command,
  options,
  config,
  registry,
  logger,
  hardware: providedHardware,
  promptRequired = false,
}) {
  const hardware = providedHardware ?? scanHardware({ config, logger });
  const taskProfile = classifyPrompt({
    explicitTask: options.task,
    prompt: options.prompt,
  });
  const requestedModel = resolveActiveModelOption(options.model, config);
  // Community Edition keeps no benchmark history.
  const benchmarkHistory = [];
  const historyAvailable = benchmarkHistory.length > 0;
  // Native validation merges into the same snapshot (Phase 1): adapter
  // identity, authoritative UMA, live memory facts, CPU CPUID+OS features.
  // The probe is cached per process; when the compiled runner is missing
  // this equals the plain registry and CPU-only behavior is unchanged.
  const capabilityRegistry = buildValidatedCapabilityRegistry({
    hardware,
    config,
    benchmarkHistory,
    logger,
  });
  saveBackendCapabilitySnapshot(config, capabilityRegistry);

  const hfContext = maybeResolveHfRunContext({
    requestedModel,
    config,
    hardware,
    mode: options.mode,
    taskProfile,
  });

  const recommendation = hfContext?.recommendation ?? buildRecommendation({
    hardware,
    registry,
    config,
    requestedMode: options.mode,
    requestedModelId: resolveRequestedModelId(requestedModel, registry),
    taskProfile,
  });
  const modelArtifact = hfContext?.modelArtifact ?? (
    requestedModel && String(requestedModel).includes(":")
      ? await resolveExplicitModel({
        requestedModel,
        config,
        registry,
        logger,
        pullIfMissing: Boolean(options.pull),
      })
      : await ensureModelForRecommendation({
        recommendation,
        config,
        registry,
        logger,
        pullIfMissing: Boolean(options.pull),
      })
  );
  const catalogEntry = hfContext?.catalogEntry ?? resolveCatalogEntry(config, modelArtifact);

  const profile = buildExecutionProfile({
    command,
    options,
    config,
    hardware,
    recommendation,
    modelArtifact,
    catalogEntry,
    historyAvailable,
    capabilityRegistry,
  });
  validateProfile({
    profile,
    modelArtifact,
    promptRequired,
    capabilityRegistry,
  });

  return {
    hardware,
    taskProfile,
    recommendation,
    modelArtifact,
    catalogEntry,
    profile,
    capabilityRegistry,
  };
}

export function buildLlamaCppArgs({
  profile,
  prompt = null,
  systemPrompt = null,
  maxTokens = null,
  host = null,
  port = null,
  serverMode = false,
}) {
  const args = [
    "-m",
    profile.modelPath,
    "-c",
    String(profile.contextSize),
    "-ngl",
    String(profile.gpuLayers),
    "-t",
    String(profile.threads),
    "-b",
    String(profile.batchSize),
    "-ub",
    String(profile.ubatchSize),
    "-ctk",
    profile.kvCache,
    "-ctv",
    profile.kvCache,
  ];

  if (!profile.mmap) {
    args.push("--no-mmap");
  }
  if (profile.mlock) {
    args.push("--mlock");
  }

  if (serverMode) {
    args.push(
      "--host",
      host ?? "127.0.0.1",
      "--port",
      String(port ?? 8080),
    );
    return args;
  }

  if (maxTokens != null) {
    args.push("-n", String(maxTokens));
  }

  args.push(
    "--simple-io",
    "--log-disable",
    "--reasoning",
    "off",
    "--no-display-prompt",
    "--single-turn",
  );
  if (profile.noConversationMode) {
    args.push("--no-conversation");
  }
  if (prompt != null) {
    args.push("-p", prompt);
  }
  if (systemPrompt) {
    args.push("-sys", systemPrompt);
  }
  return args;
}
