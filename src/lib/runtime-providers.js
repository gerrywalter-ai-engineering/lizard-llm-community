const PROVIDERS = [
  {
    id: "caterpillar",
    label: "Caterpillar",
    // Promoted to stable 2026-08-04. The run-04 performance campaign took the
    // shipped default from 4.18 to 10.71 tok/s (2.56x) with the selftest green
    // in every dispatch configuration and 11/11 native CTests passing; the
    // engine is no longer the moving target it was when it was labelled
    // experimental. Deeper architectural work now happens in Ostrich so that
    // Caterpillar stays the stable baseline it is measured against.
    implementationStatus: "stable",
    description: "Stable standalone graph-plan inference engine: recorded D3D12 decode with stable VRAM fit, integer-dot GEMM with a lane-wise Q4_K payload and four-row Q6_K tile, resident KV attention, IQ4_NL/IQ4_XS quants, and lossless speculative decode.",
    promotion: {
      status: "stable",
      enginePhase: 8,
      decidedAt: "2026-08-04",
    },
    capabilities: {
      dashboardExecution: true,
      localWarmPool: true,
      localQuantization: true,
      ollamaKeepWarm: false,
      sessionRestore: true,
    },
  },
  {
    id: "lizard-native",
    label: "lizard-native",
    implementationStatus: "experimental",
    description: "Lizard-owned native runtime using a standalone worker/server boundary and explicit benchmarkable contract fields.",
    capabilities: {
      dashboardExecution: true,
      localWarmPool: true,
      localQuantization: true,
      ollamaKeepWarm: false,
      sessionRestore: true,
    },
  },
  {
    id: "llama.cpp",
    label: "llama.cpp",
    implementationStatus: "implemented",
    description: "Managed local GGUF runtime with quantization, warm-server orchestration, and direct dashboard execution.",
    capabilities: {
      dashboardExecution: true,
      localWarmPool: true,
      localQuantization: true,
      ollamaKeepWarm: true,
      sessionRestore: true,
    },
  },
  {
    id: "Ollama",
    label: "Ollama",
    implementationStatus: "partial",
    description: "External Ollama runtime entry registered for future direct dashboard execution.",
    capabilities: {
      dashboardExecution: false,
      localWarmPool: false,
      localQuantization: false,
      ollamaKeepWarm: true,
      sessionRestore: true,
    },
  },
  {
    id: "vLLM",
    label: "vLLM",
    implementationStatus: "planned",
    description: "Provider stub for OpenAI-compatible vLLM serving endpoints.",
    capabilities: {
      dashboardExecution: false,
      localWarmPool: false,
      localQuantization: false,
      ollamaKeepWarm: false,
      sessionRestore: false,
    },
  },
  {
    id: "SGLang",
    label: "SGLang",
    implementationStatus: "planned",
    description: "Provider stub for SGLang-based serving and routing.",
    capabilities: {
      dashboardExecution: false,
      localWarmPool: false,
      localQuantization: false,
      ollamaKeepWarm: false,
      sessionRestore: false,
    },
  },
  {
    id: "TGI",
    label: "TGI",
    implementationStatus: "planned",
    description: "Provider stub for Hugging Face Text Generation Inference.",
    capabilities: {
      dashboardExecution: false,
      localWarmPool: false,
      localQuantization: false,
      ollamaKeepWarm: false,
      sessionRestore: false,
    },
  },
];

export const DEFAULT_RUNTIME_PROVIDER = "llama.cpp";
// Standard provider a fresh model runtime config is seeded with: lizard runs
// default to the Lizard-owned native engine. Kept separate from
// DEFAULT_RUNTIME_PROVIDER, which stays "llama.cpp" as the normalize fallback
// for unknown values, warm-pool labeling, and the stock llama.cpp baseline lane.
export const DEFAULT_MODEL_RUNTIME_PROVIDER = "lizard-native";
export const MODEL_LIFECYCLE_STATES = [
  "downloaded",
  "verified",
  "benchmarked",
  "activated",
  "loaded",
  "warm",
  "stopped",
];

const providerMap = new Map(PROVIDERS.map((provider) => [provider.id, provider]));

// The UI must not re-derive status wording; it is decided here so every surface
// (onboarding, runner selection, dashboards, logs) agrees. Section 15 requires
// "Caterpillar - Stable" and "Ostrich - Experimental".
function statusSuffix(implementationStatus) {
  switch (implementationStatus) {
    case "stable": return "Stable";
    case "experimental": return "Experimental";
    case "partial": return "Partial";
    case "planned": return "Planned";
    default: return null;
  }
}

function cloneProvider(provider) {
  const suffix = statusSuffix(provider.implementationStatus);
  return {
    ...provider,
    displayLabel: suffix ? `${provider.label} — ${suffix}` : provider.label,
    isExperimental: provider.implementationStatus === "experimental",
    capabilities: { ...(provider.capabilities || {}) },
    ...(provider.promotion ? { promotion: { ...provider.promotion } } : {}),
  };
}

function normalizedText(value) {
  return String(value ?? "").trim().toLowerCase();
}

function modelMatches(left, right) {
  if (!left || !right) {
    return false;
  }
  return normalizedText(left) === normalizedText(right);
}

function findRecentBenchmark(recentRuns, modelId) {
  return (recentRuns || []).some((entry) => {
    const commandId = String(entry?.commandId || entry?.command_id || "").toLowerCase();
    const runModelId = entry?.modelId || entry?.model_id || null;
    return commandId.includes("benchmark") && modelMatches(runModelId, modelId);
  });
}

export function listRuntimeProviders() {
  return PROVIDERS.map(cloneProvider);
}

// Short names people actually type on a command line. "lizard" in particular
// reads as "the Lizard engine", but it is not a provider id, so it used to
// normalize to the DEFAULT_RUNTIME_PROVIDER fallback - meaning
// `--runtime-provider lizard` silently benchmarked stock llama.cpp and
// reported it as the lizard lane.
const RUNTIME_PROVIDER_ALIASES = new Map([
  ["lizard", "lizard-native"],
  ["native", "lizard-native"],
  ["lizard native", "lizard-native"],
  ["llamacpp", "llama.cpp"],
  ["llama-cpp", "llama.cpp"],
  ["llama_cpp", "llama.cpp"],
]);

// Resolves an id or alias, or null when the value names no provider at all.
// Callers that take user input should use this and report the bad value,
// rather than normalizeRuntimeProvider's silent fallback.
export function resolveRuntimeProviderId(providerId) {
  const candidate = String(providerId ?? "").trim();
  if (candidate === "") {
    return null;
  }
  if (providerMap.has(candidate)) {
    return candidate;
  }
  const lowered = candidate.toLowerCase();
  const caseInsensitiveMatch = PROVIDERS.find((provider) => provider.id.toLowerCase() === lowered);
  if (caseInsensitiveMatch) {
    return caseInsensitiveMatch.id;
  }
  return RUNTIME_PROVIDER_ALIASES.get(lowered) ?? null;
}

export function runtimeProviderIdsForHelp() {
  return PROVIDERS.map((provider) => provider.id);
}

export function normalizeRuntimeProvider(providerId) {
  return resolveRuntimeProviderId(providerId) ?? DEFAULT_RUNTIME_PROVIDER;
}

export function getRuntimeProvider(providerId) {
  return cloneProvider(providerMap.get(normalizeRuntimeProvider(providerId)) ?? providerMap.get(DEFAULT_RUNTIME_PROVIDER));
}

export function runtimeProviderLabel(providerId) {
  return getRuntimeProvider(providerId).label;
}

export function runtimeProviderSupportsCapability(providerId, capability) {
  return Boolean(getRuntimeProvider(providerId).capabilities?.[capability]);
}

// --- Experimental runner access control (Ostrich, 2026-08-04) ---------------
//
// Ostrich is admin-only until it has been benchmarked against Caterpillar.
// These helpers are the single place that decides access, so the API layer,
// the CLI and the dashboard cannot drift apart. The backend MUST call
// assertRuntimeProviderAllowed before launching: hiding the option in the UI
// is not access control, and a hand-written request must fail the same way.

export function runtimeProviderRequiresExperimentalAccess(providerId) {
  return Boolean(getRuntimeProvider(providerId).requiresExperimentalAccess);
}

export function runtimeProviderAccessFlag(providerId) {
  return getRuntimeProvider(providerId).experimentalAccessFlag ?? null;
}

// `user` is any shape carrying { isAdmin, roles, flags } - callers pass what
// their layer has. Absent/unknown users get no experimental access.
export function userCanUseRuntimeProvider(providerId, user = null) {
  if (!runtimeProviderRequiresExperimentalAccess(providerId)) {
    return true;
  }
  if (!user) {
    return false;
  }
  const isAdmin = Boolean(user.isAdmin ?? user.admin)
    || (Array.isArray(user.roles) && user.roles.some((role) => String(role).toLowerCase() === "admin"));
  if (isAdmin) {
    return true;
  }
  const flag = runtimeProviderAccessFlag(providerId);
  if (!flag) {
    return false;
  }
  // Per-user grant, set by an admin in the user-profile view.
  return Boolean(user.flags?.[flag] ?? user[flag]);
}

// Providers a given user may actually see and select. Used by every listing
// surface so an unauthorized user never receives Ostrich in a payload.
export function listRuntimeProvidersForUser(user = null) {
  return listRuntimeProviders().filter((provider) => userCanUseRuntimeProvider(provider.id, user));
}

export class RuntimeProviderAccessError extends Error {
  constructor(providerId) {
    super(`runtime provider "${providerId}" requires experimental access`);
    this.name = "RuntimeProviderAccessError";
    this.code = "RUNTIME_PROVIDER_FORBIDDEN";
    this.statusCode = 403;
    this.providerId = providerId;
  }
}

// Throws when the caller may not use this provider. Fails closed.
export function assertRuntimeProviderAllowed(providerId, user = null) {
  const resolved = normalizeRuntimeProvider(providerId);
  if (!userCanUseRuntimeProvider(resolved, user)) {
    throw new RuntimeProviderAccessError(resolved);
  }
  return resolved;
}

// The request-facing resolver every write path should use.
//
// Only a *requested change* is access-checked. An absent request keeps the
// current value untouched, so existing saved configurations keep working even
// if they name a provider the current user could not newly select — that is
// the "existing configurations must continue working" rule, and it is why this
// is not simply assertRuntimeProviderAllowed at every call site.
export function resolveRequestedRuntimeProvider({
  requested = undefined,
  current = null,
  fallback = DEFAULT_RUNTIME_PROVIDER,
  user = null,
} = {}) {
  if (requested === undefined || requested === null || requested === "") {
    return normalizeRuntimeProvider(current ?? fallback);
  }
  const resolved = normalizeRuntimeProvider(requested);
  if (resolved === normalizeRuntimeProvider(current)) {
    return resolved;
  }
  return assertRuntimeProviderAllowed(resolved, user);
}

export function deriveModelLifecycleState({
  modelId,
  catalogEntry = null,
  runtimeConfig = null,
  activeResolution = null,
  warmState = null,
  recentRuns = [],
  lastStoppedModel = null,
}) {
  if (!modelId) {
    return null;
  }

  const activeModelId = runtimeConfig?.activeModelId ?? null;
  const warmLocalModelId = warmState?.lizard?.modelId ?? null;
  const warmOllamaModelId = warmState?.ollama?.model ?? null;
  const warmLocal = ["ready", "warming"].includes(String(warmState?.lizard?.status || "").toLowerCase());
  const warmOllama = ["ready", "warming"].includes(String(warmState?.ollama?.status || "").toLowerCase());
  const lastStoppedModelId = lastStoppedModel?.modelId ?? lastStoppedModel?.runtimeConfig?.activeModelId ?? null;

  if ((warmLocal && modelMatches(modelId, warmLocalModelId)) || (warmOllama && modelMatches(modelId, warmOllamaModelId))) {
    return "warm";
  }
  if (modelMatches(modelId, activeModelId) && activeResolution?.status === "ready") {
    return "loaded";
  }
  if (modelMatches(modelId, activeModelId)) {
    return "activated";
  }
  if (modelMatches(modelId, lastStoppedModelId)) {
    return "stopped";
  }
  if (findRecentBenchmark(recentRuns, modelId)) {
    return "benchmarked";
  }
  if (catalogEntry?.sha256 || String(catalogEntry?.status || "").toLowerCase() === "ready") {
    return "verified";
  }
  if (catalogEntry?.importedPath || catalogEntry?.filename || catalogEntry?.source) {
    return "downloaded";
  }
  return null;
}

function runtimeEntry({
  modelId,
  providerId,
  quantizationLevel = null,
  lifecycleState = null,
  status = null,
  note = null,
}) {
  if (!modelId) {
    return null;
  }
  const provider = getRuntimeProvider(providerId);
  return {
    modelId,
    providerId: provider.id,
    providerLabel: provider.label,
    providerStatus: provider.implementationStatus,
    quantizationLevel,
    lifecycleState,
    status,
    note,
  };
}

export function buildRuntimeHandoffState({
  runtimeConfig = null,
  activeResolution = null,
  warmState = null,
  recentRuns = [],
  lastStoppedModel = null,
  catalogEntries = [],
  freeTesting = false,
}) {
  const activeModelId = runtimeConfig?.activeModelId ?? null;
  const activeProviderId = normalizeRuntimeProvider(runtimeConfig?.runtimeProvider);
  const warmLocalModelId = warmState?.lizard?.modelId ?? null;
  const warmOllamaModelId = warmState?.ollama?.model ?? null;
  const warmLocalStatus = warmState?.lizard?.status ?? null;
  const warmOllamaStatus = warmState?.ollama?.status ?? null;
  const warmModelId = warmLocalModelId || warmOllamaModelId || null;
  const warmProviderId = warmLocalModelId ? DEFAULT_RUNTIME_PROVIDER : (warmOllamaModelId ? "Ollama" : activeProviderId);
  const activeCatalogEntry = catalogEntries.find((entry) => modelMatches(entry.id, activeModelId)) ?? null;
  const warmCatalogEntry = catalogEntries.find((entry) => modelMatches(entry.id, warmModelId)) ?? null;
  const lastStoppedModelId = lastStoppedModel?.modelId ?? lastStoppedModel?.runtimeConfig?.activeModelId ?? null;
  const lastStoppedProviderId = normalizeRuntimeProvider(lastStoppedModel?.runtimeProvider ?? lastStoppedModel?.runtimeConfig?.runtimeProvider);

  const activeModel = runtimeEntry({
    modelId: activeModelId,
    providerId: activeProviderId,
    quantizationLevel: runtimeConfig?.activeQuantizationLevel ?? null,
    lifecycleState: deriveModelLifecycleState({
      modelId: activeModelId,
      catalogEntry: activeCatalogEntry,
      runtimeConfig,
      activeResolution,
      warmState,
      recentRuns,
      lastStoppedModel,
    }),
    status: activeResolution?.status ?? null,
    note: activeResolution?.note ?? null,
  });

  const warmModel = runtimeEntry({
    modelId: warmModelId,
    providerId: warmProviderId,
    quantizationLevel: warmState?.lizard?.compressionLevel ?? null,
    lifecycleState: deriveModelLifecycleState({
      modelId: warmModelId,
      catalogEntry: warmCatalogEntry,
      runtimeConfig,
      activeResolution,
      warmState,
      recentRuns,
      lastStoppedModel,
    }),
    status: warmLocalModelId ? warmLocalStatus : warmOllamaStatus,
    note: warmLocalModelId ? warmState?.lizard?.note ?? warmState?.note ?? null : warmState?.ollama?.note ?? null,
  });

  const conflictModel = activeModelId && warmLocalModelId && !modelMatches(activeModelId, warmLocalModelId)
    ? runtimeEntry({
      modelId: warmLocalModelId,
      providerId: DEFAULT_RUNTIME_PROVIDER,
      quantizationLevel: warmState?.lizard?.compressionLevel ?? null,
      lifecycleState: "warm",
      status: warmLocalStatus,
      note: warmState?.lizard?.note ?? warmState?.note ?? null,
    })
    : null;

  const stoppedModel = runtimeEntry({
    modelId: lastStoppedModelId,
    providerId: lastStoppedProviderId,
    quantizationLevel: lastStoppedModel?.quantizationLevel ?? lastStoppedModel?.runtimeConfig?.activeQuantizationLevel ?? null,
    lifecycleState: "stopped",
    status: "stopped",
    note: lastStoppedModel?.note ?? null,
  });

  return {
    selectedProvider: getRuntimeProvider(activeProviderId),
    activeModel,
    warmModel,
    conflictModel,
    lastStoppedModel: stoppedModel,
    canRestoreLastStoppedModel: Boolean(lastStoppedModelId),
    freeTesting: Boolean(freeTesting),
  };
}
