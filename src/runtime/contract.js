export const NATIVE_RUNTIME_PROVIDER = "lizard-native";
export const REFERENCE_RUNTIME_PROVIDER = "llama.cpp";
export const OLLAMA_RUNTIME_PROVIDER = "Ollama";

export const NATIVE_RESULT_FIELDS = [
  "runtimeProvider",
  "nativeBackend",
  "nativeModelStatus",
  "loadMs",
  "prefillMs",
  "decodeMs",
  "tokensPerSecond",
  "kvCacheBytes",
  "fallbackReason",
];

function normalizeNumber(value) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : null;
}

export function buildRuntimeContractFields({
  runtimeProvider = REFERENCE_RUNTIME_PROVIDER,
  nativeBackend = null,
  nativeModelStatus = null,
  loadMs = null,
  prefillMs = null,
  decodeMs = null,
  tokensPerSecond = null,
  kvCacheBytes = null,
  fallbackReason = null,
} = {}) {
  return {
    runtimeProvider,
    nativeBackend,
    nativeModelStatus,
    loadMs: normalizeNumber(loadMs),
    prefillMs: normalizeNumber(prefillMs),
    decodeMs: normalizeNumber(decodeMs),
    tokensPerSecond: normalizeNumber(tokensPerSecond),
    kvCacheBytes: normalizeNumber(kvCacheBytes),
    fallbackReason: fallbackReason ? String(fallbackReason) : null,
  };
}

export function inferRuntimeProvider(runtime) {
  const normalized = String(runtime ?? "").trim().toLowerCase();
  if (!normalized) {
    return REFERENCE_RUNTIME_PROVIDER;
  }
  if (normalized.includes("ollama")) {
    return OLLAMA_RUNTIME_PROVIDER;
  }
  if (normalized.includes("lizard-native")) {
    return NATIVE_RUNTIME_PROVIDER;
  }
  return REFERENCE_RUNTIME_PROVIDER;
}

export function mergeRuntimeContract(target = {}, fields = {}) {
  return {
    ...target,
    ...buildRuntimeContractFields(fields),
  };
}
