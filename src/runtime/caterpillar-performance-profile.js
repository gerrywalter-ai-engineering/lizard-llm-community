export const CATERPILLAR_BALANCED_PROFILE = "balanced";
export const CATERPILLAR_TTFT_PROFILE = "cpil_ttft";

const TTFT_FLAG_KEYS = Object.freeze([
  "cpil_ttft",
  "cpil-ttft",
  // Accept the transposed spelling used in early notes, but never emit it.
  "cpli_ttft",
  "cpli-ttft",
]);

export function normalizeCaterpillarPerformanceProfile(value, fallback = CATERPILLAR_BALANCED_PROFILE) {
  const normalized = String(value ?? "").trim().toLowerCase().replaceAll("-", "_");
  if (["cpil_ttft", "cpli_ttft", "ttft", "latency", "b1"].includes(normalized)) {
    return CATERPILLAR_TTFT_PROFILE;
  }
  if (["balanced", "normal", "throughput", "batch"].includes(normalized)) {
    return CATERPILLAR_BALANCED_PROFILE;
  }
  return fallback;
}

export function caterpillarPerformanceProfileFromOptions(
  options = {},
  { defaultProfile = CATERPILLAR_BALANCED_PROFILE, env = process.env } = {},
) {
  if (TTFT_FLAG_KEYS.some((key) => options?.[key] === true)) {
    return CATERPILLAR_TTFT_PROFILE;
  }
  const explicit = options?.caterpillarProfile
    ?? options?.["caterpillar-profile"]
    ?? options?.performanceProfile
    ?? options?.["performance-profile"]
    ?? env?.LIZARD_CATERPILLAR_PROFILE;
  return normalizeCaterpillarPerformanceProfile(explicit, defaultProfile);
}

export function isCaterpillarTtftProfile(profile = null) {
  return normalizeCaterpillarPerformanceProfile(
    profile?.caterpillarPerformanceProfile
      ?? profile?.nativeRuntime?.caterpillarPerformanceProfile,
  ) === CATERPILLAR_TTFT_PROFILE;
}

/**
 * Applies the product-level Caterpillar latency preference.
 *
 * This profile deliberately changes scheduling and device preference, not
 * model math: the rejected approximate integer-dot candidate is never enabled.
 * Greedy output therefore stays on the accepted correctness path.
 */
export function applyCaterpillarPerformanceProfile(profile = {}, {
  runtimeProvider = "lizard-native",
  performanceProfile = CATERPILLAR_BALANCED_PROFILE,
} = {}) {
  if (runtimeProvider !== "caterpillar") {
    return profile;
  }
  const selected = normalizeCaterpillarPerformanceProfile(performanceProfile);
  if (selected !== CATERPILLAR_TTFT_PROFILE) {
    return {
      ...profile,
      caterpillarPerformanceProfile: CATERPILLAR_BALANCED_PROFILE,
      nativeRuntime: {
        ...(profile?.nativeRuntime ?? {}),
        caterpillarPerformanceProfile: CATERPILLAR_BALANCED_PROFILE,
      },
    };
  }
  return {
    ...profile,
    caterpillarPerformanceProfile: CATERPILLAR_TTFT_PROFILE,
    promptCache: true,
    nativeRuntime: {
      ...(profile?.nativeRuntime ?? {}),
      caterpillarPerformanceProfile: CATERPILLAR_TTFT_PROFILE,
      latencyPreference: "ttft",
      // B=1 is the contract for this profile. Do not queue a lone chat turn
      // behind the concurrent decode coordinator or force the CPU batch lane.
      batchedDecode: false,
      alwaysBatch: false,
      preferredBatchSize: 1,
      batchTargetSize: 1,
      batchFallbackOrder: [1],
      // Repeated prompts/retries copy a private exact KV snapshot. Chat's
      // append-only prompt cache remains enabled independently.
      prefillSnapshotCache: true,
      keepWorkerLoaded: true,
      // Prefer Caterpillar's device graph when it fits. The runner still falls
      // back to its exact CPU implementation when no compatible device exists.
      enableHybridGpuPrefill: true,
      forceCpuOnlyPlan: false,
      gpuDecodePolicy: "allowed",
    },
  };
}

export function caterpillarProfileCliArgs(performanceProfile) {
  return normalizeCaterpillarPerformanceProfile(performanceProfile) === CATERPILLAR_TTFT_PROFILE
    ? ["--cpil_ttft"]
    : [];
}
