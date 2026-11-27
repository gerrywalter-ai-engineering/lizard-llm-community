import { buildKvCachePlan } from "./kv-cache/index.js";

export const NATIVE_BATCH_LANES = Object.freeze([16, 8, 4, 1]);
export const PRIMARY_NATIVE_BATCH_LANES = Object.freeze([16, 8]);
export const FALLBACK_NATIVE_BATCH_LANES = Object.freeze([4, 1]);

function positiveInteger(value) {
  const numeric = Number(value);
  return Number.isFinite(numeric) && numeric > 0 ? Math.trunc(numeric) : null;
}

function laneAtOrBelow(limit) {
  const ceiling = Math.max(1, positiveInteger(limit) ?? 16);
  return NATIVE_BATCH_LANES.find((lane) => lane <= ceiling) ?? 1;
}

function memoryLane(profile) {
  const availableGb = Number(
    profile?.capabilityRegistry?.memory?.availableGb
      ?? profile?.capabilityRegistry?.memory?.systemAvailableGb,
  );
  if (!Number.isFinite(availableGb) || availableGb <= 0) return 16;
  if (availableGb >= 8) return 16;
  if (availableGb >= 4) return 8;
  if (availableGb >= 2) return 4;
  return 1;
}

/**
 * Product concurrency policy shared by lizard-native and Caterpillar.
 *
 * B=16 is the throughput target. B=8 is the normal reduced-memory target.
 * B=4 and B=1 remain explicit fallbacks, never preferred lanes. The byte
 * allocator still checks every real KV allocation; this policy only decides
 * how many requests the scheduler/coordinator should try to advance together.
 */
export function resolveNativeConcurrencyPolicy({
  runtimeProvider = "lizard-native",
  profile = null,
  gguf = null,
  requestedBatchSize = null,
  measuredBatchLevels = null,
} = {}) {
  const explicit = positiveInteger(
    requestedBatchSize
      ?? profile?.nativeRuntime?.preferredBatchSize
      ?? profile?.nativeRuntime?.maxConcurrentRequests
      ?? profile?.nativeRuntime?.maxConcurrentSlots,
  );
  const kvPlan = buildKvCachePlan({
    modelId: profile?.modelId ?? null,
    profile,
    gguf: gguf ?? profile?.gguf ?? null,
  });
  const capacityCeiling = Math.min(
    explicit ?? 16,
    positiveInteger(kvPlan.maxSlots) ?? 16,
    memoryLane(profile),
  );
  const capacityBatchSize = laneAtOrBelow(capacityCeiling);
  const eligibleMeasuredLevels = {};
  for (const lane of NATIVE_BATCH_LANES) {
    if (lane <= capacityBatchSize) {
      eligibleMeasuredLevels[lane] = measuredBatchLevels?.[lane]
        ?? measuredBatchLevels?.[String(lane)]
        ?? null;
    }
  }
  const measuredSelection = selectMeasuredNativeBatch(eligibleMeasuredLevels);
  // A low-concurrency smoke must not permanently make the primary lanes
  // unreachable. Previously a repeated B=1-only measurement was persisted,
  // selected on the next launch, and reduced HTTP admission to one request;
  // B=8/B=16 could then never be measured or used. Apply a measured fallback
  // only when capacity itself is already constrained to a fallback lane.
  // Primary measurements may still choose the faster of B=16 and B=8.
  const measuredSelectionApplicable = measuredSelection.selectedBatchSize != null
    && (measuredSelection.role === "primary" || !PRIMARY_NATIVE_BATCH_LANES.includes(capacityBatchSize));
  const selectedBatchSize = measuredSelectionApplicable
    ? measuredSelection.selectedBatchSize
    : capacityBatchSize;
  const fallbackOrder = NATIVE_BATCH_LANES.filter((lane) => lane <= selectedBatchSize);
  const role = PRIMARY_NATIVE_BATCH_LANES.includes(selectedBatchSize) ? "primary" : "fallback";
  const reason = measuredSelectionApplicable
    ? `${measuredSelection.reason} Selection is based on completed native HTTP throughput measurements.`
    : explicit != null
      ? `Explicit concurrency ceiling ${explicit} selected ${selectedBatchSize}.`
      : selectedBatchSize === 16
        ? "B=16 selected as the fastest verified aggregate-throughput lane."
        : selectedBatchSize === 8
          ? "B=8 selected because the current memory/session budget does not qualify for B=16."
          : `B=${selectedBatchSize} selected only as a constrained-capacity fallback.`;

  return {
    mode: "adaptive",
    runtimeProvider,
    preferredBatchSize: 16,
    secondaryBatchSize: 8,
    fallbackBatchSizes: [...FALLBACK_NATIVE_BATCH_LANES],
    selectedBatchSize,
    role,
    source: measuredSelectionApplicable ? "measured-http" : "capacity-policy",
    measuredSelection: measuredSelectionApplicable ? measuredSelection : null,
    ignoredMeasuredSelection: measuredSelection.selectedBatchSize && !measuredSelectionApplicable
      ? measuredSelection
      : null,
    fallbackOrder,
    capacityCeiling,
    capacity: {
      kvMaxSlots: kvPlan.maxSlots,
      guaranteedSlots: kvPlan.guaranteedSlots,
      memoryLane: memoryLane(profile),
      explicitCeiling: explicit,
    },
    reason,
  };
}

/**
 * Picks the fastest measured primary lane. B=4/B=1 are considered only when
 * neither B=16 nor B=8 completed with a usable throughput value.
 */
export function selectMeasuredNativeBatch(levels = {}) {
  const read = (batch) => {
    const level = levels instanceof Map ? levels.get(batch) : levels?.[batch] ?? levels?.[String(batch)];
    const throughput = Number(level?.aggregateTokensPerSecond);
    return level && level.failed === 0 && Number.isFinite(throughput) && throughput > 0
      ? { batch, throughput, level }
      : null;
  };
  const chooseFastest = (lanes) => lanes
    .map(read)
    .filter(Boolean)
    .sort((left, right) => right.throughput - left.throughput)[0] ?? null;
  const primary = chooseFastest(PRIMARY_NATIVE_BATCH_LANES);
  const selected = primary ?? chooseFastest(FALLBACK_NATIVE_BATCH_LANES);
  if (!selected) {
    return {
      selectedBatchSize: null,
      role: "unavailable",
      aggregateTokensPerSecond: null,
      reason: "No measured B=16/B=8 result or B=4/B=1 fallback completed.",
    };
  }
  return {
    selectedBatchSize: selected.batch,
    role: primary ? "primary" : "fallback",
    aggregateTokensPerSecond: selected.throughput,
    reason: primary
      ? `Fastest measured primary lane is B=${selected.batch}.`
      : `Primary B=16/B=8 lanes were unavailable; using B=${selected.batch} fallback.`,
  };
}
