import fs from "node:fs";
import path from "node:path";
import { ensureDirectory } from "./paths.js";
import { contextBucket } from "./native-turn-history.js";
import { selectMeasuredNativeBatch } from "../runtime/concurrency-policy.js";

const SCHEMA = "lizard.native-http-performance.v1";
const MAX_PROFILES = 100;

function normalized(value, fallback = "unknown") {
  const text = String(value ?? "").trim().toLowerCase();
  return text || fallback;
}

function finite(value) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : null;
}

function storePath(config) {
  return path.join(config.runtimeDirectory, "native-http-performance.json");
}

function readStore(config) {
  try {
    const parsed = JSON.parse(fs.readFileSync(storePath(config), "utf8"));
    if (parsed?.schema === SCHEMA && Array.isArray(parsed.profiles)) return parsed;
  } catch {
    // A missing, old, or interrupted cache is equivalent to no measurements.
  }
  return { schema: SCHEMA, updatedAt: null, profiles: [] };
}

function writeStore(config, store) {
  const filePath = storePath(config);
  ensureDirectory(path.dirname(filePath));
  const temporaryPath = `${filePath}.${process.pid}.tmp`;
  fs.writeFileSync(temporaryPath, `${JSON.stringify(store, null, 2)}\n`, "utf8");
  fs.renameSync(temporaryPath, filePath);
}

function profileIdentity({ runtimeProvider, modelId, quant, contextTokens, optimizationProfile = null }) {
  const identity = [
    normalized(runtimeProvider),
    normalized(modelId),
    normalized(quant),
    contextBucket(contextTokens),
  ].join("|");
  // Preserve the legacy/balanced identity while giving Caterpillar's
  // load-affecting B=1 profile its own tuning history.
  return normalized(runtimeProvider) === "caterpillar"
      && normalized(optimizationProfile, "balanced") !== "balanced"
    ? `${identity}|${normalized(optimizationProfile)}`
    : identity;
}

function projectLevel(level) {
  if (!level || typeof level !== "object") return null;
  const aggregateTokensPerSecond = finite(level.aggregateTokensPerSecond);
  const ttftMs = finite(level.streamingTtft?.ttftMs ?? level.streamingTtft?.firstTokenMs);
  return {
    completed: finite(level.completed),
    failed: finite(level.failed) ?? 0,
    aggregateTokensPerSecond,
    latestAggregateTokensPerSecond: aggregateTokensPerSecond,
    bestAggregateTokensPerSecond: aggregateTokensPerSecond,
    medianLatencyMs: finite(level.medianLatencyMs),
    meanLatencyMs: finite(level.meanLatencyMs),
    p95LatencyMs: finite(level.p95LatencyMs),
    ttftMs,
    latestTtftMs: ttftMs,
    bestTtftMs: ttftMs,
    peakWorkingSetBytes: finite(level.resources?.peakWorkingSetBytes),
    gpuUtilizationPercent: finite(level.resources?.gpuUtilizationPercent),
    measuredAt: new Date().toISOString(),
    evidenceType: "single-run",
    repetitions: 1,
  };
}

function preferLevel(existing, incoming) {
  if (!existing) return incoming;
  if (!incoming) return existing;
  const existingTps = finite(existing.bestAggregateTokensPerSecond)
    ?? finite(existing.aggregateTokensPerSecond);
  const incomingTps = finite(incoming.aggregateTokensPerSecond);
  const bestAggregateTokensPerSecond = Math.max(existingTps ?? 0, incomingTps ?? 0) || null;
  const existingTtft = finite(existing.bestTtftMs) ?? finite(existing.ttftMs);
  const incomingTtft = finite(incoming.ttftMs);
  const ttftCandidates = [existingTtft, incomingTtft].filter((value) => value > 0);
  const bestTtftMs = ttftCandidates.length > 0 ? Math.min(...ttftCandidates) : null;
  const recency = {
    latestAggregateTokensPerSecond: incomingTps,
    bestAggregateTokensPerSecond,
    latestTtftMs: incomingTtft,
    bestTtftMs,
  };
  if (existing.evidenceType === "repeated" && incoming.evidenceType !== "repeated") {
    // Stable medians remain the runtime tuning value, while the most recent
    // and best observations stay visible for the UI and regression review.
    return { ...existing, ...recency };
  }
  if (incoming.evidenceType === "repeated") return { ...incoming, ...recency };
  const winner = incomingTps != null && (existingTps == null || incomingTps >= existingTps)
    ? incoming
    : existing;
  return { ...winner, ...recency };
}

function upsertProfile(config, profile) {
  const store = readStore(config);
  const existing = store.profiles.find((entry) => entry?.id === profile.id) ?? null;
  const levels = {};
  for (const lane of [16, 8, 4, 1]) {
    const selected = preferLevel(
      existing?.levels?.[lane] ?? existing?.levels?.[String(lane)],
      profile.levels?.[lane] ?? profile.levels?.[String(lane)],
    );
    if (selected) levels[lane] = selected;
  }
  const selection = selectMeasuredNativeBatch(levels);
  const merged = {
    ...existing,
    ...profile,
    levels,
    selectedBatchSize: selection.selectedBatchSize,
    selectionRole: selection.role,
    aggregateTokensPerSecond: selection.aggregateTokensPerSecond,
    stable: [16, 8].some((lane) => levels[lane]?.evidenceType === "repeated"),
  };
  store.profiles = [
    merged,
    ...store.profiles.filter((entry) => entry?.id !== merged.id),
  ].slice(0, MAX_PROFILES);
  store.updatedAt = new Date().toISOString();
  writeStore(config, store);
  return merged;
}

/**
 * Persist only completed, measured HTTP lanes. This is a local tuning cache,
 * not a benchmark claim: provenance and llama.cpp evidence remain in the full
 * artifact. Runtime startup reads this cache solely to decide B=16 versus B=8.
 */
export function recordNativeHttpPerformance(config, artifact, { artifactPath = null } = {}) {
  if (artifact?.transport !== "http") return null;
  const candidate = artifact?.candidate ?? artifact?.lizard;
  if (!candidate?.model || !candidate?.provider || !candidate?.levels) return null;
  const contextTokens = finite(artifact?.settings?.contextTokensPerRequest);
  const quant = candidate.quant
    ?? artifact?.model?.quant
    ?? artifact?.quant
    ?? "unknown";
  const optimizationProfile = candidate.caterpillarPerformanceProfile
    ?? artifact?.settings?.caterpillarPerformanceProfile
    ?? "balanced";
  const levels = {};
  for (const lane of [16, 8, 4, 1]) {
    const projected = projectLevel(candidate.levels[lane] ?? candidate.levels[String(lane)]);
    if (projected?.failed === 0 && projected.aggregateTokensPerSecond > 0) {
      levels[lane] = projected;
    }
  }
  const selection = selectMeasuredNativeBatch(levels);
  if (!selection.selectedBatchSize) return null;
  const recordedAt = artifact.finishedAt ?? new Date().toISOString();
  const profile = {
    id: profileIdentity({
      runtimeProvider: candidate.provider,
      modelId: candidate.model,
      quant,
      contextTokens,
      optimizationProfile,
    }),
    recordedAt,
    runtimeProvider: normalized(candidate.provider),
    modelId: normalized(candidate.model),
    quant: normalized(quant),
    contextBucket: contextBucket(contextTokens),
    contextTokens,
    transport: "http",
    optimizationProfile: normalized(optimizationProfile, "balanced"),
    selectedBatchSize: selection.selectedBatchSize,
    selectionRole: selection.role,
    aggregateTokensPerSecond: selection.aggregateTokensPerSecond,
    levels,
    artifactPath,
  };
  return upsertProfile(config, profile);
}

/**
 * Promote a completed fresh-process series to stable runtime-selection
 * evidence. The median is used for lane choice; mean/min/max/stddev/P95 remain
 * attached so the choice is auditable.
 */
export function recordRepeatedNativeHttpPerformance(config, manifest, {
  quant = null,
  artifactPath = null,
  minimumRepetitions = 3,
} = {}) {
  if (manifest?.schema !== "lizard.smoke-bench.repeated-http.v1"
      || manifest?.status !== "completed"
      || !Array.isArray(manifest.runs)
      || manifest.runs.length < minimumRepetitions) {
    return null;
  }
  const report = manifest.runs[0]?.report;
  const candidate = report?.candidate ?? report?.lizard;
  if (report?.transport !== "http" || !candidate?.model || !candidate?.provider) return null;
  const contextTokens = finite(report?.settings?.contextTokensPerRequest);
  const resolvedQuant = quant
    ?? candidate.quant
    ?? report?.model?.quant
    ?? report?.quant
    ?? "unknown";
  const optimizationProfile = candidate.caterpillarPerformanceProfile
    ?? report?.settings?.caterpillarPerformanceProfile
    ?? "balanced";
  const levels = {};
  for (const lane of [16, 8, 4, 1]) {
    const summary = manifest.summary?.[lane] ?? manifest.summary?.[String(lane)];
    const throughput = summary?.candidate?.aggregateTokensPerSecond
      ?? summary?.candidateTokensPerSecond;
    const repetitions = finite(summary?.repetitions) ?? 0;
    const failed = finite(summary?.candidate?.failedRequests?.max) ?? 0;
    if (repetitions < minimumRepetitions
        || failed > 0
        || !(finite(throughput?.median) > 0)) {
      continue;
    }
    levels[lane] = {
      completed: lane,
      failed: 0,
      aggregateTokensPerSecond: finite(throughput.median),
      latestAggregateTokensPerSecond: finite(throughput.median),
      bestAggregateTokensPerSecond: finite(throughput.max) ?? finite(throughput.median),
      meanAggregateTokensPerSecond: finite(throughput.mean),
      minAggregateTokensPerSecond: finite(throughput.min),
      maxAggregateTokensPerSecond: finite(throughput.max),
      stddevAggregateTokensPerSecond: finite(throughput.stddev),
      p90AggregateTokensPerSecond: finite(throughput.p90),
      p95AggregateTokensPerSecond: finite(throughput.p95),
      coefficientOfVariationPercent: finite(throughput.coefficientOfVariationPercent),
      stability: throughput.stability ?? null,
      ttftMs: finite(summary?.candidate?.streamingProbe?.ttftMs?.median),
      latestTtftMs: finite(summary?.candidate?.streamingProbe?.ttftMs?.median),
      bestTtftMs: finite(summary?.candidate?.streamingProbe?.ttftMs?.min)
        ?? finite(summary?.candidate?.streamingProbe?.ttftMs?.median),
      peakWorkingSetBytes: finite(summary?.candidate?.resources?.peakProcessWorkingSetMb?.max) == null
        ? null
        : finite(summary.candidate.resources.peakProcessWorkingSetMb.max) * 1024 ** 2,
      gpuUtilizationPercent: finite(
        summary?.candidate?.resources?.averageGpuUtilizationPercent?.mean,
      ),
      measuredAt: manifest.finishedAt ?? new Date().toISOString(),
      evidenceType: "repeated",
      repetitions,
    };
  }
  const selection = selectMeasuredNativeBatch(levels);
  if (!selection.selectedBatchSize) return null;
  return upsertProfile(config, {
    id: profileIdentity({
      runtimeProvider: candidate.provider,
      modelId: candidate.model,
      quant: resolvedQuant,
      contextTokens,
      optimizationProfile,
    }),
    recordedAt: manifest.finishedAt ?? new Date().toISOString(),
    runtimeProvider: normalized(candidate.provider),
    modelId: normalized(candidate.model),
    quant: normalized(resolvedQuant),
    contextBucket: contextBucket(contextTokens),
    contextTokens,
    transport: "http",
    optimizationProfile: normalized(optimizationProfile, "balanced"),
    selectedBatchSize: selection.selectedBatchSize,
    selectionRole: selection.role,
    aggregateTokensPerSecond: selection.aggregateTokensPerSecond,
    levels,
    artifactPath,
    seriesLabel: manifest.label ?? null,
  });
}

/**
 * Prefer an exact provider/model/quant/context match, then the newest measured
 * profile for the same provider and model.
 */
export function findNativeHttpPerformance(config, {
  runtimeProvider,
  modelId,
  quant = null,
  contextTokens = null,
  optimizationProfile = null,
} = {}) {
  const wantedOptimizationProfile = optimizationProfile == null
    ? null
    : normalized(optimizationProfile, "balanced");
  const profiles = readStore(config).profiles
    .filter((entry) => (
      entry?.transport === "http"
      && entry.runtimeProvider === normalized(runtimeProvider)
      && entry.modelId === normalized(modelId)
      && (wantedOptimizationProfile == null
        || normalized(entry.optimizationProfile, "balanced") === wantedOptimizationProfile)
    ));
  if (profiles.length === 0) return null;
  const wantedQuant = normalized(quant);
  const wantedContext = contextBucket(contextTokens);
  const exact = profiles.find((entry) => (
    entry.quant === wantedQuant
    && entry.contextBucket === wantedContext
  ));
  const sameQuant = profiles.find((entry) => entry.quant === wantedQuant);
  const selected = exact ?? sameQuant ?? profiles[0];
  return {
    ...selected,
    match: exact ? "exact" : sameQuant ? "same-quant" : "same-model",
  };
}

export function nativeHttpPerformancePath(config) {
  return storePath(config);
}
