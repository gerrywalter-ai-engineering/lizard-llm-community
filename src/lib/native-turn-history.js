import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { ensureDirectory } from "./paths.js";

const HISTORY_LIMIT = 1000;

function finite(value) {
  if (value == null || value === "") return null;
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : null;
}

function text(value, fallback = null) {
  const normalized = String(value ?? "").trim();
  return normalized || fallback;
}

export function contextBucket(tokens) {
  const count = Math.max(0, finite(tokens) ?? 0);
  for (const ceiling of [1024, 2048, 4096, 8192, 16384, 32768]) {
    if (count <= ceiling) return `ctx-${ceiling}`;
  }
  return "ctx-32768+";
}

function historyPath(config) {
  return path.join(config.runtimeDirectory, "native-turn-metrics.jsonl");
}

export function appendNativeTurnMetric(config, sample = {}) {
  const decodeTokensPerSecond = finite(
    sample.decodeTokensPerSecond ?? sample.tokensPerSecond,
  );
  const tokensPerSecond = finite(
    sample.userVisibleTokensPerSecond ?? sample.tokensPerSecond,
  );
  if (!(tokensPerSecond > 0)) return null;
  const promptTokens = finite(sample.promptTokens);
  const evidence = sample.evidenceMetrics ?? null;
  const record = {
    id: crypto.randomUUID(),
    recordedAt: new Date().toISOString(),
    userId: text(sample.userId, "local"),
    surface: text(sample.surface, "chat"),
    modelId: text(sample.modelId, "unknown"),
    quant: text(sample.quant, "unknown"),
    runtimeProvider: text(sample.runtimeProvider, "unknown"),
    optimizationProfile: text(sample.optimizationProfile, "balanced"),
    backend: text(sample.backend, "unknown"),
    servingPath: text(sample.servingPath, sample.batched ? "batched" : "single-shot"),
    batched: sample.batched === true,
    concurrencyLevel: Math.max(1, Math.trunc(finite(sample.concurrencyLevel) ?? 1)),
    contextBucket: contextBucket(sample.contextTokens ?? promptTokens),
    contextTokens: finite(sample.contextTokens),
    promptTokens,
    generatedTokens: finite(sample.generatedTokens),
    prefillMs: finite(sample.prefillMs),
    decodeMs: finite(sample.decodeMs),
    endToEndMs: finite(sample.endToEndMs),
    ttftMs: finite(sample.ttftMs),
    latencyPerTokenMs: finite(sample.latencyPerTokenMs),
    decodeTokensPerSecond,
    userVisibleTokensPerSecond: tokensPerSecond,
    tokensPerSecond,
    transport: text(sample.transport, "direct"),
    normalizedCpuUtilizationPercent: finite(evidence?.normalizedCpuUtilizationPercent),
    workerWorkingSetBytes: finite(evidence?.peakWorkingSetBytes ?? evidence?.workingSetBytes),
    workerPrivateBytes: finite(evidence?.peakPrivateBytes ?? evidence?.privateBytes),
    tokensPerSecondPerFullCpu: finite(evidence?.tokensPerSecondPerFullCpu),
    evidenceMeasurementScope: text(evidence?.measurementScope),
    evidenceSampleSource: text(evidence?.sampleSource),
  };
  const filePath = historyPath(config);
  ensureDirectory(path.dirname(filePath));
  fs.appendFileSync(filePath, `${JSON.stringify(record)}\n`, "utf8");
  return record;
}

export function listNativeTurnMetrics(config, {
  userId = null,
  modelId = null,
  quant = null,
  runtimeProvider = null,
  optimizationProfile = null,
  context = null,
  limit = 100,
} = {}) {
  const filePath = historyPath(config);
  if (!fs.existsSync(filePath)) return [];
  let lines;
  try {
    lines = fs.readFileSync(filePath, "utf8").trim().split(/\r?\n/);
  } catch {
    return [];
  }
  const wantedUser = text(userId);
  const wantedModel = text(modelId);
  const wantedQuant = text(quant);
  const wantedProvider = text(runtimeProvider);
  const wantedOptimizationProfile = text(optimizationProfile);
  const wantedContext = text(context);
  const records = [];
  for (let index = lines.length - 1; index >= 0 && records.length < HISTORY_LIMIT; index -= 1) {
    try {
      const record = JSON.parse(lines[index]);
      if (wantedUser && record.userId !== wantedUser) continue;
      if (wantedModel && record.modelId !== wantedModel) continue;
      if (wantedQuant && record.quant !== wantedQuant) continue;
      if (wantedProvider && record.runtimeProvider !== wantedProvider) continue;
      if (wantedOptimizationProfile
        && text(record.optimizationProfile, "balanced") !== wantedOptimizationProfile) continue;
      if (wantedContext && record.contextBucket !== wantedContext) continue;
      records.push(record);
    } catch {
      // A torn final append must not make the rest of the local history unreadable.
    }
  }
  return records.slice(0, Math.max(1, Math.trunc(finite(limit) ?? 100)));
}

function median(values) {
  const sorted = values.filter((value) => Number.isFinite(value)).sort((a, b) => a - b);
  if (sorted.length === 0) return null;
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? (sorted[middle - 1] + sorted[middle]) / 2
    : sorted[middle];
}

export function summarizeNativeTurnMetrics(records = []) {
  const valid = records.filter((record) => finite(record?.tokensPerSecond) > 0);
  const rates = valid.map((record) => Number(record.tokensPerSecond));
  const latest = valid[0] ?? null;
  const decodeRates = valid.map((record) => finite(record.decodeTokensPerSecond)).filter(Number.isFinite);
  const endToEnd = valid.map((record) => finite(record.endToEndMs)).filter(Number.isFinite);
  const ttft = valid.map((record) => finite(record.ttftMs)).filter(Number.isFinite);
  const perToken = valid.map((record) => finite(record.latencyPerTokenMs)).filter(Number.isFinite);
  const normalizedCpu = valid.map((record) => finite(record.normalizedCpuUtilizationPercent)).filter(Number.isFinite);
  const workingSet = valid.map((record) => finite(record.workerWorkingSetBytes)).filter(Number.isFinite);
  const privateBytes = valid.map((record) => finite(record.workerPrivateBytes)).filter(Number.isFinite);
  const cpuEfficiency = valid.map((record) => finite(record.tokensPerSecondPerFullCpu)).filter(Number.isFinite);
  return {
    sampleCount: valid.length,
    latestTokensPerSecond: latest?.tokensPerSecond ?? null,
    medianTokensPerSecond: median(rates),
    bestTokensPerSecond: rates.length > 0 ? Math.max(...rates) : null,
    latestDecodeTokensPerSecond: finite(latest?.decodeTokensPerSecond),
    medianDecodeTokensPerSecond: median(decodeRates),
    bestDecodeTokensPerSecond: decodeRates.length > 0 ? Math.max(...decodeRates) : null,
    latestEndToEndMs: finite(latest?.endToEndMs),
    medianEndToEndMs: median(endToEnd),
    latestTtftMs: finite(latest?.ttftMs),
    medianTtftMs: median(ttft),
    latestLatencyPerTokenMs: finite(latest?.latencyPerTokenMs),
    medianLatencyPerTokenMs: median(perToken),
    latestNormalizedCpuUtilizationPercent: finite(latest?.normalizedCpuUtilizationPercent),
    medianNormalizedCpuUtilizationPercent: median(normalizedCpu),
    latestWorkerWorkingSetBytes: finite(latest?.workerWorkingSetBytes),
    medianWorkerWorkingSetBytes: median(workingSet),
    latestWorkerPrivateBytes: finite(latest?.workerPrivateBytes),
    medianWorkerPrivateBytes: median(privateBytes),
    latestTokensPerSecondPerFullCpu: finite(latest?.tokensPerSecondPerFullCpu),
    medianTokensPerSecondPerFullCpu: median(cpuEfficiency),
    latestTransport: latest?.transport ?? null,
    latestConcurrencyLevel: latest?.concurrencyLevel ?? null,
    latestServingPath: latest?.servingPath ?? null,
    latestRuntimeProvider: latest?.runtimeProvider ?? null,
    latestOptimizationProfile: latest?.optimizationProfile ?? "balanced",
    latestModelId: latest?.modelId ?? null,
  };
}
