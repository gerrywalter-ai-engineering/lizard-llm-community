function finite(value) {
  if (value == null || value === "") return null;
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : null;
}

function sessionMetricsFrom(source) {
  return source?.nativeExecution?.sessionMetrics
    ?? source?.runtime?.nativeExecution?.sessionMetrics
    ?? source?.runtime?.kvCache?.workerSessionMetrics
    ?? source?.sessionMetrics
    ?? null;
}

// One additive contract shared by command output, chat, serve, benchmark
// artifacts, and the leaderboard. Older workers/results simply return null.
export function extractEvidenceMetrics(source = null) {
  const session = sessionMetricsFrom(source);
  const raw = source?.evidenceMetrics ?? session?.evidenceMetrics ?? null;
  if (!raw || typeof raw !== "object") return null;

  const normalizedCpuUtilizationPercent = finite(raw.normalizedCpuUtilizationPercent);
  const tokensPerSecond = finite(raw.tokensPerSecond)
    ?? finite(source?.nativeTokensPerSecond)
    ?? finite(source?.tokensPerSecond)
    ?? finite(source?.runtime?.tokensPerSecond);
  const tokensPerSecondPerFullCpu = finite(raw.tokensPerSecondPerFullCpu)
    ?? (tokensPerSecond > 0 && normalizedCpuUtilizationPercent > 0
      ? tokensPerSecond / (normalizedCpuUtilizationPercent / 100)
      : null);

  return {
    schemaVersion: Math.max(1, Math.trunc(finite(raw.schemaVersion) ?? 1)),
    measurementScope: String(raw.measurementScope ?? "decode"),
    sampleSource: String(raw.sampleSource ?? "native-worker-process"),
    tokensPerSecond,
    normalizedCpuUtilizationPercent,
    cpuSeconds: finite(raw.cpuSeconds),
    elapsedMs: finite(raw.elapsedMs),
    logicalCpuCount: finite(raw.logicalCpuCount),
    workingSetBytes: finite(raw.workingSetBytes),
    privateBytes: finite(raw.privateBytes),
    peakWorkingSetBytes: finite(raw.peakWorkingSetBytes) ?? finite(raw.workingSetBytes),
    peakPrivateBytes: finite(raw.peakPrivateBytes) ?? finite(raw.privateBytes),
    tokensPerSecondPerFullCpu,
  };
}

export function evidenceMetricFields(source = null) {
  const evidence = extractEvidenceMetrics(source);
  if (!evidence) return {};
  return {
    evidenceMetrics: evidence,
    normalizedCpuUtilizationPercent: evidence.normalizedCpuUtilizationPercent,
    workerWorkingSetBytes: evidence.peakWorkingSetBytes,
    workerPrivateBytes: evidence.peakPrivateBytes,
    tokensPerSecondPerFullCpu: evidence.tokensPerSecondPerFullCpu,
    evidenceMeasurementScope: evidence.measurementScope,
    evidenceSampleSource: evidence.sampleSource,
  };
}

export function formatEvidenceMetrics(evidence, { indent = "" } = {}) {
  const normalized = extractEvidenceMetrics({ evidenceMetrics: evidence });
  if (!normalized) return [];
  const mib = (bytes) => bytes == null ? null : bytes / (1024 * 1024);
  const parts = [];
  if (normalized.normalizedCpuUtilizationPercent != null) {
    parts.push(`${indent}Normalized CPU: ${normalized.normalizedCpuUtilizationPercent.toFixed(2)}%`);
  }
  if (normalized.peakWorkingSetBytes != null) {
    parts.push(`${indent}Worker working set: ${mib(normalized.peakWorkingSetBytes).toFixed(1)} MiB`);
  }
  if (normalized.peakPrivateBytes != null) {
    parts.push(`${indent}Worker private memory: ${mib(normalized.peakPrivateBytes).toFixed(1)} MiB`);
  }
  if (normalized.tokensPerSecondPerFullCpu != null) {
    parts.push(`${indent}CPU efficiency: ${normalized.tokensPerSecondPerFullCpu.toFixed(2)} tok/s per 100% CPU`);
  }
  parts.push(`${indent}Evidence: ${normalized.measurementScope} via ${normalized.sampleSource}`);
  return parts;
}

export function compactEvidenceMetricsLine(evidence) {
  const normalized = extractEvidenceMetrics({ evidenceMetrics: evidence });
  if (!normalized) return null;
  const mib = (bytes) => bytes == null ? null : bytes / (1024 * 1024);
  return [
    normalized.tokensPerSecond == null ? null : `decode=${normalized.tokensPerSecond.toFixed(2)} tok/s`,
    normalized.normalizedCpuUtilizationPercent == null
      ? null : `cpu=${normalized.normalizedCpuUtilizationPercent.toFixed(2)}%`,
    normalized.peakWorkingSetBytes == null
      ? null : `workingSet=${mib(normalized.peakWorkingSetBytes).toFixed(1)} MiB`,
    normalized.peakPrivateBytes == null
      ? null : `private=${mib(normalized.peakPrivateBytes).toFixed(1)} MiB`,
    normalized.tokensPerSecondPerFullCpu == null
      ? null : `efficiency=${normalized.tokensPerSecondPerFullCpu.toFixed(2)} tok/s/100%-CPU`,
    `scope=${normalized.measurementScope}`,
  ].filter(Boolean).join(" ");
}
