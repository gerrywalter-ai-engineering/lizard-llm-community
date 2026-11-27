import { extractEvidenceMetrics } from "./evidence-metrics.js";

function finiteNumber(value) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : null;
}

function nonEmptyString(value) {
  const normalized = String(value ?? "").trim();
  return normalized || null;
}

export function buildQuickChatExecutionTelemetry(result = null) {
  const nativeExecution = result?.nativeExecution ?? null;
  const metrics = nativeExecution?.sessionMetrics ?? null;
  const plan = nativeExecution?.executionPlan ?? null;
  const adapter = metrics?.gpuAdapter ?? null;
  const memory = metrics?.gpuMemory ?? null;
  const hardwareTuning = nativeExecution?.hardwareTuning ?? null;
  const evidenceMetrics = extractEvidenceMetrics(result);

  return {
    cpuBuildProfile: nonEmptyString(nativeExecution?.cpuBuildProfile),
    servingPath: nonEmptyString(nativeExecution?.servingPath)
      ?? (nativeExecution?.batched === true ? "batched" : "single-shot"),
    batched: nativeExecution?.batched === true,
    concurrencyLevel: finiteNumber(nativeExecution?.concurrencyLevel)
      ?? finiteNumber(metrics?.concurrencyLevel)
      ?? (nativeExecution?.batched === true ? null : 1),
    batchTargetSize: finiteNumber(nativeExecution?.batchTargetSize),
    concurrencyPolicy: nativeExecution?.concurrencyPolicy ?? null,
    plannedPlacement: nonEmptyString(plan?.selectedPlacement),
    planMode: nonEmptyString(plan?.mode),
    adapter: adapter
      ? {
        name: nonEmptyString(adapter.name),
        luidHigh: finiteNumber(adapter.luidHigh),
        luidLow: finiteNumber(adapter.luidLow),
        vendorId: finiteNumber(adapter.vendorId),
        deviceId: finiteNumber(adapter.deviceId),
        uma: typeof adapter.uma === "boolean" ? adapter.uma : null,
        selectionReason: nonEmptyString(adapter.selectionReason),
      }
      : null,
    prefillBackend: nonEmptyString(metrics?.prefillBackend) ?? "cpu",
    decodeBackend: nonEmptyString(metrics?.decodeBackend) ?? nonEmptyString(result?.nativeBackend) ?? "cpu",
    prefillDetail: nonEmptyString(metrics?.prefillBackendDetail),
    decodeDetail: nonEmptyString(metrics?.decodeBackendDetail),
    fallbackReason: nonEmptyString(result?.fallbackReason),
    gpuDecodePolicy: nonEmptyString(metrics?.gpuDecodePolicy),
    gpuMemory: memory
      ? {
        confidence: nonEmptyString(memory.confidence),
        localBudgetBytes: finiteNumber(memory.localBudgetBytes),
        localUsageBytes: finiteNumber(memory.localUsageBytes),
        reserveBytes: finiteNumber(memory.reserveBytes),
        requestedWeightBudgetBytes: finiteNumber(memory.requestedWeightBudgetBytes),
        effectiveWeightBudgetBytes: finiteNumber(memory.effectiveWeightBudgetBytes),
        residentWeightBytes: finiteNumber(memory.residentWeightBytes),
      }
      : null,
    gpuWeightCacheHits: finiteNumber(metrics?.gpuWeightCacheHits),
    gpuWeightCacheMisses: finiteNumber(metrics?.gpuWeightCacheMisses),
    gpuWeightCacheEvictions: finiteNumber(metrics?.gpuWeightCacheEvictions),
    gpuWeightUploadBytes: finiteNumber(metrics?.gpuWeightUploadBytes),
    gpuInputUploadBytes: finiteNumber(metrics?.gpuInputUploadBytes),
    gpuOutputReadbackBytes: finiteNumber(metrics?.gpuOutputReadbackBytes),
    gpuSubmissionCount: finiteNumber(metrics?.gpuSubmissionCount),
    evidenceMetrics,
    hardwareTuning: hardwareTuning
      ? {
        recommendedDecodeBackend: nonEmptyString(hardwareTuning.recommendedDecodeBackend),
        reason: nonEmptyString(hardwareTuning.reason),
        speedup: finiteNumber(hardwareTuning.speedup),
        source: nonEmptyString(hardwareTuning.source),
        recordedAt: nonEmptyString(hardwareTuning.recordedAt),
      }
      : null,
  };
}
