// One authoritative native fit calculation + ranked placement candidates
// (First Big Refactor Phase 1, Detection stage D / Step 5). The fit report
// combines weights, context-dependent KV cache, workspace, graph/activation
// buffers, upload/readback staging, and a safety reserve into one place;
// the candidate ranking makes every rejection inspectable - the selected
// plan always states why each higher-ranked candidate was not chosen.
//
// The report is honest about confidence: "measured" only when the weight
// bytes and the KV inputs came from real metadata; anything padded with
// defaults reports "estimated".

import { estimateKvCacheBytes } from "./kv-cache/index.js";

const MB = 1024 * 1024;

function normalizeNumber(value) {
  if (value == null) {
    return null;
  }
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : null;
}

export const PLACEMENT_MODES = Object.freeze({
  GPU_FULL_RESIDENT: "gpu-full-resident",
  HYBRID_BOUNDED: "hybrid-bounded",
  CPU: "cpu",
});

/**
 * @param {object} options
 * @param {object} [options.gguf] catalog/inspection metadata (fileSizeBytes,
 *   blockCount, embeddingLength, contextLength, quant).
 * @param {number} [options.contextTokens] requested context; falls back to
 *   the GGUF's trained context, then a conservative 4096.
 * @param {number} [options.parallelSessions]
 * @param {object} [options.budgets] the execution plan's budgets object.
 */
export function buildModelFitReport({
  gguf = null,
  contextTokens = null,
  parallelSessions = 1,
  budgets = null,
} = {}) {
  const weightBytes = normalizeNumber(gguf?.fileSizeBytes);
  const layers = normalizeNumber(gguf?.blockCount);
  const hiddenSize = normalizeNumber(gguf?.embeddingLength);
  const trainedContext = normalizeNumber(gguf?.contextLength);
  const effectiveContext = normalizeNumber(contextTokens) ?? trainedContext ?? 4096;
  const sessions = Math.max(1, normalizeNumber(parallelSessions) ?? 1);

  // Native KV state is stored f32 today (KV q8_0 is roadmap work).
  const kvBytesPerSession = estimateKvCacheBytes({
    layers,
    hiddenSize,
    tokenCount: effectiveContext,
    bytesPerElement: 4,
  });
  const kvEstimated = kvBytesPerSession == null;
  // Conservative stand-in when the GGUF metadata is incomplete: 64 MiB per
  // 1k context tokens, per session.
  const kvBytes = (kvBytesPerSession ?? Math.trunc((effectiveContext / 1024) * 64 * MB)) * sessions;

  const workspaceBytes = normalizeNumber(budgets?.gpuWorkspaceBudgetBytes) ?? 256 * MB;
  // Activation/graph buffers scale with the hidden width; a handful of
  // hidden-sized f32 vectors per concurrent session plus dispatch overhead.
  const graphBytes = hiddenSize != null
    ? Math.max(64 * MB, Math.trunc(hiddenSize * 4 * 64 * sessions))
    : 64 * MB;
  const stagingBytes = 128 * MB; // persistent upload + readback + command buffers
  const safetyReserveBytes = 128 * MB;

  const totalGpuResidentBytes = weightBytes != null
    ? weightBytes + kvBytes + workspaceBytes + graphBytes + stagingBytes + safetyReserveBytes
    : null;
  const totalCpuBytes = weightBytes != null
    ? weightBytes + kvBytes + graphBytes
    : null;

  return {
    schemaVersion: 1,
    weights: {
      bytes: weightBytes,
      source: weightBytes != null ? "gguf-file-size" : null,
    },
    kvCache: {
      bytes: kvBytes,
      contextTokens: effectiveContext,
      perSessionBytes: kvBytesPerSession ?? null,
      estimated: kvEstimated,
    },
    workspace: { bytes: workspaceBytes },
    graph: { bytes: graphBytes },
    staging: { bytes: stagingBytes },
    safetyReserve: { bytes: safetyReserveBytes },
    parallelSessions: sessions,
    totalGpuResidentBytes,
    totalCpuBytes,
    confidence: weightBytes != null && !kvEstimated ? "measured" : "estimated",
  };
}

/**
 * Rank complete-GPU, bounded-hybrid, and CPU candidates against the fit and
 * the plan budgets. Returns every candidate (with a rejection reason on the
 * ones not usable) plus the selected mode - the first usable candidate in
 * rank order. CPU is the unconditional last resort per the product
 * guarantee; when it is memory-tight it stays selectable with a warning.
 */
export function rankPlacementCandidates({
  fit = null,
  budgets = null,
  gpuPresent = false,
  gpuValidated = null,
  uma = false,
} = {}) {
  const candidates = [];
  const gpuStateBudget = uma
    ? normalizeNumber(budgets?.unifiedBudgetBytes)
    : normalizeNumber(budgets?.gpuBudgetBytes);
  const cpuBudget = normalizeNumber(budgets?.cpuBudgetBytes);

  // 1. Full resident GPU: only when the complete required state safely fits.
  {
    let ok = false;
    let rejectionReason = null;
    if (!gpuPresent) {
      rejectionReason = "no usable GPU adapter is present";
    } else if (gpuValidated === false) {
      rejectionReason = "the selected adapter has not passed native validation";
    } else if (fit?.totalGpuResidentBytes == null) {
      rejectionReason = "the model's weight footprint is unknown; full residency cannot be proven safe";
    } else if (gpuStateBudget == null) {
      rejectionReason = "no GPU memory budget is available to prove the complete state fits";
    } else if (fit.totalGpuResidentBytes > gpuStateBudget) {
      rejectionReason = `complete GPU state ${Math.trunc(fit.totalGpuResidentBytes / MB)} MiB exceeds the budget ${Math.trunc(gpuStateBudget / MB)} MiB`;
    } else {
      ok = true;
    }
    candidates.push({ mode: PLACEMENT_MODES.GPU_FULL_RESIDENT, ok, rejectionReason });
  }

  // 2. Bounded hybrid: resident hot projections within the live weight
  // budget, per-layer CPU fallback. Usable whenever a validated GPU exists.
  {
    let ok = false;
    let rejectionReason = null;
    if (!gpuPresent) {
      rejectionReason = "no usable GPU adapter is present";
    } else if (gpuValidated === false) {
      rejectionReason = "the selected adapter has not passed native validation";
    } else {
      ok = true;
    }
    candidates.push({ mode: PLACEMENT_MODES.HYBRID_BOUNDED, ok, rejectionReason });
  }

  // 3. CPU: the unconditional fallback.
  {
    const warnings = [];
    if (fit?.totalCpuBytes != null && cpuBudget != null && fit.totalCpuBytes > cpuBudget) {
      warnings.push(
        `estimated CPU footprint ${Math.trunc(fit.totalCpuBytes / MB)} MiB exceeds the planner budget ${Math.trunc(cpuBudget / MB)} MiB; expect paging`,
      );
    }
    candidates.push({ mode: PLACEMENT_MODES.CPU, ok: true, rejectionReason: null, warnings });
  }

  const selected = candidates.find((candidate) => candidate.ok) ?? candidates[candidates.length - 1];
  return { candidates, selected: selected.mode };
}
