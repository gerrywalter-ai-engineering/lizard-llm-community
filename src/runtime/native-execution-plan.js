import { buildModelFitReport, rankPlacementCandidates } from "./native-model-fit.js";
import { selectPreferredAdapter } from "../lib/hardware-contracts.js";

const GIB = 1024 * 1024 * 1024;

function normalizeNumber(value) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : null;
}

function toBytesFromGb(value, ratio = 1) {
  const gb = normalizeNumber(value);
  if (gb == null || gb <= 0) {
    return null;
  }
  return Math.trunc(gb * GIB * ratio);
}

function detectPrimaryGpu(capabilityRegistry = null) {
  return Array.isArray(capabilityRegistry?.gpu) && capabilityRegistry.gpu.length > 0
    ? capabilityRegistry.gpu[0]
    : null;
}

function selectedAdapterForRuntime(capabilityRegistry, runtimeProvider) {
  const selected = capabilityRegistry?.selectedAdapter ?? null;
  if (runtimeProvider !== "caterpillar") {
    return selected;
  }

  // Caterpillar's current zero-copy state/KV layout uses CPU-mapped L0
  // write-back heaps. That layout is only supported when the native D3D12
  // architecture probe proves cache-coherent UMA. Do not pin Caterpillar to
  // the registry's normal dGPU preference on hybrid machines: choose from
  // the validated inventory by capabilities, never by Intel/AMD names.
  const inventory = capabilityRegistry?.adapterInventory;
  if (!Array.isArray(inventory)) {
    // Older/test registries without the native inventory retain their prior
    // behavior. The worker still performs the authoritative D3D12 check.
    return selected;
  }
  const compatible = inventory.filter((record) => (
    (record?.memory?.isUma ?? record?.isUma) === true
    && (record?.memory?.isCacheCoherentUma ?? record?.isCacheCoherentUma) === true
  ));
  const preferred = selectPreferredAdapter(compatible, { preferUma: true });
  if (!preferred) {
    return null;
  }
  return {
    identity: { ...preferred.identity },
    vendor: preferred.vendor,
    isUma: preferred.memory?.isUma ?? preferred.isUma ?? null,
    dedicatedVideoMemoryBytes: preferred.memory?.dedicatedVideoMemoryBytes
      ?? preferred.dedicatedVideoMemoryBytes
      ?? null,
    reason: "Caterpillar-compatible validated cache-coherent UMA adapter",
  };
}

function inferUma(primaryGpu = null) {
  if (!primaryGpu) {
    return false;
  }
  const vendor = String(primaryGpu.vendor ?? "").trim().toLowerCase();
  const name = String(primaryGpu.name ?? "").trim().toLowerCase();
  const vramGb = normalizeNumber(primaryGpu.vramGb ?? primaryGpu.vramGB);
  if (vendor === "intel") {
    return true;
  }
  return name.includes("iris")
    || name.includes("uhd")
    || name.includes("integrated")
    || (vramGb != null && vramGb <= 1.5);
}

function buildWarnings({
  backend,
  gpuPresent,
  uma,
  gpuVramGb,
  hybridPrefillRequested,
  windowsHybridCapable,
  forceCpuOnly = false,
}) {
  const warnings = [];
  if (forceCpuOnly) {
    warnings.push("CPU-only execution plan was forced by nativeRuntime.forceCpuOnlyPlan.");
    return warnings;
  }
  if (!gpuPresent) {
    warnings.push("No GPU was detected in the capability registry. Native execution should remain CPU-only.");
    return warnings;
  }
  if (backend === "cpu" && !hybridPrefillRequested && !windowsHybridCapable) {
    warnings.push("Requested native backend is CPU. GPU planning data is advisory only.");
  }
  if (uma) {
    warnings.push("Integrated/UMA GPU memory should be planned as a shared system-memory budget, not as isolated VRAM.");
  }
  if (gpuVramGb != null && gpuVramGb <= 2) {
    warnings.push("2 GB or smaller GPU mode must stay hybrid: keep GGUF weights memory-mapped on CPU and reserve GPU memory for transient projections, hot layers, and staging buffers.");
  }
  return warnings;
}

// Discrete (non-UMA) GPUs whose dedicated VRAM is smaller than the model
// normally demote decode to CPU (the Phase 2 "no-thrash" policy) to avoid
// cycling weights through the resident cache every token. This opt-in wrapper
// instead keeps decode on the D3D12 per-layer resident lane, which streams and
// evicts weights within the live VRAM budget - so a model far larger than the
// card (verified on a 2 GB adapter, designed up to 256 GB) still decodes on the
// GPU over DirectX. It is deliberately separate from the Iris/UMA
// complete-token-graph lane (that path is untouched) and only engages when GPU
// decode is explicitly REQUIRED (e.g. the remote DirectX benchmark) or opted in
// via nativeRuntime.allowStreamingGpuDecode; ordinary dedicated-card runs keep
// the no-thrash CPU decode default.
export function resolveStreamingDiscreteDecode({
  allowGpuPrefill = false,
  oversizedForResidentDecode = false,
  uma = false,
  gpuDecodePolicy = null,
  allowStreamingGpuDecode = false,
} = {}) {
  const optedIn = gpuDecodePolicy === "required" || allowStreamingGpuDecode === true;
  const stream = allowGpuPrefill === true
    && oversizedForResidentDecode === true
    && uma !== true
    && optedIn;
  return {
    stream,
    reason: stream
      ? "Discrete GPU decode streams weights through the D3D12 resident lane (bounded VRAM, per-token eviction) because GPU decode was required for a model larger than the resident weight budget."
      : null,
  };
}

export function buildNativeExecutionPlan({
  capabilityRegistry = null,
  profile = null,
  gguf = null,
  requestedBackend = null,
  runtimeProvider = "lizard-native",
} = {}) {
  const backend = String(requestedBackend ?? profile?.backend ?? "cpu").trim().toLowerCase() || "cpu";
  const forceCpuOnly = Boolean(profile?.nativeRuntime?.forceCpuOnlyPlan);
  // Explicit adapter selection from the merged registry (Phase 1 Finding 1)
  // takes precedence over gpu[0]. Its identity is carried through the plan
  // into the worker launch so DXGI provably selects the same physical GPU.
  const selectedAdapter = selectedAdapterForRuntime(capabilityRegistry, runtimeProvider);
  // Once a native inventory exists, a missing Caterpillar selection means
  // the probe found no adapter compatible with its current memory layout.
  // Do not fall back to the unvalidated legacy gpu[0] heuristic in that case.
  const providerSelectionRequired = runtimeProvider === "caterpillar"
    && Array.isArray(capabilityRegistry?.adapterInventory);
  const primaryGpu = providerSelectionRequired ? null : detectPrimaryGpu(capabilityRegistry);
  const gpuPresent = Boolean(selectedAdapter ?? primaryGpu) && !forceCpuOnly;
  const gpuVramGb = selectedAdapter?.dedicatedVideoMemoryBytes != null
    ? Number((selectedAdapter.dedicatedVideoMemoryBytes / GIB).toFixed(2))
    : normalizeNumber(primaryGpu?.vramGb ?? primaryGpu?.vramGB);
  const systemAvailableGb = normalizeNumber(capabilityRegistry?.memory?.availableGb);
  // Native architecture facts beat the vendor/name heuristic when present.
  const uma = typeof selectedAdapter?.isUma === "boolean"
    ? selectedAdapter.isUma
    : inferUma(primaryGpu);
  const lowVram = gpuVramGb != null && gpuVramGb <= 2;
  const hybridPrefillRequested = Boolean(profile?.nativeRuntime?.enableHybridGpuPrefill);
  const windowsHybridCapable = capabilityRegistry?.os == null
    ? true
    : capabilityRegistry.os === "windows";

  const mode = !gpuPresent
    ? "cpu-only"
    : backend === "cpu"
      ? windowsHybridCapable
        ? lowVram || uma
          ? "hybrid-low-vram"
          : "hybrid"
        : "cpu-only"
    : lowVram || uma
      ? "hybrid-low-vram"
      : "hybrid";

  const prefillChunkTokens = mode === "cpu-only"
    ? (systemAvailableGb != null && systemAvailableGb < 12 ? 32 : 64)
    : lowVram
      ? 32
      : 64;
  const microBatchTokens = mode === "cpu-only"
    ? 32
    : lowVram
      ? 8
      : 16;
  // Resident quantized GPU weights make short-prompt offload profitable, so
  // the gate only needs to filter out trivial prompts.
  const minHybridGpuPrefillTokens = mode === "hybrid-low-vram"
    ? 12
    : mode === "hybrid"
      ? 8
      : 0;

  // Stage D: one authoritative fit calculation + ranked candidates with
  // rejection reasons, when enough model metadata is available. Legacy
  // callers without gguf metadata get fit: null and no candidate list.
  const resolvedGguf = gguf ?? profile?.gguf ?? null;
  const budgets = {
    unifiedBudgetBytes: uma ? toBytesFromGb(systemAvailableGb, 0.7) : null,
    cpuBudgetBytes: toBytesFromGb(systemAvailableGb, uma ? 0.55 : 0.75),
    gpuBudgetBytes: toBytesFromGb(gpuVramGb, 0.75),
    gpuWorkspaceBudgetBytes: toBytesFromGb(gpuVramGb, lowVram ? 0.2 : 0.3),
    gpuKvBudgetBytes: toBytesFromGb(gpuVramGb, lowVram ? 0.1 : 0.2),
  };
  const fit = resolvedGguf?.fileSizeBytes != null || resolvedGguf?.blockCount != null
    ? buildModelFitReport({
      gguf: resolvedGguf,
      contextTokens: profile?.contextSize ?? null,
      budgets,
    })
    : null;
  const placementRanking = fit
    ? rankPlacementCandidates({
      fit,
      budgets,
      gpuPresent,
      gpuValidated: selectedAdapter ? true : null,
      uma,
    })
    : null;
  const selectedPlacement = placementRanking?.selected ?? null;
  const residentWeightBudgetBytes = uma
    ? budgets.unifiedBudgetBytes
    : Math.max(
      0,
      (budgets.gpuBudgetBytes ?? 0)
        - (budgets.gpuWorkspaceBudgetBytes ?? 0)
        - (budgets.gpuKvBudgetBytes ?? 0),
    );
  const fullGpuSelected = selectedPlacement === "gpu-full-resident";
  const cpuSelected = selectedPlacement === "cpu";
  const oversizedForResidentDecode = fit?.weights?.bytes != null
    && residentWeightBudgetBytes != null
    && fit.weights.bytes > residentWeightBudgetBytes;
  const allowGpuPrefill = mode !== "cpu-only" && !cpuSelected;
  // Opt-in streaming decode for a discrete GPU smaller than the model. Keeps
  // the Iris/UMA and default no-thrash paths below unchanged.
  const gpuDecodePolicy = String(profile?.nativeRuntime?.gpuDecodePolicy ?? "").trim().toLowerCase() || null;
  const allowStreamingGpuDecode = profile?.nativeRuntime?.allowStreamingGpuDecode === true;
  const streamingDiscreteDecode = resolveStreamingDiscreteDecode({
    allowGpuPrefill,
    oversizedForResidentDecode,
    uma,
    gpuDecodePolicy,
    allowStreamingGpuDecode,
  });
  const allowGpuDecode = allowGpuPrefill
    && (fullGpuSelected || !oversizedForResidentDecode || streamingDiscreteDecode.stream);
  const decodeStrategy = !allowGpuDecode
    ? "cpu-no-thrash"
    : fullGpuSelected
      ? "gpu-full-resident"
      : streamingDiscreteDecode.stream
        ? "gpu-streaming-resident"
        : "gpu-bounded-resident";
  const decodeReason = !allowGpuPrefill
    ? "CPU placement was selected or no validated GPU is available."
    : streamingDiscreteDecode.stream
      ? streamingDiscreteDecode.reason
      : oversizedForResidentDecode
        ? `Model weights exceed the ${Math.trunc(residentWeightBudgetBytes / 1024 / 1024)} MiB resident decode budget; CPU decode avoids per-token GPU weight eviction and re-upload.`
        : fullGpuSelected
          ? "The complete model, KV cache, graph, and workspace fit the planned GPU budget."
          : "The bounded resident decode set fits the planned GPU weight budget.";

  return {
    backend,
    mode,
    architecture: gguf?.architecture ?? profile?.gguf?.architecture ?? null,
    quantization: gguf?.quant ?? profile?.gguf?.quant ?? null,
    fit,
    placementCandidates: placementRanking?.candidates ?? null,
    selectedPlacement,
    hardware: {
      gpuName: selectedAdapter?.identity?.name ?? primaryGpu?.name ?? null,
      gpuVendor: selectedAdapter?.vendor ?? primaryGpu?.vendor ?? null,
      gpuVramGb,
      unifiedMemory: uma,
      systemAvailableGb,
      // Stable DXGI identity of the planned adapter; null while planning
      // from a registry without native validation (legacy path).
      adapterIdentity: selectedAdapter?.identity ? { ...selectedAdapter.identity } : null,
    },
    budgets,
    placement: {
      weights: mode === "cpu-only"
        ? "cpu-mmap"
        : lowVram
          ? "cpu-mmap-plus-staged-hot-layers"
          : "partial-gpu-offload",
      kvCache: mode === "cpu-only"
        ? "cpu"
        : lowVram
          ? "cpu-or-shared-q8-first"
          : "hybrid",
      activations: mode === "cpu-only" ? "cpu" : "gpu-transient",
      logits: "cpu",
    },
    execution: {
      prefillChunkTokens,
      microBatchTokens,
      decodeBatchTokens: 1,
      minHybridGpuPrefillTokens,
      // `lowVram` is derived from the adapter's DEDICATED video memory. On a
      // UMA part that number is a driver carve-out, not a capability: an Iris
      // Xe reports 128 MiB dedicated while addressing ~8 GiB of shared system
      // memory, and every other budget in this plan already accounts for that
      // through unifiedBudgetBytes. Keying the layer count off lowVram alone
      // therefore staged ONE layer onto a GPU the same plan had just proven
      // could hold the complete model - decodeStrategy read
      // "gpu-full-resident" while 27 of 28 layers decoded on the CPU, which is
      // the slow single-stream path this run exists to fix. Discrete cards
      // that genuinely have little VRAM keep the 1-layer no-thrash behaviour.
      recommendedGpuLayers: mode === "cpu-only"
        ? 0
        : fullGpuSelected || (uma && !oversizedForResidentDecode)
          ? "auto"
          : lowVram
            ? 1
            : "auto",
      keepDeviceStateResident: mode !== "cpu-only",
      prefillStrategy: allowGpuPrefill ? "auto-hybrid" : "cpu",
      decodeStrategy,
      decodeReason,
      allowGpuPrefill,
      allowGpuDecode,
      // True only on the new discrete-GPU streaming lane (D3D12 decode of a
      // model larger than VRAM via per-token weight eviction). Iris/UMA and
      // full-resident runs leave this false.
      streamingResidentDecode: streamingDiscreteDecode.stream,
      gpuDecodePolicy,
      // On UMA the bounded-resident and full-resident lanes draw from the
      // same unified pool, so the all-or-nothing complete token graph can
      // stage within the identical LRU budget the bounded path uses — and
      // measured 2026-07-12 on Iris Xe it is 17-24% faster than the
      // per-layer lane it was demoted to (199/199 graphs stable,
      // byte-identical output). Dedicated small-VRAM cards keep the
      // Phase 2 no-thrash demotion unchanged.
      allowCompleteTokenGraph: allowGpuDecode
        && (fullGpuSelected || (uma === true && !oversizedForResidentDecode)),
      residentWeightBudgetBytes,
      fallbackPolicy: "explicit-and-inspectable",
    },
    warnings: buildWarnings({
      backend,
      gpuPresent,
      uma,
      gpuVramGb,
      hybridPrefillRequested,
      windowsHybridCapable,
      forceCpuOnly,
    }),
  };
}
