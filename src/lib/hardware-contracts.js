// Versioned hardware-detection contracts (First Big Refactor Phase 1,
// .Agent/firstBgRefactor/01_firstPhaseBigRefactor.md, Step 1).
//
// One adapter identity and one status vocabulary shared by the UI scan,
// capability registry, native execution plan, worker RPC, health endpoint,
// onboarding, and benchmarks. The existing src/lib/hardware.js scan remains
// the product-level entry point; these contracts describe how its records
// are normalized, merged with native validation results, and consumed.
//
// Two levels of truth are kept per adapter (never collapsed):
//   discovered  - a UI-level scan (nvidia-smi, Win32_VideoController, lspci,
//                 system_profiler) found the adapter;
//   validated   - the native runner proved a backend can execute on it
//                 (device creation + feature probe + compute self-test).
// Only validated, non-software adapters may be selected automatically.

export const HARDWARE_CONTRACT_VERSION = 1;

// UI state progression for the hardware view (doc §"Existing UI hardware
// detection is the foundation"). Detection failures must never blank
// onboarding - they resolve to CPU_FALLBACK_READY or ACTION_REQUIRED.
export const HARDWARE_VALIDATION_STATES = Object.freeze({
  DETECTING: "detecting",
  VALIDATING: "validating",
  READY: "ready",
  CPU_FALLBACK_READY: "cpu-fallback-ready",
  ACTION_REQUIRED: "action-required",
});

// Backend probe outcome vocabulary (doc §"Detection stage C").
export const BACKEND_PROBE_STATES = Object.freeze({
  NOT_PACKAGED: "not-packaged",
  DRIVER_UNAVAILABLE: "driver-unavailable",
  DEVICE_UNSUPPORTED: "device-unsupported",
  FEATURE_LIMITED: "feature-limited",
  USABLE: "usable",
});

const PCI_VENDOR_LABELS = Object.freeze({
  0x10de: "nvidia",
  0x1002: "amd",
  0x1022: "amd",
  0x8086: "intel",
  0x106b: "apple",
  0x1414: "microsoft",
});

function normalizeNumber(value) {
  if (value == null) {
    // Number(null) === 0 - an absent LUID/memory field must stay absent,
    // not silently become zero.
    return null;
  }
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : null;
}

function normalizeName(value) {
  return String(value ?? "").trim().replace(/\s+/g, " ");
}

function nameKey(value) {
  return normalizeName(value).toLowerCase();
}

export function vendorLabelFromPciId(vendorId) {
  const id = normalizeNumber(vendorId);
  return id != null ? (PCI_VENDOR_LABELS[id] ?? "unknown") : "unknown";
}

export function vendorLabelFromName(name) {
  const lower = nameKey(name);
  if (lower.includes("nvidia") || lower.includes("geforce") || lower.includes("quadro")) return "nvidia";
  if (lower.includes("amd") || lower.includes("radeon")) return "amd";
  if (lower.includes("intel") || lower.includes("iris") || lower.includes("uhd graphics")) return "intel";
  if (lower.includes("apple")) return "apple";
  if (lower.includes("microsoft basic render")) return "microsoft";
  return "unknown";
}

export function hasAdapterLuid(record) {
  const identity = record?.identity ?? record;
  return normalizeNumber(identity?.luidHigh) != null && normalizeNumber(identity?.luidLow) != null;
}

// Stable identity key. DXGI LUID when present (stable within Windows
// graphics APIs for the current boot); otherwise PCI ids + name. Records
// without a LUID can collide by design - the merge disambiguates them by
// position and they are never auto-selectable anyway (not validated).
export function adapterIdentityKey(record) {
  const identity = record?.identity ?? record ?? {};
  if (hasAdapterLuid(identity)) {
    return `luid:${normalizeNumber(identity.luidHigh)}:${normalizeNumber(identity.luidLow)}`;
  }
  const vendorId = normalizeNumber(identity.vendorId);
  const deviceId = normalizeNumber(identity.deviceId);
  return `pci:${vendorId ?? "?"}:${deviceId ?? "?"}:${nameKey(identity.name)}`;
}

export function sameAdapterIdentity(a, b) {
  // LUID comparison never falls back to weaker fields: two same-name
  // adapters with different LUIDs are different devices.
  if (hasAdapterLuid(a) || hasAdapterLuid(b)) {
    return hasAdapterLuid(a) && hasAdapterLuid(b) && adapterIdentityKey(a) === adapterIdentityKey(b);
  }
  return adapterIdentityKey(a) === adapterIdentityKey(b);
}

/**
 * Canonical adapter record. Every producer (UI scan sources, the native
 * DXGI probe) normalizes into this shape before merging.
 */
export function normalizeAdapterRecord(raw = {}, { source = "unknown" } = {}) {
  const identity = raw.identity ?? raw;
  const vendorId = normalizeNumber(identity.vendorId);
  const name = normalizeName(identity.name ?? raw.name);
  const vendor = raw.vendor
    ?? (vendorId != null ? vendorLabelFromPciId(vendorId) : vendorLabelFromName(name));
  const dedicatedBytes = normalizeNumber(
    raw.memory?.dedicatedVideoMemoryBytes ?? raw.dedicatedVideoMemoryBytes,
  );
  const vramGb = normalizeNumber(raw.vramGb ?? raw.vramGB);
  return {
    schemaVersion: HARDWARE_CONTRACT_VERSION,
    identity: {
      luidHigh: normalizeNumber(identity.luidHigh),
      luidLow: normalizeNumber(identity.luidLow),
      vendorId,
      deviceId: normalizeNumber(identity.deviceId),
      name,
      nvidiaUuid: identity.nvidiaUuid ?? raw.nvidiaUuid ?? null,
    },
    vendor,
    memory: {
      dedicatedVideoMemoryBytes: dedicatedBytes ?? (vramGb != null ? Math.trunc(vramGb * 1024 * 1024 * 1024) : null),
      sharedSystemMemoryBytes: normalizeNumber(
        raw.memory?.sharedSystemMemoryBytes ?? raw.sharedSystemMemoryBytes,
      ),
      isUma: raw.memory?.isUma ?? raw.isUma ?? null,
      isCacheCoherentUma: raw.memory?.isCacheCoherentUma ?? raw.isCacheCoherentUma ?? null,
    },
    driver: {
      version: raw.driver?.version ?? raw.driverVersion ?? null,
    },
    isSoftware: Boolean(raw.isSoftware),
    discovered: raw.discovered ?? true,
    validated: Boolean(raw.validated),
    sources: Array.isArray(raw.sources) && raw.sources.length > 0 ? [...raw.sources] : [source],
    probes: { ...(raw.probes ?? {}) },
    validation: {
      state: raw.validation?.state
        ?? (raw.validated ? HARDWARE_VALIDATION_STATES.READY : HARDWARE_VALIDATION_STATES.DETECTING),
      reason: raw.validation?.reason ?? null,
    },
  };
}

function attachUiMetadata(nativeRecord, uiRecord) {
  const merged = nativeRecord;
  merged.sources = [...new Set([...merged.sources, ...uiRecord.sources])];
  merged.discovered = true;
  if (!merged.driver.version && uiRecord.driver.version) {
    merged.driver.version = uiRecord.driver.version;
  }
  if (!merged.identity.nvidiaUuid && uiRecord.identity.nvidiaUuid) {
    merged.identity.nvidiaUuid = uiRecord.identity.nvidiaUuid;
  }
  if (merged.memory.dedicatedVideoMemoryBytes == null && uiRecord.memory.dedicatedVideoMemoryBytes != null) {
    merged.memory.dedicatedVideoMemoryBytes = uiRecord.memory.dedicatedVideoMemoryBytes;
  }
  for (const [backend, probe] of Object.entries(uiRecord.probes)) {
    if (merged.probes[backend] == null) {
      merged.probes[backend] = probe;
    }
  }
  return merged;
}

/**
 * Merge UI-discovered adapters with native (DXGI-level) adapter records.
 *
 * Native records are canonical: each one is a distinct physical/logical
 * adapter (distinct LUIDs never collapse, even with identical names). UI
 * records enrich the matching native record with vendor-tool metadata
 * (nvidia-smi driver version/UUID, backend hints). UI records with no
 * native counterpart remain in the inventory as discovered-only entries -
 * visible in the UI, never auto-selectable.
 *
 * Match rule, in order:
 *   1. exact normalized-name match with the same vendor (first unclaimed);
 *   2. sole-vendor match: exactly one unclaimed native record and one UI
 *      record of that vendor (names from different tools rarely align).
 */
export function mergeAdapterInventories({ uiAdapters = [], nativeAdapters = [] } = {}) {
  const native = nativeAdapters.map((record) => normalizeAdapterRecord(record, {
    source: record?.sources?.[0] ?? "native-probe",
  }));
  const claimed = new Set();
  const unmatchedUi = [];

  const uiNormalized = uiAdapters.map((record) => normalizeAdapterRecord(record, {
    source: record?.sources?.[0] ?? "ui-scan",
  }));

  for (const uiRecord of uiNormalized) {
    let matchIndex = native.findIndex((candidate, index) => !claimed.has(index)
      && nameKey(candidate.identity.name) === nameKey(uiRecord.identity.name)
      && (candidate.vendor === uiRecord.vendor || candidate.vendor === "unknown" || uiRecord.vendor === "unknown"));
    if (matchIndex === -1 && uiRecord.vendor !== "unknown") {
      const vendorCandidates = native
        .map((candidate, index) => ({ candidate, index }))
        .filter(({ candidate, index }) => !claimed.has(index) && candidate.vendor === uiRecord.vendor);
      const uiVendorCount = uiNormalized.filter((record) => record.vendor === uiRecord.vendor).length;
      if (vendorCandidates.length === 1 && uiVendorCount === 1) {
        matchIndex = vendorCandidates[0].index;
      }
    }
    if (matchIndex !== -1) {
      claimed.add(matchIndex);
      attachUiMetadata(native[matchIndex], uiRecord);
    } else {
      unmatchedUi.push(uiRecord);
    }
  }

  return [...native, ...unmatchedUi];
}

/**
 * Adapters eligible for automatic acceleration: natively validated,
 * hardware (not software/WARP), with a passed d3d12 self-test when a d3d12
 * probe is present. Discovered-only records never qualify.
 */
export function selectableAdapters(inventory = []) {
  return inventory.filter((record) => record.validated
    && !record.isSoftware
    && (record.probes?.d3d12 == null || record.probes.d3d12.state === BACKEND_PROBE_STATES.USABLE));
}

/**
 * Deterministic preferred-adapter policy (doc §Finding 1, step 7 of the
 * implementation work): only selectable (validated, hardware) adapters are
 * candidates; enumeration order never decides.
 *
 *  - preferUma === true  -> prefer UMA adapters (a UMA plan wants the
 *    adapter that shares system memory);
 *  - preferUma === false -> prefer dedicated-VRAM adapters (a dGPU plan);
 *  - default             -> dedicated-VRAM adapters first (highest dedicated
 *    memory), UMA adapters after, ties broken by stable identity key.
 */
export function selectPreferredAdapter(inventory = [], { preferUma = null } = {}) {
  const candidates = selectableAdapters(inventory);
  if (candidates.length === 0) {
    return null;
  }
  const dedicatedBytes = (record) => record.memory.dedicatedVideoMemoryBytes ?? 0;
  const isUma = (record) => record.memory.isUma === true;
  const sorted = [...candidates].sort((a, b) => {
    if (preferUma === true && isUma(a) !== isUma(b)) {
      return isUma(a) ? -1 : 1;
    }
    if (preferUma !== true && isUma(a) !== isUma(b)) {
      return isUma(a) ? 1 : -1;
    }
    if (dedicatedBytes(a) !== dedicatedBytes(b)) {
      return dedicatedBytes(b) - dedicatedBytes(a);
    }
    return adapterIdentityKey(a) < adapterIdentityKey(b) ? -1 : 1;
  });
  return sorted[0];
}

// ---- CPU runner tiers (doc §"Detection stage A") ----
//
// Instruction dispatch requires CPUID *and* operating-system state (XGETBV
// for AVX-family registers). A feature counts only when both are true.
// The ladder is ordered lowest -> highest; selection walks down from the
// highest tier whose requirements are all enabled.

export const CPU_RUNNER_TIERS = Object.freeze(["baseline", "avx", "avx2", "avx512"]);

const TIER_REQUIREMENTS = Object.freeze({
  baseline: [],
  avx: ["avx"],
  avx2: ["avx", "avx2", "fma"],
  // The Run 06 pinned ggml backend compiles the AVX-512 package with
  // F/BW/VL/VNNI. Requiring only AVX512F would dispatch it to CPUs that trap on
  // VNNI (or BW/VL) before the process-level self-test can report a profile.
  avx512: ["avx", "avx2", "fma", "avx512f", "avx512bw", "avx512vl", "avx512Vnni"],
});

function featureEnabled(features, name) {
  const feature = features?.[name];
  if (feature == null) {
    return false;
  }
  if (typeof feature === "boolean") {
    // A bare boolean cannot prove OS-enabled register state; treat it as
    // CPU-only evidence, which is insufficient for AVX-family dispatch.
    return false;
  }
  return Boolean(feature.cpu) && Boolean(feature.os);
}

/**
 * @param {object} features map of feature name -> { cpu: bool, os: bool }
 * @param {string[]} [disabledTiers] tiers whose startup self-test failed
 *   this launch (doc: retry the next lower tier).
 */
export function selectCpuRunnerTier(features = {}, { disabledTiers = [] } = {}) {
  const disabled = new Set(disabledTiers);
  for (let index = CPU_RUNNER_TIERS.length - 1; index >= 0; index -= 1) {
    const tier = CPU_RUNNER_TIERS[index];
    if (disabled.has(tier)) {
      continue;
    }
    const requirements = TIER_REQUIREMENTS[tier];
    if (requirements.every((name) => featureEnabled(features, name))) {
      return tier;
    }
  }
  // Baseline is the unconditional fallback even when listed as disabled -
  // there is nothing lower to retry.
  return "baseline";
}
