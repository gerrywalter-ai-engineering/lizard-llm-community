// JS side of `lizard-native-runner --probe-hardware` (First Big Refactor
// Phase 1, iteration 2). Spawns the compiled runner's probe mode and
// normalizes its JSON into the shared hardware contracts, so the capability
// registry can merge natively validated adapter facts (DXGI LUID, real UMA
// architecture, live memory budgets, compute self-test) into the same
// records the UI scan discovered.
//
// Every failure mode - no compiled runner, spawn failure, timeout, garbage
// output - returns null. The caller keeps the UI-scan-only snapshot and CPU
// execution stays available; a missing probe is a degraded-confidence fact,
// never an error that blocks onboarding.

import { execFileSync } from "node:child_process";
import { resolveCompiledNativeRunnerPath } from "../runtime/worker/process-client.js";
import { normalizeAdapterRecord, BACKEND_PROBE_STATES } from "./hardware-contracts.js";

const PROBE_TIMEOUT_MS = 20000;

function normalizeNumber(value) {
  if (value == null) {
    return null;
  }
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : null;
}

function normalizeProbeAdapter(raw = {}) {
  const d3d12 = raw.d3d12 ?? {};
  const deviceCreated = Boolean(d3d12.deviceCreated);
  const state = d3d12.state
    ?? (deviceCreated ? BACKEND_PROBE_STATES.FEATURE_LIMITED : BACKEND_PROBE_STATES.DEVICE_UNSUPPORTED);
  return normalizeAdapterRecord({
    identity: {
      luidHigh: raw.luidHigh,
      luidLow: raw.luidLow,
      vendorId: raw.vendorId,
      deviceId: raw.deviceId,
      name: raw.name,
    },
    memory: {
      dedicatedVideoMemoryBytes: normalizeNumber(raw.dedicatedVideoMemoryBytes),
      sharedSystemMemoryBytes: normalizeNumber(raw.sharedSystemMemoryBytes),
      // UMA facts come from the native D3D12_FEATURE_ARCHITECTURE query -
      // authoritative, unlike vendor/name inference.
      isUma: typeof d3d12.uma === "boolean" ? d3d12.uma : null,
      isCacheCoherentUma: typeof d3d12.cacheCoherentUma === "boolean" ? d3d12.cacheCoherentUma : null,
    },
    isSoftware: Boolean(raw.isSoftware),
    driverVersion: raw.driverVersion ?? null,
    validated: deviceCreated && state === BACKEND_PROBE_STATES.USABLE,
    probes: {
      d3d12: {
        state,
        deviceCreated,
        selfTest: d3d12.selfTest ?? null,
        detail: d3d12.detail ?? null,
        highestShaderModel: d3d12.highestShaderModel ?? null,
        waveOps: typeof d3d12.waveOps === "boolean" ? d3d12.waveOps : null,
        waveLaneCountMin: normalizeNumber(d3d12.waveLaneCountMin),
        // Live DXGI budgets (QueryVideoMemoryInfo) captured at probe time.
        videoMemory: raw.videoMemory ?? null,
      },
    },
  }, { source: "native-probe" });
}

/**
 * Normalize the raw probe JSON (already parsed) into contract shapes.
 * Exported separately so tests can exercise the mapping without spawning.
 */
export function normalizeHardwareProbeReport(raw) {
  if (!raw || typeof raw !== "object" || !Array.isArray(raw.adapters)) {
    return null;
  }
  return {
    schemaVersion: normalizeNumber(raw.schemaVersion) ?? 0,
    platform: raw.platform ?? "unknown",
    cpu: raw.cpu ?? null,
    memory: raw.memory ?? null,
    adapters: raw.adapters.map((adapter) => normalizeProbeAdapter(adapter)),
  };
}

/**
 * Run the compiled runner's `--probe-hardware` and return the normalized
 * report, or null when the probe is unavailable or fails.
 */
// One probe per process by default: the probe spawns the runner and executes
// a GPU self-test (~1 s), which must never run per request. Stage E
// revalidation triggers (driver change, device removal, resume) will force
// refreshes in a later iteration; `maxAgeMs` provides basic freshness.
let cachedProbe = null;

export function clearCachedNativeHardwareProbe() {
  cachedProbe = null;
}

export function getCachedNativeHardwareProbe({
  config = null,
  logger = null,
  maxAgeMs = 5 * 60 * 1000,
  force = false,
  runnerPath = null,
  exec = execFileSync,
  now = Date.now,
} = {}) {
  // Kill switch for deterministic test/CI runs and emergencies: the merged
  // registry then equals the plain UI-scan registry everywhere.
  if (process.env.LIZARD_NATIVE_HARDWARE_PROBE === "0") {
    return null;
  }
  const timestamp = now();
  if (!force && cachedProbe && (timestamp - cachedProbe.at) < maxAgeMs) {
    return cachedProbe.report;
  }
  const report = runNativeHardwareProbe({ config, logger, runnerPath, exec });
  // Failed probes are cached too: retrying a missing/broken binary on every
  // registry build would re-pay the spawn attempt for the same null.
  cachedProbe = { report, at: timestamp };
  return report;
}

export function runNativeHardwareProbe({
  config = null,
  runnerPath = null,
  exec = execFileSync,
  timeoutMs = PROBE_TIMEOUT_MS,
  logger = null,
} = {}) {
  const binary = runnerPath ?? resolveCompiledNativeRunnerPath(config);
  if (!binary) {
    logger?.info?.("native-hardware-probe-skipped", { reason: "no compiled runner binary" });
    return null;
  }
  let output;
  try {
    output = exec(binary, ["--probe-hardware"], {
      encoding: "utf8",
      timeout: timeoutMs,
      windowsHide: true,
      stdio: ["ignore", "pipe", "ignore"],
    });
  } catch (error) {
    logger?.warn?.("native-hardware-probe-failed", { message: String(error?.message ?? error).slice(0, 200) });
    return null;
  }
  try {
    const report = normalizeHardwareProbeReport(JSON.parse(String(output)));
    if (!report) {
      logger?.warn?.("native-hardware-probe-failed", { message: "probe output missing adapters" });
    }
    return report;
  } catch {
    logger?.warn?.("native-hardware-probe-failed", { message: "probe output was not valid JSON" });
    return null;
  }
}
