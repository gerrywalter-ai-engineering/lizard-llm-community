import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";

import { ensureDirectory } from "../lib/paths.js";

const SCHEMA_VERSION = 1;
const MAX_HINT_AGE_MS = 30 * 24 * 60 * 60 * 1000;

function finiteNumber(value) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : null;
}

function contextClass(value) {
  const tokens = finiteNumber(value) ?? 4096;
  if (tokens <= 2048) return "ctx-2k";
  if (tokens <= 4096) return "ctx-4k";
  if (tokens <= 8192) return "ctx-8k";
  return "ctx-large";
}

function memoryBucket(value) {
  const bytes = finiteNumber(value);
  if (bytes == null) return "unknown";
  const gib = bytes / 1024 / 1024 / 1024;
  return `${Math.max(1, Math.floor(gib / 2) * 2)}g`;
}

export function buildHardwareTuningIdentity({ profile = null, modelArtifact = null, executionPlan = null } = {}) {
  const registry = profile?.capabilityRegistry ?? {};
  const adapter = registry.selectedAdapter ?? {};
  const identity = adapter.identity ?? executionPlan?.hardware?.adapterIdentity ?? {};
  const raw = {
    controller: "lizard-native-phase2-v1",
    os: registry.os ?? process.platform,
    arch: registry.arch ?? process.arch,
    cpu: registry.cpu?.name ?? null,
    cpuTier: registry.cpu?.runnerTier ?? "baseline",
    adapterLuid: identity.luidHigh != null && identity.luidLow != null
      ? `${identity.luidHigh}:${identity.luidLow}`
      : null,
    adapterVendorId: identity.vendorId ?? null,
    adapterDeviceId: identity.deviceId ?? null,
    adapterName: identity.name ?? executionPlan?.hardware?.gpuName ?? null,
    driver: adapter.driver?.version ?? null,
    uma: adapter.isUma ?? executionPlan?.hardware?.unifiedMemory ?? null,
    modelId: modelArtifact?.modelId ?? profile?.modelId ?? null,
    modelBytes: modelArtifact?.gguf?.fileSizeBytes ?? profile?.gguf?.fileSizeBytes ?? null,
    architecture: modelArtifact?.gguf?.architecture ?? profile?.gguf?.architecture ?? null,
    quant: modelArtifact?.gguf?.quant ?? profile?.gguf?.quant ?? null,
    contextClass: contextClass(profile?.contextSize),
    memoryBucket: memoryBucket(executionPlan?.execution?.residentWeightBudgetBytes),
  };
  return {
    key: crypto.createHash("sha256").update(JSON.stringify(raw)).digest("hex"),
    facts: raw,
  };
}

export function evaluateHardwareCalibration(calibration, { minimumGpuSpeedup = 1.1 } = {}) {
  const cpuMs = finiteNumber(calibration?.cpuMsPerIteration);
  const gpuMs = finiteNumber(calibration?.gpuMsPerIteration);
  const parityPassed = calibration?.parityPassed === true;
  if (cpuMs == null || gpuMs == null || cpuMs <= 0 || gpuMs <= 0) {
    return {
      recommendedDecodeBackend: "cpu",
      reason: calibration?.reason ?? "Calibration did not produce comparable CPU and GPU timings.",
      speedup: null,
    };
  }
  const speedup = cpuMs / gpuMs;
  if (!parityPassed) {
    return {
      recommendedDecodeBackend: "cpu",
      reason: "GPU calibration output did not pass the CPU parity check.",
      speedup,
    };
  }
  if (speedup < minimumGpuSpeedup) {
    return {
      recommendedDecodeBackend: "cpu",
      reason: `GPU calibration speedup ${speedup.toFixed(2)}x is below the required ${minimumGpuSpeedup.toFixed(2)}x.`,
      speedup,
    };
  }
  return {
    recommendedDecodeBackend: "gpu",
    reason: `GPU calibration passed parity and measured ${speedup.toFixed(2)}x CPU speed.`,
    speedup,
  };
}

export function applyHardwareTuningHint(executionPlan, hint) {
  if (!executionPlan || !hint || hint.recommendedDecodeBackend !== "cpu") {
    return executionPlan;
  }
  return {
    ...executionPlan,
    execution: {
      ...executionPlan.execution,
      allowGpuDecode: false,
      allowCompleteTokenGraph: false,
      decodeStrategy: "cpu-calibrated",
      decodeReason: hint.reason,
      hardwareTuning: hint,
    },
  };
}

function tuningPath(config) {
  return config?.runtimeDirectory ? path.join(config.runtimeDirectory, "hardware-tuning.json") : null;
}

function readStore(config) {
  const filePath = tuningPath(config);
  if (!filePath || !fs.existsSync(filePath)) return { schemaVersion: SCHEMA_VERSION, hints: {} };
  try {
    const parsed = JSON.parse(fs.readFileSync(filePath, "utf8"));
    return parsed?.schemaVersion === SCHEMA_VERSION && parsed.hints
      ? parsed
      : { schemaVersion: SCHEMA_VERSION, hints: {} };
  } catch {
    return { schemaVersion: SCHEMA_VERSION, hints: {} };
  }
}

export function loadHardwareTuningHint(config, identity, { now = Date.now() } = {}) {
  if (!identity?.key) return null;
  const hint = readStore(config).hints[identity.key] ?? null;
  const recordedAt = Date.parse(hint?.recordedAt ?? "");
  if (!hint || !Number.isFinite(recordedAt) || now - recordedAt > MAX_HINT_AGE_MS) return null;
  return hint;
}

export function saveHardwareTuningHint(config, identity, hint) {
  const filePath = tuningPath(config);
  if (!filePath || !identity?.key || !hint) return null;
  const store = readStore(config);
  const record = {
    ...hint,
    identity: identity.facts,
    recordedAt: new Date().toISOString(),
  };
  store.hints[identity.key] = record;
  ensureDirectory(path.dirname(filePath));
  fs.writeFileSync(filePath, JSON.stringify(store, null, 2));
  return record;
}
