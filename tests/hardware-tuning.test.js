import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import {
  applyHardwareTuningHint,
  buildHardwareTuningIdentity,
  evaluateHardwareCalibration,
  loadHardwareTuningHint,
  saveHardwareTuningHint,
} from "../src/runtime/hardware-tuning.js";

function fixture() {
  return {
    profile: {
      contextSize: 4096,
      gguf: { architecture: "gemma", quant: "q4_k_m", fileSizeBytes: 2_000_000_000 },
      capabilityRegistry: {
        os: "windows",
        arch: "x64",
        cpu: { name: "Intel i5", runnerTier: "avx2" },
        selectedAdapter: {
          identity: { luidHigh: 0, luidLow: 42, vendorId: 0x10de, deviceId: 1, name: "GTX" },
          isUma: false,
        },
      },
    },
    modelArtifact: { modelId: "gemma", gguf: { architecture: "gemma", quant: "q4_k_m", fileSizeBytes: 2_000_000_000 } },
    executionPlan: { execution: { residentWeightBudgetBytes: 1_000_000_000 }, hardware: {} },
  };
}

test("hardware tuning identity changes with material model or device facts", () => {
  const first = buildHardwareTuningIdentity(fixture());
  const changed = fixture();
  changed.profile.capabilityRegistry.selectedAdapter.identity.luidLow = 43;
  const second = buildHardwareTuningIdentity(changed);
  assert.notEqual(first.key, second.key);
});

test("calibration requires parity and a material GPU win", () => {
  assert.equal(evaluateHardwareCalibration({ cpuMsPerIteration: 10, gpuMsPerIteration: 5, parityPassed: true }).recommendedDecodeBackend, "gpu");
  assert.equal(evaluateHardwareCalibration({ cpuMsPerIteration: 10, gpuMsPerIteration: 9.5, parityPassed: true }).recommendedDecodeBackend, "cpu");
  assert.equal(evaluateHardwareCalibration({ cpuMsPerIteration: 10, gpuMsPerIteration: 5, parityPassed: false }).recommendedDecodeBackend, "cpu");
});

test("CPU tuning hint becomes an authoritative decode plan", () => {
  const plan = { execution: { allowGpuDecode: true, allowCompleteTokenGraph: true } };
  const tuned = applyHardwareTuningHint(plan, { recommendedDecodeBackend: "cpu", reason: "GPU was slower." });
  assert.equal(tuned.execution.allowGpuDecode, false);
  assert.equal(tuned.execution.allowCompleteTokenGraph, false);
  assert.equal(tuned.execution.decodeStrategy, "cpu-calibrated");
});

test("hardware tuning hints persist by stable identity", () => {
  const runtimeDirectory = fs.mkdtempSync(path.join(os.tmpdir(), "lizard-tuning-"));
  const config = { runtimeDirectory };
  const identity = buildHardwareTuningIdentity(fixture());
  saveHardwareTuningHint(config, identity, { recommendedDecodeBackend: "cpu", reason: "slower" });
  const loaded = loadHardwareTuningHint(config, identity);
  assert.equal(loaded.recommendedDecodeBackend, "cpu");
  assert.equal(loaded.identity.adapterLuid, "0:42");
});
