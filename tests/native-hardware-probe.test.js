import assert from "node:assert/strict";
import test from "node:test";

import {
  normalizeHardwareProbeReport,
  runNativeHardwareProbe,
} from "../src/lib/native-hardware-probe.js";
import {
  BACKEND_PROBE_STATES,
  hasAdapterLuid,
  selectableAdapters,
  selectCpuRunnerTier,
} from "../src/lib/hardware-contracts.js";

// Mirrors real `--probe-hardware` output captured on the Iris Xe dev machine
// (trimmed): an integrated UMA adapter, the WARP software adapter, and a
// synthetic failed dGPU.
const PROBE_FIXTURE = {
  schemaVersion: 1,
  platform: "windows",
  cpu: {
    architecture: "x86-64",
    cpuidAvailable: true,
    vendor: "GenuineIntel",
    brand: "11th Gen Intel(R) Core(TM) i7-1165G7 @ 2.80GHz",
    family: 6,
    model: 140,
    stepping: 1,
    logicalProcessors: 8,
    physicalCores: 4,
    processorGroups: 1,
    features: {
      sse2: { cpu: true, os: true },
      avx: { cpu: true, os: true },
      fma: { cpu: true, os: true },
      avx2: { cpu: true, os: true },
      avx512f: { cpu: true, os: false },
      avx512bw: { cpu: true, os: false },
      avx512vl: { cpu: true, os: false },
      avx512Vnni: { cpu: true, os: false },
    },
  },
  memory: { totalBytes: 16957816832, availableBytes: 3113222144, commitLimitBytes: 33915633664, memoryLoadPercent: 81 },
  adapters: [
    {
      luidHigh: 0,
      luidLow: 73790,
      vendorId: 0x8086,
      deviceId: 0x9a49,
      driverVersion: "31.0.101.5333",
      name: "Intel(R) Iris(R) Xe Graphics",
      dedicatedVideoMemoryBytes: 134152192,
      sharedSystemMemoryBytes: 8478908416,
      isSoftware: false,
      videoMemory: {
        local: { budgetBytes: 7774095769, currentUsageBytes: 0, availableForReservationBytes: 3887047884 },
        nonLocal: { budgetBytes: 0, currentUsageBytes: 0, availableForReservationBytes: 0 },
      },
      d3d12: {
        deviceCreated: true,
        uma: true,
        cacheCoherentUma: true,
        highestShaderModel: "6.6",
        waveOps: true,
        waveLaneCountMin: 8,
        selfTest: "passed",
        state: "usable",
      },
    },
    {
      luidHigh: 0,
      luidLow: 74580,
      vendorId: 0x1414,
      deviceId: 0x8c,
      name: "Microsoft Basic Render Driver",
      dedicatedVideoMemoryBytes: 0,
      sharedSystemMemoryBytes: 8478908416,
      isSoftware: true,
      videoMemory: { local: null, nonLocal: null },
      d3d12: { deviceCreated: true, uma: true, cacheCoherentUma: true, selfTest: "passed", state: "usable" },
    },
    {
      luidHigh: 0,
      luidLow: 90001,
      vendorId: 0x10de,
      deviceId: 0x139a,
      driverVersion: "32.0.15.6094",
      name: "NVIDIA GeForce GTX 950M",
      dedicatedVideoMemoryBytes: 2147483648,
      sharedSystemMemoryBytes: 8478908416,
      isSoftware: false,
      videoMemory: { local: null, nonLocal: null },
      d3d12: {
        deviceCreated: false,
        state: "device-unsupported",
        detail: "D3D12CreateDevice(FL 11_0) failed (0x887a0004)",
      },
    },
  ],
};

test("probe adapters normalize into contract records with LUIDs and authoritative UMA", () => {
  const report = normalizeHardwareProbeReport(PROBE_FIXTURE);
  assert.equal(report.adapters.length, 3);
  const [intel, warp, nvidia] = report.adapters;

  assert.ok(hasAdapterLuid(intel));
  assert.equal(intel.vendor, "intel");
  // UMA comes from D3D12_FEATURE_ARCHITECTURE, not vendor/name inference.
  assert.equal(intel.memory.isUma, true);
  assert.equal(intel.memory.isCacheCoherentUma, true);
  assert.equal(intel.validated, true);
  assert.equal(intel.driver.version, "31.0.101.5333");
  assert.equal(intel.probes.d3d12.state, BACKEND_PROBE_STATES.USABLE);
  assert.equal(intel.probes.d3d12.videoMemory.local.budgetBytes, 7774095769);

  assert.equal(warp.isSoftware, true);
  assert.equal(nvidia.validated, false);
  assert.equal(nvidia.probes.d3d12.state, BACKEND_PROBE_STATES.DEVICE_UNSUPPORTED);
  assert.match(nvidia.probes.d3d12.detail, /D3D12CreateDevice/);
});

test("only validated hardware adapters with a usable probe are selectable", () => {
  const report = normalizeHardwareProbeReport(PROBE_FIXTURE);
  const selectable = selectableAdapters(report.adapters);
  assert.equal(selectable.length, 1);
  assert.equal(selectable[0].vendor, "intel");
});

test("probe CPU features drive the runner tier through the CPUID+OS rule", () => {
  const report = normalizeHardwareProbeReport(PROBE_FIXTURE);
  // avx512f is CPU-capable but OS-disabled in the fixture -> avx2 tier.
  assert.equal(selectCpuRunnerTier(report.cpu.features), "avx2");
});

test("malformed or empty probe output normalizes to null", () => {
  assert.equal(normalizeHardwareProbeReport(null), null);
  assert.equal(normalizeHardwareProbeReport({}), null);
  assert.equal(normalizeHardwareProbeReport({ adapters: "nope" }), null);
});

test("runNativeHardwareProbe returns null when the runner is missing or fails", () => {
  const missing = runNativeHardwareProbe({
    runnerPath: null,
    config: { engine: { native: { buildDirectory: "Z:\\does\\not\\exist" } } },
    exec: () => {
      throw new Error("must not spawn without a binary");
    },
  });
  assert.equal(missing, null);

  const crashed = runNativeHardwareProbe({
    runnerPath: "C:\\fake\\lizard-native-runner.exe",
    exec: () => {
      throw new Error("spawn failed");
    },
  });
  assert.equal(crashed, null);

  const garbage = runNativeHardwareProbe({
    runnerPath: "C:\\fake\\lizard-native-runner.exe",
    exec: () => "this is not json",
  });
  assert.equal(garbage, null);
});

test("runNativeHardwareProbe parses real probe output through the injected exec", () => {
  const report = runNativeHardwareProbe({
    runnerPath: "C:\\fake\\lizard-native-runner.exe",
    exec: (binary, args) => {
      assert.deepEqual(args, ["--probe-hardware"]);
      return JSON.stringify(PROBE_FIXTURE);
    },
  });
  assert.equal(report.platform, "windows");
  assert.equal(report.adapters.length, 3);
  assert.equal(report.cpu.brand.includes("i7-1165G7"), true);
});
