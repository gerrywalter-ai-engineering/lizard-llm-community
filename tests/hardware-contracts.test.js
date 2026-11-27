import assert from "node:assert/strict";
import test from "node:test";

import {
  HARDWARE_CONTRACT_VERSION,
  HARDWARE_VALIDATION_STATES,
  BACKEND_PROBE_STATES,
  adapterIdentityKey,
  sameAdapterIdentity,
  hasAdapterLuid,
  normalizeAdapterRecord,
  mergeAdapterInventories,
  selectableAdapters,
  selectCpuRunnerTier,
  vendorLabelFromPciId,
} from "../src/lib/hardware-contracts.js";

// A native (DXGI-level) record as the future native probe reports it.
function nativeAdapter(overrides = {}) {
  return {
    identity: {
      luidHigh: 0,
      luidLow: 51234,
      vendorId: 0x10de,
      deviceId: 0x139a,
      name: "NVIDIA GeForce GTX 950M",
      ...(overrides.identity ?? {}),
    },
    memory: {
      dedicatedVideoMemoryBytes: 2 * 1024 * 1024 * 1024,
      sharedSystemMemoryBytes: 8 * 1024 * 1024 * 1024,
      isUma: false,
      ...(overrides.memory ?? {}),
    },
    isSoftware: overrides.isSoftware ?? false,
    validated: overrides.validated ?? true,
    probes: overrides.probes ?? { d3d12: { state: BACKEND_PROBE_STATES.USABLE } },
    sources: ["native-probe"],
  };
}

test("adapter identity prefers the DXGI LUID and never matches across different LUIDs", () => {
  const a = normalizeAdapterRecord(nativeAdapter());
  const b = normalizeAdapterRecord(nativeAdapter({ identity: { luidLow: 99999 } }));
  assert.ok(hasAdapterLuid(a));
  assert.notEqual(adapterIdentityKey(a), adapterIdentityKey(b));
  // Same-name adapters retain distinct stable identities.
  assert.equal(sameAdapterIdentity(a, b), false);
  // A LUID record never matches a LUID-less record even with identical PCI ids.
  const luidless = normalizeAdapterRecord({
    identity: { vendorId: 0x10de, deviceId: 0x139a, name: "NVIDIA GeForce GTX 950M" },
  });
  assert.equal(sameAdapterIdentity(a, luidless), false);
});

test("nvidia-smi and DXGI records deduplicate into one adapter that keeps the LUID", () => {
  const smiRecord = {
    vendor: "nvidia",
    name: "NVIDIA GeForce GTX 950M",
    vramGB: 2,
    driverVersion: "552.22",
    sources: ["nvidia-smi"],
  };
  const merged = mergeAdapterInventories({
    uiAdapters: [smiRecord],
    nativeAdapters: [nativeAdapter()],
  });
  assert.equal(merged.length, 1);
  const record = merged[0];
  assert.ok(hasAdapterLuid(record), "merged record keeps the DXGI LUID");
  assert.equal(record.driver.version, "552.22", "vendor-tool metadata attached");
  assert.deepEqual([...record.sources].sort(), ["native-probe", "nvidia-smi"]);
  assert.equal(record.discovered, true);
  assert.equal(record.validated, true);
});

test("merge result does not depend on enumeration order", () => {
  const intel = nativeAdapter({
    identity: { luidLow: 777, vendorId: 0x8086, deviceId: 0x9a49, name: "Intel(R) Iris(R) Xe Graphics" },
    memory: { dedicatedVideoMemoryBytes: 128 * 1024 * 1024, isUma: true },
  });
  const nvidia = nativeAdapter();
  const ui = [
    { vendor: "nvidia", name: "NVIDIA GeForce GTX 950M", vramGB: 2, driverVersion: "552.22" },
    { vendor: "intel", name: "Intel(R) Iris(R) Xe Graphics", vramGB: 0.125 },
  ];
  const forward = mergeAdapterInventories({ uiAdapters: ui, nativeAdapters: [intel, nvidia] });
  const reversed = mergeAdapterInventories({ uiAdapters: [...ui].reverse(), nativeAdapters: [nvidia, intel] });
  const keysForward = forward.map(adapterIdentityKey).sort();
  const keysReversed = reversed.map(adapterIdentityKey).sort();
  assert.deepEqual(keysForward, keysReversed);
  assert.equal(forward.length, 2);
  const nvidiaRecord = forward.find((r) => r.vendor === "nvidia");
  assert.equal(nvidiaRecord.driver.version, "552.22");
});

test("sole-vendor match attaches metadata despite name variance between tools", () => {
  const merged = mergeAdapterInventories({
    uiAdapters: [{ vendor: "nvidia", name: "GeForce GTX 950M", driverVersion: "552.22" }],
    nativeAdapters: [nativeAdapter()],
  });
  assert.equal(merged.length, 1);
  assert.equal(merged[0].driver.version, "552.22");
});

test("a UI-discovered but native-unsupported GPU stays visible and is not selectable", () => {
  const merged = mergeAdapterInventories({
    uiAdapters: [{ vendor: "amd", name: "AMD Radeon HD 6450", vramGB: 0.5 }],
    nativeAdapters: [nativeAdapter()],
  });
  assert.equal(merged.length, 2, "the unmatched UI adapter remains in the inventory");
  const amd = merged.find((record) => record.vendor === "amd");
  assert.equal(amd.validated, false);
  assert.equal(hasAdapterLuid(amd), false);
  const selectable = selectableAdapters(merged);
  assert.equal(selectable.length, 1);
  assert.equal(selectable[0].vendor, "nvidia");
});

test("software adapters and failed self-tests are excluded from automatic acceleration", () => {
  const warp = nativeAdapter({
    identity: { luidLow: 1, vendorId: 0x1414, deviceId: 0x8c, name: "Microsoft Basic Render Driver" },
    isSoftware: true,
  });
  const failed = nativeAdapter({
    identity: { luidLow: 2 },
    validated: false,
    probes: { d3d12: { state: BACKEND_PROBE_STATES.DEVICE_UNSUPPORTED } },
  });
  const usable = nativeAdapter({ identity: { luidLow: 3 } });
  const selectable = selectableAdapters(mergeAdapterInventories({
    uiAdapters: [],
    nativeAdapters: [warp, failed, usable],
  }));
  assert.equal(selectable.length, 1);
  assert.equal(selectable[0].identity.luidLow, 3);
});

test("normalizeAdapterRecord derives vendor from PCI id ahead of the name", () => {
  assert.equal(vendorLabelFromPciId(0x8086), "intel");
  const record = normalizeAdapterRecord({
    identity: { vendorId: 0x10de, deviceId: 1, name: "Mystery Adapter" },
  });
  assert.equal(record.vendor, "nvidia");
  assert.equal(record.schemaVersion, HARDWARE_CONTRACT_VERSION);
  assert.equal(record.validation.state, HARDWARE_VALIDATION_STATES.DETECTING);
});

test("CPU runner tier selection requires both CPUID and OS-enabled state", () => {
  const cpuOnlyAvx2 = {
    avx: { cpu: true, os: false },
    avx2: { cpu: true, os: false },
    fma: { cpu: true, os: false },
  };
  // AVX-capable CPU without OS XSAVE state must stay on baseline.
  assert.equal(selectCpuRunnerTier(cpuOnlyAvx2), "baseline");

  const fullAvx2 = {
    avx: { cpu: true, os: true },
    avx2: { cpu: true, os: true },
    fma: { cpu: true, os: true },
  };
  assert.equal(selectCpuRunnerTier(fullAvx2), "avx2");

  // Bare booleans (legacy name-inferred data) cannot prove OS state.
  assert.equal(selectCpuRunnerTier({ avx: true, avx2: true, fma: true }), "baseline");

  // Every optional feature unavailable -> baseline bootstrap still selected.
  assert.equal(selectCpuRunnerTier({}), "baseline");
});

test("a failed optimized-runner self-test retries the next lower tier down to baseline", () => {
  const features = {
    avx: { cpu: true, os: true },
    avx2: { cpu: true, os: true },
    fma: { cpu: true, os: true },
    avx512f: { cpu: true, os: true },
    avx512bw: { cpu: true, os: true },
    avx512vl: { cpu: true, os: true },
    avx512Vnni: { cpu: true, os: true },
  };
  assert.equal(selectCpuRunnerTier(features), "avx512");
  assert.equal(selectCpuRunnerTier(features, { disabledTiers: ["avx512"] }), "avx2");
  assert.equal(selectCpuRunnerTier(features, { disabledTiers: ["avx512", "avx2"] }), "avx");
  assert.equal(
    selectCpuRunnerTier(features, { disabledTiers: ["avx512", "avx2", "avx", "baseline"] }),
    "baseline",
    "baseline is the unconditional fallback even when marked disabled",
  );
});

test("AVX-512 tier requires every instruction subset compiled into the binary", () => {
  const features = {
    avx: { cpu: true, os: true },
    avx2: { cpu: true, os: true },
    fma: { cpu: true, os: true },
    avx512f: { cpu: true, os: true },
    avx512bw: { cpu: false, os: true },
    avx512vl: { cpu: true, os: true },
    avx512Vnni: { cpu: true, os: true },
  };
  assert.equal(selectCpuRunnerTier(features), "avx2");
  features.avx512bw = { cpu: true, os: true };
  features.avx512Vnni = { cpu: false, os: true };
  assert.equal(selectCpuRunnerTier(features), "avx2");
});
