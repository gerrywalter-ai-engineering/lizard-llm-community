import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { ensureDirectory } from "./paths.js";
import {
  HARDWARE_VALIDATION_STATES,
  mergeAdapterInventories,
  selectableAdapters,
  selectPreferredAdapter,
  selectCpuRunnerTier,
} from "./hardware-contracts.js";
import { getCachedNativeHardwareProbe } from "./native-hardware-probe.js";

const SCHEMA_VERSION = 1;

function clone(value) {
  return JSON.parse(JSON.stringify(value));
}

function detectOsLabel(platform = process.platform) {
  if (platform === "win32") {
    return "windows";
  }
  if (platform === "darwin") {
    return "macos";
  }
  return "linux";
}

function detectCpuFeatures({ cpuModel = "", arch = process.arch }) {
  const features = [];
  const model = String(cpuModel).toLowerCase();
  if (arch === "x64") {
    if (/avx512/.test(model)) {
      features.push("avx512");
    }
    if (/avx2|ryzen|xeon|core\(tm\)|epyc/.test(model)) {
      features.push("avx2");
    }
  }
  if (arch === "arm64") {
    features.push("neon");
  }
  return features;
}

function binarySupportSummary(config) {
  const cli = resolveBinary(config, false);
  const server = resolveBinary(config, true);
  return {
    cli,
    server,
    installed: Boolean(cli || server),
  };
}

function findExecutable(name) {
  const envPath = process.env.PATH ?? "";
  const extensions = process.platform === "win32" ? [".exe", ".cmd", ".bat", ""] : [""];
  for (const directory of envPath.split(path.delimiter)) {
    for (const extension of extensions) {
      const fullPath = path.join(directory, `${name}${extension}`);
      if (fs.existsSync(fullPath)) {
        return fullPath;
      }
    }
  }
  return null;
}

function findExplicitCandidate(candidate) {
  const extensions = process.platform === "win32" ? [".exe", ".cmd", ".bat", ""] : [""];
  for (const extension of extensions) {
    const withExtension = candidate.endsWith(extension) ? candidate : `${candidate}${extension}`;
    if (fs.existsSync(withExtension)) {
      return path.resolve(withExtension);
    }
  }
  return null;
}

function resolveBinary(config, preferServer = true) {
  const candidates = [];
  for (const directory of config?.engine?.preferredBinaryDirectories ?? []) {
    candidates.push(path.join(directory, preferServer ? config.engine.serverBinary : config.engine.cliBinary));
  }
  candidates.push(preferServer ? config?.engine?.serverBinary : config?.engine?.cliBinary);
  for (const candidate of candidates) {
    if (!candidate) {
      continue;
    }
    const explicit = findExplicitCandidate(candidate);
    if (explicit) {
      return explicit;
    }
    const resolved = findExecutable(candidate);
    if (resolved) {
      return resolved;
    }
  }
  return null;
}

function backendState({ supported, recommended, reason, warnings = [], dependency = null }) {
  return {
    supported: Boolean(supported),
    recommended: Boolean(recommended),
    reason,
    warnings,
    dependencyHint: dependency,
  };
}

export function buildBackendCapabilityRegistry({
  hardware,
  config,
  benchmarkHistory = [],
  // Normalized report from src/lib/native-hardware-probe.js. Optional during
  // migration (Phase 1 step 1): absent -> the registry behaves exactly as
  // before; present -> adapters gain stable identity, discovered/validated
  // truth levels, and the CPU gains CPUID+OS-state feature facts.
  nativeProbe = null,
} = {}) {
  const osLabel = detectOsLabel(hardware?.platform);
  const arch = hardware?.arch ?? process.arch;
  const cpuName = hardware?.cpu?.model ?? os.cpus()?.[0]?.model ?? "Unknown CPU";
  const cpuFeatures = detectCpuFeatures({ cpuModel: cpuName, arch });

  // One canonical adapter inventory (Finding 1): native DXGI records are
  // authoritative, UI-scan records enrich them or remain discovered-only.
  const adapterInventory = mergeAdapterInventories({
    uiAdapters: (hardware?.gpus ?? []).map((gpu) => ({
      vendor: gpu.vendor ?? undefined,
      name: gpu.name,
      vramGB: gpu.vramGB ?? gpu.vramGb ?? null,
      driverVersion: gpu.driverVersion ?? null,
      probes: gpu.backends
        ? Object.fromEntries(Object.entries(gpu.backends)
          .filter(([, value]) => Boolean(value))
          .map(([key]) => [key, { state: "hint" }]))
        : {},
      sources: [gpu.source ?? "ui-scan"],
    })),
    nativeAdapters: nativeProbe?.adapters ?? [],
  });
  const preferredAdapter = selectPreferredAdapter(adapterInventory);
  const hardwareValidationState = nativeProbe == null
    ? HARDWARE_VALIDATION_STATES.DETECTING
    : (preferredAdapter ? HARDWARE_VALIDATION_STATES.READY : HARDWARE_VALIDATION_STATES.CPU_FALLBACK_READY);

  const legacyGpuEntry = (gpu) => ({
    name: gpu.name,
    vendor: gpu.vendor ?? "unknown",
    vramGb: gpu.vramGB ?? gpu.vramGb ?? null,
    backends: Object.entries(gpu.backends ?? {})
      .filter(([, value]) => Boolean(value))
      .map(([key]) => key),
  });
  const inventoryGpuEntry = (record) => ({
    name: record.identity.name,
    vendor: record.vendor,
    vramGb: record.memory.dedicatedVideoMemoryBytes != null
      ? Number((record.memory.dedicatedVideoMemoryBytes / (1024 * 1024 * 1024)).toFixed(2))
      : null,
    backends: Object.keys(record.probes ?? {}),
    identity: { ...record.identity },
    validated: record.validated,
    isSoftware: record.isSoftware,
    isUma: record.memory.isUma,
    probes: record.probes ?? {},
  });
  // With a probe, selectable adapters lead (preference order) so legacy
  // gpu[0] consumers stop depending on raw enumeration order; without one,
  // the legacy list is byte-identical to the previous behavior.
  const gpuList = nativeProbe == null
    ? (hardware?.gpus ?? []).map(legacyGpuEntry)
    : (() => {
      const selectable = selectableAdapters(adapterInventory);
      const ordered = preferredAdapter
        ? [preferredAdapter, ...selectable.filter((record) => record !== preferredAdapter)]
        : selectable;
      const rest = adapterInventory.filter((record) => !selectable.includes(record));
      return [...ordered, ...rest].map(inventoryGpuEntry);
    })();
  const binaries = binarySupportSummary(config);
  const hasHistory = Array.isArray(benchmarkHistory) && benchmarkHistory.length > 0;
  const lowRam = (hardware?.memory?.plannerBudgetGB ?? hardware?.memory?.availableGB ?? 0) < 8;

  const cpuSupported = true;
  const cudaSupported = Boolean(hardware?.backends?.cuda && binaries.installed);
  const metalSupported = Boolean(hardware?.backends?.metal && binaries.installed);
  const vulkanSupported = Boolean((hardware?.backends?.vulkan || hardware?.backends?.rocm) && binaries.installed);
  const openclSupported = false;

  const preferredBackend = metalSupported
    ? "metal"
    : cudaSupported
      ? "cuda"
      : vulkanSupported
        ? "vulkan"
        : "cpu";

  const backends = {
    cpu: backendState({
      supported: cpuSupported,
      recommended: preferredBackend === "cpu",
      reason: lowRam ? "CPU mode selected because system memory is constrained." : "CPU fallback is always available.",
      warnings: lowRam ? ["Reduce context size and batch size on low-RAM systems."] : [],
    }),
    cuda: backendState({
      supported: cudaSupported,
      recommended: preferredBackend === "cuda",
      reason: cudaSupported ? "NVIDIA GPU detected and llama.cpp binaries are available." : "CUDA support is unavailable on this machine.",
      warnings: cudaSupported ? [] : ["Install CUDA-capable llama.cpp binaries or use CPU fallback."],
      dependency: cudaSupported ? null : "llama.cpp build with CUDA support",
    }),
    metal: backendState({
      supported: metalSupported,
      recommended: preferredBackend === "metal",
      reason: metalSupported ? "Apple GPU detected and llama.cpp binaries are available." : "Metal support is unavailable on this machine.",
      warnings: metalSupported ? [] : ["Metal requires Apple Silicon/macOS and a compatible llama.cpp build."],
      dependency: metalSupported ? null : "llama.cpp build with Metal support",
    }),
    vulkan: backendState({
      supported: vulkanSupported,
      recommended: preferredBackend === "vulkan",
      reason: vulkanSupported ? "AMD/Intel-compatible GPU path detected and llama.cpp binaries are available." : "Vulkan support is unavailable on this machine.",
      warnings: vulkanSupported ? [] : ["Vulkan requires a compatible GPU/runtime and llama.cpp build."],
      dependency: vulkanSupported ? null : "llama.cpp build with Vulkan support",
    }),
    opencl: backendState({
      supported: openclSupported,
      recommended: false,
      reason: "OpenCL is not currently modeled as a supported backend in Lizard.",
      warnings: ["OpenCL backend selection is unavailable."],
      dependency: "future capability",
    }),
  };

  // CPU facts: the native probe's CPUID+XGETBV result replaces name-based
  // inference when present. The legacy string array keeps its shape but is
  // rebuilt from validated (cpu AND os enabled) features only.
  const probeCpuFeatures = nativeProbe?.cpu?.features ?? null;
  const validatedCpuFeatureNames = probeCpuFeatures
    ? Object.entries(probeCpuFeatures)
      .filter(([, state]) => Boolean(state?.cpu) && Boolean(state?.os))
      .map(([name]) => name)
    : null;

  return {
    schemaVersion: SCHEMA_VERSION,
    os: osLabel,
    arch,
    cpu: {
      name: cpuName,
      cores: nativeProbe?.cpu?.physicalCores
        ?? hardware?.cpu?.physicalCores ?? hardware?.cpu?.logicalThreads ?? os.cpus().length,
      threads: nativeProbe?.cpu?.logicalProcessors
        ?? hardware?.cpu?.logicalThreads ?? os.cpus().length,
      features: validatedCpuFeatureNames ?? cpuFeatures,
      ...(probeCpuFeatures
        ? {
          featureStates: probeCpuFeatures,
          runnerTier: selectCpuRunnerTier(probeCpuFeatures),
          featureSource: "cpuid",
        }
        : { featureSource: "name-inference" }),
    },
    memory: {
      ramGb: hardware?.memory?.totalGB ?? null,
      availableGb: hardware?.memory?.plannerBudgetGB ?? hardware?.memory?.availableGB ?? null,
    },
    gpu: gpuList,
    adapterInventory,
    selectedAdapter: preferredAdapter
      ? {
        identity: { ...preferredAdapter.identity },
        vendor: preferredAdapter.vendor,
        isUma: preferredAdapter.memory.isUma,
        dedicatedVideoMemoryBytes: preferredAdapter.memory.dedicatedVideoMemoryBytes,
        reason: "preferred selectable adapter (validated, hardware, deterministic policy)",
      }
      : null,
    hardwareValidation: {
      state: hardwareValidationState,
      nativeProbe: nativeProbe != null,
    },
    backends,
    binaries,
    previousBenchmarkProfiles: hasHistory ? benchmarkHistory.slice(0, 3) : [],
    selectedBackend: preferredBackend,
    selectionReason: backends[preferredBackend]?.reason ?? "No backend recommendation available.",
    warnings: Object.entries(backends)
      .filter(([, state]) => state.supported === false && state.recommended === false)
      .flatMap(([, state]) => state.warnings ?? [])
      .slice(0, 4),
  };
}

/**
 * Registry with native validation merged in (Phase 1 flow: UI scan ->
 * native capability probe -> merged snapshot). The probe result is cached
 * per process; when the compiled runner is unavailable the result equals
 * the plain registry and hardwareValidation.state stays "detecting".
 */
export function buildValidatedCapabilityRegistry({
  hardware,
  config,
  benchmarkHistory = [],
  logger = null,
  probe = null,
} = {}) {
  const nativeProbe = probe ?? getCachedNativeHardwareProbe({ config, logger });
  return buildBackendCapabilityRegistry({ hardware, config, benchmarkHistory, nativeProbe });
}

export function selectBackendForProfile({
  capabilityRegistry,
  requestedBackend = null,
  estimate = null,
  mode = "generate",
} = {}) {
  const registry = capabilityRegistry ?? {};
  const requested = String(requestedBackend ?? "").trim().toLowerCase();
  const warnings = [];
  const memoryRisk = (() => {
    const availableGb = registry.memory?.availableGb ?? null;
    const requiredGb = estimate?.ramRequiredGB ?? estimate?.totalGB ?? null;
    if (availableGb == null || requiredGb == null) {
      return "medium";
    }
    if (requiredGb <= availableGb * 0.8) {
      return "low";
    }
    if (requiredGb <= availableGb) {
      return "medium";
    }
    return "high";
  })();

  if (requested) {
    const backend = registry.backends?.[requested];
    if (backend?.supported) {
      return {
        backend: requested,
        reason: `User override selected ${requested}. ${backend.reason}`,
        warnings,
        memoryRisk,
      };
    }
    warnings.push(`Requested backend ${requested} is not supported on this machine.`);
  }

  const selected = registry.selectedBackend ?? "cpu";
  const backendStateValue = registry.backends?.[selected] ?? registry.backends?.cpu ?? null;
  if (memoryRisk === "high" && selected !== "cpu" && registry.backends?.cpu?.supported) {
    warnings.push("Estimated memory risk is high; CPU fallback may be safer for this profile.");
  }

  return {
    backend: selected,
    reason: backendStateValue?.reason ?? `Selected ${selected} backend for ${mode}.`,
    warnings: [
      ...warnings,
      ...(backendStateValue?.warnings ?? []),
    ],
    memoryRisk,
  };
}

function snapshotPath(config) {
  return path.join(config.runtimeDirectory, "backend-capability-snapshot.json");
}

export function saveBackendCapabilitySnapshot(config, capabilityRegistry) {
  const filePath = snapshotPath(config);
  ensureDirectory(path.dirname(filePath));
  fs.writeFileSync(filePath, JSON.stringify(clone(capabilityRegistry), null, 2));
  return filePath;
}

export function loadBackendCapabilitySnapshot(config) {
  const filePath = snapshotPath(config);
  if (!fs.existsSync(filePath)) {
    return null;
  }
  try {
    return JSON.parse(fs.readFileSync(filePath, "utf8"));
  } catch {
    return null;
  }
}
