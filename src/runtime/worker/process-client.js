import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import { spawn, spawnSync } from "node:child_process";
import { buildNativeExecutionPlan } from "../native-execution-plan.js";
import { resolveAppPath } from "../../lib/app-root.js";

const DEFAULT_RPC_VERSION = 1;
const MB = 1024 * 1024;
export const MAX_NATIVE_RPC_LINE_BYTES = 16 * MB;

export function attachBoundedLineReader(stream, {
  maxBytes = MAX_NATIVE_RPC_LINE_BYTES,
  onLine,
  onError,
} = {}) {
  let parts = [];
  let lineBytes = 0;
  let failed = false;
  const fail = () => {
    if (failed) return;
    failed = true;
    parts = [];
    lineBytes = 0;
    const error = new Error(`Native worker RPC line exceeds the ${maxBytes}-byte limit.`);
    error.code = "NATIVE_WORKER_RPC_LINE_TOO_LARGE";
    onError?.(error);
  };
  const append = (segment) => {
    if (failed || segment.length === 0) return;
    lineBytes += segment.length;
    if (lineBytes > maxBytes) {
      fail();
      return;
    }
    parts.push(segment);
  };
  stream.on("data", (chunk) => {
    if (failed) return;
    let start = 0;
    for (let index = 0; index < chunk.length; index += 1) {
      if (chunk[index] !== 0x0A) continue;
      append(chunk.subarray(start, index));
      if (failed) return;
      const line = Buffer.concat(parts, lineBytes).toString("utf8").replace(/\r$/, "");
      parts = [];
      lineBytes = 0;
      onLine?.(line);
      start = index + 1;
    }
    append(chunk.subarray(start));
  });
  stream.on("end", () => {
    if (!failed && lineBytes > 0) {
      onLine?.(Buffer.concat(parts, lineBytes).toString("utf8").replace(/\r$/, ""));
    }
  });
  return { get failed() { return failed; } };
}

// Deadlines for the worker's lifecycle RPCs.
//
// A wedged native worker - a stalled GPU driver call, a shader compile that
// never returns, a half-written GGUF from an interrupted download - used to
// leave its request promise pending forever. Nothing rejected it, so every
// `.catch()` guarding the load path was unreachable and the caller simply never
// resumed. That is the "select a model and the app hangs" report: the load
// never fails, it just never finishes.
//
// Only the lifecycle calls are bounded. prefill/decode/generate are legitimately
// open-ended (a long generation on a slow CPU can run for minutes) and keep
// their previous unbounded behaviour, so this cannot cut a working answer short.
const RPC_TIMEOUTS_MS = {
  handshake: 60_000,
  load_model: 15 * 60_000,
  unload_model: 60_000,
  shutdown: 5_000,
  tensor_benchmark: 5 * 60_000,
};

// How long a killed worker gets to actually exit before the kill is escalated,
// and how long a caller waits for that exit. Both matter on Windows, where the
// GGUF stays mapped - and therefore undeletable - until the process is gone.
const WORKER_KILL_ESCALATION_MS = 1_500;
const WORKER_EXIT_WAIT_MS = 5_000;

// Env overrides exist so a deliberately slow run (a cold shader cache on a
// first-ever launch, a model on a network drive) can raise its own ceiling
// without a rebuild. `0` disables the deadline for that call entirely.
const RPC_TIMEOUT_ENV_KEYS = {
  handshake: "LIZARD_NATIVE_HANDSHAKE_TIMEOUT_MS",
  load_model: "LIZARD_NATIVE_LOAD_TIMEOUT_MS",
  shutdown: "LIZARD_NATIVE_SHUTDOWN_TIMEOUT_MS",
};

export function resolveRpcTimeoutMs(method, env = process.env) {
  const envKey = RPC_TIMEOUT_ENV_KEYS[method];
  const override = envKey ? Number(env?.[envKey]) : Number.NaN;
  if (Number.isFinite(override) && override >= 0) {
    return override;
  }
  return RPC_TIMEOUTS_MS[method] ?? 0;
}

export function validateWorkerRpcHandshake(result, clientVersion = DEFAULT_RPC_VERSION) {
  const selected = Number(result?.rpcVersion);
  const minimum = Number(result?.rpcMinVersion ?? selected);
  const maximum = Number(result?.rpcMaxVersion ?? selected);
  if (!Number.isInteger(selected)
    || !Number.isInteger(minimum)
    || !Number.isInteger(maximum)
    || clientVersion < minimum
    || clientVersion > maximum
    || selected !== clientVersion) {
    const error = new Error(
      `Native worker RPC version mismatch: client=${clientVersion}, `
      + `worker=${Number.isInteger(selected) ? selected : "unknown"}, `
      + `range=${Number.isInteger(minimum) ? minimum : "unknown"}-${Number.isInteger(maximum) ? maximum : "unknown"}.`,
    );
    error.code = "NATIVE_WORKER_RPC_VERSION_UNSUPPORTED";
    throw error;
  }
  return result;
}

function scriptRunnerPath() {
  return resolveAppPath("scripts", "lizard-native-runner.mjs");
}

// Exported because it is also the answer to "does this worker implement
// decode_batch": the batched decode loop lives in the compiled runner only, and
// prototype models stay on the JS worker (see resolveNativeWorkerLaunchSpec).
export function prefersCompiledNativeWorker(modelArtifact = null) {
  if (modelArtifact?.runtimePrototype?.format) {
    return false;
  }
  const modelPath = String(modelArtifact?.modelPath ?? "");
  if (!modelPath) {
    return true;
  }
  return path.extname(modelPath).toLowerCase() === ".gguf";
}

function detectedRunnerCandidates(config = null, runtimeProvider = "lizard-native") {
  // Ostrich is a separate binary, not an alias: resolving it to
  // lizard-native-runner would silently run a different engine.
  const RUNNER_BY_PROVIDER = {
    caterpillar: { key: "caterpillar", name: "caterpillar-runner" },
    ostrich: { key: "ostrich", name: "ostrich-runner" },
  };
  const resolved = RUNNER_BY_PROVIDER[runtimeProvider] ?? { key: "native", name: "lizard-native-runner" };
  const providerKey = resolved.key;
  const runnerName = resolved.name;
  const buildDirectory = config?.engine?.[providerKey]?.buildDirectory
    ? path.resolve(config.engine[providerKey].buildDirectory)
    : resolveAppPath("native", runnerName, "build");
  const runtimeDirectory = config?.runtimeDirectory
    ? path.resolve(config.runtimeDirectory)
    : null;
  const executableName = runnerName;
  const names = process.platform === "win32"
    ? [`${executableName}.exe`, `Release\\${executableName}.exe`, `Debug\\${executableName}.exe`]
    : [executableName, `Release/${executableName}`, `Debug/${executableName}`];
  const candidates = names.map((name) => path.join(buildDirectory, name));
  // The Windows installer stages the compiled worker at {app}\bin\ only
  // (see stageNativeRunner in build-windows-installer.mjs); it never ships
  // native/lizard-native-runner/build/. Any caller that omits `config` (or
  // whose config lacks engine.native.buildDirectory/runnerBinary) would
  // otherwise silently fall through to the JS-only stub worker in an
  // installed app - this candidate is the safety net for that case.
  candidates.push(resolveAppPath("bin", names[0]));
  if (runtimeDirectory) {
    candidates.unshift(path.join(runtimeDirectory, names[0]));
  }
  return candidates;
}

function normalizeNumber(value) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : null;
}

// The model's weight footprint in bytes. Catalog metadata when present;
// otherwise the GGUF file's real on-disk size - the Test Chat path resolves
// its artifact outside the catalog and would otherwise report no size at
// all, silently skipping every size-aware budget decision below.
function resolveModelWeightBytes(modelArtifact = null, profile = null) {
  const known = normalizeNumber(modelArtifact?.gguf?.fileSizeBytes)
    ?? normalizeNumber(profile?.gguf?.fileSizeBytes);
  if (known != null && known > 0) {
    return known;
  }
  const modelPath = String(modelArtifact?.modelPath ?? "");
  if (modelPath && path.extname(modelPath).toLowerCase() === ".gguf") {
    try {
      const size = fs.statSync(modelPath).size;
      return Number.isFinite(size) && size > 0 ? size : null;
    } catch {
      return null;
    }
  }
  return null;
}

export function resolveNativeGpuWeightBudgetMb({
  profile = null,
  modelArtifact = null,
  runtimeProvider = "lizard-native",
} = {}) {
  const executionPlan = buildNativeExecutionPlan({
    capabilityRegistry: profile?.capabilityRegistry ?? null,
    profile,
    gguf: profile?.gguf ?? null,
    requestedBackend: profile?.backend ?? "cpu",
    runtimeProvider,
  });
  if (!executionPlan || executionPlan.mode === "cpu-only") {
    return null;
  }

  const gpuBudgetBytes = normalizeNumber(executionPlan?.budgets?.gpuBudgetBytes);
  const unifiedBudgetBytes = normalizeNumber(executionPlan?.budgets?.unifiedBudgetBytes);
  const gpuWorkspaceBudgetBytes = normalizeNumber(executionPlan?.budgets?.gpuWorkspaceBudgetBytes);
  const gpuKvBudgetBytes = normalizeNumber(executionPlan?.budgets?.gpuKvBudgetBytes);
  const unifiedMemory = Boolean(executionPlan?.hardware?.unifiedMemory);

  let candidateBytes = gpuBudgetBytes;
  if (unifiedMemory && unifiedBudgetBytes != null) {
    // Integrated GPUs use system RAM for resident weights. Basing this cache
    // on the small reported "dedicated VRAM" budget leaves most layers on CPU
    // and causes resident-weight eviction/upload churn on every decoded token.
    const sharedWeightBytes = Math.trunc(unifiedBudgetBytes * 0.45);
    candidateBytes = Math.max(candidateBytes ?? 0, sharedWeightBytes);
  }
  if (candidateBytes == null || candidateBytes <= 0) {
    return null;
  }

  candidateBytes -= gpuWorkspaceBudgetBytes ?? 0;
  candidateBytes -= gpuKvBudgetBytes ?? 0;

  if (unifiedMemory && unifiedBudgetBytes != null) {
    // The default 45% share is a model-agnostic guess. When the caller
    // supplies the model actually about to load, its real on-disk weight
    // size is a much better answer than a guess - a model whose full
    // weight set exceeds the guess (e.g. gemma4-e4b Q8_0 at ~8 GB on a
    // 15-16 GB machine, well above the ~4 GB the 45% share produces) would
    // otherwise stop resident staging partway through and hard-fail with
    // NATIVE_GPU_DECODE_REQUIRED under gpuDecodePolicy "required" (found
    // 2026-07-07, iteration 29). Bounded by unifiedBudgetBytes itself (the
    // existing 70%-of-available-memory ceiling) so this only unlocks
    // headroom the hardware genuinely has, never overrides it.
    const knownWeightBytes = resolveModelWeightBytes(modelArtifact, profile);
    if (knownWeightBytes != null && knownWeightBytes > candidateBytes && knownWeightBytes <= unifiedBudgetBytes) {
      candidateBytes = knownWeightBytes;
    }
  }

  let candidateMb = Math.trunc(candidateBytes / MB);
  if (candidateMb <= 0) {
    return null;
  }

  const minMb = unifiedMemory ? 256 : 512;
  // On UMA hardware, unifiedBudgetBytes is already a machine-sized,
  // conservative ceiling (70% of available system memory) - re-capping at
  // a flat 4096 MB regardless of that number was the root cause of the
  // same NATIVE_GPU_DECODE_REQUIRED failure (it clamped the budget BELOW
  // an already-safe, hardware-derived ceiling on any machine with more
  // than ~5.7 GB available). Dedicated-GPU machines keep the flat 4096 MB
  // cap unchanged - no evidence they need this, and gpuBudgetBytes there
  // is already VRAM-derived, not a guess.
  const maxMb = unifiedMemory && unifiedBudgetBytes != null
    ? Math.trunc(unifiedBudgetBytes / MB)
    : 4096;
  candidateMb = Math.max(minMb, Math.min(maxMb, candidateMb));
  return candidateMb;
}

export function resolveNativeWorkerEnvironment({
  profile = null,
  modelArtifact = null,
  env = process.env,
  runtimeProvider = "lizard-native",
} = {}) {
  const overrides = {};
  if (!prefersCompiledNativeWorker(modelArtifact)) {
    return overrides;
  }
  // Q8_K integer-dot remains an explicit performance opt-in. It is materially
  // faster, but the Phase 8 lane-matched 64-token gate proved that its
  // approximate accumulation can change greedy output relative to llama.cpp.
  // The runner defaults to the full-precision accumulation path; an explicit
  // CATERPILLAR_CPU_INTEGER_DOT=1 is inherited by the child for operators and
  // future model-specific qualification.
  // The benchmark/profile thread count must reach the compiled Lizard worker.
  // Previously it was recorded as six in evidence while the persistent row
  // pool silently used all eight logical CPUs, invalidating equal-thread
  // comparisons with llama.cpp. Keep Caterpillar's separately qualified
  // full-hardware policy below, but make Lizard's declared setting effective.
  const requestedThreads = Math.trunc(normalizeNumber(profile?.threads));
  // Caterpillar's persistent row executor defaults to full hardware
  // concurrency inside the runner. The profile's thread count is tuned for
  // the llama.cpp server lane (it reserves headroom for that engine's own
  // loop) and clamping the dedicated runner to it measured -38% decode
  // tok/s on mistral-7B (6 of 8 threads). Only an explicit
  // CATERPILLAR_CPU_THREADS env override throttles the runner.
  // Adapter identity pinning (Phase 1 Finding 1): when the plan carries a
  // DXGI LUID, the worker must create its device on exactly that adapter
  // (select_adapter refuses substitutes). An explicit env override wins.
  const plan = buildNativeExecutionPlan({
    capabilityRegistry: profile?.capabilityRegistry ?? null,
    profile,
    gguf: modelArtifact?.gguf ?? profile?.gguf ?? null,
    requestedBackend: profile?.backend ?? "cpu",
    runtimeProvider,
  });
  if (runtimeProvider === "lizard-native"
    && env?.LIZARD_NATIVE_CPU_WORKERS == null
    && Number.isInteger(requestedThreads)
    && requestedThreads > 0) {
    // On the GPU lane the decode loop is waiting on the device, not computing:
    // measured CPU utilisation during hybrid decode is 3-8% of the machine. The
    // CPU workers there do the parts that stay on the processor, and running
    // more of them than there are physical cores puts two of them on the same
    // core's SMT siblings, which is slower rather than faster. Measured on an
    // i7-1165G7 (4C/8T, llama-3.2-3B Q4_K_M, directx-hybrid, 64 tokens):
    //
    //   2 workers 6.30 tok/s | 4 workers 7.16 (7.24/7.02/7.23)
    //   6 workers 6.90 (6.92/6.98/6.79) | 8 workers 6.25
    //
    // So the answer to "would more CPU make it faster" is no: one worker per
    // physical core is the peak, and the top of the range is 13% BELOW it. The
    // CPU-only lane keeps its own separately qualified count (Enhancement Run
    // 06 tuned 5-of-8 for this model shape) - this cap is GPU-lane only.
    // capabilityRegistry.cpu.cores is the physical count (native CPUID probe
    // first, hardware scan second); cpu.threads is the logical one.
    const physicalCores = Math.trunc(normalizeNumber(profile?.capabilityRegistry?.cpu?.cores));
    const gpuLane = plan?.mode !== "cpu-only";
    const effectiveThreads = gpuLane && Number.isInteger(physicalCores) && physicalCores > 0
      ? Math.min(requestedThreads, physicalCores)
      : requestedThreads;
    overrides.LIZARD_NATIVE_CPU_WORKERS = String(effectiveThreads);
    if (effectiveThreads !== requestedThreads) {
      // Declared and effective must never silently disagree - that mismatch is
      // what invalidated equal-thread comparisons before.
      overrides.LIZARD_NATIVE_CPU_WORKERS_REASON =
        `capped from ${requestedThreads} to ${effectiveThreads} physical cores for the GPU lane`;
    }
  }
  const requiredAdapterLuid = profile?.remoteExecution?.requiredAdapterLuid ?? null;
  if (requiredAdapterLuid !== null) {
    if (!/^-?\d+:\d+$/.test(String(requiredAdapterLuid))) {
      throw new Error("Remote execution requiredAdapterLuid is invalid");
    }
    overrides.LIZARD_NATIVE_GPU_ADAPTER_LUID = String(requiredAdapterLuid);
  } else if (env?.LIZARD_NATIVE_GPU_ADAPTER_LUID == null) {
    const identity = plan?.hardware?.adapterIdentity;
    if (plan?.mode !== "cpu-only"
      && identity
      && normalizeNumber(identity.luidHigh) != null
      && normalizeNumber(identity.luidLow) != null) {
      overrides.LIZARD_NATIVE_GPU_ADAPTER_LUID = `${identity.luidHigh}:${identity.luidLow}`;
    }
  }
  let budgetMb = normalizeNumber(env?.LIZARD_NATIVE_GPU_WEIGHT_BUDGET_MB);
  if (budgetMb == null) {
    budgetMb = resolveNativeGpuWeightBudgetMb({ profile, modelArtifact, runtimeProvider });
    if (budgetMb != null) {
      overrides.LIZARD_NATIVE_GPU_WEIGHT_BUDGET_MB = String(budgetMb);
    }
  }
  // Split encode/decode flow: the complete resident token graph is
  // all-or-nothing - it stages EVERY projection weight, and once any prior
  // token has GPU KV-cache state, a mid-sequence staging failure ("D3D12
  // resident staging stopped at the configured GPU weight budget") is a hard
  // NATIVE_GPU_DECODE_REQUIRED error even under gpuDecodePolicy "allowed"
  // (main.cpp gemma consume-token path). When the model's weight footprint
  // is known to exceed the budget, the graph can never fully stage, so skip
  // it up front: GPU hybrid prefill (encode) still offloads, and decode runs
  // the per-layer bounded-resident path with explicit CPU fallback. This is
  // what a dedicated-VRAM card that is smaller than the model (e.g. 2 GB
  // NVIDIA/AMD, no shared-memory budget) and an over-budget UMA model both
  // need. An explicit LIZARD_NATIVE_GPU_TOKEN_GRAPH in the environment
  // always wins.
  if (env?.LIZARD_NATIVE_GPU_TOKEN_GRAPH == null && budgetMb != null) {
    const knownWeightBytes = resolveModelWeightBytes(modelArtifact, profile);
    // +1 MiB slack: the budget was truncated to whole MiB, so a budget that
    // was raised exactly to the model's size must not read as "over budget".
    if (knownWeightBytes != null && knownWeightBytes > budgetMb * MB + MB) {
      overrides.LIZARD_NATIVE_GPU_TOKEN_GRAPH = "0";
    }
  }
  // Phase 2 no-thrash policy: prefill and decode are independent. A model
  // whose weights cannot remain inside the planned resident decode budget
  // may still use Direct3D for batched prefill, but decode stays on CPU
  // instead of cycling the full model through an LRU cache every token.
  // Explicit environment overrides remain available to benchmark forced GPU
  // behavior and never get silently replaced here.
  if (env?.LIZARD_NATIVE_GPU_DECODE == null && plan?.execution?.allowGpuDecode === false) {
    overrides.LIZARD_NATIVE_GPU_DECODE = "0";
    overrides.LIZARD_NATIVE_GPU_DECODE_REASON = plan.execution.decodeReason;
  }
  // A CPU-only plan cannot consume resident D3D12 weights. Pre-staging them
  // nevertheless duplicated roughly 2 GiB of this model in shared GPU memory,
  // increased peak RAM, and displaced the memory-mapped CPU weights that the
  // decode loop immediately needed. Keep pre-stage available for split
  // GPU-prefill/CPU-decode plans, where it still removes real first-prefill
  // work, and preserve an explicit operator override.
  if (env?.LIZARD_NATIVE_GPU_PRESTAGE == null && plan?.mode === "cpu-only") {
    overrides.LIZARD_NATIVE_GPU_PRESTAGE = "0";
  }
  if (env?.LIZARD_NATIVE_GPU_TOKEN_GRAPH == null
    && plan?.execution?.allowCompleteTokenGraph === false) {
    overrides.LIZARD_NATIVE_GPU_TOKEN_GRAPH = "0";
  }
  // Discrete-GPU streaming decode (model larger than VRAM): tell the native
  // resident-weight cache to evict/stream per layer instead of failing at the
  // budget, and skip the load-time pre-stage - for a model that can never fully
  // fit, pre-staging only churns the cache before decode even starts.
  if (env?.LIZARD_NATIVE_GPU_WEIGHT_STREAMING == null
    && plan?.execution?.streamingResidentDecode === true) {
    overrides.LIZARD_NATIVE_GPU_WEIGHT_STREAMING = "1";
    if (env?.LIZARD_NATIVE_GPU_PRESTAGE == null) {
      overrides.LIZARD_NATIVE_GPU_PRESTAGE = "0";
    }
  }
  return overrides;
}

// The compiled runner binary on disk, or null when only the JS stub worker
// is available. Used by callers that need the real native executable (the
// hardware probe cannot run on the stub).
export function resolveCompiledNativeRunnerPath(config = null) {
  const configuredBinary = config?.engine?.native?.runnerBinary ?? null;
  if (configuredBinary && fs.existsSync(path.resolve(configuredBinary))) {
    return path.resolve(configuredBinary);
  }
  return detectedRunnerCandidates(config).find((candidate) => fs.existsSync(candidate)) ?? null;
}

function optimizedRunnerPath(baselineBinary, variant) {
  const extension = path.extname(baselineBinary);
  return path.join(
    path.dirname(baselineBinary),
    `${path.basename(baselineBinary, extension)}-${variant}${extension}`,
  );
}

function nativeRunnerSelfTest(binary) {
  const result = spawnSync(binary, ["--cpu-selftest"], {
    encoding: "utf8",
    timeout: 10_000,
    windowsHide: true,
  });
  return result.status === 0 && !result.error;
}

export function selectNativeRunnerBinary({
  baselineBinary,
  runnerTier = "baseline",
  env = process.env,
  selfTest = nativeRunnerSelfTest,
} = {}) {
  if (!baselineBinary) {
    return null;
  }
  const detectedTier = String(runnerTier).toLowerCase();
  const requestedTier = String(env?.LIZARD_NATIVE_CPU_VARIANT ?? "").toLowerCase();
  if (requestedTier === "baseline") {
    return baselineBinary;
  }
  if (!["avx2", "avx512"].includes(detectedTier)) {
    return baselineBinary;
  }
  // AVX-512 is an additional release binary, not an alias for AVX2. When it is
  // explicitly requested on an AVX-512-capable machine, retain AVX2 and the
  // portable binary as fail-closed fallbacks. A missing binary, OS-disabled
  // extended state, or illegal-instruction self-test can therefore never make
  // model loading unavailable.
  // The 2026-08-01 interleaved A/B on an AVX-512-capable i7-1165G7 measured
  // only +0.47% median warm throughput over AVX2, below the 5% noise floor,
  // and a materially worse outlier. Treat AVX-512 as a capability ceiling and
  // an explicit experiment until a model/hardware calibration proves it wins;
  // highest ISA is not synonymous with highest sustained throughput because
  // wide-vector downclocking is CPU-specific.
  const variants = detectedTier === "avx512" && requestedTier === "avx512"
    ? ["avx512", "avx2"]
    : ["avx2"];
  for (const variant of variants) {
    const optimizedBinary = optimizedRunnerPath(baselineBinary, variant);
    if (!fs.existsSync(optimizedBinary)) {
      continue;
    }
    try {
      if (selfTest(optimizedBinary)) {
        return optimizedBinary;
      }
    } catch {
      // Continue down the capability ladder.
    }
  }
  return baselineBinary;
}

export function resolveNativeWorkerLaunchSpec({
  config = null,
  modelArtifact = null,
  profile = null,
  env = process.env,
  selfTest = nativeRunnerSelfTest,
  runtimeProvider = "lizard-native",
} = {}) {
  if (!prefersCompiledNativeWorker(modelArtifact)) {
    return {
      command: process.execPath,
      args: [scriptRunnerPath()],
    };
  }
  // Ostrich, like Caterpillar, is its own binary with its own engine key. It
  // used to fall into "native" here, so `serve --provider ostrich` resolved
  // config.engine.native.runnerBinary and launched lizard-native-runner-avx2:
  // a different engine reporting itself as Ostrich, at 3.94 tok/s instead of
  // the runner's 7.33. Neither binary ships an -avx2 sibling, so both also
  // skip the SIMD-variant ladder below.
  const OWN_BINARY_PROVIDERS = new Set(["caterpillar", "ostrich"]);
  const providerKey = OWN_BINARY_PROVIDERS.has(runtimeProvider) ? runtimeProvider : "native";
  const configuredBinary = config?.engine?.[providerKey]?.runnerBinary ?? null;
  if (configuredBinary) {
    if (OWN_BINARY_PROVIDERS.has(runtimeProvider)) {
      return { command: path.resolve(configuredBinary), args: [] };
    }
    return {
      command: selectNativeRunnerBinary({
        baselineBinary: path.resolve(configuredBinary),
        runnerTier: profile?.capabilityRegistry?.cpu?.runnerTier,
        env,
        selfTest,
      }),
      args: [],
    };
  }
  const detectedBinary = detectedRunnerCandidates(config, runtimeProvider).find((candidate) => fs.existsSync(candidate)) ?? null;
  if (detectedBinary) {
    if (OWN_BINARY_PROVIDERS.has(runtimeProvider)) {
      return { command: detectedBinary, args: [] };
    }
    return {
      command: selectNativeRunnerBinary({
        baselineBinary: detectedBinary,
        runnerTier: profile?.capabilityRegistry?.cpu?.runnerTier,
        env,
        selfTest,
      }),
      args: [],
    };
  }
  if (runtimeProvider === "caterpillar" && prefersCompiledNativeWorker(modelArtifact)) {
    throw new Error("Caterpillar GGUF inference requires a built caterpillar-runner binary.");
  }
  return {
    command: process.execPath,
    args: [scriptRunnerPath()],
  };
}

export function describeNativeWorkerExit(code, signal = null) {
  if (code == null) {
    return signal
      ? `Native worker was terminated by signal ${signal}.`
      : "Native worker exited without reporting an exit code.";
  }
  const numericCode = Number(code);
  const unsignedCode = Number.isFinite(numericCode) ? (numericCode >>> 0) : null;
  const hexCode = unsignedCode == null
    ? null
    : `0x${unsignedCode.toString(16).toUpperCase().padStart(8, "0")}`;
  if (unsignedCode === 0xC000001D) {
    return `Native worker exited with code ${numericCode} (${hexCode}, STATUS_ILLEGAL_INSTRUCTION). `
      + "The worker binary contains a CPU instruction this processor cannot execute. "
      + "Install a portable Lizard build; release binaries must use the portable x86-64 CPU profile.";
  }
  if (unsignedCode === 0xC0000135) {
    return `Native worker exited with code ${numericCode} (${hexCode}, STATUS_DLL_NOT_FOUND). `
      + "A required Windows runtime or system DLL could not be loaded.";
  }
  if (unsignedCode === 0xC0000005) {
    return `Native worker exited with code ${numericCode} (${hexCode}, STATUS_ACCESS_VIOLATION).`;
  }
  return `Native worker exited with code ${numericCode}${hexCode ? ` (${hexCode})` : ""}.`;
}

export class NativeWorkerProcessClient {
  constructor({
    config = null,
    modelArtifact = null,
    profile = null,
    rpcVersion = DEFAULT_RPC_VERSION,
    launchSpec = null,
    runtimeProvider = "lizard-native",
  } = {}) {
    this.rpcVersion = rpcVersion;
    this._launchSpec = launchSpec ?? resolveNativeWorkerLaunchSpec({
      config,
      modelArtifact,
      profile,
      runtimeProvider,
    });
    this._nextId = 1;
    this._pending = new Map();
    this._closed = false;
    this._onTokenEvent = null;
    this._sessionTokenListeners = new Map();
    this._launchEnv = {
      ...process.env,
      ...resolveNativeWorkerEnvironment({ profile, modelArtifact, runtimeProvider }),
    };

    this.child = spawn(this._launchSpec.command, this._launchSpec.args, {
      stdio: ["pipe", "pipe", "pipe"],
      env: this._launchEnv,
      windowsHide: true,
    });
    this._stderr = [];
    this.child.stderr.on("data", (chunk) => {
      this._stderr.push(chunk.toString("utf8"));
      if (this._stderr.length > 32) {
        this._stderr.shift();
      }
    });
    this.child.on("exit", (code, signal) => {
      const exitDescription = describeNativeWorkerExit(code, signal);
      const message = `${exitDescription}${this._stderr.length ? ` ${this._stderr.join("").trim()}` : ""}`;
      for (const pending of this._pending.values()) {
        pending.reject(new Error(message));
      }
      this._pending.clear();
      this._closed = true;
    });
    this.child.on("error", (error) => {
      for (const pending of this._pending.values()) {
        pending.reject(error);
      }
      this._pending.clear();
      this._closed = true;
    });

    attachBoundedLineReader(this.child.stdout, {
      onError: (error) => {
        for (const pending of this._pending.values()) pending.reject(error);
        this._pending.clear();
        this._forceTerminate("emitted an oversized RPC line");
      },
      onLine: (line) => {
      if (!line.trim()) {
        return;
      }
      let payload;
      try {
        payload = JSON.parse(line);
      } catch (error) {
        for (const pending of this._pending.values()) {
          pending.reject(new Error(`Native worker emitted invalid JSON: ${error.message}`));
        }
        this._pending.clear();
        return;
      }
      // JSON-RPC notifications carry no id; the only one the worker emits is
      // the per-token generate progress event used for live streaming.
      //
      // EnhancementRun_04: route live tokens by the native session identity.
      // Caterpillar omitted sessionId until protocol Run 04 Phase 1, which made
      // its per-session listeners unreachable and could drop concurrent live
      // tokens. An event for a session nobody is listening to is dropped,
      // never thrown: it is a normal race when generate resolves while its
      // final notification is still in flight.
      if (payload.id === undefined && payload.method === "generate.token") {
        const params = payload.params ?? {};
        const listener = this._sessionTokenListeners.get(params.sessionId)
          ?? this._onTokenEvent;
        try {
          listener?.(params);
        } catch {
          // A listener failure must never break the RPC channel.
        }
        return;
      }
      const pending = this._pending.get(payload.id);
      if (!pending) {
        return;
      }
      this._pending.delete(payload.id);
      if (payload.ok) {
        pending.resolve(payload.result ?? null);
        return;
      }
      const error = new Error(payload.error?.message ?? "Native worker request failed.");
      error.code = payload.error?.code ?? "NATIVE_WORKER_ERROR";
      error.details = payload.error?.details ?? null;
      pending.reject(error);
      },
    });
  }

  async request(method, params = {}, { timeoutMs = null } = {}) {
    if (this._closed) {
      throw new Error("Native worker process client is closed.");
    }
    // The worker only reads profile.mmap/mlock and profile.nativeRuntime.*;
    // it parses the GGUF file itself. The JS-side profile however carries
    // the full gguf metadata (incl. the 262k-token vocabulary, ~9.6 MB
    // serialized) plus the capability registry - stringifying, piping, and
    // worker-parsing that on EVERY prefill/generate request cost hundreds
    // of milliseconds per request (measured 2026-07-06). Strip the fields
    // the worker never reads before serializing.
    //
    // One exception: load_model is the request whose compatibility verdict is
    // computed FROM the capability registry (does this model's weight footprint
    // fit the memory budget). Stripping it there did not save anything - it
    // happens once per model - and it silently turned the memory guard into a
    // no-op: the worker saw no memory facts, reported "could not be determined",
    // and every model loaded regardless of size. It gets the memory block only,
    // which is a few hundred bytes; the 9.6 MB vocabulary still never ships.
    let rpcParams = params;
    if (params && typeof params === "object"
      && params.profile && typeof params.profile === "object"
      && (params.profile.gguf !== undefined || params.profile.capabilityRegistry !== undefined)) {
      const { gguf, capabilityRegistry, ...slimProfile } = params.profile;
      rpcParams = method === "load_model" && capabilityRegistry?.memory
        ? { ...params, profile: { ...slimProfile, capabilityRegistry: { memory: capabilityRegistry.memory } } }
        : { ...params, profile: slimProfile };
    }
    const id = this._nextId++;
    const payload = {
      id,
      rpcVersion: this.rpcVersion,
      method,
      params: rpcParams,
    };
    const message = `${JSON.stringify(payload)}\n`;
    if (Buffer.byteLength(message, "utf8") > MAX_NATIVE_RPC_LINE_BYTES) {
      const error = new Error(`Native worker RPC request exceeds the ${MAX_NATIVE_RPC_LINE_BYTES}-byte line limit.`);
      error.code = "NATIVE_WORKER_RPC_LINE_TOO_LARGE";
      return Promise.reject(error);
    }
    const deadlineMs = timeoutMs ?? resolveRpcTimeoutMs(method);
    return new Promise((resolve, reject) => {
      let timer = null;
      // Every settle path clears the deadline and drops the pending entry, so a
      // reply, a worker exit and a timeout can race without the losers firing
      // later against an already-settled promise.
      const settle = (finish) => (value) => {
        if (timer) {
          clearTimeout(timer);
          timer = null;
        }
        this._pending.delete(id);
        finish(value);
      };
      const entry = { resolve: settle(resolve), reject: settle(reject) };
      this._pending.set(id, entry);
      if (Number.isFinite(deadlineMs) && deadlineMs > 0) {
        timer = setTimeout(() => {
          const error = new Error(
            `Native worker did not answer '${method}' within ${Math.round(deadlineMs / 1000)}s. `
            + "The worker was stopped; the model was not loaded.",
          );
          error.code = "NATIVE_WORKER_TIMEOUT";
          entry.reject(error);
          // A worker that missed a lifecycle deadline is not coming back, and
          // leaving it running is what made the failure permanent: it keeps the
          // GGUF mapped, so the retry cannot load and a delete cannot unlink.
          this._forceTerminate(`timed out on '${method}'`);
        }, deadlineMs);
        // Never hold the process open just to watch for a timeout.
        timer.unref?.();
      }
      this.child.stdin.write(message, "utf8", (error) => {
        if (error) {
          entry.reject(error);
        }
      });
    });
  }

  // Ends the worker for good and fails everything still waiting on it.
  //
  // `child.kill()` alone is not enough: on Windows it is a TerminateProcess on
  // the direct child only, which a runner wedged inside a driver call - or one
  // that spawned helpers - can outlive. Until it is really gone it holds the
  // model file mapped, which is what turned one bad load into "nothing works
  // until reboot".
  _forceTerminate(reason = "terminated") {
    const alreadyTerminating = this._terminating === true;
    this._terminating = true;
    // Marked closed immediately: the client is unusable either way, and a fast
    // "closed" rejection is what callers need instead of another silent wait.
    this._closed = true;
    for (const pending of this._pending.values()) {
      const error = new Error(`Native worker ${reason}.`);
      error.code = "NATIVE_WORKER_TERMINATED";
      pending.reject(error);
    }
    this._pending.clear();
    if (alreadyTerminating) {
      return;
    }
    try {
      this.child.stdin?.destroy?.();
    } catch {
      // The pipe may already be torn down; the kill below is what matters.
    }
    try {
      this.child.kill();
    } catch {
      // Already gone.
    }
    const pid = this.child?.pid;
    const escalation = setTimeout(() => {
      if (this.child?.exitCode != null || this.child?.signalCode != null) {
        return;
      }
      try {
        if (process.platform === "win32" && pid) {
          // /t takes the worker's own children with it, /f skips the polite ask
          // the process is by definition too wedged to answer.
          spawn("taskkill.exe", ["/pid", String(pid), "/t", "/f"], {
            stdio: "ignore",
            windowsHide: true,
          }).on("error", () => {});
        } else {
          this.child.kill("SIGKILL");
        }
      } catch {
        // Nothing further to try; the caller's deadline has already fired.
      }
    }, WORKER_KILL_ESCALATION_MS);
    escalation.unref?.();
  }

  // Resolves true once the OS has actually reaped the worker. Callers that are
  // about to unlink the model file need the exit, not just the kill request.
  _awaitExit(timeoutMs = WORKER_EXIT_WAIT_MS) {
    if (this.child?.exitCode != null || this.child?.signalCode != null) {
      return Promise.resolve(true);
    }
    return new Promise((resolve) => {
      const timer = setTimeout(() => resolve(false), timeoutMs);
      timer.unref?.();
      this.child.once("exit", () => {
        clearTimeout(timer);
        resolve(true);
      });
    });
  }

  async handshake() {
    const result = await this.request("handshake", {});
    return validateWorkerRpcHandshake(result, this.rpcVersion);
  }

  probe() {
    return this.request("probe", {});
  }

  loadModel(params) {
    return this.request("load_model", params);
  }

  unloadModel(params = {}) {
    return this.request("unload_model", params);
  }

  openSession(params) {
    return this.request("session_open", params);
  }

  cloneSession(params) {
    return this.request("session_clone", params);
  }

  closeSession(params) {
    return this.request("session_close", params);
  }

  prefill(params) {
    return this.request("prefill", params);
  }

  decode(params) {
    return this.request("decode", params);
  }

  async generate(params, onToken = null) {
    if (typeof onToken !== "function") {
      return this.request("generate", params);
    }
    // Registered under this call's sessionId so two concurrent generates
    // deliver to their own listeners rather than the last one registered
    // winning. `_onTokenEvent` remains as the fallback for callers that
    // stream without a sessionId.
    const sessionId = params?.sessionId ?? null;
    if (sessionId != null) {
      this._sessionTokenListeners.set(sessionId, onToken);
    } else {
      this._onTokenEvent = onToken;
    }
    try {
      return await this.request("generate", { ...params, emitTokenEvents: true });
    } finally {
      if (sessionId != null) {
        this._sessionTokenListeners.delete(sessionId);
      } else {
        this._onTokenEvent = null;
      }
    }
  }

  cancel(params) {
    return this.request("cancel", params);
  }

  metrics(params = {}) {
    return this.request("metrics", params);
  }

  tensorBenchmark(params = {}) {
    return this.request("tensor_benchmark", params);
  }

  // Returns once the worker is gone - not once it has been asked to leave.
  //
  // The graceful `shutdown` RPC used to be awaited without a deadline, which was
  // the second half of the model-selection hang: a wedged worker never answers
  // it, so the await never returned, the `finally` that kills the child never
  // ran, and every caller - Unload, Delete, releaseModelWorker, the SIGINT
  // handler - stopped there. Each of them wraps this in `.catch()`, but a
  // promise that never settles never reaches a catch, so the dashboard went
  // unreachable and the port stayed bound by a process nothing could dislodge.
  async shutdown() {
    if (this._closed) {
      // Still wait out an in-flight termination: the caller may be about to
      // unlink the model file this process has mapped.
      await this._awaitExit();
      return;
    }
    try {
      await this.request("shutdown", {});
    } catch {
      // Deliberate: a worker that will not leave politely still has to go, and
      // the contract here is "this returns once it is gone", not "it agreed".
    } finally {
      this._forceTerminate("shut down");
      await this._awaitExit();
    }
  }
}
