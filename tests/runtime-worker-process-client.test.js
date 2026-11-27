import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import process from "node:process";
import {
  describeNativeWorkerExit,
  resolveNativeGpuWeightBudgetMb,
  resolveNativeWorkerEnvironment,
  resolveNativeWorkerLaunchSpec,
  selectNativeRunnerBinary,
  validateWorkerRpcHandshake,
} from "../src/runtime/worker/process-client.js";

test("worker RPC handshake accepts the selected version inside its advertised range", () => {
  const handshake = {
    rpcVersion: 1,
    rpcMinVersion: 1,
    rpcMaxVersion: 1,
    worker: "caterpillar-runner",
  };
  assert.equal(validateWorkerRpcHandshake(handshake, 1), handshake);
});

test("worker RPC handshake fails closed on missing or incompatible versions", () => {
  for (const handshake of [{}, { rpcVersion: 2 }, { rpcVersion: 1, rpcMinVersion: 2, rpcMaxVersion: 3 }]) {
    assert.throws(
      () => validateWorkerRpcHandshake(handshake, 1),
      (error) => error.code === "NATIVE_WORKER_RPC_VERSION_UNSUPPORTED",
    );
  }
});

test("describeNativeWorkerExit identifies incompatible CPU instruction crashes", () => {
  const description = describeNativeWorkerExit(3221225501);
  assert.match(description, /0xC000001D/);
  assert.match(description, /STATUS_ILLEGAL_INSTRUCTION/);
  assert.match(description, /portable x86-64 CPU profile/);
  assert.doesNotMatch(description, /missing.*runtime/i);
});

test("describeNativeWorkerExit distinguishes a missing DLL from an illegal instruction", () => {
  const description = describeNativeWorkerExit(3221225781);
  assert.match(description, /0xC0000135/);
  assert.match(description, /STATUS_DLL_NOT_FOUND/);
});

function tempDir() {
  return fs.mkdtempSync(path.join(os.tmpdir(), "lizard-native-worker-"));
}

function touch(filePath) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, "");
}

function scriptRunnerPath() {
  return path.resolve(process.cwd(), "scripts", "lizard-native-runner.mjs");
}

function nativeBinaryName() {
  return process.platform === "win32" ? "lizard-native-runner.exe" : "lizard-native-runner";
}

function caterpillarBinaryName() {
  return process.platform === "win32" ? "caterpillar-runner.exe" : "caterpillar-runner";
}

function avx2BinaryName() {
  return process.platform === "win32" ? "lizard-native-runner-avx2.exe" : "lizard-native-runner-avx2";
}

function avx512BinaryName() {
  return process.platform === "win32" ? "lizard-native-runner-avx512.exe" : "lizard-native-runner-avx512";
}

test("native runner selection uses AVX2 only after capability detection and self-test", () => {
  const root = tempDir();
  const baseline = path.join(root, nativeBinaryName());
  const avx2 = path.join(root, avx2BinaryName());
  touch(baseline);
  touch(avx2);
  assert.equal(selectNativeRunnerBinary({
    baselineBinary: baseline,
    runnerTier: "avx2",
    selfTest: (candidate) => candidate === avx2,
  }), avx2);
  assert.equal(selectNativeRunnerBinary({
    baselineBinary: baseline,
    runnerTier: "avx2",
    selfTest: () => false,
  }), baseline);
});

test("native runner selection honors baseline override and older CPUs", () => {
  const root = tempDir();
  const baseline = path.join(root, nativeBinaryName());
  touch(baseline);
  touch(path.join(root, avx2BinaryName()));
  assert.equal(selectNativeRunnerBinary({
    baselineBinary: baseline,
    runnerTier: "avx2",
    env: { LIZARD_NATIVE_CPU_VARIANT: "baseline" },
    selfTest: () => true,
  }), baseline);
  assert.equal(selectNativeRunnerBinary({
    baselineBinary: baseline,
    runnerTier: "baseline",
    selfTest: () => true,
  }), baseline);
});

test("AVX-512 is explicit opt-in and falls back through AVX2", () => {
  const root = tempDir();
  const baseline = path.join(root, nativeBinaryName());
  const avx2 = path.join(root, avx2BinaryName());
  const avx512 = path.join(root, avx512BinaryName());
  touch(baseline);
  touch(avx2);
  touch(avx512);
  assert.equal(selectNativeRunnerBinary({
    baselineBinary: baseline,
    runnerTier: "avx512",
    env: { LIZARD_NATIVE_CPU_VARIANT: "avx512" },
    selfTest: () => true,
  }), avx512);
  assert.equal(selectNativeRunnerBinary({
    baselineBinary: baseline,
    runnerTier: "avx512",
    env: { LIZARD_NATIVE_CPU_VARIANT: "avx512" },
    selfTest: (candidate) => candidate === avx2,
  }), avx2);
  assert.equal(selectNativeRunnerBinary({
    baselineBinary: baseline,
    runnerTier: "avx512",
    selfTest: () => true,
  }), avx2, "an uncalibrated wider ISA must not replace the stable default");
  assert.equal(selectNativeRunnerBinary({
    baselineBinary: baseline,
    runnerTier: "avx512",
    env: { LIZARD_NATIVE_CPU_VARIANT: "avx2" },
    selfTest: () => true,
  }), avx2);
});

test("resolveNativeWorkerLaunchSpec keeps prototype models on the JS worker even when a compiled binary exists", () => {
  const root = tempDir();
  const buildDirectory = path.join(root, "build");
  touch(path.join(buildDirectory, nativeBinaryName()));

  const spec = resolveNativeWorkerLaunchSpec({
    config: {
      engine: {
        native: {
          buildDirectory,
        },
      },
    },
    modelArtifact: {
      modelId: "tiny-native",
      modelPath: path.join(root, "tiny-native-model.json"),
      runtimePrototype: {
        format: "lizard-tiny-linear",
      },
    },
  });

  assert.equal(spec.command, process.execPath);
  assert.deepEqual(spec.args, [scriptRunnerPath()]);
});

test("resolveNativeWorkerLaunchSpec prefers an explicitly configured binary for GGUF models", () => {
  const root = tempDir();
  const configuredBinary = path.join(root, nativeBinaryName());
  touch(configuredBinary);

  const spec = resolveNativeWorkerLaunchSpec({
    config: {
      engine: {
        native: {
          runnerBinary: configuredBinary,
        },
      },
    },
    modelArtifact: {
      modelId: "llama",
      modelPath: path.join(root, "demo.gguf"),
    },
  });

  assert.equal(spec.command, configuredBinary);
  assert.deepEqual(spec.args, []);
});

test("resolveNativeWorkerLaunchSpec launches the ostrich binary, never a native SIMD sibling", () => {
  // Ostrich once fell into the "native" engine key, so `serve --provider
  // ostrich` resolved engine.native.runnerBinary and then picked its -avx2
  // sibling: a different engine answering as Ostrich (measured 3.94 tok/s
  // against the ostrich runner's own 7.33). Ostrich ships one binary and must
  // never enter the SIMD-variant ladder.
  const root = tempDir();
  const ostrichBinary = path.join(root, "bin", "ostrich-runner.exe");
  const nativeBinary = path.join(root, "bin", "lizard-native-runner.exe");
  touch(ostrichBinary);
  touch(nativeBinary);
  touch(path.join(root, "bin", "lizard-native-runner-avx2.exe"));

  const spec = resolveNativeWorkerLaunchSpec({
    config: {
      engine: {
        native: { runnerBinary: nativeBinary },
        ostrich: { runnerBinary: ostrichBinary },
      },
    },
    modelArtifact: {
      modelId: "llama",
      modelPath: path.join(root, "demo.gguf"),
    },
    profile: { capabilityRegistry: { cpu: { runnerTier: "avx2" } } },
    runtimeProvider: "ostrich",
    selfTest: () => true,
  });

  assert.equal(spec.command, ostrichBinary);
  assert.deepEqual(spec.args, []);
});

test("resolveNativeWorkerLaunchSpec detects compiled build outputs for GGUF models", () => {
  const root = tempDir();
  const buildDirectory = path.join(root, "build");
  const detectedBinary = path.join(buildDirectory, nativeBinaryName());
  touch(detectedBinary);

  const spec = resolveNativeWorkerLaunchSpec({
    config: {
      engine: {
        native: {
          buildDirectory,
        },
      },
    },
    modelArtifact: {
      modelId: "llama",
      modelPath: path.join(root, "demo.gguf"),
    },
  });

  assert.equal(spec.command, detectedBinary);
  assert.deepEqual(spec.args, []);
});

test("resolveNativeWorkerLaunchSpec selects the distinct Caterpillar binary for GGUF models", () => {
  const root = tempDir();
  const buildDirectory = path.join(root, "caterpillar-build");
  const detectedBinary = path.join(buildDirectory, caterpillarBinaryName());
  touch(detectedBinary);

  const spec = resolveNativeWorkerLaunchSpec({
    config: {
      engine: {
        caterpillar: {
          buildDirectory,
        },
      },
    },
    modelArtifact: {
      modelId: "llama",
      modelPath: path.join(root, "demo.gguf"),
    },
    runtimeProvider: "caterpillar",
  });

  assert.equal(spec.command, detectedBinary);
  assert.deepEqual(spec.args, []);
});

test("resolveNativeWorkerLaunchSpec never silently substitutes the reference worker for a missing Caterpillar binary", () => {
  const root = tempDir();

  assert.throws(() => resolveNativeWorkerLaunchSpec({
    config: {
      engine: {
        caterpillar: {
          buildDirectory: path.join(root, "missing-caterpillar-build"),
        },
      },
    },
    modelArtifact: {
      modelId: "llama",
      modelPath: path.join(root, "demo.gguf"),
    },
    runtimeProvider: "caterpillar",
  }), /requires a built caterpillar-runner binary/i);
});

test("resolveNativeWorkerLaunchSpec finds the installer's {app}/bin binary even when config carries no native.buildDirectory/runnerBinary", () => {
  // Reproduces the installed-app bug (2026-07-08): standalone.js's warm-load
  // call omitted `config` entirely, so this function saw config === null and
  // fell through to detectedRunnerCandidates(), whose only candidate
  // directory was native/lizard-native-runner/build/ - a path the Windows
  // installer never ships (it stages the compiled worker at {app}\bin\ only,
  // see stageNativeRunner in build-windows-installer.mjs). That silently
  // selected the JS-only stub worker, which rejects every real GGUF, and the
  // broken worker got cached for the model's lifetime even though later
  // request-time code passed a correct config. resolveAppPath("bin", ...)
  // must be checked regardless of what config supplies.
  const originalAppRoot = process.env.LIZARD_APP_ROOT;
  const root = tempDir();
  const binBinary = path.join(root, "bin", nativeBinaryName());
  touch(binBinary);
  process.env.LIZARD_APP_ROOT = root;

  try {
    const spec = resolveNativeWorkerLaunchSpec({
      config: null,
      modelArtifact: {
        modelId: "llama",
        modelPath: path.join(root, "demo.gguf"),
      },
    });

    assert.equal(spec.command, binBinary);
    assert.deepEqual(spec.args, []);
  } finally {
    if (originalAppRoot === undefined) {
      delete process.env.LIZARD_APP_ROOT;
    } else {
      process.env.LIZARD_APP_ROOT = originalAppRoot;
    }
  }
});

test("resolveNativeWorkerLaunchSpec falls back to the JS worker when no compiled binary is available", () => {
  const root = tempDir();

  const spec = resolveNativeWorkerLaunchSpec({
    config: {
      engine: {
        native: {
          buildDirectory: path.join(root, "missing-build"),
        },
      },
    },
    modelArtifact: {
      modelId: "llama",
      modelPath: path.join(root, "demo.gguf"),
    },
  });

  assert.equal(spec.command, process.execPath);
  assert.deepEqual(spec.args, [scriptRunnerPath()]);
});

test("resolveNativeGpuWeightBudgetMb uses shared memory for resident weights on UMA plans", () => {
  const budgetMb = resolveNativeGpuWeightBudgetMb({
    profile: {
      backend: "cpu",
      capabilityRegistry: {
        os: "windows",
        memory: {
          availableGb: 13.4,
        },
        gpu: [
          {
            name: "Intel Iris Xe Graphics",
            vendor: "intel",
            vramGb: 1,
          },
        ],
      },
    },
  });

  assert.equal(budgetMb, 4015);
});

test("resolveNativeGpuWeightBudgetMb gives all supported UMA quantizations the same resident budget", () => {
  const budgetMb = resolveNativeGpuWeightBudgetMb({
    profile: {
      backend: "cpu",
      gguf: {
        quant: "q6_k",
      },
      capabilityRegistry: {
        os: "windows",
        memory: {
          availableGb: 13.4,
        },
        gpu: [
          {
            name: "Intel Iris Xe Graphics",
            vendor: "intel",
            vramGb: 1,
          },
        ],
      },
    },
  });

  assert.equal(budgetMb, 4015);
});

function umaHardwareProfile() {
  return {
    backend: "cpu",
    capabilityRegistry: {
      os: "windows",
      memory: {
        availableGb: 13.4,
      },
      gpu: [
        {
          name: "Intel Iris Xe Graphics",
          vendor: "intel",
          vramGb: 1,
        },
      ],
    },
  };
}

test("resolveNativeGpuWeightBudgetMb grows the budget to cover a known large model instead of hard-capping at 4096 MB", () => {
  // gemma4-e4b Q8_0's real downloaded file size (2026-07-07, iteration 29
  // bug: this model failed to serve with NATIVE_GPU_DECODE_REQUIRED
  // because the old flat 4096 MB cap sat below its ~8 GB weight need).
  const budgetMb = resolveNativeGpuWeightBudgetMb({
    profile: umaHardwareProfile(),
    modelArtifact: { gguf: { fileSizeBytes: 8031240160 } },
  });
  assert.equal(budgetMb, 7659);
  assert.ok(budgetMb > 4096, "budget must exceed the old flat cap for a model that needs more");
});

test("resolveNativeGpuWeightBudgetMb does not shrink the budget for a model smaller than the default share", () => {
  const budgetMb = resolveNativeGpuWeightBudgetMb({
    profile: umaHardwareProfile(),
    modelArtifact: { gguf: { fileSizeBytes: 1_000_000 } },
  });
  assert.equal(budgetMb, 4015);
});

test("resolveNativeGpuWeightBudgetMb never exceeds the hardware-derived ceiling even for an oversized model", () => {
  const budgetMb = resolveNativeGpuWeightBudgetMb({
    profile: umaHardwareProfile(),
    modelArtifact: { gguf: { fileSizeBytes: 40 * 1024 * 1024 * 1024 } }, // 40 GB, will not fit
  });
  // Falls back to the default share rather than lying about fitting a
  // model that plainly exceeds available memory - gpuDecodePolicy
  // "required" should still (correctly) refuse this model.
  assert.equal(budgetMb, 4015);
});

test("resolveNativeGpuWeightBudgetMb ignores model file size on dedicated-VRAM (non-UMA) hardware", () => {
  const budgetMb = resolveNativeGpuWeightBudgetMb({
    profile: {
      backend: "cpu",
      capabilityRegistry: {
        os: "windows",
        memory: { availableGb: 32 },
        gpu: [{ name: "RTX 4090", vendor: "nvidia", vramGb: 24 }],
      },
    },
    modelArtifact: { gguf: { fileSizeBytes: 8031240160 } },
  });
  assert.equal(budgetMb, 4096);
});

test("resolveNativeGpuWeightBudgetMb keeps a bounded hot-weight budget on a 2 GB NVIDIA GPU", () => {
  const budgetMb = resolveNativeGpuWeightBudgetMb({
    profile: {
      backend: "cpu",
      capabilityRegistry: {
        os: "windows",
        memory: { availableGb: 12 },
        gpu: [{ name: "NVIDIA GeForce MX", vendor: "nvidia", vramGb: 2 }],
      },
    },
    modelArtifact: { gguf: { fileSizeBytes: 3 * 1024 * 1024 * 1024 } },
  });
  assert.equal(budgetMb, 921);
});

test("resolveNativeWorkerEnvironment applies the derived GPU weight budget only to compiled GGUF workers", () => {
  const compiledEnv = resolveNativeWorkerEnvironment({
    profile: {
      backend: "cpu",
      capabilityRegistry: {
        os: "windows",
        memory: {
          availableGb: 13.4,
        },
        gpu: [
          {
            name: "Intel Iris Xe Graphics",
            vendor: "intel",
            vramGb: 1,
          },
        ],
      },
    },
    modelArtifact: {
      modelId: "gemma4-e4b",
      modelPath: "E:\\demo.gguf",
    },
    env: {},
  });
  const prototypeEnv = resolveNativeWorkerEnvironment({
    profile: {
      backend: "cpu",
      capabilityRegistry: {
        memory: {
          availableGb: 13.4,
        },
      },
    },
    modelArtifact: {
      modelId: "tiny-native",
      modelPath: "E:\\demo.json",
      runtimePrototype: {
        format: "lizard-tiny-linear",
      },
    },
    env: {},
  });

  assert.equal(compiledEnv.LIZARD_NATIVE_GPU_WEIGHT_BUDGET_MB, "4015");
  assert.deepEqual(prototypeEnv, {});
});

// One worker per physical core on the GPU lane.
//
// Measured on an i7-1165G7 (4 cores / 8 threads, llama-3.2-3B Q4_K_M,
// directx-hybrid, 64 tokens): 2 workers 6.30 tok/s, 4 workers 7.16, 6 workers
// 6.90, 8 workers 6.25. More CPU is not faster there - the decode loop is
// waiting on the device (3-8% CPU) and a second worker on the same core's SMT
// sibling costs throughput.
function gpuLaneProfile(threads, { cores = 4, threads: logical = 8 } = {}) {
  return {
    backend: "cpu",
    threads,
    gguf: { quant: "Q4_K_M" },
    capabilityRegistry: {
      os: "windows",
      cpu: { name: "Test CPU", cores, threads: logical },
      memory: { availableGb: 13.4 },
      gpu: [{ name: "Intel Iris Xe Graphics", vendor: "intel", vramGb: 1 }],
    },
    nativeRuntime: { enableHybridGpuPrefill: true, gpuDecodePolicy: "allowed" },
  };
}

const gpuLaneArtifact = {
  modelId: "llama-3.2-3b-instruct-gguf-q4_k_m",
  modelPath: "E:\\demo.gguf",
  gguf: { quant: "Q4_K_M" },
};

test("GPU-lane native workers are capped at the physical core count", () => {
  const environment = resolveNativeWorkerEnvironment({
    profile: gpuLaneProfile(5),
    modelArtifact: gpuLaneArtifact,
    env: {},
    runtimeProvider: "lizard-native",
  });
  assert.equal(environment.LIZARD_NATIVE_CPU_WORKERS, "4");
  // Declared and effective must never disagree silently: that mismatch is what
  // invalidated equal-thread comparisons before.
  assert.match(environment.LIZARD_NATIVE_CPU_WORKERS_REASON, /capped from 5 to 4/);
});

test("a thread count already at or below the core count is left alone", () => {
  const environment = resolveNativeWorkerEnvironment({
    profile: gpuLaneProfile(3),
    modelArtifact: gpuLaneArtifact,
    env: {},
    runtimeProvider: "lizard-native",
  });
  assert.equal(environment.LIZARD_NATIVE_CPU_WORKERS, "3");
  assert.equal(environment.LIZARD_NATIVE_CPU_WORKERS_REASON, undefined);
});

test("the CPU-only lane keeps its separately qualified thread count", () => {
  // Enhancement Run 06 tuned 5-of-8 for this model shape on the pinned ggml CPU
  // backend. The GPU-lane cap must not reach into that.
  const profile = gpuLaneProfile(5);
  profile.nativeRuntime = { forceCpuOnlyPlan: true };
  const environment = resolveNativeWorkerEnvironment({
    profile,
    modelArtifact: gpuLaneArtifact,
    env: {},
    runtimeProvider: "lizard-native",
  });
  assert.equal(environment.LIZARD_NATIVE_CPU_WORKERS, "5");
  assert.equal(environment.LIZARD_NATIVE_CPU_WORKERS_REASON, undefined);
});

test("an explicit worker count from the environment always wins", () => {
  const environment = resolveNativeWorkerEnvironment({
    profile: gpuLaneProfile(5),
    modelArtifact: gpuLaneArtifact,
    env: { LIZARD_NATIVE_CPU_WORKERS: "8" },
    runtimeProvider: "lizard-native",
  });
  assert.equal(environment.LIZARD_NATIVE_CPU_WORKERS, undefined);
});

test("resolveNativeWorkerEnvironment gives a signed remote adapter pin precedence over local environment", () => {
  const environment = resolveNativeWorkerEnvironment({
    profile: {
      backend: "directx-hybrid",
      remoteExecution: { requiredAdapterLuid: "0:42" },
    },
    modelArtifact: { format: "gguf", nativeRuntime: "compiled" },
    env: { LIZARD_NATIVE_GPU_ADAPTER_LUID: "0:99" },
  });
  assert.equal(environment.LIZARD_NATIVE_GPU_ADAPTER_LUID, "0:42");
});

test("Lizard native receives the profile's declared CPU worker count", () => {
  const environment = resolveNativeWorkerEnvironment({
    profile: { backend: "cpu", threads: 6 },
    modelArtifact: { modelPath: "E:\\demo.gguf" },
    runtimeProvider: "lizard-native",
    env: {},
  });
  assert.equal(environment.LIZARD_NATIVE_CPU_WORKERS, "6");
  assert.equal(environment.LIZARD_NATIVE_GPU_PRESTAGE, "0");

  const explicit = resolveNativeWorkerEnvironment({
    profile: { backend: "cpu", threads: 6 },
    modelArtifact: { modelPath: "E:\\demo.gguf" },
    runtimeProvider: "lizard-native",
    env: { LIZARD_NATIVE_CPU_WORKERS: "4" },
  });
  assert.equal(explicit.LIZARD_NATIVE_CPU_WORKERS, undefined);
});

test("CPU-only Lizard preserves an explicit GPU pre-stage override", () => {
  const environment = resolveNativeWorkerEnvironment({
    profile: { backend: "cpu", threads: 6 },
    modelArtifact: { modelPath: "E:\\demo.gguf" },
    runtimeProvider: "lizard-native",
    env: { LIZARD_NATIVE_GPU_PRESTAGE: "1" },
  });
  assert.equal(environment.LIZARD_NATIVE_GPU_PRESTAGE, undefined);
});

test("Caterpillar does not clamp its worker threads to the llama.cpp profile", () => {
  const environment = resolveNativeWorkerEnvironment({
    profile: { backend: "cpu", threads: 6 },
    modelArtifact: { modelPath: "E:\\demo.gguf" },
    runtimeProvider: "caterpillar",
    env: {},
  });

  // The runner defaults to full hardware concurrency; only an explicit
  // CATERPILLAR_CPU_THREADS env override may throttle it.
  assert.equal(environment.CATERPILLAR_CPU_THREADS, undefined);
});

test("resolveNativeWorkerEnvironment applies the shared UMA resident budget to heavy quantizations", () => {
  const compiledEnv = resolveNativeWorkerEnvironment({
    profile: {
      backend: "cpu",
      gguf: {
        quant: "q6_k",
      },
      capabilityRegistry: {
        os: "windows",
        memory: {
          availableGb: 13.4,
        },
        gpu: [
          {
            name: "Intel Iris Xe Graphics",
            vendor: "intel",
            vramGb: 1,
          },
        ],
      },
    },
    modelArtifact: {
      modelId: "gemma4-e4b",
      modelPath: "E:\\demo.gguf",
    },
    env: {},
  });

  assert.equal(compiledEnv.LIZARD_NATIVE_GPU_WEIGHT_BUDGET_MB, "4015");
});

test("resolveNativeWorkerEnvironment passes the catalog-known file size through to raise the budget for a real Q8_0 download", () => {
  // Mirrors the actual "serve" path: modelArtifact.gguf comes from the
  // local catalog entry (~/.lizard-llm/model-catalog.json), which carries
  // the real downloaded file's fileSizeBytes.
  const compiledEnv = resolveNativeWorkerEnvironment({
    profile: {
      backend: "cpu",
      capabilityRegistry: {
        os: "windows",
        memory: { availableGb: 13.4 },
        gpu: [{ name: "Intel Iris Xe Graphics", vendor: "intel", vramGb: 1 }],
      },
    },
    modelArtifact: {
      modelId: "gemma4-e4b-q8_0",
      modelPath: "E:\\gemma4-e4b-q8_0.gguf",
      gguf: { fileSizeBytes: 8031240160 },
    },
    env: {},
  });

  assert.equal(compiledEnv.LIZARD_NATIVE_GPU_WEIGHT_BUDGET_MB, "7659");
});

test("resolveNativeWorkerEnvironment splits encode/decode on a dedicated-VRAM card smaller than the model", () => {
  // 2 GB NVIDIA (no shared-memory budget): the complete resident token graph
  // can never stage a multi-GB model inside the ~921 MB hot-weight budget and
  // hard-fails mid-sequence once GPU KV state exists. The launcher must skip
  // the graph up front: GPU prefill (encode) still offloads, decode runs the
  // per-layer bounded-resident path with explicit CPU fallback.
  const compiledEnv = resolveNativeWorkerEnvironment({
    profile: {
      backend: "cpu",
      capabilityRegistry: {
        os: "windows",
        memory: { availableGb: 12 },
        gpu: [{ name: "NVIDIA GeForce MX", vendor: "nvidia", vramGb: 2 }],
      },
    },
    modelArtifact: {
      modelId: "gemma4-e4b-q4_k_m",
      modelPath: "E:\\gemma4-e4b-q4_k_m.gguf",
      gguf: { fileSizeBytes: 3 * 1024 * 1024 * 1024 },
    },
    env: {},
  });

  assert.equal(compiledEnv.LIZARD_NATIVE_GPU_WEIGHT_BUDGET_MB, "921");
  assert.equal(compiledEnv.LIZARD_NATIVE_GPU_TOKEN_GRAPH, "0");
  assert.equal(compiledEnv.LIZARD_NATIVE_GPU_DECODE, "0");
  assert.match(compiledEnv.LIZARD_NATIVE_GPU_DECODE_REASON, /eviction and re-upload/);
});

test("resolveNativeWorkerEnvironment streams decode over D3D12 on a discrete GPU when GPU decode is required", () => {
  // Same 2 GB NVIDIA card + oversized model, but the remote benchmark REQUIRES
  // GPU decode: instead of dropping to CPU, stream weights through the resident
  // lane (evict per layer) and skip the pointless load-time pre-stage.
  const compiledEnv = resolveNativeWorkerEnvironment({
    profile: {
      backend: "cpu",
      nativeRuntime: { gpuDecodePolicy: "required" },
      capabilityRegistry: {
        os: "windows",
        memory: { availableGb: 12 },
        gpu: [{ name: "NVIDIA GeForce GTX 950M", vendor: "nvidia", vramGb: 2 }],
      },
    },
    modelArtifact: {
      modelId: "llama-3.2-3b-q4_k_m",
      modelPath: "E:\\llama-3.2-3b-q4_k_m.gguf",
      gguf: { fileSizeBytes: 2 * 1024 * 1024 * 1024 },
    },
    env: {},
  });

  assert.equal(compiledEnv.LIZARD_NATIVE_GPU_WEIGHT_STREAMING, "1");
  assert.equal(compiledEnv.LIZARD_NATIVE_GPU_PRESTAGE, "0");
  // Streaming keeps decode ON the D3D12 lane (never forced to CPU) ...
  assert.notEqual(compiledEnv.LIZARD_NATIVE_GPU_DECODE, "0");
  // ... on the per-layer lane, not the all-or-nothing complete token graph.
  assert.equal(compiledEnv.LIZARD_NATIVE_GPU_TOKEN_GRAPH, "0");
});

test("resolveNativeWorkerEnvironment keeps the complete token graph when the model fits the budget", () => {
  // UMA machine where the known file size raised the budget to cover the
  // model: the graph can fully stage, so it stays enabled.
  const compiledEnv = resolveNativeWorkerEnvironment({
    profile: {
      backend: "cpu",
      capabilityRegistry: {
        os: "windows",
        memory: { availableGb: 13.4 },
        gpu: [{ name: "Intel Iris Xe Graphics", vendor: "intel", vramGb: 1 }],
      },
    },
    modelArtifact: {
      modelId: "gemma4-e4b-q8_0",
      modelPath: "E:\\gemma4-e4b-q8_0.gguf",
      gguf: { fileSizeBytes: 8031240160 },
    },
    env: {},
  });

  assert.equal(compiledEnv.LIZARD_NATIVE_GPU_TOKEN_GRAPH, undefined);
  assert.equal(compiledEnv.LIZARD_NATIVE_GPU_DECODE, undefined);
});

test("resolveNativeWorkerEnvironment disables the token graph for an over-budget UMA model", () => {
  // The 40 GB model cannot fit the ~3.9 GB UMA share; staging would stop at
  // the budget and hard-fail the required lane mid-sequence.
  const compiledEnv = resolveNativeWorkerEnvironment({
    profile: umaHardwareProfile(),
    modelArtifact: {
      modelId: "huge",
      modelPath: "E:\\huge.gguf",
      gguf: { fileSizeBytes: 40 * 1024 * 1024 * 1024 },
    },
    env: {},
  });

  assert.equal(compiledEnv.LIZARD_NATIVE_GPU_TOKEN_GRAPH, "0");
});

test("resolveNativeWorkerEnvironment respects explicit token-graph and budget environment overrides", () => {
  const withExplicitGraph = resolveNativeWorkerEnvironment({
    profile: umaHardwareProfile(),
    modelArtifact: {
      modelId: "huge",
      modelPath: "E:\\huge.gguf",
      gguf: { fileSizeBytes: 40 * 1024 * 1024 * 1024 },
    },
    env: { LIZARD_NATIVE_GPU_TOKEN_GRAPH: "1" },
  });
  assert.equal(withExplicitGraph.LIZARD_NATIVE_GPU_TOKEN_GRAPH, undefined);

  // A user-preset budget is not overridden, but it still drives the
  // encode/decode split decision for an over-budget model.
  const withPresetBudget = resolveNativeWorkerEnvironment({
    profile: umaHardwareProfile(),
    modelArtifact: {
      modelId: "gemma4-e4b-q8_0",
      modelPath: "E:\\gemma4-e4b-q8_0.gguf",
      gguf: { fileSizeBytes: 8031240160 },
    },
    env: { LIZARD_NATIVE_GPU_WEIGHT_BUDGET_MB: "2048" },
  });
  assert.equal(withPresetBudget.LIZARD_NATIVE_GPU_WEIGHT_BUDGET_MB, undefined);
  assert.equal(withPresetBudget.LIZARD_NATIVE_GPU_TOKEN_GRAPH, "0");
});

test("explicit GPU decode override wins over the automatic no-thrash policy", () => {
  const compiledEnv = resolveNativeWorkerEnvironment({
    profile: {
      backend: "cpu",
      capabilityRegistry: {
        os: "windows",
        memory: { availableGb: 12 },
        gpu: [{ name: "NVIDIA GeForce MX", vendor: "nvidia", vramGb: 2 }],
      },
    },
    modelArtifact: {
      modelId: "gemma4-e4b-q4_k_m",
      modelPath: "E:\\gemma4-e4b-q4_k_m.gguf",
      gguf: { fileSizeBytes: 3 * 1024 * 1024 * 1024 },
    },
    env: { LIZARD_NATIVE_GPU_DECODE: "1" },
  });

  assert.equal(compiledEnv.LIZARD_NATIVE_GPU_DECODE, undefined);
  assert.equal(compiledEnv.LIZARD_NATIVE_GPU_DECODE_REASON, undefined);
});
