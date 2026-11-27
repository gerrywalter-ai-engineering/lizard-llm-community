import { generateWithExistingBackend } from "../lib/generate.js";
import {
  buildNativeBackendFeatureRegistry,
  buildNativeBackendTestMatrix,
  resolveNativeBackendRoute,
} from "./backend/index.js";
import { ExistingBackendRuntime, InferenceRuntime } from "./core/existing-backend-runtime.js";
import { ModelSession } from "./core/model-session.js";
import {
  buildArchitectureCompatibilityMatrix,
  buildCaterpillarArchitectureCompatibilityMatrix,
  buildCaterpillarGgufFamilyMatrix,
  caterpillarRunnerLoadsFamily,
  buildCaterpillarQuantizationCompatibilityMatrix,
  buildNativeGgufFamilyMatrix,
  buildQuantizationCompatibilityMatrix,
  evaluateNativeGgufCompatibility,
  evaluateNativeCpuPrototypeSupport,
  resolveGgufFamilyKey,
} from "./compatibility/index.js";
import { buildRuntimeContractFields } from "./contract.js";
import { NativeCpuRuntime } from "./core/native-cpu-runtime.js";

export {
  ExistingBackendRuntime,
  InferenceRuntime,
  ModelSession,
  NativeCpuRuntime,
  buildNativeBackendFeatureRegistry,
  buildNativeBackendTestMatrix,
  resolveNativeBackendRoute,
  buildArchitectureCompatibilityMatrix,
  buildCaterpillarArchitectureCompatibilityMatrix,
  buildCaterpillarGgufFamilyMatrix,
  caterpillarRunnerLoadsFamily,
  buildCaterpillarQuantizationCompatibilityMatrix,
  buildNativeGgufFamilyMatrix,
  buildQuantizationCompatibilityMatrix,
  evaluateNativeGgufCompatibility,
  evaluateNativeCpuPrototypeSupport,
  resolveGgufFamilyKey,
};

export function resolveRuntimeEngine({
  requestedEngine = null,
  env = process.env,
} = {}) {
  const normalized = String(requestedEngine ?? "").trim().toLowerCase();
  const caterpillarFlag = String(env.LIZARD_CATERPILLAR_INFERENCE ?? "").trim() === "1";
  const nativeFlag = String(env.LIZARD_NATIVE_INFERENCE ?? "").trim() === "1";
  // experimental:true is what selects NativeCpuRuntime over
  // ExistingBackendRuntime in createInferenceRuntime. Without it Ostrich got
  // the llama.cpp-shaped runtime, which has no ensureModelLoaded - so
  // switching chat to Ostrich failed with
  // "getSharedRuntime(...).ensureModelLoaded is not a function" after the
  // model had already been released from Caterpillar.
  if (normalized === "ostrich") {
    return { engine: "ostrich", experimental: true, enabledBy: "cli" };
  }
  if (normalized === "caterpillar" || caterpillarFlag) {
    return {
      engine: "caterpillar",
      experimental: true,
      enabledBy: normalized === "caterpillar" ? "cli" : "env",
    };
  }
  if (normalized === "lizard-native" || nativeFlag) {
    return {
      engine: "lizard-native",
      experimental: true,
      enabledBy: normalized === "lizard-native" ? "cli" : "env",
    };
  }
  return {
    engine: "llama.cpp",
    experimental: false,
    enabledBy: null,
  };
}

export function resolveModelRuntimeEngine({
  requestedEngine = null,
  env = process.env,
  modelArtifact = null,
  profile = null,
} = {}) {
  const resolution = resolveRuntimeEngine({
    requestedEngine,
    env,
  });
  if (!resolution.experimental || modelArtifact?.runtimePrototype?.format) {
    return resolution;
  }
  const gguf = profile?.gguf ?? modelArtifact?.gguf ?? null;
  const looksLikeGguf = Boolean(
    gguf
    || String(modelArtifact?.modelPath ?? "").toLowerCase().endsWith(".gguf"),
  );
  const hasCompatibilityMetadata = Boolean(gguf?.architecture) && Boolean(gguf?.quant);
  if (!looksLikeGguf || !hasCompatibilityMetadata) {
    return resolution;
  }

  const compatibility = evaluateNativeGgufCompatibility({
    gguf,
    modelArtifact,
    profile,
  });
  if (compatibility.supported) {
    return resolution;
  }

  const architectureLabel = compatibility.architecture.key || "unknown";
  const quantizationLabel = compatibility.quantization.key || "unknown";
  return {
    engine: "llama.cpp",
    experimental: false,
    enabledBy: resolution.enabledBy,
    experimentalRequested: true,
    fallbackReason:
      `Native runtime is not yet implemented for GGUF architecture '${architectureLabel}' `
      + `with quantization '${quantizationLabel}'. Falling back to llama.cpp. `
      + compatibility.reason,
    nativeCompatibility: compatibility,
  };
}

export function resolveRuntimeProviderForModel({
  requestedProvider = null,
  env = process.env,
  modelArtifact = null,
  profile = null,
} = {}) {
  const normalizedProvider = String(requestedProvider ?? "").trim().toLowerCase();
  if (!["lizard-native", "caterpillar", "ostrich"].includes(normalizedProvider)) {
    return {
      provider: "llama.cpp",
      engine: "llama.cpp",
      experimental: false,
      experimentalRequested: false,
      enabledBy: null,
      fallbackReason: null,
      nativeCompatibility: null,
    };
  }

  const resolution = resolveModelRuntimeEngine({
    requestedEngine: normalizedProvider,
    env,
    modelArtifact,
    profile,
  });
  return {
    provider: resolution.engine === normalizedProvider ? normalizedProvider : "llama.cpp",
    engine: resolution.engine,
    experimental: Boolean(resolution.experimental),
    experimentalRequested: true,
    enabledBy: resolution.enabledBy ?? "provider",
    fallbackReason: resolution.fallbackReason ?? null,
    nativeCompatibility: resolution.nativeCompatibility ?? null,
  };
}

export function createInferenceRuntime({
  requestedEngine = null,
  env = process.env,
  generate = generateWithExistingBackend,
  modelArtifact = null,
  profile = null,
  maxConcurrentRequests = 1,
} = {}) {
  const resolution = resolveModelRuntimeEngine({
    requestedEngine,
    env,
    modelArtifact,
    profile,
  });
  return {
    resolution,
    runtime: resolution.experimental
      ? new NativeCpuRuntime({
        engine: resolution.engine,
        generate,
        maxConcurrentRequests,
      })
      : new ExistingBackendRuntime({
        engine: resolution.engine,
        generate,
      }),
  };
}

export function resolveRuntimeForCommand({
  command,
  requestedEngine = null,
  env = process.env,
  modelArtifact = null,
  profile = null,
  supportedExperimentalCommands = ["generate", "run", "benchmark"],
} = {}) {
  const resolution = resolveModelRuntimeEngine({
    requestedEngine,
    env,
    modelArtifact,
    profile,
  });
  if ((resolution.experimental || resolution.experimentalRequested) && !supportedExperimentalCommands.includes(command)) {
    throw new Error(`Experimental engine '${resolution.engine}' is not yet supported for ${command}. Supported experimental commands: ${supportedExperimentalCommands.join(", ")}.`);
  }
  return resolution;
}

export async function generateWithRuntimeSession({
  requestedEngine = null,
  env = process.env,
  recommendation,
  modelArtifact,
  profile = null,
  prompt,
  systemPrompt,
  messages = null,
  maxTokens,
  enginePath,
  config,
  logger,
  generate = generateWithExistingBackend,
  allowOracleFallback = false,
  onDelta,
  runtime: providedRuntime,
}) {
  const resolution = resolveRuntimeForCommand({
    command: "generate",
    requestedEngine,
    env,
    modelArtifact,
    profile,
  });
  // A caller that keeps its own long-lived runtime across requests (e.g. the
  // onboarding UI's chat dialog) can pass it in here instead of getting a
  // fresh one - a fresh NativeCpuRuntime has an empty worker cache, so it
  // reloads the model from disk (tens of seconds to minutes) on every call
  // even though the previous call's worker process is still alive and
  // holding it resident.
  const { runtime } = providedRuntime
    ? { runtime: providedRuntime }
    : createInferenceRuntime({
      requestedEngine: resolution.engine,
      env,
      generate,
      modelArtifact,
      profile,
    });
  const effectiveProfile = {
    ...(profile ?? {}),
    nativeRuntime: {
      ...(profile?.nativeRuntime ?? {}),
      allowOracleFallback,
    },
  };
  const session = await runtime.createSession({
    recommendation,
    modelArtifact,
    profile: effectiveProfile,
    config,
    logger,
    enginePath,
  });

  let output = "";
  let completed = null;
  try {
    for await (const event of session.generateStream({
      prompt,
      systemPrompt,
      // Structured turns go through the model's own chat template, so a
      // multi-turn chat ends on the model's end-of-turn token.
      messages: Array.isArray(messages) && messages.length ? messages : null,
      maxTokens,
      temperature: config?.generation?.temperature ?? null,
      topK: config?.generation?.topK ?? null,
      topP: config?.generation?.topP ?? null,
      seed: config?.generation?.seed ?? null,
      metadata: {
        requestedEngine: resolution.engine,
      },
    })) {
      if (event.type === "delta") {
        output += event.text ?? "";
        onDelta?.(event.text ?? "");
      }
      if (event.type === "complete") {
        completed = event;
      }
    }
  } finally {
    await session.close();
  }

  const contract = buildRuntimeContractFields(completed?.metadata ?? {});
  if (resolution.experimental && contract.runtimeProvider !== "lizard-native" && !contract.fallbackReason) {
    contract.fallbackReason = "native worker execution was unavailable; the oracle runtime handled this request";
  }
  if (!contract.fallbackReason && resolution.fallbackReason) {
    contract.fallbackReason = resolution.fallbackReason;
  }

  return {
    status: "completed",
    modelId: modelArtifact.modelId,
    modelPath: modelArtifact.modelPath,
    runtime: completed?.runtime ?? runtime.engine,
    engine: resolution.engine,
    experimentalEngine: resolution.experimental,
    durationMs: completed?.durationMs ?? null,
    profile: effectiveProfile,
    promptArtifact: completed?.metadata?.promptArtifact ?? null,
    kvCache: completed?.metadata?.kvCache ?? null,
    scheduler: completed?.metadata?.scheduler ?? null,
    nativeExecution: completed?.metadata?.nativeExecution ?? null,
    ...contract,
    output: completed?.output ?? output,
    stderr: "",
    note: (resolution.experimental || resolution.experimentalRequested)
      ? contract.fallbackReason
        ? resolution.experimental
          ? `Experimental native runtime requested via ${resolution.enabledBy}, but execution used the oracle fallback: ${contract.fallbackReason}`
          : `Experimental native runtime requested via ${resolution.enabledBy}, but execution used llama.cpp fallback: ${contract.fallbackReason}`
        : completed?.metadata?.nativeExecution?.mode === "worker-rpc"
          ? `Experimental native worker active via ${resolution.enabledBy}.`
          : `Experimental native runtime shell active via ${resolution.enabledBy}.`
      : undefined,
  };
}
