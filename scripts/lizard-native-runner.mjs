#!/usr/bin/env node

import readline from "node:readline";
import { inspectGgufFile } from "../src/lib/gguf-metadata.js";
import { createCpuPrototypeExecutor } from "../src/runtime/executor/index.js";
import { CPU_PROTOTYPE_FORMAT, loadCpuPrototypeModel } from "../src/runtime/loader/index.js";
import { evaluateNativeCpuPrototypeSupport } from "../src/runtime/compatibility/index.js";
import { NATIVE_RUNTIME_PROVIDER } from "../src/runtime/contract.js";

const workerState = {
  loadedModel: null,
  loadMs: null,
  lastMetrics: null,
  sessions: new Map(),
  protocolVersion: 1,
};

function respond(payload) {
  process.stdout.write(`${JSON.stringify(payload)}\n`);
}

function ok(id, result) {
  respond({ id, ok: true, result });
}

function fail(id, error) {
  respond({
    id,
    ok: false,
    error: {
      code: error.code ?? "NATIVE_WORKER_ERROR",
      message: error.message,
      details: error.details ?? null,
    },
  });
}

function unsupported(message, details = null) {
  const error = new Error(message);
  error.code = "NATIVE_MODEL_UNSUPPORTED";
  error.details = details;
  return error;
}

function normalizePrototype(modelArtifact = {}) {
  return modelArtifact?.runtimePrototype?.format === CPU_PROTOTYPE_FORMAT
    || String(modelArtifact?.modelPath ?? "").toLowerCase().endsWith(".json");
}

function tokenRate(tokenCount, decodeMs) {
  return decodeMs > 0 ? Number(((tokenCount / decodeMs) * 1000).toFixed(2)) : null;
}

function compatibilityForModel(model, params = {}) {
  return evaluateNativeCpuPrototypeSupport({
    model,
    profile: params.profile ?? null,
    promptArtifact: params.promptArtifact ?? null,
    kvPlan: params.kvPlan ?? null,
  });
}

async function handle(method, params = {}) {
  switch (method) {
    case "handshake":
      return {
        worker: "lizard-native-runner-js",
        rpcVersion: workerState.protocolVersion,
        runtimeProvider: NATIVE_RUNTIME_PROVIDER,
        methods: [
          "handshake",
          "probe",
          "load_model",
          "unload_model",
          "session_open",
          "prefill",
          "decode",
          "generate",
          "cancel",
          "metrics",
          "shutdown",
        ],
      };
    case "probe":
      return {
        runtimeProvider: NATIVE_RUNTIME_PROVIDER,
        backend: "cpu",
        architectures: ["tiny-linear"],
        quantizations: ["f32"],
        experimental: true,
      };
    case "load_model": {
      const startedAt = Date.now();
      if (!params?.modelArtifact?.modelPath) {
        throw new Error("load_model requires modelArtifact.modelPath.");
      }
      if (!normalizePrototype(params.modelArtifact)) {
        const gguf = inspectGgufFile(params.modelArtifact.modelPath, {
          fallbackQuant: params.profile?.gguf?.quant ?? params.modelArtifact?.gguf?.quant ?? null,
          fallbackArchitecture: params.profile?.gguf?.architecture ?? params.modelArtifact?.gguf?.architecture ?? null,
        });
        throw unsupported(
          `Native worker can parse GGUF metadata for ${gguf.architecture ?? "unknown"} ${gguf.quant ?? "unknown"}, but executable GGUF graph support is not implemented in this worker yet.`,
          { gguf },
        );
      }
      const model = loadCpuPrototypeModel({
        modelArtifact: params.modelArtifact,
      });
      workerState.loadedModel = {
        model,
        executor: createCpuPrototypeExecutor(),
        modelArtifact: params.modelArtifact,
        profile: params.profile ?? null,
      };
      workerState.loadMs = Math.max(1, Date.now() - startedAt);
      const compatibility = compatibilityForModel(model, {
        profile: params.profile ?? null,
      });
      return {
        runtimeProvider: NATIVE_RUNTIME_PROVIDER,
        nativeBackend: "cpu",
        nativeModelStatus: "loaded",
        // Reported, not asserted. This used to be a hard-coded `true`, which
        // made the load-time compatibility verdict decorative: a model whose
        // weights cannot fit the memory budget loaded happily and only failed
        // once it was generating. `supported` already folds in architecture,
        // quantization and memory; context stays null here because no prompt
        // exists yet, and is checked again at prefill.
        executionReady: compatibility?.supported !== false,
        loadMs: workerState.loadMs,
        model: {
          format: model.format,
          architecture: model.architecture,
          quantization: model.quantization,
          maxContextTokens: model.maxContextTokens,
          estimatedModelBytes: model.estimatedModelBytes,
        },
        compatibility,
      };
    }
    case "unload_model":
      workerState.loadedModel = null;
      workerState.sessions.clear();
      workerState.lastMetrics = null;
      return {
        nativeModelStatus: "unloaded",
      };
    case "session_open": {
      const sessionId = params.sessionId;
      if (!sessionId) {
        throw new Error("session_open requires sessionId.");
      }
      workerState.sessions.set(sessionId, {
        sessionId,
        cancelled: false,
        prompt: null,
        promptArtifact: null,
        kvPlan: null,
      });
      return {
        sessionId,
        nativeModelStatus: workerState.loadedModel ? "loaded" : "idle",
      };
    }
    case "prefill": {
      const session = workerState.sessions.get(params.sessionId);
      if (!session) {
        throw new Error(`Unknown session ${params.sessionId}.`);
      }
      const startedAt = Date.now();
      session.prompt = String(params.promptArtifact?.renderedPrompt ?? params.prompt ?? "");
      session.promptArtifact = params.promptArtifact ?? null;
      session.kvPlan = params.kvPlan ?? null;
      session.maxTokens = params.maxTokens ?? null;
      return {
        sessionId: session.sessionId,
        prefillMs: Math.max(1, Date.now() - startedAt),
        kvCacheBytes: session.kvPlan?.estimatedBytes ?? null,
      };
    }
    case "decode":
    case "generate": {
      if (!workerState.loadedModel) {
        throw new Error("No native model is loaded.");
      }
      const sessionId = params.sessionId;
      const session = workerState.sessions.get(sessionId) ?? {
        sessionId,
        cancelled: false,
      };
      if (session.cancelled) {
        const error = new Error(`Session ${sessionId} was cancelled.`);
        error.code = "NATIVE_CANCELLED";
        throw error;
      }
      const promptArtifact = params.promptArtifact ?? session.promptArtifact ?? null;
      const kvPlan = params.kvPlan ?? session.kvPlan ?? null;
      const compatibility = compatibilityForModel(workerState.loadedModel.model, {
        profile: params.profile ?? workerState.loadedModel.profile ?? null,
        promptArtifact,
        kvPlan,
      });
      if (!compatibility.supported) {
        throw unsupported("Native worker rejected the request because the prototype compatibility checks failed.", compatibility);
      }
      const prefillMs = Math.max(1, Number(params.prefillMs ?? 1));
      const decodeStartedAt = Date.now();
      const result = workerState.loadedModel.executor.generate({
        model: workerState.loadedModel.model,
        prompt: promptArtifact?.renderedPrompt ?? session.prompt ?? params.prompt ?? "",
        maxTokens: params.maxTokens ?? session.maxTokens ?? null,
      });
      const decodeMs = Math.max(1, Date.now() - decodeStartedAt);
      const generatedTokenCount = result.generatedTokenIds.length;
      const metrics = {
        runtimeProvider: NATIVE_RUNTIME_PROVIDER,
        nativeBackend: "cpu",
        nativeModelStatus: "loaded",
        loadMs: workerState.loadMs,
        prefillMs,
        decodeMs,
        tokensPerSecond: tokenRate(generatedTokenCount, decodeMs),
        kvCacheBytes: kvPlan?.estimatedBytes ?? null,
        output: result.output,
        chunks: result.chunks,
        generatedTokenIds: result.generatedTokenIds,
        finishReason: "max-tokens",
        seedToken: result.seedToken,
        compatibility,
      };
      workerState.lastMetrics = metrics;
      return metrics;
    }
    case "cancel": {
      const session = workerState.sessions.get(params.sessionId);
      if (!session) {
        return { cancelled: false };
      }
      session.cancelled = true;
      return { cancelled: true };
    }
    case "metrics":
      return workerState.lastMetrics ?? {
        runtimeProvider: NATIVE_RUNTIME_PROVIDER,
        nativeBackend: "cpu",
        nativeModelStatus: workerState.loadedModel ? "loaded" : "idle",
      };
    case "shutdown":
      setTimeout(() => process.exit(0), 10);
      return { shuttingDown: true };
    default:
      throw new Error(`Unsupported native worker method '${method}'.`);
  }
}

const rl = readline.createInterface({
  input: process.stdin,
  crlfDelay: Infinity,
});

rl.on("line", async (line) => {
  if (!line.trim()) {
    return;
  }
  let request;
  try {
    request = JSON.parse(line);
  } catch (error) {
    fail(null, error);
    return;
  }
  try {
    const result = await handle(request.method, request.params);
    ok(request.id, result);
  } catch (error) {
    fail(request.id, error);
  }
});
