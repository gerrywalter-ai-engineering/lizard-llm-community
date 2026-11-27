#!/usr/bin/env node

import http from "node:http";
import crypto from "node:crypto";
import fs from "node:fs";
import process from "node:process";
import { assertProPlanFeatureAllowed } from "../../lib/edition.js";
import { loadConfig } from "../../lib/config.js";
import { NativeCpuRuntime } from "../core/native-cpu-runtime.js";
import { buildNativeExecutionPlan } from "../native-execution-plan.js";
import { resolveNativeConcurrencyPolicy } from "../concurrency-policy.js";
import {
  appendNativeTurnMetric,
  contextBucket,
  listNativeTurnMetrics,
  summarizeNativeTurnMetrics,
} from "../../lib/native-turn-history.js";
import { findNativeHttpPerformance } from "../../lib/native-http-performance.js";
import {
  compactEvidenceMetricsLine,
  extractEvidenceMetrics,
} from "../../lib/evidence-metrics.js";
import {
  buildPromptArtifact,
  detokenizeTokenIds,
  tokenizeRenderedPrompt,
} from "../tokenizer/index.js";
import { hydrateLaunchGguf } from "./launch-gguf.js";
import { buildProtocolCapabilities } from "./protocol/capabilities.js";
import { createRequestCancellation } from "./protocol/cancellation.js";
import {
  classifyProtocolHttpError,
  ProtocolHttpError,
  protocolErrorPayload,
} from "./protocol/errors.js";
import {
  attachHttpRequestId,
  DEFAULT_JSON_BODY_TIMEOUT_MS,
  DEFAULT_MAX_JSON_BODY_BYTES,
  readJsonBody,
} from "./protocol/http.js";
import { writeWithBackpressure } from "./protocol/stream.js";
import {
  applyCaterpillarPerformanceProfile,
  caterpillarPerformanceProfileFromOptions,
  isCaterpillarTtftProfile,
} from "../caterpillar-performance-profile.js";
import { validateChatRequest } from "./protocol/validation.js";
import { normalizeChatMessages } from "./protocol/messages.js";
import {
  assertConformantChatTemplate,
  buildOpenAiChatCompletion,
  writeOpenAiChatSse,
} from "./protocol/openai.js";
import {
  buildLlamaCompletion,
  buildOpenAiTextCompletion,
  validateDetokenizeRequest,
  validateLlamaCompletionRequest,
  validateTokenizeRequest,
  writeLlamaCompletionSse,
  writeOpenAiTextCompletionSse,
} from "./protocol/llama.js";
import {
  buildOllamaResponse,
  validateOllamaChatRequest,
  validateOllamaGenerateRequest,
  writeOllamaNdjson,
} from "./protocol/ollama.js";
import {
  buildResponsesResponse,
  validateResponsesRequest,
  writeResponsesSse,
} from "./protocol/responses.js";
import {
  incidentUserFields,
  reportMajorFailure,
  retryPendingMajorFailureIncidents,
} from "../../lib/major-failure-intelligence.js";

let failureIntelligenceConfig = null;

function parseArgs(argv) {
  const options = {};
  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];
    if (!token.startsWith("--")) {
      continue;
    }
    const key = token.slice(2);
    const next = argv[index + 1];
    if (!next || next.startsWith("--")) {
      options[key] = true;
      continue;
    }
    options[key] = next;
    index += 1;
  }
  return options;
}

function readLaunchPayload(filePath) {
  try {
    return JSON.parse(fs.readFileSync(filePath, "utf8"));
  } finally {
    try {
      fs.unlinkSync(filePath);
    } catch {
      // A stale credential-free launch file is safe; startup continues with
      // the original parse/read outcome.
    }
  }
}

// Serialize BEFORE committing the status line. Writing headers first means a
// payload that cannot be stringified (a cycle, a BigInt) leaves a half-sent
// response, and the catch below then calls this again -- the second writeHead
// throws ERR_HTTP_HEADERS_SENT outside any handler and takes the whole server
// down with it. A serialization bug should cost one request, not the process.
function sendJson(res, statusCode, payload) {
  let body;
  try {
    body = JSON.stringify(payload);
  } catch (error) {
    body = JSON.stringify({
      error: {
        message: `Response could not be serialized: ${error?.message ?? String(error)}`,
        type: "server_error",
        code: "unserializable_response",
      },
    });
    statusCode = 500;
  }
  if (res.headersSent) {
    res.end();
    return;
  }
  res.writeHead(statusCode, {
    "Content-Type": "application/json; charset=utf-8",
    "Cache-Control": "no-store",
  });
  res.end(body);
}

function formatOllamaParameterSize(value) {
  const count = Number(value);
  if (!Number.isFinite(count) || count <= 0) return "unknown";
  if (count >= 1e9) return `${(count / 1e9).toFixed(1)}B`;
  if (count >= 1e6) return `${(count / 1e6).toFixed(1)}M`;
  if (count >= 1e3) return `${(count / 1e3).toFixed(1)}K`;
  return String(Math.trunc(count));
}

async function sha256File(filePath) {
  return new Promise((resolve, reject) => {
    const hash = crypto.createHash("sha256");
    const input = fs.createReadStream(filePath);
    input.on("error", reject);
    input.on("data", (chunk) => hash.update(chunk));
    input.on("end", () => resolve(hash.digest("hex")));
  });
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  if (!options.launch) {
    throw new Error("--launch <json> is required.");
  }
  const launch = readLaunchPayload(options.launch);
  const {
    modelArtifact: launchModelArtifact,
    profile: launchProfile,
    recommendation,
    config,
    host = "127.0.0.1",
    port = 8080,
    launchId = null,
    runtimeProvider = "lizard-native",
    // Internal benchmark/warm servers are part of the (feed-activated)
    // Benchmark feature and skip the user-facing "Native API server" Pro gate;
    // the `serve`/`run` CLI paths never set this, so that product feature
    // stays gated.
    internalBenchmarkServer = false,
  } = launch;
  const incidentConfig = {
    ...config,
    failureIntelligence: loadConfig().failureIntelligence,
  };
  failureIntelligenceConfig = incidentConfig;
  if (!["lizard-native", "caterpillar", "ostrich"].includes(runtimeProvider)) {
    throw new Error(`Unsupported standalone runtime provider: ${runtimeProvider}`);
  }
  // An API server that answers one request at a time is not an API server. The
  // ceiling matches the KV planner's slot cap (MAX_PLANNED_SLOTS), so the
  // scheduler cannot admit more sequences than the cache can hold; past that the
  // slot manager would be evicting live conversations to make room.
  const requestedMaxConcurrentRequests = Number.isFinite(launch?.maxConcurrentRequests)
    && launch.maxConcurrentRequests > 0
    ? Math.trunc(launch.maxConcurrentRequests)
    : null;
  // Label from the provider so logs name the engine actually running. A
  // hardcoded ternary reported "Caterpillar server ready" for an Ostrich
  // launch, which is exactly the kind of thing that makes a wrong-engine bug
  // invisible.
  const PROVIDER_LABELS = { caterpillar: "Caterpillar", ostrich: "Ostrich" };
  const providerLabel = PROVIDER_LABELS[runtimeProvider] ?? "lizard-native";
  const loopbackHosts = new Set(["127.0.0.1", "::1", "localhost"]);
  const remoteBinding = !loopbackHosts.has(String(host).toLowerCase());
  const bearerToken = launch?.bearerToken
    ?? config?.server?.bearerToken
    ?? process.env.LIZARD_API_BEARER_TOKEN
    ?? null;
  if (remoteBinding) {
    console.warn(`${providerLabel} warning: HTTP server is explicitly bound to non-loopback host '${host}'.`);
  }
  const requiresProEntitlement = !internalBenchmarkServer;
  // Re-read the GGUF when the vocabulary is ABSENT, not when a flag says it
  // might be.
  //
  // Vocabularies are large -- gemma-4 carries 262,144 tokens and 514,906 merges
  // -- so they are stripped before the metadata is passed around (see the note
  // in src/ui/server.js about vocabularies the onboarding UI never reads). The
  // re-read used to be gated on `metadataCompacted`, so any path that dropped
  // the vocabulary without setting that flag handed the runtime a GGUF with no
  // tokens in it. Prefill then refused every request with
  // NATIVE_TOKENIZER_UNAVAILABLE (tokenizer=gemma4, strategy=estimate) on a file
  // whose vocabulary was complete on disk and whose tokenizer resolves fine when
  // it is actually given it.
  //
  // Testing for the thing we need cannot go stale the way a flag can. This runs
  // once at launch; inspectGgufFile reads up to 128 MB of header and must never
  // move onto the request path.
  const fullGguf = hydrateLaunchGguf({
    modelArtifact: launchModelArtifact,
    profile: launchProfile,
  });
  const modelArtifact = {
    ...(launchModelArtifact ?? {}),
    ...(fullGguf ? { gguf: fullGguf } : {}),
  };
  const serverCreated = Math.floor(Date.now() / 1000);
  const modelAliases = [...new Set([
    ...(Array.isArray(modelArtifact.aliases) ? modelArtifact.aliases : []),
    ...(Array.isArray(modelArtifact.modelAliases) ? modelArtifact.modelAliases : []),
    ...(Array.isArray(recommendation?.selected?.aliases) ? recommendation.selected.aliases : []),
  ].filter((value) => typeof value === "string" && value.length > 0))];
  modelArtifact.modelAliases = modelAliases;
  const declaredDigest = String(modelArtifact.sha256 ?? modelArtifact.gguf?.sha256 ?? "")
    .replace(/^sha256:/i, "").toLowerCase();
  let ollamaDigest = /^[0-9a-f]{64}$/.test(declaredDigest) ? declaredDigest : null;
  let ollamaDigestPending = null;
  const resolveOllamaDigest = async () => {
    if (ollamaDigest) return ollamaDigest;
    ollamaDigestPending ??= (modelArtifact.modelPath && fs.existsSync(modelArtifact.modelPath)
      ? sha256File(modelArtifact.modelPath)
      : Promise.resolve(crypto.createHash("sha256").update(String(modelArtifact.modelId)).digest("hex")))
      .then((digest) => {
        ollamaDigest = digest;
        return digest;
      });
    return ollamaDigestPending;
  };
  let ollamaModifiedAt = new Date(serverCreated * 1000).toISOString();
  if (modelArtifact.modelPath && fs.existsSync(modelArtifact.modelPath)) {
    const fileModified = fs.statSync(modelArtifact.modelPath).mtime;
    if (Number.isFinite(fileModified.getTime())) ollamaModifiedAt = fileModified.toISOString();
  }
  const ollamaSize = Math.max(0, Math.trunc(Number(
    modelArtifact.gguf?.fileSizeBytes ?? modelArtifact.fileSizeBytes ?? 0,
  ) || 0));
  const ollamaFamily = String(modelArtifact.gguf?.architecture ?? "unknown");
  const ollamaDetails = {
    parent_model: "",
    format: String(modelArtifact.modelPath ?? "").toLowerCase().endsWith(".gguf") ? "gguf" : "prototype",
    family: ollamaFamily,
    families: [ollamaFamily],
    parameter_size: formatOllamaParameterSize(modelArtifact.gguf?.parameterCount),
    quantization_level: String(modelArtifact.gguf?.quant ?? modelArtifact.quant ?? "unknown"),
  };
  const caterpillarCpuGraph = runtimeProvider === "caterpillar"
    && String(modelArtifact?.modelPath ?? "").toLowerCase().endsWith(".gguf");
  const caterpillarExecutionProvider = caterpillarCpuGraph
    ? "caterpillar-cpu-graph-fit"
    : "lizard-native-reference";
  // The compiled Caterpillar worker advertises engine Phase 7 (including
  // resident KV, quality quants, and exact speculative verification). The
  // standalone facade used to publish Phase 2 for every real GGUF even after
  // the Phase 7 runner shipped, making a genuine Caterpillar HTTP response
  // look like the old CPU-fit prototype. Keep Phase 0 only for non-GGUF
  // compatibility/prototype launches.
  const caterpillarPhase = caterpillarCpuGraph ? 7 : 0;
  // Pre-warm the tokenizer definition cache at launch, before the HTTP
  // server accepts requests: the first request otherwise pays the
  // vocabulary build (~1.2-1.6 s measured for gemma's 262k vocab) at
  // onGenerate entry, which can race ahead of any post-model-load warm-up.
  if (modelArtifact?.modelPath) {
    setImmediate(() => {
      try {
        buildPromptArtifact({
          prompt: "warmup",
          modelPath: modelArtifact.modelPath,
          gguf: modelArtifact.gguf ?? null,
        });
      } catch {
        // Best-effort; the request path builds it on demand.
      }
    });
  }
  const profile = {
    ...(launchProfile ?? {}),
    ...(fullGguf ? { gguf: fullGguf } : {}),
  };
  if (requiresProEntitlement) {
    assertProPlanFeatureAllowed(config, "Native API server");
  }
  let effectiveProfile = {
    ...(profile ?? {}),
    chatMode: true,
    nativeRuntime: {
      // Batched decode is on here and nowhere else.
      //
      // EnhancementRun_04 measured it at 1.531x llama-server `-np 16` on
      // aggregate throughput, then left it opt-in and off -- so the campaign
      // closed with the honest note that "no user gets the 1.531x". This is the
      // one path where the premise it needs actually holds: several clients
      // against one loaded model, so there are real requests in flight to batch.
      //
      // Chat uses the same capability policy but only joins the coordinator
      // when turns overlap. Serve always opens the coordinator because several
      // clients are the expected workload.
      batchedDecode: true,
      // Keep isolated, deep-copy prefill snapshots for exact repeated prompts.
      // Burst API clients (and retries) then traverse the model once before
      // entering the shared B=16 decode lane, instead of serially prefilling
      // the same token sequence for every socket.
      prefillSnapshotCache: true,
      // Serve admits a concurrent HTTP cohort directly to the coordinator.
      // A genuinely lone request remains on the faster single-sequence path;
      // B=1 is a fallback lane, never a reason to pay one RPC per token.
      // Chat leaves this off and routes adaptively from overlapping generations.
      alwaysBatch: true,
      ...(profile?.nativeRuntime ?? {}),
      keepWorkerLoaded: true,
    },
  };
  effectiveProfile = applyCaterpillarPerformanceProfile(effectiveProfile, {
    runtimeProvider,
    performanceProfile: caterpillarPerformanceProfileFromOptions({
      caterpillarProfile: effectiveProfile?.caterpillarPerformanceProfile,
    }),
  });
  const measuredHttpPerformance = findNativeHttpPerformance(config, {
    runtimeProvider,
    modelId: modelArtifact.modelId,
    quant: effectiveProfile?.gguf?.quantization ?? effectiveProfile?.gguf?.quant ?? null,
    contextTokens: effectiveProfile?.contextSize ?? null,
    optimizationProfile: effectiveProfile?.caterpillarPerformanceProfile ?? null,
  });
  const concurrencyPolicy = resolveNativeConcurrencyPolicy({
    runtimeProvider,
    profile: effectiveProfile,
    gguf: effectiveProfile?.gguf ?? modelArtifact?.gguf ?? null,
    requestedBatchSize: requestedMaxConcurrentRequests,
    measuredBatchLevels: measuredHttpPerformance?.levels ?? null,
  });
  effectiveProfile = {
    ...effectiveProfile,
    nativeRuntime: {
      ...(effectiveProfile.nativeRuntime ?? {}),
      preferredBatchSize: concurrencyPolicy.preferredBatchSize,
      batchTargetSize: concurrencyPolicy.selectedBatchSize,
      batchFallbackOrder: concurrencyPolicy.fallbackOrder,
      concurrencyPolicy,
    },
  };
  const ttftPreference = runtimeProvider === "caterpillar"
    && isCaterpillarTtftProfile(effectiveProfile);
  const effectiveConcurrencyPolicy = ttftPreference
    ? {
      ...concurrencyPolicy,
      mode: "latency-preferred",
      preferredBatchSize: 1,
      selectedBatchSize: 1,
      role: "primary",
      source: "cpil_ttft",
      fallbackOrder: [1],
      reason: "cpil_ttft selects Caterpillar's B=1 user-latency lane.",
    }
    : concurrencyPolicy;
  effectiveProfile = {
    ...effectiveProfile,
    nativeRuntime: {
      ...(effectiveProfile.nativeRuntime ?? {}),
      concurrencyPolicy: effectiveConcurrencyPolicy,
    },
  };
  const maxConcurrentRequests = effectiveConcurrencyPolicy.selectedBatchSize;
  const runtime = new NativeCpuRuntime({ engine: runtimeProvider, maxConcurrentRequests });
  const modelQuant = effectiveProfile?.gguf?.quantization ?? effectiveProfile?.gguf?.quant ?? null;
  const recordCompletionMetric = ({
    metadata,
    body,
    req,
    surface,
    requestStartedAt,
    firstContentAt,
    completedAt = Date.now(),
  }) => {
    try {
      const nativeExecution = metadata?.nativeExecution ?? null;
      const evidenceMetrics = extractEvidenceMetrics(metadata);
      const userId = body?.user ?? req?.headers?.["x-lizard-user-id"] ?? "local-http";
      const generatedTokens = nativeExecution?.generatedTokenIds?.length ?? 0;
      const endToEndMs = Math.max(1, completedAt - requestStartedAt);
      const ttftMs = Math.max(1, (firstContentAt ?? completedAt) - requestStartedAt);
      const userVisibleTokensPerSecond = generatedTokens > 0
        ? (generatedTokens / endToEndMs) * 1000
        : null;
      const current = appendNativeTurnMetric(config, {
        userId,
        surface,
        modelId: modelArtifact.modelId,
        quant: modelQuant,
        runtimeProvider,
        backend: metadata?.nativeBackend ?? nativeExecution?.backend,
        servingPath: nativeExecution?.servingPath,
        batched: nativeExecution?.batched,
        concurrencyLevel: nativeExecution?.concurrencyLevel,
        contextTokens: effectiveProfile?.contextSize,
        promptTokens: metadata?.promptArtifact?.tokenCount,
        generatedTokens,
        prefillMs: metadata?.prefillMs,
        decodeMs: metadata?.decodeMs,
        decodeTokensPerSecond: metadata?.tokensPerSecond,
        userVisibleTokensPerSecond,
        endToEndMs,
        ttftMs,
        latencyPerTokenMs: generatedTokens > 0 ? endToEndMs / generatedTokens : null,
        transport: "http",
        optimizationProfile: effectiveProfile.caterpillarPerformanceProfile ?? null,
        evidenceMetrics,
      });
      const entries = listNativeTurnMetrics(config, {
        userId,
        modelId: modelArtifact.modelId,
        quant: modelQuant ?? "unknown",
        runtimeProvider,
        context: contextBucket(effectiveProfile?.contextSize),
        limit: 100,
      });
      const compactMetrics = compactEvidenceMetricsLine(evidenceMetrics);
      if (compactMetrics) {
        console.log(`[metrics] provider=${runtimeProvider} model=${modelArtifact.modelId} ${compactMetrics}`);
      }
      return {
        current,
        summary: summarizeNativeTurnMetrics(entries),
      };
    } catch (error) {
      console.error(`native turn metric append failed: ${error?.message ?? error}`);
      return null;
    }
  };
  const executionPlan = buildNativeExecutionPlan({
    capabilityRegistry: effectiveProfile?.capabilityRegistry ?? null,
    profile: effectiveProfile,
    gguf: effectiveProfile?.gguf ?? modelArtifact?.gguf ?? null,
    requestedBackend: effectiveProfile?.backend ?? "cpu",
    runtimeProvider,
  });
  // Load the model (and pre-stage GPU weights) once at startup so the first
  // chat request runs at warm speed. Failures surface on the first request.
  let warmLoadResult = null;
  let warmLoadState = "loading";
  let warmLoadError = null;
  let warmLoadIncident = null;
  const reportCriticalFailure = async (error, context = {}) => {
    try {
      return await reportMajorFailure({
        error,
        config: incidentConfig,
        context: {
          component: "native-http-server",
          runtimeProvider,
          backend: executionPlan?.backend ?? effectiveProfile?.backend ?? null,
          architecture: effectiveProfile?.gguf?.architecture ?? modelArtifact?.gguf?.architecture ?? null,
          quantization: effectiveProfile?.gguf?.quant ?? modelArtifact?.gguf?.quant ?? null,
          modelId: modelArtifact?.modelId ?? null,
          ...context,
        },
        hardware: effectiveProfile?.hardware ?? null,
        user: incidentConfig?.failureIntelligence?.userEmail
          ? { email: incidentConfig.failureIntelligence.userEmail }
          : null,
        runtimeState: {
          warmLoadState,
          warmLoadError,
          executionPlan,
        },
      });
    } catch (reportError) {
      console.error(`major-failure reporting failed safely: ${reportError?.message ?? reportError}`);
      return { reported: false, reason: "reporting_failed" };
    }
  };
  const beginWarmLoad = () => runtime.ensureModelLoaded({
      recommendation,
      modelArtifact,
      profile: effectiveProfile,
      config,
    }).then((worker) => {
      warmLoadResult = worker.loadResult ?? null;
      warmLoadState = "ready";
      console.log(`${providerLabel} model ready: ${modelArtifact.modelId}`
        + ` (loadMs=${warmLoadResult?.loadMs ?? "?"},`
        + ` gpuResidentWeightBytes=${warmLoadResult?.gpuResidentWeightBytes ?? 0},`
        + ` gpuWeightBudgetBytes=${warmLoadResult?.gpuWeightBudgetBytes ?? 0})`);
      return worker;
    }).catch(async (error) => {
      warmLoadState = "failed";
      warmLoadError = error.message;
      console.error(`${providerLabel} model warm load failed: ${error.message}`);
      warmLoadIncident = await reportCriticalFailure(error, {
        operation: "warm-model-load",
        unrecoverable: true,
        reproduction: {
          operation: "warm-model-load",
          runtimeProvider,
          modelId: modelArtifact?.modelId ?? null,
          architecture: effectiveProfile?.gguf?.architecture ?? null,
          quantization: effectiveProfile?.gguf?.quant ?? null,
        },
      });
      if (warmLoadIncident?.reported) {
        console.error(`Ticket ${warmLoadIncident.ticketId}: ${warmLoadIncident.message}`);
      }
      return null;
    });

  let activeHttpCompletions = 0;
  let draining = false;
  const activeCancellations = new Set();
  const maxJsonBodyBytes = Number.isFinite(Number(config?.server?.maxJsonBodyBytes))
    ? Math.max(1, Math.trunc(Number(config.server.maxJsonBodyBytes)))
    : DEFAULT_MAX_JSON_BODY_BYTES;
  const jsonBodyTimeoutMs = Number.isFinite(Number(config?.server?.jsonBodyTimeoutMs))
    ? Math.max(1, Math.trunc(Number(config.server.jsonBodyTimeoutMs)))
    : DEFAULT_JSON_BODY_TIMEOUT_MS;
  const readRequestBody = (req) => readJsonBody(req, {
    maxBytes: maxJsonBodyBytes,
    timeoutMs: jsonBodyTimeoutMs,
  });
  const server = http.createServer(async (req, res) => {
    // Disable Nagle's algorithm on every connection. The token stream writes a
    // small SSE chunk per decoded token; with Nagle on (Node's default, unlike
    // Go's net/http) each write can stall ~40 ms waiting on the client's
    // delayed ACK, adding tens of ms per token to end-to-end serving latency.
    req.socket.setNoDelay(true);
    const httpRequestId = attachHttpRequestId(req, res);
    let responseProtocol = null;
    try {
      if (bearerToken && !["/health", "/v1/health"].includes(req.url)) {
        const supplied = String(req.headers.authorization ?? "");
        const expected = `Bearer ${bearerToken}`;
        const valid = Buffer.byteLength(supplied) === Buffer.byteLength(expected)
          && crypto.timingSafeEqual(Buffer.from(supplied), Buffer.from(expected));
        if (!valid) {
          throw new ProtocolHttpError(401, "invalid_api_key", "Bearer authentication is required.", {
            type: "authentication_error",
          });
        }
      }
      if (req.method === "GET" && req.url === "/health") {
        sendJson(res, 200, {
          status: "ok",
          processId: process.pid,
          launchId,
          // In-flight completions right now. Unloading a model kills this
          // server, and with it every response still streaming - so anything
          // that unloads has to be able to see whether it is about to cut
          // someone off, and say so before it does. The counter was already
          // maintained for admission control; it just was not published.
          activeRequests: activeHttpCompletions,
          // Every field below that reads from warmLoadResult is null/0/"cpu"
          // until the asynchronous warm load finishes, and the port answers
          // long before that. A caller that samples /health on the first
          // successful response therefore records a CPU-only, zero-resident
          // server that then serves every request from the GPU - which is
          // exactly how a benchmark artifact came to claim
          // gpuResidentWeightBytes: 0 for a fully resident model. Publishing
          // the load state lets callers wait for the configuration they mean
          // to record instead of guessing from a 200.
          warmLoadState,
          modelReady: warmLoadState === "ready" && !draining,
          draining,
          activeRequests: activeHttpCompletions,
          warmLoadError: warmLoadError?.message ?? warmLoadError ?? null,
          warmLoadIncident: incidentUserFields(warmLoadIncident),
          runtimeProvider,
          executionProvider: runtimeProvider === "caterpillar" ? caterpillarExecutionProvider : "lizard-native",
          caterpillarPhase: ["caterpillar", "ostrich"].includes(runtimeProvider) ? caterpillarPhase : null,
          nativeBackend: (warmLoadResult?.hybridBackends?.length ?? 0) > 0 && executionPlan.mode !== "cpu-only"
            ? warmLoadResult.hybridBackends[0]
            : "cpu",
          executionPlanMode: executionPlan.mode,
          hybridBackends: warmLoadResult?.hybridBackends ?? [],
          gpuResidentWeightBytes: warmLoadResult?.gpuResidentWeightBytes ?? 0,
          gpuWeightBudgetBytes: warmLoadResult?.gpuWeightBudgetBytes ?? 0,
          // Which CPU kernel actually ran. A declined integer-dot sidecar
          // silently drops decode onto the float path and costs ~6x, and that
          // was invisible in benchmark output - a fallback result once printed
          // as a clean provenance-VERIFIED 0.124x table that looked like an
          // engine regression. Publishing the kernel makes the fallback
          // impossible to mistake for a measurement.
          cpuKernel: warmLoadResult?.cpuKernel ?? null,
          integerDotActive: typeof warmLoadResult?.cpuKernel === "string"
            ? warmLoadResult.cpuKernel.includes("integer")
            : null,
          fitDecision: warmLoadResult?.fitDecision ?? null,
          fitReason: warmLoadResult?.fitReason ?? null,
          fitStable: warmLoadResult?.fitStable ?? null,
          fitBudgetLive: warmLoadResult?.fitBudgetLive ?? null,
          fitBudgetSource: warmLoadResult?.fitBudgetSource ?? null,
          fitAdapterName: warmLoadResult?.fitAdapterName ?? null,
          residentLayerCount: warmLoadResult?.residentLayerCount ?? null,
          totalLayerCount: warmLoadResult?.totalLayerCount ?? null,
          budgetBytes: warmLoadResult?.budgetBytes ?? null,
          fitContextTokens: warmLoadResult?.fitContextTokens ?? null,
          fitKvCacheBytes: warmLoadResult?.fitKvCacheBytes ?? null,
          fitWorkingSetBytes: warmLoadResult?.fitWorkingSetBytes ?? null,
          gpuExecution: warmLoadResult?.gpuExecution ?? null,
          gpuExecutionReason: warmLoadResult?.gpuExecutionReason ?? null,
          gpuAdapterName: warmLoadResult?.gpuAdapterName ?? null,
          planGpuSpans: warmLoadResult?.planGpuSpans ?? null,
          planCpuSpans: warmLoadResult?.planCpuSpans ?? null,
          gpuKvCapacityTokens: warmLoadResult?.gpuKvCapacityTokens ?? null,
          gpuResidentWeightBytesActual: warmLoadResult?.gpuResidentWeightBytesActual ?? null,
          gpuIntegerDotAvailable: warmLoadResult?.gpuIntegerDotAvailable ?? null,
          gpuIntegerDotActive: warmLoadResult?.gpuIntegerDotActive ?? null,
          gpuIntegerDotReason: warmLoadResult?.gpuIntegerDotReason ?? null,
          gpuQ8_1QuantizationCount: warmLoadResult?.gpuQ8_1QuantizationCount ?? null,
          gpuIntegerDotMatvecCount: warmLoadResult?.gpuIntegerDotMatvecCount ?? null,
          fitReserveBytes: warmLoadResult?.fitReserveBytes ?? null,
          totalStagedWeightBytes: warmLoadResult?.totalStagedWeightBytes ?? null,
          gpuWeightCacheEvictions: warmLoadResult?.gpuWeightCacheEvictions ?? null,
          q4_0Fp16ScaleStaging: warmLoadResult?.q4_0Fp16ScaleStaging ?? null,
          loadMs: warmLoadResult?.loadMs ?? null,
          gpuPrestageMs: warmLoadResult?.gpuPrestageMs ?? null,
          modelLoaded: warmLoadResult != null,
          modelLoadState: warmLoadState,
          modelLoadError: warmLoadError,
          fallbackAllowed: false,
          modelId: modelArtifact.modelId,
          // The weights file this server actually opened. Published so another
          // process can decide whether this server is the one it needs rather
          // than starting a second and losing to the residency guard - the
          // benchmark reuses a running serve only when this path matches the
          // GGUF it was asked to measure, since a modelId alone does not prove
          // two servers hold the same bytes.
          modelPath: modelArtifact.modelPath ?? null,
          modelQuant: modelArtifact.quant ?? modelArtifact.gguf?.quant ?? null,
          chatCompletionsUrl: `http://${host}:${port}/v1/chat/completions`,
          requestTimeoutMode: effectiveProfile.timeoutMs == null ? "unlimited" : "bounded",
          requestTimeoutMs: effectiveProfile.timeoutMs ?? null,
          concurrencyPolicy: effectiveConcurrencyPolicy,
          caterpillarPerformanceProfile: runtimeProvider === "caterpillar"
            ? effectiveProfile.caterpillarPerformanceProfile
            : null,
          measuredHttpPerformance,
          maxConcurrentRequests,
        });
        return;
      }

      if (req.method === "GET" && req.url === "/v1/models") {
        sendJson(res, 200, {
          object: "list",
          data: [
            {
              id: modelArtifact.modelId,
              object: "model",
              created: serverCreated,
              owned_by: runtimeProvider,
              aliases: modelAliases,
              meta: {
                n_ctx: effectiveProfile?.contextSize ?? modelArtifact.gguf?.contextLength ?? null,
                n_vocab: modelArtifact.gguf?.tokenVocabularySize ?? null,
                n_params: modelArtifact.gguf?.parameterCount ?? null,
                size: modelArtifact.gguf?.fileSizeBytes ?? modelArtifact.fileSizeBytes ?? null,
              },
            },
          ],
        });
        return;
      }

      if (req.method === "GET" && req.url === "/api/version") {
        sendJson(res, 200, { version: `${process.env.npm_package_version ?? "1.2.60"}-caterpillar` });
        return;
      }

      if (req.method === "GET" && req.url === "/api/tags") {
        const digest = await resolveOllamaDigest();
        sendJson(res, 200, {
          models: [{
            name: modelArtifact.modelId,
            model: modelArtifact.modelId,
            modified_at: ollamaModifiedAt,
            size: ollamaSize,
            digest,
            details: ollamaDetails,
            capabilities: ["completion"],
          }],
        });
        return;
      }

      if (req.method === "GET" && req.url === "/api/ps") {
        const digest = await resolveOllamaDigest();
        sendJson(res, 200, {
          models: warmLoadState === "ready" && !draining ? [{
            name: modelArtifact.modelId,
            model: modelArtifact.modelId,
            size: ollamaSize,
            digest,
            details: ollamaDetails,
            // A fixed Caterpillar server keeps the model until process exit;
            // this sentinel expresses process-lifetime residency without
            // promising request-controlled keep-alive support.
            expires_at: "9999-12-31T23:59:59.999Z",
            size_vram: Math.max(0, Math.trunc(Number(
              warmLoadResult?.gpuResidentWeightBytesActual ?? warmLoadResult?.gpuResidentWeightBytes ?? 0,
            ) || 0)),
            context_length: Math.max(0, Math.trunc(Number(
              effectiveProfile?.contextSize ?? modelArtifact.gguf?.contextLength ?? 0,
            ) || 0)),
          }] : [],
        });
        return;
      }

      if (req.method === "GET" && req.url === "/v1/health") {
        if (warmLoadState === "ready" && !draining) {
          sendJson(res, 200, { status: "ok" });
        } else {
          sendJson(res, 503, {
            status: "unavailable",
            ...protocolErrorPayload(new ProtocolHttpError(
              503,
              draining ? "server_draining" : "model_not_ready",
              draining ? "Server is draining." : `Model is ${warmLoadState}.`,
              { type: "service_unavailable" },
            )),
          });
        }
        return;
      }

      if (req.method === "GET" && req.url === "/capabilities") {
        sendJson(res, 200, buildProtocolCapabilities({
          runtimeProvider,
          modelArtifact,
          modelReady: warmLoadState === "ready",
          productVersion: process.env.npm_package_version ?? null,
          buildCommit: process.env.LIZARD_BUILD_COMMIT ?? null,
          constraints: { maxJsonBodyBytes, jsonBodyTimeoutMs },
        }));
        return;
      }

      if (req.method === "GET" && req.url === "/metrics") {
        const worker = await runtime.ensureModelLoaded({
          recommendation,
          modelArtifact,
          profile: effectiveProfile,
          config,
        });
        const metrics = await worker.client.metrics();
        sendJson(res, 200, {
          ...metrics,
          product: {
            transport: "http",
            runtimeProvider,
            modelId: modelArtifact.modelId,
            concurrencyPolicy: effectiveConcurrencyPolicy,
            caterpillarPerformanceProfile: runtimeProvider === "caterpillar"
              ? effectiveProfile.caterpillarPerformanceProfile
              : null,
            measuredHttpPerformance,
          },
        });
        return;
      }
      if (req.method === "GET" && req.url?.startsWith("/metrics/history")) {
        const requestUrl = new URL(req.url, `http://${host}:${port}`);
        const userId = requestUrl.searchParams.get("user")
          ?? req.headers["x-lizard-user-id"]
          ?? "local-http";
        const entries = listNativeTurnMetrics(config, { userId, limit: 100 });
        sendJson(res, 200, { userId, summary: summarizeNativeTurnMetrics(entries), entries });
        return;
      }

      // llama-server-compatible prompt identity endpoints. They let the
      // benchmark prove that both runtimes receive the same rendered bytes and
      // exact token IDs instead of assuming that identically shaped chat JSON
      // implies an identical model input.
      if (req.method === "POST" && req.url === "/apply-template") {
        const body = await readRequestBody(req);
        const unknown = Object.keys(body).find((field) => !["messages", "add_generation_prompt"].includes(field));
        if (unknown) {
          throw new ProtocolHttpError(400, "invalid_parameter", `Unsupported parameter: ${unknown}.`, {
            type: "invalid_request_error", param: unknown,
          });
        }
        if (body.add_generation_prompt != null && body.add_generation_prompt !== true) {
          throw new ProtocolHttpError(400, "unsupported_parameter",
            "Caterpillar /apply-template supports only add_generation_prompt=true.", {
              type: "invalid_request_error", param: "add_generation_prompt",
            });
        }
        const messages = normalizeChatMessages(body.messages);
        const lastUser = [...messages].reverse().find((message) => message.role === "user");
        const artifact = assertConformantChatTemplate(buildPromptArtifact({
          prompt: lastUser.content,
          messages,
          modelPath: modelArtifact.modelPath,
          gguf: modelArtifact.gguf ?? null,
          chatMode: true,
        }));
        sendJson(res, 200, {
          prompt: artifact.renderedPrompt,
          prompt_tokens: artifact.tokenCount,
          template_mode: artifact.templateMode,
        });
        return;
      }

      if (req.method === "POST" && req.url === "/tokenize") {
        const body = validateTokenizeRequest(await readRequestBody(req));
        const tokenized = tokenizeRenderedPrompt({
          content: body.content,
          modelPath: modelArtifact.modelPath,
          gguf: modelArtifact.gguf ?? null,
          addSpecial: body.addSpecial,
          parseSpecial: body.parseSpecial,
          withPieces: body.withPieces,
        });
        if (!Array.isArray(tokenized.tokenIds)) {
          sendJson(res, 422, {
            error: {
              message: "Exact prompt token IDs are unavailable for this model.",
              type: "tokenizer_unavailable",
              code: "NATIVE_TOKENIZER_UNAVAILABLE",
            },
          });
          return;
        }
        sendJson(res, 200, {
          tokens: body.withPieces ? tokenized.tokenPieces : tokenized.tokenIds,
          token_id_mode: tokenized.tokenIdMode,
          tokenizer: tokenized.tokenizer,
        });
        return;
      }

      if (req.method === "POST" && req.url === "/detokenize") {
        const body = validateDetokenizeRequest(await readRequestBody(req));
        let detokenized;
        try {
          detokenized = detokenizeTokenIds({
            tokenIds: body.tokens,
            modelPath: modelArtifact.modelPath,
            gguf: modelArtifact.gguf ?? null,
          });
        } catch (error) {
          if (error?.code === "TOKEN_ID_INVALID") {
            throw new ProtocolHttpError(400, "invalid_parameter", error.message, {
              type: "invalid_request_error", param: "tokens",
            });
          }
          throw error;
        }
        sendJson(res, 200, { content: detokenized.text });
        return;
      }

      if (req.method === "POST" && ["/api/generate", "/api/chat"].includes(req.url)) {
        const requestStartedAt = Date.now();
        const requestStartedNs = process.hrtime.bigint();
        const ollamaChat = req.url === "/api/chat";
        try {
          if (requiresProEntitlement) assertProPlanFeatureAllowed(config, "Native API server");
        } catch (error) {
          throw new ProtocolHttpError(403, error.code ?? "pro_plan_required", error.message, {
            type: "pro_plan_required",
          });
        }
        const requestBody = await readRequestBody(req);
        const ollamaRequest = ollamaChat
          ? validateOllamaChatRequest(requestBody, { modelId: modelArtifact.modelId, modelAliases })
          : validateOllamaGenerateRequest(requestBody, { modelId: modelArtifact.modelId, modelAliases });
        if (draining || warmLoadState !== "ready") {
          throw new ProtocolHttpError(503, draining ? "server_draining" : "model_not_ready",
            draining ? "Server is draining." : `Model is ${warmLoadState}.`, {
              type: "service_unavailable",
            });
        }
        if (activeHttpCompletions >= maxConcurrentRequests) {
          throw new ProtocolHttpError(429, "server_busy",
            `Server already has its maximum of ${maxConcurrentRequests} active generation requests.`, {
              type: "rate_limit_error",
            });
        }
        activeHttpCompletions += 1;
        let admissionReleased = false;
        const releaseAdmission = () => {
          if (admissionReleased) return;
          admissionReleased = true;
          activeHttpCompletions = Math.max(0, activeHttpCompletions - 1);
        };
        const cancellation = createRequestCancellation({
          req,
          res,
          timeoutMs: effectiveProfile.timeoutMs,
        });
        activeCancellations.add(cancellation);
        try {
          const session = await runtime.createSession({
            recommendation, modelArtifact, profile: effectiveProfile, config, logger: null,
          });
          cancellation.bindSession(session);
          if (cancellation.signal.aborted) {
            throw new ProtocolHttpError(499, "client_cancelled", "Request was cancelled before generation started.");
          }
          const events = session.generateStream({
            requestId: httpRequestId,
            prompt: ollamaRequest.prompt,
            messages: ollamaRequest.messages,
            stopSequences: ollamaRequest.stop,
            maxTokens: ollamaRequest.maxTokens ?? config?.generation?.maxTokens ?? 64,
            temperature: 0,
            topK: 1,
            topP: 1,
            metadata: {
              rawPrompt: ollamaRequest.rawPrompt,
              cachePrompt: true,
              stream: ollamaRequest.stream,
              getAdmittedHttpConcurrency: () => activeHttpCompletions,
            },
          });
          if (ollamaRequest.stream) {
            responseProtocol = "ollama-ndjson";
            res.writeHead(200, {
              "Content-Type": "application/x-ndjson",
              "Cache-Control": "no-store, no-transform",
              Connection: "keep-alive",
              "X-Accel-Buffering": "no",
            });
            const write = (chunk) => writeWithBackpressure(res, chunk, cancellation.signal);
            let succeeded = false;
            try {
              await writeOllamaNdjson({ events, write, request: ollamaRequest, requestStartedNs });
              succeeded = true;
            } catch (error) {
              cancellation.cancel(cancellation.reason ?? "stream_failure");
              if (!res.destroyed) res.destroy();
              throw error;
            } finally {
              await session.close();
              if (succeeded && !res.destroyed && !res.writableEnded) res.end();
            }
            return;
          }
          let completed = null;
          try {
            for await (const event of events) {
              if (event.type === "complete") completed = event;
              else if (event.type === "cancelled") {
                throw new ProtocolHttpError(499, "client_cancelled", "Generation was cancelled.");
              } else if (event.type === "error") {
                throw new ProtocolHttpError(500, "generation_failed", event.message ?? "Generation failed.");
              }
            }
          } finally {
            await session.close();
          }
          if (!completed) {
            throw new ProtocolHttpError(500, "terminal_event_missing", "Generation ended without a terminal event.");
          }
          recordCompletionMetric({
            metadata: completed.metadata,
            body: requestBody,
            req,
            surface: ollamaChat ? "ollama-chat-json" : "ollama-generate-json",
            requestStartedAt,
            firstContentAt: Date.now(),
          });
          sendJson(res, 200, buildOllamaResponse({
            completed,
            request: ollamaRequest,
            requestStartedNs,
          }));
          return;
        } catch (error) {
          if (cancellation.reason === "deadline_exceeded") {
            throw new ProtocolHttpError(504, "generation_timeout", "Generation exceeded its configured deadline.", {
              type: "timeout_error",
            });
          }
          if (cancellation.reason === "client_disconnect") {
            throw new ProtocolHttpError(499, "client_cancelled", "Client disconnected during generation.");
          }
          throw error;
        } finally {
          activeCancellations.delete(cancellation);
          await cancellation.dispose();
          releaseAdmission();
        }
      }

      if (req.method === "POST" && req.url === "/v1/responses") {
        const requestStartedAt = Date.now();
        const responseId = `resp_${crypto.randomUUID().replaceAll("-", "")}`;
        const messageId = `msg_${crypto.randomUUID().replaceAll("-", "")}`;
        const createdAt = Math.floor(requestStartedAt / 1000);
        try {
          if (requiresProEntitlement) assertProPlanFeatureAllowed(config, "Native API server");
        } catch (error) {
          throw new ProtocolHttpError(403, error.code ?? "pro_plan_required", error.message, {
            type: "pro_plan_required",
          });
        }
        const requestBody = await readRequestBody(req);
        const responsesRequest = validateResponsesRequest(requestBody, {
          modelId: modelArtifact.modelId,
          modelAliases,
        });
        if (draining || warmLoadState !== "ready") {
          throw new ProtocolHttpError(503, draining ? "server_draining" : "model_not_ready",
            draining ? "Server is draining." : `Model is ${warmLoadState}.`, {
              type: "service_unavailable",
            });
        }
        if (activeHttpCompletions >= maxConcurrentRequests) {
          throw new ProtocolHttpError(429, "server_busy",
            `Server already has its maximum of ${maxConcurrentRequests} active generation requests.`, {
              type: "rate_limit_error",
            });
        }
        activeHttpCompletions += 1;
        let admissionReleased = false;
        const releaseAdmission = () => {
          if (admissionReleased) return;
          admissionReleased = true;
          activeHttpCompletions = Math.max(0, activeHttpCompletions - 1);
        };
        const cancellation = createRequestCancellation({ req, res, timeoutMs: effectiveProfile.timeoutMs });
        activeCancellations.add(cancellation);
        try {
          const session = await runtime.createSession({
            recommendation, modelArtifact, profile: effectiveProfile, config, logger: null,
          });
          cancellation.bindSession(session);
          if (cancellation.signal.aborted) {
            throw new ProtocolHttpError(499, "client_cancelled", "Request was cancelled before generation started.");
          }
          const events = session.generateStream({
            requestId: httpRequestId,
            prompt: responsesRequest.prompt,
            messages: responsesRequest.messages,
            stopSequences: [],
            maxTokens: responsesRequest.maxOutputTokens ?? config?.generation?.maxTokens ?? 64,
            temperature: 0,
            topK: 1,
            topP: 1,
            metadata: {
              stream: responsesRequest.stream,
              getAdmittedHttpConcurrency: () => activeHttpCompletions,
            },
          });
          if (responsesRequest.stream) {
            responseProtocol = "responses-sse";
            res.writeHead(200, {
              "Content-Type": "text/event-stream; charset=utf-8",
              "Cache-Control": "no-store, no-transform",
              Connection: "keep-alive",
              "X-Accel-Buffering": "no",
            });
            const write = (chunk) => writeWithBackpressure(res, chunk, cancellation.signal);
            let succeeded = false;
            try {
              await writeResponsesSse({
                events, write, id: responseId, messageId, createdAt, request: responsesRequest,
              });
              succeeded = true;
            } catch (error) {
              cancellation.cancel(cancellation.reason ?? "stream_failure");
              if (!res.destroyed) res.destroy();
              throw error;
            } finally {
              await session.close();
              if (succeeded && !res.destroyed && !res.writableEnded) res.end();
            }
            return;
          }
          let completed = null;
          try {
            for await (const event of events) {
              if (event.type === "complete") completed = event;
              else if (event.type === "cancelled") {
                throw new ProtocolHttpError(499, "client_cancelled", "Generation was cancelled.");
              } else if (event.type === "error") {
                throw new ProtocolHttpError(500, "generation_failed", event.message ?? "Generation failed.");
              }
            }
          } finally {
            await session.close();
          }
          if (!completed) {
            throw new ProtocolHttpError(500, "terminal_event_missing", "Generation ended without a terminal event.");
          }
          recordCompletionMetric({
            metadata: completed.metadata,
            body: requestBody,
            req,
            surface: "openai-responses-json",
            requestStartedAt,
            firstContentAt: Date.now(),
          });
          sendJson(res, 200, buildResponsesResponse({
            id: responseId, messageId, createdAt, request: responsesRequest, completed,
          }));
          return;
        } catch (error) {
          if (cancellation.reason === "deadline_exceeded") {
            throw new ProtocolHttpError(504, "generation_timeout", "Generation exceeded its configured deadline.", {
              type: "timeout_error",
            });
          }
          if (cancellation.reason === "client_disconnect") {
            throw new ProtocolHttpError(499, "client_cancelled", "Client disconnected during generation.");
          }
          throw error;
        } finally {
          activeCancellations.delete(cancellation);
          await cancellation.dispose();
          releaseAdmission();
        }
      }

      if (req.method === "POST" && ["/completion", "/v1/completions"].includes(req.url)) {
        const requestStartedAt = Date.now();
        const openAi = req.url === "/v1/completions";
        try {
          if (requiresProEntitlement) assertProPlanFeatureAllowed(config, "Native API server");
        } catch (error) {
          throw new ProtocolHttpError(403, error.code ?? "pro_plan_required", error.message, {
            type: "pro_plan_required",
          });
        }
        const completionRequest = validateLlamaCompletionRequest(
          await readRequestBody(req),
          { openAi, modelId: modelArtifact.modelId, modelAliases },
        );
        let prompt = completionRequest.prompt;
        if (completionRequest.promptTokenIds) {
          try {
            prompt = detokenizeTokenIds({
              tokenIds: completionRequest.promptTokenIds,
              modelPath: modelArtifact.modelPath,
              gguf: modelArtifact.gguf ?? null,
            }).text;
          } catch (error) {
            if (error?.code === "TOKEN_ID_INVALID") {
              throw new ProtocolHttpError(400, "invalid_parameter", error.message, {
                type: "invalid_request_error", param: "prompt",
              });
            }
            throw error;
          }
        }
        if (draining || warmLoadState !== "ready") {
          throw new ProtocolHttpError(503, draining ? "server_draining" : "model_not_ready",
            draining ? "Server is draining." : `Model is ${warmLoadState}.`, {
              type: "service_unavailable",
            });
        }
        if (activeHttpCompletions >= maxConcurrentRequests) {
          throw new ProtocolHttpError(429, "server_busy",
            `Server already has its maximum of ${maxConcurrentRequests} active generation requests.`, {
              type: "rate_limit_error",
            });
        }
        activeHttpCompletions += 1;
        let admissionReleased = false;
        const releaseAdmission = () => {
          if (admissionReleased) return;
          admissionReleased = true;
          activeHttpCompletions = Math.max(0, activeHttpCompletions - 1);
        };
        const cancellation = createRequestCancellation({
          req,
          res,
          timeoutMs: effectiveProfile.timeoutMs,
        });
        activeCancellations.add(cancellation);
        const id = `${openAi ? "cmpl" : "completion"}-${crypto.randomUUID().replaceAll("-", "")}`;
        const created = Math.floor(requestStartedAt / 1000);
        try {
          const session = await runtime.createSession({
            recommendation, modelArtifact, profile: effectiveProfile, config, logger: null,
          });
          cancellation.bindSession(session);
          const events = session.generateStream({
            requestId: httpRequestId,
            prompt,
            promptTokenIds: completionRequest.promptTokenIds,
            stopSequences: completionRequest.stop,
            maxTokens: completionRequest.maxTokens ?? config?.generation?.maxTokens ?? 64,
            temperature: 0,
            topK: 1,
            topP: 1,
            seed: completionRequest.seed,
            metadata: {
              rawPrompt: true,
              cachePrompt: completionRequest.cachePrompt,
              stream: completionRequest.stream,
              getAdmittedHttpConcurrency: () => activeHttpCompletions,
            },
          });
          if (completionRequest.stream) {
            responseProtocol = openAi ? "openai-completion-sse" : "llama-completion-sse";
            res.writeHead(200, {
              "Content-Type": "text/event-stream",
              "Cache-Control": "no-store, no-transform",
              Connection: "keep-alive",
              "X-Accel-Buffering": "no",
            });
            const write = (chunk) => writeWithBackpressure(res, chunk, cancellation.signal);
            let keepalivePending = false;
            const keepalive = setInterval(() => {
              if (keepalivePending) return;
              keepalivePending = true;
              void write(": keepalive\n\n").then((written) => {
                if (!written) cancellation.cancel("client_disconnect");
              }).catch(() => cancellation.cancel("client_disconnect"))
                .finally(() => { keepalivePending = false; });
            }, 10_000);
            let succeeded = false;
            try {
              if (openAi) {
                await writeOpenAiTextCompletionSse({
                  events, write, id, created, model: modelArtifact.modelId, request: completionRequest,
                });
              } else {
                await writeLlamaCompletionSse({
                  events, write, model: modelArtifact.modelId, request: completionRequest,
                });
              }
              succeeded = true;
            } catch (error) {
              cancellation.cancel(cancellation.reason ?? "stream_failure");
              if (!res.destroyed) res.destroy();
              throw error;
            } finally {
              clearInterval(keepalive);
              await session.close();
              if (succeeded && !res.destroyed && !res.writableEnded) res.end();
            }
            return;
          }
          let completed = null;
          try {
            for await (const event of events) {
              if (event.type === "complete") completed = event;
            }
          } finally {
            await session.close();
          }
          if (!completed) {
            throw new ProtocolHttpError(500, "terminal_event_missing", "Generation ended without a terminal event.", {
              type: "server_error",
            });
          }
          sendJson(res, 200, openAi
            ? buildOpenAiTextCompletion({
              id, created, model: modelArtifact.modelId, completed, request: completionRequest,
            })
            : buildLlamaCompletion({ completed, model: modelArtifact.modelId, request: completionRequest }));
          return;
        } catch (error) {
          if (cancellation.reason === "deadline_exceeded") {
            throw new ProtocolHttpError(504, "generation_timeout", "Generation exceeded its configured deadline.", {
              type: "timeout_error",
            });
          }
          if (cancellation.reason === "client_disconnect") {
            throw new ProtocolHttpError(499, "client_cancelled", "Client disconnected during generation.");
          }
          throw error;
        } finally {
          activeCancellations.delete(cancellation);
          await cancellation.dispose();
          releaseAdmission();
        }
      }

      if (req.method === "POST" && req.url === "/v1/chat/completions") {
        const requestStartedAt = Date.now();
        const completionId = `chatcmpl-${crypto.randomUUID().replaceAll("-", "")}`;
        const completionCreated = Math.floor(requestStartedAt / 1000);
        // TTFT attribution for the pre-runtime path, under
        // LIZARD_TTFT_TRACE=1. The runner's own phases account for ~15 ms of a
        // 582 ms first token, so the rest is spent here, before the runtime is
        // entered - and nothing was measuring it.
        //
        // Deliberately cheap: integer deltas and a short line, no JSON.stringify
        // of request objects. The previous server trace stringified a 43 MB
        // profile per request, which allocated ~86 MB of garbage, and the major
        // GC that followed an idle period showed up as a 3.7 s "TTFT spike" that
        // did not exist without the diagnostic. A profiler that changes the
        // number it reports is worse than none.
        const ttftTrace = process.env.LIZARD_TTFT_TRACE === "1"
          ? (label) => {
            try {
              fs.mkdirSync(".logs", { recursive: true });
              fs.appendFileSync(
                ".logs/ttft-trace.log",
                `[ttft] ${label} +${Date.now() - requestStartedAt}ms\n`,
              );
            } catch {}
          }
          : null;
        try {
          if (requiresProEntitlement) {
            assertProPlanFeatureAllowed(config, "Native API server");
          }
        } catch (error) {
          throw new ProtocolHttpError(403, error.code ?? "pro_plan_required", error.message, {
            type: "pro_plan_required",
          });
        }
        ttftTrace?.("handler-entered");
        if (draining || warmLoadState !== "ready") {
          throw new ProtocolHttpError(
            503,
            draining ? "server_draining" : "model_not_ready",
            draining ? "Server is draining." : `Model is ${warmLoadState}.`,
            { type: "service_unavailable" },
          );
        }
        const body = validateChatRequest(
          await readRequestBody(req),
          { modelId: modelArtifact.modelId, modelAliases, runtimeProvider },
        );
        ttftTrace?.("body-read");
        const lastUser = [...body.messages].reverse().find((message) => message.role === "user");
        ttftTrace?.("prompt-rendered");
        if (activeHttpCompletions >= maxConcurrentRequests) {
          throw new ProtocolHttpError(
            429,
            "server_busy",
            `Server already has its maximum of ${maxConcurrentRequests} active generation requests.`,
            { type: "rate_limit_error" },
          );
        }
        activeHttpCompletions += 1;
        let admissionReleased = false;
        const releaseAdmission = () => {
          if (admissionReleased) return;
          admissionReleased = true;
          activeHttpCompletions = Math.max(0, activeHttpCompletions - 1);
        };
        const cancellation = createRequestCancellation({
          req,
          res,
          timeoutMs: effectiveProfile.timeoutMs,
        });
        activeCancellations.add(cancellation);
        ttftTrace?.("pre-createSession");
        try {
          const session = await runtime.createSession({
            recommendation,
            modelArtifact,
            profile: effectiveProfile,
            config,
            logger: null,
          });
          cancellation.bindSession(session);
          if (cancellation.signal.aborted) {
            throw new ProtocolHttpError(499, "client_cancelled", "Request was cancelled before generation started.");
          }
          ttftTrace?.("session-created");
          const events = session.generateStream({
            requestId: httpRequestId,
            prompt: lastUser.content,
            messages: body.messages,
            stopSequences: body.stop,
            maxTokens: body.max_tokens ?? config?.generation?.maxTokens ?? 64,
            temperature: runtimeProvider === "caterpillar"
              ? 0
              : body.temperature ?? config?.generation?.temperature ?? 0,
            topK: runtimeProvider === "caterpillar"
              ? 1
              : body.top_k ?? config?.generation?.topK ?? null,
            topP: runtimeProvider === "caterpillar"
              ? 1
              : body.top_p ?? config?.generation?.topP ?? null,
            // Caterpillar remains greedy regardless of seed, but the profile
            // accepts a seed only when the requested value is preserved as
            // deterministic execution metadata.
            seed: body.seed ?? config?.generation?.seed ?? null,
            metadata: {
              stream: Boolean(body.stream),
              chunkDelayMs: effectiveProfile?.nativeRuntime?.chunkDelayMs ?? null,
              getAdmittedHttpConcurrency: () => activeHttpCompletions,
            },
          });

          if (body.stream) {
            responseProtocol = "openai-sse";
            res.writeHead(200, {
              "Content-Type": "text/event-stream; charset=utf-8",
              "Cache-Control": "no-store, no-transform",
              Connection: "keep-alive",
              "X-Accel-Buffering": "no",
            });
            const writeStreamChunk = (chunk) => writeWithBackpressure(res, chunk, cancellation.signal);
            // Prefill can run for many seconds before the worker's first live
            // token event. Comments are protocol-valid and use the same bounded
            // backpressure path as data frames.
            let keepalivePending = false;
            const keepalive = setInterval(() => {
              if (keepalivePending) return;
              keepalivePending = true;
              void writeStreamChunk(": keepalive\n\n").then((written) => {
                if (!written) cancellation.cancel("client_disconnect");
              }).catch(() => cancellation.cancel("client_disconnect"))
                .finally(() => { keepalivePending = false; });
            }, 10000);
            let firstContentAt = null;
            let streamSucceeded = false;
            try {
              await writeOpenAiChatSse({
                events,
                write: writeStreamChunk,
                id: completionId,
                created: completionCreated,
                model: modelArtifact.modelId,
                includeUsage: body.stream_options?.include_usage === true,
                includeRuntime: String(req.headers["x-lizard-extensions"] ?? "runtime").toLowerCase() !== "none",
                onFirstContent: () => {
                  ttftTrace?.("first-delta-written");
                  firstContentAt ??= Date.now();
                },
                completeRuntime: (event) => {
                  const userPerformance = recordCompletionMetric({
                    metadata: event.metadata,
                    body,
                    req,
                    surface: "openai-http-stream",
                    requestStartedAt,
                    firstContentAt,
                  });
                  return {
                    ...(event.metadata ?? {}),
                    runtimeProvider,
                    transport: "http",
                    http: userPerformance?.current ?? null,
                    userPerformance: userPerformance?.summary ?? null,
                    concurrencyPolicy: effectiveConcurrencyPolicy,
                    correlation: {
                      httpRequestId,
                      sessionId: event.sessionId,
                      generationRequestId: event.requestId,
                    },
                    ...(["caterpillar", "ostrich"].includes(runtimeProvider)
                      ? {
                        executionProvider: caterpillarExecutionProvider,
                        caterpillarPhase,
                        caterpillarPerformanceProfile: effectiveProfile.caterpillarPerformanceProfile,
                      }
                      : {}),
                  };
                },
              });
              streamSucceeded = true;
            } catch (error) {
              cancellation.cancel(cancellation.reason ?? "stream_failure");
              if (!res.destroyed) res.destroy();
              throw error;
            } finally {
              clearInterval(keepalive);
              await session.close();
              if (streamSucceeded && !res.writableEnded && !res.destroyed) res.end();
            }
            return;
          }

          let completed = null;
          try {
            for await (const event of events) {
              if (event.type === "complete") {
                completed = event;
              }
            }
          } finally {
            await session.close();
          }
          const userPerformance = recordCompletionMetric({
          metadata: completed?.metadata,
          body,
          req,
          surface: "openai-http-json",
          requestStartedAt,
          firstContentAt: Date.now(),
        });
          const runtimeMetadata = {
            ...(completed?.metadata ?? {}),
            runtimeProvider,
            transport: "http",
            http: userPerformance?.current ?? null,
            userPerformance: userPerformance?.summary ?? null,
            concurrencyPolicy: effectiveConcurrencyPolicy,
            correlation: {
              httpRequestId,
              sessionId: completed?.sessionId ?? null,
              generationRequestId: completed?.requestId ?? null,
            },
            ...(["caterpillar", "ostrich"].includes(runtimeProvider) ? {
              executionProvider: caterpillarExecutionProvider,
              caterpillarPhase,
              caterpillarPerformanceProfile: effectiveProfile.caterpillarPerformanceProfile,
            } : {}),
          };
          sendJson(res, 200, buildOpenAiChatCompletion({
            id: completionId,
            created: completionCreated,
            model: modelArtifact.modelId,
            completed,
            runtime: runtimeMetadata,
            includeRuntime: String(req.headers["x-lizard-extensions"] ?? "runtime").toLowerCase() !== "none",
          }));
          return;
        } catch (error) {
          if (cancellation.reason === "deadline_exceeded") {
            throw new ProtocolHttpError(504, "generation_timeout", "Generation exceeded its configured deadline.", {
              type: "timeout_error",
            });
          }
          if (cancellation.reason === "client_disconnect") {
            throw new ProtocolHttpError(499, "client_cancelled", "Client disconnected during generation.");
          }
          throw error;
        } finally {
          activeCancellations.delete(cancellation);
          await cancellation.dispose();
          releaseAdmission();
        }
      }

      throw new ProtocolHttpError(404, "route_not_found", "Route not found.", { param: null });
    } catch (error) {
      // Map known runtime errors to 4xx so a bad request (empty prompt,
      // unsupported model, missing tokenizer vocabulary) is not reported as a
      // server fault. Everything else stays a 500.
      const classified = classifyProtocolHttpError(error);
      const status = classified.statusCode;
      const incident = status >= 500 && status !== 503 && status !== 504
        ? await reportCriticalFailure(error, {
          operation: "chat-completion",
          route: req.url,
          surface: "openai-http",
        })
        : null;
      const incidentFields = incidentUserFields(incident);
      // A stream commits 200 before generation so role/deltas and keepalives
      // can flow. Once that happens, a failure must close the transport without
      // a finish chunk or [DONE]; appending JSON would forge a successful SSE
      // terminal. Buffered responses still return a structured JSON error when
      // their headers have not yet been committed.
      if (res.headersSent) {
        const message = error?.message ?? String(error);
        console.error(`chat completion failed after the response began: ${message}`);
        if (responseProtocol?.endsWith("-sse") || responseProtocol === "ollama-ndjson") {
          if (!res.destroyed) res.destroy();
          return;
        }
        if (res.writableEnded || res.destroyed) {
          return;
        }
        try {
          res.end(JSON.stringify({
            error: {
              message,
              type: status === 500 ? "server_error" : "unprocessable_entity",
              code: classified.code,
              note: "The response status was already committed to 200 before generation began.",
              ...incidentFields,
            },
          }));
        } catch {
          if (!res.writableEnded && !res.destroyed) {
            res.destroy();
          }
        }
        return;
      }
      if (req.url?.startsWith("/api/")) {
        sendJson(res, status, { error: error?.message ?? String(error) });
      } else {
        sendJson(res, status, protocolErrorPayload({
          message: error?.message ?? String(error),
          type: classified.type,
          param: classified.param,
          code: classified.code,
        }, incidentFields));
      }
    }
  });

  await new Promise((resolve, reject) => {
    server.listen(Number(port), host, () => resolve());
    server.on("error", reject);
  });

  // Own the port before any model worker starts. A preflight probe reduces
  // stale-port mistakes, but only the successful bind closes the final race;
  // loading first could briefly hold a second multi-GB model and then fail
  // with EADDRINUSE.
  beginWarmLoad();
  let shutdownPromise = null;
  const shutdown = (signal) => {
    if (shutdownPromise) return shutdownPromise;
    draining = true;
    shutdownPromise = (async () => {
      for (const cancellation of activeCancellations) cancellation.cancel("server_shutdown");
      const serverClosed = new Promise((resolve) => server.close(resolve));
      server.closeIdleConnections?.();
      await Promise.all([...activeCancellations].map((cancellation) => cancellation.dispose()));
      await runtime.releaseAllWorkers();
      server.closeAllConnections?.();
      await serverClosed;
      console.log(`${providerLabel} server stopped (${signal}).`);
    })();
    return shutdownPromise;
  };
  for (const signal of ["SIGINT", "SIGTERM"]) {
    process.once(signal, () => {
      shutdown(signal).then(() => {
        process.exitCode = 0;
      }).catch((error) => {
        console.error(`${providerLabel} shutdown failed: ${error.message}`);
        process.exitCode = 1;
      });
    });
  }
  retryPendingMajorFailureIncidents({ config: incidentConfig }).catch((error) => {
    console.error(`major-failure delivery retry failed safely: ${error?.message ?? error}`);
  });
  console.log(`${providerLabel} server ready on http://${host}:${port}`);
}

main().catch(async (error) => {
  if (failureIntelligenceConfig) {
    try {
      const incident = await reportMajorFailure({
        error,
        config: failureIntelligenceConfig,
        context: {
          component: "native-http-server",
          operation: "server-startup",
          unrecoverable: true,
        },
        user: failureIntelligenceConfig?.failureIntelligence?.userEmail
          ? { email: failureIntelligenceConfig.failureIntelligence.userEmail }
          : null,
      });
      if (incident?.reported) {
        console.error(`Ticket ${incident.ticketId}: ${incident.message}`);
      }
    } catch {
      // Reporting must never obscure or replace the startup failure.
    }
  }
  console.error(error.message);
  process.exitCode = 1;
});
