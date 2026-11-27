import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";
import { snapshotLifecycleState } from "../../lib/model-lifecycle.js";
import { inspectGgufFile } from "../../lib/gguf-metadata.js";
import {
  buildNativeBackendFeatureRegistry,
  resolveNativeBackendRoute,
} from "../backend/index.js";
import { mergeRuntimeContract, NATIVE_RUNTIME_PROVIDER } from "../contract.js";
import {
  createStopSequenceFilter,
  DEFAULT_STOP_MARKERS,
  truncateAtStopSequence,
} from "./stop-sequences.js";
import { buildKvCachePlan, createKvCacheRuntime } from "../kv-cache/index.js";
import { createRequestScheduler } from "../scheduler/index.js";
import { createBatchCoordinator } from "./batch-coordinator.js";
import {
  buildRawPromptArtifact,
  buildPromptArtifact,
  createIncrementalTokenDecoder,
  decodeGeneratedTokenIds,
} from "../tokenizer/index.js";
import {
  NativeWorkerProcessClient,
  prefersCompiledNativeWorker,
  resolveNativeWorkerEnvironment,
} from "../worker/process-client.js";
import { buildNativeExecutionPlan } from "../native-execution-plan.js";
import {
  applyHardwareTuningHint,
  buildHardwareTuningIdentity,
  evaluateHardwareCalibration,
  loadHardwareTuningHint,
  saveHardwareTuningHint,
} from "../hardware-tuning.js";
import { ExistingBackendRuntime, InferenceRuntime } from "./existing-backend-runtime.js";
import { ModelSession } from "./model-session.js";

// A GGUF whose header cannot even fit is not a model, it is a leftover.
const GGUF_MIN_FILE_BYTES = 1024;

// Fails a doomed load before a worker process is spawned for it.
//
// The failure this catches is a leftover: a model file the catalog still vouches
// for but that a previous install removed, or a file that was replaced by
// something that is not a GGUF at all. Handed such a path, the worker maps it
// and reads a header that is not there - and the symptom is a wedged process
// rather than a message, because the wedge happens below the level where errors
// are reported. Four bytes of checking here turn "the app hangs when I pick a
// model" into a sentence naming the file and what to do about it.
//
// Deliberately narrow: it proves the file is present and is a GGUF, nothing
// more. Judging whether the model FITS is the worker's job (it has the live
// memory facts); this only refuses to start a conversation with a file that
// cannot answer.
export function assertLoadableModelFile(modelArtifact) {
  const modelPath = modelArtifact?.modelPath;
  // Prototype artifacts run on the JS worker and are not GGUF-backed; they have
  // nothing on disk to verify.
  if (!modelPath || path.extname(String(modelPath)).toLowerCase() !== ".gguf") {
    return;
  }
  const modelId = modelArtifact?.modelId ?? "the selected model";
  let stats = null;
  try {
    stats = fs.statSync(modelPath);
  } catch {
    const error = new Error(
      `The model file for '${modelId}' is missing: ${modelPath}. `
      + "Remove the model from your library and download it again.",
    );
    error.code = "MODEL_FILE_MISSING";
    throw error;
  }
  if (!stats.isFile() || stats.size < GGUF_MIN_FILE_BYTES) {
    const error = new Error(
      `The model file for '${modelId}' is incomplete (${stats.size} bytes): ${modelPath}. `
      + "The download did not finish. Remove the model from your library and download it again.",
    );
    error.code = "MODEL_FILE_INCOMPLETE";
    throw error;
  }
  let magic = null;
  let handle = null;
  try {
    handle = fs.openSync(modelPath, "r");
    const header = Buffer.alloc(4);
    fs.readSync(handle, header, 0, 4, 0);
    magic = header.toString("latin1");
  } catch {
    // Unreadable for some other reason (permissions, a lock). Let the worker
    // report that in its own words rather than guessing at it here.
  } finally {
    if (handle != null) {
      try {
        fs.closeSync(handle);
      } catch {
        // Nothing depends on the close succeeding.
      }
    }
  }
  if (magic != null && magic !== "GGUF") {
    const error = new Error(
      `The model file for '${modelId}' is not a GGUF model: ${modelPath}. `
      + "It is corrupt or was replaced. Remove the model from your library and download it again.",
    );
    error.code = "MODEL_FILE_CORRUPT";
    throw error;
  }
}

function cancelledError(requestId) {
  const error = new Error(`Scheduler request ${requestId} was cancelled before execution.`);
  error.code = "SCHEDULER_CANCELLED";
  return error;
}

// LIZARD_SERVER_TRACE=1 appends per-request step timings to
// .logs/server-trace.log (cwd-relative) so the pre-prefill overhead
// (~1.9 s unattributed as of 2026-07-06) can be attributed without a
// debugger. A file is used because launchers pipe the server's stderr
// without reading it. Elapsed values are cumulative from onGenerate entry;
// per-step cost is the delta between consecutive lines.
const serverTraceEnabled = process.env.LIZARD_SERVER_TRACE === "1";
function traceStep(requestId, step, entryStartedAt) {
  if (!serverTraceEnabled) {
    return;
  }
  const line =
    `[server-trace] ${new Date().toISOString()} ${requestId} ${step} `
    + `+${(performance.now() - entryStartedAt).toFixed(1)}ms\n`;
  try {
    fs.mkdirSync(".logs", { recursive: true });
    fs.appendFileSync(path.join(".logs", "server-trace.log"), line);
  } catch {
    process.stderr.write(line);
  }
}

function compatibilityFailureMessage(details = null) {
  const architectureSupported = details?.architecture?.status === "supported";
  const quantizationSupported = details?.quantization?.status === "supported";
  const reasons = [
    architectureSupported ? null : details?.architecture?.reason,
    quantizationSupported ? null : details?.quantization?.reason,
    architectureSupported && quantizationSupported && details?.executor?.status !== "supported"
      ? details?.executor?.reason
      : null,
    details?.memory?.fits === false ? details.memory.reason : null,
    details?.context?.fits === false ? details.context.reason : null,
  ].filter(Boolean);
  return reasons.length ? reasons.join(" ") : null;
}

function executionUnavailableError(loadResult = null) {
  const message = compatibilityFailureMessage(loadResult?.compatibility)
    ?? "The native worker loaded model metadata, but executable native inference is not available for this model yet.";
  const error = new Error(message);
  error.code = loadResult?.compatibility?.executor?.status === "planned"
    ? "NATIVE_EXECUTOR_UNIMPLEMENTED"
    : "NATIVE_MODEL_UNSUPPORTED";
  error.details = loadResult?.compatibility ?? null;
  return error;
}

async function collectFallbackResult(session, request) {
  let completed = null;
  let output = "";
  for await (const event of session.generateStream(request)) {
    if (event.type === "delta") {
      output += event.text ?? "";
    } else if (event.type === "complete") {
      completed = event;
    }
  }
  return {
    output: completed?.output ?? output,
    durationMs: completed?.durationMs ?? null,
    runtime: completed?.runtime ?? null,
    metadata: completed?.metadata ?? {},
    chunks: [],
  };
}

function isGgufNativeModel(modelArtifact) {
  return !modelArtifact?.runtimePrototype?.format
    && String(modelArtifact?.modelPath ?? "").toLowerCase().endsWith(".gguf");
}

// The exact tokenizer (e.g. Gemma sentencepiece/BPE) needs the GGUF token
// vocabulary from `gguf.raw["tokenizer.ggml.tokens"]`. Profiles handed to the
// runtime often carry only thin metadata (or null), which yields no exact
// token IDs and makes the native worker reject prefill with "prompt token IDs
// are unavailable" (HTTP 500). Detect that and parse the full GGUF once.
function ggufHasVocabulary(gguf) {
  const tokens = gguf?.raw?.["tokenizer.ggml.tokens"];
  return Array.isArray(tokens) && tokens.length > 0;
}

const _fullGgufCache = new Map();

function resolveTokenizerGguf({ profile, modelArtifact }) {
  const providedGguf = profile?.gguf ?? modelArtifact?.gguf ?? null;
  if (ggufHasVocabulary(providedGguf)) {
    return providedGguf;
  }
  const modelPath = modelArtifact?.modelPath ?? profile?.modelPath ?? null;
  if (!modelPath || !String(modelPath).toLowerCase().endsWith(".gguf")) {
    return providedGguf;
  }
  if (_fullGgufCache.has(modelPath)) {
    return _fullGgufCache.get(modelPath) ?? providedGguf;
  }
  let inspected = null;
  try {
    inspected = inspectGgufFile(modelPath, {
      fallbackQuant: providedGguf?.quant ?? null,
      fallbackArchitecture: providedGguf?.architecture ?? null,
    });
  } catch {
    inspected = null;
  }
  const usable = ggufHasVocabulary(inspected) ? inspected : null;
  _fullGgufCache.set(modelPath, usable);
  return usable ?? providedGguf;
}

// EnhancementRun_04 Phase 4 -- adapter, so the batched path is a drop-in.
//
// The coordinator resolves with token ids and text; everything downstream of
// `generate` expects the worker's richer result (session metrics, backend
// label, timings). Rather than teach the caller about two shapes, this returns
// the one it already handles -- which is what keeps the swap above a single
// ternary instead of a rewrite of the streaming and telemetry code below it.
//
// One coordinator per worker, created on first use: batching is only
// meaningful between sequences sharing a model, and the worker record is
// exactly that scope.
export function calculateBatchedTokensPerSecond(generatedTokenCount, decodeMs) {
  const tokens = Number(generatedTokenCount);
  const milliseconds = Number(decodeMs);
  return Number.isFinite(tokens) && tokens >= 0
    && Number.isFinite(milliseconds) && milliseconds > 0
    ? tokens / (milliseconds / 1000)
    : 0;
}

export function shouldUseBatchedDecode({
  enabled,
  compiledWorker,
  supported,
  alreadyBusy,
  alwaysBatch,
  admittedHttpConcurrency,
  temperature,
} = {}) {
  const greedy = Number(temperature ?? 0) <= 0;
  const overlappingHttpRequests = Number(admittedHttpConcurrency) > 1;
  return enabled === true
    && compiledWorker === true
    && supported === true
    && greedy
    && (alreadyBusy === true || (alwaysBatch === true && overlappingHttpRequests));
}

async function generateViaBatchCoordinator({
  worker,
  workerClient,
  sessionId,
  prefill,
  onToken,
  batchFloor,
  maxBatchSize,
}) {
  if (!worker._batchCoordinator) {
    worker._batchCoordinator = createBatchCoordinator({
      client: workerClient,
      ...(Number.isFinite(batchFloor) ? { batchFloor } : {}),
    });
  }
  const started = Date.now();
  const settled = await worker._batchCoordinator.submit({
    sessionId,
    maxBatchSize,
    onToken: typeof onToken === "function"
      // The coordinator hands over the raw token exactly as the worker's
      // streaming notification does, so the stop filter downstream sees the
      // same input on both paths.
      // Preserve the worker metadata too. In particular, the incremental GGUF
      // decoder requires tokenId; dropping it made GPT-2 BPE pieces such as
      // "Ġcache" escape into HTTP output as mojibake on the batched path.
      ? (token, emitted) => onToken({ ...emitted, token })
      : null,
  });
  const decodeMs = Date.now() - started;
  // Metrics are fetched after the fact rather than carried through the batch:
  // a step serves many sessions, so per-session metrics belong to the session,
  // not to the step that happened to advance it.
  const metrics = await workerClient.metrics({ sessionId }).catch(() => null);
  const workerSessionMetrics = metrics?.sessions?.find?.((entry) => entry.sessionId === sessionId)
    ?? metrics?.sessionMetrics
    ?? prefill?.sessionMetrics
    ?? null;
  const sessionMetrics = {
    ...(workerSessionMetrics ?? {}),
    ...(settled.batchMetrics ?? {}),
    concurrencyLevel: Math.max(1, Number(settled.batchMetrics?.maxBatchSize) || 1),
    servingPath: "batched",
  };
  const generatedTokenCount = settled.tokenIds?.length ?? 0;
  const tokensPerSecond = calculateBatchedTokensPerSecond(generatedTokenCount, decodeMs);
  return {
    output: settled.text,
    chunks: undefined,
    generatedTokenIds: settled.tokenIds,
    finishReason: settled.finishReason,
    nativeBackend: sessionMetrics?.decodeBackend ?? "cpu",
    // The batched loop runs the same llama graph as `generate`; only the number
    // of sequences advanced per step differs. Reporting "loaded" here said the
    // weights were in memory but the graph had not run, which is what the
    // single-sequence path reports before it decodes -- a real under-report that
    // made a batched answer look like it came from nowhere.
    nativeModelStatus: "llama-graph-executed",
    prefillMs: prefill?.prefillMs ?? 0,
    decodeMs,
    tokensPerSecond,
    sessionMetrics,
    compatibility: null,
    batched: true,
  };
}

function prefillSnapshotKey(promptArtifact) {
  const hash = crypto.createHash("sha256");
  hash.update(String(promptArtifact?.tokenIdMode ?? ""));
  hash.update("\0");
  for (const tokenId of promptArtifact?.tokenIds ?? []) {
    hash.update(`${tokenId},`);
  }
  return hash.digest("hex");
}

async function trimPrefillSnapshots(worker, workerClient, maxEntries) {
  const cache = worker._prefillSnapshots;
  if (!(cache instanceof Map) || cache.size <= maxEntries) {
    return;
  }
  const candidates = [...cache.entries()]
    .filter(([, entry]) => entry.ready && entry.users === 0)
    .sort((left, right) => left[1].lastUsedAt - right[1].lastUsedAt);
  while (cache.size > maxEntries && candidates.length > 0) {
    const [key, entry] = candidates.shift();
    if (cache.get(key) !== entry || entry.users !== 0) continue;
    cache.delete(key);
    await workerClient.closeSession({ sessionId: entry.sessionId }).catch(() => {});
  }
}

// Publishes an already-prefilled session as a snapshot for `key`, by cloning it
// into a session the cache owns. Cloning is a KV copy, not a prefill, so this
// costs a memcpy rather than another pass over the weights - and it is what lets
// each turn hand the next turn a prefix that ends where this one does.
//
// Failure is non-fatal by design: the request that triggered it has already
// produced its answer, and a missing snapshot only costs the next turn some
// prefill.
async function registerPrefillSnapshot({
  worker,
  workerClient,
  cache,
  key,
  sourceSessionId,
  tokenIds,
  prefillParams,
}) {
  if (cache.has(key) || typeof workerClient.cloneSession !== "function") {
    return;
  }
  const sessionId = `prefill-snapshot-${key.slice(0, 24)}`;
  const entry = {
    sessionId,
    users: 0,
    ready: false,
    lastUsedAt: Date.now(),
    promise: null,
    tokenIds: Array.isArray(tokenIds) ? tokenIds : null,
  };
  entry.promise = (async () => {
    await workerClient.cloneSession({
      sourceSessionId,
      sessionId,
      maxTokens: prefillParams.maxTokens,
      stopSequences: prefillParams.stopSequences ?? [],
    });
    entry.ready = true;
    return null;
  })().catch(async (error) => {
    if (cache.get(key) === entry) cache.delete(key);
    await workerClient.closeSession({ sessionId }).catch(() => {});
    throw error;
  });
  cache.set(key, entry);
  try {
    await entry.promise;
  } catch {
    // Already removed from the cache above.
  }
}

// Longest usable prefix among the cached snapshots.
//
// A chat transcript only ever grows: turn N+1's prompt is turn N's prompt, plus
// turn N's answer, plus the new question. So turn N's snapshot is a proper
// prefix of turn N+1's prompt, and the only work turn N+1 genuinely owes is the
// delta.
//
// Without this, every turn re-prefilled the whole transcript and TTFT grew with
// conversation length - measured 1404, 2909, 5026, 6605, 9352 ms over five turns
// of one ordinary conversation. By turn five a user waited 9.4 s for the first
// character, against llama-server's flat 427 ms (it caches prompt prefixes
// across requests, which is exactly what this restores).
function findPrefixSnapshot(cache, tokenIds, minPrefixTokens = 16) {
  if (!Array.isArray(tokenIds) || tokenIds.length === 0) {
    return null;
  }
  let best = null;
  let bestCommon = 0;
  for (const candidate of cache.values()) {
    const candidateTokens = candidate.tokenIds;
    if (!Array.isArray(candidateTokens) || candidateTokens.length === 0) continue;
    // Longest COMMON prefix, not full containment. Requiring the whole cached
    // sequence to match looked right and never fired once: BPE is not
    // prefix-stable at a boundary, so appending "\nassistant: ..." to a prompt
    // re-tokenises its final token or two. Turn 1 cached 23 tokens, turn 2 asked
    // for 46, and a full-containment test rejected it - measured, and the reason
    // the first version of this changed nothing.
    //
    // A shorter common prefix is still safe to reuse: the worker's
    // try_reuse_kv_prefix computes the same common length and truncates the
    // cloned KV to it, validating every lane before mutating anything.
    const limit = Math.min(candidateTokens.length, tokenIds.length);
    let common = 0;
    while (common < limit && candidateTokens[common] === tokenIds[common]) {
      common += 1;
    }
    if (common < minPrefixTokens) continue;
    if (common <= bestCommon) continue;
    best = candidate;
    bestCommon = common;
  }
  return best ? { entry: best, common: bestCommon } : null;
}

// Prefill snapshots close the largest HTTP-vs-direct gap without sharing mutable
// KV. A dedicated source session is prefetched once; every request gets a deep,
// independently writable clone. Concurrent misses share the same promise, so a
// burst of identical API calls performs one prefill, not one prefill per socket.
//
// Exact matches clone and are done. A prompt that merely EXTENDS a cached one
// clones the prefix and then prefills the full token list against the clone -
// the worker's own try_reuse_kv_prefix compares the incoming tokens against the
// session's existing sequence and processes only what is new, so the delta is
// computed in the runner rather than guessed here.
export async function prefillWithSnapshotCache({
  worker,
  workerClient,
  targetSessionId,
  promptArtifact,
  prefillParams,
  maxEntries = 8,
}) {
  if (!worker._prefillSnapshots) {
    worker._prefillSnapshots = new Map();
  }
  const cache = worker._prefillSnapshots;
  const key = prefillSnapshotKey(promptArtifact);
  const promptTokenIds = Array.isArray(promptArtifact?.tokenIds) ? promptArtifact.tokenIds : null;
  let entry = cache.get(key);
  const hit = Boolean(entry);
  let prefixMatch = null;

  if (!entry && promptTokenIds) {
    prefixMatch = findPrefixSnapshot(cache, promptTokenIds);
  }
  const prefixEntry = prefixMatch?.entry ?? null;
  if (serverTraceEnabled) {
    traceStep(
      prefillParams?.sessionId ?? "snapshot",
      `snapshot-lookup cacheSize=${cache.size} tokens=${promptTokenIds?.length ?? "none"} `
      + `exact=${hit} reusedPrefix=${prefixMatch ? prefixMatch.common : "none"} `
      + `cached=[${[...cache.values()].map((e) => e.tokenIds?.length ?? "?").join(",")}]`,
      performance.now(),
    );
  }

  if (!entry && prefixEntry) {
    // Reuse the prefix. The clone starts with that KV already populated, and the
    // prefill below only pays for the tokens beyond it.
    prefixEntry.users += 1;
    prefixEntry.lastUsedAt = Date.now();
    try {
      await prefixEntry.promise;
      await workerClient.cloneSession({
        sourceSessionId: prefixEntry.sessionId,
        sessionId: targetSessionId,
        maxTokens: prefillParams.maxTokens,
        stopSequences: prefillParams.stopSequences ?? [],
      });
      const extended = await workerClient.prefill({
        ...prefillParams,
        sessionId: targetSessionId,
      });
      // Register the fuller state so the NEXT turn finds a prefix that ends
      // where this one does. Without this the reusable prefix stays frozen at
      // the first turn and the delta grows again every turn.
      await registerPrefillSnapshot({
        worker,
        workerClient,
        cache,
        key,
        sourceSessionId: targetSessionId,
        tokenIds: promptTokenIds,
        prefillParams,
      });
      return {
        ...extended,
        prefillSnapshotHit: false,
        prefillSnapshotPrefixHit: true,
        prefillSnapshotPrefixTokens: prefixMatch.common,
        prefillSnapshotKey: key,
      };
    } finally {
      prefixEntry.users = Math.max(0, prefixEntry.users - 1);
      prefixEntry.lastUsedAt = Date.now();
      await trimPrefillSnapshots(
        worker,
        workerClient,
        Math.max(1, Math.trunc(Number(maxEntries) || 8)),
      );
    }
  }

  if (!entry) {
    const sessionId = `prefill-snapshot-${key.slice(0, 24)}`;
    entry = {
      sessionId,
      users: 0,
      ready: false,
      lastUsedAt: Date.now(),
      promise: null,
      // Needed for prefix matching by later turns.
      tokenIds: promptTokenIds,
    };
    entry.promise = (async () => {
      await workerClient.openSession({
        sessionId,
        modelId: prefillParams.modelId,
      });
      const result = await workerClient.prefill({
        ...prefillParams,
        sessionId,
      });
      entry.ready = true;
      return result;
    })().catch(async (error) => {
      if (cache.get(key) === entry) cache.delete(key);
      await workerClient.closeSession({ sessionId }).catch(() => {});
      throw error;
    });
    cache.set(key, entry);
  }

  entry.users += 1;
  entry.lastUsedAt = Date.now();
  try {
    await entry.promise;
    const cloned = await workerClient.cloneSession({
      sourceSessionId: entry.sessionId,
      sessionId: targetSessionId,
      maxTokens: prefillParams.maxTokens,
      stopSequences: prefillParams.stopSequences ?? [],
    });
    return {
      ...cloned,
      prefillSnapshotHit: hit,
      prefillSnapshotKey: key,
    };
  } finally {
    entry.users = Math.max(0, entry.users - 1);
    entry.lastUsedAt = Date.now();
    await trimPrefillSnapshots(
      worker,
      workerClient,
      Math.max(1, Math.trunc(Number(maxEntries) || 8)),
    );
  }
}

export function resolveCachedPromptTokens(prefill = null) {
  if (!prefill || typeof prefill !== "object") return 0;
  if (prefill.promptCacheReused === true) {
    return Math.max(0, Math.trunc(Number(prefill.promptCacheReusedTokens) || 0));
  }
  if (prefill.prefillSnapshotPrefixHit === true) {
    return Math.max(0, Math.trunc(Number(prefill.prefillSnapshotPrefixTokens) || 0));
  }
  if (prefill.prefillSnapshotHit === true) {
    return Math.max(0, Math.trunc(Number(prefill.kvPrefixReusedTokens) || 0));
  }
  // A cache miss still clones the snapshot built during that same request.
  // The clone contains the full prompt, but none of it was cached before the
  // request began, so OpenAI cached_tokens must remain zero.
  if (prefill.prefillSnapshotHit === false) return 0;
  return Math.max(0, Math.trunc(Number(prefill.kvPrefixReusedTokens) || 0));
}

export class NativeCpuRuntime extends InferenceRuntime {
  constructor({
    engine = NATIVE_RUNTIME_PROVIDER,
    generate,
    // The scheduler defaults to maxConcurrent=1, which serialises every request
    // before it can reach the batch coordinator. That is correct for a CLI that
    // runs one generation at a time, and it is the reason EnhancementRun_04's
    // batched decode never engaged for anyone: three chokepoints were found and
    // fixed above the engine, and this fourth one -- the outermost of all --
    // meant the coordinator could never be handed a second sequence to batch
    // with. Callers that serve several requests at once pass their own limit.
    maxConcurrentRequests = 1,
    scheduler = createRequestScheduler({ maxConcurrent: maxConcurrentRequests }),
    workerClientFactory = null,
    fallbackRuntime = null,
  } = {}) {
    super({ engine });
    this._scheduler = scheduler;
    // Whether loads go to a real worker process. The on-disk model check below
    // guards THAT process - it exists because a real worker handed a missing or
    // non-GGUF file wedges instead of erroring. A caller supplying its own
    // client is not bound by that contract and may have no file at all, so the
    // check applies only to the spawning path.
    this._spawnsRealWorker = !workerClientFactory;
    this._workerClientFactory = workerClientFactory
      ?? (({ config, modelArtifact, profile }) => new NativeWorkerProcessClient({
        config,
        modelArtifact,
        profile,
        runtimeProvider: engine,
      }));
    this._kvCaches = new Map();
    this._workers = new Map();
    this._fallbackRuntime = fallbackRuntime ?? new ExistingBackendRuntime({
      engine: "llama.cpp",
      generate,
    });
  }

  kvCacheForModel({ modelId, profile = null, gguf = null }) {
    const cacheKey = String(modelId);
    if (!this._kvCaches.has(cacheKey)) {
      this._kvCaches.set(cacheKey, createKvCacheRuntime({
        modelId,
        profile,
        gguf,
      }));
    }
    return this._kvCaches.get(cacheKey);
  }

  _workerCacheKey(modelArtifact, profile = null) {
    const performanceProfile = String(
      profile?.caterpillarPerformanceProfile
        ?? profile?.nativeRuntime?.caterpillarPerformanceProfile
        ?? "balanced",
    );
    return `${modelArtifact.modelId}:${modelArtifact.modelPath}:performance-profile=${performanceProfile}`;
  }

  async ensureModelLoaded({
    recommendation,
    modelArtifact,
    profile = null,
    config = null,
  }) {
    const cacheKey = this._workerCacheKey(modelArtifact, profile);
    if (!this._workers.has(cacheKey)) {
      // Checked before anything is spawned or cached, so a leftover file fails
      // as a message instead of as a wedged worker process.
      if (this._spawnsRealWorker) {
        assertLoadableModelFile(modelArtifact);
      }
      // Spawned out here rather than inside the async body so the worker process
      // has a handle from the instant it exists. Loading a multi-gigabyte model
      // is precisely when a user gives up and hits Unload or Delete, and those
      // paths need something they can shut down now - not "whatever the load I
      // am trying to interrupt eventually returns".
      const client = this._workerClientFactory({ config, modelArtifact, profile, recommendation });
      const loadPromise = (async () => {
        try {
          const handshake = await client.handshake();
          const workerEnvironment = resolveNativeWorkerEnvironment({
            profile,
            modelArtifact,
            runtimeProvider: this.engine,
          });
          const loadResult = await client.loadModel({
            recommendation,
            modelArtifact,
            profile,
          });
          if (loadResult?.gpuWeightBudgetBytes == null) {
            // The worker's live DXGI-clamped budget (gpuMemory, Phase 1
            // Finding 2) is authoritative; the launch-side request is only
            // the fallback when the GPU lane never initialized.
            const effectiveBudget = Number(loadResult?.gpuMemory?.effectiveWeightBudgetBytes);
            if (Number.isFinite(effectiveBudget) && effectiveBudget > 0) {
              loadResult.gpuWeightBudgetBytes = effectiveBudget;
            } else if (workerEnvironment.LIZARD_NATIVE_GPU_WEIGHT_BUDGET_MB) {
              loadResult.gpuWeightBudgetBytes = Number(workerEnvironment.LIZARD_NATIVE_GPU_WEIGHT_BUDGET_MB) * 1024 * 1024;
            }
          }
          if (loadResult?.executionReady === false) {
            throw executionUnavailableError(loadResult);
          }
          const calibrationPlan = buildNativeExecutionPlan({
            capabilityRegistry: profile?.capabilityRegistry ?? null,
            profile,
            gguf: modelArtifact?.gguf ?? profile?.gguf ?? null,
            requestedBackend: profile?.backend ?? "cpu",
            runtimeProvider: this.engine,
          });
          loadResult.cpuBuildProfile = handshake?.cpuBuildProfile ?? null;
          const tuningIdentity = buildHardwareTuningIdentity({
            profile,
            modelArtifact,
            executionPlan: calibrationPlan,
          });
          let hardwareTuning = loadHardwareTuningHint(config, tuningIdentity);
          if (!hardwareTuning
            && calibrationPlan?.execution?.allowGpuDecode
            && process.env.LIZARD_NATIVE_HARDWARE_CALIBRATION !== "0") {
            try {
              const benchmark = await client.tensorBenchmark({
                backend: "compare",
                iterations: 1,
              });
              const evaluated = evaluateHardwareCalibration(benchmark?.calibration);
              hardwareTuning = saveHardwareTuningHint(config, tuningIdentity, {
                ...evaluated,
                calibration: benchmark?.calibration ?? null,
                tensorName: benchmark?.tensorName ?? null,
                source: "bounded-native-tensor-calibration",
              }) ?? evaluated;
            } catch (error) {
              hardwareTuning = {
                recommendedDecodeBackend: "cpu",
                reason: `Hardware calibration failed safely: ${error.message}`,
                source: "calibration-failure",
              };
            }
          }
          loadResult.hardwareTuning = hardwareTuning;
          // Pre-warm the tokenizer definition cache off the load path so the
          // first real request does not pay the vocabulary build (~1.2 s
          // measured for the 262k-entry gemma vocab, 2026-07-06).
          setImmediate(() => {
            try {
              // A real (non-empty) prompt is required: an empty prompt
              // short-circuits before the vocabulary build.
              buildPromptArtifact({
                prompt: "warmup",
                modelPath: modelArtifact.modelPath,
                gguf: resolveTokenizerGguf({ profile, modelArtifact }),
              });
            } catch {
              // Best-effort warm-up; the request path builds it on demand.
            }
          });
          return {
            client,
            loadResult,
            handshake,
          };
        } catch (error) {
          // Order matters: drop the cache entry first, so a retry starts a fresh
          // worker even if shutting the failed one down is itself slow. Leaving
          // the entry in place while awaiting the shutdown meant a second
          // attempt joined the dead load instead of replacing it.
          this._workers.delete(cacheKey);
          await client.shutdown().catch(() => {});
          throw error;
        }
      })();
      // A load can outlive whoever asked for it - the page closes, the model is
      // deleted, the user activates something else. Node treats an unhandled
      // rejection as fatal, so without this an abandoned failing load would take
      // the whole app down instead of just failing.
      loadPromise.catch(() => {});
      this._workers.set(cacheKey, { client, promise: loadPromise });
    }
    return this._workers.get(cacheKey).promise;
  }

  async releaseModelWorker(modelArtifact, profile = null) {
    const entries = profile
      ? [[this._workerCacheKey(modelArtifact, profile), this._workers.get(this._workerCacheKey(modelArtifact, profile))]]
      : Array.from(this._workers.entries()).filter(([cacheKey]) => cacheKey.startsWith(
        `${modelArtifact.modelId}:${modelArtifact.modelPath}:performance-profile=`,
      ));
    if (entries.length === 0) return;
    // Deleted up front so an in-flight load cannot re-register itself behind us,
    // and so the next ensureModelLoaded builds a new worker rather than waiting
    // on the one being torn down.
    for (const [cacheKey] of entries) this._workers.delete(cacheKey);
    // Shut the process down without waiting for the load to finish. Awaiting the
    // load first is what made Unload and Delete appear to hang during a slow
    // load: the button did nothing for as long as the load ran, which for a
    // wedged worker was forever. The client exists from spawn time, so it can
    // always be stopped - and stopping it is what makes the pending load fail
    // and unblock its own awaiters.
    //
    // The load promise is deliberately NOT awaited here, not even to observe its
    // failure. Waiting on it would put the unbounded wait straight back where it
    // was taken out; it is already guarded against an unhandled rejection where
    // it is created, which is the only reason anything needed to await it.
    await Promise.all(entries.map(async ([, entry]) => {
      await entry?.client?.shutdown?.().catch(() => {});
    }));
  }

  // Used on process shutdown (see the SIGINT/SIGTERM handler in
  // src/ui/server.js) to free every resident model at once, not just one
  // known modelArtifact - the cache can hold more than one worker if the
  // active model was switched during the process lifetime.
  async releaseAllWorkers() {
    const entries = Array.from(this._workers.values());
    this._workers.clear();
    await Promise.all(entries.map(async (entry) => {
      // Same reasoning as releaseModelWorker: shut down the process we hold,
      // rather than waiting on a load that may never return. This runs on
      // SIGINT/SIGTERM, where waiting on a wedged load meant the app refused to
      // exit and left its port bound.
      await entry.client?.shutdown?.().catch(() => {});
    }));
  }

  async createSession({
    recommendation,
    modelArtifact,
    profile = null,
    config,
    logger,
    enginePath,
  }) {
    const allowBackendFallback = Boolean(
      profile?.nativeRuntime?.allowBackendFallback
      ?? modelArtifact?.runtimePrototype?.allowBackendFallback
      ?? false,
    );
    const keepWorkerLoaded = Boolean(profile?.nativeRuntime?.keepWorkerLoaded ?? false);
    const allowOracleFallback = Boolean(
      profile?.nativeRuntime?.allowOracleFallback
      ?? false,
    );
    const requestedBackend = profile?.backend ?? "cpu";
    const backendRoute = resolveNativeBackendRoute({
      capabilityRegistry: profile?.capabilityRegistry ?? null,
      requestedBackend,
      allowBackendFallback,
    });
    const backendRegistry = buildNativeBackendFeatureRegistry({
      capabilityRegistry: profile?.capabilityRegistry ?? null,
      requestedBackend,
    });
    const executionPlan = buildNativeExecutionPlan({
      capabilityRegistry: profile?.capabilityRegistry ?? null,
      profile,
      gguf: profile?.gguf ?? modelArtifact?.gguf ?? null,
      requestedBackend,
      runtimeProvider: this.engine,
    });

    if (backendRoute.fallbackUsed || backendRoute.backend !== "cpu") {
      if (!allowOracleFallback && !allowBackendFallback) {
        throw new Error(backendRoute.fallbackReason ?? `Native backend '${backendRoute.backend}' is unavailable for this request.`);
      }
      return this._fallbackRuntime.createSession({
        recommendation,
        modelArtifact,
        profile,
        config,
        logger,
        enginePath,
      });
    }

    try {
      await this.ensureModelLoaded({
        recommendation,
        modelArtifact,
        profile,
        config,
      });
    } catch (error) {
      if (!allowOracleFallback) {
        throw error;
      }
      return this._fallbackRuntime.createSession({
        recommendation,
        modelArtifact,
        profile,
        config,
        logger,
        enginePath,
      });
    }

    let activeKvSlotId = null;
    let activeRequestId = null;
    let activeWorkerSessionId = null;

    return new ModelSession({
      modelId: modelArtifact.modelId,
      engine: this.engine,
      onGenerate: async ({ request, onToken }) => {
        const traceStartedAt = performance.now();
        activeRequestId = request.requestId;
        activeWorkerSessionId = `native-${request.requestId}`;
        const tokenizerGguf = resolveTokenizerGguf({ profile, modelArtifact });
        traceStep(request.requestId, "tokenizer-gguf-resolved", traceStartedAt);
        const promptArtifact = request.metadata?.rawPrompt === true
          ? buildRawPromptArtifact({
            prompt: request.prompt,
            promptTokenIds: request.promptTokenIds,
            modelPath: modelArtifact.modelPath,
            gguf: tokenizerGguf,
          })
          : buildPromptArtifact({
            prompt: request.prompt,
            systemPrompt: request.systemPrompt,
            messages: request.messages,
            modelPath: modelArtifact.modelPath,
            gguf: tokenizerGguf,
            // Chat asks for instruction structure on models whose GGUF carries no
            // chat template (base models); benchmarks keep the plain form.
            chatMode: profile?.chatMode === true,
          });
        if (modelArtifact?.runtimePrototype?.format
          && (!Array.isArray(promptArtifact.tokenIds) || promptArtifact.tokenCountMode === "estimate")) {
          promptArtifact.tokenCount = String(promptArtifact.renderedPrompt ?? "").trim().split(/\s+/).filter(Boolean).length;
          promptArtifact.tokenCountMode = "prototype-exact";
        }
        traceStep(request.requestId, "prompt-artifact-built", traceStartedAt);
        if (request.messages
          && ["unsupported_jinja", "unresolved_placeholders", "unsupported_template_multiturn"]
            .includes(promptArtifact.templateRenderer)) {
          const templateError = new Error("The loaded chat template cannot render this ordered conversation safely.");
          templateError.code = "CHAT_TEMPLATE_UNSUPPORTED";
          throw templateError;
        }
        if (isGgufNativeModel(modelArtifact)
          && (!Array.isArray(promptArtifact.tokenIds) || promptArtifact.tokenIds.length === 0)) {
          const tokenizerError = new Error(
            "The native worker could not produce exact prompt token IDs for this GGUF. "
            + "The tokenizer vocabulary was unavailable, so prefill cannot run. "
            + `(tokenizer=${promptArtifact.tokenizer ?? "unknown"}, strategy=${promptArtifact.tokenizerStrategy ?? "unknown"})`,
          );
          tokenizerError.code = "NATIVE_TOKENIZER_UNAVAILABLE";
          throw tokenizerError;
        }
        const kvPlan = buildKvCachePlan({
          modelId: modelArtifact.modelId,
          profile,
          promptArtifact,
          gguf: profile?.gguf ?? modelArtifact?.gguf ?? null,
        });
        if (kvPlan.fits === false) {
          // Say what would fit, not just that this does not. The caller can act
          // on a token count (shorten the turn, lower the context); it cannot
          // act on two raw byte totals.
          const gib = (bytes) => `${(bytes / (1024 ** 3)).toFixed(2)} GB`;
          const error = new Error(
            `This turn needs ${gib(kvPlan.estimatedBytes)} of KV cache for ${kvPlan.effectiveTokens} tokens, `
            + `but only ${gib(kvPlan.totalBudgetBytes)} is budgeted on this machine`
            + `${kvPlan.maxTokensForBudget ? ` (about ${kvPlan.maxTokensForBudget} tokens)` : ""}. `
            + "Shorten the conversation or reduce the context size.",
          );
          error.code = "KV_BUDGET_EXCEEDED";
          error.kvPlan = {
            estimatedBytes: kvPlan.estimatedBytes,
            totalBudgetBytes: kvPlan.totalBudgetBytes,
            effectiveTokens: kvPlan.effectiveTokens,
            maxTokensForBudget: kvPlan.maxTokensForBudget ?? null,
          };
          throw error;
        }
        const kvCache = this.kvCacheForModel({
          modelId: modelArtifact.modelId,
          profile,
          gguf: profile?.gguf ?? modelArtifact?.gguf ?? null,
        });
        const kvSlot = kvCache.slots.assignSlot({
          prefixKey: kvPlan.prefixKey ?? promptArtifact.renderedPromptHash,
          tokenCount: kvPlan.effectiveTokens ?? promptArtifact.tokenCount ?? 0,
          bytes: kvPlan.estimatedBytes ?? 0,
          metadata: {
            bytesPerToken: kvPlan.bytesPerToken,
            promptHash: promptArtifact.renderedPromptHash,
            templateMode: promptArtifact.templateMode,
          },
        });
        activeKvSlotId = kvSlot.slotId;
        traceStep(request.requestId, "kv-plan-and-slot", traceStartedAt);

        const priority = Number.isFinite(Number(request.metadata?.priority))
          ? Number(request.metadata.priority)
          : 0;
        const timeoutMs = Number.isFinite(Number(request.metadata?.timeoutMs))
          ? Number(request.metadata.timeoutMs)
          : profile?.timeoutMs ?? null;
        const chunkSize = Number.isFinite(Number(request.metadata?.chunkSize))
          ? Number(request.metadata.chunkSize)
          : 24;
        const chunkDelayMs = Number.isFinite(Number(request.metadata?.chunkDelayMs))
          ? Number(request.metadata.chunkDelayMs)
          : 0;

        try {
          const result = await this._scheduler.enqueue({
            requestId: request.requestId,
            priority,
            timeoutMs,
            chunkSize,
            chunkDelayMs,
            task: async ({ isCancelled }) => {
              let workerClient = null;
              if (isCancelled()) {
                throw cancelledError(request.requestId);
              }
              try {
                traceStep(request.requestId, "scheduler-dequeued", traceStartedAt);
                const worker = await this.ensureModelLoaded({
                  recommendation,
                  modelArtifact,
                  profile,
                  config,
                });
                traceStep(request.requestId, "model-load-ensured", traceStartedAt);
                workerClient = worker.client;
                const effectiveExecutionPlan = applyHardwareTuningHint(
                  executionPlan,
                  worker.loadResult?.hardwareTuning,
                );
                await workerClient.openSession({
                  sessionId: activeWorkerSessionId,
                  modelId: modelArtifact.modelId,
                });
                traceStep(request.requestId, "worker-session-opened", traceStartedAt);
                const requestStopSequences = [...new Set([
                  ...(Array.isArray(profile?.stopSequences) ? profile.stopSequences : []),
                  ...(Array.isArray(request.stopSequences) ? request.stopSequences : []),
                ])];
                const prefillParams = {
                  sessionId: activeWorkerSessionId,
                  modelId: modelArtifact.modelId,
                  prompt: request.prompt,
                  promptArtifact,
                  kvPlan,
                  profile,
                  executionPlan: effectiveExecutionPlan,
                  maxTokens: request.maxTokens,
                  // Configured on the session here (handle_prefill), where
                  // max_tokens is also set, and read by the decode loop.
                  stopSequences: requestStopSequences,
                  // Caterpillar only: allows its resident KV prompt cache. Off
                  // for benchmarks so their prefill cost stays real.
                  promptCache: profile?.promptCache === true,
                };
                const supportsSnapshotClone =
                  worker.handshake?.methods?.includes?.("session_clone") === true
                  // Caterpillar's GpuModel currently owns one device-global KV
                  // allocation. Its CPU sessions are deeply cloneable; GPU
                  // counters are not. Keep the cache off for that execution
                  // class until device KV becomes per-session.
                  && (this.engine !== "caterpillar"
                    || worker.loadResult?.nativeBackend === "cpu");
                const prefill = profile?.nativeRuntime?.prefillSnapshotCache === true
                  && request.metadata?.cachePrompt !== false
                  && supportsSnapshotClone
                  ? await prefillWithSnapshotCache({
                    worker,
                    workerClient,
                    targetSessionId: activeWorkerSessionId,
                    promptArtifact,
                    prefillParams,
                    maxEntries: profile?.nativeRuntime?.prefillSnapshotCacheEntries,
                  })
                  : await workerClient.prefill(prefillParams);
                traceStep(request.requestId, "prefill-complete", traceStartedAt);
                const ggufDetokenizationAvailable = isGgufNativeModel(modelArtifact);
                const incrementalDecoder = ggufDetokenizationAvailable
                  ? createIncrementalTokenDecoder({
                    modelPath: modelArtifact.modelPath,
                    gguf: tokenizerGguf,
                  })
                  : null;
                const streamedChunks = [];
                const decodedTokenSpans = [];
                let decodedCharacterCount = 0;
                let streamDecodeMs = 0;
                const decodeLiveToken = (event) => {
                  const rawToken = typeof event?.token === "string" ? event.token : "";
                  if (!incrementalDecoder || !Number.isFinite(Number(event?.tokenId))) {
                    return rawToken;
                  }
                  const startedAt = performance.now();
                  try {
                    return incrementalDecoder.decodeToken(Number(event.tokenId));
                  } catch {
                    return rawToken;
                  } finally {
                    streamDecodeMs += performance.now() - startedAt;
                  }
                };
                // Always request worker token events. This produces the final
                // output incrementally and avoids rebuilding/decoding the full
                // generated sequence after the worker completes.
                let firstTokenTraced = false;
                // The worker only stops on the GGUF's base EOS, which is the
                // wrong token for an instruct model (see stop-sequences.js).
                // Enforce the real end of turn here so nothing past it is ever
                // streamed or stored.
                const stopFilter = createStopSequenceFilter({
                  markers: [...new Set([...DEFAULT_STOP_MARKERS, ...requestStopSequences])],
                });
                const liveTokenListener = (event) => {
                  if (!firstTokenTraced) {
                    firstTokenTraced = true;
                    traceStep(request.requestId, "first-token", traceStartedAt);
                  }
                  if (stopFilter.stopped) {
                    return;
                  }
                  const decodedText = decodeLiveToken(event);
                  const tokenId = Number(event?.tokenId);
                  const spanStart = decodedCharacterCount;
                  decodedCharacterCount += decodedText.length;
                  if (Number.isSafeInteger(tokenId) && decodedText.length > 0) {
                    decodedTokenSpans.push({
                      tokenId,
                      start: spanStart,
                      end: decodedCharacterCount,
                    });
                  }
                  const safeText = stopFilter.push(decodedText);
                  if (!safeText) {
                    return;
                  }
                  streamedChunks.push(safeText);
                  if (typeof onToken === "function") {
                    onToken({ text: safeText, tokenId: Number.isSafeInteger(tokenId) ? tokenId : null });
                  }
                };
                if (serverTraceEnabled) {
                  // Deliberately does NOT stringify `profile`. That object
                  // carries the full GGUF metadata including the vocabulary
                  // (~43 MB serialized), so measuring its size cost ~300-500 ms
                  // and ~86 MB of garbage PER REQUEST - which then surfaced as a
                  // 3.7 s "TTFT spike" on the first request after an idle
                  // period, when the major GC finally ran. The spike did not
                  // exist without the trace. A diagnostic that changes the
                  // number it reports is worse than no diagnostic.
                  //
                  // The profile is stripped before it reaches the worker anyway
                  // (see process-client.request), so its JS-side size was never
                  // the wire cost it looked like.
                  traceStep(
                    request.requestId,
                    `payload-sizes kvPlan=${JSON.stringify(kvPlan ?? {}).length}b `
                    + `promptArtifact=${JSON.stringify(promptArtifact ?? {}).length}b`,
                    traceStartedAt);
                }
                traceStep(request.requestId, "generate-sent", traceStartedAt);
                // EnhancementRun_04 Phase 4, routed adaptively.
                //
                // Three things have to agree before a request is batched.
                //
                // 1. The profile asks for it. On by default for serve and chat.
                // 2. The worker can do it. `decode_batch` exists in the compiled
                //    runner only; prototype models stay on the JS worker, and
                //    asking it for a method it lacks does not degrade -- it kills
                //    the server mid-request (ECONNRESET on the serve path).
                // 3. There is something to batch WITH. This is the adaptive part
                //    and it is why chat can share the setting safely: batching a
                //    single sequence buys nothing and costs one RPC per token
                //    instead of one RPC for the whole completion. A lone request
                //    therefore takes the single-shot `generate` path, and only a
                //    request that arrives while the worker is already busy joins
                //    the coordinator, where the shared weight read is what pays.
                //
                // So one person chatting gets the fast path, and a loaded server
                // gets the batched one, from the same setting.
                const supportsBatchedDecode = worker.handshake?.methods?.includes?.("decode_batch") === true;
                const alreadyBusy = (worker._inFlightGenerations ?? 0) > 0
                  || (worker._batchCoordinator?.pendingCount?.() ?? 0) > 0;
                const admittedHttpConcurrency = typeof request.metadata?.getAdmittedHttpConcurrency === "function"
                  ? request.metadata.getAdmittedHttpConcurrency()
                  : null;
                const batchedDecode = shouldUseBatchedDecode({
                  enabled: profile?.nativeRuntime?.batchedDecode === true,
                  compiledWorker: prefersCompiledNativeWorker(modelArtifact),
                  supported: supportsBatchedDecode,
                  alreadyBusy,
                  alwaysBatch: profile?.nativeRuntime?.alwaysBatch === true,
                  admittedHttpConcurrency,
                  temperature: request.temperature,
                });
                worker._inFlightGenerations = (worker._inFlightGenerations ?? 0) + 1;
                let nativeResult;
                try {
                  // The counter has to come back down on the failure path too,
                  // or one thrown request leaves the worker looking permanently
                  // busy and every later single request pays for a batch of one.
                  nativeResult = batchedDecode
                  ? await generateViaBatchCoordinator({
                    worker,
                    workerClient,
                    sessionId: activeWorkerSessionId,
                    prefill,
                    onToken: liveTokenListener,
                    batchFloor: profile?.nativeRuntime?.batchFloor,
                    maxBatchSize: profile?.nativeRuntime?.batchTargetSize,
                  })
                  : await workerClient.generate({
                  sessionId: activeWorkerSessionId,
                  prompt: request.prompt,
                  promptArtifact,
                  kvPlan,
                  profile,
                  maxTokens: request.maxTokens,
                  prefillMs: prefill.prefillMs,
                  // Stop strings for the worker's decode loop (turn-enders,
                  // instruction headers). Without these a model that does not
                  // emit the GGUF EOS - notably a base model - decodes the whole
                  // budget after its answer is finished.
                  stopSequences: requestStopSequences,
                  sampling: {
                    temperature: request.temperature,
                    topK: request.topK,
                    topP: request.topP,
                    seed: request.seed,
                  },
                }, liveTokenListener);
                } finally {
                  worker._inFlightGenerations = Math.max(0, (worker._inFlightGenerations ?? 1) - 1);
                }
                // Release whatever the stop filter was still holding back, so a
                // clean answer does not lose its last few characters.
                const tail = stopFilter.finish();
                if (tail) {
                  streamedChunks.push(tail);
                  if (typeof onToken === "function") {
                    onToken(tail);
                  }
                }
                // `generatedTokenIds` from the worker are every sampled token,
                // including EOS and the tokens spelling a hidden stop marker.
                // OpenAI usage is defined from the completion actually emitted
                // after filtering. Map the filter's exact visible character
                // boundary back to the decoded token spans; a token contributes
                // when at least one of its decoded characters is visible.
                const emittedCharacterCount = stopFilter.text.length;
                const unfilteredNativeOutput = String(nativeResult.output ?? "");
                const nativeOutputStop = truncateAtStopSequence(unfilteredNativeOutput);
                const emittedTokenIds = decodedTokenSpans.length > 0
                  ? decodedTokenSpans
                    .filter((span) => span.start < emittedCharacterCount && span.end > 0)
                    .map((span) => span.tokenId)
                  : nativeResult.finishReason === "max-tokens"
                    && nativeOutputStop.text === unfilteredNativeOutput
                    && Array.isArray(nativeResult.generatedTokenIds)
                    ? [...nativeResult.generatedTokenIds]
                    : null;
                const ggufDetokenizationEligible = isGgufNativeModel(modelArtifact);
                // Once the filter has cut a turn short, the streamed text IS the
                // answer. The chunk-count equality below can no longer hold (we
                // deliberately dropped tokens), and without this the fallback
                // would re-introduce the full runaway sequence from the worker.
                const streamedOutputComplete = stopFilter.stopped
                  || (ggufDetokenizationEligible
                    && streamedChunks.length === nativeResult.generatedTokenIds?.length);
                const detokenized = !streamedOutputComplete
                  && ggufDetokenizationEligible
                  && Array.isArray(nativeResult.generatedTokenIds)
                  && nativeResult.generatedTokenIds.length > 0
                  ? decodeGeneratedTokenIds({
                    tokenIds: nativeResult.generatedTokenIds,
                    modelPath: modelArtifact.modelPath,
                    gguf: tokenizerGguf,
                  })
                  : null;
                return {
                  status: "completed",
                  runtime: ({ caterpillar: "caterpillar-cpu", ostrich: "ostrich-cpu" })[this.engine] ?? "lizard-native-cpu",
                  durationMs: (nativeResult.prefillMs ?? 0) + (nativeResult.decodeMs ?? 0),
                  // Defensive truncation on the non-streamed path too: a model
                  // whose end-of-turn token never reached the live listener
                  // must still not deliver the continuation after it.
                  output: streamedOutputComplete
                    ? streamedChunks.join("")
                    : truncateAtStopSequence(detokenized?.text ?? nativeResult.output).text,
                  chunks: streamedOutputComplete
                    ? streamedChunks
                    : (detokenized?.chunks ?? nativeResult.chunks),
                  nativeExecution: {
                    mode: "worker-rpc",
                    backend: nativeResult.nativeBackend ?? "cpu",
                    deterministic: Number(request.temperature ?? 0) <= 0
                      || Number.isFinite(Number(request.seed)),
                    sampling: {
                      requested: {
                        temperature: request.temperature,
                        topK: request.topK,
                        topP: request.topP,
                        seed: request.seed,
                      },
                      effective: nativeResult.sampling ?? (nativeResult.batched === true
                        ? {
                          temperature: 0,
                          topK: 1,
                          topP: 1,
                          seed: request.seed,
                        }
                        : null),
                      mode: nativeResult.batched === true ? "greedy-batch" : "worker-sampler",
                    },
                    generatedTokenIds: nativeResult.generatedTokenIds ?? [],
                    emittedTokenIds,
                    finishReason: stopFilter.stopped
                      ? "stop-sequence"
                      : nativeResult.finishReason ?? "unknown",
                    stopReason: stopFilter.reason ?? null,
                    cachedPromptTokens: resolveCachedPromptTokens(prefill),
                    compatibility: nativeResult.compatibility ?? null,
                    sessionMetrics: nativeResult.sessionMetrics ?? prefill.sessionMetrics ?? null,
                    backendRoute,
                    backendRegistry,
                    executionPlan: effectiveExecutionPlan,
                    hardwareTuning: worker.loadResult?.hardwareTuning ?? null,
                    cpuBuildProfile: worker.loadResult?.cpuBuildProfile ?? null,
                    prefillSnapshotHit: prefill.prefillSnapshotHit === true,
                    prefillSnapshotKey: prefill.prefillSnapshotKey ?? null,
                    batched: nativeResult.batched === true,
                    servingPath: nativeResult.batched === true ? "batched" : "single-shot",
                    concurrencyLevel: nativeResult.batched === true
                      ? Math.max(1, Number(nativeResult.sessionMetrics?.maxBatchSize) || 1)
                      : 1,
                    batchTargetSize: profile?.nativeRuntime?.batchTargetSize ?? null,
                    concurrencyPolicy: profile?.nativeRuntime?.concurrencyPolicy ?? null,
                    streamDecodeMs,
                  },
                  contract: mergeRuntimeContract({}, {
                    runtimeProvider: this.engine,
                    nativeBackend: nativeResult.nativeBackend ?? "cpu",
                    nativeModelStatus: nativeResult.nativeModelStatus ?? "loaded",
                    loadMs: nativeResult.loadMs,
                    prefillMs: nativeResult.prefillMs,
                    decodeMs: nativeResult.decodeMs,
                    tokensPerSecond: nativeResult.tokensPerSecond,
                    kvCacheBytes: nativeResult.kvCacheBytes ?? kvPlan.estimatedBytes,
                    fallbackReason: null,
                  }),
                };
              } catch (error) {
                const enrichedMessage = compatibilityFailureMessage(error.details);
                if (enrichedMessage && !allowOracleFallback) {
                  const enriched = new Error(enrichedMessage);
                  enriched.code = error.code;
                  throw enriched;
                }
                if (!allowOracleFallback) {
                  throw error;
                }
                const fallbackSession = await this._fallbackRuntime.createSession({
                  recommendation,
                  modelArtifact,
                  profile,
                  config,
                  logger,
                  enginePath,
                });
                try {
                  const fallbackResult = await collectFallbackResult(fallbackSession, request);
                  return {
                    status: "completed",
                    runtime: fallbackResult.runtime,
                    durationMs: fallbackResult.durationMs,
                    output: fallbackResult.output,
                    chunks: fallbackResult.chunks,
                    scheduler: fallbackResult.metadata?.scheduler ?? null,
                    contract: mergeRuntimeContract({}, {
                      runtimeProvider: fallbackResult.runtime?.toLowerCase?.().includes("ollama")
                        ? "Ollama"
                        : "llama.cpp",
                      nativeBackend: "cpu",
                      nativeModelStatus: "fallback-oracle",
                      fallbackReason: error.message,
                    }),
                    fallbackMetadata: fallbackResult.metadata ?? {},
                  };
                } finally {
                  await fallbackSession.close();
                }
              } finally {
                if (workerClient && activeWorkerSessionId && typeof workerClient.closeSession === "function") {
                  await workerClient.closeSession({
                    sessionId: activeWorkerSessionId,
                  }).catch(() => {});
                }
              }
            },
          });
          kvCache.slots.completeSlot(kvSlot.slotId);
          activeKvSlotId = null;
          return {
            output: result.output,
            durationMs: result.durationMs,
            runtime: result.runtime,
            chunks: result.chunks,
            metadata: {
              ...result.contract,
              lifecycleState: snapshotLifecycleState(config),
              status: result.status,
              promptArtifact,
              kvCache: {
                ...kvPlan,
                slotId: kvSlot.slotId,
                reused: kvSlot.reused,
                activeSlots: kvCache.slots.snapshot(),
                simulated: result.nativeExecution?.mode === "worker-rpc"
                  ? !isGgufNativeModel(modelArtifact)
                  : true,
                workerSessionMetrics: result.nativeExecution?.sessionMetrics ?? null,
                nativeSessionBytes: result.nativeExecution?.sessionMetrics?.nativeSessionBytes ?? null,
                workerKvCacheBytes: result.nativeExecution?.sessionMetrics?.kvCacheBytes ?? result.contract.kvCacheBytes ?? null,
              },
              scheduler: result.scheduler ?? null,
              nativeExecution: result.nativeExecution ?? null,
              fallbackMetadata: result.fallbackMetadata ?? null,
            },
          };
        } catch (error) {
          kvCache.slots.completeSlot(kvSlot.slotId);
          activeKvSlotId = null;
          throw error;
        } finally {
          activeRequestId = null;
          activeWorkerSessionId = null;
        }
      },
      onClose: async () => {
        if (activeRequestId) {
          this._scheduler.cancel(activeRequestId);
        }
        if (activeWorkerSessionId) {
          try {
            const worker = await this.ensureModelLoaded({
              recommendation,
              modelArtifact,
              profile,
              config,
            });
            await worker.client.cancel({
              sessionId: activeWorkerSessionId,
            });
            if (typeof worker.client.closeSession === "function") {
              await worker.client.closeSession({
                sessionId: activeWorkerSessionId,
              });
            }
          } catch {
            // Ignore close-time worker errors.
          }
        }
        if (!activeKvSlotId) {
          if (!keepWorkerLoaded) {
            await this.releaseModelWorker(modelArtifact, profile);
          }
          return;
        }
        const kvCache = this._kvCaches.get(String(modelArtifact.modelId));
        kvCache?.slots.completeSlot(activeKvSlotId);
        if (!keepWorkerLoaded) {
          await this.releaseModelWorker(modelArtifact, profile);
        }
      },
    });
  }
}
