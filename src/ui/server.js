// Local onboarding UI server for `lizard-llm serve-ui`.
//
// Serves a single-page dark-mode onboarding flow (system scan -> model selector
// -> download progress -> benchmark -> activate) and a small JSON/SSE API that
// reuses the existing scan, recommendation, Hugging Face, benchmark and
// activation modules. Bound to localhost by default â€” this is an operator tool.

import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { spawn } from "node:child_process";
import { resolveAppPath } from "../lib/app-root.js";
import { scanHardware } from "../lib/hardware.js";
import { buildValidatedCapabilityRegistry } from "../lib/backend-capability-registry.js";
import { resolveExecutionContext } from "../lib/execution-profile.js";
import { generateWithRuntimeSession, createInferenceRuntime, resolveModelRuntimeEngine } from "../runtime/index.js";
import { loadModelRegistry } from "../lib/model-registry.js";
import { searchCompatibleGgufModels, SIZE_BUCKETS, classifyCompatibility } from "../lib/huggingface.js";
import { searchHubModels, getHubModelDetail } from "../lib/hf-hub.js";
import { MODEL_TYPES, MODEL_FORMATS } from "../lib/model-taxonomy.js";
import { isAllowlistedRepo } from "../lib/hf-model-catalog.js";
import { checkHubModelNativeReady, createNativeCompatCache, mapWithConcurrency } from "../lib/hf-native-compat.js";
import { getQuantizationCompatibility, listCompatibilityMatrix } from "../runtime/compatibility/index.js";
import { listLocalCatalogModels, listAvailableModels, importOllamaModel } from "../lib/model-sources.js";
import { runHfDownloadWorkflow } from "../lib/hf-workflow.js";
import { activateModel, deactivateModel, getActiveModel, findCatalogModel, deleteModel } from "../lib/download-manager.js";
import { freeModelMemory } from "../lib/model-memory.js";
import { getCreditState, assertProPlanFeatureAllowed, PRO_OFFER, EDITION } from "../lib/edition.js";
import {
  armLocalEndpoint,
  describeLocalEndpoint,
  disarmLocalEndpoint,
  ENDPOINT_STOP_REASONS,
  getLocalEndpointState,
  probeEndpointPort,
  resolveEndpointAddress,
  suspendLocalEndpointForBenchmark,
} from "../lib/local-endpoint.js";
import { loadCheckNote, runTwoTokenLoadCheck } from "../lib/model-load-check.js";
import { resolveHelperExecutable, resolveServerExecutable } from "../lib/engine-adapter.js";
import { readOllamaManifest, listInstalledOllamaModels } from "../lib/ollama.js";
import {
  automaticModelLoadSuspended,
  getCurrentLoadedModel,
  getLastStoppedRuntimeModel,
  recoverStaleModelResidencyClaim,
  setModelResidencyIntent,
  stopCurrentModel,
} from "../lib/model-lifecycle.js";
import {
  DEFAULT_MODEL_RUNTIME_PROVIDER,
  listRuntimeProvidersForUser,
  normalizeRuntimeProvider,
  resolveRequestedRuntimeProvider,
  RuntimeProviderAccessError,
} from "../lib/runtime-providers.js";
import { createLoopbackGuard } from "../lib/loopback-guard.js";
import { buildQuickChatExecutionTelemetry } from "../lib/native-execution-telemetry.js";
import { appendNativeTurnMetric } from "../lib/native-turn-history.js";
import { resolveNativeConcurrencyPolicy } from "../runtime/concurrency-policy.js";
import { findNativeHttpPerformance } from "../lib/native-http-performance.js";
import {
  CATERPILLAR_TTFT_PROFILE,
  applyCaterpillarPerformanceProfile,
  caterpillarPerformanceProfileFromOptions,
} from "../runtime/caterpillar-performance-profile.js";
import {
  classifyMajorFailure,
  incidentUserFields,
  listMajorFailureIncidents,
  readMajorFailureIncident,
  reportMajorFailure,
} from "../lib/major-failure-intelligence.js";
import { handleChatApi, selectChatEngineForActivatedModel } from "./chat-routes.js";
import { invalidateEngineProbeCache } from "../lib/chat/engine-runtime.js";
import { disconnectAllPlugins } from "../lib/chat/mcp-manager.js";
import { randomUUID } from "node:crypto";

// Development bypasses (direct credit grant/upgrade, simulated checkout) are
// permitted only outside a production-authorization build. Turning on remote
// entitlements (authorization.remoteEntitlements) â€” or setting
// LIZARD_RELEASE_BUILD=1 â€” removes them, so a release build cannot mutate
// plan/credit state locally (auth_phases.md Phase 6 task 9, Â§9 DoD).
function devBypassesAllowed(config) {
  if (process.env.LIZARD_RELEASE_BUILD === "1") {
    return false;
  }
  return !config?.authorization?.remoteEntitlements;
}

// Caller identity for experimental-runner access checks (Ostrich, 2026-08-04).
//
// This dashboard server is single-operator: whoever reaches it already controls
// the machine and can run any binary on it, so gating Ostrich from them would
// be theatre rather than security. The operator is therefore treated as admin.
//
// Real multi-user enforcement belongs to the hosted control plane, which
// forwards an identity. When a request arrives through that path we honour the
// forwarded claims instead of assuming admin, so a proxied non-admin user is
// correctly refused. Header names follow the BFF's existing lowercase
// convention; absent claims fail closed to a non-admin user.
function requestUser(req) {
  const forwarded = req?.headers?.["x-lizard-forwarded-user"];
  if (!forwarded) {
    return { isAdmin: true, local: true };
  }
  const admin = String(req.headers["x-lizard-forwarded-admin"] ?? "").toLowerCase() === "true";
  let flags = {};
  try {
    flags = JSON.parse(req.headers["x-lizard-forwarded-flags"] ?? "{}") ?? {};
  } catch {
    flags = {};
  }
  return { id: String(forwarded), isAdmin: admin, flags, local: false };
}

// Native-readiness verdicts for Hub repos, cached on disk per repo revision
// and deduplicated while a check is in flight. The verdict logic itself lives
// in src/lib/hf-native-compat.js.
let nativeCompatCache = null;
const nativeCompatInFlight = new Map();

async function resolveNativeCompat({ config, model, token, ggufFiles = null }) {
  if (!nativeCompatCache) {
    const artifactDirectory = config?.benchmark?.ci?.artifactDirectory;
    const cacheDirectory = artifactDirectory ? path.dirname(artifactDirectory) : path.join(process.cwd(), ".lizard-llm");
    nativeCompatCache = createNativeCompatCache(path.join(cacheDirectory, "hf-native-compat-cache.json"));
  }
  const repo = model.repo;
  const lastModified = model.lastModified ?? null;
  const cached = nativeCompatCache.get(repo, lastModified);
  if (cached) {
    return cached;
  }
  if (nativeCompatInFlight.has(repo)) {
    return nativeCompatInFlight.get(repo);
  }
  const pending = (async () => {
    try {
      const files = ggufFiles ?? (await getHubModelDetail(repo, { token })).ggufFiles;
      const verdict = await checkHubModelNativeReady({ repo, ggufFiles: files, token });
      nativeCompatCache.set(repo, lastModified, verdict);
      return verdict;
    } catch (error) {
      // Unverifiable repos are reported (and filtered) as not ready rather
      // than failing the whole search page.
      return { ready: false, reason: `Native readiness could not be verified: ${error.message}` };
    } finally {
      nativeCompatInFlight.delete(repo);
    }
  })();
  nativeCompatInFlight.set(repo, pending);
  return pending;
}

// Each Lizard-owned engine is a separate binary on its own port, so an engine
// missing from these tables silently answers on another engine's port - which
// is how "serve --provider ostrich" once talked to Caterpillar.
const UI_ENGINE_KEY_BY_PROVIDER = { caterpillar: "caterpillar" };
// The port half of that pairing now lives in local-endpoint.js, which is what
// actually binds it - see ENDPOINT_PORT_BY_PROVIDER. One table, one owner.

// The address of a provider's local API endpoint. Delegates to the endpoint
// supervisor so the URL the UI advertises and the port the supervisor actually
// binds can never drift apart - they used to be two independent copies of the
// same table.
function serverEndpoint(config, runtimeProvider = "lizard-native") {
  const address = resolveEndpointAddress(config, runtimeProvider);
  return { ...address, proFeature: true };
}

const uiDirectory = path.dirname(fileURLToPath(import.meta.url));


// The hardware scan shells out to PowerShell and takes seconds. The onboarding
// asks for it on every step, so keep one result for a minute; "Scan again"
// passes fresh=1. Model loads still scan fresh (they need live free memory).
const HARDWARE_CACHE_MS = 60_000;
let hardwareCache = null;
function cachedHardware({ config, logger, fresh = false }) {
  if (!fresh && hardwareCache && Date.now() - hardwareCache.at < HARDWARE_CACHE_MS) {
    return hardwareCache.hardware;
  }
  const hardware = scanHardware({ config, logger });
  hardwareCache = { at: Date.now(), hardware };
  return hardware;
}
const STATIC_FILES = {
  "/": { file: "index.html", type: "text/html; charset=utf-8" },
  "/index.html": { file: "index.html", type: "text/html; charset=utf-8" },
  "/app.js": { file: "app.js", type: "text/javascript; charset=utf-8" },
  "/chat.js": { file: "chat.js", type: "text/javascript; charset=utf-8" },
  "/chat-charts.js": { file: "chat-charts.js", type: "text/javascript; charset=utf-8" },
  "/styles.css": { file: "styles.css", type: "text/css; charset=utf-8" },
  "/chat.css": { file: "chat.css", type: "text/css; charset=utf-8" },
};

const ASSET_TYPES = {
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".webp": "image/webp",
  ".gif": "image/gif",
  ".mp4": "video/mp4",
  ".webm": "video/webm",
};

// Serve images/video from src/ui/assets (including the screenshots/ and video/
// subfolders), with path-traversal protection.
function serveAsset(res, pathname) {
  const rel = decodeURIComponent(pathname.replace(/^\/assets\//, ""));
  const ext = path.extname(rel).toLowerCase();
  const type = ASSET_TYPES[ext];
  if (!type || !/^[a-z0-9/._-]+$/i.test(rel) || rel.includes("..")) {
    sendJson(res, 404, { error: "Not found" });
    return;
  }
  const assetsRoot = path.resolve(uiDirectory, "assets");
  const filePath = path.resolve(assetsRoot, rel);
  if (!filePath.startsWith(assetsRoot)) {
    sendJson(res, 404, { error: "Not found" });
    return;
  }
  fs.readFile(filePath, (error, data) => {
    if (error) {
      sendJson(res, 404, { error: "Not found" });
      return;
    }
    res.writeHead(200, { "Content-Type": type, "Cache-Control": "no-store", "Accept-Ranges": "bytes" });
    res.end(data);
  });
}

function sendJson(res, statusCode, payload) {
  // Defense in depth: callers sometimes forward `error.status` from a thrown
  // error whose `.status` isn't actually an HTTP status at all - e.g. Node's
  // child_process exec*Sync attaches `.status` = the CHILD PROCESS's exit
  // code (often 1) to failures, and a generic `catch (error) { sendJson(res,
  // error.status ?? 400, ...) }` pattern used to forward that straight into
  // res.writeHead(), which throws its own "Invalid status code" RangeError
  // and replaces the real error message with a confusing, unrelated one.
  // Any value outside the valid HTTP status range falls back to 500.
  const safeStatus = Number.isInteger(statusCode) && statusCode >= 100 && statusCode <= 599 ? statusCode : 500;
  const body = JSON.stringify(payload);
  res.writeHead(safeStatus, {
    "Content-Type": "application/json; charset=utf-8",
    "Cache-Control": "no-store",
  });
  res.end(body);
}

function serveStatic(res, entry) {
  const filePath = path.join(uiDirectory, entry.file);
  fs.readFile(filePath, (error, data) => {
    if (error) {
      sendJson(res, 500, { error: `Failed to read ${entry.file}` });
      return;
    }
    res.writeHead(200, { "Content-Type": entry.type, "Cache-Control": "no-store" });
    res.end(data);
  });
}

// The chart library ships inside the app (it is a runtime dependency and the
// installer stages node_modules), so this reads it from the install rather than
// fetching it. Cached hard: the file is versioned with the app.
function serveD3Bundle(res) {
  const bundlePath = resolveAppPath("node_modules", "d3", "dist", "d3.min.js");
  fs.readFile(bundlePath, (error, data) => {
    if (error) {
      res.writeHead(404, { "Content-Type": "text/plain; charset=utf-8" });
      res.end("The chart library is not installed.");
      return;
    }
    res.writeHead(200, {
      "Content-Type": "text/javascript; charset=utf-8",
      "Cache-Control": "public, max-age=86400",
    });
    res.end(data);
  });
}

function readBody(req) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    req.on("data", (chunk) => chunks.push(chunk));
    req.on("end", () => {
      const raw = Buffer.concat(chunks).toString("utf8");
      if (!raw) {
        resolve({});
        return;
      }
      try {
        resolve(JSON.parse(raw));
      } catch (error) {
        reject(error);
      }
    });
    req.on("error", reject);
  });
}

function normalizeModelId(value) {
  return String(value ?? "").trim().toLowerCase();
}

function normalizeQuantizationLevel(value) {
  const match = String(value ?? "").toUpperCase().match(/Q([1-9])/);
  return match ? `Q${match[1]}` : "Q4";
}

// Shared across every /api/quick-chat and /api/warm-model call in this
// server process. A fresh NativeCpuRuntime has an empty worker cache, so
// creating one per HTTP request meant the model reloaded from disk (tens
// of seconds to minutes) on every single chat message even though the
// previous request's worker process was still alive and holding it
// resident - found 2026-07-07 via real use of the onboarding chat dialog.
// The runtime's own internal cache is keyed by modelId+modelPath
// (native-cpu-runtime.js _workerCacheKey), so reusing one instance across
// a later re-activation of a different model is safe - it just grows an
// extra cache entry, it does not serve stale weights.
//
// Keyed by engine because chat can switch between lizard-native and
// caterpillar: they are different worker binaries, so one runtime instance
// cannot serve both, and a caterpillar switch must not evict the native
// worker that is already warm.
const sharedNativeRuntimes = new Map();
function getSharedNativeRuntime(engineId = "lizard-native") {
  // One cached runtime per engine binary. Ostrich is a third binary, so it
  // needs its own entry: sharing Caterpillar's would hand it a worker running
  // the wrong runner.
  const requested = String(engineId);
  const key = UI_ENGINE_KEY_BY_PROVIDER[requested] ? requested : "lizard-native";
  if (!sharedNativeRuntimes.has(key)) {
    // Product chat may receive a B=16 burst. The per-profile concurrency
    // policy below can reduce each decode step to B=8/B=4/B=1, but the outer
    // scheduler must first admit enough requests for the primary B=16 lane.
    sharedNativeRuntimes.set(key, createInferenceRuntime({
      requestedEngine: key,
      maxConcurrentRequests: 16,
    }).runtime);
  }
  return sharedNativeRuntimes.get(key);
}

function activeNativeProvider(config) {
  // Return the provider itself, not a category. Collapsing Ostrich onto
  // "caterpillar" here would pick the Caterpillar runtime and port for a model
  // the user activated on Ostrich.
  const active = getActiveModel(config)?.runtimeProvider;
  return UI_ENGINE_KEY_BY_PROVIDER[active] ? active : "lizard-native";
}

function withAdaptiveNativeConcurrency(profile, runtimeProvider, config, modelId = null) {
  const performanceProfile = caterpillarPerformanceProfileFromOptions({
    caterpillarProfile: profile?.caterpillarPerformanceProfile,
  }, {
    defaultProfile: runtimeProvider === "caterpillar"
      ? CATERPILLAR_TTFT_PROFILE
      : "balanced",
  });
  const measuredHttpPerformance = findNativeHttpPerformance(config, {
    runtimeProvider,
    modelId: modelId ?? profile?.modelId,
    quant: profile?.gguf?.quantization ?? profile?.gguf?.quant ?? null,
    contextTokens: profile?.contextSize ?? null,
    optimizationProfile: runtimeProvider === "caterpillar" ? performanceProfile : null,
  });
  const concurrencyPolicy = resolveNativeConcurrencyPolicy({
    runtimeProvider,
    profile,
    gguf: profile?.gguf ?? null,
    measuredBatchLevels: measuredHttpPerformance?.levels ?? null,
  });
  const adaptiveProfile = {
    ...profile,
    nativeRuntime: {
      ...(profile?.nativeRuntime ?? {}),
      batchedDecode: true,
      prefillSnapshotCache: true,
      preferredBatchSize: concurrencyPolicy.preferredBatchSize,
      batchTargetSize: concurrencyPolicy.selectedBatchSize,
      batchFallbackOrder: concurrencyPolicy.fallbackOrder,
      concurrencyPolicy,
      measuredHttpPerformance,
      keepWorkerLoaded: true,
    },
  };
  return applyCaterpillarPerformanceProfile(adaptiveProfile, {
    runtimeProvider,
    performanceProfile,
  });
}

// Local OAuth/OIDC + session runtime (Phase 4 of .Agent/auth_phases.md).
// Lazily created so the server keeps starting instantly (and fully offline)
// while auth.enabled is false - no discovery request happens until a login
// is actually initiated. Shared across requests: it owns the in-memory
// access token, the one-time authorization transactions and the local
// dashboard sessions.
let sharedAuthRuntime = null;
async function reportUiMajorFailure({ error, config, logger, req = null, context = {} }) {
  const classification = classifyMajorFailure(error, context);
  if (!classification.major && !classification.crashCandidate) {
    return { reported: false, reason: classification.reason };
  }
  try {
    const session = null;
    let hardware = null;
    try {
      hardware = scanHardware({ config, logger });
    } catch {
      // The incident collector still records platform/CPU/memory without it.
    }
    const active = getActiveModel(config);
    const catalog = active?.id ? findCatalogModel(config, active.id) : null;
    const result = await reportMajorFailure({
      error,
      config,
      logger,
      hardware,
      user: session?.data ?? (
        config?.failureIntelligence?.userEmail
          ? { email: config.failureIntelligence.userEmail }
          : null
      ),
      context: {
        component: "onboarding-ui",
        runtimeProvider: active?.runtimeProvider ?? null,
        modelId: active?.id ?? null,
        architecture: catalog?.gguf?.architecture ?? null,
        quantization: catalog?.gguf?.quant ?? null,
        ...context,
      },
    });
    if (result?.reported) {
      logger?.error?.("ui-major-failure-reported", {
        ticketId: result.ticketId,
        component: context.component ?? "onboarding-ui",
      });
    }
    return result;
  } catch (reportError) {
    logger?.error?.("ui-major-failure-reporting-failed", {
      message: String(reportError?.message ?? reportError).slice(0, 200),
    });
    return { reported: false, reason: "reporting_failed" };
  }
}

let sharedRemoteAgentManager = null;
let sharedLoopbackGuard = null;
function getLoopbackGuard() {
  if (!sharedLoopbackGuard) {
    // Loopback IP-literal matching accepts any port, so the dashboard's actual
    // bind port (5567, or a test port) needs no configuration here.
    sharedLoopbackGuard = createLoopbackGuard();
  }
  return sharedLoopbackGuard;
}

const MUTATING_METHODS = new Set(["POST", "PUT", "PATCH", "DELETE"]);
const LOOPBACK_ORIGIN_HOSTS = new Set(["127.0.0.1", "localhost", "[::1]"]);

// A browser always sends Origin on a cross-site POST, so an absent Origin is a
// same-origin or non-browser client (curl, the CLI). A present Origin must be
// a loopback http origin.
function originIsSameLoopback(headers) {
  const origin = headers?.origin;
  if (!origin || origin === "null") return !origin;
  try {
    const parsed = new URL(origin);
    return parsed.protocol === "http:" && LOOPBACK_ORIGIN_HOSTS.has(parsed.hostname);
  } catch {
    return false;
  }
}

// ---- Session-scoped chat memory for the Test Chat dialog ----
// The dialog remembers only what was said while it is open: the client mints
// a fresh session id every time the dialog opens and explicitly resets it on
// close (POST /api/quick-chat/reset), so no transcript survives the dialog.
// History is kept server-side (keyed by that id) instead of round-tripping
// the whole transcript through the SSE query string on every message.
const chatSessions = new Map();
const CHAT_SESSION_LIMIT = 20;       // stale-session safety net (evict oldest)
const CHAT_HISTORY_MAX_TURNS = 12;   // 6 user/assistant exchanges
const CHAT_HISTORY_MAX_CHARS = 6000; // keeps the composed prompt bounded

function getChatSessionHistory(sessionId) {
  return (sessionId && chatSessions.get(sessionId)) || [];
}

// Reasoning-tuned models prefix their answer with a <think>...</think>
// block; only the visible answer belongs in the conversation memory.
function stripReasoningMarkup(text) {
  return String(text ?? "").replace(/<(think|thinking|reasoning)>[\s\S]*?<\/\1>/gi, "").trim();
}

function recordChatExchange(sessionId, userText, assistantText) {
  if (!sessionId || !assistantText) {
    return;
  }
  if (!chatSessions.has(sessionId) && chatSessions.size >= CHAT_SESSION_LIMIT) {
    chatSessions.delete(chatSessions.keys().next().value);
  }
  const history = chatSessions.get(sessionId) ?? [];
  history.push({ role: "user", text: userText }, { role: "assistant", text: assistantText });
  // Drop the oldest exchange first so recent context always survives.
  while (history.length > 0 && (
    history.length > CHAT_HISTORY_MAX_TURNS ||
    history.reduce((total, turn) => total + turn.text.length, 0) > CHAT_HISTORY_MAX_CHARS
  )) {
    history.splice(0, 2);
  }
  chatSessions.set(sessionId, history);
}

// The native runtime takes a single prompt string, so prior turns are folded
// into a plain-text transcript in front of the new message. This deliberately
// leaves the generation flow itself untouched - memory is purely prompt
// composition.
function composeChatPrompt(history, prompt) {
  if (history.length === 0) {
    return prompt;
  }
  const transcript = history
    .map((turn) => `${turn.role === "user" ? "User" : "Assistant"}: ${turn.text}`)
    .join("\n");
  return [
    "You are continuing an ongoing chat session. Conversation so far:",
    "",
    transcript,
    `User: ${prompt}`,
    "",
    "Reply to the user's last message, using the conversation above as context when relevant.",
  ].join("\n");
}

// Keeps the warmed model resident across page reloads / SPA navigation,
// but frees it once the onboarding page is genuinely closed (not just
// reloaded). A browser reload and a real tab-close fire the exact same
// unload events client-side, so there is no reliable way to tell them
// apart from the page's own JS - instead /api/warm-model's SSE connection
// is treated as a heartbeat: the client (warmUpModelForChat in app.js)
// keeps it open for as long as the page is alive and reopens it
// immediately on load/reload. The server only starts a grace-period timer
// when the LAST such connection drops, and cancels it if a new one arrives
// within the window (which a reload does almost instantly) - only a
// timer that survives the full grace period means the page is actually
// gone, at which point the resident worker is released.
let activeWarmConnections = 0;
let pendingReleaseTimer = null;
let warmModelArtifact = null;
// Which engine currently holds a resident model. Chat reads this so a model
// warmed by onboarding is reported as warm the moment chat opens, instead of
// looking cold and re-loading weights that are already in memory.
let warmEngineId = "lizard-native";
const WARM_MODEL_RELEASE_GRACE_MS = 20000;

function getWarmState() {
  return {
    modelId: warmModelArtifact?.modelId ?? null,
    engineId: warmModelArtifact ? warmEngineId : null,
    warm: Boolean(warmModelArtifact),
    connections: activeWarmConnections,
  };
}

// Chat records its own warm model here after an engine switch so the shared
// release path (page close, shutdown) frees it like any other.
function setWarmState({ modelArtifact, engineId }) {
  warmModelArtifact = modelArtifact ?? null;
  warmEngineId = engineId ?? "lizard-native";
  if (warmModelArtifact) {
    cancelPendingModelRelease();
  }
  // Switching chat's engine releases the outgoing engine's workers. If the
  // endpoint was armed on that engine, its worker has just gone: leaving the
  // port open would advertise a model that is no longer loaded. Fire-and-forget
  // because this is a synchronous state setter on the chat hot path - the stop
  // reason it records is what the endpoint panel reads.
  const armed = getLocalEndpointState();
  if (
    armed.status === "listening" &&
    (armed.runtimeProvider !== warmEngineId || armed.modelId !== (warmModelArtifact?.modelId ?? null))
  ) {
    disarmLocalEndpoint({ reason: ENDPOINT_STOP_REASONS.PROVIDER_SWITCH }).catch(() => {});
  }
}

function cancelPendingModelRelease() {
  if (pendingReleaseTimer) {
    clearTimeout(pendingReleaseTimer);
    pendingReleaseTimer = null;
  }
}

// Whether a model id is the one the user activated. Activation is the durable
// intent - it is written to ~/.lizard-llm/config.json and survives a reboot -
// so this is what separates "warm because a page happened to be open" from
// "resident because the user asked for it".
function isActivatedModel(config, modelId) {
  if (!config || !modelId) return false;
  try {
    return normalizeModelId(getActiveModel(config)?.id) === normalizeModelId(modelId);
  } catch {
    return false;
  }
}

function scheduleModelRelease(config = null) {
  cancelPendingModelRelease();
  const modelArtifact = warmModelArtifact;
  if (!modelArtifact) return;
  // An activated model outlives the page that warmed it. Closing onboarding is
  // not "I am done with this model" - only Unload (POST /api/deactivate) and
  // process shutdown free it, which is what lets the model be resident again
  // after a machine restart without the user re-activating anything.
  if (isActivatedModel(config, modelArtifact.modelId)) return;
  pendingReleaseTimer = setTimeout(() => {
    pendingReleaseTimer = null;
    if (activeWarmConnections > 0) return; // a connection re-arrived in the meantime
    getSharedNativeRuntime(warmEngineId).releaseModelWorker(modelArtifact).catch(() => {});
  }, WARM_MODEL_RELEASE_GRACE_MS);
}

// Frees every resident worker immediately - used on process shutdown,
// where there is no reload to wait for and no reason to keep the grace
// period.
async function releaseAllWarmModels({ endpointStopReason = ENDPOINT_STOP_REASONS.SHUTDOWN } = {}) {
  cancelPendingModelRelease();
  // The port closes before the weights do. An endpoint still accepting requests
  // against a model being released either answers with an error or re-loads
  // what is being freed.
  await disarmLocalEndpoint({ reason: endpointStopReason }).catch(() => {});
  // Chat plugins are child processes we spawned; they go down with us.
  await disconnectAllPlugins().catch(() => {});
  await Promise.all(
    [...sharedNativeRuntimes.values()].map((runtime) => runtime.releaseAllWorkers().catch(() => {})),
  );
  warmModelArtifact = null;
}

// ---- The local API endpoint (external http:// clients) ----------------------
//
// The endpoint serves the model this process already holds warm, so there is
// one copy of the weights and one residency claim. Everything it needs is
// captured once, at arm time, from the same execution context the warm load
// used - re-resolving hardware and the registry per request would put a scan on
// the critical path of every external call.

function buildLocalEndpointHandlers({ config, logger, runtimeProvider, context, profile }) {
  return {
    createCompletion: async ({ prompt, systemPrompt = null, maxTokens = 512, onDelta = null } = {}) => {
      const result = await generateWithRuntimeSession({
        requestedEngine: runtimeProvider,
        runtime: getSharedNativeRuntime(runtimeProvider),
        recommendation: context.recommendation,
        modelArtifact: context.modelArtifact,
        // timeoutMs: null for the same reason chat sets it: the scheduler's
        // 120 s default is sized for short CLI calls, and an external client
        // asking for a long answer is not a hang.
        profile: { ...profile, timeoutMs: null },
        prompt,
        systemPrompt,
        maxTokens,
        config,
        logger,
        onDelta,
      });
      return {
        output: result.output ?? "",
        modelId: result.modelId ?? context.modelArtifact.modelId,
        runtimeProvider: result.runtimeProvider ?? runtimeProvider,
        promptTokens: result.promptArtifact?.tokenCount ?? null,
        generatedTokens: result.nativeExecution?.generatedTokenIds?.length ?? null,
      };
    },
  };
}

/**
 * Ask the freshly-loaded model for two tokens, then - only if it answers - open
 * the port.
 *
 * The order is the whole point. A model that loads but cannot decode used to
 * light up "Active" and (before this) would have had a port opened onto it, so
 * the first external client discovered the failure instead of the person who
 * pressed Activate.
 */
async function verifyAndArmLocalEndpoint({ config, logger, runtimeProvider, context, profile }) {
  const handlers = buildLocalEndpointHandlers({ config, logger, runtimeProvider, context, profile });
  // Already proven and already serving. The warm stream reopens on every page
  // load, and spending a generation to re-answer a question this exact model
  // already answered would put seconds onto every reload for no new
  // information. A worker that has died since is caught by the next real
  // request, which reloads it.
  const armed = getLocalEndpointState();
  if (
    armed.status === "listening" &&
    armed.runtimeProvider === runtimeProvider &&
    armed.modelId === context.modelArtifact.modelId &&
    armed.verification?.ok
  ) {
    return { verification: armed.verification, endpoint: armed };
  }
  const verification = await runTwoTokenLoadCheck({
    generate: ({ prompt, maxTokens, onDelta }) => handlers.createCompletion({ prompt, maxTokens, onDelta }),
  });
  if (!verification.ok) {
    logger?.warn?.("model-load-check-failed", {
      modelId: context.modelArtifact.modelId,
      runtimeProvider,
      reason: verification.reason,
      message: verification.message,
      durationMs: verification.durationMs,
    });
    await disarmLocalEndpoint({
      reason: ENDPOINT_STOP_REASONS.LOAD_CHECK_FAILED,
      note: verification.note ?? loadCheckNote(verification.reason, verification.message),
      logger,
    });
    return { verification, endpoint: getLocalEndpointState() };
  }
  logger?.info?.("model-load-check-passed", {
    modelId: context.modelArtifact.modelId,
    runtimeProvider,
    durationMs: verification.durationMs,
  });

  // The Pro gate is evaluated HERE and nowhere else. Checking it per request -
  // which is what `lizard-llm serve` does - means an entitlement that lapses
  // while an integration is mid-conversation turns every subsequent call into a
  // 403 on a port that is still open, which is precisely the "it disconnects
  // when the session expires" report. Arming is the durable act, so that is
  // where the gate belongs; a lapsed entitlement is caught the next time a
  // model is loaded, which happens on every app start.
  let entitlement = null;
  try {
    const state = assertProPlanFeatureAllowed(config, "Native API server");
    entitlement = {
      checkedAt: new Date().toISOString(),
      tier: state?.tier ?? null,
      remote: Boolean(state?.remoteEntitlement),
    };
  } catch (error) {
    logger?.info?.("local-endpoint-not-entitled", { message: String(error?.message ?? error).slice(0, 200) });
    await disarmLocalEndpoint({
      reason: ENDPOINT_STOP_REASONS.NOT_ENTITLED,
      note: `${error.message} The model itself is loaded and chat works; only the external http:// endpoint is gated.`,
      logger,
    });
    return { verification, endpoint: getLocalEndpointState() };
  }

  const endpoint = await armLocalEndpoint({
    config,
    logger,
    runtimeProvider,
    modelId: context.modelArtifact.modelId,
    handlers,
    verification,
    entitlement,
  });
  return { verification, endpoint };
}

/**
 * Stop the endpoint because the model behind it is going away.
 *
 * Unload, Free RAM and shutdown all reach this. The port must close BEFORE the
 * weights are released: an endpoint that is still accepting requests against a
 * model being torn down answers with an error at best, and re-loads the model
 * the user just freed at worst.
 */
async function stopLocalEndpointFor(reason, logger = null) {
  try {
    return await disarmLocalEndpoint({ reason, logger });
  } catch (error) {
    logger?.warn?.("local-endpoint-stop-failed", { message: String(error?.message ?? error).slice(0, 200) });
    return getLocalEndpointState();
  }
}

/**
 * Drops every model this server holds resident, for callers outside it.
 *
 * The dashboard runs in the same process and mounts these routes, so its
 * "free the machine" path has to be able to release the chat/onboarding worker
 * too. Without this it stopped the tracked process tree and Ollama while the
 * in-process worker - usually the largest single consumer - stayed loaded, and
 * the freed memory came straight back on the next page poll.
 */
export async function releaseOnboardingWarmModels({ endpointStopReason = ENDPOINT_STOP_REASONS.FREE_MEMORY } = {}) {
  await releaseAllWarmModels({ endpointStopReason });
  invalidateEngineProbeCache();
}

// Everything currently holding model weights, and what a benchmark of
// `modelId` would have to do about it.
//
// Two independent things can be resident: the worker THIS server keeps warm for
// onboarding and chat, and a model started in its own process tree by a
// benchmark or `lizard-llm serve` (tracked in runtime-state.json). They were
// never reconciled, so the benchmark loaded its own copy alongside the warm one.
function describeBenchmarkModelState({ config, modelId }) {
  const wanted = modelId ? normalizeModelId(modelId) : null;
  const warm = warmModelArtifact
    ? { modelId: warmModelArtifact.modelId, engineId: warmEngineId }
    : null;
  let tracked = null;
  try {
    const record = getCurrentLoadedModel(config);
    if (record?.modelId) {
      tracked = {
        modelId: record.modelId,
        transport: record.transport ?? record.mode ?? null,
        pid: record.pid ?? null,
      };
    }
  } catch {
    // An unreadable runtime state is reported as "nothing tracked" rather than
    // failing the page that asked.
  }
  const warmMatches = Boolean(warm && wanted && normalizeModelId(warm.modelId) === wanted);
  const trackedMatches = Boolean(tracked && wanted && normalizeModelId(tracked.modelId) === wanted);
  return {
    selectedModelId: modelId ?? null,
    warm,
    tracked,
    // Already up as the model that was asked for: the run reuses it and pays no
    // load at all.
    willReuse: warmMatches || trackedMatches,
    // A different model is holding memory and has to go first.
    willRelease: [
      ...(warm && !warmMatches ? [{ scope: "warm", modelId: warm.modelId }] : []),
      ...(tracked && !trackedMatches ? [{ scope: "tracked", modelId: tracked.modelId }] : []),
    ],
  };
}

// Enforces "one model in memory" before a run starts.
//
// Releasing goes through the ordinary release path - releaseModelWorker for our
// own worker, stopCurrentModel for a tracked tree - never by killing whatever a
// residency claim points at. A claim is settled by the code that owns it; that
// rule is what keeps a crashed borrower recoverable instead of stranded.
async function ensureSingleResidentModel({ config, logger, modelId }) {
  // A benchmark takes the engine port for itself - the HTTP transport starts
  // its own server on exactly the address the endpoint is holding - and it may
  // swap the resident model out from under it. So the endpoint stands down
  // first, and it records that a benchmark is why, which is what lets the UI
  // say "load the model once more to get your endpoint back" instead of leaving
  // the user to discover a dead port from their own application's logs.
  const endpointBefore = getLocalEndpointState();
  if (endpointBefore.status === "listening") {
    await suspendLocalEndpointForBenchmark({ logger });
    logger?.info?.("local-endpoint-suspended-for-benchmark", {
      port: endpointBefore.port,
      runtimeProvider: endpointBefore.runtimeProvider,
      modelId: endpointBefore.modelId,
    });
  }
  const state = describeBenchmarkModelState({ config, modelId });
  if (state.willReuse && state.willRelease.length === 0) {
    logger?.info?.("benchmark-model-reused", { modelId });
    return { ...state, released: [], reused: true };
  }
  const released = [];
  for (const target of state.willRelease) {
    if (target.scope === "warm" && warmModelArtifact) {
      cancelPendingModelRelease();
      await getSharedNativeRuntime(warmEngineId).releaseModelWorker(warmModelArtifact).catch(() => {});
      warmModelArtifact = null;
      invalidateEngineProbeCache();
      released.push(target);
      continue;
    }
    if (target.scope === "tracked") {
      const stop = await stopCurrentModel(config).catch((error) => ({
        stopped: false,
        reason: error?.code ?? "stop_failed",
      }));
      if (stop?.stopped) {
        try {
          recoverStaleModelResidencyClaim(config, { allowLiveOwnerWithDeadTree: true });
        } catch (error) {
          logger?.warn?.("benchmark-claim-recovery-skipped", { message: error.message });
        }
        released.push(target);
      } else {
        // Reported rather than swallowed: the run is about to load a second
        // model into a machine that could not free the first.
        logger?.warn?.("benchmark-model-release-failed", {
          modelId: target.modelId,
          reason: stop?.reason ?? null,
        });
      }
    }
  }
  logger?.info?.("benchmark-model-residency", { modelId, released: released.length, reused: state.willReuse });
  return {
    ...state,
    released,
    reused: state.willReuse,
    // Surfaced so the run's own result can carry the endpoint consequence,
    // rather than the user finding out from a failing integration later.
    endpointSuspended: endpointBefore.status === "listening"
      ? {
        url: endpointBefore.url,
        runtimeProvider: endpointBefore.runtimeProvider,
        modelId: endpointBefore.modelId,
        note: "The local API endpoint was stopped for this run because a benchmark needs the same engine port. Load the model once more when the run finishes and it comes back on the same address.",
      }
      : null,
  };
}

let shutdownHandlersRegistered = false;
function registerShutdownHandlers(server) {
  if (shutdownHandlersRegistered) return;
  shutdownHandlersRegistered = true;
  const shutdown = async () => {
    try {
      await releaseAllWarmModels();
    } finally {
      server.close();
      // A lingering keep-alive SSE connection can hold the server's
      // internal socket set open indefinitely, which would otherwise
      // keep server.close()'s callback (and process exit) from ever
      // firing - force the process down instead of waiting on it.
      setTimeout(() => process.exit(0), 2000).unref();
    }
  };
  process.on("SIGINT", shutdown);
  process.on("SIGTERM", shutdown);
  // Last-resort net under the per-request guard in createUiServer. Node's
  // default for an unhandled rejection is to exit, and for a locally-hosted app
  // that exit IS the bug the user sees: the browser tab is still open, the URL
  // still looks right, and nothing answers it. A background task that fails - a
  // boot-time model resume, a release timer, a detached probe - must cost its
  // own result, never the app. Both handlers log and continue; nothing here
  // swallows a signal or a deliberate exit.
  process.on("unhandledRejection", (reason) => {
    // eslint-disable-next-line no-console
    console.error("[lizard] unhandled rejection (app kept running):", reason);
  });
  process.on("uncaughtException", (error) => {
    // eslint-disable-next-line no-console
    console.error("[lizard] uncaught exception (app kept running):", error);
  });
}

async function handleApi({ req, res, url, config, logger, dashboardIntegration }) {

  // The Lizard Chat workspace (conversations, documents, prompts, MCP plugins).
  // It shares this process's warm native runtime rather than loading its own
  // copy of the model - see getSharedNativeRuntime above.
  if (await handleChatApi({
    req,
    res,
    url,
    config,
    logger,
    getSharedRuntime: getSharedNativeRuntime,
    getWarmState,
    setWarmState,
    reportFailure: (error, context = {}) => reportUiMajorFailure({
      error,
      config,
      logger,
      req,
      context: {
        component: "chat-workspace",
        surface: "chat-workspace",
        ...context,
      },
    }),
  })) {
    return true;
  }

  if (req.method === "GET" && url.pathname === "/api/scan") {
    const hardware = cachedHardware({ config, logger, fresh: url.searchParams.get("fresh") === "1" });
    const capabilityRegistry = buildValidatedCapabilityRegistry({ hardware, config });
    sendJson(res, 200, { hardware, capabilityRegistry });
    return true;
  }

  if (req.method === "GET" && url.pathname === "/api/huggingface/status") {
    const token = process.env.HF_TOKEN ?? process.env.HUGGING_FACE_HUB_TOKEN ?? null;
    try {
      const response = await fetch(token ? "https://huggingface.co/api/whoami-v2" : "https://huggingface.co/api/models?limit=1", {
        headers: {
          Accept: "application/json",
          "User-Agent": "lizard-llm/1.0 onboarding",
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
        signal: AbortSignal.timeout(8000),
      });
      if (!response.ok) {
        throw new Error(token && response.status === 401 ? "The configured Hugging Face token was rejected." : `Hugging Face returned HTTP ${response.status}.`);
      }
      const payload = await response.json();
      sendJson(res, 200, {
        connected: true,
        authenticated: Boolean(token),
        user: token ? (payload?.name ?? payload?.fullname ?? null) : null,
        catalog: "curated-gguf",
      });
    } catch (error) {
      sendJson(res, 503, { connected: false, authenticated: false, error: error.message });
    }
    return true;
  }

  // Model wrapper Phase 1: one deduplicated model list across the curated
  // registry, the local catalog and installed Ollama models, each entry with
  // per-runtime verdicts (native / llama.cpp / ollama) computed by the same
  // compatibility code the runtimes consult.
  if (req.method === "GET" && url.pathname === "/api/models/unified") {
    try {
      const models = await listAvailableModels({
        config,
        registry: loadModelRegistry(),
        logger,
        unified: true,
      });
      sendJson(res, 200, { models });
    } catch (error) {
      sendJson(res, 500, { error: error.message });
    }
    return true;
  }

  // Model wrapper Phase 3: the compatibility matrix as data, projected from
  // the same constants the runtime consults.
  if (req.method === "GET" && url.pathname === "/api/compatibility/matrix") {
    sendJson(res, 200, listCompatibilityMatrix());
    return true;
  }

  if (req.method === "GET" && url.pathname === "/api/incidents") {
    sendJson(res, 200, {
      incidents: listMajorFailureIncidents(config, {
        limit: Number(url.searchParams.get("limit")) || 100,
      }),
    });
    return true;
  }

  if (req.method === "GET" && /^\/api\/incidents\/LIZ-\d{8}-[A-F0-9]{10}$/.test(url.pathname)) {
    const ticketId = url.pathname.slice("/api/incidents/".length);
    const incident = readMajorFailureIncident(config, ticketId);
    if (!incident) {
      sendJson(res, 404, { error: "Unknown incident Ticket ID.", ticketId });
    } else {
      sendJson(res, 200, { incident });
    }
    return true;
  }

  // Model wrapper Phase 2: one-click "use natively" for an installed Ollama
  // model. Wraps the existing importOllamaModel (blob -> catalog, no
  // re-download) and returns the refreshed unified entry so the UI can swap
  // chips in place.
  if (req.method === "POST" && url.pathname === "/api/models/import-ollama") {
    const body = await readBody(req);
    const tag = String(body.tag ?? "").trim();
    if (!tag) {
      sendJson(res, 400, { error: "A model tag is required (e.g. \"llama3.2:3b\")." });
      return true;
    }
    try {
      const registry = loadModelRegistry();
      await importOllamaModel({ tag, config, registry, logger });
      const models = await listAvailableModels({ config, registry, logger, unified: true });
      const normalizedTag = tag.toLowerCase().includes(":") ? tag.toLowerCase() : `${tag.toLowerCase()}:latest`;
      const model = models.find((entry) => (entry.ollamaTags ?? []).includes(normalizedTag)) ?? null;
      sendJson(res, 200, { ok: true, model, models });
    } catch (error) {
      sendJson(res, 500, { ok: false, error: error.message });
    }
    return true;
  }

  // Which engines this caller may activate on. Ostrich is admin-gated, and the
  // gate is enforced on every activation path regardless - hiding the option
  // is not access control - but the picker still has to know what to offer.
  if (req.method === "GET" && url.pathname === "/api/runtime-providers") {
    const providers = listRuntimeProvidersForUser(requestUser(req));
    sendJson(res, 200, {
      providers: providers.map((provider) => ({
        id: provider.id,
        label: provider.label,
        implementationStatus: provider.implementationStatus,
        requiresExperimentalAccess: Boolean(provider.requiresExperimentalAccess),
        // Only the engines a model can actually be pinned resident on. Ollama
        // and llama.cpp run the model themselves, so they are chat engines
        // rather than activation targets.
        activationTarget: ["lizard-native", "caterpillar"].includes(provider.id),
      })),
    });
    return;
  }
  if (req.method === "GET" && url.pathname === "/api/models") {
    const hardware = cachedHardware({ config, logger });
    const task = url.searchParams.get("task") || undefined;
    const mode = url.searchParams.get("mode") || "balanced";
    const sizeBucket = url.searchParams.get("bucket") || undefined;
    const models = searchCompatibleGgufModels({ hardware, task, mode, sizeBucket });
    const unfiltered = sizeBucket ? searchCompatibleGgufModels({ hardware, task, mode }) : models;
    sendJson(res, 200, {
      hardware: {
        cpu: hardware.cpu.model,
        ramGB: hardware.memory.totalGB,
        plannerBudgetGB: hardware.memory.plannerBudgetGB,
        gpu: hardware.primaryGpu?.name ?? "CPU only",
        vramGB: hardware.primaryGpu?.vramGB ?? 0,
      },
      task,
      mode,
      bucket: sizeBucket ?? null,
      sizeBuckets: SIZE_BUCKETS.map((entry) => ({
        id: entry.id,
        label: entry.label,
        count: unfiltered.filter((model) => model.quantOptions.some((option) => option.sizeBucket === entry.id)).length,
      })),
      models,
    });
    return true;
  }

  // Live Hugging Face Hub search (paginated). Unlike /api/models (the curated
  // allowlist tier), this browses the entire Hub of GGUF repos and attaches a
  // pre-download compatibility estimate derived from each repo's name-inferred
  // parameter count - nothing is downloaded to compute it.
  if (req.method === "GET" && url.pathname === "/api/hf/search") {
    const hardware = cachedHardware({ config, logger });
    const token = process.env.HF_TOKEN ?? process.env.HUGGING_FACE_HUB_TOKEN ?? null;
    try {
      const commaList = (name) => (url.searchParams.get(name) || "")
        .split(",").map((value) => value.trim()).filter(Boolean);
      const { models, nextCursor, scanned, filtered } = await searchHubModels({
        query: url.searchParams.get("q") || undefined,
        author: url.searchParams.get("author") || undefined,
        pipeline: url.searchParams.get("pipeline") || undefined,
        sort: url.searchParams.get("sort") || "downloads",
        gguf: url.searchParams.get("gguf") !== "false",
        // Model-type / format taxonomy filters (base, instruct, reasoning,
        // multimodal, embedding, ...) - see src/lib/model-taxonomy.js.
        types: commaList("types"),
        formats: commaList("formats"),
        chatOnly: url.searchParams.get("chatOnly") === "true",
        limit: Number(url.searchParams.get("limit")) || 24,
        cursor: url.searchParams.get("cursor") || undefined,
        token,
      });
      let enriched = models.map((model) => ({
        ...model,
        verified: isAllowlistedRepo(model.repo),
        compatibility: model.paramsB
          ? classifyCompatibility({ paramsB: model.paramsB, quant: "Q4_K_M", hardware })
          : null,
      }));
      // "Lizard-native ready" filter: verify each repo against the same
      // checks the local runtime-provider pre-flight applies (architecture
      // family, supported quant, complete tensor directory) via a ranged
      // header fetch, then drop everything that would need runtime
      // enhancements. Verdicts are cached per repo revision.
      if (url.searchParams.get("nativeReady") === "true") {
        await mapWithConcurrency(enriched, 4, async (model) => {
          model.nativeCompat = await resolveNativeCompat({ config, model, token });
        });
        enriched = enriched.filter((model) => model.nativeCompat?.ready);
      }
      sendJson(res, 200, { models: enriched, nextCursor, scanned, filtered });
    } catch (error) {
      sendJson(res, 502, { error: error.message });
    }
    return true;
  }

  // The discovery taxonomy the Select step's filter chips are built from:
  // model types (base / instruct / reasoning / multimodal / ...) and weight
  // formats, each with the one-line explanation shown under it.
  if (req.method === "GET" && url.pathname === "/api/hf/taxonomy") {
    sendJson(res, 200, { types: MODEL_TYPES, formats: MODEL_FORMATS });
    return true;
  }

  // Full detail for one Hub repo: metadata + the concrete GGUF files it ships,
  // each with a per-quant compatibility estimate for the quant chooser.
  if (req.method === "GET" && url.pathname === "/api/hf/model") {
    const repo = url.searchParams.get("repo");
    if (!repo) {
      sendJson(res, 400, { error: "A repo query parameter is required." });
      return true;
    }
    const hardware = cachedHardware({ config, logger });
    const token = process.env.HF_TOKEN ?? process.env.HUGGING_FACE_HUB_TOKEN ?? null;
    try {
      const detail = await getHubModelDetail(repo, { token });
      const ggufFiles = detail.ggufFiles.map((file) => ({
        ...file,
        compatibility: classifyCompatibility({
          paramsB: detail.paramsB,
          quant: file.quant || "Q4_K_M",
          hardware,
          contextTokens: 4096,
          sizeGB: file.sizeGB,
        }),
        // Per-quant native support for the quant chooser: files outside the
        // native worker's quant set can still run on llama.cpp, so this is
        // an annotation, not a hard block.
        nativeQuantSupported: getQuantizationCompatibility(file.quant ?? null).status === "supported",
      }));
      const nativeCompat = await resolveNativeCompat({ config, model: detail, token, ggufFiles: detail.ggufFiles });
      const compatibleAlternatives = nativeCompat?.ready
        ? []
        : searchCompatibleGgufModels({
          hardware,
          task: detail.pipeline_tag ?? "text-generation",
          mode: config.defaultMode,
        }).slice(0, 3).map((model) => ({
          id: model.id,
          label: model.label,
          family: model.family,
          paramsB: model.paramsB,
          reason: "Compatible with this machine and the native GGUF runtime.",
        }));
      sendJson(res, 200, {
        ...detail,
        verified: isAllowlistedRepo(repo),
        ggufFiles,
        nativeCompat,
        compatibilityRouting: {
          available: nativeCompat?.ready === true,
          reason: nativeCompat?.ready ? null : nativeCompat?.reason ?? "The selected model is not native-compatible.",
          alternatives: compatibleAlternatives,
        },
      });
    } catch (error) {
      const status = /not found/i.test(error.message) ? 404 : 502;
      sendJson(res, status, { error: error.message });
    }
    return true;
  }

  if (req.method === "GET" && url.pathname === "/api/active") {
    // Strip `gguf` metadata from the active model and every catalog entry:
    // it carries the full raw GGUF key/value dump including tokenizer
    // vocabularies (many MB per entry), which the onboarding UI never reads
    // - serializing it made this endpoint ship a multi-MB (with a bloated
    // catalog: ~90MB) JSON response on every page load.
    const { gguf: _activeGguf, ...activeModel } = getActiveModel(config) ?? {};
    sendJson(res, 200, {
      active: Object.keys(activeModel).length ? activeModel : null,
      catalog: listLocalCatalogModels(config).map(({ gguf, ...entry }) => entry),
      server: serverEndpoint(config, activeModel.runtimeProvider),
      credits: getCreditState(config),
      edition: EDITION,
      proOffer: PRO_OFFER,
    });
    return true;
  }

  // Live state of the local API endpoint, plus the address every engine would
  // use. Read-only and deliberately not session-gated beyond the community
  // default: an external integration that has stopped answering is exactly the
  // moment a user needs to be able to see why, and being signed out of the
  // local UI is one of the reasons it might have.
  if (req.method === "GET" && url.pathname === "/api/endpoint") {
    const requested = url.searchParams.get("provider");
    const active = getActiveModel(config);
    const description = describeLocalEndpoint(config, {
      selectedProvider: requested ?? active?.runtimeProvider ?? null,
    });
    // Whether the port answers is a fact about the machine, not about our
    // bookkeeping - and the two disagreeing is itself the useful signal
    // ("stopped" here plus an occupied port means something else took it).
    let portAnswering = null;
    if (description.selected?.port) {
      portAnswering = await probeEndpointPort({
        host: description.selected.host,
        port: description.selected.port,
      }).catch(() => null);
    }
    sendJson(res, 200, {
      ...description,
      portAnswering,
      activeModelId: active?.id ?? null,
      activeRuntimeProvider: active?.runtimeProvider ?? null,
    });
    return true;
  }

  // Server-Sent Events stream for download progress.
  if (req.method === "GET" && url.pathname === "/api/download") {
    const repo = url.searchParams.get("repo");
    const filename = url.searchParams.get("file");
    const modelId = url.searchParams.get("id") || undefined;
    const license = url.searchParams.get("license") || undefined;
    res.writeHead(200, {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
    });
    const emit = (event, data) => {
      res.write(`event: ${event}\n`);
      res.write(`data: ${JSON.stringify(data)}\n\n`);
    };
    try {
      const result = await runHfDownloadWorkflow({
        repo,
        filename,
        modelId,
        license,
        config,
        logger,
        onPhase: (phase) => emit("phase", phase),
        onProgress: (progress) => emit("progress", progress),
        // Re-downloading a model that is currently loaded means replacing a file
        // this server has mapped, and Windows will not unlink a mapped file - it
        // failed with a bare EPERM after transferring the whole model. The
        // warm worker is released here for the same reason /api/models/delete
        // releases it before unlinking, and only when the download is actually
        // about to overwrite the resident model.
        onBeforeReplace: async ({ modelId: resolvedId }) => {
          if (!warmModelArtifact) return;
          if (normalizeModelId(warmModelArtifact.modelId) !== normalizeModelId(resolvedId)) return;
          emit("phase", {
            phase: "release",
            message: "Unloading the current copy so it can be replaced",
          });
          cancelPendingModelRelease();
          await getSharedNativeRuntime(warmEngineId).releaseModelWorker(warmModelArtifact).catch(() => {});
          warmModelArtifact = null;
          invalidateEngineProbeCache();
        },
      });
      emit("done", {
        record: result.record,
        verification: result.verification,
        bytes: result.download.bytes,
      });
    } catch (error) {
      const incident = await reportUiMajorFailure({
        error,
        config,
        logger,
        req,
        context: { operation: "gguf-download", route: url.pathname, surface: "onboarding-download" },
      });
      emit("error", { message: error.message, ...incidentUserFields(incident) });
    }
    res.end();
    return true;
  }

  // Deletes a downloaded model from disk, memory and configuration: first
  // releases the warm in-memory worker if this model is resident (a live
  // worker keeps the GGUF mapped, which would block the unlink on Windows),
  // then removes the file, the catalog entry and - if it was the active
  // model - the activeModel selection.
  if (req.method === "POST" && url.pathname === "/api/models/delete") {
    const body = await readBody(req);
    const modelId = String(body.id ?? "").trim();
    if (!modelId) {
      sendJson(res, 400, { error: "A model id is required." });
      return true;
    }
    try {
      if (warmModelArtifact && normalizeModelId(warmModelArtifact.modelId) === normalizeModelId(modelId)) {
        cancelPendingModelRelease();
        await getSharedNativeRuntime(warmEngineId).releaseModelWorker(warmModelArtifact).catch(() => {});
        warmModelArtifact = null;
      }
      const result = deleteModel({ config, modelId });
      logger?.info("ui-model-deleted", result);
      sendJson(res, 200, result);
    } catch (error) {
      const incident = await reportUiMajorFailure({
        error,
        config,
        logger,
        req,
        context: { operation: "delete-model", route: url.pathname, surface: "onboarding" },
      });
      sendJson(res, 400, { error: error.message, ...incidentUserFields(incident) });
    }
    return true;
  }

  if (req.method === "POST" && url.pathname === "/api/activate") {
    const body = await readBody(req);
    try {
      const active = activateModel({
        config,
        modelId: body.id,
        runtimeProvider: resolveRequestedRuntimeProvider({
          requested: body.runtimeProvider,
          current: config?.runtimeConfig?.runtimeProvider ?? null,
          fallback: DEFAULT_MODEL_RUNTIME_PROVIDER,
          user: requestUser(req),
        }),
      });
      selectChatEngineForActivatedModel(active.runtimeProvider);
      // Only one engine serves the local API at a time. If the binding just
      // moved to a different model or engine, the port that is still answering
      // for the old one is now lying about what it serves - close it here and
      // let the warm load re-open it for the new binding once that model has
      // proven it can answer.
      const armed = getLocalEndpointState();
      if (
        armed.status === "listening" &&
        (armed.runtimeProvider !== active.runtimeProvider || armed.modelId !== active.id)
      ) {
        await stopLocalEndpointFor(ENDPOINT_STOP_REASONS.PROVIDER_SWITCH, logger);
      }
      // The bound model changed, so chat's cached engine availability and
      // model readiness are stale - recompute them on the next read.
      invalidateEngineProbeCache();
      // Activating is the user asking for this model, which lifts an earlier
      // "I freed the memory" latch: from here the warm paths may load again.
      setModelResidencyIntent(config, "resident", { reason: "activate" });
      sendJson(res, 200, { active, endpoint: describeLocalEndpoint(config, { selectedProvider: active.runtimeProvider }) });
    } catch (error) {
      const incident = await reportUiMajorFailure({
        error,
        config,
        logger,
        req,
        context: { operation: "activate-model", route: url.pathname, surface: "onboarding" },
      });
      sendJson(res, 400, { error: error.message, ...incidentUserFields(incident) });
    }
    return true;
  }

  // Unload: the other half of the Activate button. Frees the resident worker
  // and clears the binding, so nothing is loaded and nothing is active until
  // the user activates again. This is the ONLY way (short of shutting the app
  // down) that an activated model leaves memory - see scheduleModelRelease.
  if (req.method === "POST" && url.pathname === "/api/deactivate") {
    try {
      // Release first, clear second: on Windows a live worker holds the GGUF
      // mapped, and a half-cleared binding with a worker still up is the state
      // that strands memory with nothing pointing at it.
      cancelPendingModelRelease();
      // The external http:// port is the first thing to go. Unload means the
      // model is gone, and a port that outlived it would either answer errors
      // or quietly load the weights straight back for the next caller.
      const endpointBeforeUnload = getLocalEndpointState();
      await stopLocalEndpointFor(ENDPOINT_STOP_REASONS.UNLOAD, logger);
      if (warmModelArtifact) {
        await getSharedNativeRuntime(warmEngineId).releaseModelWorker(warmModelArtifact).catch(() => {});
        warmModelArtifact = null;
      }
      // Unload must free the machine's memory, not just this process's handle
      // on it. Releasing warmModelArtifact only ends the worker THIS server
      // started; a model loaded by a benchmark or a `lizard-llm serve` runs in
      // its own process tree, recorded in runtime-state.json, and used to
      // survive Unload untouched - the button reported success while a
      // multi-gigabyte runner stayed resident with nothing pointing at it.
      // stopCurrentModel is the same path `lizard-llm model stop` uses: it
      // terminates the tracked tree and only releases the residency claim once
      // termination is proven, so a failure here is reported rather than
      // silently leaving a claim behind.
      // What this unload is about to interrupt. Probed BEFORE the stop, while
      // the server is still answering - afterwards there is nothing left to
      // ask. Reported rather than enforced: the surface that has the user in
      // front of it decides whether to confirm, and a probe that cannot reach
      // the server must never be able to block an unload the user asked for.
      let unloadImpact = null;
      try {
        const { describeUnloadImpact } = await import("../lib/model-activity.js");
        unloadImpact = await describeUnloadImpact(config);
      } catch {
        unloadImpact = null;
      }
      let trackedStop = { stopped: false, reason: "no_tracked_model" };
      try {
        trackedStop = await stopCurrentModel(config);
      } catch (error) {
        const incident = await reportUiMajorFailure({
          error,
          config,
          logger,
          req,
          context: {
            operation: "unload-model",
            route: url.pathname,
            surface: "onboarding",
            unrecoverable: error?.code === "RESIDENCY_RELEASE_FAILED",
          },
        });
        trackedStop = {
          stopped: false,
          reason: error.code ?? "stop_failed",
          error: error.message,
          ...incidentUserFields(incident),
        };
        logger?.error?.("ui-model-deactivate-tracked-stop-failed", {
          message: error.message,
          code: error.code ?? null,
        });
      }
      // Freeing the memory is only half of it: a claim left behind blocks the
      // next load, so Unload would hand back the RAM and still refuse to load
      // anything into it. The claim is released by stopCurrentModel when its
      // modelPid matches the tracked root, but a claim this server owns on
      // behalf of a differently-recorded load needs the evidence-based
      // recovery instead. It only clears a claim whose entire recorded tree is
      // provably dead - never a force-clear, and never a live model's claim.
      if (trackedStop.stopped) {
        try {
          recoverStaleModelResidencyClaim(config, { allowLiveOwnerWithDeadTree: true });
        } catch (error) {
          logger?.warn?.("ui-model-deactivate-claim-recovery-skipped", { message: error.message });
        }
      }
      const released = deactivateModel({ config });
      invalidateEngineProbeCache();
      // Unload means unloaded. Without the latch the next page load reopened
      // the warm stream and put the weights straight back.
      setModelResidencyIntent(config, "released", { reason: "unload" });
      logger?.info?.("ui-model-deactivated", {
        modelId: released.id,
        trackedStopped: trackedStop.stopped === true,
        trackedReason: trackedStop.reason ?? null,
      });
      // "The port is closed" is a claim about the machine, so it is verified
      // rather than asserted: an occupied port after an unload means something
      // else owns it (a leftover `lizard-llm serve`, a benchmark still running)
      // and the user needs to know that, not be told everything is fine.
      let endpointPortStillAnswering = null;
      if (endpointBeforeUnload.port) {
        endpointPortStillAnswering = await probeEndpointPort({
          host: endpointBeforeUnload.host ?? "127.0.0.1",
          port: endpointBeforeUnload.port,
        }).catch(() => null);
      }
      sendJson(res, 200, {
        deactivated: released,
        warm: getWarmState(),
        endpoint: {
          ...describeLocalEndpoint(config),
          portStillAnswering: endpointPortStillAnswering,
          portNote: endpointPortStillAnswering === true
            ? `Port ${endpointBeforeUnload.port} is still answering after the unload, so another process owns it - a leftover \`lizard-llm serve\` or a benchmark. Lizard's own endpoint is closed.`
            : endpointPortStillAnswering === false
              ? `Port ${endpointBeforeUnload.port} is closed. No external app can reach a model until you activate one again.`
              : null,
        },
        // Surfaced so the UI can say "unloaded, but a runner is still holding
        // memory" instead of claiming a clean unload it did not achieve.
        residency: {
          trackedStopped: trackedStop.stopped === true,
          reason: trackedStop.reason ?? null,
          error: trackedStop.error ?? null,
        },
        // What was in flight when the unload started, so the surface can tell
        // the user what it interrupted instead of leaving them to infer it
        // from a truncated reply.
        unloadImpact,
      });
    } catch (error) {
      const incident = await reportUiMajorFailure({
        error,
        config,
        logger,
        req,
        context: { operation: "deactivate-model", route: url.pathname, surface: "onboarding" },
      });
      sendJson(res, 400, { error: error.message, ...incidentUserFields(incident) });
    }
    return true;
  }

  // Free RAM: hand back every byte the model runtimes are holding, whether or
  // not this app is the one holding it.
  //
  // Unload (/api/deactivate above) releases what THIS server owns and clears the
  // binding. This is the other half: it keeps the user's selection but takes the
  // weights out of memory, and it also collects runners that nothing owns any
  // more - a worker orphaned by a crash, a llama.cpp server left over from a
  // benchmark, models still resident in Ollama. Those are invisible to the
  // binding and were the ones actually holding the machine's memory.
  //
  // Order is deliberate: the graceful release runs FIRST, so this server's own
  // workers go back through the normal path and the residency claim is released
  // by the code that owns it. Killing a worker out from under the server would
  // leave a claim pointing at a process that no longer exists - which the
  // residency rules exist to prevent.
  if (req.method === "POST" && url.pathname === "/api/free-memory") {
    try {
      const report = await freeModelMemory({
        log: (line) => logger?.info?.("ui-free-memory", { line }),
        releaseResidentWorkers: async () => {
          cancelPendingModelRelease();
          // Free RAM keeps the selection but takes the weights out of memory,
          // so the endpoint has to close with them - leaving it open would let
          // the next external request pull the model straight back in, which is
          // the exact instruction the user just gave the opposite of.
          await releaseAllWarmModels({ endpointStopReason: ENDPOINT_STOP_REASONS.FREE_MEMORY });
          warmModelArtifact = null;
          // Models started outside this server (a benchmark, `lizard-llm serve`)
          // run in their own process tree and are tracked in runtime-state.json.
          // Releasing only our own worker would report success while the biggest
          // consumer stayed resident.
          const trackedStop = await stopCurrentModel(config).catch((error) => ({
            stopped: false,
            reason: error?.code ?? "stop_failed",
          }));
          if (trackedStop?.stopped) {
            try {
              recoverStaleModelResidencyClaim(config, { allowLiveOwnerWithDeadTree: true });
            } catch (error) {
              logger?.warn?.("ui-free-memory-claim-recovery-skipped", { message: error.message });
            }
          }
        },
      });
      // Chat and the header pill both read warm state; nothing is resident now.
      invalidateEngineProbeCache();
      // Freeing memory is an instruction, not a moment. Without this latch the
      // next page load reopened /api/warm-model and loaded the model straight
      // back in, which is exactly what "I cleared memory and it came back"
      // looks like from the outside.
      setModelResidencyIntent(config, "released", { reason: "free-memory" });
      logger?.info?.("ui-free-memory-complete", {
        freedGb: report.freedGb,
        stopped: report.stoppedEngineProcesses,
        ollama: report.unloadedOllamaModels,
      });
      sendJson(res, 200, { ...report, warm: getWarmState(), endpoint: describeLocalEndpoint(config) });
    } catch (error) {
      const incident = await reportUiMajorFailure({
        error,
        config,
        logger,
        req,
        context: { operation: "free-model-memory", route: url.pathname, surface: "onboarding" },
      });
      sendJson(res, 500, { error: error.message, ...incidentUserFields(incident) });
    }
    return true;
  }

  // Server-Sent Events stream that pre-loads the active model into the
  // shared runtime (see getSharedNativeRuntime above) without generating
  // anything, so the onboarding UI can show a "warming up" state right
  // after activation and only enable chat testing once loading is
  // actually done - and so that first chat message is fast too.
  //
  // Unlike a typical one-shot SSE request, this connection stays open
  // after "ready" for as long as the page does - see the grace-period
  // release logic above (registerShutdownHandlers/scheduleModelRelease).
  // The client (warmUpModelForChat in app.js) reopens it immediately on
  // every page load/reload, so activeWarmConnections only drops to 0, and
  // the release timer only runs to completion, when the page is actually
  // closed.
  if (req.method === "GET" && url.pathname === "/api/warm-model") {
    res.writeHead(200, {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
    });
    let keepAliveTimer = null;
    let closed = false;
    // Loading a model takes long enough that the user can close the tab, reload,
    // or navigate away in the middle of it - and then the load finishes and
    // writes its result into a socket that is gone. On an already-destroyed
    // response that write throws, and before the guard in createUiServer that
    // throw ended the process. Emitting is best-effort by nature here: the load
    // itself is the point, the progress report is not.
    const emit = (event, data) => {
      if (closed || res.writableEnded || res.destroyed) return;
      try {
        res.write(`event: ${event}\n`);
        res.write(`data: ${JSON.stringify(data)}\n\n`);
      } catch {
        closed = true;
      }
    };
    activeWarmConnections += 1;
    cancelPendingModelRelease();
    const onClose = () => {
      if (closed) return;
      closed = true;
      if (keepAliveTimer) clearInterval(keepAliveTimer);
      activeWarmConnections = Math.max(0, activeWarmConnections - 1);
      if (activeWarmConnections === 0) {
        scheduleModelRelease(config);
      }
    };
    req.on("close", onClose);
    // A keep-alive write to a dead socket is the same hazard as `emit`, and this
    // one fires on a timer with no request context to catch it.
    const startKeepAlive = () => {
      keepAliveTimer = setInterval(() => {
        if (closed || res.writableEnded || res.destroyed) {
          clearInterval(keepAliveTimer);
          keepAliveTimer = null;
          return;
        }
        try {
          res.write(": keep-alive\n\n");
        } catch {
          clearInterval(keepAliveTimer);
          keepAliveTimer = null;
        }
      }, 15000);
      keepAliveTimer.unref?.();
    };
    const startedAt = Date.now();
    // This stream is opened by every page load, so it is the loader that most
    // often resurrected a model the user had just freed. It now respects the
    // release latch: report the state honestly and load nothing. Anything the
    // user actually asks for (activate, a chat message, a run) clears the latch
    // and warms again from there.
    if (automaticModelLoadSuspended(config)) {
      emit("idle", {
        modelId: getActiveModel(config)?.id ?? null,
        reason: "released",
        note: "Memory was freed on purpose, so nothing is being loaded. Send a message or activate the model to load it again.",
      });
      startKeepAlive();
      return true;
    }
    try {
      const hardware = scanHardware({ config, logger });
      const registry = loadModelRegistry();
      const context = await resolveExecutionContext({
        command: "generate",
        options: { model: "active", mode: config.defaultMode, backend: "cpu", "gpu-layers": "0" },
        config,
        registry,
        hardware,
        logger,
      });
      const requestedProvider = activeNativeProvider(config);
      const runtimeResolution = resolveModelRuntimeEngine({
        requestedEngine: requestedProvider,
        modelArtifact: context.modelArtifact,
        profile: context.profile,
      });
      if (runtimeResolution.engine !== requestedProvider) {
        emit("ready", {
          modelId: context.modelArtifact.modelId,
          loadMs: Date.now() - startedAt,
          runtimeProvider: "llama.cpp",
          note: runtimeResolution.fallbackReason,
        });
        startKeepAlive();
        return true;
      }
      warmModelArtifact = context.modelArtifact;
      warmEngineId = requestedProvider;
      const adaptiveProfile = withAdaptiveNativeConcurrency(
        context.profile,
        requestedProvider,
        config,
        context.modelArtifact.modelId,
      );
      emit("warming", {
        modelId: context.modelArtifact.modelId,
        runtimeProvider: requestedProvider,
        concurrencyPolicy: adaptiveProfile.nativeRuntime.concurrencyPolicy,
      });
      await getSharedNativeRuntime(requestedProvider).ensureModelLoaded({
        recommendation: context.recommendation,
        modelArtifact: context.modelArtifact,
        profile: adaptiveProfile,
        config,
      });
      const loadMs = Date.now() - startedAt;
      // Loaded is not the same as working. Between here and "ready" the model
      // is asked for two tokens, and only an answer promotes it - see
      // verifyAndArmLocalEndpoint. The client shows this as "Testing" so the
      // extra second is legible rather than a stall.
      emit("verifying", {
        modelId: context.modelArtifact.modelId,
        runtimeProvider: requestedProvider,
        loadMs,
      });
      const { verification, endpoint } = await verifyAndArmLocalEndpoint({
        config,
        logger,
        runtimeProvider: requestedProvider,
        context,
        profile: adaptiveProfile,
      });
      if (!verification.ok) {
        // The weights are resident but the model cannot answer, so nothing
        // downstream may treat it as usable: no "Active" pill, no open port.
        // The worker is left alone rather than killed - the user may want to
        // retry, and tearing it down here would race the residency claim.
        warmModelArtifact = null;
        emit("load-error", {
          message: verification.note ?? verification.message ?? "The model did not pass its two-token answer check.",
          stage: "load-check",
          reason: verification.reason,
          detail: verification.message,
          loadMs,
          checkMs: verification.durationMs,
          endpoint,
        });
        onClose();
        try {
          res.end();
        } catch {
          // Already torn down by the disconnect that got us here.
        }
        return true;
      }
      emit("ready", {
        modelId: context.modelArtifact.modelId,
        loadMs,
        runtimeProvider: requestedProvider,
        concurrencyPolicy: adaptiveProfile.nativeRuntime.concurrencyPolicy,
        verification,
        endpoint,
      });
      startKeepAlive();
    } catch (error) {
      // The load did not happen, so nothing is resident. warmModelArtifact was
      // set optimistically before the load began; leaving it set reports a warm
      // model that is not there, and sends the next Unload and the release timer
      // chasing a worker that never existed.
      warmModelArtifact = null;
      // Reporting the incident is bookkeeping and can itself fail (disk full, no
      // network for the support bundle). It must never be the reason the client
      // is left waiting: the whole point of this catch is that the UI gets a
      // verdict, so a failure to file the ticket costs the ticket, not the
      // verdict.
      let incident = null;
      try {
        incident = await reportUiMajorFailure({
          error,
          config,
          logger,
          req,
          context: {
            operation: "warm-model-load",
            route: url.pathname,
            surface: "onboarding",
            unrecoverable: true,
          },
        });
      } catch (reportError) {
        logger?.warn?.("warm-model-incident-report-failed", {
          message: String(reportError?.message ?? reportError).slice(0, 200),
        });
      }
      // Named distinctly from EventSource's own native "error" event (fired
      // client-side on any connection drop, including an ordinary reload)
      // so the client can tell a genuine load failure apart from the
      // connection just closing.
      emit("load-error", { message: error.message, ...incidentUserFields(incident) });
      onClose();
      try {
        res.end();
      } catch {
        // Already torn down by the disconnect that got us here.
      }
    }
    return true;
  }

  // Server-Sent Events stream for the onboarding "Test Chat" dialog - talks
  // directly to the just-activated Lizard native model, streaming tokens as
  // they decode (no Ollama, no oracle fallback).
  // Ends a Test Chat session explicitly - called when the dialog closes so
  // the session-scoped memory never outlives the dialog.
  if (req.method === "POST" && url.pathname === "/api/quick-chat/reset") {
    const body = await readBody(req);
    const sessionId = String(body.session ?? "").trim();
    if (sessionId) {
      chatSessions.delete(sessionId);
    }
    sendJson(res, 200, { ok: true });
    return true;
  }

  if (req.method === "GET" && url.pathname === "/api/quick-chat") {
    const prompt = String(url.searchParams.get("prompt") ?? "").trim();
    const sessionId = String(url.searchParams.get("session") ?? "").trim() || null;
    const requestedMaxTokens = Number(url.searchParams.get("maxTokens"));
    res.writeHead(200, {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
    });
    const emit = (event, data) => {
      res.write(`event: ${event}\n`);
      res.write(`data: ${JSON.stringify(data)}\n\n`);
    };
    if (!prompt) {
      emit("error", { message: "A prompt is required." });
      res.end();
      return true;
    }
    const startedAt = Date.now();
    try {
      // Fold this dialog session's prior exchanges into the prompt so the
      // model knows what was already said - see composeChatPrompt above.
      const contextPrompt = composeChatPrompt(getChatSessionHistory(sessionId), prompt);
      const hardware = scanHardware({ config, logger });
      const registry = loadModelRegistry();
      const context = await resolveExecutionContext({
        command: "generate",
        // backend: "cpu" + gpu-layers: "0" mirror the `serve` command
        // (src/cli.js) exactly: they steer the PROFILE builder away from a
        // traditional llama.cpp GPU backend (e.g. vulkan), which validates
        // that gpuLayers > 0 and throws "Backend vulkan requires GPU
        // layers greater than 0." otherwise. The Lizard native GPU-resident
        // decode path used below is a separate mechanism entirely (gated by
        // nativeRuntime.gpuDecodePolicy, not profile.gpuLayers), so this
        // profile field is irrelevant to how the request actually executes.
        options: { model: "active", prompt: contextPrompt, mode: config.defaultMode, backend: "cpu", "gpu-layers": "0" },
        config,
        registry,
        hardware,
        logger,
      });
      const requestedProvider = activeNativeProvider(config);
      const runtimeResolution = resolveModelRuntimeEngine({
        requestedEngine: requestedProvider,
        modelArtifact: context.modelArtifact,
        profile: context.profile,
      });
      const maxTokens = Number.isFinite(requestedMaxTokens) && requestedMaxTokens > 0
        ? Math.min(requestedMaxTokens, 2048)
        : runtimeResolution.engine === requestedProvider
          ? 512
          : 96;
      // "thinking" covers tokenizer/session/prefill setup - there is no
      // visible output yet, but the request is genuinely working.
      emit("thinking", {});
      let firstDeltaSeen = false;
      let firstContentAt = null;
      if (runtimeResolution.engine !== requestedProvider) {
        firstDeltaSeen = true;
        emit("start", {});
      }
      // Prefer the native runtime when this GGUF is inside the compiled
      // worker's current execution subset; otherwise route to llama.cpp
      // immediately instead of letting an unsupported family hang in the
      // native worker load path.
      const result = await generateWithRuntimeSession({
        requestedEngine: runtimeResolution.engine,
        runtime: runtimeResolution.engine === requestedProvider
          ? getSharedNativeRuntime(requestedProvider)
          : null,
        recommendation: context.recommendation,
        modelArtifact: context.modelArtifact,
        // keepWorkerLoaded: true (same flag `serve` in src/cli.js sets) is
        // required IN ADDITION TO reusing the shared runtime above -
        // without it, ModelSession's onClose (native-cpu-runtime.js)
        // unconditionally releases the worker at the end of THIS request,
        // deleting the cache entry regardless of the runtime instance
        // persisting. This was the actual root cause of every chat
        // message reloading the whole model from scratch (90-140s) even
        // after the first one had already loaded it.
        //
        // timeoutMs: null disables the scheduler's hard request timeout,
        // which otherwise defaults to 120000ms (DEFAULT_TIMEOUT_MS,
        // src/lib/execution-profile.js) - sized for short CLI/benchmark
        // calls, not a live chat answer. At this hardware's ~3.9 tok/s, a
        // full 512-token response takes ~130s, so a genuinely-still-
        // working generation was being killed mid-stream ("Scheduler
        // request timed out after 120000 ms"). Same "long generations are
        // legitimate, let them run" reasoning already used for the same
        // reason in scripts/benchmark-big.ts. The user watches the tokens
        // stream live in the dialog, so there's no silent hang risk.
        profile: withAdaptiveNativeConcurrency({
          ...context.profile,
          timeoutMs: null,
        }, requestedProvider, config, context.modelArtifact.modelId),
        prompt: contextPrompt,
        maxTokens,
        config,
        logger,
        onDelta: (text) => {
          if (!firstDeltaSeen) {
            firstDeltaSeen = true;
            firstContentAt = Date.now();
            emit("start", {});
          }
          if (text) {
            emit("delta", { text });
          }
        },
      });
      recordChatExchange(sessionId, prompt, stripReasoningMarkup(result.output));
      const execution = buildQuickChatExecutionTelemetry(result);
      const responseTimeMs = Date.now() - startedAt;
      const generatedTokens = result.nativeExecution?.generatedTokenIds?.length ?? null;
      const userVisibleTokensPerSecond = generatedTokens > 0 && responseTimeMs > 0
        ? (generatedTokens / responseTimeMs) * 1000
        : result.tokensPerSecond ?? null;
      const ttftMs = (firstContentAt ?? Date.now()) - startedAt;
      appendNativeTurnMetric(config, {
        userId: "local-onboarding",
        surface: "onboarding-chat",
        modelId: result.modelId ?? context.modelArtifact?.modelId,
        quant: context.profile?.gguf?.quantization ?? context.profile?.gguf?.quant ?? null,
        runtimeProvider: result.runtimeProvider ?? runtimeResolution.engine,
        backend: result.nativeBackend,
        servingPath: execution.servingPath,
        batched: execution.batched,
        concurrencyLevel: execution.concurrencyLevel,
        contextTokens: context.profile?.contextSize,
        promptTokens: result.promptArtifact?.tokenCount,
        generatedTokens,
        prefillMs: result.prefillMs,
        decodeMs: result.decodeMs,
        decodeTokensPerSecond: result.tokensPerSecond,
        userVisibleTokensPerSecond,
        endToEndMs: responseTimeMs,
        ttftMs,
        latencyPerTokenMs: generatedTokens > 0 ? responseTimeMs / generatedTokens : null,
        transport: "onboarding-http-stream",
        optimizationProfile: context.profile?.caterpillarPerformanceProfile
          ?? (requestedProvider === "caterpillar" ? CATERPILLAR_TTFT_PROFILE : "balanced"),
      });
      emit("done", {
        response: result.output ?? "",
        modelId: result.modelId,
        runtimeProvider: result.runtimeProvider ?? runtimeResolution.engine,
        nativeBackend: result.nativeBackend ?? null,
        tokensPerSecond: userVisibleTokensPerSecond,
        userVisibleTokensPerSecond,
        decodeTokensPerSecond: result.tokensPerSecond ?? null,
        ttftMs,
        durationMs: result.durationMs ?? (Date.now() - startedAt),
        responseTimeMs,
        sessionTurns: getChatSessionHistory(sessionId).length,
        execution,
      });
    } catch (error) {
      const incident = await reportUiMajorFailure({
        error,
        config,
        logger,
        req,
        context: { operation: "quick-chat", route: url.pathname, surface: "onboarding-chat" },
      });
      emit("error", {
        message: error.message,
        responseTimeMs: Date.now() - startedAt,
        ...incidentUserFields(incident),
      });
    }
    res.end();
    return true;
  }

  return false;
}

// Mountable onboarding router. `basePath` lets the onboarding UI be hosted under
// a path prefix (e.g. "/onboarding" inside the dashboard) so everything lives on
// one origin. Returns true if the request was handled.
export async function routeOnboarding({ req, res, config, logger, basePath = "", dashboardIntegration = null }) {
  let url;
  try {
    url = new URL(req.url, "http://localhost");
  } catch {
    sendJson(res, 400, { error: "Bad request URL" });
    return true;
  }

  // Strip the mount prefix so internal routing stays absolute ("/api/...", "/").
  let pathname = url.pathname;
  if (basePath) {
    if (pathname === basePath) {
      // Normalise to a trailing slash so relative asset URLs resolve correctly.
      res.writeHead(302, { Location: `${basePath}/` });
      res.end();
      return true;
    }
    if (pathname.startsWith(`${basePath}/`)) {
      pathname = pathname.slice(basePath.length) || "/";
    } else {
      return false; // not ours
    }
  }

  const scopedUrl = { pathname, searchParams: url.searchParams };

  // Loopback hardening. The app binds to 127.0.0.1 only, but a web page can
  // still point a DNS name at 127.0.0.1 (DNS rebinding) or post a form to it
  // (CSRF). So the Host header must be a loopback authority on every request,
  // and a state-changing request that carries an Origin must come from this
  // same loopback origin. There are no accounts, so there is no session check.
  {
    const verdict = getLoopbackGuard().check({
      method: req.method,
      headers: req.headers,
      requiresSession: false,
      session: null,
      csrfOk: false,
      enforceMutations: false,
    });
    if (!verdict.ok) {
      logger?.warn?.("ui-request-blocked", { path: pathname, code: verdict.code });
      sendJson(res, verdict.status, { error: verdict.message, code: verdict.code });
      return true;
    }
    if (MUTATING_METHODS.has(req.method) && !originIsSameLoopback(req.headers)) {
      logger?.warn?.("ui-request-blocked", { path: pathname, code: "ORIGIN_MISMATCH" });
      sendJson(res, 403, { error: "Cross-origin request refused.", code: "ORIGIN_MISMATCH" });
      return true;
    }
  }

  if (pathname.startsWith("/api/")) {
    try {
      const handled = await handleApi({ req, res, url: scopedUrl, config, logger, dashboardIntegration });
      if (!handled) {
        sendJson(res, 404, { error: `Unknown endpoint: ${pathname}` });
      }
    } catch (error) {
      logger?.error("ui-api-error", { path: pathname, message: error.message });
      const incident = await reportUiMajorFailure({
        error,
        config,
        logger,
        req,
        context: {
          operation: "api-request",
          route: pathname,
          surface: "onboarding-api",
        },
      });
      if (!res.headersSent) {
        sendJson(res, 500, { error: error.message, ...incidentUserFields(incident) });
      } else {
        res.end();
      }
    }
    return true;
  }

  if (req.method === "GET" && pathname.startsWith("/assets/")) {
    serveAsset(res, pathname);
    return true;
  }

  // D3, served from the installed node_modules rather than a CDN: the app
  // promises everything runs on this machine, and a chart that needs the
  // network would quietly break that on the first offline conversation.
  // Fetched lazily by chat-charts.js on the first chart of a session.
  if (req.method === "GET" && pathname === "/vendor/d3.min.js") {
    serveD3Bundle(res);
    return true;
  }

  const staticEntry = STATIC_FILES[pathname];
  if (staticEntry && req.method === "GET") {
    serveStatic(res, staticEntry);
    return true;
  }

  return false;
}

export function createUiServer({ config, logger, dashboardIntegration = null }) {
  const server = http.createServer(async (req, res) => {
    // The handler is async, so anything it throws becomes a rejected promise
    // that Node - which since v15 treats an unhandled rejection as fatal - would
    // turn into a process exit. That is how one bad model load took the whole
    // dashboard down: the load failed, the failure path wrote to a response
    // whose client had already gone away, that throw escaped, and the app was
    // gone with the port still in TIME_WAIT. A request may fail; the server it
    // arrived on may not.
    if (req.url === "/onboarding" || req.url.startsWith("/onboarding/") || req.url.startsWith("/onboarding?")) {
      req.url = req.url.slice("/onboarding".length) || "/";
      if (!req.url.startsWith("/")) req.url = `/${req.url}`;
    }
    try {
      const handled = await routeOnboarding({
        req,
        res,
        config,
        logger,
        basePath: "",
        dashboardIntegration,
      });
      if (!handled && !res.headersSent) {
        sendJson(res, 404, { error: "Not found" });
      }
    } catch (error) {
      logger?.error?.("ui-request-failed", {
        method: req.method,
        url: req.url,
        message: String(error?.message ?? error).slice(0, 300),
      });
      try {
        if (!res.headersSent) {
          sendJson(res, 500, { error: "The request failed. The app is still running - please try again." });
        } else {
          res.end();
        }
      } catch {
        // The client is already gone; there is nobody left to tell.
      }
    }
  });
  // A socket that dies mid-response emits an error on the request/response
  // stream. Unhandled, that is another uncaught throw and another dead app.
  server.on("clientError", (error, socket) => {
    logger?.warn?.("ui-client-error", { message: String(error?.message ?? error).slice(0, 200) });
    if (socket.writable) {
      socket.end("HTTP/1.1 400 Bad Request\r\n\r\n");
    }
  });
  return server;
}

// Load the activated model back into memory on app launch.
//
// Activation is durable - it lives in ~/.lizard-llm/config.json - but memory is
// not, so after a machine restart the binding survives and the weights do not.
// This closes that gap: if a model is activated, it goes resident again without
// anyone opening onboarding, and stays that way until Unload (see
// /api/deactivate) or shutdown. Best-effort by design: a load failure here must
// never stop the app from starting, so it is logged and dropped - the model
// still loads on first use the way it always did.
export function resumeActivatedModelOnBoot(config, logger) {
  Promise.resolve().then(async () => {
    try {
      const active = getActiveModel(config);
      if (!active?.id) return;
      // A model the user freed before closing the app stays freed. Resuming it
      // on the next launch would undo an explicit instruction, and on a laptop
      // that is several GB back in memory nobody asked for.
      if (automaticModelLoadSuspended(config)) {
        logger?.info?.("activated-model-resume-skipped", {
          modelId: active.id,
          reason: "residency-released-by-user",
        });
        return;
      }
      const hardware = scanHardware({ config, logger });
      const registry = loadModelRegistry();
      const context = await resolveExecutionContext({
        command: "generate",
        options: { model: "active", mode: config.defaultMode, backend: "cpu", "gpu-layers": "0" },
        config,
        registry,
        hardware,
        logger,
      });
      const requestedProvider = activeNativeProvider(config);
      const runtimeResolution = resolveModelRuntimeEngine({
        requestedEngine: requestedProvider,
        modelArtifact: context.modelArtifact,
        profile: context.profile,
      });
      // llama.cpp spawns per request - there is nothing to hold resident, and
      // pretending otherwise would report a warm model that is not there.
      if (runtimeResolution.engine !== requestedProvider) return;
      warmModelArtifact = context.modelArtifact;
      warmEngineId = requestedProvider;
      const adaptiveProfile = withAdaptiveNativeConcurrency(
        context.profile,
        requestedProvider,
        config,
        context.modelArtifact.modelId,
      );
      const startedAt = Date.now();
      await getSharedNativeRuntime(requestedProvider).ensureModelLoaded({
        recommendation: context.recommendation,
        modelArtifact: context.modelArtifact,
        profile: adaptiveProfile,
        config,
      });
      // This is the path that makes the endpoint survive a reboot: the machine
      // comes back, the app starts, the activated model loads, it proves it can
      // answer, and the port re-opens - with nobody opening a browser and
      // nobody signed in. Everything an external integration needs is back
      // before anyone touches the UI.
      const { verification, endpoint } = await verifyAndArmLocalEndpoint({
        config,
        logger,
        runtimeProvider: requestedProvider,
        context,
        profile: adaptiveProfile,
      });
      if (!verification.ok) {
        warmModelArtifact = null;
      }
      logger?.info?.("activated-model-resumed", {
        modelId: context.modelArtifact.modelId,
        runtimeProvider: requestedProvider,
        concurrencyPolicy: adaptiveProfile.nativeRuntime.concurrencyPolicy,
        loadMs: Date.now() - startedAt,
        loadCheck: verification.ok ? "passed" : verification.reason,
        endpoint: endpoint.status === "listening" ? endpoint.url : null,
      });
    } catch (error) {
      warmModelArtifact = null;
      logger?.warn?.("activated-model-resume-failed", {
        message: String(error?.message ?? error).slice(0, 200),
      });
      const incident = await reportUiMajorFailure({
        error,
        config,
        logger,
        context: {
          operation: "resume-activated-model",
          surface: "app-startup",
          unrecoverable: true,
        },
      });
      if (incident?.reported) {
        logger?.error?.("activated-model-resume-ticket", {
          ticketId: incident.ticketId,
        });
      }
    }
  });
}

export function startUiServer({ config, logger, port = 8088, host = "127.0.0.1", dashboardIntegration = null } = {}) {
  const server = createUiServer({ config, logger, dashboardIntegration });
  registerShutdownHandlers(server);
  return new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(port, host, () => {
      resolve({ server, url: `http://${host}:${port}` });
    });
  });
}
