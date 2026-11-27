// The engine behind a chat turn.
//
// The MODEL is bound by onboarding and is not selectable here (see
// resolveBoundModel in src/ui/chat-routes.js). The ENGINE that runs it is:
// the same GGUF can be decoded by Lizard's own native runtime, by the
// caterpillar graph-plan engine, by stock llama.cpp, or handed to a local
// Ollama daemon. Which one is fastest is a property of the machine, not of
// the model, so it belongs in front of the user rather than in a config file.
//
// Three of the four go through the existing runtime stack
// (generateWithRuntimeSession); Ollama is an HTTP daemon and has its own path.
// This module is the seam that makes them look the same to the chat route.

import fs from "node:fs";
import path from "node:path";
import { scanHardware } from "../hardware.js";
import { resolveExecutionContext } from "../execution-profile.js";
import { generateWithRuntimeSession, resolveModelRuntimeEngine } from "../../runtime/index.js";
import { loadModelRegistry } from "../model-registry.js";
import { findCatalogModel, getActiveModel } from "../download-manager.js";
import { loadTokenizerDefinition } from "../../runtime/tokenizer/index.js";
import { resolveCliExecutable, resolveCompletionExecutable } from "../engine-adapter.js";
import {
  isOllamaReachable,
  listInstalledOllamaModels,
  pullOllamaModel,
  streamOllamaChat,
  readOllamaManifest,
} from "../ollama.js";
import { kvBytesPerToken } from "./context-budget.js";
import { NATIVE_STOP_STRINGS, INSTRUCTION_STOP_STRINGS } from "../../runtime/core/stop-sequences.js";
import { resolveNativeConcurrencyPolicy } from "../../runtime/concurrency-policy.js";
import { findNativeHttpPerformance } from "../native-http-performance.js";
import {
  CATERPILLAR_TTFT_PROFILE,
  applyCaterpillarPerformanceProfile,
  caterpillarPerformanceProfileFromOptions,
} from "../../runtime/caterpillar-performance-profile.js";

export const CHAT_ENGINES = Object.freeze([
  {
    id: "lizard-native",
    label: "Lizard Native",
    summary: "Lizard's own engine. Streams tokens, keeps the model resident between messages.",
    streams: true,
    kind: "native",
  },
  {
    id: "caterpillar",
    label: "Caterpillar",
    summary: "Graph-plan engine with GPU-resident decode. Fastest on some integrated GPUs, experimental.",
    streams: true,
    kind: "native",
  },
  {
    id: "ostrich",
    label: "Ostrich",
    summary: "Experimental fork of Caterpillar with graph-level worker dispatch. Admin-gated; Caterpillar remains the stable engine.",
    streams: true,
    kind: "native",
    requiresExperimentalAccess: true,
  },
  {
    id: "llama.cpp",
    label: "llama.cpp",
    summary: "The stock upstream runtime. Returns the whole answer at once rather than streaming it.",
    streams: false,
    kind: "llamacpp",
  },
  {
    id: "ollama",
    label: "Ollama",
    summary: "Hands the model to your local Ollama daemon. Pulls it first if Ollama does not have it.",
    streams: true,
    kind: "ollama",
  },
]);

export const DEFAULT_CHAT_ENGINE = "lizard-native";

export function normalizeChatEngine(value) {
  const candidate = String(value ?? "").trim().toLowerCase();
  const match = CHAT_ENGINES.find((engine) => engine.id.toLowerCase() === candidate);
  return match?.id ?? DEFAULT_CHAT_ENGINE;
}

export function getChatEngine(value) {
  const id = normalizeChatEngine(value);
  return CHAT_ENGINES.find((engine) => engine.id === id);
}

// ---- availability ---------------------------------------------------------

// Each Lizard engine is its own binary under its own config key. A provider
// missing here resolves to lizard-native-runner and reports itself available
// whenever that unrelated binary exists.
const CHAT_RUNNER_BY_PROVIDER = {
  caterpillar: { key: "caterpillar", name: "caterpillar-runner" },
  ostrich: { key: "ostrich", name: "ostrich-runner" },
};

function nativeRunnerExists(config, provider) {
  const runner = CHAT_RUNNER_BY_PROVIDER[provider] ?? { key: "native", name: "lizard-native-runner" };
  const providerKey = runner.key;
  const runnerName = runner.name;
  const executable = process.platform === "win32" ? `${runnerName}.exe` : runnerName;
  const candidates = [];
  const configured = config?.engine?.[providerKey]?.runnerBinary;
  if (configured) {
    candidates.push(configured);
  }
  const buildDirectory = config?.engine?.[providerKey]?.buildDirectory;
  if (buildDirectory) {
    candidates.push(
      path.join(buildDirectory, executable),
      path.join(buildDirectory, "Release", executable),
    );
  }
  if (config?.runtimeDirectory) {
    candidates.push(path.join(config.runtimeDirectory, executable));
  }
  // The Windows installer stages the compiled workers at {app}\bin only.
  candidates.push(path.resolve(process.cwd(), "bin", executable));
  try {
    const appRoot = path.dirname(path.dirname(new URL("../../", import.meta.url).pathname.replace(/^\//, "")));
    candidates.push(path.join(appRoot, "bin", executable));
  } catch {
    // best effort - the cwd candidate above covers the installed layout
  }
  return candidates.some((candidate) => {
    try {
      return Boolean(candidate) && fs.existsSync(candidate);
    } catch {
      return false;
    }
  });
}

function llamaCppExists(config) {
  for (const resolve of [resolveCliExecutable, resolveCompletionExecutable]) {
    try {
      const executable = resolve(config, null);
      if (executable && fs.existsSync(executable)) {
        return true;
      }
    } catch {
      // fall through to the next candidate
    }
  }
  return false;
}

/**
 * Is the bound model actually built for conversation?
 *
 * A GGUF that carries no chat template is almost always a BASE model - a text
 * completer, not an assistant. Lizard can structure the prompt for one (see
 * renderInstructionPrompt) and it will stop cleanly, but it will still answer
 * a greeting by repeating the greeting. That is the model, not the app, and
 * saying so beats letting someone conclude Lizard is broken.
 */
export function describeModelChatReadiness(config, modelId) {
  if (!modelId) {
    return { kind: "unknown", chatTemplate: false, note: null };
  }
  // No TTL: whether a model carries a chat template is a property of the file,
  // not of the clock. Reading it costs ~2s (tokenizer + catalog parse) and
  // resolveBoundModel runs on every turn and every state poll.
  const cached = readinessCache.get(modelId);
  if (cached) {
    return cached.value;
  }
  const value = computeModelChatReadiness(config, modelId);
  readinessCache.set(modelId, { at: Date.now(), value });
  return value;
}

function computeModelChatReadiness(config, modelId) {
  const entry = findCatalogModel(config, modelId);
  const modelPath = entry?.importedPath ?? entry?.path ?? entry?.modelPath ?? null;
  let chatTemplate = false;
  try {
    chatTemplate = Boolean(loadTokenizerDefinition({ modelPath, gguf: entry?.gguf ?? null })?.chatTemplate);
  } catch {
    // An unreadable tokenizer is not worth failing the state request over.
  }
  const namedInstruct = /(instruct|chat|-it\b|_it\b|sft|dpo|tulu|hermes|vicuna)/i.test(modelId);
  if (chatTemplate || namedInstruct) {
    return { kind: "instruct", chatTemplate, note: null };
  }
  return {
    kind: "base",
    chatTemplate,
    note: `${modelId} is a base model — it completes text rather than answering questions, so replies will be `
      + "short and literal. Activate an “instruct” build of the same model in onboarding for real conversation.",
  };
}

/** The Ollama tag this model would use, whether or not it is pulled yet. */
export function ollamaTagForModel(config, modelId) {
  const entry = findCatalogModel(config, modelId);
  if (entry?.ollamaTag) {
    return entry.ollamaTag;
  }
  // Fall back to the model id itself: Lizard's ids follow the same
  // "family-size-variant" shape Ollama uses, so "llama-3.2-3b-instruct"
  // resolves to a real library tag often enough to be worth trying, and the
  // pull reports a clear error when it does not.
  const id = String(modelId ?? "").trim();
  if (!id) {
    return null;
  }
  return id.includes(":") ? id : `${id}:latest`;
}

export function isOllamaModelPresent(config, tag) {
  if (!tag) {
    return false;
  }
  try {
    if (readOllamaManifest(tag, config)) {
      return true;
    }
  } catch {
    // manifest directory may not exist
  }
  const bare = tag.replace(/:latest$/, "");
  return listInstalledOllamaModels().some(
    (model) => model.tag === tag || model.tag === `${bare}:latest` || model.tag === bare,
  );
}

// Engine availability and model readiness are stable between messages but
// expensive to compute (a tokenizer/GGUF load, an Ollama probe, filesystem
// checks, a catalog read - ~3s together). The chat state endpoint is polled
// every few seconds, so without a cache that cost is paid on every tick and
// shows up as the "10 seconds before anything appears" the user reported.
// Cache briefly and let activation / an engine switch bust it explicitly.
const PROBE_CACHE_TTL_MS = 20_000;
let enginesCache = null;
let readinessCache = new Map();

// Resolving the execution context - hardware scan, model artifact, capability
// registry, backend selection - measured 10.3s, and it ran on EVERY turn. None
// of it depends on the message: same model, same machine, same answer. That
// wait was the "ten seconds before the model starts" the user was watching.
//
// Deliberately NOT time-expired. These are facts about the activated model and
// this machine; they change when the model is activated, not on a clock. A TTL
// here was actively harmful: a caterpillar turn takes 50-90s, so a 120s window
// expired mid-conversation and the NEXT question paid an 18s re-resolve before
// anything happened - which is precisely what "the follow-up gets stuck" was.
// invalidateEngineProbeCache() is called on activation.
let turnRuntimeCache = new Map();

export function invalidateEngineProbeCache() {
  enginesCache = null;
  readinessCache = new Map();
  turnRuntimeCache = new Map();
}

/**
 * Which engines this machine can actually run right now, and why not when it
 * cannot. Reported up front so the picker never offers something that fails
 * only once the user has typed a message.
 */
export async function listChatEngines(config, { force = false } = {}) {
  const active = getActiveModel(config);
  const cacheKey = active?.id ?? "none";
  if (!force && enginesCache && enginesCache.key === cacheKey
    && Date.now() - enginesCache.at < PROBE_CACHE_TTL_MS) {
    return enginesCache.value;
  }
  const value = await computeChatEngines(config, active);
  enginesCache = { key: cacheKey, at: Date.now(), value };
  return value;
}

async function computeChatEngines(config, active) {
  const ollamaUp = await isOllamaReachable();
  const tag = active ? ollamaTagForModel(config, active.id) : null;

  return CHAT_ENGINES.map((engine) => {
    let available = true;
    let reason = null;
    let needsDownload = false;

    if (engine.kind === "native") {
      available = nativeRunnerExists(config, engine.id);
      if (!available) {
        // Name the engine. With three native engines in the list, "the native
        // runner is not installed" does not say which one the user just tried.
        reason = `The ${engine.id} runner is not installed with this build.`;
      }
    } else if (engine.kind === "llamacpp") {
      available = llamaCppExists(config);
      if (!available) {
        reason = "llama.cpp was not found. Install it with: winget install ggml.llamacpp";
      }
    } else if (engine.kind === "ollama") {
      available = ollamaUp;
      if (!available) {
        reason = "Ollama is not running. Start it, then reopen this menu.";
      } else if (tag && !isOllamaModelPresent(config, tag)) {
        needsDownload = true;
        reason = `Ollama does not have ${tag} yet — switching will download it first.`;
      }
    }

    return {
      ...engine,
      available,
      reason,
      needsDownload,
      ...(engine.kind === "ollama" ? { tag } : {}),
    };
  });
}

// ---- shared execution context --------------------------------------------

/**
 * Resolves everything a turn needs from the runtime layer once: the model
 * artifact, the profile, the GGUF geometry that sizes the KV cache, and the
 * engine actually chosen (which can differ from the one requested when a model
 * is outside an experimental engine's supported set).
 */
export async function resolveTurnRuntime({ config, logger, engineId, prompt = "warmup" }) {
  const engine = getChatEngine(engineId);
  // The prompt is deliberately NOT part of the key. It only feeds the profile
  // builder's sizing hints, and the chat overrides context size, timeout and
  // stop sequences per turn in buildTurnProfile anyway - so the resolved
  // artifact, recommendation and capability registry are identical whatever
  // the user typed.
  const activeId = getActiveModel(config)?.id ?? "none";
  const cacheKey = `${activeId}::${engine.id}`;
  const cached = turnRuntimeCache.get(cacheKey);
  if (cached) {
    return cached.value;
  }

  const hardware = scanHardware({ config, logger });
  const registry = loadModelRegistry();
  const context = await resolveExecutionContext({
    command: "generate",
    // backend/gpu-layers mirror `serve` and /api/quick-chat: they steer the
    // profile builder away from a llama.cpp GPU backend that rejects
    // gpuLayers = 0. Native GPU-resident decode is gated separately.
    options: { model: "active", prompt, mode: config.defaultMode, backend: "cpu", "gpu-layers": "0" },
    config,
    registry,
    hardware,
    logger,
  });
  const requestedEngine = engine.kind === "ollama" ? "llama.cpp" : engine.id;
  const resolution = resolveModelRuntimeEngine({
    requestedEngine,
    modelArtifact: context.modelArtifact,
    profile: context.profile,
  });
  const gguf = context.profile?.gguf ?? context.modelArtifact?.gguf ?? null;
  // Stop strings for the native decode loop, so it stops when the turn ends
  // instead of decoding the whole budget. The explicit turn-enders are always
  // safe; the instruction-format headers are added only for base models (no
  // chat template), where "### " is the model continuing our own format rather
  // than a Markdown heading in a real answer.
  const readiness = describeModelChatReadiness(config, context.modelArtifact.modelId);
  const stopSequences = readiness.kind === "base"
    ? [...NATIVE_STOP_STRINGS, ...INSTRUCTION_STOP_STRINGS]
    : [...NATIVE_STOP_STRINGS];
  const value = {
    engine,
    context,
    resolution,
    gguf,
    stopSequences,
    bytesPerToken: kvBytesPerToken(gguf),
    modelContextTokens: context.profile?.contextSize ?? gguf?.contextLength ?? 4096,
    availableMemoryGb: context.profile?.capabilityRegistry?.memory?.availableGb ?? null,
    measuredHttpPerformance: findNativeHttpPerformance(config, {
      runtimeProvider: resolution.engine,
      modelId: context.modelArtifact.modelId,
      quant: gguf?.quantization ?? gguf?.quant ?? null,
      contextTokens: context.profile?.contextSize ?? null,
      optimizationProfile: resolution.engine === "caterpillar"
        ? caterpillarPerformanceProfileFromOptions({}, {
          defaultProfile: CATERPILLAR_TTFT_PROFILE,
        })
        : null,
    }),
  };
  turnRuntimeCache.set(cacheKey, { at: Date.now(), value });
  return value;
}

/** Profile for one turn: the resolved profile plus this turn's context size. */
export function buildTurnProfile(turnRuntime, { contextSize, maxTokens = 512 } = {}) {
  const base = turnRuntime.context.profile;
  const sizedProfile = {
    ...base,
    ...(Number.isFinite(contextSize) && contextSize > 0 ? { contextSize } : {}),
  };
  const concurrencyPolicy = resolveNativeConcurrencyPolicy({
    runtimeProvider: turnRuntime.resolution.engine,
    profile: sizedProfile,
    gguf: turnRuntime.gguf,
    measuredBatchLevels: turnRuntime.measuredHttpPerformance?.levels ?? null,
  });
  const turnProfile = {
    ...sizedProfile,
    // Handed to the native worker's decode loop so it stops at the end of the
    // turn instead of running the whole token budget.
    stopSequences: turnRuntime.stopSequences ?? [],
    // Lets Caterpillar keep the previous turn's KV and ingest only the new
    // tail. Chat-only on purpose: a benchmark re-running the same prompt would
    // hit the cache and report a prefill time it did not actually pay.
    promptCache: true,
    // A finite, generous ceiling rather than none. Long streamed answers are
    // legitimate - the scheduler's stock 120s timeout is sized for CLI calls
    // and used to kill working generations - but no timeout at all means a
    // wedged engine (e.g. a Caterpillar prefill that never produces a token)
    // shows "thinking" forever. Budget for a slow ~2 tok/s decode of the whole
    // answer plus a cold-load allowance, so a real answer always finishes and a
    // genuine hang surfaces as an error instead.
    timeoutMs: Math.max(180_000, Math.round(maxTokens * 900) + 120_000),
    // Models with no chat template in their GGUF - base (non-instruct) builds -
    // otherwise receive a bare passage and complete it rather than answering.
    chatMode: true,
    nativeRuntime: {
      // Chat opts into batched decode, but the runtime routes it adaptively: a
      // lone turn takes the single-shot path and only a turn that starts while
      // the worker is already busy joins the coordinator. So two conversations
      // answering at once share one weight read per step, and the far more
      // common case -- one person, one question -- pays nothing for the option.
      batchedDecode: true,
      // Exact retries and repeated prompts reuse an isolated native prefill
      // snapshot. Each turn receives its own writable KV copy, so conversation
      // state cannot leak between chats.
      prefillSnapshotCache: true,
      ...(base.nativeRuntime ?? {}),
      preferredBatchSize: concurrencyPolicy.preferredBatchSize,
      batchTargetSize: concurrencyPolicy.selectedBatchSize,
      batchFallbackOrder: concurrencyPolicy.fallbackOrder,
      concurrencyPolicy,
      keepWorkerLoaded: true,
    },
  };
  return applyCaterpillarPerformanceProfile(turnProfile, {
    runtimeProvider: turnRuntime.resolution.engine,
    // Chat is a B=1 user surface. Caterpillar's newest latency profile is the
    // default here; operators can restore balanced batching with the process
    // environment when comparing the two modes.
    performanceProfile: caterpillarPerformanceProfileFromOptions({}, {
      defaultProfile: CATERPILLAR_TTFT_PROFILE,
    }),
  });
}

// ---- running a turn -------------------------------------------------------

/**
 * One generation, on whichever engine is selected. Returns the same shape
 * regardless of engine so the chat route does not branch on it.
 */
export async function runEngineTurn({
  config,
  logger,
  turnRuntime,
  prompt,
  systemPrompt = null,
  maxTokens,
  temperature,
  topP,
  contextSize,
  getSharedRuntime,
  onDelta,
  signal = null,
}) {
  const { engine } = turnRuntime;

  if (engine.kind === "ollama") {
    const active = getActiveModel(config);
    const tag = ollamaTagForModel(config, active?.id);
    if (!tag) {
      throw new Error("No model is active, so there is nothing to run on Ollama.");
    }
    const result = await streamOllamaChat({
      tag,
      prompt,
      systemPrompt,
      maxTokens,
      temperature,
      topP,
      signal,
      onDelta,
    });
    return {
      output: result.output,
      modelId: tag,
      runtimeProvider: "ollama",
      nativeBackend: null,
      tokensPerSecond: result.tokensPerSecond,
      durationMs: result.durationMs,
    };
  }

  // Caterpillar only supports deterministic (temperature 0, greedy) sampling -
  // its speculative decode path is exact only for argmax - and rejects anything
  // else at generate time, which is the "Caterpillar currently accepts
  // deterministic temperature=0 sampling" error the user hit after a long wait.
  // Force greedy for it rather than letting a workspace's temperature fail the
  // whole turn.
  // Ostrich is a fork of that same engine and rejects the same way.
  const isCaterpillarFamily = ["caterpillar", "ostrich"].includes(turnRuntime.resolution.engine);
  const effectiveTemperature = isCaterpillarFamily ? 0 : temperature;
  const effectiveTopP = isCaterpillarFamily ? 1 : topP;

  // Sampling is configuration rather than a runtime argument (see
  // runtime/index.js), so per-workspace values ride in on a config clone.
  const samplingConfig = {
    ...config,
    generation: { ...(config.generation ?? {}), temperature: effectiveTemperature, topP: effectiveTopP },
  };
  const usesNativeWorker = ["lizard-native", "caterpillar", "ostrich"]
    .includes(turnRuntime.resolution.engine);

  return generateWithRuntimeSession({
    requestedEngine: turnRuntime.resolution.engine,
    runtime: usesNativeWorker ? getSharedRuntime(turnRuntime.resolution.engine) : null,
    recommendation: turnRuntime.context.recommendation,
    modelArtifact: turnRuntime.context.modelArtifact,
    profile: buildTurnProfile(turnRuntime, { contextSize, maxTokens }),
    prompt,
    // Handed to the model's own chat template rather than pasted in front of
    // the question, so the model sees a system turn and not a document.
    systemPrompt,
    maxTokens,
    config: samplingConfig,
    logger,
    onDelta,
  });
}

// ---- warming --------------------------------------------------------------

/**
 * Gets an engine ready to answer, reporting each phase so the UI can show
 * progress instead of a spinner. Ollama downloads the model here if it does
 * not have it; the native engines load the worker; llama.cpp has nothing to
 * warm because it spawns per request.
 *
 * `onProgress({ phase, message, percent })` is called throughout.
 */
export async function warmChatEngine({ config, logger, engineId, getSharedRuntime, onProgress, signal = null }) {
  const engine = getChatEngine(engineId);
  const report = (phase, message, extra = {}) => onProgress?.({ phase, message, ...extra });

  if (engine.kind === "ollama") {
    report("checking", "Checking that Ollama is running…");
    if (!await isOllamaReachable()) {
      throw new Error("Ollama is not running. Start Ollama and try again.");
    }
    const active = getActiveModel(config);
    const tag = ollamaTagForModel(config, active?.id);
    if (!tag) {
      throw new Error("No model is active, so there is nothing to hand to Ollama.");
    }
    if (!isOllamaModelPresent(config, tag)) {
      report("downloading", `Ollama does not have ${tag} yet. Downloading it…`, { percent: 0 });
      await pullOllamaModel(tag, {
        signal,
        onProgress: ({ status, completed, total }) => {
          const percent = Number.isFinite(completed) && Number.isFinite(total) && total > 0
            ? Math.round((completed / total) * 100)
            : null;
          report("downloading", `${status || "downloading"}${percent === null ? "" : ` — ${percent}%`}`, { percent });
        },
      });
    }
    report("warming", `Loading ${tag} into Ollama…`);
    // A tiny generation is the only reliable way to make Ollama resident; the
    // keep_alive on it is what holds the model for the conversation.
    await streamOllamaChat({ tag, prompt: "ok", maxTokens: 1, temperature: 0, signal });
    report("ready", `Ollama is holding ${tag}.`, { modelId: tag });
    return { engineId: engine.id, modelId: tag, warm: true };
  }

  if (engine.kind === "llamacpp") {
    report("checking", "Checking for llama.cpp…");
    if (!llamaCppExists(config)) {
      throw new Error("llama.cpp was not found. Install it with: winget install ggml.llamacpp");
    }
    // Nothing stays resident: llama.cpp is spawned per request. Saying so is
    // better than a fake "warming" step that changes nothing.
    report("ready", "llama.cpp runs per message — there is nothing to keep warm.", { transient: true });
    return { engineId: engine.id, modelId: getActiveModel(config)?.id ?? null, warm: false };
  }

  report("resolving", "Working out how this model fits on your hardware…");
  const turnRuntime = await resolveTurnRuntime({ config, logger, engineId: engine.id });
  if (turnRuntime.resolution.engine !== engine.id) {
    throw new Error(
      turnRuntime.resolution.fallbackReason
      ?? `${engine.label} cannot run ${turnRuntime.context.modelArtifact.modelId}.`,
    );
  }
  report("loading", `Loading ${turnRuntime.context.modelArtifact.modelId} into ${engine.label}…`);
  await getSharedRuntime(engine.id).ensureModelLoaded({
    recommendation: turnRuntime.context.recommendation,
    modelArtifact: turnRuntime.context.modelArtifact,
    profile: buildTurnProfile(turnRuntime, { contextSize: turnRuntime.modelContextTokens }),
    config,
  });
  report("ready", `${engine.label} is holding ${turnRuntime.context.modelArtifact.modelId}.`, {
    modelId: turnRuntime.context.modelArtifact.modelId,
  });
  return {
    engineId: engine.id,
    modelId: turnRuntime.context.modelArtifact.modelId,
    modelArtifact: turnRuntime.context.modelArtifact,
    warm: true,
  };
}
