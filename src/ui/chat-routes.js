// HTTP surface for the Lizard Chat workspace (/api/chat/*).
//
// Split out of server.js because it is a self-contained feature: conversations,
// workspaces, a document library, a prompt library and MCP plugins. The one
// thing it does NOT own is the model. Chat always runs on the activated model,
// resolved the same way `lizard-llm generate` resolves it, and the only place
// that changes is onboarding's Activate step. Everything below therefore takes
// the bound model as a given and never accepts a model id from the client.
//
// The ENGINE running that model is selectable (lizard-native, caterpillar,
// llama.cpp, Ollama) - see src/lib/chat/engine-runtime.js.

import { getActiveModel } from "../lib/download-manager.js";
import { setModelResidencyIntent } from "../lib/model-lifecycle.js";
import { buildQuickChatExecutionTelemetry } from "../lib/native-execution-telemetry.js";
import { extractEvidenceMetrics } from "../lib/evidence-metrics.js";
import {
  appendNativeTurnMetric,
  contextBucket,
  listNativeTurnMetrics,
  summarizeNativeTurnMetrics,
} from "../lib/native-turn-history.js";
import { incidentUserFields } from "../lib/major-failure-intelligence.js";
import {
  listChatEngines,
  normalizeChatEngine,
  getChatEngine,
  resolveTurnRuntime,
  runEngineTurn,
  warmChatEngine,
  describeModelChatReadiness,
  DEFAULT_CHAT_ENGINE,
} from "../lib/chat/engine-runtime.js";
import { planTurnBudget } from "../lib/chat/context-budget.js";
import { resolveKvBudgetBytes } from "../runtime/kv-cache/index.js";
import {
  CATERPILLAR_TTFT_PROFILE,
  caterpillarPerformanceProfileFromOptions,
} from "../runtime/caterpillar-performance-profile.js";
import {
  listWorkspaces,
  getWorkspace,
  saveWorkspace,
  deleteWorkspace,
  createConversation,
  getConversation,
  listConversations,
  deleteConversation,
  updateConversation,
  appendMessage,
  truncateFrom,
  recentTurns,
  exportConversationMarkdown,
  writeConversation,
} from "../lib/chat/chat-store.js";
import {
  addDocument,
  listDocuments,
  deleteDocument,
  getDocumentText,
  retrieve,
  openingChunks,
  formatContextBlock,
  SUPPORTED_EXTENSIONS,
} from "../lib/chat/knowledge-store.js";
import {
  listPrompts,
  savePrompt,
  deletePrompt,
  expandPrompt,
} from "../lib/chat/prompt-library.js";
import {
  listPlugins,
  addPlugin,
  setPluginEnabled,
  removePlugin,
  connectPlugin,
  collectTools,
  callTool,
  marketplaceView,
} from "../lib/chat/mcp-manager.js";
import { MCP_CATEGORIES } from "../lib/chat/mcp-catalog.js";
import {
  buildToolInstructions,
  parseToolCall,
  stripToolCalls,
  formatToolResult,
  composePrompt,
  createToolCallStreamGate,
  MAX_TOOL_ITERATIONS,
} from "../lib/chat/tool-protocol.js";
import {
  looksLikeChartRequest,
  suggestChartType,
  buildClassifierPrompt,
  parseClassifierReply,
  buildChartInstruction,
  extractChartSpec,
  normalizeChartSpec,
} from "../lib/chat/chart-agent.js";

// Reasoning-tuned models wrap their scratchpad in <think>. The workspace shows
// it in a fold, so it is separated here rather than stripped.
const THINK_PATTERN = /<(think|thinking|reasoning)>([\s\S]*?)<\/\1>/gi;

// Chat templates end a turn with a control token. When a model emits one
// literally instead of the runtime consuming it, it lands in the transcript as
// visible junk ("...What would you like to know?<|eot_id|>"). Strip the ones
// the common families use rather than showing tokenizer internals to the user.
const CONTROL_TOKEN_PATTERN =
  /<\|(?:eot_id|end_of_text|endoftext|im_end|im_start|start_header_id|end_header_id|eom_id)\|>|<\/s>|<end_of_turn>/g;

export function stripControlTokens(text) {
  return String(text ?? "").replace(CONTROL_TOKEN_PATTERN, "").trim();
}

export function splitReasoning(text) {
  const source = String(text ?? "");
  const parts = [];
  let match;
  THINK_PATTERN.lastIndex = 0;
  while ((match = THINK_PATTERN.exec(source)) !== null) {
    parts.push(match[2].trim());
  }
  return {
    reasoning: stripControlTokens(parts.join("\n\n")),
    answer: stripControlTokens(source.replace(THINK_PATTERN, "")),
  };
}

function sendJson(res, statusCode, payload) {
  const safeStatus = Number.isInteger(statusCode) && statusCode >= 100 && statusCode <= 599 ? statusCode : 500;
  res.writeHead(safeStatus, {
    "Content-Type": "application/json; charset=utf-8",
    "Cache-Control": "no-store",
  });
  res.end(JSON.stringify(payload));
}

function readBody(req, limitBytes = 12 * 1024 * 1024) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    let size = 0;
    req.on("data", (chunk) => {
      size += chunk.length;
      if (size > limitBytes) {
        reject(new Error("That payload is too large. Documents are capped at 12 MB."));
        req.destroy();
        return;
      }
      chunks.push(chunk);
    });
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

// ---- Bound model ----------------------------------------------------------

/** What the header chip states, and the only model chat will ever use. */
export function resolveBoundModel(config) {
  const active = getActiveModel(config);
  if (!active) {
    return {
      bound: false,
      modelId: null,
      message: "No model is active yet. Finish onboarding — Scan, Select, Download, Benchmark, Activate — and chat binds to it.",
    };
  }
  const readiness = describeModelChatReadiness(config, active.id);
  return {
    bound: true,
    modelId: active.id,
    quantization: active.quantization ?? null,
    runtimeProvider: active.runtimeProvider ?? "lizard-native",
    kind: readiness.kind,
    note: readiness.note,
    ...describeVision(active),
    message: `Bound to ${active.id}. Change it in onboarding → Activate.`,
  };
}

/**
 * Can this model look at a picture, and can this build actually show it one?
 *
 * Two separate answers, because they are separately true. A Gemma 3 or a
 * Qwen-VL genuinely accepts images - `sees` says so - but the native runner has
 * no projector path yet (see .agent/vision/01_native_vision_plan.md), so
 * `analyzes` is what decides whether an attached image is sent or only held.
 * Collapsing them would make the UI claim an answer looked at a picture it
 * never received, which is the one failure this whole feature must not have.
 */
function describeVision(active) {
  const gguf = active?.gguf ?? {};
  const tensors = Array.isArray(gguf.tensorsPreview)
    ? gguf.tensorsPreview.map((tensor) => String(tensor?.name ?? tensor).toLowerCase())
    : [];
  const architecture = String(gguf.architecture ?? "").toLowerCase();
  const id = String(active?.id ?? "").toLowerCase();
  // A multimodal projector in the weights is proof. The family patterns are the
  // fallback for a catalog entry recorded before tensor previews were kept.
  const sees = tensors.some((name) => name.includes("mmproj") || name.includes("v.blk") || name.includes("clip"))
    || /vl|vision|llava|pixtral|gemma-?3|gemma3/.test(architecture)
    || /vl|vision|llava|pixtral|moondream|internvl|gemma-?3/.test(id);
  return {
    sees,
    // Flipped on by phase 5 of the native vision plan. Until then no runtime
    // Lizard drives can consume an image, so nothing claims to.
    analyzesImages: false,
    visionNote: sees
      ? "This model accepts images, but Lizard cannot feed it one yet — images stay attached and only your text is sent."
      : "This model reads text, not pictures.",
  };
}

// ---- Engine selection -----------------------------------------------------

// Which engine chat is running on. Kept in memory rather than per workspace:
// it describes this machine's current runtime, not the conversation, and it
// must survive switching workspaces mid-session.
let selectedEngineId = null;

// Turns currently generating. An engine switch releases the outgoing engine's
// workers, which would pull the model out from under a turn that is mid-decode
// and surface as a failed answer the user did not cause. The switch waits (or
// refuses) instead.
let turnsInFlight = 0;

// Largest context size each conversation has used, so a later turn never asks
// for a smaller one. The native prefill snapshot that makes turn N+1 pay only
// for its new tail is keyed to a KV shape; a context size that shrinks for a
// short follow-up and grows again for the next question invalidates it every
// other turn. Process-local and bounded - a lost entry costs one re-prefill,
// never a wrong answer.
const conversationContextHighWater = new Map();
const CONTEXT_HIGH_WATER_MAX = 64;

function contextFloorFor(conversationId) {
  return conversationContextHighWater.get(conversationId) ?? 0;
}

function recordContextSize(conversationId, contextSize) {
  if (!conversationId || !Number.isFinite(contextSize) || contextSize <= 0) return;
  if (contextSize <= contextFloorFor(conversationId)) return;
  // Re-inserting moves the key to the end, so the delete below evicts the
  // least recently grown conversation.
  conversationContextHighWater.delete(conversationId);
  conversationContextHighWater.set(conversationId, contextSize);
  while (conversationContextHighWater.size > CONTEXT_HIGH_WATER_MAX) {
    conversationContextHighWater.delete(conversationContextHighWater.keys().next().value);
  }
}

export function getSelectedEngineId(config = null) {
  if (selectedEngineId) return selectedEngineId;
  const activatedProvider = getActiveModel(config)?.runtimeProvider;
  return normalizeChatEngine(activatedProvider ?? DEFAULT_CHAT_ENGINE);
}

export function selectChatEngineForActivatedModel(runtimeProvider) {
  selectedEngineId = normalizeChatEngine(runtimeProvider ?? DEFAULT_CHAT_ENGINE);
  return selectedEngineId;
}

// ---- Router ---------------------------------------------------------------

/**
 * Handles every /api/chat/* route. Returns true when the request was ours.
 * `getSharedRuntime(engineId)` is injected so chat reuses server.js's warm
 * runtimes instead of loading a second copy of the model, and
 * `getWarmState`/`setWarmState` keep chat and onboarding agreeing on what is
 * currently resident.
 */
export async function handleChatApi({
  req,
  res,
  url,
  config,
  logger,
  getSharedRuntime,
  getWarmState = () => ({ warm: false, modelId: null, engineId: null }),
  setWarmState = () => {},
  reportFailure = async () => ({ reported: false }),
}) {
  const { pathname } = url;
  if (!pathname.startsWith("/api/chat/")) {
    return false;
  }

  // Asking for an answer is asking for the weights. Freeing memory sets a latch
  // that stops the automatic loaders (boot resume, the page-load warm stream)
  // from putting the model back on their own; a real request lifts it, so chat
  // never has to explain why it will not answer.
  if (req.method === "POST" && (pathname === "/api/chat/turn" || pathname.startsWith("/api/chat/warm"))) {
    try {
      setModelResidencyIntent(config, "resident", { reason: "chat" });
    } catch { /* state file unavailable: the turn still runs */ }
  }

  try {
    // ---- state ----------------------------------------------------------
    if (req.method === "GET" && pathname === "/api/chat/state") {
      const workspaces = listWorkspaces(config);
      const warm = getWarmState();
      const bound = resolveBoundModel(config);
      sendJson(res, 200, {
        boundModel: bound,
        // Warmth only counts when it is THIS model on THIS engine. Reporting a
        // stale warm model as warm is how the header ends up claiming "ready"
        // while the first message quietly reloads several GB of weights.
        warm: {
          ...warm,
          matchesBound: Boolean(warm.warm && bound.modelId && warm.modelId === bound.modelId),
          engineId: warm.engineId ?? null,
        },
        engine: getSelectedEngineId(config),
        engines: await listChatEngines(config),
        workspaces,
        conversations: listConversations(config),
        prompts: listPrompts(config),
        plugins: listPlugins(config),
        supportedExtensions: SUPPORTED_EXTENSIONS,
      });
      return true;
    }

    // ---- engine ---------------------------------------------------------
    if (req.method === "GET" && pathname === "/api/chat/engines") {
      sendJson(res, 200, { engine: getSelectedEngineId(config), engines: await listChatEngines(config) });
      return true;
    }
    if (req.method === "GET" && pathname === "/api/chat/performance") {
      const entries = listNativeTurnMetrics(config, { userId: "local-chat", limit: 50 });
      const caterpillarProfile = caterpillarPerformanceProfileFromOptions({}, {
        defaultProfile: CATERPILLAR_TTFT_PROFILE,
      });
      // Ostrich is its own binary and reports its own rows; leaving it out of
      // this list silently dropped every Ostrich turn from the breakdown.
      const providers = ["lizard-native", "caterpillar", "ostrich", "llama.cpp", "ollama"];
      const byProvider = Object.fromEntries(providers.map((provider) => [
        provider,
        summarizeNativeTurnMetrics(entries.filter((entry) => entry.runtimeProvider === provider
          && (provider !== "caterpillar"
            || (entry.optimizationProfile ?? "balanced") === caterpillarProfile))),
      ]));
      sendJson(res, 200, {
        summary: summarizeNativeTurnMetrics(entries),
        byProvider,
        entries,
      });
      return true;
    }

    // Preloads the selected engine's model in the background so the first
    // message is not a cold load (~20s here). Fire-and-forget: returns
    // immediately and the state endpoint reports warmth once it is ready.
    if (req.method === "POST" && pathname === "/api/chat/warm") {
      warmSelectedEngineInBackground({ config, logger, getSharedRuntime, setWarmState });
      sendJson(res, 202, { ok: true, engine: getSelectedEngineId(config) });
      return true;
    }

    // Streaming warm: loads and warms the selected engine while reporting each
    // phase, so the chat can hold a "getting your model ready" gate open until
    // it is genuinely usable. Unlike the switch, it does not change engines.
    if (req.method === "POST" && pathname === "/api/chat/warm/stream") {
      await streamWarm({ req, res, config, logger, getSharedRuntime, getWarmState, setWarmState });
      return true;
    }

    // Switching engines is a long operation - it can release a resident model,
    // download several GB from Ollama and load weights again - so it streams
    // its progress instead of blocking on one opaque request.
    if (req.method === "POST" && pathname === "/api/chat/engine/switch") {
      const body = await readBody(req);
      await streamEngineSwitch({
        req,
        res,
        config,
        logger,
        engineId: normalizeChatEngine(body.engine),
        getSharedRuntime,
        getWarmState,
        setWarmState,
      });
      return true;
    }

    // ---- workspaces -----------------------------------------------------
    if (req.method === "POST" && pathname === "/api/chat/workspaces") {
      const body = await readBody(req);
      sendJson(res, 200, { workspace: saveWorkspace(config, body) });
      return true;
    }
    if (req.method === "POST" && pathname === "/api/chat/workspaces/delete") {
      const body = await readBody(req);
      sendJson(res, 200, deleteWorkspace(config, body.id));
      return true;
    }

    // ---- conversations --------------------------------------------------
    if (req.method === "POST" && pathname === "/api/chat/conversations") {
      const body = await readBody(req);
      sendJson(res, 200, { conversation: createConversation(config, body) });
      return true;
    }
    if (req.method === "GET" && pathname === "/api/chat/conversation") {
      const conversation = getConversation(config, url.searchParams.get("id"));
      if (!conversation) {
        sendJson(res, 404, { error: "Conversation not found." });
        return true;
      }
      sendJson(res, 200, { conversation });
      return true;
    }
    if (req.method === "POST" && pathname === "/api/chat/conversations/update") {
      const body = await readBody(req);
      sendJson(res, 200, { conversation: updateConversation(config, body.id, body) });
      return true;
    }
    if (req.method === "POST" && pathname === "/api/chat/conversations/delete") {
      const body = await readBody(req);
      sendJson(res, 200, deleteConversation(config, body.id));
      return true;
    }
    if (req.method === "POST" && pathname === "/api/chat/conversations/truncate") {
      const body = await readBody(req);
      sendJson(res, 200, {
        conversation: truncateFrom(config, body.id, body.messageId, { inclusive: body.inclusive !== false }),
      });
      return true;
    }
    if (req.method === "GET" && pathname === "/api/chat/conversation/export") {
      const conversation = getConversation(config, url.searchParams.get("id"));
      if (!conversation) {
        sendJson(res, 404, { error: "Conversation not found." });
        return true;
      }
      const format = url.searchParams.get("format") === "json" ? "json" : "markdown";
      const workspace = getWorkspace(config, conversation.workspaceId);
      const filename = `${conversation.title.replace(/[^\w -]/g, "").trim() || "chat"}.${format === "json" ? "json" : "md"}`;
      const body = format === "json"
        ? JSON.stringify({ conversation, workspace: { id: workspace.id, name: workspace.name } }, null, 2)
        : exportConversationMarkdown(conversation, workspace);
      res.writeHead(200, {
        "Content-Type": format === "json" ? "application/json; charset=utf-8" : "text/markdown; charset=utf-8",
        "Content-Disposition": `attachment; filename="${filename}"`,
        "Cache-Control": "no-store",
      });
      res.end(body);
      return true;
    }

    // ---- documents ------------------------------------------------------
    if (req.method === "GET" && pathname === "/api/chat/documents") {
      sendJson(res, 200, { documents: listDocuments(config, url.searchParams.get("workspaceId")) });
      return true;
    }
    if (req.method === "POST" && pathname === "/api/chat/documents") {
      const body = await readBody(req);
      const workspace = getWorkspace(config, body.workspaceId);
      const document = addDocument(config, workspace.id, {
        name: body.name,
        text: body.text,
        source: body.source ?? "upload",
      });
      sendJson(res, 200, { document });
      return true;
    }
    if (req.method === "POST" && pathname === "/api/chat/documents/delete") {
      const body = await readBody(req);
      sendJson(res, 200, deleteDocument(config, body.workspaceId, body.id));
      return true;
    }
    if (req.method === "GET" && pathname === "/api/chat/document") {
      const document = getDocumentText(
        config,
        url.searchParams.get("workspaceId"),
        url.searchParams.get("id"),
      );
      if (!document) {
        sendJson(res, 404, { error: "Document not found." });
        return true;
      }
      sendJson(res, 200, { document });
      return true;
    }

    // ---- prompt library -------------------------------------------------
    if (req.method === "GET" && pathname === "/api/chat/prompts") {
      sendJson(res, 200, { prompts: listPrompts(config) });
      return true;
    }
    if (req.method === "POST" && pathname === "/api/chat/prompts") {
      const body = await readBody(req);
      sendJson(res, 200, { prompt: savePrompt(config, body) });
      return true;
    }
    if (req.method === "POST" && pathname === "/api/chat/prompts/delete") {
      const body = await readBody(req);
      sendJson(res, 200, deletePrompt(config, body.id));
      return true;
    }

    // ---- plugins / marketplace ------------------------------------------
    if (req.method === "GET" && pathname === "/api/chat/marketplace") {
      sendJson(res, 200, {
        categories: MCP_CATEGORIES,
        catalog: marketplaceView(config),
        installed: listPlugins(config),
      });
      return true;
    }
    if (req.method === "POST" && pathname === "/api/chat/plugins/add") {
      const body = await readBody(req);
      sendJson(res, 200, { plugin: addPlugin(config, body) });
      return true;
    }
    if (req.method === "POST" && pathname === "/api/chat/plugins/toggle") {
      const body = await readBody(req);
      sendJson(res, 200, { plugin: setPluginEnabled(config, body.id, body.enabled) });
      return true;
    }
    if (req.method === "POST" && pathname === "/api/chat/plugins/remove") {
      const body = await readBody(req);
      sendJson(res, 200, await removePlugin(config, body.id));
      return true;
    }
    if (req.method === "POST" && pathname === "/api/chat/plugins/connect") {
      const body = await readBody(req);
      const client = await connectPlugin(config, body.id);
      sendJson(res, 200, {
        ok: true,
        server: client.serverInfo,
        tools: client.tools.map((tool) => ({ name: tool.name, description: tool.description ?? "" })),
      });
      return true;
    }

    // ---- the turn --------------------------------------------------------
    if (req.method === "POST" && pathname === "/api/chat/turn") {
      const body = await readBody(req);
      await streamTurn({
        req,
        res,
        config,
        logger,
        body,
        getSharedRuntime,
        setWarmState,
        reportFailure,
      });
      return true;
    }

    sendJson(res, 404, { error: "Unknown chat route." });
    return true;
  } catch (error) {
    const incident = await reportFailure(error, {
      operation: "chat-api-request",
      route: pathname,
    });
    if (!res.headersSent) {
      sendJson(res, 400, { error: error.message, ...incidentUserFields(incident) });
    } else {
      res.end();
    }
    return true;
  }
}

// Background warm of the selected engine. Guarded so repeated chat opens (or a
// state poll racing an open) do not stack multiple loads of the same model.
let warmInFlight = false;

function warmSelectedEngineInBackground({ config, logger, getSharedRuntime, setWarmState }) {
  const engine = getChatEngine(getSelectedEngineId(config));
  // llama.cpp spawns per request; there is nothing to hold warm. Ollama warms
  // itself on first use and a background pull is a heavy surprise - leave it to
  // the explicit switch. Only the resident native engines benefit here.
  if (warmInFlight || engine.kind !== "native") {
    return;
  }
  warmInFlight = true;
  warmChatEngine({ config, logger, engineId: engine.id, getSharedRuntime })
    .then((result) => {
      if (result?.modelArtifact) {
        setWarmState({ modelArtifact: result.modelArtifact, engineId: engine.id });
      }
    })
    .catch((error) => logger?.warn?.("chat-warm-failed", { message: error.message }))
    .finally(() => { warmInFlight = false; });
}

/**
 * Loads and warms the selected engine, streaming each phase, so chat can hold
 * a "getting your model ready" gate open until the model can actually answer.
 * Activation does not keep a model resident, so opening chat after activating
 * still finds it cold - this is what makes that wait visible and bounded
 * instead of surfacing as a mysteriously slow first message.
 */
async function streamWarm({ req, res, config, logger, getSharedRuntime, getWarmState, setWarmState }) {
  res.writeHead(200, {
    "Content-Type": "text/event-stream; charset=utf-8",
    "Cache-Control": "no-store",
    Connection: "keep-alive",
    "X-Accel-Buffering": "no",
  });
  const emit = (event, data) => {
    if (res.writableEnded) return;
    res.write(`event: ${event}\ndata: ${JSON.stringify(data ?? {})}\n\n`);
  };
  const controller = new AbortController();
  req.on("close", () => controller.abort());
  const keepAlive = setInterval(() => {
    if (!res.writableEnded) res.write(": keep-alive\n\n");
  }, 15000);

  const engine = getChatEngine(getSelectedEngineId(config));
  try {
    const bound = resolveBoundModel(config);
    if (!bound.bound) {
      emit("error", { message: bound.message });
    } else if (getWarmState().warm && getWarmState().modelId === bound.modelId
      && getWarmState().engineId === engine.id) {
      // Already resident on this engine - nothing to wait for.
      emit("done", { engine: engine.id, label: engine.label, modelId: bound.modelId, warm: true });
    } else {
      const result = await warmChatEngine({
        config,
        logger,
        engineId: engine.id,
        getSharedRuntime,
        signal: controller.signal,
        onProgress: (progress) => emit("phase", progress),
      });
      if (result?.modelArtifact) {
        setWarmState({ modelArtifact: result.modelArtifact, engineId: engine.id });
      }
      emit("done", {
        engine: engine.id,
        label: engine.label,
        modelId: result.modelId,
        warm: result.warm,
      });
    }
  } catch (error) {
    logger?.warn?.("chat-warm-stream-failed", { engine: engine.id, message: error.message });
    emit("error", { message: error.message, engine: engine.id, label: engine.label });
  }
  clearInterval(keepAlive);
  res.end();
}

// ---- Switching engine -----------------------------------------------------

/**
 * Releases the outgoing engine, prepares the incoming one, and streams every
 * phase while it happens. The download of a missing Ollama model runs here too,
 * so a multi-gigabyte pull shows progress rather than a frozen dialog.
 *
 * The selection is committed only once the engine is genuinely ready: a failed
 * switch leaves chat on the engine that still works.
 */
async function streamEngineSwitch({
  req, res, config, logger, engineId, getSharedRuntime, getWarmState, setWarmState,
}) {
  res.writeHead(200, {
    "Content-Type": "text/event-stream; charset=utf-8",
    "Cache-Control": "no-store",
    Connection: "keep-alive",
    "X-Accel-Buffering": "no",
  });
  const emit = (event, data) => {
    if (res.writableEnded) return;
    res.write(`event: ${event}\ndata: ${JSON.stringify(data ?? {})}\n\n`);
  };
  const controller = new AbortController();
  req.on("close", () => controller.abort());

  const previous = getSelectedEngineId(config);
  const engine = getChatEngine(engineId);
  try {
    if (turnsInFlight > 0) {
      // Releasing the outgoing engine now would kill a reply that is still
      // being decoded. Say so rather than breaking that answer.
      emit("error", {
        message: "A reply is still being generated. Let it finish (or stop it) before switching engines.",
        engine: engine.id,
        revertedTo: previous,
        revertedLabel: getChatEngine(previous).label,
      });
      res.end();
      return;
    }
    emit("phase", { phase: "starting", message: `Switching to ${engine.label}…`, engine: engine.id });

    // Free the outgoing engine's worker first. Two engines each holding the
    // same multi-gigabyte model is exactly the memory pressure that makes the
    // second one thrash.
    if (previous !== engine.id) {
      const outgoing = getChatEngine(previous);
      if (outgoing.kind === "native") {
        emit("phase", { phase: "releasing", message: `Releasing ${outgoing.label}…` });
        await getSharedRuntime(outgoing.id).releaseAllWorkers().catch(() => {});
        if (getWarmState().engineId === outgoing.id) {
          setWarmState({ modelArtifact: null, engineId: null });
        }
      }
    }

    const result = await warmChatEngine({
      config,
      logger,
      engineId: engine.id,
      getSharedRuntime,
      signal: controller.signal,
      onProgress: (progress) => emit("phase", progress),
    });

    selectedEngineId = engine.id;
    if (result.modelArtifact) {
      setWarmState({ modelArtifact: result.modelArtifact, engineId: engine.id });
    }
    emit("done", {
      engine: engine.id,
      label: engine.label,
      modelId: result.modelId,
      warm: result.warm,
      streams: engine.streams,
    });
  } catch (error) {
    logger?.warn?.("chat-engine-switch-failed", { engine: engine.id, message: error.message });
    // Stay on whatever was working rather than stranding chat on a dead engine.
    selectedEngineId = previous;
    emit("error", {
      message: error.message,
      engine: engine.id,
      revertedTo: previous,
      revertedLabel: getChatEngine(previous).label,
    });
  }
  res.end();
}

// ---- One chat turn --------------------------------------------------------

// How many tokens the classifier is allowed. It is asked for one word; this is
// headroom for a model that answers "A bar chart." instead, which the parser
// accepts. Small enough that the step costs a fraction of a second on a warm
// model, which is what lets it run before every plausible chart question.
const CHART_CLASSIFIER_TOKENS = 12;

/**
 * Which chart this turn should offer, if any.
 *
 * The gate is free and skips everything for the overwhelming majority of
 * messages ("hi", "thanks", "explain this error"), so ordinary chat pays
 * nothing for the feature at all.
 *
 * When it fires, the type is suggested from the wording and the model is told
 * it may override it - so the decision still belongs to the model, which emits
 * the "type" field and can drop the chart entirely, but it costs one call
 * instead of two. That matters: measured on this machine, llama-3.2-3B runs at
 * ~5 tok/s with ~140 ms/token prefill, making a separate classify call 15-22 s
 * of silence before the answer starts. LIZARD_CHART_PRECHECK=1 spends it anyway
 * for engines fast enough not to care.
 *
 * Never throws. No chart is exactly what the user had before this existed.
 */
async function decideChartType({
  config,
  logger,
  turnRuntime,
  question,
  contextSize,
  getSharedRuntime,
  isAborted,
}) {
  const gate = looksLikeChartRequest(question);
  if (!gate.asked || isAborted()) {
    return null;
  }
  if (process.env.LIZARD_CHART_PRECHECK !== "1") {
    return suggestChartType(question);
  }
  const { systemPrompt, prompt } = buildClassifierPrompt(question);
  try {
    const result = await runEngineTurn({
      config,
      logger,
      turnRuntime,
      prompt,
      systemPrompt,
      maxTokens: CHART_CLASSIFIER_TOKENS,
      // The classification is a lookup, not a creative act: same question, same
      // chart, every time.
      temperature: 0,
      topP: 1,
      contextSize,
      getSharedRuntime,
    });
    const type = parseClassifierReply(result?.output ?? "");
    logger?.debug?.("chat-chart-classified", { gate: gate.reason, type });
    return type;
  } catch (error) {
    // A failed pre-check falls back to the free suggestion rather than
    // dropping the chart: the user asked for one either way.
    logger?.warn?.("chat-chart-precheck-failed", { message: error.message });
    return suggestChartType(question);
  }
}

async function streamTurn({
  req,
  res,
  config,
  logger,
  body,
  getSharedRuntime,
  setWarmState,
  reportFailure = async () => ({ reported: false }),
}) {
  res.writeHead(200, {
    "Content-Type": "text/event-stream; charset=utf-8",
    "Cache-Control": "no-store",
    Connection: "keep-alive",
    "X-Accel-Buffering": "no",
  });
  const emit = (event, data) => {
    if (res.writableEnded) return;
    res.write(`event: ${event}\ndata: ${JSON.stringify(data ?? {})}\n\n`);
  };

  // Loading a cold model can take minutes, during which the turn produces no
  // events at all. Without a heartbeat that silence looks like a dead
  // connection to anything with an idle timeout between here and the browser.
  const keepAlive = setInterval(() => {
    if (!res.writableEnded) res.write(": keep-alive\n\n");
  }, 15000);

  // The client aborts this fetch to stop generation. The runtime cannot be
  // interrupted mid-token, so "stop" means: stop streaming, stop the tool loop,
  // and keep whatever the model had already produced.
  let aborted = false;
  req.on("close", () => { aborted = true; });

  const startedAt = Date.now();
  // Phase trace for diagnosing pre-generation latency. Off unless asked for.
  const traceTurn = process.env.LIZARD_TRACE_TURN === "1";
  const mark = (phase) => {
    if (traceTurn) process.stderr.write(`TURNPHASE ${phase} +${Date.now() - startedAt}ms\n`);
  };
  turnsInFlight += 1;
  try {
    mark("enter");
    const conversation = getConversation(config, body.conversationId);
    mark("conversation-loaded");
    if (!conversation) {
      emit("error", { message: "That conversation no longer exists." });
      res.end();
      return;
    }
    const workspace = getWorkspace(config, conversation.workspaceId);
    mark("workspace-loaded");
    const bound = resolveBoundModel(config);
    mark("bound-model-resolved");
    if (!bound.bound) {
      emit("error", { message: bound.message });
      res.end();
      return;
    }

    // Regenerating means rewinding to just before the answer being replaced.
    if (body.regenerateFrom) {
      truncateFrom(config, conversation.id, body.regenerateFrom, { inclusive: true });
    }

    const prompts = listPrompts(config);
    let userText = String(body.text ?? "").trim();
    let usedPrompt = null;
    if (userText) {
      const expansion = expandPrompt(prompts, userText);
      if (expansion.expanded) {
        userText = expansion.text;
        usedPrompt = expansion.prompt;
      }
    }

    const attachments = Array.isArray(body.attachments)
      ? body.attachments.slice(0, 4).map((attachment) => ({
        name: String(attachment.name ?? "attachment.txt").slice(0, 120),
        text: String(attachment.text ?? "").slice(0, 60_000),
      }))
      : [];

    let userMessage = null;
    if (!body.regenerateFrom) {
      if (!userText && attachments.length === 0) {
        emit("error", { message: "Type a message first." });
        res.end();
        return;
      }
      userMessage = appendMessage(config, conversation.id, {
        role: "user",
        content: userText || `(sent ${attachments.length} file${attachments.length === 1 ? "" : "s"})`,
        ...(attachments.length > 0
          ? { attachments: attachments.map((attachment) => ({ name: attachment.name, chars: attachment.text.length })) }
          : {}),
        ...(usedPrompt ? { promptUsed: usedPrompt } : {}),
      });
      emit("user-message", { message: userMessage, title: getConversation(config, conversation.id)?.title });
      mark("user-message-emitted");
    }

    const current = getConversation(config, conversation.id);
    const history = recentTurns(current, workspace.params.historyTurns).slice(0, -1);
    const lastUser = [...current.messages].reverse().find((message) => message.role === "user");
    const question = body.regenerateFrom ? (lastUser?.content ?? userText) : userText;

    // ---- retrieval ----
    emit("status", { stage: "context" });
    let matches = [];
    const citedIds = Array.isArray(body.documentIds) && body.documentIds.length > 0 ? body.documentIds : null;
    if (workspace.retrieval?.enabled !== false) {
      const topK = workspace.retrieval?.topK ?? 4;
      matches = retrieve(config, workspace.id, question, { topK, documentIds: citedIds });

      // Citing a document with #name is an instruction to USE it, not a hint
      // about where to search. BM25 scores the question's terms, and a question
      // like "#notes.txt tell me what is it about" has none once stop words go
      // - so retrieval came back empty and the model answered, correctly from
      // where it sat, that the content was not provided. When the user named a
      // document and the search found nothing in it, hand over its opening
      // instead: they asked for that document by name.
      if (citedIds && matches.length === 0) {
        matches = openingChunks(config, workspace.id, citedIds, topK);
      }
    }

    // ---- tools ----
    let tools = [];
    const enabledPlugins = Array.isArray(workspace.tools) ? workspace.tools : [];
    if (enabledPlugins.length > 0) {
      emit("status", { stage: "tools" });
      const collected = await collectTools(config, enabledPlugins);
      tools = collected.tools;
      for (const failure of collected.failures) {
        emit("tool-error", failure);
      }
      if (tools.length > 0) {
        emit("tools-ready", { count: tools.length, names: tools.map((tool) => tool.name) });
      }
    }
    const toolInstructions = buildToolInstructions(tools);

    // ---- size this turn to the machine ----
    //
    // The KV cache is planned from the context size and costs ~344 KB per
    // token on a 3B model, so inheriting the model's full context reserves
    // gigabytes for a three-message conversation - and fails outright once an
    // attachment pushes the prompt past it. Plan the turn instead: work out
    // what fits, ask for only that, and drop the least valuable inputs when it
    // still does not. See src/lib/chat/context-budget.js.
    emit("status", { stage: "planning" });
    mark("retrieval-done");
    const turnRuntime = await resolveTurnRuntime({
      config,
      logger,
      engineId: getSelectedEngineId(config),
      prompt: question,
    });
    mark("turn-runtime-resolved");
    let plan = planTurnBudget({
      systemPrompt: workspace.systemPrompt,
      toolInstructions,
      question,
      history,
      contextChunks: matches,
      attachments,
      answerTokens: workspace.params.maxTokens,
      modelContextTokens: turnRuntime.modelContextTokens,
      bytesPerToken: turnRuntime.bytesPerToken,
      budgetBytes: resolveKvBudgetBytes(turnRuntime.context.profile),
      minContextSize: contextFloorFor(conversation.id),
    });
    if (!plan.fits) {
      emit("error", { message: plan.notes[0] ?? "This message does not fit in the available context." });
      res.end();
      return;
    }
    // ---- does this turn want a picture? ----
    //
    // Deliberately quiet: no status stage, no control, nothing in the UI. The
    // user asked a question; whether the answer carries a chart is the model's
    // call, and the plumbing behind that should be invisible. The classifier
    // runs on the plan's context size so the answer that follows reuses the
    // same KV shape rather than forcing a re-plan between two calls.
    const chartType = await decideChartType({
      config,
      logger,
      turnRuntime,
      question,
      contextSize: plan.contextSize,
      getSharedRuntime,
      isAborted: () => aborted,
    });
    mark(`chart-decided-${chartType ?? "none"}`);
    const chartInstruction = chartType ? buildChartInstruction(chartType) : "";
    // The instruction is another system-block section, exactly like the tool
    // block, so it rides in the same slot and is accounted for in the budget.
    const instructions = [toolInstructions, chartInstruction].filter(Boolean).join("\n\n");
    if (chartInstruction) {
      // Re-plan: the instruction is ~150 tokens the first plan did not know
      // about, and a turn that is close to its ceiling must not silently lose
      // history to make room for it.
      const replanned = planTurnBudget({
        systemPrompt: workspace.systemPrompt,
        toolInstructions: instructions,
        question,
        history,
        contextChunks: matches,
        attachments,
        answerTokens: workspace.params.maxTokens,
        modelContextTokens: turnRuntime.modelContextTokens,
        bytesPerToken: turnRuntime.bytesPerToken,
        budgetBytes: resolveKvBudgetBytes(turnRuntime.context.profile),
        minContextSize: contextFloorFor(conversation.id),
      });
      if (replanned.fits) {
        plan = replanned;
      }
    }
    // Whatever this turn settled on becomes the floor for the next one, so the
    // KV shape stops oscillating and the prefill snapshot stays reusable.
    recordContextSize(conversation.id, plan.contextSize);

    if (plan.notes.length > 0) {
      emit("budget", {
        notes: plan.notes,
        contextSize: plan.contextSize,
        promptTokens: plan.promptTokens,
        ceiling: plan.ceiling,
      });
    }

    const citations = plan.contextChunks.map((match) => ({
      rank: match.rank,
      documentId: match.documentId,
      documentName: match.documentName,
      locator: match.locator,
      score: match.score,
      excerpt: match.text.slice(0, 320),
    }));
    if (citations.length > 0) {
      emit("citations", { citations });
    }
    const contextBlock = formatContextBlock(plan.contextChunks);

    // ---- generate, with a tool loop around it ----
    const toolTranscript = [];
    const toolCalls = [];
    let finalText = "";
    let lastResult = null;
    let streamedAny = false;
    let firstContentAt = null;

    for (let iteration = 0; iteration <= MAX_TOOL_ITERATIONS; iteration += 1) {
      if (aborted) break;
      const composed = composePrompt({
        systemPrompt: workspace.systemPrompt,
        toolInstructions: instructions,
        contextBlock,
        history: plan.history,
        userMessage: question,
        toolTranscript,
        attachments: plan.attachments,
        // Caterpillar re-runs the whole prompt through its CPU engine on every
        // prefill. Laying the conversation out append-only lets its runner keep
        // the previous turn's KV and ingest only the new tail.
        // Ostrich forks that engine and reuses the same KV, so it wants the
        // same append-only layout.
        inlineHistory: ["caterpillar", "ostrich"].includes(turnRuntime.engine.id),
      });
      // Prompt size rides along so the UI can name the wait. Everything before
      // the first token is prefill, and on a long transcript that is seconds of
      // silence; "reading 449 tokens of context" is a wait the user can read,
      // where a bare spinner looks like an answer that never streamed.
      emit("status", {
        stage: iteration === 0 ? "thinking" : "resuming",
        iteration,
        promptTokens: plan.promptTokens,
        contextSize: plan.contextSize,
      });
      mark(`generate-start-${iteration}`);

      // Tool round-trips are still kept out of the transcript - the user should
      // see the answer, not the model negotiating with a plugin - but that used
      // to cost the stream itself. `streamThisPass` was false whenever ANY tool
      // was available, so the model generated in complete silence and the
      // finished answer arrived as a single delta. Plugins are on by default, so
      // in practice most answers never streamed: a burst of preflight events, a
      // few seconds of nothing, then the whole reply at once.
      //
      // The gate below streams optimistically instead, which the protocol makes
      // safe: a tool call is the WHOLE reply ("reply with nothing but this tag"),
      // so it is recognisable from its first characters. Prose is held back only
      // while it could still be the opening of the tag - at most 11 characters -
      // and released the moment it cannot be. A real tool call is still never
      // shown.
      const emitDelta = (text) => {
        if (!streamedAny) {
          streamedAny = true;
          firstContentAt ??= Date.now();
          emit("start", {});
        }
        emit("delta", { text });
      };
      // Rebuilt per iteration: each pass decides for itself whether it is a tool
      // call, and the pass after a tool result is a fresh decision.
      const gate = createToolCallStreamGate(emitDelta);
      let passText = "";
      lastResult = await runEngineTurn({
        config,
        logger,
        turnRuntime,
        prompt: composed.prompt,
        systemPrompt: composed.systemPrompt,
        maxTokens: plan.answerTokens,
        temperature: workspace.params.temperature,
        topP: workspace.params.topP,
        contextSize: plan.contextSize,
        getSharedRuntime,
        onDelta: (text) => {
          if (aborted || !text) return;
          passText += text;
          gate.push(text);
        },
      });
      // Whatever the gate is still holding was never a tool call - release it.
      gate.finish();
      const output = lastResult.output ?? passText;
      const call = tools.length > 0 ? parseToolCall(output) : null;
      if (!call) {
        finalText = output;
        if (!streamedAny) {
          // The gate withheld this pass because it opened like a tool call, but
          // it did not parse as one (a malformed tag, most often). The text is
          // still the answer, so deliver it in one piece rather than losing it.
          firstContentAt ??= Date.now();
          emit("start", {});
          emit("delta", { text: stripToolCalls(output) });
          streamedAny = true;
        }
        break;
      }
      if (!call.name) {
        emit("tool-error", { name: "unknown", message: "The model emitted a malformed tool call; answering without tools." });
        toolTranscript.push(formatToolResult("unknown", {
          isError: true,
          text: "That tool call was not valid JSON. Answer the user directly instead.",
        }));
        continue;
      }
      emit("tool-call", { name: call.name, arguments: call.arguments });
      let result;
      try {
        result = await callTool(config, tools, call.name, call.arguments);
      } catch (error) {
        result = { isError: true, text: error.message };
      }
      toolCalls.push({
        name: call.name,
        arguments: call.arguments,
        isError: Boolean(result.isError),
        result: String(result.text ?? "").slice(0, 2000),
      });
      emit("tool-result", {
        name: call.name,
        isError: Boolean(result.isError),
        preview: String(result.text ?? "").slice(0, 600),
      });
      toolTranscript.push(`Assistant: ${call.raw}`, formatToolResult(call.name, result));
      finalText = stripToolCalls(output);
    }

    // A native turn leaves its worker resident, so record it as the warm model:
    // onboarding's release path frees it, and the header can report it warm.
    if (turnRuntime.engine.kind === "native" && !aborted) {
      setWarmState({ modelArtifact: turnRuntime.context.modelArtifact, engineId: turnRuntime.engine.id });
    }

    // ---- lift the chart out of the answer ----
    //
    // The block is cut from the stored prose so the transcript never shows raw
    // JSON, and the spec is stored beside the message. A model that ignored the
    // instruction, or emitted something that would draw a misleading chart,
    // simply produces no chart here - the prose answer is untouched either way.
    let chart = null;
    let answerText = finalText;
    if (chartType) {
      const extracted = extractChartSpec(finalText);
      // Prompt work on a small model needs to see which of the three steps
      // gave way - classify, emit, or repair. LIZARD_TRACE_CHART=1 reports it
      // without printing the conversation itself.
      if (process.env.LIZARD_TRACE_CHART === "1") {
        process.stderr.write(`CHARTTRACE type=${chartType} answerChars=${finalText.length} blockFound=${Boolean(extracted.raw)}\n`);
      }
      if (extracted.raw) {
        chart = normalizeChartSpec(extracted.raw, { requestedType: chartType });
        // Remove the block whenever it was OUR fence, chart or no chart.
        //
        // This used to keep the raw text on the theory that the model's numbers
        // should stay visible rather than vanish without trace. That reasoning
        // does not survive contact with a small model that degenerates: one
        // repetition loop emitted ~500 tokens of {"label":"5800 Hz","value":0}
        // and ran out of budget mid-object, so the block never closed, never
        // parsed, and the whole wall of JSON was published as the answer.
        // `lizard-chart` is our own private syntax - it is never prose, and a
        // block that failed is a failed chart, not data worth reading.
        //
        // A guessed block (a bare object, or a generic ```json fence) is still
        // left alone unless it really became a chart: that might be an answer
        // legitimately showing JSON, and taking it would delete real content.
        if (chart || extracted.source === "fence") {
          answerText = extracted.text;
        }
        if (!chart) {
          logger?.warn?.("chat-chart-discarded", {
            source: extracted.source,
            chars: extracted.raw.length,
            truncated: !/}\s*$/.test(extracted.raw.trim()),
          });
        }
      }
    }

    const { reasoning, answer } = splitReasoning(answerText);
    const responseTimeMs = Date.now() - startedAt;
    const generatedTokens = lastResult?.nativeExecution?.generatedTokenIds?.length ?? null;
    const userVisibleTokensPerSecond = generatedTokens > 0 && responseTimeMs > 0
      ? (generatedTokens / responseTimeMs) * 1000
      : lastResult?.tokensPerSecond ?? null;
    const ttftMs = firstContentAt == null ? responseTimeMs : firstContentAt - startedAt;
    const evidenceMetrics = extractEvidenceMetrics(lastResult);
    const stats = {
      tokensPerSecond: userVisibleTokensPerSecond,
      userVisibleTokensPerSecond,
      decodeTokensPerSecond: lastResult?.tokensPerSecond ?? null,
      runtimeProvider: lastResult?.runtimeProvider ?? turnRuntime.engine.id,
      engineId: turnRuntime.engine.id,
      nativeBackend: lastResult?.nativeBackend ?? null,
      durationMs: lastResult?.durationMs ?? (Date.now() - startedAt),
      responseTimeMs,
      endToEndMs: responseTimeMs,
      ttftMs,
      latencyPerTokenMs: generatedTokens > 0 ? responseTimeMs / generatedTokens : null,
      transport: "chat-http-stream",
      modelId: lastResult?.modelId ?? bound.modelId,
      maxTokens: plan.answerTokens,
      contextSize: plan.contextSize,
      promptTokens: plan.promptTokens,
      temperature: workspace.params.temperature,
      stopped: aborted,
      execution: lastResult ? buildQuickChatExecutionTelemetry(lastResult) : null,
      evidenceMetrics,
    };
    const currentMetric = appendNativeTurnMetric(config, {
      userId: "local-chat",
      surface: "chat-workspace",
      modelId: stats.modelId,
      quant: turnRuntime.gguf?.quantization ?? turnRuntime.gguf?.quant ?? null,
      runtimeProvider: stats.runtimeProvider,
      optimizationProfile: stats.runtimeProvider === "caterpillar"
        ? caterpillarPerformanceProfileFromOptions({}, {
          defaultProfile: CATERPILLAR_TTFT_PROFILE,
        })
        : "balanced",
      backend: stats.nativeBackend,
      servingPath: stats.execution?.servingPath,
      batched: stats.execution?.batched,
      concurrencyLevel: stats.execution?.concurrencyLevel,
      contextTokens: plan.contextSize,
      promptTokens: plan.promptTokens,
      generatedTokens,
      prefillMs: lastResult?.prefillMs,
      decodeMs: lastResult?.decodeMs,
      decodeTokensPerSecond: stats.decodeTokensPerSecond,
      userVisibleTokensPerSecond: stats.userVisibleTokensPerSecond,
      endToEndMs: stats.endToEndMs,
      ttftMs: stats.ttftMs,
      latencyPerTokenMs: stats.latencyPerTokenMs,
      transport: stats.transport,
      evidenceMetrics,
    });
    const comparableMetrics = listNativeTurnMetrics(config, {
      userId: "local-chat",
      modelId: stats.modelId,
      quant: turnRuntime.gguf?.quantization ?? turnRuntime.gguf?.quant ?? "unknown",
      runtimeProvider: stats.runtimeProvider,
      optimizationProfile: stats.runtimeProvider === "caterpillar"
        ? caterpillarPerformanceProfileFromOptions({}, {
          defaultProfile: CATERPILLAR_TTFT_PROFILE,
        })
        : "balanced",
      context: contextBucket(plan.contextSize),
      limit: 50,
    });
    stats.performance = {
      current: currentMetric,
      summary: summarizeNativeTurnMetrics(comparableMetrics),
    };
    const stored = appendMessage(config, conversation.id, {
      role: "assistant",
      content: answer || (aborted ? "(stopped)" : ""),
      ...(reasoning ? { reasoning } : {}),
      ...(citations.length > 0 ? { citations } : {}),
      ...(toolCalls.length > 0 ? { toolCalls } : {}),
      ...(chart ? { chart } : {}),
      stats,
    });
    // Regenerating leaves the transcript's updatedAt stale otherwise.
    writeConversation(config, getConversation(config, conversation.id));
    emit("done", { message: stored, stats, citations, toolCalls });
  } catch (error) {
    logger?.warn?.("chat-turn-failed", { message: error.message });
    const incident = await reportFailure(error, {
      operation: "chat-turn",
      route: "/api/chat/turn",
    });
    emit("error", {
      message: error.message,
      responseTimeMs: Date.now() - startedAt,
      ...incidentUserFields(incident),
    });
  } finally {
    turnsInFlight = Math.max(0, turnsInFlight - 1);
  }
  clearInterval(keepAlive);
  res.end();
}
