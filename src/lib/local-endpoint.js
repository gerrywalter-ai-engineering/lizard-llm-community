// The local API endpoint: one long-running, OpenAI-compatible HTTP port that
// external apps talk to.
//
// Why this exists at all
// ----------------------
// Until now the only way to reach an activated model over http:// was to run
// `lizard-llm serve` in a terminal. That process is a foreground child of a
// console: closing the window, signing out of Windows, or a reboot ends it, and
// nothing brings it back. Meanwhile the model the user actually activated was
// held resident *inside* the app process as a stdio worker, which no external
// client can address. So the thing that survived was unreachable, and the thing
// that was reachable did not survive.
//
// This module closes that gap by putting the endpoint in the SAME process that
// already holds the warm worker:
//
//   - it serves the model that is already loaded, so there is no second copy of
//     the weights and no fight with the single-residency claim,
//   - its lifetime is the app's lifetime - which on Windows is the tray icon's
//     lifetime - not a browser page's and not a login session's,
//   - it is never behind the local session cookie or a per-request entitlement
//     re-check, so a signed-out browser or an expired local session cannot take
//     an external integration down mid-conversation.
//
// Exactly one endpoint is armed at a time. Arming for a different provider
// disarms the previous one first: lizard-native, caterpillar and ostrich are
// three different binaries on three different ports, and two of them answering
// at once is how "which engine am I actually talking to" becomes unanswerable.
//
// Stopping is deliberate and always attributed. The endpoint goes down only
// when the user unloads the model, frees memory, a benchmark takes the port, or
// the app shuts down - and every one of those records a reason the UI can show,
// so a dead port is never a mystery.

import http from "node:http";
import net from "node:net";

// Each engine is its own binary on its own port. A provider missing from this
// table would silently answer on another engine's port.
export const ENDPOINT_PORT_BY_PROVIDER = Object.freeze({
  "lizard-native": 8013,
  caterpillar: 8014,
  ostrich: 8015,
});

const ENGINE_CONFIG_KEY_BY_PROVIDER = Object.freeze({
  "lizard-native": "native",
  caterpillar: "caterpillar",
  ostrich: "ostrich",
});

export const ENDPOINT_PROVIDERS = Object.freeze(Object.keys(ENDPOINT_PORT_BY_PROVIDER));

/** Normalize to a provider that has an endpoint, or null. */
export function endpointProvider(runtimeProvider) {
  const value = String(runtimeProvider ?? "").trim().toLowerCase();
  return ENDPOINT_PORT_BY_PROVIDER[value] ? value : null;
}

/**
 * Where a provider's endpoint lives, from config with the built-in defaults as
 * the floor. Pure: safe to call whether or not anything is armed, which is what
 * lets the UI show the address for the provider the user is *about* to pick.
 */
export function resolveEndpointAddress(config, runtimeProvider = "lizard-native") {
  const provider = endpointProvider(runtimeProvider) ?? "lizard-native";
  const engineKey = ENGINE_CONFIG_KEY_BY_PROVIDER[provider];
  const providerConfig = config?.engine?.[engineKey] ?? null;
  const host = providerConfig?.host ?? config?.engine?.host ?? "127.0.0.1";
  const port = Number(providerConfig?.port ?? ENDPOINT_PORT_BY_PROVIDER[provider]);
  // The bind host and the address a user pastes into their code are not the
  // same thing: 0.0.0.0 is a valid bind and a useless URL.
  const displayHost = host === "0.0.0.0" || host === "::" ? "127.0.0.1" : host;
  return {
    runtimeProvider: provider,
    host,
    displayHost,
    port,
    url: `http://${displayHost}:${port}`,
    baseUrl: `http://${displayHost}:${port}/v1`,
    chatCompletionsUrl: `http://${displayHost}:${port}/v1/chat/completions`,
    modelsUrl: `http://${displayHost}:${port}/v1/models`,
    healthUrl: `http://${displayHost}:${port}/health`,
  };
}

// Why a stop happened, in the user's terms. The UI reads `stopReason` to decide
// whether to say "you turned it off" or "something took it and here is how to
// get it back".
export const ENDPOINT_STOP_REASONS = Object.freeze({
  UNLOAD: "unload",
  FREE_MEMORY: "free-memory",
  BENCHMARK: "benchmark",
  PROVIDER_SWITCH: "provider-switch",
  SHUTDOWN: "shutdown",
  LOAD_CHECK_FAILED: "load-check-failed",
  PORT_UNAVAILABLE: "port-unavailable",
  NOT_ENTITLED: "not-entitled",
});

// Stops after which the port comes back by loading the model again, rather than
// by anything the user has to configure. These are the ones worth putting a
// "reload the model once" call to action next to.
const RELOAD_RECOVERS = new Set([
  ENDPOINT_STOP_REASONS.BENCHMARK,
  ENDPOINT_STOP_REASONS.FREE_MEMORY,
  ENDPOINT_STOP_REASONS.LOAD_CHECK_FAILED,
  ENDPOINT_STOP_REASONS.PORT_UNAVAILABLE,
]);

const STOP_NOTES = Object.freeze({
  [ENDPOINT_STOP_REASONS.UNLOAD]:
    "You unloaded the model, so the local API port was closed with it. It stays closed until you activate a model again.",
  [ENDPOINT_STOP_REASONS.FREE_MEMORY]:
    "Free RAM took the weights out of memory, so the local API port was closed. Load the model again to reopen it.",
  [ENDPOINT_STOP_REASONS.BENCHMARK]:
    "A benchmark needs the same engine port, so the local API endpoint was stopped for the run. Load the model once more when the run finishes and the endpoint comes back.",
  [ENDPOINT_STOP_REASONS.PROVIDER_SWITCH]:
    "You switched engine. Only one engine can serve the local API at a time, so the previous port was closed.",
  [ENDPOINT_STOP_REASONS.SHUTDOWN]:
    "Lizard is shutting down. The endpoint comes back on the next start, as long as a model is still activated.",
  [ENDPOINT_STOP_REASONS.LOAD_CHECK_FAILED]:
    "The model loaded but did not pass its two-token answer check, so the port was not opened - an endpoint that cannot answer is worse than no endpoint.",
  [ENDPOINT_STOP_REASONS.PORT_UNAVAILABLE]:
    "Another process is already listening on this engine's port, so Lizard did not take it over.",
  [ENDPOINT_STOP_REASONS.NOT_ENTITLED]:
    "The local API server is a Pro feature and no valid entitlement was found on this machine.",
});

function stopNoteFor(reason, fallback = null) {
  return STOP_NOTES[reason] ?? fallback;
}

// ---------------------------------------------------------------------------
// Supervisor state
// ---------------------------------------------------------------------------

const supervisor = {
  server: null,
  sockets: new Set(),
  // The intent flag. True from the moment arming starts until something
  // deliberately disarms - which is what tells an unexpected `close` apart from
  // one we asked for, and therefore whether to re-listen.
  armed: false,
  generation: 0,
  relistenTimer: null,
  relistenAttempts: 0,
  handlers: null,
  logger: null,
  state: freshState(),
};

function freshState() {
  return {
    status: "stopped",
    runtimeProvider: null,
    modelId: null,
    host: null,
    port: null,
    url: null,
    startedAt: null,
    stoppedAt: null,
    stopReason: null,
    stopNote: null,
    needsModelReload: false,
    lastError: null,
    verification: null,
    entitlement: null,
    requests: 0,
    lastRequestAt: null,
    restarts: 0,
  };
}

/**
 * A snapshot of the endpoint for the UI and for `GET /api/endpoint`.
 * Always safe to call; never throws; never exposes anything token-shaped.
 */
export function getLocalEndpointState() {
  return { ...supervisor.state };
}

/** Whether an external client can reach a model right now. */
export function localEndpointIsListening() {
  return supervisor.state.status === "listening";
}

function setState(patch) {
  supervisor.state = { ...supervisor.state, ...patch };
}

// ---------------------------------------------------------------------------
// Port probing
// ---------------------------------------------------------------------------

/**
 * Is something already listening here? Used both as an arm-time preflight and
 * as the check behind "is the port actually accessible" after an unload - the
 * UI states that the port is closed, so it has to be able to prove it.
 */
export function probeEndpointPort({ host = "127.0.0.1", port, timeoutMs = 900 } = {}) {
  return new Promise((resolve) => {
    const socket = new net.Socket();
    let settled = false;
    const finish = (occupied) => {
      if (settled) return;
      settled = true;
      socket.removeAllListeners();
      socket.destroy();
      resolve(occupied);
    };
    socket.setTimeout(timeoutMs);
    socket.once("connect", () => finish(true));
    socket.once("timeout", () => finish(false));
    socket.once("error", () => finish(false));
    // A bare IPv6/IPv4 wildcard bind still answers on the loopback literal.
    socket.connect(Number(port), host === "0.0.0.0" || host === "::" ? "127.0.0.1" : host);
  });
}

// ---------------------------------------------------------------------------
// OpenAI wire format
// ---------------------------------------------------------------------------

// A local API is only useful if the browser-based tools people already have can
// call it, and those are cross-origin by definition (a page on localhost:3000
// talking to localhost:8014). This mirrors what llama.cpp's server does.
const CORS_HEADERS = Object.freeze({
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type, Authorization, X-Requested-With",
  "Access-Control-Max-Age": "86400",
});

function sendJson(res, statusCode, payload) {
  let body;
  try {
    body = JSON.stringify(payload);
  } catch (error) {
    statusCode = 500;
    body = JSON.stringify({
      error: {
        message: `Response could not be serialized: ${error?.message ?? String(error)}`,
        type: "server_error",
        code: "unserializable_response",
      },
    });
  }
  if (res.headersSent) {
    res.end();
    return;
  }
  res.writeHead(statusCode, {
    ...CORS_HEADERS,
    "Content-Type": "application/json; charset=utf-8",
    "Cache-Control": "no-store",
  });
  res.end(body);
}

function sendError(res, statusCode, message, { type = "server_error", code = null } = {}) {
  sendJson(res, statusCode, { error: { message, type, code } });
}

function readBody(req, { limitBytes = 32 * 1024 * 1024 } = {}) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    let size = 0;
    req.on("data", (chunk) => {
      size += chunk.length;
      if (size > limitBytes) {
        reject(Object.assign(new Error("Request body is too large."), { statusCode: 413 }));
        req.destroy();
        return;
      }
      chunks.push(chunk);
    });
    req.on("end", () => {
      try {
        const raw = Buffer.concat(chunks).toString("utf8").replace(/^﻿/, "");
        resolve(raw ? JSON.parse(raw) : {});
      } catch (error) {
        reject(Object.assign(new Error(`Request body is not valid JSON: ${error.message}`), {
          statusCode: 400,
          invalidJson: true,
        }));
      }
    });
    req.on("error", reject);
  });
}

// OpenAI's chat format carries roles; the runtime takes one prompt string plus
// an optional system prompt. Multi-part content (the array form the SDKs send
// for images) is flattened to its text parts - the native runner has no vision
// path yet, and dropping the text with it would fail a request it could answer.
function renderMessages(messages = []) {
  const textOf = (content) => {
    if (typeof content === "string") return content;
    if (Array.isArray(content)) {
      return content
        .map((part) => (typeof part === "string" ? part : part?.type === "text" ? String(part.text ?? "") : ""))
        .filter(Boolean)
        .join("\n");
    }
    return "";
  };
  const system = messages
    .filter((entry) => entry?.role === "system")
    .map((entry) => textOf(entry.content).trim())
    .filter(Boolean)
    .join("\n\n");
  const conversation = messages.filter((entry) => entry?.role !== "system");
  // One turn is just the prompt. Several turns become a transcript, the same
  // shape the onboarding chat composes, so the model sees who said what.
  const prompt = conversation.length <= 1
    ? textOf(conversation[0]?.content).trim()
    : [
      "You are continuing an ongoing chat session. Conversation so far:",
      "",
      ...conversation.map((entry) => `${entry.role === "assistant" ? "Assistant" : "User"}: ${textOf(entry.content).trim()}`),
      "",
      "Reply to the user's last message, using the conversation above as context when relevant.",
    ].join("\n");
  return { prompt, systemPrompt: system || null };
}

function completionId() {
  return `chatcmpl-${Math.random().toString(36).slice(2)}${Date.now().toString(36)}`;
}

function streamChunk({ id, created, modelId, delta, finishReason = null }) {
  return `data: ${JSON.stringify({
    id,
    object: "chat.completion.chunk",
    created,
    model: modelId,
    choices: [{ index: 0, delta, finish_reason: finishReason }],
  })}\n\n`;
}

async function handleChatCompletions(req, res, { handlers, logger }) {
  let body;
  try {
    body = await readBody(req);
  } catch (error) {
    sendError(res, error.statusCode ?? 400, error.message, {
      type: "invalid_request_error",
      code: error.invalidJson ? "invalid_json" : "invalid_request",
    });
    return;
  }
  const { prompt, systemPrompt } = renderMessages(body.messages ?? []);
  if (!prompt) {
    sendError(res, 400, "A non-empty user message is required. The request contained no user prompt text.", {
      type: "invalid_request_error",
      code: "missing_prompt",
    });
    return;
  }
  const maxTokens = Number.isFinite(Number(body.max_tokens)) && Number(body.max_tokens) > 0
    ? Math.min(Number(body.max_tokens), 8192)
    : 512;
  const wantsStream = body.stream === true;
  const created = Math.floor(Date.now() / 1000);
  const id = completionId();

  setState({ requests: supervisor.state.requests + 1, lastRequestAt: new Date().toISOString() });

  if (!wantsStream) {
    const result = await handlers.createCompletion({
      prompt,
      systemPrompt,
      maxTokens,
      temperature: body.temperature,
      topP: body.top_p,
    });
    sendJson(res, 200, {
      id,
      object: "chat.completion",
      created,
      model: result.modelId ?? supervisor.state.modelId,
      choices: [{
        index: 0,
        message: { role: "assistant", content: result.output ?? "" },
        finish_reason: "stop",
      }],
      usage: {
        prompt_tokens: result.promptTokens ?? null,
        completion_tokens: result.generatedTokens ?? null,
        total_tokens: (result.promptTokens ?? 0) + (result.generatedTokens ?? 0) || null,
      },
    });
    return;
  }

  res.writeHead(200, {
    ...CORS_HEADERS,
    "Content-Type": "text/event-stream; charset=utf-8",
    "Cache-Control": "no-cache, no-transform",
    Connection: "keep-alive",
    // Any reverse proxy in front of a local endpoint (people do put one there)
    // must not sit on the stream waiting for a full body.
    "X-Accel-Buffering": "no",
  });
  let clientGone = false;
  req.on("close", () => { clientGone = true; });
  const write = (chunk) => {
    if (clientGone || res.writableEnded || res.destroyed) return false;
    try {
      res.write(chunk);
      return true;
    } catch {
      clientGone = true;
      return false;
    }
  };
  // The role chunk goes out immediately so the client sees a live stream while
  // prefill runs. On a long prompt that can be tens of seconds, and a socket
  // with nothing on it is exactly what an HTTP client's idle timeout kills.
  write(streamChunk({ id, created, modelId: supervisor.state.modelId, delta: { role: "assistant" } }));
  let firstDelta = false;
  const heartbeat = setInterval(() => {
    if (firstDelta) return;
    write(": prefill\n\n");
  }, 10000);
  heartbeat.unref?.();
  try {
    const result = await handlers.createCompletion({
      prompt,
      systemPrompt,
      maxTokens,
      temperature: body.temperature,
      topP: body.top_p,
      onDelta: (text) => {
        if (!text) return;
        firstDelta = true;
        write(streamChunk({ id, created, modelId: supervisor.state.modelId, delta: { content: text } }));
      },
    });
    // A runtime that answers in one piece rather than streaming still has to
    // produce a complete OpenAI stream, or the client ends up with an empty
    // message and no error to explain it.
    if (!firstDelta && result?.output) {
      write(streamChunk({ id, created, modelId: result.modelId ?? supervisor.state.modelId, delta: { content: result.output } }));
    }
    write(streamChunk({ id, created, modelId: result?.modelId ?? supervisor.state.modelId, delta: {}, finishReason: "stop" }));
    write("data: [DONE]\n\n");
  } catch (error) {
    logger?.warn?.("local-endpoint-stream-failed", { message: String(error?.message ?? error).slice(0, 200) });
    // The 200 is already committed, so the only place an error can still be
    // reported is the stream body. Ending silently would look like a complete
    // but empty answer.
    write(`data: ${JSON.stringify({
      id,
      object: "chat.completion.chunk",
      created,
      model: supervisor.state.modelId,
      choices: [{ index: 0, delta: {}, finish_reason: "error" }],
      error: { message: String(error?.message ?? error), type: "server_error" },
    })}\n\n`);
    write("data: [DONE]\n\n");
  } finally {
    clearInterval(heartbeat);
    if (!res.writableEnded && !res.destroyed) res.end();
  }
}

function createRequestListener({ handlers, logger }) {
  return (req, res) => {
    const respond = async () => {
      const pathname = String(req.url ?? "/").split("?")[0];
      if (req.method === "OPTIONS") {
        res.writeHead(204, CORS_HEADERS);
        res.end();
        return;
      }
      if (req.method === "GET" && (pathname === "/health" || pathname === "/health/ready")) {
        const state = supervisor.state;
        sendJson(res, 200, {
          status: state.status === "listening" ? "ok" : state.status,
          model: state.modelId,
          runtimeProvider: state.runtimeProvider,
          port: state.port,
          startedAt: state.startedAt,
          uptimeSeconds: state.startedAt ? Math.round((Date.now() - Date.parse(state.startedAt)) / 1000) : 0,
          verification: state.verification,
          requests: state.requests,
        });
        return;
      }
      if (req.method === "GET" && (pathname === "/v1/models" || pathname === "/models")) {
        const modelId = supervisor.state.modelId;
        sendJson(res, 200, {
          object: "list",
          data: modelId
            ? [{
              id: modelId,
              object: "model",
              created: supervisor.state.startedAt ? Math.floor(Date.parse(supervisor.state.startedAt) / 1000) : Math.floor(Date.now() / 1000),
              // The provider name already carries the brand ("lizard-native"),
              // so prefixing it produced "lizard-lizard-native" in a field
              // clients display verbatim.
              owned_by: supervisor.state.runtimeProvider ?? "lizard-native",
            }]
            : [],
        });
        return;
      }
      if (req.method === "POST" && (pathname === "/v1/chat/completions" || pathname === "/chat/completions")) {
        await handleChatCompletions(req, res, { handlers, logger });
        return;
      }
      sendError(res, 404, `Unknown route ${req.method} ${pathname}. This endpoint serves /v1/chat/completions, /v1/models and /health.`, {
        type: "invalid_request_error",
        code: "unknown_route",
      });
    };
    respond().catch((error) => {
      logger?.error?.("local-endpoint-request-failed", { message: String(error?.message ?? error).slice(0, 200) });
      if (!res.headersSent) {
        sendError(res, 500, String(error?.message ?? error));
        return;
      }
      if (!res.writableEnded && !res.destroyed) res.end();
    });
  };
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

// Node's HTTP defaults are sized for a request/response API behind a load
// balancer, not for a local endpoint streaming a several-minute answer. Left
// alone, `requestTimeout` (300 s since Node 18) and `keepAliveTimeout` (5 s)
// each produce exactly the symptom this whole module exists to remove: a
// connection that dies on its own with nothing wrong.
function applyLongLivedSocketPolicy(server) {
  server.keepAliveTimeout = 10 * 60 * 1000;
  // Must stay above keepAliveTimeout or Node races the two and drops sockets.
  server.headersTimeout = 11 * 60 * 1000;
  server.requestTimeout = 0;      // a generation is allowed to take as long as it takes
  server.timeout = 0;             // no inactivity timeout on an open socket
  server.maxRequestsPerSocket = 0; // never force a client to reconnect
}

function trackSockets(server) {
  server.on("connection", (socket) => {
    // TCP keep-alive so a client that vanishes is noticed by the OS rather
    // than leaving a half-open socket holding a slot.
    socket.setKeepAlive(true, 30000);
    socket.setNoDelay(true);
    supervisor.sockets.add(socket);
    socket.once("close", () => supervisor.sockets.delete(socket));
  });
}

function destroySockets() {
  for (const socket of supervisor.sockets) {
    try {
      socket.destroy();
    } catch {
      // Already gone; nothing to release.
    }
  }
  supervisor.sockets.clear();
}

function listen(server, { host, port }) {
  return new Promise((resolve, reject) => {
    const onError = (error) => {
      server.off("listening", onListening);
      reject(error);
    };
    const onListening = () => {
      server.off("error", onError);
      resolve();
    };
    server.once("error", onError);
    server.once("listening", onListening);
    server.listen(Number(port), host);
  });
}

function closeServer(server) {
  return new Promise((resolve) => {
    if (!server) {
      resolve();
      return;
    }
    let settled = false;
    const finish = () => {
      if (settled) return;
      settled = true;
      resolve();
    };
    server.close(finish);
    // An in-flight stream would otherwise hold close() open indefinitely, and
    // a stop the user asked for has to be immediate.
    server.closeIdleConnections?.();
    server.closeAllConnections?.();
    destroySockets();
    setTimeout(finish, 1500).unref?.();
  });
}

/**
 * Bring the endpoint up for a model that is loaded and has answered.
 *
 * `handlers.createCompletion` is injected rather than imported so this module
 * never reaches back into the UI server: the caller passes the function that
 * talks to the already-warm shared runtime, which is what keeps the endpoint on
 * the same single copy of the weights.
 */
export async function armLocalEndpoint({
  config,
  logger = null,
  runtimeProvider,
  modelId,
  handlers,
  verification = null,
  entitlement = null,
} = {}) {
  const provider = endpointProvider(runtimeProvider);
  if (!provider) {
    // llama.cpp and Ollama run the model themselves; there is no Lizard-owned
    // port to open for them, and pretending otherwise would advertise an
    // address that never answers.
    await disarmLocalEndpoint({ reason: ENDPOINT_STOP_REASONS.PROVIDER_SWITCH, logger });
    return getLocalEndpointState();
  }
  const address = resolveEndpointAddress(config, provider);

  // Already serving this exact model on this exact provider: arming again is a
  // no-op, not a restart. Onboarding calls this on every page load.
  if (
    supervisor.armed &&
    supervisor.state.status === "listening" &&
    supervisor.state.runtimeProvider === provider &&
    supervisor.state.modelId === modelId
  ) {
    if (verification) setState({ verification });
    if (entitlement) setState({ entitlement });
    supervisor.handlers = handlers ?? supervisor.handlers;
    return getLocalEndpointState();
  }

  // Anything else armed has to go first - one engine, one port, one answer to
  // "what am I talking to".
  if (supervisor.server) {
    await disarmLocalEndpoint({ reason: ENDPOINT_STOP_REASONS.PROVIDER_SWITCH, logger, quiet: true });
  }

  supervisor.armed = true;
  supervisor.generation += 1;
  supervisor.handlers = handlers;
  supervisor.logger = logger;
  supervisor.relistenAttempts = 0;
  setState({
    status: "starting",
    runtimeProvider: provider,
    modelId: modelId ?? null,
    host: address.host,
    port: address.port,
    url: address.url,
    stopReason: null,
    stopNote: null,
    needsModelReload: false,
    lastError: null,
    verification: verification ?? supervisor.state.verification,
    entitlement: entitlement ?? supervisor.state.entitlement,
    requests: 0,
    lastRequestAt: null,
  });

  const server = http.createServer(createRequestListener({ handlers, logger }));
  applyLongLivedSocketPolicy(server);
  trackSockets(server);
  supervisor.server = server;

  const generation = supervisor.generation;
  // An unexpected close (a socket-level fault, an OS hiccup) is not a stop the
  // user asked for. The intent flag is what separates the two, and while it is
  // set the endpoint puts itself back up.
  server.on("close", () => {
    if (!supervisor.armed || supervisor.generation !== generation) return;
    logger?.warn?.("local-endpoint-closed-unexpectedly", { port: address.port });
    scheduleRelisten({ config, logger });
  });
  server.on("error", (error) => {
    if (supervisor.generation !== generation) return;
    logger?.warn?.("local-endpoint-error", { message: String(error?.message ?? error).slice(0, 200) });
    setState({ lastError: String(error?.message ?? error) });
    if (supervisor.armed) scheduleRelisten({ config, logger });
  });

  try {
    await listen(server, address);
  } catch (error) {
    supervisor.server = null;
    const occupied = error?.code === "EADDRINUSE";
    setState({
      status: "failed",
      lastError: String(error?.message ?? error),
      stopReason: occupied ? ENDPOINT_STOP_REASONS.PORT_UNAVAILABLE : null,
      stopNote: occupied
        ? `${stopNoteFor(ENDPOINT_STOP_REASONS.PORT_UNAVAILABLE)} Free port ${address.port} (a leftover \`lizard-llm serve\`, or a benchmark that is still running) and load the model again.`
        : `The local API endpoint could not open port ${address.port}: ${error?.message ?? error}`,
      needsModelReload: true,
      stoppedAt: new Date().toISOString(),
    });
    supervisor.armed = false;
    logger?.error?.("local-endpoint-listen-failed", {
      port: address.port,
      code: error?.code ?? null,
      message: String(error?.message ?? error).slice(0, 200),
    });
    return getLocalEndpointState();
  }

  setState({
    status: "listening",
    startedAt: new Date().toISOString(),
    stoppedAt: null,
    lastError: null,
  });
  logger?.info?.("local-endpoint-listening", {
    runtimeProvider: provider,
    modelId: modelId ?? null,
    url: address.url,
  });
  return getLocalEndpointState();
}

// Losing the listener is recoverable; giving up on it is not. Backoff is capped
// so a long-running machine keeps trying for as long as the model is loaded.
const RELISTEN_DELAYS_MS = [500, 1000, 2000, 5000, 10000, 30000];

function scheduleRelisten({ config, logger }) {
  if (!supervisor.armed || supervisor.relistenTimer) return;
  const attempt = supervisor.relistenAttempts;
  const delay = RELISTEN_DELAYS_MS[Math.min(attempt, RELISTEN_DELAYS_MS.length - 1)];
  supervisor.relistenAttempts += 1;
  setState({ status: "restarting" });
  supervisor.relistenTimer = setTimeout(() => {
    supervisor.relistenTimer = null;
    if (!supervisor.armed) return;
    const { runtimeProvider, modelId } = supervisor.state;
    armLocalEndpoint({
      config,
      logger,
      runtimeProvider,
      modelId,
      handlers: supervisor.handlers,
    }).then((state) => {
      if (state.status === "listening") {
        supervisor.relistenAttempts = 0;
        setState({ restarts: supervisor.state.restarts + 1 });
      } else if (supervisor.armed) {
        scheduleRelisten({ config, logger });
      }
    }).catch(() => {
      if (supervisor.armed) scheduleRelisten({ config, logger });
    });
  }, delay);
  supervisor.relistenTimer.unref?.();
}

/**
 * Take the endpoint down and say why. Every caller supplies a reason, because a
 * closed port with no explanation is the failure mode this replaces.
 */
export async function disarmLocalEndpoint({ reason = ENDPOINT_STOP_REASONS.UNLOAD, note = null, logger = null, quiet = false } = {}) {
  supervisor.armed = false;
  supervisor.generation += 1;
  if (supervisor.relistenTimer) {
    clearTimeout(supervisor.relistenTimer);
    supervisor.relistenTimer = null;
  }
  supervisor.relistenAttempts = 0;
  const server = supervisor.server;
  supervisor.server = null;
  supervisor.handlers = null;
  if (server) {
    await closeServer(server);
  }
  const previous = supervisor.state;
  setState({
    status: "stopped",
    startedAt: null,
    stoppedAt: new Date().toISOString(),
    stopReason: reason,
    stopNote: note ?? stopNoteFor(reason),
    needsModelReload: RELOAD_RECOVERS.has(reason),
    // The address stays in the snapshot on purpose: after an unload the UI
    // still has to be able to name the port it just closed.
    runtimeProvider: previous.runtimeProvider,
    port: previous.port,
    url: previous.url,
  });
  if (!quiet && server) {
    (logger ?? supervisor.logger)?.info?.("local-endpoint-stopped", {
      reason,
      runtimeProvider: previous.runtimeProvider,
      port: previous.port,
    });
  }
  return getLocalEndpointState();
}

/**
 * Record that something outside this module is about to take the port (a
 * benchmark), stop cleanly, and leave behind the "load the model once more"
 * marker the UI reads.
 */
export async function suspendLocalEndpointForBenchmark({ logger = null } = {}) {
  if (!supervisor.server && supervisor.state.status !== "listening") {
    return getLocalEndpointState();
  }
  return disarmLocalEndpoint({ reason: ENDPOINT_STOP_REASONS.BENCHMARK, logger });
}

/**
 * Full picture for the UI: live state plus the address every provider would
 * use, so the connection overview can be rendered for the provider the user has
 * selected even before anything is armed.
 */
export function describeLocalEndpoint(config, { selectedProvider = null } = {}) {
  const state = getLocalEndpointState();
  const providers = ENDPOINT_PROVIDERS.map((provider) => ({
    ...resolveEndpointAddress(config, provider),
    active: state.runtimeProvider === provider && state.status === "listening",
  }));
  const selected = endpointProvider(selectedProvider) ?? state.runtimeProvider ?? null;
  return {
    ...state,
    proFeature: true,
    selected: selected ? resolveEndpointAddress(config, selected) : null,
    providers,
  };
}

/** Test seam: drop all supervisor state without going through a real stop. */
export async function __resetLocalEndpointForTests() {
  await disarmLocalEndpoint({ reason: ENDPOINT_STOP_REASONS.SHUTDOWN, quiet: true });
  supervisor.state = freshState();
  supervisor.sockets.clear();
}
