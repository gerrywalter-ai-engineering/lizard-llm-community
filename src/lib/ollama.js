import fs from "node:fs";
import path from "node:path";
import { execFileSync } from "node:child_process";

// `ollama list` is not a local file read - it is an RPC to the Ollama daemon,
// which answers only when it feels like it. A daemon that is starting up,
// loading a model, or simply wedged leaves the CLI waiting, and because this
// call is SYNCHRONOUS that wait is the whole Node event loop: no page renders,
// no API answers, the local server stops responding entirely and a browser
// reload does nothing. This is what froze the onboarding Select page, which
// enumerates Ollama models before it can draw the list.
//
// Ollama is an optional integration. Waiting on it is never worth the app.
const OLLAMA_CLI_TIMEOUT_MS = 2_500;

function runOllama(args) {
  return execFileSync("ollama", args, {
    encoding: "utf8",
    stdio: ["ignore", "pipe", "pipe"],
    windowsHide: true,
    timeout: OLLAMA_CLI_TIMEOUT_MS,
    killSignal: "SIGKILL",
  }).trim();
}

// listInstalledOllamaModels is polled by the dashboard's /api/state route as
// often as every 2s while a benchmark is running (see benchmark-dashboard.ts
// PENDING_REFRESH_INTERVAL_MS), and each call previously spawned a fresh
// "ollama list" process synchronously. Even with windowsHide, that's a
// process spawn + event-loop stall on every poll tick, so cache the result
// briefly - installed models don't change within a few seconds.
const OLLAMA_LIST_CACHE_TTL_MS = 4000;
// After a failure the answer is cached for far longer than after a success.
// The 4s success TTL is tuned for "installed models rarely change"; applied to a
// failure it is a disaster, because the failure mode that matters costs the full
// CLI timeout every time it is retried. A dashboard polling every 2s with a 4s
// TTL and a 2.5s stall per miss spends most of its life with the event loop
// blocked - the freeze survives the timeout that was supposed to end it. Ollama
// missing or wedged is a stable condition; there is nothing to gain by asking
// again immediately, and a user who starts Ollama mid-session waits a minute.
const OLLAMA_LIST_FAILURE_TTL_MS = 60_000;
let ollamaListCache = null;

// `run` is a test seam (same shape as selectNativeRunnerBinary's `selfTest`):
// the freeze this guards against cannot be reproduced by invoking a real
// `ollama` binary, so the contract is verified against a stand-in that behaves
// like a wedged daemon.
export function listInstalledOllamaModels(logger, { run = runOllama } = {}) {
  const now = Date.now();
  if (ollamaListCache) {
    const ttl = ollamaListCache.failed ? OLLAMA_LIST_FAILURE_TTL_MS : OLLAMA_LIST_CACHE_TTL_MS;
    if (now - ollamaListCache.at < ttl) {
      return ollamaListCache.models;
    }
  }
  try {
    const output = run(["list"]);
    const lines = output.split(/\r?\n/).slice(1).filter(Boolean);
    const models = lines.map((line) => {
      const match = line.trim().match(/^(\S+)\s+(\S+)\s+(.+?)\s{2,}(.+)$/);
      if (!match) {
        return {
          tag: line.trim(),
          id: null,
          size: null,
          modified: null,
        };
      }
      return {
        tag: match[1],
        id: match[2],
        size: match[3].trim(),
        modified: match[4].trim(),
      };
    });
    logger?.info("ollama-list", { count: models.length });
    ollamaListCache = { at: now, models, failed: false };
    return models;
  } catch (error) {
    logger?.warn("ollama-list-failed", { message: error.message });
    ollamaListCache = { at: now, models: [], failed: true };
    return [];
  }
}

// Lets a user who starts Ollama mid-session see their models without waiting out
// the failure backoff. Called by the paths where the user has explicitly asked
// for a fresh answer, not on every poll.
export function resetOllamaListCache() {
  ollamaListCache = null;
}

function manifestPathFromTag(tag, config) {
  const [name, rawVariant = "latest"] = String(tag).split(":");
  return path.join(
    config.ollama.modelsDirectory,
    "manifests",
    "registry.ollama.ai",
    "library",
    name,
    rawVariant,
  );
}

export function readOllamaManifest(tag, config) {
  const manifestPath = manifestPathFromTag(tag, config);
  if (!fs.existsSync(manifestPath)) {
    return null;
  }
  return {
    manifestPath,
    manifest: JSON.parse(fs.readFileSync(manifestPath, "utf8")),
  };
}

export function ensureOllamaPulled(tag, config, logger) {
  const existing = readOllamaManifest(tag, config);
  if (existing) {
    return existing;
  }
  logger?.info("ollama-pull-start", { tag });
  runOllama(["pull", tag]);
  logger?.info("ollama-pull-complete", { tag });
  const pulled = readOllamaManifest(tag, config);
  if (!pulled) {
    throw new Error(`Ollama pull completed but manifest was not found for ${tag}`);
  }
  return pulled;
}

export async function requestOllamaGenerate({
  tag,
  prompt,
  systemPrompt,
  maxTokens,
  temperature = 0.2,
  timeoutMs = 600000,
  keepAlive = undefined,
  signal = null,
}) {
  const controller = new AbortController();
  const abortFromCaller = () => controller.abort(signal?.reason);
  if (signal?.aborted) {
    abortFromCaller();
  } else {
    signal?.addEventListener("abort", abortFromCaller, { once: true });
  }
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const messages = [
      ...(systemPrompt ? [{ role: "system", content: systemPrompt }] : []),
      { role: "user", content: prompt },
    ];
    let response;
    try {
      response = await fetch("http://127.0.0.1:11434/api/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        signal: controller.signal,
        body: JSON.stringify({
          model: tag,
          messages,
          stream: false,
          think: false,
          keep_alive: keepAlive,
          options: {
            temperature,
            num_predict: maxTokens,
          },
        }),
      });
    } catch (error) {
      throw new Error(`Ollama chat request failed for ${tag}: ${error.message}`);
    }
    const text = await response.text();
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${text}`);
    }
    const parsed = text ? JSON.parse(text) : {};
    return {
      ...parsed,
      response: parsed.response ?? parsed.message?.content ?? "",
      eval_count: parsed.eval_count ?? parsed.usage?.completion_tokens ?? null,
    };
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener("abort", abortFromCaller);
  }
}

/**
 * Is the Ollama daemon answering? Used to decide whether to offer Ollama as a
 * chat engine at all, rather than offering it and failing at send time.
 */
export async function isOllamaReachable({ timeoutMs = 1500 } = {}) {
  try {
    const response = await fetch("http://127.0.0.1:11434/api/tags", {
      cache: "no-store",
      signal: AbortSignal.timeout(timeoutMs),
    });
    return response.ok;
  } catch {
    return false;
  }
}

/**
 * Pulls a model over Ollama's HTTP API so download progress can be reported
 * live. `ensureOllamaPulled` above shells out to `ollama pull` and blocks with
 * no progress, which is fine for a CLI step but not for a user watching a
 * warm-up dialog wondering whether a 4 GB download has stalled.
 */
export async function pullOllamaModel(tag, { onProgress, signal = null } = {}) {
  const response = await fetch("http://127.0.0.1:11434/api/pull", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model: tag, stream: true }),
    signal,
  });
  if (!response.ok || !response.body) {
    throw new Error(`Ollama could not pull ${tag}: HTTP ${response.status}`);
  }
  for await (const event of readNdjson(response.body)) {
    if (event.error) {
      throw new Error(`Ollama could not pull ${tag}: ${event.error}`);
    }
    onProgress?.({
      status: event.status ?? "",
      completed: Number.isFinite(event.completed) ? event.completed : null,
      total: Number.isFinite(event.total) ? event.total : null,
    });
  }
  return { ok: true, tag };
}

/**
 * Streaming chat against Ollama. requestOllamaGenerate above is deliberately
 * non-streaming (benchmarks want one timed round trip); a chat needs tokens as
 * they arrive, so this is a separate call rather than a flag on that one.
 */
export async function streamOllamaChat({
  tag,
  prompt,
  systemPrompt = null,
  maxTokens = 512,
  temperature = 0.7,
  topP = 0.9,
  keepAlive = "10m",
  signal = null,
  onDelta,
}) {
  const startedAt = Date.now();
  const response = await fetch("http://127.0.0.1:11434/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    signal,
    body: JSON.stringify({
      model: tag,
      messages: [
        ...(systemPrompt ? [{ role: "system", content: systemPrompt }] : []),
        { role: "user", content: prompt },
      ],
      stream: true,
      think: false,
      keep_alive: keepAlive,
      options: { temperature, top_p: topP, num_predict: maxTokens },
    }),
  });
  if (!response.ok || !response.body) {
    const detail = await response.text().catch(() => "");
    throw new Error(`Ollama chat failed for ${tag}: HTTP ${response.status} ${detail.slice(0, 200)}`);
  }

  let output = "";
  let evalCount = null;
  let evalDurationNs = null;
  for await (const event of readNdjson(response.body)) {
    if (event.error) {
      throw new Error(`Ollama chat failed for ${tag}: ${event.error}`);
    }
    const delta = event.message?.content ?? "";
    if (delta) {
      output += delta;
      onDelta?.(delta);
    }
    if (event.done) {
      evalCount = event.eval_count ?? null;
      evalDurationNs = event.eval_duration ?? null;
    }
  }
  const durationMs = Date.now() - startedAt;
  // Prefer Ollama's own decode timing; it excludes prompt evaluation, which is
  // what "tokens per second" means everywhere else in Lizard.
  const tokensPerSecond = Number.isFinite(evalCount) && Number.isFinite(evalDurationNs) && evalDurationNs > 0
    ? evalCount / (evalDurationNs / 1e9)
    : null;
  return { output, durationMs, tokensPerSecond, evalCount };
}

// Ollama streams newline-delimited JSON on both /api/pull and /api/chat.
async function* readNdjson(body) {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (value) {
      buffer += decoder.decode(value, { stream: true });
    }
    let newline = buffer.indexOf("\n");
    while (newline !== -1) {
      const line = buffer.slice(0, newline).trim();
      buffer = buffer.slice(newline + 1);
      newline = buffer.indexOf("\n");
      if (!line) continue;
      try {
        yield JSON.parse(line);
      } catch {
        // Ollama occasionally emits a partial line before a flush; skip it.
      }
    }
    if (done) break;
  }
  const tail = buffer.trim();
  if (tail) {
    try {
      yield JSON.parse(tail);
    } catch {
      // ignore a trailing fragment
    }
  }
}

export function resolveOllamaBlob(tag, config) {
  const resolved = readOllamaManifest(tag, config);
  if (!resolved) {
    return null;
  }
  const modelLayer = resolved.manifest.layers.find((layer) => layer.mediaType === "application/vnd.ollama.image.model");
  if (!modelLayer) {
    throw new Error(`No model layer found in Ollama manifest for ${tag}`);
  }
  const blobFile = modelLayer.digest.replace(":", "-");
  const blobPath = path.join(config.ollama.modelsDirectory, "blobs", blobFile);
  if (!fs.existsSync(blobPath)) {
    throw new Error(`Expected Ollama blob not found: ${blobPath}`);
  }
  return {
    ...resolved,
    modelLayer,
    blobPath,
  };
}

export function verifyGgufMagic(filePath) {
  const handle = fs.openSync(filePath, "r");
  const buffer = Buffer.alloc(4);
  fs.readSync(handle, buffer, 0, 4, 0);
  fs.closeSync(handle);
  return buffer.toString("utf8") === "GGUF";
}
