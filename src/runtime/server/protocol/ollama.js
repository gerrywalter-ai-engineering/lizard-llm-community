import { ProtocolHttpError } from "./errors.js";
import { normalizeChatMessages } from "./messages.js";

const MAX_OUTPUT_TOKENS = 4096;
const MAX_STOP_SEQUENCES = 16;
const MAX_STOP_BYTES = 16 * 1024;

function invalid(param, message, statusCode = 400, code = "invalid_parameter") {
  throw new ProtocolHttpError(statusCode, code, message, { param });
}

function rejectUnknown(value, allowed, prefix = "") {
  const unknown = Object.keys(value).find((field) => !allowed.has(field));
  if (unknown) invalid(`${prefix}${unknown}`, `Unsupported parameter: ${prefix}${unknown}.`, 400, "unsupported_parameter");
}

function modelNames(modelId, aliases) {
  const names = new Set([modelId, ...(Array.isArray(aliases) ? aliases : [])]
    .filter((value) => typeof value === "string" && value.length > 0));
  for (const name of [...names]) {
    if (name.endsWith(":latest")) names.add(name.slice(0, -7));
    else if (!name.includes(":")) names.add(`${name}:latest`);
  }
  return names;
}

function normalizeModel(model, { modelId, modelAliases = [], chat = false }) {
  if (typeof model !== "string" || model.length === 0) {
    invalid("model", "model is required", chat ? 400 : 404, "model_not_found");
  }
  if (!modelNames(modelId, modelAliases).has(model)) {
    invalid("model", `model '${model}' not found`, 404, "model_not_found");
  }
  return model;
}

function normalizeStop(stop) {
  if (stop == null) return [];
  const values = typeof stop === "string" ? [stop] : stop;
  if (!Array.isArray(values) || values.length > MAX_STOP_SEQUENCES
    || values.some((value) => typeof value !== "string" || value.length === 0)
    || Buffer.byteLength(values.join(""), "utf8") > MAX_STOP_BYTES) {
    invalid("options.stop", "options.stop must contain one to sixteen non-empty strings within the byte limit.");
  }
  return [...values];
}

function normalizeOptions(options) {
  if (options == null) options = {};
  if (!options || typeof options !== "object" || Array.isArray(options)) {
    invalid("options", "options must be an object.");
  }
  rejectUnknown(options, new Set(["temperature", "top_p", "top_k", "num_predict", "stop"]), "options.");
  const defaults = { temperature: 0, top_p: 1, top_k: 1 };
  for (const [field, expected] of Object.entries(defaults)) {
    if (options[field] != null && options[field] !== expected) {
      invalid(`options.${field}`, `options.${field} supports only ${expected} in deterministic Caterpillar mode.`, 400,
        "unsupported_parameter");
    }
  }
  if (options.num_predict != null
    && (!Number.isSafeInteger(options.num_predict) || options.num_predict < 1 || options.num_predict > MAX_OUTPUT_TOKENS)) {
    invalid("options.num_predict", `options.num_predict must be an integer from 1 to ${MAX_OUTPUT_TOKENS}.`);
  }
  return {
    maxTokens: options.num_predict ?? null,
    stop: normalizeStop(options.stop),
  };
}

function common(body, context) {
  if (!body || typeof body !== "object" || Array.isArray(body)) {
    invalid(null, "Request body must be a JSON object.");
  }
  if (body.stream != null && typeof body.stream !== "boolean") invalid("stream", "stream must be boolean.");
  if (body.keep_alive != null) {
    invalid("keep_alive",
      "keep_alive is unsupported because this endpoint serves one process-resident bound model.",
      400, "unsupported_parameter");
  }
  return {
    model: normalizeModel(body.model, context),
    stream: body.stream !== false,
    ...normalizeOptions(body.options),
  };
}

export function validateOllamaGenerateRequest(body, context = {}) {
  rejectUnknown(body ?? {}, new Set(["model", "prompt", "system", "stream", "keep_alive", "options"]));
  const request = common(body, { ...context, chat: false });
  if (typeof body.prompt !== "string" || body.prompt.length === 0) invalid("prompt", "prompt must be a non-empty string.");
  if (body.system != null && (typeof body.system !== "string" || body.system.length === 0)) {
    invalid("system", "system must be a non-empty string.");
  }
  const messages = body.system == null ? null : [
    { role: "system", content: body.system },
    { role: "user", content: body.prompt },
  ];
  return { ...request, prompt: body.prompt, messages, rawPrompt: messages == null, kind: "generate" };
}

export function validateOllamaChatRequest(body, context = {}) {
  rejectUnknown(body ?? {}, new Set(["model", "messages", "stream", "keep_alive", "options", "tools"]));
  // ollama-python 0.6.2 serializes tools=[] even when its caller omits tools.
  // Accept that inert SDK default only; any actual tool declaration remains
  // outside CAT-OLLAMA-CHAT-1 and is rejected explicitly.
  if (body?.tools != null && (!Array.isArray(body.tools) || body.tools.length !== 0)) {
    invalid("tools", "tools are unsupported; only the official SDK default tools=[] is accepted.",
      400, "unsupported_parameter");
  }
  const request = common(body, { ...context, chat: true });
  const messages = normalizeChatMessages(body.messages);
  const lastUser = [...messages].reverse().find((message) => message.role === "user");
  return { ...request, prompt: lastUser.content, messages, rawPrompt: false, kind: "chat" };
}

function facts(completed, request, requestStartedNs) {
  const metadata = completed?.metadata ?? {};
  const native = metadata.nativeExecution ?? {};
  const promptIds = metadata.promptArtifact?.tokenIds;
  const emittedIds = native.emittedTokenIds;
  if (!Array.isArray(promptIds) || !Array.isArray(emittedIds)) {
    throw new ProtocolHttpError(500, "exact_usage_unavailable", "Exact Ollama token facts are unavailable.", {
      type: "server_error",
    });
  }
  let doneReason;
  if (native.finishReason === "max-tokens") doneReason = "length";
  else if (["eos", "stop-sequence", "finished", "repetition-loop"].includes(native.finishReason)) doneReason = "stop";
  else throw new ProtocolHttpError(500, "terminal_reason_unavailable", "A supported terminal reason is unavailable.");
  const promptEvalDuration = Math.max(0, Math.round((Number(metadata.prefillMs) || 0) * 1_000_000));
  const evalDuration = Math.max(0, Math.round((Number(metadata.decodeMs) || 0) * 1_000_000));
  const elapsed = requestStartedNs == null ? 0 : Number(process.hrtime.bigint() - requestStartedNs);
  return {
    promptIds,
    emittedIds,
    doneReason,
    totalDuration: Math.max(promptEvalDuration + evalDuration, Math.max(0, Math.round(elapsed))),
    loadDuration: 0,
    promptEvalDuration,
    evalDuration,
    request,
  };
}

function messageOrResponse(kind, text) {
  return kind === "chat"
    ? { message: { role: "assistant", content: text } }
    : { response: text };
}

function terminal({ completed, request, requestStartedNs, stream }) {
  const exact = facts(completed, request, requestStartedNs);
  return {
    model: request.model,
    created_at: new Date().toISOString(),
    ...messageOrResponse(request.kind, stream ? "" : completed.output ?? ""),
    done: true,
    done_reason: exact.doneReason,
    ...(request.kind === "generate" ? { context: [...exact.promptIds, ...exact.emittedIds] } : {}),
    total_duration: exact.totalDuration,
    load_duration: exact.loadDuration,
    prompt_eval_count: exact.promptIds.length,
    prompt_eval_duration: exact.promptEvalDuration,
    eval_count: exact.emittedIds.length,
    eval_duration: exact.evalDuration,
  };
}

export function buildOllamaResponse({ completed, request, requestStartedNs = null }) {
  return terminal({ completed, request, requestStartedNs, stream: false });
}

export async function writeOllamaNdjson({ events, write, request, requestStartedNs = null }) {
  let completed = null;
  for await (const event of events) {
    if (event.type === "delta" && event.text) {
      const payload = {
        model: request.model,
        created_at: new Date().toISOString(),
        ...messageOrResponse(request.kind, event.text),
        done: false,
      };
      if (await write(`${JSON.stringify(payload)}\n`) === false) {
        throw new ProtocolHttpError(499, "client_cancelled", "Client disconnected during streaming.");
      }
    } else if (event.type === "complete") {
      if (completed) throw new ProtocolHttpError(500, "duplicate_terminal_event", "Duplicate terminal event.");
      completed = event;
    } else if (event.type === "cancelled") {
      throw new ProtocolHttpError(499, "client_cancelled", "Generation was cancelled.");
    } else if (event.type === "error") {
      throw new ProtocolHttpError(500, "generation_failed", event.message ?? "Generation failed.");
    }
  }
  if (!completed) throw new ProtocolHttpError(500, "terminal_event_missing", "Generation ended without a terminal event.");
  const final = terminal({ completed, request, requestStartedNs, stream: true });
  if (await write(`${JSON.stringify(final)}\n`) === false) {
    throw new ProtocolHttpError(499, "client_cancelled", "Client disconnected before completion.");
  }
  return completed;
}
