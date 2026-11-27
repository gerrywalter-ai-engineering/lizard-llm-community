import { ProtocolHttpError } from "./errors.js";
import { buildOpenAiUsage, mapOpenAiFinishReason } from "./openai.js";

const MAX_COMPLETION_TOKENS = 4096;
const MAX_PROMPT_TOKEN_IDS = 131_072;
const MAX_STOP_SEQUENCES = 16;
const MAX_STOP_BYTES = 16 * 1024;

function invalid(param, message, statusCode = 400) {
  throw new ProtocolHttpError(statusCode, "invalid_parameter", message, {
    type: statusCode === 422 ? "unprocessable_entity" : "invalid_request_error",
    param,
  });
}

function rejectUnknown(body, allowed) {
  const unknown = Object.keys(body).find((field) => !allowed.has(field));
  if (unknown) invalid(unknown, `Unsupported parameter: ${unknown}.`);
}

function defaultOnly(body, field, expected) {
  if (body[field] != null && body[field] !== expected) {
    invalid(field, `${field} supports only ${JSON.stringify(expected)} in deterministic Caterpillar mode.`);
  }
}

function normalizeStop(stop) {
  if (stop == null) return [];
  const values = typeof stop === "string" ? [stop] : stop;
  if (!Array.isArray(values) || values.length > MAX_STOP_SEQUENCES
    || values.some((value) => typeof value !== "string" || value.length === 0)
    || Buffer.byteLength(values.join(""), "utf8") > MAX_STOP_BYTES) {
    invalid("stop", "stop must contain one to sixteen non-empty strings within the byte limit.");
  }
  return [...values];
}

function normalizeTokenLimit(value, field) {
  if (value == null) return null;
  if (!Number.isSafeInteger(value) || value < 1 || value > MAX_COMPLETION_TOKENS) {
    invalid(field, `${field} must be an integer from 1 to ${MAX_COMPLETION_TOKENS}.`);
  }
  return value;
}

function normalizePrompt(prompt, { allowTokenIds }) {
  if (typeof prompt === "string") {
    if (prompt.length === 0) invalid("prompt", "prompt must not be empty.");
    return { prompt, promptTokenIds: null };
  }
  if (allowTokenIds && Array.isArray(prompt)) {
    if (prompt.length === 0 || prompt.length > MAX_PROMPT_TOKEN_IDS
      || prompt.some((tokenId) => !Number.isSafeInteger(tokenId) || tokenId < 0)) {
      invalid("prompt", `prompt token IDs must be a non-empty array of at most ${MAX_PROMPT_TOKEN_IDS} non-negative integers.`);
    }
    return { prompt: null, promptTokenIds: [...prompt] };
  }
  invalid("prompt", allowTokenIds
    ? "prompt must be one string or one token-ID array."
    : "prompt must be one string; prompt arrays are unsupported.");
}

function assertModel(model, modelId, modelAliases = []) {
  if (model == null) return;
  if (typeof model !== "string") invalid("model", "model must be a string.");
  if (![modelId, ...modelAliases].includes(model)) {
    throw new ProtocolHttpError(404, "model_not_found", `Model '${model}' is not loaded.`, {
      type: "invalid_request_error",
      param: "model",
    });
  }
}

export function validateLlamaCompletionRequest(body, {
  openAi = false,
  modelId = null,
  modelAliases = [],
} = {}) {
  if (!body || typeof body !== "object" || Array.isArray(body)) {
    invalid(null, "Request body must be a JSON object.");
  }
  const allowed = openAi
    ? new Set([
      "model", "prompt", "max_tokens", "stream", "stop", "temperature", "top_p", "seed",
      "n", "logprobs", "echo", "suffix", "best_of", "user",
    ])
    : new Set([
      "prompt", "n_predict", "stream", "stop", "temperature", "top_k", "top_p", "seed",
      "cache_prompt", "return_tokens", "n_probs", "ignore_eos",
    ]);
  rejectUnknown(body, allowed);
  if (openAi) assertModel(body.model, modelId, modelAliases);
  const normalizedPrompt = normalizePrompt(body.prompt, { allowTokenIds: !openAi });
  if (body.stream != null && typeof body.stream !== "boolean") invalid("stream", "stream must be boolean.");
  if (body.seed != null && (!Number.isSafeInteger(body.seed) || body.seed < 0)) {
    invalid("seed", "seed must be a non-negative safe integer.");
  }
  defaultOnly(body, "temperature", 0);
  defaultOnly(body, "top_p", 1);
  if (openAi) {
    defaultOnly(body, "n", 1);
    defaultOnly(body, "echo", false);
    defaultOnly(body, "best_of", 1);
    if (body.logprobs != null && body.logprobs !== false && body.logprobs !== 0) {
      invalid("logprobs", "logprobs are unsupported.");
    }
    if (body.suffix != null && body.suffix !== "") invalid("suffix", "suffix is unsupported.");
    if (body.user != null && (typeof body.user !== "string" || body.user.length > 256)) {
      invalid("user", "user must be a string of at most 256 characters.");
    }
  } else {
    defaultOnly(body, "top_k", 1);
    defaultOnly(body, "n_probs", 0);
    defaultOnly(body, "ignore_eos", false);
    if (body.cache_prompt != null && typeof body.cache_prompt !== "boolean") {
      invalid("cache_prompt", "cache_prompt must be boolean.");
    }
    if (body.return_tokens != null && typeof body.return_tokens !== "boolean") {
      invalid("return_tokens", "return_tokens must be boolean.");
    }
  }
  return {
    ...normalizedPrompt,
    model: modelId,
    stream: body.stream === true,
    stop: normalizeStop(body.stop),
    maxTokens: normalizeTokenLimit(openAi ? body.max_tokens : body.n_predict,
      openAi ? "max_tokens" : "n_predict"),
    seed: body.seed ?? null,
    cachePrompt: openAi ? true : body.cache_prompt !== false,
    returnTokens: openAi ? false : body.return_tokens === true,
    user: openAi ? body.user ?? null : null,
  };
}

export function validateTokenizeRequest(body) {
  if (!body || typeof body !== "object" || Array.isArray(body)) invalid(null, "Request body must be an object.");
  rejectUnknown(body, new Set(["content", "add_special", "parse_special", "with_pieces"]));
  if (typeof body.content !== "string") invalid("content", "content must be a string.");
  for (const field of ["add_special", "parse_special", "with_pieces"]) {
    if (body[field] != null && typeof body[field] !== "boolean") invalid(field, `${field} must be boolean.`);
  }
  return {
    content: body.content,
    addSpecial: body.add_special ?? false,
    parseSpecial: body.parse_special ?? true,
    withPieces: body.with_pieces ?? false,
  };
}

export function validateDetokenizeRequest(body) {
  if (!body || typeof body !== "object" || Array.isArray(body)) invalid(null, "Request body must be an object.");
  rejectUnknown(body, new Set(["tokens"]));
  if (!Array.isArray(body.tokens) || body.tokens.length > MAX_PROMPT_TOKEN_IDS
    || body.tokens.some((tokenId) => !Number.isSafeInteger(tokenId) || tokenId < 0)) {
    invalid("tokens", `tokens must be an array of at most ${MAX_PROMPT_TOKEN_IDS} non-negative integers.`);
  }
  return { tokens: [...body.tokens] };
}

function completionFacts(completed, requestStop = []) {
  const metadata = completed?.metadata ?? {};
  const native = metadata.nativeExecution ?? {};
  const promptTokens = Number(metadata.promptArtifact?.tokenCount);
  const emittedTokenIds = native.emittedTokenIds;
  if (!Number.isSafeInteger(promptTokens) || !Array.isArray(emittedTokenIds)) {
    throw new ProtocolHttpError(500, "exact_usage_unavailable", "Exact completion facts are unavailable.", {
      type: "server_error",
    });
  }
  const finishReason = mapOpenAiFinishReason(native.finishReason);
  const stopWord = requestStop.includes(native.stopReason) ? native.stopReason : "";
  const stopType = native.finishReason === "max-tokens" ? "limit" : stopWord ? "word" : "eos";
  const prefillMs = Math.max(0, Number(metadata.prefillMs) || 0);
  const decodeMs = Math.max(0, Number(metadata.decodeMs) || 0);
  const cached = Math.max(0, Math.trunc(Number(native.cachedPromptTokens) || 0));
  return {
    metadata,
    native,
    promptTokens,
    emittedTokenIds,
    finishReason,
    stopType,
    stopWord,
    cached,
    timings: {
      cache_n: cached,
      prompt_n: Math.max(0, promptTokens - cached),
      prompt_ms: prefillMs,
      predicted_n: emittedTokenIds.length,
      predicted_ms: decodeMs,
      predicted_per_second: decodeMs > 0 ? (emittedTokenIds.length / decodeMs) * 1000 : null,
    },
  };
}

function llamaTerminal({ completed, model, request, stream }) {
  const facts = completionFacts(completed, request.stop);
  const response = {
    index: 0,
    content: stream ? "" : completed.output ?? "",
    ...(request.returnTokens || stream ? { tokens: stream ? [] : facts.emittedTokenIds } : {}),
    id_slot: 0,
    stop: true,
    model,
    tokens_predicted: facts.emittedTokenIds.length,
    tokens_evaluated: facts.promptTokens,
    generation_settings: {
      seed: request.seed,
      temperature: 0,
      top_k: 1,
      top_p: 1,
      stop: request.stop,
      max_tokens: request.maxTokens,
      n_predict: request.maxTokens,
      stream,
    },
    prompt: facts.metadata.promptArtifact?.renderedPrompt ?? request.prompt ?? "",
    truncated: false,
    stop_type: facts.stopType,
    stopping_word: facts.stopWord,
    tokens_cached: facts.cached,
    timings: facts.timings,
  };
  return { response, facts };
}

export function buildLlamaCompletion({ completed, model, request }) {
  return llamaTerminal({ completed, model, request, stream: false }).response;
}

export async function writeLlamaCompletionSse({ events, write, model, request }) {
  let completed = null;
  for await (const event of events) {
    if (event.type === "delta" && event.text) {
      const payload = {
        index: 0,
        content: event.text,
        tokens: Number.isSafeInteger(event.tokenId) ? [event.tokenId] : [],
        stop: false,
        id_slot: -1,
      };
      if (await write(`data: ${JSON.stringify(payload)}\n\n`) === false) {
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
  const terminal = llamaTerminal({ completed, model, request, stream: true }).response;
  if (await write(`data: ${JSON.stringify(terminal)}\n\n`) === false) {
    throw new ProtocolHttpError(499, "client_cancelled", "Client disconnected during streaming.");
  }
  return completed;
}

function openAiEnvelope({ id, created, model, text, finishReason, usage, timings }) {
  return {
    id,
    object: "text_completion",
    created,
    model,
    choices: [{ text, index: 0, logprobs: null, finish_reason: finishReason }],
    ...(usage ? { usage } : {}),
    ...(timings ? { timings } : {}),
  };
}

export function buildOpenAiTextCompletion({ id, created, model, completed, request }) {
  const facts = completionFacts(completed, request.stop);
  return openAiEnvelope({
    id,
    created,
    model,
    text: completed.output ?? "",
    finishReason: facts.finishReason,
    usage: buildOpenAiUsage(facts.metadata),
    timings: facts.timings,
  });
}

export async function writeOpenAiTextCompletionSse({ events, write, id, created, model, request }) {
  let completed = null;
  for await (const event of events) {
    if (event.type === "delta" && event.text) {
      const payload = openAiEnvelope({
        id, created, model, text: event.text, finishReason: null,
      });
      if (await write(`data: ${JSON.stringify(payload)}\n\n`) === false) {
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
  const facts = completionFacts(completed, request.stop);
  const terminal = openAiEnvelope({
    id,
    created,
    model,
    text: "",
    finishReason: facts.finishReason,
    usage: buildOpenAiUsage(facts.metadata),
    timings: facts.timings,
  });
  // As in the chat adapter, derive exact terminal facts before any successful
  // finish marker reaches the wire.
  if (await write(`data: ${JSON.stringify(terminal)}\n\n`) === false
    || await write("data: [DONE]\n\n") === false) {
    throw new ProtocolHttpError(499, "client_cancelled", "Client disconnected before completion.");
  }
  return completed;
}
