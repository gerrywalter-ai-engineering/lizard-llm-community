import { ProtocolHttpError } from "./errors.js";
import { normalizeChatMessages } from "./messages.js";

export const CHAT_REQUEST_LIMITS = Object.freeze({
  maxMessages: 256,
  maxContentBytes: 512 * 1024,
  maxStopSequences: 16,
  maxStopBytes: 16 * 1024,
  maxOutputTokens: 4096,
});

function invalid(param, message, code = "invalid_parameter") {
  throw new ProtocolHttpError(400, code, message, { param });
}

export function validateChatRequest(body, {
  modelId,
  modelAliases = [],
  runtimeProvider = "lizard-native",
  limits = CHAT_REQUEST_LIMITS,
} = {}) {
  if (body.model != null && typeof body.model !== "string") {
    invalid("model", "model must be a string.");
  }
  const boundModelIds = new Set([
    modelId,
    ...(Array.isArray(modelAliases) ? modelAliases : []),
  ].filter((value) => typeof value === "string" && value.length > 0));
  if (body.model && boundModelIds.size > 0 && !boundModelIds.has(body.model)) {
    throw new ProtocolHttpError(404, "model_not_found", `Model '${body.model}' is not served by this endpoint.`, { param: "model" });
  }
  if (!Array.isArray(body.messages)) invalid("messages", "messages must be an array.");
  if (body.messages.length > limits.maxMessages) invalid("messages", `messages exceeds the ${limits.maxMessages}-entry limit.`, "content_too_large");
  let contentBytes = 0;
  for (let index = 0; index < body.messages.length; index += 1) {
    const message = body.messages[index];
    if (!message || typeof message !== "object" || Array.isArray(message)) invalid(`messages.${index}`, "Each message must be an object.");
    const contentValues = Array.isArray(message.content)
      ? message.content.map((part) => part?.text ?? "")
      : [message.content ?? ""];
    contentBytes += contentValues.reduce((sum, value) => sum + Buffer.byteLength(String(value), "utf8"), 0);
    if (contentBytes > limits.maxContentBytes) {
      invalid("messages", `Message content exceeds the ${limits.maxContentBytes}-byte limit.`, "content_too_large");
    }
  }
  if (body.stream != null && typeof body.stream !== "boolean") invalid("stream", "stream must be a boolean.");
  if (body.max_tokens != null
    && (!Number.isInteger(body.max_tokens) || body.max_tokens < 1 || body.max_tokens > limits.maxOutputTokens)) {
    invalid("max_tokens", `max_tokens must be an integer from 1 through ${limits.maxOutputTokens}.`);
  }
  for (const field of ["temperature", "top_p"]) {
    if (body[field] != null && !Number.isFinite(body[field])) invalid(field, `${field} must be a finite number.`);
  }
  if (body.top_p != null && (body.top_p < 0 || body.top_p > 1)) invalid("top_p", "top_p must be between 0 and 1.");
  if (body.temperature != null && body.temperature < 0) invalid("temperature", "temperature must be non-negative.");
  if (body.top_k != null && (!Number.isInteger(body.top_k) || body.top_k < 0 || body.top_k > 100000)) {
    invalid("top_k", "top_k must be an integer from 0 through 100000.");
  }
  if (body.seed != null && !Number.isSafeInteger(body.seed)) invalid("seed", "seed must be a safe integer.");
  if (body.user != null && (typeof body.user !== "string" || Buffer.byteLength(body.user, "utf8") > 256)) {
    invalid("user", "user must be a string no longer than 256 bytes.");
  }
  if (runtimeProvider === "caterpillar" && Number(body.temperature ?? 0) !== 0) {
    invalid("temperature", "Caterpillar currently supports deterministic greedy sampling only.", "unsupported_parameter");
  }
  if (runtimeProvider === "caterpillar" && body.top_p != null && body.top_p !== 1) {
    invalid("top_p", "Caterpillar currently supports top_p=1 only.", "unsupported_parameter");
  }
  if (runtimeProvider === "caterpillar" && body.top_k != null && body.top_k !== 1) {
    invalid("top_k", "Caterpillar currently supports top_k=1 only.", "unsupported_parameter");
  }

  const stops = body.stop == null ? [] : Array.isArray(body.stop) ? body.stop : [body.stop];
  if (stops.some((value) => typeof value !== "string")) invalid("stop", "stop must be a string or an array of strings.");
  if (stops.length > limits.maxStopSequences
    || stops.reduce((sum, value) => sum + Buffer.byteLength(value, "utf8"), 0) > limits.maxStopBytes) {
    invalid("stop", "stop exceeds the configured sequence or byte limit.", "content_too_large");
  }

  if (body.max_completion_tokens != null
    && (!Number.isInteger(body.max_completion_tokens)
      || body.max_completion_tokens < 1
      || body.max_completion_tokens > limits.maxOutputTokens)) {
    invalid("max_completion_tokens", `max_completion_tokens must be an integer from 1 through ${limits.maxOutputTokens}.`);
  }
  if (body.max_tokens != null && body.max_completion_tokens != null && body.max_tokens !== body.max_completion_tokens) {
    invalid("max_completion_tokens", "max_tokens and max_completion_tokens must match when both are provided.");
  }
  if (body.stream_options != null) {
    if (!body.stream_options || typeof body.stream_options !== "object" || Array.isArray(body.stream_options)
      || typeof body.stream_options.include_usage !== "boolean"
      || Object.keys(body.stream_options).some((field) => field !== "include_usage")) {
      invalid("stream_options", "stream_options supports only the boolean include_usage field.");
    }
    if (body.stream !== true) {
      invalid("stream_options", "stream_options requires stream=true.");
    }
  }
  if (body.response_format != null
    && (body.response_format?.type !== "text" || Object.keys(body.response_format).some((field) => field !== "type"))) {
    invalid("response_format", "Only response_format={\"type\":\"text\"} is supported.", "unsupported_parameter");
  }
  if (body.tools != null && (!Array.isArray(body.tools) || body.tools.length !== 0)) {
    invalid("tools", "Tools are not supported by this profile.", "unsupported_parameter");
  }
  if (body.tool_choice != null && body.tool_choice !== "none") {
    invalid("tool_choice", "tool_choice supports only 'none' without the tools profile.", "unsupported_parameter");
  }
  if (body.parallel_tool_calls != null && body.parallel_tool_calls !== false) {
    invalid("parallel_tool_calls", "parallel_tool_calls supports only false without the tools profile.", "unsupported_parameter");
  }

  const defaultOnly = {
    n: 1,
    frequency_penalty: 0,
    presence_penalty: 0,
    logprobs: false,
  };
  for (const [field, expected] of Object.entries(defaultOnly)) {
    if (body[field] !== undefined && body[field] !== expected) {
      invalid(field, `${field}=${JSON.stringify(body[field])} is not supported.`, "unsupported_parameter");
    }
  }
  const supported = new Set([
    "model", "messages", "stream", "stream_options", "max_tokens", "max_completion_tokens",
    "temperature", "top_k", "top_p", "seed", "stop", "user", "response_format", "tools", "tool_choice",
    "parallel_tool_calls",
    ...Object.keys(defaultOnly),
  ]);
  const unsupported = Object.keys(body).find((field) => !supported.has(field));
  if (unsupported) invalid(unsupported, `${unsupported} is not supported.`, "unsupported_parameter");
  return {
    ...body,
    messages: normalizeChatMessages(body.messages),
    stop: stops,
    max_tokens: body.max_tokens ?? body.max_completion_tokens,
  };
}
