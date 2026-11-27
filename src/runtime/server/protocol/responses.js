import { ProtocolHttpError } from "./errors.js";
import { buildOpenAiUsage } from "./openai.js";

const MAX_OUTPUT_TOKENS = 4096;
const MAX_INPUT_ITEMS = 256;
const MAX_INPUT_BYTES = 512 * 1024;

function invalid(param, message, code = "invalid_parameter") {
  throw new ProtocolHttpError(400, code, message, { param });
}

function defaultOnly(body, field, expected) {
  if (body[field] !== undefined && JSON.stringify(body[field]) !== JSON.stringify(expected)) {
    invalid(field, `${field} supports only ${JSON.stringify(expected)} in CAT-RESPONSES-1.`, "unsupported_parameter");
  }
}

function normalizeContent(content, param, role) {
  if (typeof content === "string") return content;
  if (!Array.isArray(content) || content.length === 0) invalid(param, "content must be text or a non-empty text-part array.");
  return content.map((part, index) => {
    if (!part || typeof part !== "object" || Array.isArray(part)) invalid(`${param}.${index}`, "content parts must be objects.");
    const allowedType = role === "assistant" ? "output_text" : "input_text";
    if (part.type !== allowedType) {
      invalid(`${param}.${index}.type`, `Only ${allowedType} parts are supported.`, "unsupported_parameter");
    }
    if (typeof part.text !== "string") invalid(`${param}.${index}.text`, "text must be a string.");
    const unknown = Object.keys(part).find((field) => !["type", "text"].includes(field));
    if (unknown) invalid(`${param}.${index}.${unknown}`, `${unknown} is unsupported.`, "unsupported_parameter");
    return part.text;
  }).join("");
}

function normalizeInput(input) {
  if (typeof input === "string") {
    if (!input.trim()) invalid("input", "input must not be empty.");
    return [{ role: "user", content: input }];
  }
  if (!Array.isArray(input) || input.length === 0 || input.length > MAX_INPUT_ITEMS) {
    invalid("input", `input must be text or one to ${MAX_INPUT_ITEMS} message items.`);
  }
  const messages = input.map((item, index) => {
    const param = `input.${index}`;
    if (!item || typeof item !== "object" || Array.isArray(item)) invalid(param, "input items must be objects.");
    if (item.type != null && item.type !== "message") {
      invalid(`${param}.type`, "Only message input items are supported.", "unsupported_parameter");
    }
    if (!["developer", "system", "user", "assistant"].includes(item.role)) {
      invalid(`${param}.role`, "Only developer, system, user, and assistant message roles are supported.",
        "unsupported_parameter");
    }
    const unknown = Object.keys(item).find((field) => !["type", "role", "content"].includes(field));
    if (unknown) invalid(`${param}.${unknown}`, `${unknown} is unsupported.`, "unsupported_parameter");
    const content = normalizeContent(item.content, `${param}.content`, item.role);
    if (!content.trim()) invalid(`${param}.content`, "message content must not be empty.");
    return { role: item.role === "developer" ? "system" : item.role, content };
  });
  const finalConversation = [...messages].reverse().find((message) => message.role !== "system");
  if (finalConversation?.role !== "user") invalid("input", "The final non-system input item must be a user message.");
  return messages;
}

function assertModel(model, modelId, aliases) {
  if (typeof model !== "string" || model.length === 0) invalid("model", "model is required.");
  if (![modelId, ...(Array.isArray(aliases) ? aliases : [])].includes(model)) {
    throw new ProtocolHttpError(404, "model_not_found", `Model '${model}' is not served by this endpoint.`, {
      param: "model",
    });
  }
}

export function validateResponsesRequest(body, { modelId, modelAliases = [] } = {}) {
  if (!body || typeof body !== "object" || Array.isArray(body)) invalid(null, "Request body must be a JSON object.");
  const allowed = new Set([
    "model", "input", "instructions", "max_output_tokens", "stream", "store", "metadata", "temperature",
    "top_p", "text", "tools", "tool_choice", "parallel_tool_calls", "previous_response_id", "conversation",
    "include", "user", "reasoning", "background", "truncation", "max_tool_calls",
  ]);
  const unknown = Object.keys(body).find((field) => !allowed.has(field));
  if (unknown) invalid(unknown, `${unknown} is unsupported.`, "unsupported_parameter");
  assertModel(body.model, modelId, modelAliases);
  if (body.stream != null && typeof body.stream !== "boolean") invalid("stream", "stream must be boolean.");
  if (body.instructions != null && (typeof body.instructions !== "string" || !body.instructions.trim())) {
    invalid("instructions", "instructions must be a non-empty string.");
  }
  if (body.max_output_tokens != null && (!Number.isSafeInteger(body.max_output_tokens)
    || body.max_output_tokens < 1 || body.max_output_tokens > MAX_OUTPUT_TOKENS)) {
    invalid("max_output_tokens", `max_output_tokens must be an integer from 1 to ${MAX_OUTPUT_TOKENS}.`);
  }
  if (body.previous_response_id != null) invalid("previous_response_id", "Stateful response chaining is unsupported.", "unsupported_parameter");
  if (body.conversation != null) invalid("conversation", "Conversation storage is unsupported.", "unsupported_parameter");
  if (body.store === true) invalid("store", "CAT-RESPONSES-1 is stateless and does not store responses.", "unsupported_parameter");
  if (body.store != null && body.store !== false) invalid("store", "store must be false when provided.");
  defaultOnly(body, "temperature", 0);
  defaultOnly(body, "top_p", 1);
  defaultOnly(body, "tools", []);
  defaultOnly(body, "tool_choice", "none");
  defaultOnly(body, "parallel_tool_calls", false);
  defaultOnly(body, "include", []);
  defaultOnly(body, "background", false);
  defaultOnly(body, "truncation", "disabled");
  if (body.reasoning != null) invalid("reasoning", "reasoning controls are unsupported.", "unsupported_parameter");
  if (body.max_tool_calls != null) invalid("max_tool_calls", "Tools are unsupported.", "unsupported_parameter");
  if (body.text != null) {
    if (!body.text || typeof body.text !== "object" || Array.isArray(body.text)) invalid("text", "text must be an object.");
    const unknownText = Object.keys(body.text).find((field) => !["format", "verbosity"].includes(field));
    if (unknownText) invalid(`text.${unknownText}`, `${unknownText} is unsupported.`, "unsupported_parameter");
    if (body.text.format != null && (body.text.format?.type !== "text"
      || Object.keys(body.text.format).some((field) => field !== "type"))) {
      invalid("text.format", "Only text.format={\"type\":\"text\"} is supported.", "unsupported_parameter");
    }
    if (body.text.verbosity != null && body.text.verbosity !== "medium") {
      invalid("text.verbosity", "Only the default verbosity 'medium' is supported.", "unsupported_parameter");
    }
  }
  if (body.metadata != null && (!body.metadata || typeof body.metadata !== "object" || Array.isArray(body.metadata)
    || Object.keys(body.metadata).length > 16
    || Object.entries(body.metadata).some(([key, value]) => key.length > 64 || typeof value !== "string" || value.length > 512))) {
    invalid("metadata", "metadata must contain at most 16 bounded string entries.");
  }
  if (body.user != null && (typeof body.user !== "string" || Buffer.byteLength(body.user, "utf8") > 256)) {
    invalid("user", "user must be a string no longer than 256 bytes.");
  }
  const messages = normalizeInput(body.input);
  if (body.instructions) messages.unshift({ role: "system", content: body.instructions });
  const contentBytes = messages.reduce((sum, message) => sum + Buffer.byteLength(message.content, "utf8"), 0);
  if (contentBytes > MAX_INPUT_BYTES) invalid("input", `input exceeds ${MAX_INPUT_BYTES} bytes.`, "content_too_large");
  return {
    model: modelId,
    requestedModel: body.model,
    messages,
    prompt: [...messages].reverse().find((message) => message.role === "user").content,
    instructions: body.instructions ?? null,
    maxOutputTokens: body.max_output_tokens ?? null,
    stream: body.stream === true,
    metadata: body.metadata ?? {},
    user: body.user ?? null,
  };
}

function usage(metadata) {
  const chat = buildOpenAiUsage(metadata);
  return {
    input_tokens: chat.prompt_tokens,
    input_tokens_details: { cached_tokens: chat.prompt_tokens_details.cached_tokens },
    output_tokens: chat.completion_tokens,
    output_tokens_details: { reasoning_tokens: 0 },
    total_tokens: chat.total_tokens,
  };
}

function messageItem({ messageId, text, status = "completed" }) {
  return {
    id: messageId,
    type: "message",
    status,
    content: [{ type: "output_text", annotations: [], logprobs: [], text }],
    role: "assistant",
  };
}

function terminalStatus(completed) {
  return completed?.metadata?.nativeExecution?.finishReason === "max-tokens"
    ? { status: "incomplete", incompleteDetails: { reason: "max_output_tokens" } }
    : { status: "completed", incompleteDetails: null };
}

function envelope({ id, createdAt, request, status, output, responseUsage = null,
  incompleteDetails = null }) {
  return {
    id,
    object: "response",
    created_at: createdAt,
    status,
    completed_at: status === "completed" ? Math.floor(Date.now() / 1000) : null,
    background: false,
    error: null,
    incomplete_details: incompleteDetails,
    instructions: request.instructions,
    max_output_tokens: request.maxOutputTokens,
    max_tool_calls: null,
    model: request.model,
    output,
    parallel_tool_calls: false,
    previous_response_id: null,
    prompt: null,
    reasoning: { effort: null, summary: null },
    service_tier: "default",
    store: false,
    temperature: 0,
    text: { format: { type: "text" }, verbosity: "medium" },
    tool_choice: "none",
    tools: [],
    top_logprobs: 0,
    top_p: 1,
    truncation: "disabled",
    usage: responseUsage,
    user: request.user,
    metadata: request.metadata,
  };
}

export function buildResponsesResponse({ id, messageId, createdAt, request, completed }) {
  const terminal = terminalStatus(completed);
  return envelope({
    id,
    createdAt,
    request,
    status: terminal.status,
    incompleteDetails: terminal.incompleteDetails,
    output: [messageItem({ messageId, text: completed.output ?? "", status: terminal.status })],
    responseUsage: usage(completed.metadata),
  });
}

export async function writeResponsesSse({ events, write, id, messageId, createdAt, request }) {
  let sequence = 0;
  const emit = async (type, fields) => {
    const payload = { type, sequence_number: sequence, ...fields };
    sequence += 1;
    if (await write(`event: ${type}\ndata: ${JSON.stringify(payload)}\n\n`) === false) {
      throw new ProtocolHttpError(499, "client_cancelled", "Client disconnected during Responses streaming.");
    }
  };
  const initial = envelope({ id, createdAt, request, status: "in_progress", output: [] });
  await emit("response.created", { response: initial });
  await emit("response.in_progress", { response: initial });
  const pendingItem = { id: messageId, type: "message", status: "in_progress", content: [], role: "assistant" };
  await emit("response.output_item.added", { output_index: 0, item: pendingItem });
  await emit("response.content_part.added", {
    item_id: messageId,
    output_index: 0,
    content_index: 0,
    part: { type: "output_text", annotations: [], logprobs: [], text: "" },
  });
  let completed = null;
  let outputText = "";
  for await (const event of events) {
    if (event.type === "delta" && event.text) {
      outputText += event.text;
      await emit("response.output_text.delta", {
        item_id: messageId, output_index: 0, content_index: 0, delta: event.text, logprobs: [],
      });
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
  const finalText = completed.output ?? outputText;
  const terminal = terminalStatus(completed);
  const finalItem = messageItem({ messageId, text: finalText, status: terminal.status });
  await emit("response.output_text.done", {
    item_id: messageId, output_index: 0, content_index: 0, text: finalText, logprobs: [],
  });
  await emit("response.content_part.done", {
    item_id: messageId, output_index: 0, content_index: 0, part: finalItem.content[0],
  });
  await emit("response.output_item.done", { output_index: 0, item: finalItem });
  const response = buildResponsesResponse({ id, messageId, createdAt, request, completed });
  await emit(terminal.status === "completed" ? "response.completed" : "response.incomplete", { response });
  return completed;
}
