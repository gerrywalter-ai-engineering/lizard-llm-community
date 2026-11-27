import { ProtocolHttpError } from "./errors.js";

export function assertConformantChatTemplate(promptArtifact) {
  if (["unsupported_jinja", "unresolved_placeholders", "unsupported_template_multiturn"]
    .includes(promptArtifact?.templateRenderer)) {
    throw new ProtocolHttpError(
      422,
      "chat_template_unsupported",
      "The loaded model chat template cannot render this ordered conversation safely.",
      { type: "unprocessable_entity", param: "messages" },
    );
  }
  return promptArtifact;
}

export function mapOpenAiFinishReason(nativeReason) {
  if (["eos", "stop-sequence", "finished", "repetition-loop"].includes(nativeReason)) return "stop";
  if (nativeReason === "max-tokens") return "length";
  if (nativeReason === "tool-calls") return "tool_calls";
  throw new ProtocolHttpError(
    500,
    "terminal_reason_unavailable",
    `Native generation ended without a supported terminal reason (${nativeReason ?? "missing"}).`,
    { type: "server_error" },
  );
}

export function buildOpenAiUsage(metadata) {
  const promptArtifact = metadata?.promptArtifact ?? null;
  const promptTokens = Number(promptArtifact?.tokenCount);
  const completionIds = metadata?.nativeExecution?.emittedTokenIds;
  if (!Number.isInteger(promptTokens) || promptTokens < 0 || promptArtifact?.tokenCountMode === "estimate") {
    throw new ProtocolHttpError(422, "exact_usage_unavailable", "Exact prompt token usage is unavailable for this model.", {
      type: "unprocessable_entity",
    });
  }
  if (!Array.isArray(completionIds)) {
    throw new ProtocolHttpError(500, "completion_usage_unavailable", "Exact emitted completion token IDs are unavailable.", {
      type: "server_error",
    });
  }
  const cachedTokens = Math.max(0, Math.min(
    promptTokens,
    Math.trunc(Number(metadata?.nativeExecution?.cachedPromptTokens) || 0),
  ));
  return {
    prompt_tokens: promptTokens,
    completion_tokens: completionIds.length,
    total_tokens: promptTokens + completionIds.length,
    prompt_tokens_details: { cached_tokens: cachedTokens },
  };
}

export function buildOpenAiChatCompletion({
  id,
  created,
  model,
  completed,
  runtime = null,
  includeRuntime = true,
}) {
  const metadata = completed?.metadata ?? {};
  const payload = {
    id,
    object: "chat.completion",
    created,
    model,
    choices: [{
      index: 0,
      message: { role: "assistant", content: completed?.output ?? "" },
      finish_reason: mapOpenAiFinishReason(metadata?.nativeExecution?.finishReason),
    }],
    usage: buildOpenAiUsage(metadata),
  };
  if (includeRuntime) payload.runtime = runtime ?? metadata;
  return payload;
}

function streamEnvelope({ id, created, model, choices, usage = undefined, runtime = undefined }) {
  return {
    id,
    object: "chat.completion.chunk",
    created,
    model,
    choices,
    ...(usage === undefined ? {} : { usage }),
    ...(runtime === undefined ? {} : { runtime }),
  };
}

export function buildOpenAiStreamStart({ id, created, model }) {
  return streamEnvelope({
    id,
    created,
    model,
    choices: [{
      index: 0,
      delta: { role: "assistant", content: "" },
      finish_reason: null,
    }],
  });
}

export function buildOpenAiStreamDelta({ id, created, model, text }) {
  return streamEnvelope({
    id,
    created,
    model,
    choices: [{ index: 0, delta: { content: String(text) }, finish_reason: null }],
  });
}

export function buildOpenAiStreamFinish({
  id,
  created,
  model,
  nativeFinishReason,
  runtime = undefined,
}) {
  return streamEnvelope({
    id,
    created,
    model,
    choices: [{
      index: 0,
      delta: {},
      finish_reason: mapOpenAiFinishReason(nativeFinishReason),
    }],
    runtime,
  });
}

export function buildOpenAiStreamUsage({ id, created, model, metadata }) {
  return streamEnvelope({
    id,
    created,
    model,
    choices: [],
    usage: buildOpenAiUsage(metadata),
  });
}

export function serializeOpenAiSseData(payload) {
  return `data: ${JSON.stringify(payload)}\n\n`;
}

export const OPENAI_SSE_DONE = "data: [DONE]\n\n";

export async function writeOpenAiChatSse({
  events,
  write,
  id,
  created,
  model,
  includeUsage = false,
  includeRuntime = true,
  completeRuntime = () => undefined,
  onFirstContent = () => {},
}) {
  if (!events || typeof events[Symbol.asyncIterator] !== "function") {
    throw new TypeError("OpenAI SSE events must be an async iterable.");
  }
  if (typeof write !== "function") throw new TypeError("OpenAI SSE write callback is required.");
  const writePayload = async (payload) => {
    if (await write(serializeOpenAiSseData(payload)) === false) {
      throw new ProtocolHttpError(499, "client_cancelled", "Client disconnected during streaming.");
    }
  };

  await writePayload(buildOpenAiStreamStart({ id, created, model }));
  let completed = null;
  let firstContent = true;
  for await (const event of events) {
    if (event?.type === "delta" && event.text) {
      if (firstContent) {
        firstContent = false;
        onFirstContent(event);
      }
      await writePayload(buildOpenAiStreamDelta({ id, created, model, text: event.text }));
    } else if (event?.type === "complete") {
      if (completed) {
        throw new ProtocolHttpError(500, "duplicate_terminal_event", "Generation emitted more than one terminal event.", {
          type: "server_error",
        });
      }
      completed = event;
    } else if (event?.type === "cancelled") {
      throw new ProtocolHttpError(499, "client_cancelled", "Generation was cancelled before completion.");
    } else if (event?.type === "error") {
      throw new ProtocolHttpError(500, "generation_failed", event.message ?? "Generation failed.", {
        type: "server_error",
      });
    }
  }
  if (!completed) {
    throw new ProtocolHttpError(500, "terminal_event_missing", "Generation ended without a terminal event.", {
      type: "server_error",
    });
  }

  // Derive all terminal facts before writing the finish chunk. If exact usage
  // or the native finish reason is missing, the stream stays incomplete and
  // can never be mistaken for a successful response.
  const runtime = await completeRuntime(completed);
  const finish = buildOpenAiStreamFinish({
    id,
    created,
    model,
    nativeFinishReason: completed.metadata?.nativeExecution?.finishReason,
    runtime: includeRuntime ? runtime : undefined,
  });
  const usage = includeUsage
    ? buildOpenAiStreamUsage({ id, created, model, metadata: completed.metadata })
    : null;
  await writePayload(finish);
  if (usage) await writePayload(usage);
  if (await write(OPENAI_SSE_DONE) === false) {
    throw new ProtocolHttpError(499, "client_cancelled", "Client disconnected before the stream completed.");
  }
  return completed;
}
