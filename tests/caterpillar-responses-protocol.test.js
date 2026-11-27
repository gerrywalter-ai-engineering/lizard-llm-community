import test from "node:test";
import assert from "node:assert/strict";
import {
  buildResponsesResponse,
  validateResponsesRequest,
  writeResponsesSse,
} from "../src/runtime/server/protocol/responses.js";

function completed(finishReason = "max-tokens") {
  return {
    type: "complete",
    output: "world hello",
    metadata: {
      promptArtifact: { tokenCount: 1, tokenCountMode: "exact-vocab" },
      nativeExecution: { emittedTokenIds: [1, 0], cachedPromptTokens: 0, finishReason },
    },
  };
}

async function* events() {
  yield { type: "delta", text: "world" };
  yield { type: "delta", text: " hello" };
  yield completed();
}

const identity = { id: "resp_abc", messageId: "msg_abc", createdAt: 1785744094 };

test("Responses validator maps the declared stateless text-only subset", () => {
  const request = validateResponsesRequest({
    model: "alias",
    instructions: "Be brief.",
    input: [
      { role: "user", content: [{ type: "input_text", text: "first" }] },
      { role: "assistant", content: [{ type: "output_text", text: "answer" }] },
      { type: "message", role: "user", content: "hello" },
    ],
    max_output_tokens: 2,
    store: false,
    text: { format: { type: "text" }, verbosity: "medium" },
    tools: [],
    tool_choice: "none",
    parallel_tool_calls: false,
    metadata: { trace: "safe" },
  }, { modelId: "tiny", modelAliases: ["alias"] });
  assert.equal(request.model, "tiny");
  assert.equal(request.requestedModel, "alias");
  assert.deepEqual(request.messages.map((message) => message.role), ["system", "user", "assistant", "user"]);
  assert.equal(request.prompt, "hello");
  assert.equal(request.stream, false);
  assert.deepEqual(request.metadata, { trace: "safe" });

  for (const [field, value] of [
    ["store", true],
    ["previous_response_id", "resp_old"],
    ["conversation", "conv_old"],
    ["tools", [{ type: "function", name: "x" }]],
    ["reasoning", { effort: "low" }],
    ["background", true],
  ]) {
    assert.throws(() => validateResponsesRequest({ model: "tiny", input: "hello", [field]: value }, {
      modelId: "tiny",
    }), { code: "unsupported_parameter" });
  }
  assert.throws(() => validateResponsesRequest({
    model: "tiny", input: "hello", text: { format: { type: "json_schema", schema: {} } },
  }, { modelId: "tiny" }), { param: "text.format", code: "unsupported_parameter" });
  assert.throws(() => validateResponsesRequest({ model: "other", input: "hello" }, { modelId: "tiny" }), {
    statusCode: 404, code: "model_not_found",
  });
});

test("Responses JSON has typed output items and exact usage", () => {
  const request = validateResponsesRequest({ model: "tiny", input: "hello", max_output_tokens: 2 }, { modelId: "tiny" });
  const response = buildResponsesResponse({ ...identity, request, completed: completed() });
  assert.equal(response.id, identity.id);
  assert.equal(response.object, "response");
  assert.equal(response.status, "incomplete");
  assert.deepEqual(response.incomplete_details, { reason: "max_output_tokens" });
  assert.equal(response.store, false);
  assert.equal(response.output[0].id, identity.messageId);
  assert.equal(response.output[0].type, "message");
  assert.equal(response.output[0].content[0].type, "output_text");
  assert.equal(response.output[0].content[0].text, "world hello");
  assert.deepEqual(response.usage, {
    input_tokens: 1,
    input_tokens_details: { cached_tokens: 0 },
    output_tokens: 2,
    output_tokens_details: { reasoning_tokens: 0 },
    total_tokens: 3,
  });

  const eosResponse = buildResponsesResponse({ ...identity, request, completed: completed("eos") });
  assert.equal(eosResponse.status, "completed");
  assert.equal(eosResponse.incomplete_details, null);
  assert.equal(eosResponse.output[0].status, "completed");
  assert.ok(Number.isSafeInteger(eosResponse.completed_at));
});

test("Responses SSE emits the typed lifecycle in sequence and no false completion", async () => {
  const request = validateResponsesRequest({ model: "tiny", input: "hello", stream: true }, { modelId: "tiny" });
  const writes = [];
  await writeResponsesSse({
    events: events(),
    write: async (chunk) => { writes.push(chunk); return true; },
    ...identity,
    request,
  });
  const payloads = writes.map((frame) => JSON.parse(frame.split("\ndata: ")[1]));
  assert.deepEqual(payloads.map((payload) => payload.sequence_number), payloads.map((_, index) => index));
  assert.deepEqual(payloads.map((payload) => payload.type), [
    "response.created",
    "response.in_progress",
    "response.output_item.added",
    "response.content_part.added",
    "response.output_text.delta",
    "response.output_text.delta",
    "response.output_text.done",
    "response.content_part.done",
    "response.output_item.done",
    "response.incomplete",
  ]);
  assert.equal(payloads.filter((payload) => payload.type === "response.incomplete").length, 1);
  assert.equal(payloads.filter((payload) => payload.type === "response.output_text.delta")
    .map((payload) => payload.delta).join(""), "world hello");
  assert.equal(payloads.at(-1).response.usage.output_tokens, 2);

  async function* failed() {
    yield { type: "delta", text: "partial" };
    throw new Error("worker failed");
  }
  const failedWrites = [];
  await assert.rejects(writeResponsesSse({
    events: failed(),
    write: async (chunk) => { failedWrites.push(chunk); return true; },
    ...identity,
    request,
  }), /worker failed/);
  assert.equal(failedWrites.some((frame) => frame.includes("response.completed") || frame.includes("response.incomplete")), false);
});
