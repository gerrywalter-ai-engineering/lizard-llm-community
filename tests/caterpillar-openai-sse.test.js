import test from "node:test";
import assert from "node:assert/strict";
import {
  buildOpenAiStreamStart,
  OPENAI_SSE_DONE,
  serializeOpenAiSseData,
  writeOpenAiChatSse,
} from "../src/runtime/server/protocol/openai.js";

const identity = {
  id: "chatcmpl-0123456789abcdef0123456789abcdef",
  created: 1785736800,
  model: "tiny-caterpillar",
};

function metadata({ finishReason = "max-tokens" } = {}) {
  return {
    promptArtifact: { tokenCount: 5, tokenCountMode: "exact-vocab" },
    nativeExecution: {
      generatedTokenIds: [10, 11],
      emittedTokenIds: [10, 11],
      cachedPromptTokens: 1,
      finishReason,
    },
  };
}

async function* successfulEvents({ finishReason = "max-tokens" } = {}) {
  yield { type: "start", sessionId: "session-a", requestId: "request-a" };
  yield { type: "delta", text: "hel", sessionId: "session-a", requestId: "request-a" };
  yield { type: "delta", text: "lo", sessionId: "session-a", requestId: "request-a" };
  yield {
    type: "complete",
    sessionId: "session-a",
    requestId: "request-a",
    metadata: metadata({ finishReason }),
  };
}

function parseDataFrame(frame) {
  assert.match(frame, /^data: /);
  const data = frame.slice(6).trim();
  return data === "[DONE]" ? data : JSON.parse(data);
}

function fragmentedFrames(source, widths) {
  let buffer = "";
  const frames = [];
  let offset = 0;
  let widthIndex = 0;
  while (offset < source.length) {
    const width = widths[widthIndex % widths.length];
    buffer += source.slice(offset, offset + width);
    offset += width;
    widthIndex += 1;
    let boundary = buffer.indexOf("\n\n");
    while (boundary !== -1) {
      frames.push(buffer.slice(0, boundary + 2));
      buffer = buffer.slice(boundary + 2);
      boundary = buffer.indexOf("\n\n");
    }
  }
  assert.equal(buffer, "");
  return frames;
}

test("OpenAI start chunk has a byte-stable assistant role envelope", () => {
  const bytes = serializeOpenAiSseData(buildOpenAiStreamStart(identity));
  assert.equal(bytes,
    "data: {\"id\":\"chatcmpl-0123456789abcdef0123456789abcdef\","
    + "\"object\":\"chat.completion.chunk\",\"created\":1785736800,"
    + "\"model\":\"tiny-caterpillar\",\"choices\":[{\"index\":0,"
    + "\"delta\":{\"role\":\"assistant\",\"content\":\"\"},"
    + "\"finish_reason\":null}]}\n\n");
});

test("OpenAI SSE emits role, ordered text, true finish, requested usage, and DONE", async () => {
  const writes = [];
  const completed = await writeOpenAiChatSse({
    events: successfulEvents(),
    write: async (chunk) => { writes.push(chunk); return true; },
    ...identity,
    includeUsage: true,
    includeRuntime: true,
    completeRuntime: (event) => ({ correlation: { sessionId: event.sessionId, requestId: event.requestId } }),
  });
  assert.equal(completed.type, "complete");
  assert.equal(writes.at(-1), OPENAI_SSE_DONE);
  const payloads = writes.map(parseDataFrame);
  assert.equal(payloads.length, 6);
  assert.deepEqual(payloads[0].choices[0].delta, { role: "assistant", content: "" });
  assert.equal(payloads[1].choices[0].delta.content, "hel");
  assert.equal(payloads[2].choices[0].delta.content, "lo");
  assert.equal(payloads[3].choices[0].finish_reason, "length");
  assert.deepEqual(payloads[3].runtime.correlation, { sessionId: "session-a", requestId: "request-a" });
  assert.deepEqual(payloads[4].choices, []);
  assert.deepEqual(payloads[4].usage, {
    prompt_tokens: 5,
    completion_tokens: 2,
    total_tokens: 7,
    prompt_tokens_details: { cached_tokens: 1 },
  });
  for (const payload of payloads.slice(0, -1)) {
    assert.equal(payload.id, identity.id);
    assert.equal(payload.created, identity.created);
    assert.equal(payload.model, identity.model);
    assert.equal(payload.object, "chat.completion.chunk");
  }
});

test("stream_options.include_usage=false emits no usage chunk and strict mode omits runtime", async () => {
  const writes = [];
  await writeOpenAiChatSse({
    events: successfulEvents({ finishReason: "eos" }),
    write: async (chunk) => { writes.push(chunk); return true; },
    ...identity,
    includeUsage: false,
    includeRuntime: false,
    completeRuntime: () => ({ mustNotAppear: true }),
  });
  const payloads = writes.map(parseDataFrame);
  assert.equal(payloads.length, 5);
  assert.equal(payloads[3].choices[0].finish_reason, "stop");
  assert.equal("runtime" in payloads[3], false);
  assert.equal(payloads.some((payload) => payload !== "[DONE]" && "usage" in payload), false);
});

test("SSE framing survives one-byte fragmentation and multiple events per chunk", async () => {
  const writes = [];
  await writeOpenAiChatSse({
    events: successfulEvents(),
    write: async (chunk) => { writes.push(chunk); return true; },
    ...identity,
    includeUsage: true,
    includeRuntime: false,
  });
  const wire = writes.join("");
  const oneByte = fragmentedFrames(wire, [1]).map(parseDataFrame);
  const mixed = fragmentedFrames(wire, [wire.length - OPENAI_SSE_DONE.length, 1, 3]).map(parseDataFrame);
  assert.equal(oneByte.length, 6);
  assert.deepEqual(mixed, oneByte);
});

test("generation failure or missing exact usage never emits finish or DONE", async () => {
  async function* failed() {
    yield { type: "delta", text: "partial" };
    throw new Error("worker failed");
  }
  const failedWrites = [];
  await assert.rejects(writeOpenAiChatSse({
    events: failed(),
    write: async (chunk) => { failedWrites.push(chunk); return true; },
    ...identity,
  }), /worker failed/);
  assert.equal(failedWrites.some((chunk) => chunk === OPENAI_SSE_DONE), false);
  assert.equal(failedWrites.some((chunk) => chunk.includes("finish_reason\":\"stop")), false);

  async function* missingUsage() {
    yield {
      type: "complete",
      metadata: {
        promptArtifact: { tokenCount: 2, tokenCountMode: "exact-vocab" },
        nativeExecution: { finishReason: "max-tokens" },
      },
    };
  }
  const usageWrites = [];
  await assert.rejects(writeOpenAiChatSse({
    events: missingUsage(),
    write: async (chunk) => { usageWrites.push(chunk); return true; },
    ...identity,
    includeUsage: true,
  }), { code: "completion_usage_unavailable" });
  assert.equal(usageWrites.length, 1, "only the initial role chunk may be committed");
  assert.equal(usageWrites.some((chunk) => chunk === OPENAI_SSE_DONE), false);
});

test("SSE serialization never outruns a slow writer and stops cleanly on backpressure cancellation", async () => {
  let writesInFlight = 0;
  let maximumWritesInFlight = 0;
  const writes = [];
  await writeOpenAiChatSse({
    events: successfulEvents(),
    write: async (chunk) => {
      writesInFlight += 1;
      maximumWritesInFlight = Math.max(maximumWritesInFlight, writesInFlight);
      await new Promise((resolve) => setTimeout(resolve, 2));
      writes.push(chunk);
      writesInFlight -= 1;
      return true;
    },
    ...identity,
    includeUsage: true,
    includeRuntime: false,
  });
  assert.equal(maximumWritesInFlight, 1);
  assert.equal(writes.at(-1), OPENAI_SSE_DONE);

  const cancelledWrites = [];
  await assert.rejects(writeOpenAiChatSse({
    events: successfulEvents(),
    write: async (chunk) => {
      cancelledWrites.push(chunk);
      return cancelledWrites.length < 2;
    },
    ...identity,
  }), { code: "client_cancelled" });
  assert.equal(cancelledWrites.some((chunk) => chunk === OPENAI_SSE_DONE), false);
  assert.equal(cancelledWrites.some((chunk) => chunk.includes('"finish_reason":"length"')), false);
});
