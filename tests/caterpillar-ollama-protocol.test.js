import test from "node:test";
import assert from "node:assert/strict";
import {
  buildOllamaResponse,
  validateOllamaChatRequest,
  validateOllamaGenerateRequest,
  writeOllamaNdjson,
} from "../src/runtime/server/protocol/ollama.js";

function completed({ finishReason = "max-tokens" } = {}) {
  return {
    type: "complete",
    output: "world hello",
    metadata: {
      promptArtifact: { tokenIds: [0], tokenCount: 1, tokenCountMode: "exact-vocab" },
      prefillMs: 4,
      decodeMs: 8,
      nativeExecution: {
        finishReason,
        emittedTokenIds: [1, 0],
        generatedTokenIds: [1, 0],
      },
    },
  };
}

async function* events(options = {}) {
  yield { type: "delta", text: "world", tokenId: 1 };
  yield { type: "delta", text: " hello", tokenId: 0 };
  yield completed(options);
}

test("Ollama validators default to streaming and accept only the deterministic subset", () => {
  const generate = validateOllamaGenerateRequest({
    model: "tiny:latest",
    prompt: "hello",
    options: { temperature: 0, top_p: 1, top_k: 1, num_predict: 2, stop: ["x"] },
  }, { modelId: "tiny" });
  assert.equal(generate.stream, true);
  assert.equal(generate.model, "tiny:latest");
  assert.equal(generate.rawPrompt, true);
  assert.equal(generate.maxTokens, 2);
  assert.deepEqual(generate.stop, ["x"]);

  const chat = validateOllamaChatRequest({
    model: "alias",
    messages: [{ role: "user", content: "hello" }],
    stream: false,
  }, { modelId: "tiny", modelAliases: ["alias"] });
  assert.equal(chat.stream, false);
  assert.equal(chat.kind, "chat");
  assert.equal(chat.rawPrompt, false);

  const multiTurn = validateOllamaChatRequest({
    model: "tiny",
    messages: [
      { role: "system", content: "brief" },
      { role: "user", content: "first" },
      { role: "assistant", content: "answer" },
      { role: "user", content: "second" },
    ],
  }, { modelId: "tiny" });
  assert.deepEqual(multiTurn.messages.map((message) => message.role), ["system", "user", "assistant", "user"]);

  const system = validateOllamaGenerateRequest({ model: "tiny", prompt: "hello", system: "brief" }, {
    modelId: "tiny",
  });
  assert.deepEqual(system.messages, [
    { role: "system", content: "brief" },
    { role: "user", content: "hello" },
  ]);

  assert.throws(() => validateOllamaGenerateRequest({ model: "other", prompt: "hello" }, { modelId: "tiny" }), {
    statusCode: 404, code: "model_not_found",
  });
  assert.throws(() => validateOllamaChatRequest({ messages: [{ role: "user", content: "hello" }] }, {
    modelId: "tiny",
  }), { statusCode: 400, code: "model_not_found" });
  assert.throws(() => validateOllamaGenerateRequest({ model: "tiny", prompt: "hello", keep_alive: 0 }, {
    modelId: "tiny",
  }), { param: "keep_alive", code: "unsupported_parameter" });
  for (const unsupported of [
    { format: "json" }, { raw: true }, { template: "{{ .Prompt }}" }, { suffix: "x" }, { images: [] },
    { think: true }, { logprobs: true },
  ]) {
    assert.throws(() => validateOllamaGenerateRequest({ model: "tiny", prompt: "hello", ...unsupported }, {
      modelId: "tiny",
    }), { code: "unsupported_parameter" });
  }
  assert.equal(validateOllamaChatRequest({
    model: "tiny", messages: [{ role: "user", content: "hello" }], tools: [],
  }, { modelId: "tiny" }).kind, "chat");
  assert.throws(() => validateOllamaChatRequest({
    model: "tiny", messages: [{ role: "user", content: "hello" }], tools: [{ type: "function" }],
  }, { modelId: "tiny" }), { code: "unsupported_parameter" });
  assert.throws(() => validateOllamaGenerateRequest({
    model: "tiny", prompt: "hello", options: { temperature: 0.5 },
  }, { modelId: "tiny" }), { param: "options.temperature", code: "unsupported_parameter" });
});

test("Ollama synchronous response carries exact counts, context, reasons, and integer nanoseconds", () => {
  const request = { model: "tiny", kind: "generate", stop: [] };
  const response = buildOllamaResponse({ completed: completed(), request });
  assert.equal(response.model, "tiny");
  assert.equal(response.response, "world hello");
  assert.equal(response.done, true);
  assert.equal(response.done_reason, "length");
  assert.deepEqual(response.context, [0, 1, 0]);
  assert.equal(response.prompt_eval_count, 1);
  assert.equal(response.eval_count, 2);
  assert.equal(response.prompt_eval_duration, 4_000_000);
  assert.equal(response.eval_duration, 8_000_000);
  for (const field of ["total_duration", "load_duration", "prompt_eval_duration", "eval_duration"]) {
    assert.equal(Number.isSafeInteger(response[field]), true);
  }
  assert.equal("message" in response, false);

  const chat = buildOllamaResponse({
    completed: completed({ finishReason: "eos" }),
    request: { model: "tiny", kind: "chat", stop: [] },
  });
  assert.deepEqual(chat.message, { role: "assistant", content: "world hello" });
  assert.equal(chat.done_reason, "stop");
  assert.equal("context" in chat, false);
});

test("Ollama NDJSON emits deltas and one exact terminal object without a false terminal", async () => {
  const writes = [];
  await writeOllamaNdjson({
    events: events(),
    write: async (chunk) => { writes.push(chunk); return true; },
    request: { model: "tiny", kind: "chat", stop: [] },
  });
  const payloads = writes.map((line) => JSON.parse(line));
  assert.equal(payloads.map((payload) => payload.message.content).join(""), "world hello");
  assert.deepEqual(payloads.map((payload) => payload.done), [false, false, true]);
  assert.equal(payloads.at(-1).message.content, "");
  assert.equal(payloads.at(-1).eval_count, 2);

  async function* failed() {
    yield { type: "delta", text: "partial" };
    throw new Error("worker failed");
  }
  const failedWrites = [];
  await assert.rejects(writeOllamaNdjson({
    events: failed(),
    write: async (chunk) => { failedWrites.push(chunk); return true; },
    request: { model: "tiny", kind: "generate", stop: [] },
  }), /worker failed/);
  assert.equal(failedWrites.some((line) => JSON.parse(line).done === true), false);

  const blockedWrites = [];
  await assert.rejects(writeOllamaNdjson({
    events: events(),
    write: async (chunk) => { blockedWrites.push(chunk); return false; },
    request: { model: "tiny", kind: "generate", stop: [] },
  }), { code: "client_cancelled" });
  assert.equal(blockedWrites.length, 1);
  assert.equal(JSON.parse(blockedWrites[0]).done, false);
});
