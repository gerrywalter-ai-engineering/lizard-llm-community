import test from "node:test";
import assert from "node:assert/strict";
import {
  buildLlamaCompletion,
  buildOpenAiTextCompletion,
  validateDetokenizeRequest,
  validateLlamaCompletionRequest,
  validateTokenizeRequest,
  writeLlamaCompletionSse,
  writeOpenAiTextCompletionSse,
} from "../src/runtime/server/protocol/llama.js";

function completed({ finishReason = "max-tokens", stopReason = null } = {}) {
  return {
    type: "complete",
    output: "world hello",
    metadata: {
      promptArtifact: {
        renderedPrompt: "hello",
        tokenCount: 1,
        tokenCountMode: "exact-vocab",
      },
      prefillMs: 4,
      decodeMs: 8,
      nativeExecution: {
        finishReason,
        stopReason,
        generatedTokenIds: [1, 0],
        emittedTokenIds: [1, 0],
        cachedPromptTokens: 1,
      },
    },
  };
}

async function* events(options = {}) {
  yield { type: "start" };
  yield { type: "delta", text: "world", tokenId: 1 };
  yield { type: "delta", text: " hello", tokenId: 0 };
  yield completed(options);
}

test("llama completion validators expose only the deterministic declared subset", () => {
  const native = validateLlamaCompletionRequest({
    prompt: [0, 1], n_predict: 2, stream: true, temperature: 0, top_k: 1,
    top_p: 1, cache_prompt: false, return_tokens: true,
  });
  assert.deepEqual(native.promptTokenIds, [0, 1]);
  assert.equal(native.cachePrompt, false);
  assert.equal(native.returnTokens, true);
  const oai = validateLlamaCompletionRequest({
    model: "alias", prompt: "hello", max_tokens: 2, temperature: 0, top_p: 1,
  }, { openAi: true, modelId: "tiny", modelAliases: ["alias"] });
  assert.equal(oai.model, "tiny");
  assert.equal(oai.prompt, "hello");
  assert.throws(() => validateLlamaCompletionRequest({ prompt: ["hello"] }), { code: "invalid_parameter" });
  assert.throws(() => validateLlamaCompletionRequest({ prompt: "hello", temperature: 0.5 }), {
    code: "invalid_parameter", param: "temperature",
  });
  assert.throws(() => validateLlamaCompletionRequest({ prompt: "hello", suffix: "x" }, { openAi: true }), {
    code: "invalid_parameter", param: "suffix",
  });
  assert.throws(() => validateLlamaCompletionRequest({ model: "other", prompt: "hello" }, {
    openAi: true, modelId: "tiny",
  }), { code: "model_not_found", statusCode: 404 });
});

test("token utility validators preserve empty content and reject unknown options", () => {
  assert.deepEqual(validateTokenizeRequest({ content: "" }), {
    content: "", addSpecial: false, parseSpecial: true, withPieces: false,
  });
  assert.deepEqual(validateDetokenizeRequest({ tokens: [] }), { tokens: [] });
  assert.throws(() => validateTokenizeRequest({ content: "hello", special: true }), {
    code: "invalid_parameter", param: "special",
  });
  assert.throws(() => validateDetokenizeRequest({ tokens: [-1] }), {
    code: "invalid_parameter", param: "tokens",
  });
});

test("llama native JSON and SSE freeze terminal shape and exact token facts", async () => {
  const request = {
    prompt: "hello", promptTokenIds: null, maxTokens: 2, stop: [], seed: 42,
    returnTokens: true,
  };
  const json = buildLlamaCompletion({ completed: completed(), model: "tiny", request });
  assert.equal(json.content, "world hello");
  assert.deepEqual(json.tokens, [1, 0]);
  assert.equal(json.stop_type, "limit");
  assert.equal(json.tokens_evaluated, 1);
  assert.equal(json.tokens_cached, 1);

  const writes = [];
  await writeLlamaCompletionSse({
    events: events(),
    write: async (chunk) => { writes.push(chunk); return true; },
    model: "tiny",
    request,
  });
  const payloads = writes.map((frame) => JSON.parse(frame.slice(6).trim()));
  assert.deepEqual(payloads[0].tokens, [1]);
  assert.deepEqual(payloads[1].tokens, [0]);
  assert.equal(payloads.at(-1).stop, true);
  assert.equal(payloads.at(-1).stop_type, "limit");
  assert.equal(writes.some((frame) => frame.includes("[DONE]")), false);
});

test("OpenAI legacy completion JSON and SSE carry stable identity, usage, finish, and DONE", async () => {
  const request = { prompt: "hello", maxTokens: 2, stop: [] };
  const identity = { id: "cmpl-abc", created: 1785744094, model: "tiny" };
  const json = buildOpenAiTextCompletion({ ...identity, completed: completed(), request });
  assert.equal(json.object, "text_completion");
  assert.equal(json.choices[0].text, "world hello");
  assert.equal(json.choices[0].finish_reason, "length");
  assert.deepEqual(json.usage, {
    prompt_tokens: 1,
    completion_tokens: 2,
    total_tokens: 3,
    prompt_tokens_details: { cached_tokens: 1 },
  });

  const writes = [];
  await writeOpenAiTextCompletionSse({
    events: events(),
    write: async (chunk) => { writes.push(chunk); return true; },
    ...identity,
    request,
  });
  assert.equal(writes.at(-1), "data: [DONE]\n\n");
  const payloads = writes.slice(0, -1).map((frame) => JSON.parse(frame.slice(6).trim()));
  assert.equal(payloads.map((payload) => payload.choices[0].text).join(""), "world hello");
  assert.equal(payloads.at(-1).choices[0].finish_reason, "length");
  assert.equal(payloads.at(-1).usage.completion_tokens, 2);
  for (const payload of payloads) {
    assert.equal(payload.id, identity.id);
    assert.equal(payload.created, identity.created);
    assert.equal(payload.model, identity.model);
  }
});

test("completion SSE failures never forge terminal success", async () => {
  async function* failed() {
    yield { type: "delta", text: "partial", tokenId: 1 };
    throw new Error("worker failed");
  }
  for (const writer of [writeLlamaCompletionSse, writeOpenAiTextCompletionSse]) {
    const writes = [];
    await assert.rejects(writer({
      events: failed(),
      write: async (chunk) => { writes.push(chunk); return true; },
      id: "cmpl-a",
      created: 1,
      model: "tiny",
      request: { prompt: "hello", maxTokens: 2, stop: [] },
    }), /worker failed/);
    assert.equal(writes.some((chunk) => chunk.includes("[DONE]")), false);
    assert.equal(writes.some((chunk) => chunk.includes('"stop":true')), false);
    assert.equal(writes.some((chunk) => chunk.includes('"finish_reason":"length"')), false);
  }
});
