import test from "node:test";
import assert from "node:assert/strict";
import { renderChatTemplate } from "../src/runtime/tokenizer/index.js";
import { normalizeChatMessages } from "../src/runtime/server/protocol/messages.js";
import { validateChatRequest } from "../src/runtime/server/protocol/validation.js";
import { resolveCachedPromptTokens } from "../src/runtime/core/native-cpu-runtime.js";
import {
  buildOpenAiChatCompletion,
  buildOpenAiUsage,
  assertConformantChatTemplate,
  mapOpenAiFinishReason,
} from "../src/runtime/server/protocol/openai.js";

test("ordered chat messages support strings and ordered text parts", () => {
  const messages = normalizeChatMessages([
    { role: "system", content: "Be concise." },
    { role: "user", content: [{ type: "text", text: "one " }, { type: "text", text: "two" }] },
    { role: "assistant", content: "three" },
    { role: "user", content: "four" },
  ]);
  assert.deepEqual(messages.map((message) => message.role), ["system", "user", "assistant", "user"]);
  assert.equal(messages[1].content, "one two");
  assert.throws(() => normalizeChatMessages([{ role: "tool", content: "x" }]), { code: "unsupported_message_role" });
  assert.throws(() => normalizeChatMessages([{ role: "user", content: [{ type: "image_url", image_url: "x" }] }]), {
    code: "unsupported_content_type",
  });
  assert.throws(() => normalizeChatMessages([{ role: "assistant", content: "not a request" }]), { code: "invalid_parameter" });
});

test("Llama 3 renderer preserves every historical role and appends one generation prompt", () => {
  const rendered = renderChatTemplate({
    template: "{% for message in messages %}<|start_header_id|>{% endfor %}",
    messages: [
      { role: "system", content: "rules" },
      { role: "user", content: "first" },
      { role: "assistant", content: "answer" },
      { role: "user", content: "second" },
    ],
    specialTokens: { bos: "<s>" },
    chatMode: true,
  });
  const text = rendered.renderedPrompt;
  assert.ok(text.indexOf("system") < text.indexOf("first"));
  assert.ok(text.indexOf("first") < text.indexOf("answer"));
  assert.ok(text.indexOf("answer") < text.indexOf("second"));
  assert.match(text, /second<\|eot_id\|><\|start_header_id\|>assistant/);
  assert.equal(rendered.templateSource, "llama3_chat");
});

test("named no-template fallback is explicit and unknown structured templates fail closed", () => {
  const fallback = renderChatTemplate({
    messages: [
      { role: "system", content: "rules" },
      { role: "user", content: "first" },
      { role: "assistant", content: "answer" },
      { role: "user", content: "second" },
    ],
    chatMode: true,
  });
  assert.equal(fallback.templateSource, "instruction_fallback");
  assert.match(fallback.renderedPrompt, /### Instruction:/);
  assert.match(fallback.renderedPrompt, /### Response:\s*$/);
  assert.doesNotThrow(() => assertConformantChatTemplate({ templateRenderer: fallback.templateSource }));

  const unknown = renderChatTemplate({
    template: "{% for unexpected in turns %}{{ unexpected }}{% endfor %}",
    messages: [{ role: "user", content: "hello" }],
    chatMode: true,
  });
  assert.equal(unknown.templateSource, "unsupported_jinja");
  assert.throws(() => assertConformantChatTemplate({ templateRenderer: unknown.templateSource }), {
    code: "chat_template_unsupported",
    statusCode: 422,
  });
});

test("chat field table accepts defaults and rejects unsupported non-defaults", () => {
  const accepted = validateChatRequest({
    model: "tiny",
    messages: [{ role: "user", content: "hello" }],
    stream: true,
    stream_options: { include_usage: true },
    max_completion_tokens: 8,
    temperature: 0,
    top_p: 1,
    top_k: 1,
    seed: 42,
    stop: ["END"],
    n: 1,
    frequency_penalty: 0,
    presence_penalty: 0,
    logprobs: false,
    response_format: { type: "text" },
    tools: [],
    tool_choice: "none",
    parallel_tool_calls: false,
    user: "sdk-user",
  }, { modelId: "tiny", runtimeProvider: "caterpillar" });
  assert.equal(accepted.max_tokens, 8);
  assert.deepEqual(accepted.stop, ["END"]);
  assert.throws(() => validateChatRequest({
    messages: [{ role: "user", content: "hello" }],
    max_tokens: 4,
    max_completion_tokens: 5,
  }), { param: "max_completion_tokens" });
  assert.throws(() => validateChatRequest({
    messages: [{ role: "user", content: "hello" }],
    response_format: { type: "json_object" },
  }), { code: "unsupported_parameter" });
  assert.throws(() => validateChatRequest({
    messages: [{ role: "user", content: "hello" }],
    tools: [{ type: "function" }],
  }), { code: "unsupported_parameter" });
});

test("every non-default or malformed chat field fails before execution", () => {
  const messages = [{ role: "user", content: "hello" }];
  const rejected = [
    ["model", { model: 42 }, "invalid_parameter"],
    ["stream", { stream: "yes" }, "invalid_parameter"],
    ["max_tokens", { max_tokens: 0 }, "invalid_parameter"],
    ["max_completion_tokens", { max_completion_tokens: 4097 }, "invalid_parameter"],
    ["max_completion_tokens", { max_tokens: 4, max_completion_tokens: 5 }, "invalid_parameter"],
    ["temperature", { temperature: 0.1 }, "unsupported_parameter"],
    ["top_p", { top_p: 0.9 }, "unsupported_parameter"],
    ["top_k", { top_k: 2 }, "unsupported_parameter"],
    ["seed", { seed: 1.5 }, "invalid_parameter"],
    ["stop", { stop: ["ok", 3] }, "invalid_parameter"],
    ["n", { n: 2 }, "unsupported_parameter"],
    ["frequency_penalty", { frequency_penalty: 0.5 }, "unsupported_parameter"],
    ["presence_penalty", { presence_penalty: 1 }, "unsupported_parameter"],
    ["logprobs", { logprobs: true }, "unsupported_parameter"],
    ["response_format", { response_format: { type: "json_object" } }, "unsupported_parameter"],
    ["tools", { tools: [{ type: "function", function: { name: "x" } }] }, "unsupported_parameter"],
    ["tool_choice", { tool_choice: "auto" }, "unsupported_parameter"],
    ["parallel_tool_calls", { parallel_tool_calls: true }, "unsupported_parameter"],
    ["stream_options", { stream_options: { include_usage: true, extra: true } }, "invalid_parameter"],
    ["user", { user: "x".repeat(257) }, "invalid_parameter"],
    ["unknown_field", { unknown_field: true }, "unsupported_parameter"],
  ];
  for (const [param, fields, code] of rejected) {
    assert.throws(
      () => validateChatRequest({ messages, ...fields }, { runtimeProvider: "caterpillar" }),
      (error) => error.param === param && error.code === code,
      param,
    );
  }
});

test("model identity accepts omission, canonical ID, and configured aliases only", () => {
  const options = {
    modelId: "canonical-model",
    modelAliases: ["friendly-model", "registry-alias"],
    runtimeProvider: "caterpillar",
  };
  const request = { messages: [{ role: "user", content: "hello" }] };
  assert.doesNotThrow(() => validateChatRequest(request, options));
  assert.equal(validateChatRequest({ ...request, model: "canonical-model" }, options).model, "canonical-model");
  assert.equal(validateChatRequest({ ...request, model: "friendly-model" }, options).model, "friendly-model");
  assert.throws(() => validateChatRequest({ ...request, model: "other-model" }, options), {
    code: "model_not_found",
    statusCode: 404,
    param: "model",
  });
});

test("OpenAI JSON derives exact usage and finish reason from native facts", () => {
  const metadata = {
    promptArtifact: { tokenCount: 7, tokenCountMode: "exact-vocab" },
    nativeExecution: {
      generatedTokenIds: [10, 11, 12],
      emittedTokenIds: [10, 11, 12],
      cachedPromptTokens: 2,
      finishReason: "max-tokens",
    },
  };
  assert.deepEqual(buildOpenAiUsage(metadata), {
    prompt_tokens: 7,
    completion_tokens: 3,
    total_tokens: 10,
    prompt_tokens_details: { cached_tokens: 2 },
  });
  assert.equal(mapOpenAiFinishReason("eos"), "stop");
  assert.equal(mapOpenAiFinishReason("stop-sequence"), "stop");
  assert.equal(mapOpenAiFinishReason("max-tokens"), "length");
  assert.throws(() => mapOpenAiFinishReason("cancelled"), {
    code: "terminal_reason_unavailable",
    statusCode: 500,
  });

  const response = buildOpenAiChatCompletion({
    id: "chatcmpl-test",
    created: 1780000000,
    model: "tiny",
    completed: { output: "answer", metadata },
    includeRuntime: false,
  });
  assert.equal(response.choices[0].finish_reason, "length");
  assert.equal(response.created, 1780000000);
  assert.equal("runtime" in response, false);
});

test("cached prompt usage distinguishes a first snapshot build from a true cache hit", () => {
  assert.equal(resolveCachedPromptTokens({
    prefillSnapshotHit: false,
    kvPrefixReusedTokens: 37,
  }), 0);
  assert.equal(resolveCachedPromptTokens({
    prefillSnapshotHit: true,
    kvPrefixReusedTokens: 37,
  }), 37);
  assert.equal(resolveCachedPromptTokens({
    prefillSnapshotHit: false,
    prefillSnapshotPrefixHit: true,
    prefillSnapshotPrefixTokens: 19,
    kvPrefixReusedTokens: 37,
  }), 19);
  assert.equal(resolveCachedPromptTokens({
    promptCacheReused: true,
    promptCacheReusedTokens: 23,
  }), 23);
});
