import assert from "node:assert/strict";
import fs from "node:fs";
import test from "node:test";

import {
  estimateTokens,
  kvBytesPerToken,
  planTurnBudget,
} from "../src/lib/chat/context-budget.js";
import {
  CHAT_ENGINES,
  DEFAULT_CHAT_ENGINE,
  normalizeChatEngine,
  getChatEngine,
  ollamaTagForModel,
  buildTurnProfile,
} from "../src/lib/chat/engine-runtime.js";
import { buildKvCachePlan, resolveKvBudgetBytes } from "../src/runtime/kv-cache/index.js";
import { composePrompt } from "../src/lib/chat/tool-protocol.js";
import {
  createStopSequenceFilter,
  truncateAtStopSequence,
  NATIVE_STOP_STRINGS,
  INSTRUCTION_STOP_STRINGS,
} from "../src/runtime/core/stop-sequences.js";
import { renderChatTemplate } from "../src/runtime/tokenizer/index.js";

// The model from the reported failure: llama-3.2-3b-instruct.
const GGUF_3B = { blockCount: 28, embeddingLength: 3072, contextLength: 131072, tokenizer: "llama" };
const BYTES_PER_TOKEN_3B = 28 * 3072 * 2 * 2; // 344,064

// ---- the reported failure -------------------------------------------------

test("kvBytesPerToken matches what the runtime actually reserves per token", () => {
  assert.equal(kvBytesPerToken(GGUF_3B), BYTES_PER_TOKEN_3B);
  assert.equal(kvBytesPerToken(null), null);
  assert.equal(kvBytesPerToken({ blockCount: 0, embeddingLength: 3072 }), null);
});

test("a short chat no longer reserves the whole context window", () => {
  // Before: contextSize came from the profile (10,240 tokens here), so every
  // turn planned 3.3 GB of KV cache regardless of how much was actually said.
  const plan = planTurnBudget({
    systemPrompt: "You are Lizard.",
    question: "hi how are you",
    history: [],
    answerTokens: 512,
    modelContextTokens: 10240,
    bytesPerToken: BYTES_PER_TOKEN_3B,
    budgetBytes: 2_877_628_088,
  });

  assert.equal(plan.fits, true);
  assert.ok(
    plan.contextSize < 2048,
    `a two-line conversation should not ask for a big context, got ${plan.contextSize}`,
  );
  assert.ok(plan.contextSize * BYTES_PER_TOKEN_3B < 2_877_628_088, "and it must fit the budget");
  assert.deepEqual(plan.notes, [], "nothing had to be trimmed, so nothing is claimed");
});

test("a conversation's context size grows but never shrinks", () => {
  // A shrinking context is a different KV shape, and the native prefill
  // snapshot that lets turn N+1 pay only for its new tail cannot be reused
  // across one. Measured on an ordinary conversation before this: turn 1
  // prefilled in 4 ms, turn 2 (context 2048 -> 1024 -> 2048) took 19.3 s, which
  // reads as an answer arriving in one block after twenty seconds of silence.
  const shared = {
    systemPrompt: "You are Lizard.",
    answerTokens: 512,
    modelContextTokens: 10240,
    bytesPerToken: BYTES_PER_TOKEN_3B,
    budgetBytes: 2_877_628_088,
  };
  const long = planTurnBudget({
    ...shared,
    question: "Summarise this: " + "the quick brown fox jumps over the lazy dog. ".repeat(120),
  });
  const shortFollowUp = planTurnBudget({ ...shared, question: "thanks" });
  assert.ok(
    shortFollowUp.contextSize < long.contextSize,
    "without a floor a short follow-up plans a smaller context, which is the bug",
  );

  const held = planTurnBudget({
    ...shared,
    question: "thanks",
    minContextSize: long.contextSize,
  });
  assert.equal(held.contextSize, long.contextSize);

  // Growing is still allowed - a longer turn needs the room.
  const longer = planTurnBudget({
    ...shared,
    question: "Summarise this: " + "the quick brown fox jumps over the lazy dog. ".repeat(240),
    minContextSize: long.contextSize,
  });
  assert.ok(longer.contextSize > long.contextSize);

  // And the machine's ceiling still wins over the floor.
  const capped = planTurnBudget({
    ...shared,
    question: "thanks",
    minContextSize: 1_000_000,
  });
  assert.equal(capped.contextSize, capped.ceiling);
  assert.ok(capped.contextSize * BYTES_PER_TOKEN_3B <= 2_877_628_088);
});

test("the exact failing turn now fits instead of throwing", () => {
  // The trigger: an image read as text produced ~60k characters of junk, which
  // pushed promptTokens past the context so the plan reserved the full 10,240.
  const junk = "�".repeat(60_000);
  const plan = planTurnBudget({
    systemPrompt: "You are Lizard.",
    question: "analyze this image",
    attachments: [{ name: "1_xUg.png", text: junk }],
    answerTokens: 512,
    modelContextTokens: 10240,
    bytesPerToken: BYTES_PER_TOKEN_3B,
    budgetBytes: 2_877_628_088,
  });

  assert.equal(plan.fits, true, "the turn degrades instead of failing");
  const reserved = plan.contextSize * BYTES_PER_TOKEN_3B;
  assert.ok(reserved <= 2_877_628_088, `reserved ${reserved} must fit the 2.68 GB budget`);
  assert.ok(plan.notes.length > 0, "and it says what it had to do");
  assert.match(plan.notes.join(" "), /truncated|left out/);
});

test("the KV plan the runtime checks passes for a planned turn", () => {
  const profile = {
    modelId: "llama-3.2-3b-instruct",
    contextSize: 10240,
    capabilityRegistry: { memory: { availableGb: 13.4 } },
    gguf: GGUF_3B,
  };
  // What used to happen: full context, prompt longer than it.
  const before = buildKvCachePlan({
    modelId: "llama-3.2-3b-instruct",
    profile,
    promptArtifact: { tokenizer: "llama", renderedPromptHash: "a", tokenCount: 40000 },
  });
  assert.equal(before.effectiveTokens, 10240, "clamped to the context window, as reported");

  // What happens now: the turn asks for the context it planned.
  const plan = planTurnBudget({
    question: "analyze this",
    answerTokens: 512,
    modelContextTokens: 10240,
    bytesPerToken: BYTES_PER_TOKEN_3B,
    budgetBytes: resolveKvBudgetBytes(profile),
  });
  const after = buildKvCachePlan({
    modelId: "llama-3.2-3b-instruct",
    profile: { ...profile, contextSize: plan.contextSize },
    promptArtifact: { tokenizer: "llama", renderedPromptHash: "a", tokenCount: 40000 },
  });
  assert.equal(after.fits, true, "the planned turn fits the budget the runtime enforces");
});

// ---- budget calibration ---------------------------------------------------

test("the KV budget uses the memory the machine actually has free", () => {
  const profile = { capabilityRegistry: { memory: { availableGb: 13.4 } } };
  const budget = resolveKvBudgetBytes(profile);
  const availableBytes = 13.4 * 1024 ** 3;

  assert.ok(budget > 2_877_628_088, "more generous than the old fixed 20% share");
  assert.ok(budget <= availableBytes, "but never more than is free");
});

test("the KV budget stays overridable for callers that must pin a footprint", () => {
  assert.equal(
    resolveKvBudgetBytes({ nativeRuntime: { kvCacheBudgetBytes: 123_456 } }),
    123_456,
  );
  const shared = resolveKvBudgetBytes({
    capabilityRegistry: { memory: { availableGb: 10 } },
    nativeRuntime: { kvCacheBudgetShare: 0.2 },
  });
  assert.equal(shared, Math.trunc(10 * 1024 ** 3 * 0.2));
});

test("a machine with almost nothing free is never promised memory it lacks", () => {
  const tiny = resolveKvBudgetBytes({ capabilityRegistry: { memory: { availableGb: 0.25 } } });
  assert.ok(tiny <= 0.25 * 1024 ** 3, "the 512 MB floor cannot exceed what is free");
});

test("buildKvCachePlan reports how long a context this machine can hold", () => {
  const plan = buildKvCachePlan({
    modelId: "demo",
    profile: {
      contextSize: 4096,
      capabilityRegistry: { memory: { availableGb: 8 } },
      gguf: GGUF_3B,
    },
    promptArtifact: { tokenizer: "llama", renderedPromptHash: "a", tokenCount: 100 },
  });
  assert.ok(plan.maxTokensForBudget > 0);
  assert.equal(plan.maxTokensForBudget, Math.floor(plan.totalBudgetBytes / plan.bytesPerToken));
});

// ---- trimming order -------------------------------------------------------

test("trimming drops history before documents, and documents before the question", () => {
  const long = (label) => `${label} `.repeat(400);
  const plan = planTurnBudget({
    systemPrompt: "S",
    question: "what did we decide?",
    history: [
      { role: "user", content: long("oldest") },
      { role: "assistant", content: long("older") },
      { role: "user", content: long("recent") },
    ],
    contextChunks: [
      { rank: 1, text: long("best"), documentName: "a.md" },
      { rank: 2, text: long("worse"), documentName: "b.md" },
    ],
    answerTokens: 256,
    modelContextTokens: 2048,
    bytesPerToken: BYTES_PER_TOKEN_3B,
    budgetBytes: 2048 * BYTES_PER_TOKEN_3B,
  });

  assert.equal(plan.fits, true);
  assert.ok(plan.history.length < 3, "history is what gives way first");
  if (plan.history.length > 0) {
    assert.match(plan.history[plan.history.length - 1].content, /recent/, "the newest turn survives");
  }
  if (plan.contextChunks.length === 1) {
    assert.equal(plan.contextChunks[0].rank, 1, "the best-ranked passage is the one kept");
  }
});

test("a question too large for the machine is refused with a way forward", () => {
  const plan = planTurnBudget({
    question: "x".repeat(400_000),
    answerTokens: 512,
    modelContextTokens: 4096,
    bytesPerToken: BYTES_PER_TOKEN_3B,
    budgetBytes: 512 * 1024 * 1024,
  });
  assert.equal(plan.fits, false);
  assert.match(plan.notes[0], /Shorten it|smaller model/);
});

test("without GGUF geometry the planner changes nothing rather than guessing", () => {
  const plan = planTurnBudget({
    question: "hello",
    answerTokens: 512,
    modelContextTokens: 4096,
    bytesPerToken: null,
    budgetBytes: null,
  });
  assert.equal(plan.fits, true);
  assert.equal(plan.ceiling, 4096, "falls back to the model's own context");
});

test("token estimates are conservative, because under-counting is what overruns", () => {
  assert.equal(estimateTokens(""), 0);
  const text = "a".repeat(340);
  assert.ok(estimateTokens(text) >= 100, "340 chars must not be counted as fewer than 100 tokens");
});

// ---- engines --------------------------------------------------------------

test("all five engines are offered, and each says whether it streams", () => {
  const ids = CHAT_ENGINES.map((engine) => engine.id);
  assert.deepEqual(ids, ["lizard-native", "caterpillar", "ostrich", "llama.cpp", "ollama"]);
  for (const engine of CHAT_ENGINES) {
    assert.ok(engine.label && engine.summary, `${engine.id} explains itself`);
    assert.equal(typeof engine.streams, "boolean");
  }
  // Ostrich is admin-gated. The flag has to live on the engine record, because
  // this list is what the chat picker renders from.
  assert.equal(getChatEngine("ostrich").requiresExperimentalAccess, true);
  // Every Lizard-owned engine streams. Their runners emit generate.token
  // notifications carrying the sessionId the client registered its listener
  // under; without that id the tokens land on the no-session fallback and are
  // dropped, which is how Caterpillar and Ostrich looked non-streaming.
  assert.equal(getChatEngine("ostrich").streams, true);
  assert.equal(getChatEngine("caterpillar").streams, true);
  assert.equal(getChatEngine("lizard-native").streams, true);
  // llama.cpp is spawned per request and returns the answer whole; claiming it
  // streams would make the UI look hung waiting for tokens that never arrive.
  assert.equal(getChatEngine("llama.cpp").streams, false);
  assert.equal(getChatEngine("ollama").streams, true);
});

test("an unknown engine falls back to the default instead of failing a turn", () => {
  assert.equal(normalizeChatEngine("nonsense"), DEFAULT_CHAT_ENGINE);
  assert.equal(normalizeChatEngine("LIZARD-NATIVE"), "lizard-native");
  assert.equal(normalizeChatEngine(undefined), DEFAULT_CHAT_ENGINE);
});

test("chat applies the fastest measured HTTP B16/B8 lane", () => {
  const gguf = { blockCount: 1, embeddingLength: 64, contextLength: 4096 };
  const profile = buildTurnProfile({
    context: {
      profile: {
        contextSize: 2048,
        capabilityRegistry: { memory: { availableGb: 12 } },
        gguf,
      },
    },
    resolution: { engine: "lizard-native" },
    gguf,
    stopSequences: [],
    measuredHttpPerformance: {
      levels: {
        16: { failed: 0, aggregateTokensPerSecond: 30 },
        8: { failed: 0, aggregateTokensPerSecond: 34 },
      },
    },
  }, { contextSize: 2048 });

  assert.equal(profile.nativeRuntime.batchTargetSize, 8);
  assert.equal(profile.nativeRuntime.concurrencyPolicy.source, "measured-http");
});

// ---- why answers ran away -------------------------------------------------

test("the question is the prompt; everything else is the system turn", () => {
  const composed = composePrompt({
    systemPrompt: "You are Lizard.",
    toolInstructions: "TOOLS",
    contextBlock: "CONTEXT",
    history: [{ role: "user", content: "earlier" }, { role: "assistant", content: "reply" }],
    userMessage: "what did we decide?",
    attachments: [{ name: "a.txt", text: "FILE BODY" }],
  });

  assert.equal(composed.prompt, "what did we decide?", "the prompt is the question, nothing else");
  for (const marker of ["You are Lizard.", "TOOLS", "CONTEXT", "FILE BODY", "earlier"]) {
    assert.ok(composed.systemPrompt.includes(marker), `${marker} belongs in the system turn`);
  }
});

test("no chat-log pattern is ever handed to the model", () => {
  // This is what produced "Assistant: hi / Assistant: hi" for 500 tokens: the
  // prompt was a transcript, so the model continued the transcript.
  const composed = composePrompt({
    systemPrompt: "You are Lizard.",
    history: [{ role: "user", content: "hi" }, { role: "assistant", content: "hello" }],
    userMessage: "hi how are you",
  });
  const everything = `${composed.systemPrompt}\n${composed.prompt}`;
  assert.ok(!/\bUser:\s/.test(everything), "no 'User:' role prefix to imitate");
  assert.ok(!/\bAssistant:\s/.test(everything), "no 'Assistant:' role prefix to imitate");
  assert.ok(!/\n\s*Assistant:\s*$/.test(composed.prompt), "and no trailing 'Assistant:' cue");
});

test("Llama 3 and ChatML get real turn markers instead of bare text", () => {
  // Both ship Jinja templates this renderer cannot execute. It used to degrade
  // to "System: ...\n\nquestion" - no turn structure, so an instruct model
  // continued the document and never emitted an end-of-turn token.
  const llama = renderChatTemplate({
    template: "{%- set x = 1 %}<|start_header_id|>user<|end_header_id|>",
    prompt: "hello",
    systemPrompt: "be brief",
    specialTokens: { bos: "<|begin_of_text|>" },
  });
  assert.equal(llama.templateMode, "rendered");
  assert.equal(llama.templateSource, "llama3_chat");
  assert.match(llama.renderedPrompt, /<\|start_header_id\|>system<\|end_header_id\|>/);
  assert.match(llama.renderedPrompt, /<\|start_header_id\|>user<\|end_header_id\|>\n\nhello<\|eot_id\|>/);
  assert.ok(llama.renderedPrompt.endsWith("<|start_header_id|>assistant<|end_header_id|>\n\n"));

  const chatml = renderChatTemplate({
    template: "{%- for m in messages %}<|im_start|>{{ m.role }}",
    prompt: "hello",
    systemPrompt: "be brief",
    specialTokens: {},
  });
  assert.equal(chatml.templateSource, "chatml");
  assert.match(chatml.renderedPrompt, /<\|im_start\|>user\nhello<\|im_end\|>/);
  assert.ok(chatml.renderedPrompt.endsWith("<|im_start|>assistant\n"));
});

test("generation stops at the end of the turn, whatever the family calls it", () => {
  for (const marker of ["<|eot_id|>", "<|im_end|>", "<end_of_turn>", "</s>"]) {
    const result = truncateAtStopSequence(`The answer is 4.${marker}And then it rambled on.`);
    assert.equal(result.stopped, true, `${marker} must end the turn`);
    assert.equal(result.text, "The answer is 4.");
  }
  const clean = truncateAtStopSequence("A complete answer.");
  assert.equal(clean.stopped, false);
  assert.equal(clean.text, "A complete answer.");
});

test("a model that starts writing the user's side is cut off there", () => {
  // The reported runaway: no special token at all, just an invented dialogue.
  const result = truncateAtStopSequence("hi\nAssistant: hw are you\nAssistant: hi");
  assert.equal(result.stopped, true);
  assert.equal(result.reason, "role-echo");
  assert.equal(result.text, "hi");
});

test("a stop marker split across tokens is still caught", () => {
  const filter = createStopSequenceFilter();
  let out = "";
  // A tokenizer emits "<|eot" and "_id|>" as separate tokens.
  for (const token of ["The ", "answer ", "is ", "4.", "<|eot", "_id|>", " and more"]) {
    out += filter.push(token);
  }
  out += filter.finish();
  assert.equal(filter.stopped, true);
  assert.equal(out, "The answer is 4.");
});

test("a greedy repetition loop is cut instead of filling the token budget", () => {
  // Exactly the "it writes the same thing 20 times" failure: no stop marker
  // appears, the model just repeats itself until max_tokens.
  const filter = createStopSequenceFilter();
  let out = "";
  for (let i = 0; i < 40 && !filter.stopped; i += 1) {
    out += filter.push("The stack principle is a data structure. ");
  }
  out += filter.finish();
  assert.equal(filter.stopped, true, "the loop must end the turn");
  assert.equal(filter.reason, "repetition-loop");
  assert.ok(out.length < 40 * 41, "and it must not have emitted every repeat");
});

test("ordinary prose with a repeated phrase is not mistaken for a loop", () => {
  const filter = createStopSequenceFilter();
  let out = "";
  const sentences = [
    "A stack is last in, first out. ",
    "You push a value onto the top of it. ",
    "You pop the most recent value back off. ",
    "Function calls use one to track return addresses. ",
    "Overflowing it is what a stack overflow means. ",
  ];
  for (const sentence of sentences) {
    out += filter.push(sentence);
  }
  out += filter.finish();
  assert.equal(filter.stopped, false, "distinct sentences are not a loop");
  assert.ok(out.includes("stack overflow"), "the whole answer survives");
});

test("a clean answer is released in full, including its tail", () => {
  const filter = createStopSequenceFilter();
  let out = "";
  for (const token of ["Hello", " there", ", friend."]) {
    out += filter.push(token);
  }
  out += filter.finish();
  assert.equal(filter.stopped, false);
  assert.equal(out, "Hello there, friend.", "nothing is held back once generation ends");
});

test("model facts are cached without a clock, so a slow turn cannot expire them", () => {
  // The pre-generation work (hardware scan, model artifact, capability
  // registry, chat-template read) measured 10.3s and ran on every turn. It was
  // cached with a TTL - which was worse than useless on a slow engine: a
  // caterpillar turn takes 50-90s, so a 120s window expired mid-conversation
  // and the NEXT question paid an 18s re-resolve before anything happened.
  // That is what "the follow-up gets stuck" was. These caches must be
  // invalidated by activation, never by elapsed time.
  const source = fs.readFileSync("src/lib/chat/engine-runtime.js", "utf8");
  const turnCache = /const cached = turnRuntimeCache\.get\(cacheKey\);\s*\n\s*if \(cached\) \{/;
  assert.match(source, turnCache, "the turn-runtime cache must not be time-expired");
  const readiness = /const cached = readinessCache\.get\(modelId\);\s*\n\s*if \(cached\) \{/;
  assert.match(source, readiness, "model readiness must not be time-expired");
  // Engine AVAILABILITY is genuinely external state (is Ollama running?) and
  // does keep its TTL - the distinction is deliberate.
  assert.match(source, /enginesCache\.at < PROBE_CACHE_TTL_MS/);
  assert.match(source, /invalidateEngineProbeCache/);
});

test("caterpillar gets an append-only transcript so its KV cache can be reused", () => {
  // Caterpillar re-runs the whole prompt through its CPU engine on every
  // prefill. Inlining prior turns makes turn N a strict extension of turn N-1,
  // which is what lets its runner keep the previous KV. With history in the
  // system block the prompt changes in the middle and nothing carries over.
  const history = [
    { role: "user", content: "Q1" },
    { role: "assistant", content: "A1" },
  ];
  const turn1 = composePrompt({ systemPrompt: "S", history: [], userMessage: "Q1", inlineHistory: true });
  const turn2 = composePrompt({ systemPrompt: "S", history, userMessage: "Q2", inlineHistory: true });

  assert.equal(turn1.systemPrompt, turn2.systemPrompt, "the system turn must not drift between turns");
  assert.equal(turn1.prompt, "Q1");
  // Turn 2 restates turn 1 verbatim, then appends - the separators match
  // renderInstructionPrompt so the rendered prompts nest exactly.
  assert.ok(turn2.prompt.startsWith("Q1\n\n### Response:\nA1\n\n### Instruction:\n\n"));
  assert.ok(turn2.prompt.endsWith("Q2"));

  // Default (non-caterpillar) still summarises history in the system turn.
  const plain = composePrompt({ systemPrompt: "S", history, userMessage: "Q2" });
  assert.equal(plain.prompt, "Q2");
  assert.match(plain.systemPrompt, /Earlier in this conversation/);
});

test("native stop strings are plain substrings the worker can match, never regexes", () => {
  // The worker does a substring find(), so every entry must be a literal that
  // cannot appear in ordinary prose.
  for (const marker of NATIVE_STOP_STRINGS) {
    assert.equal(typeof marker, "string");
    assert.ok(marker.length > 0);
    assert.ok(/[<>|#]/.test(marker), `${marker} should be a special marker, not prose`);
  }
  assert.ok(NATIVE_STOP_STRINGS.includes("<|eot_id|>"), "Llama 3 turn-ender");
  assert.ok(NATIVE_STOP_STRINGS.includes("<|im_end|>"), "ChatML turn-ender");
  // Instruction-format stops are kept separate because "### " and "User:" can
  // legitimately appear in an instruct model's answer - they apply only to
  // template-less base models, which continue a chat by writing both sides.
  assert.ok(INSTRUCTION_STOP_STRINGS.some((marker) => marker.includes("###")));
  assert.ok(INSTRUCTION_STOP_STRINGS.includes("\nAssistant:"), "base-model role echo");
  assert.ok(!NATIVE_STOP_STRINGS.some((marker) => marker.includes("###")),
    "### must not be an always-on stop, or it would truncate Markdown answers");
  assert.ok(!NATIVE_STOP_STRINGS.some((marker) => /assistant/i.test(marker)),
    "a role echo must not be an always-on stop, or it truncates quoted dialogue");
});

test("the buffered filter tolerates a huge stop marker without unbounded hold-back", () => {
  const filter = createStopSequenceFilter();
  let out = "";
  // A long clean answer streams through without the filter hoarding it.
  for (let i = 0; i < 50; i += 1) {
    out += filter.push(`word${i} `);
  }
  out += filter.finish();
  assert.ok(out.startsWith("word0 word1"), "text flows as it arrives");
  assert.ok(out.includes("word49"), "nothing is lost at the end");
});

test("an Ollama tag is derived when the catalog does not carry one", () => {
  const config = { runtimeDirectory: "/tmp/does-not-exist", catalogPath: "/tmp/none.json" };
  assert.equal(ollamaTagForModel(config, "llama-3.2-3b-instruct"), "llama-3.2-3b-instruct:latest");
  assert.equal(ollamaTagForModel(config, "qwen2.5:7b"), "qwen2.5:7b");
  assert.equal(ollamaTagForModel(config, ""), null);
});
