import assert from "node:assert/strict";
import test from "node:test";

import {
  MODEL_TYPES,
  MODEL_FORMATS,
  RUNTIME_ROUTES,
  detectModelType,
  detectModelFormats,
  routeRuntime,
  describeModel,
  matchesModelFilters,
  normalizeTypeFilter,
  getModelType,
} from "../src/lib/model-taxonomy.js";

// ---- the distinction that matters for local chat --------------------------

test("base, instruct and reasoning are told apart by name", () => {
  const type = (repoId) => detectModelType({ repoId, pipelineTag: "text-generation" }).type;
  assert.equal(type("meta-llama/Meta-Llama-3.1-8B-Instruct"), "instruct");
  assert.equal(type("meta-llama/Meta-Llama-3.1-8B"), "base");
  assert.equal(type("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"), "reasoning");
  assert.equal(type("Qwen/QwQ-32B"), "reasoning");
  assert.equal(type("google/gemma-2-9b-it"), "instruct");
});

test("a text model with no tuning signal is called base, and says so is a weak read", () => {
  // Being honest about this is the point: a base model shipped as the chat
  // model is what made Lizard answer a greeting by repeating it.
  const verdict = detectModelType({ repoId: "someone/Mystery-7B", pipelineTag: "text-generation" });
  assert.equal(verdict.type, "base");
  assert.equal(verdict.confidence, "low");
  assert.match(verdict.reason, /no instruct, chat-template or reasoning signal/);
});

test("a chat template outranks a silent name", () => {
  const verdict = detectModelType({ repoId: "someone/Mystery-7B", chatTemplate: "{% for m in messages %}" });
  assert.equal(verdict.type, "instruct");
  assert.match(verdict.reason, /chat template/);
});

test("the author prefix never hides the marker", () => {
  // Repo ids are "author/name", so the boundary class must treat "/" as a
  // separator - without it BAAI/bge and openai/whisper both read as base.
  const type = (repoId) => detectModelType({ repoId, pipelineTag: "text-generation" }).type;
  assert.equal(type("BAAI/bge-large-en-v1.5"), "embedding");
  assert.equal(type("openai/whisper-large-v3"), "audio");
  assert.equal(type("stabilityai/stable-diffusion-3-medium"), "image");
  assert.equal(type("BAAI/bge-reranker-v2-m3"), "rerank");
});

test("run-together embedding names are still embeddings", () => {
  assert.equal(detectModelType({ repoId: "ggml-org/embeddinggemma-300M-GGUF" }).type, "embedding");
});

test("an instruction-tuned code model stays findable under Instruct", () => {
  // Someone filtering Instruct to get a working chat model should still see
  // Coder-Instruct; a plain Coder with no tuning marker stays code.
  assert.equal(detectModelType({ repoId: "Qwen/Qwen2.5-Coder-7B-Instruct" }).type, "instruct");
  assert.equal(detectModelType({ repoId: "Qwen/Qwen2.5-Coder-7B" }).type, "code");
});

test("the pipeline tag beats the name for non-text models", () => {
  const verdict = detectModelType({
    repoId: "unsloth/Qwen3-27B-GGUF",
    pipelineTag: "image-text-to-text",
  });
  assert.equal(verdict.type, "multimodal");
  assert.match(verdict.reason, /pipeline tag/);
});

// ---- chat capability ------------------------------------------------------

test("only genuinely non-conversational kinds are excluded from chat", () => {
  const chat = (id) => getModelType(id).localChat;
  for (const id of ["base", "instruct", "reasoning", "code", "multimodal"]) {
    assert.equal(chat(id), true, `${id} can hold a conversation`);
  }
  for (const id of ["embedding", "rerank", "audio", "image"]) {
    assert.equal(chat(id), false, `${id} cannot`);
  }
});

test("a vision-language instruct model is not filtered out of chat", () => {
  // gemma-4-...-it is multimodal AND instruct; excluding it would hide some of
  // the best chat models on the Hub.
  const described = describeModel({
    repoId: "unsloth/gemma-4-26B-A4B-it-GGUF",
    tags: ["gguf"],
    pipelineTag: "image-text-to-text",
  });
  assert.equal(described.type, "multimodal");
  assert.equal(described.runtime.chatCapable, true);
  assert.equal(matchesModelFilters(described, { chatOnly: true }), true);
});

test("an embedding GGUF loads but is not offered as a chat model", () => {
  const described = describeModel({
    repoId: "ggml-org/embeddinggemma-300M-GGUF",
    tags: ["gguf"],
  });
  assert.equal(described.type, "embedding");
  assert.equal(described.runtime.runnable, true, "it is still a valid GGUF");
  assert.equal(described.runtime.chatCapable, false);
  assert.match(described.runtime.reason, /it does not chat/);
  assert.equal(matchesModelFilters(described, { chatOnly: true }), false);
});

// ---- formats and routing --------------------------------------------------

test("every weight format a repo ships is detected, from tags and files", () => {
  const formats = detectModelFormats({
    repoId: "org/model-AWQ",
    tags: ["safetensors", "onnx"],
    files: [{ rfilename: "model.gguf" }, { rfilename: "model.safetensors" }],
  });
  for (const expected of ["gguf", "safetensors", "onnx", "awq"]) {
    assert.ok(formats.includes(expected), `${expected} detected`);
  }
});

test("routing: GGUF runs here, everything else says why not and what to do", () => {
  const gguf = routeRuntime({ formats: ["gguf"], type: "instruct" });
  assert.equal(gguf.route, RUNTIME_ROUTES.NATIVE);
  assert.equal(gguf.runnable, true);
  assert.deepEqual(gguf.engines, ["lizard-native", "caterpillar", "llama.cpp"]);

  const safet = routeRuntime({ formats: ["safetensors"], type: "instruct" });
  assert.equal(safet.route, RUNTIME_ROUTES.OLLAMA);
  assert.equal(safet.runnable, false);
  assert.match(safet.reason, /Import it through Ollama|GGUF build/);

  for (const format of ["gptq", "awq", "mlx", "onnx"]) {
    const verdict = routeRuntime({ formats: [format], type: "instruct" });
    assert.equal(verdict.route, RUNTIME_ROUTES.UNSUPPORTED, `${format} is not runnable here`);
    assert.match(verdict.reason, /GGUF/, `${format} points at the GGUF alternative`);
  }
});

test("a pre-download verdict is labelled as a prediction, not a fact", () => {
  // The authoritative check needs the GGUF header and belongs to the runtime's
  // own decider; discovery must not present a guess as a contract.
  assert.equal(routeRuntime({ formats: ["gguf"] }).predicted, true);
});

test("a repo with no runnable weights is refused clearly", () => {
  const verdict = routeRuntime({ formats: [], type: null });
  assert.equal(verdict.route, RUNTIME_ROUTES.UNSUPPORTED);
  assert.equal(verdict.runnable, false);
});

// ---- filtering ------------------------------------------------------------

test("filters combine, and an empty filter means no filtering", () => {
  const described = describeModel({ repoId: "org/Llama-3-8B-Instruct-GGUF", tags: ["gguf"] });
  assert.equal(matchesModelFilters(described, {}), true);
  assert.equal(matchesModelFilters(described, { types: ["instruct"] }), true);
  assert.equal(matchesModelFilters(described, { types: ["base"] }), false);
  assert.equal(matchesModelFilters(described, { formats: ["gguf"] }), true);
  assert.equal(matchesModelFilters(described, { formats: ["mlx"] }), false);
  assert.equal(matchesModelFilters(described, { types: ["instruct"], formats: ["mlx"] }), false);
});

test("unknown filter values are dropped rather than emptying the results", () => {
  assert.deepEqual(normalizeTypeFilter(["instruct", "nonsense", "BASE"]), ["instruct", "base"]);
  assert.deepEqual(normalizeTypeFilter("reasoning,embedding"), ["reasoning", "embedding"]);
});

// ---- taxonomy shape -------------------------------------------------------

test("every type and format is presentable in the filter UI", () => {
  assert.ok(MODEL_TYPES.length >= 9);
  for (const type of MODEL_TYPES) {
    assert.ok(type.id && type.label && type.summary, `${type.id} explains itself`);
    assert.equal(typeof type.localChat, "boolean");
  }
  for (const format of MODEL_FORMATS) {
    assert.ok(format.id && format.label && format.summary, `${format.id} explains itself`);
  }
  // The kinds the goal called out are all represented.
  const ids = MODEL_TYPES.map((type) => type.id);
  for (const expected of ["base", "instruct", "reasoning", "multimodal", "embedding", "audio", "image"]) {
    assert.ok(ids.includes(expected), `${expected} is offered as a filter`);
  }
});
