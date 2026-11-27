// Universal model adapter: what a model IS, and which runtime can run it.
//
// Discovery used to ask one question - "is this repo GGUF?" - and sort the
// rest into coarse pipeline buckets. That hides the distinction that actually
// decides whether a local chat works: a BASE model completes text, an
// INSTRUCT model answers, and a REASONING model thinks first. Lizard shipped a
// base model as the default chat model for exactly this reason.
//
// This module classifies a model on three independent axes and never collapses
// them into one filter:
//
//   TYPE     base | instruct | reasoning | multimodal | embedding | audio |
//            image | rerank | code | moe (a trait, reported alongside)
//   FORMAT   gguf | safetensors | gptq | awq | mlx | onnx | ...
//   RUNTIME  which of Lizard's engines can execute it, and why not otherwise
//
// Everything here reads metadata a Hub listing already carries (repo id, tags,
// pipeline_tag, file names, config). Nothing is fabricated: a field that cannot
// be determined comes back null with a stated reason, and a routing verdict
// taken before the weights are on disk is marked `predicted` - the authoritative
// check is the runtime's own decider once the GGUF header can be read.

// ---- model type -----------------------------------------------------------

export const MODEL_TYPES = Object.freeze([
  {
    id: "reasoning",
    label: "Reasoning",
    summary: "Thinks step by step before answering. Slowest, strongest on hard problems.",
    localChat: true,
  },
  {
    id: "instruct",
    label: "Instruct / Chat",
    summary: "Tuned to follow instructions and hold a conversation. The right default for chat.",
    localChat: true,
  },
  {
    id: "code",
    label: "Code",
    summary: "Specialised for reading and writing code.",
    localChat: true,
  },
  {
    id: "base",
    label: "Base / Pretrained",
    summary: "Raw completion model. Continues text rather than answering it - poor at chat.",
    localChat: true,
  },
  {
    id: "multimodal",
    label: "Multimodal / Vision",
    summary: "Accepts images alongside text. Chats normally here; the image side needs a vision runtime.",
    // Chat-capable on purpose. A vision-language model is instruction-tuned -
    // "gemma-4-26B-A4B-it" is both multimodal and an instruct model - so
    // excluding it from the chat filter would hide some of the best chat
    // models on the Hub. Lizard just cannot feed it images yet.
    localChat: true,
  },
  {
    id: "embedding",
    label: "Embedding",
    summary: "Turns text into vectors for search and retrieval. Does not chat.",
    localChat: false,
  },
  {
    id: "rerank",
    label: "Reranker",
    summary: "Scores how well a document answers a query. Does not chat.",
    localChat: false,
  },
  {
    id: "audio",
    label: "Audio / Speech",
    summary: "Speech recognition or synthesis.",
    localChat: false,
  },
  {
    id: "image",
    label: "Image generation",
    summary: "Diffusion and image models.",
    localChat: false,
  },
]);

const TYPE_IDS = new Set(MODEL_TYPES.map((type) => type.id));

export function getModelType(id) {
  return MODEL_TYPES.find((type) => type.id === id) ?? null;
}

// Name fragments are checked as whole-ish tokens so "based" does not read as
// "base" and "Codestral" does not swallow every repo containing "code".
//
// The boundary class includes "/" because a repo id is "author/name": without
// it "BAAI/bge-large" and "openai/whisper-large" were classified as base text
// models, since the marker sits immediately after the slash.
const B = "[-_. /]";
const INSTRUCT_PATTERN = new RegExp(`(^|${B})(instruct|instruction|chat|it|sft|dpo|orpo|rlhf|assistant|hermes|vicuna|alpaca|tulu|zephyr|openchat|dolphin|wizardlm)(${B}|$)`, "i");
const REASONING_PATTERN = new RegExp(`(^|${B})(reason(er|ing)?|thinking|thought|cot|r1|qwq|marco-?o1|deepseek-?r1|distill-?r1)(${B}|$)`, "i");
const CODE_PATTERN = new RegExp(`(^|${B})(code[rl]?|coding|codestral|starcoder|codegemma|codellama|deepseek-?coder|qwen-?coder|granite-?code|sql)(${B}|$)`, "i");
const BASE_PATTERN = new RegExp(`(^|${B})(base|pt|pretrain(ed)?|foundation)(${B}|$)`, "i");
const RERANK_PATTERN = new RegExp(`(^|${B})(rerank(er)?|cross-?encoder)(${B}|$)`, "i");
// "embed[a-z]*" so run-together names like "embeddinggemma" are caught too.
const EMBED_PATTERN = new RegExp(`(^|${B})(embed[a-z]*|bge|gte|e5|nomic-?embed|minilm|sentence-?transformers?)(${B}|$)`, "i");
const AUDIO_PATTERN = new RegExp(`(^|${B})(whisper|wav2vec|bark|vits|x?tts|speecht5|parler|musicgen|encodec|distil-?whisper)(${B}|$)`, "i");
const IMAGE_PATTERN = new RegExp(`(^|${B})(stable-?diffusion|sdxl|sd3|flux|diffusion|kandinsky|pixart|playground-v)(${B}|$)`, "i");
const VISION_PATTERN = new RegExp(`(^|${B})(vl|vlm|vision|llava|moondream|internvl|minicpm-?v|pixtral)(${B}|$)`, "i");

// pipeline_tag is the strongest non-text signal: a model tagged for speech or
// image generation is not a chat model whatever its name says.
const PIPELINE_TYPES = new Map([
  ["feature-extraction", "embedding"],
  ["sentence-similarity", "embedding"],
  ["text-ranking", "rerank"],
  ["image-text-to-text", "multimodal"],
  ["visual-question-answering", "multimodal"],
  ["image-to-text", "multimodal"],
  ["video-text-to-text", "multimodal"],
  ["any-to-any", "multimodal"],
  ["automatic-speech-recognition", "audio"],
  ["text-to-speech", "audio"],
  ["text-to-audio", "audio"],
  ["audio-classification", "audio"],
  ["audio-text-to-text", "audio"],
  ["text-to-image", "image"],
  ["image-to-image", "image"],
  ["text-to-video", "image"],
  ["unconditional-image-generation", "image"],
]);

/**
 * What kind of model this is, with the evidence that decided it.
 *
 * Order matters and encodes the local-inference hierarchy the user cares about:
 * a non-text pipeline wins outright, then reasoning, then instruct, then code,
 * and BASE is the honest default for a text model with no tuning signal - not
 * a guess dressed up as a fact.
 */
export function detectModelType({
  repoId = "",
  tags = [],
  pipelineTag = null,
  chatTemplate = null,
  architecture = null,
} = {}) {
  const name = String(repoId ?? "");
  const lowerTags = (Array.isArray(tags) ? tags : []).map((tag) => String(tag).toLowerCase());
  const tagSet = new Set(lowerTags);
  const pipeline = String(pipelineTag ?? "").toLowerCase();

  const decide = (id, reason, confidence = "high") => ({
    type: id,
    label: getModelType(id)?.label ?? id,
    reason,
    confidence,
  });

  if (PIPELINE_TYPES.has(pipeline)) {
    return decide(PIPELINE_TYPES.get(pipeline), `Hugging Face pipeline tag "${pipeline}".`);
  }
  if (tagSet.has("diffusers") || tagSet.has("stable-diffusion") || IMAGE_PATTERN.test(name)) {
    return decide("image", "Named or tagged as a diffusion / image model.");
  }
  if (AUDIO_PATTERN.test(name)) {
    return decide("audio", "Named as a speech / audio model.");
  }
  if (RERANK_PATTERN.test(name)) {
    return decide("rerank", "Named as a reranker / cross-encoder.");
  }
  if (tagSet.has("sentence-transformers") || EMBED_PATTERN.test(name)) {
    return decide("embedding", "Named or tagged as an embedding model.");
  }
  if (tagSet.has("multimodal") || tagSet.has("vision") || VISION_PATTERN.test(name)) {
    return decide("multimodal", "Named or tagged as a vision-language model.");
  }
  if (REASONING_PATTERN.test(name)) {
    return decide("reasoning", "Named as a reasoning / thinking model.");
  }
  // Instruct is tested before code on purpose: "Qwen2.5-Coder-7B-Instruct" is
  // an instruction-tuned model that happens to specialise in code, and someone
  // filtering for Instruct to get a working chat model should still find it.
  // A plain "Coder" with no tuning marker stays code.
  if (INSTRUCT_PATTERN.test(name)) {
    return decide("instruct", "Named as an instruct / chat model.");
  }
  if (CODE_PATTERN.test(name)) {
    return decide("code", "Named as a code model.");
  }
  // A chat template in the metadata is proof the model was trained with turns,
  // even when the name says nothing - stronger evidence than any pattern.
  if (chatTemplate) {
    return decide("instruct", "Carries a chat template, so it was tuned on conversations.");
  }
  if (BASE_PATTERN.test(name)) {
    return decide("base", "Named as a base / pretrained model.");
  }
  if (pipeline === "text-generation" || pipeline === "text2text-generation" || architecture) {
    return decide(
      "base",
      "A text model with no instruct, chat-template or reasoning signal - treated as base.",
      "low",
    );
  }
  return { type: null, label: null, reason: "Not enough metadata to classify this repo.", confidence: "none" };
}

// ---- format ---------------------------------------------------------------

export const MODEL_FORMATS = Object.freeze([
  { id: "gguf", label: "GGUF", summary: "llama.cpp's format. What Lizard runs natively." },
  { id: "safetensors", label: "Safetensors", summary: "Full-precision weights. Convertible; not directly runnable here." },
  { id: "gptq", label: "GPTQ", summary: "GPU-quantised weights for GPTQ runtimes." },
  { id: "awq", label: "AWQ", summary: "Activation-aware quantised weights." },
  { id: "mlx", label: "MLX", summary: "Apple-silicon format." },
  { id: "onnx", label: "ONNX", summary: "Portable graph format." },
  { id: "pytorch", label: "PyTorch", summary: "Legacy .bin checkpoints." },
]);

const FORMAT_FILE_PATTERNS = [
  ["gguf", /\.gguf$/i],
  ["safetensors", /\.safetensors$/i],
  ["onnx", /\.onnx$/i],
  ["mlx", /(^|\/)mlx[-_]?/i],
  ["pytorch", /(pytorch_model.*\.bin|\.pt$|\.pth$)/i],
];

/** Every weight format a repo actually ships, from its tags and file list. */
export function detectModelFormats({ tags = [], files = [], repoId = "" } = {}) {
  const found = new Set();
  const lowerTags = (Array.isArray(tags) ? tags : []).map((tag) => String(tag).toLowerCase());
  const name = String(repoId ?? "").toLowerCase();

  for (const tag of lowerTags) {
    if (tag === "gguf") found.add("gguf");
    if (tag === "safetensors") found.add("safetensors");
    if (tag === "onnx") found.add("onnx");
    if (tag === "mlx") found.add("mlx");
    if (tag.includes("gptq")) found.add("gptq");
    if (tag.includes("awq")) found.add("awq");
  }
  // Quantisation families are usually only in the repo name.
  if (/(^|[-_. ])gptq([-_. ]|$)/i.test(name)) found.add("gptq");
  if (/(^|[-_. ])awq([-_. ]|$)/i.test(name)) found.add("awq");
  if (/(^|[-_. ])mlx([-_. ]|$)/i.test(name)) found.add("mlx");

  for (const file of Array.isArray(files) ? files : []) {
    const filename = String(file?.rfilename ?? file?.path ?? file ?? "");
    for (const [format, pattern] of FORMAT_FILE_PATTERNS) {
      if (pattern.test(filename)) found.add(format);
    }
  }
  return [...found];
}

// ---- runtime routing ------------------------------------------------------

export const RUNTIME_ROUTES = Object.freeze({
  NATIVE: "lizard-native",
  CATERPILLAR: "caterpillar",
  LLAMACPP: "llama.cpp",
  OLLAMA: "ollama",
  UNSUPPORTED: "unsupported",
});

/**
 * Which engine would run this model, and why not when none would.
 *
 * BEFORE DOWNLOAD this is a prediction from repo metadata and is returned with
 * `predicted: true`. The authoritative answer needs the GGUF header and comes
 * from the runtime's own decider (evaluateNativeGgufCompatibility) once the
 * file is local - this never second-guesses it with parallel rules.
 */
export function routeRuntime({ formats = [], type = null, architecture = null } = {}) {
  const has = (format) => formats.includes(format);
  const typeInfo = type ? getModelType(type) : null;

  if (has("gguf")) {
    // Non-text GGUFs exist (embedding and vision models ship them), and they
    // load, but they are not chat models - say so rather than implying chat.
    const chatCapable = !typeInfo || typeInfo.localChat;
    return {
      route: RUNTIME_ROUTES.NATIVE,
      engines: [RUNTIME_ROUTES.NATIVE, RUNTIME_ROUTES.CATERPILLAR, RUNTIME_ROUTES.LLAMACPP],
      runnable: true,
      chatCapable,
      predicted: true,
      reason: chatCapable
        ? "GGUF: runs on Lizard's native engine, with caterpillar and llama.cpp as alternatives."
        : `GGUF, but this is a ${typeInfo.label.toLowerCase()} model - it loads, it does not chat.`,
    };
  }
  if (has("safetensors")) {
    return {
      route: RUNTIME_ROUTES.OLLAMA,
      engines: [RUNTIME_ROUTES.OLLAMA],
      runnable: false,
      chatCapable: Boolean(!typeInfo || typeInfo.localChat),
      predicted: true,
      reason: "Safetensors: not directly runnable here. Import it through Ollama, or pick a GGUF build.",
    };
  }
  for (const [format, note] of [
    ["gptq", "GPTQ needs a GPTQ-capable serving runtime; Lizard runs GGUF."],
    ["awq", "AWQ needs an AWQ-capable serving runtime; Lizard runs GGUF."],
    ["mlx", "MLX is Apple-silicon only."],
    ["onnx", "ONNX needs an ONNX runtime; Lizard runs GGUF."],
  ]) {
    if (has(format)) {
      return {
        route: RUNTIME_ROUTES.UNSUPPORTED,
        engines: [],
        runnable: false,
        chatCapable: false,
        predicted: true,
        reason: `${note} Look for a GGUF conversion of the same model.`,
      };
    }
  }
  return {
    route: RUNTIME_ROUTES.UNSUPPORTED,
    engines: [],
    runnable: false,
    chatCapable: false,
    predicted: true,
    reason: "No weight format Lizard can run was found in this repo.",
  };
}

/**
 * One adapter record for a discovered model: what it is, what it ships, and
 * how it would run. This is the shape discovery surfaces filter on.
 */
export function describeModel({
  repoId = "",
  tags = [],
  pipelineTag = null,
  files = [],
  chatTemplate = null,
  architecture = null,
  contextLength = null,
  quantization = null,
} = {}) {
  let typed = detectModelType({ repoId, tags, pipelineTag, chatTemplate, architecture });
  const formats = detectModelFormats({ tags, files, repoId });
  // A GGUF repo that matched nothing is still, in practice, a text LLM - GGUF
  // is a language-model format. Calling it base (weakly) beats leaving it
  // unclassified, because an unknown type would otherwise be waved through the
  // chat filter as though it were known to converse.
  if (!typed.type && formats.includes("gguf")) {
    typed = {
      type: "base",
      label: getModelType("base").label,
      reason: "A GGUF text model with no instruct, chat-template or reasoning signal.",
      confidence: "low",
    };
  }
  const runtime = routeRuntime({ formats, type: typed.type, architecture });
  const lowerTags = (Array.isArray(tags) ? tags : []).map((tag) => String(tag).toLowerCase());
  return {
    repoId,
    type: typed.type,
    typeLabel: typed.label,
    typeReason: typed.reason,
    typeConfidence: typed.confidence,
    formats,
    architecture: architecture ?? null,
    contextLength: Number.isFinite(contextLength) ? contextLength : null,
    quantization: quantization ?? null,
    chatTemplate: chatTemplate ? true : null,
    moe: lowerTags.includes("moe") || /(^|[-_. ])(moe|a\d+b)([-_. ]|$)/i.test(String(repoId)),
    runtime,
  };
}

/** Filter a described model against the discovery selections. Empty = allow. */
export function matchesModelFilters(described, { types = [], formats = [], chatOnly = false } = {}) {
  if (chatOnly && !described?.runtime?.chatCapable) {
    return false;
  }
  if (types.length > 0 && !types.includes(described?.type)) {
    return false;
  }
  if (formats.length > 0 && !formats.some((format) => (described?.formats ?? []).includes(format))) {
    return false;
  }
  return true;
}

export function normalizeTypeFilter(values) {
  const list = Array.isArray(values) ? values : String(values ?? "").split(",");
  return list.map((value) => String(value).trim().toLowerCase()).filter((value) => TYPE_IDS.has(value));
}
