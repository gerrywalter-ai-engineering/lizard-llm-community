const GIB = 1024 * 1024 * 1024;
const NATIVE_CPU_MEMORY_HEADROOM_RATIO = 0.8;

// Quantizations the compiled worker accepts for the layered GGUF graphs.
const NATIVE_WORKER_QUANTIZATIONS = ["f32", "f16", "bf16", "q8_0", "q4_k_m", "q5_k_m", "q6_k", "iq4_xs"];

// Canonical native-worker family matrix for the GGUF single-wave expansion.
// One row per canonical group: GGUF architecture aliases, tokenizer family,
// graph family, quant expectations, and the current worker status. This is
// the single source every surface (chat, onboarding, dashboard, benchmark,
// CI, docs) must report from.
const NATIVE_GGUF_FAMILY_MATRIX = [
  {
    family: "llama",
    ggufArchitectures: ["llama"],
    tokenizerFamily: "gpt2-bpe / sentencepiece",
    graphFamily: "llama",
    quantizations: NATIVE_WORKER_QUANTIZATIONS,
    workerStatus: "implemented",
    reason: "The compiled native CPU worker executes the Llama GGUF graph natively.",
  },
  {
    family: "gemma",
    ggufArchitectures: ["gemma", "gemma2", "gemma3", "gemma4"],
    tokenizerFamily: "sentencepiece / gemma4-bpe",
    graphFamily: "gemma",
    quantizations: ["q4_k_m", "q8_0", "f32"],
    workerStatus: "implemented",
    reason: "The compiled native CPU worker executes the guarded Gemma GGUF subset natively.",
  },
  {
    family: "qwen2",
    ggufArchitectures: ["qwen2"],
    tokenizerFamily: "gpt2-bpe",
    graphFamily: "llama + attention biases, NEOX rope",
    quantizations: NATIVE_WORKER_QUANTIZATIONS,
    workerStatus: "implemented",
    reason: "The compiled native CPU worker executes the qwen2 GGUF family natively.",
  },
  {
    family: "qwen3",
    ggufArchitectures: ["qwen3"],
    tokenizerFamily: "gpt2-bpe",
    graphFamily: "llama + per-head q/k RMS norm, explicit head dim, NEOX rope",
    quantizations: NATIVE_WORKER_QUANTIZATIONS,
    workerStatus: "implemented",
    reason: "The compiled native CPU worker executes the qwen3 GGUF family natively.",
  },
  {
    family: "mistral",
    ggufArchitectures: ["mistral", "llama"],
    tokenizerFamily: "sentencepiece / gpt2-bpe",
    graphFamily: "llama",
    quantizations: NATIVE_WORKER_QUANTIZATIONS,
    workerStatus: "implemented",
    reason: "Mistral GGUFs convert to the llama graph, which the compiled native CPU worker executes natively.",
  },
  {
    family: "mixtral",
    ggufArchitectures: ["llama (expert_count > 0)"],
    tokenizerFamily: "sentencepiece",
    graphFamily: "llama + softmax top-k MoE routing",
    quantizations: NATIVE_WORKER_QUANTIZATIONS,
    workerStatus: "implemented",
    reason: "The compiled native CPU worker executes Mixtral-style llama MoE GGUFs natively.",
  },
  {
    family: "phi",
    ggufArchitectures: ["phi2"],
    tokenizerFamily: "gpt2-bpe",
    graphFamily: "LayerNorm + parallel residual + partial NEOX rope + GELU FFN",
    quantizations: NATIVE_WORKER_QUANTIZATIONS,
    workerStatus: "implemented",
    reason: "The compiled native CPU worker executes the phi (phi2) GGUF family natively.",
  },
  {
    family: "phi3",
    ggufArchitectures: ["phi3"],
    tokenizerFamily: "sentencepiece",
    graphFamily: "fused QKV + fused SwiGLU + longrope factors (covers phi3, phi3.5, phi4)",
    quantizations: NATIVE_WORKER_QUANTIZATIONS,
    workerStatus: "implemented",
    reason: "The compiled native CPU worker executes the phi3/phi4 GGUF family natively.",
  },
  {
    family: "phi4",
    ggufArchitectures: ["phi3"],
    tokenizerFamily: "gpt2-bpe / sentencepiece",
    graphFamily: "phi3 graph (phi4 GGUFs convert to the phi3 architecture)",
    quantizations: NATIVE_WORKER_QUANTIZATIONS,
    workerStatus: "implemented",
    reason: "Phi-4 GGUFs use the phi3 architecture, which the compiled native CPU worker executes natively.",
  },
  {
    family: "qwen2moe",
    ggufArchitectures: ["qwen2moe"],
    tokenizerFamily: "gpt2-bpe",
    graphFamily: "qwen2 + softmax top-k MoE + sigmoid-gated shared expert",
    quantizations: NATIVE_WORKER_QUANTIZATIONS,
    workerStatus: "implemented",
    reason: "The compiled native CPU worker executes the qwen2moe GGUF family natively.",
  },
  {
    family: "qwen3moe",
    ggufArchitectures: ["qwen3moe"],
    tokenizerFamily: "gpt2-bpe",
    graphFamily: "qwen3 + softmax top-k MoE with renormalized weights",
    quantizations: NATIVE_WORKER_QUANTIZATIONS,
    workerStatus: "implemented",
    reason: "The compiled native CPU worker executes the qwen3moe GGUF family natively.",
  },
  {
    family: "deepseek",
    ggufArchitectures: ["deepseek"],
    tokenizerFamily: "gpt2-bpe",
    graphFamily: "llama + leading dense blocks + MoE with shared experts",
    quantizations: NATIVE_WORKER_QUANTIZATIONS,
    workerStatus: "implemented",
    reason: "The compiled native CPU worker executes the deepseek (V1) GGUF family natively.",
    notes: "deepseek2/deepseek3 (MLA attention) are tracked as blocked below.",
  },
  {
    family: "glm",
    ggufArchitectures: ["glm4"],
    tokenizerFamily: "gpt2-bpe",
    graphFamily: "llama + GLM4 post-attention/post-FFN norms + partial rope + fused SwiGLU",
    quantizations: NATIVE_WORKER_QUANTIZATIONS,
    workerStatus: "implemented",
    reason: "The compiled native CPU worker executes the glm (glm4) GGUF family natively.",
    notes: "Legacy chatglm GGUFs are tracked as blocked below.",
  },
  {
    family: "starcoder2",
    ggufArchitectures: ["starcoder2"],
    tokenizerFamily: "gpt2-bpe",
    graphFamily: "LayerNorm + biased projections + NEOX rope + GELU FFN",
    quantizations: NATIVE_WORKER_QUANTIZATIONS,
    workerStatus: "implemented",
    reason: "The compiled native CPU worker executes the starcoder2 GGUF family natively.",
  },
  {
    family: "deepseek2",
    ggufArchitectures: ["deepseek2", "deepseek3"],
    tokenizerFamily: "gpt2-bpe",
    graphFamily: "Multi-head Latent Attention (not implemented)",
    quantizations: [],
    workerStatus: "blocked",
    reason: "deepseek2/deepseek3 GGUFs require Multi-head Latent Attention (q_a/kv_a latent compression projections and decoupled RoPE keys), which the native worker does not implement yet.",
    missingWorkerCapability: "MLA attention graph topology (latent kv compression, decoupled rope key)",
    willBeUpdatedSoon: true,
  },
  {
    family: "chatglm",
    ggufArchitectures: ["chatglm"],
    tokenizerFamily: "gpt2-bpe",
    graphFamily: "pre-GLM4 packed QKV/rope layout (not implemented)",
    quantizations: [],
    workerStatus: "blocked",
    reason: "Legacy chatglm GGUFs use the pre-GLM4 packed QKV/rope layout, which the native worker does not implement yet; the glm4 lineage is supported natively.",
    missingWorkerCapability: "legacy ChatGLM packed QKV tensor mapping and rope layout",
    willBeUpdatedSoon: true,
  },
  // Model wrapper Phase 8 (spec s4): gpt-oss as a FIRST-CLASS family. First
  // class = correctly described, routable and benchmarkable - NOT natively
  // executed. The GGUF conversions run on llama.cpp/Ollama; the native worker
  // does not implement this MoE architecture.
  {
    family: "gpt-oss",
    ggufArchitectures: ["gpt-oss", "gptoss"],
    tokenizerFamily: "o200k-harmony",
    graphFamily: "sparse MoE with attention sinks (not implemented)",
    quantizations: [],
    workerStatus: "blocked",
    reason: "gpt-oss (OpenAI open-weight) is a sparse Mixture-of-Experts model with reasoning and tool calling. Its GGUF conversions run on llama.cpp/Ollama; the native worker does not implement MoE expert routing yet. gpt-oss-20b fits ~16 GB; gpt-oss-120b needs a large machine although active parameters are far below total.",
    missingWorkerCapability: "MoE expert routing and attention-sink graph",
    capabilities: ["reasoning", "tool_calling", "moe"],
    willBeUpdatedSoon: false,
  },
  {
    family: "kimi",
    ggufArchitectures: ["kimi", "kimi-k2", "deepseek2"],
    tokenizerFamily: "tiktoken-derived",
    graphFamily: "very large MoE / MLA (not implemented)",
    quantizations: [],
    workerStatus: "blocked",
    reason: "Kimi K2 (Moonshot AI) is a very large agentic/coding MoE. GGUF conversions are version-dependent on llama.cpp/Ollama; the native worker does not implement this MoE/MLA architecture. Treat native as unsupported and check the runtime version for GGUF support.",
    missingWorkerCapability: "very large MoE + latent attention",
    capabilities: ["reasoning", "tool_calling", "moe", "code_generation"],
    willBeUpdatedSoon: false,
  },
  {
    family: "qwen2vl",
    ggufArchitectures: ["qwen2vl", "qwen2-vl"],
    tokenizerFamily: "gpt2-bpe",
    graphFamily: "Qwen2 text graph + vision encoder/projector (not implemented)",
    quantizations: [],
    workerStatus: "blocked",
    reason: "Qwen2-VL requires a vision encoder/projector graph. Choose a text-only Qwen2 GGUF for native execution or use a multimodal llama.cpp/Ollama build.",
  },
  {
    family: "qwen3vl",
    ggufArchitectures: ["qwen3vl", "qwen3-vl"],
    tokenizerFamily: "gpt2-bpe",
    graphFamily: "Qwen3 text graph + vision encoder/projector (not implemented)",
    quantizations: [],
    workerStatus: "blocked",
    reason: "Qwen3-VL requires a vision encoder/projector graph. Choose a text-only Qwen3 GGUF for native execution or use a multimodal llama.cpp/Ollama build.",
  },
  {
    family: "gemma3n",
    ggufArchitectures: ["gemma3n"],
    tokenizerFamily: "sentencepiece",
    graphFamily: "mobile multimodal/per-layer-input Gemma graph (not implemented)",
    quantizations: [],
    workerStatus: "blocked",
    reason: "Gemma 3n's mobile multimodal/per-layer-input graph is not implemented. Choose a supported text Gemma GGUF.",
  },
  {
    family: "mamba",
    ggufArchitectures: ["mamba", "mamba2"],
    tokenizerFamily: "model-specific",
    graphFamily: "state-space model (not implemented)",
    quantizations: [],
    workerStatus: "blocked",
    reason: "Mamba requires recurrent state-space updates that the native transformer worker does not implement.",
  },
  {
    family: "rwkv",
    ggufArchitectures: ["rwkv", "rwkv6", "rwkv7"],
    tokenizerFamily: "model-specific",
    graphFamily: "recurrent time-mixing (not implemented)",
    quantizations: [],
    workerStatus: "blocked",
    reason: "RWKV requires recurrent time-mixing state that the native transformer worker does not implement.",
  },
  {
    family: "t5",
    ggufArchitectures: ["t5", "t5encoder"],
    tokenizerFamily: "sentencepiece",
    graphFamily: "encoder-decoder cross-attention (not implemented)",
    quantizations: [],
    workerStatus: "blocked",
    reason: "T5 uses an encoder-decoder graph with cross-attention; the native worker currently executes decoder-only generation graphs.",
  },
  {
    family: "bert",
    ggufArchitectures: ["bert", "nomic-bert", "jina-bert-v2"],
    tokenizerFamily: "wordpiece / model-specific",
    graphFamily: "bidirectional encoder (not implemented)",
    quantizations: [],
    workerStatus: "blocked",
    reason: "BERT-family GGUFs are encoders; the native chat worker currently executes decoder-only generation graphs.",
  },
  {
    family: "falcon",
    ggufArchitectures: ["falcon"],
    tokenizerFamily: "gpt2-bpe",
    graphFamily: "Falcon attention/tensor layout (not implemented)",
    quantizations: [],
    workerStatus: "blocked",
    reason: "Falcon's GGUF tensor layout and attention graph are not implemented by the native worker.",
  },
  {
    family: "dbrx",
    ggufArchitectures: ["dbrx"],
    tokenizerFamily: "gpt2-bpe",
    graphFamily: "DBRX sparse MoE (not implemented)",
    quantizations: [],
    workerStatus: "blocked",
    reason: "DBRX uses a distinct sparse-MoE graph that the native worker does not implement.",
  },
  {
    family: "cohere2",
    ggufArchitectures: ["cohere2", "command-r"],
    tokenizerFamily: "sentencepiece",
    graphFamily: "Cohere/Command-R attention (not implemented)",
    quantizations: [],
    workerStatus: "blocked",
    reason: "Cohere/Command-R attention and normalization topology is not implemented by the native worker.",
  },
];

const ARCHITECTURE_MATRIX = {
  "tiny-linear": {
    status: "supported",
    backends: ["cpu"],
    quantizations: ["f32"],
    reason: "The native CPU runtime supports the tiny-linear reference architecture through the JS prototype path.",
  },
  llama: {
    status: "supported",
    backends: ["cpu"],
    quantizations: ["f32", "f16", "bf16", "q8_0", "q4_k_m", "q5_k_m", "q6_k"],
    reason: "The compiled native CPU worker can execute the current Llama GGUF subset.",
  },
  gemma: {
    status: "supported",
    backends: ["cpu"],
    quantizations: ["q4_k_m", "q8_0", "f32"],
    reason: "The compiled native CPU worker executes the guarded Gemma/Gemma2/Gemma3 GGUF subset.",
  },
  gemma4: {
    status: "supported",
    backends: ["cpu"],
    quantizations: ["q4_k_m"],
    reason: "The compiled native CPU worker can execute the guarded Gemma4 GGUF subset.",
  },
  qwen2: {
    status: "supported",
    backends: ["cpu"],
    quantizations: NATIVE_WORKER_QUANTIZATIONS,
    reason: "The compiled native CPU worker executes the qwen2 GGUF family natively.",
  },
  qwen3: {
    status: "supported",
    backends: ["cpu"],
    quantizations: NATIVE_WORKER_QUANTIZATIONS,
    reason: "The compiled native CPU worker executes the qwen3 GGUF family natively.",
  },
  phi: {
    status: "supported",
    backends: ["cpu"],
    quantizations: NATIVE_WORKER_QUANTIZATIONS,
    reason: "The compiled native CPU worker executes the phi (phi2) GGUF family natively.",
  },
  phi3: {
    status: "supported",
    backends: ["cpu"],
    quantizations: NATIVE_WORKER_QUANTIZATIONS,
    reason: "The compiled native CPU worker executes the phi3/phi4 GGUF family natively.",
  },
  mistral: {
    status: "supported",
    backends: ["cpu"],
    quantizations: NATIVE_WORKER_QUANTIZATIONS,
    reason: "Mistral GGUFs use the llama graph, which the compiled native CPU worker executes natively.",
  },
  mixtral: {
    status: "supported",
    backends: ["cpu"],
    quantizations: NATIVE_WORKER_QUANTIZATIONS,
    reason: "The compiled native CPU worker executes Mixtral-style llama MoE GGUFs natively.",
  },
  qwen2moe: {
    status: "supported",
    backends: ["cpu"],
    quantizations: NATIVE_WORKER_QUANTIZATIONS,
    reason: "The compiled native CPU worker executes the qwen2moe GGUF family natively.",
  },
  qwen3moe: {
    status: "supported",
    backends: ["cpu"],
    quantizations: NATIVE_WORKER_QUANTIZATIONS,
    reason: "The compiled native CPU worker executes the qwen3moe GGUF family natively.",
  },
  deepseek: {
    status: "supported",
    backends: ["cpu"],
    quantizations: NATIVE_WORKER_QUANTIZATIONS,
    reason: "The compiled native CPU worker executes the deepseek (V1) GGUF family natively.",
  },
  glm: {
    status: "supported",
    backends: ["cpu"],
    quantizations: NATIVE_WORKER_QUANTIZATIONS,
    reason: "The compiled native CPU worker executes the glm (glm4) GGUF family natively.",
  },
  starcoder2: {
    status: "supported",
    backends: ["cpu"],
    quantizations: NATIVE_WORKER_QUANTIZATIONS,
    reason: "The compiled native CPU worker executes the starcoder2 GGUF family natively.",
  },
  deepseek2: {
    status: "blocked",
    backends: [],
    quantizations: [],
    reason: "deepseek2/deepseek3 GGUFs require Multi-head Latent Attention (q_a/kv_a latent compression projections and decoupled RoPE keys), which the native worker does not implement yet. Status: blocked; will be updated soon.",
  },
  chatglm: {
    status: "blocked",
    backends: [],
    quantizations: [],
    reason: "Legacy chatglm GGUFs use the pre-GLM4 packed QKV/rope layout, which the native worker does not implement yet; the glm4 lineage is supported natively. Status: blocked; will be updated soon.",
  },
  // Phase 8: gpt-oss / kimi resolve to an explicit blocked verdict (not
  // "unknown") so a user selecting one is told exactly why native cannot run
  // it and that llama.cpp/Ollama can - rather than a generic failure.
  "gpt-oss": {
    status: "blocked",
    backends: [],
    quantizations: [],
    reason: "gpt-oss is a sparse MoE with reasoning and tool calling. GGUF conversions run on llama.cpp/Ollama; the native worker does not implement MoE expert routing yet.",
  },
  kimi: {
    status: "blocked",
    backends: [],
    quantizations: [],
    reason: "Kimi K2 is a very large MoE. GGUF support is runtime-version-dependent on llama.cpp/Ollama; the native worker does not implement this architecture.",
  },
  qwen2vl: {
    status: "blocked",
    backends: [],
    quantizations: [],
    reason: "Qwen2-VL GGUFs require a vision encoder/projector graph that the native text worker does not implement. Use a compatible text-only Qwen2 model or a multimodal llama.cpp/Ollama build.",
  },
  qwen3vl: {
    status: "blocked",
    backends: [],
    quantizations: [],
    reason: "Qwen3-VL GGUFs require a vision encoder/projector graph that the native text worker does not implement. Use a compatible text-only Qwen3 model or a multimodal llama.cpp/Ollama build.",
  },
  gemma3n: {
    status: "blocked",
    backends: [],
    quantizations: [],
    reason: "Gemma 3n uses an additional mobile multimodal/per-layer-input graph that is not implemented by the native worker. Use a supported text Gemma GGUF instead.",
  },
  mamba: {
    status: "blocked",
    backends: [],
    quantizations: [],
    reason: "Mamba GGUFs use a state-space model graph rather than transformer attention. The native worker does not implement recurrent SSM state updates.",
  },
  rwkv: {
    status: "blocked",
    backends: [],
    quantizations: [],
    reason: "RWKV GGUFs require recurrent time-mixing state that the native transformer worker does not implement.",
  },
  t5: {
    status: "blocked",
    backends: [],
    quantizations: [],
    reason: "T5 GGUFs use an encoder-decoder graph with cross-attention; the native decoder-only worker does not implement that topology.",
  },
  bert: {
    status: "blocked",
    backends: [],
    quantizations: [],
    reason: "BERT-family GGUFs are encoder models. The native chat worker currently executes decoder-only generation graphs.",
  },
  falcon: {
    status: "blocked",
    backends: [],
    quantizations: [],
    reason: "Falcon GGUF tensor layout and attention graph are not implemented by the native worker.",
  },
  dbrx: {
    status: "blocked",
    backends: [],
    quantizations: [],
    reason: "DBRX uses a distinct sparse-MoE graph that the native worker does not implement.",
  },
  cohere2: {
    status: "blocked",
    backends: [],
    quantizations: [],
    reason: "Cohere/Command-R GGUF attention and normalization topology is not implemented by the native worker.",
  },
  unknown: {
    status: "unsupported",
    backends: [],
    quantizations: [],
    reason: "Unknown architectures cannot be executed by the native CPU prototype.",
  },
};

const NATIVE_WORKER_FAMILIES = [
  "llama",
  "qwen2",
  "qwen3",
  "mistral",
  "mixtral",
  "phi",
  "phi3",
  "qwen2moe",
  "qwen3moe",
  "deepseek",
  "glm",
  "starcoder2",
];

const QUANTIZATION_MATRIX = {
  f32: {
    status: "supported",
    backends: ["cpu"],
    architectures: ["tiny-linear", ...NATIVE_WORKER_FAMILIES, "gemma", "gemma4"],
    bytesPerElement: 4,
    reason: "The native CPU runtime supports unquantized float32 weights.",
  },
  f16: {
    status: "supported",
    backends: ["cpu"],
    architectures: [...NATIVE_WORKER_FAMILIES],
    bytesPerElement: 2,
    reason: "The compiled native CPU worker supports float16 GGUF tensor dequantization.",
  },
  bf16: {
    status: "supported",
    backends: ["cpu"],
    architectures: [...NATIVE_WORKER_FAMILIES],
    bytesPerElement: 2,
    reason: "The compiled native CPU worker supports bfloat16 GGUF tensor dequantization.",
  },
  q8_0: {
    status: "supported",
    backends: ["cpu"],
    architectures: [...NATIVE_WORKER_FAMILIES, "gemma", "gemma4"],
    bytesPerElement: 1,
    reason: "The compiled native CPU worker supports Q8_0 GGUF tensor dequantization.",
  },
  q6_k: {
    status: "supported",
    backends: ["cpu"],
    architectures: [...NATIVE_WORKER_FAMILIES],
    bytesPerElement: null,
    reason: "The compiled native CPU worker supports Q6_K GGUF tensor dequantization.",
  },
  q5_k_m: {
    status: "supported",
    backends: ["cpu"],
    architectures: [...NATIVE_WORKER_FAMILIES],
    bytesPerElement: null,
    reason: "The compiled native CPU worker supports Q5_K_M GGUF tensor dequantization.",
  },
  q4_k_m: {
    status: "supported",
    backends: ["cpu"],
    architectures: [...NATIVE_WORKER_FAMILIES, "gemma", "gemma4"],
    bytesPerElement: null,
    reason: "The compiled native CPU worker supports the current Q4_K_M GGUF execution subset.",
  },
  iq4_xs: {
    status: "supported",
    backends: ["cpu", "gpu"],
    architectures: [...NATIVE_WORKER_FAMILIES],
    bytesPerElement: null,
    reason: "Caterpillar and the compiled native worker support IQ4_XS CPU dequantization; Caterpillar adds a resident D3D12 matvec kernel.",
  },
  tq2_0: {
    status: "planned",
    backends: [],
    architectures: [],
    bytesPerElement: null,
    reason: "TQ2_0 native CPU dequantization is not implemented in the prototype yet.",
  },
  tq1_0: {
    status: "planned",
    backends: [],
    architectures: [],
    bytesPerElement: null,
    reason: "TQ1_0 native CPU dequantization is not implemented in the prototype yet.",
  },
  unknown: {
    status: "unsupported",
    backends: [],
    architectures: [],
    bytesPerElement: null,
    reason: "Unknown quantization formats cannot be executed by the native CPU prototype.",
  },
};

function clone(value) {
  return JSON.parse(JSON.stringify(value));
}

function normalizeKey(value, fallback = "unknown") {
  const normalized = String(value ?? "").trim().toLowerCase();
  return normalized || fallback;
}

const ARCHITECTURE_ALIASES = {
  qwen: "qwen2",
  "qwen2.5": "qwen2",
  qwen2_5: "qwen2",
  qwen2vl: "qwen2vl",
  "qwen2-vl": "qwen2vl",
  qwen3vl: "qwen3vl",
  "qwen3-vl": "qwen3vl",
  "phi-3": "phi3",
  "phi-3.5": "phi3",
  "phi-4": "phi3",
  phi2: "phi",
  phi4: "phi3",
  glm4: "glm",
  gemma2: "gemma",
  gemma3: "gemma",
  "gemma-2": "gemma",
  "gemma-3": "gemma",
  "gemma-3n": "gemma3n",
  deepseek3: "deepseek2",
  "deepseek-v2": "deepseek2",
  "deepseek-v3": "deepseek2",
  gptoss: "gpt-oss",
  "gpt_oss": "gpt-oss",
  "kimi-k2": "kimi",
  kimik2: "kimi",
  mamba2: "mamba",
  "mamba-2": "mamba",
  rwkv6: "rwkv",
  rwkv7: "rwkv",
  t5encoder: "t5",
  "nomic-bert": "bert",
  "jina-bert-v2": "bert",
  "command-r": "cohere2",
  commandr: "cohere2",
};

function normalizeArchitectureKey(value, fallback = "unknown") {
  const normalized = normalizeKey(value, fallback);
  return ARCHITECTURE_ALIASES[normalized] ?? normalized;
}

// Canonical family for a GGUF: architecture aliases plus the Mixtral case,
// where the GGUF architecture stays `llama` and only expert_count reveals
// the MoE graph.
export function resolveGgufFamilyKey({ architecture = null, expertCount = 0, modelId = null } = {}) {
  const identity = String(modelId ?? "").toLowerCase();
  if (/gpt[-_. ]?oss/.test(identity)) return "gpt-oss";
  if (/\bkimi(?:[-_. ]?k2)?\b/.test(identity)) return "kimi";
  const normalized = normalizeArchitectureKey(architecture);
  if (normalized === "llama" && Number(expertCount) > 0) {
    return "mixtral";
  }
  return normalized;
}

export function buildNativeGgufFamilyMatrix() {
  return clone(NATIVE_GGUF_FAMILY_MATRIX);
}

// Model wrapper Phase 3 (spec s8/s13): the complete compatibility picture as
// one machine-readable document, assembled from the SAME constants every
// runtime decision consults - the code stays canonical, this is a projection.
// Consumed by /api/compatibility/matrix and the docs generator, so the docs
// physically cannot drift from what the runtime actually does.
export function listCompatibilityMatrix() {
  const families = clone(NATIVE_GGUF_FAMILY_MATRIX).map((row) => ({
    family: row.family,
    ggufArchitectures: row.ggufArchitectures,
    tokenizerFamily: row.tokenizerFamily ?? null,
    graphFamily: row.graphFamily ?? null,
    quantizations: row.quantizations ?? [],
    workerStatus: row.workerStatus,
    reason: row.reason ?? null,
    missingWorkerCapability: row.missingWorkerCapability ?? null,
    capabilities: row.capabilities ?? [],
    runtimes: {
      // The native worker's own verdict for this family.
      native: row.workerStatus === "implemented" ? "supported" : row.workerStatus,
      // llama.cpp executes every GGUF architecture in this matrix - that is
      // where these families were converted from in the first place.
      "llama.cpp": "supported",
      // Ollama wraps llama.cpp for GGUF, so the same set applies.
      ollama: "supported",
    },
  }));
  return {
    schema: "lizard.compatibility-matrix.v1",
    nativeWorkerQuantizations: [...NATIVE_WORKER_QUANTIZATIONS],
    families,
    quantizations: Object.entries(clone(QUANTIZATION_MATRIX)).map(([id, entry]) => ({
      id,
      status: entry.status,
      backends: entry.backends ?? [],
      bytesPerElement: entry.bytesPerElement ?? null,
      reason: entry.reason ?? null,
    })),
    architectures: Object.entries(clone(ARCHITECTURE_MATRIX)).map(([id, entry]) => ({
      id,
      status: entry.status,
      backends: entry.backends ?? [],
      quantizations: entry.quantizations ?? [],
      reason: entry.reason ?? null,
    })),
  };
}

// The decoder families the Caterpillar/Ostrich runner will actually load.
// FamilySpec::validate() accepts family llama, gemma or qwen2 and throws
// otherwise, and family_name() folds every gemma revision onto "gemma" and
// mistral onto "llama" (native/caterpillar-runner/src/family_spec.cpp and
// gguf_loader.cpp).
//
// This matrix used to be the native worker's list with the provider relabelled,
// so Caterpillar advertised fourteen families and rejected ten of them at load
// with "Caterpillar Phase 1 supports llama, gemma, and qwen2 decoder families".
// A model that plans and then dies on load is worse than one that says up front
// it is not supported, so the claim is derived from the runner's own gate.
const CATERPILLAR_RUNNER_FAMILIES = new Set([
  "llama", "mistral", "gemma", "gemma2", "gemma3", "gemma4", "qwen2",
]);

export function caterpillarRunnerLoadsFamily(family) {
  return CATERPILLAR_RUNNER_FAMILIES.has(String(family ?? "").trim().toLowerCase());
}

export function buildCaterpillarGgufFamilyMatrix() {
  return clone(NATIVE_GGUF_FAMILY_MATRIX).map((entry) => {
    const loadable = caterpillarRunnerLoadsFamily(entry.family);
    return {
      ...entry,
      provider: "caterpillar",
      implementationStatus: "experimental",
      executionProvider: "lizard-native-reference",
      workerStatus: loadable ? entry.workerStatus : "blocked",
      reason: loadable
        ? entry.reason
        : `The Caterpillar runner loads only the llama, gemma and qwen2 decoder`
          + ` families; ${entry.family} must run on lizard-native or llama.cpp.`,
    };
  });
}

function resolveMatrixEntry(matrix, key, normalizer = normalizeKey) {
  const normalizedKey = normalizer(key);
  return {
    key: normalizedKey,
    ...(clone(matrix[normalizedKey] ?? matrix.unknown)),
  };
}

function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes < 0) {
    return "unknown";
  }
  if (bytes < 1024) {
    return `${bytes} B`;
  }
  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(1)} KiB`;
  }
  if (bytes < GIB) {
    return `${(bytes / (1024 * 1024)).toFixed(1)} MiB`;
  }
  return `${(bytes / GIB).toFixed(2)} GiB`;
}

export function buildArchitectureCompatibilityMatrix() {
  return clone(ARCHITECTURE_MATRIX);
}

export function buildQuantizationCompatibilityMatrix() {
  return clone(QUANTIZATION_MATRIX);
}

export function buildCaterpillarArchitectureCompatibilityMatrix() {
  return Object.fromEntries(Object.entries(buildArchitectureCompatibilityMatrix()).map(([key, entry]) => [key, {
    ...entry,
    provider: "caterpillar",
    implementationStatus: "experimental",
  }]));
}

export function buildCaterpillarQuantizationCompatibilityMatrix() {
  return Object.fromEntries(Object.entries(buildQuantizationCompatibilityMatrix()).map(([key, entry]) => [key, {
    ...entry,
    provider: "caterpillar",
    implementationStatus: "experimental",
  }]));
}

export function getArchitectureCompatibility(architecture) {
  return resolveMatrixEntry(ARCHITECTURE_MATRIX, architecture, normalizeArchitectureKey);
}

export function getQuantizationCompatibility(quantization) {
  return resolveMatrixEntry(QUANTIZATION_MATRIX, quantization);
}

export function evaluateNativeGgufCompatibility({
  gguf = null,
  modelArtifact = null,
  profile = null,
} = {}) {
  const familyKey = resolveGgufFamilyKey({
    architecture: gguf?.architecture
      ?? profile?.gguf?.architecture
      ?? modelArtifact?.gguf?.architecture
      ?? null,
    expertCount: gguf?.expertCount
      ?? profile?.gguf?.expertCount
      ?? modelArtifact?.gguf?.expertCount
      ?? 0,
    modelId: modelArtifact?.modelId
      ?? modelArtifact?.id
      ?? profile?.modelId
      ?? null,
  });
  const architecture = getArchitectureCompatibility(familyKey);
  const quantization = getQuantizationCompatibility(
    gguf?.quant
      ?? profile?.gguf?.quant
      ?? modelArtifact?.gguf?.quant
      ?? null,
  );
  // Tensor-directory pre-flight: architectures that share attention value
  // projections across layers (Gemma 4 omits blk.N.attn_v.weight on V-share
  // layers) parse as a supported family but cannot execute on the native
  // worker yet. Blocking here routes the model to llama.cpp automatically
  // instead of failing the native load mid-run.
  const attentionValueGaps = gguf?.attentionValueGaps
    ?? profile?.gguf?.attentionValueGaps
    ?? modelArtifact?.gguf?.attentionValueGaps
    ?? null;
  const tensors = attentionValueGaps && attentionValueGaps.missingLayerCount > 0
    ? {
      key: "cross-layer-value-sharing",
      status: "blocked",
      reason: `This GGUF shares attention value projections across layers: attn_v.weight is absent on `
        + `${attentionValueGaps.missingLayerCount} of ${attentionValueGaps.totalLayers} layers `
        + `(e.g. ${attentionValueGaps.missingLayers.slice(0, 4).join(", ")}). `
        + `The native executor does not implement cross-layer value sharing yet.`,
    }
    : { key: null, status: "supported", reason: null };
  const familyAcceptsQuantization = quantization.status !== "supported"
    || !Array.isArray(quantization.architectures)
    || quantization.architectures.includes(familyKey);
  const effectiveQuantization = quantization.status === "supported" && !familyAcceptsQuantization
    ? {
      ...quantization,
      status: "blocked",
      reason: `${quantization.key} is implemented, but not for the ${familyKey} native graph. Choose one of the family matrix's supported quantizations.`,
    }
    : quantization;
  const supported = architecture.status === "supported"
    && effectiveQuantization.status === "supported"
    && tensors.status === "supported";
  const reasons = [
    architecture.status === "supported" ? null : architecture.reason,
    effectiveQuantization.status === "supported" ? null : effectiveQuantization.reason,
    tensors.status === "supported" ? null : tensors.reason,
  ].filter(Boolean);
  return {
    supported,
    architecture,
    quantization: effectiveQuantization,
    tensors,
    reason: reasons.join(" "),
  };
}

export function estimatePrototypeModelBytes({
  embeddings = [],
  outputWeights = [],
  bias = [],
  bytesPerElement = 4,
} = {}) {
  const embeddingElements = embeddings.reduce((sum, row) => sum + row.length, 0);
  const outputElements = outputWeights.reduce((sum, row) => sum + row.length, 0);
  const biasElements = Array.isArray(bias) ? bias.length : 0;
  return Math.trunc((embeddingElements + outputElements + biasElements) * bytesPerElement);
}

export function evaluateNativeCpuPrototypeSupport({
  model,
  profile = null,
  promptArtifact = null,
  kvPlan = null,
} = {}) {
  const architecture = getArchitectureCompatibility(model?.architecture);
  const quantization = getQuantizationCompatibility(model?.quantization);
  const availableGb = profile?.capabilityRegistry?.memory?.availableGb ?? null;
  const availableBytes = Number.isFinite(availableGb) && availableGb > 0
    ? Math.trunc(availableGb * GIB)
    : null;
  const budgetBytes = Number.isFinite(availableBytes)
    ? Math.trunc(availableBytes * NATIVE_CPU_MEMORY_HEADROOM_RATIO)
    : null;
  const requiredBytes = Math.trunc((model?.estimatedModelBytes ?? 0) + (kvPlan?.estimatedBytes ?? 0));
  const memoryFits = budgetBytes == null ? null : requiredBytes <= budgetBytes;
  const memory = {
    availableBytes,
    budgetBytes,
    requiredBytes,
    fits: memoryFits,
    reason: memoryFits === false
      ? `Native CPU prototype requires ${formatBytes(requiredBytes)} but only ${formatBytes(budgetBytes)} is budgeted.`
      : memoryFits === true
        ? `Native CPU prototype requires ${formatBytes(requiredBytes)} within a ${formatBytes(budgetBytes)} budget.`
        : "Native CPU prototype memory budget could not be determined from the profile.",
  };

  const profileContextLimit = Number.isFinite(Number(profile?.contextSize)) && Number(profile.contextSize) > 0
    ? Math.floor(Number(profile.contextSize))
    : null;
  const modelContextLimit = Number.isFinite(Number(model?.maxContextTokens)) && Number(model.maxContextTokens) > 0
    ? Math.floor(Number(model.maxContextTokens))
    : null;
  const promptTokens = promptArtifact?.tokenCount ?? null;
  const contextFits = modelContextLimit == null || promptTokens == null
    ? null
    : promptTokens <= modelContextLimit;
  const truncatedByProfile = profileContextLimit != null && promptTokens != null
    ? promptTokens > profileContextLimit
    : null;
  const context = {
    promptTokens,
    profileContextLimit,
    modelContextLimit,
    effectiveContextLimit: kvPlan?.effectiveTokens ?? profileContextLimit ?? modelContextLimit,
    truncatedByProfile,
    fits: contextFits,
    reason: contextFits === false
      ? `Prompt requires ${promptTokens} tokens but the native CPU prototype supports only ${modelContextLimit} tokens for this model.`
      : contextFits === true
        ? truncatedByProfile
          ? `Prompt exceeds the configured runtime context of ${profileContextLimit} tokens, so the native CPU prototype will use the most recent ${kvPlan?.effectiveTokens ?? profileContextLimit} tokens.`
          : `Prompt fits within the native CPU prototype context limit of ${modelContextLimit ?? profileContextLimit} tokens.`
        : "Native CPU prototype context fit could not be determined from the profile or prompt artifact.",
  };

  const supported = architecture.status === "supported"
    && quantization.status === "supported"
    && memory.fits !== false
    && context.fits !== false;

  return {
    supported,
    backend: "cpu",
    architecture,
    quantization,
    memory,
    context,
  };
}
