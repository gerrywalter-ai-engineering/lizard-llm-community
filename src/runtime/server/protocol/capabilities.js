export const CATERPILLAR_PROTOCOL_SPEC_VERSION = "1.0";
export const CATERPILLAR_WORKER_RPC_VERSION = 1;

const CURRENT_ROUTES = Object.freeze([
  Object.freeze({ method: "GET", path: "/health", profile: "legacy-lizard" }),
  Object.freeze({ method: "GET", path: "/v1/health", profile: "CAT-BASE-1" }),
  Object.freeze({ method: "GET", path: "/v1/models", profile: "legacy-openai" }),
  Object.freeze({ method: "GET", path: "/metrics", profile: "legacy-lizard" }),
  Object.freeze({ method: "GET", path: "/metrics/history", profile: "legacy-lizard" }),
  Object.freeze({ method: "GET", path: "/capabilities", profile: "discovery" }),
  Object.freeze({ method: "POST", path: "/apply-template", profile: "CAT-LLAMA-SERVER-1" }),
  Object.freeze({ method: "POST", path: "/tokenize", profile: "CAT-LLAMA-SERVER-1" }),
  Object.freeze({ method: "POST", path: "/detokenize", profile: "CAT-LLAMA-SERVER-1" }),
  Object.freeze({ method: "POST", path: "/completion", profile: "CAT-LLAMA-SERVER-1" }),
  Object.freeze({ method: "POST", path: "/v1/completions", profile: "CAT-LLAMA-SERVER-1" }),
  Object.freeze({ method: "POST", path: "/v1/chat/completions", profile: "CAT-OAI-CHAT-1" }),
  Object.freeze({ method: "GET", path: "/api/version", profile: "CAT-OLLAMA-CHAT-1" }),
  Object.freeze({ method: "GET", path: "/api/tags", profile: "CAT-OLLAMA-CHAT-1" }),
  Object.freeze({ method: "GET", path: "/api/ps", profile: "CAT-OLLAMA-CHAT-1" }),
  Object.freeze({ method: "POST", path: "/api/generate", profile: "CAT-OLLAMA-CHAT-1" }),
  Object.freeze({ method: "POST", path: "/api/chat", profile: "CAT-OLLAMA-CHAT-1" }),
  Object.freeze({ method: "POST", path: "/v1/responses", profile: "CAT-RESPONSES-1" }),
]);

const CATERPILLAR_LIMITATIONS = Object.freeze([
  "deterministic-greedy-sampling-only",
  "text-content-only",
  "single-bound-model",
  "tools-not-yet-conformant",
  "structured-output-not-yet-conformant",
  "responses-api-stateless-text-only",
  "slots-administration-not-supported",
  "lora-mutation-not-supported",
  "model-router-autoload-not-supported",
  "embeddings-reranking-infill-not-supported",
  "prometheus-metrics-not-supported-json-metrics-only",
]);

export function buildProtocolCapabilities({
  runtimeProvider = "lizard-native",
  modelArtifact = null,
  modelReady = false,
  productVersion = null,
  buildCommit = null,
  constraints = {},
} = {}) {
  const caterpillar = runtimeProvider === "caterpillar";
  const modelAliases = [...new Set([
    ...(Array.isArray(modelArtifact?.aliases) ? modelArtifact.aliases : []),
    ...(Array.isArray(modelArtifact?.modelAliases) ? modelArtifact.modelAliases : []),
  ].filter((value) => typeof value === "string" && value.length > 0))];
  return {
    object: "lizard.protocol.capabilities",
    version: CATERPILLAR_PROTOCOL_SPEC_VERSION,
    product: {
      name: "lizard-llm",
      version: productVersion,
      buildCommit,
    },
    runtime: {
      provider: runtimeProvider,
      modelId: modelArtifact?.modelId ?? null,
      modelAliases,
      modelReady: Boolean(modelReady),
      samplingModes: caterpillar ? ["greedy"] : ["greedy", "sampling"],
    },
    profiles: {
      "CAT-BASE-1": caterpillar ? "supported" : "not-applicable",
      "CAT-OAI-CHAT-1": caterpillar ? "supported" : "not-applicable",
      "CAT-LLAMA-SERVER-1": caterpillar ? "supported" : "not-applicable",
      "CAT-OLLAMA-CHAT-1": caterpillar ? "supported" : "not-applicable",
      "CAT-STRUCTURED-1": "unsupported",
      "CAT-TOOLS-1": "unsupported",
      "CAT-RESPONSES-1": caterpillar ? "supported" : "not-applicable",
    },
    routes: CURRENT_ROUTES.map((route) => ({ ...route })),
    constraints: {
      maxJsonBodyBytes: 1024 * 1024,
      jsonBodyTimeoutMs: 30000,
      maxJsonDepth: 32,
      maxMessages: 256,
      maxMessageContentBytes: 512 * 1024,
      maxStopSequences: 16,
      maxStopBytes: 16 * 1024,
      maxOutputTokens: 4096,
      ...constraints,
    },
    requestFields: {
      "/v1/chat/completions": {
        supported: [
          "model", "messages", "stream", "stream_options", "max_tokens", "max_completion_tokens",
          "seed", "stop", "user",
        ],
        defaults: { stream: false, stream_options: { include_usage: false } },
        defaultOnly: {
          temperature: 0,
          top_p: 1,
          top_k: 1,
          n: 1,
          frequency_penalty: 0,
          presence_penalty: 0,
          logprobs: false,
          response_format: { type: "text" },
          tools: [],
          tool_choice: "none",
          parallel_tool_calls: false,
        },
        aliases: { max_completion_tokens: "max_tokens" },
        conflicts: [["max_tokens", "max_completion_tokens"]],
        streamOptions: { supported: ["include_usage"], requires: { stream: true } },
        unsupported: ["audio", "modalities", "prediction", "reasoning_effort", "service_tier"],
      },
      "/tokenize": {
        supported: ["content", "add_special", "parse_special", "with_pieces"],
        defaults: { add_special: false, parse_special: true, with_pieces: false },
      },
      "/detokenize": { supported: ["tokens"] },
      "/completion": {
        supported: [
          "prompt", "n_predict", "stream", "stop", "seed", "cache_prompt", "return_tokens",
        ],
        defaultOnly: { temperature: 0, top_k: 1, top_p: 1, n_probs: 0, ignore_eos: false },
      },
      "/v1/completions": {
        supported: ["model", "prompt", "max_tokens", "stream", "stop", "seed", "user"],
        defaultOnly: {
          temperature: 0, top_p: 1, n: 1, logprobs: false, echo: false, suffix: "", best_of: 1,
        },
      },
      "/api/generate": {
        supported: ["model", "prompt", "system", "stream", "options"],
        defaults: { stream: true },
        options: {
          supported: ["num_predict", "stop"],
          defaultOnly: { temperature: 0, top_p: 1, top_k: 1 },
        },
        unsupported: ["keep_alive", "format", "raw", "template", "suffix", "images", "think", "logprobs"],
      },
      "/api/chat": {
        supported: ["model", "messages", "stream", "options"],
        defaults: { stream: true },
        defaultOnly: { tools: [] },
        options: {
          supported: ["num_predict", "stop"],
          defaultOnly: { temperature: 0, top_p: 1, top_k: 1 },
        },
        unsupported: ["keep_alive", "format", "think", "images", "logprobs"],
      },
      "/v1/responses": {
        supported: ["model", "input", "instructions", "max_output_tokens", "stream", "metadata", "user"],
        defaults: { stream: false, store: false },
        defaultOnly: {
          temperature: 0,
          top_p: 1,
          text: { format: { type: "text" }, verbosity: "medium" },
          tools: [],
          tool_choice: "none",
          parallel_tool_calls: false,
          include: [],
          background: false,
          truncation: "disabled",
        },
        unsupported: ["previous_response_id", "conversation", "reasoning", "max_tool_calls"],
      },
    },
    streaming: {
      openai: caterpillar ? "supported" : "text/event-stream",
      llamaCompletion: caterpillar ? "supported" : "text/event-stream",
      ollama: caterpillar ? "supported-ndjson" : "application/x-ndjson",
      responses: caterpillar ? "supported-typed-sse" : "text/event-stream",
      backpressure: "bounded",
    },
    extensionFieldPolicy: "reject-unknown-request-fields",
    excludedRoutes: caterpillar ? [
      "/slots", "/lora-adapters", "/models/load", "/models/unload", "/v1/embeddings",
      "/reranking", "/infill", "/v1/messages",
      "/api/create", "/api/copy", "/api/delete", "/api/pull", "/api/push", "/api/show",
      "/api/embed", "/api/embeddings", "/api/blobs/:digest", "/api/cloud", "/api/search", "/api/import",
    ] : [],
    metrics: { route: "/metrics", format: "lizard-json", prometheus: false },
    workerRpc: caterpillar
      ? {
        minVersion: CATERPILLAR_WORKER_RPC_VERSION,
        maxVersion: CATERPILLAR_WORKER_RPC_VERSION,
        selectedVersion: CATERPILLAR_WORKER_RPC_VERSION,
      }
      : null,
    limitations: caterpillar ? [...CATERPILLAR_LIMITATIONS] : [],
  };
}

function markdownCell(value) {
  return String(value ?? "-").replaceAll("|", "\\|").replaceAll("\n", " ");
}

function fieldList(value) {
  if (Array.isArray(value)) return value.length ? value.join(", ") : "-";
  if (!value || typeof value !== "object") return "-";
  return Object.keys(value).length ? Object.keys(value).join(", ") : "-";
}

export function renderProtocolCompatibilityMarkdown(capabilities) {
  const lines = [
    "# Caterpillar Protocol Compatibility Matrix",
    "",
    "> Generated from `buildProtocolCapabilities()` by",
    "> `scripts/write-caterpillar-compatibility-matrix.mjs`. Do not edit by hand.",
    "",
    `Protocol specification version: \`${markdownCell(capabilities?.version)}\`.`,
    "",
    "## Profiles",
    "",
    "| Profile | Status |",
    "| --- | --- |",
  ];
  for (const [profile, status] of Object.entries(capabilities?.profiles ?? {})) {
    lines.push(`| \`${markdownCell(profile)}\` | ${markdownCell(status)} |`);
  }
  lines.push("", "## Routes", "", "| Method | Path | Profile |", "| --- | --- | --- |");
  for (const route of capabilities?.routes ?? []) {
    lines.push(`| ${markdownCell(route.method)} | \`${markdownCell(route.path)}\` | \`${markdownCell(route.profile)}\` |`);
  }
  lines.push(
    "", "## Request fields", "",
    "| Route | Supported | Default-only | Explicitly unsupported |",
    "| --- | --- | --- | --- |",
  );
  for (const [route, fields] of Object.entries(capabilities?.requestFields ?? {})) {
    lines.push(
      `| \`${markdownCell(route)}\` | ${markdownCell(fieldList(fields.supported))} `
      + `| ${markdownCell(fieldList(fields.defaultOnly))} | ${markdownCell(fieldList(fields.unsupported))} |`,
    );
  }
  lines.push(
    "", "## Streaming and exclusions", "",
    `- OpenAI Chat SSE: ${markdownCell(capabilities?.streaming?.openai)}.`,
    `- llama completion SSE: ${markdownCell(capabilities?.streaming?.llamaCompletion)}.`,
    `- Ollama NDJSON: ${markdownCell(capabilities?.streaming?.ollama)}.`,
    `- Responses typed SSE: ${markdownCell(capabilities?.streaming?.responses)}.`,
    `- Backpressure: ${markdownCell(capabilities?.streaming?.backpressure)}.`,
    `- Extension policy: ${markdownCell(capabilities?.extensionFieldPolicy)}.`,
    `- Excluded routes: ${(capabilities?.excludedRoutes ?? []).map((route) => `\`${markdownCell(route)}\``).join(", ") || "none"}.`,
    "", "## Declared constraints", "",
    "| Constraint | Value |", "| --- | ---: |",
  );
  for (const [name, value] of Object.entries(capabilities?.constraints ?? {})) {
    lines.push(`| ${markdownCell(name)} | ${markdownCell(value)} |`);
  }
  lines.push("");
  return lines.join("\n");
}
