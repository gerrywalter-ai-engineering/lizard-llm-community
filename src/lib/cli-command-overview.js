// Command reference for `lizard help` and `lizard commands`.
// Lizard LLM Community Edition: Ollama-style verbs first, the detailed
// commands they map onto underneath.

export const CLI_COMMANDS = [
  { name: "pull", usage: "lizard pull <owner/repo> [file.gguf] [--quant Q4_K_M] [--activate]", summary: "Download a GGUF model from Hugging Face and verify it." },
  { name: "pull ollama:", usage: "lizard pull ollama:<tag>", summary: "Import a model Ollama already has on this machine (no second download)." },
  { name: "run", usage: "lizard run [model] [prompt] [--engine lizard-native|caterpillar] [--system TEXT]", summary: "Chat in the terminal, or answer one prompt and exit." },
  { name: "list", usage: "lizard list", summary: "Models on this machine. Alias: ls." },
  { name: "ps", usage: "lizard ps", summary: "The model that is loaded right now." },
  { name: "stop", usage: "lizard stop [--yes]", summary: "Unload the loaded model." },
  { name: "rm", usage: "lizard rm <model> [--force]", summary: "Delete a downloaded model." },
  { name: "show", usage: "lizard show [model]", summary: "Details of a model: source, file, quantization, size." },
  { name: "serve", usage: "lizard serve [model] [--engine lizard-native|caterpillar] [--port N] [--cpu-only]", summary: "OpenAI- and Ollama-compatible API on 127.0.0.1 (default port 8013, caterpillar 8014)." },
  { name: "app", usage: "lizard app [--port 7817] [--no-open]", summary: "Open the guided onboarding (Scan, Select, Download, Activate, Chat) in the browser." },
  { name: "scan", usage: "lizard scan [--json]", summary: "Profile CPU, RAM, GPU and VRAM." },
  { name: "recommend", usage: "lizard recommend [--task chat|code|reasoning] [--mode fast|balanced|quality] [--json]", summary: "Recommend a model, context size and offload plan for this machine." },
  { name: "models hf-search", usage: "lizard models hf-search [--task TASK] [--mode MODE] [--json]", summary: "Curated GGUF models that fit this machine." },
  { name: "models activate", usage: "lizard models activate --model MODEL_ID", summary: "Make a downloaded model the active one." },
  { name: "generate", usage: "lizard generate --prompt TEXT [--model MODEL_ID] [--engine lizard-native|caterpillar] [--max-tokens N] [--json]", summary: "One completion without starting a server." },
  { name: "engines", usage: "lizard engines", summary: "The two open-source inference engines." },
  { name: "pro", usage: "lizard pro", summary: "What the paid Lizard LLM app adds (EUR 5, one-time)." },
  { name: "version", usage: "lizard version", summary: "Print the version." },
];

export const CLI_CONSOLE_COMMANDS = [];

export function formatCliCommandList() {
  const width = Math.max(...CLI_COMMANDS.map((command) => command.name.length));
  return CLI_COMMANDS.map((command) => `  ${command.name.padEnd(width)}  ${command.summary}`).join("\n");
}

export function formatCliHelp() {
  return [
    "Lizard LLM Community Edition - run local LLMs on your own hardware, fully offline.",
    "",
    "Usage:",
    "  lizard <command> [options]",
    "",
    "Commands:",
    formatCliCommandList(),
    "",
    "Examples:",
    "  lizard pull Qwen/Qwen2.5-0.5B-Instruct-GGUF",
    "  lizard run",
    "  lizard run --engine caterpillar \"Explain KV caching in two sentences\"",
    "  lizard serve            # then POST http://127.0.0.1:8013/v1/chat/completions",
    "  lizard app              # guided setup in the browser",
    "",
    "Details for one command: lizard commands",
  ].join("\n");
}
