// Ollama-style command surface for Lizard LLM Community Edition.
//
//   lizard pull <hf-repo> [file.gguf]     download a GGUF from Hugging Face
//   lizard pull ollama:<tag>              import a model Ollama already has
//   lizard run [model] [prompt]           chat in the terminal (or one-shot)
//   lizard list | ls                      models on this machine
//   lizard ps                             model loaded right now
//   lizard stop                           unload it
//   lizard rm <model>                     delete a downloaded model
//   lizard show <model>                   details of one model
//   lizard serve [--engine caterpillar]   OpenAI- and Ollama-compatible API
//   lizard app                            open the onboarding app in the browser
//   lizard engines                        the two open-source inference engines
//   lizard pro                            what the paid app adds
//
// Verbs that map onto an existing command are rewritten in place
// (translateOllamaStyleArgv); the rest are handled here
// (runOllamaStyleCommand). Everything else falls through to src/cli.js.

import process from "node:process";
import path from "node:path";
import { spawn } from "node:child_process";
import { createInterface } from "node:readline/promises";
import { fileURLToPath } from "node:url";
import { PRO_OFFER } from "./edition.js";

const CLI_PATH = path.join(path.dirname(fileURLToPath(import.meta.url)), "..", "cli.js");
const ENGINE_ALIASES = new Map([
  ["lizard", "lizard-native"],
  ["native", "lizard-native"],
  ["lizard-native", "lizard-native"],
  ["caterpillar", "caterpillar"],
]);

function splitArgs(argv) {
  const positional = [];
  const flags = [];
  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];
    if (token.startsWith("--")) {
      flags.push(token);
      const next = argv[index + 1];
      if (next !== undefined && !next.startsWith("--")) {
        flags.push(next);
        index += 1;
      }
    } else {
      positional.push(token);
    }
  }
  return { positional, flags };
}

function flagValue(argv, name) {
  const index = argv.indexOf(`--${name}`);
  if (index < 0) return undefined;
  const next = argv[index + 1];
  return next === undefined || next.startsWith("--") ? true : next;
}

function normalizeEngine(value) {
  if (value == null || value === true) return null;
  const engine = ENGINE_ALIASES.get(String(value).trim().toLowerCase());
  if (!engine) {
    throw new Error(`Unknown engine "${value}". Use lizard-native or caterpillar.`);
  }
  return engine;
}

// `--engine` is the friendly spelling; the core commands call it --provider.
function engineFlagsToProvider(flags) {
  const out = [];
  for (let index = 0; index < flags.length; index += 1) {
    if (flags[index] === "--engine") {
      out.push("--provider", normalizeEngine(flags[index + 1]));
      index += 1;
    } else {
      out.push(flags[index]);
    }
  }
  return out;
}

export function translateOllamaStyleArgv(argv) {
  const [verb, ...rest] = argv;
  const { positional, flags } = splitArgs(rest);
  switch (verb) {
    case "list":
    case "ls":
      return ["models", "list", "--source", "local", ...flags];
    case "ps":
      return ["model", "status", ...flags];
    case "stop":
      return ["model", "stop", ...flags];
    case "app":
    case "ui":
      return ["serve-ui", ...(flags.includes("--no-open") ? flags.filter((f) => f !== "--no-open") : ["--open", ...flags])];
    case "serve": {
      const model = positional[0];
      return ["serve", ...(model ? ["--model", model] : []), ...engineFlagsToProvider(flags)];
    }
    case "pull": {
      const target = positional[0];
      if (!target) return argv;
      if (target.startsWith("ollama:")) {
        return ["models", "pull", "--ollama", target.slice("ollama:".length), ...flags];
      }
      return argv; // Hugging Face pulls are resolved in runOllamaStyleCommand.
    }
    case "version":
    case "--version":
    case "-v":
      return ["version"];
    default:
      return argv;
  }
}

async function loadCore() {
  const [{ loadConfig }, downloads, sources, registryModule] = await Promise.all([
    import("./config.js"),
    import("./download-manager.js"),
    import("./model-sources.js"),
    import("./model-registry.js"),
  ]);
  return {
    config: loadConfig(),
    downloads,
    sources,
    registry: registryModule.loadModelRegistry(),
  };
}

function formatBytes(bytes) {
  const value = Number(bytes);
  if (!Number.isFinite(value) || value <= 0) return "-";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let size = value;
  let unit = 0;
  while (size >= 1024 && unit < units.length - 1) {
    size /= 1024;
    unit += 1;
  }
  return `${size.toFixed(unit >= 3 ? 1 : 0)} ${units[unit]}`;
}

async function pullFromHuggingFace(positional, argv) {
  const [repo, explicitFile] = positional;
  if (!repo || !repo.includes("/")) {
    throw new Error("Usage: lizard pull <owner/repo> [file.gguf]   (or: lizard pull ollama:<tag>)");
  }
  const { config } = await loadCore();
  const { getHubModelDetail } = await import("./hf-hub.js");
  const { runHfDownloadWorkflow } = await import("./hf-workflow.js");
  const { activateModel, getActiveModel } = await import("./download-manager.js");
  const token = flagValue(argv, "token") ?? process.env.HF_TOKEN ?? undefined;

  let filename = explicitFile;
  if (!filename) {
    const detail = await getHubModelDetail(repo, { token });
    const files = detail.ggufFiles ?? [];
    if (!files.length) {
      throw new Error(`${repo} has no downloadable GGUF file.`);
    }
    const wantedQuant = String(flagValue(argv, "quant") ?? "Q4_K_M").toUpperCase();
    const pick = files.find((file) => String(file.quant ?? "").toUpperCase() === wantedQuant)
      ?? files.find((file) => /Q4_K/i.test(file.quant ?? ""))
      ?? files[0];
    filename = pick.filename;
    console.log(`pulling ${repo} -> ${filename} (${formatBytes(pick.sizeBytes)})`);
  }

  let lastPercent = -1;
  const workflow = await runHfDownloadWorkflow({
    repo,
    filename,
    modelId: flagValue(argv, "alias") === true ? undefined : flagValue(argv, "alias"),
    token,
    config,
    logger: null,
    onPhase: (phase) => {
      if (lastPercent >= 0) { process.stdout.write("\n"); lastPercent = -1; }
      console.log(`${phase.message}...`);
    },
    onProgress: (progress) => {
      const percent = Math.floor(Number(progress?.ratio) * 100);
      if (Number.isFinite(percent) && percent !== lastPercent && percent % 5 === 0) {
        lastPercent = percent;
        process.stdout.write(`\r  ${String(percent).padStart(3)}%`);
      }
    },
  });
  const id = workflow.record.id;
  console.log(`success: ${id}`);
  if (argv.includes("--activate") || !getActiveModel(config)?.id) {
    activateModel({ config, modelId: id });
    console.log(`${id} is now the active model. Try: lizard run`);
  } else {
    console.log(`Run it with: lizard run ${id}`);
  }
}

async function showModel(positional) {
  const { config, downloads } = await loadCore();
  const id = positional[0] ?? downloads.getActiveModel(config)?.id;
  if (!id) throw new Error("Usage: lizard show <model>");
  const entry = downloads.findCatalogModel(config, id);
  if (!entry) throw new Error(`Model not found: ${id}. See: lizard list`);
  const active = downloads.getActiveModel(config)?.id === entry.id;
  const gguf = entry.gguf ?? {};
  const rows = [
    ["model", entry.id],
    ["active", active ? "yes" : "no"],
    ["source", entry.hfRepo ? `huggingface.co/${entry.hfRepo}` : (entry.ollamaTag ? `ollama:${entry.ollamaTag}` : "local")],
    ["file", entry.filename ?? path.basename(entry.importedPath ?? "")],
    ["path", entry.importedPath ?? "-"],
    ["quantization", entry.quant ?? gguf.quant ?? "-"],
    ["architecture", gguf.architecture ?? "-"],
    ["parameters", entry.paramsB ? `${entry.paramsB}B` : "-"],
    ["context length", gguf.contextLength ?? "-"],
    ["license", entry.license ?? "-"],
    ["sha256", entry.sha256 ?? "-"],
  ];
  const width = Math.max(...rows.map(([key]) => key.length));
  for (const [key, value] of rows) console.log(`  ${key.padEnd(width)}  ${value}`);
}

async function removeModel(positional, argv) {
  const id = positional[0];
  if (!id) throw new Error("Usage: lizard rm <model>");
  const { config, downloads } = await loadCore();
  const { getCurrentLoadedModel, stopCurrentModel } = await import("./model-lifecycle.js");
  const loaded = getCurrentLoadedModel(config);
  if (loaded?.modelId && String(loaded.modelId).toLowerCase() === id.toLowerCase()) {
    if (!argv.includes("--force")) {
      throw new Error(`${id} is loaded right now. Run "lizard stop" first, or pass --force.`);
    }
    await stopCurrentModel(config);
  }
  const result = downloads.deleteModel({ config, modelId: id });
  console.log(`deleted '${result.id}'${result.removedFile ? "" : " (catalog entry only, the file was already gone)"}`);
}

function printEngines() {
  console.log(`Lizard LLM Community Edition ships two open-source inference engines.
Both run GGUF models fully offline; pick one with --engine.

  lizard-native  (default)  Resident-GPU engine on Direct3D 12. Weights stay on the
                            GPU between calls; fused kernels, a zero-copy path for
                            integrated GPUs and an on-disk shader cache. Falls back to
                            a pinned ggml CPU backend.
  caterpillar               Standalone engine built around a pre-compiled compute
                            graph plan. Optimised Q4_K decode, speculative decoding
                            and its own shader cache.

Build them with: npm run build:engines`);
}

function printPro() {
  console.log(`Lizard LLM Pro - ${PRO_OFFER.price}, ${PRO_OFFER.billing}

The Community Edition is complete on its own: onboarding, chat, both engines
and the local API are free and open source. The paid app adds:
${PRO_OFFER.includes.map((line) => `  - ${line}`).join("\n")}

Download: ${PRO_OFFER.downloadUrl}`);
}

// Interactive chat. Starts the engine's API server in the background, talks to
// it over the same OpenAI-compatible endpoint any client uses, and unloads the
// model when the session ends.
async function runChat(positional, argv) {
  const { config, downloads } = await loadCore();
  const [modelArg, ...promptWords] = positional;
  let modelId = modelArg;
  if (modelId && !downloads.findCatalogModel(config, modelId)) {
    // `lizard run "a question"` with no model: treat it as the prompt.
    promptWords.unshift(modelId);
    modelId = undefined;
  }
  modelId = modelId ?? downloads.getActiveModel(config)?.id;
  if (!modelId) {
    throw new Error("No model yet. Pull one first, for example:\n  lizard pull bartowski/Qwen2.5-1.5B-Instruct-GGUF\nor open the guided setup: lizard app");
  }
  const engine = normalizeEngine(flagValue(argv, "engine")) ?? "lizard-native";
  const portDefault = engine === "caterpillar" ? config.engine?.caterpillar?.port ?? 8014 : config.engine?.native?.port ?? 8013;
  const port = Number.parseInt(flagValue(argv, "port") ?? portDefault, 10);
  const base = `http://127.0.0.1:${port}`;
  const oneShot = promptWords.join(" ").trim();

  const serverArgs = [CLI_PATH, "serve", "--model", modelId, "--provider", engine, "--port", String(port)];
  if (argv.includes("--cpu-only")) serverArgs.push("--cpu-only");
  const server = spawn(process.execPath, serverArgs, { stdio: ["ignore", "pipe", "pipe"], windowsHide: true });
  let serverLog = "";
  const keepTail = (chunk) => { serverLog = (serverLog + String(chunk)).slice(-4000); };
  server.stdout.on("data", keepTail);
  server.stderr.on("data", keepTail);
  let serverExited = false;
  server.once("exit", () => { serverExited = true; });

  const stopServer = async () => {
    if (serverExited) return;
    try {
      const { stopCurrentModel } = await import("./model-lifecycle.js");
      await stopCurrentModel(config);
    } catch {
      // Fall through to stopping the serve process itself.
    }
    if (!serverExited) server.kill();
  };

  process.stderr.write(`loading ${modelId} on ${engine}...`);
  const deadline = Date.now() + 10 * 60 * 1000;
  let ready = false;
  while (!ready && !serverExited && Date.now() < deadline) {
    try {
      const health = await fetch(`${base}/health`, { signal: AbortSignal.timeout(1500) });
      if (health.ok) {
        const body = await health.json().catch(() => ({}));
        if (body?.warmLoadState === "failed" || body?.warmLoadError) {
          throw Object.assign(new Error(String(body.warmLoadError ?? "model load failed")), { fatal: true });
        }
        ready = body?.modelReady === true;
      }
    } catch (error) {
      if (error?.fatal) {
        await stopServer();
        throw new Error(`The ${engine} engine could not load ${modelId}: ${error.message}`);
      }
    }
    if (!ready) await new Promise((resolve) => setTimeout(resolve, 500));
  }
  process.stderr.write("\r\x1b[K");
  if (!ready) {
    await stopServer();
    throw new Error(`The ${engine} engine did not start.${serverLog.trim() ? `\n${serverLog.trim()}` : ""}`);
  }

  const messages = [];
  const system = flagValue(argv, "system");
  if (typeof system === "string") messages.push({ role: "system", content: system });

  const ask = async (text) => {
    messages.push({ role: "user", content: text });
    const response = await fetch(`${base}/v1/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: modelId, messages, stream: true }),
    });
    if (!response.ok || !response.body) {
      throw new Error(`The engine answered HTTP ${response.status}: ${(await response.text()).slice(0, 300)}`);
    }
    let answer = "";
    let buffer = "";
    const decoder = new TextDecoder();
    for await (const chunk of response.body) {
      buffer += decoder.decode(chunk, { stream: true });
      let newline;
      while ((newline = buffer.indexOf("\n")) >= 0) {
        const line = buffer.slice(0, newline).trim();
        buffer = buffer.slice(newline + 1);
        if (!line.startsWith("data:")) continue;
        const data = line.slice(5).trim();
        if (data === "[DONE]") continue;
        try {
          const delta = JSON.parse(data)?.choices?.[0]?.delta?.content ?? "";
          if (delta) {
            answer += delta;
            process.stdout.write(delta);
          }
        } catch {
          // ignore keep-alive or partial lines
        }
      }
    }
    process.stdout.write("\n");
    messages.push({ role: "assistant", content: answer });
  };

  const onSignal = () => { stopServer().finally(() => process.exit(130)); };
  process.once("SIGINT", onSignal);
  try {
    if (oneShot) {
      await ask(oneShot);
      return;
    }
    console.log(`>>> ${modelId} on ${engine}. Type /bye to quit, /clear to forget the conversation.`);
    const rl = createInterface({ input: process.stdin, output: process.stdout, prompt: ">>> " });
    rl.prompt();
    for await (const line of rl) {
      const text = line.trim();
      if (text === "/bye" || text === "/exit" || text === "/quit") break;
      if (text === "/clear") {
        messages.splice(system ? 1 : 0);
        console.log("Conversation cleared.");
      } else if (text === "/?" || text === "/help") {
        console.log("/clear  forget the conversation\n/bye    quit and unload the model");
      } else if (text) {
        await ask(text);
      }
      rl.prompt();
    }
    rl.close();
  } finally {
    process.removeListener("SIGINT", onSignal);
    await stopServer();
  }
}

async function printVersion() {
  const fs = await import("node:fs");
  const pkg = JSON.parse(fs.readFileSync(path.join(path.dirname(CLI_PATH), "..", "package.json"), "utf8"));
  console.log(`lizard ${pkg.version} (Community Edition)`);
}

// Returns true when the verb was handled here.
export async function runOllamaStyleCommand(argv) {
  const [verb, ...rest] = argv;
  const { positional } = splitArgs(rest);
  switch (verb) {
    case "run":
      await runChat(positional, rest);
      return true;
    case "pull":
      if (positional[0]?.startsWith("ollama:")) return false;
      await pullFromHuggingFace(positional, rest);
      return true;
    case "rm":
    case "remove":
      await removeModel(positional, rest);
      return true;
    case "show":
      await showModel(positional);
      return true;
    case "engines":
      printEngines();
      return true;
    case "pro":
      printPro();
      return true;
    case "version":
      await printVersion();
      return true;
    default:
      return false;
  }
}
