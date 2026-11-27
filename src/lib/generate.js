import { spawn } from "node:child_process";
import path from "node:path";
import { resolveCliExecutable, resolveCompletionExecutable } from "./engine-adapter.js";
import { requestOllamaGenerate } from "./ollama.js";
import { buildLlamaCppArgs } from "./execution-profile.js";
import { buildRuntimeContractFields, inferRuntimeProvider } from "../runtime/contract.js";

function normalizeOutput(rawOutput, prompt) {
  let text = rawOutput.replace(/\r/g, "").trim();
  const promptMarker = `> ${prompt}`;
  const promptIndex = text.indexOf(promptMarker);
  if (promptIndex >= 0) {
    text = text.slice(promptIndex + promptMarker.length).trim();
  }
  text = text.replace(/\n*\[ Prompt:[\s\S]*$/m, "").trim();
  text = text.replace(/\n*Exiting\.\.\.\s*$/m, "").trim();
  return text;
}

function isLocalLoadFailure(message) {
  return /failed to load the model|wrong number of tensors|failed to load model|llama_model_load/i.test(String(message ?? ""));
}

// llama.cpp prints its own "[end of text]" sentinel (not an empty string)
// when generation stops before producing any real content - which is exactly
// what a misclassified special/EOG token produces. A bare-empty-string check
// missed this, so the completion->cli retry below silently never fired for
// this failure shape.
export function isEmptyCompletionOutput(output) {
  const normalized = String(output ?? "").trim().toLowerCase();
  return normalized === "" || normalized === "[end of text]";
}

function shouldRetryEmptyCompletionOutput(execution, executable) {
  if (!execution) {
    return false;
  }
  const runtimeName = String(execution.runtime ?? "").toLowerCase();
  const executableName = path.basename(executable).toLowerCase();
  if (runtimeName !== "llama-completion" && executableName.includes("completion") === false) {
    return false;
  }
  return isEmptyCompletionOutput(execution.output);
}

async function runLocalGeneration({ executable, args, prompt }) {
  const startedAt = Date.now();
  const stdoutChunks = [];
  const stderrChunks = [];
  const executableName = path.basename(executable).toLowerCase();
  await new Promise((resolve, reject) => {
    const child = spawn(executable, args, { stdio: ["ignore", "pipe", "pipe"] });
    child.stdout.on("data", (chunk) => stdoutChunks.push(chunk));
    child.stderr.on("data", (chunk) => stderrChunks.push(chunk));
    child.on("error", reject);
    child.on("exit", (code) => {
      if (code === 0) {
        resolve();
        return;
      }
      const stdoutText = Buffer.concat(stdoutChunks).toString("utf8");
      const stderrText = Buffer.concat(stderrChunks).toString("utf8");
      reject(new Error(`llama-cli exited with code ${code}: ${stderrText || stdoutText}`));
    });
  });

  return {
    runtime: executableName.includes("completion") ? "llama-completion" : "llama-cli",
    durationMs: Date.now() - startedAt,
    output: normalizeOutput(Buffer.concat(stdoutChunks).toString("utf8"), prompt),
    stderr: Buffer.concat(stderrChunks).toString("utf8").trim(),
  };
}

async function runOllamaFallback({ modelArtifact, prompt, systemPrompt, maxTokens, config }) {
  const startedAt = Date.now();
  const response = await requestOllamaGenerate({
    tag: modelArtifact.ollamaTag,
    prompt,
    systemPrompt,
    maxTokens,
    temperature: config.generation.temperature,
  });
  return {
    runtime: "ollama",
    durationMs: Date.now() - startedAt,
    output: String(response.response ?? "").trim(),
    stderr: "",
    fallbackReason: "local model load failed; used Ollama runtime",
  };
}

export async function generateWithExistingBackend({
  recommendation,
  modelArtifact,
  profile = null,
  prompt,
  systemPrompt,
  maxTokens,
  enginePath,
  config,
  logger,
  localRunner = runLocalGeneration,
  ollamaRunner = runOllamaFallback,
  executableOverride,
}) {
  const resolvedProfile = profile ?? {
    modelPath: modelArtifact.modelPath,
    contextSize: recommendation.selected.contextTokens,
    gpuLayers: recommendation.selected.gpuLayers,
    threads: recommendation.selected.threads,
    batchSize: config.execution?.batchSize ?? 512,
    ubatchSize: config.execution?.ubatchSize ?? 128,
    kvCache: recommendation.selected.kvCache,
    mmap: true,
    mlock: false,
    noConversationMode: true,
  };
  const executable = executableOverride ?? (
    resolvedProfile.noConversationMode
      ? (resolveCompletionExecutable(config, enginePath) ?? resolveCliExecutable(config, enginePath))
      : resolveCliExecutable(config, enginePath)
  );
  if (!executable) {
    throw new Error("Unable to locate a llama.cpp CLI binary. Run 'lizard-llm engine install' or set --engine-path.");
  }
  const args = buildLlamaCppArgs({
    profile: resolvedProfile,
    prompt,
    systemPrompt,
    maxTokens,
  });
  args.splice(args.indexOf("--simple-io"), 0, "--temp", String(config.generation.temperature));
  if (path.basename(executable).toLowerCase().includes("completion")) {
    const logDisableIndex = args.indexOf("--log-disable");
    if (logDisableIndex >= 0) {
      args.splice(logDisableIndex, 1);
    }
  }

  logger?.info("generate-start", {
    executable,
    args,
    modelPath: modelArtifact.modelPath,
    modelId: modelArtifact.modelId,
    profileReason: profile?.reason,
  });

  let execution;
  try {
    execution = await localRunner({
      executable,
      args,
      prompt,
    });
    if (shouldRetryEmptyCompletionOutput(execution, executable)) {
      const cliExecutable = executableOverride ?? resolveCliExecutable(config, enginePath);
      if (cliExecutable && cliExecutable !== executable) {
        logger?.warn("generate-empty-completion-output-retrying-cli", {
          modelId: modelArtifact.modelId,
          modelPath: modelArtifact.modelPath,
          completionExecutable: executable,
          cliExecutable,
        });
        execution = await localRunner({
          executable: cliExecutable,
          args,
          prompt,
        });
      }
    }
  } catch (error) {
    if (!modelArtifact.ollamaTag || !isLocalLoadFailure(error.message)) {
      throw error;
    }
    logger?.warn("generate-local-load-failed-fallback-ollama", {
      modelId: modelArtifact.modelId,
      modelPath: modelArtifact.modelPath,
      ollamaTag: modelArtifact.ollamaTag,
      message: error.message,
    });
    execution = await ollamaRunner({
      modelArtifact,
      prompt,
      systemPrompt,
      maxTokens,
      config,
    });
  }

  return {
    status: "completed",
    modelId: modelArtifact.modelId,
    modelPath: modelArtifact.modelPath,
    runtime: execution.runtime,
    durationMs: execution.durationMs,
    profile,
    output: execution.output,
    stderr: execution.stderr,
    ...buildRuntimeContractFields({
      runtimeProvider: inferRuntimeProvider(execution.runtime),
      nativeModelStatus: "reference",
      fallbackReason: execution.fallbackReason ?? null,
    }),
    note: execution.fallbackReason ?? undefined,
  };
}

export async function generateResponse(options) {
  return generateWithExistingBackend(options);
}
