import fs from "node:fs";
import path from "node:path";
import { spawn } from "node:child_process";
import { buildLlamaCppArgs } from "./execution-profile.js";
import { requireTrackedProcessTreeTermination } from "./model-lifecycle.js";

function findExecutable(name) {
  const envPath = process.env.PATH ?? "";
  const extensions = process.platform === "win32" ? [".exe", ".cmd", ".bat", ""] : [""];
  for (const directory of envPath.split(path.delimiter)) {
    for (const extension of extensions) {
      const fullPath = path.join(directory, `${name}${extension}`);
      if (fs.existsSync(fullPath)) {
        return fullPath;
      }
    }
  }
  return null;
}

function findExplicitCandidate(candidate) {
  const extensions = process.platform === "win32" ? [".exe", ".cmd", ".bat", ""] : [""];
  for (const extension of extensions) {
    const withExtension = candidate.endsWith(extension) ? candidate : `${candidate}${extension}`;
    if (fs.existsSync(withExtension)) {
      return path.resolve(withExtension);
    }
  }
  return null;
}

function resolveEngineExecutable(config, overridePath, preferServer = true) {
  const candidates = [];
  if (overridePath) {
    candidates.push(overridePath);
  }
  for (const directory of config.engine.preferredBinaryDirectories ?? []) {
    if (preferServer) {
      candidates.push(path.join(directory, config.engine.serverBinary));
    } else {
      candidates.push(path.join(directory, config.engine.cliBinary));
      candidates.push(path.join(directory, config.engine.serverBinary));
    }
  }
  if (preferServer) {
    candidates.push(config.engine.serverBinary);
  } else {
    candidates.push(config.engine.cliBinary, config.engine.serverBinary);
  }

  for (const candidate of candidates) {
    if (!candidate) {
      continue;
    }
    const explicit = findExplicitCandidate(candidate);
    if (explicit) {
      return explicit;
    }
    const resolved = findExecutable(candidate);
    if (resolved) {
      return resolved;
    }
  }

  return null;
}

function resolveModelPath(config, recommendation) {
  const modelFile = `${recommendation.selected.modelId.toLowerCase()}-${recommendation.selected.quant.toLowerCase()}.gguf`;
  return path.resolve(process.cwd(), config.modelDirectory, modelFile);
}

export function buildEnginePlan({ recommendation, profile = null, config, enginePath, modelPathOverride }) {
  const executable = resolveEngineExecutable(config, enginePath, true);
  const modelPath = modelPathOverride ?? profile?.modelPath ?? resolveModelPath(config, recommendation);
  const resolvedProfile = profile ?? {
    modelPath,
    contextSize: recommendation.selected.contextTokens,
    gpuLayers: recommendation.selected.gpuLayers,
    threads: recommendation.selected.threads,
    batchSize: config.execution?.batchSize ?? 512,
    ubatchSize: config.execution?.ubatchSize ?? 128,
    kvCache: recommendation.selected.kvCache,
    mmap: true,
    mlock: false,
  };
  const args = buildLlamaCppArgs({
    profile: {
      ...resolvedProfile,
      modelPath,
    },
    host: config.engine.host,
    port: config.engine.port,
    serverMode: true,
  });

  return {
    executable,
    exists: Boolean(executable),
    args,
    modelPath,
    profile: {
      ...resolvedProfile,
      modelPath,
    },
    recommendation,
    displayCommand: `${executable ?? config.engine.serverBinary} ${args.join(" ")}`,
  };
}

export async function runEnginePlan(plan, logger, hooks = {}) {
  if (!plan.exists) {
    throw new Error(`Unable to locate llama.cpp server binary. Planned command:\n${plan.displayCommand}`);
  }

  logger?.info("engine-start", {
    executable: plan.executable,
    args: plan.args,
    modelPath: plan.modelPath,
  });
  const child = spawn(plan.executable, plan.args, {
    stdio: "inherit",
  });
  const completion = new Promise((resolve, reject) => {
    child.once("exit", resolve);
    child.once("error", reject);
  });
  // Attach a rejection observer immediately. If lifecycle registration throws
  // below, caller-owned cleanup may not terminate the child until this stack
  // unwinds; a concurrent spawn error must not become an unhandled rejection.
  void completion.catch(() => {});
  try {
    await hooks?.onSpawn?.(child);
  } catch (error) {
    if (hooks?.callerOwnsCleanup !== true && child.pid) {
      try {
        await requireTrackedProcessTreeTermination(child.pid);
      } catch (releaseError) {
        releaseError.cause = error;
        throw releaseError;
      }
    }
    throw error;
  }
  const code = await completion;
  hooks?.onExit?.(code);
  if (code === 0) {
    logger?.info("engine-stop", { code });
    return;
  }
  throw new Error(`Engine exited with code ${code}`);
}

export function resolveBenchmarkExecutable(config, enginePath) {
  return resolveEngineExecutable(config, enginePath, false);
}

export function resolveCliExecutable(config, enginePath) {
  return resolveEngineExecutable(config, enginePath, false);
}

export function resolveCompletionExecutable(config, enginePath) {
  const candidates = [];
  if (enginePath) {
    candidates.push(enginePath);
  }
  for (const directory of config.engine.preferredBinaryDirectories ?? []) {
    candidates.push(path.join(directory, config.engine.completionBinary));
  }
  candidates.push(config.engine.completionBinary);

  for (const candidate of candidates) {
    if (!candidate) {
      continue;
    }
    const explicit = findExplicitCandidate(candidate);
    if (explicit) {
      return explicit;
    }
    const resolved = findExecutable(candidate);
    if (resolved) {
      return resolved;
    }
  }

  return null;
}

export function resolveServerExecutable(config, enginePath) {
  return resolveEngineExecutable(config, enginePath, true);
}

export function resolveHelperExecutable(config, executableName) {
  const candidates = [];
  if (executableName) {
    candidates.push(executableName);
    for (const directory of config.engine.preferredBinaryDirectories ?? []) {
      candidates.push(path.join(directory, executableName));
    }
  }

  for (const candidate of candidates) {
    const explicit = findExplicitCandidate(candidate);
    if (explicit) {
      return explicit;
    }
    const resolved = findExecutable(candidate);
    if (resolved) {
      return resolved;
    }
  }

  return null;
}
