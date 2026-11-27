import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";

function ensureDirectory(directoryPath) {
  fs.mkdirSync(directoryPath, { recursive: true, mode: 0o700 });
}

const BULK_GGUF_RAW_FIELDS = new Set([
  "tokenizer.ggml.tokens",
  "tokenizer.ggml.merges",
  "tokenizer.ggml.scores",
  "tokenizer.ggml.token_type",
]);

export function compactGgufMetadata(gguf = null) {
  if (!gguf || typeof gguf !== "object") {
    return gguf;
  }
  const compactRaw = gguf.raw && typeof gguf.raw === "object"
    ? Object.fromEntries(
      Object.entries(gguf.raw).filter(([key]) => !BULK_GGUF_RAW_FIELDS.has(key)),
    )
    : gguf.raw;
  return {
    ...gguf,
    raw: compactRaw,
    metadataCompacted: true,
  };
}

function compactLaunchConfig(config = null) {
  if (!config || typeof config !== "object") {
    return config;
  }
  return {
    ...config,
    // Delivery credentials and contact addresses are inherited through the
    // child process environment and reloaded by standalone.js. They must never
    // be serialized into a long-lived launch file beside model metadata.
    failureIntelligence: config.failureIntelligence
      ? {
        ...config.failureIntelligence,
        userEmail: null,
        nodebb: {
          ...(config.failureIntelligence.nodebb ?? {}),
          apiToken: null,
        },
        email: {
          ...(config.failureIntelligence.email ?? {}),
          apiToken: null,
          adminTo: [],
        },
      }
      : config.failureIntelligence,
    activeModel: config.activeModel
      ? {
        ...config.activeModel,
        gguf: compactGgufMetadata(config.activeModel.gguf),
      }
      : config.activeModel,
  };
}

export function writeNativeServerLaunchConfig({
  config,
  modelArtifact,
  profile,
  recommendation,
  host = "127.0.0.1",
  port = 8080,
  internalBenchmarkServer = false,
  launchId = crypto.randomUUID(),
  runtimeProvider = "lizard-native",
} = {}) {
  const runtimeDirectory = config?.runtimeDirectory ?? process.cwd();
  // Per-provider launch directory: sharing one would let two engines
  // overwrite each other's launch config.
  const SERVER_DIRS = { caterpillar: "caterpillar-server", ostrich: "ostrich-server" };
  const directory = path.join(runtimeDirectory, SERVER_DIRS[runtimeProvider] ?? "native-server");
  ensureDirectory(directory);
  const filePath = path.join(directory, `${crypto.randomUUID()}.json`);
  fs.writeFileSync(filePath, JSON.stringify({
    host,
    port,
    launchId,
    runtimeProvider,
    internalBenchmarkServer: Boolean(internalBenchmarkServer),
    modelArtifact: modelArtifact
      ? {
        ...modelArtifact,
        gguf: compactGgufMetadata(modelArtifact.gguf),
      }
      : modelArtifact,
    profile: profile
      ? {
        ...profile,
        gguf: compactGgufMetadata(profile.gguf),
      }
      : profile,
    recommendation,
    config: compactLaunchConfig(config),
  }, null, 2), { encoding: "utf8", mode: 0o600 });
  return filePath;
}
