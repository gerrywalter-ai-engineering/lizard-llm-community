import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import process from "node:process";
import { Readable } from "node:stream";
import { findModel, findModelByOllamaTag, listRegistryModels } from "./model-registry.js";
import { ensureDirectory, sameVolume, sanitizeSlug } from "./paths.js";
import {
  ensureOllamaPulled,
  listInstalledOllamaModels,
  readOllamaManifest,
  resolveOllamaBlob,
  verifyGgufMagic,
} from "./ollama.js";
import { parseVerifiedGgufMetadata } from "./download-manager.js";
import { buildUniversalModelList, computeRuntimeVerdicts, deriveModelCapabilities } from "./universal-model.js";

// The catalog file can be tens of MB (entries carry full gguf.raw metadata
// including tokenizer vocabularies), and loadCatalog is hit several times per
// dashboard /api/state request (listModelsState, previewResolution,
// findLocalModelPath, ...). Re-reading + JSON.parse-ing that file on every
// call pegged the installed dashboard's event loop at ~70% CPU and made
// localhost:5567 completely unresponsive (found 2026-07-09 via CPU profile of
// the hung process: nearly all samples inside loadCatalog/readFileSync).
// Cache the parsed catalog keyed by mtime+size so it is parsed once per
// actual file change.
let catalogCache = null;

function loadCatalog(config) {
  if (!fs.existsSync(config.catalogPath)) {
    return { models: [] };
  }
  const stat = fs.statSync(config.catalogPath);
  const fingerprint = `${config.catalogPath}:${stat.size}:${stat.mtimeMs}`;
  if (catalogCache?.fingerprint === fingerprint) {
    return catalogCache.catalog;
  }
  const catalog = JSON.parse(fs.readFileSync(config.catalogPath, "utf8"));
  catalogCache = { fingerprint, catalog };
  return catalog;
}

function saveCatalog(config, catalog) {
  ensureDirectory(path.dirname(config.catalogPath));
  fs.writeFileSync(config.catalogPath, JSON.stringify(catalog, null, 2));
  const stat = fs.statSync(config.catalogPath);
  catalogCache = {
    fingerprint: `${config.catalogPath}:${stat.size}:${stat.mtimeMs}`,
    catalog,
  };
}

function updateCatalogEntry(config, entry, registry) {
  const catalog = loadCatalog(config);
  const matchedModel = findModel(registry, entry.id) ?? (entry.ollamaTag ? findModelByOllamaTag(registry, entry.ollamaTag) : null);
  const relatedIds = new Set([entry.id]);
  const relatedTags = new Set(entry.ollamaTag ? [entry.ollamaTag] : []);

  if (matchedModel) {
    relatedIds.add(matchedModel.id);
    for (const alias of matchedModel.aliases ?? []) {
      relatedIds.add(alias);
    }
    if (matchedModel.ollamaTag) {
      relatedTags.add(matchedModel.ollamaTag);
    }
    for (const aliasTag of matchedModel.ollamaTags ?? []) {
      relatedTags.add(aliasTag);
    }
  }

  catalog.models = [
    ...catalog.models.filter((model) => !relatedIds.has(model.id) && !relatedTags.has(model.ollamaTag)),
    entry,
  ];
  saveCatalog(config, catalog);
  return entry;
}

export function listLocalCatalogModels(config) {
  return loadCatalog(config).models;
}

function pickImportTarget({ tag, alias, config }) {
  const slug = sanitizeSlug(alias || tag).replace(/:/g, "-");
  return path.join(config.modelDirectory, `${slug}.gguf`);
}

function importWithStrategy(sourcePath, targetPath, strategy) {
  ensureDirectory(path.dirname(targetPath));
  if (fs.existsSync(targetPath)) {
    return "existing";
  }
  if (strategy === "hardlink" && sameVolume(sourcePath, targetPath)) {
    fs.linkSync(sourcePath, targetPath);
    return "hardlink";
  }
  if (strategy === "symlink") {
    fs.symlinkSync(sourcePath, targetPath);
    return "symlink";
  }
  fs.copyFileSync(sourcePath, targetPath);
  return "copy";
}

function inferModelId(tag, registry) {
  const exact = findModelByOllamaTag(registry, tag);
  if (exact) {
    return exact.id;
  }
  return sanitizeSlug(tag).replace(/:/g, "-");
}

export async function importOllamaModel({ tag, alias, strategy, pullIfMissing, config, registry, logger }) {
  const resolved = pullIfMissing || config.ollama.autoPullIfMissing
    ? ensureOllamaPulled(tag, config, logger)
    : readOllamaManifest(tag, config);
  if (!resolved) {
    throw new Error(`Ollama model not found locally: ${tag}`);
  }
  const blob = resolveOllamaBlob(tag, config);
  if (!verifyGgufMagic(blob.blobPath)) {
    throw new Error(`Ollama blob for ${tag} is not a GGUF model`);
  }

  const inferredId = alias || inferModelId(tag, registry);
  const gguf = parseVerifiedGgufMetadata({
    filePath: blob.blobPath,
    fallbackArchitecture: matchedArchitectureFromRegistry(registry, inferredId),
  });
  const entry = updateCatalogEntry(config, {
    id: inferredId,
    source: "ollama",
    ollamaTag: tag,
    importedPath: pickImportTarget({
      tag,
      alias: inferredId,
      config,
    }),
    blobPath: blob.blobPath,
    manifestPath: resolved.manifestPath,
    gguf,
    importMode: importWithStrategy(
      blob.blobPath,
      pickImportTarget({ tag, alias: inferredId, config }),
      strategy || config.ollama.importStrategy,
    ),
    importedAt: new Date().toISOString(),
  }, registry);
  logger?.info("ollama-import", { tag, targetPath: entry.importedPath, importMode: entry.importMode });
  return {
    models: [
      {
        type: "local",
        id: entry.id,
        source: entry.source,
        ollamaTag: entry.ollamaTag,
        importedPath: entry.importedPath,
        importMode: entry.importMode,
      },
    ],
  };
}

function matchedArchitectureFromRegistry(registry, modelId) {
  return findModel(registry, modelId)?.family ?? null;
}

// Model wrapper Phase 2: native-compat probe for an INSTALLED Ollama model
// that was never imported. Resolves the underlying blob, verifies it is GGUF,
// parses the header (header-only read - no full-file hashing) and returns the
// same gguf metadata shape the catalog stores, so the universal layer can run
// the real native-compatibility verdict on it. Never writes anything.
//
// Cached by blob path + size: `ollama list` models do not change under a
// stable blob, and the Select page may render many models per request.
const ollamaProbeCache = new Map();

export function probeOllamaModelGguf({ tag, config, logger }) {
  try {
    const blob = resolveOllamaBlob(tag, config);
    if (!blob?.blobPath || !fs.existsSync(blob.blobPath)) {
      return { ok: false, reason: "Ollama blob not found on disk." };
    }
    const stat = fs.statSync(blob.blobPath);
    const cacheKey = `${blob.blobPath}:${stat.size}`;
    if (ollamaProbeCache.has(cacheKey)) {
      return ollamaProbeCache.get(cacheKey);
    }
    if (!verifyGgufMagic(blob.blobPath)) {
      const result = { ok: false, reason: "The Ollama blob is not a GGUF file (Safetensors-based or multi-layer model)." };
      ollamaProbeCache.set(cacheKey, result);
      return result;
    }
    const gguf = parseVerifiedGgufMetadata({ filePath: blob.blobPath });
    const result = { ok: true, gguf, blobPath: blob.blobPath };
    ollamaProbeCache.set(cacheKey, result);
    return result;
  } catch (error) {
    logger?.warn?.("ollama-probe-failed", { tag, message: error.message });
    return { ok: false, reason: `Could not inspect the Ollama model: ${error.message}` };
  }
}

export async function listAvailableModels({ config, registry, source, logger, unified = false }) {
  // Model wrapper Phase 1: the unified view merges every source into one
  // deduplicated UniversalModel list with per-runtime verdicts, so a model
  // installed through Ollama and the same model downloaded from Hugging Face
  // appear as ONE model with several sources - never duplicate entries.
  if (unified) {
    const models = buildUniversalModelList({
      registryModels: listRegistryModels(registry),
      catalogModels: listLocalCatalogModels(config),
      ollamaModels: listInstalledOllamaModels(logger),
    });
    // Phase 2: for installed Ollama models the merge could not inspect (no
    // catalog import yet), probe the blob so "importable" becomes a REAL
    // verdict - supported-after-import, or blocked with the matrix's reason.
    return models.map((model) => {
      if (model.gguf || !model.ollamaInstalled || !model.ollamaTag) return model;
      const probe = probeOllamaModelGguf({ tag: model.ollamaTag, config, logger });
      if (!probe.ok) {
        return {
          ...model,
          probe: { ok: false, reason: probe.reason },
        };
      }
      const probed = { ...model, gguf: probe.gguf, probe: { ok: true } };
      return {
        ...probed,
        capabilities: deriveModelCapabilities(probe.gguf, {
          reasoning: /reasoning|r1|think|qwq/i.test(String(model.id)),
          vision: /vl|vision|llava|pixtral/i.test(String(model.id)),
          embeddings: /embed|bge|gte|e5/i.test(String(model.id)),
        }),
        runtimes: computeRuntimeVerdicts(probed),
      };
    });
  }

  const selectedSource = source ?? "all";
  const results = [];

  if (selectedSource === "registry" || selectedSource === "all") {
    results.push(...listRegistryModels(registry).map((model) => ({ ...model, type: "registry" })));
  }
  if (selectedSource === "ollama" || selectedSource === "all") {
    results.push(...listInstalledOllamaModels(logger).map((model) => ({ ...model, type: "ollama" })));
  }
  if (selectedSource === "local" || selectedSource === "all") {
    results.push(...listLocalCatalogModels(config).map((model) => ({ ...model, type: "local" })));
  }

  return results;
}

function findLocalModelPath(modelId, config) {
  const entry = listLocalCatalogModels(config).find((model) => model.id === modelId);
  if (entry?.importedPath && fs.existsSync(entry.importedPath)) {
    return entry;
  }
  return null;
}

function resolveRegistryModelForRequest(requestedModel, registry) {
  if (!requestedModel) {
    return null;
  }
  return String(requestedModel).includes(":")
    ? findModelByOllamaTag(registry, requestedModel)
    : findModel(registry, requestedModel);
}

function normalizeQuantKey(value) {
  return String(value ?? "")
    .trim()
    .toUpperCase()
    .replace(/[^A-Z0-9]+/g, "_")
    .replace(/_+/g, "_")
    .replace(/^_|_$/g, "");
}

function huggingFaceHubDirectory() {
  if (process.env.HUGGINGFACE_HUB_CACHE) {
    return process.env.HUGGINGFACE_HUB_CACHE;
  }
  if (process.env.HF_HOME) {
    return path.join(process.env.HF_HOME, "hub");
  }
  return path.join(os.homedir(), ".cache", "huggingface", "hub");
}

function huggingFaceRepoDirectory(repo) {
  return path.join(huggingFaceHubDirectory(), `models--${String(repo).replace(/\//g, "--")}`);
}

function findHuggingFaceSnapshotFile(repo, filename) {
  const snapshotsDirectory = path.join(huggingFaceRepoDirectory(repo), "snapshots");
  if (!fs.existsSync(snapshotsDirectory)) {
    return null;
  }

  const snapshots = fs.readdirSync(snapshotsDirectory, { withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .map((entry) => {
      const snapshotPath = path.join(snapshotsDirectory, entry.name);
      const candidatePath = path.join(snapshotPath, filename);
      let modifiedMs = 0;
      try {
        modifiedMs = fs.statSync(snapshotPath).mtimeMs;
      } catch {
        modifiedMs = 0;
      }
      return {
        snapshotPath,
        candidatePath,
        modifiedMs,
      };
    })
    .sort((left, right) => right.modifiedMs - left.modifiedMs);

  for (const snapshot of snapshots) {
    if (fs.existsSync(snapshot.candidatePath)) {
      return {
        path: snapshot.candidatePath,
        snapshotPath: snapshot.snapshotPath,
      };
    }
  }

  return null;
}

function huggingFaceManualSnapshotDirectory(repo) {
  return path.join(huggingFaceRepoDirectory(repo), "snapshots", "manual-download");
}

function huggingFaceResolveUrl(repo, filename) {
  return `https://huggingface.co/${repo}/resolve/main/${filename}?download=true`;
}

async function downloadFile(url, outputPath) {
  const response = await fetch(url, {
    redirect: "follow",
    headers: {
      "User-Agent": "lizard-llm/0.1 native-source-fetch",
    },
  });
  if (!response.ok) {
    throw new Error(`HTTP ${response.status} while downloading ${url}`);
  }
  if (!response.body) {
    throw new Error(`No response body while downloading ${url}`);
  }

  ensureDirectory(path.dirname(outputPath));
  const tempPath = `${outputPath}.download`;
  fs.rmSync(tempPath, { force: true });

  await new Promise((resolve, reject) => {
    const input = Readable.fromWeb(response.body);
    const output = fs.createWriteStream(tempPath);
    let settled = false;

    const finish = (error = null) => {
      if (settled) {
        return;
      }
      settled = true;
      if (error) {
        fs.rmSync(tempPath, { force: true });
        reject(error);
        return;
      }
      resolve();
    };

    input.on("error", finish);
    output.on("error", finish);
    output.on("finish", () => finish());
    input.pipe(output);
  });

  fs.rmSync(outputPath, { force: true });
  fs.renameSync(tempPath, outputPath);
}

async function ensureHuggingFaceSnapshotFile({ repo, filename, logger }) {
  const existing = findHuggingFaceSnapshotFile(repo, filename);
  if (existing) {
    return existing;
  }

  const snapshotDirectory = huggingFaceManualSnapshotDirectory(repo);
  const outputPath = path.join(snapshotDirectory, filename);
  if (fs.existsSync(outputPath)) {
    return {
      path: outputPath,
      snapshotPath: snapshotDirectory,
    };
  }

  const url = huggingFaceResolveUrl(repo, filename);
  logger?.info("native-source-download-start", {
    source: "huggingface-cache",
    repo,
    filename,
    outputPath,
    url,
  });
  await downloadFile(url, outputPath);
  logger?.info("native-source-download-complete", {
    source: "huggingface-cache",
    repo,
    filename,
    outputPath,
  });
  return {
    path: outputPath,
    snapshotPath: snapshotDirectory,
  };
}

function preferredNativeQuantType(model, compressionLevel) {
  const nativeLlamaCpp = model?.nativeLlamaCpp;
  if (!nativeLlamaCpp) {
    return null;
  }
  const requestedLevel = String(compressionLevel ?? "").trim().toUpperCase();
  if (requestedLevel && nativeLlamaCpp.preferredSourceQuantByCompression?.[requestedLevel]) {
    return nativeLlamaCpp.preferredSourceQuantByCompression[requestedLevel];
  }
  return nativeLlamaCpp.defaultQuantType ?? null;
}

function artifactFromCachedNative({ model, nativeLlamaCpp, quantKey, cached, logger }) {
  logger?.info("native-source-resolved", {
    modelId: model.id,
    source: "huggingface-cache",
    repo: nativeLlamaCpp.repo,
    quantType: quantKey,
    modelPath: cached.path,
  });
  return {
    modelId: model.id,
    modelPath: cached.path,
    source: "huggingface-cache",
    sourceRuntime: "llama.cpp",
    sourceRepo: nativeLlamaCpp.repo,
    sourceQuantType: quantKey,
    ollamaTag: model.ollamaTag,
  };
}

async function resolveNativeLlamaCppArtifact({
  model,
  compressionLevel,
  strictSourceQuant = false,
  allowDownload = false,
  logger,
}) {
  const nativeLlamaCpp = model?.nativeLlamaCpp;
  if (!nativeLlamaCpp || nativeLlamaCpp.provider !== "huggingface-cache" || !nativeLlamaCpp.repo || !nativeLlamaCpp.files) {
    return null;
  }

  const filesByQuant = Object.fromEntries(
    Object.entries(nativeLlamaCpp.files).map(([quantType, filename]) => [normalizeQuantKey(quantType), filename]),
  );
  const preferredQuant = preferredNativeQuantType(model, compressionLevel);
  const preferredQuantKey = normalizeQuantKey(preferredQuant);
  const orderedQuantKeys = [
    preferredQuantKey,
    normalizeQuantKey(nativeLlamaCpp.defaultQuantType),
    ...Object.keys(filesByQuant),
  ].filter((value, index, values) => value && values.indexOf(value) === index);

  if (strictSourceQuant && preferredQuantKey) {
    const preferredFile = filesByQuant[preferredQuantKey];
    let preferredCached = preferredFile ? findHuggingFaceSnapshotFile(nativeLlamaCpp.repo, preferredFile) : null;
    if (!preferredCached && allowDownload && preferredFile) {
      preferredCached = await ensureHuggingFaceSnapshotFile({
        repo: nativeLlamaCpp.repo,
        filename: preferredFile,
        logger,
      });
    }
    if (!preferredCached) {
      const error = new Error(
        `Native llama.cpp source unavailable for ${model.id} at ${compressionLevel}: missing Hugging Face cached file ${preferredFile ?? preferredQuant}.`,
      );
      error.code = "NATIVE_SOURCE_UNAVAILABLE";
      throw error;
    }
    return artifactFromCachedNative({
      model,
      nativeLlamaCpp,
      quantKey: preferredQuantKey,
      cached: preferredCached,
      logger,
    });
  }

  for (const quantKey of orderedQuantKeys) {
    const filename = filesByQuant[quantKey];
    if (!filename) {
      continue;
    }
    const cached = findHuggingFaceSnapshotFile(nativeLlamaCpp.repo, filename);
    if (!cached) {
      continue;
    }
    return artifactFromCachedNative({
      model,
      nativeLlamaCpp,
      quantKey,
      cached,
      logger,
    });
  }

  if (allowDownload && preferredQuantKey && filesByQuant[preferredQuantKey]) {
    const downloaded = await ensureHuggingFaceSnapshotFile({
      repo: nativeLlamaCpp.repo,
      filename: filesByQuant[preferredQuantKey],
      logger,
    });
    return artifactFromCachedNative({
      model,
      nativeLlamaCpp,
      quantKey: preferredQuantKey,
      cached: downloaded,
      logger,
    });
  }

  return null;
}

async function ensureRegistryModelImported({
  model,
  config,
  registry,
  logger,
  pullIfMissing,
  compressionLevel,
  strictSourceQuant = false,
}) {
  const nativeArtifact = await resolveNativeLlamaCppArtifact({
    model,
    compressionLevel,
    strictSourceQuant,
    allowDownload: Boolean(pullIfMissing),
    logger,
  });
  if (nativeArtifact) {
    return nativeArtifact;
  }

  const local = findLocalModelPath(model.id, config);
  if (local) {
    return {
      modelId: local.id,
      modelPath: local.importedPath,
      source: "local",
      ollamaTag: local.ollamaTag,
      sha256: local.sha256 ?? null,
      gguf: local.gguf ?? null,
    };
  }
  if (!model.ollamaTag) {
    throw new Error(`Model ${model.id} is not imported and has no configured Ollama source`);
  }
  await importOllamaModel({
    tag: model.ollamaTag,
    alias: model.id,
    strategy: config.ollama.importStrategy,
    pullIfMissing,
    config,
    registry,
    logger,
  });
  const imported = findLocalModelPath(model.id, config);
  return {
    modelId: imported.id,
    modelPath: imported.importedPath,
    source: "ollama",
    ollamaTag: model.ollamaTag,
    sha256: imported.sha256 ?? null,
    gguf: imported.gguf ?? null,
  };
}

export async function resolveExplicitModel({
  requestedModel,
  config,
  registry,
  logger,
  pullIfMissing,
  compressionLevel,
  strictSourceQuant = false,
}) {
  const registryModel = resolveRegistryModelForRequest(requestedModel, registry);
  if (registryModel) {
    const artifact = await ensureRegistryModelImported({
      model: registryModel,
      config,
      registry,
      logger,
      pullIfMissing,
      compressionLevel,
      strictSourceQuant,
    });
    if (artifact) {
      return artifact;
    }
  }

  if (String(requestedModel).includes(":")) {
    const modelId = inferModelId(requestedModel, registry);
    await importOllamaModel({
      tag: requestedModel,
      alias: modelId,
      strategy: config.ollama.importStrategy,
      pullIfMissing,
      config,
      registry,
      logger,
    });
    const local = findLocalModelPath(modelId, config);
    return {
      modelId: local.id,
      modelPath: local.importedPath,
      source: "ollama",
      ollamaTag: requestedModel,
      sha256: local.sha256 ?? null,
      gguf: local.gguf ?? null,
      sourceQuantType: local.gguf?.quant ?? null,
    };
  }

  const local = findLocalModelPath(requestedModel, config);
  if (local) {
    return {
      modelId: local.id,
      modelPath: local.importedPath,
      source: "local",
      ollamaTag: local.ollamaTag,
      sha256: local.sha256 ?? null,
      gguf: local.gguf ?? null,
      sourceQuantType: local.gguf?.quant ?? null,
    };
  }

  try {
    const modelId = inferModelId(requestedModel, registry);
    await importOllamaModel({
      tag: requestedModel,
      alias: modelId,
      strategy: config.ollama.importStrategy,
      pullIfMissing,
      config,
      registry,
      logger,
    });
    const imported = findLocalModelPath(modelId, config);
    return {
      modelId: imported.id,
      modelPath: imported.importedPath,
      source: "ollama",
      ollamaTag: requestedModel,
      sha256: imported.sha256 ?? null,
      gguf: imported.gguf ?? null,
      sourceQuantType: imported.gguf?.quant ?? null,
    };
  } catch {
    // Fall through to the explicit resolution error below.
  }

  throw new Error(`Unable to resolve a runnable model artifact for ${requestedModel}`);
}

export async function ensureModelForRecommendation({ recommendation, config, registry, logger, pullIfMissing }) {
  const selected = findModel(registry, recommendation.selected.modelId);
  if (!selected) {
    throw new Error(`Unknown recommended model: ${recommendation.selected.modelId}`);
  }

  try {
    return await ensureRegistryModelImported({
      model: selected,
      config,
      registry,
      logger,
      pullIfMissing,
      strictSourceQuant: false,
    });
  } catch (error) {
    for (const alternative of recommendation.alternatives) {
      const altModel = findModel(registry, alternative.modelId);
      if (!altModel) {
        continue;
      }
      try {
        return await ensureRegistryModelImported({
          model: altModel,
          config,
          registry,
          logger,
          pullIfMissing,
          strictSourceQuant: false,
        });
      } catch {
        continue;
      }
    }
    throw error;
  }
}
