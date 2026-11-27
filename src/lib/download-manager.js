// Hugging Face GGUF download, integrity verification, catalog registration and
// model activation. Downloads are restricted to the curated allowlist and land
// in ~/.lizard-llm/models with metadata recorded in ~/.lizard-llm/model-catalog.json
// (the same catalog the Ollama import path uses), so downloaded models become
// runnable through the existing `run`, `generate`, and `benchmark` commands.

import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import crypto from "node:crypto";
import { Readable } from "node:stream";
import { ensureDirectory, sanitizeSlug } from "./paths.js";
import { verifyGgufMagic } from "./ollama.js";
import { findAllowlistedFile } from "./hf-model-catalog.js";
import { detectQuantFromFilename } from "./hf-hub.js";
import { inspectGgufFile } from "./gguf-metadata.js";

function loadCatalog(config) {
  if (!fs.existsSync(config.catalogPath)) {
    return { models: [] };
  }
  return JSON.parse(fs.readFileSync(config.catalogPath, "utf8"));
}

function saveCatalog(config, catalog) {
  ensureDirectory(path.dirname(config.catalogPath));
  fs.writeFileSync(config.catalogPath, JSON.stringify(catalog, null, 2));
}

export function quantizationCatalogGuidance(quantization) {
  const normalized = String(quantization ?? "").trim().toUpperCase();
  if (normalized !== "IQ4_XS" && normalized !== "IQ4_NL") {
    return null;
  }
  return {
    importanceMatrix: "preferred",
    preferredSource: "imatrix-calibrated",
    warning: "Prefer an imatrix-calibrated upload; an uncalibrated I-quant can be lower quality than Q4_K_M.",
  };
}

export function targetModelPath({ config, modelId, filename }) {
  const slug = sanitizeSlug(modelId).replace(/:/g, "-");
  const extension = filename && filename.toLowerCase().endsWith(".gguf") ? "" : ".gguf";
  return path.join(config.modelDirectory, `${slug}${extension || path.extname(filename) || ".gguf"}`);
}

function huggingFaceResolveUrl(repo, filename) {
  return `https://huggingface.co/${repo}/resolve/main/${filename}?download=true`;
}

function authHeaders(token) {
  const resolvedToken = token ?? process.env.HF_TOKEN ?? process.env.HUGGING_FACE_HUB_TOKEN;
  return resolvedToken ? { Authorization: `Bearer ${resolvedToken}` } : {};
}

// Fetch the expected sha256 (LFS oid) for a file from the Hugging Face API.
// Returns null when offline or when the file is not LFS-tracked.
export async function fetchExpectedSha256({ repo, filename, token }) {
  const url = `https://huggingface.co/api/models/${repo}/paths-info/main`;
  try {
    const response = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "User-Agent": "lizard-llm/0.1 hf-download",
        ...authHeaders(token),
      },
      body: JSON.stringify({ paths: [filename] }),
    });
    if (!response.ok) {
      return null;
    }
    const entries = await response.json();
    const match = Array.isArray(entries) ? entries.find((entry) => entry.path === filename) : null;
    return match?.lfs?.oid ?? null;
  } catch {
    return null;
  }
}

export function computeSha256(filePath) {
  return new Promise((resolve, reject) => {
    const hash = crypto.createHash("sha256");
    const stream = fs.createReadStream(filePath);
    stream.on("error", reject);
    stream.on("data", (chunk) => hash.update(chunk));
    stream.on("end", () => resolve(hash.digest("hex")));
  });
}

export async function verifyDownloadedModel({ path: filePath, expectedSha256 }) {
  if (!fs.existsSync(filePath)) {
    return { ok: false, reason: "missing", message: `File not found: ${filePath}` };
  }
  if (!verifyGgufMagic(filePath)) {
    return { ok: false, reason: "not-gguf", message: "File does not start with the GGUF magic header." };
  }
  const sha256 = await computeSha256(filePath);
  if (expectedSha256 && sha256.toLowerCase() !== String(expectedSha256).toLowerCase()) {
    return {
      ok: false,
      reason: "sha256-mismatch",
      message: `sha256 mismatch: expected ${expectedSha256}, got ${sha256}`,
      sha256,
    };
  }
  return {
    ok: true,
    sha256,
    verified: Boolean(expectedSha256),
    reason: expectedSha256 ? "verified" : "magic-only",
  };
}

export function parseVerifiedGgufMetadata({ filePath, fallbackQuant = null, fallbackArchitecture = null, fallbackParamsB = null }) {
  return inspectGgufFile(filePath, {
    fallbackQuant,
    fallbackArchitecture,
    fallbackParamsB,
  });
}

// Moves a finished download into place, or explains why it could not.
//
// `fs.rmSync(outputPath, { force: true })` was doing this job, and `force` does
// not mean what it looks like it means here: it suppresses ENOENT, not EPERM. On
// Windows a file that a process has memory-mapped cannot be unlinked at all, and
// a loaded model holds its own GGUF - so re-downloading the active model failed
// at the very last step with a bare "EPERM, Permission denied" naming a path,
// after transferring the entire file.
//
// Two things go wrong there and both are fixed here: the message now says what
// is holding the file and what to do about it, and the temporary file is removed
// on the way out. Previously it survived, because the only cleanup was on the
// download stream's own error path - so every failed attempt left another full
// copy of the model on disk under a .download name.
function replaceDownloadedFile({ tempPath, outputPath, modelId }) {
  try {
    fs.rmSync(outputPath, { force: true });
    fs.renameSync(tempPath, outputPath);
  } catch (error) {
    fs.rmSync(tempPath, { force: true });
    if (error?.code === "EPERM" || error?.code === "EBUSY" || error?.code === "EACCES") {
      const blocked = new Error(
        `The model file for '${modelId}' is in use and could not be replaced: ${outputPath}. `
        + "It is loaded in memory - unload it (or press Free RAM) and download again.",
      );
      blocked.code = "MODEL_FILE_IN_USE";
      blocked.cause = error;
      throw blocked;
    }
    throw error;
  }
}

export async function downloadGgufModel({
  repo,
  filename,
  modelId,
  license,
  token,
  config,
  logger,
  onProgress,
  fetchImpl = fetch,
  // Called with the resolved id and path just before the finished download is
  // moved into place, so a caller that owns the runtime can let go of the file
  // first. Replacing a model that is currently loaded is otherwise impossible on
  // Windows: the running worker holds its own GGUF mapped and the unlink fails.
  // The knowledge of what is resident lives in the server, not here, so this is
  // a hook rather than a call - the same split /api/models/delete already uses.
  onBeforeReplace = null,
}) {
  // The curated allowlist is now a TRUST signal, not a hard gate: any GGUF
  // repo the user browsed on the live Hub is downloadable. We still reject
  // obvious non-GGUF requests, and GGUF magic + sha256 verification downstream
  // is unchanged - a live download is verified exactly like a curated one.
  if (!repo || typeof repo !== "string") {
    throw new Error("A Hugging Face repository id is required.");
  }
  if (!filename || !/\.gguf$/i.test(filename)) {
    throw new Error(`Not a GGUF file: ${filename}`);
  }
  const allow = findAllowlistedFile(repo, filename);
  const trusted = Boolean(allow);
  const quant = allow?.quant ?? detectQuantFromFilename(filename) ?? "unknown";
  const resolvedLicense = allow?.model.license ?? license ?? null;
  const baseId = allow ? allow.model.id : sanitizeSlug(repo.split("/").pop() || repo);

  const resolvedModelId = modelId ?? `${baseId}-${String(quant).toLowerCase()}`;
  const outputPath = targetModelPath({ config, modelId: resolvedModelId, filename });
  ensureDirectory(path.dirname(outputPath));

  const url = huggingFaceResolveUrl(repo, filename);
  logger?.info("hf-download-start", { repo, filename, url, outputPath, modelId: resolvedModelId });

  const response = await fetchImpl(url, {
    redirect: "follow",
    headers: {
      "User-Agent": "lizard-llm/0.1 hf-download",
      ...authHeaders(token),
    },
  });
  if (!response.ok) {
    throw new Error(`HTTP ${response.status} while downloading ${url}`);
  }
  if (!response.body) {
    throw new Error(`No response body while downloading ${url}`);
  }

  const totalBytes = Number(response.headers.get("content-length")) || null;
  const tempPath = `${outputPath}.download`;

  // Is this file already here, in full? Checked now - after the headers, before
  // a single byte of body - because the answer decides whether to transfer
  // gigabytes at all.
  //
  // It also avoids a failure that could not be recovered from. Replacing the
  // file means unlinking it first, and Windows refuses to unlink a file that a
  // process still has memory-mapped: a model that is currently loaded holds its
  // own GGUF, so re-downloading it used to transfer the whole thing and only
  // then fail with a bare EPERM. When the bytes on disk already match what the
  // server is offering there is nothing to replace, so both the transfer and the
  // conflict simply do not happen.
  if (totalBytes && fs.existsSync(outputPath)) {
    const existing = fs.statSync(outputPath);
    if (existing.isFile() && existing.size === totalBytes && verifyGgufMagic(outputPath)) {
      // Nothing will read the body; releasing it frees the socket immediately
      // instead of leaving the connection to time out.
      await response.body.cancel().catch(() => {});
      // A temp file from an earlier interrupted attempt is dead weight - it is
      // the same gigabytes a second time, and nothing will ever resume it.
      fs.rmSync(tempPath, { force: true });
      logger?.info("hf-download-already-present", {
        repo,
        filename,
        outputPath,
        bytes: existing.size,
      });
      return {
        modelId: resolvedModelId,
        repo,
        filename,
        quant,
        trusted,
        license: resolvedLicense,
        path: outputPath,
        bytes: existing.size,
        alreadyPresent: true,
        hfModel: allow?.model ?? null,
      };
    }
  }

  fs.rmSync(tempPath, { force: true });

  let downloadedBytes = 0;
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

    input.on("data", (chunk) => {
      downloadedBytes += chunk.length;
      if (onProgress) {
        onProgress({
          downloadedBytes,
          totalBytes,
          ratio: totalBytes ? downloadedBytes / totalBytes : null,
        });
      }
    });
    input.on("error", finish);
    output.on("error", finish);
    output.on("finish", () => finish());
    input.pipe(output);
  });

  if (typeof onBeforeReplace === "function") {
    // Best-effort: if the release fails, the replace below still gets its
    // chance and reports the real reason it could not happen.
    await Promise.resolve(onBeforeReplace({ modelId: resolvedModelId, outputPath })).catch((error) => {
      logger?.warn?.("hf-download-release-before-replace-failed", {
        modelId: resolvedModelId,
        message: String(error?.message ?? error).slice(0, 200),
      });
    });
  }
  replaceDownloadedFile({ tempPath, outputPath, modelId: resolvedModelId });
  logger?.info("hf-download-complete", { repo, filename, outputPath, downloadedBytes });

  return {
    modelId: resolvedModelId,
    repo,
    filename,
    quant,
    trusted,
    license: resolvedLicense,
    path: outputPath,
    bytes: downloadedBytes,
    hfModel: allow?.model ?? null,
  };
}

export function registerDownloadedModel({ config, entry }) {
  const catalog = loadCatalog(config);
  const record = {
    id: entry.id,
    source: "huggingface",
    hfRepo: entry.hfRepo,
    filename: entry.filename,
    quant: entry.quant ?? null,
    quantizationGuidance: entry.quantizationGuidance
      ?? quantizationCatalogGuidance(entry.quant)
      ?? null,
    importedPath: entry.importedPath,
    sha256: entry.sha256 ?? null,
    license: entry.license ?? null,
    paramsB: entry.paramsB ?? null,
    gguf: entry.gguf ?? null,
    status: entry.status ?? "ready",
    benchmark: entry.benchmark ?? {},
    importedAt: entry.importedAt ?? new Date().toISOString(),
  };
  catalog.models = [...catalog.models.filter((model) => model.id !== record.id), record];
  saveCatalog(config, catalog);
  return record;
}

export function updateCatalogModel({ config, modelId, patch }) {
  const catalog = loadCatalog(config);
  let updated = null;
  catalog.models = catalog.models.map((model) => {
    if (model.id !== modelId) {
      return model;
    }
    updated = { ...model, ...patch };
    return updated;
  });
  if (updated) {
    saveCatalog(config, catalog);
  }
  return updated;
}

export function findCatalogModel(config, modelId) {
  return loadCatalog(config).models.find((model) => model.id === modelId) ?? null;
}

function homeConfigPath(config) {
  const runtimeDirectory = config?.runtimeDirectory ?? path.join(os.homedir(), ".lizard-llm");
  return path.join(runtimeDirectory, "config.json");
}

function readHomeConfig(config) {
  const filePath = homeConfigPath(config);
  if (!fs.existsSync(filePath)) {
    return {};
  }
  try {
    return JSON.parse(fs.readFileSync(filePath, "utf8"));
  } catch {
    return {};
  }
}

// Persist the active model selection into ~/.lizard-llm/config.json. We only
// touch the `activeModel` key so user-provided config overrides are preserved.
// Providers a model can be bound to: the three Lizard-owned engines, each with
// its own runner binary and its own port. llama.cpp and Ollama are selectable
// in chat but are not activation targets - they run the model themselves rather
// than through a Lizard-resident worker.
const ACTIVATABLE_RUNTIME_PROVIDERS = new Set(["lizard-native", "caterpillar", "ostrich"]);

export function activateModel({ config, modelId, runtimeProvider = "lizard-native" }) {
  const entry = findCatalogModel(config, modelId);
  if (!entry) {
    throw new Error(`Model not found in local catalog: ${modelId}`);
  }
  if (!entry.importedPath || !fs.existsSync(entry.importedPath)) {
    throw new Error(`Model artifact missing for ${modelId}; download it before activating.`);
  }

  const homeConfig = readHomeConfig(config);
  // Every provider that has its own runner binary is written through as itself.
  // The previous ternary collapsed anything that was not "caterpillar" onto
  // lizard-native, which silently downgraded an Ostrich activation the caller
  // had already access-checked: the binding then said lizard-native, so chat,
  // the resumed boot load and the local API endpoint all ran the wrong engine
  // on the wrong port while the UI reported Ostrich. Access control belongs to
  // the caller (resolveRequestedRuntimeProvider); this function's job is to
  // record faithfully what it was given.
  const requested = String(runtimeProvider ?? "").trim().toLowerCase();
  const selectedProvider = ACTIVATABLE_RUNTIME_PROVIDERS.has(requested) ? requested : "lizard-native";
  const activeModel = {
    id: entry.id,
    source: entry.source,
    modelPath: entry.importedPath,
    quant: entry.quant ?? null,
    gguf: entry.gguf ?? null,
    runtimeProvider: selectedProvider,
    activatedAt: new Date().toISOString(),
  };
  homeConfig.activeModel = activeModel;
  const filePath = homeConfigPath(config);
  ensureDirectory(path.dirname(filePath));
  fs.writeFileSync(filePath, JSON.stringify(homeConfig, null, 2));

  updateCatalogModel({ config, modelId, patch: { status: "active" } });
  return activeModel;
}

// The inverse of activateModel: clears the activeModel selection so nothing is
// bound any more. The model itself is left on disk and in the catalog, so
// re-activating it costs a click rather than another download.
//
// Freeing the resident worker is the caller's job, for the same reason as in
// deleteModel below: the runtime lives with the caller, not here.
export function deactivateModel({ config }) {
  const active = getActiveModel(config);
  if (!active) {
    return { id: null, wasActive: false };
  }
  const homeConfig = readHomeConfig(config);
  delete homeConfig.activeModel;
  const filePath = homeConfigPath(config);
  ensureDirectory(path.dirname(filePath));
  fs.writeFileSync(filePath, JSON.stringify(homeConfig, null, 2));

  // Back to "downloaded": still on disk, no longer the one Lizard uses.
  if (findCatalogModel(config, active.id)) {
    updateCatalogModel({ config, modelId: active.id, patch: { status: "downloaded" } });
  }
  return { id: active.id, wasActive: true };
}

// Removes a downloaded model completely: the artifact on disk, its catalog
// entry, and - when it is the active model - the activeModel selection in
// ~/.lizard-llm/config.json. Releasing any warm in-memory worker is the
// caller's job (the runtime lives with the caller, not here), and must
// happen BEFORE this runs: on Windows a live worker holds the GGUF mapped
// and the unlink would fail.
export function deleteModel({ config, modelId }) {
  const entry = findCatalogModel(config, modelId);
  if (!entry) {
    throw new Error(`Model not found in local catalog: ${modelId}`);
  }
  const active = getActiveModel(config);
  const wasActive = active?.id === modelId;

  let removedFile = false;
  if (entry.importedPath && fs.existsSync(entry.importedPath)) {
    fs.rmSync(entry.importedPath, { force: true });
    removedFile = true;
  }

  const catalog = loadCatalog(config);
  catalog.models = catalog.models.filter((model) => model.id !== modelId);
  saveCatalog(config, catalog);

  if (wasActive) {
    const homeConfig = readHomeConfig(config);
    delete homeConfig.activeModel;
    const filePath = homeConfigPath(config);
    ensureDirectory(path.dirname(filePath));
    fs.writeFileSync(filePath, JSON.stringify(homeConfig, null, 2));
  }

  return { id: modelId, wasActive, removedFile, removedPath: entry.importedPath ?? null };
}

export function getActiveModel(config) {
  return readHomeConfig(config).activeModel ?? null;
}
