// High-level Hugging Face onboarding workflow. Composes the download manager,
// the estimator/generate pipeline, and the local catalog so
// both the CLI and the onboarding UI can drive: download -> verify -> register
// -> activate, while reusing the planning logic.

import {
  downloadGgufModel,
  fetchExpectedSha256,
  parseVerifiedGgufMetadata,
  verifyDownloadedModel,
  registerDownloadedModel,
  updateCatalogModel,
  findCatalogModel,
  getActiveModel,
} from "./download-manager.js";
import { findAllowlistedHfModel, syntheticHfModelFromGguf } from "./hf-model-catalog.js";
import { buildHfRecommendation } from "./execution-profile.js";
export { buildHfRecommendation } from "./execution-profile.js";


// Resolve the planning metadata for a DOWNLOADED catalog entry. Prefer the
// curated allowlist (hand-verified shape); otherwise reconstruct an
// allowlist-shaped model from the GGUF header the download step parsed, so a
// live-discovered model plans/benchmarks/serves through the same pipeline.
function resolveHfModelForEntry(entry) {
  const allow = findAllowlistedHfModel(entry.hfRepo);
  if (allow) {
    return allow;
  }
  const synthetic = syntheticHfModelFromGguf(entry);
  if (!synthetic) {
    throw new Error(`Cannot plan a run for ${entry.id}: the downloaded GGUF is missing the layer/embedding shape needed to size it.`);
  }
  return synthetic;
}

export async function runHfDownloadWorkflow({
  repo,
  filename,
  modelId,
  license,
  token,
  config,
  logger,
  onProgress,
  onPhase,
  // Threaded straight through to downloadGgufModel - see the note there. The
  // server passes this so a model that is currently loaded releases its own file
  // before the download replaces it.
  onBeforeReplace = null,
  verifyOnline = true,
}) {
  // May be null for a live-discovered (non-allowlist) model - metadata is then
  // derived from the parsed GGUF below rather than the curated catalog.
  const hfModel = findAllowlistedHfModel(repo);

  onPhase?.({ phase: "download", message: `Downloading ${filename} from ${repo}` });
  const downloaded = await downloadGgufModel({
    repo,
    filename,
    modelId,
    license,
    token,
    config,
    logger,
    onProgress,
    onBeforeReplace,
  });

  onPhase?.({ phase: "verify", message: "Verifying model integrity" });
  let expectedSha256 = null;
  if (verifyOnline) {
    expectedSha256 = await fetchExpectedSha256({ repo, filename, token });
  }
  const verification = await verifyDownloadedModel({
    path: downloaded.path,
    expectedSha256,
  });
  if (!verification.ok) {
    const error = new Error(`Model verification failed (${verification.reason}): ${verification.message}`);
    error.code = ["not-gguf", "sha256-mismatch"].includes(verification.reason)
      ? "GGUF_CORRUPT"
      : "MODEL_DOWNLOAD_VERIFICATION_FAILED";
    error.details = {
      verificationReason: verification.reason,
      sha256: verification.sha256 ?? null,
    };
    throw error;
  }
  const gguf = parseVerifiedGgufMetadata({
    filePath: downloaded.path,
    fallbackQuant: downloaded.quant,
    fallbackArchitecture: downloaded.hfModel?.family ?? null,
    fallbackParamsB: hfModel?.paramsB ?? null,
  });
  // Prefer curated paramsB, else the GGUF header's parameter estimate. Stored on
  // the entry so syntheticHfModelFromGguf() can size live models later.
  const paramsB = hfModel?.paramsB ?? gguf.parameterEstimateB ?? null;

  onPhase?.({ phase: "register", message: "Registering model in the local catalog" });
  const record = registerDownloadedModel({
    config,
    entry: {
      id: downloaded.modelId,
      hfRepo: repo,
      filename,
      quant: downloaded.quant,
      importedPath: downloaded.path,
      sha256: verification.sha256,
      license: downloaded.license,
      paramsB,
      gguf,
      status: "ready",
      benchmark: {},
    },
  });

  return {
    hfModel,
    download: downloaded,
    verification,
    gguf,
    record,
  };
}

export async function smokeGenerateHfModel({
  modelId,
  config,
  hardware,
  logger,
  mode = "balanced",
  task,
  prompt = "Reply with the single word: READY.",
  maxTokens = 16,
  enginePath,
}) {
  const entry = findCatalogModel(config, modelId);
  if (!entry) {
    throw new Error(`Model not found in local catalog: ${modelId}`);
  }
  const hfModel = resolveHfModelForEntry(entry);
  const quant = entry.quant ?? hfModel.quantizations[0];
  const recommendation = buildHfRecommendation({ hfModel, quant, hardware, config, task, mode });

  const { generateResponse } = await import("./generate.js");
  return generateResponse({
    recommendation,
    modelArtifact: { modelId: entry.id, modelPath: entry.importedPath },
    prompt,
    maxTokens,
    enginePath,
    config,
    logger,
  });
}
