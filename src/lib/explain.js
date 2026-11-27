import { findModel } from "./model-registry.js";
import { estimateConfiguration, estimateGpuLayers } from "./estimator.js";
import { executionModes } from "./profiles.js";

export function buildExplanation({ registry, hardware, config, modelId, quant, contextTokens, mode }) {
  const model = findModel(registry, modelId);
  if (!model) {
    throw new Error(`Unknown model: ${modelId}`);
  }

  const modeProfile = executionModes[mode || config.defaultMode];
  const selectedQuant = quant || modeProfile.preferredQuants.find((entry) => model.quantizations.includes(entry)) || model.quantizations[0];
  const selectedContext = contextTokens || Math.min(model.defaultContext, modeProfile.contextCap);
  const kvCache = modeProfile.preferredKvCache;
  const gpuLayers = estimateGpuLayers({
    model,
    quant: selectedQuant,
    contextTokens: selectedContext,
    kvCache,
    hardware,
    config,
  });
  const estimate = estimateConfiguration({
    model,
    quant: selectedQuant,
    contextTokens: selectedContext,
    gpuLayers,
    kvCache,
    hardware,
    config,
  });

  return {
    modelId,
    quant: selectedQuant,
    contextTokens: selectedContext,
    kvCache,
    gpuLayers,
    estimate,
    verdict: estimate.fits ? "recommended" : "not_recommended",
    rationale: estimate.fits
      ? "The configuration fits within the configured RAM/VRAM policy."
      : estimate.fitsRam
        ? "VRAM budget is exceeded; reduce GPU layers, context, or quantization."
        : "RAM budget is exceeded; choose a smaller model or lower quantization.",
  };
}

