import { estimateConfiguration, estimateGpuLayers } from "./estimator.js";
import { executionModes, kvCacheProfiles, quantizationProfiles } from "./profiles.js";
import { findModel } from "./model-registry.js";

function clamp(value, min, max) {
  return Math.max(min, Math.min(value, max));
}

// Run 06 Phase 0e qualified this exact CPU/model/quant shape with three
// independently memory-cleared, five-sample campaigns. Five workers retained
// 99.18% of the previous six-worker throughput while reducing normalized CPU
// from 71.82% to 61.54%. A separate equal-thread llama.cpp qualification then
// measured a 1.039x median-of-medians Lizard lead. Keep the rule deliberately
// narrow: other model shapes, quantizations, and CPU sizes remain on the prior
// 75%-of-logical-threads policy until separately qualified.
export function recommendedCpuThreads({ hardware, model, quant } = {}) {
  const logicalThreads = Math.max(1, Math.trunc(Number(hardware?.cpu?.logicalThreads) || 1));
  const qualifiedRun06Shape = logicalThreads === 8
    && model?.family === "llama"
    && Number(model?.totalLayers) === 28
    && Number(model?.hiddenSize) === 3072
    && String(quant).toUpperCase() === "Q4_K_M";
  return qualifiedRun06Shape
    ? 5
    : Math.max(1, Math.floor(logicalThreads * 0.75));
}

function buildContextCandidates(targetContext, modeContextCap, model) {
  const preferred = clamp(targetContext, 2048, Math.min(modeContextCap, model.maxContext));
  const candidates = [preferred, 8192, 6144, 4096, 3072, 2048]
    .filter((value, index, values) => value <= model.maxContext && values.indexOf(value) === index);
  if (!candidates.includes(model.defaultContext)) {
    candidates.push(model.defaultContext);
  }
  return candidates.sort((left, right) => right - left);
}

function scoreCandidate({ candidate, taskProfile, mode }) {
  const quantProfile = quantizationProfiles[candidate.quant];
  const taskFit = candidate.model.tasks.includes(taskProfile.taskType) ? 2 : 0;
  const modelSizeFit = -Math.abs(candidate.model.paramsB - mode.targetParamsB) * 0.15;
  const contextPenalty = candidate.contextTokens < taskProfile.contextNeed ? -1.25 : 0;
  const memoryReward = Math.min(candidate.estimate.ramHeadroomGB, 6) * 0.15 + Math.min(candidate.estimate.vramHeadroomGB || 2, 2) * 0.15;
  const tightPenalty = candidate.estimate.tight ? -0.5 : 0;
  return Number((quantProfile.qualityScore + taskFit + modelSizeFit + contextPenalty + memoryReward + tightPenalty).toFixed(3));
}

function buildCandidate({ model, quant, contextTokens, kvCache, hardware, config, taskProfile, modeName, mode }) {
  const gpuLayers = estimateGpuLayers({ model, quant, contextTokens, kvCache, hardware, config });
  const estimate = estimateConfiguration({
    model,
    quant,
    contextTokens,
    gpuLayers,
    kvCache,
    hardware,
    config,
  });

  const candidate = {
    model,
    quant,
    kvCache,
    contextTokens,
    gpuLayers,
    threads: recommendedCpuThreads({ hardware, model, quant }),
    mode: modeName,
    estimate,
  };
  candidate.score = scoreCandidate({ candidate, taskProfile, mode });
  return candidate;
}

function selectCandidates({ models, taskProfile, modeName, hardware, config }) {
  const mode = executionModes[modeName];
  const candidates = [];

  for (const model of models) {
    for (const quant of mode.preferredQuants.filter((entry) => model.quantizations.includes(entry))) {
      const contextCandidates = buildContextCandidates(taskProfile.contextNeed, mode.contextCap, model);
      for (const contextTokens of contextCandidates) {
        const candidate = buildCandidate({
          model,
          quant,
          contextTokens,
          kvCache: mode.preferredKvCache,
          hardware,
          config,
          taskProfile,
          modeName,
          mode,
        });
        if (candidate.estimate.fits) {
          candidates.push(candidate);
          break;
        }
      }

      if (mode.elasticFallback) {
        const elasticCandidate = buildCandidate({
          model,
          quant,
          contextTokens: Math.min(4096, model.defaultContext),
          kvCache: mode.preferredKvCache,
          hardware,
          config,
          taskProfile,
          modeName,
          mode,
        });
        if (elasticCandidate.estimate.fits || (elasticCandidate.estimate.ramHeadroomGB > -1.0 && elasticCandidate.estimate.vramHeadroomGB > -0.5)) {
          elasticCandidate.elasticWarning = !elasticCandidate.estimate.fits;
          elasticCandidate.score -= elasticCandidate.elasticWarning ? 1.5 : 0;
          candidates.push(elasticCandidate);
        }
      }
    }
  }

  return candidates.sort((left, right) => right.score - left.score);
}

function buildReason(candidate, taskProfile) {
  const reasons = [
    `${candidate.model.id} fits the detected hardware budget`,
    `${candidate.quant} matches ${candidate.mode} mode`,
    `${candidate.contextTokens} tokens cover the ${taskProfile.taskType} workload`,
  ];
  if (candidate.gpuLayers > 0) {
    reasons.push(`${candidate.gpuLayers}/${candidate.model.totalLayers} layers can be offloaded to GPU`);
  } else {
    reasons.push("This plan stays CPU-first because usable GPU VRAM is unavailable");
  }
  if (candidate.elasticWarning) {
    reasons.push("Elastic mode accepts a small headroom deficit and may degrade throughput");
  }
  return reasons;
}

export function buildRecommendation({ hardware, registry, config, requestedMode, requestedModelId, taskProfile }) {
  const modeName = requestedMode || taskProfile.recommendedMode || config.defaultMode;
  const mode = executionModes[modeName];
  if (!mode) {
    throw new Error(`Unsupported mode: ${modeName}`);
  }

  const models = requestedModelId && requestedModelId !== "auto"
    ? [findModel(registry, requestedModelId)].filter(Boolean)
    : registry.models;
  if (models.length === 0) {
    throw new Error(`Unknown model: ${requestedModelId}`);
  }

  const candidates = selectCandidates({
    models,
    taskProfile,
    modeName,
    hardware,
    config,
  });

  if (candidates.length === 0) {
    throw new Error("No feasible configuration was found for the detected hardware and requested mode.");
  }

  const selected = candidates[0];
  return {
    hardwareSummary: {
      ramAvailableGB: hardware.memory.availableGB,
      gpu: hardware.primaryGpu?.name ?? "CPU only",
      vramGB: hardware.primaryGpu?.vramGB ?? 0,
    },
    taskProfile,
    selected: {
      modelId: selected.model.id,
      family: selected.model.family,
      paramsB: selected.model.paramsB,
      hiddenSize: selected.model.hiddenSize,
      quant: selected.quant,
      kvCache: selected.kvCache,
      contextTokens: selected.contextTokens,
      gpuLayers: selected.gpuLayers,
      totalLayers: selected.model.totalLayers,
      threads: selected.threads,
      mode: selected.mode,
      estimate: selected.estimate,
      elasticWarning: Boolean(selected.elasticWarning),
    },
    reasons: buildReason(selected, taskProfile),
    alternatives: candidates.slice(1, 4).map((candidate) => ({
      modelId: candidate.model.id,
      quant: candidate.quant,
      contextTokens: candidate.contextTokens,
      mode: candidate.mode,
      score: candidate.score,
      ramRequiredGB: candidate.estimate.ramRequiredGB,
      vramRequiredGB: candidate.estimate.vramRequiredGB,
    })),
    catalogSize: registry.models.length,
  };
}

