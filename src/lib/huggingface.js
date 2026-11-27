// Hugging Face GGUF discovery and compatibility scoring.
//
// Search is restricted to the curated allowlist (see hf-model-catalog.js). For
// each allowlisted model we pick the best quantization for the requested mode,
// estimate its memory footprint against the scanned hardware, and produce a
// compatibility score so the CLI and onboarding UI can rank corporate-safe
// choices for the detected machine.

import { quantizationProfiles, executionModes } from "./profiles.js";
import { listAllowlistedHfModels, findAllowlistedHfModel } from "./hf-model-catalog.js";
import { quantBitsPerWeight } from "./hf-hub.js";

const RUNTIME_OVERHEAD_GB = 1.2;

// Named download-size categories for the onboarding "browse by size" selector.
// Bucketed on the raw GGUF file size (sizeGB), not the RAM/VRAM footprint
// estimate, since "how big is the download" is the more intuitive framing
// when a user is comparing models across families/quants.
export const SIZE_BUCKETS = [
  { id: "2gb", label: "~2 GB", maxGB: 2.5 },
  { id: "5gb", label: "~5 GB", maxGB: 6.5 },
  { id: "9gb", label: "~9 GB", maxGB: 11 },
  { id: "14gb", label: "~14 GB+", maxGB: Infinity },
];

export function sizeBucketForGB(sizeGB) {
  const bucket = SIZE_BUCKETS.find((entry) => sizeGB <= entry.maxGB);
  return bucket ?? SIZE_BUCKETS[SIZE_BUCKETS.length - 1];
}

export function estimateQuantFootprint({ paramsB, quant, contextTokens, hiddenSize, totalLayers }) {
  // Prefer the curated quant profile's bits-per-weight; fall back to the wider
  // Hub quant table (Q4_0, IQ*, F16, ...) so live-discovered quants estimate too.
  const bitsPerWeight = quantizationProfiles[quant]?.bitsPerWeight ?? quantBitsPerWeight(quant) ?? quantizationProfiles.Q4_K_M.bitsPerWeight;
  // Weight memory: params (billions) * bits-per-weight / 8 ~= GiB.
  const weightsGB = (paramsB * bitsPerWeight) / 8;
  // KV cache: 2 (K+V) * layers * hidden * ctx * 2 bytes (f16) -> GiB.
  const kvCacheGB = (2 * totalLayers * hiddenSize * contextTokens * 2) / 1024 ** 3;
  const totalGB = weightsGB + kvCacheGB + RUNTIME_OVERHEAD_GB;
  return {
    weightsGB: Number(weightsGB.toFixed(2)),
    kvCacheGB: Number(kvCacheGB.toFixed(2)),
    totalGB: Number(totalGB.toFixed(2)),
  };
}

// Rough transformer shape (hidden size + layer count) inferred from parameter
// count, used ONLY for the pre-download compatibility estimate of live Hub
// models where the real GGUF header (and thus the exact shape) is not yet
// available. Post-download, the parsed GGUF values are used instead.
export function approxModelShape(paramsB) {
  const points = [
    [0.5, { hiddenSize: 1024, totalLayers: 24 }],
    [1, { hiddenSize: 2048, totalLayers: 24 }],
    [3, { hiddenSize: 3072, totalLayers: 28 }],
    [7, { hiddenSize: 4096, totalLayers: 32 }],
    [13, { hiddenSize: 5120, totalLayers: 40 }],
    [34, { hiddenSize: 7168, totalLayers: 48 }],
    [70, { hiddenSize: 8192, totalLayers: 80 }],
  ];
  const p = Number(paramsB) || 7;
  let chosen = points[0][1];
  for (const [threshold, shape] of points) {
    if (p >= threshold) {
      chosen = shape;
    }
  }
  return chosen;
}

// Very rough expected decode throughput (tokens/sec) from model size, quant and
// the detected memory bandwidth. Decode is memory-bandwidth bound: tok/s ~=
// effective bandwidth / bytes-read-per-token (~= quantized weight bytes). This
// is an ESTIMATE surfaced with a "~" and an "est" flag - never a benchmark.
export function estimateDecodeTokensPerSec({ weightsGB, hardware }) {
  if (!weightsGB || weightsGB <= 0) {
    return null;
  }
  const hasGpu = (hardware?.primaryGpu?.vramGB ?? 0) > 0;
  const gpuFits = hasGpu && weightsGB <= (hardware.primaryGpu.vramGB ?? 0);
  // Effective sustained bandwidth (GB/s) actually reachable for decode. Coarse:
  // a dedicated GPU that fits the weights is far faster than system RAM.
  const effectiveBandwidthGBs = gpuFits ? 220 : hasGpu ? 60 : 28;
  const tps = effectiveBandwidthGBs / weightsGB;
  return Number(tps.toFixed(tps >= 10 ? 0 : 1));
}

// Rich, human-facing compatibility classification for one model+quant against
// the scanned machine. Returns a single primary label (the spec's vocabulary:
// Recommended / Runs Great / Runs Well / Runs with Reduced Speed / Too Large /
// Requires More RAM), a tone for styling, the numeric estimates the card shows
// (RAM / VRAM / disk / est tok·s), and supporting reasons. All derived from
// metadata - nothing is downloaded.
export function classifyCompatibility({ paramsB, quant, hardware, contextTokens = 4096, hiddenSize, totalLayers, sizeGB } = {}) {
  const shape = approxModelShape(paramsB);
  const estimate = estimateQuantFootprint({
    paramsB: Number(paramsB) || 7,
    quant,
    contextTokens,
    hiddenSize: hiddenSize ?? shape.hiddenSize,
    totalLayers: totalLayers ?? shape.totalLayers,
  });
  // When we know the real GGUF file size, trust it for the weight/disk figure.
  const weightsGB = sizeGB != null ? Number(sizeGB) : estimate.weightsGB;
  const totalGB = Number((weightsGB + estimate.kvCacheGB + RUNTIME_OVERHEAD_GB).toFixed(2));

  const ramBudget = hardware?.memory?.plannerBudgetGB ?? hardware?.memory?.availableGB ?? 8;
  const vramBudget = hardware?.primaryGpu?.vramGB ?? 0;
  const headroom = ramBudget - totalGB;
  const gpuFits = vramBudget > 0 && weightsGB <= vramBudget;
  const tokensPerSec = estimateDecodeTokensPerSec({ weightsGB, hardware });

  const reasons = [];
  let label;
  let tone;
  if (totalGB > ramBudget * 1.15) {
    label = "Too Large";
    tone = "bad";
    reasons.push(`Needs ~${totalGB} GB but only ~${Math.round(ramBudget)} GB is budgeted.`);
  } else if (totalGB > ramBudget) {
    label = "Requires More RAM";
    tone = "warn";
    reasons.push(`Just over budget (~${totalGB} GB vs ~${Math.round(ramBudget)} GB) — close a few apps or pick a smaller quant.`);
  } else if (gpuFits && headroom >= 2) {
    label = "Recommended";
    tone = "good";
    reasons.push(`Weights (~${weightsGB} GB) fit in ${hardware.primaryGpu.name} VRAM and leave RAM headroom.`);
  } else if (gpuFits || headroom >= ramBudget * 0.4) {
    label = "Runs Great";
    tone = "good";
    reasons.push(gpuFits ? "Fits GPU VRAM for fast decode." : "Comfortable RAM headroom.");
  } else if (headroom >= 1.5) {
    label = "Runs Well";
    tone = "ok";
  } else {
    label = "Runs with Reduced Speed";
    tone = "warn";
    reasons.push("Fits, but with little headroom — expect slower decode.");
  }

  if (vramBudget === 0 && Number(paramsB) >= 7 && tone !== "bad") {
    reasons.push("GPU Recommended for interactive speed on a model this size.");
  }

  return {
    label,
    tone,
    ramGB: totalGB,
    weightsGB: Number(weightsGB.toFixed(2)),
    vramGB: vramBudget > 0 ? Number(Math.min(weightsGB, vramBudget).toFixed(2)) : 0,
    gpuFits,
    diskGB: sizeGB != null ? Number(Number(sizeGB).toFixed(2)) : estimate.weightsGB,
    tokensPerSec,
    tokensPerSecEstimated: true,
    reasons,
  };
}

function preferredQuantOrder(mode, model) {
  const executionMode = executionModes[mode] ?? executionModes.balanced;
  const available = new Set(model.quantizations);
  const ordered = executionMode.preferredQuants.filter((quant) => available.has(quant) && model.files[quant]);
  // Fall back to any remaining quants the model offers, highest quality first.
  const remaining = Object.keys(model.files)
    .filter((quant) => !ordered.includes(quant))
    .sort((left, right) => (quantizationProfiles[right]?.qualityScore ?? 0) - (quantizationProfiles[left]?.qualityScore ?? 0));
  return [...ordered, ...remaining];
}

function scoreCandidate({ model, quant, estimate, hardware, task, mode }) {
  const profile = quantizationProfiles[quant] ?? quantizationProfiles.Q4_K_M;
  const ramBudget = hardware?.memory?.plannerBudgetGB ?? hardware?.memory?.availableGB ?? 8;
  const vramBudget = hardware?.primaryGpu?.vramGB ?? 0;

  const ramHeadroom = ramBudget - estimate.totalGB;
  const vramHeadroom = vramBudget > 0 ? vramBudget - estimate.weightsGB : 0;

  // Quality contributes, but the requested mode's preferred quants dominate the
  // quant choice so "balanced" doesn't always max out at the largest build.
  let score = profile.qualityScore * 0.4;
  const preferredQuants = (executionModes[mode] ?? executionModes.balanced).preferredQuants;
  const modeIndex = preferredQuants.indexOf(quant);
  if (modeIndex >= 0) {
    score += 4 - modeIndex; // first preferred quant gets the strongest boost
  }
  if (task && model.tasks.includes(task)) {
    score += 3;
  }
  // Reward fitting comfortably in RAM, penalise overruns hard.
  if (ramHeadroom >= 2) {
    score += 2;
  } else if (ramHeadroom >= 0) {
    score += 0.5;
  } else {
    score += ramHeadroom; // negative -> pushes infeasible models down
  }
  // Reward GPU offload potential.
  if (vramBudget > 0 && vramHeadroom >= 0) {
    score += 1.5;
  }
  // Prefer smaller downloads on equal footing.
  score -= Math.min(estimate.totalGB, 24) * 0.05;

  return Number(score.toFixed(3));
}

function classifyFit(estimate, hardware) {
  const ramBudget = hardware?.memory?.plannerBudgetGB ?? hardware?.memory?.availableGB ?? 8;
  if (estimate.totalGB <= ramBudget) {
    return "fits";
  }
  if (estimate.totalGB <= ramBudget * 1.1) {
    return "tight";
  }
  return "exceeds";
}

function evaluateModel({ model, hardware, task, mode }) {
  const contextTokens = Math.min(model.defaultContext, (executionModes[mode] ?? executionModes.balanced).contextCap);
  const quantOrder = preferredQuantOrder(mode, model);
  const quantOptions = quantOrder.map((quant) => {
    const file = model.files[quant];
    const estimate = estimateQuantFootprint({
      paramsB: model.paramsB,
      quant,
      contextTokens,
      hiddenSize: model.hiddenSize,
      totalLayers: model.totalLayers,
    });
    return {
      quant,
      filename: file.filename,
      sizeGB: file.sizeGB,
      sizeBucket: file.sizeGB != null ? sizeBucketForGB(file.sizeGB).id : null,
      sha256: file.sha256 ?? null,
      estimate,
      fit: classifyFit(estimate, hardware),
      compatibilityScore: scoreCandidate({ model, quant, estimate, hardware, task, mode }),
    };
  });

  // The recommended quant is the highest-scoring option that still fits; if none
  // fit, fall back to the smallest footprint option so the user still sees it.
  const feasible = quantOptions.filter((option) => option.fit !== "exceeds");
  const recommended = (feasible.length > 0 ? feasible : quantOptions)
    .slice()
    .sort((left, right) => right.compatibilityScore - left.compatibilityScore)[0];

  return {
    id: model.id,
    label: model.label,
    family: model.family,
    architecture: model.family,
    hfRepo: model.hfRepo,
    license: model.license,
    paramsB: model.paramsB,
    tasks: model.tasks,
    taskMatch: Boolean(task && model.tasks.includes(task)),
    exactQuantCompatibility: quantOrder.length > 0,
    architectureSupport: ["llama", "mistral", "qwen", "gemma", "gemma4", "phi"].includes(String(model.family).toLowerCase()),
    tokenizerSupport: true,
    enterpriseSuitability: model.license && !/unknown/i.test(model.license) ? "allowlisted" : "review",
    contextTokens,
    recommendedQuant: recommended.quant,
    recommendedFile: recommended.filename,
    fit: recommended.fit,
    compatibilityScore: recommended.compatibilityScore,
    estimate: recommended.estimate,
    sizeGB: recommended.sizeGB,
    sizeBucket: recommended.sizeBucket,
    featured: Boolean(model.featured),
    nativeBackend: Boolean(model.nativeBackend),
    note: model.note ?? null,
    ollamaTag: model.ollamaTag ?? null,
    ollamaTags: Array.isArray(model.ollamaTags) ? model.ollamaTags : [],
    quantOptions,
  };
}

// When browsing a specific size bucket, a model should be represented by
// WHICHEVER of its quants best matches that bucket - not by the single
// mode-driven "recommended" quant, which might sit in a different bucket
// (e.g. gemma4-e4b's Q4_K_M is recommended under "balanced" mode, but a
// user browsing "~9 GB" wants to see it as Q8_0). Returns null if none of
// the model's quants fall in the requested bucket.
function rebaseOnBucketQuant(evaluatedModel, bucketId) {
  const candidates = evaluatedModel.quantOptions.filter((option) => option.sizeBucket === bucketId);
  if (candidates.length === 0) {
    return null;
  }
  const best = candidates.slice().sort((left, right) => right.compatibilityScore - left.compatibilityScore)[0];
  return {
    ...evaluatedModel,
    recommendedQuant: best.quant,
    recommendedFile: best.filename,
    fit: best.fit,
    compatibilityScore: best.compatibilityScore,
    estimate: best.estimate,
    sizeGB: best.sizeGB,
    sizeBucket: best.sizeBucket,
  };
}

export function searchCompatibleGgufModels({ hardware, task, mode = "balanced", sizeBucket } = {}) {
  let models = listAllowlistedHfModels().map((model) => evaluateModel({ model, hardware, task, mode }));
  if (sizeBucket) {
    models = models
      .map((model) => rebaseOnBucketQuant(model, sizeBucket))
      .filter((model) => model !== null);
  }
  // Task relevance still wins first - a featured model that does not cover
  // the requested task should not outrank one that does. "Featured" pins
  // Lizard-native models above other options WITHIN the same relevance tier.
  return models.sort((left, right) => {
    if (left.taskMatch !== right.taskMatch) {
      return left.taskMatch ? -1 : 1;
    }
    if (left.featured !== right.featured) {
      return left.featured ? -1 : 1;
    }
    return right.compatibilityScore - left.compatibilityScore;
  });
}

// Produce a registry-shaped model object so the existing recommendation /
// estimator / benchmark pipeline can plan a downloaded HF GGUF.
export function buildHfRegistryModel(model, { id, quant } = {}) {
  return {
    id: id ?? model.id,
    aliases: model.id && id && id !== model.id ? [model.id] : [],
    family: model.family,
    paramsB: model.paramsB,
    totalLayers: model.totalLayers,
    hiddenSize: model.hiddenSize,
    defaultContext: model.defaultContext,
    maxContext: model.maxContext,
    tasks: model.tasks,
    quantizations: quant ? [quant, ...model.quantizations.filter((entry) => entry !== quant)] : model.quantizations,
    source: "huggingface",
  };
}

export function augmentRegistryWithHfModel(registry, hfModel) {
  const filtered = registry.models.filter((model) => model.id !== hfModel.id);
  return { ...registry, models: [...filtered, hfModel] };
}

export { findAllowlistedHfModel };
