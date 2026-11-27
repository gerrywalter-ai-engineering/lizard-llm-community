import { kvCacheProfiles, quantizationProfiles } from "./profiles.js";

const GIB = 1024 ** 3;

function clamp(value, min, max) {
  return Math.max(min, Math.min(value, max));
}

function requireQuantProfile(quant) {
  const profile = quantizationProfiles[String(quant ?? "").toUpperCase()];
  if (!profile) {
    throw new Error(
      `Unsupported quantization profile: ${quant ?? "unknown"}. `
      + "Choose a compatible GGUF quantization in onboarding.",
    );
  }
  return profile;
}

export function estimateGpuLayers({ model, quant, contextTokens, kvCache, hardware, config }) {
  const gpu = hardware.primaryGpu;
  if (!gpu?.vramGB) {
    return 0;
  }

  const quantProfile = requireQuantProfile(quant);
  const kvProfile = kvCacheProfiles[kvCache];
  const weightGB = (model.paramsB * 1e9 * quantProfile.bitsPerWeight / 8) / GIB;
  const kvGB = (contextTokens * model.totalLayers * model.hiddenSize * 2 * kvProfile.bytesPerElement) / GIB;
  const allowedVram = gpu.vramGB * (config.maxVramUsagePercent / 100);
  const reservedGB = 0.6 + (kvGB * 0.65);
  const usableForWeights = Math.max(0, allowedVram - reservedGB);
  const perLayerGB = weightGB / model.totalLayers;
  return clamp(Math.floor(usableForWeights / perLayerGB), 0, model.totalLayers);
}

export function estimateConfiguration({ model, quant, contextTokens, gpuLayers, kvCache, hardware, config }) {
  const quantProfile = requireQuantProfile(quant);
  const kvProfile = kvCacheProfiles[kvCache];
  const gpu = hardware.primaryGpu;

  const weightGB = (model.paramsB * 1e9 * quantProfile.bitsPerWeight / 8) / GIB;
  const runtimeOverheadGB = weightGB * 0.12 + 0.35;
  const kvCacheGB = (contextTokens * model.totalLayers * model.hiddenSize * 2 * kvProfile.bytesPerElement) / GIB;
  const gpuShare = gpu ? clamp(gpuLayers / model.totalLayers, 0, 1) : 0;
  const usesGpu = Boolean(gpu && gpuLayers > 0);

  const vramRequiredGB = usesGpu ? Number((weightGB * gpuShare + kvCacheGB * 0.65 + 0.25).toFixed(2)) : 0;
  const ramRequiredGB = Number((weightGB * (1 - gpuShare) + runtimeOverheadGB + kvCacheGB * (usesGpu ? 0.35 : 1) + 0.3).toFixed(2));
  const allowedRamGB = Number((hardware.memory.totalGB * (config.maxRamUsagePercent / 100)).toFixed(2));
  const allowedVramGB = gpu?.vramGB ? Number((gpu.vramGB * (config.maxVramUsagePercent / 100)).toFixed(2)) : 0;

  const ramHeadroomGB = Number((allowedRamGB - ramRequiredGB).toFixed(2));
  const vramHeadroomGB = Number((allowedVramGB - vramRequiredGB).toFixed(2));
  const fitsRam = ramHeadroomGB >= 0;
  const fitsVram = !gpu || vramHeadroomGB >= 0;
  const fits = fitsRam && fitsVram;
  const tight = fits && (ramHeadroomGB < 1.5 || (gpu && vramHeadroomGB < 0.75));

  return {
    weightGB: Number(weightGB.toFixed(2)),
    runtimeOverheadGB: Number(runtimeOverheadGB.toFixed(2)),
    kvCacheGB: Number(kvCacheGB.toFixed(2)),
    ramRequiredGB,
    vramRequiredGB,
    allowedRamGB,
    allowedVramGB,
    ramHeadroomGB,
    vramHeadroomGB,
    fits,
    tight,
    fitsRam,
    fitsVram,
  };
}
