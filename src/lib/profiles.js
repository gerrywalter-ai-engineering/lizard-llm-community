export const quantizationProfiles = {
  TQ1_0: { bitsPerWeight: 1.69, qualityScore: 0, label: "Ternary minimum footprint" },
  IQ1_S: { bitsPerWeight: 1.6, qualityScore: 0, label: "Importance-aware minimum footprint" },
  IQ1_M: { bitsPerWeight: 1.75, qualityScore: 0, label: "Importance-aware minimum footprint" },
  IQ2_XXS: { bitsPerWeight: 2.1, qualityScore: 1, label: "Importance-aware ultra-low memory" },
  IQ2_XS: { bitsPerWeight: 2.3, qualityScore: 1, label: "Importance-aware ultra-low memory" },
  IQ2_S: { bitsPerWeight: 2.5, qualityScore: 1, label: "Importance-aware low memory" },
  IQ2_M: { bitsPerWeight: 2.7, qualityScore: 1, label: "Importance-aware low memory" },
  Q2_K: { bitsPerWeight: 2.6, qualityScore: 1, label: "Emergency low-memory" },
  IQ3_XXS: { bitsPerWeight: 3.1, qualityScore: 2, label: "Importance-aware low memory" },
  IQ3_XS: { bitsPerWeight: 3.3, qualityScore: 2, label: "Importance-aware low memory" },
  IQ3_S: { bitsPerWeight: 3.4, qualityScore: 2, label: "Importance-aware balanced" },
  IQ3_M: { bitsPerWeight: 3.6, qualityScore: 2, label: "Importance-aware balanced" },
  Q3_K_M: { bitsPerWeight: 3.4, qualityScore: 2, label: "Low-memory balanced" },
  // Q4_0 is exactly 18 bytes per 32 weights (fp16 scale + 16 packed bytes).
  Q4_0: { bitsPerWeight: 4.5, qualityScore: 2, label: "Legacy 4-bit" },
  IQ4_XS: { bitsPerWeight: 4.25, qualityScore: 3, label: "Importance-aware balanced" },
  IQ4_NL: { bitsPerWeight: 4.5, qualityScore: 3, label: "Importance-aware balanced" },
  Q4_K_M: { bitsPerWeight: 4.5, qualityScore: 3, label: "Default balanced" },
  Q5_K_M: { bitsPerWeight: 5.5, qualityScore: 4, label: "Higher quality" },
  Q6_K: { bitsPerWeight: 6.4, qualityScore: 5, label: "High quality" },
  Q8_0: { bitsPerWeight: 8, qualityScore: 6, label: "Near full quality" },
  F16: { bitsPerWeight: 16, qualityScore: 7, label: "Half precision" },
  BF16: { bitsPerWeight: 16, qualityScore: 7, label: "Brain floating-point" },
  F32: { bitsPerWeight: 32, qualityScore: 8, label: "Full precision" },
};

export const kvCacheProfiles = {
  q4_0: { bytesPerElement: 0.5, qualityScore: 1 },
  q8_0: { bytesPerElement: 1, qualityScore: 2 },
  f16: { bytesPerElement: 2, qualityScore: 3 },
};

export const executionModes = {
  fast: {
    preferredQuants: ["Q4_K_M", "Q3_K_M", "Q2_K"],
    preferredKvCache: "q4_0",
    contextCap: 4096,
    targetParamsB: 7,
    elasticFallback: false,
  },
  balanced: {
    preferredQuants: ["Q4_K_M", "Q5_K_M", "Q3_K_M"],
    preferredKvCache: "q8_0",
    contextCap: 8192,
    targetParamsB: 10,
    elasticFallback: false,
  },
  quality: {
    preferredQuants: ["Q5_K_M", "Q6_K", "Q4_K_M"],
    preferredKvCache: "f16",
    contextCap: 16384,
    targetParamsB: 14,
    elasticFallback: false,
  },
  elastic: {
    preferredQuants: ["Q3_K_M", "Q2_K", "Q4_K_M"],
    preferredKvCache: "q4_0",
    contextCap: 4096,
    targetParamsB: 14,
    elasticFallback: true,
  },
};

