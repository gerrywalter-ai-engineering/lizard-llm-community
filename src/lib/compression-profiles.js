export const compressionProfiles = [
  {
    // "Q1_0" was a placeholder that does not exist in llama.cpp, so every Q1
    // row could never quantize and silently fell back to Ollama. Q4_0 is the
    // real legacy baseline quant and runs natively end to end.
    level: "Q1",
    quantType: "Q4_0",
    family: "legacy",
    strategy: "legacy round-to-nearest 4-bit baseline",
    expectedUse: "legacy-format baseline for comparing modern K-Quants at similar size",
  },
  {
    level: "Q2",
    quantType: "Q2_K",
    family: "k-quant",
    strategy: "very aggressive K-Quant",
    expectedUse: "lowest-memory comparison using established llama.cpp K-Quant",
  },
  {
    level: "Q3",
    quantType: "Q3_K_M",
    family: "k-quant",
    strategy: "balanced low-memory K-Quant",
    expectedUse: "practical low-memory benchmark candidate",
  },
  {
    level: "Q4",
    quantType: "Q4_K_M",
    family: "k-quant",
    strategy: "default balanced K-Quant",
    expectedUse: "baseline app-side quality and memory tradeoff",
  },
  {
    level: "Q5",
    quantType: "Q5_K_M",
    family: "k-quant",
    strategy: "higher-quality K-Quant",
    expectedUse: "quality-biased local inference",
  },
  {
    level: "Q6",
    quantType: "Q6_K",
    family: "k-quant",
    strategy: "high-quality K-Quant",
    expectedUse: "quality-first local inference when memory allows",
  },
  {
    level: "Q7",
    quantType: "IQ4_XS",
    family: "i-quant",
    strategy: "non-linear IQ compact quality candidate",
    expectedUse: "Ollama-parity candidate with better compression than common Q4_K_M-style baselines",
  },
  {
    // Q8 used to map to TQ2_0 (2-bit ternary): CPU-only, degraded output by
    // design, and nothing like what "Q8" suggests. Q8_0 is the real 8-bit
    // quant, runs on the native resident GPU decode lane, and gives the
    // benchmark a true quality-ceiling row below F16.
    level: "Q8",
    quantType: "Q8_0",
    family: "legacy",
    strategy: "8-bit round-to-nearest reference quality",
    expectedUse: "near-lossless quality ceiling for judging what lower quants lose; GPU-resident decode",
  },
  {
    level: "Q9",
    quantType: "TQ1_0",
    family: "ternary",
    strategy: "maximum experimental ternary compression",
    expectedUse: "smallest no-imatrix advanced comparison; not expected to match Ollama quality on all prompts",
  },
];

export function getCompressionProfiles() {
  return compressionProfiles.map((profile) => ({ ...profile }));
}

export function getCompressionProfile(level) {
  return compressionProfiles.find((profile) => profile.level === String(level).toUpperCase()) ?? null;
}

export function getCompressionLevels() {
  return compressionProfiles.map((profile) => profile.level);
}

export function getCompressionQuantMap() {
  return Object.fromEntries(compressionProfiles.map((profile) => [profile.level, profile.quantType]));
}
