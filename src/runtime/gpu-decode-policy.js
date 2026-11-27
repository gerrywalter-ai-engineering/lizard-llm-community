export function nativeGpuDecodeQuantEligible(value) {
  const normalized = String(value ?? "").trim().toLowerCase();
  return normalized === "f32"
    || normalized === "f16"
    || normalized === "bf16"
    || normalized === "q8_0"
    || normalized === "q4_0"
    || normalized.startsWith("q4_k")
    || normalized.startsWith("q5_k")
    || normalized.startsWith("q6_k")
    || normalized === "iq4_xs"
    || normalized === "iq4_nl";
}

export function gpuDecodePolicyForQuant(value, {
  gpuEnabled = true,
  requireGpu = false,
} = {}) {
  if (!gpuEnabled || !nativeGpuDecodeQuantEligible(value)) {
    return "disabled";
  }
  // Consumer/server paths must remain runnable when a supported GPU has too
  // little dedicated memory for the selected model. "allowed" uses D3D12
  // where profitable and falls back to the portable CPU kernels. Benchmark
  // gates opt into "required" explicitly so a GPU result can never silently
  // become a CPU measurement.
  return requireGpu ? "required" : "allowed";
}
