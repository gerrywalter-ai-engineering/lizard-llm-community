// Phase 3 of .agent/vision/01_native_vision_plan.md: turning a picture into the
// pixel buffer a vision tower expects.
//
// Deliberately outside the native runner. Image decoding is a codec problem,
// not an inference problem, and keeping it here means the runner stays free of
// image libraries and its tower function is pure maths over a float buffer -
// which is what makes that function diffable against a reference.
//
// This is where families diverge hardest, and where being subtly wrong is
// invisible: a mis-normalized image produces a confident, fluent, wrong answer
// with nothing to catch it. So every knob is stated per family rather than
// defaulted, and the resize is exact rather than "close enough".

/**
 * How each family wants its pixels. Values are the mean/std applied per
 * channel AFTER scaling to 0..1.
 *
 *   siglip  : mean 0.5, std 0.5  -> the [-1, 1] range SigLIP was trained on
 *   clip    : OpenAI CLIP's per-channel statistics, which are NOT 0.5
 *
 * Getting these two the wrong way round shifts every patch embedding and is
 * exactly the failure this table exists to prevent.
 */
export const VISION_NORMALIZATIONS = Object.freeze({
  siglip: { mean: [0.5, 0.5, 0.5], std: [0.5, 0.5, 0.5] },
  clip: {
    mean: [0.48145466, 0.4578275, 0.40821073],
    std: [0.26862954, 0.26130258, 0.27577711],
  },
});

/**
 * Picks the normalization for a family spec reported by `vision_probe`.
 *
 * Keyed off the tower's own traits rather than a model name: a gated RMSNorm
 * tower is Gemma-lineage and uses SigLIP statistics; a LayerNorm tower with a
 * soft-embedding norm is gemma-3's SigLIP mmproj; anything else LayerNorm is
 * CLIP-lineage. An unknown combination returns null rather than a guess.
 */
export function normalizationForFamily(family) {
  if (!family) return null;
  if (family.softEmbeddingNorm || family.gatedFfn) return VISION_NORMALIZATIONS.siglip;
  if (family.layerNorm) return VISION_NORMALIZATIONS.clip;
  return null;
}

/**
 * Bilinear resize of an RGB byte buffer to a square edge.
 *
 * Bilinear rather than nearest-neighbour: nearest introduces aliasing that
 * shows up as high-frequency noise in the patch embeddings, which is precisely
 * the kind of small wrongness that survives a smoke test and ruins an answer.
 */
export function resizeRgb(pixels, width, height, edge) {
  if (width <= 0 || height <= 0 || edge <= 0) {
    throw new Error("Image dimensions must be positive.");
  }
  if (pixels.length < width * height * 3) {
    throw new Error(`Pixel buffer holds ${pixels.length} values; ${width}x${height} RGB needs ${width * height * 3}.`);
  }
  const out = new Float32Array(edge * edge * 3);
  // Map destination centres back into source space, so the sampled grid is
  // centred rather than biased toward the top-left corner.
  const scaleX = width / edge;
  const scaleY = height / edge;
  for (let y = 0; y < edge; y += 1) {
    const sourceY = Math.min(height - 1, Math.max(0, (y + 0.5) * scaleY - 0.5));
    const y0 = Math.floor(sourceY);
    const y1 = Math.min(height - 1, y0 + 1);
    const wy = sourceY - y0;
    for (let x = 0; x < edge; x += 1) {
      const sourceX = Math.min(width - 1, Math.max(0, (x + 0.5) * scaleX - 0.5));
      const x0 = Math.floor(sourceX);
      const x1 = Math.min(width - 1, x0 + 1);
      const wx = sourceX - x0;
      for (let c = 0; c < 3; c += 1) {
        const p00 = pixels[(y0 * width + x0) * 3 + c];
        const p01 = pixels[(y0 * width + x1) * 3 + c];
        const p10 = pixels[(y1 * width + x0) * 3 + c];
        const p11 = pixels[(y1 * width + x1) * 3 + c];
        const top = p00 + (p01 - p00) * wx;
        const bottom = p10 + (p11 - p10) * wx;
        out[(y * edge + x) * 3 + c] = top + (bottom - top) * wy;
      }
    }
  }
  return out;
}

/**
 * Full preprocessing: resize to the tower's square, scale to 0..1, then
 * normalize per channel. Returns the flat interleaved RGB buffer the runner's
 * `vision_encode` consumes, alongside the edge it was built for.
 */
export function preprocessForTower(pixels, width, height, { edge, normalization }) {
  if (!Number.isInteger(edge) || edge <= 0) {
    throw new Error("The tower's image size is unknown; cannot preprocess without it.");
  }
  if (!normalization) {
    throw new Error("No normalization is known for this vision family; refusing to guess.");
  }
  const resized = resizeRgb(pixels, width, height, edge);
  const { mean, std } = normalization;
  for (let index = 0; index < resized.length; index += 3) {
    for (let c = 0; c < 3; c += 1) {
      resized[index + c] = (resized[index + c] / 255 - mean[c]) / std[c];
    }
  }
  return { pixels: resized, width: edge, height: edge };
}

/**
 * How many patches a preprocessed frame becomes. Callers need this before
 * encoding: it is how many prompt positions the image will occupy, and the
 * budget planner has to know that before the turn is sized.
 */
export function patchCountForTower(edge, patchSize) {
  if (!patchSize || edge % patchSize !== 0) {
    throw new Error(`A ${edge}px frame does not divide into ${patchSize}px patches.`);
  }
  return (edge / patchSize) ** 2;
}
