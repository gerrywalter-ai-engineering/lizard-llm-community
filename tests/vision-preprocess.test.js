// Phase 3 of .agent/vision/01_native_vision_plan.md.
//
// Preprocessing is where a vision pipeline goes wrong invisibly: the wrong
// normalization or a biased resize still produces a plausible answer, just not
// the right one. These tests pin the arithmetic rather than the vibe.

import test from "node:test";
import assert from "node:assert/strict";

import {
  VISION_NORMALIZATIONS,
  normalizationForFamily,
  resizeRgb,
  preprocessForTower,
  patchCountForTower,
} from "../src/lib/chat/vision-preprocess.js";

test("SigLIP normalization lands on [-1, 1], CLIP's does not", () => {
  // SigLIP's mean/std of 0.5 is what maps 0..255 onto [-1, 1]. CLIP uses real
  // per-channel statistics, and swapping the two is a silent quality loss.
  const { mean, std } = VISION_NORMALIZATIONS.siglip;
  assert.equal((0 / 255 - mean[0]) / std[0], -1);
  assert.equal((255 / 255 - mean[0]) / std[0], 1);
  assert.notDeepEqual(VISION_NORMALIZATIONS.clip.mean, VISION_NORMALIZATIONS.siglip.mean);
});

test("the family spec picks the normalization, and refuses when it cannot", () => {
  // Gemma's inline tower: gated, RMSNorm -> SigLIP statistics.
  assert.deepEqual(
    normalizationForFamily({ gatedFfn: true, layerNorm: false }),
    VISION_NORMALIZATIONS.siglip,
  );
  // gemma-3's mmproj: LayerNorm with a soft-embedding norm -> still SigLIP.
  assert.deepEqual(
    normalizationForFamily({ layerNorm: true, softEmbeddingNorm: true }),
    VISION_NORMALIZATIONS.siglip,
  );
  // A plain LayerNorm tower is CLIP-lineage.
  assert.deepEqual(
    normalizationForFamily({ layerNorm: true }),
    VISION_NORMALIZATIONS.clip,
  );
  // Unknown shape: null, never a guess.
  assert.equal(normalizationForFamily({}), null);
  assert.equal(normalizationForFamily(null), null);
});

test("resizing is centre-sampled, so a symmetric image stays symmetric", () => {
  // 2x2: black, white / white, black. Any correct centred resize keeps that
  // diagonal symmetry; a top-left-biased one does not.
  const pixels = [
    0, 0, 0, 255, 255, 255,
    255, 255, 255, 0, 0, 0,
  ];
  const out = resizeRgb(pixels, 2, 2, 4);
  const at = (x, y) => out[(y * 4 + x) * 3];
  assert.equal(at(0, 0), at(3, 3), "corners on one diagonal must match");
  assert.equal(at(3, 0), at(0, 3), "corners on the other diagonal must match");
  assert.notEqual(at(0, 0), at(3, 0), "and the two diagonals must differ");
});

test("a solid image survives resize and normalization unchanged", () => {
  // Nothing to interpolate, so every output pixel must be the same value -
  // this catches an off-by-one in the sampling grid, which otherwise only
  // shows up as a faint edge artefact.
  const edge = 8;
  const pixels = new Array(4 * 4 * 3).fill(255);
  const { pixels: out, width, height } = preprocessForTower(pixels, 4, 4, {
    edge,
    normalization: VISION_NORMALIZATIONS.siglip,
  });
  assert.equal(width, edge);
  assert.equal(height, edge);
  assert.equal(out.length, edge * edge * 3);
  for (const value of out) {
    assert.ok(Math.abs(value - 1) < 1e-6, `expected 1, got ${value}`);
  }
});

test("preprocessing refuses rather than guessing", () => {
  const pixels = new Array(4 * 4 * 3).fill(0);
  assert.throws(
    () => preprocessForTower(pixels, 4, 4, { edge: 0, normalization: VISION_NORMALIZATIONS.siglip }),
    /image size is unknown/,
  );
  assert.throws(
    () => preprocessForTower(pixels, 4, 4, { edge: 8, normalization: null }),
    /refusing to guess/,
  );
  assert.throws(() => resizeRgb([0, 0, 0], 4, 4, 8), /needs 48/);
});

test("patch count is exact, and a frame that does not divide is refused", () => {
  // 896 / 14 = 64 -> 4096 patches, which is what a full gemma-3 frame costs in
  // prompt positions. Callers budget the turn from this number.
  assert.equal(patchCountForTower(896, 14), 4096);
  assert.equal(patchCountForTower(64, 16), 16);
  assert.throws(() => patchCountForTower(100, 14), /does not divide/);
});
