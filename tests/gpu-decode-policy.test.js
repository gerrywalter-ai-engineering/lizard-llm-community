import assert from "node:assert/strict";
import test from "node:test";
import {
  gpuDecodePolicyForQuant,
  nativeGpuDecodeQuantEligible,
} from "../src/runtime/gpu-decode-policy.js";

test("consumer GPU decode is optional so low-VRAM machines can fall back to CPU", () => {
  assert.equal(nativeGpuDecodeQuantEligible("Q4_K_M"), true);
  assert.equal(nativeGpuDecodeQuantEligible("IQ4_XS"), true);
  assert.equal(gpuDecodePolicyForQuant("Q4_K_M"), "allowed");
});

test("benchmark gates can explicitly require the GPU lane", () => {
  assert.equal(gpuDecodePolicyForQuant("Q4_K_M", { requireGpu: true }), "required");
});

test("CPU-only and unsupported quantizations disable the GPU lane", () => {
  assert.equal(gpuDecodePolicyForQuant("Q4_K_M", { gpuEnabled: false }), "disabled");
  assert.equal(gpuDecodePolicyForQuant("TQ2_0"), "disabled");
});
