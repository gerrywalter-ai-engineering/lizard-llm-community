import test from "node:test";
import assert from "node:assert/strict";
import {
  resolveNativeConcurrencyPolicy,
  selectMeasuredNativeBatch,
} from "../src/runtime/concurrency-policy.js";

function profile(availableGb, nativeRuntime = {}) {
  return {
    contextSize: 128,
    nativeRuntime,
    capabilityRegistry: { memory: { availableGb } },
    gguf: { blockCount: 1, embeddingLength: 64, contextLength: 4096 },
  };
}

test("native concurrency prefers B=16 and uses the same policy for both providers", () => {
  const lizard = resolveNativeConcurrencyPolicy({
    runtimeProvider: "lizard-native",
    profile: profile(12),
  });
  const caterpillar = resolveNativeConcurrencyPolicy({
    runtimeProvider: "caterpillar",
    profile: profile(12),
  });
  assert.equal(lizard.selectedBatchSize, 16);
  assert.equal(caterpillar.selectedBatchSize, 16);
  assert.deepEqual(caterpillar.fallbackOrder, [16, 8, 4, 1]);
  assert.equal(caterpillar.role, "primary");
});

test("native concurrency selects B=8 before constrained fallbacks", () => {
  const policy = resolveNativeConcurrencyPolicy({ profile: profile(6) });
  assert.equal(policy.selectedBatchSize, 8);
  assert.equal(policy.role, "primary");
  assert.deepEqual(policy.fallbackOrder, [8, 4, 1]);
});

test("B=4 and B=1 are only capacity fallbacks", () => {
  const b4 = resolveNativeConcurrencyPolicy({ profile: profile(3) });
  const b1 = resolveNativeConcurrencyPolicy({ profile: profile(1) });
  assert.equal(b4.selectedBatchSize, 4);
  assert.equal(b4.role, "fallback");
  assert.equal(b1.selectedBatchSize, 1);
  assert.equal(b1.role, "fallback");
});

test("an explicit ceiling is snapped to a supported lane", () => {
  const policy = resolveNativeConcurrencyPolicy({
    profile: profile(16, { preferredBatchSize: 10 }),
  });
  assert.equal(policy.selectedBatchSize, 8);
  assert.equal(policy.capacity.explicitCeiling, 10);
});

test("measured selection chooses the fastest B=16/B=8 and only then fallbacks", () => {
  const selected = selectMeasuredNativeBatch({
    16: { failed: 0, aggregateTokensPerSecond: 28 },
    8: { failed: 0, aggregateTokensPerSecond: 15 },
    4: { failed: 0, aggregateTokensPerSecond: 40 },
  });
  assert.equal(selected.selectedBatchSize, 16);
  assert.equal(selected.role, "primary");

  const fallback = selectMeasuredNativeBatch({
    16: { failed: 1, aggregateTokensPerSecond: null },
    8: { failed: 1, aggregateTokensPerSecond: null },
    4: { failed: 0, aggregateTokensPerSecond: 7 },
    1: { failed: 0, aggregateTokensPerSecond: 5 },
  });
  assert.equal(fallback.selectedBatchSize, 4);
  assert.equal(fallback.role, "fallback");
});

test("runtime policy uses the fastest measured HTTP primary lane within capacity", () => {
  const policy = resolveNativeConcurrencyPolicy({
    profile: profile(12),
    measuredBatchLevels: {
      16: { failed: 0, aggregateTokensPerSecond: 30 },
      8: { failed: 0, aggregateTokensPerSecond: 35 },
      4: { failed: 0, aggregateTokensPerSecond: 90 },
    },
  });
  assert.equal(policy.selectedBatchSize, 8);
  assert.equal(policy.source, "measured-http");
  assert.equal(policy.role, "primary");
  assert.deepEqual(policy.fallbackOrder, [8, 4, 1]);
});

test("a fallback-only measurement cannot make primary admission unreachable", () => {
  const b16 = resolveNativeConcurrencyPolicy({
    profile: profile(12),
    measuredBatchLevels: {
      1: { failed: 0, aggregateTokensPerSecond: 5 },
    },
  });
  assert.equal(b16.selectedBatchSize, 16);
  assert.equal(b16.source, "capacity-policy");
  assert.equal(b16.measuredSelection, null);
  assert.equal(b16.ignoredMeasuredSelection.selectedBatchSize, 1);

  const b8 = resolveNativeConcurrencyPolicy({
    profile: profile(6),
    measuredBatchLevels: {
      4: { failed: 0, aggregateTokensPerSecond: 7 },
      1: { failed: 0, aggregateTokensPerSecond: 5 },
    },
  });
  assert.equal(b8.selectedBatchSize, 8);
  assert.equal(b8.source, "capacity-policy");

  const constrained = resolveNativeConcurrencyPolicy({
    profile: profile(3),
    measuredBatchLevels: {
      4: { failed: 0, aggregateTokensPerSecond: 6 },
      1: { failed: 0, aggregateTokensPerSecond: 7 },
    },
  });
  assert.equal(constrained.selectedBatchSize, 1);
  assert.equal(constrained.source, "measured-http");
});
