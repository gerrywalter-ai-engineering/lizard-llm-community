import test from "node:test";
import assert from "node:assert/strict";
import { prefillWithSnapshotCache } from "../src/runtime/core/native-cpu-runtime.js";

test("concurrent identical prompts share one prefill and clone isolated target sessions", async () => {
  const calls = {
    open: [],
    prefill: [],
    clone: [],
    close: [],
  };
  const workerClient = {
    async openSession(params) {
      calls.open.push(params);
    },
    async prefill(params) {
      calls.prefill.push(params);
      await new Promise((resolve) => setTimeout(resolve, 10));
      return { prefillMs: 10, cachedTokenCount: params.promptArtifact.tokenIds.length };
    },
    async cloneSession(params) {
      calls.clone.push(params);
      return {
        sessionId: params.sessionId,
        prefillMs: 1,
        cachedTokenCount: 3,
        prefillSnapshotHit: true,
      };
    },
    async closeSession(params) {
      calls.close.push(params);
    },
  };
  const worker = {};
  const promptArtifact = {
    tokenIdMode: "exact-vocab",
    tokenIds: [1, 2, 3],
  };
  const prefillParams = {
    modelId: "demo",
    promptArtifact,
    maxTokens: 8,
    stopSequences: ["</s>"],
  };

  const [first, second] = await Promise.all([
    prefillWithSnapshotCache({
      worker,
      workerClient,
      targetSessionId: "request-a",
      promptArtifact,
      prefillParams,
    }),
    prefillWithSnapshotCache({
      worker,
      workerClient,
      targetSessionId: "request-b",
      promptArtifact,
      prefillParams,
    }),
  ]);

  assert.equal(calls.open.length, 1);
  assert.equal(calls.prefill.length, 1);
  assert.deepEqual(calls.clone.map((call) => call.sessionId).sort(), ["request-a", "request-b"]);
  assert.equal(calls.clone[0].sourceSessionId, calls.clone[1].sourceSessionId);
  assert.equal(first.prefillSnapshotHit, false);
  assert.equal(second.prefillSnapshotHit, true);
  assert.equal(first.sessionId, "request-a");
  assert.equal(second.sessionId, "request-b");
});

test("different token streams create different prefill snapshots", async () => {
  let prefills = 0;
  const workerClient = {
    openSession: async () => {},
    prefill: async () => {
      prefills += 1;
      return { prefillMs: 1 };
    },
    cloneSession: async ({ sessionId }) => ({ sessionId, prefillMs: 1 }),
    closeSession: async () => {},
  };
  const worker = {};
  await prefillWithSnapshotCache({
    worker,
    workerClient,
    targetSessionId: "one",
    promptArtifact: { tokenIdMode: "exact-vocab", tokenIds: [1, 2] },
    prefillParams: { modelId: "demo", maxTokens: 1 },
  });
  await prefillWithSnapshotCache({
    worker,
    workerClient,
    targetSessionId: "two",
    promptArtifact: { tokenIdMode: "exact-vocab", tokenIds: [1, 3] },
    prefillParams: { modelId: "demo", maxTokens: 1 },
  });
  assert.equal(prefills, 2);
});
