// EnhancementRun_04 Phase 4 wiring -- batch coordinator.
//
// Driven against a stub client rather than a real worker: the decisions under
// test are scheduling ones (when to batch, when not to, what to append, when to
// retire), and they must be provable on a machine with no GGUF. The end-to-end
// throughput and stream-equality proof lives in
// scripts/enhancement04-coordinator-bench.mjs.

import test from "node:test";
import assert from "node:assert/strict";
import { createBatchCoordinator } from "../src/runtime/core/batch-coordinator.js";

// Emits `script[sessionId]` one token per step, recording how each call was
// batched so the floor rule can be asserted directly.
function stubClient(script, { calls = [] } = {}) {
  const cursor = new Map();
  return {
    calls,
    async request(method, params) {
      assert.equal(method, "decode_batch");
      const sessionIds = params.sessionIds;
      calls.push([...sessionIds]);
      const tokens = sessionIds.map((sessionId) => {
        const index = cursor.get(sessionId) ?? 0;
        cursor.set(sessionId, index + 1);
        const planned = script[sessionId] ?? [];
        const entry = planned[index] ?? { tokenId: 0, finished: true, finishReason: "max-tokens" };
        return { sessionId, ...entry };
      });
      return { tokens, batchSize: sessionIds.length };
    },
  };
}

function run(count, finished = 3) {
  return Array.from({ length: finished }, (unused, index) => ({
    tokenId: 100 + index,
    token: `t${index}`,
    suppressed: false,
    finished: index === finished - 1,
    finishReason: index === finished - 1 ? "max-tokens" : null,
  })).map((entry) => ({ ...entry, tokenId: entry.tokenId + count * 1000 }));
}

test("coordinator batches once when at or above the floor", async () => {
  const calls = [];
  const client = stubClient({ a: run(0), b: run(1), c: run(2) }, { calls });
  const coordinator = createBatchCoordinator({ client, batchFloor: 3 });

  await Promise.all(["a", "b", "c"].map((sessionId) => coordinator.submit({ sessionId })));

  // Three sequences of three tokens is nine tokens. Batched properly that is
  // three calls, not nine -- the assertion is on the saving, not on an exact
  // count, because the tail of a run legitimately drops below the floor as
  // sequences retire at different steps.
  assert.ok(calls.length < 9, `expected batching to save calls, made ${calls.length}`);
  const batched = calls.filter((call) => call.length === 3);
  assert.ok(batched.length >= 2, `expected batched steps, saw ${JSON.stringify(calls)}`);
  // Never a two-row batch: that is the configuration measured slower than two
  // sequential single-row steps.
  assert.ok(calls.every((call) => call.length !== 2), `two-row batch found: ${JSON.stringify(calls)}`);
});

test("formation window includes sessions that finish cloning in the same HTTP burst", async () => {
  const calls = [];
  const client = stubClient({
    a: run(0, 1),
    b: run(1, 1),
    c: run(2, 1),
  }, { calls });
  const coordinator = createBatchCoordinator({
    client,
    batchFloor: 3,
    batchFormationMs: 50,
  });

  const first = coordinator.submit({ sessionId: "a" });
  await new Promise((resolve) => setTimeout(resolve, 10));
  const rest = [
    coordinator.submit({ sessionId: "b" }),
    coordinator.submit({ sessionId: "c" }),
  ];
  await Promise.all([first, ...rest]);

  assert.deepEqual(calls, [["a", "b", "c"]]);
});

test("formation window resets while a serialized clone burst is still arriving", async () => {
  const calls = [];
  const client = stubClient({
    a: run(0, 1),
    b: run(1, 1),
    c: run(2, 1),
  }, { calls });
  const coordinator = createBatchCoordinator({
    client,
    batchFloor: 3,
    batchFormationMs: 50,
  });

  const submitted = [coordinator.submit({ sessionId: "a" })];
  await new Promise((resolve) => setTimeout(resolve, 30));
  submitted.push(coordinator.submit({ sessionId: "b" }));
  await new Promise((resolve) => setTimeout(resolve, 30));
  submitted.push(coordinator.submit({ sessionId: "c" }));
  await Promise.all(submitted);

  assert.deepEqual(calls, [["a", "b", "c"]]);
});

test("coordinator steps sequences individually below the floor", async () => {
  // The floor is not tuning: B=2 measured SLOWER than two sequential B=1 runs
  // on the target box (4.56 vs 7.11 tok/s), so batching a pair would make the
  // most likely real-world case worse.
  const calls = [];
  const client = stubClient({ a: run(0), b: run(1) }, { calls });
  const coordinator = createBatchCoordinator({ client, batchFloor: 3 });

  await Promise.all(["a", "b"].map((sessionId) => coordinator.submit({ sessionId })));

  assert.ok(calls.length >= 2, "expected per-session calls below the floor");
  for (const call of calls) {
    assert.equal(call.length, 1, `expected single-session calls, saw ${JSON.stringify(call)}`);
  }
});

test("coordinator caps each shared step at the adaptive B=8 target", async () => {
  const calls = [];
  const sessions = Object.fromEntries(
    Array.from({ length: 16 }, (unused, index) => [`s${index}`, run(index, 1)]),
  );
  const coordinator = createBatchCoordinator({
    client: stubClient(sessions, { calls }),
    batchFloor: 3,
  });

  await Promise.all(Object.keys(sessions).map((sessionId) => coordinator.submit({
    sessionId,
    maxBatchSize: 8,
  })));

  assert.equal(calls.length, 2);
  assert.ok(calls.every((call) => call.length === 8), JSON.stringify(calls));
});

test("a suppressed token is not part of the answer", async () => {
  // EOS is consumed, not shown. Appending it would make a batched answer one
  // token longer than the same prompt decoded alone -- the exact failure the
  // first wiring run hit.
  const client = stubClient({
    a: [
      { tokenId: 11, token: "x", suppressed: false, finished: false },
      { tokenId: 99, token: "<eos>", suppressed: true, finished: true, finishReason: "eos" },
    ],
  });
  const coordinator = createBatchCoordinator({ client, batchFloor: 1 });
  const result = await coordinator.submit({ sessionId: "a" });

  assert.deepEqual(result.tokenIds, [11]);
  assert.equal(result.finishReason, "eos");
  assert.equal(result.text, "x");
});

test("token listeners receive the worker token id and metadata", async () => {
  const client = stubClient({
    a: [
      {
        tokenId: 42,
        token: "Ġcache",
        suppressed: false,
        finished: true,
        finishReason: "max-tokens",
      },
    ],
  });
  const seen = [];
  const coordinator = createBatchCoordinator({ client, batchFloor: 1 });
  await coordinator.submit({
    sessionId: "a",
    onToken: (token, emitted) => seen.push({ token, emitted }),
  });

  assert.equal(seen.length, 1);
  assert.equal(seen[0].token, "Ġcache");
  assert.equal(seen[0].emitted.tokenId, 42);
  assert.equal(seen[0].emitted.finishReason, "max-tokens");
});

test("sequences retire independently and resolve with their own tokens", async () => {
  const client = stubClient({
    a: run(0, 2),   // finishes after 2
    b: run(1, 4),   // keeps going
    c: run(2, 4),
  });
  const coordinator = createBatchCoordinator({ client, batchFloor: 3 });

  const [first, second, third] = await Promise.all(
    ["a", "b", "c"].map((sessionId) => coordinator.submit({ sessionId })));

  assert.equal(first.tokenIds.length, 2);
  assert.equal(second.tokenIds.length, 4);
  assert.equal(third.tokenIds.length, 4);
  // No cross-contamination: each sequence carries only its own ids.
  assert.ok(first.tokenIds.every((id) => id < 1000));
  assert.ok(second.tokenIds.every((id) => id >= 1000 && id < 2000));
});

test("a listener that throws does not take down the batch", async () => {
  const client = stubClient({ a: run(0), b: run(1), c: run(2) });
  const coordinator = createBatchCoordinator({ client, batchFloor: 3 });

  const results = await Promise.all([
    coordinator.submit({ sessionId: "a", onToken: () => { throw new Error("listener blew up"); } }),
    coordinator.submit({ sessionId: "b" }),
    coordinator.submit({ sessionId: "c" }),
  ]);

  assert.equal(results.length, 3);
  for (const result of results) {
    assert.equal(result.tokenIds.length, 3);
  }
});

test("a worker failure rejects every in-flight sequence rather than hanging one", async () => {
  const client = {
    async request() { throw new Error("worker died"); },
  };
  const coordinator = createBatchCoordinator({ client, batchFloor: 1 });

  await assert.rejects(() => coordinator.submit({ sessionId: "a" }), /worker died/);
});

test("submitting the same session twice is refused", async () => {
  const client = stubClient({ a: run(0, 6) });
  const coordinator = createBatchCoordinator({ client, batchFloor: 1 });
  const first = coordinator.submit({ sessionId: "a" });
  await assert.rejects(() => coordinator.submit({ sessionId: "a" }), /already being decoded/);
  await first;
});
