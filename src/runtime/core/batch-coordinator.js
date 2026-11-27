// EnhancementRun_04 Phase 4 wiring -- continuous batching for the request path.
//
// Phase 4 proved that stepping B sequences through one weight read beats
// llama-server on this hardware (1.531x at B=16). That result was reachable
// only from the benchmarks, because every product request calls the worker's
// single-sequence `generate` and the runner computes it start to finish.
//
// This coordinator closes that gap. Sequences register after their own prefill,
// a single loop advances everything currently registered with one `decode_batch`
// per step, and each sequence resolves when the runner says it is finished.
// Requests join and leave at step boundaries, which is what makes it continuous
// rather than a fixed batch that has to drain.
//
// THE BATCH FLOOR IS NOT AN OPTIMISATION, IT IS A CORRECTNESS-OF-INTENT RULE.
// Measured on the target box: B=1 is 7.11 tok/s, B=2 is 4.56, B=4 is 7.76.
// Batching two sequences is *worse* than running them one after the other,
// because the batched path carries roughly 380 ms of fixed per-step cost that
// only amortises from about three rows upward. A coordinator that batched
// whatever it found would therefore make life worse for exactly the two-user
// case that is most likely in practice. Below the floor it steps each sequence
// individually, which routes to the single-row matvec kernel in the runner.

const DEFAULT_BATCH_FLOOR = 3;
const DEFAULT_BATCH_FORMATION_MS = 100;

export function createBatchCoordinator({
  client,
  batchFloor = DEFAULT_BATCH_FLOOR,
  batchFormationMs = DEFAULT_BATCH_FORMATION_MS,
  onError = null,
} = {}) {
  if (!client) {
    throw new Error("Batch coordinator requires a worker client.");
  }

  const active = new Map();
  let running = false;
  let formationTimer = null;

  function retire(sessionId, finishReason) {
    const entry = active.get(sessionId);
    if (!entry) return;
    active.delete(sessionId);
    entry.resolve({
      sessionId,
      tokenIds: entry.tokenIds,
      text: entry.text,
      finishReason,
      generatedTokenCount: entry.tokenIds.length,
      batchMetrics: {
        batched: true,
        batchSteps: entry.batchSteps,
        maxBatchSize: entry.maxBatchSize,
        maxEffectiveRows: entry.maxEffectiveRows,
        coalescedSessionSteps: entry.coalescedSessionSteps,
        weightBytesAcrossSteps: entry.weightBytesAcrossSteps,
      },
    });
  }

  function failAll(error) {
    for (const [sessionId, entry] of [...active.entries()]) {
      active.delete(sessionId);
      entry.reject(error);
    }
  }

  function applyStep(tokens, stepResult = null) {
    for (const emitted of tokens ?? []) {
      const entry = active.get(emitted.sessionId);
      if (!entry) continue;
      entry.batchSteps += 1;
      entry.maxBatchSize = Math.max(
        entry.maxBatchSize,
        Number(stepResult?.batchSize) || 0,
      );
      entry.maxEffectiveRows = Math.max(
        entry.maxEffectiveRows,
        Number(stepResult?.effectiveRows) || Number(stepResult?.batchSize) || 0,
      );
      entry.coalescedSessionSteps +=
        Number(stepResult?.coalescedSessions) || 0;
      entry.weightBytesAcrossSteps +=
        Number(stepResult?.weightBytesPerStep) || 0;
      // A suppressed token is the runner telling us this one is consumed, not
      // shown -- EOS. Appending it would make a batched answer one token longer
      // than the same prompt decoded alone.
      if (emitted.suppressed) {
        retire(emitted.sessionId, emitted.finishReason ?? "finished");
        continue;
      }
      entry.tokenIds.push(emitted.tokenId);
      if (typeof emitted.token === "string" && emitted.token.length > 0) {
        entry.text += emitted.token;
        // The listener sees the raw token exactly as the single-sequence
        // streaming path does; whitespace decoding and stop-marker trimming
        // stay the caller's job so both paths present identically.
        try {
          entry.onToken?.(emitted.token, emitted);
        } catch {
          // A listener failure must never take down the batch.
        }
      }
      if (emitted.finished) {
        retire(emitted.sessionId, emitted.finishReason ?? "finished");
      }
    }
  }

  async function step() {
    const sessionIds = [...active.keys()];
    if (sessionIds.length === 0) return;
    // Every active request carries the concurrency policy selected for its
    // profile. Use the smallest target while those requests overlap: a B=8
    // memory-qualified turn must never be pulled into a B=16 step merely
    // because another request was admitted with a larger budget. All chunks
    // are stepped once before the next decode round so a second B=8 cohort
    // cannot starve behind the first.
    const targetBatchSize = Math.max(1, Math.min(
      ...sessionIds.map((sessionId) => active.get(sessionId)?.targetBatchSize ?? 16),
    ));
    for (let offset = 0; offset < sessionIds.length; offset += targetBatchSize) {
      const chunk = sessionIds.slice(offset, offset + targetBatchSize)
        .filter((sessionId) => active.has(sessionId));
      if (chunk.length >= batchFloor) {
        const result = await client.request("decode_batch", { sessionIds: chunk });
        applyStep(result?.tokens, result);
        continue;
      }
      // Below the floor, step each sequence on its own. Sequential single-row
      // calls beat a two-row batch on this hardware by a wide margin.
      for (const sessionId of chunk) {
        if (!active.has(sessionId)) continue;
        const result = await client.request("decode_batch", { sessionIds: [sessionId] });
        applyStep(result?.tokens, result);
      }
    }
  }

  async function pump() {
    if (running) return;
    if (formationTimer != null) {
      clearTimeout(formationTimer);
      formationTimer = null;
    }
    running = true;
    try {
      while (active.size > 0) {
        await step();
      }
    } catch (error) {
      onError?.(error);
      failAll(error);
    } finally {
      running = false;
    }
  }

  function schedulePump() {
    if (running || active.size === 0) return;
    const targetBatchSize = Math.max(1, Math.min(
      ...[...active.values()].map((entry) => entry.targetBatchSize),
    ));
    if (active.size >= targetBatchSize || batchFormationMs <= 0) {
      if (formationTimer != null) {
        clearTimeout(formationTimer);
        formationTimer = null;
      }
      queueMicrotask(() => { void pump(); });
      return;
    }
    if (formationTimer != null) {
      clearTimeout(formationTimer);
    }
    // Session clones are serialized through one worker RPC stream. Measure the
    // formation window from the latest arrival, not the first: a B16 burst can
    // take more than 100 ms to finish cloning even though every adjacent clone
    // arrives only a few milliseconds apart. Starting from the first arrival
    // split that burst into two token positions and doubled the physical rows.
    // Reaching the configured B8/B16 target still cancels the wait immediately.
    formationTimer = setTimeout(() => {
      formationTimer = null;
      void pump();
    }, Math.max(0, Number(batchFormationMs) || 0));
  }

  return {
    // Registers an already-prefilled session. Resolves when the runner reports
    // the sequence finished -- EOS, a stop sequence, its token budget, or
    // cancellation. Termination is decided runner-side on purpose: EOS ids and
    // the raw-BPE stop-tail scan are runner facts, and a second implementation
    // here would drift.
    submit({ sessionId, onToken = null, maxBatchSize = 16 } = {}) {
      if (!sessionId) {
        return Promise.reject(new Error("Batch coordinator requires a sessionId."));
      }
      if (active.has(sessionId)) {
        return Promise.reject(new Error(`Session ${sessionId} is already being decoded.`));
      }
      return new Promise((resolve, reject) => {
        active.set(sessionId, {
          tokenIds: [],
          text: "",
          onToken,
          resolve,
          reject,
          batchSteps: 0,
          maxBatchSize: 0,
          maxEffectiveRows: 0,
          coalescedSessionSteps: 0,
          weightBytesAcrossSteps: 0,
          targetBatchSize: Math.max(1, Math.trunc(Number(maxBatchSize) || 16)),
        });
        schedulePump();
      });
    },

    // How many sequences are mid-decode right now. The runtime asks this to
    // decide whether a new request has anything to batch with: joining an
    // occupied coordinator shares the weight read, while opening one for a
    // single sequence just trades one RPC per completion for one per token.
    pendingCount() {
      return active.size;
    },

    snapshot() {
      const targetBatchSize = active.size > 0
        ? Math.min(...[...active.values()].map((entry) => entry.targetBatchSize))
        : null;
      return {
        active: active.size,
        running,
        forming: formationTimer != null,
        batchFloor,
        batchFormationMs,
        targetBatchSize,
      };
    },
  };
}
