function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function chunkText(text, chunkSize = 24) {
  const normalized = String(text ?? "");
  if (!normalized) {
    return [];
  }
  const chunks = [];
  for (let index = 0; index < normalized.length; index += chunkSize) {
    chunks.push(normalized.slice(index, index + chunkSize));
  }
  return chunks;
}

function timeoutError(timeoutMs) {
  const error = new Error(`Scheduler request timed out after ${timeoutMs} ms.`);
  error.code = "SCHEDULER_TIMEOUT";
  return error;
}

function cancelledError(requestId) {
  const error = new Error(`Scheduler request ${requestId} was cancelled.`);
  error.code = "SCHEDULER_CANCELLED";
  return error;
}

// EnhancementRun_04 Phase 2.
//
// `maxConcurrent` defaults to 1, which reproduces the previous single-active
// behaviour exactly -- the scheduler held `let active = null` and admitted the
// next request only when that slot emptied. That default is the point: the
// refactor has to be provably behaviour-preserving before any concurrency is
// switched on, and the proof is that every pre-existing scheduler test passes
// unmodified.
//
// Raising it is what lets the runner's step loop see more than one live
// sequence. Until the runner can batch (Phase 4), a higher value only means
// requests queue inside the worker instead of in front of it, so it stays 1.
export function createRequestScheduler({
  now = () => Date.now(),
  sleepImpl = sleep,
  maxConcurrent = 1,
} = {}) {
  const queued = [];
  const activeEntries = new Map();
  const concurrencyLimit = Math.max(1, Number(maxConcurrent) || 1);

  function sortQueue() {
    queued.sort((left, right) => {
      if (left.priority !== right.priority) {
        return right.priority - left.priority;
      }
      return left.enqueuedAt - right.enqueuedAt;
    });
  }

  function describe(entry) {
    return {
      requestId: entry.requestId,
      priority: entry.priority,
      state: entry.state,
      enqueuedAt: entry.enqueuedAt,
      startedAt: entry.startedAt,
    };
  }

  function snapshot() {
    const running = [...activeEntries.values()];
    return {
      // `active` stays a single entry (or null) so existing consumers keep
      // working. At the default limit of 1 it is exactly what it always was;
      // above it, it is the longest-running request and `activeList` carries
      // the rest rather than silently hiding them.
      active: running.length > 0 ? describe(running[0]) : null,
      activeList: running.map(describe),
      maxConcurrent: concurrencyLimit,
      queued: queued.map((item) => ({
        requestId: item.requestId,
        priority: item.priority,
        state: item.state,
        enqueuedAt: item.enqueuedAt,
      })),
    };
  }

  function cancel(requestId) {
    const running = activeEntries.get(requestId);
    if (running) {
      running.cancelled = true;
      running.state = "cancelled";
      return true;
    }
    const index = queued.findIndex((entry) => entry.requestId === requestId);
    if (index === -1) {
      return false;
    }
    const [entry] = queued.splice(index, 1);
    entry.state = "cancelled";
    entry.reject(cancelledError(requestId));
    return true;
  }

  async function runEntry(active) {
    let timeoutHandle = null;

    try {
      const execute = async () => active.task({
        requestId: active.requestId,
        priority: active.priority,
        isCancelled: () => Boolean(active.cancelled),
      });
      const timeoutMs = active.timeoutMs;
      const result = timeoutMs != null
        ? await Promise.race([
          execute(),
          new Promise((_, reject) => {
            timeoutHandle = setTimeout(() => reject(timeoutError(timeoutMs)), timeoutMs);
          }),
        ])
        : await execute();
      if (timeoutHandle) {
        clearTimeout(timeoutHandle);
      }

      if (active.cancelled) {
        throw cancelledError(active.requestId);
      }

      const chunkSize = active.chunkSize ?? 24;
      const chunkDelayMs = active.chunkDelayMs ?? 0;
      const chunks = Array.isArray(result?.chunks) && result.chunks.length
        ? result.chunks.map((chunk) => String(chunk))
        : chunkText(result?.output ?? "", chunkSize);

      for (const chunk of chunks) {
        if (active.cancelled) {
          throw cancelledError(active.requestId);
        }
        if (chunkDelayMs > 0) {
          await sleepImpl(chunkDelayMs);
        }
      }

      active.state = "completed";
      active.resolve({
        ...result,
        chunks,
        scheduler: {
          requestId: active.requestId,
          priority: active.priority,
          queuedMs: active.startedAt - active.enqueuedAt,
          runtimeMs: now() - active.startedAt,
          chunkCount: chunks.length,
        },
      });
    } catch (error) {
      if (timeoutHandle) {
        clearTimeout(timeoutHandle);
      }
      active.state = error.code === "SCHEDULER_CANCELLED" ? "cancelled" : "failed";
      active.reject(error);
    } finally {
      activeEntries.delete(active.requestId);
      queueMicrotask(() => {
        void pump();
      });
    }
  }

  // Admits as many queued requests as the limit allows, in priority order.
  // Each admitted entry runs detached: at the default limit of 1 that is the
  // same serialisation as before, because the next admission cannot happen
  // until the running entry's finally block removes it.
  async function pump() {
    while (activeEntries.size < concurrencyLimit && queued.length > 0) {
      sortQueue();
      const next = queued.shift();
      next.state = "running";
      next.startedAt = now();
      activeEntries.set(next.requestId, next);
      void runEntry(next);
    }
  }

  function enqueue({
    requestId,
    priority = 0,
    timeoutMs = null,
    chunkSize = 24,
    chunkDelayMs = 0,
    task,
  }) {
    if (!requestId) {
      throw new Error("Scheduler enqueue requires a requestId.");
    }
    if (typeof task !== "function") {
      throw new Error("Scheduler enqueue requires a task function.");
    }
    return new Promise((resolve, reject) => {
      queued.push({
        requestId,
        priority,
        timeoutMs,
        chunkSize,
        chunkDelayMs,
        task,
        resolve,
        reject,
        enqueuedAt: now(),
        startedAt: null,
        state: "queued",
        cancelled: false,
      });
      queueMicrotask(() => {
        void pump();
      });
    });
  }

  return {
    enqueue,
    cancel,
    snapshot,
  };
}
