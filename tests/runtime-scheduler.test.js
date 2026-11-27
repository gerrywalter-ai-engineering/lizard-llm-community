import test from "node:test";
import assert from "node:assert/strict";
import { createRequestScheduler } from "../src/runtime/scheduler/index.js";
import { ExistingBackendRuntime } from "../src/runtime/core/existing-backend-runtime.js";

async function collect(stream) {
  const items = [];
  for await (const item of stream) {
    items.push(item);
  }
  return items;
}

test("request scheduler runs higher priority queued requests first", async () => {
  let current = 0;
  const order = [];
  const scheduler = createRequestScheduler({
    now: () => ++current,
    sleepImpl: async () => {},
  });

  const first = scheduler.enqueue({
    requestId: "low",
    priority: 1,
    task: async () => {
      order.push("low");
      return { output: "low" };
    },
  });
  const second = scheduler.enqueue({
    requestId: "high",
    priority: 5,
    task: async () => {
      order.push("high");
      return { output: "high" };
    },
  });

  const results = await Promise.all([first, second]);
  assert.deepEqual(order, ["high", "low"]);
  assert.deepEqual(results.map((result) => result.output).sort(), ["high", "low"]);
});

test("request scheduler cancels queued requests before execution", async () => {
  let release;
  const blocked = new Promise((resolve) => {
    release = resolve;
  });
  const scheduler = createRequestScheduler();

  const running = scheduler.enqueue({
    requestId: "running",
    priority: 0,
    task: async () => {
      await blocked;
      return { output: "done" };
    },
  });
  const queued = scheduler.enqueue({
    requestId: "queued",
    priority: 0,
    task: async () => ({ output: "never" }),
  });

  assert.equal(scheduler.cancel("queued"), true);
  release();
  await running;
  await assert.rejects(() => queued, /cancelled/i);
});

test("request scheduler enforces timeout errors", async () => {
  const scheduler = createRequestScheduler();
  await assert.rejects(() => scheduler.enqueue({
    requestId: "timeout",
    timeoutMs: 5,
    task: async () => {
      await new Promise((resolve) => setTimeout(resolve, 20));
      return { output: "late" };
    },
  }), /timed out/i);
});

test("runtime shell emits multiple simulated delta events from scheduler chunks", async () => {
  const runtime = new ExistingBackendRuntime({
    generate: async ({ prompt }) => ({
      status: "completed",
      runtime: "llama-cli",
      durationMs: 12,
      output: `${prompt}::scheduler`,
    }),
  });

  const session = await runtime.createSession({
    recommendation: { selected: {} },
    modelArtifact: { modelId: "demo-local", modelPath: "E:\\demo.gguf" },
    profile: {
      contextSize: 128,
      timeoutMs: 1000,
      gguf: {
        blockCount: 4,
        embeddingLength: 128,
        tokenizer: "llama",
      },
      capabilityRegistry: {
        memory: {
          availableGb: 1,
        },
      },
    },
    config: { runtimeDirectory: "E:\\tmp" },
  });

  const events = await collect(session.generateStream({
    requestId: "sched-1",
    prompt: "chunk me",
    metadata: {
      chunkSize: 5,
    },
  }));

  const deltaEvents = events.filter((event) => event.type === "delta");
  assert.ok(deltaEvents.length > 1);
  assert.equal(events.at(-1).metadata.scheduler.chunkCount, deltaEvents.length);
});
