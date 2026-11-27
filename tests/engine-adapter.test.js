import test from "node:test";
import assert from "node:assert/strict";
import { runEnginePlan } from "../src/lib/engine-adapter.js";

function processAlive(pid) {
  try {
    process.kill(Number(pid), 0);
    return true;
  } catch {
    return false;
  }
}

async function waitForExit(child, timeoutMs = 5000) {
  const deadline = Date.now() + timeoutMs;
  while (child?.pid && processAlive(child.pid) && Date.now() < deadline) {
    await new Promise((resolve) => setTimeout(resolve, 20));
  }
  return !child?.pid || !processAlive(child.pid);
}

function longRunningEnginePlan() {
  return {
    exists: true,
    executable: process.execPath,
    args: ["-e", "setInterval(() => {}, 1000)"],
    modelPath: process.execPath,
    displayCommand: `${process.execPath} -e ...`,
  };
}

test("llama.cpp spawn-callback failure is self-cleaned by default", async () => {
  const callbackError = new Error("llama lifecycle registration failed");
  let child = null;
  await assert.rejects(
    runEnginePlan(longRunningEnginePlan(), null, {
      onSpawn(spawned) {
        child = spawned;
        throw callbackError;
      },
    }),
    (error) => error === callbackError,
  );
  assert.equal(await waitForExit(child), true);
});

test("llama.cpp caller-owned spawn-callback failure leaves one captured cleanup owner", async () => {
  const callbackError = new Error("CLI owns failed registration cleanup");
  let child = null;
  try {
    await assert.rejects(
      runEnginePlan(longRunningEnginePlan(), null, {
        callerOwnsCleanup: true,
        onSpawn(spawned) {
          child = spawned;
          throw callbackError;
        },
      }),
      (error) => error === callbackError,
    );
    assert.ok(child?.pid);
    assert.equal(processAlive(child.pid), true);
  } finally {
    if (child?.pid && processAlive(child.pid)) {
      try { process.kill(child.pid, "SIGKILL"); } catch {}
      await waitForExit(child);
    }
  }
});
