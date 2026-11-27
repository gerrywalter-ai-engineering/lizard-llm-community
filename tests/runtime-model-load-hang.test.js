// The "select a model and the app freezes" contract.
//
// Every case here is a step on the path a model selection actually takes, and
// every one of them used to be able to stop forever rather than fail. The
// through-line: a promise that never settles is not an error anyone can catch,
// so a `.catch()` around it is decoration - the caller simply never resumes, and
// with it the dashboard that was waiting on the caller.
import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import process from "node:process";
import {
  NativeWorkerProcessClient,
  resolveRpcTimeoutMs,
} from "../src/runtime/worker/process-client.js";
import { NativeCpuRuntime, assertLoadableModelFile } from "../src/runtime/core/native-cpu-runtime.js";

function tempDir() {
  return fs.mkdtempSync(path.join(os.tmpdir(), "lizard-load-hang-"));
}

// A worker that connects, reads its stdin, and answers nothing - the observable
// behaviour of a runner wedged inside a driver call or mapping a broken file.
// It is deliberately not a crash: a crash was always handled (the child's exit
// rejects everything pending). Silence was not.
function deafWorkerSpec(root) {
  const scriptPath = path.join(root, "deaf-worker.mjs");
  fs.writeFileSync(scriptPath, "process.stdin.resume();\nsetInterval(() => {}, 60000);\n");
  return { command: process.execPath, args: [scriptPath] };
}

function waitForExit(client, timeoutMs = 10000) {
  if (client.child.exitCode != null || client.child.signalCode != null) {
    return Promise.resolve(true);
  }
  return new Promise((resolve) => {
    const timer = setTimeout(() => resolve(false), timeoutMs);
    client.child.once("exit", () => {
      clearTimeout(timer);
      resolve(true);
    });
  });
}

test("a silent worker fails the handshake instead of hanging on it", async () => {
  const root = tempDir();
  const previous = process.env.LIZARD_NATIVE_HANDSHAKE_TIMEOUT_MS;
  process.env.LIZARD_NATIVE_HANDSHAKE_TIMEOUT_MS = "300";
  const client = new NativeWorkerProcessClient({ launchSpec: deafWorkerSpec(root) });
  try {
    const error = await client.handshake().then(
      () => null,
      (rejection) => rejection,
    );
    assert.ok(error, "a worker that never answers must reject, not leave the caller pending");
    assert.equal(error.code, "NATIVE_WORKER_TIMEOUT");
    assert.match(error.message, /handshake/);

    // Rejecting is only half of it. A worker left running holds the model file
    // mapped, which is what turned one bad load into "nothing works until I
    // reboot": the retry could not load and the delete could not unlink.
    assert.equal(await waitForExit(client), true, "the timed-out worker must be stopped, not abandoned");
  } finally {
    process.env.LIZARD_NATIVE_HANDSHAKE_TIMEOUT_MS = previous ?? "";
    if (previous === undefined) delete process.env.LIZARD_NATIVE_HANDSHAKE_TIMEOUT_MS;
    await client.shutdown().catch(() => {});
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test("requests after a timed-out worker fail immediately rather than queueing", async () => {
  const root = tempDir();
  const previous = process.env.LIZARD_NATIVE_HANDSHAKE_TIMEOUT_MS;
  process.env.LIZARD_NATIVE_HANDSHAKE_TIMEOUT_MS = "300";
  const client = new NativeWorkerProcessClient({ launchSpec: deafWorkerSpec(root) });
  try {
    await client.handshake().catch(() => {});
    // Without this, every later call joined the same silence and the failure
    // spread from one load to the whole app.
    await assert.rejects(() => client.loadModel({}), /closed/i);
  } finally {
    process.env.LIZARD_NATIVE_HANDSHAKE_TIMEOUT_MS = previous ?? "";
    if (previous === undefined) delete process.env.LIZARD_NATIVE_HANDSHAKE_TIMEOUT_MS;
    await client.shutdown().catch(() => {});
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test("shutdown returns once a deaf worker is gone, not once it agrees to go", async () => {
  const root = tempDir();
  const previous = process.env.LIZARD_NATIVE_SHUTDOWN_TIMEOUT_MS;
  process.env.LIZARD_NATIVE_SHUTDOWN_TIMEOUT_MS = "300";
  const client = new NativeWorkerProcessClient({ launchSpec: deafWorkerSpec(root) });
  try {
    // This await used to be unbounded, so the `finally` that killed the child
    // never ran. Unload, Delete, releaseModelWorker and the SIGINT handler all
    // funnel through here - which is how a single wedged worker made the whole
    // app unreachable and left its port bound on the way out.
    await client.shutdown();
    assert.ok(
      client.child.exitCode != null || client.child.signalCode != null,
      "shutdown must not report success while the worker is still running",
    );
  } finally {
    process.env.LIZARD_NATIVE_SHUTDOWN_TIMEOUT_MS = previous ?? "";
    if (previous === undefined) delete process.env.LIZARD_NATIVE_SHUTDOWN_TIMEOUT_MS;
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test("only lifecycle calls carry a deadline", () => {
  assert.ok(resolveRpcTimeoutMs("load_model") > 0);
  assert.ok(resolveRpcTimeoutMs("handshake") > 0);
  // A long generation on a slow CPU is legitimately slow. Bounding it would
  // turn a working answer into a failure, which is the opposite of the point.
  assert.equal(resolveRpcTimeoutMs("generate"), 0);
  assert.equal(resolveRpcTimeoutMs("decode"), 0);
  assert.equal(resolveRpcTimeoutMs("prefill"), 0);
});

test("a load deadline can be raised, and lifted, from the environment", () => {
  assert.equal(resolveRpcTimeoutMs("load_model", { LIZARD_NATIVE_LOAD_TIMEOUT_MS: "1200000" }), 1200000);
  // A deliberately slow run - a first-ever launch compiling shaders, a model on
  // a network drive - can opt out rather than being forced to fail.
  assert.equal(resolveRpcTimeoutMs("load_model", { LIZARD_NATIVE_LOAD_TIMEOUT_MS: "0" }), 0);
});

test("unloading during a slow load does not wait for the load", async () => {
  let shutdownCalls = 0;
  const runtime = new NativeCpuRuntime({
    // A load that never returns. Releasing used to await the cached load
    // promise first, so Unload and Delete did nothing for as long as the load
    // ran - forever, when the load was the thing that had wedged.
    workerClientFactory: () => ({
      handshake: () => new Promise(() => {}),
      loadModel: () => new Promise(() => {}),
      shutdown: async () => {
        shutdownCalls += 1;
      },
    }),
  });
  const modelArtifact = { modelId: "stuck-model", modelPath: path.join(tempDir(), "stuck.gguf") };

  const load = runtime.ensureModelLoaded({ modelArtifact, recommendation: null });
  load.catch(() => {});

  await runtime.releaseModelWorker(modelArtifact);
  assert.equal(shutdownCalls, 1, "the worker process must be stopped even mid-load");
  assert.equal(runtime._workers.size, 0, "and the cache entry must not survive the release");
});

test("a failed load does not poison the next attempt", async () => {
  let attempts = 0;
  const runtime = new NativeCpuRuntime({
    workerClientFactory: () => {
      attempts += 1;
      const failing = attempts === 1;
      return {
        handshake: async () => ({}),
        loadModel: async () => {
          if (failing) throw new Error("load failed");
          return { executionReady: true };
        },
        shutdown: async () => {},
      };
    },
  });
  const modelArtifact = { modelId: "retry-model", modelPath: path.join(tempDir(), "retry.gguf") };

  await assert.rejects(() => runtime.ensureModelLoaded({ modelArtifact, recommendation: null }));
  assert.equal(runtime._workers.size, 0, "a failed load must not stay cached as the answer");

  // Retrying is the first thing a user does. It has to be able to succeed.
  await runtime.ensureModelLoaded({ modelArtifact, recommendation: null });
  assert.equal(attempts, 2, "the retry must build a new worker rather than rejoin the failed one");
});

test("a leftover model file is named, not loaded", () => {
  const root = tempDir();
  const missing = path.join(root, "gone.gguf");
  const truncated = path.join(root, "truncated.gguf");
  const corrupt = path.join(root, "corrupt.gguf");
  const good = path.join(root, "good.gguf");
  fs.writeFileSync(truncated, "GG");
  fs.writeFileSync(corrupt, Buffer.concat([Buffer.from("ZIP\0"), Buffer.alloc(4096)]));
  fs.writeFileSync(good, Buffer.concat([Buffer.from("GGUF"), Buffer.alloc(4096)]));

  // These are the states a reinstall leaves behind: a catalog entry whose file
  // a previous uninstall took, and a file that is not what it claims to be.
  // Handed to a real worker, both wedge it - so they are refused before one is
  // spawned, with the path and the fix in the message.
  assert.throws(
    () => assertLoadableModelFile({ modelId: "m", modelPath: missing }),
    (error) => error.code === "MODEL_FILE_MISSING" && error.message.includes(missing),
  );
  assert.throws(
    () => assertLoadableModelFile({ modelId: "m", modelPath: truncated }),
    (error) => error.code === "MODEL_FILE_INCOMPLETE",
  );
  assert.throws(
    () => assertLoadableModelFile({ modelId: "m", modelPath: corrupt }),
    (error) => error.code === "MODEL_FILE_CORRUPT",
  );

  assert.doesNotThrow(() => assertLoadableModelFile({ modelId: "m", modelPath: good }));
  // Prototype artifacts have no GGUF on disk to vouch for; the check is about
  // what the compiled worker will be asked to map, and must not invent a
  // requirement for models that never go near it.
  assert.doesNotThrow(() => assertLoadableModelFile({ modelId: "m", modelPath: path.join(root, "p.bin") }));
  assert.doesNotThrow(() => assertLoadableModelFile({ modelId: "m", modelPath: null }));

  fs.rmSync(root, { recursive: true, force: true });
});
