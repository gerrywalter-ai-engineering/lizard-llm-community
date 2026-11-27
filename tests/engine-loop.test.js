// EnhancementRun_04 Phase 2 -- the step-loop engine.
//
// The phase's claim is that it changed structure and nothing else, so the tests
// are about equivalence rather than about features: the same requests, through
// the same binary, with the flag on and off, must produce the same bytes.
//
// These drive the compiled runner directly over stdio rather than through
// NativeWorkerProcessClient, because the thing under test IS the transport.
// They skip when the binary has not been built, the way the other native tests
// do -- a checkout without a build should not fail the suite.

import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { spawn } from "node:child_process";

const RUNNER = path.resolve("native/lizard-native-runner/build/lizard-native-runner.exe");
const RUNNER_AVAILABLE = fs.existsSync(RUNNER);

// Fields that legitimately differ between two runs of the same request. Timings
// and probe-derived hardware readings are not part of the protocol contract, so
// comparing them would make the equivalence test flaky rather than strict.
const NONDETERMINISTIC_KEYS = new Set([
  "loadMs", "prefillMs", "decodeMs", "elapsedMs", "lastPrefillMs", "lastDecodeMs",
  "gpuPrestageMs", "durationMs", "uptimeMs", "timestamp", "availableGb",
  "availableBytes", "budgetBytes", "freeBytes", "currentUsageBytes",
  "currentReservationBytes", "availableForReservationBytes", "temperatureC",
  "decodeTokensPerSecond", "effectiveWeightGBs",
]);

function stripVolatile(value) {
  if (Array.isArray(value)) {
    return value.map(stripVolatile);
  }
  if (value && typeof value === "object") {
    const result = {};
    for (const [key, item] of Object.entries(value)) {
      if (NONDETERMINISTIC_KEYS.has(key)) continue;
      result[key] = stripVolatile(item);
    }
    return result;
  }
  return value;
}

// Sends every request line, collects one response line per request, and exits.
function runRunner(requests, env = {}) {
  return new Promise((resolve, reject) => {
    const child = spawn(RUNNER, [], {
      stdio: ["pipe", "pipe", "pipe"],
      env: { ...process.env, ...env },
      windowsHide: true,
    });
    let stdout = "";
    let stderr = "";
    const timer = setTimeout(() => {
      child.kill();
      reject(new Error(`runner timed out\nstdout: ${stdout}\nstderr: ${stderr}`));
    }, 60_000);

    child.stdout.on("data", (chunk) => { stdout += chunk.toString(); });
    child.stderr.on("data", (chunk) => { stderr += chunk.toString(); });
    child.on("error", (error) => { clearTimeout(timer); reject(error); });
    child.on("close", () => {
      clearTimeout(timer);
      const lines = stdout.split("\n").map((line) => line.trim()).filter(Boolean);
      resolve({ lines: lines.map((line) => JSON.parse(line)), stderr });
    });

    for (const request of requests) {
      child.stdin.write(`${JSON.stringify(request)}\n`);
    }
    child.stdin.end();
  });
}

const BASIC_REQUESTS = [
  { id: 1, method: "handshake", params: {} },
  { id: 2, method: "probe", params: {} },
  { id: 3, method: "session_open", params: { sessionId: "engine-loop-a" } },
  { id: 4, method: "session_open", params: { sessionId: "engine-loop-b" } },
  { id: 5, method: "metrics", params: {} },
  { id: 6, method: "session_close", params: { sessionId: "engine-loop-a" } },
  { id: 7, method: "session_close", params: { sessionId: "engine-loop-b" } },
];

test("engine loop is opt-in and off by default", { skip: !RUNNER_AVAILABLE }, async () => {
  // With the flag unset the runner must take the synchronous path. Proven by
  // behaviour rather than by reading the env: the loop reports itself in the
  // metrics telemetry, and `running` is false when it never started.
  const { lines } = await runRunner([{ id: 1, method: "metrics", params: {} }]);
  assert.equal(lines.length, 1);
  assert.equal(lines[0].ok, true);
  const engine = lines[0].result?.engineLoop;
  if (engine) {
    assert.equal(engine.running, false, "engine loop must not run unless opted in");
  }
});

test("engine loop answers every request, in order, with matching ids", {
  skip: !RUNNER_AVAILABLE,
}, async () => {
  const { lines } = await runRunner(BASIC_REQUESTS, { LIZARD_NATIVE_ENGINE_LOOP: "1" });
  assert.equal(lines.length, BASIC_REQUESTS.length);
  assert.deepEqual(lines.map((line) => line.id), BASIC_REQUESTS.map((request) => request.id));
  for (const line of lines) {
    assert.equal(line.ok, true, `request ${line.id} failed: ${JSON.stringify(line.error)}`);
  }
});

test("engine loop output is byte-identical to the synchronous path", {
  skip: !RUNNER_AVAILABLE,
}, async () => {
  // The gate for this phase. Both paths call the same process_request_line, so
  // a difference here means the transport changed observable behaviour, which
  // is precisely what the phase promised not to do.
  const synchronous = await runRunner(BASIC_REQUESTS);
  const stepLoop = await runRunner(BASIC_REQUESTS, { LIZARD_NATIVE_ENGINE_LOOP: "1" });

  assert.equal(stepLoop.lines.length, synchronous.lines.length);
  for (let index = 0; index < synchronous.lines.length; index += 1) {
    const expected = stripVolatile(synchronous.lines[index]);
    const actual = stripVolatile(stepLoop.lines[index]);
    // engineLoop telemetry is expected to differ -- it is the one thing the
    // flag is supposed to change.
    delete expected.result?.engineLoop;
    delete actual.result?.engineLoop;
    assert.deepEqual(actual, expected, `response ${index} differs between paths`);
  }
});

test("engine loop reports its telemetry once running", {
  skip: !RUNNER_AVAILABLE,
}, async () => {
  const { lines } = await runRunner([
    { id: 1, method: "handshake", params: {} },
    { id: 2, method: "metrics", params: {} },
  ], { LIZARD_NATIVE_ENGINE_LOOP: "1" });

  const engine = lines[1].result?.engineLoop;
  assert.ok(engine, "metrics must carry engineLoop telemetry when the loop is running");
  assert.equal(engine.running, true);
  // Two requests were dispatched before this response was serialised; the
  // metrics request itself is counted as handled only after it returns, so the
  // observable count is at least one.
  assert.ok(engine.steps >= 2, `expected at least 2 steps, saw ${engine.steps}`);
  assert.ok(engine.commandsHandled >= 1, `expected at least 1 handled, saw ${engine.commandsHandled}`);
});

test("engine loop survives malformed input without dropping later requests", {
  skip: !RUNNER_AVAILABLE,
}, async () => {
  const { lines } = await runRunner([
    { id: 1, method: "handshake", params: {} },
    // A method the runner does not know: it must answer with an error and keep
    // serving, not close the channel.
    { id: 2, method: "not_a_method", params: {} },
    { id: 3, method: "probe", params: {} },
  ], { LIZARD_NATIVE_ENGINE_LOOP: "1" });

  assert.equal(lines.length, 3);
  assert.equal(lines[0].ok, true);
  assert.equal(lines[1].ok, false);
  assert.equal(lines[2].ok, true, "the loop must keep serving after an error");
});
