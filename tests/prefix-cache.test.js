// EnhancementRun_04 Phase 5 -- cross-session prefix cache.
//
// Like the block-pool test, the correctness proof lives in C++ and this file's
// job is to put it inside `npm test` and to name the cases so the suite cannot
// quietly shrink. Reimplementing the chained hash in JS would prove only that
// two implementations agree with each other, not that either is right.

import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { spawnSync } from "node:child_process";

const RUNNER = path.resolve("native/lizard-native-runner/build/lizard-native-runner.exe");
const RUNNER_AVAILABLE = fs.existsSync(RUNNER);

function runSelftest() {
  const result = spawnSync(RUNNER, ["--prefix-cache-selftest"], {
    encoding: "utf8",
    timeout: 120_000,
    windowsHide: true,
  });
  const lines = (result.stdout ?? "").split("\n").map((line) => line.trim()).filter(Boolean);
  return {
    status: result.status,
    lines,
    failures: lines.filter((line) => line.startsWith("FAIL")),
    passes: lines.filter((line) => line.startsWith("ok")),
    stderr: result.stderr ?? "",
  };
}

test("prefix cache selftest passes every case", { skip: !RUNNER_AVAILABLE }, () => {
  const result = runSelftest();
  assert.deepEqual(
    result.failures,
    [],
    `prefix cache selftest reported failures:\n${result.failures.join("\n")}\n${result.stderr}`,
  );
  assert.equal(result.status, 0, `selftest exited ${result.status}\n${result.lines.join("\n")}`);
});

test("prefix cache selftest covers the properties the design depends on", {
  skip: !RUNNER_AVAILABLE,
}, () => {
  const covered = runSelftest().passes.join("\n");
  for (const required of [
    // The chain property is what makes "stop at the first miss" correct rather
    // than an approximation; without it the matcher could assemble a prefix
    // out of individually-colliding blocks.
    "a change in block 0 changes the final chain value",
    "the same block hashes differently at a different position",
    "matching stops where the prompts diverge",
    // Evicting an in-use block is a correctness bug, not a performance one:
    // it hands one session another session's context.
    "a block held by a live sequence survives eviction",
    "the live sequence still reads its own bytes after eviction",
    "declining to free an in-use block is recorded",
  ]) {
    assert.ok(covered.includes(required), `selftest no longer covers: ${required}`);
  }
});
