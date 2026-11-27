// EnhancementRun_04 Phase 3 -- paged KV block pool.
//
// The pool's correctness is proven in C++ (`--kv-blocks-selftest`) rather than
// reimplemented here, because the thing under test is the memory layout the
// attention kernels read, and a JS mirror of it would prove only that two
// implementations agree with each other.
//
// This test's job is to make that selftest part of `npm test`, and to surface
// which case failed rather than a bare exit code -- a paged-KV failure is the
// kind that otherwise shows up much later as fluent, confident, wrong text.

import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { spawnSync } from "node:child_process";

const RUNNER = path.resolve("native/lizard-native-runner/build/lizard-native-runner.exe");
const RUNNER_AVAILABLE = fs.existsSync(RUNNER);

function runSelftest() {
  const result = spawnSync(RUNNER, ["--kv-blocks-selftest"], {
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

test("paged KV block pool selftest passes every case", { skip: !RUNNER_AVAILABLE }, () => {
  const result = runSelftest();
  assert.deepEqual(
    result.failures,
    [],
    `paged KV selftest reported failures:\n${result.failures.join("\n")}\n${result.stderr}`,
  );
  assert.equal(result.status, 0, `selftest exited ${result.status}\n${result.lines.join("\n")}`);
});

test("paged KV selftest covers the cases the phase gate names", { skip: !RUNNER_AVAILABLE }, () => {
  // Guards against the suite quietly shrinking: a selftest that stops running
  // the bit-identity or copy-on-write case would still exit 0 and would still
  // look green above.
  const result = runSelftest();
  const covered = result.passes.join("\n");
  for (const required of [
    "bit-identical to contiguous, block_tokens=8",
    "bit-identical to contiguous, block_tokens=16",
    "bit-identical to contiguous, block_tokens=32",
    "out-of-range read returns null",
    "donor is unchanged after the borrower writes",
    "oversized request refused as TooLarge",
    "fuzz keeps the pool accounting consistent",
    "releasing every sequence empties the pool",
  ]) {
    assert.ok(covered.includes(required), `selftest no longer covers: ${required}`);
  }
});
