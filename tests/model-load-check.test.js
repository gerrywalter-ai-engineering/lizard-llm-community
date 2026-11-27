import test from "node:test";
import assert from "node:assert/strict";

import {
  DEFAULT_LOAD_CHECK_TIMEOUT_MS,
  LOAD_CHECK_FAILURES,
  LOAD_CHECK_PROMPT,
  loadCheckNote,
  runTwoTokenLoadCheck,
} from "../src/lib/model-load-check.js";

test("a model that answers passes, and the check asks for exactly two tokens", async () => {
  const seen = [];
  const verdict = await runTwoTokenLoadCheck({
    generate: async ({ prompt, maxTokens, onDelta }) => {
      seen.push({ prompt, maxTokens });
      onDelta("ok");
      return { output: "ok" };
    },
  });
  assert.equal(verdict.ok, true);
  assert.equal(verdict.reason, null);
  assert.equal(verdict.text, "ok");
  assert.equal(verdict.deltas, 1);
  assert.equal(seen[0].maxTokens, 2);
  assert.equal(seen[0].prompt, LOAD_CHECK_PROMPT);
  assert.equal(typeof verdict.checkedAt, "string");
});

test("a worker that loads but decodes nothing fails, with a note naming the likely cause", async () => {
  const verdict = await runTwoTokenLoadCheck({
    // This is the case the check exists for: ensureModelLoaded returned
    // happily, so everything downstream believed the model was ready.
    generate: async () => ({ output: "   " }),
  });
  assert.equal(verdict.ok, false);
  assert.equal(verdict.reason, LOAD_CHECK_FAILURES.NO_OUTPUT);
  assert.match(verdict.note, /produced no tokens/);
  assert.match(verdict.note, /not supported by the selected engine/);
});

test("a generation that never returns is reported as a timeout, not left hanging", async () => {
  const verdict = await runTwoTokenLoadCheck({
    generate: () => new Promise(() => {}),
    timeoutMs: 40,
  });
  assert.equal(verdict.ok, false);
  assert.equal(verdict.reason, LOAD_CHECK_FAILURES.TIMEOUT);
  assert.match(verdict.message, /within 0 seconds|within \d+ seconds/);
  assert.match(verdict.note, /did not answer a two-token test in time/);
});

test("a thrown error is captured as a verdict rather than propagating", async () => {
  const verdict = await runTwoTokenLoadCheck({
    generate: async () => { throw new Error("scheduler request failed"); },
  });
  assert.equal(verdict.ok, false);
  assert.equal(verdict.reason, LOAD_CHECK_FAILURES.ERROR);
  assert.match(verdict.message, /scheduler request failed/);
  assert.match(verdict.note, /scheduler request failed/);
});

test("partial output streamed before a timeout is kept, so the report is not blank", async () => {
  const verdict = await runTwoTokenLoadCheck({
    generate: ({ onDelta }) => {
      onDelta("o");
      return new Promise(() => {});
    },
    timeoutMs: 40,
  });
  assert.equal(verdict.reason, LOAD_CHECK_FAILURES.TIMEOUT);
  assert.equal(verdict.text, "o");
  assert.equal(verdict.deltas, 1);
});

test("streamed deltas count as an answer even when the result carries no output field", async () => {
  const verdict = await runTwoTokenLoadCheck({
    generate: async ({ onDelta }) => {
      onDelta("hi");
      return {};
    },
  });
  assert.equal(verdict.ok, true);
  assert.equal(verdict.text, "hi");
});

test("a missing generate function is a verdict, never a throw", async () => {
  const verdict = await runTwoTokenLoadCheck({});
  assert.equal(verdict.ok, false);
  assert.equal(verdict.reason, LOAD_CHECK_FAILURES.ERROR);
});

test("the default timeout leaves room for a first-run shader compile", () => {
  assert.ok(
    DEFAULT_LOAD_CHECK_TIMEOUT_MS >= 60_000,
    "a cold first run legitimately spends tens of seconds before the first token",
  );
  assert.match(loadCheckNote(LOAD_CHECK_FAILURES.TIMEOUT), /compiling shaders|thrashing/);
});
