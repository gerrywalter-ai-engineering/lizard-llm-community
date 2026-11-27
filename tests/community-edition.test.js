import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import { assertProPlanFeatureAllowed, getCreditState, PRO_OFFER } from "../src/lib/edition.js";
import { listRuntimeProviders } from "../src/lib/runtime-providers.js";
import { reportMajorFailure } from "../src/lib/major-failure-intelligence.js";
import { translateOllamaStyleArgv } from "../src/lib/ollama-style-cli.js";

test("community edition never gates a feature", () => {
  assert.equal(assertProPlanFeatureAllowed({}, "Native API server").allowed, true);
  assert.equal(getCreditState().edition, "community");
  assert.equal(PRO_OFFER.price, "€5");
  assert.ok(PRO_OFFER.includes.length >= 3);
});

test("only the two open-source native engines are offered", () => {
  const ids = listRuntimeProviders().map((provider) => provider.id);
  assert.ok(ids.includes("lizard-native"));
  assert.ok(ids.includes("caterpillar"));
  assert.ok(!ids.includes("ostrich"));
});

test("a major failure is recorded locally and never sent anywhere", async () => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "lizard-community-incidents-"));
  let fetchCalls = 0;
  const config = { failureIntelligence: { enabled: true, directory, repeatThreshold: 3, repeatWindowMs: 60_000 } };
  const error = Object.assign(new Error("model file is corrupt"), { code: "GGUF_CORRUPT" });
  const result = await reportMajorFailure({
    error,
    config,
    context: { component: "test", unrecoverable: true },
    fetchImpl: async () => { fetchCalls += 1; throw new Error("network must not be used"); },
  });
  assert.equal(fetchCalls, 0);
  if (result.reported) {
    assert.equal(result.status, "recorded_locally");
    assert.equal(result.communityUrl, null);
  }
  fs.rmSync(directory, { recursive: true, force: true });
});

test("ollama-style verbs map onto the core commands", () => {
  assert.deepEqual(translateOllamaStyleArgv(["ls"]), ["models", "list", "--source", "local"]);
  assert.deepEqual(translateOllamaStyleArgv(["ps"]), ["model", "status"]);
  assert.deepEqual(translateOllamaStyleArgv(["pull", "ollama:llama3.2"]), ["models", "pull", "--ollama", "llama3.2"]);
  assert.deepEqual(translateOllamaStyleArgv(["serve", "--engine", "caterpillar"]), ["serve", "--provider", "caterpillar"]);
  assert.deepEqual(translateOllamaStyleArgv(["app", "--no-open"]), ["serve-ui"]);
});
