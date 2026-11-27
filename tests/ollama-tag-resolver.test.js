import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import {
  ensureOllamaModelAvailable,
  matchInstalledOllamaTag,
  modelIdentity,
  ollamaTagCandidates,
  resolveOllamaTag,
} from "../src/lib/ollama-tag-resolver.js";

const registry = {
  models: [
    {
      id: "qwen2.5-coder-7b-instruct",
      aliases: ["qwen-coder"],
      ollamaTag: "qwen2.5-coder:7b",
    },
  ],
};

function writeCatalogFixture(models) {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "lizard-tag-resolver-"));
  const catalogPath = path.join(directory, "model-catalog.json");
  fs.writeFileSync(catalogPath, JSON.stringify({ models }));
  return { catalogPath };
}

const catalogConfig = writeCatalogFixture([
  { id: "qwen2.5-3b", ollamaTag: "qwen2.5:3b" },
  { id: "gemma4-e4b", ollamaTag: "gemma4:e4b" },
  { id: "no-tag-model" },
]);

test("model identity extracts family and parameter size from ids and tags", () => {
  assert.deepEqual(modelIdentity("qwen2.5-3b"), { key: "qwen253b", family: "qwen2.5", sizeB: 3 });
  assert.deepEqual(modelIdentity("qwen2.5:3b"), { key: "qwen253b", family: "qwen2.5", sizeB: 3 });
  assert.equal(modelIdentity("Qwen2.5-3B-Instruct-Q4_K_M").family, "qwen2.5");
  assert.equal(modelIdentity("Qwen2.5-3B-Instruct-Q4_K_M").sizeB, 3);
});

test("candidates come from the direct tag, curated registry, and runtime catalog", () => {
  assert.deepEqual(
    ollamaTagCandidates({ modelRef: "qwen2.5-coder-7b-instruct", registry, config: catalogConfig }),
    [{ tag: "qwen2.5-coder:7b", source: "registry" }],
  );
  assert.deepEqual(
    ollamaTagCandidates({ modelRef: "qwen2.5-3b", registry, config: catalogConfig }),
    [{ tag: "qwen2.5:3b", source: "catalog" }],
  );
  assert.deepEqual(
    ollamaTagCandidates({ modelRef: "llama3.2:3b", registry, config: catalogConfig }),
    [{ tag: "llama3.2:3b", source: "direct" }],
  );
});

test("an installed model matching family and size is reused without a pull", () => {
  const match = matchInstalledOllamaTag("qwen2.5-3b-instruct", ["gemma4:e4b", "qwen2.5:3b"]);
  assert.deepEqual(match, { tag: "qwen2.5:3b", match: "family-size" });
  assert.equal(matchInstalledOllamaTag("qwen2.5-3b", ["qwen2.5:3b"]).match, "exact");
  assert.equal(matchInstalledOllamaTag("mistral-7b", ["qwen2.5:3b"]), null);
});

test("catalog ids resolve to their installed ollama tag without pulling", () => {
  const resolution = ensureOllamaModelAvailable({
    modelRef: "qwen2.5-3b",
    registry,
    config: catalogConfig,
    installedTags: ["qwen2.5:3b", "gemma4:latest"],
    pull: () => {
      throw new Error("pull should not run when the tag is installed");
    },
  });
  assert.equal(resolution.tag, "qwen2.5:3b");
  assert.equal(resolution.installed, true);
  assert.equal(resolution.pulled, false);
});

test("missing models pull the catalog tag instead of the raw catalog id", () => {
  const pulled = [];
  const resolution = ensureOllamaModelAvailable({
    modelRef: "qwen2.5-3b",
    registry,
    config: catalogConfig,
    installedTags: ["gemma4:latest"],
    pull: (tag) => pulled.push(tag),
  });
  assert.deepEqual(pulled, ["qwen2.5:3b"]);
  assert.equal(resolution.tag, "qwen2.5:3b");
  assert.equal(resolution.pulled, true);
});

test("a failed pull falls back to an installed family-size match", () => {
  const resolution = ensureOllamaModelAvailable({
    modelRef: "qwen2.5-3b-instruct",
    registry,
    config: catalogConfig,
    installedTags: ["qwen2.5:3b"],
    pull: () => {
      throw new Error("pull model manifest: file does not exist");
    },
  });
  assert.equal(resolution.tag, "qwen2.5:3b");
  assert.equal(resolution.substituted, true);
});

test("exhausted resolution throws a diagnosable error listing the attempts", () => {
  assert.throws(() => ensureOllamaModelAvailable({
    modelRef: "qwen2.5-3b",
    registry,
    config: catalogConfig,
    installedTags: [],
    pull: () => {
      throw new Error("pull model manifest: file does not exist");
    },
  }), /Unable to resolve an Ollama model for "qwen2\.5-3b".*pull qwen2\.5:3b \(catalog\)/s);
});

test("resolveOllamaTag prefers installed tags and never pulls", () => {
  const installed = resolveOllamaTag({
    modelRef: "qwen2.5-3b",
    registry,
    config: catalogConfig,
    installedTags: ["qwen2.5:3b"],
  });
  assert.deepEqual(installed, { tag: "qwen2.5:3b", source: "catalog", installed: true, substituted: false });

  const uninstalled = resolveOllamaTag({
    modelRef: "qwen2.5-3b",
    registry,
    config: catalogConfig,
    installedTags: [],
  });
  assert.equal(uninstalled.tag, "qwen2.5:3b");
  assert.equal(uninstalled.installed, false);

  assert.equal(resolveOllamaTag({
    modelRef: "unknown-model",
    registry,
    config: catalogConfig,
    installedTags: [],
  }), null);
});
