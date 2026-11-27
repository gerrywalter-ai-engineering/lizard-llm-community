import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { importOllamaModel, listLocalCatalogModels, resolveExplicitModel } from "../src/lib/model-sources.js";
import { loadModelRegistry } from "../src/lib/model-registry.js";

function createTempConfig() {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "lizard-gemma4-"));
  return {
    root,
    config: {
      modelDirectory: path.join(root, "models"),
      catalogPath: path.join(root, "catalog.json"),
      ollama: {
        modelsDirectory: path.join(root, "ollama"),
        importStrategy: "copy",
        autoPullIfMissing: false,
      },
    },
  };
}

function writeManifestFixture(config, tag) {
  const [name, variant] = tag.split(":");
  const digest = "sha256:gemma4fixture";
  const manifestPath = path.join(
    config.ollama.modelsDirectory,
    "manifests",
    "registry.ollama.ai",
    "library",
    name,
    variant,
  );
  const blobPath = path.join(config.ollama.modelsDirectory, "blobs", digest.replace(":", "-"));

  fs.mkdirSync(path.dirname(manifestPath), { recursive: true });
  fs.mkdirSync(path.dirname(blobPath), { recursive: true });
  fs.writeFileSync(manifestPath, JSON.stringify({
    schemaVersion: 2,
    layers: [
      {
        mediaType: "application/vnd.ollama.image.model",
        digest,
        size: 16,
      },
    ],
  }, null, 2));
  fs.writeFileSync(blobPath, Buffer.concat([Buffer.from("GGUF"), Buffer.alloc(12)]));
}

async function withTempHuggingFaceCache(root, callback) {
  const previous = process.env.HUGGINGFACE_HUB_CACHE;
  process.env.HUGGINGFACE_HUB_CACHE = path.join(root, "hf-cache");
  try {
    return await callback();
  } finally {
    if (previous == null) {
      delete process.env.HUGGINGFACE_HUB_CACHE;
      return;
    }
    process.env.HUGGINGFACE_HUB_CACHE = previous;
  }
}

function writeHuggingFaceSnapshotFixture(root, repo, filename) {
  const snapshotPath = path.join(
    root,
    "hf-cache",
    `models--${repo.replace(/\//g, "--")}`,
    "snapshots",
    "fixture-snapshot",
  );
  fs.mkdirSync(snapshotPath, { recursive: true });
  const filePath = path.join(snapshotPath, filename);
  fs.writeFileSync(filePath, Buffer.concat([Buffer.from("GGUF"), Buffer.alloc(32)]));
  return filePath;
}

test("imports gemma4:e4b into a canonical local model entry and replaces the stale alias catalog entry", async () => {
  const { root, config } = createTempConfig();
  const registry = loadModelRegistry();

  try {
    fs.mkdirSync(path.dirname(config.catalogPath), { recursive: true });
    fs.writeFileSync(config.catalogPath, JSON.stringify({
      models: [
        {
          id: "gemma-2-9b-instruct",
          source: "ollama",
          ollamaTag: "gemma4:latest",
          importedPath: path.join(config.modelDirectory, "gemma-2-9b-instruct.gguf"),
          importMode: "existing",
        },
      ],
    }, null, 2));

    writeManifestFixture(config, "gemma4:e4b");

    const result = await importOllamaModel({
      tag: "gemma4:e4b",
      config,
      registry,
    });

    assert.equal(result.models[0].id, "gemma4-e4b");
    assert.match(result.models[0].importedPath, /gemma4-e4b\.gguf$/);
    assert.equal(fs.existsSync(result.models[0].importedPath), true);

    const catalog = listLocalCatalogModels(config);
    assert.equal(catalog.length, 1);
    assert.equal(catalog[0].id, "gemma4-e4b");
    assert.equal(catalog[0].ollamaTag, "gemma4:e4b");
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test("importing gemma4:e4b preserves unrelated local models", async () => {
  const { root, config } = createTempConfig();
  const registry = loadModelRegistry();

  try {
    fs.mkdirSync(path.dirname(config.catalogPath), { recursive: true });
    fs.writeFileSync(config.catalogPath, JSON.stringify({
      models: [
        {
          id: "llama-3.2-3b-instruct",
          source: "ollama",
          ollamaTag: "llama3.2:3b",
          importedPath: path.join(config.modelDirectory, "llama-3.2-3b-instruct.gguf"),
          importMode: "existing",
        },
        {
          id: "gemma-2-9b-instruct",
          source: "ollama",
          ollamaTag: "gemma4:latest",
          importedPath: path.join(config.modelDirectory, "gemma-2-9b-instruct.gguf"),
          importMode: "existing",
        },
      ],
    }, null, 2));

    writeManifestFixture(config, "gemma4:e4b");

    await importOllamaModel({
      tag: "gemma4:e4b",
      config,
      registry,
    });

    const catalog = listLocalCatalogModels(config).map((entry) => entry.id).sort();
    assert.deepEqual(catalog, ["gemma4-e4b", "llama-3.2-3b-instruct"]);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test("resolveExplicitModel prefers the native Hugging Face Gemma source over a stale local Ollama import", async () => {
  const { root, config } = createTempConfig();
  const registry = loadModelRegistry();

  try {
    fs.mkdirSync(path.dirname(config.catalogPath), { recursive: true });
    fs.writeFileSync(config.catalogPath, JSON.stringify({
      models: [
        {
          id: "gemma4-e4b",
          source: "ollama",
          ollamaTag: "gemma4:e4b",
          importedPath: path.join(config.modelDirectory, "gemma4-e4b.gguf"),
          importMode: "existing",
        },
      ],
    }, null, 2));
    fs.mkdirSync(config.modelDirectory, { recursive: true });
    fs.writeFileSync(path.join(config.modelDirectory, "gemma4-e4b.gguf"), Buffer.concat([Buffer.from("GGUF"), Buffer.alloc(8)]));

    const nativePath = writeHuggingFaceSnapshotFixture(root, "ggml-org/gemma-4-E4B-it-GGUF", "gemma-4-E4B-it-Q4_K_M.gguf");

    await withTempHuggingFaceCache(root, async () => {
      const artifact = await resolveExplicitModel({
        requestedModel: "gemma4:e4b",
        config,
        registry,
        pullIfMissing: false,
      });

      assert.equal(artifact.source, "huggingface-cache");
      assert.equal(artifact.modelPath, nativePath);
      assert.equal(artifact.sourceQuantType, "Q4_K_M");
    });
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test("resolveExplicitModel uses the stricter native Q8 source for Gemma Q5 compression when it is cached", async () => {
  const { root, config } = createTempConfig();
  const registry = loadModelRegistry();

  try {
    writeHuggingFaceSnapshotFixture(root, "ggml-org/gemma-4-E4B-it-GGUF", "gemma-4-E4B-it-Q4_K_M.gguf");
    const q8Path = writeHuggingFaceSnapshotFixture(root, "ggml-org/gemma-4-E4B-it-GGUF", "gemma-4-E4B-it-Q8_0.gguf");

    await withTempHuggingFaceCache(root, async () => {
      const artifact = await resolveExplicitModel({
        requestedModel: "gemma4:e4b",
        config,
        registry,
        pullIfMissing: false,
        compressionLevel: "Q5",
        strictSourceQuant: true,
      });

      assert.equal(artifact.source, "huggingface-cache");
      assert.equal(artifact.modelPath, q8Path);
      assert.equal(artifact.sourceQuantType, "Q8_0");
    });
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test("resolveExplicitModel fails explicitly when a strict native Gemma source is required but not cached", async () => {
  const { root, config } = createTempConfig();
  const registry = loadModelRegistry();

  try {
    writeHuggingFaceSnapshotFixture(root, "ggml-org/gemma-4-E4B-it-GGUF", "gemma-4-E4B-it-Q4_K_M.gguf");

    await withTempHuggingFaceCache(root, async () => {
      await assert.rejects(() => resolveExplicitModel({
        requestedModel: "gemma4:e4b",
        config,
        registry,
        pullIfMissing: false,
        compressionLevel: "Q5",
        strictSourceQuant: true,
      }), /Native llama\.cpp source unavailable/);
    });
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test("resolveExplicitModel downloads the missing strict native Gemma source when pullIfMissing is enabled", async () => {
  const { root, config } = createTempConfig();
  const registry = loadModelRegistry();
  const previousFetch = globalThis.fetch;

  try {
    writeHuggingFaceSnapshotFixture(root, "ggml-org/gemma-4-E4B-it-GGUF", "gemma-4-E4B-it-Q4_K_M.gguf");

    globalThis.fetch = async (url) => {
      assert.match(String(url), /gemma-4-E4B-it-Q8_0\.gguf/);
      return new Response(Buffer.concat([Buffer.from("GGUF"), Buffer.alloc(64)]), {
        status: 200,
        headers: {
          "Content-Type": "application/octet-stream",
        },
      });
    };

    await withTempHuggingFaceCache(root, async () => {
      const artifact = await resolveExplicitModel({
        requestedModel: "gemma4:e4b",
        config,
        registry,
        pullIfMissing: true,
        compressionLevel: "Q5",
        strictSourceQuant: true,
      });

      assert.equal(artifact.source, "huggingface-cache");
      assert.equal(artifact.sourceQuantType, "Q8_0");
      assert.match(artifact.modelPath, /manual-download[\\/]gemma-4-E4B-it-Q8_0\.gguf$/);
      assert.equal(fs.existsSync(artifact.modelPath), true);
    });
  } finally {
    globalThis.fetch = previousFetch;
    fs.rmSync(root, { recursive: true, force: true });
  }
});
