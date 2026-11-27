import test from "node:test";
import assert from "node:assert/strict";
import os from "node:os";
import path from "node:path";
import { mkdtempSync, rmSync, writeFileSync, mkdirSync } from "node:fs";
import { resolveOllamaBlob, verifyGgufMagic } from "../src/lib/ollama.js";

test("resolves an ollama manifest blob path", () => {
  const tempRoot = mkdtempSync(path.join(os.tmpdir(), "lizard-ollama-"));
  const modelsDirectory = path.join(tempRoot, "models");
  const manifestDir = path.join(modelsDirectory, "manifests", "registry.ollama.ai", "library", "llama3.2");
  const blobsDir = path.join(modelsDirectory, "blobs");
  mkdirSync(manifestDir, { recursive: true });
  mkdirSync(blobsDir, { recursive: true });
  const digest = "sha256:abc123";
  writeFileSync(path.join(manifestDir, "3b"), JSON.stringify({
    layers: [
      {
        mediaType: "application/vnd.ollama.image.model",
        digest,
      },
    ],
  }));
  writeFileSync(path.join(blobsDir, "sha256-abc123"), Buffer.from("GGUFdemo"));

  const resolved = resolveOllamaBlob("llama3.2:3b", {
    ollama: {
      modelsDirectory,
    },
  });
  assert.equal(resolved.blobPath.endsWith(path.join("blobs", "sha256-abc123")), true);
  assert.equal(verifyGgufMagic(resolved.blobPath), true);
  rmSync(tempRoot, { recursive: true, force: true });
});
