import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { inspectGgufFile } from "../src/lib/gguf-metadata.js";
import { NativeWorkerProcessClient } from "../src/runtime/worker/process-client.js";
import {
  NATIVE_FAMILY_FIXTURES,
  compiledWorkerPath,
  writeFamilyGguf,
} from "./native-gguf-fixtures.js";

function tempDir() {
  return fs.mkdtempSync(path.join(os.tmpdir(), "lizard-native-family-"));
}

async function runFamilyGeneration(binary, family) {
  const root = tempDir();
  const modelPath = path.join(root, `${family}.gguf`);
  writeFamilyGguf(modelPath, family);
  const gguf = inspectGgufFile(modelPath);

  const client = new NativeWorkerProcessClient({
    launchSpec: {
      command: binary,
      args: [],
    },
  });
  try {
    const handshake = await client.handshake();
    assert.equal(handshake.worker, "lizard-native-runner-cpp");

    const loaded = await client.loadModel({
      modelArtifact: {
        modelId: `family-${family}`,
        modelPath,
      },
      profile: {
        mmap: true,
        mlock: false,
        gguf,
      },
    });
    assert.equal(loaded.executionReady, true, `${family} fixture must be execution-ready: ${JSON.stringify(loaded.compatibility?.executor ?? null)}`);
    assert.equal(loaded.nativeModelStatus, `${family}-graph-ready`);
    assert.equal(loaded.compatibility.architecture.status, "supported");
    assert.equal(loaded.compatibility.executor.status, "supported");

    await client.openSession({
      sessionId: `family-${family}`,
      modelId: `family-${family}`,
    });
    const prefill = await client.prefill({
      sessionId: `family-${family}`,
      prompt: "hello world",
      promptArtifact: {
        renderedPrompt: "hello world",
        tokenIds: [0, 1, 2, 3],
        tokenIdMode: "exact-vocab",
      },
      kvPlan: {
        estimatedBytes: 0,
      },
      maxTokens: 4,
    });
    assert.equal(prefill.nativeModelStatus, `${family}-graph-prefilled`);
    assert.equal(prefill.promptTokenCount, 4);
    assert.equal(prefill.cachedTokenCount, 4);
    assert.ok(Array.isArray(prefill.embeddingPreview) && prefill.embeddingPreview.length > 0);

    const generated = await client.generate({
      sessionId: `family-${family}`,
      maxTokens: 4,
      prefillMs: prefill.prefillMs,
    });
    assert.equal(generated.runtimeProvider, "lizard-native");
    assert.equal(generated.nativeModelStatus, `${family}-graph-executed`);
    assert.equal(generated.family, family);
    // The decode loop ends at the GGUF EOS (token 4 here), so a greedy
    // synthetic model may stop before the budget; it must never exceed it.
    assert.ok(generated.generatedTokenIds.length <= 4, `${family} generated ${generated.generatedTokenIds.length} tokens`);
    for (const tokenId of generated.generatedTokenIds) {
      assert.ok(Number.isInteger(tokenId) && tokenId >= 0 && tokenId < 5, `${family} produced token id ${tokenId}`);
    }
    for (const value of generated.logitsPreview) {
      assert.ok(Number.isFinite(value), `${family} produced non-finite logits`);
    }

    // Determinism: a second identical session must reproduce the tokens.
    await client.openSession({
      sessionId: `family-${family}-repeat`,
      modelId: `family-${family}`,
    });
    const repeatPrefill = await client.prefill({
      sessionId: `family-${family}-repeat`,
      prompt: "hello world",
      promptArtifact: {
        renderedPrompt: "hello world",
        tokenIds: [0, 1, 2, 3],
        tokenIdMode: "exact-vocab",
      },
      kvPlan: {
        estimatedBytes: 0,
      },
      maxTokens: 4,
    });
    const repeated = await client.generate({
      sessionId: `family-${family}-repeat`,
      maxTokens: 4,
      prefillMs: repeatPrefill.prefillMs,
    });
    assert.deepEqual(repeated.generatedTokenIds, generated.generatedTokenIds);
    return generated;
  } finally {
    await client.shutdown().catch(() => {});
  }
}

for (const family of NATIVE_FAMILY_FIXTURES) {
  test(`compiled native worker executes the ${family} GGUF family natively`, async (t) => {
    const binary = compiledWorkerPath();
    if (!binary) {
      t.skip("compiled native worker binary is not present in build/ or build-cmake/");
      return;
    }
    await runFamilyGeneration(binary, family);
  });
}

test("compiled native worker keeps family-spec GGUFs out of the plain llama graph", async (t) => {
  const binary = compiledWorkerPath();
  if (!binary) {
    t.skip("compiled native worker binary is not present in build/ or build-cmake/");
    return;
  }
  const generated = await runFamilyGeneration(binary, "qwen2");
  // qwen2 must report its own graph, not the llama one it would otherwise
  // silently (and wrongly) execute through.
  assert.notEqual(generated.nativeModelStatus, "llama-graph-executed");
});

test("compiled native worker reports deepseek2 as blocked with the exact MLA reason", async (t) => {
  const binary = compiledWorkerPath();
  if (!binary) {
    t.skip("compiled native worker binary is not present in build/ or build-cmake/");
    return;
  }
  const root = tempDir();
  const modelPath = path.join(root, "deepseek2.gguf");
  writeFamilyGguf(modelPath, "deepseek2");
  const gguf = inspectGgufFile(modelPath);

  const client = new NativeWorkerProcessClient({
    launchSpec: {
      command: binary,
      args: [],
    },
  });
  try {
    await client.handshake();
    let loadError = null;
    let loaded = null;
    try {
      loaded = await client.loadModel({
        modelArtifact: {
          modelId: "family-deepseek2",
          modelPath,
        },
        profile: {
          mmap: true,
          mlock: false,
          gguf,
        },
      });
    } catch (error) {
      loadError = error;
    }
    // Whether load rejects or reports non-ready, the surfaced compatibility
    // must carry the explicit blocked reason - never a silent switch.
    const compatibility = loaded?.compatibility ?? loadError?.details?.compatibility ?? null;
    assert.ok(compatibility, "deepseek2 load must surface compatibility details");
    assert.equal(loaded?.executionReady ?? false, false);
    assert.equal(compatibility.architecture.status, "blocked");
    assert.match(compatibility.architecture.reason, /Multi-head Latent Attention/);
    assert.match(compatibility.architecture.reason, /will be updated soon/);
  } finally {
    await client.shutdown().catch(() => {});
  }
});
