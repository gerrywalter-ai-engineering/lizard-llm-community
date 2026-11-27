import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import {
  parseGgufHeaderForCompat,
  checkHubModelNativeReady,
  createNativeCompatCache,
  mapWithConcurrency,
} from "../src/lib/hf-native-compat.js";

// ---- synthetic GGUF builder -------------------------------------------------

function ggufString(value) {
  const body = Buffer.from(value, "utf8");
  const len = Buffer.alloc(8);
  len.writeBigUInt64LE(BigInt(body.length));
  return Buffer.concat([len, body]);
}

function u32(value) {
  const buf = Buffer.alloc(4);
  buf.writeUInt32LE(value);
  return buf;
}

function u64(value) {
  const buf = Buffer.alloc(8);
  buf.writeBigUInt64LE(BigInt(value));
  return buf;
}

function buildGguf({ architecture, blockCount, tensorNames }) {
  const parts = [Buffer.from("GGUF", "ascii"), u32(3), u64(tensorNames.length), u64(2)];
  parts.push(ggufString("general.architecture"), u32(8), ggufString(architecture));
  parts.push(ggufString(`${architecture}.block_count`), u32(4), u32(blockCount));
  for (const name of tensorNames) {
    parts.push(ggufString(name), u32(2), u64(16), u64(16), u32(0), u64(0));
  }
  return Buffer.concat(parts);
}

function layerNames(blockCount, { skipValueOn = [] } = {}) {
  const names = ["token_embd.weight", "output_norm.weight"];
  for (let layer = 0; layer < blockCount; layer += 1) {
    names.push(`blk.${layer}.attn_q.weight`, `blk.${layer}.attn_k.weight`);
    if (!skipValueOn.includes(layer)) names.push(`blk.${layer}.attn_v.weight`);
  }
  return names;
}

// Range-aware fetch mock: serves slices of `fileBuffer` per the Range header.
function rangedFetchFor(fileBuffer, { onRequest = () => {} } = {}) {
  return async (url, options = {}) => {
    onRequest(url, options);
    const match = /bytes=0-(\d+)/.exec(options.headers?.Range ?? "");
    const end = match ? Number(match[1]) + 1 : fileBuffer.length;
    const slice = fileBuffer.subarray(0, Math.min(end, fileBuffer.length));
    return {
      ok: true,
      status: 206,
      arrayBuffer: async () => slice.buffer.slice(slice.byteOffset, slice.byteOffset + slice.byteLength),
    };
  };
}

// ---- tests ------------------------------------------------------------------

test("parseGgufHeaderForCompat reads architecture, block count, and tensor names", () => {
  const buffer = buildGguf({ architecture: "qwen2", blockCount: 2, tensorNames: layerNames(2) });
  const header = parseGgufHeaderForCompat(buffer);
  assert.equal(header.architecture, "qwen2");
  assert.equal(header.blockCount, 2);
  assert.ok(header.tensorNames.includes("blk.1.attn_v.weight"));
});

test("checkHubModelNativeReady blocks Gemma-4-style value sharing with the pre-flight reason", async () => {
  const buffer = buildGguf({
    architecture: "gemma4",
    blockCount: 12,
    tensorNames: layerNames(12, { skipValueOn: [5, 11] }),
  });
  const verdict = await checkHubModelNativeReady({
    repo: "someone/gemma-4-coder-gguf",
    ggufFiles: [{ filename: "gemma-4-coder-Q4_K_M.gguf", quant: "Q4_K_M", sizeBytes: 1000 }],
    fetchImpl: rangedFetchFor(buffer),
  });
  assert.equal(verdict.ready, false);
  assert.match(verdict.reason, /value projections across layers/i);
});

test("checkHubModelNativeReady passes a complete supported model", async () => {
  const buffer = buildGguf({ architecture: "qwen2", blockCount: 4, tensorNames: layerNames(4) });
  const verdict = await checkHubModelNativeReady({
    repo: "someone/qwen2-gguf",
    ggufFiles: [
      { filename: "qwen2-Q4_K_M.gguf", quant: "Q4_K_M", sizeBytes: 2000 },
      { filename: "qwen2-Q8_0.gguf", quant: "Q8_0", sizeBytes: 4000 },
    ],
    fetchImpl: rangedFetchFor(buffer),
  });
  assert.equal(verdict.ready, true);
  assert.equal(verdict.probedFile, "qwen2-Q4_K_M.gguf");
});

test("checkHubModelNativeReady rejects repos without any natively supported quant, before any fetch", async () => {
  let fetched = 0;
  const verdict = await checkHubModelNativeReady({
    repo: "someone/exotic-quants",
    ggufFiles: [{ filename: "model-IQ2_XXS.gguf", quant: "IQ2_XXS", sizeBytes: 500 }],
    fetchImpl: async () => { fetched += 1; throw new Error("should not fetch"); },
  });
  assert.equal(verdict.ready, false);
  assert.equal(fetched, 0);
  assert.match(verdict.reason, /quantizations/i);
});

test("checkHubModelNativeReady skips split shards and mmproj files when picking the probe", async () => {
  const buffer = buildGguf({ architecture: "qwen2", blockCount: 2, tensorNames: layerNames(2) });
  const probed = [];
  const verdict = await checkHubModelNativeReady({
    repo: "someone/qwen2-gguf",
    ggufFiles: [
      { filename: "model-00001-of-00002.gguf", quant: "Q4_K_M", sizeBytes: 10 },
      { filename: "mmproj-model.gguf", quant: "F16", sizeBytes: 20 },
      { filename: "qwen2-Q4_K_M.gguf", quant: "Q4_K_M", sizeBytes: 2000 },
    ],
    fetchImpl: rangedFetchFor(buffer, { onRequest: (url) => probed.push(url) }),
  });
  assert.equal(verdict.ready, true);
  assert.equal(probed.length, 1);
  assert.match(probed[0], /qwen2-Q4_K_M\.gguf/);
});

test("header fetch escalates the byte range until the tensor directory fits", async () => {
  const buffer = buildGguf({ architecture: "qwen2", blockCount: 8, tensorNames: layerNames(8) });
  const ranges = [];
  // Serve at most 64 bytes on the first (4 MB) request to force escalation.
  const fetchImpl = async (url, options) => {
    const match = /bytes=0-(\d+)/.exec(options.headers?.Range ?? "");
    const requested = match ? Number(match[1]) + 1 : buffer.length;
    ranges.push(requested);
    const cap = ranges.length === 1 ? 64 : buffer.length;
    const slice = buffer.subarray(0, Math.min(requested, cap));
    return {
      ok: true,
      status: 206,
      arrayBuffer: async () => slice.buffer.slice(slice.byteOffset, slice.byteOffset + slice.byteLength),
    };
  };
  const verdict = await checkHubModelNativeReady({
    repo: "someone/qwen2-gguf",
    ggufFiles: [{ filename: "qwen2-Q4_K_M.gguf", quant: "Q4_K_M", sizeBytes: 2000 }],
    fetchImpl,
  });
  assert.equal(verdict.ready, true);
  assert.equal(ranges.length, 2, "expected exactly one escalation");
  assert.ok(ranges[1] > ranges[0]);
});

test("createNativeCompatCache round-trips per revision and invalidates on newer lastModified", () => {
  const cachePath = path.join(fs.mkdtempSync(path.join(os.tmpdir(), "lz-compat-")), "cache.json");
  const cache = createNativeCompatCache(cachePath);
  const verdict = { ready: true, reason: "ok" };
  cache.set("someone/qwen2-gguf", "2026-07-01", verdict);
  assert.deepEqual(cache.get("someone/qwen2-gguf", "2026-07-01"), verdict);
  assert.equal(cache.get("someone/qwen2-gguf", "2026-07-09"), null);
  // A fresh instance reads the persisted file.
  const reloaded = createNativeCompatCache(cachePath);
  assert.deepEqual(reloaded.get("someone/qwen2-gguf", "2026-07-01"), verdict);
  fs.rmSync(path.dirname(cachePath), { recursive: true, force: true });
});

test("mapWithConcurrency preserves order while limiting parallelism", async () => {
  let active = 0;
  let peak = 0;
  const items = [30, 10, 20, 5, 15];
  const results = await mapWithConcurrency(items, 2, async (delay) => {
    active += 1;
    peak = Math.max(peak, active);
    await new Promise((resolve) => setTimeout(resolve, delay));
    active -= 1;
    return delay * 2;
  });
  assert.deepEqual(results, [60, 20, 40, 10, 30]);
  assert.ok(peak <= 2, `peak concurrency ${peak} exceeded limit`);
});
