// Native-readiness pre-flight for live Hugging Face Hub models.
//
// "Lizard-native ready" means the model would pass the same checks the local
// runtime-provider pre-flight applies after download: a supported GGUF
// architecture family, at least one downloadable file in a natively supported
// quantization, and a complete per-layer tensor directory (no cross-layer
// value sharing gaps - see summarizeAttentionValueGaps).
//
// The tensor directory lives in the GGUF header, before any tensor data, so
// the verdict comes from a ranged fetch of the file's first megabytes - the
// model is never downloaded. Verdicts are cached on disk keyed by
// repo + lastModified so each repo is inspected once per revision.

import fs from "node:fs";
import path from "node:path";
import { evaluateNativeGgufCompatibility, getQuantizationCompatibility } from "../runtime/compatibility/index.js";
import { summarizeAttentionValueGaps } from "./gguf-metadata.js";
import { isDownloadableGguf } from "./hf-hub.js";

const RANGE_STEPS_BYTES = [4 * 1024 * 1024, 16 * 1024 * 1024, 48 * 1024 * 1024];
const FETCH_TIMEOUT_MS = 20000;

class NeedMoreData extends Error {}

// Minimal GGUF header reader: metadata KVs we care about plus the full tensor
// name directory. Throws NeedMoreData when the buffer ends before the tensor
// directory does, so the caller can escalate the ranged fetch.
export function parseGgufHeaderForCompat(buffer) {
  let pos = 0;
  const need = (bytes) => {
    if (pos + bytes > buffer.length) throw new NeedMoreData();
  };
  const u32 = () => { need(4); const v = buffer.readUInt32LE(pos); pos += 4; return v; };
  const u64 = () => { need(8); const v = buffer.readBigUInt64LE(pos); pos += 8; return Number(v); };
  const str = () => { const n = u64(); need(n); const v = buffer.toString("utf8", pos, pos + n); pos += n; return v; };
  const skip = (bytes) => { need(bytes); pos += bytes; };
  const SCALAR_SIZES = { 0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8 };
  const skipValue = (type) => {
    if (type === 8) { str(); return; }
    if (type === 9) {
      const elementType = u32();
      const count = u64();
      if (elementType === 8) { for (let i = 0; i < count; i += 1) str(); return; }
      if (elementType === 9) { for (let i = 0; i < count; i += 1) skipValue(9); return; }
      const size = SCALAR_SIZES[elementType];
      if (!size) throw new Error(`Unknown GGUF array element type ${elementType}`);
      skip(count * size);
      return;
    }
    const size = SCALAR_SIZES[type];
    if (!size) throw new Error(`Unknown GGUF value type ${type}`);
    skip(size);
  };
  const readScalar = (type) => {
    if (type === 4) return u32();
    if (type === 10 || type === 11) return u64();
    if (type === 5) { need(4); const v = buffer.readInt32LE(pos); pos += 4; return v; }
    skipValue(type);
    return null;
  };

  need(4);
  if (buffer.toString("ascii", 0, 4) !== "GGUF") {
    throw new Error("Not a GGUF file (bad magic).");
  }
  pos = 4;
  const version = u32();
  const tensorCount = u64();
  const kvCount = u64();
  let architecture = null;
  let blockCount = null;
  let expertCount = 0;
  for (let i = 0; i < kvCount; i += 1) {
    const key = str();
    const type = u32();
    if (key === "general.architecture" && type === 8) {
      architecture = str();
    } else if (key.endsWith(".block_count")) {
      blockCount = readScalar(type) ?? blockCount;
    } else if (key.endsWith(".expert_count")) {
      expertCount = readScalar(type) ?? expertCount;
    } else {
      skipValue(type);
    }
  }
  const tensorNames = [];
  for (let i = 0; i < tensorCount; i += 1) {
    tensorNames.push(str());
    const dims = u32();
    skip(dims * 8);
    u32(); // ggml type
    u64(); // offset
  }
  return { version, architecture, blockCount, expertCount, tensorNames };
}

async function fetchGgufHeader({ repo, filename, token, fetchImpl }) {
  const url = `https://huggingface.co/${repo}/resolve/main/${encodeURIComponent(filename)}`;
  let lastError = null;
  for (const rangeBytes of RANGE_STEPS_BYTES) {
    const response = await fetchImpl(url, {
      headers: {
        Range: `bytes=0-${rangeBytes - 1}`,
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      },
      redirect: "follow",
      signal: AbortSignal.timeout(FETCH_TIMEOUT_MS),
    });
    if (!response.ok && response.status !== 206) {
      throw new Error(`Header fetch failed: HTTP ${response.status}`);
    }
    const buffer = Buffer.from(await response.arrayBuffer());
    try {
      return parseGgufHeaderForCompat(buffer);
    } catch (error) {
      if (error instanceof NeedMoreData) {
        lastError = error;
        continue; // escalate to the next range step
      }
      throw error;
    }
  }
  throw new Error(`GGUF tensor directory extends past ${RANGE_STEPS_BYTES.at(-1) / 1024 / 1024} MB.`, { cause: lastError });
}

// Model-level verdict. Layout (architecture + tensor directory) is a property
// of the model, so one header - the smallest downloadable GGUF, the cheapest
// range fetch - answers for every quant variant. Quantization support is
// per-file: the model stays "ready" when at least one downloadable file uses
// a natively supported quant.
export async function checkHubModelNativeReady({
  repo,
  ggufFiles = [],
  token = null,
  fetchImpl = fetch,
} = {}) {
  const downloadable = ggufFiles.filter((file) => isDownloadableGguf(file.filename));
  if (downloadable.length === 0) {
    return { ready: false, reason: "This repo ships no single-file GGUF the runtime could load." };
  }
  const supportedQuantFiles = downloadable.filter(
    (file) => getQuantizationCompatibility(file.quant ?? null).status === "supported",
  );
  if (supportedQuantFiles.length === 0) {
    return {
      ready: false,
      reason: "None of this repo's GGUF quantizations are in the native worker's supported set.",
    };
  }
  const probe = [...downloadable].sort((a, b) => (a.sizeBytes ?? Infinity) - (b.sizeBytes ?? Infinity))[0];
  let header;
  try {
    header = await fetchGgufHeader({ repo, filename: probe.filename, token, fetchImpl });
  } catch (error) {
    return { ready: false, reason: `Native readiness could not be verified: ${error.message}` };
  }
  const gguf = {
    architecture: header.architecture,
    quant: supportedQuantFiles[0].quant,
    expertCount: header.expertCount,
    attentionValueGaps: summarizeAttentionValueGaps(header.tensorNames, header.blockCount),
  };
  const compatibility = evaluateNativeGgufCompatibility({ gguf });
  return {
    ready: compatibility.supported,
    reason: compatibility.supported
      ? `Verified against ${probe.filename}: runs fully on the Lizard native engine.`
      : compatibility.reason,
    architecture: header.architecture,
    probedFile: probe.filename,
  };
}

// Disk-backed verdict cache keyed by repo + revision timestamp. Tolerates a
// missing or unwritable cache file - the check simply runs again.
export function createNativeCompatCache(cacheFilePath) {
  let entries = {};
  try {
    entries = JSON.parse(fs.readFileSync(cacheFilePath, "utf8"));
    if (typeof entries !== "object" || entries === null) entries = {};
  } catch {
    entries = {};
  }
  const persist = () => {
    try {
      fs.mkdirSync(path.dirname(cacheFilePath), { recursive: true });
      fs.writeFileSync(cacheFilePath, JSON.stringify(entries));
    } catch {
      // Cache persistence is best-effort.
    }
  };
  return {
    get(repo, lastModified) {
      const entry = entries[repo];
      return entry && entry.lastModified === (lastModified ?? null) ? entry.verdict : null;
    },
    set(repo, lastModified, verdict) {
      entries[repo] = { lastModified: lastModified ?? null, verdict, checkedAt: new Date().toISOString() };
      persist();
    },
  };
}

export async function mapWithConcurrency(items, limit, worker) {
  const results = new Array(items.length);
  let next = 0;
  const lanes = Array.from({ length: Math.min(limit, items.length) }, async () => {
    while (next < items.length) {
      const index = next;
      next += 1;
      results[index] = await worker(items[index], index);
    }
  });
  await Promise.all(lanes);
  return results;
}
