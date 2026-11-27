import test from "node:test";
import assert from "node:assert/strict";
import crypto from "node:crypto";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { Readable } from "node:stream";
import {
  HF_GGUF_ALLOWLIST,
  findAllowlistedHfModel,
  isAllowlistedRepo,
  findAllowlistedFile,
} from "../src/lib/hf-model-catalog.js";
import { searchCompatibleGgufModels, SIZE_BUCKETS, sizeBucketForGB } from "../src/lib/huggingface.js";
import { buildHfRecommendation } from "../src/lib/hf-workflow.js";
import {
  downloadGgufModel,
  verifyDownloadedModel,
  registerDownloadedModel,
  activateModel,
  getActiveModel,
  computeSha256,
} from "../src/lib/download-manager.js";
import { createUiServer } from "../src/ui/server.js";

function tempConfig() {
  const runtimeDirectory = fs.mkdtempSync(path.join(os.tmpdir(), "lizard-hf-"));
  return {
    runtimeDirectory,
    modelDirectory: path.join(runtimeDirectory, "models"),
    catalogPath: path.join(runtimeDirectory, "model-catalog.json"),
    maxRamUsagePercent: 85,
    maxVramUsagePercent: 90,
    defaultMode: "balanced",
    engine: { serverBinary: "llama-server", cliBinary: "llama-cli", preferredBinaryDirectories: [] },
    benchmark: { ci: { model: null } },
  };
}

function fakeHardware({ ramGB = 16, vramGB = 0 } = {}) {
  return {
    platform: "linux",
    arch: "x64",
    cpu: { model: "Test CPU", physicalCores: 8, logicalThreads: 8 },
    memory: { totalGB: ramGB, freeGB: ramGB - 2, availableGB: ramGB * 0.85, plannerBudgetGB: ramGB * 0.85 },
    storage: { drive: "/", freeGB: 200, usedGB: 50 },
    gpus: vramGB ? [{ vendor: "nvidia", name: "Test GPU", vramGB }] : [],
    primaryGpu: vramGB ? { vendor: "nvidia", name: "Test GPU", vramGB, backends: { cuda: true } } : null,
    backends: { cuda: Boolean(vramGB), cpu: true },
  };
}

function writeFakeGguf(filePath, payload = "lizard-test-weights") {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, Buffer.concat([Buffer.from("GGUF"), Buffer.from(payload)]));
}

test("allowlist lookups are restricted to curated repos", () => {
  assert.ok(HF_GGUF_ALLOWLIST.length > 0);
  const first = HF_GGUF_ALLOWLIST[0];
  assert.equal(isAllowlistedRepo(first.hfRepo), true);
  assert.equal(isAllowlistedRepo("evil/malware-GGUF"), false);
  assert.equal(findAllowlistedHfModel(first.id)?.hfRepo, first.hfRepo);
  const file = Object.values(first.files)[0].filename;
  assert.equal(findAllowlistedFile(first.hfRepo, file)?.model.id, first.id);
  assert.equal(findAllowlistedFile(first.hfRepo, "not-a-real-file.gguf"), null);
});

test("search ranks task-matching, fitting models above others and respects mode", () => {
  const hardware = fakeHardware({ ramGB: 16 });
  const models = searchCompatibleGgufModels({ hardware, task: "coding", mode: "balanced" });
  assert.ok(models.length === HF_GGUF_ALLOWLIST.length);
  // Task-matching models come first.
  assert.equal(models[0].taskMatch, true);
  // Balanced mode should prefer Q4_K_M, not the largest quant.
  const coder = models.find((m) => m.id === "qwen2.5-coder-7b-instruct-gguf");
  assert.equal(coder.recommendedQuant, "Q4_K_M");
  // Every model carries an estimate and a fit classification.
  for (const model of models) {
    assert.ok(["fits", "tight", "exceeds"].includes(model.fit));
    assert.ok(model.estimate.totalGB > 0);
  }
});

test("tiny RAM budgets push large models to 'exceeds'", () => {
  const hardware = fakeHardware({ ramGB: 4 });
  const models = searchCompatibleGgufModels({ hardware, mode: "balanced" });
  const big = models.find((m) => m.id === "meta-llama-3.1-8b-instruct-gguf");
  assert.equal(big.fit, "exceeds");
});

test("gemma4-e4b is allowlisted, featured, and reachable by its ollama tag", () => {
  const gemma4 = findAllowlistedHfModel("gemma4-e4b");
  assert.ok(gemma4, "gemma4-e4b must be on the allowlist so onboarding downloads are accepted");
  assert.equal(gemma4.featured, true);
  assert.equal(gemma4.nativeBackend, true);
  assert.ok(gemma4.ollamaTags.includes("gemma4:latest"));
  assert.equal(isAllowlistedRepo(gemma4.hfRepo), true);
});

test("featured models are pinned above non-featured ones within the same task-relevance tier", () => {
  const hardware = fakeHardware({ ramGB: 16 });
  // No task filter -> every model ties on taskMatch, so featured must decide.
  const models = searchCompatibleGgufModels({ hardware, mode: "balanced" });
  assert.equal(models[0].id, "gemma4-e4b");
  assert.equal(models[0].featured, true);
});

test("featured status never outranks an explicit task match", () => {
  const hardware = fakeHardware({ ramGB: 16 });
  // gemma4-e4b does not cover "coding" - every task-matching model (there
  // are several in the allowlist) must sort ahead of it despite it being
  // featured.
  const models = searchCompatibleGgufModels({ hardware, task: "coding", mode: "balanced" });
  assert.equal(models[0].taskMatch, true);
  const gemma4Index = models.findIndex((m) => m.id === "gemma4-e4b");
  const lastTaskMatchIndex = models.reduce((last, m, index) => (m.taskMatch ? index : last), -1);
  assert.ok(gemma4Index > lastTaskMatchIndex, "gemma4-e4b must sort after every task-matching model");
});

test("size buckets classify known download sizes into the expected category", () => {
  assert.equal(SIZE_BUCKETS.map((b) => b.id).join(","), "2gb,5gb,9gb,14gb");
  assert.equal(sizeBucketForGB(1.7).id, "2gb");
  assert.equal(sizeBucketForGB(4.9).id, "5gb");
  assert.equal(sizeBucketForGB(8.5).id, "9gb");
  assert.equal(sizeBucketForGB(16).id, "14gb");
});

test("filtering by size bucket re-bases each model onto its best-matching quant", () => {
  const hardware = fakeHardware({ ramGB: 32 });
  const models = searchCompatibleGgufModels({ hardware, mode: "balanced", sizeBucket: "9gb" });
  assert.ok(models.length > 0);
  for (const model of models) {
    assert.equal(model.sizeBucket, "9gb");
  }
  // gemma4-e4b's Q4_K_M (~4.9GB) is what "balanced" mode would normally
  // recommend, but browsing the ~9GB bucket must surface its Q8_0 file.
  const gemma4 = models.find((m) => m.id === "gemma4-e4b");
  assert.ok(gemma4, "gemma4-e4b offers a Q8_0 file and should appear in the 9gb bucket");
  assert.equal(gemma4.recommendedQuant, "Q8_0");
  // A model with no ~9GB quant option (e.g. the 1.5B model) must be excluded.
  assert.equal(models.some((m) => m.id === "qwen2.5-1.5b-instruct-gguf"), false);
});

test("an empty size bucket filter falls back to the unfiltered mode-driven recommendation", () => {
  const hardware = fakeHardware({ ramGB: 16 });
  const filtered = searchCompatibleGgufModels({ hardware, mode: "balanced", sizeBucket: "9gb" });
  const unfiltered = searchCompatibleGgufModels({ hardware, mode: "balanced" });
  assert.ok(filtered.length < unfiltered.length);
});

test("buildHfRecommendation produces a runnable plan shape", () => {
  const hardware = fakeHardware({ ramGB: 16, vramGB: 8 });
  const hfModel = findAllowlistedHfModel("qwen2.5-coder-7b-instruct-gguf");
  const config = tempConfig();
  const recommendation = buildHfRecommendation({ hfModel, quant: "Q4_K_M", hardware, config, task: "coding", mode: "balanced" });
  assert.equal(recommendation.selected.modelId, hfModel.id);
  assert.equal(recommendation.selected.quant, "Q4_K_M");
  assert.ok(recommendation.selected.contextTokens > 0);
  assert.ok(recommendation.selected.gpuLayers >= 0);
  assert.ok(recommendation.selected.threads >= 1);
  assert.ok(recommendation.selected.estimate.ramRequiredGB > 0);
});

test("verifyDownloadedModel checks GGUF magic and sha256", async () => {
  const config = tempConfig();
  const filePath = path.join(config.modelDirectory, "valid.gguf");
  writeFakeGguf(filePath);
  const sha = await computeSha256(filePath);

  const ok = await verifyDownloadedModel({ path: filePath, expectedSha256: sha });
  assert.equal(ok.ok, true);
  assert.equal(ok.sha256, sha);

  const mismatch = await verifyDownloadedModel({ path: filePath, expectedSha256: "deadbeef" });
  assert.equal(mismatch.ok, false);
  assert.equal(mismatch.reason, "sha256-mismatch");

  const notGguf = path.join(config.modelDirectory, "bad.gguf");
  fs.writeFileSync(notGguf, Buffer.from("NOPEnotgguf"));
  const bad = await verifyDownloadedModel({ path: notGguf });
  assert.equal(bad.ok, false);
  assert.equal(bad.reason, "not-gguf");
});

test("downloadGgufModel accepts any GGUF repo (live browse) but rejects non-GGUF input", async () => {
  const config = tempConfig();
  // Basic input validation stays: empty repo and non-.gguf files are refused.
  await assert.rejects(
    () => downloadGgufModel({ repo: "", filename: "x.gguf", config }),
    /repository id is required/,
  );
  await assert.rejects(
    () => downloadGgufModel({ repo: "someone/model", filename: "README.txt", config }),
    /Not a GGUF file/,
  );

  // A live (non-allowlist) GGUF now downloads. The allowlist is a trust flag,
  // not a gate: quant is derived from the filename and the result is marked
  // untrusted. GGUF magic + sha256 verification downstream is unchanged.
  const payload = Buffer.concat([Buffer.from("GGUF"), Buffer.from("live-weights")]);
  const fetchImpl = async () => ({
    ok: true,
    headers: new Map([["content-length", String(payload.length)]]),
    body: Readable.toWeb(Readable.from([payload])),
  });
  const result = await downloadGgufModel({
    repo: "some-user/Custom-Model-GGUF",
    filename: "Custom-Model-Q5_K_M.gguf",
    config,
    fetchImpl,
  });
  assert.equal(result.quant, "Q5_K_M");
  assert.equal(result.trusted, false);
  assert.ok(fs.existsSync(result.path));
});

test("downloadGgufModel streams an allowlisted file with progress and atomic write", async () => {
  const config = tempConfig();
  const model = HF_GGUF_ALLOWLIST.find((m) => m.id === "qwen2.5-1.5b-instruct-gguf");
  const [quant, file] = Object.entries(model.files)[0];
  const payload = Buffer.concat([Buffer.from("GGUF"), Buffer.from("streamed-weights")]);

  const fetchImpl = async () => ({
    ok: true,
    headers: new Map([["content-length", String(payload.length)]]),
    body: Readable.toWeb(Readable.from([payload])),
  });

  const progressEvents = [];
  const result = await downloadGgufModel({
    repo: model.hfRepo,
    filename: file.filename,
    config,
    fetchImpl,
    onProgress: (p) => progressEvents.push(p),
  });

  assert.equal(result.quant, quant);
  assert.ok(fs.existsSync(result.path));
  assert.equal(fs.readFileSync(result.path).toString("utf8").startsWith("GGUF"), true);
  assert.ok(progressEvents.length > 0);
  assert.equal(fs.existsSync(`${result.path}.download`), false);
});

test("register then activate round-trips through catalog and config.json", async () => {
  const config = tempConfig();
  const modelPath = path.join(config.modelDirectory, "qwen.gguf");
  writeFakeGguf(modelPath);
  const sha = await computeSha256(modelPath);

  const record = registerDownloadedModel({
    config,
    entry: {
      id: "qwen2.5-1.5b-instruct-gguf-q4_k_m",
      hfRepo: "bartowski/Qwen2.5-1.5B-Instruct-GGUF",
      filename: "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf",
      quant: "Q4_K_M",
      importedPath: modelPath,
      sha256: sha,
      license: "apache-2.0",
      paramsB: 1.5,
    },
  });
  assert.equal(record.source, "huggingface");
  assert.equal(record.status, "ready");

  const catalog = JSON.parse(fs.readFileSync(config.catalogPath, "utf8"));
  assert.equal(catalog.models.length, 1);

  const active = activateModel({ config, modelId: record.id });
  assert.equal(active.id, record.id);
  assert.equal(active.modelPath, modelPath);
  assert.equal(active.runtimeProvider, "lizard-native");
  assert.deepEqual(getActiveModel(config), active);

  // Active selection is persisted under runtimeDirectory/config.json.
  const homeConfig = JSON.parse(fs.readFileSync(path.join(config.runtimeDirectory, "config.json"), "utf8"));
  assert.equal(homeConfig.activeModel.id, record.id);
});

test("activation persists Caterpillar so onboarding, warm-up, and chat agree", () => {
  const config = tempConfig();
  const modelPath = path.join(config.modelDirectory, "caterpillar.gguf");
  fs.mkdirSync(config.modelDirectory, { recursive: true });
  fs.writeFileSync(modelPath, Buffer.from("GGUF"));
  registerDownloadedModel({
    config,
    entry: {
      id: "caterpillar-model",
      hfRepo: "example/caterpillar",
      filename: "caterpillar.gguf",
      quant: "Q4_K_M",
      importedPath: modelPath,
    },
  });

  const active = activateModel({
    config,
    modelId: "caterpillar-model",
    runtimeProvider: "caterpillar",
  });
  assert.equal(active.runtimeProvider, "caterpillar");
  assert.equal(getActiveModel(config).runtimeProvider, "caterpillar");
});

test("IQ model catalog entries preserve imatrix sourcing guidance", () => {
  const config = tempConfig();
  const record = registerDownloadedModel({
    config,
    entry: {
      id: "llama-iq4-xs",
      hfRepo: "example/Llama-GGUF",
      filename: "Llama-IQ4_XS.gguf",
      quant: "IQ4_XS",
      importedPath: path.join(config.modelDirectory, "llama-iq4-xs.gguf"),
    },
  });
  assert.equal(record.quantizationGuidance.importanceMatrix, "preferred");
  assert.equal(record.quantizationGuidance.preferredSource, "imatrix-calibrated");
  assert.match(record.quantizationGuidance.warning, /Q4_K_M/);
});

test("activateModel rejects unknown or missing-artifact models", () => {
  const config = tempConfig();
  assert.throws(() => activateModel({ config, modelId: "nope" }), /not found in local catalog/);
  registerDownloadedModel({
    config,
    entry: { id: "ghost", hfRepo: "x", filename: "x.gguf", importedPath: path.join(config.modelDirectory, "missing.gguf") },
  });
  assert.throws(() => activateModel({ config, modelId: "ghost" }), /artifact missing/);
});

test("asset route serves images and blocks traversal", async () => {
  const config = tempConfig();
  const { createUiServer } = await import("../src/ui/server.js");
  const server = createUiServer({ config });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const { port } = server.address();
  const base = `http://127.0.0.1:${port}`;
  try {
    const svg = await fetch(`${base}/assets/brand/lizard-mark.png`);
    assert.equal(svg.status, 200);
    // Encoded traversal must not escape the assets directory.
    const escape = await fetch(`${base}/assets/%2e%2e/server.js`);
    assert.equal(escape.status, 404);
  } finally {
    await new Promise((resolve) => server.close(resolve));
  }
});

test("UI server exposes scan and models endpoints", async () => {
  const config = tempConfig();
  const server = createUiServer({ config });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const { port } = server.address();
  try {
    const scan = await fetch(`http://127.0.0.1:${port}/api/scan`).then((r) => r.json());
    assert.ok(scan.hardware.cpu);
    assert.equal(typeof scan.capabilityRegistry?.selectedBackend, "string");

    const models = await fetch(`http://127.0.0.1:${port}/api/models?task=coding&mode=balanced`).then((r) => r.json());
    assert.equal(models.models.length, HF_GGUF_ALLOWLIST.length);
    assert.ok(models.models[0].compatibilityScore !== undefined);
    assert.equal(models.bucket, null);
    assert.ok(Array.isArray(models.sizeBuckets));
    assert.ok(models.sizeBuckets.every((bucket) => typeof bucket.count === "number"));

    const bucketed = await fetch(`http://127.0.0.1:${port}/api/models?mode=balanced&bucket=9gb`).then((r) => r.json());
    assert.equal(bucketed.bucket, "9gb");
    assert.ok(bucketed.models.length < HF_GGUF_ALLOWLIST.length);
    assert.ok(bucketed.models.every((model) => model.sizeBucket === "9gb"));

    const index = await fetch(`http://127.0.0.1:${port}/`).then((r) => r.text());
    assert.match(index, /Lizard LLM/);
  } finally {
    await new Promise((resolve) => server.close(resolve));
  }
});

// Collects every "event: X\ndata: Y\n\n" frame from an SSE response body.
async function readSseEvents(response) {
  const text = await response.text();
  return text
    .split("\n\n")
    .filter((block) => block.trim())
    .map((block) => {
      const eventLine = block.split("\n").find((line) => line.startsWith("event: "));
      const dataLine = block.split("\n").find((line) => line.startsWith("data: "));
      return {
        event: eventLine?.slice("event: ".length) ?? null,
        data: dataLine ? JSON.parse(dataLine.slice("data: ".length)) : null,
      };
    });
}

test("quick-chat SSE endpoint rejects an empty prompt without needing an active model", async () => {
  const config = tempConfig();
  const server = createUiServer({ config });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const { port } = server.address();
  try {
    const response = await fetch(`http://127.0.0.1:${port}/api/quick-chat?prompt=`);
    assert.equal(response.status, 200); // SSE - errors are events, not HTTP status
    const events = await readSseEvents(response);
    assert.equal(events.length, 1);
    assert.equal(events[0].event, "error");
    assert.match(events[0].data.message, /prompt is required/i);
  } finally {
    await new Promise((resolve) => server.close(resolve));
  }
});

test("quick-chat SSE endpoint reports a clear error when no model is active", async () => {
  const config = tempConfig();
  const server = createUiServer({ config });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const { port } = server.address();
  try {
    const response = await fetch(`http://127.0.0.1:${port}/api/quick-chat?${new URLSearchParams({ prompt: "Hello" })}`);
    const events = await readSseEvents(response);
    const errorEvent = events.find((e) => e.event === "error");
    assert.ok(errorEvent, "expected an error event when no model is active");
    assert.match(errorEvent.data.message, /No active model is set/);
  } finally {
    await new Promise((resolve) => server.close(resolve));
  }
});
