import test from "node:test";
import assert from "node:assert/strict";
import {
  buildUniversalModelList,
  computeRuntimeVerdicts,
  dedupeUniversalModels,
  normalizeCatalogModel,
  normalizeOllamaModel,
  normalizeRegistryModel,
  RUNTIME_STATUS,
} from "../src/lib/universal-model.js";

// Fixtures mirror the real shapes: src/data/models.json entries,
// ~/.lizard-llm/model-catalog.json entries, and `ollama list` rows.
const registryLlama = {
  id: "llama-3.2-3b-instruct",
  family: "llama",
  paramsB: 3.2,
  tasks: ["chat", "summarization"],
  quantizations: ["Q4_K_M", "Q5_K_M"],
  defaultContext: 8192,
  maxContext: 131072,
  ollamaTag: "llama3.2:3b",
  ollamaTags: ["llama3.2:3b"],
};

const catalogImported = {
  id: "llama-3.2-3b-instruct",
  source: "ollama",
  ollamaTag: "llama3.2:3b",
  importedPath: "C:/models/llama-3.2-3b-instruct.gguf",
  gguf: { architecture: "llama", quant: "Q4_K_M", paramsB: 3.2 },
};

const ollamaInstalled = { tag: "llama3.2:3b", id: "abc123", size: "2.0 GB", modified: "2 days ago" };

test("the same model from three sources merges into ONE entry (spec §9.8)", () => {
  const models = buildUniversalModelList({
    registryModels: [registryLlama],
    catalogModels: [catalogImported],
    ollamaModels: [ollamaInstalled],
  });
  assert.equal(models.length, 1, "registry + catalog + ollama must dedupe to one model");
  const [model] = models;
  assert.deepEqual(model.sources.map((s) => s.kind).sort(), ["local", "ollama", "registry"]);
  // Curated facts survive the merge; on-disk facts survive the merge.
  assert.equal(model.family, "llama");
  assert.equal(model.parameterCountB, 3.2);
  assert.equal(model.localPath, "C:/models/llama-3.2-3b-instruct.gguf");
  assert.equal(model.ollamaInstalled, true);
});

test("runtime verdicts come from real metadata, not guesses", () => {
  const models = buildUniversalModelList({
    registryModels: [registryLlama],
    catalogModels: [catalogImported],
    ollamaModels: [ollamaInstalled],
  });
  const runtimes = Object.fromEntries(models[0].runtimes.map((r) => [r.id, r]));
  assert.equal(runtimes.native.status, RUNTIME_STATUS.SUPPORTED, runtimes.native.reason ?? "");
  assert.equal(runtimes["llama.cpp"].status, RUNTIME_STATUS.SUPPORTED);
  assert.equal(runtimes.ollama.status, RUNTIME_STATUS.SUPPORTED);
});

test("an unsupported quantization blocks native with the matrix's own reason", () => {
  const model = {
    ...normalizeCatalogModel({
      id: "weird-quant",
      source: "huggingface",
      importedPath: "C:/models/weird.gguf",
      gguf: { architecture: "llama", quant: "IQ1_S", paramsB: 7 },
    }),
  };
  const runtimes = Object.fromEntries(computeRuntimeVerdicts(model).map((r) => [r.id, r]));
  assert.equal(runtimes.native.status, RUNTIME_STATUS.BLOCKED);
  assert.ok(runtimes.native.reason && runtimes.native.reason.length > 10, "a blocked verdict must say why");
  // llama.cpp still runs any local GGUF.
  assert.equal(runtimes["llama.cpp"].status, RUNTIME_STATUS.SUPPORTED);
});

test("an installed-but-unimported Ollama model is importable, never claimed native", () => {
  const model = normalizeOllamaModel({ tag: "mistral:7b", id: "x", size: "4 GB", modified: "now" });
  const runtimes = Object.fromEntries(computeRuntimeVerdicts(model).map((r) => [r.id, r]));
  assert.equal(runtimes.native.status, RUNTIME_STATUS.IMPORTABLE);
  assert.match(runtimes.native.reason, /import/i);
  assert.equal(runtimes.ollama.status, RUNTIME_STATUS.SUPPORTED);
  assert.equal(runtimes["llama.cpp"].status, RUNTIME_STATUS.UNKNOWN);
});

test("a registry model with a known but uninstalled tag is pullable", () => {
  const model = normalizeRegistryModel(registryLlama);
  const runtimes = Object.fromEntries(computeRuntimeVerdicts(model).map((r) => [r.id, r]));
  assert.equal(runtimes.ollama.status, RUNTIME_STATUS.PULLABLE);
  assert.match(runtimes.ollama.reason, /llama3\.2:3b/);
  assert.equal(runtimes.native.status, RUNTIME_STATUS.UNKNOWN);
});

test("tags compare with an implicit :latest and case-insensitively", () => {
  const merged = dedupeUniversalModels([
    normalizeRegistryModel({ ...registryLlama, id: "m", ollamaTag: "Gemma2", ollamaTags: [] }),
    normalizeOllamaModel({ tag: "gemma2:latest", id: "y", size: "5 GB", modified: "now" }),
  ]);
  assert.equal(merged.length, 1, "Gemma2 and gemma2:latest are the same tag");
});

test("unrelated models never merge", () => {
  const merged = dedupeUniversalModels([
    normalizeRegistryModel(registryLlama),
    normalizeOllamaModel({ tag: "qwen3:8b", id: "z", size: "5 GB", modified: "now" }),
  ]);
  assert.equal(merged.length, 2);
});

test("no Ollama installed degrades cleanly to registry + catalog", () => {
  const models = buildUniversalModelList({
    registryModels: [registryLlama],
    catalogModels: [catalogImported],
    ollamaModels: [],
  });
  assert.equal(models.length, 1);
  assert.equal(models[0].ollamaInstalled, false);
  const runtimes = Object.fromEntries(models[0].runtimes.map((r) => [r.id, r]));
  // The tag linkage still says where it CAN come from.
  assert.equal(runtimes.ollama.status, RUNTIME_STATUS.PULLABLE);
});

// ---------------------------------------------------------------------------
// Endpoint coverage, fully sandboxed: an in-process UI server on an ephemeral
// port with a temp-directory config and a seeded fake catalog. Deliberately
// NEVER touches the installed app on localhost:7817, and never asserts on
// this machine's real Ollama state (which may or may not exist) - only on the
// structural contract plus the entry we seeded ourselves.
// ---------------------------------------------------------------------------
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { createUiServer } from "../src/ui/server.js";

function sandboxConfig() {
  const runtimeDirectory = fs.mkdtempSync(path.join(os.tmpdir(), "lizard-universal-"));
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

test("/api/models/unified serves the merged list from a sandboxed server", async () => {
  const config = sandboxConfig();
  const seededPath = path.join(config.modelDirectory, "seeded.gguf");
  fs.mkdirSync(config.modelDirectory, { recursive: true });
  fs.writeFileSync(seededPath, Buffer.from("GGUFfake"));
  fs.writeFileSync(config.catalogPath, JSON.stringify({
    models: [{
      id: "sandbox-seeded-model",
      source: "huggingface",
      importedPath: seededPath,
      gguf: { architecture: "llama", quant: "Q4_K_M", paramsB: 1.1 },
    }],
  }));

  const server = createUiServer({ config, logger: { info() {}, warn() {}, error() {} } });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const { port } = server.address();
  try {
    const response = await fetch(`http://127.0.0.1:${port}/api/models/unified`);
    assert.equal(response.status, 200);
    const payload = await response.json();
    assert.ok(Array.isArray(payload.models) && payload.models.length > 0);

    // Structural contract: every entry carries sources + the three verdicts.
    for (const model of payload.models) {
      assert.ok(Array.isArray(model.sources) && model.sources.length > 0, `${model.id} needs sources`);
      const ids = model.runtimes.map((r) => r.id).sort();
      assert.deepEqual(ids, ["llama.cpp", "native", "ollama"], `${model.id} needs all three verdicts`);
    }

    // The entry we seeded is present, native-supported, exactly once.
    const seeded = payload.models.filter((m) => m.id === "sandbox-seeded-model");
    assert.equal(seeded.length, 1);
    const runtimes = Object.fromEntries(seeded[0].runtimes.map((r) => [r.id, r]));
    assert.equal(runtimes.native.status, RUNTIME_STATUS.SUPPORTED, runtimes.native.reason ?? "");
    assert.equal(runtimes["llama.cpp"].status, RUNTIME_STATUS.SUPPORTED);
  } finally {
    await new Promise((resolve) => server.close(resolve));
    fs.rmSync(config.runtimeDirectory, { recursive: true, force: true });
  }
});

// ---------------------------------------------------------------------------
// Phase 2: probed-Ollama verdicts. A compatible blob without a local file must
// be "importable" (one click away), never "supported" - the native runtime
// has nothing to load until the import creates the catalog entry.
// ---------------------------------------------------------------------------
test("a probed compatible Ollama blob is importable, not supported (no local file yet)", () => {
  const model = {
    ...normalizeOllamaModel({ tag: "llama3.2:3b", id: "x", size: "2 GB", modified: "now" }),
    gguf: { architecture: "llama", quant: "Q4_K_M", paramsB: 3.2 },
  };
  const runtimes = Object.fromEntries(computeRuntimeVerdicts(model).map((r) => [r.id, r]));
  assert.equal(runtimes.native.status, RUNTIME_STATUS.IMPORTABLE);
  assert.match(runtimes.native.reason, /no re-download/i);
});

test("a probed incompatible Ollama blob is blocked with the matrix's reason", () => {
  const model = {
    ...normalizeOllamaModel({ tag: "odd:latest", id: "x", size: "2 GB", modified: "now" }),
    gguf: { architecture: "some-unknown-arch", quant: "Q4_K_M" },
  };
  const runtimes = Object.fromEntries(computeRuntimeVerdicts(model).map((r) => [r.id, r]));
  assert.equal(runtimes.native.status, RUNTIME_STATUS.BLOCKED);
  assert.ok(runtimes.native.reason.length > 10);
  // Still perfectly usable through Ollama - blocked native is not a dead model.
  assert.equal(runtimes.ollama.status, RUNTIME_STATUS.SUPPORTED);
});

// ---------------------------------------------------------------------------
// Phase 3: the compatibility matrix as data + drift guard. The generated doc
// must contain every native family the code declares, so it can never quietly
// claim (or omit) support the runtime actually has.
// ---------------------------------------------------------------------------
test("listCompatibilityMatrix projects every native family from code", async () => {
  const { listCompatibilityMatrix, buildNativeGgufFamilyMatrix } = await import("../src/runtime/compatibility/index.js");
  const matrix = listCompatibilityMatrix();
  const codeFamilies = buildNativeGgufFamilyMatrix().map((r) => r.family).sort();
  const docFamilies = matrix.families.map((r) => r.family).sort();
  assert.deepEqual(docFamilies, codeFamilies, "the projection must list exactly the code's families");
  assert.equal(matrix.schema, "lizard.compatibility-matrix.v1");
  assert.ok(matrix.families.every((f) => f.runtimes && f.runtimes.native), "every family needs a native verdict");
});

test("generated compatibility doc contains every family in code (drift guard)", async () => {
  const { listCompatibilityMatrix } = await import("../src/runtime/compatibility/index.js");
  const docPath = new URL("../docs/model-runtime-compatibility.md", import.meta.url);
  let doc = "";
  try {
    doc = fs.readFileSync(docPath, "utf8");
  } catch {
    // Doc not generated in this checkout - skip rather than fail a fresh clone.
    return;
  }
  for (const family of listCompatibilityMatrix().families) {
    assert.ok(doc.includes(family.family), `docs/model-runtime-compatibility.md is missing "${family.family}" - re-run scripts/write-model-compatibility-docs.mjs`);
  }
});

// ---------------------------------------------------------------------------
// Phase 4: capability detection. Detected from metadata, never guessed; an
// absent capability means "not detected", not a claim.
// ---------------------------------------------------------------------------
test("capabilities are derived from GGUF metadata, not invented", async () => {
  const { deriveModelCapabilities } = await import("../src/lib/universal-model.js");

  // A plain chat model: chat yes, everything exotic no.
  const plain = deriveModelCapabilities({
    architecture: "llama",
    chatTemplate: "{% for message in messages %}{{ message.role }}{% endfor %}",
    contextLength: 8192,
  });
  assert.equal(plain.chat, true);
  assert.equal(plain.toolCalling, false);
  assert.equal(plain.vision, false);
  assert.equal(plain.moe, false);
  assert.equal(plain.longContext, false);

  // A tool + long-context + MoE model.
  const rich = deriveModelCapabilities({
    architecture: "qwen3moe",
    chatTemplate: "{% if tools %}You may call a function/tool{% endif %}",
    contextLength: 131072,
    expertCount: 8,
  });
  assert.equal(rich.toolCalling, true);
  assert.equal(rich.longContext, true);
  assert.equal(rich.moe, true);

  // A vision model detected from projector tensors.
  const vision = deriveModelCapabilities({
    architecture: "llama",
    tensorsPreview: [{ name: "mmproj.weight" }, { name: "blk.0.attn_q.weight" }],
  });
  assert.equal(vision.vision, true);
});

test("no metadata yields the safe default (text only, nothing claimed)", async () => {
  const { deriveModelCapabilities } = await import("../src/lib/universal-model.js");
  const caps = deriveModelCapabilities(null);
  assert.equal(caps.text, true);
  assert.equal(caps.chat, false);
  assert.equal(caps.toolCalling, false);
  assert.equal(caps.vision, false);
});

test("the unified list attaches detected capabilities to each model", () => {
  const models = buildUniversalModelList({
    registryModels: [registryLlama],
    catalogModels: [{
      id: "llama-3.2-3b-instruct",
      source: "ollama",
      ollamaTag: "llama3.2:3b",
      importedPath: "C:/models/llama-3.2-3b-instruct.gguf",
      gguf: { architecture: "llama", quant: "Q4_K_M", contextLength: 131072 },
    }],
    ollamaModels: [],
  });
  assert.ok(models[0].capabilities, "every unified model must carry a capabilities object");
  assert.equal(models[0].capabilities.longContext, true);
});

// ---------------------------------------------------------------------------
// Phase 5: MoE-aware sizing. Parameter count != runtime cost (spec s12).
// ---------------------------------------------------------------------------
test("MoE active params are far below total; dense are equal", async () => {
  const { deriveParameterSizing } = await import("../src/lib/universal-model.js");

  // Dense: active == total.
  const dense = deriveParameterSizing({ architecture: "llama", parameterEstimateB: 3.2 }, 3.2);
  assert.equal(dense.isMoe, false);
  assert.equal(dense.activeParameterCountB, 3.2);
  assert.equal(dense.totalParameterCountB, 3.2);

  // MoE: 8 experts, 2 used -> active ~= 1/4 of total.
  const moe = deriveParameterSizing({ expertCount: 8, expertUsedCount: 2, parameterEstimateB: 47 }, 47);
  assert.equal(moe.isMoe, true);
  assert.equal(moe.totalParameterCountB, 47);
  assert.ok(moe.activeParameterCountB < moe.totalParameterCountB, "active must be below total for MoE");
  assert.equal(moe.activeIsEstimate, true, "MoE active is an estimate and must be labelled so");
});

test("MoE sizing degrades to total when the header lacks expert_used_count", async () => {
  const { deriveParameterSizing } = await import("../src/lib/universal-model.js");
  // expert_count present but not expert_used_count: cannot estimate active,
  // so do not fabricate one.
  const partial = deriveParameterSizing({ expertCount: 8, parameterEstimateB: 47 }, 47);
  assert.equal(partial.activeParameterCountB, 47);
  assert.equal(partial.isMoe, true);
});

// ---------------------------------------------------------------------------
// Phase 6: runtime adapter seam. canLoad and health are real; the interface
// exists uniformly; the Safetensors slot tells the truth. Zero behavior
// change - execution still runs through the existing engine paths.
// ---------------------------------------------------------------------------
test("every adapter exposes the full interface", async () => {
  const { createRuntimeAdapter, ADAPTER_IDS } = await import("../src/runtime/adapters/index.js");
  for (const id of ADAPTER_IDS) {
    const adapter = createRuntimeAdapter(id);
    for (const verb of ["canLoad", "prepare", "load", "generate", "stream", "unload", "health"]) {
      assert.equal(typeof adapter[verb], "function", `${id}.${verb} must exist`);
    }
  }
});

test("canLoad answers from the model's runtime verdicts", async () => {
  const { createRuntimeAdapter, resolveLoadableAdapters } = await import("../src/runtime/adapters/index.js");
  const [model] = buildUniversalModelList({
    registryModels: [registryLlama],
    catalogModels: [catalogImported],
    ollamaModels: [ollamaInstalled],
  });
  assert.equal(createRuntimeAdapter("native").canLoad(model).ok, true);
  assert.equal(createRuntimeAdapter("llama.cpp").canLoad(model).ok, true);
  assert.equal(createRuntimeAdapter("ollama").canLoad(model).ok, true);
  const loadable = resolveLoadableAdapters(model);
  assert.ok(loadable.includes("native") && loadable.includes("llama.cpp"));
});

test("the Safetensors adapter refuses honestly", async () => {
  const { createRuntimeAdapter } = await import("../src/runtime/adapters/index.js");
  const verdict = createRuntimeAdapter("safetensors").canLoad({ runtimes: [] });
  assert.equal(verdict.ok, false);
  assert.match(verdict.reason, /Safetensors|routed/i);
});

test("an execution verb without a delegate refuses clearly (no silent path)", async () => {
  const { createRuntimeAdapter } = await import("../src/runtime/adapters/index.js");
  await assert.rejects(
    createRuntimeAdapter("native").load({}, {}),
    /no load delegate wired/i,
  );
});

// ---------------------------------------------------------------------------
// Phase 7: Safetensors detection + honest routing. Never claims native/llama
// execution for Safetensors; routes to Ollama import where the arch allows.
// ---------------------------------------------------------------------------
test("detectSafetensors distinguishes safetensors-only from GGUF repos", async () => {
  const { detectSafetensors } = await import("../src/lib/universal-model.js");
  assert.equal(detectSafetensors([{ rfilename: "model.safetensors" }, { rfilename: "config.json" }]).safetensorsOnly, true);
  assert.equal(detectSafetensors([{ rfilename: "model.gguf" }]).safetensorsOnly, false);
  assert.equal(detectSafetensors([{ rfilename: "model.safetensors" }, { rfilename: "q4.gguf" }]).safetensorsOnly, false);
});

test("a safetensors-only model is blocked native/llama, routed to Ollama by arch", () => {
  const importable = buildUniversalModelList({
    safetensorsModels: [{ id: "some/llama-ft", repo: "some/llama-ft", architecture: "llama", paramsB: 8 }],
  })[0];
  const r = Object.fromEntries(importable.runtimes.map((x) => [x.id, x]));
  assert.equal(r.native.status, RUNTIME_STATUS.BLOCKED);
  assert.match(r.native.reason, /Safetensors|GGUF/);
  assert.equal(r["llama.cpp"].status, RUNTIME_STATUS.BLOCKED);
  assert.equal(r.ollama.status, RUNTIME_STATUS.IMPORTABLE);

  const notImportable = buildUniversalModelList({
    safetensorsModels: [{ id: "some/exotic", repo: "some/exotic", architecture: "rwkv", paramsB: 3 }],
  })[0];
  assert.equal(Object.fromEntries(notImportable.runtimes.map((x) => [x.id, x])).ollama.status, RUNTIME_STATUS.BLOCKED);
});

// ---------------------------------------------------------------------------
// Phase 8: gpt-oss / Kimi as first-class families. First-class = correctly
// described + routable + honest, NOT natively executed.
// ---------------------------------------------------------------------------
test("gpt-oss and kimi are first-class matrix families with capabilities", async () => {
  const { listCompatibilityMatrix } = await import("../src/runtime/compatibility/index.js");
  const families = Object.fromEntries(listCompatibilityMatrix().families.map((f) => [f.family, f]));
  for (const name of ["gpt-oss", "kimi"]) {
    assert.ok(families[name], `${name} must be a first-class family`);
    assert.equal(families[name].runtimes.native, "blocked", `${name} native must be honestly blocked`);
    assert.equal(families[name].runtimes["llama.cpp"], "supported", `${name} must be routable via llama.cpp`);
    assert.ok(families[name].capabilities.includes("moe"), `${name} must record its MoE capability`);
  }
});

test("selecting a gpt-oss GGUF yields a blocked native verdict with a real reason (not 'unknown')", async () => {
  const { evaluateNativeGgufCompatibility } = await import("../src/runtime/compatibility/index.js");
  const verdict = evaluateNativeGgufCompatibility({ gguf: { architecture: "gpt-oss", quant: "Q4_K_M", expertCount: 32 } });
  assert.equal(verdict.supported, false);
  assert.match(verdict.reason, /MoE|expert/i);
  assert.doesNotMatch(verdict.reason, /Unknown architectures/);
});

test("blocked and newly released GGUFs expose only native-compatible alternatives", () => {
  const models = buildUniversalModelList({
    catalogModels: [
      {
        id: "moonshot-kimi-k2-instruct",
        importedPath: "C:/models/kimi.gguf",
        gguf: { architecture: "deepseek2", quant: "Q4_K_M", paramsB: 32 },
      },
      {
        id: "llama-3.2-3b-instruct",
        importedPath: "C:/models/llama.gguf",
        gguf: {
          architecture: "llama",
          quant: "Q4_K_M",
          paramsB: 3.2,
          chatTemplate: "{{ messages }}",
        },
      },
      {
        id: "future-ssm",
        importedPath: "C:/models/future.gguf",
        gguf: { architecture: "mamba2", quant: "Q4_K_M", paramsB: 3 },
      },
    ],
  });
  const kimi = models.find((model) => model.id === "moonshot-kimi-k2-instruct");
  assert.equal(kimi.compatibilityRouting.available, true);
  assert.equal(kimi.compatibilityRouting.selectedRuntime, "llama.cpp");
  assert.match(kimi.compatibilityRouting.reason, /Kimi K2/i);
  assert.deepEqual(kimi.compatibilityRouting.alternatives.map((entry) => entry.id), ["llama-3.2-3b-instruct"]);
  assert.ok(kimi.compatibilityRouting.alternatives.every((entry) => ["supported", "importable"].includes(entry.nativeStatus)));

  const future = models.find((model) => model.id === "future-ssm");
  assert.match(future.compatibilityRouting.reason, /state-space/i);
  assert.deepEqual(future.compatibilityRouting.alternatives.map((entry) => entry.id), ["llama-3.2-3b-instruct"]);
});
