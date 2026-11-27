// Re-downloading a model that is already on disk.
//
// The reported failure: picking Llama-3.2-3B again transferred the whole 1.9 GB
// and then died on
//   EPERM, Permission denied: ...\models\llama-3.2-3b-instruct-gguf-q4_k_m.gguf
//
// Two separate mistakes met there. `fs.rmSync(outputPath, { force: true })` was
// used to clear the way for the rename, and `force` suppresses ENOENT - not
// EPERM; on Windows a memory-mapped file cannot be unlinked at all, and a loaded
// model holds its own GGUF. And nothing checked whether the file already needed
// downloading, so the entire transfer was spent to reach a step that could not
// succeed.
import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { downloadGgufModel } from "../src/lib/download-manager.js";

function sandbox() {
  const runtimeDirectory = fs.mkdtempSync(path.join(os.tmpdir(), "lizard-dl-"));
  const modelDirectory = path.join(runtimeDirectory, "models");
  fs.mkdirSync(modelDirectory, { recursive: true });
  return {
    runtimeDirectory,
    modelDirectory,
    catalogPath: path.join(runtimeDirectory, "model-catalog.json"),
  };
}

// A GGUF-shaped payload: the magic is what verifyGgufMagic checks.
function ggufBytes(size) {
  const buffer = Buffer.alloc(size);
  buffer.write("GGUF", 0, "latin1");
  return buffer;
}

function responseFor(body) {
  let cancelled = false;
  return {
    response: {
      ok: true,
      headers: new Map([["content-length", String(body.length)]]),
      body: {
        cancel: async () => { cancelled = true; },
      },
    },
    wasCancelled: () => cancelled,
  };
}

test("an identical file already on disk is not downloaded again", async () => {
  const config = sandbox();
  const payload = ggufBytes(4096);
  const target = path.join(config.modelDirectory, "demo-repo-q4_k_m.gguf");
  fs.writeFileSync(target, payload);

  let bodyRead = false;
  const { response, wasCancelled } = responseFor(payload);
  // Reading the body at all would mean the transfer happened.
  Object.defineProperty(response.headers, "get", {
    value: (key) => (key === "content-length" ? String(payload.length) : null),
  });

  const result = await downloadGgufModel({
    repo: "demo/repo",
    filename: "demo.Q4_K_M.gguf",
    modelId: "demo-repo-q4_k_m",
    config,
    fetchImpl: async () => {
      bodyRead = false;
      return response;
    },
  });

  assert.equal(result.alreadyPresent, true, "the file was already complete");
  assert.equal(result.bytes, payload.length);
  assert.equal(bodyRead, false);
  assert.equal(wasCancelled(), true, "the response body must be released, not left to time out");
  // Untouched: never unlinked, so a loaded model cannot collide with itself.
  assert.deepEqual(fs.readFileSync(target), payload);

  fs.rmSync(config.runtimeDirectory, { recursive: true, force: true });
});

test("a stale .download from an interrupted attempt is not left behind", async () => {
  const config = sandbox();
  const payload = ggufBytes(4096);
  const target = path.join(config.modelDirectory, "demo-repo-q4_k_m.gguf");
  const stale = `${target}.download`;
  fs.writeFileSync(target, payload);
  // The orphan a previous failure left: a full second copy of the model.
  fs.writeFileSync(stale, payload);

  await downloadGgufModel({
    repo: "demo/repo",
    filename: "demo.Q4_K_M.gguf",
    modelId: "demo-repo-q4_k_m",
    config,
    fetchImpl: async () => responseFor(payload).response,
  });

  assert.equal(fs.existsSync(stale), false, "nothing will ever resume it; it is just gigabytes");
  fs.rmSync(config.runtimeDirectory, { recursive: true, force: true });
});

test("a differing file still downloads, and releases the loaded copy first", async () => {
  const config = sandbox();
  const onDisk = ggufBytes(4096);
  const incoming = ggufBytes(8192);
  const target = path.join(config.modelDirectory, "demo-repo-q4_k_m.gguf");
  fs.writeFileSync(target, onDisk);

  const released = [];
  const result = await downloadGgufModel({
    repo: "demo/repo",
    filename: "demo.Q4_K_M.gguf",
    modelId: "demo-repo-q4_k_m",
    config,
    onBeforeReplace: ({ modelId }) => { released.push(modelId); },
    fetchImpl: async () => ({
      ok: true,
      headers: { get: (key) => (key === "content-length" ? String(incoming.length) : null) },
      body: (await import("node:stream")).Readable.toWeb((await import("node:stream")).Readable.from([incoming])),
    }),
  });

  assert.equal(result.alreadyPresent, undefined, "a different file must really be replaced");
  assert.equal(result.bytes, incoming.length);
  // The hook fires before the replace, which is the only moment at which
  // releasing the file still helps.
  assert.deepEqual(released, ["demo-repo-q4_k_m"]);
  assert.equal(fs.statSync(target).size, incoming.length);
  assert.equal(fs.existsSync(`${target}.download`), false);

  fs.rmSync(config.runtimeDirectory, { recursive: true, force: true });
});

test("a file that cannot be replaced fails with a message, not an errno", async () => {
  const config = sandbox();
  const onDisk = ggufBytes(4096);
  const incoming = ggufBytes(8192);
  const target = path.join(config.modelDirectory, "demo-repo-q4_k_m.gguf");
  fs.writeFileSync(target, onDisk);

  // Stand in for "a process still has this mapped", which is what Windows
  // reports as EPERM on the unlink.
  const realRm = fs.rmSync;
  fs.rmSync = (targetPath, options) => {
    if (targetPath === target) {
      const error = new Error("EPERM: operation not permitted");
      error.code = "EPERM";
      throw error;
    }
    return realRm(targetPath, options);
  };

  try {
    await assert.rejects(
      () => downloadGgufModel({
        repo: "demo/repo",
        filename: "demo.Q4_K_M.gguf",
        modelId: "demo-repo-q4_k_m",
        config,
        fetchImpl: async () => ({
          ok: true,
          headers: { get: (key) => (key === "content-length" ? String(incoming.length) : null) },
          body: (await import("node:stream")).Readable.toWeb((await import("node:stream")).Readable.from([incoming])),
        }),
      }),
      (error) => {
        // The user needs to know what is holding it and what to do - not a path
        // and a POSIX error name.
        assert.equal(error.code, "MODEL_FILE_IN_USE");
        assert.match(error.message, /in use/i);
        assert.match(error.message, /unload|free ram/i);
        return true;
      },
    );
  } finally {
    fs.rmSync = realRm;
  }

  // The failed attempt must not leave its gigabytes behind.
  assert.equal(fs.existsSync(`${target}.download`), false);
  fs.rmSync(config.runtimeDirectory, { recursive: true, force: true });
});
