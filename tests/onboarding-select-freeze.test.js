// The onboarding Select page froze the whole local server, and this is why.
//
// Select renders by enumerating every model source, and one of those sources is
// `ollama list` - which is not a file read but an RPC to the Ollama daemon,
// spawned SYNCHRONOUSLY. A daemon that is starting, loading a model, or wedged
// answers late or never, and a synchronous wait is a wait on the entire Node
// event loop: no models appear, no other request is served, and reloading the
// page does nothing because there is nobody left to serve the reload.
//
// The symptom pointed at model loading. It was never the loader - the page could
// not even finish listing.
import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import { listInstalledOllamaModels, resetOllamaListCache } from "../src/lib/ollama.js";

test("the ollama CLI is spawned with a deadline", () => {
  // Asserted against the source because the failure only reproduces against a
  // daemon that hangs, which a test cannot conjure - but a missing `timeout`
  // here is exactly the regression that froze the app, so it is worth pinning.
  const source = fs.readFileSync("src/lib/ollama.js", "utf8");
  const spawnBlock = source.slice(source.indexOf("function runOllama"), source.indexOf("OLLAMA_LIST_CACHE_TTL_MS"));
  assert.match(spawnBlock, /timeout: OLLAMA_CLI_TIMEOUT_MS/);
  assert.match(spawnBlock, /killSignal: "SIGKILL"/);
});

test("a wedged ollama is asked once, not on every render", () => {
  resetOllamaListCache();
  let calls = 0;
  const wedged = () => {
    calls += 1;
    const error = new Error("spawnSync ollama ETIMEDOUT");
    error.code = "ETIMEDOUT";
    throw error;
  };

  assert.deepEqual(listInstalledOllamaModels(null, { run: wedged }), []);
  assert.equal(calls, 1);

  // Bounding the spawn is not on its own enough. The success cache lives ~4s and
  // the dashboard polls every ~2s, so a failure cached under the success TTL
  // would pay the full CLI timeout again seconds later, and again after that -
  // the freeze would survive the timeout meant to end it. A failure has to be
  // remembered for much longer than a success.
  for (let i = 0; i < 20; i += 1) {
    listInstalledOllamaModels(null, { run: wedged });
  }
  assert.equal(calls, 1, "a failing ollama must not be re-spawned on every poll");

  resetOllamaListCache();
});

test("Select still lists ollama models when the daemon answers", () => {
  resetOllamaListCache();
  // The fix must not achieve stability by simply never reporting anything: a
  // working Ollama has to keep showing up in the list.
  const models = listInstalledOllamaModels(null, {
    run: () => [
      "NAME                ID              SIZE      MODIFIED",
      "llama3.2:3b         a80c4f17acd5    2.0 GB    2 days ago",
      "qwen2.5:7b          845dbda0ea48    4.7 GB    3 weeks ago",
    ].join("\n"),
  });
  assert.deepEqual(models.map((model) => model.tag), ["llama3.2:3b", "qwen2.5:7b"]);
  assert.equal(models[0].size, "2.0 GB");
  resetOllamaListCache();
});

test("an explicit refresh can clear the failure backoff", () => {
  resetOllamaListCache();
  let calls = 0;
  const failing = () => {
    calls += 1;
    throw new Error("ollama not running");
  };
  listInstalledOllamaModels(null, { run: failing });
  listInstalledOllamaModels(null, { run: failing });
  assert.equal(calls, 1);

  // Someone who starts Ollama mid-session should not have to wait out the whole
  // backoff to see their models.
  resetOllamaListCache();
  listInstalledOllamaModels(null, { run: failing });
  assert.equal(calls, 2);
  resetOllamaListCache();
});
