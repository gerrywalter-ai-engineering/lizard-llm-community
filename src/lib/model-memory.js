// Freeing the RAM/VRAM that local model runtimes are holding.
//
// One implementation, three callers: the CLI (`npm run free:memory`), the Pro
// Dashboard's "Free Model Memory" command, and the app's Free RAM button
// (POST /api/free-memory). The Windows uninstaller generates its own kill list
// from NATIVE_RUNNER_PROCESS_NAMES below, so it cannot drift either.
//
// That sharing is the point. The list used to be written out separately in the
// CLI script and in the installer, and they disagreed: the installer stopped
// lizard-native-runner.exe, its optimized CPU variants, and
// caterpillar-runner.exe, while the script knew only the first. Because
// `taskkill /IM` matches the image name EXACTLY, and because the AVX2 variant is
// the one that actually runs wherever the CPU supports it
// (selectNativeRunnerBinary prefers it after a self-test), "Free Model Memory"
// reported success while freeing nothing from the native lane - a runner stayed
// resident holding gigabytes with nothing pointing at it. A list that has to be
// correct in two files will eventually be correct in one.
import { execFile } from "node:child_process";
import os from "node:os";
import process from "node:process";

// Lizard's own runners. Kept separate because the installer stops exactly these
// (it has no business killing a user's llama.cpp or Ollama) while the free-memory
// sweep below stops all of them.
export const NATIVE_RUNNER_PROCESS_NAMES = [
  "lizard-native-runner",
  // NOT covered by the entry above - taskkill /IM is an exact match.
  "lizard-native-runner-avx2",
  "lizard-native-runner-avx512",
  "caterpillar-runner",
];

export const ENGINE_PROCESS_NAMES = [
  ...NATIVE_RUNNER_PROCESS_NAMES,
  "llama-server",
  "llama-cli",
  "llama-bench",
  "ollama_llama_server",
];

const OLLAMA_BASE_URL = "http://127.0.0.1:11434";

function run(command, args, { timeoutMs = 15000 } = {}) {
  return new Promise((resolve) => {
    execFile(command, args, { timeout: timeoutMs, windowsHide: true }, (error, stdout, stderr) => {
      resolve({
        ok: !error,
        stdout: String(stdout ?? ""),
        stderr: String(stderr ?? ""),
      });
    });
  });
}

async function fetchJson(url, init = null, timeoutMs = 5000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(url, { ...(init ?? {}), signal: controller.signal });
    const text = await response.text();
    return { ok: response.ok, status: response.status, body: text ? JSON.parse(text) : null };
  } catch (error) {
    return { ok: false, status: 0, body: null, error: error.message };
  } finally {
    clearTimeout(timer);
  }
}

export function availableMemoryGb() {
  return Number((os.freemem() / 1024 / 1024 / 1024).toFixed(2));
}

// Evicts loaded Ollama models without stopping the Ollama service itself -
// killing the service would be a bigger action than the user asked for, and it
// comes back with the models still loaded anyway.
async function unloadOllamaModels(log, requestJson) {
  if (isDryRun()) {
    log("[ollama] dry run: nothing was unloaded.");
    return 0;
  }
  const ps = await requestJson(`${OLLAMA_BASE_URL}/api/ps`);
  if (!ps.ok || !Array.isArray(ps.body?.models)) {
    log("[ollama] service not reachable or no models endpoint; skipping unload.");
    return 0;
  }
  const loaded = ps.body.models.map((entry) => entry?.name).filter(Boolean);
  if (loaded.length === 0) {
    log("[ollama] no models loaded.");
    return 0;
  }
  let unloaded = 0;
  for (const name of loaded) {
    // keep_alive: 0 tells Ollama to evict the model immediately.
    const result = await requestJson(`${OLLAMA_BASE_URL}/api/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: name, keep_alive: 0 }),
    }, 30000);
    log(`[ollama] unload ${name}: ${result.ok ? "ok" : `failed (${result.error ?? result.status})`}`);
    if (result.ok) {
      unloaded += 1;
    }
  }
  return unloaded;
}

// Reports what would be stopped without stopping it. Set
// LIZARD_FREE_MEMORY_DRY_RUN=1 to inspect the sweep on a machine you do not want
// disturbed - and used by the route test, which drives the real endpoint and
// would otherwise unload whatever the developer running `npm test` had warm.
function isDryRun() {
  return String(process.env.LIZARD_FREE_MEMORY_DRY_RUN ?? "") === "1";
}

async function stopEngineProcesses(log, runCommand) {
  const stopped = [];
  if (isDryRun()) {
    log("[engine] dry run: no processes were stopped.");
    return stopped;
  }
  for (const name of ENGINE_PROCESS_NAMES) {
    const result = process.platform === "win32"
      // /T takes the runner's children with it; a runner that spawned helpers
      // would otherwise leave them holding the weights it was asked to release.
      ? await runCommand("taskkill", ["/IM", `${name}.exe`, "/F", "/T"])
      : await runCommand("pkill", ["-f", name]);
    if (result.ok) {
      stopped.push(name);
      log(`[engine] stopped ${name}${process.platform === "win32" ? ".exe" : ""}`);
    }
  }
  if (stopped.length === 0) {
    log("[engine] no engine processes were running.");
  }
  return stopped;
}

/**
 * Frees model memory and reports what it actually freed.
 *
 * `releaseResidentWorkers` lets the in-process caller (the app server) hand its
 * own workers back through the normal release path FIRST, so the sweep below is
 * only ever a backstop for processes nothing owns. That ordering matters: a
 * worker killed out from under the server leaves the server believing a model is
 * still warm, and the residency rules are explicit that a claim is released
 * through the release functions, never by removing what it points at.
 *
 * Always resolves. Freeing an already-free machine is a success, not an error.
 */
export async function freeModelMemory({
  log = () => {},
  releaseResidentWorkers = null,
  settleMs = 1500,
  // Seams, same convention as selectNativeRunnerBinary's `selfTest`. This
  // function's whole job is killing processes on the real machine, so a test
  // that calls it for its reporting behaviour would otherwise kill whatever
  // runner a sibling test had spawned - which is exactly what happened, and
  // showed up as an unrelated engine-loop failure that only appeared under a
  // full parallel run.
  runCommand = run,
  requestJson = fetchJson,
} = {}) {
  const freeRamGbBefore = availableMemoryGb();
  log(`Free model memory: start (free RAM ${freeRamGbBefore} GB)`);

  let releasedResident = false;
  let releaseError = null;
  if (typeof releaseResidentWorkers === "function") {
    try {
      await releaseResidentWorkers();
      releasedResident = true;
    } catch (error) {
      // A failed graceful release is a reason to fall through to the sweep, not
      // a reason to leave the memory held.
      releaseError = String(error?.message ?? error);
      log(`[resident] graceful release failed, falling back to process stop: ${releaseError}`);
    }
  }

  const unloadedOllamaModels = await unloadOllamaModels(log, requestJson);
  const stopped = await stopEngineProcesses(log, runCommand);

  // Give the OS a moment to reclaim pages before measuring, or the report
  // understates what was freed.
  await new Promise((resolve) => setTimeout(resolve, settleMs));
  const freeRamGbAfter = availableMemoryGb();

  return {
    ok: true,
    releasedResident,
    releaseError,
    unloadedOllamaModels,
    stoppedEngineProcesses: stopped.length,
    stoppedProcessNames: stopped,
    freeRamGbBefore,
    freeRamGbAfter,
    freedGb: Number((freeRamGbAfter - freeRamGbBefore).toFixed(2)),
  };
}
