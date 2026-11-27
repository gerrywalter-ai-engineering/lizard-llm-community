import path from "node:path";
import process from "node:process";
import { randomUUID } from "node:crypto";
import { execFileSync, spawn } from "node:child_process";
import { assertTcpPortAvailable, waitForServerReady } from "../../lib/advanced-runtime.js";
import { resolveAppPath } from "../../lib/app-root.js";
import { writeNativeServerLaunchConfig } from "./launch-config.js";

function serverScriptPath(config = null, provider = "lizard-native") {
  const providerConfig = config?.engine?.[{ caterpillar: "caterpillar", ostrich: "ostrich" }[provider] ?? "native"];
  return providerConfig?.serverScript
    ? path.resolve(providerConfig.serverScript)
    : resolveAppPath("src", "runtime", "server", "standalone.js");
}

function buildProviderServerPlan({
  config,
  recommendation,
  modelArtifact,
  profile,
  host = null,
  port = null,
  // Internal benchmark/warm servers (dashboard warm pool, benchmark-big,
  // ci-benchmark) are infrastructure for the (feed-activated) Benchmark
  // feature, not the user-facing "Native API server" product that the `serve`
  // command exposes - so they are exempt from that server's Pro entitlement
  // gate. The user-facing `serve`/`run` commands leave this false and keep the
  // gate.
  internalBenchmarkServer = false,
  runtimeProvider = "lizard-native",
} = {}) {
  // Keyed per provider so Ostrich gets its own engine config, port and binary.
  // Reusing caterpillar's here made `serve --provider ostrich` start a
  // Caterpillar server on 8014 - a superficial alias, which the runner spec
  // explicitly forbids.
  const PROVIDER_ENGINE_KEY = { caterpillar: "caterpillar", ostrich: "ostrich" };
  const PROVIDER_DEFAULT_PORT = { caterpillar: 8014, ostrich: 8015 };
  const engineKey = PROVIDER_ENGINE_KEY[runtimeProvider] ?? "native";
  const providerConfig = config?.engine?.[engineKey];
  const resolvedHost = host ?? providerConfig?.host ?? config?.engine?.host ?? "127.0.0.1";
  const resolvedPort = Number(port ?? providerConfig?.port ?? config?.engine?.port ?? (PROVIDER_DEFAULT_PORT[runtimeProvider] ?? 8013));
  const launchId = randomUUID();
  const launchPath = writeNativeServerLaunchConfig({
    config,
    recommendation,
    modelArtifact,
    profile,
    host: resolvedHost,
    port: resolvedPort,
    internalBenchmarkServer,
    launchId,
    runtimeProvider,
  });
  const scriptPath = serverScriptPath(config, runtimeProvider);
  return {
    executable: process.execPath,
    args: [scriptPath, "--launch", launchPath],
    displayCommand: `${process.execPath} ${JSON.stringify(scriptPath)} --launch ${JSON.stringify(launchPath)}`,
    modelPath: modelArtifact?.modelPath ?? null,
    exists: true,
    host: resolvedHost,
    port: resolvedPort,
    launchPath,
    launchId,
    runtimeProvider,
  };
}

export function buildNativeServerPlan(options = {}) {
  return buildProviderServerPlan({ ...options, runtimeProvider: "lizard-native" });
}

export function buildCaterpillarServerPlan(options = {}) {
  return buildProviderServerPlan({ ...options, runtimeProvider: "caterpillar" });
}

export function buildOstrichServerPlan(options = {}) {
  return buildProviderServerPlan({ ...options, runtimeProvider: "ostrich" });
}

// Default readiness wait for the native GPU-resident server. Loading GGUF
// weights onto the GPU (plus first-run D3D12 shader compilation) can
// legitimately take well over a minute on a cold start - a fixed 60s here
// previously caused warm-up to report "failed" on slower/first-run hardware
// while the llama.cpp fallback path (startWarmLizardServer) already used a
// configurable 180s default via config.benchmark.ci.healthTimeoutSeconds.
// Critically, ensureWarmSelection swallows this failure and the benchmark
// proceeds anyway, so a too-short timeout here doesn't just delay warm-up -
// it makes every native (Q1/Q4/etc.) row fail near-instantly with an opaque
// "no stream fragments received" instead of ever reaching a working server.
const DEFAULT_NATIVE_SERVER_READY_TIMEOUT_SECONDS = 180;

// A readiness timeout/identity mismatch still requires caller-owned process
// cleanup. Previously neither caller did, so a timed-out warm-up left the
// native server (and any
// lizard-native-runner.exe grandchild it had already spawned to load the
// model onto the GPU) running as an orphan indefinitely: every failed
// warm-up permanently leaked GPU VRAM + RAM until the whole app was killed.
// spawn() doesn't put children in a Job Object on Windows, so a plain
// child.kill() wouldn't reach a grandchild - use taskkill's tree kill (/T)
// like the rest of the app's process cleanup (dashboard-command-center.js
// killProcessTree, benchmark-dashboard.ts stopProcessTree).
function processIsAlive(pid) {
  try {
    process.kill(Number(pid), 0);
    return true;
  } catch {
    return false;
  }
}

async function waitForProcessExit(pid, timeoutMs = 5000) {
  const deadline = Date.now() + timeoutMs;
  while (processIsAlive(pid) && Date.now() < deadline) {
    await new Promise((resolve) => setTimeout(resolve, 50));
  }
  return !processIsAlive(pid);
}

function windowsProcessRows() {
  try {
    const output = execFileSync("powershell", [
      "-NoProfile",
      "-NonInteractive",
      "-Command",
      "Get-CimInstance Win32_Process | Select-Object ProcessId,ParentProcessId | ConvertTo-Json -Compress",
    ], {
      encoding: "utf8",
      windowsHide: true,
      timeout: 15000,
    }).trim();
    if (!output) return [];
    const parsed = JSON.parse(output);
    return (Array.isArray(parsed) ? parsed : [parsed]).map((row) => ({
      pid: Number(row.ProcessId),
      ppid: Number(row.ParentProcessId),
    })).filter((row) => Number.isFinite(row.pid) && Number.isFinite(row.ppid));
  } catch {
    return null;
  }
}

function descendantProcessIds(rows, rootPid) {
  if (!Array.isArray(rows)) return [];
  const descendants = new Set();
  let parents = new Set([Number(rootPid)]);
  while (parents.size > 0) {
    const children = rows
      .filter((row) => parents.has(row.ppid) && !descendants.has(row.pid))
      .map((row) => row.pid);
    if (children.length === 0) break;
    parents = new Set(children);
    for (const pid of children) descendants.add(pid);
  }
  return [...descendants];
}

async function runTaskkill(pid) {
  return await new Promise((resolve) => {
    const killer = spawn("taskkill", ["/PID", String(pid), "/T", "/F"], {
      stdio: "ignore",
      windowsHide: true,
    });
    killer.once("error", () => resolve(false));
    killer.once("exit", (code) => resolve(code === 0));
  });
}

async function killChildTree(child) {
  if (!child?.pid) {
    return true;
  }
  const pid = Number(child.pid);
  if (process.platform === "win32") {
    const beforeRows = windowsProcessRows();
    if (beforeRows == null) {
      const error = new Error(
        `Native server process-tree inventory was unavailable for pid ${pid}; refusing a replacement model load.`,
      );
      error.code = "RESIDENCY_RELEASE_FAILED";
      throw error;
    }
    const tracked = new Set([pid, ...descendantProcessIds(beforeRows, pid)]);
    if (processIsAlive(pid)) {
      await runTaskkill(pid);
    } else {
      // Windows preserves ParentProcessId after the parent exits. Kill any
      // worker that outlived the HTTP process directly, then verify below.
      for (const descendantPid of [...tracked].filter((value) => value !== pid && processIsAlive(value))) {
        await runTaskkill(descendantPid);
      }
    }
    const afterRows = windowsProcessRows();
    if (afterRows == null) {
      const error = new Error(
        `Native server process-tree shutdown could not be verified for pid ${pid}; refusing a replacement model load.`,
      );
      error.code = "RESIDENCY_RELEASE_FAILED";
      throw error;
    }
    for (const descendantPid of descendantProcessIds(afterRows, pid)) {
      if (!tracked.has(descendantPid) && processIsAlive(descendantPid)) {
        await runTaskkill(descendantPid);
      }
      tracked.add(descendantPid);
    }
    const deadline = Date.now() + 5000;
    let alive = [...tracked].filter(processIsAlive);
    while (alive.length > 0 && Date.now() < deadline) {
      await new Promise((resolve) => setTimeout(resolve, 50));
      alive = [...tracked].filter(processIsAlive);
    }
    if (alive.length > 0) {
      const error = new Error(
        `Native server process-tree shutdown was not proven for pid ${pid}; still alive: ${alive.join(", ")}.`,
      );
      error.code = "RESIDENCY_RELEASE_FAILED";
      error.livePids = alive;
      throw error;
    }
    return true;
  }
  try {
    child.kill("SIGKILL");
  } catch {
    // Already exited.
  }
  if (!(await waitForProcessExit(pid))) {
    const error = new Error(`Native server pid ${pid} did not exit after SIGKILL.`);
    error.code = "RESIDENCY_RELEASE_FAILED";
    throw error;
  }
  return true;
}

export async function runNativeServerPlan(plan, {
  onSpawn = null,
  onExit = null,
  timeoutSeconds = DEFAULT_NATIVE_SERVER_READY_TIMEOUT_SECONDS,
  callerOwnsCleanup = false,
} = {}) {
  await assertTcpPortAvailable({ host: plan.host ?? "127.0.0.1", port: plan.port });
  const child = spawn(plan.executable, plan.args, {
    stdio: "inherit",
    windowsHide: true,
  });
  try {
    // Capture must happen inside the cleanup boundary. CLI callers record the
    // PID/tree monitor first and opt into owning the one verified cleanup;
    // other callers retain the historical self-cleaning default.
    await onSpawn?.(child);
    await waitForServerReady({
      port: plan.port,
      timeoutSeconds,
      expectedProcessId: plan.launchId ? child.pid : null,
      expectedLaunchId: plan.launchId ?? null,
      requireModelLoadTerminal: true,
    });
    await new Promise((resolve, reject) => {
      child.on("exit", () => resolve());
      child.on("error", reject);
    });
    onExit?.();
  } catch (error) {
    if (!callerOwnsCleanup) {
      try {
        await killChildTree(child);
      } catch (releaseError) {
        releaseError.cause = error;
        throw releaseError;
      }
    }
    throw error;
  }
}

export async function startNativeServerProcess(plan, {
  stdout = "pipe",
  stderr = "pipe",
  env = undefined,
  timeoutSeconds = DEFAULT_NATIVE_SERVER_READY_TIMEOUT_SECONDS,
  onSpawn = null,
  signal = null,
  callerOwnsAbortCleanup = false,
  callerOwnsCleanup = false,
} = {}) {
  if (signal?.aborted) {
    throw signal.reason instanceof Error ? signal.reason : new Error("Native server startup aborted.");
  }
  await assertTcpPortAvailable({ host: plan.host ?? "127.0.0.1", port: plan.port });
  // Cancellation may arrive while the asynchronous port preflight is in
  // progress. Recheck immediately before spawn; there is no await between
  // this guard and spawn, so a superseded warm transition cannot slip in.
  if (signal?.aborted) {
    throw signal.reason instanceof Error ? signal.reason : new Error("Native server startup aborted.");
  }
  const child = spawn(plan.executable, plan.args, {
    stdio: ["ignore", stdout, stderr],
    windowsHide: true,
    ...(env ? { env } : {}),
  });
  // Track the child's own exit in addition to the launch-id/PID health check.
  // Capture a stderr/stdout tail too so a startup crash
  // surfaces its real reason instead of an opaque timeout, and draining the
  // pipes prevents a full OS pipe buffer from stalling the server.
  let exited = null;
  let outputTail = "";
  const readinessController = new AbortController();
  const abortReadiness = () => readinessController.abort(
    signal?.reason instanceof Error ? signal.reason : new Error("Native server startup aborted."),
  );
  signal?.addEventListener("abort", abortReadiness, { once: true });
  const appendTail = (chunk) => { outputTail = (outputTail + String(chunk)).slice(-2000); };
  try {
    // Callers that request cleanup ownership register the child synchronously
    // here, before any readiness work can fail. The same owner must then
    // handle every post-spawn failure, not only an explicit abort.
    await onSpawn?.(child);
    if (child.stdout && typeof child.stdout.on === "function") child.stdout.on("data", appendTail);
    if (child.stderr && typeof child.stderr.on === "function") child.stderr.on("data", appendTail);
    await new Promise((resolve, reject) => {
      let settled = false;
      const settle = (fn, arg) => { if (!settled) { settled = true; fn(arg); } };
      waitForServerReady({
        port: plan.port,
        timeoutSeconds,
        signal: readinessController.signal,
        expectedProcessId: plan.launchId ? child.pid : null,
        expectedLaunchId: plan.launchId ?? null,
        requireModelLoadTerminal: true,
      }).then(
        () => settle(resolve),
        (error) => settle(reject, error),
      );
      // Plain callback (not a promise chain), so a late exit after readiness
      // simply records `exited` and no-ops - no unhandled rejection.
      child.once("exit", (code, signal) => {
        exited = { code, signal };
        const error = new Error(
          `native server process exited before becoming ready (code=${code} signal=${signal})`
          + (outputTail.trim() ? `: ${outputTail.trim()}` : ""),
        );
        readinessController.abort(error);
        settle(reject, error);
      });
    });
  } catch (error) {
    readinessController.abort(error);
    // Dashboard warm transitions register the child synchronously and own the
    // one authoritative /T /F cleanup. Letting both this readiness path and
    // the dashboard race taskkill makes the loser report process-not-found and
    // falsely fail closed even though the winner proved cleanup.
    if (!(callerOwnsCleanup || (callerOwnsAbortCleanup && signal?.aborted))) {
      try {
        await killChildTree(child);
      } catch (releaseError) {
        releaseError.cause = error;
        throw releaseError;
      }
    }
    throw error;
  } finally {
    signal?.removeEventListener("abort", abortReadiness);
  }
  readinessController.abort();
  // Readiness passed - but confirm it was THIS child that came up, not a stale
  // process on the same port. A child that already exited must never be
  // reported as a live warm server.
  if (exited || child.exitCode != null) {
    const error = new Error(
      `native server process exited immediately after readiness (code=${exited?.code ?? child.exitCode})`
      + (outputTail.trim() ? `: ${outputTail.trim()}` : ""),
    );
    if (!callerOwnsCleanup) {
      try {
        await killChildTree(child);
      } catch (releaseError) {
        releaseError.cause = error;
        throw releaseError;
      }
    }
    throw error;
  }
  return child;
}
