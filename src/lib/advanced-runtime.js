import { execFileSync } from "node:child_process";
import fs from "node:fs";
import net from "node:net";
import process from "node:process";

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export async function assertTcpPortAvailable({
  port,
  host = "127.0.0.1",
} = {}) {
  await new Promise((resolve, reject) => {
    const probe = net.createServer();
    let settled = false;
    const finish = (error = null) => {
      if (settled) return;
      settled = true;
      probe.removeAllListeners();
      if (error) reject(error);
      else resolve();
    };
    probe.once("error", (cause) => {
      // The most likely occupant is now Lizard's own long-running local API
      // endpoint, which holds this exact port while a model is activated. Say
      // so: "already occupied" on its own sends people hunting for a stray
      // process that is actually the app working as designed.
      const error = new Error(
        `Refusing to start a model server because ${host}:${port} is already occupied or unavailable (${cause.message}). `
        + "If Lizard is running with a model activated, its local API endpoint already owns this port and is serving that model - "
        + "use it directly, or unload the model in onboarding first.",
      );
      error.code = cause?.code === "EADDRINUSE" ? "SERVER_PORT_OCCUPIED" : (cause?.code ?? "SERVER_PORT_UNAVAILABLE");
      error.cause = cause;
      finish(error);
    });
    probe.listen(Number(port), host, () => {
      probe.close((error) => finish(error));
    });
  });
  return true;
}

function linuxListeningSocketInodes(port) {
  const inodes = new Set();
  for (const tablePath of ["/proc/net/tcp", "/proc/net/tcp6"]) {
    let rows;
    try {
      rows = fs.readFileSync(tablePath, "utf8").trim().split(/\r?\n/).slice(1);
    } catch {
      continue;
    }
    for (const row of rows) {
      const columns = row.trim().split(/\s+/);
      const portHex = String(columns[1] ?? "").split(":").pop();
      if (columns[3] === "0A"
        && Number.parseInt(portHex, 16) === Number(port)
        && columns[9]) {
        inodes.add(columns[9]);
      }
    }
  }
  return inodes;
}

export async function processOwnsListeningPort(pid, port) {
  const numericPid = Number(pid);
  const numericPort = Number(port);
  if (!Number.isFinite(numericPid) || numericPid <= 0 || !Number.isFinite(numericPort)) {
    return false;
  }
  if (process.platform === "win32") {
    try {
      const output = execFileSync("netstat", ["-ano", "-p", "TCP"], {
        encoding: "utf8",
        windowsHide: true,
        timeout: 5000,
      });
      return output.split(/\r?\n/).some((line) => {
        const columns = line.trim().split(/\s+/);
        if (String(columns[0] ?? "").toUpperCase() !== "TCP" || columns.length < 5) return false;
        const localPort = Number.parseInt(String(columns[1] ?? "").split(":").pop(), 10);
        const foreignPort = Number.parseInt(String(columns[2] ?? "").split(":").pop(), 10);
        return localPort === numericPort
          && foreignPort === 0
          && Number.parseInt(columns.at(-1), 10) === numericPid;
      });
    } catch {
      return false;
    }
  }
  if (process.platform === "linux") {
    const inodes = linuxListeningSocketInodes(numericPort);
    if (inodes.size === 0) return false;
    try {
      return fs.readdirSync(`/proc/${numericPid}/fd`).some((fd) => {
        try {
          const match = /^socket:\[(\d+)\]$/.exec(fs.readlinkSync(`/proc/${numericPid}/fd/${fd}`));
          return Boolean(match && inodes.has(match[1]));
        } catch {
          return false;
        }
      });
    } catch {
      return false;
    }
  }
  try {
    const output = execFileSync("lsof", [
      "-nP", "-a", "-p", String(numericPid), `-iTCP:${numericPort}`, "-sTCP:LISTEN", "-t",
    ], { encoding: "utf8", timeout: 5000 });
    return output.trim().split(/\s+/).map(Number).includes(numericPid);
  } catch {
    return false;
  }
}

export function assertServerHealthIdentity(response, {
  expectedProcessId = null,
  expectedLaunchId = null,
} = {}) {
  const actualProcessId = Number(response?.processId);
  const expectedPid = Number(expectedProcessId);
  const pidMatches = expectedProcessId == null
    || (Number.isFinite(actualProcessId) && actualProcessId === expectedPid);
  const launchMatches = expectedLaunchId == null
    || String(response?.launchId ?? "") === String(expectedLaunchId);
  if (pidMatches && launchMatches) {
    return true;
  }
  const error = new Error(
    "Server readiness came from a different process/launch than the model server that was just started.",
  );
  error.code = "SERVER_IDENTITY_MISMATCH";
  error.expectedProcessId = expectedProcessId;
  error.actualProcessId = Number.isFinite(actualProcessId) ? actualProcessId : null;
  error.expectedLaunchId = expectedLaunchId;
  error.actualLaunchId = response?.launchId ?? null;
  throw error;
}

async function fetchJsonWithTimeout({
  url,
  init,
  timeoutMs = 600000,
  fetchImpl = fetch,
  signal = null,
}) {
  const controller = new AbortController();
  const abortFromCaller = () => controller.abort(signal?.reason);
  if (signal?.aborted) abortFromCaller();
  else signal?.addEventListener("abort", abortFromCaller, { once: true });
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetchImpl(url, {
      ...init,
      signal: controller.signal,
    });
    const text = await response.text();
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${text}`);
    }
    return text ? JSON.parse(text) : {};
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener("abort", abortFromCaller);
  }
}

export function bytesToMiB(bytes) {
  if (bytes == null) {
    return null;
  }
  return Number((bytes / 1024 / 1024).toFixed(1));
}

function parseWorkingSetFromWindowsProcess(id) {
  try {
    const output = execFileSync("powershell.exe", [
      "-NoProfile",
      "-Command",
      `Get-Process -Id ${id} | Select-Object -ExpandProperty WorkingSet64`,
    ], {
      encoding: "utf8",
      stdio: ["ignore", "pipe", "ignore"],
      windowsHide: true,
      // Synchronous, so a PowerShell that is slow to start blocks the whole
      // event loop. A memory reading is worth degrading; the app is not.
      timeout: 5000,
      killSignal: "SIGKILL",
    }).trim();
    return Number.parseInt(output, 10);
  } catch {
    return null;
  }
}

function parseWorkingSetFromUnixProcess(id) {
  try {
    const output = execFileSync("ps", ["-o", "rss=", "-p", String(id)], {
      encoding: "utf8",
      stdio: ["ignore", "pipe", "ignore"],
      timeout: 5000,
      killSignal: "SIGKILL",
    }).trim();
    return Number.parseInt(output, 10) * 1024;
  } catch {
    return null;
  }
}

export function getProcessWorkingSetBytes(id) {
  if (!id) {
    return null;
  }
  if (process.platform === "win32") {
    return parseWorkingSetFromWindowsProcess(id);
  }
  return parseWorkingSetFromUnixProcess(id);
}

export class AdvancedRuntimeError extends Error {
  constructor(message, { stage = "unknown", metrics = null, cause = null } = {}) {
    super(message);
    this.name = "AdvancedRuntimeError";
    this.stage = stage;
    this.metrics = metrics;
    this.cause = cause;
  }
}

export async function waitForServerReady({
  port,
  timeoutSeconds,
  healthPath = "/health",
  fetchJsonImpl = fetchJsonWithTimeout,
  sleepImpl = sleep,
  signal = null,
  expectedProcessId = null,
  expectedLaunchId = null,
  requireModelLoadTerminal = false,
}) {
  const deadline = Date.now() + (timeoutSeconds * 1000);
  while (Date.now() < deadline) {
    if (signal?.aborted) {
      throw signal.reason ?? new Error("Server readiness wait aborted.");
    }
    try {
      const response = await fetchJsonImpl({
        url: `http://127.0.0.1:${port}${healthPath}`,
        init: { method: "GET" },
        timeoutMs: 5000,
        signal,
      });
      const modelLoadTerminal = response.warmLoadState === "ready" || response.warmLoadState === "failed";
      if (response.status === "ok") {
        assertServerHealthIdentity(response, { expectedProcessId, expectedLaunchId });
        if (!requireModelLoadTerminal || modelLoadTerminal) {
          return true;
        }
      }
    } catch (error) {
      if (signal?.aborted) {
        throw signal.reason ?? new Error("Server readiness wait aborted.");
      }
      if (error?.code === "SERVER_IDENTITY_MISMATCH") {
        throw error;
      }
      // keep polling until timeout
    }
    await sleepImpl(250);
  }
  throw new Error(`llama-server on port ${port} did not become ready within ${timeoutSeconds} seconds.`);
}

export async function requestServerChat({
  port,
  prompt,
  maxTokens,
  timeoutSeconds,
  temperature = 0,
  fetchJsonImpl = fetchJsonWithTimeout,
}) {
  return fetchJsonImpl({
    url: `http://127.0.0.1:${port}/v1/chat/completions`,
    init: {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        messages: [
          { role: "user", content: prompt },
        ],
        max_tokens: maxTokens,
        temperature,
      }),
    },
    timeoutMs: timeoutSeconds * 1000,
  });
}

function extractStreamDelta(payload) {
  const choices = Array.isArray(payload?.choices) ? payload.choices : [];
  return choices
    .map((choice) => choice?.delta?.content ?? choice?.message?.content ?? "")
    .join("");
}

export async function requestServerChatStream({
  port,
  prompt,
  maxTokens,
  timeoutSeconds,
  temperature = 0,
  now = () => Date.now(),
  fetchImpl = fetch,
}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutSeconds * 1000);
  const startedAt = now();
  try {
    const response = await fetchImpl(`http://127.0.0.1:${port}/v1/chat/completions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        messages: [
          { role: "user", content: prompt },
        ],
        max_tokens: maxTokens,
        temperature,
        stream: true,
      }),
      signal: controller.signal,
    });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    if (!response.body?.getReader) {
      throw new Error("Streaming response body is not readable.");
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let content = "";
    let firstTokenMs = null;
    let lastFragmentAt = null;
    let intervalTotalMs = 0;
    let intervalCount = 0;
    let fragmentsReceived = 0;

    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        break;
      }
      buffer += decoder.decode(value, { stream: true });
      while (true) {
        const boundaryIndex = buffer.indexOf("\n\n");
        if (boundaryIndex === -1) {
          break;
        }
        const rawEvent = buffer.slice(0, boundaryIndex);
        buffer = buffer.slice(boundaryIndex + 2);
        const dataLines = rawEvent
          .split(/\r?\n/)
          .filter((line) => line.startsWith("data:"))
          .map((line) => line.slice(5).trim())
          .filter(Boolean);
        for (const line of dataLines) {
          if (line === "[DONE]") {
            return {
              content: content.trim(),
              firstTokenMs,
              averageTokenIntervalMs: intervalCount > 0 ? Number((intervalTotalMs / intervalCount).toFixed(1)) : null,
              totalStreamMs: now() - startedAt,
              fragmentsReceived,
            };
          }
          let parsed;
          try {
            parsed = JSON.parse(line);
          } catch {
            continue;
          }
          const delta = extractStreamDelta(parsed);
          if (!delta) {
            continue;
          }
          const fragmentAt = now();
          if (firstTokenMs == null) {
            firstTokenMs = fragmentAt - startedAt;
          } else if (lastFragmentAt != null) {
            intervalTotalMs += fragmentAt - lastFragmentAt;
            intervalCount += 1;
          }
          lastFragmentAt = fragmentAt;
          fragmentsReceived += 1;
          content += delta;
        }
      }
    }

    return {
      content: content.trim(),
      firstTokenMs,
      averageTokenIntervalMs: intervalCount > 0 ? Number((intervalTotalMs / intervalCount).toFixed(1)) : null,
      totalStreamMs: now() - startedAt,
      fragmentsReceived,
    };
  } finally {
    clearTimeout(timer);
  }
}

async function timedStep(step, fn, now) {
  const startedAt = now();
  const result = await fn();
  return {
    result,
    durationMs: now() - startedAt,
    step,
  };
}

export async function measureAdvancedRuntimeSession({
  prompt,
  maxTokens,
  timeoutSeconds,
  temperature = 0,
  multipleRequestCount = 3,
  now = () => Date.now(),
  requestChatImpl = requestServerChat,
  requestStreamImpl = requestServerChatStream,
  sampleMemoryImpl = async () => null,
  requestContext = {},
}) {
  let peakRamMb = null;
  let peakVramMb = null;

  async function sampleMemory() {
    const sample = await sampleMemoryImpl();
    if (sample?.ramMb != null) {
      peakRamMb = peakRamMb == null ? sample.ramMb : Math.max(peakRamMb, sample.ramMb);
    }
    if (sample?.vramMb != null) {
      peakVramMb = peakVramMb == null ? sample.vramMb : Math.max(peakVramMb, sample.vramMb);
    }
  }

  await sampleMemory();

  const first = await timedStep("first_request", () => requestChatImpl({
    ...requestContext,
    prompt,
    maxTokens,
    timeoutSeconds,
    temperature,
  }), now);
  await sampleMemory();

  const repeated = await timedStep("repeated_prompt", () => requestChatImpl({
    ...requestContext,
    prompt,
    maxTokens,
    timeoutSeconds,
    temperature,
  }), now);
  await sampleMemory();

  const multipleRequestMs = [];
  for (let index = 0; index < multipleRequestCount; index += 1) {
    const measured = await timedStep(`multiple_request_${index + 1}`, () => requestChatImpl({
      ...requestContext,
      prompt: `${prompt}\n\n[advanced-runtime probe ${index + 1}]`,
      maxTokens,
      timeoutSeconds,
      temperature,
    }), now);
    multipleRequestMs.push(measured.durationMs);
    await sampleMemory();
  }

  const stream = await requestStreamImpl({
    ...requestContext,
    prompt,
    maxTokens,
    timeoutSeconds,
    temperature,
    now,
  });
  await sampleMemory();

  const firstContent = first.result?.choices?.[0]?.message?.content?.trim?.() ?? "";
  const firstTokens = first.result?.usage?.completion_tokens ?? null;

  return {
    firstResponseContent: firstContent,
    firstResponseTokens: firstTokens,
    firstRequestMs: first.durationMs,
    repeatedPromptMs: repeated.durationMs,
    repeatedPromptImprovementPercent: first.durationMs > 0
      ? Number((((first.durationMs - repeated.durationMs) / first.durationMs) * 100).toFixed(1))
      : null,
    multipleRequestMs,
    averageMultipleRequestMs: multipleRequestMs.length
      ? Number((multipleRequestMs.reduce((sum, value) => sum + value, 0) / multipleRequestMs.length).toFixed(1))
      : null,
    firstTokenMs: stream.firstTokenMs ?? null,
    averageTokenIntervalMs: stream.averageTokenIntervalMs ?? null,
    totalStreamMs: stream.totalStreamMs ?? null,
    fragmentsReceived: stream.fragmentsReceived ?? 0,
    peakRamMb,
    peakVramMb,
  };
}

export async function runManagedAdvancedRuntimeBenchmark({
  startRuntime,
  waitForReady,
  measureSession,
  stopRuntime,
  now = () => Date.now(),
}) {
  const metrics = {
    enabled: true,
    status: "running",
    startupMs: null,
    readyMs: null,
    firstRequestMs: null,
    repeatedPromptMs: null,
    repeatedPromptImprovementPercent: null,
    multipleRequestMs: [],
    averageMultipleRequestMs: null,
    firstTokenMs: null,
    averageTokenIntervalMs: null,
    totalStreamMs: null,
    peakRamMb: null,
    peakVramMb: null,
    fragmentsReceived: 0,
    cleanupMs: null,
    failureStage: null,
    error: null,
  };

  let runtime = null;
  try {
    const startupStartedAt = now();
    runtime = await startRuntime();
    metrics.startupMs = now() - startupStartedAt;

    const readyStartedAt = now();
    await waitForReady(runtime);
    metrics.readyMs = now() - readyStartedAt;

    const measured = await measureSession(runtime);
    Object.assign(metrics, measured);
    metrics.status = "completed";
    return {
      runtime,
      metrics,
    };
  } catch (error) {
    const stage = error?.stage
      ?? (metrics.readyMs == null ? "readiness" : metrics.firstRequestMs == null ? "measurement" : "runtime");
    metrics.status = "failed";
    metrics.failureStage = stage;
    metrics.error = error.message;
    throw new AdvancedRuntimeError(error.message, {
      stage,
      metrics,
      cause: error,
    });
  } finally {
    if (runtime) {
      const cleanupStartedAt = now();
      try {
        await stopRuntime(runtime);
      } finally {
        metrics.cleanupMs = now() - cleanupStartedAt;
      }
    }
  }
}
