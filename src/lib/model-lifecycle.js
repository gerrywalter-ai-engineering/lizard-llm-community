import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { randomUUID } from "node:crypto";
import { spawn } from "node:child_process";
import { ensureDirectory } from "./paths.js";
import {
  classifyTrackedPids,
  pidLooksRecycled,
  processIdentityIndex,
  processRows,
  snapshotProcessTree,
} from "./process-identity.js";

const SCHEMA_VERSION = 1;

function statePath(config) {
  return config?.runtimeStatePath
    ?? path.join(config?.runtimeDirectory ?? path.join(os.homedir(), ".lizard-llm"), "runtime-state.json");
}

function modelReleaseProofPath(config, pid) {
  return path.join(
    config?.runtimeDirectory ?? path.join(os.homedir(), ".lizard-llm"),
    "model-release-proofs",
    `${Number(pid)}.json`,
  );
}

function defaults() {
  return {
    schemaVersion: SCHEMA_VERSION,
    currentLoadedModel: null,
    lastStoppedModel: null,
    residencyIntent: null,
  };
}

// gguf (full metadata incl. tensor previews) and capabilityRegistry (full
// hardware/architecture compatibility matrix) are each independently ~10MB
// and were being persisted into runtime-state.json on every model start,
// then echoed back verbatim on every chat-completion response via
// snapshotLifecycleState() - a "Hello!" reply was riding along with ~38MB of
// dead weight. Neither field is read back for anything: restart-last only
// needs record.executable/record.args (see restartLastModel below).
const PROFILE_PERSIST_STRIP_KEYS = ["gguf", "capabilityRegistry"];

function trimProfileForPersistence(profile) {
  if (!profile || typeof profile !== "object") {
    return profile;
  }
  const trimmed = { ...profile };
  for (const key of PROFILE_PERSIST_STRIP_KEYS) {
    delete trimmed[key];
  }
  return trimmed;
}

export function isProcessAlive(pid) {
  if (!pid) {
    return false;
  }
  try {
    process.kill(Number(pid), 0);
    return true;
  } catch {
    return false;
  }
}

// Wall-clock instant the OS last booted. A process - and any GPU VRAM it held -
// recorded before this cannot possibly still exist: a reboot (the usual
// aftermath of a crash or BSOD mid-run) kills every process and releases all
// device memory. This is the one piece of evidence strong enough to lift the
// fail-closed residency guard WITHOUT a live process inventory, which is what
// otherwise leaves a machine permanently "unable to run again" after a crash.
export function osBootTimeMs(uptimeSeconds = os.uptime()) {
  const uptimeS = Number(uptimeSeconds);
  if (!Number.isFinite(uptimeS) || uptimeS <= 0) {
    return null;
  }
  return Date.now() - uptimeS * 1000;
}

// True when a recorded timestamp provably predates the current OS boot. The 3s
// tolerance absorbs uptime rounding and clock skew, so only records clearly
// from a previous session resolve as dead.
export function recordPredatesCurrentBoot(recordedAtIso, { bootTimeMs = osBootTimeMs() } = {}) {
  if (bootTimeMs == null) {
    return false;
  }
  const recordedMs = Date.parse(recordedAtIso ?? "");
  if (!Number.isFinite(recordedMs)) {
    return false;
  }
  return recordedMs < bootTimeMs - 3000;
}

export function loadRuntimeState(config) {
  const filePath = statePath(config);
  if (!fs.existsSync(filePath)) {
    return defaults();
  }
  try {
    return {
      ...defaults(),
      ...JSON.parse(fs.readFileSync(filePath, "utf8")),
    };
  } catch {
    return defaults();
  }
}

export function saveRuntimeState(config, partialState) {
  const next = {
    ...loadRuntimeState(config),
    ...partialState,
    schemaVersion: SCHEMA_VERSION,
  };
  const filePath = statePath(config);
  ensureDirectory(path.dirname(filePath));
  fs.writeFileSync(filePath, JSON.stringify(next, null, 2));
  return next;
}

export function refreshRuntimeState(config) {
  const state = loadRuntimeState(config);
  const current = state.currentLoadedModel;
  if (!current?.pid) {
    return state;
  }
  // A model recorded before the current OS boot is provably gone - the reboot
  // killed it and freed its VRAM. Resolve it as stopped even when its PID now
  // reads "alive" (recycled into an unrelated process) and even without a
  // readable process inventory, so a crash/BSOD mid-run does not wedge the
  // machine "unable to run again" forever. This takes precedence over the
  // fail-closed residencyReleaseUnproven retention below.
  if (recordPredatesCurrentBoot(current.startedAt)) {
    return saveRuntimeState(config, {
      currentLoadedModel: null,
      lastStoppedModel: {
        ...current,
        status: "stopped",
        residencyReleaseUnproven: false,
        stoppedAt: new Date().toISOString(),
        restartable: Boolean(current.executable && Array.isArray(current.args)),
        note: "Tracked runtime process predates the current OS boot; resolved as stopped after reboot.",
      },
    });
  }
  if (isProcessAlive(current.pid)) {
    return state;
  }
  // A vanished HTTP/root PID is not proof that its native worker descendants
  // released their model mappings. When a previous tree-kill attempt could
  // not be verified, retain ownership in persistent state so a later refresh
  // cannot silently convert "unknown residency" into "safe to reload".
  if (current.residencyReleaseUnproven === true) {
    return state;
  }
  return saveRuntimeState(config, {
    currentLoadedModel: null,
    lastStoppedModel: {
      ...current,
      status: "stopped",
      stoppedAt: new Date().toISOString(),
      restartable: Boolean(current.executable && Array.isArray(current.args)),
      note: current.note ?? "Tracked runtime process is no longer alive.",
    },
  });
}

// ---- residency intent ------------------------------------------------------
//
// Whether the user wants a model in memory right now.
//
// Loading is triggered from several places that each looked reasonable on their
// own: opening the onboarding page reopens the warm SSE, and app start resumes
// the activated model. Together they meant that freeing memory did not stick -
// you unloaded the model, watched the RAM come back, and the next page poll
// loaded it again. From the outside that is the app refusing to let go.
//
// So an explicit release records an intent, and every AUTOMATIC load checks it
// first. Nothing here blocks a load the user actually asked for: activating a
// model, sending a chat message, or starting a run all clear the latch, because
// asking for work is asking for the weights.
export function setModelResidencyIntent(config, intent, { reason = null } = {}) {
  const next = intent === "released" ? "released" : "resident";
  return saveRuntimeState(config, {
    residencyIntent: {
      state: next,
      reason: reason ?? null,
      at: new Date().toISOString(),
    },
  });
}

export function getModelResidencyIntent(config) {
  return loadRuntimeState(config).residencyIntent ?? null;
}

/**
 * True when the user has explicitly freed memory and has not since asked for a
 * model. Automatic loaders (boot resume, page-load warm-up) must honour this;
 * user-initiated loads must clear it with setModelResidencyIntent(…,"resident").
 */
export function automaticModelLoadSuspended(config) {
  return getModelResidencyIntent(config)?.state === "released";
}

export function getCurrentLoadedModel(config) {
  return refreshRuntimeState(config).currentLoadedModel ?? null;
}

export function getLastStoppedRuntimeModel(config) {
  return refreshRuntimeState(config).lastStoppedModel ?? null;
}

export function assertNoUnprovenModelResidency(config, {
  allowedPid = null,
  claimToken = null,
} = {}) {
  const state = refreshRuntimeState(config);
  const current = state.currentLoadedModel;
  const claim = readModelResidencyClaim(config);
  // Only a claim backed by a live process is real residency. One left by a
  // crashed run - dead owner, dead (or recycled) model tree - must be cleared
  // here rather than refusing every future load until someone deletes the lock
  // by hand. Recovery is evidence-based and returns false whenever anything is
  // still alive or unprovable, in which case the refusal below stands.
  if (claim && (!claimToken || claim.token !== claimToken)
    && !recoverStaleModelResidencyClaim(config)) {
    const error = new Error(
      claim.modelPid
        ? `Model residency is claimed by pid ${claim.modelPid} for ${claim.key?.provider ?? "another provider"}; refusing a second load.`
        : `A model load claim is ${claim.status ?? "unproven"} for owner pid ${claim.ownerPid ?? "unknown"}; refusing to cross the assert-to-spawn boundary.`,
    );
    error.code = "RESIDENCY_RELEASE_FAILED";
    error.pid = Number(claim.modelPid ?? claim.ownerPid) || null;
    error.claim = claim;
    error.lockPath = claim.lockPath ?? null;
    throw error;
  }
  const currentIsLive = Boolean(current?.pid && isProcessAlive(current.pid));
  // Legacy records predate residencyReleaseUnproven. A live legacy root still
  // owns model residency and must block a second load; only a dead legacy
  // record may be demoted by refreshRuntimeState above.
  if (currentIsLive || current?.residencyReleaseUnproven === true) {
    const sameLiveOwner = allowedPid != null
      && Number(current.pid) === Number(allowedPid)
      && currentIsLive;
    if (sameLiveOwner) return state;
    // A model that finished or crashed on its own leaves a dead root that
    // still reads residencyReleaseUnproven - and a recycled PID even reads
    // "alive". Recorded process-tree identity can prove that residency is
    // really gone; only that proof lifts the refusal.
    if (resolveExitedTrackedModelFromEvidence(config, current)) {
      return refreshRuntimeState(config);
    }
    const error = new Error(
      currentIsLive
        ? `Model residency is already owned by tracked pid ${current.pid}; refusing to load or register a second model.`
        : `Model residency for tracked pid ${current.pid} is unproven after its root exited; refusing to load or register another model.`,
    );
    error.code = "RESIDENCY_RELEASE_FAILED";
    error.pid = Number(current.pid);
    throw error;
  }
  return state;
}

export function registerRunningModel(config, {
  modelId,
  pid,
  profile = null,
  command = null,
  executable = null,
  args = [],
  source = "unknown",
  status = "running",
  note = null,
  claimToken = null,
}) {
  assertNoUnprovenModelResidency(config, { allowedPid: pid, claimToken });
  // Capture the tree WITH per-process start times, mirroring the residency
  // claim's model.json: this is the evidence that later proves "everything
  // exited on its own" (or "this PID was recycled") even when no claim
  // covers the record. An unavailable inventory records nothing - absence of
  // evidence must keep failing closed.
  const tree = snapshotProcessTree(pid);
  const payload = {
    modelId,
    pid,
    startedAt: new Date().toISOString(),
    ...(tree.inventoryAvailable ? {
      processIds: tree.processIds,
      processes: tree.processes,
    } : {}),
    profile: trimProfileForPersistence(profile),
    command,
    executable,
    args,
    source,
    status,
    note,
    // Active tree ownership remains unproven until the lifecycle manager (or
    // a verified external handoff proof) terminates the complete tree.
    residencyReleaseUnproven: true,
  };
  return saveRuntimeState(config, {
    currentLoadedModel: payload,
  }).currentLoadedModel;
}

export function markCurrentModelStopped(config, note = null, expectedPid = null) {
  const state = refreshRuntimeState(config);
  const current = state.currentLoadedModel;
  if (!current) {
    return state;
  }
  if (expectedPid != null && Number(current.pid) !== Number(expectedPid)) {
    return state;
  }
  writeModelResidencyReleaseProof(config, current, note);
  return saveRuntimeState(config, {
    currentLoadedModel: null,
    lastStoppedModel: {
      ...current,
      status: "stopped",
      residencyReleaseUnproven: false,
      stoppedAt: new Date().toISOString(),
      restartable: Boolean(current.executable && Array.isArray(current.args)),
      note: note ?? current.note ?? null,
    },
  });
}

function writeModelResidencyReleaseProof(config, current, note = null) {
  const pid = Number(current?.pid);
  const claim = readModelResidencyClaim(config);
  const claimMatches = claim?.modelPid != null && Number(claim.modelPid) === pid;
  const processIds = [...new Set([
    ...(Array.isArray(current?.processIds) ? current.processIds : []),
    ...(claimMatches && Array.isArray(claim.processIds) ? claim.processIds : []),
  ].map(Number).filter((value) => Number.isFinite(value) && value > 0))];
  const processes = [];
  const seenProcessIds = new Set();
  for (const entry of [
    ...(Array.isArray(current?.processes) ? current.processes : []),
    ...(claimMatches && Array.isArray(claim.processes) ? claim.processes : []),
  ]) {
    const entryPid = Number(entry?.pid);
    if (!Number.isFinite(entryPid) || entryPid <= 0 || seenProcessIds.has(entryPid)) continue;
    seenProcessIds.add(entryPid);
    processes.push(entry);
  }
  if (!Number.isFinite(pid) || pid <= 0 || !processIds.includes(pid)) {
    return null;
  }
  const filePath = modelReleaseProofPath(config, pid);
  const payload = {
    schemaVersion: 1,
    pid,
    startedAt: current.startedAt ?? null,
    releasedAt: new Date().toISOString(),
    processIds,
    processes,
    modelId: current.modelId ?? null,
    note: note ?? current.note ?? null,
  };
  ensureDirectory(path.dirname(filePath));
  const temporaryPath = `${filePath}.tmp-${process.pid}-${randomUUID()}`;
  try {
    fs.writeFileSync(temporaryPath, JSON.stringify(payload, null, 2));
    fs.renameSync(temporaryPath, filePath);
  } finally {
    fs.rmSync(temporaryPath, { force: true });
  }
  return payload;
}

// A lifecycle stop writes this certificate only after its captured root and
// every recorded descendant have been verified dead. The process start value
// binds it to the exact handed-off process rather than merely a reusable PID.
export function readVerifiedModelResidencyReleaseProof(config, {
  pid,
  startedAt,
  notBeforeMs = null,
} = {}) {
  const expectedPid = Number(pid);
  if (!Number.isFinite(expectedPid) || expectedPid <= 0 || !startedAt) return null;
  const filePath = modelReleaseProofPath(config, expectedPid);
  let proof;
  try {
    proof = JSON.parse(fs.readFileSync(filePath, "utf8"));
  } catch {
    return null;
  }
  if (Number(proof?.pid) !== expectedPid || proof?.startedAt !== startedAt) return null;
  const releasedAtMs = Date.parse(proof?.releasedAt ?? "");
  if (!Number.isFinite(releasedAtMs)) return null;
  if (Number.isFinite(Number(notBeforeMs)) && releasedAtMs < Number(notBeforeMs)) return null;
  if (!Array.isArray(proof.processIds) || !proof.processIds.map(Number).includes(expectedPid)) return null;
  const trackedPids = [...new Set(proof.processIds.map(Number)
    .filter((value) => Number.isFinite(value) && value > 0))];
  const identities = Array.isArray(proof.processes) ? proof.processes : [];
  if (!identities.some((entry) => Number(entry?.pid) === expectedPid && entry?.startedAt)) return null;
  return {
    ...proof,
    filePath,
    trackedPids,
  };
}

async function waitForProcessExit(pid, timeoutMs = 8000) {
  const startedAt = Date.now();
  while (Date.now() - startedAt < timeoutMs) {
    if (!isProcessAlive(pid)) {
      return true;
    }
    await new Promise((resolve) => setTimeout(resolve, 200));
  }
  return !isProcessAlive(pid);
}

async function runTaskkill(pid, force = false) {
  return new Promise((resolve) => {
    const args = ["/PID", String(pid), "/T", ...(force ? ["/F"] : [])];
    const child = spawn("taskkill", args, {
      stdio: "ignore",
      windowsHide: true,
    });
    child.once("error", () => resolve(false));
    // Only exit code 0 proves that Windows accepted the /T request for the
    // captured tree. A root that disappears concurrently can still leave its
    // native worker grandchild resident.
    child.once("exit", (code) => resolve(code === 0));
  });
}

async function readPosixProcessRows() {
  return await new Promise((resolve) => {
    const child = spawn("ps", ["-eo", "pid=,ppid="], {
      stdio: ["ignore", "pipe", "ignore"],
    });
    let output = "";
    child.stdout?.on("data", (chunk) => { output += String(chunk); });
    child.once("error", () => resolve(null));
    child.once("exit", (code) => {
      if (code !== 0) {
        resolve(null);
        return;
      }
      resolve(output.split(/\r?\n/).map((line) => {
        const [pid, ppid] = line.trim().split(/\s+/).map(Number);
        return { pid, ppid };
      }).filter((row) => Number.isFinite(row.pid) && Number.isFinite(row.ppid)));
    });
  });
}

async function readWindowsProcessRows() {
  return await new Promise((resolve) => {
    const child = spawn("powershell", [
      "-NoProfile",
      "-NonInteractive",
      "-Command",
      "Get-CimInstance Win32_Process | Select-Object ProcessId,ParentProcessId | ConvertTo-Json -Compress",
    ], {
      stdio: ["ignore", "pipe", "ignore"],
      windowsHide: true,
    });
    let output = "";
    let settled = false;
    const finish = (value) => {
      if (settled) return;
      settled = true;
      clearTimeout(killTimer);
      resolve(value);
    };
    // A hung WMI query must not stall shutdown (stopAndSnapshot awaits this).
    // Kill it and degrade to "inventory unavailable" after a bounded wait.
    const killTimer = setTimeout(() => {
      try { child.kill(); } catch { /* already gone */ }
      finish(null);
    }, 8000);
    child.stdout?.on("data", (chunk) => { output += String(chunk); });
    child.once("error", () => finish(null));
    child.once("exit", (code) => {
      if (code !== 0) {
        finish(null);
        return;
      }
      try {
        const parsed = JSON.parse(output.trim() || "[]");
        finish((Array.isArray(parsed) ? parsed : [parsed]).map((row) => ({
          pid: Number(row.ProcessId),
          ppid: Number(row.ParentProcessId),
        })).filter((row) => Number.isFinite(row.pid) && Number.isFinite(row.ppid)));
      } catch {
        finish(null);
      }
    });
  });
}

async function readPlatformProcessRows() {
  return process.platform === "win32"
    ? readWindowsProcessRows()
    : readPosixProcessRows();
}

function descendantProcessIds(rows, rootPid) {
  const descendants = [];
  let parents = new Set([Number(rootPid)]);
  while (parents.size > 0) {
    const children = rows
      .filter((row) => parents.has(row.ppid) && !descendants.includes(row.pid))
      .map((row) => row.pid);
    if (children.length === 0) break;
    descendants.push(...children);
    parents = new Set(children);
  }
  return descendants;
}

export async function captureTrackedProcessTreeSnapshot(pid, {
  readProcessRows = readPlatformProcessRows,
  processAlive = isProcessAlive,
} = {}) {
  const rootPid = Number(pid);
  const rootWasAliveBeforeCapture = processAlive(rootPid);
  if (!rootWasAliveBeforeCapture) {
    return {
      rootPid,
      capturedWhileRootAlive: false,
      trackedPids: [],
      capturedAt: null,
      reason: "root_not_alive_before_capture",
    };
  }

  const rows = await readProcessRows();
  const rootObserved = Array.isArray(rows)
    && rows.some((row) => Number(row.pid) === rootPid);
  const rootStillAlive = processAlive(rootPid);
  if (!rootObserved || !rootStillAlive) {
    return {
      rootPid,
      capturedWhileRootAlive: false,
      trackedPids: [],
      capturedAt: null,
      reason: !Array.isArray(rows)
        ? "process_inventory_unavailable"
        : "root_changed_state_during_capture",
    };
  }

  return {
    rootPid,
    capturedWhileRootAlive: true,
    trackedPids: [rootPid, ...descendantProcessIds(rows, rootPid)],
    capturedAt: new Date().toISOString(),
  };
}

export function monitorTrackedProcessTree(pid, {
  readProcessRows = readPlatformProcessRows,
  processAlive = isProcessAlive,
  // Each sample spawns a FULL process-inventory query (on Windows a
  // `Get-CimInstance Win32_Process` WMI enumeration). The old 50ms default ran
  // that ~20x/second for the ENTIRE model lifetime - indefinitely for `serve` -
  // hammering the WMI service and, alongside a GPU-resident runner, destabilizing
  // the machine. A model's native workers spawn within the first moments, so we
  // sample at a modest interval, then back off HARD once the tree stops changing
  // (a slow keepalive still catches a late child within `idlePollIntervalMs`).
  pollIntervalMs = 500,
  idlePollIntervalMs = 5000,
  stableSamplesToIdle = 4,
} = {}) {
  const rootPid = Number(pid);
  const trackedPids = new Set([rootPid]);
  let capturedWhileRootAlive = false;
  let capturedAt = null;
  let stopRequested = false;

  const sampling = (async () => {
    let stableSamples = 0;
    let lastSize = trackedPids.size;
    while (!stopRequested && processAlive(rootPid)) {
      const snapshot = await captureTrackedProcessTreeSnapshot(rootPid, {
        readProcessRows,
        processAlive,
      });
      if (snapshot.capturedWhileRootAlive) {
        capturedWhileRootAlive = true;
        capturedAt = snapshot.capturedAt;
        for (const processId of snapshot.trackedPids) trackedPids.add(processId);
      }
      if (stopRequested || !processAlive(rootPid)) break;
      // Back off once the captured tree stabilizes: after several unchanged
      // samples a native runner has finished spawning its workers, so a slow
      // keepalive interval replaces the fast one - turning thousands of WMI
      // enumerations per run into a handful.
      if (capturedWhileRootAlive && trackedPids.size === lastSize) {
        stableSamples += 1;
      } else {
        stableSamples = 0;
        lastSize = trackedPids.size;
      }
      const interval = stableSamples >= stableSamplesToIdle ? idlePollIntervalMs : pollIntervalMs;
      await new Promise((resolve) => setTimeout(resolve, interval));
    }
  })();

  return {
    rootPid,
    async stopAndSnapshot() {
      stopRequested = true;
      await sampling;
      return {
        rootPid,
        capturedWhileRootAlive,
        trackedPids: capturedWhileRootAlive ? [...trackedPids] : [],
        capturedAt,
      };
    },
  };
}

export async function runPosixProcessTreeKill(pid, force = true, {
  readProcessRows = readPosixProcessRows,
  processAlive = isProcessAlive,
  signalProcess = (processId, signal) => process.kill(processId, signal),
  wait = (milliseconds) => new Promise((resolve) => setTimeout(resolve, milliseconds)),
  verificationTimeoutMs = 5000,
  maxFreezePasses = 32,
} = {}) {
  const rootPid = Number(pid);
  // Once the root is already gone its children may have been re-parented and
  // cannot be proven from a fresh process table. Unknown is not success.
  if (!Number.isFinite(rootPid) || rootPid <= 0 || !processAlive(rootPid)) return false;

  const tracked = new Set([rootPid]);
  const frozen = new Set();
  const signalOne = (processId, signal) => {
    try {
      signalProcess(processId, signal);
      return true;
    } catch (error) {
      if (error?.code === "ESRCH") return false;
      throw error;
    }
  };
  const killTrackedBestEffort = () => {
    for (const processId of [...tracked].reverse()) {
      try {
        signalProcess(processId, force ? "SIGKILL" : "SIGTERM");
      } catch {
        // Verification below is authoritative.
      }
    }
  };

  try {
    // Freeze the root before inventory so it cannot fork between the tree walk
    // and termination. Freeze each discovered generation and repeat until a
    // complete pass finds no new descendants.
    if (!signalOne(rootPid, "SIGSTOP")) return false;
    frozen.add(rootPid);
    let stabilized = false;
    for (let pass = 0; pass < maxFreezePasses; pass += 1) {
      const rows = await readProcessRows();
      if (!Array.isArray(rows) || !rows.some((row) => Number(row.pid) === rootPid)) {
        throw new Error(`POSIX model root pid ${rootPid} disappeared before its descendant tree was frozen.`);
      }
      const descendants = descendantProcessIds(rows, rootPid);
      const newlyDiscovered = descendants.filter((processId) => !tracked.has(processId));
      for (const processId of newlyDiscovered) tracked.add(processId);
      for (const processId of descendants) {
        if (frozen.has(processId)) continue;
        if (signalOne(processId, "SIGSTOP")) frozen.add(processId);
        else if (processAlive(processId)) {
          throw new Error(`POSIX model descendant pid ${processId} could not be frozen.`);
        }
      }
      if (newlyDiscovered.length === 0) {
        stabilized = true;
        break;
      }
    }
    if (!stabilized) {
      throw new Error(`POSIX model process tree for pid ${rootPid} did not stabilize while freezing descendants.`);
    }

    // Descendants first and root last. Every captured process is stopped, so
    // none can create an untracked branch after the final inventory.
    killTrackedBestEffort();
    const deadline = Date.now() + Math.max(0, Number(verificationTimeoutMs) || 0);
    while ([...tracked].some(processAlive) && Date.now() < deadline) {
      await wait(50);
    }
    return ![...tracked].some(processAlive);
  } catch {
    // Never resume a partially frozen tree. Make a best-effort kill and retain
    // the residency claim by returning an unproven result.
    killTrackedBestEffort();
    return false;
  }
}

async function terminateTrackedProcessTree(pid, force = true) {
  return process.platform === "win32"
    ? runTaskkill(pid, force)
    : runPosixProcessTreeKill(pid, force);
}

export async function requireTrackedProcessTreeTermination(pid, {
  terminateTree = terminateTrackedProcessTree,
} = {}) {
  const processId = Number(pid);
  if (await terminateTree(processId, true) !== true) {
    const error = new Error(
      `Tracked model process ${processId} changed state before descendant-tree termination was proven.`,
    );
    error.code = "RESIDENCY_RELEASE_FAILED";
    throw error;
  }
  return true;
}

export async function recordTrackedProcessExit(config, pid, {
  note = "Tracked CLI model process exited and no live descendants remained.",
  processAlive = isProcessAlive,
  preExitTreeSnapshot = null,
  trustedLeafCleanExit = false,
} = {}) {
  const processId = Number(pid);
  const state = loadRuntimeState(config);
  const current = state.currentLoadedModel;
  if (!current || Number(current.pid) !== processId) {
    return { stopped: false, reason: "not_current_model", state };
  }

  let failureReason = null;
  let liveDescendantPids = [];
  if (processAlive(processId)) {
    failureReason = `tracked root pid ${processId} is still alive`;
  } else if (trustedLeafCleanExit === true) {
    // This escape hatch is intentionally explicit and narrow. Callers may
    // use it only for a process they authoritatively know cannot create child
    // processes; CLI model servers use captured process-tree snapshots.
  } else {
    const snapshotIsValid = preExitTreeSnapshot?.capturedWhileRootAlive === true
      && Number(preExitTreeSnapshot.rootPid) === processId
      && Array.isArray(preExitTreeSnapshot.trackedPids)
      && preExitTreeSnapshot.trackedPids.some((trackedPid) => Number(trackedPid) === processId);
    if (!snapshotIsValid) {
      // A post-exit PPID walk is never proof on POSIX: descendants are
      // reparented as soon as the root exits. Require an inventory captured
      // while the root was alive so those PIDs remain independently testable.
      failureReason = `no valid process-tree snapshot was captured while pid ${processId} was alive`;
    } else {
      liveDescendantPids = [...new Set(preExitTreeSnapshot.trackedPids.map(Number))]
        .filter((trackedPid) => trackedPid !== processId && processAlive(trackedPid));
      if (liveDescendantPids.length > 0) {
        failureReason = `live captured descendant pid(s) remain: ${liveDescendantPids.join(", ")}`;
      }
    }
  }

  if (failureReason) {
    const error = new Error(
      `Model residency release after CLI process exit was not proven (${failureReason}).`,
    );
    error.code = "RESIDENCY_RELEASE_FAILED";
    error.pid = processId;
    error.liveDescendantPids = liveDescendantPids;
    saveRuntimeState(config, {
      currentLoadedModel: {
        ...current,
        status: "release_failed",
        residencyReleaseUnproven: true,
        note: error.message,
      },
    });
    throw error;
  }

  const stoppedState = markCurrentModelStopped(config, note, processId);
  releaseModelResidencyClaim(config, { pid: processId, treeReleaseProven: true });
  return { stopped: true, state: stoppedState };
}

function modelResidencyClaimRoot(config, directory = null) {
  return path.resolve(
    directory
      ?? config?.modelLifecycleClaimDirectory
      ?? process.env.LIZARD_MODEL_CLAIM_DIRECTORY
      ?? path.join(os.homedir(), ".lizard-llm", "locks"),
  );
}

function modelResidencyClaimPath(config, directory = null) {
  return path.join(modelResidencyClaimRoot(config, directory), ".model-residency-claim.lock");
}

function readClaimJson(filePath) {
  try {
    return JSON.parse(fs.readFileSync(filePath, "utf8"));
  } catch {
    return null;
  }
}

const CLAIM_MUTATION_WAIT_CELL = new Int32Array(new SharedArrayBuffer(4));

function waitForClaimMutation(milliseconds) {
  Atomics.wait(CLAIM_MUTATION_WAIT_CELL, 0, 0, Math.max(1, Number(milliseconds) || 1));
}

function renameClaimDirectorySync(source, destination, timeoutMs = 1000) {
  const deadline = Date.now() + Math.max(0, Number(timeoutMs) || 0);
  while (true) {
    try {
      fs.renameSync(source, destination);
      return;
    } catch (error) {
      // A concurrent reader can transiently deny a directory rename on
      // Windows even though its file handle is immediately closed.
      if (!["EPERM", "EACCES"].includes(error?.code) || Date.now() >= deadline) throw error;
      waitForClaimMutation(5);
    }
  }
}

function modelResidencyMutationLockPath(config, directory = null) {
  return `${modelResidencyClaimPath(config, directory)}.mutation-lock`;
}

function residencyMutationError(message, details = {}) {
  const error = new Error(message);
  error.code = "RESIDENCY_RELEASE_FAILED";
  Object.assign(error, details);
  return error;
}

// Model claims are shared across the dashboard, benchmark, and CLI processes.
// Token validation followed by a later write/rename is not a CAS: another
// token owner could remove the old directory and create a new generation in
// between. This short per-user mutex serializes every claim creator, mutator,
// and remover, and each operation revalidates its token while holding it.
function acquireModelResidencyMutationLock(config, {
  directory = null,
  waitTimeoutMs = 10000,
  initializationGraceMs = 30000,
  processAlive = isProcessAlive,
  listProcesses = processRows,
} = {}) {
  const lockPath = modelResidencyMutationLockPath(config, directory);
  const ownerPath = path.join(lockPath, "owner.json");
  const deadline = Date.now() + Math.max(0, Number(waitTimeoutMs) || 0);
  ensureDirectory(path.dirname(lockPath));

  while (true) {
    const token = randomUUID();
    try {
      fs.mkdirSync(lockPath);
      try {
        fs.writeFileSync(ownerPath, JSON.stringify({
          token,
          ownerPid: process.pid,
          startedAt: new Date().toISOString(),
        }, null, 2));
      } catch (error) {
        fs.rmSync(lockPath, { recursive: true, force: true });
        throw error;
      }

      let released = false;
      return () => {
        if (released) return false;
        const owner = readClaimJson(ownerPath);
        if (owner?.token !== token) {
          throw residencyMutationError(
            "Model residency mutation-lock ownership changed before release.",
            { mutationLockOwner: owner },
          );
        }
        const quarantine = `${lockPath}.release-${process.pid}-${randomUUID()}`;
        renameClaimDirectorySync(lockPath, quarantine);
        const movedOwner = readClaimJson(path.join(quarantine, "owner.json"));
        if (movedOwner?.token !== token) {
          try {
            renameClaimDirectorySync(quarantine, lockPath);
          } catch {
            // Preserve the quarantined evidence and fail closed if restoration
            // cannot be completed.
          }
          throw residencyMutationError(
            "Model residency mutation-lock token changed during atomic release.",
            { mutationLockOwner: movedOwner },
          );
        }
        fs.rmSync(quarantine, { recursive: true, force: true });
        released = true;
        return true;
      };
    } catch (error) {
      if (error?.code !== "EEXIST") throw error;
    }

    const owner = readClaimJson(ownerPath);
    const ownerPid = Number(owner?.ownerPid);
    // Same PID-is-not-identity rule as the claim acquire: a mutation lock whose
    // owner PID was recycled into an unrelated process must not be waited on
    // (that just stalls until the deadline, then throws) - treat it as stale and
    // reclaim it below. pidLooksRecycled fails closed, so a genuinely-held lock
    // is still honored.
    const ownerHeld = owner?.token && Number.isFinite(ownerPid) && ownerPid > 0 && processAlive(ownerPid)
      && !pidLooksRecycled(ownerPid, owner.startedAt ?? null, { listProcesses });
    if (ownerHeld) {
      // A synchronous re-entry in this process can never make progress while
      // the current stack owns the lock; surface it instead of deadlocking.
      if (ownerPid === process.pid) {
        throw residencyMutationError("Re-entrant model residency claim mutation is not allowed.", {
          mutationLockOwner: owner,
        });
      }
      if (Date.now() >= deadline) {
        throw residencyMutationError(
          `Timed out waiting for model residency claim mutation owned by pid ${ownerPid}.`,
          { mutationLockOwner: owner },
        );
      }
      waitForClaimMutation(Math.min(10, Math.max(1, deadline - Date.now())));
      continue;
    }

    if (!owner?.token) {
      let ageMs;
      try {
        ageMs = Date.now() - fs.statSync(lockPath).mtimeMs;
      } catch (error) {
        if (error?.code === "ENOENT") continue;
        throw error;
      }
      if (ageMs < initializationGraceMs) {
        if (Date.now() >= deadline) {
          throw residencyMutationError("Timed out waiting for a model residency claim mutation to initialize.");
        }
        waitForClaimMutation(Math.min(10, Math.max(1, deadline - Date.now())));
        continue;
      }
    }

    const quarantine = `${lockPath}.stale-${process.pid}-${randomUUID()}`;
    try {
      renameClaimDirectorySync(lockPath, quarantine);
    } catch (error) {
      if (["ENOENT", "EEXIST", "EPERM", "EACCES"].includes(error?.code)) continue;
      throw error;
    }
    fs.rmSync(quarantine, { recursive: true, force: true });
  }
}

function withModelResidencyMutationLock(config, { directory = null } = {}, operation) {
  const release = acquireModelResidencyMutationLock(config, { directory });
  try {
    return operation();
  } finally {
    release();
  }
}

export function normalizeModelResidencyKey({ provider, modelPath, quant } = {}) {
  let canonicalPath;
  try {
    canonicalPath = fs.realpathSync.native(String(modelPath ?? ""));
  } catch {
    canonicalPath = path.resolve(String(modelPath ?? ""));
  }
  if (process.platform === "win32") canonicalPath = canonicalPath.toLowerCase();
  const key = {
    provider: String(provider ?? "").trim().toLowerCase(),
    modelPath: canonicalPath,
    quant: String(quant ?? "").trim().toUpperCase(),
  };
  if (!key.provider || !String(modelPath ?? "").trim() || !key.quant) {
    const error = new Error("Model residency claims require an exact provider, model path, and quantization key.");
    error.code = "INVALID_MODEL_RESIDENCY_KEY";
    throw error;
  }
  return key;
}

export function modelResidencyKeyForOllama(config, tag) {
  const normalizedTag = String(tag ?? "").trim().toLowerCase();
  if (!normalizedTag) {
    const error = new Error("An Ollama model tag is required for a residency key.");
    error.code = "INVALID_MODEL_RESIDENCY_KEY";
    throw error;
  }
  const canonicalTag = normalizedTag.includes(":") || normalizedTag.includes("@")
    ? normalizedTag
    : `${normalizedTag}:latest`;
  const runtimeRoot = config?.runtimeDirectory
    ?? path.join(os.homedir(), ".lizard-llm");
  return normalizeModelResidencyKey({
    provider: "ollama",
    modelPath: path.resolve(
      runtimeRoot,
      "ollama-residency",
      encodeURIComponent(canonicalTag),
    ),
    quant: "NONE",
  });
}

export function readModelResidencyClaim(config, { directory = null } = {}) {
  const lockPath = modelResidencyClaimPath(config, directory);
  if (!fs.existsSync(lockPath)) return null;
  const owner = readClaimJson(path.join(lockPath, "owner.json"));
  const model = readClaimJson(path.join(lockPath, "model.json"));
  const failure = readClaimJson(path.join(lockPath, "failure.json"));
  if (!owner) {
    return {
      status: "initializing",
      lockPath,
      ownerPid: null,
      modelPid: model?.modelPid ?? null,
      key: null,
      token: null,
    };
  }
  return {
    ...owner,
    ...(model ?? {}),
    ...(failure ?? {}),
    status: failure?.status ?? (model?.modelPid ? "resident" : (owner.status ?? "loading")),
    lockPath,
  };
}

function residencyKeysEqual(left, right) {
  return left?.provider === right?.provider
    && left?.modelPath === right?.modelPath
    && left?.quant === right?.quant;
}

function requireClaimToken(config, token, { directory = null } = {}) {
  const claim = readModelResidencyClaim(config, { directory });
  if (!claim || !token || claim.token !== token) {
    const error = new Error("Model residency claim token is missing, stale, or owned by another process.");
    error.code = "RESIDENCY_RELEASE_FAILED";
    error.claim = claim;
    throw error;
  }
  return claim;
}

function writeClaimJsonAtomically(filePath, payload) {
  const tempPath = `${filePath}.tmp-${process.pid}-${randomUUID()}`;
  try {
    fs.writeFileSync(tempPath, JSON.stringify(payload, null, 2));
    fs.renameSync(tempPath, filePath);
  } finally {
    fs.rmSync(tempPath, { force: true });
  }
}

export function validateModelResidencyClaim(config, {
  token,
  key,
  pid = null,
  directory = null,
} = {}) {
  const claim = requireClaimToken(config, token, { directory });
  const normalizedKey = normalizeModelResidencyKey(key);
  if (!residencyKeysEqual(claim.key, normalizedKey)) {
    const error = new Error("Model residency claim does not match the exact provider/model/Q handoff key.");
    error.code = "RESIDENCY_RELEASE_FAILED";
    error.claim = claim;
    error.expectedKey = normalizedKey;
    throw error;
  }
  if (pid != null && Number(claim.modelPid) !== Number(pid)) {
    const error = new Error(
      `Model residency claim pid ${claim.modelPid ?? "unknown"} does not match handoff pid ${pid}.`,
    );
    error.code = "RESIDENCY_RELEASE_FAILED";
    error.claim = claim;
    throw error;
  }
  return claim;
}

export function updateModelResidencyClaim(config, options = {}) {
  const { directory = null } = options;
  return withModelResidencyMutationLock(config, { directory }, () => (
    updateModelResidencyClaimUnlocked(config, options)
  ));
}

function updateModelResidencyClaimUnlocked(config, {
  token,
  key,
  modelPid = null,
  previousPid = null,
  previousTreeReleaseProven = false,
  previousResidencyEmptyProven = false,
  directory = null,
} = {}) {
  const claim = requireClaimToken(config, token, { directory });
  const normalizedKey = normalizeModelResidencyKey(key);
  const nextHasPid = modelPid != null;
  const nextPid = nextHasPid ? Number(modelPid) : null;
  if (nextHasPid && (!Number.isFinite(nextPid) || nextPid <= 0)) {
    const error = new Error("A positive spawned model PID is required to attach a residency claim.");
    error.code = "INVALID_MODEL_RESIDENCY_PID";
    throw error;
  }

  if (claim.modelPid != null) {
    if (nextHasPid && Number(claim.modelPid) === nextPid && residencyKeysEqual(claim.key, normalizedKey)) {
      // Re-attaching the same model refreshes (and merges) its process tree.
      // The first attach runs immediately after spawn and may see the root plus
      // a conhost but not the native worker yet; "more than one PID" therefore
      // is not proof that the snapshot is complete. The post-readiness attach
      // records every late descendant without dropping identities captured
      // earlier.
      return writeClaimModelJson(config, claim, {
        modelPid: nextPid,
        key: normalizedKey,
        attachedAt: claim.attachedAt ?? new Date().toISOString(),
        previousPid: claim.previousPid ?? null,
        previousTreeReleaseProven: claim.previousTreeReleaseProven === true,
        previousResidencyEmptyProven: claim.previousResidencyEmptyProven === true,
        mergeExistingTree: true,
        directory,
      });
    }
    if (previousTreeReleaseProven !== true || Number(previousPid) !== Number(claim.modelPid)) {
      const error = new Error(
        `Cannot move residency claim from pid ${claim.modelPid} to ${nextPid ?? "PID-less residency"} without verified release of the prior tree.`,
      );
      error.code = "RESIDENCY_RELEASE_FAILED";
      error.claim = claim;
      throw error;
    }
  } else if (!nextHasPid && residencyKeysEqual(claim.key, normalizedKey)) {
    return claim;
  } else if (!residencyKeysEqual(claim.key, normalizedKey)
    && previousResidencyEmptyProven !== true) {
    const error = new Error(
      "Cannot change a PID-less residency claim key without authoritative proof that the prior provider has no resident model.",
    );
    error.code = "RESIDENCY_RELEASE_FAILED";
    error.claim = claim;
    throw error;
  }

  return writeClaimModelJson(config, claim, {
    modelPid: nextPid,
    key: normalizedKey,
    attachedAt: new Date().toISOString(),
    previousPid: previousPid == null ? null : Number(previousPid),
    previousTreeReleaseProven: previousTreeReleaseProven === true,
    previousResidencyEmptyProven: previousResidencyEmptyProven === true,
    directory,
  });
}

// The claim records the model's WHOLE process tree, with each member's start
// time, not just the root PID. Those start times are what let a later run tell
// the processes we tracked apart from whatever inherits their PIDs after a
// crash - and the descendants are what make "nothing is resident" provable at
// all, since a dead root says nothing about a native worker still holding VRAM.
function writeClaimModelJson(config, claim, {
  modelPid,
  key,
  attachedAt,
  previousPid = null,
  previousTreeReleaseProven = false,
  previousResidencyEmptyProven = false,
  mergeExistingTree = false,
  directory = null,
} = {}) {
  const freshTree = modelPid == null
    ? { processIds: [], processes: [] }
    : snapshotProcessTree(modelPid);
  const processIds = mergeExistingTree
    ? [...new Set([
      ...(Array.isArray(claim.processIds) ? claim.processIds : []),
      ...freshTree.processIds,
    ].map(Number).filter((pid) => Number.isFinite(pid) && pid > 0))]
    : freshTree.processIds;
  const processByPid = new Map(
    (mergeExistingTree && Array.isArray(claim.processes) ? claim.processes : [])
      .map((entry) => [Number(entry?.pid), entry])
      .filter(([pid]) => Number.isFinite(pid) && pid > 0),
  );
  for (const entry of freshTree.processes) {
    const pid = Number(entry?.pid);
    if (Number.isFinite(pid) && pid > 0) processByPid.set(pid, entry);
  }
  writeClaimJsonAtomically(path.join(claim.lockPath, "model.json"), {
    modelPid,
    key,
    processIds,
    processes: processIds.map((pid) => processByPid.get(pid) ?? {
      pid,
      name: null,
      startedAt: null,
    }),
    attachedAt,
    previousPid,
    previousTreeReleaseProven,
    previousResidencyEmptyProven,
  });
  fs.rmSync(path.join(claim.lockPath, "failure.json"), { force: true });
  return readModelResidencyClaim(config, { directory });
}

// A claim left behind by a crashed run is stale bookkeeping, not residency: its
// owner is gone and its model tree is dead. Because the claim now carries that
// tree, it can prove this on its own evidence - it no longer depends on the
// benchmark run lease to unwedge it, which is what turned one crash into a
// permanently blocked machine (every load, not just benchmarks, refuses while a
// foreign claim exists).
//
// Fail-closed by construction: a live owner, a still-live tracked process, or a
// claim too old to record its tree all return false and keep the refusal.
function modelResidencyClaimOwnerIsLive(claim, {
  isAlive = isProcessAlive,
  listProcesses = processRows,
} = {}) {
  const ownerPid = Number(claim?.ownerPid);
  if (!Number.isFinite(ownerPid) || ownerPid <= 0 || !isAlive(ownerPid)) return false;
  const { livePids } = classifyTrackedPids([{
    pids: [ownerPid],
    processes: claim?.ownerProcessStartedAt
      ? [{
          pid: ownerPid,
          name: claim.ownerProcessName ?? null,
          startedAt: claim.ownerProcessStartedAt,
        }]
      : [],
    recordedAtIso: claim?.startedAt ?? null,
  }], { isAlive, listProcesses });
  return livePids.includes(ownerPid);
}

export function recoverStaleModelResidencyClaim(config, {
  directory = null,
  isAlive = isProcessAlive,
  listProcesses = processRows,
  // A long-lived coordinator (the benchmark dashboard) can remain alive after
  // a handed-off benchmark child and its model tree have both exited. In that
  // case owner liveness is not residency. Only callers that already hold an
  // independent exclusivity proof (the benchmark run lease), or an explicit
  // operator recovery action, may opt into reclaiming such a claim. The full
  // tracked model tree still has to be present and conclusively dead.
  allowLiveOwnerWithDeadTree = false,
  // Callers already inside withModelResidencyMutationLock (the reservation
  // acquire loop) must pass releaseModelResidencyClaimUnlocked to avoid a
  // re-entrant mutation-lock error; the default self-locks for outside callers.
  release = releaseModelResidencyClaim,
} = {}) {
  const claim = readModelResidencyClaim(config, { directory });
  if (!claim?.token) return false;

  // A claim from before the current OS boot is stale bookkeeping: its owner and
  // model tree died with the previous session and all VRAM was released on
  // reboot. Release it without needing a live process inventory - this is the
  // claim-lock counterpart of the runtime-state reboot recovery, and together
  // they stop a crash/BSOD from permanently blocking every future load.
  if (recordPredatesCurrentBoot(claim.attachedAt ?? claim.startedAt ?? null)) {
    release(config, {
      token: claim.token,
      ...(claim.modelPid != null
        ? { pid: Number(claim.modelPid), treeReleaseProven: true }
        : { residencyEmptyProven: true }),
      directory,
    });
    return true;
  }

  const ownerPid = Number(claim.ownerPid);
  const sameOwnerStillLive = modelResidencyClaimOwnerIsLive(claim, { isAlive, listProcesses });
  if (sameOwnerStillLive && allowLiveOwnerWithDeadTree !== true) {
    return false;
  }

  if (claim.modelPid == null) {
    // No model was ever attached, and the owner that would have attached one is
    // gone, so there is nothing loaded to protect.
    if (sameOwnerStillLive) return false;
    release(config, { token: claim.token, residencyEmptyProven: true, directory });
    return true;
  }

  const trackedPids = Array.isArray(claim.processIds) && claim.processIds.length > 0
    ? claim.processIds
    : null;
  if (!trackedPids) {
    // A claim written before trees were recorded knows only its root PID. A dead
    // root is not proof its native workers released the GPU, so stay refused.
    return false;
  }

  const { livePids } = classifyTrackedPids([{
    pids: trackedPids,
    processes: Array.isArray(claim.processes) ? claim.processes : [],
    recordedAtIso: claim.attachedAt ?? claim.startedAt ?? null,
  }], { isAlive, listProcesses });
  if (livePids.length > 0) return false;

  release(config, {
    token: claim.token,
    pid: Number(claim.modelPid),
    treeReleaseProven: true,
    directory,
  });
  return true;
}

export function markModelResidencyClaimFailed(config, options = {}) {
  const { directory = null } = options;
  return withModelResidencyMutationLock(config, { directory }, () => (
    markModelResidencyClaimFailedUnlocked(config, options)
  ));
}

function markModelResidencyClaimFailedUnlocked(config, {
  token,
  note = "Model residency release or transition could not be proven.",
  directory = null,
} = {}) {
  const claim = requireClaimToken(config, token, { directory });
  writeClaimJsonAtomically(path.join(claim.lockPath, "failure.json"), {
    status: "release_failed",
    note,
    failedAt: new Date().toISOString(),
  });
  return readModelResidencyClaim(config, { directory });
}

export function borrowModelResidencyClaim(config, options = {}) {
  const claim = validateModelResidencyClaim(config, options);
  const { token, directory = null } = options;
  return {
    token,
    claim,
    update(next) {
      return updateModelResidencyClaim(config, { token, directory, ...next });
    },
    markFailed(note) {
      return markModelResidencyClaimFailed(config, { token, directory, note });
    },
    release({
      pid,
      treeReleaseProven = false,
      residencyEmptyProven = false,
    } = {}) {
      return releaseModelResidencyClaim(config, {
        token,
        pid,
        treeReleaseProven,
        residencyEmptyProven,
        directory,
      });
    },
  };
}

function removeClaimDirectory(lockPath, { expectedToken = null, expectedOwnerless = false } = {}) {
  const quarantine = `${lockPath}.release-${process.pid}-${randomUUID()}`;
  renameClaimDirectorySync(lockPath, quarantine);
  const movedOwner = readClaimJson(path.join(quarantine, "owner.json"));
  const tokenMatches = expectedToken != null && movedOwner?.token === expectedToken;
  const ownerlessMatches = expectedOwnerless === true && !movedOwner?.token;
  if (!tokenMatches && !ownerlessMatches) {
    try {
      renameClaimDirectorySync(quarantine, lockPath);
    } catch {
      // Keep the mismatched generation quarantined as fail-closed evidence if
      // an external actor occupied the canonical path before restoration.
    }
    throw residencyMutationError(
      "Model residency claim generation changed during atomic removal.",
      { claim: movedOwner },
    );
  }
  fs.rmSync(quarantine, { recursive: true, force: true });
  return true;
}

export function releaseModelResidencyClaim(config, options = {}) {
  const { directory = null } = options;
  return withModelResidencyMutationLock(config, { directory }, () => (
    releaseModelResidencyClaimUnlocked(config, options)
  ));
}

function releaseModelResidencyClaimUnlocked(config, {
  token = null,
  pid,
  treeReleaseProven = false,
  residencyEmptyProven = false,
  directory = null,
} = {}) {
  const claim = readModelResidencyClaim(config, { directory });
  if (!claim) return false;
  if (token != null && claim.token !== token) {
    const error = new Error("Model residency claim token does not own the claim being released.");
    error.code = "RESIDENCY_RELEASE_FAILED";
    error.claim = claim;
    throw error;
  }
  if (claim.modelPid == null) {
    if (!token || claim.token !== token || residencyEmptyProven !== true) {
      const error = new Error(
        "A PID-less model residency claim requires its owner token and authoritative empty-residency proof before release.",
      );
      error.code = "RESIDENCY_RELEASE_FAILED";
      error.claim = claim;
      throw error;
    }
    return removeClaimDirectory(claim.lockPath, { expectedToken: claim.token });
  }
  if (treeReleaseProven !== true) {
    const error = new Error("A PID-backed model residency claim can be released only after process-tree release is proven.");
    error.code = "RESIDENCY_RELEASE_FAILED";
    error.claim = claim;
    throw error;
  }
  if (Number(claim.modelPid) !== Number(pid)) {
    const error = new Error(
      `Residency claim pid ${claim.modelPid ?? "unknown"} does not match verified released pid ${pid}.`,
    );
    error.code = "RESIDENCY_RELEASE_FAILED";
    error.claim = claim;
    throw error;
  }
  return removeClaimDirectory(claim.lockPath, { expectedToken: claim.token });
}

export function acquireModelLoadReservation(config, options = {}) {
  const { directory = null } = options;
  return withModelResidencyMutationLock(config, { directory }, () => (
    acquireModelLoadReservationUnlocked(config, options)
  ));
}

function acquireModelLoadReservationUnlocked(config, {
  pid = process.pid,
  processAlive = isProcessAlive,
  initializationGraceMs = 30000,
  directory = null,
  key,
  listProcesses = processRows,
  // Benchmark callers set this only after acquiring the per-user benchmark
  // run lease. It repairs the dashboard-handoff failure where the dashboard
  // owner remains alive but the claimed model tree is already fully dead.
  // Ordinary model loads leave it false, preserving the strict single-loader
  // boundary during live owner transitions.
  recoverLiveOwnerWithDeadTree = false,
} = {}) {
  const lockPath = modelResidencyClaimPath(config, directory);
  const ownerPath = path.join(lockPath, "owner.json");
  const modelPath = path.join(lockPath, "model.json");
  const token = randomUUID();
  const normalizedKey = normalizeModelResidencyKey(key);
  ensureDirectory(path.dirname(lockPath));

  for (let attempt = 0; attempt < 20; attempt += 1) {
    try {
      fs.mkdirSync(lockPath);
      // The normal owner is this process. Derive its creation identity from
      // uptime without a synchronous WMI/ps scan on every model reservation;
      // custom owner PIDs (primarily recovery/tests) still use the supplied
      // process inventory.
      const ownerIdentity = Number(pid) === Number(process.pid)
        ? {
            name: path.basename(process.execPath),
            startedAt: new Date(Date.now() - (process.uptime() * 1000)).toISOString(),
          }
        : (processIdentityIndex(listProcesses)?.get(Number(pid)) ?? null);
      const owner = {
        ownerPid: Number(pid),
        ...(ownerIdentity?.startedAt ? { ownerProcessStartedAt: ownerIdentity.startedAt } : {}),
        ...(ownerIdentity?.name ? { ownerProcessName: ownerIdentity.name } : {}),
        token,
        startedAt: new Date().toISOString(),
        status: "loading",
        key: normalizedKey,
      };
      try {
        fs.writeFileSync(ownerPath, JSON.stringify(owner, null, 2));
      } catch (error) {
        fs.rmSync(lockPath, { recursive: true, force: true });
        throw error;
      }

      let cancelled = false;
      const cancelBeforeSpawnUnlocked = () => {
        if (cancelled) return false;
        const currentOwner = readClaimJson(ownerPath);
        if (currentOwner?.token !== token || Number(currentOwner?.ownerPid) !== Number(pid)) {
          return false;
        }
        if (fs.existsSync(modelPath)) {
          const error = new Error("Cannot cancel a model residency claim after a spawned PID was attached.");
          error.code = "RESIDENCY_RELEASE_FAILED";
          throw error;
        }
        removeClaimDirectory(lockPath, { expectedToken: token });
        cancelled = true;
        return true;
      };
      const reservation = {
        lockPath,
        owner,
        token,
        key: normalizedKey,
        attachModel(modelPid) {
          return updateModelResidencyClaim(config, {
            token,
            key: normalizedKey,
            modelPid,
            directory,
          });
        },
        cancelBeforeSpawn() {
          return withModelResidencyMutationLock(config, { directory }, cancelBeforeSpawnUnlocked);
        },
      };
      try {
        assertNoUnprovenModelResidency(config, { claimToken: token });
      } catch (error) {
        cancelBeforeSpawnUnlocked();
        throw error;
      }
      return reservation;
    } catch (error) {
      if (error?.code !== "EEXIST") throw error;
    }

    const claim = readModelResidencyClaim(config, { directory });
    const ownerPid = Number(claim?.ownerPid);
    // "PID alive" is not "owner alive": the OS hands a finished run's PID to the
    // next process, so a crashed/timed-out run whose PID got recycled into an
    // unrelated live process (an editor worker, a browser helper) would read as
    // "a model run is active" and block every future run until reboot. Only
    // refuse when the live PID is provably the SAME process that made the claim
    // (start time within tolerance of the claim's startedAt). When it is a
    // recycled PID, fall through to the evidence-based recovery below, exactly
    // as recoverStaleModelResidencyClaim already does at its owner-held gate.
    const ownerHeld = claim?.token
      && modelResidencyClaimOwnerIsLive(claim, { isAlive: processAlive, listProcesses });
    if (ownerHeld) {
      if (recoverLiveOwnerWithDeadTree === true
        && recoverStaleModelResidencyClaim(config, {
          directory,
          isAlive: processAlive,
          listProcesses,
          allowLiveOwnerWithDeadTree: true,
          // The acquire loop already owns the mutation mutex.
          release: releaseModelResidencyClaimUnlocked,
        })) {
        continue;
      }
      const error = new Error(`Another CLI model load/run is active (pid ${ownerPid}).`);
      error.code = "RESIDENCY_RELEASE_FAILED";
      error.pid = ownerPid;
      throw error;
    }
    if (!claim?.token) {
      let ageMs;
      try {
        ageMs = Date.now() - fs.statSync(lockPath).mtimeMs;
      } catch (error) {
        if (error?.code === "ENOENT") continue;
        throw error;
      }
      if (ageMs < initializationGraceMs) {
        const error = new Error("Another CLI model load is initializing its lifecycle reservation.");
        error.code = "RESIDENCY_RELEASE_FAILED";
        throw error;
      }
    } else {
      // The owner is gone - either its PID is dead, or (per the gate above) its
      // PID was recycled into an unrelated process. A written loading/resident
      // claim can still be protecting a live native worker tree, so clearing it
      // must be evidence-based rather than automatic: recoverStaleModelResidencyClaim
      // releases only a claim that is provably finished - a dead owner that
      // never attached a model (exactly the crashed/aborted warm-load case), a
      // fully-dead recorded process tree, or a claim from before the current
      // boot - and returns false while anything is still alive or unprovable.
      // Retry the acquire once it clears; otherwise keep refusing.
      if (recoverStaleModelResidencyClaim(config, {
        directory,
        isAlive: processAlive,
        listProcesses,
        // Already holding the mutation lock in this acquire loop, so recovery
        // must release through the unlocked primitive to avoid re-entrancy.
        release: releaseModelResidencyClaimUnlocked,
      })) {
        continue;
      }
      const error = new Error(
        `An unproven model residency claim remains for owner pid ${ownerPid || "unknown"}.`,
      );
      error.code = "RESIDENCY_RELEASE_FAILED";
      error.claim = claim;
      throw error;
    }

    try {
      removeClaimDirectory(lockPath, { expectedOwnerless: true });
    } catch (error) {
      if (["ENOENT", "EEXIST", "EPERM", "EACCES"].includes(error?.code)) continue;
      throw error;
    }
  }

  const error = new Error("Unable to acquire the CLI model-load lifecycle reservation.");
  error.code = "RESIDENCY_RELEASE_FAILED";
  throw error;
}

// Recorded process-tree identity for the tracked model: the runtime-state
// record (captured at registration) plus the residency claim's model tree
// (captured at attach). Either can prove that everything the model spawned is
// gone - or that a tracked PID now belongs to an unrelated process.
function recordedTrackedTreeRecords(config, current) {
  const processId = Number(current?.pid);
  const records = [];
  if (Array.isArray(current?.processIds) && current.processIds.length > 0) {
    records.push({
      pids: current.processIds,
      processes: Array.isArray(current.processes) ? current.processes : [],
      recordedAtIso: current.startedAt ?? null,
    });
  }
  const claim = readModelResidencyClaim(config);
  if (claim?.modelPid != null && Number(claim.modelPid) === processId
    && Array.isArray(claim.processIds) && claim.processIds.length > 0) {
    records.push({
      pids: claim.processIds,
      processes: Array.isArray(claim.processes) ? claim.processes : [],
      recordedAtIso: claim.attachedAt ?? claim.startedAt ?? null,
    });
  }
  return records;
}

function classifyTrackedModelResidency(config, current, {
  isAlive = isProcessAlive,
  listProcesses = processRows,
} = {}) {
  const records = recordedTrackedTreeRecords(config, current);
  if (records.length === 0) {
    return { hasEvidence: false, trackedPids: [], livePids: [], recycledPids: [] };
  }
  return { hasEvidence: true, ...classifyTrackedPids(records, { isAlive, listProcesses }) };
}

function releaseResidencyClaimIfHeldByPid(config, pid, {
  ownerAlive = isProcessAlive,
} = {}) {
  const claim = readModelResidencyClaim(config);
  if (claim?.modelPid == null || Number(claim.modelPid) !== Number(pid)) return false;
  const ownerPid = Number(claim.ownerPid);
  // A verified external `model stop` may run while a live dashboard/benchmark
  // has transferred this exact claim. Keep the token in place so that owner
  // can move it to the next engine after observing the durable release proof.
  // Dead owners and this process's own claims are released immediately.
  if (Number.isFinite(ownerPid) && ownerPid > 0 && ownerPid !== process.pid && ownerAlive(ownerPid)) {
    return false;
  }
  return releaseModelResidencyClaim(config, { pid: Number(pid), treeReleaseProven: true });
}

const TRACKED_EXIT_PROVEN_NOTE =
  "Tracked model process exited on its own; residency release proven from its recorded process-tree identity.";

// A tracked model whose recorded processes all provably exited (or whose PIDs
// were recycled into unrelated processes) is finished - record it as stopped
// instead of failing every later stop/load against a PID that no longer means
// anything. Fail-closed: missing evidence or any genuinely live tracked
// process returns false and the caller keeps refusing.
function resolveExitedTrackedModelFromEvidence(config, current, {
  isAlive = isProcessAlive,
  listProcesses = processRows,
} = {}) {
  const processId = Number(current?.pid);
  if (!Number.isFinite(processId) || processId <= 0) return false;
  const residency = classifyTrackedModelResidency(config, current, { isAlive, listProcesses });
  if (!residency.hasEvidence || residency.livePids.length > 0) return false;
  markCurrentModelStopped(config, TRACKED_EXIT_PROVEN_NOTE, processId);
  releaseResidencyClaimIfHeldByPid(config, processId);
  return true;
}

async function stopExitedTrackedModel(config, current, residency, {
  forceAfterMs = 8000,
  isAlive = isProcessAlive,
  listProcesses = processRows,
} = {}) {
  const processId = Number(current.pid);
  const failStop = (message) => {
    const error = new Error(message);
    error.code = "RESIDENCY_RELEASE_FAILED";
    error.pid = processId;
    saveRuntimeState(config, {
      currentLoadedModel: {
        ...current,
        status: "release_failed",
        residencyReleaseUnproven: true,
        note: message,
      },
    });
    return error;
  };

  if (!residency.hasEvidence) {
    throw failStop(
      `Tracked model process ${processId} already exited and no recorded process-tree identity proves its descendants released residency.`,
    );
  }

  // The root is gone (or its PID recycled) but recorded descendants survive -
  // a native runner outliving its HTTP parent is exactly the case the
  // fail-closed guard exists for. Each survivor was reparented when the root
  // died, so terminate it as the root of its own tree, then re-verify against
  // the recorded inventory.
  let survivors = residency.livePids.map(Number).filter((pid) => pid !== processId);
  if (survivors.length > 0) {
    for (const survivorPid of survivors) {
      try {
        await requireTrackedProcessTreeTermination(survivorPid);
      } catch {
        // Re-verification below is authoritative.
      }
    }
    const deadline = Date.now() + Math.max(0, Number(forceAfterMs) || 0);
    while (survivors.some((pid) => isAlive(pid)) && Date.now() < deadline) {
      await new Promise((resolve) => setTimeout(resolve, 200));
    }
    const recheck = classifyTrackedModelResidency(config, current, { isAlive, listProcesses });
    survivors = recheck.livePids.map(Number).filter((pid) => pid !== processId);
    if (survivors.length > 0) {
      throw failStop(
        `Tracked model process ${processId} exited but its recorded descendant pid(s) ${survivors.join(", ")} could not be stopped.`,
      );
    }
  }

  const stoppedState = markCurrentModelStopped(config, TRACKED_EXIT_PROVEN_NOTE, processId);
  releaseResidencyClaimIfHeldByPid(config, processId);
  return { stopped: true, reason: "already_exited", state: stoppedState };
}

export async function stopCurrentModel(config, {
  forceAfterMs = 8000,
  isAlive = isProcessAlive,
  listProcesses = processRows,
} = {}) {
  const state = refreshRuntimeState(config);
  const current = state.currentLoadedModel;
  if (!current?.pid) {
    return {
      stopped: false,
      reason: "no_current_model",
      state,
    };
  }

  // Identity first: a root that already exited - or whose PID the OS handed
  // to an unrelated process - cannot be stopped by killing that PID. A kill
  // proves nothing for the dead case and hits an innocent process in the
  // recycled case; the recorded tree identity settles both, and a run whose
  // model simply finished or failed on its own resolves as stopped instead
  // of failing the whole stop.
  const residency = classifyTrackedModelResidency(config, current, { isAlive, listProcesses });
  const rootGenuinelyLive = residency.hasEvidence
    ? residency.livePids.some((pid) => Number(pid) === Number(current.pid))
    : isAlive(current.pid);
  if (!rootGenuinelyLive) {
    return await stopExitedTrackedModel(config, current, residency, {
      forceAfterMs,
      isAlive,
      listProcesses,
    });
  }

  saveRuntimeState(config, {
    currentLoadedModel: {
      ...current,
      status: "stopping",
      // Cleared only after the complete captured process tree has been
      // terminated and verified. This flag deliberately survives a dead root.
      residencyReleaseUnproven: true,
    },
  });

  try {
    await requireTrackedProcessTreeTermination(current.pid);
  } catch (error) {
    saveRuntimeState(config, {
      currentLoadedModel: {
        ...current,
        status: "release_failed",
        residencyReleaseUnproven: true,
        note: `Model residency release could not be proven: ${error.message}`,
      },
    });
    throw error;
  }

  if (!(await waitForProcessExit(current.pid, forceAfterMs))) {
    await requireTrackedProcessTreeTermination(current.pid);
    await waitForProcessExit(current.pid, 3000);
  }

  if (isProcessAlive(current.pid)) {
    const error = new Error(`Tracked model process ${current.pid} could not be stopped.`);
    error.code = "RESIDENCY_RELEASE_FAILED";
    throw error;
  }

  const stoppedState = markCurrentModelStopped(config, "Stopped by Lizard lifecycle manager.");
  releaseResidencyClaimIfHeldByPid(config, current.pid, { ownerAlive: isAlive });
  return {
    stopped: true,
    reason: "stopped",
    state: stoppedState,
  };
}

export function clearRuntimeState(config, { force = false } = {}) {
  if (!force) {
    const claim = readModelResidencyClaim(config);
    // `model clear-state` is an explicit recovery command. It still refuses a
    // live tracked model, but it may remove a live coordinator's stale claim
    // when the claim's complete PID/start-time tree proves that every model
    // process has already exited.
    if (claim?.token) {
      recoverStaleModelResidencyClaim(config, {
        allowLiveOwnerWithDeadTree: true,
      });
    }
    assertNoUnprovenModelResidency(config);
  }
  return saveRuntimeState(config, defaults());
}

export function buildRestartCommand(record) {
  if (!record?.executable || !Array.isArray(record.args)) {
    return null;
  }
  return `${record.executable} ${record.args.join(" ")}`;
}

export function restartLastModel(config) {
  const state = refreshRuntimeState(config);
  const record = state.lastStoppedModel;
  if (!record?.restartable || !record.executable || !Array.isArray(record.args)) {
    throw new Error("No restartable model session is available.");
  }
  const reservation = acquireModelLoadReservation(config, {
    key: {
      provider: record.profile?.runtimeProvider ?? record.source ?? "restart",
      modelPath: record.profile?.modelPath ?? record.args.find((value) => /\.(?:gguf|bin)$/i.test(String(value))) ?? record.executable,
      quant: record.profile?.quantizationLevel ?? record.profile?.gguf?.quant ?? record.profile?.quant ?? "UNKNOWN",
    },
  });
  let attachedPid = null;
  try {
    const child = spawn(record.executable, record.args, {
      detached: true,
      stdio: "ignore",
      windowsHide: true,
    });
    attachedPid = child.pid;
    reservation.attachModel(child.pid);
    child.unref();
    const currentLoadedModel = registerRunningModel(config, {
      modelId: record.modelId,
      pid: child.pid,
      profile: record.profile ?? null,
      command: buildRestartCommand(record),
      executable: record.executable,
      args: record.args,
      source: `${record.source ?? "unknown"}:restart`,
      note: "Restarted from last stopped model session.",
      claimToken: reservation.token,
    });
    return {
      restarted: true,
      currentLoadedModel,
    };
  } finally {
    if (!attachedPid) reservation.cancelBeforeSpawn();
  }
}

export function snapshotLifecycleState(config) {
  return refreshRuntimeState(config);
}
