import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { spawn } from "node:child_process";
import {
  acquireModelLoadReservation,
  assertNoUnprovenModelResidency,
  borrowModelResidencyClaim,
  captureTrackedProcessTreeSnapshot,
  clearRuntimeState,
  getCurrentLoadedModel,
  getLastStoppedRuntimeModel,
  loadRuntimeState,
  monitorTrackedProcessTree,
  modelResidencyKeyForOllama,
  normalizeModelResidencyKey,
  osBootTimeMs,
  readModelResidencyClaim,
  readVerifiedModelResidencyReleaseProof,
  recordPredatesCurrentBoot,
  recoverStaleModelResidencyClaim,
  releaseModelResidencyClaim,
  requireTrackedProcessTreeTermination,
  refreshRuntimeState,
  recordTrackedProcessExit,
  registerRunningModel,
  restartLastModel,
  runPosixProcessTreeKill,
  saveRuntimeState,
  stopCurrentModel,
  updateModelResidencyClaim,
  validateModelResidencyClaim,
} from "../src/lib/model-lifecycle.js";

function tempConfig() {
  const runtimeDirectory = fs.mkdtempSync(path.join(os.tmpdir(), "lizard-lifecycle-"));
  return {
    runtimeDirectory,
    modelLifecycleClaimDirectory: path.join(runtimeDirectory, "claim-locks"),
  };
}

function spawnSleeper() {
  return spawn(process.execPath, ["-e", "setInterval(() => {}, 1000)"], {
    stdio: "ignore",
    windowsHide: true,
  });
}

async function killChild(child) {
  if (!child || child.killed) {
    return;
  }
  try {
    child.kill("SIGKILL");
  } catch {
    return;
  }
  await new Promise((resolve) => {
    child.once("exit", resolve);
    setTimeout(resolve, 3000);
  });
}

const claimActorSource = String.raw`
  import fs from "node:fs";
  const sleepCell = new Int32Array(new SharedArrayBuffer(4));
  const sleep = (ms) => Atomics.wait(sleepCell, 0, 0, ms);
  const deadline = () => Date.now() + 10000;
  const waitFor = (filePath) => {
    const until = deadline();
    while (!fs.existsSync(filePath) && Date.now() < until) sleep(5);
    if (!fs.existsSync(filePath)) throw new Error("Timed out waiting for " + filePath);
  };
  try {
    const lifecycle = await import(process.env.LIZARD_TEST_LIFECYCLE_URL);
    const config = JSON.parse(process.env.LIZARD_TEST_CONFIG);
    const oldToken = process.env.LIZARD_TEST_OLD_TOKEN;
    const nextKey = JSON.parse(process.env.LIZARD_TEST_NEXT_KEY);
    fs.writeFileSync(process.env.LIZARD_TEST_READY, "ready");
    waitFor(process.env.LIZARD_TEST_START);
    const releasePrior = (prior) => {
      if (prior?.token !== oldToken) return;
      if (prior.modelPid == null) {
        lifecycle.releaseModelResidencyClaim(config, { token: oldToken, residencyEmptyProven: true });
      } else {
        lifecycle.releaseModelResidencyClaim(config, {
          token: oldToken,
          pid: Number(prior.modelPid),
          treeReleaseProven: true,
        });
      }
    };
    try {
      releasePrior(lifecycle.readModelResidencyClaim(config));
    } catch (error) {
      // The old owner may finish attaching its PID while this process waits
      // for the mutation mutex. Re-read and present the matching proof.
      releasePrior(lifecycle.readModelResidencyClaim(config));
    }
    const reservation = lifecycle.acquireModelLoadReservation(config, { key: nextKey });
    fs.writeFileSync(process.env.LIZARD_TEST_RESULT, JSON.stringify({ token: reservation.token }));
    waitFor(process.env.LIZARD_TEST_ACK);
    reservation.cancelBeforeSpawn();
  } catch (error) {
    fs.writeFileSync(process.env.LIZARD_TEST_RESULT, JSON.stringify({
      error: error?.stack ?? String(error),
    }));
    process.exitCode = 1;
  }
`;

function spawnClaimSwapActor({ config, oldToken, nextKey, readyPath, startPath, resultPath, ackPath }) {
  let stderr = "";
  const child = spawn(process.execPath, ["--input-type=module", "-e", claimActorSource], {
    stdio: ["ignore", "ignore", "pipe"],
    windowsHide: true,
    env: {
      ...process.env,
      LIZARD_TEST_LIFECYCLE_URL: new URL("../src/lib/model-lifecycle.js", import.meta.url).href,
      LIZARD_TEST_CONFIG: JSON.stringify(config),
      LIZARD_TEST_OLD_TOKEN: oldToken,
      LIZARD_TEST_NEXT_KEY: JSON.stringify(nextKey),
      LIZARD_TEST_READY: readyPath,
      LIZARD_TEST_START: startPath,
      LIZARD_TEST_RESULT: resultPath,
      LIZARD_TEST_ACK: ackPath,
    },
  });
  child.stderr?.on("data", (chunk) => { stderr += String(chunk); });
  const exited = new Promise((resolve, reject) => {
    child.once("error", reject);
    child.once("exit", (code) => resolve({ code, stderr }));
  });
  return { child, exited };
}

async function waitForFile(filePath, timeoutMs = 10000) {
  const deadline = Date.now() + timeoutMs;
  while (!fs.existsSync(filePath) && Date.now() < deadline) {
    await new Promise((resolve) => setTimeout(resolve, 10));
  }
  assert.equal(fs.existsSync(filePath), true, `Timed out waiting for ${filePath}`);
}

test("refreshRuntimeState demotes legacy stale records that did not track tree ownership", async () => {
  const config = tempConfig();
  const child = spawnSleeper();
  registerRunningModel(config, {
    modelId: "demo-model",
    pid: child.pid,
    executable: process.execPath,
    args: ["-e", "setInterval(() => {}, 1000)"],
    source: "test",
  });
  const legacyRecord = { ...getCurrentLoadedModel(config) };
  delete legacyRecord.residencyReleaseUnproven;
  saveRuntimeState(config, { currentLoadedModel: legacyRecord });
  await killChild(child);

  const state = refreshRuntimeState(config);
  assert.equal(state.currentLoadedModel, null);
  assert.equal(state.lastStoppedModel?.modelId, "demo-model");
  assert.equal(state.lastStoppedModel?.status, "stopped");
});

test("stopCurrentModel terminates a tracked process and records restart metadata", async () => {
  const config = tempConfig();
  const child = spawnSleeper();
  registerRunningModel(config, {
    modelId: "demo-model",
    pid: child.pid,
    executable: process.execPath,
    args: ["-e", "setInterval(() => {}, 1000)"],
    source: "test",
  });

  const result = await stopCurrentModel(config);
  assert.equal(result.stopped, true);
  assert.equal(getCurrentLoadedModel(config), null);
  assert.equal(getLastStoppedRuntimeModel(config)?.modelId, "demo-model");
  assert.equal(getLastStoppedRuntimeModel(config)?.restartable, true);
});

test("registerRunningModel records process-tree identity evidence", async () => {
  const config = tempConfig();
  const child = spawnSleeper();
  try {
    const record = registerRunningModel(config, {
      modelId: "tree-evidence-model",
      pid: child.pid,
      source: "test",
    });
    assert.ok(record.processIds?.map(Number).includes(child.pid));
    assert.ok(record.processes?.some((entry) => Number(entry.pid) === child.pid));
  } finally {
    await killChild(child);
  }
});

test("stopCurrentModel records a self-exited tracked model as stopped", async () => {
  const config = tempConfig();
  const child = spawnSleeper();
  registerRunningModel(config, {
    modelId: "self-exit-model",
    pid: child.pid,
    executable: process.execPath,
    args: ["-e", "setInterval(() => {}, 1000)"],
    source: "test",
  });
  await killChild(child);

  const result = await stopCurrentModel(config);
  assert.equal(result.stopped, true);
  assert.equal(result.reason, "already_exited");
  assert.equal(getCurrentLoadedModel(config), null);
  assert.equal(getLastStoppedRuntimeModel(config)?.modelId, "self-exit-model");
  assert.equal(getLastStoppedRuntimeModel(config)?.status, "stopped");
});

test("stopCurrentModel fails closed when a dead tracked model left no tree evidence", async () => {
  const config = tempConfig();
  saveRuntimeState(config, {
    currentLoadedModel: {
      modelId: "legacy-dead-model",
      pid: 2147483000,
      status: "release_failed",
      residencyReleaseUnproven: true,
    },
  });

  await assert.rejects(
    stopCurrentModel(config),
    (error) => error.code === "RESIDENCY_RELEASE_FAILED"
      && /no recorded process-tree identity/i.test(error.message),
  );
  assert.equal(loadRuntimeState(config).currentLoadedModel?.status, "release_failed");
});

test("stopCurrentModel treats a recycled tracked pid as already exited", async () => {
  const config = tempConfig();
  // A model registered THIS boot session whose PID was later recycled (no
  // reboot). Pin the recorded start after the current boot so the reboot-
  // recovery path does not fire - this is the same-session recycling case.
  const recordedStart = new Date(Math.max(osBootTimeMs() + 10_000, Date.now() - 3_600_000)).toISOString();
  saveRuntimeState(config, {
    currentLoadedModel: {
      modelId: "recycled-pid-model",
      pid: 2147480123,
      status: "running",
      residencyReleaseUnproven: true,
      startedAt: recordedStart,
      processIds: [2147480123],
      processes: [{ pid: 2147480123, name: "lizard-native-runner", startedAt: recordedStart }],
    },
  });

  const result = await stopCurrentModel(config, {
    isAlive: (pid) => Number(pid) === 2147480123,
    listProcesses: () => [
      { pid: 2147480123, ppid: 1, name: "unrelated-editor", startedAt: new Date().toISOString() },
    ],
  });
  assert.equal(result.stopped, true);
  assert.equal(result.reason, "already_exited");
  assert.equal(getCurrentLoadedModel(config), null);
});

test("recordPredatesCurrentBoot flags only timestamps from before the current boot", () => {
  const bootTimeMs = Date.parse("2026-07-15T10:00:00.000Z");
  assert.equal(recordPredatesCurrentBoot("2026-07-15T09:00:00.000Z", { bootTimeMs }), true);
  assert.equal(recordPredatesCurrentBoot("2026-07-15T10:30:00.000Z", { bootTimeMs }), false);
  // Within the 3s tolerance of the boot instant is NOT treated as pre-boot.
  assert.equal(recordPredatesCurrentBoot("2026-07-15T09:59:58.000Z", { bootTimeMs }), false);
  assert.equal(recordPredatesCurrentBoot(null, { bootTimeMs }), false);
  assert.equal(recordPredatesCurrentBoot("2026-07-15T09:00:00.000Z", { bootTimeMs: null }), false);
});

test("refreshRuntimeState resolves a model recorded before the current boot as stopped (reboot recovery)", () => {
  const config = tempConfig();
  // A wedged record left by a crash/BSOD: residency unproven, PID from a prior
  // session (now recycled into a live, unrelated process), started long before
  // this boot. Without reboot recovery this refuses every future load forever.
  saveRuntimeState(config, {
    currentLoadedModel: {
      modelId: "pre-boot-model",
      pid: process.pid, // a genuinely-alive PID (recycled) - would otherwise read "resident"
      status: "running",
      residencyReleaseUnproven: true,
      startedAt: new Date(osBootTimeMs() - 60_000).toISOString(),
      executable: "lizard",
      args: ["serve"],
    },
  });

  const state = refreshRuntimeState(config);
  assert.equal(state.currentLoadedModel, null);
  assert.equal(state.lastStoppedModel?.modelId, "pre-boot-model");
  assert.equal(state.lastStoppedModel?.residencyReleaseUnproven, false);
  assert.match(state.lastStoppedModel?.note ?? "", /predates the current OS boot/);
  // And a fresh load is allowed again.
  assert.doesNotThrow(() => assertNoUnprovenModelResidency(config));
});

test("recoverStaleModelResidencyClaim releases a claim from before the current boot", () => {
  const config = tempConfig();
  const reservation = acquireModelLoadReservation(config, {
    key: { provider: "lizard-native", modelPath: path.join(config.runtimeDirectory, "m.gguf"), quant: "Q4_K_M" },
  });
  reservation.attachModel(process.pid);
  // Backdate the claim's owner/attach records to before the current boot, as a
  // crash/BSOD would leave them relative to the next session's uptime.
  const preBoot = new Date(osBootTimeMs() - 60_000).toISOString();
  const lockPath = readModelResidencyClaim(config).lockPath;
  for (const file of ["owner.json", "model.json"]) {
    const filePath = path.join(lockPath, file);
    if (!fs.existsSync(filePath)) continue;
    const json = JSON.parse(fs.readFileSync(filePath, "utf8"));
    if ("startedAt" in json) json.startedAt = preBoot;
    if ("attachedAt" in json) json.attachedAt = preBoot;
    fs.writeFileSync(filePath, JSON.stringify(json, null, 2));
  }

  assert.equal(recoverStaleModelResidencyClaim(config), true);
  assert.equal(readModelResidencyClaim(config), null);
});

test("monitorTrackedProcessTree captures the tree then backs off to the idle interval", async () => {
  let calls = 0;
  const rows = [{ pid: 4242, ppid: 1 }, { pid: 4243, ppid: 4242 }];
  let alive = true;
  const monitor = monitorTrackedProcessTree(4242, {
    readProcessRows: async () => { calls += 1; return rows; },
    processAlive: () => alive,
    pollIntervalMs: 10,
    idlePollIntervalMs: 500,
    stableSamplesToIdle: 3,
  });
  // Fast phase stabilizes within a few samples, then switches to the 500ms idle
  // interval - proving we do NOT keep spawning a process inventory ~continuously
  // (the old 50ms-forever behavior that stormed WMI).
  await new Promise((r) => setTimeout(r, 120));
  const callsAfterStabilize = calls;
  await new Promise((r) => setTimeout(r, 150));
  const callsDuringIdle = calls - callsAfterStabilize;

  alive = false;
  const snapshot = await monitor.stopAndSnapshot();
  assert.equal(snapshot.capturedWhileRootAlive, true);
  assert.deepEqual([...snapshot.trackedPids].sort((a, b) => a - b), [4242, 4243]);
  assert.ok(callsDuringIdle <= 1, `expected idle backoff after stabilization, saw ${callsDuringIdle} extra samples`);
});

test("assertNoUnprovenModelResidency lifts the refusal after a tracked model provably exited", async () => {
  const config = tempConfig();
  const child = spawnSleeper();
  registerRunningModel(config, {
    modelId: "crashed-model",
    pid: child.pid,
    source: "test",
  });
  await killChild(child);

  const state = assertNoUnprovenModelResidency(config);
  assert.equal(state.currentLoadedModel, null);
  assert.equal(state.lastStoppedModel?.modelId, "crashed-model");
  const replacement = registerRunningModel(config, {
    modelId: "replacement-model",
    pid: process.pid,
    source: "test",
  });
  assert.equal(replacement.modelId, "replacement-model");
});

test("tracked cleanup fails closed when descendant-tree termination is unproven", async () => {
  await assert.rejects(
    requireTrackedProcessTreeTermination(424242, {
      terminateTree: async () => false,
    }),
    (error) => error.code === "RESIDENCY_RELEASE_FAILED",
  );
});

test("POSIX tree cleanup freezes and captures descendants that appear during inventory", async () => {
  const alive = new Set([51000, 51001, 51002]);
  const signals = [];
  let pass = 0;
  const result = await runPosixProcessTreeKill(51000, true, {
    processAlive: (pid) => alive.has(Number(pid)),
    readProcessRows: async () => {
      pass += 1;
      return pass === 1
        ? [{ pid: 51000, ppid: 1 }, { pid: 51001, ppid: 51000 }]
        : [
            { pid: 51000, ppid: 1 },
            { pid: 51001, ppid: 51000 },
            { pid: 51002, ppid: 51001 },
          ];
    },
    signalProcess: (pid, signal) => {
      signals.push([Number(pid), signal]);
      if (signal === "SIGKILL") alive.delete(Number(pid));
    },
    wait: async () => {},
  });

  assert.equal(result, true);
  assert.ok(pass >= 3, "tree inventory must stabilize after a complete no-new-child pass");
  for (const pid of [51000, 51001, 51002]) {
    assert.ok(signals.some(([signaledPid, signal]) => signaledPid === pid && signal === "SIGSTOP"));
    assert.ok(signals.some(([signaledPid, signal]) => signaledPid === pid && signal === "SIGKILL"));
  }
});

test("refresh retains failed residency ownership after the tracked root disappears", async () => {
  const config = tempConfig();
  const child = spawnSleeper();
  registerRunningModel(config, {
    modelId: "orphan-risk-model",
    pid: child.pid,
    executable: process.execPath,
    args: ["-e", "setInterval(() => {}, 1000)"],
    source: "test",
  });
  saveRuntimeState(config, {
    currentLoadedModel: {
      ...getCurrentLoadedModel(config),
      status: "release_failed",
      residencyReleaseUnproven: true,
    },
  });
  await killChild(child);

  const state = refreshRuntimeState(config);
  assert.equal(state.currentLoadedModel?.modelId, "orphan-risk-model");
  assert.equal(state.currentLoadedModel?.status, "release_failed");
  assert.equal(state.currentLoadedModel?.residencyReleaseUnproven, true);
  assert.equal(state.lastStoppedModel, null);
});

test("unproven persisted residency blocks a new model registration", () => {
  const config = tempConfig();
  saveRuntimeState(config, {
    currentLoadedModel: {
      modelId: "unknown-resident",
      pid: 2147483000,
      status: "release_failed",
      residencyReleaseUnproven: true,
    },
  });

  assert.throws(
    () => assertNoUnprovenModelResidency(config),
    (error) => error?.code === "RESIDENCY_RELEASE_FAILED",
  );
  assert.throws(
    () => registerRunningModel(config, {
      modelId: "replacement",
      pid: process.pid,
      source: "test",
    }),
    (error) => error?.code === "RESIDENCY_RELEASE_FAILED",
  );
  assert.equal(loadRuntimeState(config).currentLoadedModel?.modelId, "unknown-resident");
});

test("a live tracked model owns residency until its tree is verified stopped", async () => {
  const config = tempConfig();
  const child = spawnSleeper();
  registerRunningModel(config, {
    modelId: "current-model",
    pid: child.pid,
    source: "test",
  });
  try {
    assert.throws(
      () => registerRunningModel(config, {
        modelId: "second-model",
        pid: process.pid,
        source: "test",
      }),
      (error) => error?.code === "RESIDENCY_RELEASE_FAILED" && /second model/i.test(error.message),
    );
  } finally {
    await stopCurrentModel(config);
  }
});

test("a live legacy lifecycle record blocks a replacement model", async () => {
  const config = tempConfig();
  const child = spawnSleeper();
  saveRuntimeState(config, {
    currentLoadedModel: {
      modelId: "legacy-live-model",
      pid: child.pid,
      status: "running",
      source: "legacy-test",
    },
  });
  try {
    assert.throws(
      () => assertNoUnprovenModelResidency(config),
      (error) => error?.code === "RESIDENCY_RELEASE_FAILED" && /already owned/i.test(error.message),
    );
    assert.throws(
      () => registerRunningModel(config, {
        modelId: "replacement",
        pid: process.pid,
        source: "test",
      }),
      (error) => error?.code === "RESIDENCY_RELEASE_FAILED",
    );
  } finally {
    await killChild(child);
  }
});

test("a normal tracked CLI child exit records verified stopped ownership", async () => {
  const config = tempConfig();
  const reservation = acquireModelLoadReservation(config, {
    key: {
      provider: "llama.cpp",
      modelPath: process.execPath,
      quant: "Q4_K_M",
    },
  });
  const child = spawnSleeper();
  reservation.attachModel(child.pid);
  registerRunningModel(config, {
    modelId: "normal-exit-model",
    pid: child.pid,
    source: "cli-run",
    claimToken: reservation.token,
  });
  const preExitTreeSnapshot = await captureTrackedProcessTreeSnapshot(child.pid);
  assert.equal(preExitTreeSnapshot.capturedWhileRootAlive, true);
  await killChild(child);

  const result = await recordTrackedProcessExit(config, child.pid, { preExitTreeSnapshot });
  assert.equal(result.stopped, true);
  assert.equal(loadRuntimeState(config).currentLoadedModel, null);
  assert.equal(loadRuntimeState(config).lastStoppedModel?.modelId, "normal-exit-model");
  assert.equal(loadRuntimeState(config).lastStoppedModel?.residencyReleaseUnproven, false);
  assert.equal(readModelResidencyClaim(config), null);
});

test("tracked CLI exit stays fail-closed when a captured live descendant was reparented", async () => {
  const config = tempConfig();
  saveRuntimeState(config, {
    currentLoadedModel: {
      modelId: "descendant-resident",
      pid: 41000,
      status: "running",
      residencyReleaseUnproven: true,
    },
  });

  await assert.rejects(
    recordTrackedProcessExit(config, 41000, {
      processAlive: (pid) => Number(pid) === 41001,
      preExitTreeSnapshot: {
        rootPid: 41000,
        capturedWhileRootAlive: true,
        trackedPids: [41000, 41001],
        capturedAt: new Date().toISOString(),
      },
    }),
    (error) => error?.code === "RESIDENCY_RELEASE_FAILED" && error.liveDescendantPids?.[0] === 41001,
  );
  assert.equal(loadRuntimeState(config).currentLoadedModel?.status, "release_failed");
  assert.equal(loadRuntimeState(config).currentLoadedModel?.residencyReleaseUnproven, true);
});

test("post-exit process inventory alone is not descendant-release proof", async () => {
  const config = tempConfig();
  saveRuntimeState(config, {
    currentLoadedModel: {
      modelId: "uncaptured-descendant-risk",
      pid: 41100,
      status: "running",
      residencyReleaseUnproven: true,
    },
  });

  await assert.rejects(
    recordTrackedProcessExit(config, 41100, {
      processAlive: () => false,
    }),
    (error) => error?.code === "RESIDENCY_RELEASE_FAILED" && /while pid 41100 was alive/i.test(error.message),
  );
  assert.equal(loadRuntimeState(config).currentLoadedModel?.status, "release_failed");
});

test("model-load reservation serializes the CLI assert-to-spawn window", () => {
  const config = tempConfig();
  const key = {
    provider: "lizard-native",
    modelPath: process.execPath,
    quant: "Q4_K_M",
  };
  const first = acquireModelLoadReservation(config, { key });
  assert.throws(
    () => assertNoUnprovenModelResidency(config),
    (error) => error?.code === "RESIDENCY_RELEASE_FAILED" && /claim/i.test(error.message),
  );
  assert.throws(
    () => acquireModelLoadReservation(config, { key }),
    (error) => error?.code === "RESIDENCY_RELEASE_FAILED" && /active/i.test(error.message),
  );
  assert.equal(first.cancelBeforeSpawn(), true);
  const next = acquireModelLoadReservation(config, { key });
  assert.equal(next.cancelBeforeSpawn(), true);
});

test("model-load reservations persist the owner process creation identity", () => {
  const config = tempConfig();
  const ownerPid = 616161;
  const ownerProcessStartedAt = "2026-07-31T01:02:03.000Z";
  const reservation = acquireModelLoadReservation(config, {
    pid: ownerPid,
    key: {
      provider: "lizard-native",
      modelPath: process.execPath,
      quant: "Q4_K_M",
    },
    listProcesses: () => [{
      pid: ownerPid,
      ppid: 1,
      name: "node.exe",
      startedAt: ownerProcessStartedAt,
    }],
  });
  try {
    const claim = readModelResidencyClaim(config);
    assert.equal(claim.ownerPid, ownerPid);
    assert.equal(claim.ownerProcessStartedAt, ownerProcessStartedAt);
    assert.equal(claim.ownerProcessName, "node.exe");
  } finally {
    reservation.cancelBeforeSpawn();
    fs.rmSync(config.runtimeDirectory, { recursive: true, force: true });
  }
});

test("a stale concurrent releaser cannot delete a successor claim generation", async () => {
  const config = tempConfig();
  const oldKey = { provider: "ollama", modelPath: process.execPath, quant: "NONE" };
  const nextKey = { provider: "lizard-native", modelPath: process.execPath, quant: "Q5_K_M" };
  const reservation = acquireModelLoadReservation(config, { key: oldKey });
  const readyPath = path.join(config.runtimeDirectory, "release-actor-ready");
  const startPath = path.join(config.runtimeDirectory, "release-actor-start");
  const resultPath = path.join(config.runtimeDirectory, "release-actor-result.json");
  const ackPath = path.join(config.runtimeDirectory, "release-actor-ack");
  const actor = spawnClaimSwapActor({
    config,
    oldToken: reservation.token,
    nextKey,
    readyPath,
    startPath,
    resultPath,
    ackPath,
  });
  await waitForFile(readyPath);

  const originalRename = fs.renameSync;
  const waitCell = new Int32Array(new SharedArrayBuffer(4));
  let interposed = false;
  fs.renameSync = function interposeClaimRelease(source, destination, ...rest) {
    if (!interposed
      && path.resolve(String(source)) === path.resolve(reservation.lockPath)
      && String(destination).includes(".release-")) {
      interposed = true;
      fs.writeFileSync(startPath, "start");
      const deadline = Date.now() + 300;
      while (!fs.existsSync(resultPath) && Date.now() < deadline) Atomics.wait(waitCell, 0, 0, 5);
    }
    return originalRename.call(fs, source, destination, ...rest);
  };
  try {
    assert.equal(releaseModelResidencyClaim(config, {
      token: reservation.token,
      residencyEmptyProven: true,
    }), true);
  } finally {
    fs.renameSync = originalRename;
  }

  await waitForFile(resultPath);
  const actorResult = JSON.parse(fs.readFileSync(resultPath, "utf8"));
  try {
    assert.equal(actorResult.error, undefined, actorResult.error);
    const active = readModelResidencyClaim(config);
    assert.equal(active?.token, actorResult.token);
    assert.deepEqual(active?.key, normalizeModelResidencyKey(nextKey));
  } finally {
    fs.writeFileSync(ackPath, "ack");
  }
  const exited = await actor.exited;
  assert.equal(exited.code, 0, exited.stderr);
});

test("a stale concurrent update cannot write into a successor claim generation", async () => {
  const config = tempConfig();
  const oldKey = { provider: "lizard-native", modelPath: process.execPath, quant: "Q4_K_M" };
  const nextKey = { provider: "ollama", modelPath: process.execPath, quant: "NONE" };
  const reservation = acquireModelLoadReservation(config, { key: oldKey });
  const readyPath = path.join(config.runtimeDirectory, "update-actor-ready");
  const startPath = path.join(config.runtimeDirectory, "update-actor-start");
  const resultPath = path.join(config.runtimeDirectory, "update-actor-result.json");
  const ackPath = path.join(config.runtimeDirectory, "update-actor-ack");
  const actor = spawnClaimSwapActor({
    config,
    oldToken: reservation.token,
    nextKey,
    readyPath,
    startPath,
    resultPath,
    ackPath,
  });
  await waitForFile(readyPath);

  const originalWrite = fs.writeFileSync;
  const waitCell = new Int32Array(new SharedArrayBuffer(4));
  let interposed = false;
  fs.writeFileSync = function interposeClaimUpdate(filePath, ...args) {
    if (!interposed && String(filePath).startsWith(path.join(reservation.lockPath, "model.json.tmp-"))) {
      interposed = true;
      originalWrite.call(fs, startPath, "start");
      const deadline = Date.now() + 300;
      while (!fs.existsSync(resultPath) && Date.now() < deadline) Atomics.wait(waitCell, 0, 0, 5);
    }
    return originalWrite.call(fs, filePath, ...args);
  };
  try {
    updateModelResidencyClaim(config, {
      token: reservation.token,
      key: oldKey,
      modelPid: 52001,
    });
  } finally {
    fs.writeFileSync = originalWrite;
  }

  await waitForFile(resultPath);
  const actorResult = JSON.parse(fs.readFileSync(resultPath, "utf8"));
  try {
    assert.equal(actorResult.error, undefined, actorResult.error);
    const active = readModelResidencyClaim(config);
    assert.equal(active?.token, actorResult.token);
    assert.equal(active?.modelPid ?? null, null);
    assert.deepEqual(active?.key, normalizeModelResidencyKey(nextKey));
  } finally {
    fs.writeFileSync(ackPath, "ack");
  }
  const exited = await actor.exited;
  assert.equal(exited.code, 0, exited.stderr);
});

test("Ollama residency keys normalize tags and use collision-safe pseudo-paths", () => {
  const config = tempConfig();
  const bare = modelResidencyKeyForOllama(config, " Library/MyModel ");
  assert.deepEqual(bare, normalizeModelResidencyKey({
    provider: "ollama",
    modelPath: path.resolve(
      config.runtimeDirectory,
      "ollama-residency",
      encodeURIComponent("library/mymodel:latest"),
    ),
    quant: "NONE",
  }));

  const digest = modelResidencyKeyForOllama(config, "Library/MyModel@SHA256:ABCDEF");
  assert.deepEqual(digest, normalizeModelResidencyKey({
    provider: "ollama",
    modelPath: path.resolve(
      config.runtimeDirectory,
      "ollama-residency",
      encodeURIComponent("library/mymodel@sha256:abcdef"),
    ),
    quant: "NONE",
  }));
  assert.notEqual(
    modelResidencyKeyForOllama(config, "a/b:q4").modelPath,
    modelResidencyKeyForOllama(config, "a%2fb:q4").modelPath,
  );
  assert.throws(
    () => modelResidencyKeyForOllama(config, " "),
    (error) => error?.code === "INVALID_MODEL_RESIDENCY_KEY",
  );
});

test("token helpers validate and move only verified exact residency handoffs", () => {
  const config = tempConfig();
  const firstKey = {
    provider: "lizard-native",
    modelPath: process.execPath,
    quant: "Q4_K_M",
  };
  const secondKey = {
    provider: "llama.cpp",
    modelPath: process.execPath,
    quant: "Q5_K_M",
  };
  const reservation = acquireModelLoadReservation(config, { key: firstKey });
  reservation.attachModel(42001);

  assert.equal(
    validateModelResidencyClaim(config, {
      token: reservation.token,
      key: firstKey,
      pid: 42001,
    }).modelPid,
    42001,
  );
  assert.throws(
    () => validateModelResidencyClaim(config, {
      token: reservation.token,
      key: secondKey,
      pid: 42001,
    }),
    (error) => error?.code === "RESIDENCY_RELEASE_FAILED" && /exact provider\/model\/Q/i.test(error.message),
  );
  assert.throws(
    () => updateModelResidencyClaim(config, {
      token: reservation.token,
      key: secondKey,
      modelPid: 42002,
    }),
    (error) => error?.code === "RESIDENCY_RELEASE_FAILED" && /verified release/i.test(error.message),
  );

  const borrowed = borrowModelResidencyClaim(config, {
    token: reservation.token,
    key: firstKey,
    pid: 42001,
  });
  const moved = borrowed.update({
    key: secondKey,
    modelPid: 42002,
    previousPid: 42001,
    previousTreeReleaseProven: true,
  });
  assert.equal(moved.modelPid, 42002);
  assert.equal(moved.key.quant, "Q5_K_M");
  assert.throws(
    () => releaseModelResidencyClaim(config, {
      token: reservation.token,
      pid: 42002,
    }),
    (error) => error?.code === "RESIDENCY_RELEASE_FAILED" && /process-tree release/i.test(error.message),
  );
  assert.equal(borrowed.release({ pid: 42002, treeReleaseProven: true }), true);
  assert.equal(readModelResidencyClaim(config), null);
});

test("PID-less residency changes and release require authoritative empty proof", () => {
  const config = tempConfig();
  const firstKey = {
    provider: "ollama",
    modelPath: process.execPath,
    quant: "Q4_K_M",
  };
  const secondKey = {
    provider: "ollama",
    modelPath: process.execPath,
    quant: "Q5_K_M",
  };
  const pidBackedKey = {
    provider: "lizard-native",
    modelPath: process.execPath,
    quant: "Q5_K_M",
  };
  const reservation = acquireModelLoadReservation(config, { key: firstKey });

  assert.throws(
    () => updateModelResidencyClaim(config, {
      token: reservation.token,
      key: pidBackedKey,
      modelPid: 43001,
    }),
    (error) => error?.code === "RESIDENCY_RELEASE_FAILED" && /authoritative proof|no resident/i.test(error.message),
  );
  assert.throws(
    () => updateModelResidencyClaim(config, {
      token: reservation.token,
      key: secondKey,
    }),
    (error) => error?.code === "RESIDENCY_RELEASE_FAILED" && /authoritative proof|no resident/i.test(error.message),
  );
  const moved = updateModelResidencyClaim(config, {
    token: reservation.token,
    key: secondKey,
    previousResidencyEmptyProven: true,
  });
  assert.equal(moved.modelPid, null);
  assert.equal(moved.key.quant, "Q5_K_M");
  assert.throws(
    () => releaseModelResidencyClaim(config, {
      token: reservation.token,
      treeReleaseProven: true,
    }),
    (error) => error?.code === "RESIDENCY_RELEASE_FAILED" && /empty-residency proof/i.test(error.message),
  );
  assert.equal(releaseModelResidencyClaim(config, {
    token: reservation.token,
    residencyEmptyProven: true,
  }), true);
  assert.equal(readModelResidencyClaim(config), null);
});

test("restartLastModel relaunches the saved command and clearRuntimeState resets records", async () => {
  const config = tempConfig();
  const child = spawnSleeper();
  registerRunningModel(config, {
    modelId: "demo-model",
    pid: child.pid,
    executable: process.execPath,
    args: ["-e", "setInterval(() => {}, 1000)"],
    source: "test",
  });
  await stopCurrentModel(config);

  const restarted = restartLastModel(config);
  assert.equal(restarted.restarted, true);
  assert.equal(getCurrentLoadedModel(config)?.modelId, "demo-model");
  assert.ok(getCurrentLoadedModel(config)?.pid);

  await stopCurrentModel(config);

  clearRuntimeState(config);
  assert.equal(getCurrentLoadedModel(config), null);
  assert.equal(getLastStoppedRuntimeModel(config), null);
});

const STALE_CLAIM_KEY = { provider: "lizard-native", modelPath: process.execPath, quant: "Q4" };

// Build the exact on-disk state a crashed run leaves behind: a claim whose owner
// process is gone, still holding a model PID.
function orphanClaim(config, { modelPid, ownerPid }) {
  const reservation = acquireModelLoadReservation(config, { key: STALE_CLAIM_KEY });
  updateModelResidencyClaim(config, { token: reservation.token, key: STALE_CLAIM_KEY, modelPid });
  const claim = readModelResidencyClaim(config);
  const ownerPath = path.join(claim.lockPath, "owner.json");
  const owner = JSON.parse(fs.readFileSync(ownerPath, "utf8"));
  delete owner.ownerProcessStartedAt;
  delete owner.ownerProcessName;
  fs.writeFileSync(ownerPath, JSON.stringify({ ...owner, ownerPid }, null, 2));
  return readModelResidencyClaim(config);
}

test("a residency claim records the model's process tree, not just its root PID", async () => {
  const config = tempConfig();
  const model = spawnSleeper();
  try {
    const reservation = acquireModelLoadReservation(config, { key: STALE_CLAIM_KEY });
    updateModelResidencyClaim(config, { token: reservation.token, key: STALE_CLAIM_KEY, modelPid: model.pid });
    const claim = readModelResidencyClaim(config);
    assert.ok(claim.processIds.includes(model.pid), "the tracked tree must contain the model root");
    const rootIdentity = claim.processes.find((entry) => Number(entry.pid) === Number(model.pid));
    assert.ok(rootIdentity?.startedAt, "each tracked PID needs a start time to stay identifiable");
  } finally {
    await killChild(model);
    fs.rmSync(config.runtimeDirectory, { recursive: true, force: true });
  }
});

test("a claim orphaned by a crashed run self-heals once its whole tree is dead", async () => {
  const config = tempConfig();
  const model = spawnSleeper();
  const owner = spawnSleeper();
  const modelPid = model.pid;
  try {
    orphanClaim(config, { modelPid, ownerPid: owner.pid });
    await killChild(owner);

    // The model is still running: this is real residency and must stay refused,
    // even though the owner that spawned it is gone.
    assert.equal(recoverStaleModelResidencyClaim(config), false);
    assert.throws(
      () => assertNoUnprovenModelResidency(config),
      (error) => error?.code === "RESIDENCY_RELEASE_FAILED" && String(error.message).includes(String(modelPid)),
    );

    // Once the tracked tree is gone, nothing is loaded, so the claim is stale
    // bookkeeping and must not block the next load.
    await killChild(model);
    assert.equal(recoverStaleModelResidencyClaim(config), true);
    assert.equal(readModelResidencyClaim(config), null);
    assert.doesNotThrow(() => assertNoUnprovenModelResidency(config));
  } finally {
    await killChild(model);
    await killChild(owner);
    fs.rmSync(config.runtimeDirectory, { recursive: true, force: true });
  }
});

test("verified external stop keeps a live coordinator claim and writes exact tree-release proof", async () => {
  const config = tempConfig();
  const model = spawnSleeper();
  const coordinator = spawnSleeper();
  const key = { provider: "lizard-native", modelPath: process.execPath, quant: "Q4" };
  const reservation = acquireModelLoadReservation(config, { key });
  try {
    reservation.attachModel(model.pid);
    const registered = registerRunningModel(config, {
      modelId: "externally-stopped-model",
      pid: model.pid,
      source: "test",
      claimToken: reservation.token,
    });
    const claim = readModelResidencyClaim(config);
    fs.writeFileSync(
      path.join(claim.lockPath, "owner.json"),
      JSON.stringify({
        ...JSON.parse(fs.readFileSync(path.join(claim.lockPath, "owner.json"), "utf8")),
        ownerPid: coordinator.pid,
      }, null, 2),
    );
    const proofNotBefore = Date.now();

    const result = await stopCurrentModel(config);
    assert.equal(result.stopped, true);
    assert.equal(readModelResidencyClaim(config)?.token, reservation.token);
    const proof = readVerifiedModelResidencyReleaseProof(config, {
      pid: model.pid,
      startedAt: registered.startedAt,
      notBeforeMs: proofNotBefore,
    });
    assert.equal(proof?.pid, model.pid);
    assert.ok(proof.trackedPids.includes(model.pid));
  } finally {
    await killChild(model);
    await killChild(coordinator);
    const claim = readModelResidencyClaim(config);
    if (claim?.token === reservation.token) {
      releaseModelResidencyClaim(config, {
        token: reservation.token,
        pid: model.pid,
        treeReleaseProven: true,
      });
    }
    fs.rmSync(config.runtimeDirectory, { recursive: true, force: true });
  }
});

test("benchmark-exclusive recovery clears a live dashboard owner's claim only after its tracked model tree is dead", async () => {
  const config = tempConfig();
  const model = spawnSleeper();
  const owner = spawnSleeper();
  try {
    orphanClaim(config, { modelPid: model.pid, ownerPid: owner.pid });
    await killChild(model);

    assert.equal(
      recoverStaleModelResidencyClaim(config),
      false,
      "ordinary recovery must continue to respect a live owner",
    );
    assert.equal(
      recoverStaleModelResidencyClaim(config, { allowLiveOwnerWithDeadTree: true }),
      true,
      "an independently serialized benchmark may repair the dead transferred tree",
    );
    assert.equal(readModelResidencyClaim(config), null);
  } finally {
    await killChild(model);
    await killChild(owner);
    fs.rmSync(config.runtimeDirectory, { recursive: true, force: true });
  }
});

test("live-owner recovery never clears a claim while any tracked model process is alive", async () => {
  const config = tempConfig();
  const model = spawnSleeper();
  const owner = spawnSleeper();
  try {
    orphanClaim(config, { modelPid: model.pid, ownerPid: owner.pid });
    assert.equal(
      recoverStaleModelResidencyClaim(config, { allowLiveOwnerWithDeadTree: true }),
      false,
    );
    assert.ok(readModelResidencyClaim(config)?.token);
  } finally {
    await killChild(model);
    await killChild(owner);
    fs.rmSync(config.runtimeDirectory, { recursive: true, force: true });
  }
});

test("acquireModelLoadReservation repairs a live coordinator's dead claim only with benchmark exclusivity", async () => {
  const config = tempConfig();
  const model = spawnSleeper();
  const owner = spawnSleeper();
  try {
    orphanClaim(config, { modelPid: model.pid, ownerPid: owner.pid });
    await killChild(model);
    assert.throws(
      () => acquireModelLoadReservation(config, { key: STALE_CLAIM_KEY }),
      (error) => error?.code === "RESIDENCY_RELEASE_FAILED" && /active/i.test(error.message),
    );

    const reservation = acquireModelLoadReservation(config, {
      key: STALE_CLAIM_KEY,
      recoverLiveOwnerWithDeadTree: true,
    });
    assert.ok(reservation.token);
    reservation.cancelBeforeSpawn();
    assert.equal(readModelResidencyClaim(config), null);
  } finally {
    await killChild(model);
    await killChild(owner);
    fs.rmSync(config.runtimeDirectory, { recursive: true, force: true });
  }
});

test("model clear-state repairs a live coordinator's dead tracked claim", async () => {
  const config = tempConfig();
  const model = spawnSleeper();
  const owner = spawnSleeper();
  try {
    orphanClaim(config, { modelPid: model.pid, ownerPid: owner.pid });
    await killChild(model);
    assert.doesNotThrow(() => clearRuntimeState(config));
    assert.equal(readModelResidencyClaim(config), null);
  } finally {
    await killChild(model);
    await killChild(owner);
    fs.rmSync(config.runtimeDirectory, { recursive: true, force: true });
  }
});

test("assertNoUnprovenModelResidency clears a stale claim instead of refusing every later load", async () => {
  const config = tempConfig();
  const model = spawnSleeper();
  const owner = spawnSleeper();
  try {
    orphanClaim(config, { modelPid: model.pid, ownerPid: owner.pid });
    await killChild(model);
    await killChild(owner);
    // The wedge this replaces: a foreign claim used to throw here forever, and
    // only hand-deleting the lock directory could clear it.
    const state = assertNoUnprovenModelResidency(config);
    assert.ok(state);
    assert.equal(readModelResidencyClaim(config), null);
  } finally {
    await killChild(model);
    await killChild(owner);
    fs.rmSync(config.runtimeDirectory, { recursive: true, force: true });
  }
});

// The exact on-disk state a crashed/refused warm load leaves: a "loading" claim
// whose owner exited before ever attaching a model PID (e.g. gemma4's native
// value-sharing refusal exiting at ~0s).
function orphanLoadingClaim(config, { ownerPid }) {
  const reservation = acquireModelLoadReservation(config, { key: STALE_CLAIM_KEY });
  const claim = readModelResidencyClaim(config);
  const ownerPath = path.join(claim.lockPath, "owner.json");
  const owner = JSON.parse(fs.readFileSync(ownerPath, "utf8"));
  delete owner.ownerProcessStartedAt;
  delete owner.ownerProcessName;
  fs.writeFileSync(ownerPath, JSON.stringify({ ...owner, ownerPid }, null, 2));
  return readModelResidencyClaim(config);
}

test("acquireModelLoadReservation recovers a stale loading claim whose owner exited before attaching a model", async () => {
  const config = tempConfig();
  const owner = spawnSleeper();
  try {
    orphanLoadingClaim(config, { ownerPid: owner.pid });
    await killChild(owner);
    // The wedge this replaces: a loading claim with a dead owner used to throw
    // "An unproven model residency claim remains" on every later acquire until
    // the lock was hand-deleted. Nothing was ever loaded, so the reservation
    // must recover it and proceed.
    const reservation = acquireModelLoadReservation(config, { key: STALE_CLAIM_KEY });
    assert.ok(reservation?.token, "a fresh reservation should be acquired after recovering the stale loading claim");
    assert.equal(readModelResidencyClaim(config)?.token, reservation.token, "the new reservation owns the claim");
    reservation.cancelBeforeSpawn();
    assert.equal(readModelResidencyClaim(config), null);
  } finally {
    await killChild(owner);
    fs.rmSync(config.runtimeDirectory, { recursive: true, force: true });
  }
});

test("acquireModelLoadReservation recovers a loading claim whose owner PID was recycled into a live process", () => {
  const config = tempConfig();
  const RECYCLED_PID = 424242;
  // A crashed/timed-out run left a loading claim, then the OS handed its PID to
  // an unrelated, still-running process. The wedge this replaces: the acquire
  // path checked only kill(pid,0), so this read as "Another CLI model load/run
  // is active (pid ...)" and blocked every future run until reboot.
  const claim = orphanLoadingClaim(config, { ownerPid: RECYCLED_PID });
  // The live holder of the PID started well AFTER the claim - provable recycling.
  const liveStartedAt = new Date(Date.parse(claim.startedAt) + 60000).toISOString();
  const injected = {
    key: STALE_CLAIM_KEY,
    processAlive: (pid) => Number(pid) === RECYCLED_PID,
    listProcesses: () => [{ pid: RECYCLED_PID, ppid: 0, name: "node.exe", startedAt: liveStartedAt }],
  };
  try {
    const reservation = acquireModelLoadReservation(config, injected);
    assert.ok(reservation?.token, "the recycled-owner claim must be recovered, not treated as an active run");
    assert.equal(readModelResidencyClaim(config)?.token, reservation.token, "the new reservation owns the claim");
    reservation.cancelBeforeSpawn();
    assert.equal(readModelResidencyClaim(config), null);
  } finally {
    fs.rmSync(config.runtimeDirectory, { recursive: true, force: true });
  }
});

test("acquireModelLoadReservation still refuses when the owner PID is the same live process (not recycled)", () => {
  const config = tempConfig();
  const OWNER_PID = 525252;
  const claim = orphanLoadingClaim(config, { ownerPid: OWNER_PID });
  // The live holder started BEFORE the claim: this is the genuine owner, still
  // loading. Recycle detection must NOT fire, so the run stays refused.
  const liveStartedAt = new Date(Date.parse(claim.startedAt) - 60000).toISOString();
  const injected = {
    key: STALE_CLAIM_KEY,
    processAlive: (pid) => Number(pid) === OWNER_PID,
    listProcesses: () => [{ pid: OWNER_PID, ppid: 0, name: "node.exe", startedAt: liveStartedAt }],
  };
  try {
    assert.throws(
      () => acquireModelLoadReservation(config, injected),
      (error) => error?.code === "RESIDENCY_RELEASE_FAILED" && /active/i.test(error.message),
    );
  } finally {
    fs.rmSync(config.runtimeDirectory, { recursive: true, force: true });
  }
});

test("a claim with no recorded tree stays fail-closed because its workers are unprovable", async () => {
  const config = tempConfig();
  const model = spawnSleeper();
  const owner = spawnSleeper();
  const modelPid = model.pid;
  try {
    const claim = orphanClaim(config, { modelPid, ownerPid: owner.pid });
    await killChild(model);
    await killChild(owner);

    // Rewrite model.json in the pre-1.0.21 shape: a bare root PID, no tree. A
    // dead root is not proof that a native worker released the GPU, so recovery
    // must decline rather than guess.
    const modelPath = path.join(claim.lockPath, "model.json");
    const stored = JSON.parse(fs.readFileSync(modelPath, "utf8"));
    delete stored.processIds;
    delete stored.processes;
    fs.writeFileSync(modelPath, JSON.stringify(stored, null, 2));

    assert.equal(recoverStaleModelResidencyClaim(config), false);
    assert.throws(
      () => assertNoUnprovenModelResidency(config),
      (error) => error?.code === "RESIDENCY_RELEASE_FAILED",
    );
  } finally {
    await killChild(model);
    await killChild(owner);
    fs.rmSync(config.runtimeDirectory, { recursive: true, force: true });
  }
});
