import { execFileSync } from "node:child_process";

// A PID is not an identity. Both Windows and Linux hand a dead process's PID
// straight back to the next process that starts, so `kill(pid, 0)` only answers
// "does something hold this number" - never "is the process I tracked still
// running". Every residency safeguard in Lizard used to probe bare PIDs, which
// meant a crashed model server whose PID got recycled into an unrelated process
// (a browser helper, an editor worker) read as "a model is still resident" and
// refused every later load, forever.
//
// PID + start time IS an identity: a process holding a tracked PID that started
// AFTER we recorded that PID cannot be the process we tracked.
//
// Every function here fails closed. When a start time cannot be obtained -
// no inventory, an unreadable row, a platform without `ps` - recycling cannot
// be proven, so the PID counts as live and the caller keeps refusing. Absence
// of evidence never becomes evidence of absence.

const PID_IDENTITY_TOLERANCE_MS = 5000;

export function processIsAlive(pid) {
  const numeric = Number(pid);
  if (!Number.isFinite(numeric) || numeric <= 0) return false;
  try {
    process.kill(numeric, 0);
    return true;
  } catch {
    return false;
  }
}

export function normalizeProcessStartTime(value) {
  if (typeof value !== "string" || !value.trim()) return null;
  const parsed = Date.parse(value.trim());
  return Number.isFinite(parsed) ? new Date(parsed).toISOString() : null;
}

function windowsProcessRows() {
  try {
    const output = execFileSync("powershell", [
      "-NoProfile",
      "-NonInteractive",
      "-Command",
      "Get-CimInstance Win32_Process | Select-Object ProcessId,ParentProcessId,Name,"
      + "@{n='StartedAt';e={ if ($_.CreationDate) { $_.CreationDate.ToUniversalTime().ToString('o') } else { $null } }}"
      + " | ConvertTo-Json -Compress",
    ], {
      encoding: "utf8",
      windowsHide: true,
      // This runs SYNCHRONOUSLY (snapshotProcessTree at model registration), so
      // the timeout is also the worst-case event-loop freeze. Keep it short: a
      // slow/flaky WMI must degrade to "inventory unavailable", never hang the
      // CLI or the server for 15 seconds.
      timeout: 5000,
      maxBuffer: 32 * 1024 * 1024,
    }).trim();
    if (!output) return [];
    const parsed = JSON.parse(output);
    return (Array.isArray(parsed) ? parsed : [parsed]).map((row) => ({
      pid: Number(row.ProcessId),
      ppid: Number(row.ParentProcessId),
      name: typeof row.Name === "string" ? row.Name : null,
      startedAt: normalizeProcessStartTime(row.StartedAt),
    })).filter((row) => Number.isFinite(row.pid) && Number.isFinite(row.ppid));
  } catch {
    return null;
  }
}

function unixProcessRows() {
  // `lstart` is the portable (Linux + macOS) way to get an absolute start time,
  // which is what makes a PID identifiable rather than merely present. Fall back
  // to the bare listing when a `ps` build rejects it.
  for (const args of [["-eo", "pid=,ppid=,lstart="], ["-eo", "pid=,ppid="]]) {
    try {
      const output = execFileSync("ps", args, { encoding: "utf8", timeout: 5000 });
      const rows = output.split(/\r?\n/).map((line) => {
        const match = line.trim().match(/^(\d+)\s+(\d+)\s*(.*)$/);
        if (!match) return null;
        const pid = Number(match[1]);
        const ppid = Number(match[2]);
        if (!Number.isFinite(pid) || !Number.isFinite(ppid)) return null;
        return { pid, ppid, name: null, startedAt: normalizeProcessStartTime(match[3]) };
      }).filter(Boolean);
      if (rows.length > 0) return rows;
    } catch {
      // Try the next `ps` form before giving up on the inventory entirely.
    }
  }
  return null;
}

// The live process table, or null when it cannot be read.
export function processRows() {
  return process.platform === "win32" ? windowsProcessRows() : unixProcessRows();
}

export function descendantProcessIds(rows, rootPid) {
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

export function processIdentityIndex(listProcesses = processRows) {
  const rows = listProcesses();
  if (!Array.isArray(rows)) return null;
  const index = new Map();
  for (const row of rows) {
    const pid = Number(row?.pid);
    if (Number.isFinite(pid) && pid > 0) index.set(pid, row);
  }
  return index;
}

// Capture a process tree WITH each member's identity, so a later run can tell
// these exact processes apart from whatever inherits their PIDs after a crash.
export function snapshotProcessTree(rootPid, listProcesses = processRows) {
  const pid = Number(rootPid);
  const rows = listProcesses();
  if (!Array.isArray(rows)) {
    return { inventoryAvailable: false, processIds: [pid], processes: [{ pid, name: null, startedAt: null }] };
  }
  const processIds = [pid, ...descendantProcessIds(rows, pid)];
  const index = new Map(rows
    .filter((row) => Number.isFinite(Number(row?.pid)))
    .map((row) => [Number(row.pid), row]));
  const processes = processIds.map((id) => ({
    pid: Number(id),
    name: index.get(Number(id))?.name ?? null,
    startedAt: index.get(Number(id))?.startedAt ?? null,
  }));
  return { inventoryAvailable: true, processIds, processes };
}

// `identity` is the live process holding the PID right now (or null if none).
export function trackedProcessLooksRecycled(identity, { recordedStartedAt = null, recordedAtIso = null } = {}) {
  const liveStartedAt = Date.parse(identity?.startedAt ?? "");
  if (!Number.isFinite(liveStartedAt)) return false;

  const recordedStart = Date.parse(recordedStartedAt ?? "");
  if (Number.isFinite(recordedStart)) {
    // We captured this exact process's start time: identity must match.
    return Math.abs(liveStartedAt - recordedStart) > PID_IDENTITY_TOLERANCE_MS;
  }

  // Records written before start times were captured still pin the moment the
  // PID was last observed alive, which is enough: a process that started after
  // that observation is provably a different one.
  const recordedAt = Date.parse(recordedAtIso ?? "");
  if (!Number.isFinite(recordedAt)) return false;
  return liveStartedAt > recordedAt + PID_IDENTITY_TOLERANCE_MS;
}

// Split tracked PIDs into the ones still held by the process we tracked (a real
// residency hazard) and the ones since recycled into unrelated processes (none).
//
// `records`: [{ pids: number[], processes?: [{pid, startedAt}], recordedAtIso }]
export function classifyTrackedPids(records, {
  isAlive = processIsAlive,
  listProcesses = processRows,
} = {}) {
  const trackedPids = [];
  const claims = new Map();
  for (const record of Array.isArray(records) ? records : []) {
    const recordedIdentities = new Map((Array.isArray(record?.processes) ? record.processes : [])
      .map((entry) => [Number(entry?.pid), entry])
      .filter(([pid]) => Number.isFinite(pid) && pid > 0));
    for (const rawPid of Array.isArray(record?.pids) ? record.pids : []) {
      const pid = Number(rawPid);
      if (!Number.isFinite(pid) || pid <= 0) continue;
      if (!claims.has(pid)) trackedPids.push(pid);
      const recordedStartedAt = recordedIdentities.get(pid)?.startedAt ?? null;
      const previous = claims.get(pid);
      // Prefer the strongest evidence recorded for this PID across records.
      if (!previous || (!previous.recordedStartedAt && recordedStartedAt)) {
        claims.set(pid, { recordedStartedAt, recordedAtIso: record?.recordedAtIso ?? null });
      }
    }
  }

  const stillHeld = trackedPids.filter((pid) => isAlive(pid));
  const index = stillHeld.length > 0 ? processIdentityIndex(listProcesses) : null;
  const livePids = [];
  const recycledPids = [];
  for (const pid of stillHeld) {
    if (index && trackedProcessLooksRecycled(index.get(pid) ?? null, claims.get(pid) ?? {})) {
      recycledPids.push(pid);
    } else {
      livePids.push(pid);
    }
  }
  return { trackedPids, livePids, recycledPids };
}

// True when `pid` is held by a process that provably started after `recordedAtIso`.
export function pidLooksRecycled(pid, recordedAtIso, { listProcesses = processRows } = {}) {
  const numeric = Number(pid);
  if (!Number.isFinite(numeric) || numeric <= 0) return false;
  const index = processIdentityIndex(listProcesses);
  if (!index) return false;
  return trackedProcessLooksRecycled(index.get(numeric) ?? null, { recordedAtIso });
}

export function recycleAwareIsAlive(recycledPids = [], isAlive = processIsAlive) {
  const recycled = new Set(recycledPids.map(Number).filter((pid) => Number.isFinite(pid) && pid > 0));
  return (pid) => !recycled.has(Number(pid)) && isAlive(pid);
}
