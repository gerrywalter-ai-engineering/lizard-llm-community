// "Is anything actually using this model right now?"
//
// Unloading a model kills its server process, and with it every response still
// streaming to a chat window or an API client. Before doing that - whether from
// the dashboard, `model stop`, or the pre-benchmark clear - the caller needs to
// be able to tell the user what they are about to interrupt, rather than
// discovering it from a truncated answer.
//
// The native server publishes `activeRequests` on /health (the admission
// counter it already keeps). Everything here is best-effort and fails toward
// "unknown", never toward a confident "nothing is running": a false "idle"
// would be worse than no check at all, because it would let a caller cut
// someone off while telling them it was safe.
import { execFileSync } from "node:child_process";
import { getCurrentLoadedModel } from "./model-lifecycle.js";

const PROBE_TIMEOUT_MS = 1500;

/**
 * Ask one server port how many completions it is serving.
 * Returns { reachable, activeRequests, modelId } - activeRequests is null when
 * the port did not answer or did not publish the field.
 */
export async function probeServerActivity(port, { host = "127.0.0.1", timeoutMs = PROBE_TIMEOUT_MS, fetchImpl = fetch } = {}) {
  if (!Number.isFinite(Number(port)) || Number(port) <= 0) {
    return { reachable: false, activeRequests: null, modelId: null };
  }
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetchImpl(`http://${host}:${port}/health`, { signal: controller.signal });
    if (!res.ok) return { reachable: false, activeRequests: null, modelId: null };
    const body = await res.json();
    return {
      reachable: true,
      // Absent field (an older server) reads as unknown, not as zero.
      activeRequests: Number.isFinite(Number(body?.activeRequests)) ? Number(body.activeRequests) : null,
      modelId: body?.modelId ?? body?.model ?? null,
    };
  } catch {
    return { reachable: false, activeRequests: null, modelId: null };
  } finally {
    clearTimeout(timer);
  }
}

/**
 * The residency record tracks processes, not sockets - it has no `port` field,
 * and the `--launch` plan file it references is deleted once the server has
 * read it. So the only durable source is the operating system: which ports the
 * tracked processes are actually listening on.
 *
 * Without this the probe had nothing to ask and every unload reported "idle" -
 * precisely the confident-but-wrong answer this guard exists to prevent.
 */
export function listeningPortsForPids(pids, { exec = null } = {}) {
  const wanted = new Set((pids ?? []).map((pid) => String(pid)).filter(Boolean));
  if (wanted.size === 0) return [];
  try {
    const run = exec ?? ((cmd, args) => execFileSync(cmd, args, { encoding: "utf8", windowsHide: true, timeout: 5000 }));
    const output = process.platform === "win32"
      ? run("netstat", ["-ano", "-p", "TCP"])
      : run("lsof", ["-nP", "-iTCP", "-sTCP:LISTEN"]);
    const ports = new Set();
    for (const line of String(output).split(/\r?\n/)) {
      if (process.platform === "win32") {
        // Proto  Local Address  Foreign Address  State  PID
        //
        // The state word is LOCALISED - a German Windows prints HERGESTELLT
        // and ABHOEREN, not ESTABLISHED and LISTENING - so filtering on it
        // silently matched nothing and every unload reported "idle". Match on
        // the owning PID instead, which is locale-independent, and take the
        // local port. A non-HTTP port that slips through is harmless: probing
        // it fails and reads as unknown, never as idle.
        const parts = line.trim().split(/\s+/);
        if (parts.length < 5) continue;
        if (!wanted.has(parts[parts.length - 1])) continue;
        const port = Number(parts[1].split(":").pop());
        if (Number.isFinite(port) && port > 0) ports.add(port);
      } else {
        const parts = line.trim().split(/\s+/);
        if (parts.length < 9 || !wanted.has(parts[1])) continue;
        const port = Number(String(parts[8]).split(":").pop());
        if (Number.isFinite(port) && port > 0) ports.add(port);
      }
    }
    return [...ports];
  } catch {
    return [];
  }
}

/**
 * What an unload is about to interrupt, for the model this machine currently
 * has resident. `busy` is only true when a server actually reported work in
 * flight - unknown never counts as busy, so a broken probe cannot block an
 * unload the user asked for.
 */
export async function describeUnloadImpact(config, { probe = probeServerActivity } = {}) {
  const current = getCurrentLoadedModel(config);
  if (!current) {
    return { busy: false, activeRequests: 0, modelId: null, ports: [], unknown: false };
  }
  const ports = [...new Set([current.port, ...(current.ports ?? []), ...listeningPortsForPids([current.pid, ...(current.processIds ?? [])])].filter(Boolean))];
  let total = 0;
  let sawCount = false;
  for (const port of ports) {
    const result = await probe(port);
    if (result.activeRequests != null) {
      sawCount = true;
      total += result.activeRequests;
    }
  }
  return {
    busy: total > 0,
    activeRequests: total,
    modelId: current.modelId ?? null,
    ports,
    // The model is resident but nothing could tell us what it is doing.
    unknown: ports.length > 0 && !sawCount,
  };
}

/**
 * The sentence a user should see before confirming. Deliberately concrete
 * about the consequence - requests are closed, not "may be affected".
 */
export function unloadImpactMessage(impact) {
  if (!impact) return null;
  if (impact.busy) {
    const n = impact.activeRequests;
    return `${n} request${n === 1 ? "" : "s"} ${n === 1 ? "is" : "are"} still being answered by ${impact.modelId ?? "the loaded model"}. Unloading closes ${n === 1 ? "it" : "them"} immediately and the ${n === 1 ? "reply" : "replies"} will be cut off. Proceed?`;
  }
  if (impact.unknown) {
    return `${impact.modelId ?? "A model"} is loaded but is not reporting activity, so any request still in flight would be closed by unloading. Proceed?`;
  }
  return null;
}
