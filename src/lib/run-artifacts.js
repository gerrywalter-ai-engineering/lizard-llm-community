// An artifact for every finished UI run.
//
// Benchmark scripts write their own rich artifacts, but a run started from the
// Command Center left nothing behind except a row in the runs table and a log
// blob in SQLite. So the only record of what was actually executed - which
// model, which quantization, which transport, what it exited with - lived in a
// database the dashboard reads and nothing else does. Comparing two runs from
// last week, or handing one to somebody else, meant screenshots.
//
// This writes one JSON file per finished run, next to the benchmark artifacts,
// covering EVERY run rather than only the benchmark-shaped ones: a failed run is
// often the one worth keeping.
//
// Deliberately additive. It never modifies the run row, never throws into the
// run's own completion path, and a failure here costs the artifact and nothing
// else - a run that succeeded must not be reported as failed because its
// paperwork could not be filed.
import fs from "node:fs";
import path from "node:path";
import { ensureDirectory } from "./paths.js";

// Logs can be large; an artifact is a record, not a transcript store.
const MAX_LOG_CHARS = 64 * 1024;

export function runArtifactDirectory(config) {
  const configured = config?.benchmark?.ci?.artifactDirectory;
  if (configured) {
    return path.resolve(configured, "command-runs");
  }
  const runtimeDirectory = config?.runtimeDirectory ?? ".";
  return path.resolve(runtimeDirectory, "benchmark-artifacts", "command-runs");
}

export function runArtifactPath(config, runId, startedAt) {
  // Timestamp first so a directory listing sorts chronologically, run id second
  // so two runs in the same second cannot collide.
  const stamp = String(startedAt ?? new Date().toISOString()).replace(/[:.]/g, "-");
  return path.join(runArtifactDirectory(config), `run-${stamp}-${runId}.json`);
}

function tail(text, limit = MAX_LOG_CHARS) {
  const value = String(text ?? "");
  if (value.length <= limit) return value;
  // The end of a log is where the failure is; keep that rather than the start.
  return `... [${value.length - limit} earlier characters omitted]\n${value.slice(-limit)}`;
}

/**
 * Builds the artifact body. Separated from writing so it can be asserted
 * against without touching a filesystem.
 */
export function buildRunArtifact({ run, command = null, logs = [], finishedAt, durationMs, status, exitCode }) {
  const streams = { stdout: [], stderr: [] };
  for (const entry of logs ?? []) {
    const stream = entry?.stream === "stderr" ? "stderr" : "stdout";
    streams[stream].push(String(entry?.message ?? entry?.text ?? ""));
  }
  return {
    kind: "lizard.command-run",
    version: 1,
    run: {
      id: run?.id ?? null,
      commandId: run?.command_id ?? run?.commandId ?? null,
      status,
      exitCode: exitCode ?? null,
      startedAt: run?.started_at ?? run?.startedAt ?? null,
      finishedAt: finishedAt ?? null,
      durationMs: durationMs ?? null,
      attemptCount: run?.attempt_count ?? run?.attemptCount ?? 1,
    },
    // What the number describes. Without these a stored figure is unattributable
    // a week later, which is the whole reason for keeping it.
    subject: {
      modelId: run?.model_id ?? run?.modelId ?? null,
      quantizationLevel: run?.quantization_level ?? run?.quantizationLevel ?? null,
      runtimeProvider: run?.runtime_provider ?? run?.runtimeProvider ?? null,
      ggufFilePath: run?.gguf_file_path ?? run?.ggufFilePath ?? null,
      fallbackModelId: run?.fallback_model_id ?? run?.fallbackModelId ?? null,
      // The transport is what separates two otherwise identical measurements:
      // one paid the HTTP hop and the other did not.
      transport: command?.transport ?? null,
      runtimeTarget: command?.runtimeTarget ?? run?.runtime_target ?? null,
      riskLevel: command?.riskLevel ?? run?.risk_level ?? null,
    },
    command: {
      name: command?.name ?? null,
      category: command?.category ?? null,
      text: run?.command_text ?? run?.commandText ?? null,
    },
    logs: {
      stdout: tail(streams.stdout.join("\n")),
      stderr: tail(streams.stderr.join("\n")),
    },
  };
}

/**
 * Writes the artifact and returns its path, or null when it could not be
 * written. Never throws: see the note at the top of this file.
 */
export function writeRunArtifact({ config, run, command = null, logs = [], finishedAt, durationMs, status, exitCode }) {
  try {
    const filePath = runArtifactPath(config, run?.id ?? "unknown", run?.started_at ?? run?.startedAt);
    ensureDirectory(path.dirname(filePath));
    const artifact = buildRunArtifact({ run, command, logs, finishedAt, durationMs, status, exitCode });
    fs.writeFileSync(filePath, `${JSON.stringify(artifact, null, 2)}\n`, "utf8");
    return filePath;
  } catch {
    return null;
  }
}
