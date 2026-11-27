import crypto from "node:crypto";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

const INCIDENT_SCHEMA = "lizard.major-failure-incident.v1";
const INDEX_SCHEMA = "lizard.major-failure-index.v1";
const DEFAULT_REPEAT_WINDOW_MS = 10 * 60 * 1000;
const DEFAULT_REPEAT_THRESHOLD = 3;

const MINOR_CODES = new Set([
  "AUTH_REQUIRED",
  "AUTH_INTERNAL",
  "AUTH_DISABLED",
  "BILLING_UNAVAILABLE",
  "CSRF_INVALID",
  "INVALID_JSON",
  "KV_BUDGET_EXCEEDED",
  "NATIVE_EXECUTOR_UNIMPLEMENTED",
  "NATIVE_MODEL_UNSUPPORTED",
  "NATIVE_TOKENIZER_UNAVAILABLE",
  "PAYMENT_REQUIRED",
  "SCHEDULER_CANCELLED",
]);

const CATEGORY_RULES = [
  {
    category: "corrupted_gguf",
    codes: ["GGUF_CORRUPT", "INVALID_GGUF", "NOT_GGUF", "GGUF_CHECKSUM_MISMATCH"],
    pattern: /\b(corrupt(?:ed|ion)?|checksum mismatch|bad gguf magic|invalid gguf|truncated gguf)\b/i,
  },
  {
    category: "unsupported_hardware",
    codes: ["NATIVE_CPU_INCOMPATIBLE", "NATIVE_HARDWARE_UNSUPPORTED", "UNSUPPORTED_HARDWARE"],
    pattern: /\b(illegal instruction|unsupported (?:cpu|hardware)|incompatible cpu|no compatible (?:gpu|adapter)|required instruction set)\b/i,
  },
  {
    category: "model_initialization",
    codes: ["MODEL_INITIALIZATION_FAILED", "NATIVE_MODEL_LOAD_FAILED", "NATIVE_WORKER_LOAD_FAILED"],
    pattern: /\b(model (?:initiali[sz]ation|load) failed|failed to (?:initiali[sz]e|load) (?:the )?model|load_model failed)\b/i,
  },
  {
    category: "runtime_crash",
    codes: ["NATIVE_WORKER_CRASH", "NATIVE_WORKER_EXIT", "RUNTIME_CRASH", "WORKER_CRASH"],
    pattern: /\b(STATUS_ACCESS_VIOLATION|worker|runner|runtime|process).*\b(crash(?:ed)?|aborted|terminated unexpectedly|exited unexpectedly|STATUS_ACCESS_VIOLATION)\b/i,
  },
  {
    category: "runtime_unrecoverable",
    codes: ["RESIDENCY_RELEASE_FAILED", "RUNTIME_UNRECOVERABLE", "UNRECOVERABLE_RUNTIME_ERROR"],
    pattern: /\b(unrecoverable|residency release failed|cannot safely continue|failed closed|STATUS_DLL_NOT_FOUND)\b/i,
  },
];

function nowIso(now = Date.now) {
  return new Date(now()).toISOString();
}

function normalizeError(error) {
  if (error instanceof Error) {
    return {
      name: error.name,
      message: error.message,
      code: error.code ?? null,
      stack: error.stack ?? null,
      cause: error.cause ? normalizeError(error.cause) : null,
      details: error.details ?? null,
    };
  }
  if (error && typeof error === "object") {
    return {
      name: String(error.name ?? "Error"),
      message: String(error.message ?? JSON.stringify(error)),
      code: error.code ?? null,
      stack: error.stack ?? null,
      cause: null,
      details: error.details ?? null,
    };
  }
  return {
    name: "Error",
    message: String(error ?? "Unknown failure"),
    code: null,
    stack: null,
    cause: null,
    details: null,
  };
}

function safeString(value, maxLength = 20_000) {
  const text = String(value ?? "");
  return text.length > maxLength ? `${text.slice(0, maxLength)}…[truncated]` : text;
}

function redactSecrets(value, key = "", seen = new WeakSet(), depth = 0) {
  if (depth > 12) return "[depth-limit]";
  if (value == null || typeof value === "number" || typeof value === "boolean") return value;
  if (typeof value === "string") {
    if (/token|secret|password|authorization|cookie|credential|private.?key|api.?key/i.test(key)) {
      return "[redacted]";
    }
    return safeString(value)
      .replace(/\b(Bearer|Basic)\s+[A-Za-z0-9._~+/=-]+/gi, "$1 [redacted]")
      .replace(/([?&](?:token|key|secret|password)=)[^&\s]+/gi, "$1[redacted]");
  }
  if (typeof value !== "object") return safeString(value);
  if (seen.has(value)) return "[circular]";
  seen.add(value);
  if (Array.isArray(value)) {
    return value.slice(0, 2_000).map((entry) => redactSecrets(entry, key, seen, depth + 1));
  }
  const output = {};
  for (const [childKey, childValue] of Object.entries(value).slice(0, 2_000)) {
    output[childKey] = redactSecrets(childValue, childKey, seen, depth + 1);
  }
  return output;
}

export function anonymizeCommunityText(value) {
  return safeString(value, 50_000)
    .replace(/[A-Z]:\\Users\\[^\\\s]+/gi, "[redacted-user-home]")
    .replace(/\/(?:home|Users)\/[^/\s]+/g, "[redacted-user-home]")
    .replace(/\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b/gi, "[redacted-email]")
    .replace(/\b(?:\d{1,3}\.){3}\d{1,3}\b/g, "[redacted-ip]")
    .replace(/\b(?:Bearer|Basic)\s+[A-Za-z0-9._~+/=-]+/gi, "[redacted-credential]")
    .replace(/\b(?:token|secret|password|api[_-]?key)\s*[:=]\s*\S+/gi, "[redacted-secret]");
}

function normalizeCode(error) {
  return String(error?.code ?? "").trim().toUpperCase();
}

export function classifyMajorFailure(error, context = {}) {
  const normalized = normalizeError(error);
  const code = normalizeCode(normalized);
  const message = `${normalized.message}\n${normalized.cause?.message ?? ""}`;
  if (MINOR_CODES.has(code) || context.minor === true) {
    return {
      major: false,
      severity: "minor",
      category: "minor",
      reason: "The failure is an expected request, compatibility, cancellation, or access-control outcome.",
    };
  }
  for (const rule of CATEGORY_RULES) {
    if (rule.codes.includes(code) || rule.pattern.test(message)) {
      return {
        major: true,
        severity: "critical",
        category: rule.category,
        reason: `Matched the ${rule.category} critical-failure policy.`,
      };
    }
  }
  if (context.unrecoverable === true) {
    return {
      major: true,
      severity: "critical",
      category: "runtime_unrecoverable",
      reason: "The caller marked the runtime failure as unrecoverable.",
    };
  }
  const crashCandidate = context.crash === true
    || /\b(worker|runner|runtime|process)\b/i.test(message)
      && /\b(exit(?:ed)?|crash(?:ed)?|abort(?:ed)?|terminated|signal)\b/i.test(message);
  return {
    major: false,
    severity: crashCandidate ? "candidate" : "minor",
    category: crashCandidate ? "runtime_crash" : "unclassified",
    crashCandidate,
    reason: crashCandidate
      ? "The failure becomes critical only after the configured repeat threshold."
      : "No critical-failure rule matched.",
  };
}

function fingerprintFor(error, context, classification) {
  const normalized = normalizeError(error);
  const stableMessage = normalized.message
    .toLowerCase()
    .replace(/[a-f0-9]{8,}/g, "#")
    .replace(/\b\d+\b/g, "#")
    .slice(0, 500);
  return crypto.createHash("sha256").update(JSON.stringify({
    code: normalizeCode(normalized),
    message: stableMessage,
    category: classification.category,
    component: context.component ?? null,
    runtimeProvider: context.runtimeProvider ?? null,
  })).digest("hex");
}

function ticketId(now = Date.now, randomBytes = crypto.randomBytes) {
  const day = new Date(now()).toISOString().slice(0, 10).replaceAll("-", "");
  return `LIZ-${day}-${randomBytes(5).toString("hex").toUpperCase()}`;
}

function ensurePrivateDirectory(directory) {
  fs.mkdirSync(directory, { recursive: true, mode: 0o700 });
  try {
    fs.chmodSync(directory, 0o700);
  } catch {
    // Windows ACLs do not map to POSIX modes; the per-user runtime directory
    // still provides the local security boundary.
  }
}

function writeJsonAtomic(filePath, value) {
  ensurePrivateDirectory(path.dirname(filePath));
  const temporary = `${filePath}.${process.pid}.${crypto.randomBytes(3).toString("hex")}.tmp`;
  fs.writeFileSync(temporary, `${JSON.stringify(value, null, 2)}\n`, { encoding: "utf8", mode: 0o600 });
  fs.renameSync(temporary, filePath);
  try {
    fs.chmodSync(filePath, 0o600);
  } catch {
    // See ensurePrivateDirectory.
  }
}

function readJson(filePath, fallback) {
  try {
    return JSON.parse(fs.readFileSync(filePath, "utf8"));
  } catch {
    return fallback;
  }
}

function incidentRoot(config) {
  return path.resolve(
    config?.failureIntelligence?.directory
      ?? path.join(config?.runtimeDirectory ?? path.join(os.homedir(), ".lizard-llm"), "incidents"),
  );
}

function incidentPath(config, id) {
  return path.join(incidentRoot(config), id, "incident.json");
}

function failureIndexPath(config) {
  return path.join(incidentRoot(config), "failure-index.json");
}

function updateRepeatIndex({ config, fingerprint, now = Date.now }) {
  const currentTime = now();
  const windowMs = Number(config?.failureIntelligence?.repeatWindowMs) || DEFAULT_REPEAT_WINDOW_MS;
  const filePath = failureIndexPath(config);
  const current = readJson(filePath, { schema: INDEX_SCHEMA, failures: {} });
  const failures = current?.schema === INDEX_SCHEMA && current.failures && typeof current.failures === "object"
    ? current.failures
    : {};
  const previous = failures[fingerprint];
  const previousTimestamps = Array.isArray(previous)
    ? previous
    : Array.isArray(previous?.timestamps)
      ? previous.timestamps
      : [];
  const recent = previousTimestamps.filter((timestamp) => currentTime - timestamp <= windowMs);
  recent.push(currentTime);
  failures[fingerprint] = {
    timestamps: recent.slice(-100),
    ticketId: previous?.ticketId ?? null,
  };
  for (const [key, record] of Object.entries(failures)) {
    const timestamps = Array.isArray(record) ? record : record?.timestamps;
    const kept = Array.isArray(timestamps)
      ? timestamps.filter((timestamp) => currentTime - timestamp <= windowMs)
      : [];
    if (kept.length) failures[key] = {
      timestamps: kept,
      ticketId: Array.isArray(record) ? null : record?.ticketId ?? null,
    };
    else delete failures[key];
  }
  writeJsonAtomic(filePath, { schema: INDEX_SCHEMA, updatedAt: new Date(currentTime).toISOString(), failures });
  return {
    count: recent.length,
    ticketId: failures[fingerprint]?.ticketId ?? null,
  };
}

function assignTicketToFingerprint({ config, fingerprint, id }) {
  const filePath = failureIndexPath(config);
  const current = readJson(filePath, { schema: INDEX_SCHEMA, failures: {} });
  const failures = current?.schema === INDEX_SCHEMA && current.failures && typeof current.failures === "object"
    ? current.failures
    : {};
  const previous = failures[fingerprint];
  failures[fingerprint] = {
    timestamps: Array.isArray(previous)
      ? previous
      : Array.isArray(previous?.timestamps)
        ? previous.timestamps
        : [],
    ticketId: id,
  };
  writeJsonAtomic(filePath, { schema: INDEX_SCHEMA, updatedAt: nowIso(), failures });
}

function environmentSnapshot(environment = process.env) {
  return Object.fromEntries(
    Object.entries(environment)
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([key, value]) => [
        key,
        /token|secret|password|authorization|cookie|credential|private.?key|api.?key/i.test(key)
          ? "[redacted]"
          : safeString(value, 4_000),
      ]),
  );
}

function copyDiagnosticLogs({ config, directory }) {
  const sourceDirectory = config?.logging?.directory;
  const targetDirectory = path.join(directory, "logs");
  ensurePrivateDirectory(targetDirectory);
  if (!sourceDirectory || !fs.existsSync(sourceDirectory)) return [];
  const copied = [];
  for (const entry of fs.readdirSync(sourceDirectory, { withFileTypes: true })) {
    if (!entry.isFile()) continue;
    const source = path.join(sourceDirectory, entry.name);
    const target = path.join(targetDirectory, path.basename(entry.name));
    try {
      fs.copyFileSync(source, target);
      const stat = fs.statSync(target);
      copied.push({ name: entry.name, path: target, bytes: stat.size });
    } catch (error) {
      copied.push({ name: entry.name, path: null, bytes: null, copyError: error.message });
    }
  }
  return copied;
}

function hardwareSummary(hardware) {
  return {
    platform: hardware?.platform ?? process.platform,
    arch: hardware?.arch ?? process.arch,
    cpu: hardware?.cpu
      ? {
        model: hardware.cpu.model ?? null,
        physicalCores: hardware.cpu.physicalCores ?? null,
        logicalThreads: hardware.cpu.logicalThreads ?? null,
      }
      : { model: os.cpus()[0]?.model ?? null, logicalThreads: os.cpus().length },
    memory: hardware?.memory ?? {
      totalGB: Number((os.totalmem() / (1024 ** 3)).toFixed(2)),
      freeGB: Number((os.freemem() / (1024 ** 3)).toFixed(2)),
    },
    gpus: Array.isArray(hardware?.gpus)
      ? hardware.gpus.map((gpu) => ({
        name: gpu.name ?? null,
        vendor: gpu.vendor ?? null,
        vramGB: gpu.vramGB ?? gpu.vramGb ?? null,
        backends: gpu.backends ?? null,
      }))
      : [],
    backends: hardware?.backends ?? null,
  };
}

function saveIncident(config, incident) {
  writeJsonAtomic(incidentPath(config, incident.ticketId), incident);
}

// Community Edition: incidents stay on this machine. Nothing is posted to a
// forum or sent by e-mail; the record is kept in the local incident folder.
async function deliverIncident(incident, { config }) {
  incident.status = "recorded_locally";
  incident.updatedAt = nowIso();
  saveIncident(config, incident);
  return incident;
}

function publicIncidentResult(incident) {
  return {
    reported: true,
    ticketId: incident.ticketId,
    status: incident.status,
    communityUrl: incident.delivery.nodebb?.url ?? null,
    message: incident.delivery.nodebb?.url
      ? `Critical failure recorded as ${incident.ticketId}. The anonymized community discussion is ${incident.delivery.nodebb.url}.`
      : `Critical failure recorded locally as ${incident.ticketId}. Nothing was sent anywhere.`,
  };
}

export async function reportMajorFailure({
  error,
  config,
  logger = null,
  context = {},
  hardware = null,
  runtimeState = null,
  user = null,
  fetchImpl = fetch,
  now = Date.now,
  randomBytes = crypto.randomBytes,
} = {}) {
  if (config?.failureIntelligence?.enabled === false) {
    return { reported: false, reason: "disabled" };
  }
  const initialClassification = classifyMajorFailure(error, context);
  if (!initialClassification.major && !initialClassification.crashCandidate) {
    return {
      reported: false,
      reason: initialClassification.reason,
      repeatCount: 0,
    };
  }
  const fingerprint = fingerprintFor(error, context, initialClassification);
  const repeat = updateRepeatIndex({ config, fingerprint, now });
  const repeatCount = repeat.count;
  const repeatThreshold = Number(config?.failureIntelligence?.repeatThreshold) || DEFAULT_REPEAT_THRESHOLD;
  const classification = !initialClassification.major
    && initialClassification.crashCandidate
    && repeatCount >= repeatThreshold
    ? {
      major: true,
      severity: "critical",
      category: "repeated_runtime_crash",
      reason: `The same runtime crash occurred ${repeatCount} times inside the repeat window.`,
    }
    : initialClassification;
  if (!classification.major) {
    logger?.warn?.("failure-intelligence-filtered", {
      code: normalizeCode(error) || null,
      category: classification.category,
      repeatCount,
    });
    return {
      reported: false,
      reason: classification.reason,
      repeatCount,
    };
  }

  if (repeat.ticketId) {
    const existing = readMajorFailureIncident(config, repeat.ticketId);
    if (existing) {
      existing.repeatCount = repeatCount;
      existing.occurrences = Number(existing.occurrences ?? 1) + 1;
      existing.updatedAt = nowIso(now);
      existing.lastOccurrence = {
        at: nowIso(now),
        code: normalizeCode(error) || null,
        component: context.component ?? null,
      };
      saveIncident(config, existing);
      logger?.error?.("major-failure-ticket-reused", {
        ticketId: existing.ticketId,
        category: existing.classification?.category ?? classification.category,
        repeatCount,
      });
      const delivered = existing.status === "delivered"
        ? existing
        : await deliverIncident(existing, { config, fetchImpl });
      return publicIncidentResult(delivered);
    }
  }

  const id = ticketId(now, randomBytes);
  const root = incidentRoot(config);
  const storageDirectory = path.join(root, id);
  ensurePrivateDirectory(storageDirectory);
  const normalizedError = redactSecrets(normalizeError(error));
  const logs = copyDiagnosticLogs({ config, directory: storageDirectory });
  const diagnostics = {
    capturedAt: nowIso(now),
    error: normalizedError,
    hardware: hardwareSummary(hardware),
    runtime: redactSecrets({
      runtimeProvider: context.runtimeProvider ?? runtimeState?.runtimeProvider ?? null,
      backend: context.backend ?? runtimeState?.backend ?? null,
      architecture: context.architecture ?? runtimeState?.architecture ?? null,
      quantization: context.quantization ?? runtimeState?.quantization ?? null,
      modelId: context.modelId ?? runtimeState?.modelId ?? null,
      state: runtimeState,
    }),
    process: {
      pid: process.pid,
      node: process.version,
      platform: process.platform,
      arch: process.arch,
      uptimeSeconds: process.uptime(),
      memoryUsage: process.memoryUsage(),
    },
    environment: environmentSnapshot(),
    logs,
    reproduction: redactSecrets(context.reproduction ?? {
      component: context.component ?? null,
      operation: context.operation ?? null,
      route: context.route ?? null,
    }),
  };
  writeJsonAtomic(path.join(storageDirectory, "diagnostics.json"), diagnostics);

  const email = String(user?.email ?? "").trim() || null;
  const incident = {
    schema: INCIDENT_SCHEMA,
    ticketId: id,
    status: "open",
    createdAt: nowIso(now),
    updatedAt: nowIso(now),
    fingerprint,
    repeatCount,
    occurrences: 1,
    classification,
    context: redactSecrets({
      component: context.component ?? null,
      operation: context.operation ?? null,
      route: context.route ?? null,
      surface: context.surface ?? null,
    }),
    storageRoot: root,
    storageDirectory,
    diagnostics,
    privateContact: {
      email,
      subjectHash: user?.subject
        ? crypto.createHash("sha256").update(String(user.subject)).digest("hex")
        : null,
    },
    delivery: {
      nodebb: { status: "pending" },
      adminEmail: { status: "pending" },
      userEmail: email ? { status: "pending" } : { status: "not_available" },
    },
  };
  saveIncident(config, incident);
  assignTicketToFingerprint({ config, fingerprint, id });
  logger?.error?.("major-failure-ticket-created", {
    ticketId: id,
    category: classification.category,
    repeatCount,
  });
  await deliverIncident(incident, { config, fetchImpl });
  return publicIncidentResult(incident);
}

export function listMajorFailureIncidents(config, { limit = 100 } = {}) {
  const root = incidentRoot(config);
  ensurePrivateDirectory(root);
  return fs.readdirSync(root, { withFileTypes: true })
    .filter((entry) => entry.isDirectory() && /^LIZ-\d{8}-[A-F0-9]{10}$/.test(entry.name))
    .map((entry) => readJson(path.join(root, entry.name, "incident.json"), null))
    .filter((entry) => entry?.schema === INCIDENT_SCHEMA)
    .sort((left, right) => String(right.createdAt).localeCompare(String(left.createdAt)))
    .slice(0, Math.max(1, Math.min(Number(limit) || 100, 1_000)))
    .map((incident) => ({
      ticketId: incident.ticketId,
      status: incident.status,
      createdAt: incident.createdAt,
      updatedAt: incident.updatedAt,
      category: incident.classification?.category ?? null,
      code: incident.diagnostics?.error?.code ?? null,
      component: incident.context?.component ?? null,
      communityUrl: incident.delivery?.nodebb?.url ?? null,
      delivery: {
        nodebb: incident.delivery?.nodebb?.status ?? null,
        adminEmail: incident.delivery?.adminEmail?.status ?? null,
        userEmail: incident.delivery?.userEmail?.status ?? null,
      },
    }));
}

export function readMajorFailureIncident(config, id) {
  if (!/^LIZ-\d{8}-[A-F0-9]{10}$/.test(String(id ?? ""))) return null;
  const incident = readJson(incidentPath(config, id), null);
  return incident?.schema === INCIDENT_SCHEMA ? incident : null;
}

export async function retryPendingMajorFailureIncidents({
  config,
  logger = null,
  fetchImpl = fetch,
  limit = 25,
} = {}) {
  const summaries = listMajorFailureIncidents(config, { limit: 1_000 })
    .filter((entry) => entry.status !== "delivered" && entry.status !== "recorded_locally")
    .slice(0, Math.max(1, Math.min(Number(limit) || 25, 100)));
  const results = [];
  for (const summary of summaries) {
    const incident = readMajorFailureIncident(config, summary.ticketId);
    if (!incident) continue;
    const delivered = await deliverIncident(incident, { config, fetchImpl });
    results.push(publicIncidentResult(delivered));
  }
  if (results.length) {
    logger?.info?.("major-failure-delivery-retry", {
      attempted: results.length,
      delivered: results.filter((entry) => entry.status === "delivered").length,
    });
  }
  return results;
}

export function incidentUserFields(result) {
  if (!result?.reported) return {};
  return {
    ticketId: result.ticketId,
    incidentStatus: result.status,
    communityUrl: result.communityUrl,
    incidentMessage: result.message,
  };
}
