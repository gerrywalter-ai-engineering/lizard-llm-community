import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import {
  anonymizeCommunityText,
  classifyMajorFailure,
  incidentUserFields,
  listMajorFailureIncidents,
  readMajorFailureIncident,
  reportMajorFailure,
  retryPendingMajorFailureIncidents,
} from "../src/lib/major-failure-intelligence.js";

function testConfig(directory, overrides = {}) {
  const runtimeDirectory = path.join(directory, "runtime");
  return {
    runtimeDirectory,
    logging: { directory: path.join(runtimeDirectory, "logs") },
    failureIntelligence: {
      enabled: true,
      directory: path.join(runtimeDirectory, "incidents"),
      repeatWindowMs: 60_000,
      repeatThreshold: 3,
      requestTimeoutMs: 1_000,
      nodebb: {
        baseUrl: null,
        apiToken: null,
        openQuestionsCategoryId: null,
      },
      email: {
        endpoint: null,
        apiToken: null,
        adminTo: [],
        from: null,
      },
      ...overrides,
    },
  };
}

function success(body) {
  return {
    ok: true,
    status: 200,
    async json() {
      return body;
    },
  };
}

test("minor compatibility and request failures never create tickets or repeat state", async () => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "lizard-major-minor-"));
  try {
    const config = testConfig(directory);
    const error = Object.assign(new Error("This model family is not implemented."), {
      code: "NATIVE_MODEL_UNSUPPORTED",
    });
    const result = await reportMajorFailure({ error, config });
    assert.equal(result.reported, false);
    assert.equal(result.repeatCount, 0);
    assert.equal(fs.existsSync(config.failureIntelligence.directory), false);
  } finally {
    fs.rmSync(directory, { recursive: true, force: true });
  }
});

test("a generic runtime crash becomes major only after the repeat threshold", async () => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "lizard-major-repeat-"));
  try {
    const config = testConfig(directory);
    const error = new Error("worker process exited with signal 11");
    const common = {
      error,
      config,
      context: { component: "native-worker", crash: true },
      now: () => Date.UTC(2026, 6, 29, 12, 0, 0),
    };
    assert.equal((await reportMajorFailure(common)).reported, false);
    assert.equal((await reportMajorFailure(common)).reported, false);
    const third = await reportMajorFailure({
      ...common,
      randomBytes: () => Buffer.from("1112131415", "hex"),
    });
    assert.equal(third.reported, true);
    assert.equal(third.ticketId, "LIZ-20260729-1112131415");
    assert.equal(readMajorFailureIncident(config, third.ticketId).classification.category, "repeated_runtime_crash");
  } finally {
    fs.rmSync(directory, { recursive: true, force: true });
  }
});

test("the same major failure reuses its Ticket ID instead of spamming duplicate incidents", async () => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "lizard-major-dedupe-"));
  try {
    const config = testConfig(directory);
    const common = {
      error: Object.assign(new Error("GGUF checksum mismatch"), { code: "GGUF_CHECKSUM_MISMATCH" }),
      config,
      context: { component: "download-verifier" },
      now: () => Date.UTC(2026, 6, 29, 12, 0, 0),
    };
    const first = await reportMajorFailure({
      ...common,
      randomBytes: () => Buffer.from("3132333435", "hex"),
    });
    const second = await reportMajorFailure({
      ...common,
      randomBytes: () => Buffer.from("4142434445", "hex"),
    });
    assert.equal(second.ticketId, first.ticketId);
    assert.equal(listMajorFailureIncidents(config).length, 1);
    const incident = readMajorFailureIncident(config, first.ticketId);
    assert.equal(incident.occurrences, 2);
    assert.equal(incident.repeatCount, 2);
  } finally {
    fs.rmSync(directory, { recursive: true, force: true });
  }
});

test("classification and community redaction cover critical hardware and personal data", () => {
  const hardware = Object.assign(new Error("Required instruction set is unavailable"), {
    code: "NATIVE_CPU_INCOMPATIBLE",
  });
  assert.deepEqual(classifyMajorFailure(hardware).category, "unsupported_hardware");
  const redacted = anonymizeCommunityText(
    "alice@example.test C:\\Users\\Alice\\model.gguf /home/alice/model.gguf 10.0.0.8 token=secret",
  );
  assert.doesNotMatch(redacted, /alice@example|Users\\Alice|home\/alice|10\.0\.0\.8|token=secret/i);
});
