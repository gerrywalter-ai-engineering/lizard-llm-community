import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { writeNativeServerLaunchConfig } from "../src/runtime/server/launch-config.js";

test("native server launch files never persist incident credentials or contact addresses", () => {
  const runtimeDirectory = fs.mkdtempSync(path.join(os.tmpdir(), "lizard-launch-secret-"));
  try {
    const launchPath = writeNativeServerLaunchConfig({
      config: {
        runtimeDirectory,
        failureIntelligence: {
          enabled: true,
          directory: path.join(runtimeDirectory, "incidents"),
          userEmail: "private-user@example.test",
          nodebb: {
            baseUrl: "https://community.example.test",
            apiToken: "nodebb-secret",
            openQuestionsCategoryId: 7,
          },
          email: {
            endpoint: "https://mail.example.test/send",
            apiToken: "mail-secret",
            adminTo: ["engineering@example.test"],
            from: "incidents@example.test",
          },
        },
      },
      modelArtifact: { modelId: "test", modelPath: "test.gguf" },
      profile: {},
      recommendation: {},
    });
    const raw = fs.readFileSync(launchPath, "utf8");
    const launch = JSON.parse(raw);
    assert.doesNotMatch(raw, /nodebb-secret|mail-secret|private-user@example|engineering@example/);
    assert.equal(launch.config.failureIntelligence.nodebb.apiToken, null);
    assert.equal(launch.config.failureIntelligence.email.apiToken, null);
    assert.deepEqual(launch.config.failureIntelligence.email.adminTo, []);
    assert.equal(launch.config.failureIntelligence.userEmail, null);
  } finally {
    fs.rmSync(runtimeDirectory, { recursive: true, force: true });
  }
});
