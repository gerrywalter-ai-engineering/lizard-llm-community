import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import {
  buildProtocolCapabilities,
  CATERPILLAR_PROTOCOL_SPEC_VERSION,
  CATERPILLAR_WORKER_RPC_VERSION,
  renderProtocolCompatibilityMarkdown,
} from "../src/runtime/server/protocol/capabilities.js";

test("Caterpillar capability discovery advertises only truthful profile states", () => {
  const capabilities = buildProtocolCapabilities({
    runtimeProvider: "caterpillar",
    modelArtifact: { modelId: "tiny-caterpillar", modelAliases: ["tiny-friendly"] },
    modelReady: true,
    productVersion: "1.2.60",
    buildCommit: "abc123",
  });

  assert.equal(capabilities.version, CATERPILLAR_PROTOCOL_SPEC_VERSION);
  assert.equal(capabilities.runtime.provider, "caterpillar");
  assert.equal(capabilities.runtime.modelId, "tiny-caterpillar");
  assert.deepEqual(capabilities.runtime.modelAliases, ["tiny-friendly"]);
  assert.equal(capabilities.runtime.modelReady, true);
  assert.deepEqual(capabilities.runtime.samplingModes, ["greedy"]);
  assert.equal(capabilities.profiles["CAT-BASE-1"], "supported");
  assert.equal(capabilities.profiles["CAT-OAI-CHAT-1"], "supported");
  assert.equal(capabilities.profiles["CAT-LLAMA-SERVER-1"], "supported");
  assert.equal(capabilities.profiles["CAT-OLLAMA-CHAT-1"], "supported");
  assert.equal(capabilities.profiles["CAT-STRUCTURED-1"], "unsupported");
  assert.equal(capabilities.profiles["CAT-TOOLS-1"], "unsupported");
  assert.equal(capabilities.profiles["CAT-RESPONSES-1"], "supported");
  assert.deepEqual(capabilities.workerRpc, {
    minVersion: CATERPILLAR_WORKER_RPC_VERSION,
    maxVersion: CATERPILLAR_WORKER_RPC_VERSION,
    selectedVersion: CATERPILLAR_WORKER_RPC_VERSION,
  });
  assert.ok(capabilities.routes.some((route) => route.path === "/capabilities"));
  assert.ok(capabilities.routes.some((route) => route.path === "/v1/health"));
  assert.ok(capabilities.routes.some((route) => (
    route.path === "/v1/chat/completions" && route.profile === "CAT-OAI-CHAT-1"
  )));
  assert.equal(capabilities.constraints.maxJsonBodyBytes, 1024 * 1024);
  assert.equal(capabilities.extensionFieldPolicy, "reject-unknown-request-fields");
  assert.equal(capabilities.streaming.openai, "supported");
  assert.equal(capabilities.streaming.llamaCompletion, "supported");
  assert.equal(capabilities.streaming.ollama, "supported-ndjson");
  assert.ok(capabilities.routes.some((route) => route.path === "/detokenize"));
  assert.ok(capabilities.routes.some((route) => route.path === "/completion"));
  assert.ok(capabilities.routes.some((route) => route.path === "/v1/completions"));
  assert.ok(capabilities.routes.some((route) => (
    route.path === "/api/chat" && route.profile === "CAT-OLLAMA-CHAT-1"
  )));
  assert.ok(capabilities.routes.some((route) => route.path === "/api/generate"));
  assert.ok(capabilities.excludedRoutes.includes("/api/pull"));
  assert.equal(capabilities.excludedRoutes.includes("/v1/responses"), false);
  assert.ok(capabilities.routes.some((route) => (
    route.path === "/v1/responses" && route.profile === "CAT-RESPONSES-1"
  )));
  assert.ok(capabilities.excludedRoutes.includes("/slots"));
  assert.deepEqual(capabilities.metrics, { route: "/metrics", format: "lizard-json", prometheus: false });
  assert.deepEqual(capabilities.requestFields["/v1/chat/completions"].supported, [
    "model", "messages", "stream", "stream_options", "max_tokens", "max_completion_tokens",
    "seed", "stop", "user",
  ]);
  assert.deepEqual(capabilities.requestFields["/v1/chat/completions"].defaultOnly.tools, []);
  assert.equal(capabilities.requestFields["/v1/chat/completions"].defaults.stream, false);
  assert.ok(capabilities.requestFields["/v1/chat/completions"].supported.includes("stream"));
  assert.ok(capabilities.requestFields["/v1/chat/completions"].supported.includes("stream_options"));
  assert.ok(capabilities.limitations.includes("deterministic-greedy-sampling-only"));
});

test("non-Caterpillar servers do not claim Caterpillar compatibility profiles", () => {
  const capabilities = buildProtocolCapabilities({ runtimeProvider: "lizard-native" });
  assert.equal(capabilities.profiles["CAT-BASE-1"], "not-applicable");
  assert.equal(capabilities.profiles["CAT-OLLAMA-CHAT-1"], "not-applicable");
  assert.equal(capabilities.workerRpc, null);
  assert.deepEqual(capabilities.limitations, []);
});
