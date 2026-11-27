import test from "node:test";
import assert from "node:assert/strict";

import {
  ENDPOINT_PORT_BY_PROVIDER,
  ENDPOINT_STOP_REASONS,
  armLocalEndpoint,
  describeLocalEndpoint,
  disarmLocalEndpoint,
  endpointProvider,
  getLocalEndpointState,
  localEndpointIsListening,
  probeEndpointPort,
  resolveEndpointAddress,
  suspendLocalEndpointForBenchmark,
  __resetLocalEndpointForTests,
} from "../src/lib/local-endpoint.js";

// Ports well outside the shipped 8013-8015 range so a developer machine with a
// real endpoint or a leftover `lizard-llm serve` running cannot make these pass
// or fail for the wrong reason.
//
// A fresh triple per test, because fetch() pools keep-alive sockets by host and
// port: reusing one port across tests hands the next test a socket to a server
// that has since been closed, and the ECONNRESET that follows is the pool's
// fault rather than the endpoint's.
const PORT_BASE = 18600;
let portCursor = 0;
let TEST_PORTS;
let config;

function nextPorts() {
  portCursor += 1;
  const base = PORT_BASE + (portCursor * 3);
  return { "lizard-native": base, caterpillar: base + 1, ostrich: base + 2 };
}

test.beforeEach(() => {
  TEST_PORTS = nextPorts();
  config = {
    engine: {
      host: "127.0.0.1",
      native: { host: "127.0.0.1", port: TEST_PORTS["lizard-native"] },
      caterpillar: { host: "127.0.0.1", port: TEST_PORTS.caterpillar },
      ostrich: { host: "127.0.0.1", port: TEST_PORTS.ostrich },
    },
  };
});

function stubHandlers({ output = "ok", deltas = ["o", "k"], fail = null } = {}) {
  const calls = [];
  return {
    calls,
    handlers: {
      createCompletion: async ({ prompt, systemPrompt, maxTokens, onDelta }) => {
        calls.push({ prompt, systemPrompt, maxTokens });
        if (fail) throw new Error(fail);
        for (const delta of deltas) onDelta?.(delta);
        return { output, modelId: "stub-model", promptTokens: 4, generatedTokens: deltas.length };
      },
    },
  };
}

async function arm(provider = "caterpillar", options = {}) {
  const { handlers, calls } = stubHandlers(options.stub ?? {});
  const state = await armLocalEndpoint({
    config,
    runtimeProvider: provider,
    modelId: options.modelId ?? "stub-model",
    handlers,
    ...options.arm,
  });
  return { state, calls };
}

test.afterEach(async () => {
  await __resetLocalEndpointForTests();
});

test("each engine gets its own port, and only the three Lizard engines have one", () => {
  assert.deepEqual(ENDPOINT_PORT_BY_PROVIDER, {
    "lizard-native": 8013,
    caterpillar: 8014,
    ostrich: 8015,
  });
  assert.equal(endpointProvider("ostrich"), "ostrich");
  assert.equal(endpointProvider("Caterpillar"), "caterpillar");
  // llama.cpp and Ollama run the model themselves - advertising a Lizard port
  // for them would name an address that never answers.
  assert.equal(endpointProvider("llama.cpp"), null);
  assert.equal(endpointProvider("ollama"), null);
});

test("the advertised address comes from config, and a wildcard bind still shows a usable URL", () => {
  const address = resolveEndpointAddress({ engine: { caterpillar: { host: "0.0.0.0", port: 9999 } } }, "caterpillar");
  assert.equal(address.host, "0.0.0.0");
  assert.equal(address.url, "http://127.0.0.1:9999");
  assert.equal(address.chatCompletionsUrl, "http://127.0.0.1:9999/v1/chat/completions");
});

test("an armed endpoint answers /health, /v1/models and a buffered completion", async () => {
  const { state } = await arm("caterpillar");
  assert.equal(state.status, "listening");
  assert.equal(state.port, TEST_PORTS.caterpillar);
  assert.ok(localEndpointIsListening());

  const health = await (await fetch(`${state.url}/health`)).json();
  assert.equal(health.status, "ok");
  assert.equal(health.model, "stub-model");
  assert.equal(health.runtimeProvider, "caterpillar");

  const models = await (await fetch(`${state.url}/v1/models`)).json();
  assert.equal(models.object, "list");
  assert.equal(models.data[0].id, "stub-model");
  // Clients display owned_by verbatim, so it names the engine once - the first
  // live run of this endpoint reported "lizard-lizard-native".
  assert.equal(models.data[0].owned_by, "caterpillar");

  const completion = await (await fetch(`${state.url}/v1/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model: "stub-model", messages: [{ role: "user", content: "hi" }] }),
  })).json();
  assert.equal(completion.object, "chat.completion");
  assert.equal(completion.choices[0].message.content, "ok");
  assert.equal(completion.choices[0].finish_reason, "stop");
});

test("a streaming completion emits a role chunk, deltas, a stop and [DONE]", async () => {
  const { state } = await arm("caterpillar", { stub: { deltas: ["He", "llo"], output: "Hello" } });
  const response = await fetch(`${state.url}/v1/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model: "stub-model", stream: true, messages: [{ role: "user", content: "hi" }] }),
  });
  const body = await response.text();
  const events = body.split("\n\n").filter((line) => line.startsWith("data: ")).map((line) => line.slice(6));
  assert.equal(events.at(-1), "[DONE]");
  const parsed = events.slice(0, -1).map((line) => JSON.parse(line));
  assert.deepEqual(parsed[0].choices[0].delta, { role: "assistant" });
  assert.deepEqual(parsed.slice(1, -1).map((chunk) => chunk.choices[0].delta.content), ["He", "llo"]);
  assert.equal(parsed.at(-1).choices[0].finish_reason, "stop");
});

test("system messages and multi-turn history reach the runtime as one prompt", async () => {
  const { handlers, calls } = stubHandlers();
  const state = await armLocalEndpoint({ config, runtimeProvider: "caterpillar", modelId: "stub-model", handlers });
  await fetch(`${state.url}/v1/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      messages: [
        { role: "system", content: "Be terse." },
        { role: "user", content: "first" },
        { role: "assistant", content: "reply" },
        { role: "user", content: "second" },
      ],
    }),
  });
  assert.equal(calls[0].systemPrompt, "Be terse.");
  assert.match(calls[0].prompt, /User: first/);
  assert.match(calls[0].prompt, /Assistant: reply/);
  assert.match(calls[0].prompt, /User: second/);
});

test("a request with no user text is refused as an invalid request, not a server error", async () => {
  const { state } = await arm("caterpillar");
  const response = await fetch(`${state.url}/v1/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ messages: [{ role: "system", content: "only a system prompt" }] }),
  });
  assert.equal(response.status, 400);
  const payload = await response.json();
  assert.equal(payload.error.code, "missing_prompt");
});

test("a generation failure mid-stream is written into the stream, not left as a silent empty answer", async () => {
  const { state } = await arm("caterpillar", { stub: { fail: "worker died" } });
  const response = await fetch(`${state.url}/v1/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ stream: true, messages: [{ role: "user", content: "hi" }] }),
  });
  const body = await response.text();
  assert.match(body, /worker died/);
  assert.match(body, /\[DONE\]/);
});

test("only one engine serves at a time: arming a second closes the first", async () => {
  const first = await arm("caterpillar");
  assert.equal(await probeEndpointPort({ port: TEST_PORTS.caterpillar }), true);

  const second = await arm("ostrich");
  assert.equal(second.state.status, "listening");
  assert.equal(second.state.runtimeProvider, "ostrich");
  assert.equal(second.state.port, TEST_PORTS.ostrich);
  assert.equal(
    await probeEndpointPort({ port: TEST_PORTS.caterpillar }),
    false,
    "the previous engine's port must be released when another engine takes over",
  );
  assert.equal(first.state.runtimeProvider, "caterpillar");
});

test("re-arming the same model on the same engine is a no-op, not a restart", async () => {
  const { handlers } = stubHandlers();
  const first = await armLocalEndpoint({ config, runtimeProvider: "caterpillar", modelId: "stub-model", handlers });
  const again = await armLocalEndpoint({ config, runtimeProvider: "caterpillar", modelId: "stub-model", handlers });
  assert.equal(again.status, "listening");
  // A restart would move startedAt; onboarding calls this on every page load.
  assert.equal(again.startedAt, first.startedAt);
});

test("unloading closes the port and says why, with no reload prompt", async () => {
  await arm("caterpillar");
  const stopped = await disarmLocalEndpoint({ reason: ENDPOINT_STOP_REASONS.UNLOAD });
  assert.equal(stopped.status, "stopped");
  assert.equal(stopped.stopReason, ENDPOINT_STOP_REASONS.UNLOAD);
  assert.match(stopped.stopNote, /until you activate a model again/);
  // An unload is what the user asked for; offering "load it again" there would
  // undo their instruction.
  assert.equal(stopped.needsModelReload, false);
  assert.equal(await probeEndpointPort({ port: TEST_PORTS.caterpillar }), false);
  // The address survives the stop so the UI can still name the port it closed.
  assert.equal(stopped.port, TEST_PORTS.caterpillar);
});

test("a benchmark stop is attributed and marked as recoverable by reloading the model", async () => {
  await arm("caterpillar");
  const stopped = await suspendLocalEndpointForBenchmark({});
  assert.equal(stopped.status, "stopped");
  assert.equal(stopped.stopReason, ENDPOINT_STOP_REASONS.BENCHMARK);
  assert.equal(stopped.needsModelReload, true);
  assert.match(stopped.stopNote, /Load the model once more/);
  assert.equal(await probeEndpointPort({ port: TEST_PORTS.caterpillar }), false);
});

test("freeing memory is also a reload-recoverable stop, and shutdown is not", async () => {
  await arm("caterpillar");
  assert.equal((await disarmLocalEndpoint({ reason: ENDPOINT_STOP_REASONS.FREE_MEMORY })).needsModelReload, true);
  await arm("caterpillar");
  assert.equal((await disarmLocalEndpoint({ reason: ENDPOINT_STOP_REASONS.SHUTDOWN })).needsModelReload, false);
});

test("a port already taken by something else fails loudly instead of silently not serving", async () => {
  const http = await import("node:http");
  const squatter = http.createServer((_req, res) => res.end("not lizard"));
  await new Promise((resolve) => squatter.listen(TEST_PORTS.ostrich, "127.0.0.1", resolve));
  try {
    const state = await armLocalEndpoint({
      config,
      runtimeProvider: "ostrich",
      modelId: "stub-model",
      handlers: stubHandlers().handlers,
    });
    assert.equal(state.status, "failed");
    assert.equal(state.stopReason, ENDPOINT_STOP_REASONS.PORT_UNAVAILABLE);
    assert.equal(state.needsModelReload, true);
    assert.match(state.stopNote, new RegExp(String(TEST_PORTS.ostrich)));
  } finally {
    await new Promise((resolve) => squatter.close(resolve));
  }
});

test("describeLocalEndpoint names every engine's address so the UI can show the selected one", () => {
  const described = describeLocalEndpoint(config, { selectedProvider: "ostrich" });
  assert.equal(described.selected.port, TEST_PORTS.ostrich);
  assert.deepEqual(
    described.providers.map((provider) => provider.runtimeProvider),
    ["lizard-native", "caterpillar", "ostrich"],
  );
  assert.equal(described.proFeature, true);
});

test("probing reports an occupied port as occupied and a free one as free", async () => {
  assert.equal(await probeEndpointPort({ port: TEST_PORTS["lizard-native"] }), false);
  await arm("lizard-native");
  assert.equal(await probeEndpointPort({ port: TEST_PORTS["lizard-native"] }), true);
  await disarmLocalEndpoint({ reason: ENDPOINT_STOP_REASONS.UNLOAD });
  assert.equal(await probeEndpointPort({ port: TEST_PORTS["lizard-native"] }), false);
});

test("sockets are configured for long generations rather than Node's request-timeout defaults", async () => {
  await arm("caterpillar");
  const state = getLocalEndpointState();
  assert.equal(state.status, "listening");
  // A 300 s request timeout (Node's default since 18) cuts a long answer off
  // mid-stream, which is indistinguishable from the endpoint dying.
  const response = await fetch(`${state.url}/health`);
  assert.equal(response.status, 200);
  assert.equal(response.headers.get("access-control-allow-origin"), "*");
});

test("a browser preflight is answered so cross-origin local tools can call the endpoint", async () => {
  const { state } = await arm("caterpillar");
  const response = await fetch(`${state.url}/v1/chat/completions`, { method: "OPTIONS" });
  assert.equal(response.status, 204);
  assert.equal(response.headers.get("access-control-allow-methods"), "GET, POST, OPTIONS");
});

test("an unknown route names what the endpoint does serve", async () => {
  const { state } = await arm("caterpillar");
  const response = await fetch(`${state.url}/v1/embeddings`, { method: "POST" });
  assert.equal(response.status, 404);
  const payload = await response.json();
  assert.match(payload.error.message, /\/v1\/chat\/completions/);
});
