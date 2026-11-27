import test from "node:test";
import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import { PassThrough } from "node:stream";
import {
  assertJsonDepth,
  readJsonBody,
  resolveHttpRequestId,
} from "../src/runtime/server/protocol/http.js";
import { createRequestCancellation } from "../src/runtime/server/protocol/cancellation.js";
import {
  classifyProtocolHttpError,
  ProtocolHttpError,
} from "../src/runtime/server/protocol/errors.js";
import { validateChatRequest } from "../src/runtime/server/protocol/validation.js";
import { writeWithBackpressure } from "../src/runtime/server/protocol/stream.js";
import {
  attachBoundedLineReader,
  MAX_NATIVE_RPC_LINE_BYTES,
} from "../src/runtime/worker/process-client.js";

function requestWith(body, headers = { "content-type": "application/json" }) {
  const request = new PassThrough();
  request.headers = headers;
  queueMicrotask(() => request.end(body));
  return request;
}

test("bounded JSON ingress accepts JSON media types and rejects malformed shapes", async () => {
  assert.deepEqual(await readJsonBody(requestWith('{"ok":true}')), { ok: true });
  assert.deepEqual(await readJsonBody(requestWith('{"ok":true}', { "content-type": "application/problem+json" })), { ok: true });

  await assert.rejects(readJsonBody(requestWith("[]")), { code: "invalid_json_type", statusCode: 400 });
  await assert.rejects(readJsonBody(requestWith("{")), { code: "invalid_json", statusCode: 400 });
  await assert.rejects(readJsonBody(requestWith("{}", { "content-type": "text/plain" })), {
    code: "invalid_content_type",
    statusCode: 400,
  });
  await assert.rejects(readJsonBody(requestWith(Buffer.from([0x7B, 0x22, 0x78, 0x22, 0x3A, 0xFF, 0x7D]))), {
    code: "invalid_utf8",
    statusCode: 400,
  });
});

test("bounded JSON ingress enforces streamed bytes, declared bytes, and nesting", async () => {
  await assert.rejects(readJsonBody(requestWith('{"value":"oversized"}'), { maxBytes: 8 }), {
    code: "request_body_too_large",
    statusCode: 413,
  });
  assert.throws(() => assertJsonDepth({ a: { b: { c: true } } }, 2), { code: "json_too_deep" });
  const declared = requestWith("{}", {
    "content-type": "application/json",
    "content-length": "200",
  });
  await assert.rejects(readJsonBody(declared, { maxBytes: 100 }), { code: "request_body_too_large" });
});

test("bounded JSON ingress rejects a stalled upload", async () => {
  const request = new PassThrough();
  request.headers = { "content-type": "application/json" };
  await assert.rejects(readJsonBody(request, { timeoutMs: 10 }), {
    code: "request_body_timeout",
    statusCode: 408,
  });
  request.destroy();
});

test("request IDs retain only bounded safe incoming values", () => {
  assert.equal(resolveHttpRequestId({ "x-request-id": "safe-id/123" }), "safe-id/123");
  assert.match(resolveHttpRequestId({ "x-request-id": "bad id\r\nInjected: yes" }), /^[0-9a-f-]{36}$/);
  assert.match(resolveHttpRequestId({ "x-request-id": "x".repeat(129) }), /^[0-9a-f-]{36}$/);
});

test("chat request validation enforces semantic limits and unsupported fields", () => {
  const valid = validateChatRequest({
    model: "tiny",
    messages: [{ role: "user", content: "hello" }],
    stream: false,
    max_tokens: 2,
    temperature: 0,
    n: 1,
  }, { modelId: "tiny", runtimeProvider: "caterpillar" });
  assert.equal(valid.messages.length, 1);
  assert.throws(() => validateChatRequest({ messages: "hello" }), { code: "invalid_parameter" });
  assert.throws(() => validateChatRequest({ messages: [], tools: [] }), { code: "invalid_parameter", param: "messages" });
  assert.throws(() => validateChatRequest({ messages: [], temperature: 0.5 }, { runtimeProvider: "caterpillar" }), {
    code: "unsupported_parameter",
  });
  assert.throws(() => validateChatRequest({ messages: [], top_p: 0.9 }, { runtimeProvider: "caterpillar" }), {
    code: "unsupported_parameter",
  });
  assert.throws(() => validateChatRequest({ messages: [], seed: 42 }, { runtimeProvider: "caterpillar" }), {
    code: "invalid_parameter",
    param: "messages",
  });
  assert.throws(() => validateChatRequest({ model: "other", messages: [] }, { modelId: "tiny" }), {
    code: "model_not_found",
    statusCode: 404,
  });
});

test("stable error classification covers the CAT-BASE status policy", () => {
  for (const statusCode of [400, 401, 404, 408, 413, 429, 503, 504]) {
    assert.equal(classifyProtocolHttpError(new ProtocolHttpError(statusCode, `code_${statusCode}`, "test")).statusCode, statusCode);
  }
  assert.equal(classifyProtocolHttpError(Object.assign(new Error("capability"), { code: "KV_BUDGET_EXCEEDED" })).statusCode, 422);
  assert.equal(classifyProtocolHttpError(Object.assign(new Error("cancelled"), { code: "SCHEDULER_CANCELLED" })).statusCode, 499);
  assert.equal(classifyProtocolHttpError(Object.assign(new Error("timeout"), { code: "SCHEDULER_TIMEOUT" })).statusCode, 504);
  assert.equal(classifyProtocolHttpError(new Error("defect")).statusCode, 500);
});

test("request cancellation is idempotent and closes only its bound session", async () => {
  const req = new EventEmitter();
  const res = new EventEmitter();
  res.writableEnded = false;
  let closes = 0;
  let otherCloses = 0;
  const cancellation = createRequestCancellation({ req, res });
  const otherCancellation = createRequestCancellation({ req: new EventEmitter(), res: new EventEmitter() });
  cancellation.bindSession({ close: async () => { closes += 1; } });
  otherCancellation.bindSession({ close: async () => { otherCloses += 1; } });
  req.emit("aborted");
  res.emit("close");
  await cancellation.dispose();
  assert.equal(cancellation.signal.aborted, true);
  assert.equal(cancellation.reason, "client_disconnect");
  assert.equal(closes, 1);
  assert.equal(otherCloses, 0);
  assert.ok(cancellation.timings.cancelToCloseStartMs >= 0);
  assert.ok(cancellation.timings.cancelToCloseCompleteMs >= cancellation.timings.cancelToCloseStartMs);
  assert.equal(req.listenerCount("aborted"), 0);
  assert.equal(res.listenerCount("close"), 0);
  await otherCancellation.dispose();
});

test("cancellation before session allocation closes a later session and completed responses do not cancel", async () => {
  const earlyReq = new EventEmitter();
  const earlyRes = new EventEmitter();
  earlyRes.writableEnded = false;
  const early = createRequestCancellation({ req: earlyReq, res: earlyRes });
  earlyReq.emit("aborted");
  let closes = 0;
  early.bindSession({ close: async () => { closes += 1; } });
  await early.dispose();
  assert.equal(closes, 1);

  const completedReq = new EventEmitter();
  const completedRes = new EventEmitter();
  completedRes.writableEnded = true;
  const completed = createRequestCancellation({ req: completedReq, res: completedRes });
  completed.bindSession({ close: async () => { closes += 1; } });
  completedRes.emit("close");
  await completed.dispose();
  assert.equal(completed.signal.aborted, false);
  assert.equal(closes, 1);
});

test("request deadline converges on the same session cancellation operation", async () => {
  const req = new EventEmitter();
  const res = new EventEmitter();
  res.writableEnded = false;
  let closes = 0;
  const cancellation = createRequestCancellation({ req, res, timeoutMs: 10 });
  cancellation.bindSession({ close: async () => { closes += 1; } });
  await new Promise((resolve) => setTimeout(resolve, 20));
  await cancellation.dispose();
  assert.equal(cancellation.reason, "deadline_exceeded");
  assert.equal(closes, 1);
  assert.ok(cancellation.timings.cancelToCloseCompleteMs >= 0);
});

test("stream writer waits for drain and exits on cancellation", async () => {
  const res = new EventEmitter();
  res.destroyed = false;
  res.writableEnded = false;
  res.write = () => false;
  const controller = new AbortController();
  const pending = writeWithBackpressure(res, "chunk", controller.signal);
  queueMicrotask(() => res.emit("drain"));
  assert.equal(await pending, true);

  const cancelled = new AbortController();
  const blocked = writeWithBackpressure(res, "chunk", cancelled.signal);
  cancelled.abort();
  assert.equal(await blocked, false);
  assert.equal(res.listenerCount("drain"), 0);
});

test("native RPC line reader accepts fragmented lines and fails at its byte ceiling", async () => {
  const stream = new PassThrough();
  const lines = [];
  let failure = null;
  attachBoundedLineReader(stream, {
    maxBytes: 8,
    onLine: (line) => lines.push(line),
    onError: (error) => { failure = error; },
  });
  stream.write("{\"a\":");
  stream.write("1}\n");
  stream.end("123456789\n");
  await new Promise((resolve) => stream.once("end", resolve));
  assert.deepEqual(lines, ['{"a":1}']);
  assert.equal(failure?.code, "NATIVE_WORKER_RPC_LINE_TOO_LARGE");
  assert.equal(MAX_NATIVE_RPC_LINE_BYTES, 16 * 1024 * 1024);
});
