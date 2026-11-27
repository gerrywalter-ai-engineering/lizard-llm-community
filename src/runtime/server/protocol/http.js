import crypto from "node:crypto";
import { ProtocolHttpError } from "./errors.js";

export const DEFAULT_MAX_JSON_BODY_BYTES = 1024 * 1024;
export const DEFAULT_MAX_JSON_DEPTH = 32;
export const DEFAULT_JSON_BODY_TIMEOUT_MS = 30_000;
const REQUEST_ID_PATTERN = /^[A-Za-z0-9._:/-]{1,128}$/;

export function resolveHttpRequestId(headers = {}) {
  const candidate = Array.isArray(headers["x-request-id"])
    ? headers["x-request-id"][0]
    : headers["x-request-id"];
  return typeof candidate === "string" && REQUEST_ID_PATTERN.test(candidate)
    ? candidate
    : crypto.randomUUID();
}

export function attachHttpRequestId(req, res) {
  const requestId = resolveHttpRequestId(req?.headers);
  res.setHeader("X-Request-ID", requestId);
  return requestId;
}

export function isJsonContentType(value) {
  const mediaType = String(value ?? "").split(";", 1)[0].trim().toLowerCase();
  return mediaType === "application/json" || mediaType.endsWith("+json");
}

export function assertJsonDepth(value, maxDepth = DEFAULT_MAX_JSON_DEPTH) {
  const pending = [{ value, depth: 1 }];
  while (pending.length) {
    const current = pending.pop();
    if (current.depth > maxDepth) {
      throw new ProtocolHttpError(400, "json_too_deep", `JSON nesting exceeds the ${maxDepth}-level limit.`);
    }
    if (!current.value || typeof current.value !== "object") continue;
    for (const nested of Array.isArray(current.value) ? current.value : Object.values(current.value)) {
      if (nested && typeof nested === "object") {
        pending.push({ value: nested, depth: current.depth + 1 });
      }
    }
  }
}

function drainRequest(req) {
  const discard = () => {};
  const done = () => {
    req.off("data", discard);
    req.off("end", done);
    req.off("error", done);
    req.off("aborted", done);
    req.off("close", done);
  };
  req.on("data", discard);
  req.once("end", done);
  req.once("error", done);
  req.once("aborted", done);
  req.once("close", done);
  req.resume();
}

export async function readJsonBody(req, {
  maxBytes = DEFAULT_MAX_JSON_BODY_BYTES,
  maxDepth = DEFAULT_MAX_JSON_DEPTH,
  timeoutMs = DEFAULT_JSON_BODY_TIMEOUT_MS,
  allowMissingContentType = false,
} = {}) {
  const contentType = req.headers?.["content-type"];
  if ((!contentType && !allowMissingContentType) || (contentType && !isJsonContentType(contentType))) {
    throw new ProtocolHttpError(
      400,
      "invalid_content_type",
      "This route requires Content-Type: application/json.",
      { param: "Content-Type" },
    );
  }
  const declaredLength = Number(req.headers?.["content-length"]);
  if (Number.isFinite(declaredLength) && declaredLength > maxBytes) {
    throw new ProtocolHttpError(413, "request_body_too_large", `JSON request body exceeds the ${maxBytes}-byte limit.`);
  }

  return new Promise((resolve, reject) => {
    const chunks = [];
    let bytes = 0;
    let settled = false;
    let timer = null;

    const cleanup = () => {
      if (timer) {
        clearTimeout(timer);
        timer = null;
      }
      req.off("data", onData);
      req.off("end", onEnd);
      req.off("error", onError);
      req.off("aborted", onAborted);
    };
    const fail = (error) => {
      if (settled) return;
      settled = true;
      cleanup();
      reject(error);
    };
    const onData = (chunk) => {
      bytes += chunk.length;
      if (bytes > maxBytes) {
        // Stop retaining bytes immediately. Resume in discard mode so Node can
        // send the 413 and close/reuse the connection without buffering the
        // remainder in userland.
        cleanup();
        drainRequest(req);
        fail(new ProtocolHttpError(413, "request_body_too_large", `JSON request body exceeds the ${maxBytes}-byte limit.`));
        return;
      }
      chunks.push(Buffer.from(chunk));
    };
    const onEnd = () => {
      if (settled) return;
      settled = true;
      cleanup();
      let decoded;
      try {
        decoded = new TextDecoder("utf-8", { fatal: true }).decode(Buffer.concat(chunks));
      } catch {
        reject(new ProtocolHttpError(400, "invalid_utf8", "JSON request body is not valid UTF-8."));
        return;
      }
      const raw = decoded.replace(/^\uFEFF/, "");
      let value;
      try {
        value = raw ? JSON.parse(raw) : {};
      } catch {
        reject(new ProtocolHttpError(400, "invalid_json", "JSON request body is malformed."));
        return;
      }
      if (!value || Array.isArray(value) || typeof value !== "object") {
        reject(new ProtocolHttpError(400, "invalid_json_type", "JSON request body must be an object."));
        return;
      }
      try {
        assertJsonDepth(value, maxDepth);
        resolve(value);
      } catch (error) {
        reject(error);
      }
    };
    const onError = (error) => fail(error);
    const onAborted = () => fail(new ProtocolHttpError(499, "client_cancelled", "Client disconnected before the request body completed."));

    req.on("data", onData);
    req.once("end", onEnd);
    req.once("error", onError);
    req.once("aborted", onAborted);
    if (Number.isFinite(timeoutMs) && timeoutMs > 0) {
      timer = setTimeout(() => {
        drainRequest(req);
        fail(new ProtocolHttpError(408, "request_body_timeout", "JSON request body was not received before the deadline."));
      }, timeoutMs);
      timer.unref?.();
    }
  });
}
