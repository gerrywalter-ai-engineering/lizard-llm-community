export class ProtocolHttpError extends Error {
  constructor(statusCode, code, message, { type = "invalid_request_error", param = null } = {}) {
    super(message);
    this.name = "ProtocolHttpError";
    this.statusCode = statusCode;
    this.code = code;
    this.type = type;
    this.param = param;
  }
}

export function protocolErrorPayload(error, extensions = {}) {
  return {
    error: {
      message: error?.message ?? String(error),
      type: error?.type ?? "server_error",
      param: error?.param ?? null,
      code: error?.code ?? "internal_error",
      ...extensions,
    },
  };
}

const UNPROCESSABLE_CODES = new Set([
  "NATIVE_TOKENIZER_UNAVAILABLE",
  "NATIVE_MODEL_UNSUPPORTED",
  "NATIVE_EXECUTOR_UNIMPLEMENTED",
  "KV_BUDGET_EXCEEDED",
  "CHAT_TEMPLATE_UNSUPPORTED",
]);

export function classifyProtocolHttpError(error) {
  if (error instanceof ProtocolHttpError) {
    return {
      statusCode: error.statusCode,
      type: error.type,
      code: error.code,
      param: error.param,
    };
  }
  if (error instanceof SyntaxError) {
    return { statusCode: 400, type: "invalid_request_error", code: "invalid_json", param: null };
  }
  if (UNPROCESSABLE_CODES.has(error?.code)) {
    return { statusCode: 422, type: "unprocessable_entity", code: error.code, param: null };
  }
  if (error?.code === "SCHEDULER_TIMEOUT" || error?.code === "NATIVE_WORKER_TIMEOUT") {
    return { statusCode: 504, type: "timeout_error", code: error.code, param: null };
  }
  if (error?.code === "SCHEDULER_CANCELLED") {
    return { statusCode: 499, type: "invalid_request_error", code: error.code, param: null };
  }
  return {
    statusCode: 500,
    type: "server_error",
    code: error?.code ?? "internal_error",
    param: null,
  };
}
