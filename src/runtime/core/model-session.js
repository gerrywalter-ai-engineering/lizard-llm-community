import crypto from "node:crypto";

/**
 * @typedef {Object} GenerateRequest
 * @property {string} prompt
 * @property {number[] | null} [promptTokenIds]
 * @property {string | null} [systemPrompt]
 * @property {Array<{role: string, content: string}> | null} [messages]
 * @property {string[]} [stopSequences]
 * @property {number | null} [maxTokens]
 * @property {number | null} [temperature]
 * @property {number | null} [topK]
 * @property {number | null} [topP]
 * @property {number | null} [seed]
 * @property {string} [requestId]
 * @property {Record<string, unknown>} [metadata]
 */

/**
 * @typedef {Object} GenerateStreamEvent
 * @property {"start" | "delta" | "complete" | "cancelled" | "error"} type
 * @property {string} sessionId
 * @property {string} modelId
 * @property {string} [requestId]
 * @property {string} [text]
 * @property {string} [output]
 * @property {string} [message]
 * @property {number | null} [durationMs]
 * @property {string | null} [runtime]
 * @property {Record<string, unknown>} [metadata]
 */

function normalizeRequest(request = {}) {
  if (!request.prompt && !Array.isArray(request.promptTokenIds)) {
    throw new Error("GenerateRequest.prompt is required.");
  }
  return {
    requestId: request.requestId ?? crypto.randomUUID(),
    prompt: String(request.prompt),
    promptTokenIds: Array.isArray(request.promptTokenIds) ? [...request.promptTokenIds] : null,
    systemPrompt: request.systemPrompt ? String(request.systemPrompt) : null,
    messages: Array.isArray(request.messages) ? request.messages.map((message) => ({ ...message })) : null,
    stopSequences: Array.isArray(request.stopSequences) ? request.stopSequences.map(String) : [],
    maxTokens: request.maxTokens ?? null,
    temperature: request.temperature ?? null,
    topK: request.topK ?? null,
    topP: request.topP ?? null,
    seed: request.seed ?? null,
    metadata: request.metadata ?? {},
  };
}

export class ModelSession {
  constructor({
    sessionId = crypto.randomUUID(),
    modelId,
    engine = "existing-backend",
    onGenerate,
    onClose = async () => {},
  }) {
    if (!modelId) {
      throw new Error("ModelSession requires a modelId.");
    }
    if (typeof onGenerate !== "function") {
      throw new Error("ModelSession requires an onGenerate handler.");
    }
    this.sessionId = sessionId;
    this.modelId = modelId;
    this.engine = engine;
    this._onGenerate = onGenerate;
    this._onClose = onClose;
    this._closed = false;
    this._activeRequest = null;
  }

  get state() {
    return {
      sessionId: this.sessionId,
      modelId: this.modelId,
      engine: this.engine,
      closed: this._closed,
      activeRequestId: this._activeRequest?.requestId ?? null,
      activeRequestState: this._activeRequest?.state ?? null,
    };
  }

  cancelRequest(requestId = this._activeRequest?.requestId ?? null) {
    if (!this._activeRequest || !requestId || this._activeRequest.requestId !== requestId) {
      return false;
    }
    this._activeRequest.cancelled = true;
    this._activeRequest.state = "cancelled";
    return true;
  }

  async *generateStream(request) {
    if (this._closed) {
      throw new Error("ModelSession is closed.");
    }
    if (this._activeRequest) {
      throw new Error("ModelSession already has an active request.");
    }

    const normalized = normalizeRequest(request);
    const active = {
      requestId: normalized.requestId,
      cancelled: false,
      state: "running",
    };
    this._activeRequest = active;

    try {
      yield {
        type: "start",
        sessionId: this.sessionId,
        modelId: this.modelId,
        request: normalized,
      };

      if (active.cancelled) {
        yield {
          type: "cancelled",
          sessionId: this.sessionId,
          modelId: this.modelId,
          requestId: normalized.requestId,
        };
        return;
      }

      // Live token deltas: the generate handler may push per-token text while
      // the native worker is still decoding. Yield those immediately so SSE
      // consumers see output as it is produced instead of one burst at the end.
      const liveQueue = [];
      let wakeConsumer = null;
      const pushLiveToken = (value) => {
        const item = value && typeof value === "object"
          ? { text: String(value.text ?? ""), tokenId: value.tokenId ?? null }
          : { text: String(value ?? ""), tokenId: null };
        if (!item.text) {
          return;
        }
        liveQueue.push(item);
        if (wakeConsumer) {
          const wake = wakeConsumer;
          wakeConsumer = null;
          wake();
        }
      };

      let result = null;
      let generateError = null;
      let generateSettled = false;
      const generatePromise = this._onGenerate({
        request: normalized,
        session: this,
        onToken: pushLiveToken,
      }).then(
        (value) => {
          result = value;
        },
        (error) => {
          generateError = error;
        },
      ).finally(() => {
        generateSettled = true;
        if (wakeConsumer) {
          const wake = wakeConsumer;
          wakeConsumer = null;
          wake();
        }
      });

      let liveDeltasStreamed = 0;
      while (true) {
        while (liveQueue.length) {
          if (active.cancelled) {
            yield {
              type: "cancelled",
              sessionId: this.sessionId,
              modelId: this.modelId,
              requestId: normalized.requestId,
            };
            return;
          }
          liveDeltasStreamed += 1;
          const live = liveQueue.shift();
          yield {
            type: "delta",
            sessionId: this.sessionId,
            modelId: this.modelId,
            requestId: normalized.requestId,
            text: live.text,
            tokenId: Number.isSafeInteger(Number(live.tokenId)) ? Number(live.tokenId) : null,
          };
        }
        if (generateSettled) {
          break;
        }
        await new Promise((resolve) => {
          wakeConsumer = resolve;
        });
      }
      await generatePromise;
      if (generateError) {
        throw generateError;
      }

      if (active.cancelled) {
        yield {
          type: "cancelled",
          sessionId: this.sessionId,
          modelId: this.modelId,
          requestId: normalized.requestId,
        };
        return;
      }

      const output = String(result?.output ?? "");
      // When no live deltas arrived (handler without token events, oracle
      // fallback, non-streaming worker), keep the buffered chunk replay.
      const chunks = liveDeltasStreamed > 0
        ? []
        : (Array.isArray(result?.chunks) && result.chunks.length
          ? result.chunks.map((chunk) => String(chunk))
          : output
            ? [output]
            : []);
      for (const chunk of chunks) {
        if (active.cancelled) {
          yield {
            type: "cancelled",
            sessionId: this.sessionId,
            modelId: this.modelId,
            requestId: normalized.requestId,
          };
          return;
        }
        yield {
          type: "delta",
          sessionId: this.sessionId,
          modelId: this.modelId,
          requestId: normalized.requestId,
          text: chunk,
        };
      }

      active.state = "completed";
      yield {
        type: "complete",
        sessionId: this.sessionId,
        modelId: this.modelId,
        requestId: normalized.requestId,
        output,
        durationMs: result?.durationMs ?? null,
        runtime: result?.runtime ?? null,
        metadata: result?.metadata ?? {},
      };
    } catch (error) {
      active.state = "failed";
      yield {
        type: "error",
        sessionId: this.sessionId,
        modelId: this.modelId,
        requestId: normalized.requestId,
        message: error.message,
      };
      throw error;
    } finally {
      this._activeRequest = null;
    }
  }

  async close() {
    if (this._closed) {
      return;
    }
    this._closed = true;
    if (this._activeRequest) {
      this.cancelRequest(this._activeRequest.requestId);
    }
    await this._onClose(this);
  }
}
