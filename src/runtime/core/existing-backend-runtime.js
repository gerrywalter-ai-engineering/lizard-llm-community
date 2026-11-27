import { generateWithExistingBackend } from "../../lib/generate.js";
import { snapshotLifecycleState } from "../../lib/model-lifecycle.js";
import { buildKvCachePlan, createKvCacheRuntime } from "../kv-cache/index.js";
import { createRequestScheduler } from "../scheduler/index.js";
import { buildPromptArtifact } from "../tokenizer/index.js";
import { ModelSession } from "./model-session.js";

/**
 * @typedef {Object} GenerateRequest
 * @property {string} prompt
 * @property {string | null} [systemPrompt]
 * @property {number | null} [maxTokens]
 * @property {number | null} [temperature]
 * @property {string} [requestId]
 * @property {Record<string, unknown>} [metadata]
 */

export class InferenceRuntime {
  constructor({ engine = "llama.cpp" } = {}) {
    this.engine = engine;
  }

  async createSession() {
    throw new Error("InferenceRuntime.createSession must be implemented by a concrete runtime.");
  }
}

export class ExistingBackendRuntime extends InferenceRuntime {
  constructor({
    engine = "llama.cpp",
    generate = generateWithExistingBackend,
    scheduler = createRequestScheduler(),
  } = {}) {
    super({ engine });
    this._generate = generate;
    this._kvCaches = new Map();
    this._scheduler = scheduler;
  }

  kvCacheForModel({ modelId, profile = null, gguf = null }) {
    const cacheKey = String(modelId);
    if (!this._kvCaches.has(cacheKey)) {
      this._kvCaches.set(cacheKey, createKvCacheRuntime({
        modelId,
        profile,
        gguf,
      }));
    }
    return this._kvCaches.get(cacheKey);
  }

  async createSession({
    recommendation,
    modelArtifact,
    profile = null,
    config,
    logger,
    enginePath,
  }) {
    if (!modelArtifact?.modelId) {
      throw new Error("ExistingBackendRuntime requires a resolved model artifact.");
    }
    let activeKvSlotId = null;
    let activeRequestId = null;
    return new ModelSession({
      modelId: modelArtifact.modelId,
      engine: this.engine,
      onGenerate: async ({ request }) => {
        activeRequestId = request.requestId;
        const promptArtifact = buildPromptArtifact({
          prompt: request.prompt,
          systemPrompt: request.systemPrompt,
          messages: request.messages,
          modelPath: modelArtifact.modelPath,
          gguf: profile?.gguf ?? modelArtifact?.gguf ?? null,
        });
        const kvPlan = buildKvCachePlan({
          modelId: modelArtifact.modelId,
          profile,
          promptArtifact,
          gguf: profile?.gguf ?? modelArtifact?.gguf ?? null,
        });
        const kvCache = this.kvCacheForModel({
          modelId: modelArtifact.modelId,
          profile,
          gguf: profile?.gguf ?? modelArtifact?.gguf ?? null,
        });
        const kvSlot = kvCache.slots.assignSlot({
          prefixKey: kvPlan.prefixKey ?? promptArtifact.renderedPromptHash,
          tokenCount: kvPlan.effectiveTokens ?? promptArtifact.tokenCount ?? 0,
          bytes: kvPlan.estimatedBytes ?? 0,
          metadata: {
            bytesPerToken: kvPlan.bytesPerToken,
            promptHash: promptArtifact.renderedPromptHash,
            templateMode: promptArtifact.templateMode,
          },
        });
        activeKvSlotId = kvSlot.slotId;
        const priority = Number.isFinite(Number(request.metadata?.priority))
          ? Number(request.metadata.priority)
          : 0;
        const timeoutMs = Number.isFinite(Number(request.metadata?.timeoutMs))
          ? Number(request.metadata.timeoutMs)
          : profile?.timeoutMs ?? null;
        const chunkSize = Number.isFinite(Number(request.metadata?.chunkSize))
          ? Number(request.metadata.chunkSize)
          : 24;
        const chunkDelayMs = Number.isFinite(Number(request.metadata?.chunkDelayMs))
          ? Number(request.metadata.chunkDelayMs)
          : 0;
        try {
          const result = await this._scheduler.enqueue({
            requestId: request.requestId,
            priority,
            timeoutMs,
            chunkSize,
            chunkDelayMs,
            task: async ({ isCancelled }) => {
              if (isCancelled()) {
                const error = new Error(`Scheduler request ${request.requestId} was cancelled before execution.`);
                error.code = "SCHEDULER_CANCELLED";
                throw error;
              }
              return this._generate({
                recommendation,
                modelArtifact,
                profile,
                prompt: request.prompt,
                systemPrompt: request.systemPrompt,
                maxTokens: request.maxTokens,
                enginePath,
                config,
                logger,
              });
            },
          });
          kvCache.slots.completeSlot(kvSlot.slotId);
          activeKvSlotId = null;
          return {
            output: result.output,
            durationMs: result.durationMs,
            runtime: result.runtime,
            chunks: result.chunks,
            metadata: {
              lifecycleState: snapshotLifecycleState(config),
              status: result.status,
              promptArtifact,
              kvCache: {
                ...kvPlan,
                slotId: kvSlot.slotId,
                reused: kvSlot.reused,
                activeSlots: kvCache.slots.snapshot(),
                simulated: true,
              },
              scheduler: result.scheduler,
            },
          };
        } catch (error) {
          kvCache.slots.completeSlot(kvSlot.slotId);
          activeKvSlotId = null;
          throw error;
        }
      },
      onClose: async () => {
        if (activeRequestId) {
          this._scheduler.cancel(activeRequestId);
        }
        if (!activeKvSlotId) {
          return;
        }
        const kvCache = this._kvCaches.get(String(modelArtifact.modelId));
        kvCache?.slots.completeSlot(activeKvSlotId);
      },
    });
  }
}
