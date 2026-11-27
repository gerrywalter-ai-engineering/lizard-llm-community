import test from "node:test";
import assert from "node:assert/strict";
import {
  buildKvCachePlan,
  buildPrefixCacheKey,
  createKvCacheAllocator,
  createKvSlotManager,
} from "../src/runtime/kv-cache/index.js";

test("buildPrefixCacheKey is stable for the same input", () => {
  const first = buildPrefixCacheKey({
    modelId: "demo",
    renderedPromptHash: "abc",
    tokenizer: "llama",
    contextSize: 4096,
    templateSource: "gguf",
  });
  const second = buildPrefixCacheKey({
    modelId: "demo",
    renderedPromptHash: "abc",
    tokenizer: "llama",
    contextSize: 4096,
    templateSource: "gguf",
  });

  assert.equal(first, second);
});

test("KV cache allocator tracks reserve, resize, and release", () => {
  const allocator = createKvCacheAllocator({
    totalBudgetBytes: 100,
  });
  allocator.reserve({ slotId: "a", bytes: 30 });
  allocator.reserve({ slotId: "b", bytes: 20 });
  allocator.reserve({ slotId: "a", bytes: 40 });

  assert.equal(allocator.usedBytes(), 60);
  assert.equal(allocator.availableBytes(), 40);
  allocator.release("b");
  assert.equal(allocator.usedBytes(), 40);
});

test("KV slot manager reuses a completed slot by prefix key", () => {
  const allocator = createKvCacheAllocator({
    totalBudgetBytes: 200,
  });
  const slots = createKvSlotManager({
    allocator,
    maxSlots: 2,
    now: (() => {
      let current = 0;
      return () => ++current;
    })(),
  });

  const first = slots.assignSlot({
    prefixKey: "prefix-a",
    tokenCount: 10,
    bytes: 50,
  });
  slots.completeSlot(first.slotId);
  const second = slots.assignSlot({
    prefixKey: "prefix-a",
    tokenCount: 12,
    bytes: 60,
  });

  assert.equal(first.slotId, second.slotId);
  assert.equal(second.reused, true);
  assert.equal(allocator.usedBytes(), 60);
});

test("KV slot manager evicts least recently used inactive slots when budget is tight", () => {
  const allocator = createKvCacheAllocator({
    totalBudgetBytes: 100,
  });
  let current = 0;
  const slots = createKvSlotManager({
    allocator,
    maxSlots: 2,
    now: () => ++current,
  });

  const first = slots.assignSlot({
    prefixKey: "prefix-a",
    tokenCount: 10,
    bytes: 40,
  });
  slots.completeSlot(first.slotId);
  const second = slots.assignSlot({
    prefixKey: "prefix-b",
    tokenCount: 10,
    bytes: 40,
  });
  slots.completeSlot(second.slotId);
  const third = slots.assignSlot({
    prefixKey: "prefix-c",
    tokenCount: 10,
    bytes: 70,
  });

  const snapshot = slots.snapshot();
  assert.equal(snapshot.some((slot) => slot.prefixKey === "prefix-a"), false);
  assert.equal(snapshot.some((slot) => slot.prefixKey === "prefix-b"), false);
  assert.equal(third.prefixKey, "prefix-c");
});

test("KV slot manager shifts context and updates token count", () => {
  const allocator = createKvCacheAllocator({
    totalBudgetBytes: 200,
  });
  const slots = createKvSlotManager({
    allocator,
    maxSlots: 2,
  });
  const slot = slots.assignSlot({
    prefixKey: "prefix-a",
    tokenCount: 20,
    bytes: 80,
    metadata: {
      bytesPerToken: 4,
    },
  });

  const shifted = slots.shiftContext({
    slotId: slot.slotId,
    maxTokens: 12,
  });

  assert.equal(shifted.tokenCount, 12);
  assert.equal(shifted.bytes, 48);
});

test("buildKvCachePlan reports memory fit and warnings from GGUF-backed estimates", () => {
  const plan = buildKvCachePlan({
    modelId: "demo",
    profile: {
      modelId: "demo",
      contextSize: 4096,
      capabilityRegistry: {
        memory: {
          availableGb: 1,
        },
      },
      gguf: {
        blockCount: 32,
        embeddingLength: 4096,
        tokenizer: "llama",
      },
    },
    promptArtifact: {
      tokenizer: "llama",
      templateSource: "gguf",
      renderedPromptHash: "abc123",
      tokenCount: 4096,
    },
  });

  assert.equal(plan.modelId, "demo");
  assert.ok(plan.prefixKey);
  assert.ok(plan.estimatedBytes > 0);
  assert.equal(typeof plan.fits, "boolean");
});

test("slot planning does not inherit the model's advertised context", () => {
  // The cache runtime builds its plan with NO promptArtifact, so effectiveTokens
  // falls back to the model's maximum. On a 131k-context model that made one
  // hypothetical full-length sequence cost more than the whole budget, and
  // maxSlots collapsed to 1 -- so the second concurrent session was refused on
  // every machine regardless of what conversations actually cost.
  const profile = {
    capabilityRegistry: { memory: { availableGb: 16 } },
    nativeRuntime: {},
  };
  const gguf = {
    blockCount: 28,
    embeddingLength: 3072,
    contextLength: 131072,
    tokenizer: "llama",
    architecture: "llama",
    raw: {
      "llama.attention.head_count_kv": 8,
      "llama.attention.key_length": 128,
      "llama.attention.value_length": 128,
    },
  };

  const plan = buildKvCachePlan({ modelId: "wide-context", profile, gguf });
  assert.equal(plan.effectiveTokens, 131072, "the plan still describes the requested sequence");
  assert.ok(
    plan.maxSlots > 1,
    `slot planning must not collapse to 1 on a wide-context model, got ${plan.maxSlots}`,
  );
});

test("a modest budget admits concurrent sessions instead of refusing them", () => {
  // The refusal that kept EnhancementRun_04's 1.531x away from every user: the
  // count cap was budget/worst-case-slot, so a 1.2 GB budget allowed 2 sessions
  // and refused the third while 44 MB of 1229 MB was in use. The third
  // conversation was rejected for memory it had not asked for and, at 200
  // tokens a turn, never would. Bytes are the allocator's job; the count cap is
  // bookkeeping.
  const gguf = {
    blockCount: 28,
    embeddingLength: 3072,
    contextLength: 131072,
    tokenizer: "llama",
    architecture: "llama",
    raw: {
      "llama.attention.head_count_kv": 8,
      "llama.attention.key_length": 128,
      "llama.attention.value_length": 128,
    },
  };

  for (const availableGb of [1, 2, 4, 8]) {
    const profile = {
      contextSize: 4096,
      capabilityRegistry: { memory: { availableGb } },
      nativeRuntime: {},
    };
    const plan = buildKvCachePlan({ modelId: "concurrent", profile, gguf });
    const allocator = createKvCacheAllocator({ totalBudgetBytes: plan.totalBudgetBytes });
    const slots = createKvSlotManager({ allocator, maxSlots: plan.maxSlots });

    // Six short conversations, all mid-flight: none is completed, so none is
    // evictable. Every one of them must get a slot.
    for (let index = 0; index < 6; index += 1) {
      const turn = buildKvCachePlan({
        modelId: "concurrent",
        profile,
        gguf,
        promptArtifact: { tokenCount: 200, renderedPromptHash: `hash-${index}` },
      });
      assert.doesNotThrow(
        () => slots.assignSlot({
          prefixKey: `session-${index}`,
          tokenCount: turn.effectiveTokens,
          bytes: turn.estimatedBytes,
          metadata: { bytesPerToken: turn.bytesPerToken },
        }),
        `session ${index + 1} refused at ${availableGb} GB free (maxSlots=${plan.maxSlots})`,
      );
    }
    assert.equal(slots.snapshot().length, 6);
  }
});

test("a long conversation evicts an idle slot rather than failing mid-turn", () => {
  // The cost of admitting optimistically: more sessions can be live than could
  // all reach full context. Growth must therefore reclaim from idle slots, not
  // throw at the person still typing.
  const bytesPerToken = 114688;
  const allocator = createKvCacheAllocator({ totalBudgetBytes: 600 * 1024 * 1024 });
  const slots = createKvSlotManager({ allocator, maxSlots: 8 });

  const assigned = [];
  for (let index = 0; index < 4; index += 1) {
    assigned.push(slots.assignSlot({
      prefixKey: `short-${index}`,
      tokenCount: 200,
      bytes: 200 * bytesPerToken,
      metadata: { bytesPerToken },
    }).slotId);
  }
  slots.completeSlot(assigned[0]);
  slots.completeSlot(assigned[1]);

  assert.doesNotThrow(() => slots.assignSlot({
    prefixKey: "short-3",
    tokenCount: 4096,
    bytes: 4096 * bytesPerToken,
    metadata: { bytesPerToken },
  }), "growing to full context must evict an idle slot, not refuse the active turn");
});

test("grouped-query attention is not over-estimated by the GQA ratio", () => {
  // llama-3.2-3B has 24 attention heads but only 8 KV heads of 128, so K+V is
  // 8 * (128 + 128) = 2048 elements per layer -- not 2 * 3072. Assuming the
  // hidden size over-estimated the cache by exactly 3x, which is what capped
  // concurrency. 28 layers * 2048 * 2 bytes = 114,688 per token, and the
  // runner's own telemetry measures 112 KiB.
  const gguf = {
    blockCount: 28,
    embeddingLength: 3072,
    contextLength: 8192,
    architecture: "llama",
    raw: {
      "llama.attention.head_count_kv": 8,
      "llama.attention.key_length": 128,
      "llama.attention.value_length": 128,
    },
  };
  const plan = buildKvCachePlan({
    modelId: "gqa",
    profile: { capabilityRegistry: { memory: { availableGb: 16 } }, contextSize: 1 },
    gguf,
  });
  assert.equal(plan.bytesPerToken, 114688);

  // A model that does not state its KV heads keeps the old hidden-size
  // assumption: over-estimating wastes capacity, under-estimating would hand
  // out slots that do not fit.
  const legacy = buildKvCachePlan({
    modelId: "legacy",
    profile: { capabilityRegistry: { memory: { availableGb: 16 } }, contextSize: 1 },
    gguf: { blockCount: 28, embeddingLength: 3072, contextLength: 8192 },
  });
  assert.equal(legacy.bytesPerToken, 28 * 3072 * 2 * 2);
});
