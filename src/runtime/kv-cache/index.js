import crypto from "node:crypto";

const GIB = 1024 * 1024 * 1024;

function stableHash(value) {
  return crypto.createHash("sha256").update(JSON.stringify(value ?? null)).digest("hex");
}

function nowMs() {
  return Date.now();
}

export function buildPrefixCacheKey({
  modelId,
  renderedPromptHash,
  tokenizer = "unknown",
  contextSize = null,
  templateSource = null,
}) {
  return stableHash({
    modelId: modelId ?? null,
    renderedPromptHash: renderedPromptHash ?? null,
    tokenizer,
    contextSize,
    templateSource,
  });
}

// Width of ONE position's K+V, in elements, per layer.
//
// Grouped-query attention makes this much narrower than the hidden size: on
// llama-3.2-3B there are 24 attention heads but only 8 KV heads of 128, so K+V
// is 8 x (128 + 128) = 2048 elements, not 2 x 3072 = 6144. Assuming the hidden
// size therefore over-estimates the cache by exactly 3x on that model, and by
// the GQA ratio in general -- which is how a machine with room for six
// concurrent sequences came to refuse the second one.
//
// Returns null when the metadata is absent, so the caller keeps the old
// hidden-size behaviour rather than guessing a narrower cache than the model
// really needs. Over-estimating wastes capacity; under-estimating hands out
// slots that do not fit.
export function resolveKvWidthElements(gguf) {
  const raw = gguf?.raw ?? null;
  if (!raw) return null;
  const architecture = gguf?.architecture ?? null;
  const lookup = (suffix) => {
    if (architecture && Number.isFinite(raw[`${architecture}.attention.${suffix}`])) {
      return raw[`${architecture}.attention.${suffix}`];
    }
    // Architecture prefixes vary by exporter, so fall back to any key that ends
    // with the attention suffix rather than maintaining an allow-list.
    const key = Object.keys(raw).find((name) => name.endsWith(`.attention.${suffix}`));
    return key && Number.isFinite(raw[key]) ? raw[key] : null;
  };
  const kvHeads = lookup("head_count_kv");
  const keyLength = lookup("key_length");
  const valueLength = lookup("value_length") ?? keyLength;
  if (!Number.isFinite(kvHeads) || kvHeads <= 0) return null;
  if (!Number.isFinite(keyLength) || keyLength <= 0) return null;
  return kvHeads * (keyLength + valueLength);
}

export function estimateKvCacheBytes({
  layers = null,
  hiddenSize = null,
  tokenCount = null,
  bytesPerElement = 2,
  kvWidthElements = null,
} = {}) {
  if (![layers, tokenCount].every((value) => Number.isFinite(value) && value > 0)) {
    return null;
  }
  // The GQA-accurate width when the model told us, otherwise the historical
  // hidden-size assumption (K and V each hiddenSize wide).
  if (Number.isFinite(kvWidthElements) && kvWidthElements > 0) {
    return Math.trunc(layers * kvWidthElements * tokenCount * bytesPerElement);
  }
  if (!Number.isFinite(hiddenSize) || hiddenSize <= 0) {
    return null;
  }
  return Math.trunc(layers * hiddenSize * tokenCount * bytesPerElement * 2);
}

export function createKvCacheAllocator({
  totalBudgetBytes,
  allocations = new Map(),
} = {}) {
  if (!Number.isFinite(totalBudgetBytes) || totalBudgetBytes <= 0) {
    throw new Error("KV cache allocator requires a positive total budget.");
  }

  function usedBytes() {
    return [...allocations.values()].reduce((sum, value) => sum + value, 0);
  }

  function availableBytes() {
    return totalBudgetBytes - usedBytes();
  }

  function canFit(bytesDelta) {
    return bytesDelta <= availableBytes();
  }

  function reserve({ slotId, bytes }) {
    if (!slotId) {
      throw new Error("KV cache allocator reserve requires a slotId.");
    }
    if (!Number.isFinite(bytes) || bytes < 0) {
      throw new Error("KV cache allocator reserve requires non-negative bytes.");
    }
    const previous = allocations.get(slotId) ?? 0;
    const delta = bytes - previous;
    if (delta > 0 && !canFit(delta)) {
      throw new Error("KV cache budget exceeded.");
    }
    allocations.set(slotId, bytes);
    return allocations.get(slotId);
  }

  function release(slotId) {
    allocations.delete(slotId);
  }

  return {
    totalBudgetBytes,
    allocations,
    usedBytes,
    availableBytes,
    canFit,
    reserve,
    release,
  };
}

export function createKvSlotManager({
  allocator,
  maxSlots = 4,
  now = nowMs,
} = {}) {
  if (!allocator) {
    throw new Error("KV slot manager requires an allocator.");
  }
  const slots = new Map();
  let sequence = 0;

  function listSlots() {
    return [...slots.values()].sort((left, right) => left.createdAt - right.createdAt);
  }

  function inactiveSlots() {
    return listSlots().filter((slot) => !slot.active);
  }

  function snapshot() {
    return listSlots().map((slot) => ({ ...slot }));
  }

  function getSlotByPrefix(prefixKey) {
    return listSlots().find((slot) => slot.prefixKey === prefixKey) ?? null;
  }

  function evictLeastRecentlyUsed() {
    const candidate = inactiveSlots()
      .sort((left, right) => left.lastUsedAt - right.lastUsedAt)[0] ?? null;
    if (!candidate) {
      return null;
    }
    allocator.release(candidate.slotId);
    slots.delete(candidate.slotId);
    return candidate;
  }

  function ensureCapacity({ slotCountDelta = 0, bytesDelta = 0 }) {
    while ((slots.size + slotCountDelta > maxSlots) || !allocator.canFit(bytesDelta)) {
      const evicted = evictLeastRecentlyUsed();
      if (!evicted) {
        throw new Error("KV cache has no evictable inactive slots and cannot satisfy the request.");
      }
    }
  }

  function assignSlot({
    prefixKey,
    tokenCount,
    bytes,
    metadata = {},
  }) {
    if (!prefixKey) {
      throw new Error("KV slot assignment requires a prefixKey.");
    }
    if (!Number.isFinite(bytes) || bytes < 0) {
      throw new Error("KV slot assignment requires non-negative bytes.");
    }

    const timestamp = now();
    const existing = getSlotByPrefix(prefixKey);
    if (existing) {
      const bytesDelta = Math.max(0, bytes - existing.bytes);
      ensureCapacity({ slotCountDelta: 0, bytesDelta });
      allocator.reserve({
        slotId: existing.slotId,
        bytes: Math.max(existing.bytes, bytes),
      });
      const updated = {
        ...existing,
        tokenCount: Math.max(existing.tokenCount ?? 0, tokenCount ?? 0),
        bytes: Math.max(existing.bytes, bytes),
        active: true,
        reused: true,
        metadata: {
          ...existing.metadata,
          ...metadata,
        },
        lastUsedAt: timestamp,
      };
      slots.set(existing.slotId, updated);
      return { ...updated };
    }

    ensureCapacity({
      slotCountDelta: 1,
      bytesDelta: bytes,
    });

    const slotId = `kv-slot-${++sequence}`;
    allocator.reserve({ slotId, bytes });
    const slot = {
      slotId,
      prefixKey,
      tokenCount: tokenCount ?? 0,
      bytes,
      active: true,
      reused: false,
      metadata: { ...metadata },
      createdAt: timestamp,
      lastUsedAt: timestamp,
    };
    slots.set(slotId, slot);
    return { ...slot };
  }

  function completeSlot(slotId) {
    const slot = slots.get(slotId);
    if (!slot) {
      return null;
    }
    const updated = {
      ...slot,
      active: false,
      lastUsedAt: now(),
    };
    slots.set(slotId, updated);
    return { ...updated };
  }

  function releaseSlot(slotId) {
    if (!slots.has(slotId)) {
      return false;
    }
    allocator.release(slotId);
    slots.delete(slotId);
    return true;
  }

  function shiftContext({ slotId, maxTokens }) {
    const slot = slots.get(slotId);
    if (!slot) {
      return null;
    }
    if (!Number.isFinite(maxTokens) || maxTokens <= 0 || slot.tokenCount <= maxTokens) {
      return { ...slot };
    }
    const bytesPerToken = slot.metadata?.bytesPerToken ?? null;
    const nextBytes = Number.isFinite(bytesPerToken)
      ? Math.trunc(maxTokens * bytesPerToken)
      : slot.bytes;
    allocator.reserve({
      slotId,
      bytes: nextBytes,
    });
    const updated = {
      ...slot,
      tokenCount: maxTokens,
      bytes: nextBytes,
      metadata: {
        ...slot.metadata,
        shifted: true,
      },
      lastUsedAt: now(),
    };
    slots.set(slotId, updated);
    return { ...updated };
  }

  return {
    allocator,
    maxSlots,
    snapshot,
    getSlotByPrefix,
    assignSlot,
    completeSlot,
    releaseSlot,
    shiftContext,
  };
}

// Share of *available* system memory the KV cache may occupy by default.
//
// This was 0.2, which is where "KV cache requires 3523215360 bytes but only
// 2877628088 are budgeted" came from: on a machine with 13 GB free, a 3B model
// at ~344 KB/token could not plan a 10k context even though the weights were
// already resident and answering. The KV cache is the working set of the
// conversation - if the machine has the memory free, refusing to use it just
// turns a slow answer into no answer. Callers that must stay small (benchmark
// lanes pinning a footprint) set nativeRuntime.kvCacheBudgetShare or an
// explicit nativeRuntime.kvCacheBudgetBytes.
const DEFAULT_KV_BUDGET_SHARE = 0.6;
const MIN_KV_BUDGET_BYTES = 512 * 1024 * 1024;

export function resolveKvBudgetBytes(profile = null) {
  const explicit = profile?.nativeRuntime?.kvCacheBudgetBytes;
  if (Number.isFinite(explicit) && explicit > 0) {
    return Math.trunc(explicit);
  }
  const availableGb = profile?.capabilityRegistry?.memory?.availableGb ?? null;
  if (!Number.isFinite(availableGb) || availableGb <= 0) {
    return 256 * 1024 * 1024;
  }
  const shareOverride = profile?.nativeRuntime?.kvCacheBudgetShare;
  const share = Number.isFinite(shareOverride) && shareOverride > 0 && shareOverride <= 1
    ? shareOverride
    : DEFAULT_KV_BUDGET_SHARE;
  const availableBytes = availableGb * GIB;
  // The floor must never exceed what is actually free, or the budget promises
  // memory the machine does not have.
  return Math.trunc(Math.min(availableBytes, Math.max(availableBytes * share, Math.min(MIN_KV_BUDGET_BYTES, availableBytes))));
}

export function buildKvCachePlan({
  modelId,
  profile = null,
  promptArtifact = null,
  gguf = null,
} = {}) {
  const resolvedGguf = gguf ?? profile?.gguf ?? null;
  const layers = resolvedGguf?.blockCount ?? null;
  const hiddenSize = resolvedGguf?.embeddingLength ?? null;
  const promptTokens = promptArtifact?.tokenCount ?? null;
  const contextTokens = profile?.contextSize ?? resolvedGguf?.contextLength ?? null;
  const effectiveTokens = Math.max(1, Math.min(
    Number.isFinite(contextTokens) ? contextTokens : promptTokens ?? 1,
    Number.isFinite(promptTokens) ? promptTokens : contextTokens ?? 1,
  ));
  const kvWidthElements = resolveKvWidthElements(resolvedGguf);
  const bytesPerToken = estimateKvCacheBytes({
    layers,
    hiddenSize,
    kvWidthElements,
    tokenCount: 1,
  });
  const estimatedBytes = estimateKvCacheBytes({
    layers,
    hiddenSize,
    kvWidthElements,
    tokenCount: effectiveTokens,
  });
  const totalBudgetBytes = resolveKvBudgetBytes(profile);
  const fits = estimatedBytes == null ? null : estimatedBytes <= totalBudgetBytes;
  // How long a context this machine can actually cache for this model. Callers
  // that can choose their context size (the chat workspace does, per turn) use
  // this to ask for what they need instead of inheriting the model's maximum
  // and failing the fit check on a conversation of three messages.
  const maxTokensForBudget = Number.isFinite(bytesPerToken) && bytesPerToken > 0
    ? Math.max(1, Math.floor(totalBudgetBytes / bytesPerToken))
    : null;
  const warnings = [];
  if (estimatedBytes == null) {
    warnings.push("KV cache size is estimated conservatively because GGUF hidden size or block count is missing.");
  } else if (!fits) {
    warnings.push("Estimated KV cache footprint exceeds the configured cache budget.");
  }

  const prefixKey = buildPrefixCacheKey({
    modelId,
    renderedPromptHash: promptArtifact?.renderedPromptHash ?? null,
    tokenizer: promptArtifact?.tokenizer ?? resolvedGguf?.tokenizer ?? "unknown",
    contextSize: contextTokens,
    templateSource: promptArtifact?.templateSource ?? null,
  });

  return {
    prefixKey,
    modelId: modelId ?? profile?.modelId ?? null,
    promptTokens,
    contextTokens,
    effectiveTokens,
    layers,
    hiddenSize,
    bytesPerToken,
    estimatedBytes,
    totalBudgetBytes,
    maxTokensForBudget,
    fits,
    warnings,
    maxSlots: resolveMaxSlots({
      profile,
      contextTokens,
      layers,
      hiddenSize,
      kvWidthElements,
      totalBudgetBytes,
    }),
    // How many of those admitted sessions could reach the planning length
    // without evicting anyone. Below maxSlots by design; see resolveMaxSlots.
    guaranteedSlots: resolveGuaranteedSlots({
      profile,
      contextTokens,
      layers,
      hiddenSize,
      kvWidthElements,
      totalBudgetBytes,
    }),
  };
}

// How many sequences may be cached at once.
//
// This used to divide the budget by `estimatedBytes`, which is sized for the
// sequence being planned. The cache RUNTIME builds its plan with no prompt at
// all, so `effectiveTokens` fell back to the model's advertised maximum --
// 131,072 on llama-3.2 -- and one hypothetical full-context sequence was
// estimated at 15 GB against an 8.6 GB budget. maxSlots therefore collapsed to
// 1 and the second concurrent session was refused on every machine, whatever
// the conversations actually cost.
//
// A model's advertised context is a capability, not a prediction of use. The
// same file already says so about `maxTokensForBudget`: callers "ask for what
// they need instead of inheriting the model's maximum and failing the fit check
// on a conversation of three messages". Slot planning now follows that rule.
//
// `estimatedBytes` and `fits` are deliberately left alone: they answer "does
// THIS sequence fit", which is a different question and is still asked per
// request with the real prompt.
//
// Second correction (EnhancementRun_04 wiring): dividing the budget by a
// worst-case slot still encoded BYTES into a COUNT cap, and the allocator
// already enforces bytes exactly. On a 1.2 GB budget that produced maxSlots=2
// and refused the third session while 44 MB of 1229 MB was in use -- the third
// conversation was rejected for memory it had not asked for and usually never
// would. The count cap is now bookkeeping only; `guaranteedSlots` keeps the
// worst-case answer for anyone who needs it.
//
// The trade this makes, stated plainly: more sessions can be admitted than
// could all simultaneously grow to the planning length. Growth is checked on
// every assignSlot, and `ensureCapacity` evicts idle slots before refusing, so
// the failure mode moves from "refuse a short conversation up front" to "evict
// an idle one when a long conversation actually needs the room". That is the
// better trade for a chat workload, where most turns never approach the cap.
const DEFAULT_SLOT_PLANNING_TOKENS = 4096;
const MAX_PLANNED_SLOTS = 16;

// How many sequences can reach the full planning length with no eviction at
// all. Reported for observability and capacity decisions; it is deliberately
// NOT the admission cap.
export function resolveGuaranteedSlots({
  profile = null,
  contextTokens = null,
  layers = null,
  hiddenSize = null,
  kvWidthElements = null,
  totalBudgetBytes = null,
} = {}) {
  const planningTokens = resolveSlotPlanningTokens({ profile, contextTokens });
  const perSlotBytes = estimateKvCacheBytes({
    layers,
    hiddenSize,
    kvWidthElements,
    tokenCount: planningTokens,
  });
  if (!Number.isFinite(perSlotBytes) || perSlotBytes <= 0
    || !Number.isFinite(totalBudgetBytes) || totalBudgetBytes <= 0) {
    return null;
  }
  return Math.max(0, Math.floor(totalBudgetBytes / perSlotBytes));
}

function resolveSlotPlanningTokens({ profile = null, contextTokens = null } = {}) {
  const explicit = profile?.nativeRuntime?.slotPlanningTokens;
  if (Number.isFinite(explicit) && explicit > 0) return explicit;
  if (Number.isFinite(profile?.contextSize) && profile.contextSize > 0) return profile.contextSize;
  return Math.min(
    Number.isFinite(contextTokens) && contextTokens > 0 ? contextTokens : DEFAULT_SLOT_PLANNING_TOKENS,
    DEFAULT_SLOT_PLANNING_TOKENS,
  );
}

export function resolveMaxSlots({
  profile = null,
  contextTokens = null,
  layers = null,
  hiddenSize = null,
  kvWidthElements = null,
  totalBudgetBytes = null,
} = {}) {
  const explicit = profile?.nativeRuntime?.maxConcurrentSlots;
  if (Number.isFinite(explicit) && explicit > 0) {
    return Math.max(1, Math.min(MAX_PLANNED_SLOTS, Math.trunc(explicit)));
  }

  const perSlotBytes = estimateKvCacheBytes({
    layers,
    hiddenSize,
    kvWidthElements,
    tokenCount: resolveSlotPlanningTokens({ profile, contextTokens }),
  });
  if (!Number.isFinite(perSlotBytes) || perSlotBytes <= 0
    || !Number.isFinite(totalBudgetBytes) || totalBudgetBytes <= 0) {
    return 4;
  }

  // A budget that cannot hold a single planning-length sequence is genuinely
  // small, and admitting eight sessions into it would thrash rather than batch.
  // Above that, the allocator's byte accounting is the real constraint, so the
  // count stops shadowing it.
  if (totalBudgetBytes < perSlotBytes) {
    return Math.max(1, Math.min(
      MAX_PLANNED_SLOTS,
      Math.floor(totalBudgetBytes / Math.max(1, Math.trunc(perSlotBytes / MAX_PLANNED_SLOTS))) || 1,
    ));
  }
  return MAX_PLANNED_SLOTS;
}

export function createKvCacheRuntime({
  modelId,
  profile = null,
  promptArtifact = null,
  gguf = null,
} = {}) {
  const plan = buildKvCachePlan({
    modelId,
    profile,
    promptArtifact,
    gguf,
  });
  const allocator = createKvCacheAllocator({
    totalBudgetBytes: plan.totalBudgetBytes,
  });
  const slots = createKvSlotManager({
    allocator,
    maxSlots: plan.maxSlots,
  });
  return {
    plan,
    allocator,
    slots,
  };
}
