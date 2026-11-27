// Sizing a chat turn to the hardware, instead of to the model's maximum.
//
// The KV cache is planned per request from `profile.contextSize` (see
// buildKvCachePlan in src/runtime/kv-cache/index.js), and its cost is linear
// in that number: layers x hidden x 2 (K and V) x 2 bytes, per token. For a
// 3B model that is ~344 KB per token, so a 10k-token context reserves ~3.3 GB
// whether the conversation is ten tokens long or ten thousand.
//
// That is what made "analyze this image" fail with
//   KV cache requires 3523215360 bytes but only 2877628088 are budgeted.
// The turn did not need a 10k context. It inherited one.
//
// So each turn is planned here: work out how many tokens the machine can
// actually afford, ask for only what this turn needs, and when the inputs
// still do not fit, drop the least valuable ones and say which. A turn should
// degrade, not fail.

// Conservative on purpose. Under-counting tokens is what overruns a budget, so
// the estimate leans low on characters-per-token (English prose is ~4; code,
// JSON and non-Latin text are denser).
const CHARS_PER_TOKEN = 3.4;

// Context is requested in coarse steps because contextSize is part of the
// prefix-cache key (buildPrefixCacheKey) - a bespoke size per turn would miss
// the cache on every message.
const CONTEXT_BUCKET = 1024;
const MIN_CONTEXT = 1024;

// Headroom for the chat template, BOS/EOS and tokenizer drift.
const SAFETY_TOKENS = 160;

// Below this an answer is not worth generating; if the budget cannot cover it
// the turn is refused with an explanation rather than silently truncated.
const MIN_ANSWER_TOKENS = 128;

export function estimateTokens(text) {
  const length = String(text ?? "").length;
  if (length === 0) {
    return 0;
  }
  return Math.ceil(length / CHARS_PER_TOKEN);
}

export function estimateTurnTokens(sections) {
  return [
    sections.systemPrompt,
    sections.toolInstructions,
    sections.contextBlock,
    sections.question,
  ].reduce((total, part) => total + estimateTokens(part), 0);
}

/**
 * Tokens per KV-cache token for this model, from its GGUF geometry.
 * Returns null when the metadata is missing, in which case the caller keeps
 * whatever context it already had rather than guessing.
 */
export function kvBytesPerToken(gguf) {
  const layers = gguf?.blockCount ?? null;
  const hidden = gguf?.embeddingLength ?? null;
  if (!Number.isFinite(layers) || !Number.isFinite(hidden) || layers <= 0 || hidden <= 0) {
    return null;
  }
  // K and V, two bytes per element - matches estimateKvCacheBytes' defaults.
  return layers * hidden * 2 * 2;
}

function roundUpToBucket(tokens) {
  return Math.max(MIN_CONTEXT, Math.ceil(tokens / CONTEXT_BUCKET) * CONTEXT_BUCKET);
}

/**
 * Plans one turn against a token ceiling.
 *
 * Trimming order is deliberate, least valuable first:
 *   1. history      - the model can be told what was dropped
 *   2. retrieved chunks, lowest-ranked first - BM25 already ordered them
 *   3. attachment bodies, truncated with a marker
 * The system prompt, tool instructions and the question itself are never
 * dropped: without them the answer is not the answer to what was asked.
 *
 * Returns the trimmed inputs, the context size to request, and notes naming
 * what was left out so the UI can tell the user rather than quietly lying.
 */
export function planTurnBudget({
  systemPrompt = "",
  toolInstructions = "",
  question = "",
  history = [],
  contextChunks = [],
  attachments = [],
  answerTokens = 512,
  modelContextTokens = 4096,
  bytesPerToken = null,
  budgetBytes = null,
  // Floor for this conversation's context size, so it can grow but never
  // shrink. See the comment at the contextSize computation below: a shrinking
  // context changes the KV shape between consecutive turns, which is what made
  // turn 2 re-prefill the whole transcript instead of just the new tail.
  minContextSize = 0,
} = {}) {
  const notes = [];
  const hardCeiling = Number.isFinite(modelContextTokens) && modelContextTokens > 0
    ? modelContextTokens
    : 4096;

  // What the machine can afford. Without GGUF geometry or a budget we cannot
  // compute it, so fall back to the model's own context and change nothing.
  const affordable = Number.isFinite(bytesPerToken) && bytesPerToken > 0
    && Number.isFinite(budgetBytes) && budgetBytes > 0
    ? Math.floor(budgetBytes / bytesPerToken)
    : hardCeiling;
  const ceiling = Math.max(MIN_CONTEXT, Math.min(hardCeiling, affordable));

  let answer = Math.min(answerTokens, Math.max(MIN_ANSWER_TOKENS, ceiling - SAFETY_TOKENS - 64));
  if (answer < answerTokens) {
    notes.push(`Answer budget reduced from ${answerTokens} to ${answer} tokens to fit this machine's cache budget.`);
  }

  const fixedTokens = estimateTokens(systemPrompt) + estimateTokens(toolInstructions) + estimateTokens(question);
  let available = ceiling - answer - SAFETY_TOKENS - fixedTokens;

  // The question alone does not fit. Give back as much of the answer budget as
  // possible before refusing, then refuse honestly.
  if (available < 0) {
    const recovered = Math.max(MIN_ANSWER_TOKENS, answer + available);
    available += answer - recovered;
    answer = recovered;
  }
  if (available < 0) {
    return {
      fits: false,
      contextSize: ceiling,
      answerTokens: answer,
      history: [],
      contextChunks: [],
      attachments: [],
      promptTokens: fixedTokens,
      ceiling,
      notes: [
        `This message alone needs about ${fixedTokens} tokens, more than the ${ceiling} this machine can cache for `
        + `${bytesPerToken ? `${(bytesPerToken / 1024).toFixed(0)} KB per token` : "this model"}. Shorten it, or activate a smaller model.`,
      ],
    };
  }

  // 1. Attachments: explicit for this turn, so they get first call on what is
  //    left, but a single huge file must not crowd out everything else.
  const keptAttachments = [];
  const attachmentCeiling = Math.floor(available * 0.6);
  let attachmentBudget = attachmentCeiling;
  for (const attachment of attachments) {
    const tokens = estimateTokens(attachment.text);
    if (tokens <= attachmentBudget) {
      keptAttachments.push(attachment);
      attachmentBudget -= tokens;
      continue;
    }
    if (attachmentBudget > MIN_ANSWER_TOKENS) {
      const keepChars = Math.floor(attachmentBudget * CHARS_PER_TOKEN);
      keptAttachments.push({
        ...attachment,
        text: `${attachment.text.slice(0, keepChars)}\n\n[truncated: ${attachment.name} is larger than this turn's budget]`,
        truncated: true,
      });
      notes.push(`${attachment.name} was truncated to fit the context.`);
      attachmentBudget = 0;
      continue;
    }
    notes.push(`${attachment.name} was left out: no room in this turn's context.`);
  }
  available -= attachmentCeiling - attachmentBudget;

  // 2. Retrieved chunks, best-ranked first.
  const keptChunks = [];
  let droppedChunks = 0;
  for (const chunk of contextChunks) {
    const tokens = estimateTokens(chunk.text);
    if (tokens <= available) {
      keptChunks.push(chunk);
      available -= tokens;
    } else {
      droppedChunks += 1;
    }
  }
  if (droppedChunks > 0) {
    notes.push(`${droppedChunks} document passage${droppedChunks === 1 ? "" : "s"} did not fit and ${droppedChunks === 1 ? "was" : "were"} left out.`);
  }

  // 3. History, newest first - older turns fall off the back.
  const keptHistory = [];
  for (let index = history.length - 1; index >= 0; index -= 1) {
    const tokens = estimateTokens(history[index].content);
    if (tokens > available) {
      break;
    }
    keptHistory.unshift(history[index]);
    available -= tokens;
  }
  const droppedTurns = history.length - keptHistory.length;
  if (droppedTurns > 0) {
    notes.push(`${droppedTurns} earlier message${droppedTurns === 1 ? "" : "s"} dropped from this turn's memory.`);
  }

  const promptTokens = fixedTokens
    + keptAttachments.reduce((total, attachment) => total + estimateTokens(attachment.text), 0)
    + keptChunks.reduce((total, chunk) => total + estimateTokens(chunk.text), 0)
    + keptHistory.reduce((total, turn) => total + estimateTokens(turn.content), 0);

  // Sized for this turn, but never smaller than this conversation has already
  // used.
  //
  // The size was previously recomputed from scratch every turn, so it moved
  // with the prompt: 2048, then 1024 for a short follow-up, then 2048 again.
  // Each change is a different KV shape, and the native prefill snapshot that
  // exists precisely so turn N+1 pays only for its new tail cannot be reused
  // across one - measured on an ordinary conversation, turn 1 prefilled in 4 ms
  // and turn 2 took 19.3 s, which the user experiences as the answer arriving
  // in one block after twenty seconds of nothing.
  //
  // Growing is still allowed (a longer turn needs the room) and the machine's
  // own ceiling still wins, so this cannot allocate more cache than the budget
  // permits. It only stops the size from oscillating downward.
  const turnContextSize = roundUpToBucket(promptTokens + answer + SAFETY_TOKENS);
  const floor = Number.isFinite(minContextSize) && minContextSize > 0
    ? roundUpToBucket(minContextSize)
    : 0;
  const contextSize = Math.min(ceiling, Math.max(turnContextSize, floor));

  return {
    fits: true,
    contextSize,
    answerTokens: answer,
    history: keptHistory,
    contextChunks: keptChunks,
    attachments: keptAttachments,
    promptTokens,
    ceiling,
    notes,
  };
}
