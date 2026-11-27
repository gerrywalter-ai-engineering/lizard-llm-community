import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";

function sha256(value) {
  return crypto.createHash("sha256").update(String(value ?? "")).digest("hex");
}

let cachedByteToUnicode = null;
let cachedUnicodeToByte = null;
const tokenizerDefinitionCache = new Map();
const tokenizerDefinitionObjectCache = new WeakMap();
const SENTENCEPIECE_SPACE = "\u2581";

function byteToUnicodeMap() {
  if (cachedByteToUnicode) {
    return cachedByteToUnicode;
  }
  const bytes = [];
  const chars = [];
  for (let code = 33; code <= 126; code += 1) {
    bytes.push(code);
    chars.push(String.fromCharCode(code));
  }
  for (let code = 161; code <= 172; code += 1) {
    bytes.push(code);
    chars.push(String.fromCharCode(code));
  }
  for (let code = 174; code <= 255; code += 1) {
    bytes.push(code);
    chars.push(String.fromCharCode(code));
  }
  let extra = 0;
  for (let byte = 0; byte < 256; byte += 1) {
    if (bytes.includes(byte)) {
      continue;
    }
    bytes.push(byte);
    chars.push(String.fromCharCode(256 + extra));
    extra += 1;
  }
  cachedByteToUnicode = new Map(bytes.map((byte, index) => [byte, chars[index]]));
  return cachedByteToUnicode;
}

function unicodeToByteMap() {
  if (cachedUnicodeToByte) {
    return cachedUnicodeToByte;
  }
  cachedUnicodeToByte = new Map(
    [...byteToUnicodeMap().entries()].map(([byte, symbol]) => [symbol, byte]),
  );
  return cachedUnicodeToByte;
}

function normalizeSpecialTokenIds(value = {}) {
  if (!value || typeof value !== "object") {
    return {};
  }
  return Object.entries(value).reduce((acc, [key, tokenId]) => {
    const numeric = Number(tokenId);
    if (Number.isInteger(numeric) && numeric >= 0) {
      acc[key] = numeric;
    }
    return acc;
  }, {});
}

function sidecarCandidates(modelPath) {
  if (!modelPath) {
    return [];
  }
  return [
    `${modelPath}.tokenizer.json`,
    `${modelPath}.tokenizer.sidecar.json`,
  ];
}

function readJson(filePath) {
  try {
    return JSON.parse(fs.readFileSync(filePath, "utf8"));
  } catch {
    return null;
  }
}

function firstExisting(paths = []) {
  return paths.find((filePath) => filePath && fs.existsSync(filePath)) ?? null;
}

function fileCacheIdentity(filePath) {
  if (!filePath) {
    return null;
  }
  const resolved = path.resolve(filePath);
  try {
    const stat = fs.statSync(resolved);
    return `${resolved}|${stat.size}|${stat.mtimeMs}`;
  } catch {
    return `${resolved}|missing`;
  }
}

function tokenizerDefinitionCacheKey({
  modelPath = null,
  gguf = null,
  sidecarPath = null,
} = {}) {
  if (modelPath && !fs.existsSync(modelPath) && !gguf?.metadataHash) {
    return null;
  }
  const modelIdentity = fileCacheIdentity(modelPath);
  if (!modelIdentity) {
    return null;
  }
  return [
    modelIdentity,
    fileCacheIdentity(sidecarPath) ?? "no-sidecar",
    String(gguf?.metadataHash ?? "no-metadata-hash"),
  ].join("|");
}

function normalizeSpecialTokens(value = {}) {
  if (!value || typeof value !== "object") {
    return {};
  }
  return Object.entries(value).reduce((acc, [key, token]) => {
    if (token == null) {
      return acc;
    }
    acc[key] = String(token);
    return acc;
  }, {});
}

function resolveGgufSpecialTokens(gguf = null) {
  const fromText = normalizeSpecialTokens(gguf?.specialTokenTexts);
  if (Object.keys(fromText).length > 0) {
    return fromText;
  }
  return normalizeSpecialTokens(gguf?.specialTokens);
}

function buildVocabularyArtifacts({ sidecarVocabulary = null, gguf = null } = {}) {
  const idToToken = [];
  const tokenToId = new Map();

  if (sidecarVocabulary && typeof sidecarVocabulary === "object") {
    for (const [key, value] of Object.entries(sidecarVocabulary)) {
      const numericKey = Number(key);
      if (Number.isInteger(numericKey) && typeof value === "string") {
        idToToken[numericKey] = value;
        if (!tokenToId.has(value)) {
          tokenToId.set(value, numericKey);
        }
        continue;
      }
      const numericValue = Number(value);
      if (Number.isInteger(numericValue)) {
        idToToken[numericValue] = String(key);
        if (!tokenToId.has(String(key))) {
          tokenToId.set(String(key), numericValue);
        }
      }
    }
  }

  const ggufTokens = Array.isArray(gguf?.raw?.["tokenizer.ggml.tokens"])
    ? gguf.raw["tokenizer.ggml.tokens"].filter((token) => typeof token === "string")
    : [];
  if (idToToken.length === 0 && ggufTokens.length > 0) {
    for (const [index, token] of ggufTokens.entries()) {
      idToToken[index] = token;
      if (!tokenToId.has(token)) {
        tokenToId.set(token, index);
      }
    }
  }

  const sidecarMerges = Array.isArray(sidecarVocabulary?.merges)
    ? sidecarVocabulary.merges.filter((merge) => typeof merge === "string")
    : [];
  const ggufMerges = Array.isArray(gguf?.raw?.["tokenizer.ggml.merges"])
    ? gguf.raw["tokenizer.ggml.merges"].filter((merge) => typeof merge === "string")
    : [];
  const merges = sidecarMerges.length > 0 ? sidecarMerges : ggufMerges;
  const mergeRanks = new Map(merges.map((merge, index) => [merge, index]));
  const ggufTokenTypes = Array.isArray(gguf?.raw?.["tokenizer.ggml.token_type"])
    ? gguf.raw["tokenizer.ggml.token_type"].map((value) => Number(value))
    : [];
  const ggufTokenScores = Array.isArray(gguf?.raw?.["tokenizer.ggml.scores"])
    ? gguf.raw["tokenizer.ggml.scores"].map((value) => Number(value))
    : [];
  const specialTokenEntries = idToToken
    .map((token, tokenId) => ({
      token,
      tokenId,
      tokenType: ggufTokenTypes[tokenId] ?? null,
    }))
    .filter(({ token, tokenType }) => (
      typeof token === "string"
      && token.length > 0
      && (tokenType === 3 || tokenType === 4)
    ))
    .sort((left, right) => right.token.length - left.token.length);

  const sortedTokens = [...tokenToId.keys()].sort((left, right) => {
    if (right.length !== left.length) {
      return right.length - left.length;
    }
    return left.localeCompare(right);
  });

  return {
    idToToken,
    tokenToId,
    merges,
    mergeRanks,
    sortedTokens,
    tokenTypes: ggufTokenTypes,
    tokenScores: ggufTokenScores,
    specialTokenEntries,
  };
}

function whitespaceTokens(text) {
  const normalized = String(text ?? "").trim();
  return normalized ? normalized.split(/\s+/).filter(Boolean) : [];
}

function estimateTokens(text) {
  const normalized = String(text ?? "").trim();
  if (!normalized) {
    return 0;
  }
  const matches = normalized.match(/[A-Za-z0-9_]+|[^\sA-Za-z0-9_]/g) ?? [];
  return matches.length;
}

function countTokensWithStrategy(text, tokenizer) {
  const strategy = tokenizer?.strategy ?? null;
  if (strategy === "whitespace") {
    return {
      tokenCount: whitespaceTokens(text).length,
      mode: "accurate",
    };
  }
  if (strategy === "character") {
    return {
      tokenCount: Array.from(String(text ?? "")).length,
      mode: "accurate",
    };
  }
  if (strategy === "whitespace-vocab") {
    return {
      tokenCount: whitespaceTokens(text).length,
      mode: "accurate",
    };
  }
  if (strategy === "gpt2-bpe") {
    const exact = resolveExactTokenIds(text, tokenizer);
    if (exact) {
      return {
        tokenCount: exact.tokenIds.length,
        mode: exact.mode,
      };
    }
  }
  return {
    tokenCount: estimateTokens(text),
    mode: "estimate",
  };
}

function encodePieceToByteBpeSymbols(piece) {
  const encoder = byteToUnicodeMap();
  return [...Buffer.from(piece, "utf8")].map((byte) => encoder.get(byte)).join("");
}

function decodeByteBpeSymbols(text) {
  const decoder = unicodeToByteMap();
  const bytes = [];
  let decoded = "";
  const flush = () => {
    if (bytes.length === 0) {
      return;
    }
    decoded += Buffer.from(bytes).toString("utf8");
    bytes.length = 0;
  };

  for (const symbol of String(text ?? "")) {
    const byte = decoder.get(symbol);
    if (byte == null) {
      flush();
      decoded += symbol;
      continue;
    }
    bytes.push(byte);
  }
  flush();
  return decoded;
}

function mergeBytePairSymbols(text, mergeRanks) {
  let symbols = Array.from(text);
  if (symbols.length <= 1 || !(mergeRanks instanceof Map) || mergeRanks.size === 0) {
    return symbols;
  }

  while (symbols.length > 1) {
    let bestIndex = -1;
    let bestRank = Number.POSITIVE_INFINITY;
    for (let index = 0; index < symbols.length - 1; index += 1) {
      const pair = `${symbols[index]} ${symbols[index + 1]}`;
      const rank = mergeRanks.get(pair);
      if (rank == null || rank >= bestRank) {
        continue;
      }
      bestRank = rank;
      bestIndex = index;
    }
    if (bestIndex === -1) {
      break;
    }
    symbols.splice(bestIndex, 2, symbols[bestIndex] + symbols[bestIndex + 1]);
  }

  return symbols;
}

function resolveGpt2BpeTokenIds(text, tokenizer) {
  const tokenToId = tokenizer?.tokenToId ?? null;
  const mergeRanks = tokenizer?.mergeRanks ?? null;
  if (!tokenToId || tokenToId.size === 0) {
    return null;
  }

  const tokenizerPre = String(tokenizer?.tokenizerPre ?? "").toLowerCase();
  const isLlama3Bpe = ["llama-bpe", "llama-v3", "llama3"].includes(tokenizerPre);
  // Llama 3 does not use GPT-2's original pre-tokenizer regex. In particular,
  // its `\s*[\r\n]` branch keeps consecutive newlines together so the common
  // header separator "\n\n" becomes vocabulary token 271. Splitting it into
  // two token-198 entries changes the model input even though the rendered
  // prompt bytes look identical. Other GPT-2-BPE families retain the legacy
  // pattern until their GGUF `tokenizer.ggml.pre` contract says otherwise.
  const pattern = isLlama3Bpe
    ? /'s|'t|'re|'ve|'m|'ll|'d|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+/gu
    : /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;
  const tokenIds = [];

  // Llama 3 and several newer GPT-2-BPE GGUFs embed control tokens directly in
  // their chat templates. Passing the whole rendered prompt through the regex
  // splits `<|start_header_id|>` into ordinary punctuation/word pieces, changing
  // both the prompt length and the model input. Partition known GGUF special
  // tokens first, exactly as the SentencePiece and Gemma BPE paths already do.
  const fragments = partitionSpecialTokens(text, tokenizer.specialTokenEntries);
  for (const fragment of fragments) {
    if (fragment.type === "special") {
      tokenIds.push(fragment.tokenId);
      continue;
    }
    const pieces = fragment.text.match(pattern) ?? [];
    for (const piece of pieces) {
      const encoded = encodePieceToByteBpeSymbols(piece);
      const mergedSymbols = mergeBytePairSymbols(encoded, mergeRanks);
      for (const symbol of mergedSymbols) {
        const tokenId = tokenToId.get(symbol);
        if (!Number.isInteger(tokenId)) {
          return null;
        }
        tokenIds.push(tokenId);
      }
    }
  }

  // Only prepend BOS when the model asks for it. qwen2/qwen3, deepseek, glm,
  // and starcoder2 GGUFs set add_bos_token=false (or default to false for
  // their BPE pre-tokenizers); prepending BOS anyway derails greedy decoding.
  const bosId = tokenizer?.specialTokenIds?.bos;
  if (tokenizer?.addBosToken && Number.isInteger(bosId) && tokenIds[0] !== bosId) {
    tokenIds.unshift(bosId);
  }

  return {
    tokenIds,
    mode: "gpt2-bpe",
  };
}

function isSentencePieceTokenizer(tokenizerName) {
  // GGUF names the SentencePiece tokenizer model "llama" (llama-2, mistral,
  // mixtral, phi3); llama-3-style BPE models use "gpt2" instead.
  return ["llama", "gemma", "gemma2", "gemma3"].includes(String(tokenizerName ?? "").toLowerCase());
}

function isGemma4Tokenizer(tokenizerName) {
  return String(tokenizerName ?? "").toLowerCase() === "gemma4";
}

function isGemma4Definition({ tokenizerName = null, architecture = null, tokenizerPre = null } = {}) {
  return isGemma4Tokenizer(tokenizerName)
    || String(architecture ?? "").toLowerCase() === "gemma4"
    || String(tokenizerPre ?? "").toLowerCase() === "gemma4";
}

function normalizeSentencePieceInput(text, { addSpacePrefix = false } = {}) {
  let normalized = String(text ?? "").replaceAll(" ", SENTENCEPIECE_SPACE);
  if (addSpacePrefix && normalized && !normalized.startsWith(SENTENCEPIECE_SPACE)) {
    normalized = `${SENTENCEPIECE_SPACE}${normalized}`;
  }
  return normalized;
}

function resolveSentencePieceTokenIds(text, tokenizer) {
  const tokenToId = tokenizer?.tokenToId ?? null;
  if (!tokenToId || tokenToId.size === 0) {
    return null;
  }
  if (!Array.isArray(tokenizer?.tokenScores) || tokenizer.tokenScores.length === 0) {
    return resolveGreedySentencePieceTokenIds(text, tokenizer);
  }

  const tokenIds = [];
  const fragments = partitionSpecialTokens(text, tokenizer.specialTokenEntries);
  let previousWasSpecial = true;

  for (const fragment of fragments) {
    if (fragment.type === "special") {
      tokenIds.push(fragment.tokenId);
      previousWasSpecial = true;
      continue;
    }
    let rawText = fragment.text;
    if (tokenizer?.addSpacePrefix && previousWasSpecial) {
      rawText = ` ${rawText}`;
    }
    const normalized = normalizeSentencePieceInput(rawText);
    const pieceTokenIds = resolveSentencePieceSymbols(Array.from(normalized), tokenizer);
    if (!pieceTokenIds) {
      return null;
    }
    tokenIds.push(...pieceTokenIds);
    previousWasSpecial = false;
  }

  const bosId = tokenizer?.specialTokenIds?.bos;
  if (tokenizer?.addBosToken && Number.isInteger(bosId) && tokenIds[0] !== bosId) {
    tokenIds.unshift(bosId);
  }

  return {
    tokenIds,
    mode: "sentencepiece-vocab",
  };
}

function resolveGreedySentencePieceTokenIds(text, tokenizer) {
  const rendered = normalizeSentencePieceInput(text, {
    addSpacePrefix: Boolean(tokenizer?.addSpacePrefix),
  });
  const tokenIds = [];
  let offset = 0;

  while (offset < rendered.length) {
    const matchedToken = tokenizer.sortedTokens.find((token) => (
      token && rendered.startsWith(token, offset)
    ));
    if (!matchedToken) {
      return null;
    }
    tokenIds.push(tokenizer.tokenToId.get(matchedToken));
    offset += matchedToken.length;
  }

  const bosId = tokenizer?.specialTokenIds?.bos;
  if (tokenizer?.addBosToken && Number.isInteger(bosId) && tokenIds[0] !== bosId) {
    tokenIds.unshift(bosId);
  }
  return {
    tokenIds,
    mode: "sentencepiece-vocab",
  };
}

function resolveSentencePieceSymbols(initialSymbols, tokenizer) {
  const symbols = [...initialSymbols];
  const tokenScores = tokenizer?.tokenScores ?? [];

  while (symbols.length > 1) {
    let bestIndex = -1;
    let bestScore = Number.NEGATIVE_INFINITY;
    for (let index = 0; index < symbols.length - 1; index += 1) {
      const merged = symbols[index] + symbols[index + 1];
      const tokenId = tokenizer.tokenToId.get(merged);
      if (!Number.isInteger(tokenId)) {
        continue;
      }
      const score = Number(tokenScores[tokenId] ?? Number.NEGATIVE_INFINITY);
      if (score > bestScore) {
        bestScore = score;
        bestIndex = index;
      }
    }
    if (bestIndex === -1) {
      break;
    }
    symbols.splice(bestIndex, 2, symbols[bestIndex] + symbols[bestIndex + 1]);
  }

  const tokenIds = [];
  for (const symbol of symbols) {
    const tokenId = tokenizer.tokenToId.get(symbol);
    if (Number.isInteger(tokenId)) {
      tokenIds.push(tokenId);
      continue;
    }
    for (const byte of Buffer.from(symbol, "utf8")) {
      const byteTokenId = tokenizer.tokenToId.get(
        `<0x${byte.toString(16).toUpperCase().padStart(2, "0")}>`,
      );
      if (!Number.isInteger(byteTokenId)) {
        return null;
      }
      tokenIds.push(byteTokenId);
    }
  }
  return tokenIds;
}

function partitionSpecialTokens(text, specialTokenEntries = []) {
  const fragments = [];
  const rendered = String(text ?? "");
  let rawStart = 0;
  let offset = 0;

  while (offset < rendered.length) {
    const special = specialTokenEntries.find(({ token }) => rendered.startsWith(token, offset));
    if (!special) {
      offset += 1;
      continue;
    }
    if (rawStart < offset) {
      fragments.push({
        type: "raw",
        text: rendered.slice(rawStart, offset),
      });
    }
    fragments.push({
      type: "special",
      tokenId: special.tokenId,
    });
    offset += special.token.length;
    rawStart = offset;
  }

  if (rawStart < rendered.length) {
    fragments.push({
      type: "raw",
      text: rendered.slice(rawStart),
    });
  }
  return fragments;
}

function resolveRawBpeSymbols(symbols, tokenizer) {
  const mergedSymbols = mergeBytePairSymbols(symbols.join(""), tokenizer?.mergeRanks);
  const tokenIds = [];

  for (const symbol of mergedSymbols) {
    const tokenId = tokenizer?.tokenToId?.get(symbol);
    if (Number.isInteger(tokenId)) {
      tokenIds.push(tokenId);
      continue;
    }
    for (const byte of Buffer.from(symbol, "utf8")) {
      const byteTokenId = tokenizer?.tokenToId?.get(
        `<0x${byte.toString(16).toUpperCase().padStart(2, "0")}>`,
      );
      if (!Number.isInteger(byteTokenId)) {
        return null;
      }
      tokenIds.push(byteTokenId);
    }
  }
  return tokenIds;
}

function resolveGemma4BpeTokenIds(text, tokenizer) {
  if (!(tokenizer?.tokenToId instanceof Map) || tokenizer.tokenToId.size === 0) {
    return null;
  }

  const tokenIds = [];
  const fragments = partitionSpecialTokens(text, tokenizer.specialTokenEntries);
  for (const fragment of fragments) {
    if (fragment.type === "special") {
      tokenIds.push(fragment.tokenId);
      continue;
    }

    const normalized = normalizeSentencePieceInput(fragment.text);
    const pieces = normalized.match(/[^\n]+|[\n]+/gu) ?? [];
    for (const piece of pieces) {
      const directNewlineId = /^\n+$/u.test(piece)
        ? tokenizer.tokenToId.get(piece)
        : null;
      if (Number.isInteger(directNewlineId)) {
        tokenIds.push(directNewlineId);
        continue;
      }
      const pieceTokenIds = resolveRawBpeSymbols(Array.from(piece), tokenizer);
      if (!pieceTokenIds) {
        return null;
      }
      tokenIds.push(...pieceTokenIds);
    }
  }

  const bosId = tokenizer?.specialTokenIds?.bos;
  if (tokenizer?.addBosToken && Number.isInteger(bosId) && tokenIds[0] !== bosId) {
    tokenIds.unshift(bosId);
  }

  return {
    tokenIds,
    mode: "gemma4-bpe",
  };
}

function resolveExactTokenIds(text, tokenizer) {
  // Try the strategy-specific resolver first, but fall through to the greedy
  // exact-vocab matcher when it cannot produce IDs (e.g. reduced synthetic
  // vocabularies without merge/score data).
  let resolved = null;
  if (tokenizer?.strategy === "gpt2-bpe") {
    resolved = resolveGpt2BpeTokenIds(text, tokenizer);
  } else if (tokenizer?.strategy === "sentencepiece-vocab") {
    resolved = resolveSentencePieceTokenIds(text, tokenizer);
  } else if (tokenizer?.strategy === "gemma4-bpe") {
    resolved = resolveGemma4BpeTokenIds(text, tokenizer);
  }
  if (resolved) {
    return resolved;
  }
  const sortedTokens = tokenizer?.sortedTokens ?? [];
  const tokenToId = tokenizer?.tokenToId ?? null;
  if (!tokenToId || sortedTokens.length === 0) {
    return null;
  }
  const rendered = String(text ?? "");
  const tokenIds = [];
  let offset = 0;

  while (offset < rendered.length) {
    let matchedToken = null;
    for (const token of sortedTokens) {
      if (token && rendered.startsWith(token, offset)) {
        matchedToken = token;
        break;
      }
    }
    if (!matchedToken) {
      return null;
    }
    tokenIds.push(tokenToId.get(matchedToken));
    offset += matchedToken.length;
  }

  return {
    tokenIds,
    mode: "exact-vocab",
  };
}

function normalizeTemplateMessages({ messages = null, prompt, systemPrompt }) {
  if (Array.isArray(messages) && messages.length > 0) {
    return messages.map((message) => ({
      role: String(message.role),
      content: String(message.content ?? ""),
    }));
  }
  return [
    ...(systemPrompt ? [{ role: "system", content: String(systemPrompt) }] : []),
    { role: "user", content: String(prompt ?? "") },
  ];
}

function byteBpeBytes(text) {
  const decoder = unicodeToByteMap();
  const bytes = [];
  for (const symbol of String(text ?? "")) {
    const byte = decoder.get(symbol);
    if (byte == null) return null;
    bytes.push(byte);
  }
  return bytes;
}

function tokenPieceWithDefinition(tokenizer, tokenId) {
  const token = tokenizer.vocabularyTokens?.[tokenId];
  if (typeof token !== "string") return null;
  const byteToken = /^<0x([0-9A-Fa-f]{2})>$/.exec(token);
  if (byteToken) return [Number.parseInt(byteToken[1], 16)];
  const byteBpe = tokenizer.strategy === "gpt2-bpe"
    || tokenizer.tokenizer === "gpt2"
    || tokenizer.tokenizer === "llama-bpe";
  if (!byteBpe) {
    return tokenizer.strategy === "sentencepiece-vocab" || tokenizer.strategy === "gemma4-bpe"
      ? token.replaceAll(SENTENCEPIECE_SPACE, " ")
      : token;
  }
  const bytes = byteBpeBytes(token);
  if (!bytes) return token;
  try {
    return new TextDecoder("utf-8", { fatal: true }).decode(Uint8Array.from(bytes));
  } catch {
    return bytes;
  }
}

function fallbackRenderedPrompt({ messages, structured = false, prompt = "", systemPrompt = null }) {
  if (!structured) {
    const parts = [];
    if (systemPrompt) parts.push(`System: ${systemPrompt}`);
    parts.push(prompt);
    return parts.join("\n\n").trim();
  }
  return messages.map((message) => `${message.role[0].toUpperCase()}${message.role.slice(1)}: ${message.content}`)
    .join("\n\n")
    .trim();
}

// Structure for models that carry no chat template - notably BASE (non-instruct)
// GGUFs, which is what "llama-3-2-1b-gguf-q8_0" is.
//
// A base model is a text completer: handed "System: ...\n\nwhat is 2 + 2?" it
// continues the document, producing "4\nwhat is 2 + 2? 4\n..." until it runs
// out of budget. It is not hallucinating - it was never asked a question, it
// was given a passage.
//
// The instruction format below is the one such models have seen most in
// pretraining, and, critically, it ends each answer with a marker the stop
// filter can cut on (createStopSequenceFilter treats a line-leading "###" as
// end of turn). Opt-in via chatMode so benchmark prompts, which must stay
// comparable across runs, are unaffected.
export const INSTRUCTION_RESPONSE_MARKER = "### Response:";

// One worked example before the real question. A base model continues the most
// likely pattern, and with a bare "### Instruction: <x> ### Response:" the most
// likely continuation is often to REPEAT x - which is exactly the "it just
// answers what I wrote" echo seen on greedy (temperature 0) decoding, the only
// mode Caterpillar supports. The exemplar makes "answer the instruction" the
// pattern being continued instead of "restate it". Cheap (~25 tokens) and it
// helps every base model, on every engine.
const INSTRUCTION_EXEMPLAR = [
  "### Instruction:",
  "Hello!",
  "",
  "### Response:",
  "Hello! How can I help you today?",
  "",
  "### Instruction:",
  "What is the capital of France?",
  "",
  "### Response:",
  "The capital of France is Paris.",
].join("\n");

function renderInstructionPrompt({ messages }) {
  const parts = [];
  parts.push(INSTRUCTION_EXEMPLAR);
  for (const message of messages) {
    if (message.role === "system") {
      parts.push(`### System:\n${message.content.trim()}`);
    } else if (message.role === "user") {
      parts.push(`### Instruction:\n${message.content.trim()}\n\n${INSTRUCTION_RESPONSE_MARKER}`);
    } else if (message.role === "assistant") {
      parts.push(message.content.trim());
    }
  }
  return `${parts.join("\n\n")}\n`;
}

function renderGemmaChatPrompt({ messages, specialTokens = {} }) {
  const parts = messages.map((message) => (
    `<|turn>${message.role === "assistant" ? "model" : message.role}\n${message.content.trim()}<turn|>`
  ));
  parts.push("<|turn>model\n");
  return `${specialTokens.bos ?? ""}${parts.join("\n")}`;
}

// Llama 3 / ChatML rendering. Both families declare their markup literally in
// the Jinja template, so the family can be identified from the template text
// even though the Jinja itself is not executed here.
function renderKnownChatFamily({ template, messages, specialTokens }) {
  const bos = specialTokens.bos ?? "";
  if (template.includes("<|start_header_id|>")) {
    const parts = [bos];
    for (const message of messages) {
      parts.push(`<|start_header_id|>${message.role}<|end_header_id|>\n\n${message.content.trim()}<|eot_id|>`);
    }
    parts.push("<|start_header_id|>assistant<|end_header_id|>\n\n");
    return {
      renderedPrompt: parts.join(""),
      templateMode: "rendered",
      templateSource: "llama3_chat",
      unresolvedPlaceholders: [],
    };
  }
  if (template.includes("<|im_start|>")) {
    const parts = [];
    for (const message of messages) {
      parts.push(`<|im_start|>${message.role}\n${message.content.trim()}<|im_end|>\n`);
    }
    parts.push("<|im_start|>assistant\n");
    return {
      renderedPrompt: parts.join(""),
      templateMode: "rendered",
      templateSource: "chatml",
      unresolvedPlaceholders: [],
    };
  }
  return null;
}

export function renderChatTemplate({
  template = null,
  prompt,
  systemPrompt = null,
  messages = null,
  specialTokens = {},
  tokenizerName = null,
  architecture = null,
  tokenizerPre = null,
  chatMode = false,
}) {
  const normalizedPrompt = String(prompt ?? "");
  const normalizedSystemPrompt = systemPrompt == null ? null : String(systemPrompt);
  const structuredMessages = Array.isArray(messages) && messages.length > 0;
  const normalizedMessages = normalizeTemplateMessages({
    messages,
    prompt: normalizedPrompt,
    systemPrompt: normalizedSystemPrompt,
  });
  const isGemma4 = isGemma4Definition({
    tokenizerName,
    architecture,
    tokenizerPre,
  });

  if (!template) {
    if (isGemma4) {
      return {
        renderedPrompt: renderGemmaChatPrompt({
          messages: normalizedMessages,
          specialTokens,
        }),
        templateMode: "rendered",
        templateSource: "gemma_chat_fallback",
        unresolvedPlaceholders: [],
      };
    }
    if (chatMode) {
      return {
        renderedPrompt: renderInstructionPrompt({
          messages: normalizedMessages,
        }),
        templateMode: "rendered",
        templateSource: "instruction_fallback",
        unresolvedPlaceholders: [],
      };
    }
    const renderedPrompt = fallbackRenderedPrompt({
      messages: normalizedMessages,
      structured: structuredMessages,
      prompt: normalizedPrompt,
      systemPrompt: normalizedSystemPrompt,
    });
    return {
      renderedPrompt,
      templateMode: "fallback",
      templateSource: "default",
      unresolvedPlaceholders: [],
    };
  }

  if (template.includes("{%") || template.includes("{#")) {
    if (isSentencePieceTokenizer(tokenizerName) || isGemma4) {
      return {
        renderedPrompt: renderGemmaChatPrompt({
          messages: normalizedMessages,
          specialTokens,
        }),
        templateMode: "rendered",
        templateSource: "gemma_chat_fallback",
        unresolvedPlaceholders: [],
      };
    }
    // Llama 3 and ChatML ship Jinja templates, which this renderer cannot
    // execute. Falling through to fallbackRenderedPrompt handed the model
    // "System: ...\n\n<question>" with no turn markers at all - so an
    // instruct model, given what looks like a plain document, continued the
    // document instead of answering, and never emitted an end-of-turn token.
    // The markup for both families is fixed and short, so render it directly
    // rather than degrading to raw text.
    const familyRender = renderKnownChatFamily({
      template,
      messages: normalizedMessages,
      specialTokens,
    });
    if (familyRender) {
      return familyRender;
    }
    const renderedPrompt = fallbackRenderedPrompt({
      messages: normalizedMessages,
      structured: structuredMessages,
      prompt: normalizedPrompt,
      systemPrompt: normalizedSystemPrompt,
    });
    return {
      renderedPrompt,
      templateMode: "fallback",
      templateSource: "unsupported_jinja",
      unresolvedPlaceholders: [],
    };
  }

  const replacements = {
    prompt: normalizedPrompt,
    user_prompt: normalizedPrompt,
    system_prompt: normalizedSystemPrompt ?? "",
    bos_token: specialTokens.bos ?? "",
    eos_token: specialTokens.eos ?? "",
    pad_token: specialTokens.pad ?? "",
    unk_token: specialTokens.unk ?? "",
  };

  let renderedPrompt = String(template);
  for (const [key, value] of Object.entries(replacements)) {
    const pattern = new RegExp(`\\{\\{\\s*${key}\\s*\\}\\}`, "g");
    renderedPrompt = renderedPrompt.replace(pattern, value);
  }
  const unresolvedPlaceholders = [...renderedPrompt.matchAll(/\{\{\s*([^}]+?)\s*\}\}/g)].map((match) => match[1].trim());
  if (unresolvedPlaceholders.length > 0 || normalizedMessages.length > (normalizedSystemPrompt ? 2 : 1)) {
    renderedPrompt = fallbackRenderedPrompt({
      messages: normalizedMessages,
      structured: structuredMessages,
      prompt: normalizedPrompt,
      systemPrompt: normalizedSystemPrompt,
    });
    return {
      renderedPrompt,
      templateMode: "fallback",
      templateSource: unresolvedPlaceholders.length > 0 ? "unresolved_placeholders" : "unsupported_template_multiturn",
      unresolvedPlaceholders,
    };
  }

  return {
    renderedPrompt,
    templateMode: "rendered",
    templateSource: "template",
    unresolvedPlaceholders: [],
  };
}

export function loadTokenizerDefinition({
  modelPath = null,
  gguf = null,
  sidecarPath = null,
} = {}) {
  if (gguf && typeof gguf === "object" && tokenizerDefinitionObjectCache.has(gguf)) {
    return tokenizerDefinitionObjectCache.get(gguf);
  }
  const resolvedSidecarPath = sidecarPath ?? firstExisting(sidecarCandidates(modelPath));
  const cacheKey = tokenizerDefinitionCacheKey({
    modelPath,
    gguf,
    sidecarPath: resolvedSidecarPath,
  });
  if (cacheKey && tokenizerDefinitionCache.has(cacheKey)) {
    return tokenizerDefinitionCache.get(cacheKey);
  }
  const sidecar = resolvedSidecarPath ? readJson(resolvedSidecarPath) : null;
  const specialTokens = sidecar
    ? normalizeSpecialTokens(sidecar?.specialTokens)
    : resolveGgufSpecialTokens(gguf);
  const specialTokenIds = sidecar
    ? normalizeSpecialTokenIds(sidecar?.specialTokenIds)
    : normalizeSpecialTokenIds(gguf?.specialTokens);
  const vocabulary = sidecar?.vocabulary && typeof sidecar.vocabulary === "object"
    ? Object.fromEntries(Object.entries(sidecar.vocabulary).map(([key, value]) => [String(key), value]))
    : null;
  const vocabularyDefinition = {
    ...(vocabulary ?? {}),
    ...(Array.isArray(sidecar?.merges) ? { merges: sidecar.merges } : {}),
  };
  const vocabularyArtifacts = buildVocabularyArtifacts({
    sidecarVocabulary: sidecar ? vocabularyDefinition : null,
    gguf,
  });
  const tokenizerName = sidecar?.tokenizer ?? gguf?.tokenizer ?? "unknown";
  const tokenizerPre = sidecar?.tokenizerPre ?? gguf?.raw?.["tokenizer.ggml.pre"] ?? null;
  const gemma4Definition = isGemma4Definition({
    tokenizerName,
    architecture: gguf?.architecture,
    tokenizerPre,
  });
  const inferredStrategy = sidecar?.strategy
    ?? (vocabulary
      ? "whitespace-vocab"
      : (isGemma4Tokenizer(tokenizerName)
        && vocabularyArtifacts.idToToken.length > 0
        && vocabularyArtifacts.merges.length > 0
        ? "gemma4-bpe"
        : ((isSentencePieceTokenizer(tokenizerName) || gemma4Definition)
        && vocabularyArtifacts.idToToken.length > 0
        ? "sentencepiece-vocab"
        : (gguf?.tokenizer === "gpt2"
        && vocabularyArtifacts.idToToken.length > 0
        && vocabularyArtifacts.merges.length > 0
        ? "gpt2-bpe"
        : null))));
  // When the GGUF does not carry add_bos_token, default per llama.cpp:
  // llama-3's BPE pre-tokenizer expects a BOS prefix, other BPE families
  // (qwen2, deepseek, glm, starcoder2, ...) do not.
  const addBosToken = sidecar?.addBosToken ?? (
    typeof gguf?.raw?.["tokenizer.ggml.add_bos_token"] === "boolean"
      ? gguf.raw["tokenizer.ggml.add_bos_token"]
      : tokenizerPre === "llama-bpe" || tokenizerPre === "llama-v3" || tokenizerPre === "llama3"
  );
  const addSpacePrefix = sidecar?.addSpacePrefix ?? (
    typeof gguf?.raw?.["tokenizer.ggml.add_space_prefix"] === "boolean"
      ? gguf.raw["tokenizer.ggml.add_space_prefix"]
      : inferredStrategy === "sentencepiece-vocab"
  );
  const explicitSpecialTokenEntries = Object.values(specialTokenIds)
    .filter((tokenId) => Number.isInteger(tokenId))
    .map((tokenId) => ({
      tokenId,
      token: vocabularyArtifacts.idToToken[tokenId],
      tokenType: vocabularyArtifacts.tokenTypes[tokenId] ?? 3,
    }))
    .filter(({ token }) => typeof token === "string" && token.length > 0);
  const specialTokenEntries = [
    ...vocabularyArtifacts.specialTokenEntries,
    ...explicitSpecialTokenEntries,
  ].filter((entry, index, entries) => (
    entries.findIndex((candidate) => candidate.tokenId === entry.tokenId) === index
  )).sort((left, right) => right.token.length - left.token.length);

  const definition = {
    tokenizer: tokenizerName,
    source: sidecar ? "sidecar" : "gguf",
    sidecarPath: sidecar ? resolvedSidecarPath : null,
    strategy: inferredStrategy,
    vocabulary,
    vocabularyTokens: vocabularyArtifacts.idToToken,
    tokenToId: vocabularyArtifacts.tokenToId,
    mergeRanks: vocabularyArtifacts.mergeRanks,
    merges: vocabularyArtifacts.merges,
    sortedTokens: vocabularyArtifacts.sortedTokens,
    tokenTypes: vocabularyArtifacts.tokenTypes,
    tokenScores: vocabularyArtifacts.tokenScores,
    specialTokenEntries,
    chatTemplate: sidecar?.chatTemplate ?? gguf?.chatTemplate ?? null,
    specialTokens,
    specialTokenIds,
    addBosToken,
    addSpacePrefix,
    metadataStatus: gguf?.status ?? null,
    architecture: gguf?.architecture ?? null,
    tokenizerPre,
  };
  if (cacheKey) {
    tokenizerDefinitionCache.set(cacheKey, definition);
  }
  if (gguf && typeof gguf === "object") {
    tokenizerDefinitionObjectCache.set(gguf, definition);
  }
  return definition;
}

export function clearTokenizerDefinitionCache() {
  tokenizerDefinitionCache.clear();
}

export function buildPromptArtifact({
  prompt,
  systemPrompt = null,
  messages = null,
  modelPath = null,
  gguf = null,
  sidecarPath = null,
  // Chat callers opt in to instruction structure for template-less models.
  // Benchmarks deliberately do not: their prompts must stay byte-comparable
  // across runs and versions.
  chatMode = false,
} = {}) {
  const tokenizer = loadTokenizerDefinition({
    modelPath,
    gguf,
    sidecarPath,
  });
  const rendered = renderChatTemplate({
    template: tokenizer.chatTemplate,
    prompt,
    systemPrompt,
    messages,
    specialTokens: tokenizer.specialTokens,
    tokenizerName: tokenizer.tokenizer,
    architecture: tokenizer.architecture,
    tokenizerPre: tokenizer.tokenizerPre,
    chatMode,
  });
  const exactTokenization = resolveExactTokenIds(rendered.renderedPrompt, tokenizer);
  const counted = countTokensWithStrategy(rendered.renderedPrompt, tokenizer);
  const tokenCount = exactTokenization
    ? {
      tokenCount: exactTokenization.tokenIds.length,
      mode: exactTokenization.mode,
    }
    : counted;

  return {
    tokenizer: tokenizer.tokenizer,
    tokenizerSource: tokenizer.source,
    tokenizerStrategy: tokenizer.strategy ?? "estimate",
    tokenizerSidecarPath: tokenizer.sidecarPath,
    templateSource: tokenizer.chatTemplate ? tokenizer.source : "default",
    templateRenderer: rendered.templateSource,
    templateMode: rendered.templateMode,
    renderedPrompt: rendered.renderedPrompt,
    renderedPromptHash: sha256(rendered.renderedPrompt),
    prompt,
    systemPrompt: systemPrompt == null ? null : String(systemPrompt),
    messages: Array.isArray(messages) ? messages.map((message) => ({ ...message })) : null,
    tokenCount: tokenCount.tokenCount,
    tokenCountMode: tokenCount.mode,
    tokenIds: exactTokenization?.tokenIds ?? null,
    tokenIdMode: exactTokenization?.mode ?? null,
    metadataStatus: tokenizer.metadataStatus,
    unresolvedPlaceholders: rendered.unresolvedPlaceholders,
  };
}

// Tokenize an already-rendered prompt without applying a chat template again.
// The benchmark truth layer uses this to compare the exact prompt bytes and
// token IDs accepted by Lizard with llama-server's /apply-template + /tokenize
// result. Keeping it beside buildPromptArtifact guarantees both paths use the
// same tokenizer definition and exact-token fallback rules.
export function tokenizeRenderedPrompt({
  content,
  modelPath = null,
  gguf = null,
  sidecarPath = null,
  addSpecial = null,
  parseSpecial = true,
  withPieces = false,
} = {}) {
  const tokenizer = loadTokenizerDefinition({ modelPath, gguf, sidecarPath });
  const specialTexts = new Set(tokenizer.specialTokenEntries.map((entry) => entry.token));
  const effectiveTokenizer = {
    ...tokenizer,
    // Add BOS once below so every tokenizer strategy observes the same caller
    // option, including the exact-vocabulary fallback.
    addBosToken: false,
    specialTokenEntries: parseSpecial ? tokenizer.specialTokenEntries : [],
    sortedTokens: parseSpecial
      ? tokenizer.sortedTokens
      : tokenizer.sortedTokens.filter((token) => !specialTexts.has(token)),
  };
  const exact = resolveExactTokenIds(String(content ?? ""), effectiveTokenizer);
  const tokenIds = exact?.tokenIds ? [...exact.tokenIds] : null;
  const shouldAddSpecial = addSpecial == null ? tokenizer.addBosToken : addSpecial === true;
  const bosId = tokenizer.specialTokenIds?.bos;
  if (tokenIds && shouldAddSpecial && Number.isInteger(bosId) && tokenIds[0] !== bosId) {
    tokenIds.unshift(bosId);
  }
  const tokenPieces = tokenIds && withPieces
    ? tokenIds.map((tokenId) => ({
      id: tokenId,
      piece: tokenPieceWithDefinition(tokenizer, tokenId),
    }))
    : null;
  return {
    tokenIds,
    tokenPieces,
    tokenIdMode: exact?.mode ?? null,
    tokenizer: tokenizer.tokenizer,
    tokenizerStrategy: tokenizer.strategy ?? "estimate",
  };
}

export function buildRawPromptArtifact({
  prompt,
  promptTokenIds = null,
  modelPath = null,
  gguf = null,
  sidecarPath = null,
} = {}) {
  const renderedPrompt = String(prompt ?? "");
  const tokenized = Array.isArray(promptTokenIds)
    ? {
      tokenIds: [...promptTokenIds],
      tokenIdMode: "provided-token-ids",
      tokenizer: loadTokenizerDefinition({ modelPath, gguf, sidecarPath }).tokenizer,
      tokenizerStrategy: "provided-token-ids",
    }
    : tokenizeRenderedPrompt({
      content: renderedPrompt,
      modelPath,
      gguf,
      sidecarPath,
    });
  return {
    tokenizer: tokenized.tokenizer,
    tokenizerSource: "raw-completion",
    tokenizerStrategy: tokenized.tokenizerStrategy,
    tokenizerSidecarPath: sidecarPath,
    templateSource: "none",
    templateRenderer: "raw",
    templateMode: "raw",
    renderedPrompt,
    renderedPromptHash: sha256(`tokens:${(tokenized.tokenIds ?? []).join(",")}`),
    prompt: renderedPrompt,
    systemPrompt: null,
    messages: null,
    tokenCount: tokenized.tokenIds?.length ?? 0,
    tokenCountMode: tokenized.tokenIdMode ?? "unavailable",
    tokenIds: tokenized.tokenIds,
    tokenIdMode: tokenized.tokenIdMode,
    metadataStatus: gguf?.status ?? null,
    unresolvedPlaceholders: [],
  };
}

export function detokenizeTokenIds({
  tokenIds,
  modelPath = null,
  gguf = null,
  sidecarPath = null,
} = {}) {
  const tokenizer = loadTokenizerDefinition({ modelPath, gguf, sidecarPath });
  if (!Array.isArray(tokenIds)) throw new TypeError("tokens must be an array");
  for (const tokenId of tokenIds) {
    if (!Number.isSafeInteger(tokenId) || tokenId < 0
      || typeof tokenizer.vocabularyTokens?.[tokenId] !== "string") {
      const error = new RangeError(`Unknown tokenizer token ID: ${tokenId}`);
      error.code = "TOKEN_ID_INVALID";
      throw error;
    }
  }
  const byteBpe = tokenizer.strategy === "gpt2-bpe"
    || tokenizer.tokenizer === "gpt2"
    || tokenizer.tokenizer === "llama-bpe";
  const sentencePiece = tokenizer.strategy === "sentencepiece-vocab"
    || tokenizer.strategy === "gemma4-bpe"
    || isSentencePieceTokenizer(tokenizer.tokenizer)
    || isGemma4Tokenizer(tokenizer.tokenizer);
  const rawTokens = tokenIds.map((tokenId) => tokenizer.vocabularyTokens[tokenId]);
  const text = byteBpe
    ? decodeByteBpeSymbols(rawTokens.join(""))
    : sentencePiece
      ? rawTokens.join("").replaceAll(SENTENCEPIECE_SPACE, " ")
      : rawTokens.join("");
  return {
    text,
    chunks: tokenIds.map((tokenId) => tokenPieceWithDefinition(tokenizer, tokenId)),
    tokenizer: tokenizer.tokenizer,
    strategy: tokenizer.strategy ?? "unknown",
  };
}

export function decodeGeneratedTokenIds({
  tokenIds = [],
  modelPath = null,
  gguf = null,
  sidecarPath = null,
} = {}) {
  const tokenizer = loadTokenizerDefinition({
    modelPath,
    gguf,
    sidecarPath,
  });
  const shouldDecodeByteBpe = tokenizer.strategy === "gpt2-bpe"
    || tokenizer.tokenizer === "gpt2"
    || tokenizer.tokenizer === "llama-bpe";
  const shouldDecodeSentencePiece = tokenizer.strategy === "sentencepiece-vocab"
    || tokenizer.strategy === "gemma4-bpe"
    || isSentencePieceTokenizer(tokenizer.tokenizer)
    || isGemma4Tokenizer(tokenizer.tokenizer);
  const chunks = [];
  for (const tokenId of tokenIds ?? []) {
    chunks.push(decodeTokenWithDefinition(tokenizer, tokenId, {
      shouldDecodeByteBpe,
      shouldDecodeSentencePiece,
    }));
  }

  const text = chunks.join("");
  return {
    text: shouldDecodeSentencePiece ? text.replace(/^\s+/, "") : text,
    chunks,
    tokenizer: tokenizer.tokenizer,
    strategy: tokenizer.strategy ?? "unknown",
  };
}

function decodeTokenWithDefinition(tokenizer, tokenId, {
  shouldDecodeByteBpe = tokenizer.strategy === "gpt2-bpe"
    || tokenizer.tokenizer === "gpt2"
    || tokenizer.tokenizer === "llama-bpe",
  shouldDecodeSentencePiece = tokenizer.strategy === "sentencepiece-vocab"
    || tokenizer.strategy === "gemma4-bpe"
    || isSentencePieceTokenizer(tokenizer.tokenizer)
    || isGemma4Tokenizer(tokenizer.tokenizer),
} = {}) {
  const token = tokenizer.vocabularyTokens?.[tokenId];
  if (typeof token !== "string") {
    return `<token:${tokenId}>`;
  }
  if (shouldDecodeByteBpe) {
    return decodeByteBpeSymbols(token);
  }
  if (shouldDecodeSentencePiece) {
    return token.replaceAll(SENTENCEPIECE_SPACE, " ");
  }
  return token;
}

export function createIncrementalTokenDecoder({
  modelPath = null,
  gguf = null,
  sidecarPath = null,
} = {}) {
  const tokenizer = loadTokenizerDefinition({
    modelPath,
    gguf,
    sidecarPath,
  });
  const shouldDecodeByteBpe = tokenizer.strategy === "gpt2-bpe"
    || tokenizer.tokenizer === "gpt2"
    || tokenizer.tokenizer === "llama-bpe";
  const shouldDecodeSentencePiece = tokenizer.strategy === "sentencepiece-vocab"
    || tokenizer.strategy === "gemma4-bpe"
    || isSentencePieceTokenizer(tokenizer.tokenizer)
    || isGemma4Tokenizer(tokenizer.tokenizer);
  let leadingStripPending = shouldDecodeSentencePiece;

  return {
    tokenizer: tokenizer.tokenizer,
    strategy: tokenizer.strategy ?? "unknown",
    decodeToken(tokenId) {
      let text = decodeTokenWithDefinition(tokenizer, tokenId, {
        shouldDecodeByteBpe,
        shouldDecodeSentencePiece,
      });
      if (leadingStripPending) {
        text = text.replace(/^\s+/, "");
        if (text) {
          leadingStripPending = false;
        }
      }
      return text;
    },
    decodeTokens(tokenIds = []) {
      return tokenIds.map((tokenId) => this.decodeToken(tokenId)).join("");
    },
    reset() {
      leadingStripPending = shouldDecodeSentencePiece;
    },
  };
}
