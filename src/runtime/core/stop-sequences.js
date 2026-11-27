// Where a generated turn actually ends.
//
// The native worker stops on the GGUF's base EOS token
// (tokenizer.ggml.eos_token_id, see main.cpp). For an *instruct* model that is
// the wrong token: Llama 3 ends a turn with <|eot_id|> while its base EOS is
// <|end_of_text|>, Qwen/ChatML uses <|im_end|>, Gemma uses <end_of_turn>. None
// of those stop the decode loop, so the model sails past the end of its answer
// and keeps writing until it hits max_tokens - inventing a conversation with
// itself:
//
//   hi
//   Assistant: hw are you
//   Assistant: hi
//   Assistant: hw are you        <- 500 more tokens of this
//
// That is not the model hallucinating in the usual sense. It answered, said it
// was done, and was ignored. This filter enforces the stop in the streaming
// layer, so every model on every engine ends its turn where it meant to.

// End-of-turn markers across the families Lizard runs. Matching on text rather
// than token id keeps this working for models whose GGUF metadata does not
// declare the instruct token at all.
export const DEFAULT_STOP_MARKERS = Object.freeze([
  "<|eot_id|>",
  "<|end_of_text|>",
  "<|endoftext|>",
  "<|im_end|>",
  "<|im_start|>",
  "<end_of_turn>",
  "<turn|>",
  "<|turn>",
  "</s>",
  "<|assistant|>",
  "<|user|>",
  "<|system|>",
]);

// Plain-string markers safe to hand the native worker's decode loop, which
// does a substring match (no regex). These never occur in ordinary prose, so
// they are safe to stop on for any model. The role-echo pattern below stays
// JS-only because it is a regex; the streaming filter applies it.
export const NATIVE_STOP_STRINGS = Object.freeze([
  "<|eot_id|>",
  "<|end_of_text|>",
  "<|endoftext|>",
  "<|im_end|>",
  "<|im_start|>",
  "<end_of_turn>",
  "<start_of_turn>",
  "<turn|>",
  "</s>",
]);

// Additional stops for the instruction format used ONLY for template-less base
// models (renderInstructionPrompt): the model finishes and starts the next
// "### Instruction:" / "### Response:" block. Kept separate because "###" can
// legitimately appear as a Markdown heading in an instruct model's answer, so
// these are applied only when the base-model instruction format is in use.
export const INSTRUCTION_STOP_STRINGS = Object.freeze([
  "### Instruction:",
  "### Response:",
  "###Instruction",
  "\n### ",
  // Base models also continue a conversation by writing the other speaker's
  // turn - "2+2=4\nUser: ...". These are safe only for base models (an instruct
  // answer might legitimately quote "User:"), which is why they live here and
  // not in NATIVE_STOP_STRINGS.
  "\nUser:",
  "\nUser :",
  "\nAssistant:",
  "\nHuman:",
  "\nAI:",
  "\nQuestion:",
]);

// A model that starts writing the next turn's header has finished this one.
// This catches the runaway above even when the model emits no special token,
// which is what happens once a plain-text transcript teaches it the pattern.
// Anchored to a line start so prose like "the User: field" is not a stop.
// Also covers the instruction format used for template-less base models
// (renderInstructionPrompt in runtime/tokenizer/index.js): the model finishes
// its answer and starts writing the next "### Instruction:" block, which is
// exactly where the turn ends. And the bracketed labels the chat layer uses
// for prior turns, which a base model will happily start imitating.
const ROLE_ECHO_PATTERN = /\n\s*(?:###\s|(?:user|assistant|system|human|ai)\s*:|\[(?:them|you)\])/i;

// Longest partial role echo worth holding: "\n  assistant " and similar.
const ROLE_ECHO_HOLD = 16;

/**
 * How many trailing characters could still turn into a stop match.
 *
 * The filter used to hold a fixed 11-character tail on every push, which is
 * correct but far more than necessary, and it cost real first-token latency:
 * at ~4 characters per token nothing could be released until 3-4 tokens had
 * been decoded, so the first SSE delta arrived ~550 ms after the first token
 * was ready (measured 2026-07-30 - 13 ms to first token inside the runtime,
 * 560-598 ms to the first byte on the socket).
 *
 * Only an *ambiguous* tail needs holding. Every marker starts with "<", and a
 * role echo has to start at a newline, so ordinary prose is unambiguous and can
 * go out immediately. This holds the shortest suffix that could still complete
 * a match and releases everything before it - same safety property, no fixed
 * latency floor.
 */
function ambiguousTailLength(text, markers, detectRoleEcho) {
  const maxHold = Math.max(
    markers.reduce((longest, marker) => Math.max(longest, marker.length), 0),
    detectRoleEcho ? ROLE_ECHO_HOLD : 0,
  ) - 1;
  if (maxHold <= 0 || !text) {
    return 0;
  }
  const from = Math.max(0, text.length - maxHold);
  for (let index = from; index < text.length; index += 1) {
    const suffix = text.slice(index);
    // Could this suffix still grow into one of the markers?
    if (markers.some((marker) => marker.length > suffix.length && marker.startsWith(suffix))) {
      return text.length - index;
    }
    // Could it still grow into a role echo? Those must begin at a newline, and
    // once a colon has arrived the question is already settled by findStop - so
    // a suffix containing ":" is never ambiguous.
    if (detectRoleEcho
      && suffix.startsWith("\n")
      && !suffix.includes(":")
      && /^\n[\s#a-z[\]]*$/i.test(suffix)) {
      return text.length - index;
    }
  }
  return 0;
}

/**
 * Streaming stop-sequence filter.
 *
 * Markers can straddle token boundaries ("<|eot" + "_id|>"), so the tail of the
 * text is held back until it is long enough to be unambiguous. push() returns
 * only the text that is safe to show.
 */
// Greedy decoding (the only mode Caterpillar supports) loops: a base model
// that runs out of things to say will repeat the same sentence until the token
// budget is gone. A stop MARKER cannot catch that - there is no marker, just
// the same text again. Detect a tail that has already occurred several times
// and end the turn there.
const LOOP_WINDOW = 48;      // chars of tail to look for
const LOOP_MIN_REPEATS = 3;  // occurrences before it is a loop, not a refrain
const LOOP_SCAN_EVERY = 16;  // chars between scans, to keep this off the hot path
const LOOP_MIN_LENGTH = 160; // don't judge anything shorter than this

function looksLikeLoop(text) {
  if (text.length < LOOP_MIN_LENGTH) {
    return false;
  }
  const tail = text.slice(-LOOP_WINDOW);
  if (!/[a-z0-9]/i.test(tail)) {
    return false;
  }
  let count = 0;
  let index = text.indexOf(tail);
  while (index !== -1) {
    count += 1;
    if (count >= LOOP_MIN_REPEATS) {
      return true;
    }
    index = text.indexOf(tail, index + 1);
  }
  return false;
}

export function createStopSequenceFilter({
  markers = DEFAULT_STOP_MARKERS,
  detectRoleEcho = true,
  detectLoop = true,
} = {}) {
  let emitted = "";   // everything released to the caller
  let pending = "";   // held back: could be the start of a marker
  let stopped = false;
  let reason = null;
  let lastLoopScanAt = 0;

  function findStop(text) {
    let best = -1;
    let bestReason = null;
    for (const marker of markers) {
      const index = text.indexOf(marker);
      if (index !== -1 && (best === -1 || index < best)) {
        best = index;
        bestReason = marker;
      }
    }
    if (detectRoleEcho) {
      const echo = ROLE_ECHO_PATTERN.exec(text);
      if (echo && (best === -1 || echo.index < best)) {
        best = echo.index;
        bestReason = "role-echo";
      }
    }
    return { index: best, reason: bestReason };
  }

  return {
    /** Feed decoded token text; returns the portion safe to emit. */
    push(text) {
      if (stopped || !text) {
        return "";
      }
      pending += text;
      const found = findStop(pending);
      if (found.index !== -1) {
        stopped = true;
        reason = found.reason;
        const release = pending.slice(0, found.index);
        pending = "";
        emitted += release;
        return release;
      }
      // Hold back only the tail that could still complete a stop match, rather
      // than a fixed-length tail. See ambiguousTailLength.
      const keep = ambiguousTailLength(pending, markers, detectRoleEcho);
      if (pending.length <= keep) {
        return "";
      }
      const release = pending.slice(0, pending.length - keep);
      pending = pending.slice(pending.length - keep);
      emitted += release;
      // Scanned on the released text (not the held-back tail) and only every
      // so often, so a long answer does not turn this into an O(n^2) sweep.
      if (detectLoop && emitted.length - lastLoopScanAt >= LOOP_SCAN_EVERY) {
        lastLoopScanAt = emitted.length;
        if (looksLikeLoop(emitted)) {
          stopped = true;
          reason = "repetition-loop";
        }
      }
      return release;
    },

    /** Flush whatever is still held back once generation ends. */
    finish() {
      if (stopped || !pending) {
        const tail = "";
        pending = "";
        return tail;
      }
      const release = pending;
      pending = "";
      emitted += release;
      return release;
    },

    get stopped() { return stopped; },
    get reason() { return reason; },
    get text() { return emitted; },
  };
}

/**
 * Non-streaming equivalent, for a final output assembled after the fact.
 * Returns the answer up to its first stop marker.
 */
export function truncateAtStopSequence(text, { markers = DEFAULT_STOP_MARKERS, detectRoleEcho = true } = {}) {
  const source = String(text ?? "");
  let cut = -1;
  let reason = null;
  for (const marker of markers) {
    const index = source.indexOf(marker);
    if (index !== -1 && (cut === -1 || index < cut)) {
      cut = index;
      reason = marker;
    }
  }
  if (detectRoleEcho) {
    const echo = ROLE_ECHO_PATTERN.exec(source);
    if (echo && (cut === -1 || echo.index < cut)) {
      cut = echo.index;
      reason = "role-echo";
    }
  }
  if (cut === -1) {
    return { text: source, stopped: false, reason: null };
  }
  return { text: source.slice(0, cut).trimEnd(), stopped: true, reason };
}
