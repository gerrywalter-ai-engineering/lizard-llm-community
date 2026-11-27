// The two-token answer check.
//
// "Loaded" used to mean `ensureModelLoaded` returned without throwing - i.e.
// the worker process started and mapped the weights. That is a necessary
// condition and not a sufficient one: a worker can come up, report itself
// loaded, and then produce nothing when asked to decode (an unsupported
// architecture that only fails at the first matmul, a GPU allocation that
// succeeded for the weights and not for the KV cache, a tokenizer whose vocab
// did not build). Everything downstream - the Activate button, the header pill,
// and now the local API port - was believing that report.
//
// So before any of them commit, the model is asked for two tokens. It is the
// cheapest question that exercises the whole path: tokenize, prefill, decode,
// detokenize. If it answers, the model works. If it does not, we say so, in
// words, instead of opening a port onto a model that cannot reply.

/** How long a two-token answer is allowed to take before it counts as a hang. */
export const DEFAULT_LOAD_CHECK_TIMEOUT_MS = 90_000;

// Deliberately trivial and deterministic: the check is about whether tokens
// come out at all, not about what they say. A prompt that invites a long answer
// would just make the check slower.
export const LOAD_CHECK_PROMPT = "Reply with the single word: ok";

export const LOAD_CHECK_FAILURES = Object.freeze({
  NO_OUTPUT: "no_output",
  TIMEOUT: "timeout",
  ERROR: "error",
});

// What the user is told, per failure kind. These are the sentences the UI
// shows; they name what happened and what to do, because the alternative the
// user reported is a button that silently says "loaded" for a model that is not.
const FAILURE_NOTES = Object.freeze({
  [LOAD_CHECK_FAILURES.NO_OUTPUT]:
    "The model loaded into memory but produced no tokens when asked for a two-token reply. The weights are resident and the engine is not decoding, which usually means this GGUF's architecture is not supported by the selected engine. Try the other native engine, or pick a different quantization.",
  [LOAD_CHECK_FAILURES.TIMEOUT]:
    "The model loaded but did not answer a two-token test in time. It is either still compiling shaders on first run, or it is thrashing because the weights do not fit the GPU budget. Wait and load it again, or activate a smaller quantization.",
  [LOAD_CHECK_FAILURES.ERROR]:
    "The model loaded but the two-token test failed with an error. The model is not usable in this state.",
});

export function loadCheckNote(reason, message = null) {
  const base = FAILURE_NOTES[reason] ?? FAILURE_NOTES[LOAD_CHECK_FAILURES.ERROR];
  return message ? `${base} (${message})` : base;
}

/**
 * Ask an already-loaded model for two tokens.
 *
 * `generate` receives `{ prompt, maxTokens, onDelta }` and resolves to
 * `{ output }` - injected so this stays a pure function of the runtime and can
 * be tested without a model.
 *
 * Never throws: a check that could fail the caller would be a second way for
 * loading to break. The verdict is always the return value.
 *
 * @returns {Promise<{ok:boolean, reason:string|null, note:string|null, message:string|null,
 *                     text:string, deltas:number, durationMs:number, checkedAt:string,
 *                     timeoutMs:number}>}
 */
export async function runTwoTokenLoadCheck({
  generate,
  prompt = LOAD_CHECK_PROMPT,
  maxTokens = 2,
  timeoutMs = DEFAULT_LOAD_CHECK_TIMEOUT_MS,
  now = () => Date.now(),
} = {}) {
  const startedAt = now();
  const verdict = (patch) => ({
    ok: false,
    reason: null,
    note: null,
    message: null,
    text: "",
    deltas: 0,
    durationMs: Math.max(0, now() - startedAt),
    checkedAt: new Date().toISOString(),
    timeoutMs,
    ...patch,
  });

  if (typeof generate !== "function") {
    return verdict({
      reason: LOAD_CHECK_FAILURES.ERROR,
      message: "No generate function was supplied to the load check.",
      note: loadCheckNote(LOAD_CHECK_FAILURES.ERROR, "internal: no generate function"),
    });
  }

  let deltas = 0;
  let streamed = "";
  let timer = null;
  try {
    const generation = generate({
      prompt,
      maxTokens,
      onDelta: (text) => {
        if (!text) return;
        deltas += 1;
        streamed += text;
      },
    });
    // The timeout races the generation rather than cancelling it: the runtime
    // owns its own request lifetime, and a check that tried to tear a worker
    // down mid-decode could leave the residency claim pointing at a process it
    // no longer controls.
    const timedOut = Symbol("load-check-timeout");
    const result = await Promise.race([
      generation,
      new Promise((resolve) => {
        timer = setTimeout(() => resolve(timedOut), timeoutMs);
        timer.unref?.();
      }),
    ]);
    if (result === timedOut) {
      return verdict({
        reason: LOAD_CHECK_FAILURES.TIMEOUT,
        message: `No reply within ${Math.round(timeoutMs / 1000)} seconds.`,
        note: loadCheckNote(LOAD_CHECK_FAILURES.TIMEOUT),
        text: streamed,
        deltas,
      });
    }
    const text = String(result?.output ?? streamed ?? "");
    if (!text.trim()) {
      return verdict({
        reason: LOAD_CHECK_FAILURES.NO_OUTPUT,
        message: "The generation completed with an empty response.",
        note: loadCheckNote(LOAD_CHECK_FAILURES.NO_OUTPUT),
        text,
        deltas,
      });
    }
    return verdict({ ok: true, text: text.trim().slice(0, 120), deltas });
  } catch (error) {
    const message = String(error?.message ?? error).slice(0, 300);
    return verdict({
      reason: LOAD_CHECK_FAILURES.ERROR,
      message,
      note: loadCheckNote(LOAD_CHECK_FAILURES.ERROR, message),
      text: streamed,
      deltas,
    });
  } finally {
    if (timer) clearTimeout(timer);
  }
}
