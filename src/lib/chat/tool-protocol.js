// Tool calling for models that were never fine-tuned for tool calling.
//
// Lizard's native runtime takes a prompt string and returns text - there is no
// structured function-calling channel, and the small quantised models that run
// on this hardware mostly have no tool-call template either. So the protocol is
// text: the system block teaches one tag, the parser reads it back, and the
// loop re-prompts with the result. Any model that can follow an instruction can
// use a plugin, and a model that ignores the tag simply answers normally.
//
// The shape is deliberately close to the widely-published <tool_call> JSON
// convention, because that is what most instruction-tuned models have seen.

export const TOOL_CALL_PATTERN = /<tool_call>\s*([\s\S]*?)\s*<\/tool_call>/i;
export const MAX_TOOL_ITERATIONS = 4;
const TOOL_RESULT_MAX_CHARS = 4000;

/** The instruction block prepended when a workspace has tools enabled. */
export function buildToolInstructions(tools) {
  if (tools.length === 0) {
    return "";
  }
  const lines = tools.map((tool) => {
    const properties = tool.inputSchema?.properties ?? {};
    const signature = Object.keys(properties)
      .map((key) => `${key}: ${properties[key]?.type ?? "any"}`)
      .join(", ");
    const required = Array.isArray(tool.inputSchema?.required) && tool.inputSchema.required.length > 0
      ? ` (required: ${tool.inputSchema.required.join(", ")})`
      : "";
    return `- ${tool.name}(${signature})${required} — ${collapse(tool.description)}`;
  });
  return [
    "You can call tools that run on this machine. Available tools:",
    ...lines,
    "",
    "To call one, reply with nothing but this tag:",
    '<tool_call>{"name": "tool.name", "arguments": {"key": "value"}}</tool_call>',
    "",
    "Rules: one call per reply; wait for the result before calling again; when you " +
    "have enough information, answer the user in plain prose without any tag. If no " +
    "tool helps, just answer.",
  ].join("\n");
}

function collapse(text) {
  return String(text ?? "").replace(/\s+/g, " ").trim().slice(0, 180);
}

/** The literal opening of a tool call, and the only thing worth withholding. */
export const TOOL_CALL_OPEN = "<tool_call>";

/**
 * Lets a tool-capable turn stream its prose without ever showing a tool call.
 *
 * Withholding the whole pass was the alternative, and it is what the chat route
 * used to do: with any plugin available the model generated in silence and the
 * finished reply arrived as one block. Since plugins are on by default, that was
 * most answers.
 *
 * Streaming optimistically is safe because the protocol above tells the model to
 * reply with NOTHING BUT the tag when it wants a tool. So a call announces
 * itself in the first characters, and text only has to be held while it is still
 * a possible prefix of `<tool_call>` - eleven characters at worst, which no
 * reader perceives. Once the text cannot be the tag, everything held is released
 * and the rest streams straight through.
 *
 * `onText` receives only text that is certainly not part of a tool call.
 */
export function createToolCallStreamGate(onText) {
  let held = "";
  // null while it could still go either way.
  let verdict = null;

  const release = () => {
    const pending = held;
    held = "";
    if (pending) onText(pending);
  };

  return {
    push(text) {
      if (verdict === "withhold") return;
      if (verdict === "stream") {
        onText(text);
        return;
      }
      held += text;
      // Leading whitespace is common before the tag and must not defeat the
      // prefix test, but it is still ordinary output if no tag follows.
      const lead = held.replace(/^\s+/, "").toLowerCase();
      if (lead.length < TOOL_CALL_OPEN.length) {
        if (TOOL_CALL_OPEN.startsWith(lead)) return;
        verdict = "stream";
      } else if (lead.startsWith(TOOL_CALL_OPEN)) {
        verdict = "withhold";
        held = "";
        return;
      } else {
        verdict = "stream";
      }
      release();
    },

    /** Ends the pass. Anything still held was never a tool call. */
    finish() {
      if (verdict === "withhold") return;
      verdict = "stream";
      release();
    },

    /** True when this pass was withheld as a tool call. */
    get withheld() {
      return verdict === "withhold";
    },
  };
}

/**
 * Reads a tool call out of model output.
 * Returns null when the model answered normally, which is the common case.
 */
export function parseToolCall(text) {
  const match = TOOL_CALL_PATTERN.exec(String(text ?? ""));
  if (!match) {
    return null;
  }
  let payload = match[1].trim();
  // Models frequently wrap the JSON in a fence out of habit.
  payload = payload.replace(/^```(?:json)?\s*/i, "").replace(/```$/i, "").trim();
  try {
    const parsed = JSON.parse(payload);
    if (!parsed?.name) {
      return null;
    }
    return {
      name: String(parsed.name),
      arguments: parsed.arguments ?? parsed.args ?? {},
      raw: match[0],
    };
  } catch {
    return { name: null, arguments: {}, raw: match[0], malformed: payload };
  }
}

/** Strips a tool-call tag from text that is about to be shown to the user. */
export function stripToolCalls(text) {
  return String(text ?? "").replace(/<tool_call>[\s\S]*?<\/tool_call>/gi, "").trim();
}

export function formatToolResult(name, result) {
  const body = String(result?.text ?? "").slice(0, TOOL_RESULT_MAX_CHARS) || "(the tool returned no output)";
  return [
    `<tool_result name="${name}">`,
    result?.isError ? `ERROR: ${body}` : body,
    "</tool_result>",
  ].join("\n");
}

/**
 * Folds everything the model needs into one prompt string: system persona,
 * tool instructions, retrieved document context, prior turns, and this turn's
 * message (plus any tool round-trips already made in this turn).
 *
 * Prompt composition is the only place this integration touches generation —
 * the runtime call itself is unchanged.
 */
export function composePrompt({
  systemPrompt,
  toolInstructions = "",
  contextBlock = "",
  history = [],
  userMessage,
  toolTranscript = [],
  attachments = [],
  // Caterpillar only. Renders prior turns INLINE in the instruction transcript
  // instead of as a summary block in the system turn, which makes each turn's
  // prompt a strict extension of the previous one. That is what lets the
  // runner's resident KV cache be reused (see the prompt cache in
  // caterpillar-runner/src/server.cpp): with history in the system block the
  // prompt changes in the middle every turn and nothing can be carried over.
  inlineHistory = false,
}) {
  // Everything that is context rather than the question goes in the system
  // block, which the runtime hands to the model's own chat template
  // (buildPromptArtifact -> renderChatTemplate). The prompt itself is the
  // user's message and nothing else.
  //
  // This used to build one flat "User: ... / Assistant: ..." transcript and
  // pass it as the prompt with no system prompt at all. An instruction-tuned
  // model handed a document that looks like a chat log does the only sensible
  // thing with it: it continues the document, writing both sides -
  //   hi
  //   Assistant: hw are you
  //   Assistant: hi            <- for 500 tokens
  // Teaching the model a "Role:" pattern and then asking it not to follow the
  // pattern is a losing game; the roles belong in the template, not the text.
  const systemSections = [];
  if (systemPrompt?.trim()) {
    systemSections.push(systemPrompt.trim());
  }
  if (toolInstructions) {
    systemSections.push(toolInstructions);
  }
  if (contextBlock) {
    systemSections.push(contextBlock);
  }
  if (attachments.length > 0) {
    systemSections.push(
      [
        "Files the user attached to this message:",
        "",
        ...attachments.map((attachment) => `--- ${attachment.name} ---\n${attachment.text}`),
      ].join("\n"),
    );
  }
  if (history.length > 0 && !inlineHistory) {
    systemSections.push(
      [
        "Earlier in this conversation (for context only - do not repeat or continue it, "
        + "and never write the user's side):",
        "",
        ...history.map((turn) => `[${turn.role === "user" ? "them" : "you"}] ${turn.content}`),
      ].join("\n"),
    );
  }

  // The separators mirror renderInstructionPrompt exactly (it emits
  // "### Instruction:\n\n<prompt>\n\n### Response:\n"), so the rendered turn N
  // is byte-for-byte turn N-1 plus the new tail.
  const promptSections = [];
  if (inlineHistory && history.length > 0) {
    let transcript = "";
    for (const turn of history) {
      transcript += turn.content;
      transcript += turn.role === "user" ? "\n\n### Response:\n" : "\n\n### Instruction:\n\n";
    }
    promptSections.push(`${transcript}${String(userMessage ?? "")}`);
  } else {
    promptSections.push(String(userMessage ?? ""));
  }
  for (const entry of toolTranscript) {
    promptSections.push(entry);
  }

  return {
    systemPrompt: systemSections.join("\n\n").trim(),
    prompt: promptSections.join("\n\n").trim(),
  };
}
