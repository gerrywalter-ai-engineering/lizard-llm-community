function hasAny(text, patterns) {
  return patterns.some((pattern) => pattern.test(text));
}

export function classifyPrompt({ explicitTask, prompt }) {
  if (explicitTask) {
    return {
      taskType: explicitTask,
      difficulty: explicitTask === "reasoning" || explicitTask === "coding" ? "medium" : "low",
      recommendedMode: explicitTask === "reasoning" ? "quality" : "balanced",
      contextNeed: explicitTask === "long_context" ? 16384 : explicitTask === "coding" ? 8192 : 4096,
      rationale: `Task forced by user: ${explicitTask}`,
    };
  }

  const text = String(prompt ?? "").toLowerCase();
  const tokenEstimate = Math.ceil(text.length / 4);

  let taskType = "chat";
  if (hasAny(text, [/\bjson\b/, /\byaml\b/, /\bschema\b/, /\bstructured\b/])) {
    taskType = "structured_output";
  } else if (hasAny(text, [/\bsummarize\b/, /\bsummary\b/, /\btl;dr\b/, /\babstract\b/])) {
    taskType = "summarization";
  } else if (hasAny(text, [/\bcode\b/, /\bfunction\b/, /\bclass\b/, /\brefactor\b/, /\bdebug\b/, /\bbug\b/, /```/])) {
    taskType = "coding";
  } else if (hasAny(text, [/\barchitecture\b/, /\btrade-?off\b/, /\bcompare\b/, /\bplan\b/, /\breason\b/, /\bdesign\b/])) {
    taskType = "reasoning";
  } else if (hasAny(text, [/\bdocument\b/, /\btranscript\b/, /\blog file\b/, /\breview the attached\b/, /\bcontext window\b/]) || tokenEstimate > 2500) {
    taskType = "long_context";
  }

  let difficulty = "low";
  if (tokenEstimate > 1500 || taskType === "long_context") {
    difficulty = "high";
  } else if (tokenEstimate > 400 || taskType === "coding" || taskType === "reasoning") {
    difficulty = "medium";
  }

  const recommendedMode =
    taskType === "reasoning" || taskType === "long_context" ? "quality" :
    taskType === "coding" || taskType === "structured_output" ? "balanced" :
    "fast";

  const contextNeed =
    taskType === "long_context" ? 16384 :
    taskType === "coding" || taskType === "reasoning" ? 8192 :
    taskType === "structured_output" ? 6144 :
    4096;

  return {
    taskType,
    difficulty,
    recommendedMode,
    contextNeed,
    rationale: prompt ? `Classified from prompt heuristics as ${taskType}` : "No prompt supplied; defaulting to chat",
  };
}

