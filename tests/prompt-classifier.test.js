import test from "node:test";
import assert from "node:assert/strict";
import { classifyPrompt } from "../src/lib/prompt-classifier.js";

test("classifies coding prompts", () => {
  const result = classifyPrompt({ prompt: "Refactor this function and explain the bug in the code block ```js```" });
  assert.equal(result.taskType, "coding");
  assert.equal(result.recommendedMode, "balanced");
  assert.equal(result.contextNeed, 8192);
});

test("classifies summarization prompts", () => {
  const result = classifyPrompt({ prompt: "Summarize this architecture note into a TL;DR." });
  assert.equal(result.taskType, "summarization");
});

