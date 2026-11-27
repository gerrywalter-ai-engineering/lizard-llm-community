// Prompt library: the "/" menu in the composer.
//
// Built-ins ship with the app and cannot be deleted (they can be copied and
// edited into your own). User prompts live in <runtimeDirectory>/chat/prompts.json.
//
// A prompt body may contain {{input}}, which is replaced with whatever the user
// typed after the command. A body with no placeholder gets the input appended,
// so "/review here is my code" behaves the way people expect either way.

import fs from "node:fs";
import path from "node:path";
import { ensureDirectory } from "../paths.js";
import { newId } from "./chat-store.js";

export const BUILT_IN_PROMPTS = Object.freeze([
  {
    id: "bp-summarize",
    command: "summarize",
    title: "Summarize",
    description: "Condense text to its load-bearing points.",
    body: "Summarize the following in at most five bullet points. Keep numbers, names and " +
      "caveats; drop everything else.\n\n{{input}}",
  },
  {
    id: "bp-explain",
    command: "explain",
    title: "Explain",
    description: "Walk through how something works.",
    body: "Explain the following clearly, from the top. State what it does before how it does it, " +
      "and name any assumption you had to make.\n\n{{input}}",
  },
  {
    id: "bp-review",
    command: "review",
    title: "Review code",
    description: "Find real defects, not style opinions.",
    body: "Review this code. Report correctness bugs, edge cases and resource leaks first — each with " +
      "the input that triggers it. Mention style only where it hides a defect.\n\n{{input}}",
  },
  {
    id: "bp-tests",
    command: "tests",
    title: "Write tests",
    description: "Cover the behaviour, including the edges.",
    body: "Write tests for this code. Cover the happy path, the boundaries, and the failure modes. " +
      "Use the test framework the code already implies.\n\n{{input}}",
  },
  {
    id: "bp-commit",
    command: "commit",
    title: "Commit message",
    description: "Describe a diff the way a maintainer would.",
    body: "Write a commit message for this diff. One imperative subject line under 72 characters, " +
      "then a body explaining why the change was needed.\n\n{{input}}",
  },
  {
    id: "bp-rewrite",
    command: "rewrite",
    title: "Rewrite",
    description: "Same meaning, better sentences.",
    body: "Rewrite the following so it is clearer and shorter. Keep the meaning and the register " +
      "exactly; do not add anything that was not there.\n\n{{input}}",
  },
  {
    id: "bp-translate",
    command: "translate",
    title: "Translate",
    description: "Into the language you name first.",
    body: "Translate the following. The first line names the target language.\n\n{{input}}",
  },
  {
    id: "bp-table",
    command: "table",
    title: "To table",
    description: "Turn prose or data into a Markdown table.",
    body: "Turn the following into a Markdown table. Choose columns that make the data comparable, " +
      "and leave a cell empty rather than inventing a value.\n\n{{input}}",
  },
  {
    id: "bp-ask-docs",
    command: "docs",
    title: "Ask my documents",
    description: "Answer strictly from the workspace library.",
    body: "Answer using only the context from my documents. Cite each claim as [1], [2]. If the " +
      "documents do not answer it, say exactly that and stop.\n\n{{input}}",
  },
]);

function promptsPath(config) {
  return path.join(config?.runtimeDirectory ?? process.cwd(), "chat", "prompts.json");
}

function readUserPrompts(config) {
  try {
    const filePath = promptsPath(config);
    if (!fs.existsSync(filePath)) {
      return [];
    }
    const parsed = JSON.parse(fs.readFileSync(filePath, "utf8"));
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function writeUserPrompts(config, prompts) {
  const filePath = promptsPath(config);
  ensureDirectory(path.dirname(filePath));
  fs.writeFileSync(filePath, JSON.stringify(prompts, null, 2));
  return prompts;
}

export function listPrompts(config) {
  const user = readUserPrompts(config).map((prompt) => ({ ...prompt, builtIn: false }));
  const builtIn = BUILT_IN_PROMPTS.map((prompt) => ({ ...prompt, builtIn: true }));
  // A user prompt with the same command wins — that is how you override a built-in.
  const overridden = new Set(user.map((prompt) => prompt.command));
  return [...user, ...builtIn.filter((prompt) => !overridden.has(prompt.command))];
}

export function savePrompt(config, patch) {
  const prompts = readUserPrompts(config);
  const command = normalizeCommand(patch.command);
  if (!command) {
    throw new Error("A prompt needs a command name.");
  }
  if (!String(patch.body ?? "").trim()) {
    throw new Error("A prompt needs a body.");
  }
  const index = prompts.findIndex((prompt) => prompt.id === patch.id);
  const entry = {
    id: patch.id && index !== -1 ? patch.id : newId("pr-"),
    command,
    title: String(patch.title ?? command).slice(0, 60),
    description: String(patch.description ?? "").slice(0, 140),
    body: String(patch.body),
  };
  if (index === -1) {
    prompts.push(entry);
  } else {
    prompts[index] = entry;
  }
  writeUserPrompts(config, prompts);
  return entry;
}

export function deletePrompt(config, promptId) {
  const remaining = readUserPrompts(config).filter((prompt) => prompt.id !== promptId);
  writeUserPrompts(config, remaining);
  return { ok: true };
}

export function normalizeCommand(value) {
  return String(value ?? "")
    .trim()
    .replace(/^\//, "")
    .toLowerCase()
    .replace(/[^a-z0-9-]/g, "")
    .slice(0, 24);
}

/** Expands "/command rest of the message" into the prompt body. */
export function expandPrompt(prompts, text) {
  const match = /^\/([a-z0-9-]+)\s*([\s\S]*)$/i.exec(String(text ?? "").trim());
  if (!match) {
    return { expanded: false, text };
  }
  const prompt = prompts.find((entry) => entry.command === match[1].toLowerCase());
  if (!prompt) {
    return { expanded: false, text };
  }
  const input = match[2].trim();
  const body = prompt.body.includes("{{input}}")
    ? prompt.body.replace(/\{\{input\}\}/g, input)
    : `${prompt.body}\n\n${input}`.trim();
  return { expanded: true, text: body, prompt: { id: prompt.id, command: prompt.command, title: prompt.title } };
}
