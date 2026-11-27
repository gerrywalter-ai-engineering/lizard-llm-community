import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import process from "node:process";
import { generateResponse, generateWithExistingBackend } from "../src/lib/generate.js";

test("generateWithExistingBackend prefers the completion binary for no-conversation mode", async () => {
  const completionDirectory = fs.mkdtempSync(path.join(os.tmpdir(), "lizard-completion-pref-"));
  const completionBinary = path.join(completionDirectory, "llama-completion.exe");
  fs.copyFileSync(process.execPath, completionBinary);
  let captured = null;
  const result = await generateWithExistingBackend({
    recommendation: {
      selected: {
        contextTokens: 128,
        gpuLayers: 0,
        threads: 4,
        kvCache: "f16",
      },
    },
    modelArtifact: {
      modelId: "demo",
      modelPath: "E:\\demo.gguf",
    },
    profile: {
      modelPath: "E:\\demo.gguf",
      contextSize: 128,
      gpuLayers: 0,
      threads: 4,
      batchSize: 512,
      ubatchSize: 128,
      kvCache: "f16",
      mmap: true,
      mlock: false,
      noConversationMode: true,
    },
    prompt: "Hello",
    maxTokens: 1,
    config: {
      generation: { temperature: 0 },
      engine: {
        cliBinary: "powershell.exe",
        completionBinary,
        preferredBinaryDirectories: [],
      },
    },
    localRunner: async ({ executable, args }) => {
      captured = { executable, args };
      return {
        runtime: "llama-completion",
        durationMs: 1,
        output: "Hello",
        stderr: "",
      };
    },
  });

  assert.equal(result.runtime, "llama-completion");
  assert.equal(captured.executable, completionBinary);
  assert.ok(captured.args.includes("--no-conversation"));
  assert.equal(captured.args.includes("--log-disable"), false);
});

test("generateWithExistingBackend retries with llama-cli when llama-completion returns no output", async () => {
  const completionDirectory = fs.mkdtempSync(path.join(os.tmpdir(), "lizard-completion-"));
  const completionBinary = path.join(completionDirectory, "llama-completion.exe");
  fs.copyFileSync(process.execPath, completionBinary);
  const calls = [];
  const result = await generateWithExistingBackend({
    recommendation: {
      selected: {
        contextTokens: 128,
        gpuLayers: 0,
        threads: 4,
        kvCache: "f16",
      },
    },
    modelArtifact: {
      modelId: "demo",
      modelPath: "E:\\demo.gguf",
    },
    profile: {
      modelPath: "E:\\demo.gguf",
      contextSize: 128,
      gpuLayers: 0,
      threads: 4,
      batchSize: 512,
      ubatchSize: 128,
      kvCache: "f16",
      mmap: true,
      mlock: false,
      noConversationMode: true,
    },
    prompt: "Hello",
    maxTokens: 1,
    config: {
      generation: { temperature: 0 },
      engine: {
        cliBinary: "powershell.exe",
        completionBinary,
        preferredBinaryDirectories: [],
      },
    },
    localRunner: async ({ executable }) => {
      calls.push(executable);
      if (executable === completionBinary) {
        return {
          runtime: "llama-completion",
          durationMs: 1,
          output: "",
          stderr: "",
        };
      }
      return {
        runtime: "llama-cli",
        durationMs: 1,
        output: "Hello",
        stderr: "",
      };
    },
  });

  assert.equal(result.runtime, "llama-cli");
  assert.equal(calls.length, 2);
  assert.equal(calls[0], completionBinary);
  assert.match(String(calls[1]).toLowerCase(), /powershell(?:\.exe)?$/);
});

const recommendation = {
  selected: {
    contextTokens: 8192,
    gpuLayers: 0,
    threads: 4,
    kvCache: "q4_0",
  },
};

const config = {
  generation: {
    temperature: 0.2,
  },
};

test("falls back to Ollama when local model loading fails", async () => {
  let localCalls = 0;
  let ollamaCalls = 0;

  const result = await generateResponse({
    recommendation,
    modelArtifact: {
      modelId: "gemma4-e4b",
      modelPath: "C:\\fake\\gemma4-e4b.gguf",
      ollamaTag: "gemma4:e4b",
    },
    prompt: "Reply with exactly: gemma4 works.",
    maxTokens: 16,
    config,
    executableOverride: "C:\\fake\\llama-cli.exe",
    localRunner: async () => {
      localCalls += 1;
      throw new Error("llama-cli exited with code 1: Failed to load the model\nwrong number of tensors");
    },
    ollamaRunner: async () => {
      ollamaCalls += 1;
      return {
        runtime: "ollama",
        durationMs: 123,
        output: "gemma4 works.",
        stderr: "",
        fallbackReason: "local model load failed; used Ollama runtime",
      };
    },
  });

  assert.equal(localCalls, 1);
  assert.equal(ollamaCalls, 1);
  assert.equal(result.runtime, "ollama");
  assert.equal(result.output, "gemma4 works.");
  assert.match(result.note, /Ollama runtime/);
});

test("does not mask non-load local generation failures", async () => {
  await assert.rejects(() => generateResponse({
    recommendation,
    modelArtifact: {
      modelId: "llama-3.2-3b-instruct",
      modelPath: "C:\\fake\\llama.gguf",
      ollamaTag: "llama3.2:3b",
    },
    prompt: "test",
    maxTokens: 8,
    config,
    executableOverride: "C:\\fake\\llama-cli.exe",
    localRunner: async () => {
      throw new Error("llama-cli exited with code 1: invalid argument");
    },
  }), /invalid argument/);
});
