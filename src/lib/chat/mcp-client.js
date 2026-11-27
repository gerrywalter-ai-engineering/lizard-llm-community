// Minimal Model Context Protocol client (stdio transport).
//
// MCP is JSON-RPC 2.0 over newline-delimited JSON on a child process's stdio.
// The handful of calls a chat client needs - initialize, tools/list, tools/call
// - fit in a couple of hundred lines, so Lizard speaks the protocol directly
// rather than taking an SDK dependency. That keeps the installer small and
// keeps the runtime honest about what it spawns on the user's machine.
//
// Reference: https://modelcontextprotocol.io/specification (2024-11-05)

import { spawn } from "node:child_process";
import { EventEmitter } from "node:events";

const PROTOCOL_VERSION = "2024-11-05";
const DEFAULT_REQUEST_TIMEOUT_MS = 30_000;
const START_TIMEOUT_MS = 45_000;

export class McpStdioClient extends EventEmitter {
  constructor({ id, command, args = [], env = {}, cwd = null, name = null }) {
    super();
    this.id = id;
    this.name = name ?? id;
    this.command = command;
    this.args = args;
    this.env = env;
    this.cwd = cwd;
    this.child = null;
    this.nextRequestId = 1;
    this.pending = new Map();
    this.buffer = "";
    this.serverInfo = null;
    this.tools = [];
    this.status = "stopped";
    this.lastError = null;
  }

  async start() {
    if (this.child) {
      return this;
    }
    this.status = "starting";
    this.child = spawn(this.command, this.args, {
      cwd: this.cwd ?? undefined,
      env: { ...process.env, ...this.env },
      stdio: ["pipe", "pipe", "pipe"],
      // Windows resolves npx/uvx through the shell; without this the spawn
      // fails with ENOENT for every server in the catalog.
      shell: process.platform === "win32",
      windowsHide: true,
    });

    this.child.stdout.setEncoding("utf8");
    this.child.stdout.on("data", (chunk) => this.#onData(chunk));
    this.child.stderr.setEncoding("utf8");
    this.child.stderr.on("data", (chunk) => {
      // MCP servers log to stderr as a matter of course; keep only the tail so
      // a failure has something to show without unbounded memory growth.
      this.lastError = String(chunk).slice(-2000);
      this.emit("log", String(chunk));
    });
    this.child.on("error", (error) => {
      this.status = "failed";
      this.lastError = error.message;
      this.#rejectAll(error);
      this.emit("closed", { code: null, error: error.message });
    });
    this.child.on("exit", (code) => {
      this.status = code === 0 ? "stopped" : "failed";
      this.child = null;
      this.#rejectAll(new Error(`MCP server "${this.name}" exited (code ${code}).`));
      this.emit("closed", { code });
    });

    const initialize = await this.request(
      "initialize",
      {
        protocolVersion: PROTOCOL_VERSION,
        capabilities: { tools: {} },
        clientInfo: { name: "lizard-chat", version: "1.2.50" },
      },
      START_TIMEOUT_MS,
    );
    this.serverInfo = initialize?.serverInfo ?? null;
    this.notify("notifications/initialized", {});
    this.status = "ready";
    return this;
  }

  async listTools() {
    const result = await this.request("tools/list", {});
    this.tools = Array.isArray(result?.tools) ? result.tools : [];
    return this.tools;
  }

  async callTool(name, args) {
    const result = await this.request("tools/call", { name, arguments: args ?? {} }, 120_000);
    return {
      isError: Boolean(result?.isError),
      text: flattenContent(result?.content),
      raw: result,
    };
  }

  request(method, params, timeoutMs = DEFAULT_REQUEST_TIMEOUT_MS) {
    if (!this.child) {
      return Promise.reject(new Error(`MCP server "${this.name}" is not running.`));
    }
    const id = this.nextRequestId++;
    const payload = { jsonrpc: "2.0", id, method, params };
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error(`MCP server "${this.name}" did not answer ${method} within ${Math.round(timeoutMs / 1000)}s.`));
      }, timeoutMs);
      this.pending.set(id, { resolve, reject, timer });
      try {
        this.child.stdin.write(`${JSON.stringify(payload)}\n`);
      } catch (error) {
        clearTimeout(timer);
        this.pending.delete(id);
        reject(error);
      }
    });
  }

  notify(method, params) {
    if (!this.child) return;
    try {
      this.child.stdin.write(`${JSON.stringify({ jsonrpc: "2.0", method, params })}\n`);
    } catch {
      // A notification that cannot be delivered is not worth failing the turn.
    }
  }

  async stop() {
    const child = this.child;
    if (!child) {
      this.status = "stopped";
      return;
    }
    this.child = null;
    this.status = "stopped";
    this.#rejectAll(new Error("MCP server stopped."));
    try {
      child.stdin.end();
    } catch {
      // already gone
    }
    child.kill();
  }

  #onData(chunk) {
    this.buffer += chunk;
    let newline = this.buffer.indexOf("\n");
    while (newline !== -1) {
      const line = this.buffer.slice(0, newline).trim();
      this.buffer = this.buffer.slice(newline + 1);
      newline = this.buffer.indexOf("\n");
      if (!line) continue;
      let message;
      try {
        message = JSON.parse(line);
      } catch {
        continue; // non-JSON banner lines are common; ignore them
      }
      this.#dispatch(message);
    }
  }

  #dispatch(message) {
    if (message.id !== undefined && this.pending.has(message.id)) {
      const entry = this.pending.get(message.id);
      this.pending.delete(message.id);
      clearTimeout(entry.timer);
      if (message.error) {
        entry.reject(new Error(message.error.message ?? "MCP error"));
      } else {
        entry.resolve(message.result);
      }
      return;
    }
    if (message.method) {
      this.emit("notification", message);
    }
  }

  #rejectAll(error) {
    for (const [, entry] of this.pending) {
      clearTimeout(entry.timer);
      entry.reject(error);
    }
    this.pending.clear();
  }
}

/** MCP tool results are a content array; chat only needs the text of it. */
export function flattenContent(content) {
  if (!Array.isArray(content)) {
    return typeof content === "string" ? content : "";
  }
  return content
    .map((part) => {
      if (part?.type === "text") return part.text ?? "";
      if (part?.type === "resource") return part.resource?.text ?? `[resource ${part.resource?.uri ?? ""}]`;
      if (part?.type === "image") return "[image returned by tool]";
      return "";
    })
    .filter(Boolean)
    .join("\n")
    .trim();
}
